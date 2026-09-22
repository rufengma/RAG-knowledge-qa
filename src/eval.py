"""Evaluation harness.

Two layers:
  1. Retrieval metrics (no LLM needed): for each golden question, check whether
     the expected source document appears in the top-k hits.
       - hit_rate@k: fraction of questions with >= 1 hit in top-k
       - recall@k:   fraction of expected docs retrieved (supports multi-doc Qs)
  2. Faithfulness via LLM-as-judge (optional, --judge): asks the LLM whether the
     generated answer is fully supported by the retrieved sources. Needs API key.

`eval/golden_qa.jsonl` — one JSON object per line:
  {"question": "...", "expected_sources": ["rag_concepts.md"], "answer": "..."}

`expected_sources` lists the doc filenames a good retriever should surface.
`answer` (optional) is a reference answer used only by the judge.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import settings  # noqa: E402
from src.embeddings import make_embedding_client  # noqa: E402
from src.qa import answer_question  # noqa: E402
from src.retriever import Retriever  # noqa: E402
from src.vectorstore import VectorStore  # noqa: E402

JUDGE_PROMPT = """\
You are a strict evaluator. Given SOURCES and an ANSWER, reply with exactly one word:
"SUPPORTED" if every factual claim in the answer is supported by the sources,
"UNSUPPORTED" otherwise.

SOURCES:
{sources}

ANSWER:
{answer}"""


def load_golden(path: Path) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {i} of {path}: {exc}") from exc
    if not items:
        raise ValueError(f"No eval items found in {path}")
    return items


def retrieval_metrics(
    retriever: Retriever, golden: list[dict], k_values: tuple[int, ...] = (1, 3, 5)
) -> dict[int, dict[str, float]]:
    """Compute hit_rate@k and recall@k per k."""
    results: dict[int, dict[str, float]] = {}
    for k in k_values:
        hits, recall_sum, n = 0, 0.0, 0
        for item in golden:
            expected = set(item["expected_sources"])
            if not expected:
                continue
            n += 1
            retrieved_sources = {r.chunk.source for r in retriever.retrieve(item["question"], top_k=k)}
            found = expected & retrieved_sources
            hits += 1 if found else 0
            recall_sum += len(found) / len(expected)
        results[k] = {
            "hit_rate": hits / n if n else 0.0,
            "recall": recall_sum / n if n else 0.0,
            "n": n,
        }
    return results


def judge_faithfulness(question: str, answer: str, sources_text: str) -> str:
    """LLM-as-judge: is the answer supported by the sources? Needs API key."""
    from openai import OpenAI

    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is required for --judge")
    client = OpenAI(api_key=settings.openai_api_key, base_url=settings.llm_base_url)
    resp = client.chat.completions.create(
        model=settings.llm_model,
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(sources=sources_text, answer=answer)}],
        temperature=0.0,
        max_tokens=10,
    )
    verdict = resp.choices[0].message.content.strip().upper()
    return "SUPPORTED" if "SUPPORTED" in verdict and "UNSUPPORTED" not in verdict else "UNSUPPORTED"


def compare_configs(
    index_dir: str,
    golden: list[dict],
    top_k_options: tuple[int, ...] = (1, 3, 5),
) -> None:
    """Rebuild-free config sweep: vary only retrieval-time knobs (top_k).

    Chunk-size/strategy comparisons require re-ingesting; see README.
    """
    embedder = make_embedding_client(
        provider=settings.embedding_provider,
        model_name=settings.embedding_model,
        api_key=settings.openai_api_key,
        base_url=settings.llm_base_url,
        openai_model=settings.openai_embedding_model,
    )
    store = VectorStore.load(index_dir, embedding_model=_embedding_model_name())
    print(f"\nIndex: {len(store)} chunks | embedding model: {_embedding_model_name()}")
    print(f"Golden set: {len(golden)} questions\n")
    print(f"{'top_k':>6} | {'hit_rate':>8} | {'recall':>8}")
    print("-" * 30)
    for k in top_k_options:
        retriever = Retriever(store, embedder, top_k=k)
        m = retrieval_metrics(retriever, golden, k_values=(k,))[k]
        print(f"{k:>6} | {m['hit_rate']:>8.2%} | {m['recall']:>8.2%}")


def _embedding_model_name() -> str:
    return settings.embedding_model if settings.embedding_provider == "local" else settings.openai_embedding_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline on the golden set.")
    parser.add_argument("--golden", default="eval/golden_qa.jsonl", help="Path to golden Q&A jsonl")
    parser.add_argument("--judge", action="store_true", help="Run LLM-as-judge faithfulness check (needs API key)")
    parser.add_argument("--top-k", type=int, nargs="+", default=[1, 3, 5], help="k values for the config table")
    args = parser.parse_args()

    settings.validate()
    golden = load_golden(Path(args.golden))

    embedder = make_embedding_client(
        provider=settings.embedding_provider,
        model_name=settings.embedding_model,
        api_key=settings.openai_api_key,
        base_url=settings.llm_base_url,
        openai_model=settings.openai_embedding_model,
    )
    store = VectorStore.load(settings.index_dir, embedding_model=_embedding_model_name())
    print(f"Loaded index: {len(store)} chunks from {settings.index_dir}")

    # 1) Retrieval metrics — no LLM needed.
    print("\n=== Retrieval metrics ===")
    compare_configs(settings.index_dir, golden, top_k_options=tuple(args.top_k))

    # 2) Optional LLM-as-judge faithfulness on generated answers.
    if args.judge:
        print("\n=== Faithfulness (LLM-as-judge) ===")
        retriever = Retriever(store, embedder, top_k=settings.top_k)
        supported, total = 0, 0
        for item in golden:
            retrieved = retriever.retrieve(item["question"])
            result = answer_question(item["question"], retrieved, settings)
            sources_text = "\n\n".join(r.chunk.text for r in result.sources)
            verdict = judge_faithfulness(item["question"], result.answer, sources_text)
            total += 1
            supported += verdict == "SUPPORTED"
            print(f"[{verdict}] {item['question'][:70]}")
        print(f"\nFaithfulness: {supported}/{total} supported ({supported / total:.0%})")


if __name__ == "__main__":
    main()
