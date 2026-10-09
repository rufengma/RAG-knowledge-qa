"""The RAG chain: retrieve -> build a cited prompt -> call the LLM.

Honesty rules (enforced in the system prompt AND in code):
  - If no chunk scores above the relevance threshold, answer "I don't know"
    instead of hallucinating.
  - Every factual claim should be traceable to a numbered source [1], [2], ...
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone

from .config import Settings
from .retriever import RetrievedChunk
from .usage import UsageRecord, append_usage

# Chunks scoring below this cosine similarity are treated as irrelevant.
# MiniLM cosine scores for on-topic chunks are usually well above 0.35.
RELEVANCE_THRESHOLD = 0.25

SYSTEM_PROMPT = """\
You answer questions using ONLY the numbered sources below.
- Cite every factual claim with the source number, e.g. "Dense retrieval uses embeddings [1]."
- If the sources do not contain the answer, say exactly: "I don't know based on the provided documents."
- Be concise. Do not invent facts beyond the sources."""


@dataclass
class QAResult:
    question: str
    answer: str
    sources: list[RetrievedChunk]
    grounded: bool  # False when we refused for lack of evidence


def build_prompt(question: str, retrieved: list[RetrievedChunk]) -> str:
    sources = "\n\n".join(
        f"[{i + 1}] ({r.chunk.source}, {r.chunk.location})\n{r.chunk.text}"
        for i, r in enumerate(retrieved)
    )
    return f"{SYSTEM_PROMPT}\n\nSources:\n{sources}\n\nQuestion: {question}\nAnswer:"


def _maybe_log_usage(
    settings: Settings,
    question: str,
    result: QAResult,
    latency_ms: float,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    total_tokens: int | None,
) -> None:
    """Append one JSONL usage record when USAGE_LOG is set; no-op otherwise."""
    if not settings.usage_log:
        return
    append_usage(
        settings.usage_log,
        UsageRecord(
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            question=question[:200],
            model=settings.llm_model,
            grounded=result.grounded,
            num_sources=len(result.sources),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=round(latency_ms, 2),
        ),
    )


def answer_question(
    question: str,
    retrieved: list[RetrievedChunk],
    settings: Settings,
) -> QAResult:
    """Run the RAG chain. Requires OPENAI_API_KEY (or another OpenAI-compatible key)."""
    t0 = time.perf_counter()
    relevant = [r for r in retrieved if r.score >= RELEVANCE_THRESHOLD]

    if not relevant:
        result = QAResult(
            question=question,
            answer="I don't know based on the provided documents.",
            sources=[],
            grounded=False,
        )
        _maybe_log_usage(
            settings, question, result, (time.perf_counter() - t0) * 1000,
            prompt_tokens=None, completion_tokens=None, total_tokens=None,
        )
        return result

    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is not set — copy .env.example to .env and add your key.")

    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key, base_url=settings.llm_base_url)
    prompt = build_prompt(question, relevant)
    resp = client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    usage = getattr(resp, "usage", None)  # some compatible endpoints omit it
    result = QAResult(
        question=question,
        answer=resp.choices[0].message.content.strip(),
        sources=relevant,
        grounded=True,
    )
    _maybe_log_usage(
        settings, question, result, (time.perf_counter() - t0) * 1000,
        prompt_tokens=getattr(usage, "prompt_tokens", None),
        completion_tokens=getattr(usage, "completion_tokens", None),
        total_tokens=getattr(usage, "total_tokens", None),
    )
    return result


def format_result(result: QAResult) -> str:
    lines = [result.answer, ""]
    if result.sources:
        lines.append("Sources:")
        for i, r in enumerate(result.sources):
            lines.append(f"  [{i + 1}] {r.chunk.source} ({r.chunk.location}) — score {r.score:.3f}")
    return "\n".join(lines)
