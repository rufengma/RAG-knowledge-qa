"""The RAG chain: retrieve -> build a cited prompt -> call the LLM.

Honesty rules (enforced in the system prompt AND in code):
  - If no chunk scores above the relevance threshold, answer "I don't know"
    instead of hallucinating.
  - Every factual claim should be traceable to a numbered source [1], [2], ...
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import Settings
from .retriever import RetrievedChunk

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


def answer_question(
    question: str,
    retrieved: list[RetrievedChunk],
    settings: Settings,
) -> QAResult:
    """Run the RAG chain. Requires OPENAI_API_KEY (or another OpenAI-compatible key)."""
    relevant = [r for r in retrieved if r.score >= RELEVANCE_THRESHOLD]

    if not relevant:
        return QAResult(
            question=question,
            answer="I don't know based on the provided documents.",
            sources=[],
            grounded=False,
        )

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
    return QAResult(
        question=question,
        answer=resp.choices[0].message.content.strip(),
        sources=relevant,
        grounded=True,
    )


def format_result(result: QAResult) -> str:
    lines = [result.answer, ""]
    if result.sources:
        lines.append("Sources:")
        for i, r in enumerate(result.sources):
            lines.append(f"  [{i + 1}] {r.chunk.source} ({r.chunk.location}) — score {r.score:.3f}")
    return "\n".join(lines)
