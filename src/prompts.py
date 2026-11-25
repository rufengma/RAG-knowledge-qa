"""Prompt construction for the RAG chain.

Kept separate from src/qa.py so the prompt text and its assembly rules can
be reviewed, tested, and tuned without touching the retrieval/LLM-call code.
`src/qa.py` re-exports the names below, so existing imports keep working.
"""

from __future__ import annotations

from .retriever import RetrievedChunk

# How many prior turns are sent back to the LLM as conversation context.
# Only the most recent turns are kept, so a long chat stays cheap and the
# retrieved sources (not stale answers) remain the ground truth.
MAX_HISTORY_TURNS = 4

SYSTEM_PROMPT = """\
You answer questions using ONLY the numbered sources below.
- Cite every factual claim with the source number, e.g. "Dense retrieval uses embeddings [1]."
- If the sources do not contain the answer, say exactly: "I don't know based on the provided documents."
- Be concise. Do not invent facts beyond the sources."""


def build_prompt(
    question: str,
    retrieved: list[RetrievedChunk],
    history: list[tuple[str, str]] | None = None,
) -> str:
    """Assemble the LLM prompt.

    `history` is a list of prior (question, answer) turns, oldest first. Only
    the most recent MAX_HISTORY_TURNS turns are included, and only so the
    model can resolve follow-up references ("it", "that", "more about that");
    factual claims must still come from the numbered sources.
    """
    sources = "\n\n".join(
        f"[{i + 1}] ({r.chunk.source}, {r.chunk.location})\n{r.chunk.text}"
        for i, r in enumerate(retrieved)
    )
    parts = [SYSTEM_PROMPT]
    turns = list(history or [])[-MAX_HISTORY_TURNS:]
    if turns:
        convo = "\n\n".join(f"User: {q}\nAssistant: {a}" for q, a in turns)
        parts.append(
            "Conversation so far — use it only to resolve follow-up references; "
            "every factual claim must still come from the numbered sources below:\n"
            + convo
        )
    parts.append(f"Sources:\n{sources}\n\nQuestion: {question}\nAnswer:")
    return "\n\n".join(parts)
