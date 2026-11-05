"""HyDE (Hypothetical Document Embeddings) query expansion.

Short questions and long document chunks live in different parts of
embedding space: a question like "What is RAG?" embeds far from the
paragraph that actually explains it. HyDE bridges the gap by asking the
LLM to write a short hypothetical passage that *would* answer the
question, then embedding that passage instead of the raw question --
the hypothetical answer is stylistically much closer to the real chunks.

When enabled (``HYDE=true`` or ``--hyde``), ``Retriever`` generates one
hypothetical document per query and uses it as the first-stage query for
whatever strategy is configured (dense embeds it, BM25 scores its terms,
hybrid fuses both). Reranking, when on, still scores candidates against
the *original* question -- the cross-encoder should judge relevance to
what the user actually asked, not to the model's guess.

Costs one extra LLM call per query, so it stays off by default. Measure
the gain on your golden set with ``python -m src.eval --hyde`` before
turning it on everywhere.
"""

from __future__ import annotations

from .config import Settings

HYDE_PROMPT = """\
Write a short passage (3-5 sentences) that directly answers the question
below, as it might appear in a technical document. Do not add caveats,
do not say you don't know, and do not mention that this is hypothetical --
just write the passage itself.

Question: {question}

Passage:"""


class HyDEGenerator:
    """Generates hypothetical answer passages via an OpenAI-compatible LLM.

    The client is built lazily from ``settings`` so importing this module
    (and running the offline test suite) never needs an API key. A custom
    client can be injected for tests -- it only needs
    ``client.chat.completions.create(model=..., messages=..., temperature=...)``
    returning an object with ``choices[0].message.content``.
    """

    def __init__(self, settings: Settings, client=None):
        self.settings = settings
        self._client = client

    @property
    def model(self) -> str:
        # HYDE_MODEL overrides the answer-generation model when set; a small
        # cheap model is fine here, the passage is only used for retrieval.
        return self.settings.hyde_model or self.settings.llm_model

    def _client_lazy(self):
        if self._client is None:
            if not self.settings.openai_api_key:
                raise ValueError(
                    "OPENAI_API_KEY is not set — copy .env.example to .env "
                    "and add your key (HyDE needs an LLM to write the "
                    "hypothetical document)."
                )
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self.settings.openai_api_key,
                base_url=self.settings.llm_base_url,
            )
        return self._client

    def generate(self, question: str) -> str:
        """Write the hypothetical document answering ``question``."""
        question = question.strip()
        if not question:
            raise ValueError("HyDE needs a non-empty question")
        client = self._client_lazy()
        resp = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": HYDE_PROMPT.format(question=question)}],
            temperature=0.2,
        )
        passage = resp.choices[0].message.content.strip()
        if not passage:
            raise ValueError("HyDE LLM returned an empty hypothetical document")
        return passage
