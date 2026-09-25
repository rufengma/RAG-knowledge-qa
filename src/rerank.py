"""Cross-encoder reranking for the retriever.

A cross-encoder scores (query, chunk) pairs jointly, which is more accurate
than bi-encoder cosine similarity but too slow to run over the whole corpus.
The intended use is second-stage reranking: fetch a larger candidate pool
with the cheap first-stage strategy (dense/bm25/hybrid), then rerank the
pool down to the final top-k.

The default model is `cross-encoder/ms-marco-MiniLM-L6-v2` (~80 MB, runs on
CPU). It is downloaded from the HF Hub on first use and cached locally; the
load is lazy so retrieval paths that don't rerank never pay for it.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .retriever import RetrievedChunk

DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


@contextlib.contextmanager
def _sanitized_no_proxy() -> Iterator[None]:
    """Work around an httpx 0.28 parsing bug with IPv6 no_proxy entries.

    When `no_proxy`/`NO_PROXY` contains entries like `[fd8b:...:1]`, httpx
    raises `InvalidURL: Invalid port: ':1]'` while building its proxy map —
    which breaks the huggingface_hub download path. Temporarily reducing the
    list to plain hostnames keeps the download working; it is restored after.
    """
    saved = (os.environ.get("no_proxy"), os.environ.get("NO_PROXY"))
    os.environ["no_proxy"] = "localhost,127.0.0.1"
    os.environ["NO_PROXY"] = "localhost,127.0.0.1"
    try:
        yield
    finally:
        for name, value in zip(("no_proxy", "NO_PROXY"), saved):
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


class Reranker:
    """Second-stage reranker backed by a sentence-transformers CrossEncoder."""

    def __init__(self, model_name: str = DEFAULT_RERANK_MODEL):
        self.model_name = model_name
        self._model = None  # loaded lazily on first rerank()

    def _load(self):
        if self._model is None:
            with _sanitized_no_proxy():
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_k: int | None = None
    ) -> list[RetrievedChunk]:
        """Score (query, chunk) pairs and return candidates best-first.

        Sets `rerank_score` on each returned candidate. `top_k` trims the
        result; the original `score` field is left untouched so downstream
        code (e.g. the relevance threshold in qa.py) keeps working.
        """
        if not candidates:
            return []
        model = self._load()
        scores = model.predict([(query, c.chunk.text) for c in candidates])
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        out = []
        for cand, s in ranked:
            cand.rerank_score = float(s)
            out.append(cand)
        return out if top_k is None else out[:top_k]
