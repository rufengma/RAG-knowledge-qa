"""Top-k retrieval with pluggable strategies: dense, bm25, or hybrid.

  - "dense"  — embed the query, cosine search over the FAISS index (default).
  - "bm25"   — keyword scoring over the chunk texts, no embeddings needed.
  - "hybrid" — weighted-sum fusion of both: each candidate list is
    min-max normalized, then ``score = alpha * dense + (1 - alpha) * bm25``.

Hybrid is the fix for acronym/keyword-heavy queries where pure dense
retrieval misses (see eval/results.md).
"""

from __future__ import annotations

from dataclasses import dataclass

from .embeddings import EmbeddingClient
from .hybrid import BM25Index
from .ingest import Chunk
from .rerank import DEFAULT_RERANK_MODEL, Reranker
from .vectorstore import VectorStore

STRATEGIES = ("dense", "bm25", "hybrid")


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float  # cosine similarity in [-1, 1] for dense; BM25 or fused score otherwise
    rerank_score: float | None = None  # cross-encoder logit, set only when reranked


class Retriever:
    def __init__(
        self,
        store: VectorStore,
        embedder: EmbeddingClient,
        top_k: int = 5,
        strategy: str = "dense",
        alpha: float = 0.4,
        rerank: bool = False,
        rerank_model: str = DEFAULT_RERANK_MODEL,
        rerank_candidates: int = 20,
    ):
        if strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}, got {strategy!r}")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha!r}")
        self.store = store
        self.embedder = embedder
        self.top_k = top_k
        self.strategy = strategy
        self.alpha = alpha
        self.rerank = rerank
        self.rerank_model = rerank_model
        self.rerank_candidates = rerank_candidates
        self._bm25: BM25Index | None = None  # built lazily; only needed for bm25/hybrid
        self._reranker: Reranker | None = None  # loaded lazily; only when reranking

    def _bm25_index(self) -> BM25Index:
        if self._bm25 is None:
            self._bm25 = BM25Index(self.store.chunks)
        return self._bm25

    def retrieve(self, query: str, top_k: int | None = None, strategy: str | None = None,
                 rerank: bool | None = None) -> list[RetrievedChunk]:
        k = top_k if top_k is not None else self.top_k
        strat = strategy or self.strategy
        do_rerank = self.rerank if rerank is None else rerank

        if do_rerank:
            # Fetch a larger candidate pool from the cheap first-stage
            # strategy, then let the cross-encoder reorder it down to k.
            pool_k = max(k * 2, min(self.rerank_candidates, len(self.store)))
            candidates = self._retrieve_first_stage(query, pool_k, strat)
            return self._reranker_lazy().rerank(query, candidates, top_k=k)

        return self._retrieve_first_stage(query, k, strat)

    def _retrieve_first_stage(self, query: str, k: int, strat: str) -> list[RetrievedChunk]:
        if strat == "dense":
            return self._dense(query, k)
        if strat == "bm25":
            return self._sparse(query, k)
        if strat == "hybrid":
            return self._hybrid(query, k)
        raise ValueError(f"strategy must be one of {STRATEGIES}, got {strat!r}")

    def _reranker_lazy(self) -> Reranker:
        if self._reranker is None:
            self._reranker = Reranker(self.rerank_model)
        return self._reranker

    # -- strategies -------------------------------------------------------
    def _dense(self, query: str, k: int) -> list[RetrievedChunk]:
        qvec = self.embedder.embed([query])[0]
        return [RetrievedChunk(chunk=c, score=s) for c, s in self.store.search(qvec, k)]

    def _sparse(self, query: str, k: int) -> list[RetrievedChunk]:
        scored = self._bm25_index().score(query)[:k]
        return [RetrievedChunk(chunk=self.store.chunks[i], score=s) for i, s in scored]

    @staticmethod
    def _minmax(scores: dict[int, float]) -> dict[int, float]:
        if not scores:
            return {}
        lo, hi = min(scores.values()), max(scores.values())
        if hi == lo:
            return {i: 1.0 for i in scores}
        return {i: (s - lo) / (hi - lo) for i, s in scores.items()}

    def _hybrid(self, query: str, k: int) -> list[RetrievedChunk]:
        if len(self.store) == 0:
            return []
        cand_k = min(max(k * 2, 10), len(self.store))
        qvec = self.embedder.embed([query])[0]

        # Map chunks back to their index via object identity.
        index_of = {id(c): i for i, c in enumerate(self.store.chunks)}
        dense = {index_of[id(c)]: s for c, s in self.store.search(qvec, cand_k)}
        sparse = dict(self._bm25_index().score(query)[:cand_k])

        dense_n = self._minmax(dense)
        sparse_n = self._minmax(sparse)
        combined = {
            i: self.alpha * dense_n.get(i, 0.0) + (1 - self.alpha) * sparse_n.get(i, 0.0)
            for i in set(dense_n) | set(sparse_n)
        }
        top = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:k]
        return [RetrievedChunk(chunk=self.store.chunks[i], score=s) for i, s in top]
