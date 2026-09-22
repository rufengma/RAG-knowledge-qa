"""Top-k retrieval: embed the query, search the FAISS index."""

from __future__ import annotations

from dataclasses import dataclass

from .embeddings import EmbeddingClient
from .ingest import Chunk
from .vectorstore import VectorStore


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float  # cosine similarity in [-1, 1]


class Retriever:
    def __init__(self, store: VectorStore, embedder: EmbeddingClient, top_k: int = 5):
        self.store = store
        self.embedder = embedder
        self.top_k = top_k

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        k = top_k if top_k is not None else self.top_k
        qvec = self.embedder.embed([query])[0]
        return [RetrievedChunk(chunk=c, score=s) for c, s in self.store.search(qvec, k)]
