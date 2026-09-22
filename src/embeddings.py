"""Embedding clients.

Two providers behind one tiny interface:
  - "local"  (default): sentence-transformers, no API key, no network needed
             after the model is cached. Great for dev/eval.
  - "openai": OpenAI-compatible embeddings API (`/v1/embeddings`).

Vectors are L2-normalized so cosine similarity == inner product in FAISS.
"""

from __future__ import annotations

import numpy as np


class EmbeddingClient:
    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts -> (n, dim) float32 normalized array."""
        raise NotImplementedError

    @property
    def dim(self) -> int:
        raise NotImplementedError


class LocalEmbeddingClient(EmbeddingClient):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        # `get_embedding_dimension()` is the current API (renamed in st 3.x);
        # fall back to the legacy name for older installs.
        get_dim = getattr(self.model, "get_embedding_dimension", None) or getattr(
            self.model, "get_sentence_embedding_dimension"
        )
        self._dim = get_dim()

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)

    @property
    def dim(self) -> int:
        return self._dim


class OpenAIEmbeddingClient(EmbeddingClient):
    def __init__(self, api_key: str, base_url: str, model: str):
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for the 'openai' embedding provider")
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self._dim: int | None = None

    def embed(self, texts: list[str]) -> np.ndarray:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vecs = vecs / norms
        if self._dim is None:
            self._dim = vecs.shape[1]
        return vecs

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = self.embed(["dimension probe"]).shape[1]
        return self._dim


def make_embedding_client(
    provider: str = "local",
    model_name: str = "all-MiniLM-L6-v2",
    api_key: str = "",
    base_url: str = "https://api.openai.com/v1",
    openai_model: str = "text-embedding-3-small",
) -> EmbeddingClient:
    if provider == "local":
        return LocalEmbeddingClient(model_name)
    if provider == "openai":
        return OpenAIEmbeddingClient(api_key, base_url, openai_model)
    raise ValueError(f"Unknown embedding provider: {provider!r}")
