"""FAISS-backed vector store.

Persists to `index_dir/`:
  - `index.faiss`  — the FAISS index (inner-product over normalized vectors)
  - `chunks.jsonl` — one JSON object per chunk: {"text": ..., "metadata": {...}}
  - `meta.json`    — embedding model + dimension, so a mismatched index fails fast

Because embeddings are L2-normalized, inner-product search == cosine similarity.
"""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from .ingest import Chunk


class VectorStore:
    def __init__(self, dim: int, index_dir: str | Path):
        self.dim = dim
        self.index_dir = Path(index_dir)
        # IndexFlatIP: exact inner-product search. Fine for portfolio scale;
        # swap for IndexIVFFlat/IndexHNSW when you outgrow brute force.
        self.index = faiss.IndexFlatIP(dim)
        self.chunks: list[Chunk] = []

    # -- building ---------------------------------------------------------
    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        if embeddings.shape[1] != self.dim:
            raise ValueError(f"Embedding dim {embeddings.shape[1]} != index dim {self.dim}")
        self.index.add(embeddings.astype(np.float32))
        self.chunks.extend(chunks)

    # -- querying ---------------------------------------------------------
    def search(self, query_vec: np.ndarray, top_k: int) -> list[tuple[Chunk, float]]:
        """Return [(chunk, cosine_similarity), ...] sorted best-first."""
        if self.index.ntotal == 0:
            return []
        q = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
        k = min(top_k, self.index.ntotal)
        scores, ids = self.index.search(q, k)
        return [(self.chunks[i], float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]

    # -- persistence ------------------------------------------------------
    def save(self, embedding_model: str) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_dir / "index.faiss"))
        with open(self.index_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps({"text": c.text, "metadata": c.metadata}, ensure_ascii=False) + "\n")
        (self.index_dir / "meta.json").write_text(
            json.dumps({"embedding_model": embedding_model, "dim": self.dim, "num_chunks": len(self.chunks)}),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, index_dir: str | Path, embedding_model: str) -> "VectorStore":
        index_dir = Path(index_dir)
        meta_path = index_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"No index found in {index_dir}. Run `python cli.py ingest <docs>` first."
            )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta["embedding_model"] != embedding_model:
            raise ValueError(
                f"Index was built with embedding model {meta['embedding_model']!r}, "
                f"but config now uses {embedding_model!r}. Re-run ingest."
            )
        store = cls(dim=meta["dim"], index_dir=index_dir)
        store.index = faiss.read_index(str(index_dir / "index.faiss"))
        with open(index_dir / "chunks.jsonl", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                store.chunks.append(Chunk(text=obj["text"], metadata=obj["metadata"]))
        return store

    def __len__(self) -> int:
        return len(self.chunks)
