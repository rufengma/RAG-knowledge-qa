"""Central configuration. Everything is overridable via environment variables
(and a local `.env` file, see `.env.example`)."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()  # no-op if there is no .env file


def _getenv(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _getint(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"Env var {name} must be an int, got {os.environ.get(name)!r}") from exc


def _getfloat(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"Env var {name} must be a float, got {os.environ.get(name)!r}") from exc


def _getbool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    if raw.strip().lower() in ("1", "true", "yes", "y"):
        return True
    if raw.strip().lower() in ("0", "false", "no", "n"):
        return False
    raise ValueError(f"Env var {name} must be a bool, got {raw!r}")


@dataclass(frozen=True)
class Settings:
    # --- LLM (OpenAI-compatible) ---
    openai_api_key: str = _getenv("OPENAI_API_KEY", "")
    llm_base_url: str = _getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    llm_model: str = _getenv("LLM_MODEL", "gpt-4o-mini")

    # --- Embeddings ---
    embedding_provider: str = _getenv("EMBEDDING_PROVIDER", "local")  # "local" | "openai"
    embedding_model: str = _getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    openai_embedding_model: str = _getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    # --- Ingestion / chunking ---
    chunk_size: int = _getint("CHUNK_SIZE", 500)          # characters per chunk
    chunk_overlap: int = _getint("CHUNK_OVERLAP", 50)     # characters of overlap
    chunk_strategy: str = _getenv("CHUNK_STRATEGY", "fixed")  # "fixed" | "paragraph"

    # --- Retrieval ---
    top_k: int = _getint("TOP_K", 5)
    retrieval_strategy: str = _getenv("RETRIEVAL_STRATEGY", "dense")  # "dense" | "bm25" | "hybrid"
    hybrid_alpha: float = _getfloat("HYBRID_ALPHA", 0.4)  # dense weight in hybrid fusion

    # --- Reranking (cross-encoder, off by default) ---
    rerank: bool = _getbool("RERANK", False)  # rerank the candidate pool with a cross-encoder
    rerank_model: str = _getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L6-v2")
    rerank_candidates: int = _getint("RERANK_CANDIDATES", 20)  # first-stage pool size to rerank

    # --- Paths ---
    index_dir: str = _getenv("INDEX_DIR", "data/index")

    def validate(self) -> None:
        if self.chunk_strategy not in ("fixed", "paragraph"):
            raise ValueError(f"CHUNK_STRATEGY must be 'fixed' or 'paragraph', got {self.chunk_strategy!r}")
        if self.embedding_provider not in ("local", "openai"):
            raise ValueError(f"EMBEDDING_PROVIDER must be 'local' or 'openai', got {self.embedding_provider!r}")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        if self.retrieval_strategy not in ("dense", "bm25", "hybrid"):
            raise ValueError(
                "RETRIEVAL_STRATEGY must be 'dense', 'bm25' or 'hybrid', "
                f"got {self.retrieval_strategy!r}"
            )
        if not 0.0 <= self.hybrid_alpha <= 1.0:
            raise ValueError(f"HYBRID_ALPHA must be in [0, 1], got {self.hybrid_alpha!r}")
        if self.rerank_candidates < 1:
            raise ValueError(f"RERANK_CANDIDATES must be >= 1, got {self.rerank_candidates!r}")


settings = Settings()
