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

    # --- Paths ---
    index_dir: str = _getenv("INDEX_DIR", "data/index")

    def validate(self) -> None:
        if self.chunk_strategy not in ("fixed", "paragraph"):
            raise ValueError(f"CHUNK_STRATEGY must be 'fixed' or 'paragraph', got {self.chunk_strategy!r}")
        if self.embedding_provider not in ("local", "openai"):
            raise ValueError(f"EMBEDDING_PROVIDER must be 'local' or 'openai', got {self.embedding_provider!r}")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")


settings = Settings()
