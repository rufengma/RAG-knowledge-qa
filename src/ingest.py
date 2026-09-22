"""Document loading + chunking.

Supported inputs: .pdf, .md, .txt (a folder is scanned recursively).

Two chunking strategies:
  - "fixed": sliding window over characters with overlap. Simple, predictable.
  - "paragraph": split on blank lines first, then greedily merge paragraphs up to
    `chunk_size` chars (with `chunk_overlap` chars carried from the previous chunk).
    Respects natural boundaries, usually better for prose/docs.

Every chunk carries metadata: source file, page number (PDF) or section (md/txt),
chunk index, and the strategy used — so citations can point back to the origin.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)

    @property
    def source(self) -> str:
        return self.metadata.get("source", "unknown")

    @property
    def location(self) -> str:
        """Human-readable origin, e.g. 'page 3' or 'section 2'."""
        return self.metadata.get("location", "")


SUPPORTED_SUFFIXES = {".pdf", ".md", ".txt", ".markdown"}


def _read_pdf(path: Path) -> list[tuple[str, str]]:
    """Return [(page_text, location), ...]."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((text, f"page {i + 1}"))
    return pages


def _read_text(path: Path) -> list[tuple[str, str]]:
    return [(path.read_text(encoding="utf-8", errors="ignore").strip(), "section 1")]


def load_documents(folder: str | Path) -> list[Chunk]:
    """Load every supported file under `folder` as one Chunk per page/section.

    (Chunking into smaller pieces happens in `chunk_documents`.)
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Document folder not found: {folder}")

    docs: list[Chunk] = []
    for path in sorted(folder.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        if path.suffix.lower() == ".pdf":
            segments = _read_pdf(path)
        else:
            segments = _read_text(path)
        for text, location in segments:
            if text:
                docs.append(Chunk(text=text, metadata={"source": path.name, "location": location}))
    if not docs:
        raise ValueError(f"No supported documents (.pdf/.md/.txt) found in {folder}")
    return docs


def _chunk_fixed(text: str, chunk_size: int, overlap: int) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return [c.strip() for c in chunks if c.strip()]


def _chunk_paragraph(text: str, chunk_size: int, overlap: int) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for para in paragraphs:
        # +2 accounts for the "\n\n" joiner
        if current and current_len + len(para) + 2 > chunk_size:
            chunks.append("\n\n".join(current))
            # carry overlap chars from the tail of the finished chunk
            tail = chunks[-1][-overlap:] if overlap else ""
            current, current_len = ([tail] if tail else []), len(tail)
        current.append(para)
        current_len += len(para) + 2
    if current:
        chunks.append("\n\n".join(current))
    return [c.strip() for c in chunks if c.strip()]


def chunk_documents(
    docs: Iterable[Chunk],
    strategy: str = "fixed",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    """Split documents into chunks, preserving source metadata + chunk index."""
    if strategy not in ("fixed", "paragraph"):
        raise ValueError(f"Unknown chunking strategy: {strategy!r}")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunker = _chunk_fixed if strategy == "fixed" else _chunk_paragraph
    out: list[Chunk] = []
    for doc in docs:
        pieces = chunker(doc.text, chunk_size, chunk_overlap)
        for i, piece in enumerate(pieces):
            meta = dict(doc.metadata)
            meta.update({"chunk_id": i, "strategy": strategy})
            out.append(Chunk(text=piece, metadata=meta))
    return out
