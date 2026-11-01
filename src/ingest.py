"""Document loading + chunking.

Supported inputs: .pdf, .md, .txt, .docx (a folder is scanned recursively),
plus a single web URL via :func:`load_url` (fetched, cleaned to text, chunked
like any other document — no new dependency, stdlib ``urllib`` only).

Two chunking strategies:
  - "fixed": sliding window over characters with overlap. Simple, predictable.
  - "paragraph": split on blank lines first, then greedily merge paragraphs up to
    `chunk_size` chars (with `chunk_overlap` chars carried from the previous chunk).
    Respects natural boundaries, usually better for prose/docs.

Every chunk carries metadata: source file, page number (PDF) or section (md/txt),
chunk index, and the strategy used — so citations can point back to the origin.
"""

from __future__ import annotations

import html
import re
import urllib.error
import urllib.request
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


SUPPORTED_SUFFIXES = {".pdf", ".md", ".txt", ".markdown", ".docx"}


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


def _read_docx(path: Path) -> list[tuple[str, str]]:
    """Return [(full_text, location)] for a Word document.

    Paragraphs are joined with blank lines so the paragraph chunker keeps
    natural boundaries; table rows are appended as extra paragraphs with
    cells joined by " | " so tabular content stays searchable.
    """
    from docx import Document

    doc = Document(str(path))
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                paragraphs.append(" | ".join(cells))
    text = "\n\n".join(paragraphs)
    return [(text, "section 1")] if text else []


def html_to_text(page: str) -> str:
    """Strip an HTML page down to readable text.

    Drops ``<script>``/``<style>`` blocks entirely, turns block-level closes
    into paragraph breaks (so the paragraph chunker keeps natural boundaries),
    strips the remaining tags, and unescapes entities. The ``<title>`` is kept
    as the first paragraph when present, so a fetched page still cites well.
    """
    title = ""
    m = re.search(r"<title[^>]*>(.*?)</title>", page, re.IGNORECASE | re.DOTALL)
    if m:
        title = re.sub(r"<[^>]+>", "", m.group(1)).strip()
    body = re.sub(r"<title[^>]*>.*?</title>", "", page, flags=re.IGNORECASE | re.DOTALL)
    body = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", body, flags=re.IGNORECASE | re.DOTALL)
    body = re.sub(r"</(p|div|section|article|li|h[1-6]|tr|blockquote)>", "\n\n", body, flags=re.IGNORECASE)
    body = re.sub(r"<br\s*/?>", "\n\n", body, flags=re.IGNORECASE)
    body = re.sub(r"<[^>]+>", "", body)
    lines = [html.unescape(line.strip()) for line in body.splitlines()]
    paragraphs = [line for line in lines if line]
    if title:
        paragraphs.insert(0, html.unescape(title))
    return "\n\n".join(paragraphs)


def fetch_url(url: str, timeout: float = 30.0) -> str:
    """Fetch `url` and return its cleaned text.

    Raises ``ValueError`` on a non-200 status, on a non-text content type, or
    when nothing readable survives cleaning; network errors from ``urllib``
    (``URLError``/``HTTPError``) propagate so callers can tell them apart.
    """
    request = urllib.request.Request(url, headers={"User-Agent": "RAG-knowledge-qa/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        status = getattr(response, "status", 200)
        if status != 200:
            raise ValueError(f"Failed to fetch {url}: HTTP {status}")
        content_type = (response.headers.get_content_type() if response.headers else "text/html")
        raw = response.read()
        charset = response.headers.get_content_charset("utf-8") if response.headers else "utf-8"
        text = raw.decode(charset, errors="ignore")
    if content_type == "text/plain":
        cleaned = text.strip()
    elif content_type in ("text/html", "application/xhtml+xml"):
        cleaned = html_to_text(text)
    else:
        raise ValueError(f"Refusing to ingest {url}: unsupported content type {content_type!r}")
    if not cleaned:
        raise ValueError(f"No readable text found at {url}")
    return cleaned


def load_url(url: str, timeout: float = 30.0) -> Chunk:
    """Fetch one web page as a single document Chunk.

    The URL itself is the chunk source, so citations point back to the page
    rather than to a downloaded file.
    """
    text = fetch_url(url, timeout=timeout)
    return Chunk(text=text, metadata={"source": url, "location": "web page"})


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
        elif path.suffix.lower() == ".docx":
            segments = _read_docx(path)
        else:
            segments = _read_text(path)
        for text, location in segments:
            if text:
                docs.append(Chunk(text=text, metadata={"source": path.name, "location": location}))
    if not docs:
        raise ValueError(f"No supported documents (.pdf/.md/.txt/.docx) found in {folder}")
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
