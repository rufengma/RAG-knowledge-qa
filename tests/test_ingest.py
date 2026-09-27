"""Tests for the chunking strategies in src/ingest.py.

Run: pytest tests/ -q   (from the repo root, with the project venv)
"""

from pathlib import Path

import pytest

from src.ingest import Chunk, chunk_documents, load_documents


def _doc(text: str, **meta) -> Chunk:
    metadata = {"source": "demo.md", "location": "section 1"}
    metadata.update(meta)
    return Chunk(text=text, metadata=metadata)


# ---------------------------------------------------------------------------
# "fixed" strategy
# ---------------------------------------------------------------------------


def test_fixed_short_text_is_one_chunk():
    chunks = chunk_documents([_doc("hello world")], strategy="fixed", chunk_size=500)
    assert len(chunks) == 1
    assert chunks[0].text == "hello world"


def test_fixed_chunk_window_and_overlap():
    text = "x" * 100
    chunks = chunk_documents(
        [_doc(text)], strategy="fixed", chunk_size=30, chunk_overlap=10
    )
    # windows: [0:30], [20:50], [40:70], [60:90], [80:100] -> 5 chunks
    assert len(chunks) == 5
    assert all(len(c.text) <= 30 for c in chunks)
    # consecutive chunks share their overlap region
    assert chunks[0].text.endswith("x" * 10)
    assert chunks[1].text.startswith("x" * 10)


def test_fixed_windows_cover_full_text():
    # with zero overlap the chunks tile the text exactly
    text = "abcdefghij" * 12  # 120 chars
    chunks = chunk_documents(
        [_doc(text)], strategy="fixed", chunk_size=50, chunk_overlap=0
    )
    assert "".join(c.text for c in chunks) == text


def test_fixed_never_yields_empty_chunks():
    chunks = chunk_documents(
        [_doc("   \n\n  abc  \n\n   ")], strategy="fixed", chunk_size=4, chunk_overlap=1
    )
    assert chunks
    assert all(c.text for c in chunks)


def test_fixed_larger_overlap_means_more_chunks():
    text = "y" * 200
    few = chunk_documents([_doc(text)], strategy="fixed", chunk_size=50, chunk_overlap=0)
    many = chunk_documents([_doc(text)], strategy="fixed", chunk_size=50, chunk_overlap=40)
    assert len(many) > len(few)


# ---------------------------------------------------------------------------
# "paragraph" strategy
# ---------------------------------------------------------------------------


def test_paragraph_never_splits_fitting_paragraph():
    paras = ["alpha " * 10, "beta " * 10, "gamma " * 10]
    text = "\n\n".join(paras)
    chunks = chunk_documents(
        [_doc(text)], strategy="paragraph", chunk_size=500, chunk_overlap=0
    )
    assert len(chunks) == 1
    for para in paras:
        assert para.strip() in chunks[0].text


def test_paragraph_starts_new_chunk_when_full():
    big = "word " * 200  # ~1000 chars: too big for two of them in 600
    chunks = chunk_documents(
        [_doc(f"{big}\n\n{big}")], strategy="paragraph", chunk_size=600, chunk_overlap=0
    )
    # one chunk per paragraph; the paragraph chunker never splits inside a
    # paragraph, so an oversized paragraph stays whole in its own chunk
    assert len(chunks) == 2
    assert all(c.text.count("word") == 200 for c in chunks)


def test_paragraph_merges_small_paragraphs_up_to_limit():
    paras = [f"para-{i}" for i in range(10)]
    chunks = chunk_documents(
        [_doc("\n\n".join(paras))],
        strategy="paragraph",
        chunk_size=10_000,
        chunk_overlap=0,
    )
    assert len(chunks) == 1
    assert all(p in chunks[0].text for p in paras)


def test_paragraph_overlap_carries_tail_of_previous_chunk():
    first = "A" * 300
    second = "B" * 300
    third = "C" * 300
    chunks = chunk_documents(
        [_doc("\n\n".join([first, second, third]))],
        strategy="paragraph",
        chunk_size=400,
        chunk_overlap=50,
    )
    assert len(chunks) >= 2
    # tail of chunk 0 should appear at the start of chunk 1
    assert chunks[1].text.startswith(chunks[0].text[-50:])


def test_paragraph_handles_single_line_text():
    chunks = chunk_documents(
        [_doc("one line, no blank lines", source="note.txt")],
        strategy="paragraph",
        chunk_size=100,
        chunk_overlap=10,
    )
    assert len(chunks) == 1
    assert chunks[0].text == "one line, no blank lines"


# ---------------------------------------------------------------------------
# metadata + validation
# ---------------------------------------------------------------------------


def test_chunk_metadata_preserves_source_and_index():
    docs = [_doc("a" * 60, source="a.md", location="section 2")]
    chunks = chunk_documents(docs, strategy="fixed", chunk_size=25, chunk_overlap=0)
    assert len(chunks) == 3
    for i, chunk in enumerate(chunks):
        assert chunk.source == "a.md"
        assert chunk.location == "section 2"
        assert chunk.metadata["chunk_id"] == i
        assert chunk.metadata["strategy"] == "fixed"


def test_strategies_tag_their_chunks():
    text = "hello " * 100
    for strategy in ("fixed", "paragraph"):
        chunks = chunk_documents(
            [_doc(text)], strategy=strategy, chunk_size=60, chunk_overlap=10
        )
        assert chunks
        assert all(c.metadata["strategy"] == strategy for c in chunks)


def test_unknown_strategy_raises():
    with pytest.raises(ValueError, match="Unknown chunking strategy"):
        chunk_documents([_doc("text")], strategy="sliding")


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError, match="chunk_overlap must be smaller"):
        chunk_documents([_doc("text")], strategy="fixed", chunk_size=50, chunk_overlap=50)


def test_empty_documents_yield_no_chunks():
    assert chunk_documents([], strategy="fixed") == []
    assert chunk_documents([_doc("   ")], strategy="paragraph") == []


# ---------------------------------------------------------------------------
# load_documents (disk -> Chunk)
# ---------------------------------------------------------------------------


def test_load_documents_reads_md_and_txt(tmp_path: Path):
    (tmp_path / "a.md").write_text("# Title\n\nSome prose here.")
    (tmp_path / "b.txt").write_text("plain text file")
    (tmp_path / "skip.bin").write_bytes(b"\x00\x01\x02")  # unsupported suffix

    docs = load_documents(tmp_path)
    assert len(docs) == 2
    by_source = {d.source: d for d in docs}
    assert "a.md" in by_source and "b.txt" in by_source
    assert by_source["a.md"].location == "section 1"
    assert "skip.bin" not in by_source


def test_load_documents_missing_folder_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path / "does-not-exist")


def test_load_documents_empty_folder_raises(tmp_path: Path):
    with pytest.raises(ValueError, match="No supported documents"):
        load_documents(tmp_path)
