"""Tests for the eval harness in src/eval.py.

The metric functions take a duck-typed retriever, so we script a fake one
instead of building a real FAISS index -- these tests stay fast and offline.

Run: pytest tests/ -q   (from the repo root, with the project venv)
"""

import json
import sys
import types

import pytest

from src.eval import judge_faithfulness, load_golden, retrieval_metrics
from src.ingest import Chunk
from src.retriever import RetrievedChunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunk(source: str, text: str = "dummy text") -> Chunk:
    return Chunk(text=text, metadata={"source": source})


class FakeRetriever:
    """Scripted retriever: maps a question to the ordered source filenames it
    'retrieves'. Honors top_k the way a real first-stage retriever would."""

    def __init__(self, script: dict[str, list[str]]):
        self.script = script

    def retrieve(self, question: str, top_k: int = 5, **kwargs) -> list[RetrievedChunk]:
        sources = self.script.get(question, [])[:top_k]
        return [RetrievedChunk(chunk=_chunk(s), score=1.0) for s in sources]


def _golden(tmp_path, rows) -> "Path":
    p = tmp_path / "golden_qa.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# load_golden
# ---------------------------------------------------------------------------


def test_load_golden_parses_jsonl_and_skips_blank_lines(tmp_path):
    p = _golden(
        tmp_path,
        [
            {"question": "q1", "expected_sources": ["a.md"]},
            {"question": "q2", "expected_sources": ["b.md"], "answer": "ref"},
        ],
    )
    # add blank lines between records
    text = p.read_text(encoding="utf-8")
    p.write_text("\n" + text + "\n", encoding="utf-8")

    items = load_golden(p)
    assert len(items) == 2
    assert items[0]["expected_sources"] == ["a.md"]
    assert items[1]["answer"] == "ref"


def test_load_golden_rejects_invalid_json_with_line_number(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"question": "ok"}\n{not json}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        load_golden(p)


def test_load_golden_rejects_empty_file(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="[Nn]o eval items"):
        load_golden(p)


# ---------------------------------------------------------------------------
# retrieval_metrics
# ---------------------------------------------------------------------------


def test_perfect_retrieval_scores_one():
    golden = [
        {"question": "q1", "expected_sources": ["a.md"]},
        {"question": "q2", "expected_sources": ["b.md"]},
    ]
    r = FakeRetriever({"q1": ["a.md", "x.md"], "q2": ["b.md", "y.md"]})
    m = retrieval_metrics(r, golden, k_values=(1, 3))[1]
    assert m["hit_rate"] == 1.0
    assert m["recall"] == 1.0
    assert m["n"] == 2


def test_hit_rate_counts_any_hit_in_top_k():
    golden = [{"question": "q1", "expected_sources": ["want.md"]}]
    r = FakeRetriever({"q1": ["noise1.md", "noise2.md", "want.md"]})
    assert retrieval_metrics(r, golden, k_values=(1,))[1]["hit_rate"] == 0.0
    assert retrieval_metrics(r, golden, k_values=(3,))[3]["hit_rate"] == 1.0


def test_recall_supports_multi_doc_questions():
    golden = [{"question": "q1", "expected_sources": ["a.md", "b.md", "c.md"]}]
    r = FakeRetriever({"q1": ["a.md", "c.md"]})
    m = retrieval_metrics(r, golden, k_values=(5,))[5]
    assert m["hit_rate"] == 1.0  # at least one hit
    assert m["recall"] == pytest.approx(2 / 3)


def test_total_miss_scores_zero():
    golden = [{"question": "q1", "expected_sources": ["want.md"]}]
    r = FakeRetriever({"q1": ["noise.md"]})
    m = retrieval_metrics(r, golden, k_values=(3,))[3]
    assert m["hit_rate"] == 0.0
    assert m["recall"] == 0.0


def test_items_without_expected_sources_are_excluded_from_n():
    golden = [
        {"question": "q1", "expected_sources": ["a.md"]},
        {"question": "q2", "expected_sources": []},  # no ground truth -> skipped
    ]
    r = FakeRetriever({"q1": ["zzz.md"]})
    m = retrieval_metrics(r, golden, k_values=(3,))[3]
    assert m["n"] == 1
    assert m["hit_rate"] == 0.0


def test_all_k_values_computed_in_one_call():
    golden = [{"question": "q1", "expected_sources": ["a.md"]}]
    r = FakeRetriever({"q1": ["noise1.md", "a.md", "noise2.md"]})
    m = retrieval_metrics(r, golden, k_values=(1, 2, 5))
    assert set(m) == {1, 2, 5}
    assert m[1]["hit_rate"] == 0.0
    assert m[2]["hit_rate"] == 1.0
    assert m[5]["recall"] == 1.0


# ---------------------------------------------------------------------------
# judge_faithfulness (verdict parsing, with the OpenAI client stubbed)
# ---------------------------------------------------------------------------


def _stub_openai(monkeypatch, verdict_text: str, seen: dict):
    """Install a fake `openai` module; `seen` captures the create() kwargs."""
    message = types.SimpleNamespace(content=verdict_text)
    choice = types.SimpleNamespace(message=message)
    completions = types.SimpleNamespace(
        create=lambda **kw: seen.update(kw) or types.SimpleNamespace(choices=[choice])
    )
    client_cls = lambda **kw: types.SimpleNamespace(chat=types.SimpleNamespace(completions=completions))
    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=client_cls))


@pytest.fixture
def fake_key(monkeypatch):
    from src.eval import settings

    old = settings.openai_api_key
    object.__setattr__(settings, "openai_api_key", "sk-test")
    yield "sk-test"
    object.__setattr__(settings, "openai_api_key", old)


def test_judge_supported_verdict(monkeypatch, fake_key):
    seen: dict = {}
    _stub_openai(monkeypatch, "SUPPORTED", seen)
    assert judge_faithfulness("q", "a", "sources") == "SUPPORTED"
    assert seen["temperature"] == 0.0  # deterministic judging
    assert "sources" in seen["messages"][0]["content"]


def test_judge_unsupported_verdict(monkeypatch, fake_key):
    seen: dict = {}
    _stub_openai(monkeypatch, "UNSUPPORTED", seen)
    assert judge_faithfulness("q", "a", "sources") == "UNSUPPORTED"


def test_judge_fuzzy_verdict_maps_to_unsupported(monkeypatch, fake_key):
    # any non-SUPPORTED wording (e.g. "the claim is unsupported") -> UNSUPPORTED
    seen: dict = {}
    _stub_openai(monkeypatch, "the answer is unsupported by the sources", seen)
    assert judge_faithfulness("q", "a", "sources") == "UNSUPPORTED"


def test_judge_requires_api_key(monkeypatch):
    from src.eval import settings

    old = settings.openai_api_key
    object.__setattr__(settings, "openai_api_key", "")
    try:
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            judge_faithfulness("q", "a", "sources")
    finally:
        object.__setattr__(settings, "openai_api_key", old)
