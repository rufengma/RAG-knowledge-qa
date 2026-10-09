"""Tests for src/usage.py and the per-query logging in src/qa.py.

All offline: the OpenAI client is stubbed the same way as in test_eval.py.
"""

from __future__ import annotations

import json
import sys
import types

from src.config import Settings
from src.ingest import Chunk
from src.qa import answer_question
from src.retriever import RetrievedChunk
from src.usage import UsageRecord, append_usage


def _record(**kw) -> UsageRecord:
    base = dict(
        timestamp="2026-10-09T00:00:00+00:00",
        question="q",
        model="gpt-4o-mini",
        grounded=True,
        num_sources=2,
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        latency_ms=12.5,
    )
    base.update(kw)
    return UsageRecord(**base)


def _read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _settings(log_path: str) -> Settings:
    # usage_log set, fake key so the LLM path doesn't fail on the key check
    return Settings(openai_api_key="sk-test", usage_log=log_path)


def _chunk(score: float = 0.9) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(text="Dense retrieval uses embeddings.", metadata={"source": "doc.md"}),
        score=score,
    )


def _stub_openai(monkeypatch, with_usage: bool):
    """Install a fake `openai` module returning a canned answer."""
    message = types.SimpleNamespace(content="Dense retrieval uses embeddings [1].")
    choice = types.SimpleNamespace(message=message)
    resp_kwargs = {"choices": [choice]}
    if with_usage:
        resp_kwargs["usage"] = types.SimpleNamespace(
            prompt_tokens=100, completion_tokens=20, total_tokens=120
        )
    completions = types.SimpleNamespace(
        create=lambda **kw: types.SimpleNamespace(**resp_kwargs)
    )
    client_cls = lambda **kw: types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=completions)
    )
    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=client_cls))


def test_append_usage_writes_valid_jsonl(tmp_path):
    log = tmp_path / "nested" / "usage.jsonl"  # parent dir does not exist yet
    append_usage(str(log), _record())
    rows = _read_rows(log)
    assert len(rows) == 1
    row = rows[0]
    assert row["model"] == "gpt-4o-mini"
    assert row["total_tokens"] == 15
    assert row["latency_ms"] == 12.5
    assert row["grounded"] is True


def test_append_usage_appends_lines(tmp_path):
    log = tmp_path / "usage.jsonl"
    append_usage(str(log), _record(question="one"))
    append_usage(str(log), _record(question="two"))
    assert [r["question"] for r in _read_rows(log)] == ["one", "two"]


def test_refusal_path_is_logged(tmp_path):
    log = tmp_path / "usage.jsonl"
    result = answer_question("What is RAG?", [], _settings(str(log)))
    assert result.grounded is False
    rows = _read_rows(log)
    assert len(rows) == 1
    row = rows[0]
    assert row["grounded"] is False
    assert row["num_sources"] == 0
    assert row["prompt_tokens"] is None and row["total_tokens"] is None
    assert row["latency_ms"] >= 0
    assert row["model"] == "gpt-4o-mini"
    assert "timestamp" in row


def test_grounded_path_logs_tokens(monkeypatch, tmp_path):
    _stub_openai(monkeypatch, with_usage=True)
    log = tmp_path / "usage.jsonl"
    result = answer_question("What is dense retrieval?", [_chunk()], _settings(str(log)))
    assert result.grounded is True
    row = _read_rows(log)[0]
    assert (row["prompt_tokens"], row["completion_tokens"], row["total_tokens"]) == (100, 20, 120)
    assert row["num_sources"] == 1
    assert row["question"] == "What is dense retrieval?"


def test_missing_usage_from_provider_logs_null_tokens(monkeypatch, tmp_path):
    _stub_openai(monkeypatch, with_usage=False)
    log = tmp_path / "usage.jsonl"
    answer_question("What is dense retrieval?", [_chunk()], _settings(str(log)))
    row = _read_rows(log)[0]
    assert row["prompt_tokens"] is None
    assert row["total_tokens"] is None


def test_no_log_when_usage_log_is_empty(monkeypatch, tmp_path):
    _stub_openai(monkeypatch, with_usage=True)
    settings = Settings(openai_api_key="sk-test", usage_log="")
    answer_question("What is dense retrieval?", [_chunk()], settings)
    assert list(tmp_path.iterdir()) == []  # nothing was created
