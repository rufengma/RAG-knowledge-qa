"""Tests for multi-turn conversation memory in src/qa.py.

All offline: the OpenAI client is stubbed the same way as in test_usage.py.
"""

from __future__ import annotations

import sys
import types

from src.config import Settings
from src.ingest import Chunk
from src.qa import MAX_HISTORY_TURNS, answer_question, build_prompt
from src.retriever import RetrievedChunk


def _chunk(score: float = 0.9) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(text="Dense retrieval uses embeddings.", metadata={"source": "doc.md"}),
        score=score,
    )


def _settings() -> Settings:
    return Settings(openai_api_key="sk-test")


def _stub_openai(monkeypatch, seen: dict):
    """Install a fake `openai` module that records the create() kwargs."""
    message = types.SimpleNamespace(content="Dense retrieval uses embeddings [1].")
    choice = types.SimpleNamespace(message=message)

    def create(**kw):
        seen.update(kw)
        return types.SimpleNamespace(choices=[choice])

    completions = types.SimpleNamespace(create=create)
    client_cls = lambda **kw: types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=completions)
    )
    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=client_cls))


def test_build_prompt_without_history_has_no_conversation_section():
    prompt = build_prompt("What is dense retrieval?", [_chunk()])
    assert "Conversation so far" not in prompt
    assert "Question: What is dense retrieval?" in prompt
    assert "[1] (doc.md" in prompt


def test_build_prompt_includes_history_turns():
    history = [
        ("What is RAG?", "RAG grounds answers in retrieved docs [1]."),
        ("How does it retrieve?", "It embeds the query and searches FAISS [1]."),
    ]
    prompt = build_prompt("And what about reranking?", [_chunk()], history=history)
    assert "Conversation so far" in prompt
    assert "User: What is RAG?" in prompt
    assert "Assistant: RAG grounds answers in retrieved docs [1]." in prompt
    assert "User: How does it retrieve?" in prompt
    # History is context only — the sources still ground the answer.
    assert "[1] (doc.md" in prompt
    assert "Question: And what about reranking?" in prompt


def test_build_prompt_keeps_only_recent_turns():
    history = [(f"Question {i}?", f"Answer {i}.") for i in range(6)]
    prompt = build_prompt("Follow-up?", [_chunk()], history=history)
    assert "Question 0?" not in prompt
    assert "Question 1?" not in prompt
    assert f"Question {6 - MAX_HISTORY_TURNS}?" in prompt
    assert "Question 5?" in prompt


def test_answer_question_sends_history_to_llm(monkeypatch):
    _stub_openai(monkeypatch, seen := {})
    history = [("What is RAG?", "RAG grounds answers in retrieved docs [1].")]
    result = answer_question(
        "Tell me more about it", [_chunk()], _settings(), history=history
    )
    assert result.grounded is True
    user_prompt = seen["messages"][1]["content"]
    assert "User: What is RAG?" in user_prompt
    assert "Tell me more about it" in user_prompt


def test_answer_question_without_history_unchanged(monkeypatch):
    _stub_openai(monkeypatch, seen := {})
    result = answer_question("What is dense retrieval?", [_chunk()], _settings())
    assert result.grounded is True
    assert "Conversation so far" not in seen["messages"][1]["content"]


def test_refusal_path_ignores_history(monkeypatch):
    _stub_openai(monkeypatch, seen := {})
    history = [("What is RAG?", "RAG grounds answers in retrieved docs [1].")]
    result = answer_question("Anything?", [], _settings(), history=history)
    assert result.grounded is False
    assert seen == {}  # no LLM call at all
