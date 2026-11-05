"""Tests for HyDE query expansion (src/hyde.py) and its Retriever wiring.

The LLM is stubbed -- these tests stay fast and offline, like the rest of
the suite. The Retriever integration tests monkeypatch the first stage so
no FAISS index or embedding model is needed.

Run: pytest tests/ -q   (from the repo root, with the project venv)
"""

import pytest

from src.config import Settings
from src.hyde import HYDE_PROMPT, HyDEGenerator
from src.ingest import Chunk
from src.retriever import RetrievedChunk, Retriever


# ---------------------------------------------------------------------------
# Stub LLM client (same shape as openai.OpenAI)
# ---------------------------------------------------------------------------


class _StubCompletions:
    def __init__(self, text: str, seen: dict):
        self._text = text
        self._seen = seen

    def create(self, **kwargs):
        self._seen.update(kwargs)
        message = type("Msg", (), {"content": self._text})()
        choice = type("Choice", (), {"message": message})()
        return type("Resp", (), {"choices": [choice]})()


class _StubClient:
    def __init__(self, text: str, seen: dict):
        self.chat = type("Chat", (), {"completions": _StubCompletions(text, seen)})()


def _settings(**overrides) -> Settings:
    base = {"openai_api_key": "sk-test", "llm_model": "gpt-4o-mini", "hyde_model": ""}
    base.update(overrides)
    return Settings(**base)


def _chunk(source: str) -> Chunk:
    return Chunk(text="dummy text", metadata={"source": source})


# ---------------------------------------------------------------------------
# HyDEGenerator
# ---------------------------------------------------------------------------


def test_generate_returns_passage_and_uses_prompt():
    seen: dict = {}
    gen = HyDEGenerator(_settings(), client=_StubClient("Hypothetical passage.", seen))
    assert gen.generate("What is RAG?") == "Hypothetical passage."
    assert "What is RAG?" in seen["messages"][0]["content"]
    assert seen["model"] == "gpt-4o-mini"
    assert seen["temperature"] == 0.2


def test_generate_uses_hyde_model_when_set():
    seen: dict = {}
    gen = HyDEGenerator(_settings(hyde_model="gpt-4o"), client=_StubClient("passage", seen))
    gen.generate("q?")
    assert seen["model"] == "gpt-4o"


def test_generate_rejects_empty_question():
    gen = HyDEGenerator(_settings(), client=_StubClient("passage", {}))
    with pytest.raises(ValueError, match="non-empty"):
        gen.generate("   ")


def test_generate_rejects_empty_llm_reply():
    gen = HyDEGenerator(_settings(), client=_StubClient("  \n ", {}))
    with pytest.raises(ValueError, match="empty hypothetical"):
        gen.generate("What is RAG?")


def test_generate_requires_api_key_when_no_client_injected():
    # No injected client -> lazy build must fail loudly without a key.
    gen = HyDEGenerator(_settings(openai_api_key=""))
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        gen.generate("What is RAG?")


def test_generate_builds_client_from_settings(monkeypatch):
    """Lazy client construction uses settings (key + base_url) -- stub openai."""
    import sys
    import types

    seen: dict = {}
    message = types.SimpleNamespace(content="lazy passage")
    choice = types.SimpleNamespace(message=message)
    completions = types.SimpleNamespace(
        create=lambda **kw: seen.update(kw) or types.SimpleNamespace(choices=[choice])
    )
    created: dict = {}

    def client_cls(**kw):
        created.update(kw)
        return types.SimpleNamespace(chat=types.SimpleNamespace(completions=completions))

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=client_cls))
    gen = HyDEGenerator(_settings(llm_base_url="http://localhost:11434/v1"))
    assert gen.generate("What is RAG?") == "lazy passage"
    assert created["api_key"] == "sk-test"
    assert created["base_url"] == "http://localhost:11434/v1"


def test_hyde_prompt_asks_for_a_passage_not_a_caveat():
    rendered = HYDE_PROMPT.format(question="What is BM25?")
    assert "What is BM25?" in rendered
    assert "hypothetical" in rendered.lower()


# ---------------------------------------------------------------------------
# Retriever wiring
# ---------------------------------------------------------------------------


class _StubHyDE:
    """Scripted HyDE generator: returns a fixed passage, records questions."""

    def __init__(self, passage: str = "hypothetical answer passage"):
        self.passage = passage
        self.seen_questions: list[str] = []

    def generate(self, question: str) -> str:
        self.seen_questions.append(question)
        return self.passage


def _retriever(monkeypatch, hyde=False, generator=None, rerank=False):
    r = Retriever(
        store=object(),
        embedder=object(),
        top_k=3,
        strategy="dense",
        hyde=hyde,
        hyde_generator=generator,
        rerank=rerank,
    )
    seen: dict = {}

    def fake_first_stage(query, k, strat):
        seen["query"] = query
        seen["k"] = k
        seen["strategy"] = strat
        return [RetrievedChunk(chunk=_chunk("a.md"), score=0.9)]

    monkeypatch.setattr(r, "_retrieve_first_stage", fake_first_stage)
    return r, seen


def test_retrieve_without_hyde_uses_raw_query(monkeypatch):
    gen = _StubHyDE()
    r, seen = _retriever(monkeypatch, hyde=False, generator=gen)
    out = r.retrieve("What is RAG?")
    assert seen["query"] == "What is RAG?"
    assert gen.seen_questions == []  # no LLM call at all
    assert r.last_hyde_document is None
    assert len(out) == 1


def test_retrieve_with_hyde_embeds_the_passage_not_the_question(monkeypatch):
    gen = _StubHyDE("RAG combines retrieval with generation...")
    r, seen = _retriever(monkeypatch, hyde=True, generator=gen)
    out = r.retrieve("What is RAG?")
    assert gen.seen_questions == ["What is RAG?"]
    assert seen["query"] == "RAG combines retrieval with generation..."
    assert r.last_hyde_document == "RAG combines retrieval with generation..."
    assert len(out) == 1


def test_retrieve_hyde_can_be_enabled_per_call(monkeypatch):
    gen = _StubHyDE("passage")
    r, seen = _retriever(monkeypatch, hyde=False, generator=gen)
    r.retrieve("What is RAG?", hyde=True)
    assert seen["query"] == "passage"


def test_hyde_without_generator_raises_clear_error():
    with pytest.raises(ValueError, match="hyde_generator"):
        Retriever(store=object(), embedder=object(), hyde=True)


def test_hyde_failure_propagates_instead_of_silently_falling_back(monkeypatch):
    class _FailingHyDE:
        def generate(self, question):
            raise ValueError("LLM unavailable")

    r, _seen = _retriever(monkeypatch, hyde=True, generator=_FailingHyDE())
    with pytest.raises(ValueError, match="LLM unavailable"):
        r.retrieve("What is RAG?")


def test_rerank_still_scores_against_the_original_question(monkeypatch):
    """First stage gets the HyDE passage, but the cross-encoder must judge
    relevance to what the user actually asked."""
    gen = _StubHyDE("hypothetical passage")
    r, seen = _retriever(monkeypatch, hyde=True, generator=gen, rerank=True)
    rerank_seen: dict = {}

    class _FakeReranker:
        def rerank(self, query, candidates, top_k):
            rerank_seen["query"] = query
            return candidates[:top_k]

    monkeypatch.setattr(r, "_reranker_lazy", lambda: _FakeReranker())
    r.store = type("Store", (), {"__len__": lambda self: 5})()
    out = r.retrieve("What is RAG?")
    assert seen["query"] == "hypothetical passage"  # first stage: HyDE
    assert rerank_seen["query"] == "What is RAG?"  # reranker: original question
    assert len(out) == 1
