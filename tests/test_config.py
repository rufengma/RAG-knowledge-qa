"""Tests for src/config.py: env parsing and Settings.validate().

All offline: env vars are monkeypatched, and the module is reloaded to
exercise the module-level `Settings()` construction path too.
"""

from __future__ import annotations

import importlib

import pytest

import src.config as config_mod
from src.config import Settings, _getbool, _getfloat, _getint


@pytest.fixture()
def fresh_config(monkeypatch):
    """Reload src.config with a clean env; restore the module afterwards."""
    for name in (
        "CHUNK_SIZE", "CHUNK_OVERLAP", "CHUNK_STRATEGY", "TOP_K",
        "RETRIEVAL_STRATEGY", "HYBRID_ALPHA", "RERANK", "RERANK_CANDIDATES",
        "USAGE_LOG", "EMBEDDING_PROVIDER",
    ):
        monkeypatch.delenv(name, raising=False)
    yield importlib.reload(config_mod)
    importlib.reload(config_mod)


def test_defaults(fresh_config):
    s = fresh_config.Settings()
    assert s.chunk_size == 500
    assert s.chunk_overlap == 50
    assert s.chunk_strategy == "fixed"
    assert s.top_k == 5
    assert s.retrieval_strategy == "dense"
    assert s.hybrid_alpha == 0.4
    assert s.rerank is False
    assert s.rerank_candidates == 20
    assert s.usage_log == ""
    assert s.embedding_provider == "local"


def test_env_overrides(monkeypatch, fresh_config):
    monkeypatch.setenv("CHUNK_SIZE", "800")
    monkeypatch.setenv("CHUNK_OVERLAP", "100")
    monkeypatch.setenv("RETRIEVAL_STRATEGY", "hybrid")
    monkeypatch.setenv("HYBRID_ALPHA", "0.7")
    monkeypatch.setenv("RERANK", "true")
    monkeypatch.setenv("RERANK_CANDIDATES", "10")
    monkeypatch.setenv("USAGE_LOG", "logs/usage.jsonl")
    s = importlib.reload(fresh_config).Settings()
    assert s.chunk_size == 800
    assert s.chunk_overlap == 100
    assert s.retrieval_strategy == "hybrid"
    assert s.hybrid_alpha == pytest.approx(0.7)
    assert s.rerank is True
    assert s.rerank_candidates == 10
    assert s.usage_log == "logs/usage.jsonl"


def test_getint_rejects_garbage(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "abc")
    with pytest.raises(ValueError, match="CHUNK_SIZE.*must be an int"):
        _getint("CHUNK_SIZE", 500)


def test_getfloat_rejects_garbage(monkeypatch):
    monkeypatch.setenv("HYBRID_ALPHA", "not-a-number")
    with pytest.raises(ValueError, match="HYBRID_ALPHA.*must be a float"):
        _getfloat("HYBRID_ALPHA", 0.4)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("1", True), ("true", True), ("True", True), ("yes", True), ("Y", True),
        ("0", False), ("false", False), ("False", False), ("no", False), ("N", False),
        ("  YES  ", True),
    ],
)
def test_getbool_truthy_falsy(monkeypatch, raw, expected):
    monkeypatch.setenv("RERANK", raw)
    assert _getbool("RERANK", False) is expected


def test_getbool_rejects_garbage(monkeypatch):
    monkeypatch.setenv("RERANK", "maybe")
    with pytest.raises(ValueError, match="RERANK.*must be a bool"):
        _getbool("RERANK", False)


def test_getbool_missing_returns_default(monkeypatch):
    monkeypatch.delenv("RERANK", raising=False)
    assert _getbool("RERANK", True) is True


def test_validate_ok():
    Settings().validate()  # defaults are consistent


def test_settings_is_frozen():
    with pytest.raises(AttributeError):
        Settings().chunk_size = 999  # type: ignore[misc]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"chunk_strategy": "semantic"},
        {"embedding_provider": "cohere"},
        {"chunk_overlap": 500, "chunk_size": 500},  # overlap must be < size
        {"chunk_overlap": 600, "chunk_size": 500},
        {"retrieval_strategy": "keyword"},
        {"hybrid_alpha": -0.1},
        {"hybrid_alpha": 1.1},
        {"rerank_candidates": 0},
        {"rerank_candidates": -5},
    ],
)
def test_validate_rejects_bad_values(kwargs):
    with pytest.raises(ValueError):
        Settings(**kwargs).validate()


def test_validate_boundary_alpha_and_candidates_ok():
    Settings(hybrid_alpha=0.0).validate()
    Settings(hybrid_alpha=1.0).validate()
    Settings(rerank_candidates=1).validate()
