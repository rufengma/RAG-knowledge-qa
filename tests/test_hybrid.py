"""Tests for the in-house BM25 sparse retriever (src/hybrid.py).

Pure offline unit tests: no embeddings, no FAISS, no network.
"""

from __future__ import annotations

import math

import pytest

from src.hybrid import BM25Index, tokenize
from src.ingest import Chunk


def _chunks(*texts: str) -> list[Chunk]:
    return [Chunk(text=t, metadata={"source": f"doc{i}.txt"}) for i, t in enumerate(texts)]


class TestTokenize:
    def test_lowercases(self):
        assert tokenize("Hello WORLD") == ["hello", "world"]

    def test_keeps_acronyms_as_single_terms(self):
        # The whole point of BM25 here: acronyms must survive as terms.
        assert tokenize("HNSW index") == ["hnsw", "index"]

    def test_strips_punctuation_and_splits_on_symbols(self):
        assert tokenize("hello, world! (v2.0)") == ["hello", "world", "v2", "0"]

    def test_empty_string(self):
        assert tokenize("") == []


class TestBM25Index:
    def test_empty_corpus(self):
        idx = BM25Index([])
        assert len(idx) == 0
        assert idx.avgdl == 0.0
        assert idx.score("anything") == []

    def test_empty_query_returns_no_results(self):
        idx = BM25Index(_chunks("some text here"))
        assert idx.score("") == []
        assert idx.score("   !!!") == []

    def test_no_term_overlap_returns_no_results(self):
        idx = BM25Index(_chunks("cats and dogs", "birds and fish"))
        assert idx.score("quantum computing") == []

    def test_matching_chunk_ranks_first(self):
        idx = BM25Index(_chunks("the weather is sunny", "HNSW indexing explained", "cooking recipes"))
        results = idx.score("HNSW")
        assert results[0][0] == 1

    def test_scores_sorted_best_first(self):
        idx = BM25Index(_chunks("apple apple apple", "apple pie", "banana split"))
        results = idx.score("apple")
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)
        assert len(results) == 2  # banana chunk has no overlap

    def test_repeated_term_boosts_but_saturates(self):
        idx = BM25Index(_chunks("apple", "apple apple apple apple apple"))
        one, five = idx.score("apple")[1][1], idx.score("apple")[0][1]
        assert five > one
        # BM25 term-frequency saturation: 5x occurrences must NOT give 5x score.
        assert five < 5 * one

    def test_rare_term_outweighs_common_term(self):
        idx = BM25Index(_chunks("common common common", "common rareterm", "common"))
        results = dict(idx.score("common rareterm"))
        assert results[1] > results[0]

    def test_long_document_penalized_by_length_norm(self):
        filler = " ".join(["filler"] * 50)
        idx = BM25Index(_chunks(f"target {filler}", "target"))
        results = dict(idx.score("target"))
        assert results[1] > results[0]

    def test_custom_k1_b_accepted(self):
        idx = BM25Index(_chunks("hello world"), k1=2.0, b=0.5)
        assert idx.k1 == 2.0 and idx.b == 0.5
        assert idx.score("hello")

    def test_idf_non_negative(self):
        idx = BM25Index(_chunks("shared text", "shared words"))
        assert all(v >= 0 for v in idx._idf.values())
        # A term in every doc still gets a positive (smoothed) IDF.
        assert idx._idf["shared"] == pytest.approx(math.log(1 + (2 - 2 + 0.5) / (2 + 0.5)))

    def test_query_term_frequency_does_not_double_count(self):
        # Query "apple apple apple" must score the same as "apple": BM25
        # iterates over distinct query terms, qtf is only used for membership.
        idx = BM25Index(_chunks("apple pie", "banana split"))
        assert idx.score("apple") == idx.score("apple apple apple")
