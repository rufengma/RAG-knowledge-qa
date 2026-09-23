"""BM25 sparse retrieval, implemented in-house (no extra dependency).

Standard BM25 with k1=1.5, b=0.75 and simple lowercase alphanumeric
tokenization. Works well for keyword/acronym-heavy queries where dense
embeddings underperform, and pairs with dense scores via weighted-sum
fusion in `retriever.Retriever` (the "hybrid" strategy).
"""

from __future__ import annotations

import math
import re
from collections import Counter

from .ingest import Chunk

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens. Keeps acronyms (e.g. "HNSW") as terms."""
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """In-memory BM25 index over a fixed list of chunks.

    Build it from the same chunks that back the FAISS index so doc indices
    line up: ``store.chunks[i]`` is the chunk for score entry ``(i, score)``.
    """

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b
        self._doc_tokens = [tokenize(c.text) for c in self.chunks]
        self._doc_lens = [len(t) for t in self._doc_tokens]
        self.avgdl = sum(self._doc_lens) / len(self._doc_lens) if self._doc_lens else 0.0
        df = Counter()
        for toks in self._doc_tokens:
            df.update(set(toks))
        n = len(self.chunks)
        # IDF with the standard +1 smoothing; always non-negative.
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def __len__(self) -> int:
        return len(self.chunks)

    def score(self, query: str) -> list[tuple[int, float]]:
        """Return [(chunk_index, bm25_score), ...] sorted best-first.

        Documents with no query-term overlap are omitted.
        """
        qtf = Counter(tokenize(query))
        if not qtf or not self.chunks:
            return []
        results: list[tuple[int, float]] = []
        for i, toks in enumerate(self._doc_tokens):
            tf = Counter(toks)
            dl = self._doc_lens[i]
            s = 0.0
            for term in qtf:
                if term not in self._idf or term not in tf:
                    continue
                denom = tf[term] + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                s += self._idf[term] * tf[term] * (self.k1 + 1) / denom
            if s > 0:
                results.append((i, s))
        results.sort(key=lambda x: x[1], reverse=True)
        return results
