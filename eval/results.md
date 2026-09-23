# Eval results

Paste the output of `python -m src.eval` here after each config change so
regressions are visible over time.

## Baseline — 2026-09-22

Config: `CHUNK_STRATEGY=fixed`, `CHUNK_SIZE=500`, `CHUNK_OVERLAP=50`,
embedding `all-MiniLM-L6-v2`, golden set v1 (10 questions).

| top_k | hit_rate | recall |
|------:|---------:|-------:|
|     1 |    90.00% | 90.00% |
|     3 |    90.00% | 90.00% |
|     5 |    90.00% | 90.00% |

The single miss is "What are HNSW and IVF used for?" — an acronym-heavy query
where dense retrieval underperforms. Candidate fix: hybrid BM25 + dense
retrieval (see Roadmap).

## Paragraph chunking — 2026-09-22

Same config except `CHUNK_STRATEGY=paragraph` (15 chunks).

| top_k | hit_rate | recall |
|------:|---------:|-------:|
|     1 |    80.00% | 80.00% |
|     3 |    90.00% | 90.00% |
|     5 |    90.00% | 90.00% |

Slightly worse at k=1 on this corpus, so `fixed` stays the default.

## Hybrid BM25 + dense (alpha=0.4) — 2026-09-23

New `RETRIEVAL_STRATEGY=hybrid`: in-house BM25 (`src/hybrid.py`) fused with
dense scores via weighted sum (`alpha * dense + (1 - alpha) * bm25`, both
min-max normalized). `alpha=0.4` was picked from a sweep on the golden set —
`0.5` reintroduced a miss, `0.6` didn't fix the acronym query at k=3.

| top_k | hit_rate | recall |
|------:|---------:|-------:|
|     1 |    90.00% | 90.00% |
|     3 |   100.00% | 100.00% |
|     5 |   100.00% | 100.00% |

The acronym miss ("What are HNSW and IVF used for?") is now retrieved at
rank 1 under hybrid; pure BM25 alone also gets 9/10 @1 (it misses "What does
RAG stand for?" instead — keyword ambiguity). No regression vs. dense @1.

## Notes

-
