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

## Notes

-
