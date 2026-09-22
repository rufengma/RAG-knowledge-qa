# RAG Knowledge-Base Q&A

A clean, minimal Retrieval-Augmented Generation pipeline: ingest your documents
(PDF/Markdown/TXT), index them with FAISS, and ask questions with cited answers.
Built as a portfolio project — readable code, real evals, no magic.

## Architecture

```
data/sample_docs/ ──► ingest.py ──► chunk (fixed | paragraph)
                                         │
                                         ▼
                              embeddings.py (sentence-transformers | OpenAI)
                                         │
                                         ▼
                              vectorstore.py (FAISS, persisted to data/index/)
                                         │
                              ┌──────────┴──────────┐
                              ▼                     ▼
                      retriever.py               eval.py
                    (top-k + scores)      (hit_rate@k, recall@k, LLM judge)
                              │
                              ▼
                        qa.py ──► answer with [1][2] citations
                              (refuses with "I don't know" when nothing relevant)
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Build the index from the sample docs (no API key needed)
python cli.py ingest data/sample_docs

# 2. Inspect retrieval (no API key needed)
python cli.py retrieve "What is RAG?"

# 3. Ask a question (needs OPENAI_API_KEY in .env)
cp .env.example .env   # then add your key
python cli.py ask "What is RAG?"

# 4. Run the eval suite (retrieval metrics need no API key)
python -m src.eval

# 5. Launch the web UI
streamlit run app.py
```

## Configuration

All settings live in `src/config.py` and are overridable via environment
variables or a `.env` file (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | Required for `ask`, the UI, and `--judge`. Not needed for ingest/retrieval/eval. |
| `LLM_BASE_URL` | `https://api.openai.com/v1` | Any OpenAI-compatible endpoint (Ollama, vLLM, Together…) |
| `LLM_MODEL` | `gpt-4o-mini` | Chat model for answer generation |
| `EMBEDDING_PROVIDER` | `local` | `local` (sentence-transformers) or `openai` |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local embedding model |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Used when `EMBEDDING_PROVIDER=openai` |
| `CHUNK_SIZE` | `500` | Characters per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between consecutive chunks |
| `CHUNK_STRATEGY` | `fixed` | `fixed` or `paragraph` |
| `TOP_K` | `5` | Retrieved chunks per query |

Changing the embedding model invalidates the index — the loader checks the
stored model name and tells you to re-run ingest.

## Evaluation

`eval/golden_qa.jsonl` holds 10 golden Q&A pairs over the sample docs, each
labeled with the source file(s) a good retriever should surface.

```bash
python -m src.eval                    # retrieval metrics only
python -m src.eval --judge             # + LLM-as-judge faithfulness (needs API key)
python -m src.eval --top-k 1 3 5 10   # sweep retrieval depth
```

Retrieval metrics (no LLM needed):

- **hit_rate@k** — fraction of questions with ≥1 expected doc in top-k
- **recall@k** — fraction of expected docs found in top-k

To compare chunking strategies: change `CHUNK_STRATEGY`/`CHUNK_SIZE` in `.env`,
re-run `python cli.py ingest data/sample_docs`, then `python -m src.eval`.
Record the table in `eval/results.md` to track regressions.

## Project layout

```
├── app.py              # Streamlit UI: upload docs, chat with citations
├── cli.py              # CLI: ingest | ask | retrieve
├── requirements.txt
├── .env.example
├── src/
│   ├── config.py       # env-based settings
│   ├── ingest.py       # load pdf/md/txt + chunking (fixed, paragraph)
│   ├── embeddings.py   # local (sentence-transformers) / OpenAI embeddings
│   ├── vectorstore.py   # FAISS wrapper + disk persistence
│   ├── retriever.py    # top-k retrieval with cosine scores
│   ├── qa.py           # RAG chain: cited answers + "I don't know" fallback
│   └── eval.py         # golden-set eval: retrieval metrics + LLM judge
├── data/
│   ├── sample_docs/    # 4 short markdown docs so the demo works out of the box
│   └── index/          # built FAISS index (gitignored)
└── eval/
    ├── golden_qa.jsonl # 10 golden Q&A pairs
    └── results.md      # (you) paste eval tables here over time
```

## Roadmap

- [ ] Hybrid retrieval: BM25 + dense with reciprocal rank fusion
- [ ] Reranker stage (cross-encoder) before generation
- [ ] Semantic chunking strategy
- [ ] Multi-query / HyDE query expansion
- [ ] Dockerfile + one-command demo
- [ ] Persisted eval history with charts in `eval/results.md`
