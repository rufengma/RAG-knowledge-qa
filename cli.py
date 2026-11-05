#!/usr/bin/env python3
"""CLI entry point.

    python cli.py ingest data/sample_docs     # build the FAISS index
    python cli.py ask "What is RAG?"           # ask a question (needs LLM key)
    python cli.py retrieve "What is RAG?"     # retrieval only (no LLM needed)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import settings  # noqa: E402
from src.embeddings import make_embedding_client  # noqa: E402
from src.hyde import HyDEGenerator  # noqa: E402
from src.ingest import chunk_documents, load_documents, load_url  # noqa: E402
from src.qa import answer_question, format_result  # noqa: E402
from src.retriever import Retriever  # noqa: E402
from src.vectorstore import VectorStore  # noqa: E402


def embedding_model_name() -> str:
    return settings.embedding_model if settings.embedding_provider == "local" else settings.openai_embedding_model


def make_embedder():
    return make_embedding_client(
        provider=settings.embedding_provider,
        model_name=settings.embedding_model,
        api_key=settings.openai_api_key,
        base_url=settings.llm_base_url,
        openai_model=settings.openai_embedding_model,
    )


def _build_index(docs, label: str) -> None:
    print(f"Loaded {len(docs)} document sections from {label}")
    chunks = chunk_documents(
        docs,
        strategy=settings.chunk_strategy,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    print(f"Chunked into {len(chunks)} chunks (strategy={settings.chunk_strategy}, "
          f"size={settings.chunk_size}, overlap={settings.chunk_overlap})")

    embedder = make_embedder()
    print(f"Embedding with {settings.embedding_provider}:{embedding_model_name()} ...")
    embeddings = embedder.embed([c.text for c in chunks])

    store = VectorStore(dim=embeddings.shape[1], index_dir=settings.index_dir)
    store.add(chunks, embeddings)
    store.save(embedding_model=embedding_model_name())
    print(f"Saved index with {len(store)} chunks to {settings.index_dir}/")


def cmd_ingest(args: argparse.Namespace) -> None:
    settings.validate()
    _build_index(load_documents(args.docs_dir), args.docs_dir)


def cmd_ingest_url(args: argparse.Namespace) -> None:
    settings.validate()
    _build_index([load_url(args.url)], args.url)


def load_retriever(hyde: bool | None = None) -> Retriever:
    settings.validate()
    store = VectorStore.load(settings.index_dir, embedding_model=embedding_model_name())
    use_hyde = settings.hyde if hyde is None else hyde
    return Retriever(
        store,
        make_embedder(),
        top_k=settings.top_k,
        strategy=settings.retrieval_strategy,
        alpha=settings.hybrid_alpha,
        rerank=settings.rerank,
        rerank_model=settings.rerank_model,
        rerank_candidates=settings.rerank_candidates,
        hyde=use_hyde,
        hyde_generator=HyDEGenerator(settings) if use_hyde else None,
    )


def cmd_ask(args: argparse.Namespace) -> None:
    retriever = load_retriever(hyde=args.hyde if args.hyde else None)
    retrieved = retriever.retrieve(args.question, rerank=args.rerank if args.rerank else None)
    result = answer_question(args.question, retrieved, settings)
    print(format_result(result))


def cmd_retrieve(args: argparse.Namespace) -> None:
    retriever = load_retriever(hyde=args.hyde if args.hyde else None)
    strategy = args.strategy or retriever.strategy
    rerank = args.rerank or retriever.rerank
    print(f"strategy={strategy} alpha={retriever.alpha} rerank={rerank} hyde={retriever.hyde}")
    retrieved = retriever.retrieve(args.question, top_k=args.top_k, strategy=strategy, rerank=rerank)
    if retriever.last_hyde_document:
        print(f"HyDE passage: {retriever.last_hyde_document[:300]}")
        print()
    for i, r in enumerate(retrieved):
        score_line = f"score={r.score:.3f}"
        if r.rerank_score is not None:
            score_line += f" rerank_score={r.rerank_score:.3f}"
        print(f"[{i + 1}] {score_line} {r.chunk.source} ({r.chunk.location})")
        print(f"    {r.chunk.text[:200].replace(chr(10), ' ')}...")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG knowledge-base Q&A")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Ingest docs and build the FAISS index")
    p_ingest.add_argument("docs_dir", help="Folder with .pdf/.md/.txt/.markdown/.docx files")

    p_ingest_url = sub.add_parser("ingest-url", help="Fetch one web page and build the FAISS index from it")
    p_ingest_url.add_argument("url", help="http(s) URL of an HTML or plain-text page")

    p_ask = sub.add_parser("ask", help="Ask a question (needs OPENAI_API_KEY)")
    p_ask.add_argument("question", help="The question to answer")
    p_ask.add_argument(
        "--rerank",
        action="store_true",
        help="Rerank the first-stage candidates with a cross-encoder (see RERANK_MODEL)",
    )
    p_ask.add_argument(
        "--hyde",
        action="store_true",
        help="Expand the query with a HyDE hypothetical answer passage (needs OPENAI_API_KEY)",
    )

    p_retrieve = sub.add_parser("retrieve", help="Show top-k retrieved chunks (no LLM needed)")
    p_retrieve.add_argument("question", help="The query")
    p_retrieve.add_argument("--top-k", type=int, default=None)
    p_retrieve.add_argument(
        "--strategy",
        choices=["dense", "bm25", "hybrid"],
        default=None,
        help="Retrieval strategy (default: RETRIEVAL_STRATEGY env, else 'dense')",
    )
    p_retrieve.add_argument(
        "--rerank",
        action="store_true",
        help="Rerank the first-stage candidates with a cross-encoder (see RERANK_MODEL)",
    )
    p_retrieve.add_argument(
        "--hyde",
        action="store_true",
        help="Expand the query with a HyDE hypothetical answer passage (needs OPENAI_API_KEY)",
    )

    args = parser.parse_args()
    {"ingest": cmd_ingest, "ingest-url": cmd_ingest_url, "ask": cmd_ask, "retrieve": cmd_retrieve}[args.command](args)


if __name__ == "__main__":
    main()
