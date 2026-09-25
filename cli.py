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
from src.ingest import chunk_documents, load_documents  # noqa: E402
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


def cmd_ingest(args: argparse.Namespace) -> None:
    settings.validate()
    docs = load_documents(args.docs_dir)
    print(f"Loaded {len(docs)} document sections from {args.docs_dir}")
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


def load_retriever() -> Retriever:
    settings.validate()
    store = VectorStore.load(settings.index_dir, embedding_model=embedding_model_name())
    return Retriever(
        store,
        make_embedder(),
        top_k=settings.top_k,
        strategy=settings.retrieval_strategy,
        alpha=settings.hybrid_alpha,
        rerank=settings.rerank,
        rerank_model=settings.rerank_model,
        rerank_candidates=settings.rerank_candidates,
    )


def cmd_ask(args: argparse.Namespace) -> None:
    retriever = load_retriever()
    retrieved = retriever.retrieve(args.question, rerank=args.rerank if args.rerank else None)
    result = answer_question(args.question, retrieved, settings)
    print(format_result(result))


def cmd_retrieve(args: argparse.Namespace) -> None:
    retriever = load_retriever()
    strategy = args.strategy or retriever.strategy
    rerank = args.rerank or retriever.rerank
    print(f"strategy={strategy} alpha={retriever.alpha} rerank={rerank}")
    for i, r in enumerate(
        retriever.retrieve(args.question, top_k=args.top_k, strategy=strategy, rerank=rerank)
    ):
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
    p_ingest.add_argument("docs_dir", help="Folder with .pdf/.md/.txt files")

    p_ask = sub.add_parser("ask", help="Ask a question (needs OPENAI_API_KEY)")
    p_ask.add_argument("question", help="The question to answer")
    p_ask.add_argument(
        "--rerank",
        action="store_true",
        help="Rerank the first-stage candidates with a cross-encoder (see RERANK_MODEL)",
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

    args = parser.parse_args()
    {"ingest": cmd_ingest, "ask": cmd_ask, "retrieve": cmd_retrieve}[args.command](args)


if __name__ == "__main__":
    main()
