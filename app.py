"""Minimal Streamlit UI: upload docs, build the index, chat with cited sources.

Run:  streamlit run app.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import settings  # noqa: E402
from src.embeddings import make_embedding_client  # noqa: E402
from src.ingest import chunk_documents, load_documents  # noqa: E402
from src.qa import answer_question  # noqa: E402
from src.retriever import Retriever  # noqa: E402
from src.vectorstore import VectorStore  # noqa: E402

st.set_page_config(page_title="RAG Knowledge Base Q&A", layout="centered")
st.title("📚 RAG Knowledge-Base Q&A")


def embedding_model_name() -> str:
    return settings.embedding_model if settings.embedding_provider == "local" else settings.openai_embedding_model


@st.cache_resource
def get_embedder():
    return make_embedding_client(
        provider=settings.embedding_provider,
        model_name=settings.embedding_model,
        api_key=settings.openai_api_key,
        base_url=settings.llm_base_url,
        openai_model=settings.openai_embedding_model,
    )


def index_exists() -> bool:
    return (Path(settings.index_dir) / "meta.json").exists()


# --- Sidebar: ingestion ---------------------------------------------------
with st.sidebar:
    st.header("Knowledge base")
    uploaded = st.file_uploader(
        "Upload .pdf / .md / .txt files",
        type=["pdf", "md", "txt", "markdown"],
        accept_multiple_files=True,
    )
    if st.button("Build index", disabled=not uploaded):
        with tempfile.TemporaryDirectory() as tmp:
            for f in uploaded:
                (Path(tmp) / f.name).write_bytes(f.getbuffer())
            docs = load_documents(tmp)
            chunks = chunk_documents(
                docs,
                strategy=settings.chunk_strategy,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )
            with st.spinner(f"Embedding {len(chunks)} chunks..."):
                embeddings = get_embedder().embed([c.text for c in chunks])
                store = VectorStore(dim=embeddings.shape[1], index_dir=settings.index_dir)
                store.add(chunks, embeddings)
                store.save(embedding_model=embedding_model_name())
        st.success(f"Indexed {len(chunks)} chunks from {len(uploaded)} files.")
        st.cache_resource.clear()

    if index_exists():
        meta = VectorStore.load(settings.index_dir, embedding_model_name())
        st.caption(f"Current index: {len(meta)} chunks")

    if st.session_state.get("history") and st.button("Clear chat history"):
        st.session_state.history = []
        st.rerun()

# --- Main: chat -----------------------------------------------------------
if not settings.openai_api_key:
    st.warning("Set OPENAI_API_KEY in `.env` to enable answers. Retrieval preview still works via CLI.")

# Each entry is (question, raw answer, [(source, location, score), ...]).
# The raw answer (not the formatted sources block) is what gets sent back
# to the LLM as conversation context for follow-up questions.
if "history" not in st.session_state:
    st.session_state.history = []

for q, a, srcs in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        st.markdown(a)
        if srcs:
            st.markdown("Sources:")
            for i, (source, location, score) in enumerate(srcs):
                st.markdown(f"  [{i + 1}] {source} ({location}) — score {score:.3f}")

question = st.chat_input("Ask about your documents...")
if question and index_exists():
    store = VectorStore.load(settings.index_dir, embedding_model_name())
    retriever = Retriever(
        store,
        get_embedder(),
        top_k=settings.top_k,
        strategy=settings.retrieval_strategy,
        alpha=settings.hybrid_alpha,
    )
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Retrieving + generating..."):
            result = answer_question(
                question,
                retriever.retrieve(question),
                settings,
                history=[(q, a) for q, a, _ in st.session_state.history],
            )
        st.markdown(result.answer)
        if result.sources:
            st.markdown("Sources:")
            for i, r in enumerate(result.sources):
                st.markdown(f"  [{i + 1}] {r.chunk.source} ({r.chunk.location}) — score {r.score:.3f}")
    st.session_state.history.append(
        (
            question,
            result.answer,
            [(r.chunk.source, r.chunk.location, r.score) for r in result.sources],
        )
    )
elif question:
    st.error("No index yet — upload documents and click 'Build index' first.")
