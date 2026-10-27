"""Streamlit front end for the PDF RAG pipeline in data/pdf.

Run locally:   streamlit run app.py
Deployed:      Streamlit Community Cloud, see the "Hosting" section of README.md.

The index is rebuilt from data/pdf at boot (two small PDFs, five chunks), so the app
needs no database, no volume and no persistence between restarts.
"""

from __future__ import annotations

import hmac
import importlib
import os
import platform
import sys
import tempfile
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from rag.retrieval import (  # noqa: E402
    EmbeddingManager,
    RAGRetriever,
    VectorStore,
    answer_question,
    build_llm,
    load_env,
    load_pdfs,
    split_documents,
)

PDF_DIR = ROOT / "data" / "pdf"
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 4
# Cosine floor. Below this the chunks are unrelated boilerplate, and letting the LLM
# answer from them is how a RAG app starts inventing things.
MIN_SCORE = 0.2
# Bump this whenever app.py changes: a deployed copy can then be identified at a
# glance, which is the difference between "my fix did not work" and "my fix is not
# deployed yet".
APP_REVISION = "2026-10-27-stdlib-fallback"
# A public URL spends the owner's Groq credits, one call per question.
MAX_QUESTIONS_PER_SESSION = 25

# Shown in the diagnostics panel. A deployed container says "ModuleNotFoundError ...
# redacted" when something is missing, which tells you nothing; this names it.
DIAGNOSTIC_MODULES = [
    "streamlit",
    "numpy",
    "torch",
    "sentence_transformers",
    "chromadb",
    "langchain",
    "langchain_core",
    "langchain_community",
    "langchain_groq",
    "groq",
    "pypdf",
    "dotenv",
]


def show_diagnostics(expanded: bool = False) -> None:
    """Report what is actually importable on this host, with versions."""
    with st.expander("Environment diagnostics (what is installed here)", expanded=expanded):
        st.write(f"App revision: {APP_REVISION}")
        st.write(f"Python {sys.version.split()[0]} on {platform.system()} {platform.machine()}")
        st.write(f"Working directory: {os.getcwd()}")

        try:
            import langchain_groq  # noqa: F401

            st.write("Groq client: langchain-groq")
        except Exception:
            st.write("Groq client: stdlib fallback (langchain-groq is NOT installed)")

        rows = []
        for name in DIAGNOSTIC_MODULES:
            try:
                module = importlib.import_module(name)
                version = getattr(module, "__version__", None) or "installed"
                rows.append({"module": name, "status": "ok", "version": str(version)})
            except Exception as exc:
                rows.append({"module": name, "status": f"MISSING ({type(exc).__name__})", "version": str(exc)[:80]})
        st.table(rows)

        try:
            import torch

            st.write(f"torch build: {torch.__version__}  (a '+cpu' suffix means the CPU wheel was installed)")
        except Exception:
            pass


def get_secret(name: str, default: str | None = None) -> str | None:
    """Streamlit secrets first (the deployed path), then the environment."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name, default)


def require_password() -> None:
    """Stop the app behind a password when APP_PASSWORD is configured.

    A public URL spends the owner's Groq credits, so this is the cheap way to keep
    it to people you hand the password to. When APP_PASSWORD is unset the app stays
    open, but it says so on screen rather than being quietly public.
    """
    expected = get_secret("APP_PASSWORD")
    if not expected:
        st.warning(
            "No APP_PASSWORD secret is set, so anyone with this URL can use the app and "
            "spend the Groq credits on the configured key. Add APP_PASSWORD in "
            "Settings > Secrets to require a password."
        )
        return
    if st.session_state.get("authenticated"):
        return

    with st.form("password_gate"):
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Enter")

    if submitted:
        # Strip both sides: a trailing space or newline inside the quotes of the
        # TOML secret is invisible in the secrets editor but changes the value.
        if hmac.compare_digest(password.strip(), str(expected).strip()):
            st.session_state["authenticated"] = True
            return
        st.error(
            "Wrong password. It is compared exactly, so check the APP_PASSWORD secret "
            "for a stray space, a quote character, or a capital letter: "
            f"the configured value is {len(str(expected).strip())} characters."
        )

    st.caption("This app is password protected.")
    st.stop()


@st.cache_resource(show_spinner="Loading the embedding model and indexing the PDFs...")
def load_pipeline():
    """Build the model + index once per container, not once per interaction."""
    load_env()
    embedder = EmbeddingManager(MODEL_NAME)
    # A temp dir keeps Chroma from writing into the repo; the index is rebuilt at boot.
    store = VectorStore(
        collection_name="pdf_documents_app",
        persist_directory=tempfile.mkdtemp(prefix="rag-streamlit-"),
    )
    chunks = split_documents(load_pdfs(PDF_DIR))
    if not chunks:
        raise RuntimeError(f"No PDF content found in {PDF_DIR}")
    store.add_documents(chunks, embedder.generate_embeddings([c.page_content for c in chunks]))
    return embedder, store, chunks


def main() -> None:
    st.set_page_config(page_title="Ask the PDFs", page_icon="📄", layout="centered")
    st.title("Ask the PDFs")
    st.caption(
        "Answers come only from the documents in data/pdf. "
        "If the answer is not in them, the app says so instead of guessing."
    )

    # Load the repo-root .env first: locally that is where GROQ_API_KEY lives, while
    # Streamlit Community Cloud supplies it through st.secrets. Without this the key
    # check below fails on a local run with a perfectly good .env sitting right there.
    load_env()

    # Gate first: an unauthenticated visitor should not even load the model.
    require_password()

    if not get_secret("GROQ_API_KEY"):
        st.error(
            "GROQ_API_KEY is not set. On Streamlit Cloud add it under Settings > Secrets, "
            "locally put it in the repo's .env file."
        )
        st.stop()

    try:
        embedder, store, chunks = load_pipeline()
    except Exception as exc:  # surface setup failures instead of a blank page
        st.error(f"Could not build the index: {type(exc).__name__}: {exc}")
        st.stop()

    sources = sorted({c.metadata.get("source_file", "?") for c in chunks})
    st.caption(f"Indexed {len(chunks)} chunks from: {', '.join(sources)}")

    asked = st.session_state.setdefault("asked", 0)

    with st.form("ask"):
        question = st.text_input("Your question", placeholder="What is Java used for?")
        submitted = st.form_submit_button("Ask")

    if submitted and question.strip():
        if asked >= MAX_QUESTIONS_PER_SESSION:
            st.warning(
                f"This demo allows {MAX_QUESTIONS_PER_SESSION} questions per session. "
                "Reload the page to start a new session."
            )
            st.stop()

        st.session_state["asked"] = asked + 1
        retriever = RAGRetriever(store, embedder)

        try:
            llm = build_llm(api_key=get_secret("GROQ_API_KEY"))
            with st.spinner("Searching the PDFs..."):
                hits = retriever.retrieve(question, top_k=TOP_K, score_threshold=MIN_SCORE)
                answer = answer_question(question, retriever, llm, top_k=TOP_K, min_score=MIN_SCORE)
        except Exception as exc:
            # Streamlit redacts deployed tracebacks, so name the failure in the UI.
            st.error(f"Could not answer: {type(exc).__name__}: {exc}")
            st.info(
                "A module is probably missing from the deployed environment. Open the "
                "diagnostics panel below; fix it by installing the package or by making "
                "sure requirements.txt lists it, then reboot the app."
            )
            show_diagnostics(expanded=True)
            st.stop()

        if not hits:
            st.warning(
                "No relevant context found in these PDFs for that question, so no answer "
                "was generated. Try rephrasing using words that appear in the documents."
            )
        else:
            st.markdown("### Answer")
            st.write(answer)

            st.markdown("### Sources")
            for hit in hits:
                meta = hit["metadata"]
                st.markdown(
                    f"- **{meta.get('source_file', '?')}** page {meta.get('page', '?')} "
                    f"— cosine {hit['similarity_score']:.3f}"
                )
                st.caption(hit["content"][:300] + "...")

    st.divider()
    st.caption(
        f"Questions this session: {st.session_state.get('asked', 0)}/{MAX_QUESTIONS_PER_SESSION}. "
        f"Embedding model: {MODEL_NAME}. Retrieved context only; no external knowledge."
    )
    show_diagnostics()


if __name__ == "__main__":
    main()
