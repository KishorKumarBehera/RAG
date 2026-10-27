"""Regression tests for the RAG retrieval path (notebook/pdf_loader.ipynb bugs).

These exercise src/rag/retrieval.py, the importable copy of the notebook pipeline:
the notebook itself is guarded by tests/test_notebook_pdf_loader.py.

Run:  uv run pytest      (or)  .venv/bin/python -m pytest
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
from pypdf import PdfReader, PdfWriter
from sentence_transformers import SentenceTransformer

from rag.retrieval import (
    EmbeddingManager,
    RAGRetriever,
    VectorStore,
    answer_question,
    load_pdfs,
    split_documents,
)

REPO = Path(__file__).resolve().parents[1]
PDF_DIR = REPO / "data" / "pdf"
MODEL_NAME = "all-MiniLM-L6-v2"

# Questions whose answers are inside Java.pdf / Python.pdf.
QUESTIONS = [
    "What is Python?",
    "What is Java?",
    "Who created Python and when?",
    "What is Java used for?",
    "What are the different types of variables in Java?",
    "Why use Python?",
]


class StubLLM:
    """Stands in for ChatGroq: no network, records the prompt it was handed."""

    def __init__(self, reply: str = "stub answer"):
        self.reply = reply
        self.prompts: list[str] = []

    def invoke(self, messages):
        self.prompts.append(messages if isinstance(messages, str) else str(messages))

        class _Response:
            content = self.reply

        return _Response()


@pytest.fixture(scope="session")
def model():
    return SentenceTransformer(MODEL_NAME)


@pytest.fixture(scope="session")
def embedder(model):
    return EmbeddingManager(MODEL_NAME)


@pytest.fixture
def store(tmp_path, embedder):
    vs = VectorStore(persist_directory=str(tmp_path / "vector_store"))
    chunks = split_documents(load_pdfs(PDF_DIR))
    vs.add_documents(chunks, embedder.generate_embeddings([c.page_content for c in chunks]))
    return vs


def test_load_pdfs_does_not_duplicate_multipage_documents(tmp_path):
    """The notebook extended `all_documents` inside the per-page loop, so an
    n-page PDF contributed n x n documents. Build a real 3-page PDF and assert
    one Document per page."""
    writer = PdfWriter()
    source = PdfReader(str(PDF_DIR / "Python.pdf"))
    for _ in range(3):
        writer.add_page(source.pages[0])
    multipage = tmp_path / "multipage.pdf"
    with multipage.open("wb") as fh:
        writer.write(fh)

    documents = load_pdfs(multipage.parent, glob="multipage.pdf")

    assert len(documents) == 3, f"expected 1 document per page, got {len(documents)}"
    assert sorted(d.metadata["page"] for d in documents) == [0, 1, 2]
    assert all(d.metadata["source_file"] == "multipage.pdf" for d in documents)


def test_similarity_score_is_true_cosine(model, store, embedder):
    """The notebook scored chunks as `1.0 - distance` on Chroma's default L2
    space, which equals 2*cos - 1: a chunk at cos 0.78 scored 0.57 and off-topic
    chunks scored negative. Scores must be a real cosine similarity."""
    query = "What is Python?"
    results = RAGRetriever(store, embedder).retrieve(query, top_k=5, score_threshold=-1.0)
    assert results, "retrieval returned nothing even with the threshold disabled"

    query_vec = model.encode([query], normalize_embeddings=True)[0]
    for hit in results:
        chunk_vec = model.encode([hit["content"]], normalize_embeddings=True)[0]
        true_cos = float(np.dot(query_vec, chunk_vec))
        assert hit["similarity_score"] == pytest.approx(true_cos, abs=1e-4), (
            f"score {hit['similarity_score']} != true cosine {true_cos}"
        )
        assert -1.0 <= hit["similarity_score"] <= 1.0


def test_questions_about_the_two_pdfs_retrieve_context(embedder, store):
    """The user-visible symptom: asking about the two PDFs returned no context.
    Every question below is answerable from Java.pdf or Python.pdf."""
    retriever = RAGRetriever(store, embedder)
    for question in QUESTIONS:
        results = retriever.retrieve(question, top_k=3)
        assert results, f"no context retrieved for {question!r}"

        expected = "Java.pdf" if "java" in question.lower() else "Python.pdf"
        assert results[0]["metadata"]["source_file"] == expected, (
            f"{question!r} matched {results[0]['metadata']['source_file']} first, expected {expected}"
        )


def test_answer_question_returns_context_not_a_keyerror(embedder, store):
    """`rag_simple` read doc['context'] (the retriever returns 'content') and
    then response.context (an AIMessage has .content), so it crashed twice."""
    llm = StubLLM()
    answer = answer_question("What is Python?", RAGRetriever(store, embedder), llm)

    assert answer == "stub answer"
    prompt = llm.prompts[0]
    assert "Python is a popular programming language" in prompt, "context never reached the LLM"
    assert "What is Python?" in prompt


def test_answer_question_reports_missing_context_instead_of_crashing(embedder, store):
    llm = StubLLM()
    answer = answer_question("quantum chromodynamics", RAGRetriever(store, embedder), llm, min_score=0.9)
    assert answer == "No relevant context found to answer the question."
    assert llm.prompts == [], "the LLM should not be called without context"


def test_reingesting_does_not_grow_or_duplicate_the_collection(embedder, store):
    """Every notebook re-run re-added the same chunks (the friend's collection
    reached 5295 rows), so top-k filled with duplicates of one chunk."""
    chunks = split_documents(load_pdfs(PDF_DIR))
    before = store.count()

    store.add_documents(chunks, embedder.generate_embeddings([c.page_content for c in chunks]))

    assert store.count() == before, "re-ingesting the same PDFs must not add rows"

    results = RAGRetriever(store, embedder).retrieve("What is Python?", top_k=3, score_threshold=-1.0)
    contents = [r["content"] for r in results]
    assert len(contents) == len(set(contents)), f"top-3 contained duplicates: {contents}"


def test_reingesting_a_modified_pdf_does_not_grow_the_collection(embedder, tmp_path):
    """Ids are position-keyed, not content-keyed: editing a PDF must overwrite its
    chunks in place rather than leaving the old ones behind to crowd out hits."""
    from langchain_core.documents import Document

    def chunk(text: str, index: int) -> Document:
        return Document(
            page_content=text,
            metadata={"source_file": "doc.pdf", "page": 0, "doc_index": index},
        )

    vs = VectorStore(persist_directory=str(tmp_path / "vs"))
    original = [chunk(f"original text {i}", i) for i in range(4)]
    vs.add_documents(original, embedder.generate_embeddings([c.page_content for c in original]))
    assert vs.count() == 4

    edited = [chunk(f"edited text {i}", i) for i in range(4)]
    vs.add_documents(edited, embedder.generate_embeddings([c.page_content for c in edited]))

    assert vs.count() == 4, "editing a document must overwrite its chunks, not append"
    stored = vs.collection.get(include=["documents"])["documents"]
    assert all(text.startswith("edited") for text in stored), stored


def test_collection_starts_empty_and_reset_clears_it(embedder, tmp_path):
    vs = VectorStore(persist_directory=str(tmp_path / "vs"))
    assert vs.count() == 0
    chunks = split_documents(load_pdfs(PDF_DIR))
    vs.add_documents(chunks, embedder.generate_embeddings([c.page_content for c in chunks]))
    assert vs.count() > 0
    vs.reset()
    assert vs.count() == 0
