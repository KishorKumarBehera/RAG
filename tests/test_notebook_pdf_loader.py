"""Executes the code cells of notebook/pdf_loader.ipynb and asserts the pipeline
answers from the two PDFs.

This guards the artifact the friend actually runs: the notebooks are the app, so a
regression in their cells has to fail a test.

The notebook is executed in a subprocess. Two reasons:

* Isolation. Its relative paths ("../data") resolve under a temp dir, so a test run
  cannot touch the repo's real data/vector_store.
* One Chroma client per process. Creating and garbage-collecting several
  ``chromadb.PersistentClient`` objects inside one pytest process makes Chroma's
  shared system tear down underneath the next client ("no such table: max_seq_id"),
  which is a test-harness artifact, not a pipeline bug. A subprocess sidesteps it.

The Groq client is stubbed, so this needs no API key and no network.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "notebook" / "pdf_loader.ipynb"
REAL_PDF_DIR = REPO / "data" / "pdf"
SENTINEL = "No relevant context found to answer the question."
STUB_ANSWER = "Python is a popular programming language."
MARKER = "###NOTEBOOK-RESULT###"

HARNESS = r'''
import json, os, shutil, sys, types

# Stub the Groq client before the notebook imports it: no network, no API key.
groq = types.ModuleType("langchain_groq")

class ChatGroq:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def invoke(self, messages):
        class _Response:
            content = __STUB_ANSWER__
        return _Response()

groq.ChatGroq = ChatGroq
sys.modules["langchain_groq"] = groq
os.environ["GROQ_API_KEY"] = "test-key-never-used"

notebook_path, pdf_dir, workdir = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(os.path.join(workdir, "notebook"))
os.makedirs(os.path.join(workdir, "data"))
shutil.copytree(pdf_dir, os.path.join(workdir, "data", "pdf"))
os.chdir(os.path.join(workdir, "notebook"))

cells = [
    "".join(c["source"])
    for c in json.load(open(notebook_path))["cells"]
    if c["cell_type"] == "code"
]
namespace = {"__name__": "__notebook__"}
for index, source in enumerate(cells):
    try:
        exec(compile(source, "<notebook cell %d>" % index, "exec"), namespace)
    except Exception as exc:
        raise AssertionError("notebook cell %d raised %s: %s" % (index, type(exc).__name__, exc))

retriever = namespace["rag_retriever"]
documents = namespace["all_pdf_documents"]

# Threshold -1.0 disables filtering so every retrieved chunk is visible.
top = retriever.retrieve("What is Python?", top_k=3, score_threshold=-1.0)
default = retriever.retrieve("What is Python?", top_k=3)

result = {
    "documents": [{"source_file": d.metadata.get("source_file"), "page": d.metadata.get("page")} for d in documents],
    "chunks": len(namespace["chunks"]),
    "collection_count": namespace["vectorstore"].count(),
    "top": [
        {
            "source_file": hit["metadata"].get("source_file"),
            "page": hit["metadata"].get("page"),
            "similarity_score": hit["similarity_score"],
            "content": hit["content"],
        }
        for hit in top
    ],
    "default_threshold_count": len(default),
    "answer": namespace.get("answer"),
}
print("###NOTEBOOK-RESULT###" + json.dumps(result))
'''.replace("__STUB_ANSWER__", json.dumps(STUB_ANSWER))


@pytest.fixture(scope="session")
def notebook_run(tmp_path_factory) -> dict:
    """Run the notebook's cells once and return what they produced."""
    workdir = tmp_path_factory.mktemp("notebook-run")
    proc = subprocess.run(
        [sys.executable, "-c", HARNESS, str(NOTEBOOK), str(REAL_PDF_DIR), str(workdir)],
        capture_output=True,
        text=True,
        cwd=REPO,
        timeout=900,
    )
    assert proc.returncode == 0, (
        f"the notebook failed to run end to end:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    )
    payload = [line for line in proc.stdout.splitlines() if line.startswith(MARKER)]
    assert payload, f"the notebook run produced no result marker:\n{proc.stdout[-3000:]}"
    return json.loads(payload[-1][len(MARKER) :])


def test_notebook_answers_a_question_about_the_pdfs(notebook_run):
    """The friend's symptom: asking about the two PDFs retrieved no context."""
    answer = notebook_run["answer"]
    assert answer is not None, "the notebook never produced an `answer`"
    assert answer != SENTINEL, "the notebook still answers with the no-context sentinel"
    assert answer == STUB_ANSWER


def test_notebook_retrieves_context_for_the_two_pdfs(notebook_run):
    assert notebook_run["default_threshold_count"] > 0, "retrieve() returned no context at its default threshold"
    assert notebook_run["top"], "retrieval returned nothing even with filtering disabled"
    assert notebook_run["top"][0]["source_file"] == "Python.pdf"


def test_notebook_ingests_each_pdf_page_once(notebook_run):
    documents = notebook_run["documents"]
    assert len(documents) == 2, f"expected 1 document per PDF page, got {len(documents)}"
    assert sorted(d["source_file"] for d in documents) == ["Java.pdf", "Python.pdf"]
    assert notebook_run["chunks"] > 0


def test_notebook_collection_is_not_polluted_by_reruns(notebook_run):
    """Re-running the notebook used to append the same chunks again (the friend's
    collection reached 5295 rows), so top-k filled with duplicate chunks."""
    assert notebook_run["collection_count"] == notebook_run["chunks"]

    contents = [hit["content"] for hit in notebook_run["top"]]
    assert len(contents) == len(set(contents)), "top-3 contained duplicate chunks"


def test_notebook_scores_are_cosine_similarities(notebook_run):
    """`1.0 - distance` on Chroma's default L2 space is 2*cos - 1, so good chunks
    scored like 0.57 and the threshold silently dropped real context."""
    top = notebook_run["top"]
    assert top
    best = top[0]["similarity_score"]
    assert 0.0 <= best <= 1.0, f"score outside the cosine range: {best}"
    assert best > 0.6, f"the best chunk for 'What is Python?' scored {best}, expected a real cosine"
