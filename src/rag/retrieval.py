"""Importable RAG pipeline: PDFs -> chunks -> Chroma -> retrieve -> answer.

This is the fixed, testable version of the pipeline in notebook/pdf_loader.ipynb.
The notebook stays the walkthrough; this module is what the tests exercise, so a
regression in the retrieval path fails CI instead of silently returning
"No relevant context found".

Bugs this module fixes, all of which were in the notebook:

1. `rag_simple` read `doc['context']` (the retriever returns ``content``) and then
   `response.context` (a LangChain AIMessage has ``content``), so it raised
   KeyError/AttributeError and no answer was ever produced.
2. Scores were computed as ``1.0 - distance`` on Chroma's default L2 space, which
   equals ``2*cos - 1`` for unit-length embeddings: good chunks scored ~0.57,
   off-topic chunks scored negative, and any threshold silently dropped real
   context. See :func:`cosine_similarity`.
3. Ingestion was not idempotent: every re-run appended the same chunks again
   (a friend's collection reached 5295 rows), so top-k filled with duplicates of
   a single chunk and crowded out the answer-bearing ones. Ingestion now uses
   deterministic ids with ``upsert``, plus :meth:`VectorStore.reset`.
4. ``process_all_pdfs`` extended its result list *inside* the per-page loop, so an
   n-page PDF contributed n x n documents.
5. ``{self.collection.count}`` printed a bound method instead of the count.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import chromadb
import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_COLLECTION_NAME = "pdf_documents"
DEFAULT_PERSIST_DIRECTORY = "data/vector_store"
# gemma2-9b-it was decommissioned by Groq; this one is live.
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"
NO_CONTEXT_MESSAGE = "No relevant context found to answer the question."


# ---------------------------------------------------------------- ingestion


def load_pdfs(
    pdf_directory: Union[str, Path], glob: str = "**/*.pdf"
) -> List[Document]:
    """Load every PDF under ``pdf_directory``, one Document per page.

    Sorted for a deterministic chunk order (and therefore deterministic ids).
    """
    pdf_dir = Path(pdf_directory)
    pdf_files = sorted(pdf_dir.glob(glob))

    print(f"Found {len(pdf_files)} PDF files to process")

    all_documents: List[Document] = []
    for pdf_file in pdf_files:
        print(f"\nProcessing: {pdf_file.name}")
        try:
            documents = PyPDFLoader(str(pdf_file)).load()
        except Exception as e:
            print(f" ✗ Error : {e}")
            continue

        # Add source information to metadata
        for doc in documents:
            doc.metadata["source_file"] = pdf_file.name
            doc.metadata["file_type"] = "pdf"

        all_documents.extend(documents)
        print(f" ✓ Loaded {len(documents)} pages")

    print(f"\nTotal documents loaded: {len(all_documents)}")
    return all_documents


def split_documents(
    documents: Sequence[Document], chunk_size: int = 1000, chunk_overlap: int = 200
) -> List[Document]:
    """Split documents into smaller chunks for better RAG performance."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    split_docs = text_splitter.split_documents(list(documents))
    print(f"Split {len(documents)} documents into {len(split_docs)} chunks")
    return split_docs


class EmbeddingManager:
    """Handles document embedding generation using SentenceTransformer."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print(
                "Model loaded successfully. "
                f"Embedding dimension: {self.model.get_embedding_dimension()}"
            )
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            raise

    def generate_embeddings(self, texts: Sequence[str]) -> np.ndarray:
        """Embed ``texts`` as unit-length vectors.

        Unit length matters: it is what makes the squared-L2 distance from Chroma
        convertible to a cosine similarity (see :func:`cosine_similarity`).
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        if not texts:
            return np.empty((0, self.model.get_embedding_dimension()), dtype=np.float32)

        print(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(
            list(texts), show_progress_bar=False, normalize_embeddings=True
        )
        print(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings


# ---------------------------------------------------------------- vector store


def cosine_similarity(distance: float, space: str) -> float:
    """Convert a Chroma *distance* into a cosine similarity.

    Chroma returns distances, and the meaning depends on the collection's space.
    With SentenceTransformer embeddings (unit length):

    * ``"l2"`` (the default): ``distance = ||a-b||^2 = 2 - 2*cos``,
      so ``cos = 1 - distance / 2``. This is the case the notebook got wrong by
      using ``1 - distance`` (= ``2*cos - 1``), which scores a perfect match 1.0
      but pushes everything below cos 0.5 negative, so ``score_threshold=0.0``
      silently discarded real context.
    * ``"cosine"``: ``distance = 1 - cos``, so ``cos = 1 - distance``.
    * ``"ip"``: inner product, which equals cos for unit-length vectors.
    """
    if space == "l2":
        return 1.0 - distance / 2.0
    if space == "cosine":
        return 1.0 - distance
    if space == "ip":
        return -distance
    raise ValueError(f"unsupported distance space: {space!r}")


class VectorStore:
    """Manages document embeddings in a ChromaDB vector store."""

    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        persist_directory: Union[str, Path] = DEFAULT_PERSIST_DIRECTORY,
    ):
        self.collection_name = collection_name
        self.persist_directory = str(persist_directory)
        self.client: Optional[chromadb.ClientAPI] = None
        self.collection: Any = None
        self._initialize_store()

    def _initialize_store(self) -> None:
        os.makedirs(self.persist_directory, exist_ok=True)
        try:
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "PDF document embeddings for RAG"},
            )
            print(f"Vector store initialized. Collection: {self.collection_name}")
            print(f"Existing documents in collection: {self.count()}")
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise

    @property
    def space(self) -> str:
        """The distance space the collection actually uses ("l2" by default)."""
        configuration = getattr(self.collection, "configuration", None) or {}
        return (configuration.get("hnsw") or {}).get("space", "l2")

    def count(self) -> int:
        return self.collection.count()

    @staticmethod
    def _document_id(document: Document, index: int) -> str:
        """Deterministic id, so re-ingesting the same PDF updates in place.

        Deliberately *not* derived from the content: if it were, editing a PDF would
        create new ids and leave the old chunks behind, which is how a collection
        grows on every run again. Position-keyed ids overwrite in place. Use
        :meth:`reset` when a document shrinks, to drop the orphaned tail.
        """
        source = document.metadata.get("source_file") or document.metadata.get("source", "")
        page = document.metadata.get("page", "?")
        return f"doc_{source}_p{page}_c{index}"

    def add_documents(self, documents: Sequence[Document], embeddings: np.ndarray) -> None:
        """Add (or update) documents and their embeddings.

        Uses deterministic ids with ``upsert``: running the pipeline twice over the
        same PDFs leaves the collection the same size instead of doubling it.
        """
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match exact number of embeddings")
        if not documents:
            print("Nothing to add: no documents were supplied")
            return

        print(f"Adding {len(documents)} documents to vector store...")

        ids, metadatas, documents_text, embeddings_list = [], [], [], []
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            ids.append(self._document_id(doc, i))
            metadata = dict(doc.metadata)
            metadata["doc_index"] = i
            metadata["content_length"] = len(doc.page_content)
            metadatas.append(metadata)
            documents_text.append(doc.page_content)
            embeddings_list.append(np.asarray(embedding, dtype=np.float32).tolist())

        try:
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=documents_text,
            )
            print(f"Successfully added {len(documents)} documents to vector store")
            print(f"Total documents in collection: {self.count()}")
        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            raise

    def reset(self) -> None:
        """Delete and recreate the collection: rebuild from data/pdf from scratch.

        Needed once when moving off a store built by the old notebook, which is
        holding thousands of duplicated chunks.
        """
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "PDF document embeddings for RAG"},
        )
        print(f"Vector store reset. Documents in collection: {self.count()}")

    def cosine_similarity(self, distance: float) -> float:
        return cosine_similarity(distance, self.space)


# ---------------------------------------------------------------- retrieval


class RAGRetriever:
    """Handles query-based retrieval from the vector store."""

    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(
        self, query: str, top_k: int = 5, score_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Return the top-k chunks whose cosine similarity clears ``score_threshold``."""
        print(f"Retrieving documents for query: '{query}'")
        print(f"Top K: {top_k}, Score threshold: {score_threshold}")

        try:
            query_embedding = self.embedding_manager.generate_embeddings([query])[0]
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()], n_results=top_k
            )

            retrieved_docs: List[Dict[str, Any]] = []
            documents = results.get("documents") or [[]]
            metadatas = results.get("metadatas") or [[]]
            distances = results.get("distances") or [[]]
            ids = results.get("ids") or [[]]

            if documents[0]:
                for rank, (doc_id, document, metadata, distance) in enumerate(
                    zip(ids[0], documents[0], metadatas[0], distances[0]), start=1
                ):
                    similarity_score = self.vector_store.cosine_similarity(distance)
                    if similarity_score >= score_threshold:
                        retrieved_docs.append(
                            {
                                "id": doc_id,
                                "content": document,
                                "metadata": metadata,
                                "similarity_score": similarity_score,
                                "distance": distance,
                                "rank": rank,
                            }
                        )

            print(f"Retrieved {len(retrieved_docs)} documents (after filtering)")
            return retrieved_docs
        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []


# ---------------------------------------------------------------- generation


def load_env() -> None:
    """Load GROQ_API_KEY from the repo-root .env (and the notebook's own .env)."""
    from dotenv import load_dotenv

    repo_root = Path(__file__).resolve().parents[2]
    load_dotenv(repo_root / ".env")
    load_dotenv()  # notebook/.env or any .env found from the working directory


class _GroqResponse:
    """Minimal stand-in for a LangChain AIMessage: just carries ``content``."""

    def __init__(self, content: str):
        self.content = content


class StdlibGroqChat:
    """Groq chat client with no third-party dependency.

    A deployed environment does not always have ``langchain-groq`` installed, and
    when it is missing the import fails at question time. The Groq API is a plain
    HTTPS POST, so this keeps an app working without the package. It exposes the one
    method the rest of the pipeline uses.
    """

    ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(
        self,
        api_key: str,
        model_name: str = DEFAULT_GROQ_MODEL,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        timeout: int = 60,
    ):
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def invoke(self, prompt):
        import json
        import urllib.error
        import urllib.request

        content = prompt if isinstance(prompt, str) else str(prompt)
        payload = json.dumps(
            {
                "model": self.model_name,
                "messages": [{"role": "user", "content": content}],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            self.ENDPOINT,
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                # Groq sits behind Cloudflare, which answers 403 with error code 1010
                # when the default "Python-urllib/x.y" signature is presented.
                "User-Agent": "rag-pdf-app/1.0",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:300]
            raise RuntimeError(f"Groq API error {e.code}: {detail}") from e

        choice = body["choices"][0]
        content = choice["message"].get("content") or ""
        if not content.strip():
            # gpt-oss models reason before answering, so a small max_tokens can be
            # spent entirely on reasoning tokens and return empty text.
            raise RuntimeError(
                f"Groq returned no text (finish_reason={choice.get('finish_reason')!r}). "
                f"{self.model_name} is a reasoning model: raise max_tokens."
            )
        return _GroqResponse(content)


def build_llm(
    model_name: str = DEFAULT_GROQ_MODEL,
    temperature: float = 0.1,
    max_tokens: int = 1024,
    api_key: Optional[str] = None,
):
    """Build a Groq chat client, failing loudly when the key is missing.

    Uses ``langchain_groq`` when it is installed and falls back to the stdlib client
    otherwise, so a missing package degrades instead of breaking the pipeline.
    """
    load_env()  # repo-root .env, so this works straight from a script
    key = api_key or os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Put it in the repo's .env file as\n"
            "  GROQ_API_KEY=your-key-here"
        )

    try:
        from langchain_groq import ChatGroq
    except ImportError as e:
        print(f"langchain_groq is not installed ({e}); using the stdlib Groq client")
        return StdlibGroqChat(
            key, model_name=model_name, temperature=temperature, max_tokens=max_tokens
        )

    return ChatGroq(
        groq_api_key=key, model_name=model_name, temperature=temperature, max_tokens=max_tokens
    )


def answer_question(
    query: str,
    retriever: RAGRetriever,
    llm,
    top_k: int = 3,
    min_score: float = 0.0,
) -> str:
    """Retrieve context for ``query`` and answer from it. Empty context short-circuits."""
    results = retriever.retrieve(query, top_k=top_k, score_threshold=min_score)
    context = "\n\n".join(doc["content"] for doc in results) if results else ""
    if not context:
        return NO_CONTEXT_MESSAGE

    prompt = f"""Use the following context to answer the question concisely.

        Context:
        {context}

        Question: {query}

        Answer: """
    response = llm.invoke(prompt)
    return response.content


def rag_advanced(
    query: str,
    retriever: RAGRetriever,
    llm,
    top_k: int = 5,
    min_score: float = 0.2,
    return_context: bool = False,
) -> Dict[str, Any]:
    """RAG pipeline that also returns sources and a confidence score."""
    results = retriever.retrieve(query, top_k=top_k, score_threshold=min_score)
    if not results:
        return {"answer": "No relevant context found.", "sources": [], "confidence": 0.0, "context": ""}

    context = "\n\n".join(doc["content"] for doc in results)
    sources = [
        {
            "source": doc["metadata"].get("source_file", doc["metadata"].get("source", "unknown")),
            "page": doc["metadata"].get("page", "unknown"),
            "score": doc["similarity_score"],
            "preview": doc["content"][:300] + "...",
        }
        for doc in results
    ]
    confidence = max(doc["similarity_score"] for doc in results)

    prompt = (
        "Use the following context to answer the question concisely.\n"
        f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    )
    response = llm.invoke(prompt)

    output: Dict[str, Any] = {
        "answer": response.content,
        "sources": sources,
        "confidence": confidence,
    }
    if return_context:
        output["context"] = context
    return output
