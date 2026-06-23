"""
vectorstore.py
--------------
ChromaDB vector store management.

Responsibilities:
    1. Initialize the embedding model
    2. Build the vector store from chunks (first run)
    3. Load an existing vector store from disk (subsequent runs)
    4. Expose a simple interface used by retriever.py

How ChromaDB works:
    - Documents are embedded into dense vectors (lists of floats)
    - Vectors are stored in a local SQLite + binary index on disk
    - At query time, the query is also embedded and compared against stored vectors
    - ChromaDB returns the N most similar vectors (and their source documents)

Persistence strategy:
    - On first run: index all chunks, persist to VECTORSTORE_DIR
    - On subsequent runs: load from disk — no re-indexing
    - To force re-indexing: delete the VECTORSTORE_DIR folder
"""

from pathlib import Path
from typing import Optional

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from src.config import (
    CHROMA_COLLECTION,
    EMBEDDING_MODEL,
    VECTORSTORE_DIR,
)
from src.loader import load_documents, split_documents
from src.logger import get_logger

log = get_logger(__name__)


def _get_embeddings() -> OllamaEmbeddings:
    """
    Initialize the Ollama embedding model.

    Using a dedicated embedding model (nomic-embed-text) instead of the LLM
    for embeddings is faster, cheaper, and produces better retrieval quality.
    Embedding models are trained specifically to map text to dense vector spaces
    where semantic similarity = vector proximity.
    """
    return OllamaEmbeddings(model=EMBEDDING_MODEL)


def build_vectorstore(docs_dir: Optional[Path] = None) -> Chroma:
    """
    Build or load the ChromaDB vector store.

    First run  → indexes all documents from docs_dir, persists to disk
    Subsequent → loads existing index from disk (fast, no re-embedding)

    Args:
        docs_dir: override the default DOCS_DIR from config (useful for testing)

    Returns:
        Chroma vector store instance ready for retrieval
    """
    embeddings  = _get_embeddings()
    vectorstore = Chroma(
        collection_name=CHROMA_COLLECTION,
        persist_directory=str(VECTORSTORE_DIR),
        embedding_function=embeddings,
    )

    count = vectorstore._collection.count()

    if count == 0:
        log.info("Vector store is empty — indexing documents...")
        documents = load_documents(docs_dir) if docs_dir else load_documents()
        chunks    = split_documents(documents)

        if not chunks:
            log.warning("No documents found — vector store remains empty")
            return vectorstore

        vectorstore.add_documents(chunks)
        log.info(f"Indexed {len(chunks)} chunks and persisted to {VECTORSTORE_DIR}")
    else:
        log.info(f"Loaded existing vector store ({count} chunks) from {VECTORSTORE_DIR}")

    return vectorstore
