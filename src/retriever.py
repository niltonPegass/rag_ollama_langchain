"""
retriever.py
------------
Retriever factory for the RAG pipeline.

A retriever wraps a vector store and exposes a simple .invoke(query) interface
that returns a list of relevant Document chunks.

Why isolate the retriever in its own module?
    Retrieval strategy is the most tuneable part of a RAG system.
    Isolating it here makes it easy to swap between:
    - Similarity search (default)
    - MMR (Maximum Marginal Relevance — reduces redundancy among returned chunks)
    - BM25 (lexical/keyword-based — better for exact term matching)
    - Ensemble (hybrid: semantic + lexical)
    without touching any other module.

Current strategy: similarity_score_threshold
    Returns up to TOP_K chunks whose cosine distance is below SCORE_THRESHOLD.
    ChromaDB distance: 0.0 = identical, 1.0 = completely different.
    Lower threshold = stricter filtering = fewer but more relevant results.
"""

from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever

from src.config import SCORE_THRESHOLD, TOP_K_RESULTS
from src.logger import get_logger

log = get_logger(__name__)


def build_retriever(vectorstore: Chroma) -> VectorStoreRetriever:
    """
    Build a retriever from the given vector store.

    Uses similarity_score_threshold search type:
    - Returns at most TOP_K_RESULTS chunks
    - Only returns chunks with distance <= SCORE_THRESHOLD
    - If no chunks pass the threshold, returns an empty list
      (the agent handles this via its fallback node)

    Args:
        vectorstore: initialized Chroma instance from vectorstore.py

    Returns:
        VectorStoreRetriever ready to use with .invoke(query)
    """
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k":               TOP_K_RESULTS,
            "score_threshold": SCORE_THRESHOLD,
        },
    )

    log.info(
        f"Retriever ready — k={TOP_K_RESULTS}, threshold={SCORE_THRESHOLD}"
    )
    return retriever
