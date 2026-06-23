"""
diagnostics.py
--------------
Vector store inspection utilities.

Use this script to debug retrieval issues:
    1 — Semantic search: finds chunks by meaning (embedding-based)
    2 — Lexical search:  finds chunks containing an exact string
    3 — List sources:    shows which files are indexed and chunk counts

When to use each:
    Semantic search  → check what the retriever actually returns for your query
                       inspect the distance scores to calibrate SCORE_THRESHOLD
    Lexical search   → verify a key term exists in the vector store at all
                       if 0 results: document may not be indexed or text extraction failed
    List sources     → confirm which documents were indexed successfully
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from src.config import CHROMA_COLLECTION, EMBEDDING_MODEL, VECTORSTORE_DIR


def _get_vectorstore() -> Chroma:
    """Load ChromaDB from disk — shared across all diagnostic functions."""
    return Chroma(
        collection_name=CHROMA_COLLECTION,
        persist_directory=str(VECTORSTORE_DIR),
        embedding_function=OllamaEmbeddings(model=EMBEDDING_MODEL),
    )


def semantic_search(term: str, k: int = 6) -> None:
    """
    Retrieve chunks by semantic similarity.

    ChromaDB returns cosine distance scores:
        0.0 = identical vectors (best match)
        1.0 = completely different vectors (no match)

    Lower score = more relevant.
    If the expected chunk scores poorly here, consider:
        - Lowering SCORE_THRESHOLD in config.py (accept more results)
        - Using hybrid search (semantic + BM25 lexical)
        - Reviewing chunk size — maybe the key term is split across chunks
    """
    vs = _get_vectorstore()
    print(f"\n--- Semantic search: '{term}' (k={k}) ---")

    results = vs.similarity_search_with_score(term, k=k)
    if not results:
        print("No results found.")
        return

    for doc, score in results:
        source = doc.metadata.get("source", "unknown")
        print(f"score: {score:.4f} | source: {source}")
        print(f"  {doc.page_content[:200]}")
        print()


def lexical_search(term: str) -> None:
    """
    Search for an exact term string across all indexed chunks.

    Bypasses embeddings entirely — purely string matching.
    If this returns 0 results, the term is not in the vector store:
        - The document was not indexed (check list_sources)
        - The PDF loader failed to extract text (scanned/image-based PDF)
        - The term appears only in a page header/footer stripped during loading
    """
    vs       = _get_vectorstore()
    all_docs = vs.get()
    total    = len(all_docs["documents"])

    matches = [
        (content, meta)
        for content, meta in zip(all_docs["documents"], all_docs["metadatas"])
        if term.lower() in content.lower()
    ]

    print(f"\n--- Lexical search: '{term}' ---")
    print(f"{len(matches)} of {total} chunks contain the term\n")

    for content, meta in matches:
        source = meta.get("source", "unknown")
        print(f"source: {source}")
        print(f"  {content[:300]}")
        print()


def list_sources() -> None:
    """
    List all indexed source files and their chunk counts.

    Use to verify which documents are in the vector store
    and detect missing files before debugging retrieval.
    """
    vs       = _get_vectorstore()
    all_docs = vs.get()
    total    = len(all_docs["documents"])

    sources: dict[str, int] = {}
    for meta in all_docs["metadatas"]:
        src = meta.get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1

    print(f"\n--- Indexed sources ({total} total chunks) ---")
    for src, count in sorted(sources.items()):
        print(f"  {count:>4} chunks | {src}")


MENU = {
    "1": ("Semantic search (embedding-based)", semantic_search),
    "2": ("Lexical search (exact term)",       lexical_search),
    "3": ("List indexed sources",              list_sources),
}

if __name__ == "__main__":
    print("\nVector store diagnostics")
    for key, (label, _) in MENU.items():
        print(f"  {key} — {label}")

    choice     = input("\nChoice: ").strip()
    menu_entry = MENU.get(choice)

    if not menu_entry:
        print("Invalid option.")
    else:
        label, func = menu_entry
        if choice in ("1", "2"):
            term = input("Term: ").strip()
            func(term)
        else:
            func()
