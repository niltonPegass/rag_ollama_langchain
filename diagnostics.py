"""
diagnostics.py
--------------
Utility script for inspecting and debugging the ChromaDB vector store.

Use this when:
- The RAG agent is not finding relevant chunks
- You want to verify which documents were indexed
- You need to calibrate the similarity score threshold
- You suspect chunking broke the context around a key term

Menu options:
    1 — Semantic search (embedding-based): finds chunks by meaning
    2 — Lexical search (exact term match): finds chunks containing a specific string
    3 — List indexed sources: shows which files are in the vector store and chunk counts
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# ── Configuration ────────────────────────────────────────────────────────────

VECTORSTORE_DIR = "vectorstore/"
EMBEDDING_MODEL = "nomic-embed-text"


# ── Vector store loader ──────────────────────────────────────────────────────

def get_vectorstore() -> Chroma:
    """
    Load the ChromaDB vector store from disk.
    Centralizing this avoids instantiating Chroma multiple times.
    """
    return Chroma(
        persist_directory=VECTORSTORE_DIR,
        embedding_function=OllamaEmbeddings(model=EMBEDDING_MODEL),
    )


# ── Diagnostic functions ─────────────────────────────────────────────────────

def semantic_search(term: str) -> None:
    """
    Retrieve chunks by semantic similarity to the query term.

    ChromaDB returns a distance score (not similarity):
        score = 0.0 → identical vectors (perfect match)
        score = 1.0 → completely different vectors (no match)

    Lower score = more relevant chunk.
    If chunks with the key term score poorly, the embedding model may not
    capture domain-specific terminology well — consider hybrid search (BM25 + semantic).
    """
    vs = get_vectorstore()
    print(f"\n--- Semantic search: '{term}' ---")

    results = vs.similarity_search_with_score(term, k=6)
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

    This bypasses embeddings entirely — useful for debugging cases where
    a key term exists in the documents but semantic search misses it.

    If this returns 0 results, the term is not in the vector store:
        - The document may not have been indexed
        - The PDF loader may have failed to extract the text (scanned PDFs, images)
        - The chunking may have split the term across chunk boundaries
    """
    vs       = get_vectorstore()
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

    Use this to verify which documents are in the vector store
    and detect cases where a file was not indexed (empty docs folder,
    unsupported format, loader error).
    """
    vs       = get_vectorstore()
    all_docs = vs.get()
    total    = len(all_docs["documents"])

    sources: dict[str, int] = {}
    for meta in all_docs["metadatas"]:
        src = meta.get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1

    print(f"\n--- Indexed sources ({total} total chunks) ---")
    for src, count in sorted(sources.items()):
        print(f"  {count:>4} chunks | {src}")


# ── Menu ─────────────────────────────────────────────────────────────────────

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
