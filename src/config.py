"""
config.py
---------
Central configuration for the RAG project.

All tuneable parameters live here — no magic strings scattered across modules.
To change a model, path, or threshold, edit this file only.

Design principle: configuration should be explicit and discoverable.
Hardcoded values inside functions are the enemy of maintainability.
"""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT_DIR        = Path(__file__).resolve().parent.parent
DOCS_DIR        = ROOT_DIR / "docs"
VECTORSTORE_DIR = ROOT_DIR / "vectorstore"
LOGS_DIR        = ROOT_DIR / "logs"

# ── Models ────────────────────────────────────────────────────────────────────

EMBEDDING_MODEL  = "nomic-embed-text"   # dedicated embedding model via Ollama
                                        # smaller and faster than using the LLM for embeddings

AVAILABLE_LLMS   = ["llama3.2", "mistral"]
DEFAULT_LLM      = "llama3.2"           # used when no model is specified at runtime

LLM_TEMPERATURE  = 0                    # 0 = deterministic, factual responses
                                        # increase for more creative/varied outputs

# ── Chunking ──────────────────────────────────────────────────────────────────

CHUNK_SIZE       = 500                  # max characters per chunk
                                        # smaller = more precise retrieval
                                        # larger  = more context per chunk

CHUNK_OVERLAP    = 50                   # characters shared between adjacent chunks
                                        # prevents losing context at chunk boundaries

# ── Retrieval ─────────────────────────────────────────────────────────────────

TOP_K_RESULTS    = 6                    # number of chunks to retrieve per query
SCORE_THRESHOLD  = 0.2                  # ChromaDB cosine distance threshold
                                        # 0.0 = accept everything
                                        # 1.0 = accept nothing
                                        # tune this based on your documents and embedding model
                                        # use diagnostics.py to inspect scores before adjusting

# ── Agent ─────────────────────────────────────────────────────────────────────

MAX_RETRIES      = 2                    # max retrieval attempts before fallback
                                        # prevents infinite loops when no chunks match

# ── ChromaDB ─────────────────────────────────────────────────────────────────

CHROMA_COLLECTION = "rag_collection"    # collection name inside ChromaDB
