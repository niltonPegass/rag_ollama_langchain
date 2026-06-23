"""
loader.py
---------
Document loading and chunking for the RAG pipeline.

Responsibilities:
    1. Discover supported files in a folder (PDF, TXT)
    2. Load them into LangChain Document objects
    3. Split documents into chunks suitable for embedding

Why split documents?
    LLMs have context limits. Embedding quality also degrades with long texts.
    Splitting into overlapping chunks ensures:
    - Each chunk is small enough to embed accurately
    - Context around chunk boundaries is preserved via overlap
    - The retriever can pinpoint the exact passage that answers a question

Supported formats:
    .pdf  — extracted via PyPDFLoader (text-based PDFs only; scanned PDFs need OCR)
    .txt  — plain text via TextLoader
"""

from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_OVERLAP, CHUNK_SIZE, DOCS_DIR
from src.logger import get_logger

log = get_logger(__name__)


def load_documents(folder: Path = DOCS_DIR) -> List[Document]:
    """
    Load all supported documents from a folder recursively.

    Returns a flat list of Document objects. Each Document contains:
        - page_content: the extracted text
        - metadata: source path, page number (for PDFs), etc.

    Args:
        folder: path to search for documents (defaults to DOCS_DIR from config)

    Returns:
        List of Document objects ready for splitting
    """
    documents: List[Document] = []

    for path in sorted(folder.rglob("*")):
        if path.suffix == ".pdf":
            try:
                docs = PyPDFLoader(str(path)).load()
                documents.extend(docs)
                log.info(f"Loaded PDF: {path.name} ({len(docs)} pages)")
            except Exception as e:
                log.warning(f"Failed to load {path.name}: {e}")

        elif path.suffix == ".txt":
            try:
                docs = TextLoader(str(path), encoding="utf-8").load()
                documents.extend(docs)
                log.info(f"Loaded TXT: {path.name} ({len(docs)} documents)")
            except Exception as e:
                log.warning(f"Failed to load {path.name}: {e}")

    log.info(f"Total: {len(documents)} pages/documents loaded from {folder}")
    return documents


def split_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into smaller chunks for embedding.

    Uses RecursiveCharacterTextSplitter, which tries to split on natural
    boundaries in this order: paragraphs → sentences → words → characters.
    This preserves semantic coherence better than fixed-size splits.

    Args:
        documents: list of Document objects from load_documents()

    Returns:
        List of smaller Document chunks, each preserving the original metadata
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # separators tried in order (falls back to next if chunk still too large)
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)
    log.info(f"Split into {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks
