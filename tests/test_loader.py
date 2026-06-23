"""
tests/test_loader.py
--------------------
Tests for document loading and chunking (src/loader.py).

These tests cover pure Python logic — no LLM calls, no Ollama, no ChromaDB.
Fast, deterministic, and runnable offline.

Run with:
    pytest tests/test_loader.py -v
"""

import tempfile
from pathlib import Path

import pytest

from src.loader import load_documents, split_documents
from langchain_core.documents import Document


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_txt_file(tmp_path: Path) -> Path:
    """Create a temporary .txt file with known content."""
    file = tmp_path / "sample.txt"
    file.write_text(
        "This is the first paragraph about artificial intelligence.\n\n"
        "This is the second paragraph about machine learning.\n\n"
        "This is the third paragraph about neural networks.",
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture
def empty_folder(tmp_path: Path) -> Path:
    """Empty folder with no supported files."""
    return tmp_path


@pytest.fixture
def unsupported_file(tmp_path: Path) -> Path:
    """Folder with only unsupported file types."""
    (tmp_path / "document.docx").write_bytes(b"fake docx content")
    return tmp_path


# ── load_documents tests ──────────────────────────────────────────────────────

def test_load_documents_txt(sample_txt_file: Path):
    """Should load text documents from a folder."""
    docs = load_documents(sample_txt_file)
    assert len(docs) >= 1
    assert any("artificial intelligence" in doc.page_content for doc in docs)


def test_load_documents_empty_folder(empty_folder: Path):
    """Should return empty list when folder has no supported files."""
    docs = load_documents(empty_folder)
    assert docs == []


def test_load_documents_unsupported_files(unsupported_file: Path):
    """Should silently skip unsupported file types."""
    docs = load_documents(unsupported_file)
    assert docs == []


def test_load_documents_metadata(sample_txt_file: Path):
    """Loaded documents should carry source metadata."""
    docs = load_documents(sample_txt_file)
    for doc in docs:
        assert "source" in doc.metadata


# ── split_documents tests ─────────────────────────────────────────────────────

def test_split_documents_produces_chunks():
    """Should produce multiple chunks from a long document."""
    long_text = "word " * 500
    docs = [Document(page_content=long_text, metadata={"source": "test.txt"})]
    chunks = split_documents(docs)
    assert len(chunks) > 1


def test_split_documents_respects_chunk_size():
    """Each chunk should not exceed CHUNK_SIZE by a large margin."""
    from src.config import CHUNK_SIZE
    long_text = "word " * 1000
    docs      = [Document(page_content=long_text, metadata={"source": "test.txt"})]
    chunks    = split_documents(docs)

    for chunk in chunks:
        # Allow some slack for overlap — no chunk should be more than 2x the size
        assert len(chunk.page_content) <= CHUNK_SIZE * 2


def test_split_documents_preserves_metadata():
    """Chunks should inherit metadata from source document."""
    docs   = [Document(page_content="word " * 200, metadata={"source": "test.txt", "page": 1})]
    chunks = split_documents(docs)

    for chunk in chunks:
        assert chunk.metadata.get("source") == "test.txt"


def test_split_documents_empty_input():
    """Should return empty list for empty input."""
    chunks = split_documents([])
    assert chunks == []
