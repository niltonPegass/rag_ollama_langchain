"""
tests/test_agent.py
-------------------
Tests for the LangGraph agent routing logic (src/agent.py).

The conditional edge function (edge_after_retrieval) is pure Python —
it takes an AgentState dict and returns a string.
This makes it trivially testable without any LLM or vector store.

These tests verify that the routing logic handles all three branches:
    - Chunks found       → "generator"
    - No chunks, retry   → "retriever"
    - No chunks, give up → "fallback"

Run with:
    pytest tests/test_agent.py -v
"""

import pytest
from langchain_core.documents import Document

from src.agent import AgentState, edge_after_retrieval
from src.config import MAX_RETRIES


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_state(
    question:   str = "test question",
    documents:  list = None,
    generation: str = "",
    attempts:   int = 0,
) -> AgentState:
    """Build a minimal AgentState for testing."""
    return AgentState(
        question=question,
        documents=documents if documents is not None else [],
        generation=generation,
        attempts=attempts,
    )


def make_doc(content: str = "relevant chunk") -> Document:
    return Document(page_content=content, metadata={"source": "test.txt"})


# ── edge_after_retrieval tests ────────────────────────────────────────────────

def test_routes_to_generator_when_chunks_found():
    """Should route to generator when documents are present."""
    state = make_state(documents=[make_doc()], attempts=1)
    assert edge_after_retrieval(state) == "generator"


def test_routes_to_retriever_on_first_empty_result():
    """Should retry retrieval on first empty result."""
    state = make_state(documents=[], attempts=1)
    assert edge_after_retrieval(state) == "retriever"


def test_routes_to_fallback_after_max_retries():
    """Should fall back after MAX_RETRIES failed attempts."""
    state = make_state(documents=[], attempts=MAX_RETRIES)
    assert edge_after_retrieval(state) == "fallback"


def test_routes_to_fallback_beyond_max_retries():
    """Should still fall back even if attempts somehow exceed MAX_RETRIES."""
    state = make_state(documents=[], attempts=MAX_RETRIES + 5)
    assert edge_after_retrieval(state) == "fallback"


def test_routes_to_generator_with_multiple_chunks():
    """Should route to generator regardless of how many chunks were found."""
    state = make_state(documents=[make_doc(), make_doc(), make_doc()], attempts=1)
    assert edge_after_retrieval(state) == "generator"


def test_chunks_take_priority_over_attempts():
    """
    If chunks are found, should route to generator even after many attempts.
    Retry count should not block generation when chunks are present.
    """
    state = make_state(documents=[make_doc()], attempts=MAX_RETRIES + 10)
    assert edge_after_retrieval(state) == "generator"
