"""
tests/test_prompts.py
---------------------
Tests for prompt templates (src/prompts.py).

Prompts are pure Python — no LLM calls needed.
These tests verify that templates render correctly and contain
the expected variables, which prevents silent bugs when prompts are edited.

Run with:
    pytest tests/test_prompts.py -v
"""

import pytest

from src.prompts import CHAIN_PROMPT, RAG_PROMPT


# ── RAG_PROMPT tests ──────────────────────────────────────────────────────────

def test_rag_prompt_has_required_variables():
    """RAG_PROMPT must expose {context} and {question} as input variables."""
    variables = set(RAG_PROMPT.input_variables)
    assert "context"  in variables
    assert "question" in variables


def test_rag_prompt_renders_correctly():
    """RAG_PROMPT should render without errors when given valid inputs."""
    rendered = RAG_PROMPT.format_messages(
        context="Python is a programming language.",
        question="What is Python?",
    )
    full_text = " ".join(m.content for m in rendered)
    assert "Python is a programming language" in full_text
    assert "What is Python?" in full_text


def test_rag_prompt_contains_grounding_instruction():
    """RAG_PROMPT must instruct the LLM to use ONLY the provided context."""
    rendered  = RAG_PROMPT.format_messages(context="x", question="y")
    full_text = " ".join(m.content for m in rendered).lower()
    assert "only" in full_text


# ── CHAIN_PROMPT tests ────────────────────────────────────────────────────────

def test_chain_prompt_has_required_variables():
    """CHAIN_PROMPT must expose {context} and {question}."""
    variables = set(CHAIN_PROMPT.input_variables)
    assert "context"  in variables
    assert "question" in variables


def test_chain_prompt_renders_correctly():
    """CHAIN_PROMPT should render without errors when given valid inputs."""
    rendered = CHAIN_PROMPT.format_messages(
        context="The sky is blue.",
        question="What color is the sky?",
    )
    full_text = " ".join(m.content for m in rendered)
    assert "The sky is blue." in full_text
    assert "What color is the sky?" in full_text
