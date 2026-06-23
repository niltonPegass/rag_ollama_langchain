"""
chain.py
--------
Phase 1 — Linear RAG chain using LangChain Expression Language (LCEL).

Architecture:
    question ──► retriever ──► format_docs ──► prompt ──► LLM ──► answer

This is the simplest possible RAG implementation. Every question follows
the same fixed path — no branching, no retries, no state.

When to use this vs the agent (Phase 2):
    - chain.py: simple Q&A, predictable queries, latency-sensitive
    - agent.py: complex queries, retry logic needed, observable decision flow

LCEL (LangChain Expression Language):
    The | operator composes Runnables — objects with a standard .invoke() interface.
    Each component receives the output of the previous one.
    RunnablePassthrough passes the original input unchanged to allow it to appear
    in multiple places downstream (here: both as retrieval query AND prompt variable).
"""

from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama

from src.config import DEFAULT_LLM, LLM_TEMPERATURE
from src.logger import get_logger
from src.prompts import CHAIN_PROMPT

log = get_logger(__name__)


def format_docs(docs: list) -> str:
    """
    Join retrieved Document chunks into a single context string.

    Double newlines between chunks help the LLM distinguish
    where one passage ends and another begins.
    """
    return "\n\n".join(doc.page_content for doc in docs)


def build_chain(vectorstore: Chroma, model: str = DEFAULT_LLM):
    """
    Build the LCEL RAG chain.

    Chain composition:
        1. Build a dict: context = retrieved+formatted chunks,
                         question = original query (passed through unchanged)
        2. Feed into the prompt template
        3. Send to the LLM
        4. Parse output as plain string

    Args:
        vectorstore: initialized Chroma instance
        model:       Ollama model name (default from config)

    Returns:
        A compiled LCEL Runnable — call with .invoke("your question")
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    llm       = ChatOllama(model=model, temperature=LLM_TEMPERATURE)

    # LCEL chain — reads left to right like a Unix pipe
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | CHAIN_PROMPT
        | llm
        | StrOutputParser()
    )

    log.info(f"RAG chain ready — model={model}")
    return chain
