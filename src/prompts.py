"""
prompts.py
----------
All prompt templates for the RAG pipeline.

Centralizing prompts here means:
    - Easy to compare, version, and iterate on prompt quality
    - No prompt strings scattered across chain.py and agent.py
    - Prompts can be tested independently of the rest of the pipeline

Prompt engineering notes:
    - "Use ONLY the context below" grounds the LLM in the documents
      and prevents hallucination from training data
    - Explicit fallback instruction ("say you don't know") is more
      reliable than hoping the LLM will admit ignorance on its own
    - temperature=0 in the LLM + grounding instruction in the prompt
      = maximum factual reliability for RAG use cases
"""

from langchain_core.prompts import ChatPromptTemplate

# ── RAG generation prompt ─────────────────────────────────────────────────────

RAG_PROMPT = ChatPromptTemplate.from_template("""
You are an assistant that answers questions based on provided documents.
Use ONLY the context below to answer. Be direct and objective.
If the context does not contain the answer, say exactly:
"I could not find this information in the documents."

Context:
{context}

Question: {question}

Answer:""")

# ── Chain prompt (used in Phase 1 / main_chain.py) ───────────────────────────
# Identical in intent but uses {question} via RunnablePassthrough
# keeping it separate makes the chain definition cleaner

CHAIN_PROMPT = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the question based ONLY on the context below.
If the answer is not in the context, say you don't know.

Context:
{context}

Question: {question}
""")
