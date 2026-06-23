"""
agent.py
--------
Phase 2 — Agentic RAG pipeline using LangGraph.

Architecture (stateful graph):

    question
        │
        ▼
    [retriever node] ──► chunks found? ──► [generator node] ──► answer
        │
        ├── no chunks, attempts < MAX_RETRIES ──► [retriever node] (retry)
        └── no chunks, attempts >= MAX_RETRIES ──► [fallback node] ──► answer

Key LangGraph concepts:
    StateGraph:   a directed graph where each node receives and returns AgentState
    AgentState:   a TypedDict shared across all nodes — the "memory" of the run
    Node:         a plain Python function (AgentState) → AgentState
    Edge:         a fixed transition between two nodes
    Conditional edge: a function that inspects state and returns the next node name
    END:          a special sentinel that terminates the graph

Why this is better than a chain for complex RAG:
    - Retries become a loop edge, not a try/except block
    - Fallback is an explicit node, not a hidden else branch
    - State is shared without passing data manually between functions
    - The graph is inspectable, serializable, and traceable via LangSmith
"""

from typing import List, TypedDict

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph

from src.config import DEFAULT_LLM, LLM_TEMPERATURE, MAX_RETRIES
from src.logger import get_logger
from src.prompts import RAG_PROMPT

log = get_logger(__name__)


# ── Agent state ───────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    """
    Shared state passed between all graph nodes.

    Every node receives the full state and returns an updated copy.
    LangGraph merges updates automatically — nodes only need to set
    the fields they modify.

    Fields:
        question:   the user's input question (immutable after entry)
        documents:  chunks retrieved from the vector store
        generation: the final answer (set by generator or fallback)
        attempts:   number of retrieval attempts (prevents infinite retry loops)
    """
    question:   str
    documents:  List[Document]
    generation: str
    attempts:   int


# ── Node factory ──────────────────────────────────────────────────────────────

def build_nodes(retriever: VectorStoreRetriever, llm: ChatOllama) -> dict:
    """
    Build all graph nodes as closures over retriever and llm.

    Using a factory function (instead of global variables) makes nodes
    testable in isolation and avoids module-level side effects.

    Args:
        retriever: initialized retriever from retriever.py
        llm:       initialized ChatOllama instance

    Returns:
        dict mapping node names to node functions
    """
    generator_chain = RAG_PROMPT | llm | StrOutputParser()

    def node_retriever(state: AgentState) -> AgentState:
        """
        Retrieve relevant chunks from the vector store.

        Uses semantic search: the question is embedded and compared
        against stored chunk embeddings via cosine similarity.
        Only chunks above SCORE_THRESHOLD are returned.
        """
        documents = retriever.invoke(state["question"])
        log.info(f"[retriever] {len(documents)} chunks above threshold")

        return {
            **state,
            "documents": documents,
            "attempts":  state.get("attempts", 0) + 1,
        }

    def node_generator(state: AgentState) -> AgentState:
        """
        Generate an answer grounded in the retrieved chunks.

        Context is built by joining all chunk texts with double newlines.
        The prompt instructs the LLM to stay within the provided context.
        """
        context = "\n\n".join(doc.page_content for doc in state["documents"])
        answer  = generator_chain.invoke({
            "context":  context,
            "question": state["question"],
        })
        log.info("[generator] answer generated")
        return {**state, "generation": answer}

    def node_fallback(state: AgentState) -> AgentState:
        """
        Fallback node: triggered after MAX_RETRIES failed retrieval attempts.

        Returns an explicit "not found" message instead of hallucinating.
        This is a core RAG safety pattern:
            explicit fallback > silent hallucination
        """
        log.warning(f"[fallback] no relevant chunks after {state['attempts']} attempts")
        return {
            **state,
            "generation": (
                "I could not find relevant information in the documents "
                "to answer this question."
            ),
        }

    return {
        "retriever": node_retriever,
        "generator": node_generator,
        "fallback":  node_fallback,
    }


# ── Conditional edge ──────────────────────────────────────────────────────────

def edge_after_retrieval(state: AgentState) -> str:
    """
    Routing logic executed after each retrieval attempt.

    LangGraph calls this function to decide which node to visit next.
    The return value must match a key in the conditional_edges mapping.

    Decision logic:
        - Chunks found          → generate answer
        - No chunks, can retry  → retry retrieval
        - No chunks, exhausted  → fallback
    """
    if len(state["documents"]) > 0:
        return "generator"

    if state.get("attempts", 0) >= MAX_RETRIES:
        log.warning(f"[router] no chunks after {MAX_RETRIES} attempts → fallback")
        return "fallback"

    log.info("[router] no chunks found → retrying")
    return "retriever"


# ── Graph assembly ────────────────────────────────────────────────────────────

def build_agent(retriever: VectorStoreRetriever, model: str = DEFAULT_LLM):
    """
    Assemble and compile the LangGraph agent.

    The compiled graph is a standard Runnable — call with:
        result = agent.invoke({
            "question":   "your question",
            "documents":  [],
            "generation": "",
            "attempts":   0,
        })

    Args:
        retriever: initialized retriever from retriever.py
        model:     Ollama model name

    Returns:
        Compiled LangGraph agent
    """
    llm   = ChatOllama(model=model, temperature=LLM_TEMPERATURE)
    nodes = build_nodes(retriever, llm)

    graph = StateGraph(AgentState)

    # register nodes
    for name, fn in nodes.items():
        graph.add_node(name, fn)

    # entry point
    graph.set_entry_point("retriever")

    # conditional routing after retrieval
    graph.add_conditional_edges(
        "retriever",
        edge_after_retrieval,
        {
            "generator": "generator",
            "retriever": "retriever",
            "fallback":  "fallback",
        },
    )

    # terminal edges
    graph.add_edge("generator", END)
    graph.add_edge("fallback",  END)

    log.info(f"Agent compiled — model={model}, max_retries={MAX_RETRIES}")
    return graph.compile()
