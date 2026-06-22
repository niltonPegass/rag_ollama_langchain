"""
rag_agent.py
------------
Phase 2 — Agentic RAG pipeline using LangGraph.

Architecture (stateful graph):

    question
        │
        ▼
    [retriever] ──► found chunks? ──► [generator] ──► answer
        │                                  
        └── no chunks, attempts < 2 ──► [retriever] (retry)
        └── no chunks, attempts >= 2 ──► [fallback]  ──► answer

Why LangGraph over a plain chain?
- Chains are linear and fixed. LangGraph allows conditional branching and loops.
- State is shared across nodes via AgentState — no need to pass data manually.
- Retries and fallbacks become explicit graph edges, not hidden logic.
- Each node has a clear single responsibility (separation of concerns).

This pattern is the foundation for more complex agents (multi-tool, multi-step).
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import subprocess
from typing import TypedDict, List

from dotenv import load_dotenv
load_dotenv()  # load .env before any LangChain import (required for LangSmith tracing)

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END

# ── Configuration ────────────────────────────────────────────────────────────

VECTORSTORE_DIR  = "vectorstore/"
EMBEDDING_MODEL  = "nomic-embed-text"
AVAILABLE_MODELS = ["llama3.2", "mistral"]
TOP_K_RESULTS    = 6
SCORE_THRESHOLD  = 0.2   # ChromaDB distance threshold (lower = more similar)
                          # 0.0 = accept all, 1.0 = accept none
                          # tune this based on your documents and embedding model
MAX_RETRIES      = 2      # max retrieval attempts before falling back


# ── Model selection ──────────────────────────────────────────────────────────

LLM_MODEL = input(
    f"Choose the LLM model ({', '.join(AVAILABLE_MODELS)}): "
).strip()

if LLM_MODEL not in AVAILABLE_MODELS:
    print(f"Invalid model. Defaulting to {AVAILABLE_MODELS[0]}.")
    LLM_MODEL = AVAILABLE_MODELS[0]


# ── Agent state ──────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    """
    Shared state passed between all graph nodes.

    LangGraph nodes receive the full state dict and return an updated version.
    This replaces the need to pass data explicitly between functions.

    Fields:
        question:   the user's input question
        documents:  chunks retrieved from the vector store
        generation: the final answer (populated by generator or fallback)
        attempts:   number of retrieval attempts (used to prevent infinite loops)
    """
    question:   str
    documents:  List[Document]
    generation: str
    attempts:   int


# ── LLM and retriever setup ──────────────────────────────────────────────────

def load_vectorstore() -> Chroma:
    """Load the existing ChromaDB vector store from disk."""
    embeddings  = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma(
        persist_directory=VECTORSTORE_DIR,
        embedding_function=embeddings,
    )
    print(f"[vectorstore] {vectorstore._collection.count()} chunks loaded")
    return vectorstore


llm = ChatOllama(model=LLM_MODEL, temperature=0)
vs  = load_vectorstore()

# similarity_score_threshold: only return chunks above a similarity threshold
# this avoids feeding irrelevant context to the LLM
retriever = vs.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": TOP_K_RESULTS, "score_threshold": SCORE_THRESHOLD},
)


# ── Graph nodes ──────────────────────────────────────────────────────────────

def node_retriever(state: AgentState) -> AgentState:
    """
    Retrieve relevant document chunks from the vector store.

    Uses semantic search: the question is embedded and compared against
    all stored chunk embeddings using cosine similarity.
    """
    documents = retriever.invoke(state["question"])
    print(f"[retriever] {len(documents)} chunks above threshold")

    state["documents"] = documents
    state["attempts"]  = state.get("attempts", 0) + 1
    return state


# Generator prompt — instructs the LLM to stay grounded in the retrieved context
_generator_prompt = ChatPromptTemplate.from_template("""
You are an assistant that answers questions based on provided documents.
Use ONLY the context below to answer. Be direct and objective.
If the context does not contain the answer, say exactly:
"I could not find this information in the documents."

Context:
{context}

Question: {question}

Answer:""")

_generator_chain = _generator_prompt | llm | StrOutputParser()


def node_generator(state: AgentState) -> AgentState:
    """
    Generate an answer grounded in the retrieved chunks.

    The context is built by joining all retrieved chunk texts.
    Temperature=0 ensures deterministic, factual responses.
    """
    context = "\n\n".join(doc.page_content for doc in state["documents"])
    answer  = _generator_chain.invoke({
        "context":  context,
        "question": state["question"],
    })
    state["generation"] = answer
    return state


def node_fallback(state: AgentState) -> AgentState:
    """
    Fallback node: triggered when retrieval fails after MAX_RETRIES attempts.

    Returns a honest "not found" message instead of hallucinating an answer.
    This is a key RAG safety pattern — explicit fallback > silent hallucination.
    """
    state["generation"] = (
        "I could not find relevant information in the documents to answer this question."
    )
    return state


# ── Conditional edges ────────────────────────────────────────────────────────

def edge_after_retrieval(state: AgentState) -> str:
    """
    Routing logic after retrieval attempt.

    This is a conditional edge — LangGraph calls this function to decide
    which node to visit next based on the current state.

    Returns:
        "generator" — chunks found, proceed to answer generation
        "retriever" — no chunks, retry retrieval
        "fallback"  — no chunks after MAX_RETRIES, give up gracefully
    """
    if len(state["documents"]) > 0:
        return "generator"

    if state.get("attempts", 0) >= MAX_RETRIES:
        print(f"[retriever] No chunks after {MAX_RETRIES} attempts — fallback")
        return "fallback"

    print("[retriever] No chunks found — retrying")
    return "retriever"


# ── Graph assembly ───────────────────────────────────────────────────────────

def build_agent():
    """
    Assemble the LangGraph state machine.

    Nodes are functions that transform AgentState.
    Edges define the flow between nodes.
    Conditional edges use a routing function to decide the next node dynamically.
    """
    graph = StateGraph(AgentState)

    # register nodes
    graph.add_node("retriever", node_retriever)
    graph.add_node("generator", node_generator)
    graph.add_node("fallback",  node_fallback)

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
        }
    )

    # terminal edges
    graph.add_edge("generator", END)
    graph.add_edge("fallback",  END)

    return graph.compile()


agent = build_agent()


# ── Main loop ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n[RAG Agent] Type your question (or /bye to exit)\n")

    while True:
        question = input("Question: ").strip()

        if question.lower() == "/bye":
            print("Shutting down...")
            subprocess.run(["ollama", "stop", LLM_MODEL])
            subprocess.run(["ollama", "stop", EMBEDDING_MODEL])
            break

        result = agent.invoke({
            "question":   question,
            "documents":  [],
            "generation": "",
            "attempts":   0,
        })

        print(f"\nAnswer: {result['generation']}\n")
