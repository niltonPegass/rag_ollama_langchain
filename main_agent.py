"""
main_agent.py
-------------
Entry point — Phase 2: agentic RAG with LangGraph.

Orchestrates the modules in src/ to run the stateful agent loop.
Accepts model selection at startup via CLI prompt.
"""

import subprocess
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv
load_dotenv()

from src.config import AVAILABLE_LLMS, DEFAULT_LLM, EMBEDDING_MODEL
from src.logger import get_logger
from src.vectorstore import build_vectorstore
from src.retriever import build_retriever
from src.agent import build_agent

log = get_logger(__name__)


def select_model() -> str:
    """Prompt the user to select an LLM model at startup."""
    options = ", ".join(AVAILABLE_LLMS)
    choice  = input(f"Choose model ({options}) [default: {DEFAULT_LLM}]: ").strip()

    if choice not in AVAILABLE_LLMS:
        if choice:
            print(f"Unknown model '{choice}' — using default: {DEFAULT_LLM}")
        return DEFAULT_LLM

    return choice


def main():
    model = select_model()
    log.info(f"Starting RAG agent (Phase 2) — model={model}")

    vectorstore = build_vectorstore()
    retriever   = build_retriever(vectorstore)
    agent       = build_agent(retriever, model=model)

    print("\n[RAG Agent] Type your question (or /bye to exit)\n")

    while True:
        question = input("Question: ").strip()

        if not question:
            continue

        if question.lower() == "/bye":
            print("Shutting down...")
            subprocess.run(["ollama", "stop", model])
            subprocess.run(["ollama", "stop", EMBEDDING_MODEL])
            break

        result = agent.invoke({
            "question":   question,
            "documents":  [],
            "generation": "",
            "attempts":   0,
        })

        print(f"\nAnswer: {result['generation']}\n")


if __name__ == "__main__":
    main()
