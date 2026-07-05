"""
main_chain.py
-------------
Entry point — Phase 1: basic RAG chain.

Orchestrates the modules in src/ to run a simple Q&A loop.
No business logic here — just wiring and the interactive loop.
"""

import subprocess
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv
load_dotenv()

from src.config import AVAILABLE_LLMS, DEFAULT_LLM, EMBEDDING_MODEL
from src.logger import get_logger
from src.vectorstore import build_vectorstore
from src.chain import build_chain

log = get_logger(__name__)


def select_model_and_retrain() -> tuple[str, bool]:
    """Prompt the user to select an LLM model and ask if they want to retrain the RAG database."""
    options = ", ".join(AVAILABLE_LLMS)
    choice  = input(f"Choose model ({options}) [default: {DEFAULT_LLM}]: ").strip()

    if choice not in AVAILABLE_LLMS:
        if choice:
            print(f"Unknown model '{choice}' — using default: {DEFAULT_LLM}")
        model = DEFAULT_LLM
    else:
        model = choice

    retrain_choice = input("Do you want to retrain the RAG database from the 'docs' folder? (y/N): ").strip().lower()
    retrain = retrain_choice in ("y", "yes")

    return model, retrain


def main():
    model, retrain = select_model_and_retrain()
    log.info(f"Starting RAG chain (Phase 1) — model={model}")
    vectorstore = build_vectorstore(force_retrain=retrain)
    chain       = build_chain(vectorstore, model=model)

    print("\n[RAG ready] Type your question (or /bye to exit)\n")

    while True:
        question = input("Question: ").strip()

        if not question:
            continue

        if question.lower() == "/bye":
            print("Shutting down...")
            subprocess.run(["ollama", "stop", model])
            subprocess.run(["ollama", "stop", EMBEDDING_MODEL])
            break

        answer = chain.invoke(question)
        print(f"\nAnswer: {answer}\n")


if __name__ == "__main__":
    main()
