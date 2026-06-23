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

from src.config import DEFAULT_LLM, EMBEDDING_MODEL
from src.logger import get_logger
from src.vectorstore import build_vectorstore
from src.chain import build_chain

log = get_logger(__name__)


def main():
    log.info("Starting RAG chain (Phase 1)")
    vectorstore = build_vectorstore()
    chain       = build_chain(vectorstore)

    print("\n[RAG ready] Type your question (or /bye to exit)\n")

    while True:
        question = input("Question: ").strip()

        if not question:
            continue

        if question.lower() == "/bye":
            print("Shutting down...")
            subprocess.run(["ollama", "stop", DEFAULT_LLM])
            subprocess.run(["ollama", "stop", EMBEDDING_MODEL])
            break

        answer = chain.invoke(question)
        print(f"\nAnswer: {answer}\n")


if __name__ == "__main__":
    main()
