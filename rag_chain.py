"""
rag_chain.py
------------
Phase 1 — Basic RAG pipeline using LangChain.

Architecture (linear chain):
    Documents → Chunking → Embeddings → ChromaDB → Retriever → Prompt → LLM → Answer

This is the simplest RAG implementation: a fixed, sequential pipeline with no
decision logic. Every question goes through the same path regardless of content.

Useful for understanding the core RAG primitives before adding agent logic.
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import glob
import subprocess

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ── Configuration ────────────────────────────────────────────────────────────

DOCS_FOLDER     = "docs/"
VECTORSTORE_DIR = "vectorstore/"
EMBEDDING_MODEL = "nomic-embed-text"  # dedicated embedding model (better than using the LLM)
LLM_MODEL       = "llama3.2"          # generation model
CHUNK_SIZE      = 500                 # max characters per chunk
CHUNK_OVERLAP   = 50                  # overlap between chunks to preserve context
TOP_K_RESULTS   = 4                   # number of chunks to retrieve per query


# ── Step 1: Load documents ───────────────────────────────────────────────────

def load_documents(folder: str = DOCS_FOLDER) -> list:
    """
    Load all PDF and TXT files from a folder recursively.

    LangChain document loaders return a list of Document objects,
    each containing page_content (text) and metadata (source, page, etc.).
    """
    documents = []
    for path in glob.glob(f"{folder}**/*", recursive=True):
        if path.endswith(".pdf"):
            documents.extend(PyPDFLoader(path).load())
        elif path.endswith(".txt"):
            documents.extend(TextLoader(path).load())

    print(f"[load] {len(documents)} pages/documents loaded")
    return documents


# ── Step 2: Split documents into chunks ─────────────────────────────────────

def split_documents(documents: list) -> list:
    """
    Split documents into smaller chunks for embedding.

    Why chunk? LLMs have context limits and embedding quality degrades with long texts.
    RecursiveCharacterTextSplitter tries to split on natural boundaries
    (paragraphs → sentences → words) before falling back to character splits.

    chunk_overlap ensures that context around boundaries is not lost —
    important when an answer spans two adjacent chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    print(f"[split] {len(chunks)} chunks generated")
    return chunks


# ── Step 3: Build or load the vector store ──────────────────────────────────

def build_vectorstore() -> Chroma:
    """
    Create or load a ChromaDB vector store.

    On first run: loads documents, splits them, generates embeddings, and persists to disk.
    On subsequent runs: loads existing embeddings from disk (fast, no re-indexing).

    Embeddings are dense vector representations of text — semantically similar
    texts will have similar vectors, enabling semantic search.
    """
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma(
        persist_directory=VECTORSTORE_DIR,
        embedding_function=embeddings,
    )

    if vectorstore._collection.count() == 0:
        print("[vectorstore] Empty — indexing documents...")
        documents = load_documents()
        chunks    = split_documents(documents)
        vectorstore.add_documents(chunks)
        print(f"[vectorstore] {len(chunks)} chunks indexed and persisted")
    else:
        print(f"[vectorstore] Loaded from disk ({vectorstore._collection.count()} chunks)")

    return vectorstore


# ── Step 4: Build the RAG chain ──────────────────────────────────────────────

def build_chain(vectorstore: Chroma):
    """
    Build the RAG chain: retriever | prompt | LLM | output parser.

    LangChain uses the | (pipe) operator to compose runnables — similar to Unix pipes.
    Each component receives the output of the previous one.

    RunnablePassthrough passes the original input (question) unchanged
    so it's available alongside the retrieved context in the prompt.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K_RESULTS})

    prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the question based ONLY on the context below.
If the answer is not in the context, say you don't know.

Context:
{context}

Question: {question}
""")

    llm = ChatOllama(model=LLM_MODEL, temperature=0)

    def format_docs(docs: list) -> str:
        """Join retrieved chunks into a single context string."""
        return "\n\n".join(doc.page_content for doc in docs)

    # Chain composition using LCEL (LangChain Expression Language)
    # 1. Build a dict with context (retrieved+formatted) and question (passed through)
    # 2. Feed into the prompt template
    # 3. Send to the LLM
    # 4. Parse the output as a plain string
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


# ── Main loop ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    vectorstore = build_vectorstore()
    chain       = build_chain(vectorstore)

    print("\n[RAG ready] Type your question (or /bye to exit)\n")
    while True:
        question = input("Question: ").strip()

        if question.lower() == "/bye":
            print("Shutting down...")
            subprocess.run(["ollama", "stop", LLM_MODEL])
            subprocess.run(["ollama", "stop", EMBEDDING_MODEL])
            break

        answer = chain.invoke(question)
        print(f"\nAnswer: {answer}\n")
