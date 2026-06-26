# Building RAG Systems from Scratch
## Part 4 of 8 — Building the Basic RAG Chain (Phase 1)

**Series:** Building RAG Systems from Scratch  
**Part:** 4 of 8  
**Covers:** Chapter 7 — Phase 1 in full detail  
**Previous:** Part 3 — Document Loading, Chunking, and LangChain  
**Next:** Part 5 — LangGraph: From Chains to Agents

---

## Chapter 7 — Building the Basic RAG Chain (Phase 1)

This chapter traces through every file involved in Phase 1, line by line.
By the end, you will understand not just *what* the code does,
but *why* each decision was made.

The files we cover:
- `src/config.py` — all settings in one place
- `src/prompts.py` — the prompt templates
- `src/vectorstore.py` — building and loading ChromaDB
- `src/retriever.py` — the retriever factory
- `src/chain.py` — the LCEL chain
- `main_chain.py` — the entry point

---

### 7.1 Starting with config.py — everything tuneable in one place

The first file to understand is `src/config.py`. Every other module imports from it.

```python
from pathlib import Path

ROOT_DIR        = Path(__file__).resolve().parent.parent
DOCS_DIR        = ROOT_DIR / "docs"
VECTORSTORE_DIR = ROOT_DIR / "vectorstore"
LOGS_DIR        = ROOT_DIR / "logs"
```

**Breaking down `Path(__file__).resolve().parent.parent`:**

```
__file__          → the path to the current file: "/home/user/project/src/config.py"
.resolve()        → converts to absolute path, resolves any symlinks
.parent           → the directory containing config.py: "/home/user/project/src"
.parent           → one level up: "/home/user/project"   ← ROOT_DIR
```

Then `ROOT_DIR / "docs"` uses pathlib's `/` operator to join path segments:
```python
Path("/home/user/project") / "docs"
# → Path("/home/user/project/docs")
```

This is how we find the `docs/` and `vectorstore/` folders relative to the project root,
regardless of where you run the script from.

**Why not just write `"docs/"` directly?**

If you run `python main_chain.py` from inside the `src/` folder instead of
from the project root, a relative path `"docs/"` would look for `src/docs/`.
Using `__file__` makes paths relative to the file's location, not the
working directory — much more robust.

---

**The model settings:**

```python
EMBEDDING_MODEL = "nomic-embed-text"

AVAILABLE_LLMS  = ["llama3.2", "mistral"]
DEFAULT_LLM     = "llama3.2"

LLM_TEMPERATURE = 0
```

`AVAILABLE_LLMS` is a Python list — an ordered, mutable collection of items.
```python
AVAILABLE_LLMS = ["llama3.2", "mistral"]
AVAILABLE_LLMS[0]   # → "llama3.2"
AVAILABLE_LLMS[1]   # → "mistral"
len(AVAILABLE_LLMS) # → 2
```

It's used in `main_agent.py` to validate user input and build the
selection prompt: `f"Choose model ({', '.join(AVAILABLE_LLMS)})"`.

---

**The chunking settings:**

```python
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50
```

These are used only in `src/loader.py`. If you want to experiment with
different chunk sizes (e.g., `CHUNK_SIZE = 800`), change it here and
delete `vectorstore/` to force re-indexing with the new settings.

---

**The retrieval settings:**

```python
TOP_K_RESULTS   = 6
SCORE_THRESHOLD = 0.2
MAX_RETRIES     = 2
```

`SCORE_THRESHOLD = 0.2` means: "only return chunks whose cosine distance
from the query vector is 0.2 or less." Distance 0.0 = identical, 1.0 = unrelated.
So 0.2 is a fairly strict threshold — only very similar chunks pass.

If your agent keeps falling back with "I could not find this information,"
the first thing to try is raising this value:
```python
SCORE_THRESHOLD = 0.5  # more permissive — returns chunks up to distance 0.5
```

Then re-run `diagnostics.py` to verify what scores your documents produce.

---

### 7.2 prompts.py — the contract with the LLM

```python
from langchain_core.prompts import ChatPromptTemplate

RAG_PROMPT = ChatPromptTemplate.from_template("""
You are an assistant that answers questions based on provided documents.
Use ONLY the context below to answer. Be direct and objective.
If the context does not contain the answer, say exactly:
"I could not find this information in the documents."

Context:
{context}

Question: {question}

Answer:""")
```

**Anatomy of this prompt:**

```
"You are an assistant that answers questions based on provided documents."
↑ Role instruction — tells the model what persona to adopt.
  Models respond differently depending on how you frame their role.

"Use ONLY the context below to answer."
↑ Grounding instruction — the single most important line.
  Without "ONLY", the model will blend retrieved context with training data.
  The word "ONLY" is not decoration — it's load-bearing.

"If the context does not contain the answer, say exactly: ..."
↑ Explicit fallback — prevents hallucination when retrieval fails.
  Without this, the model will try to answer from general knowledge
  even when the retrieved chunks are irrelevant.

"Context:\n{context}"
↑ {context} is a template variable — filled at runtime with the
  retrieved and formatted chunk texts.

"Question: {question}"
↑ {question} is a template variable — filled with the user's question.

"Answer:"
↑ Prompts the model to start generating immediately after this word.
  Without it, the model might add unnecessary preamble like
  "Sure, I'd be happy to help! Based on the context..."
```

**`from_template()` vs `from_messages()`:**

```python
# from_template — creates one human message (simpler, covers most cases)
RAG_PROMPT = ChatPromptTemplate.from_template("Answer: {question}")

# from_messages — explicit control over roles
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You only answer from provided documents."),
    ("human",  "Context: {context}\nQuestion: {question}"),
])
```

We use `from_template()` here. The system instruction is embedded in the
human message, which works well for Ollama-hosted models.

**The `input_variables` property:**

```python
RAG_PROMPT.input_variables
# → ["context", "question"]
```

This is what our prompt tests verify — that the template exposes the
expected variables before we ever run the LLM:
```python
# tests/test_prompts.py
def test_rag_prompt_has_required_variables():
    variables = set(RAG_PROMPT.input_variables)
    assert "context"  in variables
    assert "question" in variables
```

---

### 7.3 vectorstore.py — building and loading ChromaDB

```python
from typing import Optional
from pathlib import Path
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from src.config import CHROMA_COLLECTION, EMBEDDING_MODEL, VECTORSTORE_DIR
from src.loader import load_documents, split_documents
from src.logger import get_logger

log = get_logger(__name__)
```

**`get_logger(__name__)`:**

`__name__` is a special Python variable that contains the module's name.
In `src/vectorstore.py`, `__name__` equals `"src.vectorstore"`.

So this logger is named `"src.vectorstore"` — when it logs a message,
you'll see `src.vectorstore` in the output:
```
10:23:45 | INFO     | src.vectorstore | Loaded existing vector store (247 chunks)
10:23:46 | INFO     | src.chain       | RAG chain ready — model=llama3.2
10:23:47 | INFO     | src.retriever   | Retriever ready — k=6, threshold=0.2
```

This is much more informative than just `print()` — you know exactly which
module logged each message.

---

**The private helper function:**

```python
def _get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=EMBEDDING_MODEL)
```

The leading underscore (`_get_embeddings`) is a Python convention meaning
"this is an internal implementation detail — don't import this from outside."

Python doesn't enforce this — it's a communication tool:
```python
from src.vectorstore import build_vectorstore   # ✓ public API
from src.vectorstore import _get_embeddings     # works, but breaks convention
```

`OllamaEmbeddings` is a LangChain class that wraps Ollama's embedding endpoint.
When you call `OllamaEmbeddings(model="nomic-embed-text")`, nothing happens yet.
The actual HTTP call to Ollama's API only happens when you call:
- `.embed_documents(["text1", "text2"])` — during indexing
- `.embed_query("search query")` — during retrieval

This lazy evaluation means instantiating the embeddings object is fast.

---

**The main function:**

```python
def build_vectorstore(docs_dir: Optional[Path] = None) -> Chroma:
    embeddings  = _get_embeddings()
    vectorstore = Chroma(
        collection_name=CHROMA_COLLECTION,
        persist_directory=str(VECTORSTORE_DIR),
        embedding_function=embeddings,
    )
```

`Chroma(...)` opens a connection to the local ChromaDB database.
This is fast — it doesn't load all vectors into memory.
The database is opened lazily; data is loaded only when needed.

`persist_directory=str(VECTORSTORE_DIR)` — ChromaDB expects a string path,
not a `pathlib.Path` object. `str()` converts it.

`embedding_function=embeddings` — this tells ChromaDB what embedding model
to use both when adding new documents AND when searching. The same model
must be used for both — otherwise the query vector and document vectors
would be in incompatible spaces and search would be meaningless.

---

**The first-run vs subsequent-run logic:**

```python
    count = vectorstore._collection.count()

    if count == 0:
        log.info("Vector store is empty — indexing documents...")
        documents = load_documents()
        chunks    = split_documents(documents)

        if not chunks:
            log.warning("No documents found — vector store remains empty")
            return vectorstore

        vectorstore.add_documents(chunks)
        log.info(f"Indexed {len(chunks)} chunks and persisted to {VECTORSTORE_DIR}")
    else:
        log.info(f"Loaded existing vector store ({count} chunks) from {VECTORSTORE_DIR}")

    return vectorstore
```

`if not chunks:` — in Python, an empty list is "falsy":
```python
bool([])        # → False
bool([doc1])    # → True
if not []:      # → True (the list is empty)
```

So `if not chunks:` means "if the chunks list is empty."
This guards against the case where `docs/` exists but contains no
supported files — without this guard, `add_documents([])` would succeed
silently but the vector store would still be empty.

`vectorstore.add_documents(chunks)` — this is the slow step.
It calls `OllamaEmbeddings.embed_documents()` for every chunk,
which makes one HTTP request to Ollama's embedding endpoint per chunk
(or batched, depending on the implementation).

With 200 chunks and `nomic-embed-text`, expect 20-60 seconds on first run.
On subsequent runs, this entire block is skipped — loading from disk
takes under 1 second.

---

### 7.4 retriever.py — the retriever factory

```python
from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever
from src.config import SCORE_THRESHOLD, TOP_K_RESULTS
from src.logger import get_logger

log = get_logger(__name__)


def build_retriever(vectorstore: Chroma) -> VectorStoreRetriever:
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k":               TOP_K_RESULTS,
            "score_threshold": SCORE_THRESHOLD,
        },
    )
    log.info(f"Retriever ready — k={TOP_K_RESULTS}, threshold={SCORE_THRESHOLD}")
    return retriever
```

**`search_kwargs` — the keyword arguments pattern:**

`search_kwargs` is a Python dict. The name "kwargs" comes from the convention
of using `**kwargs` to pass arbitrary keyword arguments to functions:

```python
def search(query, **kwargs):
    # kwargs is a dict of any extra keyword arguments
    k = kwargs.get("k", 4)
    threshold = kwargs.get("score_threshold", 0.5)
```

Passing `search_kwargs={"k": 6, "score_threshold": 0.2}` is how LangChain
lets you configure the underlying search call without exposing every possible
parameter as a separate argument.

---

**Why `build_retriever` is its own module (not inside chain.py or agent.py):**

Consider what might change about retrieval in the future:
- You might add BM25 hybrid search
- You might add metadata filtering
- You might add a self-querying retriever that uses the LLM to generate filters
- You might add MMR for diversity

All of these changes belong in `retriever.py`. Nothing else needs to change.
This is the Single Responsibility Principle in practice.

---

**`VectorStoreRetriever` — the return type:**

The function signature says it returns `VectorStoreRetriever`.
This is the LangChain type for retrievers that wrap a vector store.

Why specify this in the return type instead of just `Chroma`?
- `VectorStoreRetriever` is a Runnable — it has `.invoke(query)` → `List[Document]`
- Other modules (`chain.py`, `agent.py`) only need to know they can call `.invoke()`
- They don't need to know the underlying implementation is Chroma
- If you switch to Pinecone tomorrow, the return type stays the same — only `vectorstore.py` changes

---

### 7.5 chain.py — the LCEL chain

```python
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama
from src.config import DEFAULT_LLM, LLM_TEMPERATURE
from src.logger import get_logger
from src.prompts import CHAIN_PROMPT

log = get_logger(__name__)
```

---

**`format_docs` — the bridge between retriever output and prompt input:**

```python
def format_docs(docs: list) -> str:
    return "\n\n".join(doc.page_content for doc in docs)
```

The retriever returns `List[Document]`. The prompt needs a plain string for `{context}`.
`format_docs` bridges this gap.

The expression `doc.page_content for doc in docs` is a **generator expression** —
a concise way to transform a sequence:

```python
# Generator expression (memory-efficient):
"\n\n".join(doc.page_content for doc in docs)

# Equivalent list comprehension:
"\n\n".join([doc.page_content for doc in docs])

# Equivalent explicit loop:
texts = []
for doc in docs:
    texts.append(doc.page_content)
"\n\n".join(texts)
```

All three produce the same result. The generator expression is idiomatic Python
for this pattern.

**Why `"\n\n"` as separator?**

A single `"\n"` looks like a line break within a paragraph.
Two `"\n\n"` signals a paragraph break — the LLM treats them as separate passages.
This helps the model distinguish where one retrieved chunk ends and another begins,
which is especially important when the chunks come from different parts of the document.

---

**`build_chain` — assembling the LCEL pipeline:**

```python
def build_chain(vectorstore: Chroma, model: str = DEFAULT_LLM):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    llm       = ChatOllama(model=model, temperature=LLM_TEMPERATURE)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | CHAIN_PROMPT
        | llm
        | StrOutputParser()
    )

    log.info(f"RAG chain ready — model={model}")
    return chain
```

Note: in `chain.py` we use `vectorstore.as_retriever(search_kwargs={"k": 4})`
directly — without the score threshold. Phase 1 is intentionally simple.
The threshold and retry logic are added in Phase 2 via `src/retriever.py`.

---

**Tracing the data flow step by step:**

When you call `chain.invoke("What is the refund policy?")`:

---

**Step 1: RunnableParallel — fan-out**

```python
{
    "context":  retriever | format_docs,
    "question": RunnablePassthrough()
}
```

LangChain sees a dict of Runnables and creates a `RunnableParallel`.
Both branches receive the same input: `"What is the refund policy?"`

Branch 1 — `retriever | format_docs`:
```
"What is the refund policy?"
    → retriever.invoke("What is the refund policy?")
    → embed("What is the refund policy?") → query_vector
    → ChromaDB: find 4 most similar vectors
    → [Document("Section 3.2: Refunds..."), Document("Returns must be..."), ...]
    → format_docs([...])
    → "Section 3.2: Refunds...\n\nReturns must be..."
```

Branch 2 — `RunnablePassthrough()`:
```
"What is the refund policy?"
    → "What is the refund policy?"   (unchanged)
```

Merge:
```python
{
    "context":  "Section 3.2: Refunds...\n\nReturns must be...",
    "question": "What is the refund policy?"
}
```

---

**Step 2: CHAIN_PROMPT — fill the template**

```python
CHAIN_PROMPT.invoke({
    "context":  "Section 3.2: Refunds...\n\nReturns must be...",
    "question": "What is the refund policy?"
})
```

Output:
```
[HumanMessage(content="""
You are a helpful assistant. Answer the question based ONLY on the context below.
If the answer is not in the context, say you don't know.

Context:
Section 3.2: Digital products are non-refundable within 30 days unless the product
is proven defective. To initiate a refund claim, customers must contact support
within 30 days of purchase.

Returns must be initiated by the customer. The company does not proactively process
returns without a customer request.

Question: What is the refund policy?
""")]
```

---

**Step 3: ChatOllama — generate the answer**

```python
llm.invoke([HumanMessage(content="...")])
```

This makes an HTTP POST to `http://localhost:11434/api/chat` with the formatted messages.
Ollama runs the `llama3.2` model and returns the generated text.

Output:
```python
AIMessage(
    content="According to the documents, digital products are non-refundable within 30 days "
            "unless proven defective. To initiate a refund, customers must contact support "
            "within 30 days of purchase. The company requires a customer request to process returns.",
    response_metadata={
        "model": "llama3.2",
        "total_tokens": 412,
        ...
    }
)
```

---

**Step 4: StrOutputParser — extract the text**

```python
StrOutputParser().invoke(AIMessage(content="According to the documents..."))
# → "According to the documents, digital products are non-refundable within 30 days..."
```

This is what `chain.invoke("What is the refund policy?")` returns:
a clean Python string ready to print to the user.

---

### 7.6 main_chain.py — the entry point

```python
import subprocess
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv
load_dotenv()
```

**`warnings.filterwarnings("ignore", category=DeprecationWarning)`:**

LangChain is evolving rapidly. Some functions we use have been replaced
by newer alternatives, and the old ones now show deprecation warnings.
These warnings don't affect functionality — they're notices that the API
may change in a future version.

Suppressing them keeps the output clean. In a production codebase, you'd
address each deprecation by migrating to the newer API.

**`load_dotenv()`:**

This reads the `.env` file and sets the variables as environment variables:

```
# .env file:
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_abc123...
LANGCHAIN_PROJECT=rag-langchain
```

After `load_dotenv()`, these are available as `os.environ["LANGCHAIN_API_KEY"]`.
LangChain reads them automatically to enable LangSmith tracing.

**Critical:** `load_dotenv()` must be called BEFORE any LangChain imports.
LangChain reads environment variables at import time.

---

```python
from src.config import DEFAULT_LLM, EMBEDDING_MODEL
from src.logger import get_logger
from src.vectorstore import build_vectorstore
from src.chain import build_chain

log = get_logger(__name__)
```

Notice the import order:
1. Standard library (`subprocess`, `warnings`)
2. Third-party (`dotenv`)
3. Local project (`src.*`)

This is the PEP 8 convention for Python import ordering.
It makes the dependency structure immediately clear when reading the file.

---

```python
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
```

**`while True:` — the infinite loop:**

`while True:` creates a loop that runs forever until explicitly broken.
The `break` statement exits the loop (triggered by `/bye`).
The `continue` statement skips the rest of the loop body and goes back
to `input()` (triggered by empty input — user pressed Enter with no text).

**`.strip()`:**

```python
question = input("Question: ").strip()
```

`input()` returns whatever the user typed, including leading/trailing whitespace.
`.strip()` removes leading and trailing whitespace:
```python
"  hello world  ".strip()   # → "hello world"
"\n".strip()                 # → ""
```

This handles the case where the user accidentally presses Enter with spaces.
`if not question: continue` then skips to the next iteration.

**`.lower()`:**

```python
if question.lower() == "/bye":
```

`.lower()` converts the string to lowercase:
```python
"/BYE".lower()   # → "/bye"
"/Bye".lower()   # → "/bye"
```

This makes the exit command case-insensitive — the user can type `/bye`, `/BYE`,
or `/Bye` and it all works the same way.

**`subprocess.run(["ollama", "stop", DEFAULT_LLM])`:**

When Ollama runs a model, it keeps the model loaded in RAM for faster
subsequent responses. `ollama stop <model>` unloads the model from RAM.

`subprocess.run()` executes a shell command from Python:
```python
subprocess.run(["ollama", "stop", "llama3.2"])
# equivalent to running in terminal: ollama stop llama3.2
```

The list form `["ollama", "stop", "llama3.2"]` is safer than a string form
`"ollama stop llama3.2"` because it avoids shell injection — the arguments
are passed directly to the OS without shell interpretation.

**`if __name__ == "__main__":`:**

This is the standard Python idiom for "only run this code if this file
is executed directly, not when imported as a module."

```python
# scenario 1: python main_chain.py
# __name__ = "__main__" → main() IS called

# scenario 2: from main_chain import something
# __name__ = "main_chain" → main() is NOT called
```

Without this guard, importing `main_chain` from another file would
immediately start the interactive Q&A loop — undesirable.

---

### 7.7 The complete startup sequence

When you run `python main_chain.py`, here is everything that happens in order:

```
1. warnings.filterwarnings() — suppress deprecation warnings

2. load_dotenv() — read .env → set LANGCHAIN_* environment variables
                   (LangSmith tracing is now active)

3. Import all modules from src/
   (config.py, logger.py, vectorstore.py, chain.py, prompts.py, etc.)

4. main() is called

5. build_vectorstore()
   a. OllamaEmbeddings("nomic-embed-text") — initialize embedding model
   b. Chroma(persist_directory="vectorstore/") — open/connect to ChromaDB
   c. vectorstore._collection.count()
      → 0? → load_documents() → split_documents() → add_documents()
              (calls Ollama API once per chunk — slow on first run)
      → N? → log "Loaded N chunks from disk" (fast)

6. build_chain(vectorstore)
   a. vectorstore.as_retriever() — wrap Chroma in retriever interface
   b. ChatOllama("llama3.2") — initialize LLM client
   c. Compose: {context: retriever|format_docs, question: passthrough}
              | CHAIN_PROMPT | llm | StrOutputParser()
   d. Return the compiled chain

7. Print "[RAG ready]" — pipeline is ready

8. while True loop — wait for user input

9. For each question:
   a. chain.invoke(question)
      → embed question → search ChromaDB → format docs → fill prompt
      → LLM generates response → extract string
   b. print the answer

10. /bye → stop Ollama models → break → program exits
```

---

### 7.8 Phase 1 limitations — why Phase 2 exists

After building Phase 1, you'll notice two problems:

**Problem 1: No quality gate on retrieved chunks**

The chain always retrieves 4 chunks (k=4) and always passes them to the LLM,
regardless of whether they're relevant to the question.

```
User: "What is the meaning of life?"
Retriever: returns 4 chunks (even if they're about "refund policy")
LLM: tries to answer "meaning of life" using refund policy text
Result: confusing, hallucinated answer
```

**Problem 2: No retry logic**

If retrieval returns nothing relevant, the LLM receives an empty context
and either refuses to answer or hallucinates.

**The solution — Phase 2 (LangGraph agent):**

- Add `score_threshold` → only pass chunks that are actually similar
- Add retry loop → if no chunks pass threshold, try again before giving up
- Add fallback node → if retries exhausted, return honest "not found" message

This is exactly what `src/agent.py` implements.
Part 5 covers LangGraph and the Phase 2 agent in full.

---

### Chapter 7 — Summary

**config.py:**
- Single source of truth for all settings
- Uses `pathlib.Path(__file__).resolve().parent.parent` for robust path resolution
- Modifying `SCORE_THRESHOLD` or `CHUNK_SIZE` changes the whole project

**prompts.py:**
- `ChatPromptTemplate.from_template()` creates parameterized prompts
- Three essential elements: role instruction, grounding ("ONLY"), explicit fallback
- Centralizing prompts enables testing and iteration independent of the LLM

**vectorstore.py:**
- `Chroma(persist_directory=...)` opens connection (fast, no data loaded)
- `_collection.count()` checks if data exists without loading everything
- `add_documents(chunks)` embeds + stores (slow on first run, skipped on subsequent)
- `Optional[Path]` parameter enables testing with temporary directories

**retriever.py:**
- `as_retriever()` wraps Chroma as a LangChain Runnable
- `search_type="similarity_score_threshold"` enables quality-gated retrieval
- Isolated module makes swapping retrieval strategies a one-file change

**chain.py:**
- `format_docs()` bridges `List[Document]` → `str` for the prompt's `{context}`
- Dict `{"context": ..., "question": ...}` creates `RunnableParallel` (fan-out)
- `RunnablePassthrough()` passes the original question unchanged to the prompt
- `StrOutputParser()` extracts `.content` from `AIMessage` → plain string

**main_chain.py:**
- `load_dotenv()` must come before all LangChain imports
- `while True:` with `break` and `continue` for the interactive loop
- `subprocess.run(["ollama", "stop", ...])` frees RAM on exit
- `if __name__ == "__main__":` prevents auto-execution when imported

---

*End of Part 4.*  
*Next: Part 5 — LangGraph: From Chains to Agents*  
*Why fixed pipelines aren't enough, and how to build stateful graphs*  
*with conditional routing, retry loops, and explicit fallback behavior.*
