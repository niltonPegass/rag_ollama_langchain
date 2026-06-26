# Building RAG Systems from Scratch
## Part 3 of 8 — Document Loading, Chunking Strategy, and LangChain

**Series:** Building RAG Systems from Scratch  
**Part:** 3 of 8  
**Covers:** Chapters 5 and 6  
**Previous:** Part 2 — Embeddings and Vector Stores  
**Next:** Part 4 — Building the Basic RAG Chain (Phase 1)

---

## Chapter 5 — Document Loading and Chunking Strategy

Chunking is one of the most underestimated parts of a RAG system.
Poor chunking leads to poor retrieval, which leads to poor answers —
regardless of how good your LLM or embedding model is.

The rule is simple: **garbage in, garbage out.**
If the right information isn't in any chunk, no amount of LLM capability
will produce a correct answer.

---

### 5.1 The two-step process

Getting documents into a RAG system always follows the same two steps:

```
Raw files (PDF, TXT, DOCX, ...)
    │
    ▼
Step 1: Load     → convert files into Document objects (structured Python objects)
    │
    ▼
Step 2: Split    → divide each Document into smaller chunks
    │
    ▼
Ready for embedding
```

Both steps are handled in `src/loader.py`.

---

### 5.2 The Document object

LangChain represents all loaded content as `Document` objects.
A `Document` is a simple dataclass with two fields:

```python
from langchain_core.documents import Document

# Structure of a Document:
doc = Document(
    page_content="Section 3.2: Digital products are non-refundable within 30 days...",
    metadata={
        "source": "/home/user/docs/policy.pdf",
        "page":   3,
    }
)

# Accessing the fields:
print(doc.page_content)         # the text content
print(doc.metadata["source"])   # where it came from
print(doc.metadata["page"])     # which page (for PDFs)
```

The `metadata` dict is crucial for traceability — it's what lets you
tell the user "this answer came from `policy.pdf`, page 3."

After chunking, every chunk inherits the metadata from its source document.
This propagation is automatic when you use `split_documents()` (not `split_text()`).

---

### 5.3 Document loaders

LangChain provides "loaders" — objects that know how to read specific file formats
and return a list of `Document` objects.

**PyPDFLoader** — for PDF files:

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("policy.pdf")
docs = loader.load()

# Returns one Document per page:
# docs[0] → Document(page_content="Page 1 text...", metadata={"source": "policy.pdf", "page": 0})
# docs[1] → Document(page_content="Page 2 text...", metadata={"source": "policy.pdf", "page": 1})
# ...
```

**TextLoader** — for plain text files:

```python
from langchain_community.document_loaders import TextLoader

loader = TextLoader("notes.txt", encoding="utf-8")
docs = loader.load()

# Returns one Document for the entire file:
# docs[0] → Document(page_content="All the text...", metadata={"source": "notes.txt"})
```

**Other loaders you'll encounter:**

```python
from langchain_community.document_loaders import (
    WebBaseLoader,      # scrapes a webpage
    CSVLoader,          # one Document per row
    UnstructuredWordDocumentLoader,  # .docx files
    JSONLoader,         # JSON files with jq path selector
    DirectoryLoader,    # loads all files in a folder
)
```

All loaders return the same type: `List[Document]`.
This is the power of LangChain's standard interfaces —
once you have `List[Document]`, the rest of the pipeline doesn't care
whether it came from a PDF, a webpage, or a database.

---

### 5.4 The critical limitation of PyPDFLoader

`PyPDFLoader` extracts text from **text-based PDFs** — PDFs where the text
is stored as actual characters in the file structure.

It does NOT work for **scanned PDFs** — PDFs where each page is an image.
In that case, `page_content` will be empty or contain garbage characters.

```python
docs = PyPDFLoader("scanned_contract.pdf").load()
print(docs[0].page_content)
# → ""   or   "�����"   (garbage)
```

How to check if your PDF is text-based:
- Open it in a PDF reader and try to select/copy text
- If you can select text → text-based → `PyPDFLoader` works
- If you can't select text → scanned image → needs OCR first

For scanned PDFs, common solutions:
- `pytesseract` + `pdf2image` for local OCR
- `UnstructuredPDFLoader` with `strategy="hi_res"` (uses OCR internally)
- Cloud OCR services (AWS Textract, Google Document AI)

In our project, we handle this gracefully with `try/except`:

```python
# src/loader.py
for path in sorted(folder.rglob("*")):
    if path.suffix == ".pdf":
        try:
            docs = PyPDFLoader(str(path)).load()
            documents.extend(docs)
            log.info(f"Loaded PDF: {path.name} ({len(docs)} pages)")
        except Exception as e:
            log.warning(f"Failed to load {path.name}: {e}")
            # continues to next file — one bad file doesn't crash everything
```

The `try/except` per file ensures that one corrupted or incompatible document
doesn't abort the entire indexing process.

---

### 5.5 How we load files in the project

In `src/loader.py`, the `load_documents()` function:

```python
def load_documents(folder: Path = DOCS_DIR) -> List[Document]:
    documents: List[Document] = []

    for path in sorted(folder.rglob("*")):
        #            ^^^^^^^^^^^^^^^^^^^^
        # rglob("*") recursively finds ALL files in the folder and subfolders
        # sorted() ensures consistent ordering across operating systems

        if path.suffix == ".pdf":
            try:
                docs = PyPDFLoader(str(path)).load()
                documents.extend(docs)
                # .extend() adds all items from a list to another list
                # equivalent to: for doc in docs: documents.append(doc)
            except Exception as e:
                log.warning(f"Failed to load {path.name}: {e}")

        elif path.suffix == ".txt":
            try:
                docs = TextLoader(str(path), encoding="utf-8").load()
                documents.extend(docs)
            except Exception as e:
                log.warning(f"Failed to load {path.name}: {e}")

    log.info(f"Total: {len(documents)} pages/documents loaded from {folder}")
    return documents
```

**`Path.rglob("*")`** — this is Python's `pathlib` module.
`rglob` stands for "recursive glob" — it finds all files matching a pattern
in a directory and all its subdirectories. The `"*"` pattern matches everything.

**Why `pathlib.Path` instead of strings?**

```python
# String approach (fragile):
folder = "/home/user/docs"
for file in os.listdir(folder):
    full_path = folder + "/" + file  # breaks on Windows (\)

# pathlib approach (robust):
folder = Path("/home/user/docs")
for file in folder.rglob("*"):
    print(file.suffix)    # ".pdf"
    print(file.name)      # "policy.pdf"
    print(file.stem)      # "policy"
    print(str(file))      # "/home/user/docs/policy.pdf"
```

`pathlib` handles path separators correctly on all operating systems
and provides clean methods for inspecting file properties.

---

### 5.6 Why we split documents

After loading, we have Document objects that may be thousands of characters long.
Entire PDFs might be 50 pages — 50,000+ characters, 60,000+ tokens.

We can't embed entire documents because:

**Problem 1: Context dilution**
An embedding of an entire document captures the "average meaning" of all its content.
A specific fact buried on page 23 gets diluted by everything else.
When searching for that specific fact, the document-level embedding may not
be close enough to the query embedding to be retrieved.

```
Document about: company history, refund policy, shipping, careers, legal notices
Query: "refund policy for digital products"

Document embedding ≈ average of all topics → weak match for specific query
Chunk embedding   ≈ the refund section specifically → strong match
```

**Problem 2: Context window limits**
If we retrieved even 3 full documents, the prompt might be 150,000 tokens —
exceeding the context window of most local models.

**Problem 3: Precision**
When the answer is in a specific paragraph, we want to retrieve that paragraph —
not an entire 50-page document where the user (and the LLM) must hunt for it.

---

### 5.7 RecursiveCharacterTextSplitter — the standard approach

LangChain's `RecursiveCharacterTextSplitter` is the most commonly used splitter.
It's what we use in `src/loader.py`.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # target max characters per chunk
    chunk_overlap=50,    # characters shared between adjacent chunks
    separators=["\n\n", "\n", ". ", " ", ""],
    # ^ tries these in order; falls back to next if chunk is still too large
)
```

**How it works — the recursive algorithm:**

```
Input: a document of 2000 characters
Target: chunks of max 500 characters

Step 1: Try to split on "\n\n" (paragraph breaks)
  → produces pieces of varying sizes

Step 2: For any piece still > 500 chars, try to split on "\n" (line breaks)

Step 3: For any piece still > 500 chars, try to split on ". " (sentences)

Step 4: For any piece still > 500 chars, try to split on " " (words)

Step 5: For any piece still > 500 chars, split on "" (characters — last resort)
```

It tries to split on natural language boundaries first.
Only as an absolute last resort does it split in the middle of a word.

---

### 5.8 Why chunk_overlap matters

Consider this text, split exactly at character 500:

```
chunk 1 ends:   "...The refund policy applies to all orders placed before"
chunk 2 starts: "December 31st. Returns must be initiated within 30 days..."
```

The key fact — "orders placed before December 31st" — is split across chunks.
Without overlap:
- chunk 1 contains an incomplete sentence (no date)
- chunk 2 starts with a date that has no context

Neither chunk contains the complete information needed to answer:
"What is the deadline for the refund policy?"

**With chunk_overlap=50**, chunk 2 starts 50 characters earlier:

```
chunk 2 starts: "orders placed before December 31st. Returns must be initiated..."
```

Now chunk 2 contains the complete fact. The overlap ensures continuity.

**The trade-off:**
Larger overlap → more context preserved at boundaries → more accurate retrieval
Larger overlap → more redundant data stored → larger vector store → slower indexing

50 characters (about half a sentence) is a reasonable default for most documents.
For highly technical content with dense sentences, consider 100-150.

---

### 5.9 Choosing chunk size

There is no universally correct chunk size. The right value depends on
your document structure and the nature of the questions users will ask.

**General guidance:**

| Document type | Recommended chunk_size | Reasoning |
|---|---|---|
| Academic papers | 800–1200 | Dense arguments need full paragraph context |
| Legal documents | 500–800 | Clauses are self-contained but can be verbose |
| FAQ / manuals | 200–400 | Each Q&A pair should be one chunk |
| News articles | 300–600 | Paragraph-length answers |
| Internal wikis | 400–700 | Section-level granularity |
| Code documentation | 500–1000 | Function-level context |

**How to validate your choice:**

```bash
python diagnostics.py
# Choose option 1 (Semantic search)
# Ask a question you know the answer to
# Check: does the returned chunk actually contain the answer?
# Check: is the chunk too long (contains unrelated info)?
# Check: is the chunk too short (truncates the answer)?
```

If the answer is split across two chunks → reduce chunk_overlap is wrong approach,
increase chunk_overlap instead, or reduce chunk_size.

If retrieved chunks contain too much irrelevant surrounding text → reduce chunk_size.

If retrieved chunks are all truncated mid-sentence → chunk_size is too small,
or your documents have very long sentences.

---

### 5.10 Metadata preservation — split_documents vs split_text

LangChain's text splitter has two methods:

```python
# split_text — takes a string, returns a list of strings (NO metadata)
chunks_as_strings = splitter.split_text("some long text...")
# → ["chunk 1 text", "chunk 2 text", ...]

# split_documents — takes Document objects, returns Document objects (WITH metadata)
chunks_as_documents = splitter.split_documents([doc1, doc2, ...])
# → [Document(page_content="chunk 1", metadata={"source": "policy.pdf", "page": 3}),
#    Document(page_content="chunk 2", metadata={"source": "policy.pdf", "page": 3}),
#    ...]
```

**Always use `split_documents()` in a RAG pipeline.**

The metadata is what allows you to:
- Tell the user which file and page the answer came from
- Filter chunks by source (e.g., "only search in `policy.pdf`")
- Debug retrieval issues (see which documents contribute to answers)

In our project:

```python
# src/loader.py
def split_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)   # ← preserves metadata
    log.info(f"Split into {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks
```

---

### 5.11 The complete load + split flow

```python
# What happens when you run main_chain.py or main_agent.py for the first time:

# 1. load_documents() reads all files from docs/
documents = load_documents()
# → [Document(page_content="Full page 1 text...", metadata={"source": "policy.pdf", "page": 0}),
#    Document(page_content="Full page 2 text...", metadata={"source": "policy.pdf", "page": 1}),
#    Document(page_content="Full file content...", metadata={"source": "notes.txt"}),
#    ...]

# 2. split_documents() divides them into chunks
chunks = split_documents(documents)
# → [Document(page_content="Section 1: Introduction...", metadata={"source": "policy.pdf", "page": 0}),
#    Document(page_content="...continuation of intro...", metadata={"source": "policy.pdf", "page": 0}),
#    Document(page_content="Section 2: Definitions...", metadata={"source": "policy.pdf", "page": 0}),
#    ...]
# Note: 2 pages → many chunks, all with correct source metadata

# 3. vectorstore.add_documents() embeds and stores them
vectorstore.add_documents(chunks)
# → For each chunk:
#   a. embed("Section 1: Introduction...") → [0.021, -0.143, ...]
#   b. Store in ChromaDB: text + metadata + vector
```

---

## Chapter 6 — LangChain — The Framework Layer

LangChain is the backbone of our project. It provides:
1. Standard interfaces for all components (LLMs, retrievers, parsers)
2. Composability — connecting components with the `|` operator
3. Integrations — adapters for 100+ LLM providers, vector stores, data sources

Understanding LangChain's design will help you read, extend, and debug the code.

---

### 6.1 What LangChain actually does (and doesn't do)

LangChain does NOT provide AI capabilities.
It orchestrates them.

Think of LangChain as plumbing: it connects components that do the actual work.
The LLM (Ollama/mistral) does the reasoning. ChromaDB does the vector search.
`nomic-embed-text` does the embedding. LangChain connects them all.

```
WITHOUT LangChain:
  embeddings = requests.post("http://localhost:11434/api/embeddings", ...)
  results = chromadb_client.query(embeddings, n_results=4)
  context = "\n".join([r["document"] for r in results])
  response = requests.post("http://localhost:11434/api/chat", json={"messages": [...]})
  answer = response.json()["message"]["content"]

WITH LangChain:
  chain = retriever | format_docs | prompt | llm | StrOutputParser()
  answer = chain.invoke("your question")
```

LangChain handles the API calls, data formatting, error handling, and tracing
so you can focus on the logic.

---

### 6.2 The Runnable interface — the foundation of everything

Every component in LangChain implements the `Runnable` interface.
A Runnable is any object that has an `.invoke(input)` method.

```python
# All of these are Runnables — they all have .invoke():
llm.invoke("hello")                         # → AIMessage(content="Hello!")
retriever.invoke("refund policy")           # → [Document, Document, ...]
prompt.invoke({"question": "?", "context": "..."})  # → [HumanMessage(...)]
StrOutputParser().invoke(ai_message)        # → "the answer text"
```

This standard interface is what makes composition possible.
If everything has `.invoke()`, you can chain them together.

---

### 6.3 LCEL — LangChain Expression Language

LCEL uses Python's `|` operator to chain Runnables.

In Python, the `|` operator normally means "bitwise OR" for integers.
LangChain overrides this behavior for Runnable objects by implementing
the `__or__` magic method:

```python
# Under the hood (simplified):
class Runnable:
    def __or__(self, other):
        return RunnableSequence(first=self, last=other)
        # Creates a new Runnable that calls self, then passes output to other
```

So when you write:
```python
chain = retriever | format_docs | prompt | llm | StrOutputParser()
```

Python actually calls:
```python
chain = retriever.__or__(format_docs).__or__(prompt).__or__(llm).__or__(StrOutputParser())
```

Which creates a `RunnableSequence` that:
1. Calls `retriever.invoke(input)` → output1
2. Calls `format_docs(output1)` → output2
3. Calls `prompt.invoke(output2)` → output3
4. Calls `llm.invoke(output3)` → output4
5. Calls `StrOutputParser().invoke(output4)` → final output

---

### 6.4 The fan-out pattern — RunnablePassthrough and RunnableParallel

Our RAG chain needs to pass the question to two places simultaneously:
- The retriever (to find relevant chunks)
- The prompt (as the "Question:" field)

A linear chain can only pass data in one direction. We need a "fan-out":

```
question ──┬──► retriever ──► format_docs ──► context ──┐
           │                                             ├──► prompt ──► llm
           └──► passthrough ──────────────── question ──┘
```

This is achieved with a dict of Runnables:

```python
# src/chain.py
chain = (
    {
        "context":  retriever | format_docs,  # branch 1: retrieve + format
        "question": RunnablePassthrough(),    # branch 2: pass question unchanged
    }
    | CHAIN_PROMPT   # receives {"context": "...", "question": "..."}
    | llm
    | StrOutputParser()
)
```

When LangChain encounters a dict of Runnables, it creates a `RunnableParallel`:
- Both branches receive the SAME input (the original question string)
- Both branches run (potentially in parallel)
- Their outputs merge into a dict with the specified keys
- That dict feeds the next step (the prompt)

**`RunnablePassthrough()`** is the identity function — it returns its input unchanged.
It's used here to "route" the original question to the prompt as `"question"`.

```python
RunnablePassthrough().invoke("What is the refund policy?")
# → "What is the refund policy?"   (exactly the same, unchanged)
```

---

### 6.5 ChatPromptTemplate — structured prompts

A prompt template is a parameterized string. Variables in `{curly_braces}` are
filled at runtime with `.invoke({"variable": "value"})`.

```python
from langchain_core.prompts import ChatPromptTemplate

# Define the template with variables
template = ChatPromptTemplate.from_template("""
Answer the question based ONLY on the context below.

Context: {context}
Question: {question}
""")

# Fill the variables at runtime
filled = template.invoke({
    "context":  "Digital products are non-refundable...",
    "question": "Can I return a digital product?",
})

# filled is a list of message objects:
# [HumanMessage(content="\nAnswer the question...\n\nContext: Digital products...\nQuestion: Can I return...")]
```

**`from_template()` vs `from_messages()`:**

```python
# from_template: single human message (simpler)
template = ChatPromptTemplate.from_template("Answer: {question}")

# from_messages: explicit control over roles (more control)
template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that only uses provided context."),
    ("human",  "Context: {context}\nQuestion: {question}"),
])
```

For most RAG use cases, `from_template()` is sufficient.
Use `from_messages()` when you need strict separation of system instructions
from user content (e.g., for multi-turn conversations).

**In our project — `src/prompts.py`:**

```python
# RAG_PROMPT — used in the agent (Phase 2)
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

Notice the "If the context does not contain the answer..." instruction.
This is the single most important line for reducing hallucination.
Without it, the LLM will try to answer even when the context is irrelevant.

---

### 6.6 StrOutputParser — extracting the text

LLMs return `AIMessage` objects, not plain strings:

```python
response = llm.invoke([HumanMessage(content="Hello")])
print(type(response))    # <class 'langchain_core.messages.ai.AIMessage'>
print(response)          # AIMessage(content='Hello! How can I help?', response_metadata={...})
print(response.content)  # 'Hello! How can I help?'
```

The `AIMessage` object contains:
- `.content` — the actual text
- `.response_metadata` — tokens used, model name, finish reason, etc.
- `.id` — a unique identifier for the message

`StrOutputParser()` extracts just the `.content` field:

```python
from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()
text = parser.invoke(response)
# → "Hello! How can I help?"  (plain string, not AIMessage)
```

Without `StrOutputParser()` at the end of your chain, `chain.invoke()` returns
an `AIMessage` object. You'd have to call `.content` manually every time.
Adding the parser makes the chain return a clean string directly.

---

### 6.7 ChatOllama — the local LLM interface

`ChatOllama` is LangChain's adapter for Ollama's API.
It implements the same interface as `ChatOpenAI`, `ChatAnthropic`, etc.

```python
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="mistral",    # must match an installed ollama model
    temperature=0,      # 0 = deterministic
    # base_url="http://localhost:11434"  ← default, no need to specify
)

# It's a Runnable — use .invoke() directly or in a chain
response = llm.invoke("What is 2+2?")
# → AIMessage(content="2+2 equals 4.", ...)
```

**Why `ChatOllama` and not just `requests.post()` to Ollama's API?**

1. **Standard interface:** it works in LCEL chains with `|`
2. **Automatic tracing:** LangSmith traces every call automatically
3. **Retry logic:** built-in retry on transient errors
4. **Streaming support:** `.stream()` for token-by-token output
5. **Portability:** swap to `ChatOpenAI` by changing one line and one import

---

### 6.8 The logging module — replacing print statements

In `src/logger.py`, we replace `print()` statements with Python's `logging` module.

**Why `logging` instead of `print()`?**

```python
# print() — simple but limited:
print("[vectorstore] 247 chunks loaded")
# - always outputs to stdout
# - always shows, even in tests
# - no timestamps
# - no severity levels
# - can't redirect to a file without shell tricks

# logging — production-ready:
log.info("Vectorstore loaded")
# - configurable output (stdout, file, both)
# - can be silenced in tests
# - timestamps + severity level
# - hierarchical loggers (src.vectorstore, src.agent, etc.)
# - file rotation support
```

**Log levels — severity hierarchy:**

```python
log.debug("Detailed diagnostic info")     # only in log file, not console
log.info("Normal operational messages")   # console + log file
log.warning("Something unexpected")       # console + log file
log.error("A failure occurred")           # console + log file
log.critical("System cannot continue")    # console + log file
```

**Our logger setup in `src/logger.py`:**

```python
def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    # name = __name__ from the calling module
    # produces loggers like: src.vectorstore, src.agent, src.loader

    # Console: shows INFO and above (normal operational flow)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)

    # File: shows DEBUG and above (everything, for post-mortem analysis)
    file_handler = logging.FileHandler(LOGS_DIR / "rag.log")
    file_handler.setLevel(logging.DEBUG)

    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger
```

**Usage in every module:**

```python
# src/vectorstore.py
from src.logger import get_logger
log = get_logger(__name__)   # __name__ = "src.vectorstore"

def build_vectorstore():
    log.info("Building vector store...")    # appears in console + file
    log.debug("Using collection: rag_collection")  # appears in file only
```

---

### 6.9 Type hints — making code readable and safe

Throughout our codebase, we use Python type hints:

```python
# Without type hints (ambiguous):
def load_documents(folder):
    ...

# With type hints (clear contract):
def load_documents(folder: Path = DOCS_DIR) -> List[Document]:
    ...
```

Type hints tell you:
- `folder` must be a `Path` object (from `pathlib`)
- If not provided, defaults to `DOCS_DIR` (from `config.py`)
- The function returns `List[Document]` — a list of LangChain Document objects

**They don't enforce types at runtime** (Python is still dynamically typed),
but they:
- Enable IDE autocomplete and error detection
- Serve as inline documentation
- Allow static analysis tools (`mypy`) to catch bugs before running

**Common type hints you'll see in our code:**

```python
from typing import List, Optional, TypedDict

def build_chain(vectorstore: Chroma, model: str = DEFAULT_LLM):
    # vectorstore must be a Chroma instance
    # model must be a string, defaults to DEFAULT_LLM

def load_vectorstore() -> Chroma:
    # this function always returns a Chroma instance

def semantic_search(term: str, k: int = 6) -> None:
    # term is a string, k is an int (default 6), returns nothing
```

---

### 6.10 Optional and default arguments

In Python, function parameters can have default values:

```python
def load_documents(folder: Path = DOCS_DIR) -> List[Document]:
    ...
```

This means:
```python
load_documents()              # uses DOCS_DIR from config.py
load_documents(Path("/tmp"))  # uses /tmp instead
```

`Optional[Path]` means the parameter can be either a `Path` or `None`:

```python
from typing import Optional

def build_vectorstore(docs_dir: Optional[Path] = None) -> Chroma:
    if docs_dir:
        documents = load_documents(docs_dir)  # use the provided path
    else:
        documents = load_documents()           # use default from config
```

You'll see this pattern in `src/vectorstore.py` — it allows tests to pass
a temporary directory while the main application uses the configured `DOCS_DIR`.

---

### Chapter 5 & 6 — Summary

**Document loading:**
- LangChain loaders convert files into `Document` objects (text + metadata)
- `PyPDFLoader` works only for text-based PDFs — not scanned images
- Always use `try/except` per file so one bad document doesn't abort everything
- `pathlib.Path` and `rglob()` handle file discovery cleanly across operating systems

**Chunking:**
- Split documents because: context dilution, context window limits, precision
- `RecursiveCharacterTextSplitter` splits on natural boundaries (paragraphs → sentences → words)
- `chunk_overlap` prevents losing context at chunk boundaries
- Use `split_documents()` (not `split_text()`) to preserve metadata
- Validate chunk size with `diagnostics.py` — there's no universal correct value

**LangChain:**
- It's plumbing — connects components, doesn't provide AI capabilities
- Everything implements `Runnable` with `.invoke()` — the basis of composition
- LCEL uses `|` to chain Runnables: `retriever | format_docs | prompt | llm | parser`
- `RunnablePassthrough` + dict → fan-out pattern for passing data to multiple branches
- `ChatPromptTemplate` for parameterized prompts; `StrOutputParser` for clean string output
- `logging` module replaces `print()` — configurable, filterable, file-writable
- Type hints document function signatures and enable IDE support

**The pipeline now:**

```
docs/ ──► load_documents() ──► split_documents()
                                    │
                                    ▼
                            ChromaDB (vectorstore/)
                                    │
                                    ▼ (at query time)
question ──► retriever | format_docs ──► context
question ──► RunnablePassthrough()  ──► question
                    │
                    ▼
              ChatPromptTemplate ──► [HumanMessage with context + question]
                    │
                    ▼
              ChatOllama (mistral) ──► AIMessage(content="the answer")
                    │
                    ▼
              StrOutputParser ──► "the answer"
```

This is the complete Phase 1 pipeline.
Part 4 walks through the actual code that implements it.

---

*End of Part 3.*  
*Next: Part 4 — Building the Basic RAG Chain (Phase 1)*  
*Tracing every line of `src/chain.py` and `main_chain.py` with full understanding.*
