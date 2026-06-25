# Building RAG Systems from Scratch
## Part 7 of 8 — Project Architecture and Testing

**Series:** Building RAG Systems from Scratch  
**Part:** 7 of 8  
**Covers:** Chapters 12 and 13  
**Previous:** Part 6 — LangSmith and Langflow  
**Next:** Part 8 — Debugging RAG Systems and What to Build Next

---

## Chapter 12 — Project Architecture: Why We Modularized

Before modularization, the project was three files:
`rag_chain.py`, `rag_agent.py`, and `diagnostics.py`.
Each file mixed configuration, loading, embedding, retrieval, prompting,
and orchestration together.

After modularization, every concept has its own home:

```
src/
├── config.py       ← all settings
├── logger.py       ← logging infrastructure
├── loader.py       ← document loading + chunking
├── vectorstore.py  ← ChromaDB build/load
├── retriever.py    ← retriever factory
├── prompts.py      ← all prompt templates
├── chain.py        ← Phase 1 LCEL chain
└── agent.py        ← Phase 2 LangGraph agent

main_chain.py       ← entry point (Phase 1)
main_agent.py       ← entry point (Phase 2)
diagnostics.py      ← debugging utilities

tests/
├── test_loader.py  ← tests for loader.py
├── test_prompts.py ← tests for prompts.py
└── test_agent.py   ← tests for agent.py
```

This chapter explains the design principles behind this structure
and why each decision was made.

---

### 12.1 Single Responsibility Principle

The most important principle in software design: **each module should have
one reason to change.**

Consider what would need to change in each scenario:

| If you want to... | Change only... |
|---|---|
| Switch from llama3.2 to mistral | `src/config.py` |
| Add support for .docx files | `src/loader.py` |
| Switch from ChromaDB to Pinecone | `src/vectorstore.py` |
| Add BM25 hybrid search | `src/retriever.py` |
| Improve the prompt | `src/prompts.py` |
| Add a query-rewriting step | `src/agent.py` |
| Change the logging format | `src/logger.py` |

If a change requires editing multiple files, the responsibility isn't
separated correctly. The Single Responsibility Principle makes code
maintainable as it grows.

---

### 12.2 Dependency direction — the import hierarchy

In our project, dependencies flow in one direction only:

```
main_chain.py ──► src/chain.py ──► src/prompts.py
                              ──► src/vectorstore.py ──► src/loader.py
                                                     ──► src/config.py
                              ──► src/config.py

main_agent.py ──► src/agent.py ──► src/prompts.py
                              ──► src/config.py
              ──► src/retriever.py ──► src/config.py
              ──► src/vectorstore.py ──► src/loader.py
```

**Higher-level modules depend on lower-level modules. Never the reverse.**

`src/config.py` never imports from `src/agent.py`.
`src/loader.py` never imports from `src/vectorstore.py`.

Why does this matter?

```python
# BAD — circular import:
# src/vectorstore.py imports from src/chain.py
# src/chain.py imports from src/vectorstore.py
# Python can't resolve this — ImportError

# GOOD — one direction:
# src/chain.py imports from src/vectorstore.py
# src/vectorstore.py imports from src/config.py and src/loader.py
# No cycles — Python resolves imports cleanly
```

Circular imports cause `ImportError` at startup and are a common source of
confusing bugs in Python projects.

---

### 12.3 Configuration centralization — why `config.py` exists

**The problem without a config file:**

```python
# loader.py
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# vectorstore.py
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# chain.py
retriever = vs.as_retriever(search_kwargs={"k": 4})
llm = ChatOllama(model="llama3.2", temperature=0)

# agent.py
retriever = vs.as_retriever(search_kwargs={"k": 6, "score_threshold": 0.2})
llm = ChatOllama(model="llama3.2", temperature=0)
```

The embedding model name appears in two files. The LLM model appears in two files.
When you want to switch to `mistral`, you search for `"llama3.2"` across all files
and hope you found every instance.

**The solution — everything in one place:**

```python
# src/config.py
EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_LLM     = "llama3.2"
CHUNK_SIZE      = 500
SCORE_THRESHOLD = 0.2
```

```python
# loader.py
from src.config import CHUNK_SIZE
splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE)

# vectorstore.py
from src.config import EMBEDDING_MODEL
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

# agent.py
from src.config import DEFAULT_LLM, SCORE_THRESHOLD
llm = ChatOllama(model=DEFAULT_LLM)
```

Change `DEFAULT_LLM = "mistral"` in one file — the whole project updates.

---

### 12.4 UPPER_SNAKE_CASE — Python constants convention

```python
# src/config.py
EMBEDDING_MODEL = "nomic-embed-text"   # constant
CHUNK_SIZE      = 500                  # constant
MAX_RETRIES     = 2                    # constant
```

`UPPER_SNAKE_CASE` (all caps, words separated by underscores) is the
Python convention for constants — values that are set once and not changed
at runtime.

Python does NOT enforce this. You could write `embedding_model = "nomic-embed-text"`
and it would work identically. The capitalization is a **communication convention**:

```python
# Reading code:
llm = ChatOllama(model=DEFAULT_LLM)   # "DEFAULT_LLM" → this is a project-wide constant
llm = ChatOllama(model=default_llm)   # "default_llm" → this might be a local variable
```

The capital letters signal to anyone reading the code:
"this value comes from a centralized config, not a local calculation."

---

### 12.5 Entry points vs modules — why main_*.py files are thin

`main_chain.py` and `main_agent.py` contain almost no logic.
They only wire modules together and run the I/O loop:

```python
# main_agent.py — the entire business logic is delegated to src/
def main():
    model       = select_model()              # user input
    vectorstore = build_vectorstore()         # src/vectorstore.py
    retriever   = build_retriever(vectorstore) # src/retriever.py
    agent       = build_agent(retriever, model) # src/agent.py

    while True:
        question = input("Question: ").strip()
        result   = agent.invoke({...})
        print(result["generation"])
```

This is the **Separation of Concerns** principle:
- Business logic lives in `src/` (testable, reusable, importable)
- I/O and orchestration lives in `main_*.py` (thin, readable, replaceable)

**Why does this matter in practice?**

Tomorrow you want to build a FastAPI web server instead of a terminal interface.
You create `main_api.py`:

```python
# main_api.py — new entry point, same src/ modules
from fastapi import FastAPI
from src.vectorstore import build_vectorstore
from src.retriever import build_retriever
from src.agent import build_agent

app = FastAPI()
agent = build_agent(build_retriever(build_vectorstore()))

@app.post("/ask")
def ask(question: str):
    result = agent.invoke({"question": question, "documents": [], "generation": "", "attempts": 0})
    return {"answer": result["generation"]}
```

All the logic in `src/` is reused unchanged. Only the entry point changes.
This is only possible because the entry point doesn't contain business logic.

---

### 12.6 The `__init__.py` file — making a directory a package

```
src/
├── __init__.py   ← this file
├── config.py
└── ...
```

`__init__.py` is what makes `src/` a Python package rather than just a folder.

Without it:
```python
from src.config import CHUNK_SIZE
# ModuleNotFoundError: No module named 'src'
```

With it:
```python
from src.config import CHUNK_SIZE  # ✓ works
```

The `__init__.py` file is usually empty (or contains a brief comment).
Its mere existence is what signals to Python: "this directory is a package."

**What is a Python package?**

A package is a directory that Python can import from.
It can contain modules (`.py` files) and sub-packages (nested directories
with their own `__init__.py`).

```
rag-langchain/     ← not a package (no __init__.py)
└── src/           ← is a package (has __init__.py)
    ├── __init__.py
    ├── config.py  ← is a module (importable as src.config)
    └── agent.py   ← is a module (importable as src.agent)
```

---

### 12.7 Explicit dependency injection — how build functions work

Every `build_*` function in our project takes its dependencies as arguments:

```python
def build_chain(vectorstore: Chroma, model: str = DEFAULT_LLM):
    retriever = vectorstore.as_retriever(...)  # uses the passed vectorstore
    llm       = ChatOllama(model=model)        # uses the passed model name
    ...

def build_retriever(vectorstore: Chroma) -> VectorStoreRetriever:
    return vectorstore.as_retriever(...)

def build_agent(retriever: VectorStoreRetriever, model: str = DEFAULT_LLM):
    llm   = ChatOllama(model=model)
    nodes = build_nodes(retriever, llm)   # passes them as arguments
    ...
```

This is called **dependency injection** — instead of having functions
create their own dependencies internally or read from global variables,
they receive what they need as arguments.

**Why dependency injection?**

```python
# WITHOUT dependency injection (hard to test):
def build_chain():
    vectorstore = Chroma(persist_directory="vectorstore/")  # hardcoded!
    llm = ChatOllama(model="llama3.2")                      # hardcoded!
    ...

# In tests: can't pass a different vectorstore or model without monkeypatching

# WITH dependency injection (easy to test):
def build_chain(vectorstore: Chroma, model: str = DEFAULT_LLM):
    llm = ChatOllama(model=model)
    ...

# In tests:
mock_vectorstore = create_in_memory_chroma()
chain = build_chain(mock_vectorstore, model="llama3.2")
# Full control — no external services needed
```

---

## Chapter 13 — Testing LLM Applications

Testing is where LLM applications diverge most sharply from traditional software.

The core challenge: **LLM outputs are non-deterministic.**

Even at temperature = 0, exact outputs can change when:
- The model is updated
- The underlying CUDA/CPU libraries change
- The prompt is modified
- The context changes

This means you cannot write tests like:
```python
assert llm.invoke("What is 2+2?") == "2+2 equals 4."
# This will fail when the model adds "!" or "The answer is" preamble
```

---

### 13.1 The testing hierarchy for LLM applications

Instead of testing LLM outputs, we test the deterministic parts of our system:

```
Level 1: Unit tests (what we have)
  → Pure Python logic: routing functions, prompt templates, chunking behavior
  → No LLMs, no ChromaDB, no Ollama required
  → Runs in milliseconds, always deterministic

Level 2: Integration tests (not in this project, but worth knowing)
  → Tests that require Ollama running and ChromaDB loaded
  → "Does the full pipeline return a non-empty answer for this question?"
  → Slower, requires infrastructure, still avoid testing exact LLM output

Level 3: Evaluation (LangSmith datasets)
  → "Is the answer correct?" judged by LLM-as-judge or human review
  → Not deterministic — used for quality tracking, not pass/fail CI
  → Runs periodically, not on every commit
```

Our `tests/` folder covers Level 1 only.
This is the correct scope for a portfolio project —
it demonstrates testing discipline without over-engineering.

---

### 13.2 pytest — Python's testing framework

pytest is the standard testing framework for Python.
It discovers and runs functions whose names start with `test_`.

```python
# A minimal test:
def test_addition():
    assert 1 + 1 == 2

# Run it:
# pytest tests/test_example.py
# → PASSED
```

**`assert` — the testing primitive:**

`assert expression` raises `AssertionError` if the expression is `False`.

```python
assert 1 + 1 == 2     # passes silently
assert 1 + 1 == 3     # raises AssertionError: assert 2 == 3
assert len([]) == 0   # passes silently
assert "hello" in "hello world"  # passes silently
```

pytest intercepts `AssertionError` and displays a detailed failure message
showing the actual vs expected values.

**pytest configuration — `pytest.ini`:**

```ini
[pytest]
testpaths = tests          ← only look for tests in the tests/ directory
python_files = test_*.py   ← only files starting with test_
python_classes = Test*     ← only classes starting with Test
python_functions = test_*  ← only functions starting with test_
addopts = -v --tb=short    ← always run verbose with short tracebacks
```

`addopts` means "additional options" — flags added to every `pytest` invocation:
- `-v` (verbose): show each test name and PASSED/FAILED
- `--tb=short`: show short tracebacks on failure (not the full stack)

Running the tests:
```bash
pytest                       # run all tests
pytest tests/test_agent.py   # run one file
pytest -k "test_loader"      # run tests matching a pattern
pytest -v                    # verbose output (already in addopts)
```

---

### 13.3 Fixtures — reusable test setup

A **fixture** is a function decorated with `@pytest.fixture` that provides
setup data or objects to tests.

```python
# tests/test_loader.py

@pytest.fixture
def sample_txt_file(tmp_path: Path) -> Path:
    """Create a temporary .txt file with known content."""
    file = tmp_path / "sample.txt"
    file.write_text(
        "This is the first paragraph about artificial intelligence.\n\n"
        "This is the second paragraph about machine learning.\n\n"
        "This is the third paragraph about neural networks.",
        encoding="utf-8",
    )
    return tmp_path  # return the folder, not the file
```

**How fixtures are used:**

When a test function has a parameter with the same name as a fixture,
pytest automatically calls the fixture and passes its return value:

```python
def test_load_documents_txt(sample_txt_file: Path):
    #                        ^^^^^^^^^^^^^^^^
    # pytest sees this parameter name, finds the matching fixture,
    # calls sample_txt_file() to get a Path, and passes it here

    docs = load_documents(sample_txt_file)
    assert len(docs) >= 1
```

**`tmp_path` — pytest's built-in fixture:**

`tmp_path` is a built-in pytest fixture that provides a temporary directory
(`pathlib.Path` object) that:
- Is unique for each test function
- Is automatically deleted after the test completes
- Lives in `/tmp/pytest-*/` during the test run

```python
@pytest.fixture
def empty_folder(tmp_path: Path) -> Path:
    return tmp_path  # tmp_path is already empty — nothing to do
```

```python
@pytest.fixture
def unsupported_file(tmp_path: Path) -> Path:
    # Write a fake .docx file (unsupported format)
    (tmp_path / "document.docx").write_bytes(b"fake docx content")
    return tmp_path
```

`write_bytes(b"fake docx content")` — the `b"..."` prefix creates a **bytes literal**
(raw bytes, not a string). We use it here because .docx is a binary format,
but the content doesn't matter — we're testing that the loader *ignores* it.

**Why fixtures instead of setup code inside each test?**

```python
# Without fixtures — repeated setup:
def test_load_documents_txt():
    tmp = Path(tempfile.mkdtemp())
    file = tmp / "sample.txt"
    file.write_text("content...")
    docs = load_documents(tmp)
    assert len(docs) >= 1
    shutil.rmtree(tmp)  # manual cleanup

def test_load_documents_metadata():
    tmp = Path(tempfile.mkdtemp())   # repeated setup
    file = tmp / "sample.txt"
    file.write_text("content...")    # repeated setup
    docs = load_documents(tmp)
    for doc in docs:
        assert "source" in doc.metadata
    shutil.rmtree(tmp)  # repeated cleanup

# With fixtures — setup and cleanup handled automatically:
def test_load_documents_txt(sample_txt_file):
    docs = load_documents(sample_txt_file)
    assert len(docs) >= 1

def test_load_documents_metadata(sample_txt_file):
    docs = load_documents(sample_txt_file)
    for doc in docs:
        assert "source" in doc.metadata
```

Fixtures eliminate repeated setup/teardown code and centralize test infrastructure.

---

### 13.4 Testing `loader.py` — what we test and why

**`test_load_documents_txt` — the happy path:**

```python
def test_load_documents_txt(sample_txt_file: Path):
    """Should load text documents from a folder."""
    docs = load_documents(sample_txt_file)
    assert len(docs) >= 1
    assert any("artificial intelligence" in doc.page_content for doc in docs)
```

`any(condition for item in iterable)` — returns `True` if the condition
is `True` for at least one item in the iterable:

```python
# Equivalent to:
found = False
for doc in docs:
    if "artificial intelligence" in doc.page_content:
        found = True
        break
assert found
```

The generator expression inside `any()` is evaluated lazily —
it stops as soon as it finds a `True` case, without checking the rest.

---

**`test_load_documents_empty_folder` — the edge case:**

```python
def test_load_documents_empty_folder(empty_folder: Path):
    """Should return empty list when folder has no supported files."""
    docs = load_documents(empty_folder)
    assert docs == []
```

Why test this? Without this test, someone could accidentally change
`load_documents()` to raise an exception on empty folders instead of
returning an empty list — and the first symptom would be a crash
in production when a user runs the pipeline with no documents.

---

**`test_load_documents_unsupported_files` — graceful degradation:**

```python
def test_load_documents_unsupported_files(unsupported_file: Path):
    """Should silently skip unsupported file types."""
    docs = load_documents(unsupported_file)
    assert docs == []
```

This tests a specific design decision: `.docx` files (and any other
unsupported format) should be silently ignored, not cause an error.
The test documents this behavior and ensures it doesn't regress.

---

**`test_split_documents_respects_chunk_size`:**

```python
def test_split_documents_respects_chunk_size():
    """Each chunk should not exceed CHUNK_SIZE by a large margin."""
    from src.config import CHUNK_SIZE
    long_text = "word " * 1000
    docs      = [Document(page_content=long_text, metadata={"source": "test.txt"})]
    chunks    = split_documents(docs)

    for chunk in chunks:
        assert len(chunk.page_content) <= CHUNK_SIZE * 2
```

Why `CHUNK_SIZE * 2` instead of `CHUNK_SIZE`?

`RecursiveCharacterTextSplitter` uses `chunk_size` as a target, not a hard limit.
Chunks can exceed it slightly due to overlap and the preference for natural
splitting boundaries. We allow 2× slack to test that chunks are roughly
the right size without being brittle about the exact character count.

**This is a key testing principle:** test the behavior (chunks are approximately
the target size), not the implementation detail (exact character count).

---

**`test_split_documents_preserves_metadata`:**

```python
def test_split_documents_preserves_metadata():
    """Chunks should inherit metadata from source document."""
    docs   = [Document(page_content="word " * 200, metadata={"source": "test.txt", "page": 1})]
    chunks = split_documents(docs)

    for chunk in chunks:
        assert chunk.metadata.get("source") == "test.txt"
```

This tests a critical behavior: source metadata must survive splitting.
If someone replaces `split_documents()` with `split_text()` (which loses metadata),
this test fails immediately — catching the bug before it reaches production.

---

### 13.5 Testing `prompts.py` — verifying the contract

Prompt templates are pure Python — no LLM needed.
We test two things: structure and content.

**Testing structure (variables):**

```python
def test_rag_prompt_has_required_variables():
    """RAG_PROMPT must expose {context} and {question} as input variables."""
    variables = set(RAG_PROMPT.input_variables)
    assert "context"  in variables
    assert "question" in variables
```

`RAG_PROMPT.input_variables` is a list of variable names extracted from
the template's `{curly_brace}` placeholders:

```python
# Template: "Context: {context}\nQuestion: {question}"
# RAG_PROMPT.input_variables → ["context", "question"]
```

`set(RAG_PROMPT.input_variables)` converts the list to a set for O(1) `in` checks.

Why test this? If someone renames `{context}` to `{ctx}` in the template,
`chain.invoke({"context": "..."})` will fail with a `KeyError` at runtime.
This test catches that refactoring mistake immediately.

---

**Testing rendering:**

```python
def test_rag_prompt_renders_correctly():
    """RAG_PROMPT should render without errors when given valid inputs."""
    rendered = RAG_PROMPT.format_messages(
        context="Python is a programming language.",
        question="What is Python?",
    )
    full_text = " ".join(m.content for m in rendered)
    assert "Python is a programming language" in full_text
    assert "What is Python?" in full_text
```

`format_messages()` fills the template variables and returns a list of message objects.

`" ".join(m.content for m in rendered)` — joins the content of all messages
into a single string for easy substring checking.

Why test rendering? The template might have a syntax error (e.g., unclosed brace `{context`)
that only manifests at runtime. This test ensures the template is valid.

---

**Testing the grounding instruction:**

```python
def test_rag_prompt_contains_grounding_instruction():
    """RAG_PROMPT must instruct the LLM to use ONLY the provided context."""
    rendered  = RAG_PROMPT.format_messages(context="x", question="y")
    full_text = " ".join(m.content for m in rendered).lower()
    assert "only" in full_text
```

This tests a semantic property of the prompt: it must contain the word "only"
(case-insensitive) as part of the grounding instruction.

This test encodes a design decision as a machine-checkable rule:
"our prompts must always instruct the LLM to use ONLY the provided context."
If someone removes that instruction to "simplify" the prompt, this test fails.

`.lower()` ensures the test doesn't break if "ONLY" becomes "only" or "Only".

---

### 13.6 Testing `agent.py` — the routing logic

This is the most valuable test file. `edge_after_retrieval` is pure Python —
it takes a dict and returns a string. No LLM, no ChromaDB, no Ollama.

**The helper functions:**

```python
def make_state(
    question:   str = "test question",
    documents:  list = None,
    generation: str = "",
    attempts:   int = 0,
) -> AgentState:
    """Build a minimal AgentState for testing."""
    return AgentState(
        question=question,
        documents=documents if documents is not None else [],
        generation=generation,
        attempts=attempts,
    )
```

`documents if documents is not None else []` — this is a Python ternary expression:

```python
# Ternary expression:
value = x if condition else y
# equivalent to:
if condition:
    value = x
else:
    value = y
```

Why not just `documents or []`?

```python
# WRONG:
documents or []
# If documents=[] (empty list, falsy), this returns []
# If documents=[doc1] (non-empty list, truthy), this returns [doc1]
# But if documents=None (the default), this also returns [] ✓
# So far so good... but:
# If documents=0 (edge case), this returns [] even though we passed 0
# The `or` operator is too broad

# CORRECT:
documents if documents is not None else []
# Only returns [] when documents is explicitly None (the default)
# [] (empty list) is returned as-is — which is what the test wants
```

`is not None` checks identity (is this literally the None object?),
not truthiness (is this truthy/falsy?). More precise.

---

**The test cases:**

```python
def test_routes_to_generator_when_chunks_found():
    state = make_state(documents=[make_doc()], attempts=1)
    assert edge_after_retrieval(state) == "generator"
```

Documents present → "generator". The simplest, most common case.

```python
def test_routes_to_retriever_on_first_empty_result():
    state = make_state(documents=[], attempts=1)
    assert edge_after_retrieval(state) == "retriever"
```

No documents, attempts=1 < MAX_RETRIES=2 → retry. Tests the retry branch.

```python
def test_routes_to_fallback_after_max_retries():
    state = make_state(documents=[], attempts=MAX_RETRIES)
    assert edge_after_retrieval(state) == "fallback"
```

No documents, attempts=MAX_RETRIES → fallback. Tests the exhaustion case.
Uses `MAX_RETRIES` from config — if someone changes `MAX_RETRIES = 3`,
this test automatically adapts.

```python
def test_routes_to_fallback_beyond_max_retries():
    state = make_state(documents=[], attempts=MAX_RETRIES + 5)
    assert edge_after_retrieval(state) == "fallback"
```

Tests a defensive edge case: what if `attempts` somehow exceeds `MAX_RETRIES`?
(Shouldn't happen normally, but a future refactor might cause it.)
The routing should still fall back, not enter an unexpected state.

```python
def test_chunks_take_priority_over_attempts():
    state = make_state(documents=[make_doc()], attempts=MAX_RETRIES + 10)
    assert edge_after_retrieval(state) == "generator"
```

This tests an important priority rule: **if chunks were found, always generate —
regardless of how many attempts it took.** The attempt counter should never
prevent the generator from running if good chunks were retrieved.

This is a subtle edge case that documents a design decision:
"attempts" is a guard against infinite loops, not a cap on successful retrieval.

---

### 13.7 What we explicitly don't test

Understanding what we choose NOT to test is as important as what we do test.

**We don't test:**

```python
# LLM output quality:
def test_llm_gives_correct_answer():
    result = llm.invoke("What is the refund policy?")
    assert "30 days" in result.content  # FRAGILE: will fail on model updates
```

**We don't test:**

```python
# Full pipeline integration:
def test_chain_returns_answer():
    vectorstore = build_vectorstore()  # requires Ollama + ChromaDB running
    chain = build_chain(vectorstore)
    answer = chain.invoke("test question")
    assert len(answer) > 0  # requires real LLM call — slow and fragile
```

**We don't test:**

```python
# ChromaDB internals:
def test_vectorstore_stores_correct_vectors():
    vs = build_vectorstore()
    vectors = vs._collection.get(include=["embeddings"])
    # testing internal state of a third-party library — breaks on updates
```

These tests would be:
- **Slow:** requiring Ollama to run (3-10 seconds per LLM call)
- **Brittle:** breaking on model updates, LangChain version changes, or hardware differences
- **Hard to run:** requiring infrastructure (Ollama running, ChromaDB populated)

The tests we wrote are:
- **Fast:** milliseconds (pure Python, no I/O)
- **Stable:** deterministic, don't break on model updates
- **Portable:** run on any machine without Ollama or ChromaDB

This is the right trade-off for a portfolio project.

---

### 13.8 Running the tests

```bash
# All tests
pytest

# One file
pytest tests/test_agent.py

# Verbose (shows each test name)
pytest -v

# Stop on first failure
pytest -x

# Show locals on failure (useful for debugging)
pytest -l

# Run tests matching a pattern
pytest -k "test_split"
```

Expected output:
```
===================== test session starts ======================
tests/test_agent.py::test_routes_to_generator_when_chunks_found  PASSED
tests/test_agent.py::test_routes_to_retriever_on_first_empty_result  PASSED
tests/test_agent.py::test_routes_to_fallback_after_max_retries  PASSED
tests/test_agent.py::test_routes_to_fallback_beyond_max_retries  PASSED
tests/test_agent.py::test_routes_to_generator_with_multiple_chunks  PASSED
tests/test_agent.py::test_chunks_take_priority_over_attempts  PASSED
tests/test_loader.py::test_load_documents_txt  PASSED
tests/test_loader.py::test_load_documents_empty_folder  PASSED
...
tests/test_prompts.py::test_rag_prompt_has_required_variables  PASSED
...
===================== 14 passed in 0.83s =======================
```

14 tests passing in under 1 second. No Ollama required.

---

### Chapter 12 & 13 — Summary

**Architecture (Chapter 12):**

- **Single Responsibility:** each module has one reason to change
- **Dependency direction:** higher-level imports lower-level, never reverse
  (avoids circular imports and keeps the dependency tree clean)
- **Config centralization:** `config.py` as the single source of truth for all settings,
  using `UPPER_SNAKE_CASE` to signal constants
- **`__init__.py`:** makes `src/` a Python package (required for `from src.x import y`)
- **Thin entry points:** `main_*.py` files only wire modules and handle I/O;
  all business logic lives in `src/` (enables reuse across different entry points)
- **Dependency injection:** build functions receive dependencies as arguments
  (not global state), enabling clean testing and multiple configurations

**Testing (Chapter 13):**

- **What to test:** pure Python logic — routing functions, prompt templates,
  chunking behavior. These are deterministic and fast.
- **What not to test:** LLM output quality, full integration, third-party internals.
  These are slow, brittle, and non-deterministic.
- **pytest:** `test_*` functions, `assert` statements, `-v` and `--tb=short` flags
- **Fixtures:** `@pytest.fixture` for reusable setup; `tmp_path` for temporary files
- **`any(condition for item in iterable)`:** idiomatic Python for "at least one"
- **`is not None` vs `or`:** precise identity check vs broad truthiness check
- **Ternary expression:** `value if condition else other` — concise conditional
- **Testing design decisions:** each test encodes a behavior that must remain true
  even as the code evolves (metadata preservation, fallback priority, etc.)

---

*End of Part 7.*  
*Next: Part 8 — Debugging RAG Systems and What to Build Next*  
*A systematic approach to finding and fixing retrieval problems,*  
*and the natural extensions that turn this project into a production system.*
