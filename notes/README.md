# Building RAG Systems from Scratch
## Complete Series Index

**8 parts · 15 chapters · Built from a real, working project**

All concepts map directly to code in the `rag-langchain` repository.
Every part is self-contained — you can read in order or jump to what you need.

---

## How to use this material

- **Learning from scratch:** read Parts 1–4 in order before jumping ahead
- **Reviewing a specific concept:** use the detailed index below to find the section
- **Debugging a problem:** go directly to Part 8
- **Understanding the architecture:** go directly to Part 7
- **Setting up the tools:** Part 6 covers LangSmith and Langflow setup in full

---

## Part 1 — The Problem RAG Solves + How LLMs Work
**File:** `ebook_part1.md`

### Chapter 1 — The Problem RAG Solves
- The three fundamental limitations of LLMs: knowledge cutoff, private data blindness, hallucination
- What RAG is and how it solves all three problems with one idea
- Concrete before/after example with a real refund policy question
- The two phases of every RAG system: indexing (offline) and querying (real-time)
- When RAG is the right tool — and when it isn't
- RAG vs fine-tuning: the key distinction and when to use each

### Chapter 2 — How Language Models Work (What You Need to Know)
- Tokens: what they are, why 1 token ≈ 0.75 words, context window limits by model
- Temperature: how it works technically, why RAG always uses temperature = 0
- System messages vs human messages: the chat format (roles explained)
- What the LLM actually receives in our RAG system: a concrete full prompt example
- Non-determinism and why it matters for testing strategy
- Local vs API-based LLMs: why we chose Ollama and what we trade off

---

## Part 2 — Embeddings and Vector Stores
**File:** `ebook_part2.md`

### Chapter 3 — Embeddings — Turning Text into Numbers
- The core idea: semantically similar texts produce geometrically close vectors
- How embedding models are trained: contrastive learning (positive/negative pairs)
- Vectors and vector spaces: a 3D intuition example before going to 768 dimensions
- Cosine similarity vs cosine distance: the formula, the range, why lower scores are better
- The SCORE_THRESHOLD parameter: what it does and how to calibrate it for your documents
- Why use a dedicated embedding model (nomic-embed-text) instead of the LLM
- Embedding quality failure modes: domain terminology, language mismatch, chunk length

### Chapter 4 — Vector Stores — Databases for Semantic Search
- Why regular databases (PostgreSQL, SQLite) can't do semantic search at scale
- How ChromaDB works: the HNSW algorithm explained with the highway system intuition
- What ChromaDB stores for each chunk: text + metadata + vector
- What's on disk: `chroma.sqlite3` and the binary HNSW index files
- The `_collection.count()` pattern: why it's the most reliable way to check for data
- Collections as namespaces: using `CHROMA_COLLECTION` to separate datasets
- The three retriever search types: `similarity`, `similarity_score_threshold`, `mmr`
- When to use `similarity_score_threshold` vs the default similarity search
- Vector store options beyond ChromaDB: Pinecone, Qdrant, Weaviate, pgvector comparison
- How to re-index from scratch: delete `vectorstore/` and re-run

---

## Part 3 — Document Loading, Chunking Strategy, and LangChain
**File:** `ebook_part3.md`

### Chapter 5 — Document Loading and Chunking Strategy
- The Document object: `page_content` + `metadata`, why metadata matters for traceability
- Document loaders: PyPDFLoader, TextLoader, and other loaders with example outputs
- The critical limitation of PyPDFLoader: text-based vs scanned PDFs, how to tell the difference
- Using `try/except` per file: one bad document shouldn't crash the whole pipeline
- `pathlib.Path` vs `os.path`: why pathlib is cleaner and more portable
- `rglob("*")`: recursive file discovery, why we `sorted()` for consistent ordering
- Why documents must be chunked: context dilution, context window limits, retrieval precision
- RecursiveCharacterTextSplitter: the recursive algorithm step by step
- Why chunk_overlap prevents losing context at boundaries: concrete before/after example
- Choosing chunk size by document type: table with recommendations and reasoning
- `split_documents()` vs `split_text()`: always use the former to preserve metadata

### Chapter 6 — LangChain — The Framework Layer
- What LangChain actually does (and doesn't do): it's plumbing, not AI
- The Runnable interface: why everything has `.invoke()` and why that matters
- LCEL (LangChain Expression Language): how `|` works under the hood (`__or__`)
- The fan-out pattern: `RunnableParallel` + `RunnablePassthrough` explained with diagram
- ChatPromptTemplate: `from_template()` vs `from_messages()`, `input_variables`
- StrOutputParser: what `AIMessage` is, why you need to extract `.content`
- ChatOllama: why use it instead of raw HTTP calls to Ollama
- The `logging` module vs `print()`: 5 reasons to use logging in production code
- Log levels: DEBUG, INFO, WARNING, ERROR and what each means
- Type hints in Python: `List[Document]`, `Optional[Path]`, what they do and don't enforce

---

## Part 4 — Building the Basic RAG Chain (Phase 1)
**File:** `ebook_part4.md`

### Chapter 7 — Building the Basic RAG Chain (Phase 1)
- `config.py` deep dive: `Path(__file__).resolve().parent.parent` dismantled step by step
- The pathlib `/` operator: joining paths without string concatenation
- Why `UPPER_SNAKE_CASE` signals constants (convention, not enforcement)
- `prompts.py` anatomy: role instruction, grounding ("ONLY"), explicit fallback, `Answer:` trigger
- Why the word "ONLY" is load-bearing in the grounding instruction
- `vectorstore.py`: lazy evaluation of Ollama, `_collection.count()` pattern, `if not chunks:`
- `retriever.py`: `search_kwargs` and the `**kwargs` pattern, why the retriever is its own module
- `chain.py`: `format_docs` as a bridge between `List[Document]` and `str`
- Full step-by-step trace of `chain.invoke("question")` with real inputs and outputs at each step
- `main_chain.py`: `warnings.filterwarnings`, `load_dotenv()` import order requirement
- PEP 8 import ordering: standard library → third-party → local
- `while True` with `break` and `continue`: the interactive loop pattern
- `.strip()` and `.lower()`: defensive input handling
- `subprocess.run()`: list vs string form, why list is safer
- `if __name__ == "__main__":` and why it matters when importing modules
- The complete startup sequence: everything that happens when you run `python main_chain.py`
- Phase 1 limitations: no quality gate, no retry — why Phase 2 was needed

---

## Part 5 — LangGraph: From Chains to Agents
**File:** `ebook_part5.md`

### Chapter 8 — LangGraph: From Chains to Agents
- Why fixed chains fail: two concrete failure scenarios with real outputs
- What LangGraph adds: conditional branching and loops
- Finite state machines: traffic light and vending machine as intuition builders
- The RAG agent as a finite state machine: states, transitions, shared data
- TypedDict: what it is, why it's NOT a regular class, how to create a state dict correctly
- `List[Document]` generic type annotation: what it means, what Python enforces vs doesn't
- The node contract: `AgentState → AgentState`, why return a new dict not mutate the input
- `{**state, "key": value}`: dictionary unpacking for immutable state updates
- Closures: how `build_nodes(retriever, llm)` captures dependencies for inner functions
- Why closures instead of global variables: explicit dependencies, testability
- Conditional edges: why they return strings, the mapping from string to next node
- The self-loop: `"retriever": "retriever"` in the mapping creates retry behavior
- The END sentinel: what happens when a node transitions to END
- `graph.compile()`: validation, CompiledGraph creation, tracing enablement

### Chapter 9 — Building the Agentic RAG (Phase 2)
- The initial state: what goes in before the first node runs
- Full trace of the happy path: retriever → generator → END with real state before/after each node
- Full trace of the failure path: retriever → retriever → fallback → END
- `state.get("attempts", 0)` vs `state["attempts"]`: defensive programming with dict access
- `select_model()`: `", ".join(list)`, `in`/`not in`, truthiness of empty strings
- Dependency injection in `main()`: vectorstore → retriever → agent, why this order matters
- f-strings: inline expressions, quote mixing inside f-strings
- Phase 1 vs Phase 2 comparison table: structure, retrieval, retries, LLM calls, observability
- When to use Phase 1 vs Phase 2: criteria for choosing the right approach
- The complete startup sequence for `python main_agent.py`

---

## Part 6 — LangSmith and Langflow
**File:** `ebook_part6.md`

### Chapter 10 — Observability with LangSmith (Phase 3)
- The observability problem: non-determinism, multi-step pipelines, invisible internals
- How LangSmith activates with zero code changes: the 4 environment variables
- Why `load_dotenv()` must come before all LangChain imports: import-time env reading
- The `.env` file: why it exists, why it's in `.gitignore`, how environment variables work in the OS
- Full dashboard trace example: per-node latency, token counts, inputs, outputs, chunk scores
- Latency analysis: where local RAG systems spend their time (almost always the LLM)
- Token analysis: input vs output tokens, what high token counts imply
- Retrieval analysis: the most valuable part — did we get the right chunks?
- Retry traces: how to spot questions that consistently fall back
- LangSmith vs MLflow: same role, different scope (inference vs training)
- Evaluation datasets: systematic quality measurement across prompt/model changes

### Chapter 11 — Visual Pipelines with Langflow (Phase 4)
- What Langflow is and isn't: visual prototyping tool, not a production system
- Why Docker for Langflow: dependency isolation, no virtual environment conflicts
- The Docker run command dismantled flag by flag: `-it`, `-p`, `-v`, `-e`
- Volume mounting: what `-v $(pwd)/vectorstore:/app/vectorstore` means and why it's required
- The Ollama connectivity problem on Linux: why `localhost` inside Docker refers to the container
- The Docker bridge gateway (`172.17.0.1`): the host machine's address from inside Docker
- `OLLAMA_HOST=0.0.0.0:11434`: why Ollama must listen on all interfaces for Docker to reach it
- Why never run `ollama serve` manually: systemd service vs manual instance, two separate model stores
- Mapping Langflow visual nodes to Python code: complete table
- Ingest vs Retrieve mode: why Python handles this automatically and Langflow requires manual switching
- Exporting flows as JSON for Git version control
- When to use Langflow vs code: prototyping/communication vs production/complex logic

---

## Part 7 — Project Architecture and Testing
**File:** `ebook_part7.md`

### Chapter 12 — Project Architecture: Why We Modularized
- Single Responsibility Principle: the "one reason to change" rule with a complete table
- Dependency direction: why higher-level imports lower-level (and never reverse)
- Circular imports: what they are, why they cause `ImportError`, how to avoid them
- Configuration centralization: before/after comparison showing the `config.py` benefit
- `UPPER_SNAKE_CASE`: Python constants convention, what it communicates, what Python enforces
- `__init__.py`: what it does, what happens without it, what makes a directory a package
- Entry points vs modules: why `main_*.py` files are thin, how this enables new entry points
- Creating `main_api.py` example: reusing all `src/` modules with a FastAPI server
- Dependency injection: `build_chain(vectorstore, model)` pattern, why not global variables
- The full import hierarchy of the project: which module imports from which

### Chapter 13 — Testing LLM Applications
- Why you can't test LLM output quality: non-determinism, model updates, output variation
- The three-level testing hierarchy: unit → integration → evaluation
- What we test (deterministic Python logic) vs what we don't (LLM outputs, full pipeline)
- pytest fundamentals: `test_*` naming, `assert`, running tests, common flags
- `pytest.ini` configuration: `testpaths`, `addopts`, `-v`, `--tb=short`
- Fixtures: `@pytest.fixture`, `tmp_path` built-in, how pytest injects them automatically
- `write_bytes(b"...")`: bytes literals for binary file creation in tests
- `any(condition for item in iterable)`: idiomatic Python for "at least one" checks
- `test_load_documents_*`: happy path, empty folder, unsupported files, metadata preservation
- `test_split_documents_*`: why `CHUNK_SIZE * 2` as upper bound, not exact character count
- `test_prompts.py`: `input_variables`, `format_messages()`, testing design decisions as code
- `test_agent.py`: `make_state()` helper, `make_doc()` helper, all routing branches covered
- `documents if documents is not None else []`: why not `documents or []` (truthiness vs identity)
- Ternary expression syntax: `value if condition else other`
- `is not None`: identity check vs truthiness check, when precision matters
- The "chunks take priority over attempts" test: documenting a design decision as a rule

---

## Part 8 — Debugging RAG Systems and What to Build Next
**File:** `ebook_part8.md`

### Chapter 14 — Debugging RAG Systems
- The core principle: wrong answers are almost always a retrieval problem, not a generation problem
- The 3-step debugging hierarchy: list sources → lexical search → semantic search
- Step 1 — Diagnosing indexing problems: 5 causes, how to identify each
- Step 2 — Diagnosing text extraction problems: scanned PDFs, encoding issues, headers/footers
- How to detect a scanned PDF (select text test), OCR solutions with `ocrmypdf`
- Step 3 — Reading semantic search output: score interpretation, what each range means
- Calibrating SCORE_THRESHOLD: the systematic process with a worked example
- BM25 hybrid search: why semantic search fails for proper names and acronyms, full code solution
- `EnsembleRetriever`: combining semantic + lexical with configurable weights
- Step 7 — Generation problems: 4 causes (model too small, context too long, contradictions, multi-hop)
- The lessons learned table: 8 real problems hit during this project, root causes, and solutions applied

### Chapter 15 — What to Build Next
- Conversational memory: `ConversationBufferMemory`, `MessagesPlaceholder`, prompt changes
- Source citation: extracting metadata from retrieved chunks in `node_generator`
- Query rewriting: new `node_rewriter` LangGraph node, updated routing logic, full code
- FastAPI server: `main_api.py` with `@app.post("/ask")`, Pydantic `BaseModel` explained
- Metadata filtering: `filter` parameter in `search_kwargs`, ChromaDB filter syntax
- LangSmith evaluation: datasets, `evaluate()`, systematic quality comparison across configurations
- The architectural progression: Phase 1 → Phase 9, each addition incremental and independent
- The complete mental model: RAG as separation of concerns, LangChain as plumbing, LangGraph as control flow, LangSmith as visibility
- Closing: what you built, what you understand, what you can now design yourself

---

## Quick reference — concepts by part

| Concept | Part |
|---|---|
| What RAG is and why it exists | 1 |
| Tokens and context windows | 1 |
| Temperature | 1 |
| Embeddings and cosine distance | 2 |
| SCORE_THRESHOLD calibration | 2 |
| ChromaDB and HNSW | 2 |
| Document loaders | 3 |
| Chunking strategy and chunk size | 3 |
| LCEL and the `\|` operator | 3 |
| RunnablePassthrough and fan-out | 3 |
| ChatPromptTemplate | 3 |
| config.py and path resolution | 4 |
| Full chain.invoke() trace | 4 |
| load_dotenv() import order | 4 |
| TypedDict | 5 |
| LangGraph nodes and edges | 5 |
| Closures | 5 |
| Retry loop via self-edge | 5 |
| LangSmith setup and traces | 6 |
| Docker run command | 6 |
| Ollama + Docker networking | 6 |
| Single Responsibility Principle | 7 |
| __init__.py | 7 |
| Dependency injection | 7 |
| pytest and fixtures | 7 |
| tmp_path | 7 |
| Debugging retrieval problems | 8 |
| BM25 hybrid search | 8 |
| FastAPI integration | 8 |
| Query rewriting node | 8 |
| LangSmith evaluation | 8 |
