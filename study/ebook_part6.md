# Building RAG Systems from Scratch
## Part 6 of 8 — Observability with LangSmith and Visual Pipelines with Langflow

**Series:** Building RAG Systems from Scratch  
**Part:** 6 of 8  
**Covers:** Chapters 10 and 11  
**Previous:** Part 5 — LangGraph: From Chains to Agents  
**Next:** Part 7 — Project Architecture and Testing

---

## Chapter 10 — Observability with LangSmith (Phase 3)

You now have a working RAG agent. But when it gives a wrong answer,
how do you know *why*?

Was it a retrieval problem — the right chunks weren't found?
Was it a prompt problem — the LLM misread the context?
Was it a model problem — llama3.2 isn't capable enough for this question?
Did the agent retry? Did it fall back?

Without observability, you're flying blind.
This chapter covers LangSmith — the tool that makes LLM applications visible.

---

### 10.1 The observability problem in LLM applications

Traditional software is relatively easy to observe:
- Log the inputs and outputs of each function
- Trace execution paths through your code
- Measure latency and error rates

LLM applications are harder because:

**Non-deterministic outputs:**
The same input can produce different outputs on different runs.
"The LLM returned a wrong answer" doesn't tell you if the context was wrong,
the prompt was wrong, or the model made a reasoning error.

**Multi-step pipelines:**
A RAG agent has at least 4 steps: embed → retrieve → prompt → generate.
An error in step 2 (wrong chunks retrieved) looks like an error in step 4
(wrong answer generated). You need per-step visibility.

**Invisible internals:**
When you call `agent.invoke("question")`, you get back a string.
You don't automatically see: which chunks were retrieved, what the full prompt
looked like, how many tokens were consumed, or how long each step took.

**LangSmith solves all of this.**

---

### 10.2 What LangSmith is

LangSmith is a SaaS observability platform built specifically for LLM applications.
It's created by the same team as LangChain and integrates with zero code changes.

It provides:
- **Tracing:** every run is recorded as a tree of steps with inputs/outputs
- **Metrics:** latency, token counts, cost per run
- **Debugging:** see the exact prompt sent to the LLM and the exact response
- **Evaluation:** run automated tests against a dataset of expected answers
- **Comparison:** compare different prompt versions or models side by side

The free tier covers everything you need for a portfolio project.

---

### 10.3 How LangSmith activates — zero code changes

LangSmith is activated entirely through environment variables.
You don't import it. You don't call any LangSmith functions.
LangChain reads these variables at startup and instruments everything automatically.

```bash
# .env file
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=lsv2_pt_your_key_here
LANGCHAIN_PROJECT=rag-langchain
```

When `LANGCHAIN_TRACING_V2=true`, LangChain wraps every:
- LLM call (`ChatOllama.invoke()`)
- Retriever call (`retriever.invoke()`)
- Chain invocation (`chain.invoke()`)
- LangGraph node execution

...with a "tracer" that sends timing, token counts, inputs, and outputs
to LangSmith's API in the background. Your business logic doesn't change at all.

**`load_dotenv()` — reading the .env file:**

```python
# main_chain.py and main_agent.py — must be the FIRST thing after imports
from dotenv import load_dotenv
load_dotenv()
```

`load_dotenv()` reads `.env` and sets each line as an OS environment variable:

```python
# After load_dotenv():
import os
os.environ["LANGCHAIN_TRACING_V2"]  # → "true"
os.environ["LANGCHAIN_API_KEY"]     # → "lsv2_pt_..."
os.environ["LANGCHAIN_PROJECT"]     # → "rag-langchain"
```

LangChain reads `os.environ["LANGCHAIN_TRACING_V2"]` at import time.
**This is why `load_dotenv()` must come before any LangChain imports** —
if LangChain is imported first, it reads the environment before `load_dotenv()`
sets the variables, and tracing never activates.

```python
# WRONG order:
from langchain_ollama import ChatOllama  # LangChain imported — reads env now
from dotenv import load_dotenv
load_dotenv()                            # too late — LangChain already checked

# CORRECT order:
from dotenv import load_dotenv
load_dotenv()                            # sets env vars first
from langchain_ollama import ChatOllama  # LangChain reads the vars correctly
```

---

### 10.4 The .env file and security

The `.env` file contains your LangSmith API key.
API keys are credentials — treat them like passwords.

**Never commit `.env` to Git.** Our `.gitignore` includes:
```
.env
```

This prevents accidentally pushing your key to a public repository
where anyone could use it (and potentially exhaust your account's quota).

**How environment variables work:**

Environment variables are key-value pairs managed by the operating system.
They're accessible to any process running under your user account.

```python
import os

# Set an environment variable (this session only):
os.environ["MY_KEY"] = "abc123"

# Read an environment variable:
value = os.environ["MY_KEY"]           # → "abc123" (raises KeyError if missing)
value = os.environ.get("MY_KEY", "")  # → "abc123" (returns "" if missing)
```

`python-dotenv` (the `dotenv` package) reads a `.env` file and calls
`os.environ[key] = value` for each line. It's a convenience tool —
you could manually set environment variables in your shell instead:

```bash
# Without .env — set manually before running:
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=lsv2_pt_...
python main_agent.py
```

The `.env` file just makes this permanent and portable.

---

### 10.5 What you see in the LangSmith dashboard

After running the agent and asking a question, go to
[smith.langchain.com](https://smith.langchain.com) → your project → Traces.

Each invocation of `agent.invoke()` creates a **trace** — a tree of nested spans:

```
Run: agent.invoke()                     [total: 9.2s, 1847 tokens]
│
├── Node: retriever                     [0.6s]
│   └── retriever.invoke("refund...")
│       ├── OllamaEmbeddings.embed_query  [0.1s]
│       │   input:  "What is the refund policy?"
│       │   output: [0.021, -0.143, ...]  (768-dim vector)
│       └── ChromaDB similarity search   [0.05s]
│           input:  query vector
│           output: 4 Documents
│               - "Section 3.2: Refunds..." (score: 0.14)
│               - "Returns must be..."     (score: 0.18)
│               - "Contact support..."     (score: 0.19)
│               - "Digital products..."   (score: 0.20)
│
└── Node: generator                     [8.6s, 1847 tokens]
    └── generator_chain.invoke(...)
        ├── ChatPromptTemplate            [<0.01s]
        │   input:  {context: "Section 3.2...", question: "What is the refund..."}
        │   output: HumanMessage(content="You are an assistant... Context:...")
        └── ChatOllama (llama3.2)         [8.6s, 1847 tokens]
            input:  [HumanMessage(content="...full prompt...")]
            output: AIMessage(content="According to the documents...")
            tokens: {input: 1203, output: 644, total: 1847}
```

Every step is clickable. You can expand any node to see:
- The exact input (the full prompt, word for word)
- The exact output (the full LLM response)
- The token counts
- The latency
- Any errors

---

### 10.6 What to look for when debugging

**Latency analysis:**

```
Node: retriever    0.6s   ← embedding call + ChromaDB search
Node: generator    8.6s   ← LLM inference (the bottleneck)
```

With local inference (CPU, no GPU), the LLM is almost always the bottleneck.
If the retriever is slow (>2s), the embedding model may be overloaded or
ChromaDB is searching a very large collection.

**Token analysis:**

```
tokens: {input: 1203, output: 644, total: 1847}
```

Input tokens = your system instruction + retrieved chunks + question.
Output tokens = the generated answer.

If input tokens are very high (>3000), your chunks may be too large
or you're retrieving too many of them (`TOP_K_RESULTS`).
High input token count increases latency proportionally with local models.

**Retrieval analysis — the most useful part:**

LangSmith shows exactly which chunks were retrieved and their similarity scores.
This lets you answer the most important diagnostic question:

> "Did the retriever return the right chunks?"

```
Chunks retrieved:
  score: 0.14 | "Section 3.2: Digital products are non-refundable..."  ← correct
  score: 0.18 | "Returns initiated within 30 days..."                  ← correct
  score: 0.19 | "Contact support@company.com for refund requests..."   ← relevant
  score: 0.20 | "Digital products include software licenses..."         ← borderline
```

If the correct chunk is NOT in this list → retrieval problem (adjust threshold or chunk size).
If the correct chunk IS in the list but the answer is wrong → generation problem
(improve the prompt or use a better model).

**Retry traces:**

If the agent retried, you'll see:
```
Node: retriever (attempt 1)   → 0 chunks
  → edge_after_retrieval → "retriever"
Node: retriever (attempt 2)   → 0 chunks
  → edge_after_retrieval → "fallback"
Node: fallback
```

This tells you: the question is genuinely out of scope for your documents,
or your `SCORE_THRESHOLD` is too strict.

---

### 10.7 LangSmith as the MLflow equivalent for LLM pipelines

If you use MLflow in your ML work, LangSmith fills the same role for LLM applications:

| MLflow concept | LangSmith equivalent |
|---|---|
| Experiment | Project |
| Run | Trace |
| Parameters | Prompt template, model name, temperature |
| Metrics | Latency, token count |
| Artifacts | Retrieved chunks, full prompts, answers |
| Run comparison | Trace comparison (different prompts/models) |

**Key difference:**

MLflow tracks training runs — you run an experiment once (or periodically)
and compare the results.

LangSmith tracks inference calls — every time a user asks a question,
that's a new trace. In production, you'd have thousands of traces per day.

For our project, LangSmith helps during development:
- Debug why a specific question got a wrong answer
- Compare responses with `llama3.2` vs `mistral`
- See how token count changes when you adjust chunk size

---

### 10.8 LangSmith evaluation — beyond tracing

LangSmith also has an **evaluation framework** — you can build a dataset
of question/expected-answer pairs and run automated scoring.

This is the systematic way to measure RAG quality:

```python
from langsmith import Client

client = Client()

# Step 1: Create a dataset
dataset = client.create_dataset("rag-eval-v1")
client.create_examples(
    inputs=[
        {"question": "What is the refund period?"},
        {"question": "Are digital products refundable?"},
        {"question": "How do I contact support?"},
    ],
    outputs=[
        {"answer": "30 days"},
        {"answer": "No, digital products are non-refundable"},
        {"answer": "support@company.com"},
    ],
    dataset_id=dataset.id,
)

# Step 2: Run the agent against the dataset
# LangSmith scores each answer against the expected answer
# and shows pass/fail rates across the dataset
```

This is how you measure improvement when you change:
- `CHUNK_SIZE` (smaller chunks → better precision?)
- `SCORE_THRESHOLD` (stricter threshold → fewer but better chunks?)
- The prompt template (does adding "be concise" improve answers?)
- The model (mistral vs llama3.2 — which gives more accurate answers?)

For the scope of this project, manual inspection of traces is sufficient.
Evaluation datasets become important when deploying to production.

---

## Chapter 11 — Visual Pipelines with Langflow (Phase 4)

Langflow is a visual, low-code interface for building LangChain pipelines.
Instead of writing Python code, you drag and drop components onto a canvas
and connect them with lines.

Phase 4 of our project reproduces the RAG pipeline visually —
not to replace the code, but to understand the same concepts
from a different perspective.

---

### 11.1 What Langflow is and isn't

**What Langflow is:**
- A visual editor for LangChain pipelines
- A drag-and-drop interface where each node = a LangChain class
- A prototyping tool for exploring pipeline architectures
- A communication tool for showing non-technical stakeholders how the pipeline works

**What Langflow is NOT:**
- A replacement for code in production
- A tool for complex logic (conditional branching, retry loops)
- A performance-optimized deployment target
- The right tool for systems that need version control and CI/CD

The Langflow flow you built visually is equivalent to `main_chain.py` (Phase 1) —
a linear pipeline. The LangGraph agent (Phase 2) with its retry loop and
conditional routing cannot be easily represented in Langflow's current interface.

---

### 11.2 Running Langflow in Docker

We run Langflow via Docker rather than installing it directly. Here's why
this was the right choice for our project:

**Docker isolation:**
Langflow has dozens of heavy dependencies (PyTorch, CUDA libraries, etc.)
that we don't need for our RAG project. Installing Langflow directly into
our virtual environment would bloat it with gigabytes of unneeded packages.

Docker runs Langflow in an isolated container with its own Python environment.
Our project's `.venv` stays clean.

**What Docker is (briefly):**

Docker packages an application with all its dependencies into a **container** —
a lightweight, isolated process that runs the same way on any machine.

```
Your machine:        Docker container:
┌──────────────┐    ┌──────────────────────┐
│ .venv/       │    │ langflow + its deps  │
│ (our project)│    │ Python 3.11          │
│              │    │ PyTorch, CUDA libs   │
│ No Langflow  │    │ etc.                 │
└──────────────┘    └──────────────────────┘
         ↑                     ↑
    separate environments, no conflicts
```

**The Docker run command explained:**

```bash
docker run -it -p 7860:7860 \
  -v $(pwd)/vectorstore:/app/vectorstore \
  -v $(pwd)/langflow_data:/app/langflow \
  -e LANGFLOW_DATABASE_URL=sqlite:////app/langflow/langflow.db \
  langflowai/langflow:latest
```

Breaking this down piece by piece:

`docker run` — create and start a new container

`-it` — two flags combined:
- `-i` (interactive): keep stdin open (so you can see output)
- `-t` (tty): attach a terminal

`-p 7860:7860` — port mapping: `host_port:container_port`
The container's port 7860 (where Langflow runs) is exposed as port 7860
on your machine. This is why you access Langflow at `http://localhost:7860`.

`-v $(pwd)/vectorstore:/app/vectorstore` — volume mount:
Maps a directory from your machine into the container.
`$(pwd)` expands to your current working directory.
Without this, the container can't access your `vectorstore/` folder.

`-v $(pwd)/langflow_data:/app/langflow` — second volume mount:
Maps `langflow_data/` from your machine to `/app/langflow` inside the container.
This is where Langflow stores its database (flows, settings).
Without it, all flows disappear when the container stops.

`-e LANGFLOW_DATABASE_URL=...` — environment variable:
Tells Langflow to store its database inside `/app/langflow/langflow.db`
(which is mapped to your `langflow_data/` folder).

`langflowai/langflow:latest` — the Docker image to use.
Docker downloads this from Docker Hub on first run (~1-2GB).

---

### 11.3 The Ollama connectivity problem on Linux

Inside a Docker container, `localhost` refers to the container itself —
not to your host machine. This is a fundamental Docker networking concept.

```
Your machine:                Docker container:
  localhost → your machine     localhost → the container
  port 11434: Ollama running   port 11434: nothing here
```

When Langflow (inside Docker) tries to connect to `http://localhost:11434`
to reach Ollama, it fails — because Ollama is on the host machine,
not inside the container.

**The solution: the Docker bridge gateway**

On Linux, Docker creates a virtual network bridge called `docker0`.
The host machine is accessible from containers via the bridge gateway IP:

```bash
ip addr show docker0 | grep "inet " | awk '{print $2}' | cut -d/ -f1
# → 172.17.0.1
```

This IP (`172.17.0.1`) is the host machine's address as seen from inside Docker.

So in Langflow, instead of `http://localhost:11434`, you use:
`http://172.17.0.1:11434`

**But first, Ollama must listen on all interfaces:**

By default, Ollama only listens on `127.0.0.1` (localhost) —
meaning it accepts connections from the same machine only,
not from the Docker bridge.

We fix this with a systemd override:

```ini
# /etc/systemd/system/ollama.service.d/override.conf
[Service]
Environment="OLLAMA_HOST=0.0.0.0:11434"
Environment="OLLAMA_MODELS=/usr/share/ollama/.ollama/models"
```

`OLLAMA_HOST=0.0.0.0:11434` means "listen on all network interfaces."
`0.0.0.0` is the wildcard address — accepts connections from any IP,
including `172.17.0.1` (Docker bridge) and `127.0.0.1` (localhost).

After this change:
- Your terminal: `curl http://localhost:11434` ✓ (still works)
- Langflow (Docker): `http://172.17.0.1:11434` ✓ (now works)

**Why `ollama serve` manually causes problems:**

The systemd service runs as user `ollama` and stores models in
`/usr/share/ollama/.ollama/models`.

When you run `ollama serve` manually (as your user), it:
1. Creates a separate instance that tries to bind to port 11434
2. Fails if the systemd service is already using that port
3. Uses `~/.ollama/models` (your home directory) instead of the shared location
4. Creates two separate model stores that don't see each other

**Always use the systemd service. Never run `ollama serve` manually.**

---

### 11.4 The pipeline you built in Langflow

The visual flow you constructed maps directly to the Phase 1 code:

```
[Read File] → [Split Text] → [Ollama Embeddings] → [Chroma DB]
                                                         ↑
                                                    [Chat Input]
                                                         ↓
                                                    [Chroma DB] (Retrieve mode)
                                                         ↓
                                                      [Parser]
                                                         ↓
                                                  [Prompt Template]
                                                         ↓
                                                    [Ollama LLM]
                                                         ↓
                                                   [Chat Output]
```

**Mapping visual nodes to Python code:**

| Langflow node | Python equivalent |
|---|---|
| Read File | `PyPDFLoader` / `TextLoader` |
| Split Text | `RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)` |
| Ollama Embeddings | `OllamaEmbeddings(model="nomic-embed-text")` |
| Chroma DB (Ingest) | `Chroma.add_documents(chunks)` |
| Chroma DB (Retrieve) | `vectorstore.as_retriever(...)` |
| Parser | `format_docs()` — joins chunk texts |
| Prompt Template | `ChatPromptTemplate.from_template(...)` |
| Ollama LLM | `ChatOllama(model="llama3.2")` |
| Chat Input | `input("Question: ")` |
| Chat Output | `print(answer)` |

Everything you clicked and configured in Langflow corresponds exactly
to a class instantiation or function call in the Python code.

---

### 11.5 Ingest vs Retrieve — two modes for the same node

The Chroma DB node in Langflow operates in two modes, and you had to switch
between them manually. Understanding why clarifies what the code does automatically.

**Ingest mode:**
```
Read File → Split Text → Ollama Embeddings → Chroma DB (Ingest)
```
This runs the indexing pipeline: load documents → split → embed → store.
Equivalent to `vectorstore.add_documents(chunks)` in Python.

**Retrieve mode:**
```
Chat Input → Chroma DB (Retrieve) → Parser → Prompt → LLM → Chat Output
```
This runs the query pipeline: embed question → search → return chunks.
Equivalent to `retriever.invoke(question)` in Python.

In our Python code, `src/vectorstore.py` handles this automatically:
```python
if vectorstore._collection.count() == 0:
    # Ingest mode: load, split, embed, store
    vectorstore.add_documents(chunks)
else:
    # Retrieve mode: already indexed, just connect
    pass
```

Langflow doesn't have this logic built in — you manage it manually.
This is one of the clearest examples of code being more powerful than visual tools
for anything beyond simple prototyping.

---

### 11.6 Saving flows — the JSON export

After building a flow in Langflow, you export it as JSON:

```
Menu (three dots) → Export → Save as rag_ollama_langchain.json
```

The JSON file is a complete description of your visual pipeline:
which nodes exist, their configurations, and how they're connected.

```json
{
  "nodes": [
    {
      "id": "File-abc123",
      "type": "File",
      "data": {
        "node": {
          "template": {
            "files": { "value": [] }
          }
        }
      }
    },
    ...
  ],
  "edges": [
    {
      "source": "File-abc123",
      "target": "SplitText-def456",
      "sourceHandle": "files",
      "targetHandle": "input"
    },
    ...
  ]
}
```

This JSON file is what you commit to Git (in the `flows/` folder) instead of
the visual interface itself. Anyone with Langflow installed can import it
and recreate the exact flow.

**Why version-control the JSON but not the database:**

`langflow_data/langflow.db` (SQLite) contains your flows, settings, and history.
It's a binary file — Git can't meaningfully diff it or show what changed.

The exported JSON is human-readable text — Git can track changes, show diffs,
and let you compare versions. This is the correct artifact for version control.

---

### 11.7 When to use Langflow vs code

After building the same pipeline both ways, you can now make an informed
judgment about when each approach is appropriate.

**Use Langflow when:**

- **Exploring a new idea quickly.** Dragging nodes is faster than writing code
  when you're not sure what architecture you want yet.

- **Showing stakeholders.** A visual pipeline is easier to explain to
  non-technical colleagues than a Python file.

- **Teaching.** The visual representation makes the data flow obvious —
  you can literally see where the question enters and where the answer exits.

- **Prototyping with non-developers.** Data analysts or domain experts can
  experiment with prompts and configurations without writing code.

**Use Python code when:**

- **Production deployment.** Code is versioned, tested, and deployed with
  standard software engineering practices. Langflow adds overhead and
  complexity in production.

- **Complex logic.** Retry loops, conditional branching, custom functions,
  error handling — these are natural in code and awkward in visual tools.

- **Performance matters.** Direct Python calls to Ollama are significantly
  faster than Langflow's HTTP API layer.

- **You need tests.** You can write pytest tests for Python functions.
  Testing Langflow flows requires a running Langflow server.

- **You need LangGraph.** The agent with retry logic (Phase 2) doesn't
  exist in Langflow's component library.

**The practical conclusion:**

Use Langflow to prototype and communicate. Use Python to build and deploy.
Many teams use both — Langflow for rapid iteration with domain experts,
Python for the production system.

---

### 11.8 The observability gap in Langflow

One important limitation: Langflow's built-in observability is minimal.
You can see if a run succeeded or failed, but you don't get:
- Per-step latency
- Token counts
- Exact prompts sent to the LLM
- Retrieved chunk details

LangSmith partially integrates with Langflow, but the tracing depth
is less detailed than what you get from the Python implementation.

This is another reason the Python agent is the production choice:
LangSmith gives you complete visibility into every node execution,
which is essential for debugging and optimization in production.

---

### Chapter 10 & 11 — Summary

**LangSmith (Chapter 10):**

- Activated by 4 environment variables in `.env` — zero code changes
- `load_dotenv()` must come before all LangChain imports (reads env at import time)
- `.env` contains credentials — always in `.gitignore`, never committed to Git
- Traces show: per-node latency, token counts, exact prompts, retrieved chunks
- Retrieval analysis is the most valuable: "did we get the right chunks?"
- Retry traces show whether the agent is falling back on certain questions
- Equivalent to MLflow but for inference calls, not training runs
- Evaluation datasets enable systematic quality measurement across prompt/model changes

**Langflow (Chapter 11):**

- Visual editor where nodes = LangChain classes, connections = data flow
- Run via Docker to avoid dependency conflicts with the project environment
- On Linux, Docker can't use `localhost` to reach Ollama on the host:
  → Ollama must listen on `0.0.0.0` (via systemd override)
  → Langflow connects via `http://172.17.0.1:11434` (Docker bridge gateway)
- Never run `ollama serve` manually — it conflicts with the systemd service
- Ingest mode = indexing pipeline; Retrieve mode = query pipeline
- Python handles this automatically; Langflow requires manual switching
- Export flows as JSON for version control in Git
- Use Langflow for prototyping and communication; Python for production

**The complementary relationship:**

```
Langflow  → explore, prototype, communicate visually
Python    → build, test, deploy, observe in production
LangSmith → understand what's happening at every step
```

All three together give you the full picture:
the visual understanding (Langflow), the production system (Python + LangGraph),
and the observability layer (LangSmith).

---

*End of Part 6.*  
*Next: Part 7 — Project Architecture and Testing*  
*Why we modularized the project, the design principles behind the structure,*  
*and how to write tests for LLM applications.*
