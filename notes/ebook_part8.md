# Building RAG Systems from Scratch
## Part 8 of 8 — Debugging RAG Systems and What to Build Next

**Series:** Building RAG Systems from Scratch  
**Part:** 8 of 8 — Final Part  
**Covers:** Chapters 14 and 15  
**Previous:** Part 7 — Project Architecture and Testing

---

## Chapter 14 — Debugging RAG Systems

When a RAG system gives a wrong answer, the problem is almost always in
**retrieval** — the right chunks weren't found, not that the LLM reasoned poorly.

This is the most important debugging insight for RAG:

> Optimize retrieval first. Optimize the LLM second.

A great LLM reading the wrong chunks will give a wrong answer.
A mediocre LLM reading the right chunks will give a correct answer.

This chapter gives you a systematic process for finding exactly where
things go wrong, using `diagnostics.py` as your primary tool.

---

### 14.1 The debugging hierarchy

When you get a wrong or missing answer, follow this decision tree:

```
Step 1: Is the document indexed?
    → python diagnostics.py → Option 3 (List sources)
    → Is the file you expect in the list?
        YES → proceed to Step 2
        NO  → indexing problem (Step 4)

Step 2: Is the key term in the vector store?
    → python diagnostics.py → Option 2 (Lexical search)
    → Enter the specific word or phrase you expect to find
        FOUND    → proceed to Step 3
        NOT FOUND → text extraction problem (Step 5)

Step 3: Is retrieval returning the right chunks?
    → python diagnostics.py → Option 1 (Semantic search)
    → Enter the question (not the term — the full question)
    → Look at the returned chunks and their scores
        RIGHT CHUNKS, LOW SCORES (< 0.3) → retrieval working correctly
        WRONG CHUNKS, HIGH SCORES (> 0.5) → threshold/embedding problem (Step 6)
        RIGHT CHUNKS but answer still wrong → generation problem (Step 7)

Step 4: Indexing problem → document not loaded
Step 5: Text extraction problem → document loaded but text not extracted
Step 6: Retrieval quality problem → chunks exist but wrong ones returned
Step 7: Generation problem → right chunks, wrong interpretation
```

---

### 14.2 Step 1: Diagnosing indexing problems

```bash
python diagnostics.py
# Choose: 3 — List indexed sources
```

Expected output when everything is correct:
```
--- Indexed sources (247 total chunks) ---
   89 chunks | /home/user/project/docs/policy.pdf
   112 chunks | /home/user/project/docs/manual.pdf
   46 chunks | /home/user/project/docs/faq.txt
```

**Problem: your document isn't in the list**

Causes and solutions:

1. **Wrong folder:** the file is in `documents/` but the loader looks in `docs/`
   → Check `DOCS_DIR` in `src/config.py`

2. **Unsupported extension:** the file is `policy.docx` but only `.pdf` and `.txt` are supported
   → Convert to PDF, or add a Word document loader to `src/loader.py`

3. **File not added before indexing:** you added the file after the vector store was built
   → Delete `vectorstore/` and run again to re-index

4. **Permission error:** the file exists but Python can't read it
   → Check file permissions: `ls -la docs/`

5. **Empty vectorstore/ but docs/ has files:** something crashed during indexing
   → Check `logs/rag.log` for error messages

---

### 14.3 Step 2: Diagnosing text extraction problems

```bash
python diagnostics.py
# Choose: 2 — Lexical search
# Enter: the exact term you expect (e.g. "IA generativa")
```

**Problem: 0 chunks containing the term**

The term exists in the document but not in the vector store.
This almost always means text extraction failed silently.

**Cause 1: Scanned PDF (most common)**

The PDF is a scanned image, not a text-based PDF.
`PyPDFLoader` can only extract text from text-based PDFs.
For scanned PDFs, the text fields are empty or contain garbage.

How to diagnose:
```bash
# Open the PDF and try to select text with your cursor
# If you CAN select text → text-based PDF, loader should work
# If you CANNOT select text → scanned image PDF
```

Solutions:
- Use an OCR tool to create a text-based version:
  ```bash
  pip install ocrmypdf
  ocrmypdf scanned.pdf text_based.pdf
  ```
- Use `UnstructuredPDFLoader` with OCR strategy:
  ```python
  from langchain_community.document_loaders import UnstructuredPDFLoader
  loader = UnstructuredPDFLoader("scanned.pdf", strategy="hi_res")
  ```

**Cause 2: Text in images or tables within the PDF**

Some PDFs have text-based pages but embed specific content as images
(charts, screenshots, scanned figures). Text in those images won't be extracted.

**Cause 3: Encoding issues**

Some PDFs use non-standard character encodings.
`PyPDFLoader` may extract garbled text:
```
"IA generativa" → "IA gene rativa" or "I A g e n e r a t i v a"
```

Solution: try `pdfplumber` as an alternative loader:
```python
pip install pdfplumber
from langchain_community.document_loaders import PDFPlumberLoader
docs = PDFPlumberLoader("file.pdf").load()
```

**Cause 4: Term in header/footer**

PDF headers and footers are sometimes stripped by the loader.
If "IA generativa" only appears in a running header, it won't be in the chunks.

---

### 14.4 Step 3: Diagnosing retrieval quality

```bash
python diagnostics.py
# Choose: 1 — Semantic search
# Enter: the full question (not just the key term)
```

Read the output carefully:

```
--- Semantic search: 'What is the refund policy for digital products?' (k=6) ---

score: 0.14 | source: policy.pdf
  Section 3.2: Digital products are non-refundable within 30 days
  unless the product is proven defective. To initiate a refund claim...

score: 0.18 | source: policy.pdf
  Returns must be initiated by the customer within 30 days. The company
  does not proactively process returns without a customer request.

score: 0.51 | source: policy.pdf
  Section 1: Company Overview. Founded in 2010, our company provides...

score: 0.63 | source: manual.pdf
  Product installation guide. Step 1: Download the installer from...
```

**Interpretation:**

- Scores 0.14 and 0.18: **excellent** — these are the right chunks, well below threshold
- Score 0.51: **borderline** — company overview, not relevant to the question
- Score 0.63: **noise** — installation guide, completely irrelevant

With `SCORE_THRESHOLD = 0.2`, only the first two chunks would be returned.
The borderline and noise chunks would be filtered out. This is correct behavior.

**Problem scenario 1: right chunks have high scores (> 0.5)**

```
score: 0.52 | source: policy.pdf
  Section 3.2: Digital products are non-refundable...  ← this is the right chunk!
```

The right chunk exists but is scoring poorly. Causes:

- **Domain terminology mismatch:** you asked "What is the refund policy" but the
  document says "reimbursement procedure." The embedding model may not map
  these to the same region of vector space.
  Solution: try rephrasing the question, or add the expected phrasing to your
  test questions to match the document vocabulary.

- **Chunk too large:** a 1000-character chunk about "refund policy AND company history
  AND team structure" produces an averaged embedding that matches no query well.
  Solution: reduce `CHUNK_SIZE` in `src/config.py` and re-index.

- **Embedding model quality:** `nomic-embed-text` may not capture domain terminology well.
  Solution: try `mxbai-embed-large` or a domain-specific embedding model.

**Problem scenario 2: no relevant chunks at all, all scores > 0.7**

```
score: 0.71 | source: policy.pdf   ← Section 1: Company overview (irrelevant)
score: 0.74 | source: manual.pdf   ← Installation guide (irrelevant)
score: 0.79 | source: faq.txt      ← Product FAQ (irrelevant)
```

The question is genuinely out of scope for the documents — or the documents
don't contain the answer in any form that the embedding model can match.

Verify with lexical search (Step 2) to confirm the content exists in the chunks at all.

---

### 14.5 Calibrating SCORE_THRESHOLD

After running `diagnostics.py → Semantic search` with several questions,
you'll have a sense of what scores your documents produce.

The calibration process:

```
1. Run semantic search for 5-10 questions you know the answers to.

2. For each question, note:
   - The score of the CORRECT chunk (the one containing the answer)
   - The score of the BEST NOISE chunk (the most similar irrelevant chunk)

3. Set SCORE_THRESHOLD between those two values.

Example results:
  Question 1: correct=0.15, best noise=0.43
  Question 2: correct=0.18, best noise=0.51
  Question 3: correct=0.22, best noise=0.39
  Question 4: correct=0.12, best noise=0.55

  Correct chunks: 0.12 - 0.22 (max = 0.22)
  Noise chunks:   0.39 - 0.55 (min = 0.39)
  
  Good threshold: 0.30
  (captures all correct chunks, filters out all noise)
```

Then update `src/config.py`:
```python
SCORE_THRESHOLD = 0.30  # was 0.20
```

And restart the agent. No re-indexing needed — the threshold is applied
at query time, not at indexing time.

---

### 14.6 Step 6: Retrieval quality — the BM25 hybrid search solution

Sometimes semantic search fails for specific types of queries:
- Proper names ("TOTVS", "Arianna Montilla")
- Product codes ("SKU-48291", "Form 17-B")
- Technical acronyms ("FIFO", "ETL", "RAG")
- Exact phrases ("clause 3.2.1")

These fail because the embedding model may not represent rare or domain-specific
terms well in vector space. Semantically, "TOTVS" might be close to nothing
that the model knows about.

**BM25 — lexical search as a complement:**

BM25 (Best Match 25) is a classical information retrieval algorithm.
It scores documents based on term frequency and document frequency —
essentially a sophisticated keyword search. It's completely immune to
the domain terminology problem because it matches exact strings.

```python
# Adding BM25 to src/retriever.py

from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

def build_hybrid_retriever(vectorstore: Chroma) -> EnsembleRetriever:
    # Get all chunks for BM25 (needs them in memory)
    all_data = vectorstore.get()
    all_docs = [
        Document(page_content=content, metadata=meta)
        for content, meta in zip(all_data["documents"], all_data["metadatas"])
    ]

    # Semantic retriever (embedding-based)
    semantic = vectorstore.as_retriever(search_kwargs={"k": TOP_K_RESULTS})

    # Lexical retriever (keyword-based)
    bm25 = BM25Retriever.from_documents(all_docs)
    bm25.k = TOP_K_RESULTS

    # Combine: 50% semantic, 50% lexical
    return EnsembleRetriever(
        retrievers=[semantic, bm25],
        weights=[0.5, 0.5],
    )
```

With hybrid search:
- Semantic search finds conceptually related content (even if keywords differ)
- BM25 finds exact term matches (even if conceptually distant)
- Combined result: the best of both approaches

To add this to the project, modify `src/retriever.py` only.
Nothing else needs to change — the function signature returns a Runnable
that the rest of the code uses identically.

---

### 14.7 Step 7: Generation problems — right chunks, wrong answer

If `diagnostics.py` confirms the correct chunks are being retrieved
but the LLM still gives wrong answers, the problem is in generation.

**Problem 1: LLM ignores the "ONLY" instruction**

Smaller models (llama3.2 3B) sometimes blend retrieved context with training
data, even when instructed to use ONLY the provided context.

Solutions:
- Switch to a larger model: `mistral 7B` follows instructions much more reliably
- Strengthen the instruction: add explicit negative instruction:
  ```
  Use ONLY the context below. Do NOT use your training knowledge.
  If the answer is not in the context, say exactly: "I could not find this."
  ```

**Problem 2: Context is too long — LLM "forgets" the beginning**

If you retrieve many large chunks, the total context might be thousands of tokens.
Local models can struggle to attend to the beginning of very long contexts.

Solutions:
- Reduce `TOP_K_RESULTS` from 6 to 3-4
- Reduce `CHUNK_SIZE` so each chunk contributes fewer tokens
- Use a model with better long-context handling

**Problem 3: Context contains contradictory information**

Multiple chunks may contain conflicting information (e.g., two versions of a policy).
The LLM may blend both or pick the wrong one.

Solution: add source filtering to the prompt:
```python
# Modified prompt that emphasizes using ONLY the most relevant part
"""
The context below contains passages from our documents.
Focus on the passage most directly relevant to the question.
Use ONLY information from these passages.

Context:
{context}

Question: {question}
"""
```

**Problem 4: The answer requires combining multiple chunks**

Some questions need information from non-adjacent passages.
RAG systems struggle with this — they're designed for single-passage answers.

Example: "Compare the refund policy for digital vs physical products"
→ requires two separate passages, potentially from different pages

Solutions:
- Increase `TOP_K_RESULTS` to retrieve more context
- Add a summarization step that pre-processes retrieved chunks
- For complex multi-hop questions, consider a more sophisticated agent
  with query decomposition

---

### 14.8 The lessons learned table — from real problems

During the development of this project, we encountered these real problems.
Each one taught a lesson worth remembering:

| Problem we hit | Root cause | Solution applied |
|---|---|---|
| Duplicate chunks on repeated runs | `Chroma.from_documents()` always adds, never checks | Check `_collection.count()` before indexing |
| Router LLM choosing "direct" for document questions | llama3.2 3B too weak for reliable instruction following | Removed router — always retrieve |
| Grader LLM eliminating relevant chunks | Small model inconsistently evaluated relevance | Replaced LLM grader with `score_threshold` (objective metric) |
| "IA generativa" not found in chunks | Term existed in documents but not in indexed chunks (PDF extraction issue) | Verified with lexical search; re-checked source document |
| Ollama invisible to Docker | Ollama listening on `127.0.0.1` only | `OLLAMA_HOST=0.0.0.0` in systemd override |
| `ollama list` showing empty models | Systemd service and manual `ollama serve` using different home directories | Always use systemd service; never `ollama serve` manually |
| Port conflict on Ollama restart | Previous `ollama serve` still running on port 11434 | `kill $(lsof -t -i:11434)` before restart |
| ChromaDB scores interpreted as similarity | Score is distance (lower=better), not similarity (higher=better) | Read ChromaDB docs; score 0.15 is better than 0.85 |
| Langflow flows lost on Docker restart | No persistent volume for Langflow database | Mount `-v langflow_data:/app/langflow` |

These mistakes are not failures — they're the actual learning.
Each one forced a deeper understanding of how the system works.

---

## Chapter 15 — What to Build Next

You now have a working, well-structured RAG system.
This chapter maps the natural extensions that turn this project into
a production-grade application.

---

### 15.1 Conversational memory — multi-turn Q&A

The current system treats every question as independent.
A follow-up question like "And what about digital products?" fails
because the agent has no memory of the previous exchange.

**What to add:**

LangChain's `ConversationBufferMemory` stores the conversation history
and appends it to each prompt:

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
)
```

The prompt changes to include the history:
```python
template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer ONLY from context."),
    MessagesPlaceholder(variable_name="chat_history"),  # ← conversation history
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])
```

**Where to implement:** modify `src/prompts.py` and `src/chain.py`.
The agent in `src/agent.py` would need `chat_history` added to `AgentState`.

---

### 15.2 Source citation — telling users where answers come from

When the LLM answers a question, it would be useful to show which document
and page the answer came from.

**What to add:**

In `src/agent.py`, `node_generator` can be modified to extract sources:

```python
def node_generator(state: AgentState) -> AgentState:
    context = "\n\n".join(doc.page_content for doc in state["documents"])
    answer  = generator_chain.invoke({"context": context, "question": state["question"]})

    # Extract unique sources from retrieved chunks
    sources = sorted(set(
        f"{doc.metadata.get('source', 'unknown')} (page {doc.metadata.get('page', '?')})"
        for doc in state["documents"]
    ))

    # Append sources to the answer
    full_answer = f"{answer}\n\nSources: {', '.join(sources)}"
    return {**state, "generation": full_answer}
```

This produces answers like:
```
According to the documents, digital products are non-refundable within 30 days...

Sources: policy.pdf (page 3), policy.pdf (page 4)
```

---

### 15.3 Query rewriting — smarter retries

Currently, when retrieval fails, the agent retries with the exact same question.
This rarely helps — if the question didn't find good chunks the first time,
the same question won't find them the second time.

A smarter approach: use the LLM to rewrite the question before retrying.

**What to add — a new node in `src/agent.py`:**

```python
rewrite_prompt = ChatPromptTemplate.from_template("""
The following question failed to retrieve relevant documents.
Rewrite it using different phrasing that might match the document vocabulary better.
Keep the same intent but use synonyms or alternative phrasings.

Original question: {question}

Rewritten question:""")

rewrite_chain = rewrite_prompt | llm | StrOutputParser()

def node_rewriter(state: AgentState) -> AgentState:
    rewritten = rewrite_chain.invoke({"question": state["question"]})
    log.info(f"[rewriter] '{state['question']}' → '{rewritten.strip()}'")
    return {**state, "question": rewritten.strip()}
```

**New graph structure:**

```python
graph.add_conditional_edges(
    "retriever",
    edge_after_retrieval,
    {
        "generator": "generator",
        "rewriter":  "rewriter",   # ← new: retry with rewritten question
        "fallback":  "fallback",
    },
)
graph.add_edge("rewriter", "retriever")  # rewriter feeds back into retriever
```

**Updated routing logic:**

```python
def edge_after_retrieval(state: AgentState) -> str:
    if len(state["documents"]) > 0:
        return "generator"
    if state.get("attempts", 0) == 1:
        return "rewriter"   # first failure → rewrite and retry
    return "fallback"       # second failure → give up
```

---

### 15.4 FastAPI server — from terminal to API

The terminal loop in `main_agent.py` is fine for exploration.
In production, you'd expose the agent as an HTTP API that other
applications can call.

**What to add — `main_api.py`:**

```python
from fastapi import FastAPI
from pydantic import BaseModel
from src.vectorstore import build_vectorstore
from src.retriever import build_retriever
from src.agent import build_agent

app = FastAPI(title="RAG API", description="Document Q&A via RAG")

# Initialize once at startup (not per request — expensive operations)
vectorstore = build_vectorstore()
retriever   = build_retriever(vectorstore)
agent       = build_agent(retriever)


class Question(BaseModel):
    text: str


class Answer(BaseModel):
    answer: str
    attempts: int


@app.post("/ask", response_model=Answer)
def ask(question: Question) -> Answer:
    result = agent.invoke({
        "question":   question.text,
        "documents":  [],
        "generation": "",
        "attempts":   0,
    })
    return Answer(
        answer=result["generation"],
        attempts=result["attempts"],
    )
```

Run with:
```bash
pip install fastapi uvicorn
uvicorn main_api:app --host 0.0.0.0 --port 8000
```

Now you can query it from any client:
```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"text": "What is the refund policy?"}'
```

**`pydantic.BaseModel`** — Pydantic is FastAPI's data validation library.
`BaseModel` subclasses define the schema for request and response bodies.
FastAPI automatically validates, serializes, and documents them.

---

### 15.5 Multi-document filtering — scoped retrieval

If you have multiple document types (policies, manuals, FAQs), you might want
to let users specify which document to search.

**What to add — metadata filtering in `src/retriever.py`:**

```python
def build_retriever(vectorstore: Chroma, source_filter: str = None):
    search_kwargs = {
        "k":               TOP_K_RESULTS,
        "score_threshold": SCORE_THRESHOLD,
    }

    if source_filter:
        # Only return chunks from documents whose source path contains this string
        search_kwargs["filter"] = {"source": {"$contains": source_filter}}

    return vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs=search_kwargs,
    )
```

Usage:
```python
# Search only in policy documents
retriever = build_retriever(vectorstore, source_filter="policy")

# Search only in manual documents
retriever = build_retriever(vectorstore, source_filter="manual")
```

ChromaDB's filter syntax supports `$eq`, `$ne`, `$contains`, `$in`, and more.

---

### 15.6 LangSmith evaluation — systematic quality measurement

As your project matures, you'll want to systematically measure whether
changes (new models, different prompts, different chunk sizes) improve quality.

**What to add:**

```python
# eval_pipeline.py
from langsmith import Client
from langsmith.evaluation import evaluate

client = Client()

# 1. Create a dataset of questions with expected answers
dataset = client.create_dataset("rag-quality-v1")
client.create_examples(
    inputs=[
        {"question": "What is the refund period for digital products?"},
        {"question": "How do I contact customer support?"},
        {"question": "What shipping methods are available?"},
    ],
    outputs=[
        {"answer": "30 days"},
        {"answer": "support@company.com or via the help portal"},
        {"answer": "standard (5-7 days) and express (1-2 days)"},
    ],
    dataset_id=dataset.id,
)

# 2. Define a function that wraps your agent
def run_agent(inputs: dict) -> dict:
    result = agent.invoke({
        "question":   inputs["question"],
        "documents":  [],
        "generation": "",
        "attempts":   0,
    })
    return {"answer": result["generation"]}

# 3. Run evaluation
results = evaluate(
    run_agent,
    data="rag-quality-v1",
    evaluators=["qa"],  # LangSmith's built-in QA evaluator
    experiment_prefix="chunk-size-500",
)
```

Now you can compare:
- `chunk-size-500` (current) vs `chunk-size-800` (experiment)
- `llama3.2` vs `mistral`
- Original prompt vs improved prompt

And see which configuration scores better on your test questions.

---

### 15.7 The architectural progression

Looking back at the full project, each phase added one specific capability:

```
Phase 1: Basic chain
    "Given a question, retrieve relevant chunks and generate an answer"
    → Proves the core concept works

Phase 2: LangGraph agent
    "Detect retrieval failure and respond honestly instead of hallucinating"
    → Makes the system trustworthy

Phase 3: LangSmith
    "See exactly what's happening inside the system"
    → Makes the system debuggable

Phase 4: Langflow
    "Represent the pipeline visually"
    → Makes the system communicable

Next steps:
Phase 5: Conversational memory → makes the system coherent across turns
Phase 6: Source citation → makes the system transparent and auditable
Phase 7: Query rewriting → makes retrieval more robust
Phase 8: FastAPI → makes the system accessible to other applications
Phase 9: Evaluation pipeline → makes quality measurable and trackable
```

Each addition is incremental and independent.
You can add Phase 5 without touching Phase 4.
You can skip Phase 6 and jump to Phase 7.
The modular architecture makes this possible.

---

### 15.8 The complete mental model

After building this project from scratch, here is the complete mental model
you should carry into any future RAG work:

**RAG is a separation of concerns:**
```
Documents         → what the system knows (your data, not the model)
Embeddings        → how the system indexes knowledge (mathematical representation)
Vector store      → where indexed knowledge lives (specialized search database)
Retrieval         → how the system finds relevant knowledge (similarity search)
Prompt            → the contract between you and the LLM (instructions + context)
LLM               → the reasoning engine (interprets context, generates answer)
```

**LangChain is plumbing:**
It connects components with standard interfaces. Swap any piece
(model, vector store, embedding model) by changing one module.

**LangGraph is control flow:**
It adds decisions, loops, and state. Every complex AI application eventually
needs conditional branching — LangGraph provides it cleanly.

**LangSmith is visibility:**
Without it, LLM applications are black boxes. With it, you see exactly
what's happening, what's slow, what's expensive, and what's wrong.

**The architecture principle:**
```
config.py      → what (all settings in one place)
loader.py      → input (get documents in)
vectorstore.py → storage (persist and retrieve)
retriever.py   → search (find relevant pieces)
prompts.py     → instruction (tell the LLM what to do)
chain.py       → flow (simple, fixed path)
agent.py       → flow (complex path with decisions)
main_*.py      → entry (wire everything together and handle I/O)
tests/         → verification (prove the logic is correct)
```

This architecture pattern scales from a portfolio project to a production system
with millions of daily queries. The concepts are the same at any scale.

---

### Chapter 14 & 15 — Summary

**Debugging (Chapter 14):**

The debugging hierarchy:
1. List sources → confirm the document was indexed
2. Lexical search → confirm the term exists in chunks
3. Semantic search → confirm the right chunks are retrieved
4. If right chunks, wrong answer → generation problem

Common problems and solutions:
- Scanned PDFs → use OCR before indexing
- Domain terminology miss → hybrid search (BM25 + semantic)
- Threshold too strict → raise `SCORE_THRESHOLD`, verify with diagnostics
- Small model ignores instructions → switch to mistral or strengthen the prompt
- Context too long → reduce `TOP_K_RESULTS` or `CHUNK_SIZE`

Lessons learned the hard way:
- Always check `_collection.count()` to prevent duplicate indexing
- Small LLMs make unreliable routers and graders — use objective metrics instead
- ChromaDB scores are distances (lower=better), not similarities (higher=better)
- Never run `ollama serve` manually — use systemd always

**What to build next (Chapter 15):**

- Conversational memory → `ConversationBufferMemory` + `MessagesPlaceholder`
- Source citation → extract metadata from retrieved chunks in `node_generator`
- Query rewriting → new `node_rewriter` node before the retry edge
- FastAPI server → `main_api.py` with `@app.post("/ask")` endpoint
- Metadata filtering → `filter` parameter in `search_kwargs`
- LangSmith evaluation → datasets + `evaluate()` for systematic quality tracking

Each extension is incremental: add one module or node, change one function.
The modular architecture makes all of this possible without rewriting anything.

---

## Closing — What You've Built

This is what you built, from scratch:

```
A fully local RAG system that:
  ✓ Loads and indexes PDF and TXT documents
  ✓ Embeds chunks with nomic-embed-text via Ollama
  ✓ Stores vectors in ChromaDB with persistence
  ✓ Retrieves relevant chunks via semantic similarity search
  ✓ Builds grounded prompts with retrieved context
  ✓ Generates answers with llama3.2 or mistral via Ollama
  ✓ Detects retrieval failure and responds honestly (LangGraph agent)
  ✓ Retries retrieval up to MAX_RETRIES times before giving up
  ✓ Falls back gracefully instead of hallucinating
  ✓ Traces every run with per-node visibility (LangSmith)
  ✓ Represents the pipeline visually (Langflow via Docker)
  ✓ Centralizes all configuration in one file
  ✓ Logs structured messages to console and file
  ✓ Tests all pure Python logic without external services (pytest)
  ✓ Documents the architecture and design decisions (README.md)
  ✓ Versions the code and flow exports (Git)
  ✓ Works completely offline, with no API costs
```

And you understand not just *what* each piece does,
but *why* it exists, *when* to use it, and *how* to change it.

That's the foundation for building any RAG system — including the ones
you'll design yourself.

---

*End of Part 8. End of the series.*

*The complete series:*  
*Part 1 — The Problem RAG Solves + How LLMs Work*  
*Part 2 — Embeddings and Vector Stores*  
*Part 3 — Document Loading, Chunking, and LangChain*  
*Part 4 — Building the Basic RAG Chain (Phase 1)*  
*Part 5 — LangGraph: From Chains to Agents*  
*Part 6 — LangSmith and Langflow*  
*Part 7 — Project Architecture and Testing*  
*Part 8 — Debugging RAG Systems and What to Build Next*
