# Building RAG Systems from Scratch
## Part 2 of 8 — Embeddings and Vector Stores

**Series:** Building RAG Systems from Scratch  
**Part:** 2 of 8  
**Covers:** Chapters 3 and 4  
**Previous:** Part 1 — The Problem RAG Solves + How LLMs Work  
**Next:** Part 3 — Document Loading, Chunking, and LangChain

---

## Chapter 3 — Embeddings — Turning Text into Numbers

This is the mathematical foundation of semantic search.
Understanding it will help you debug retrieval problems, choose the right
embedding model, and reason about why your RAG system does or doesn't find
the right passages.

---

### 3.1 The core idea

An embedding is a function that maps text to a point in a high-dimensional space:

```
embed("The cat sat on the mat")    → [0.021, -0.143, 0.087, ..., 0.312]  # 768 numbers
embed("A feline rested on a rug")  → [0.019, -0.138, 0.091, ..., 0.308]  # very similar!
embed("The stock market crashed")  → [-0.412, 0.834, -0.221, ..., 0.091] # very different
```

The output — that list of floating-point numbers — is called a **vector** or **embedding**.
The number of dimensions varies by model: common sizes are 384, 768, 1024, and 1536.

The key property that makes this useful:

> **Semantically similar texts produce geometrically close vectors.**

"The cat sat on the mat" and "A feline rested on a rug" mean nearly the same thing.
Their embeddings are numerically close — the distance between them is small.

"The stock market crashed" means something completely different.
Its embedding is far away from both cat sentences.

This is what makes semantic search possible: instead of matching keywords,
we match *meaning*.

---

### 3.2 How embedding models are trained

Embedding models are not magic — they are neural networks trained with a specific
objective that forces similar meanings to cluster together in vector space.

The most common training approach is **contrastive learning**:

```
Training pairs:
  POSITIVE: ("The cat sat on the mat", "A feline rested on a rug")  → pull vectors together
  NEGATIVE: ("The cat sat on the mat", "The economy is struggling")  → push vectors apart
```

The model is trained to minimize distance between positive pairs and
maximize distance between negative pairs. After training on millions of such pairs,
the model learns a vector space where semantic proximity = geometric proximity.

**This is fundamentally different from how the generative LLM is trained.**
The LLM is trained to predict the next token.
The embedding model is trained to map meaning to geometry.
Different objective → different architecture → different use case.

---

### 3.3 Vectors and vector spaces — the geometry

If you've studied linear algebra or machine learning, you already know vectors.
An embedding vector is the same concept applied to text.

A 3-dimensional example to build intuition:

```
Imagine a 3D space where:
  axis 1: "animal-ness" (0 = no animals, 1 = very animal-related)
  axis 2: "financial-ness" (0 = unrelated to finance, 1 = very finance-related)
  axis 3: "technology-ness" (0 = unrelated to tech, 1 = very tech-related)

"The cat sat on the mat"   → [0.9, 0.0, 0.0]  (animal, not finance, not tech)
"A feline rested on a rug" → [0.8, 0.0, 0.1]  (animal, not finance, barely tech)
"Stock market crash"       → [0.0, 0.9, 0.1]  (not animal, finance, barely tech)
"Machine learning model"   → [0.0, 0.1, 0.9]  (not animal, barely finance, tech)
```

Real embedding models use 768+ dimensions, not 3. The dimensions don't have
human-interpretable labels — they emerge from training. But the principle is the same:
similar meanings cluster in the same region of the space.

---

### 3.4 Cosine similarity and cosine distance

There are multiple ways to measure "closeness" between two vectors.
ChromaDB uses **cosine similarity** (and converts it to distance).

**Why cosine similarity and not Euclidean distance?**

Euclidean distance measures the straight-line distance between two points.
It's sensitive to vector magnitude (length), not just direction.

Two documents about "cats" might produce vectors pointing in the same direction
but with different magnitudes (e.g., because one document is longer and its
embedding has larger values). Euclidean distance would say they're far apart.
Cosine similarity only looks at the angle between them — direction, not magnitude.

```
cosine_similarity(A, B) = (A · B) / (|A| × |B|)

where:
  A · B  = dot product = Σ(Aᵢ × Bᵢ) for all dimensions i
  |A|    = magnitude of A = √(Σ Aᵢ²)
  |B|    = magnitude of B = √(Σ Bᵢ²)
```

Cosine similarity ranges from -1 to 1:
```
 1.0  → vectors point in identical direction (same meaning)
 0.0  → vectors are perpendicular (unrelated)
-1.0  → vectors point in opposite directions (opposite meaning)
```

**ChromaDB converts this to cosine distance:**
```
cosine_distance = 1 - cosine_similarity

0.0  → identical (best possible match)
1.0  → completely unrelated
2.0  → maximally opposite (rare in practice)
```

**This is why lower scores are better in our diagnostics output:**

```python
# diagnostics.py output — LOWER score = MORE relevant
score: 0.15 | "Section 3.2: Digital products are non-refundable..."  ← great match
score: 0.31 | "Our company was founded in 2010..."                   ← weak match
score: 0.87 | "The weather in São Paulo is warm..."                  ← irrelevant
```

When you see high scores (0.7+), the retriever is returning chunks that are
semantically distant from your question.

---

### 3.5 The SCORE_THRESHOLD parameter

In `src/config.py`:

```python
SCORE_THRESHOLD = 0.2  # only return chunks with cosine distance <= 0.2
```

This threshold acts as a quality filter:
- Chunks with distance ≤ 0.2 → returned to the agent
- Chunks with distance > 0.2 → discarded (not sent to the LLM)

**How to calibrate it for your documents:**

```
Step 1: Run diagnostics.py → Option 1 (Semantic search)
        with a question you KNOW the answer to.

Step 2: Find the chunk that SHOULD be returned.
        Note its score.

Step 3: Set SCORE_THRESHOLD slightly above that score.

Example output:
  score: 0.15 | "Section 3.2: Refunds are processed within..."  ← correct chunk
  score: 0.31 | "Our company was founded in 2010..."            ← irrelevant noise
  score: 0.44 | "Contact support@company.com..."               ← irrelevant noise

  Good threshold: 0.25
  (captures the correct chunk at 0.15, excludes noise at 0.31+)
```

Setting the threshold too low → agent gets no chunks → always falls back
Setting the threshold too high → agent gets irrelevant chunks → LLM hallucinates

---

### 3.6 Why a dedicated embedding model?

In our project, we use two separate models:
- `nomic-embed-text` for generating embeddings
- `llama3.2` or `mistral` for generating text

You might wonder: why not just use the LLM for both?

**Reason 1: Task mismatch**

The LLM (mistral, llama3.2) is trained to predict the next token and generate
fluent, coherent text. Its internal representations reflect this objective.

The embedding model (`nomic-embed-text`) is trained specifically to map text
to a vector space where semantic similarity = geometric proximity.
Different training objective → better results for the embedding task.

**Reason 2: Speed**

Generating a single embedding with `nomic-embed-text`:
```
~50ms per chunk
1000 chunks → ~50 seconds to index
```

Generating an embedding by running `mistral` in "embedding mode":
```
~3-8 seconds per chunk (because it runs the full model)
1000 chunks → hours to index
```

For indexing thousands of documents, this difference is critical.

**Reason 3: Consistency and stability**

Embeddings must be reproducible: the same text must always produce the same vector.
If you re-embed a document and get slightly different vectors, the search results
change unpredictably.

Embedding models are deterministic by design — same input always produces same output.

LLMs with temperature > 0 are stochastic. Even at temperature = 0, small
implementation differences can cause output variation.

**In our project:**

```python
# src/config.py
EMBEDDING_MODEL = "nomic-embed-text"   # dedicated, fast, deterministic
DEFAULT_LLM     = "llama3.2"           # for generation only
```

```python
# src/vectorstore.py
from langchain_ollama import OllamaEmbeddings

def _get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=EMBEDDING_MODEL)
    # This is what converts text chunks to vectors during indexing
    # and converts the user's question to a vector during querying
```

---

### 3.7 Embedding quality and domain mismatch

The quality of your embeddings directly determines the quality of your retrieval.

**Common failure modes:**

**Domain terminology**

`nomic-embed-text` is trained on general internet text. It may not represent
specialized terminology well.

```
"TOTVS"    → the model may embed this near random words
             instead of near "ERP", "enterprise resource planning", "fiscal module"

"Prophet"  → the model may embed this near "biblical prophet"
             instead of near "time series forecasting", "Facebook", "trend"
```

Solution: use a domain-specific embedding model, or add hybrid search (Chapter 14).

**Language quality**

Most popular embedding models are trained primarily on English text.
Portuguese content may produce lower-quality embeddings.

`nomic-embed-text` has some multilingual capability, but dedicated multilingual
models (`multilingual-e5-large`, `paraphrase-multilingual-mpnet-base-v2`) perform
better on non-English content.

**Chunk length mismatch**

Very long chunks produce "averaged" embeddings that dilute the specific meaning
of individual sentences within the chunk.

```
Long chunk (2000 chars) about "quarterly revenue and team structure and product roadmap"
→ embedding captures all three topics weakly

Short chunk (200 chars) about "quarterly revenue increased by 23%"
→ embedding captures this specific fact strongly
```

If your queries are specific questions, shorter chunks usually produce better
retrieval. We discuss chunk sizing in detail in Part 3.

---

### 3.8 How to think about embedding models

From a practical standpoint, think of an embedding model as a black box with
one contract:

```
input:  any text string
output: a fixed-size vector of floats
guarantee: similar inputs → similar outputs (geometrically close)
```

The internal mechanism (transformer architecture, training data, number of layers)
matters for understanding performance differences between models,
but you don't need to know it to use them effectively.

What you do need to know:
- The **dimension** of the output (384, 768, 1024, 1536)
  → must match what your vector store expects
- The **maximum input length** (many models cap at 512 tokens)
  → chunks longer than this will be truncated
- The **language support**
  → multilingual vs English-only
- The **domain coverage**
  → general vs specialized (legal, medical, code)

`nomic-embed-text` via Ollama:
- Dimension: 768
- Max input: 8192 tokens
- Language: primarily English, some multilingual capability
- Domain: general purpose

---

## Chapter 4 — Vector Stores — Databases for Semantic Search

A vector store is a database optimized for one specific operation:

> Given a query vector, find the N stored vectors most similar to it.

This is called **Approximate Nearest Neighbor (ANN) search**,
and it's what makes real-time semantic search possible at scale.

---

### 4.1 Why you can't use a regular database

You might wonder: why not just store the embeddings in PostgreSQL or SQLite
and compute distances at query time?

```sql
-- Naive approach: compute distance from all stored vectors
SELECT id, text, cosine_distance(embedding, query_embedding) AS dist
FROM chunks
ORDER BY dist ASC
LIMIT 5;
```

This works for small collections. At 1000 chunks, it's fast.
At 1,000,000 chunks, you're computing millions of distance calculations per query.
At 100,000,000 chunks (a large enterprise knowledge base), it's unusable.

Vector stores solve this with specialized indexing algorithms that find
approximate nearest neighbors in sublinear time — much faster than checking every vector.

---

### 4.2 How ChromaDB works

ChromaDB uses the **HNSW** algorithm (Hierarchical Navigable Small World)
to build a graph-based index over the stored vectors.

**The intuition behind HNSW:**

Think of it like a highway system:
- At the top level (highway): few nodes, long-range connections
- At the bottom level (local roads): all nodes, short-range connections

To find the nearest neighbors of a query:
1. Start at the top level — jump quickly to the approximate right region
2. Move down through levels — refine the search in progressively smaller neighborhoods
3. At the bottom level — return the final candidates

This gives O(log N) search complexity instead of O(N), at the cost of some
approximation (hence "approximate" nearest neighbor — you might miss the
theoretically closest vector by a tiny margin, but this is almost never a
problem in practice).

---

### 4.3 What ChromaDB stores for each document

For every chunk you index, ChromaDB stores three things:

```
1. The original text (page_content)
   → "Section 3.2: Digital products are non-refundable within 30 days..."

2. The metadata (source file, page number, etc.)
   → {"source": "policy.pdf", "page": 3}

3. The embedding vector
   → [0.021, -0.143, 0.087, ..., 0.312]  (768 numbers)
```

When you search, ChromaDB:
1. Takes your query text
2. Calls the embedding function → converts query to a vector
3. Uses HNSW to find the K closest stored vectors
4. Returns the corresponding texts + metadata + distances

```python
# Under the hood of retriever.invoke("refund policy for digital products"):

query_vector = embed("refund policy for digital products")
# → [0.019, -0.138, 0.091, ..., 0.308]

results = hnsw_search(query_vector, k=6)
# → [(chunk_text_1, metadata_1, distance_1),
#    (chunk_text_2, metadata_2, distance_2),
#    ...]
```

---

### 4.4 Persistence — what's on disk

ChromaDB writes two types of files to the `vectorstore/` directory:

```
vectorstore/
├── chroma.sqlite3              ← metadata, document text, chunk IDs
└── <collection-uuid>/
    ├── data_level0.bin         ← the HNSW index (the actual vectors)
    ├── header.bin              ← index metadata
    ├── length.bin              ← sizes of stored elements
    └── link_lists.bin          ← the graph connections in HNSW
```

The SQLite file stores what you can read as text (the actual chunk content,
metadata, and mapping from IDs to chunks).

The binary files store the numerical index — the HNSW graph with all the vectors.
These are not human-readable; they're the internal data structure
that makes fast search possible.

**This is why you can reload the vector store without re-embedding:**

```python
# src/vectorstore.py

# First run: builds and writes to disk
vectorstore = Chroma(persist_directory=str(VECTORSTORE_DIR), ...)
vectorstore.add_documents(chunks)     # embeds + writes binary files

# Subsequent runs: reads from disk (fast, no Ollama call for embedding)
vectorstore = Chroma(persist_directory=str(VECTORSTORE_DIR), ...)
count = vectorstore._collection.count()   # confirms data exists
# → 247 chunks already indexed
```

---

### 4.5 The `_collection.count()` pattern

In `src/vectorstore.py`:

```python
vectorstore = Chroma(
    collection_name=CHROMA_COLLECTION,
    persist_directory=str(VECTORSTORE_DIR),
    embedding_function=embeddings,
)

count = vectorstore._collection.count()

if count == 0:
    # First run — need to index
    ...
else:
    # Already indexed — load from disk
    log.info(f"Loaded existing vector store ({count} chunks)")
```

**Why `_collection.count()` and not just checking if the directory exists?**

The directory might exist but be empty (e.g., you deleted the binary files
but not the folder). Or the directory might exist with corrupted data.

`_collection.count()` queries the SQLite database directly — it's the most
reliable way to confirm that actual chunks are stored and available.

The underscore prefix (`_collection`) signals that this is a semi-private
attribute of the LangChain Chroma wrapper — not part of the official public API.
It works reliably in current versions and is the standard approach used across
the LangChain ecosystem, but be aware it could change in future versions.

---

### 4.6 Collections — namespacing your data

ChromaDB organizes data into **collections** — named namespaces within
a single vector store instance.

```python
# src/config.py
CHROMA_COLLECTION = "rag_collection"
```

```python
# src/vectorstore.py
vectorstore = Chroma(
    collection_name=CHROMA_COLLECTION,    # ← the namespace
    persist_directory=str(VECTORSTORE_DIR),
    ...
)
```

Why does this matter? If you build multiple RAG applications using the same
`vectorstore/` directory (e.g., one for HR documents, one for technical manuals),
you can keep them in separate collections within the same database.

For this project, we have a single collection (`rag_collection`).
For a multi-tenant system, you'd create one collection per tenant or domain.

---

### 4.7 The retriever abstraction

In LangChain, a "retriever" is any object that:
- Takes a string (the query)
- Returns a list of `Document` objects

```python
# The simplest retriever — wraps the vector store
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# Call it:
docs = retriever.invoke("refund policy for digital products")
# → [Document(page_content="Section 3.2...", metadata={...}),
#    Document(page_content="Returns must...", metadata={...}),
#    ...]
```

`as_retriever()` converts the Chroma vector store into a LangChain `Runnable`
(the standard composable interface). This is what lets you use it in LCEL chains:

```python
chain = retriever | format_docs | prompt | llm | output_parser
```

Without `as_retriever()`, you'd have a Chroma object (not a Runnable)
and the `|` composition wouldn't work.

---

### 4.8 Search types in ChromaDB via LangChain

LangChain exposes three search strategies through the `search_type` parameter:

**`"similarity"` (default)**
Returns the K most similar chunks, regardless of how similar they are.
Always returns exactly K results, even if they're all irrelevant.

```python
retriever = vs.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)
```

**`"similarity_score_threshold"` (what we use)**
Returns up to K chunks, but only those with distance ≤ threshold.
May return 0 results if nothing is similar enough.
This is what enables our fallback behavior in the agent.

```python
# src/retriever.py
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": TOP_K_RESULTS,          # max 6 chunks
        "score_threshold": SCORE_THRESHOLD,  # only if distance ≤ 0.2
    },
)
```

**`"mmr"` — Maximum Marginal Relevance**
Returns diverse chunks — balances relevance with variety.
Useful when you want to avoid returning 4 chunks that all say the same thing.

```python
retriever = vs.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.5}
    # fetch_k: initially retrieves 20, then selects 4 most diverse
    # lambda_mult: 0=max diversity, 1=max relevance
)
```

We use `similarity_score_threshold` because it enables our fallback logic:
if nothing passes the threshold → `node_retriever` returns empty list →
`edge_after_retrieval` routes to `node_fallback` → honest "not found" message.

---

### 4.9 Vector stores beyond ChromaDB

ChromaDB is our choice for this project because:
- Runs entirely locally (no external service)
- Persists to disk automatically
- Zero configuration required
- Good performance for small to medium collections (up to ~1M chunks)

Other vector stores you'll encounter:

| Vector store | Best for | Notes |
|---|---|---|
| ChromaDB | Local dev, small projects | What we use |
| Pinecone | Production at scale | Managed cloud service |
| Weaviate | Production, semantic search | Self-hosted or cloud |
| Qdrant | Production, filtering | Fast, Rust-based |
| pgvector | PostgreSQL users | Extension for Postgres |
| FAISS | Research, CPU-intensive | Facebook AI, no persistence layer |

LangChain wraps all of these with the same interface. Switching from ChromaDB
to Pinecone in production would require changing `src/vectorstore.py` only —
nothing else in the project needs to change.

---

### 4.10 To re-index from scratch

If you change your documents, add new files, or want to change chunk settings:

```bash
# Delete the existing index
rm -rf vectorstore/

# Re-run — will automatically re-index all documents in docs/
python main_chain.py
# or
python main_agent.py
```

The vectorstore directory is in `.gitignore` because:
1. It can be large (gigabytes for big document collections)
2. It's reproducible — anyone can recreate it by running the indexing step
3. It contains binary files that are not meaningful to diff in version control

---

### Chapter 3 & 4 — Summary

**Embeddings:**
- Convert text to vectors where semantic similarity = geometric proximity
- Trained with contrastive learning: pull similar pairs together, push different pairs apart
- ChromaDB measures cosine distance: 0.0 = identical, 1.0 = completely different
- Lower score = more relevant (this is the opposite of what "score" implies intuitively)
- Use a dedicated embedding model (nomic-embed-text), not the LLM, for speed + quality
- Calibrate SCORE_THRESHOLD using diagnostics.py on your actual documents

**Vector stores:**
- Specialized databases for ANN (Approximate Nearest Neighbor) search
- ChromaDB uses HNSW: fast O(log N) search instead of O(N) brute force
- Stores text + metadata + vectors; persists to disk in binary format
- `as_retriever()` wraps the vector store as a LangChain Runnable for use in chains
- `similarity_score_threshold` enables quality-gated retrieval + fallback behavior
- Switching vector stores only requires changing `src/vectorstore.py`

**The pipeline so far:**

```
docs/           ← your raw documents
    │
    ▼
load_documents() ← reads PDF/TXT into Document objects
    │
    ▼
split_documents() ← splits into 500-char chunks with 50-char overlap
    │
    ▼
OllamaEmbeddings ← converts each chunk to a 768-dim vector
    │
    ▼
ChromaDB         ← stores text + metadata + vectors on disk
    │
    ▼ (at query time)
retriever.invoke("question") ← embeds question, finds nearest chunks
    │
    ▼
[Document, Document, Document, Document]  ← the retrieved context
```

Everything from here feeds into the prompt and the LLM.
That's what Parts 3 and 4 cover.

---

*End of Part 2.*  
*Next: Part 3 — Document Loading, Chunking Strategy, and LangChain*  
*How to get data in, how to split it intelligently, and the framework that connects everything.*
