# Building RAG Systems from Scratch
## Part 1 of 8 — The Problem RAG Solves + How LLMs Work

**Series:** Building RAG Systems from Scratch  
**Part:** 1 of 8  
**Covers:** Chapters 1 and 2  
**Next:** Part 2 — Embeddings and Vector Stores

---

> This material was built from a real, working project: `rag-langchain`.
> Every concept here maps directly to code you can run.
> The goal is not just to understand *what* things are,
> but *why* they exist, *when* to use them, and *how* they fit together.

---

## Chapter 1 — The Problem RAG Solves

### 1.1 The fundamental limitation of LLMs

A large language model is trained on a massive corpus of text — books, websites,
academic papers, code repositories — frozen at a specific point in time.
When you ask it a question, it answers from that frozen snapshot of the world.

This creates three hard problems in production systems:

---

**Problem 1: Knowledge cutoff**

The model doesn't know anything that happened after its training date.
If you ask about last month's news, it can't answer accurately.
It may try — and produce a fluent, confident-sounding response that is simply wrong.

```
User:  "What happened at the company board meeting last Tuesday?"
LLM:   [has no idea, but generates something plausible-sounding]
```

---

**Problem 2: Private data blindness**

The model was never trained on your organization's internal documents:
contracts, policies, wikis, reports, product manuals.
No matter how capable the model is, it cannot answer questions about
data it has never seen.

```
User:  "What does our refund policy say about digital products?"
LLM:   [invents a policy that sounds reasonable but doesn't match yours]
```

---

**Problem 3: Hallucination**

When the model doesn't know the answer, it doesn't say "I don't know."
It generates a response that is grammatically correct, stylistically confident,
and potentially completely fabricated.

This is called **hallucination** — and it is the most dangerous property of LLMs
in any system where accuracy matters.

Hallucination happens because the model's training objective is to predict the
next token, not to verify factual accuracy. The model has no internal signal
distinguishing "I know this" from "I'm making this up."

---

### 1.2 The RAG approach

RAG — **Retrieval-Augmented Generation** — solves all three problems with one idea:

> **Before generating an answer, retrieve the relevant information
> and inject it into the prompt as context.**

The LLM becomes a reasoning engine, not a knowledge store.
Knowledge lives in your documents.
The LLM's job is to read, synthesize, and answer based on what it was given.

```
WITHOUT RAG:
  User question → LLM (uses training data) → answer

WITH RAG:
  User question
      → search your documents
      → retrieve relevant passages
      → build prompt: [instructions + passages + question]
      → LLM (uses ONLY the provided passages) → grounded answer
```

A concrete example:

```
User:  "What does our refund policy say about digital products?"

RAG pipeline:
  1. Embed the question → search policy document
  2. Retrieve: "Section 3.2: Digital products are non-refundable
                within 30 days unless the product is defective..."
  3. Prompt: "Answer based ONLY on the context below.
              Context: [retrieved passage]
              Question: What does our refund policy say about digital products?"
  4. LLM answers based on the exact text from your document
```

No hallucination. No invented policy. The answer is grounded in your document.

---

### 1.3 How RAG addresses each problem

| Problem | How RAG solves it |
|---|---|
| Knowledge cutoff | Documents are retrieved fresh on every query — you can add new docs anytime |
| Private data blindness | Your documents are indexed; the LLM answers from them |
| Hallucination | Prompt instructs the LLM to use ONLY the provided context; fallback if nothing found |

---

### 1.4 The two phases of a RAG system

A RAG system operates in two completely separate phases:

---

**Phase A: Indexing (runs once, offline)**

This is the preparation phase. You run it when you add or update documents.

```
Raw files (PDF, TXT, ...)
    │
    ▼
Load documents          ← read the files into memory
    │
    ▼
Split into chunks       ← divide each document into smaller pieces (~500 chars)
    │
    ▼
Embed each chunk        ← convert each piece to a vector (list of numbers)
    │
    ▼
Store in vector DB      ← save vectors + original text to disk
```

In our project, this is handled by:
- `src/loader.py` — loading and splitting
- `src/vectorstore.py` — embedding and storing

---

**Phase B: Query (runs on every user question)**

This is the real-time phase. It runs every time a user asks something.

```
User question
    │
    ▼
Embed the question      ← convert question to vector
    │
    ▼
Search vector DB        ← find the K chunks most similar to the question
    │
    ▼
Build prompt            ← combine: instructions + retrieved chunks + question
    │
    ▼
LLM generates answer    ← reads only what was provided
    │
    ▼
Return answer to user
```

In our project, this is handled by:
- `src/retriever.py` — searching the vector DB
- `src/prompts.py` — building the prompt
- `src/chain.py` (Phase 1) or `src/agent.py` (Phase 2) — orchestrating the flow

---

### 1.5 When RAG is the right tool

RAG is the right choice when:

- You have a corpus of documents (PDFs, wikis, manuals, reports, contracts)
- Users ask factual questions that require specific answers from those documents
- The information changes over time (new docs can be added without retraining)
- You cannot or do not want to fine-tune a model (cost, time, data sensitivity)
- Accuracy and traceability matter (you need to cite sources)

---

RAG is NOT the right tool when:

- **You need the model to learn a new behavior or skill** → fine-tuning
  (e.g., "always respond in a specific tone" or "learn our domain terminology deeply")

- **Your documents change continuously at very high frequency** → streaming pipelines
  (e.g., live stock prices — by the time you index it, it's outdated)

- **Questions require reasoning across hundreds of documents simultaneously**
  → more sophisticated multi-agent architectures with planning

- **The answer is simple and doesn't require documents**
  → a direct LLM call without retrieval is faster and cheaper

---

### 1.6 RAG vs fine-tuning — the key distinction

This is a common point of confusion, so it deserves a clear explanation.

**Fine-tuning** modifies the model's weights — it changes what the model "knows"
at a fundamental level. It's expensive, time-consuming, and requires retraining
whenever your knowledge changes.

**RAG** doesn't touch the model at all. It changes what the model *sees* at
inference time. Knowledge lives outside the model, in documents. You can update,
add, or remove documents without any retraining.

```
Fine-tuning: knowledge baked INTO the model
RAG:         knowledge provided TO the model at query time
```

For most enterprise use cases (internal knowledge bases, document Q&A,
policy assistants), RAG is the better approach:
- Faster to build (days, not weeks)
- Cheaper to maintain (update documents, not retrain models)
- More transparent (you can see exactly what passages informed the answer)
- More controllable (change the prompt, change the behavior)

---

## Chapter 2 — How Language Models Work (What You Need to Know)

You don't need to understand the full transformer architecture to build RAG systems.
But there are five properties of LLMs that directly affect how you design a RAG pipeline.

---

### 2.1 Tokens, not words or characters

LLMs don't process text character by character or word by word.
They process **tokens** — subword units produced by a tokenizer.

A tokenizer splits text into pieces based on frequency in the training data.
Common words become single tokens; rare words are split into subword pieces.

```
"artificial"     → ["art", "ificial"]          (2 tokens)
"intelligence"   → ["intelligence"]             (1 token)
"RAG"            → ["R", "AG"]                  (2 tokens)
"hello"          → ["hello"]                    (1 token)
"rag-langchain"  → ["rag", "-", "lang", "chain"] (4 tokens)
```

As a rough approximation: **1 token ≈ 0.75 words** in English.
A 1000-word document is approximately 1300 tokens.

**Why does this matter for RAG?**

Every model has a **context window** — a hard limit on how many tokens it can
process in a single call. Common limits:

| Model | Context window |
|---|---|
| llama3.2 3B | 128,000 tokens |
| mistral 7B  | 32,000 tokens  |
| GPT-4o      | 128,000 tokens |

If your prompt (instructions + retrieved chunks + question) exceeds the context window,
the model will truncate the input or throw an error.

This is one reason we chunk documents into ~500-character pieces:
a large document might be 100,000 tokens — we can't paste it all into every prompt.
By retrieving only the 4-6 most relevant chunks, we keep the prompt manageable.

---

### 2.2 Temperature — controlling randomness

Temperature is the parameter that controls how "creative" or "deterministic"
the model's output is.

**How it works technically:**

At each generation step, the model computes a probability distribution over
all possible next tokens (the vocabulary may have 32,000+ tokens).

Temperature scales this distribution before sampling:

```
Low temperature (→ 0):  peak probabilities become sharper → model picks the most likely token
High temperature (→ 2): probabilities flatten → model picks more varied tokens
```

In practice:

```
temperature = 0:    deterministic — always picks the single most likely token
temperature = 0.3:  slightly varied — mostly consistent, small creative variation
temperature = 0.7:  balanced — good for creative writing
temperature = 1.0:  sampling proportionally — unpredictable, often diverges
temperature > 1.0:  often incoherent
```

**For RAG, always use temperature = 0.**

We want the model to extract and synthesize information from the provided context,
not to creatively interpret or embellish it. Consistency is more important than
variety in a Q&A system.

In our project:
```python
# src/config.py
LLM_TEMPERATURE = 0   # deterministic — same input → same output
```

```python
# src/chain.py and src/agent.py
llm = ChatOllama(model=model, temperature=LLM_TEMPERATURE)
```

---

### 2.3 System messages vs human messages — the chat format

Modern LLMs (like llama3.2 and mistral) are "chat models" — they're trained
with a specific conversation format that distinguishes between roles.

The three roles:

```
system    → background instructions set by the developer (invisible to the end user)
human     → the user's message
assistant → the model's response
```

In code, a conversation is a list of messages:

```python
messages = [
    {"role": "system",    "content": "You are a helpful assistant that answers only from provided documents."},
    {"role": "human",     "content": "What is the refund policy?"},
    # the model generates the next "assistant" message
]
```

For multi-turn conversations (follow-up questions), you append each exchange:

```python
messages = [
    {"role": "system",    "content": "You are a helpful assistant..."},
    {"role": "human",     "content": "What is the refund policy?"},
    {"role": "assistant", "content": "Digital products are non-refundable within 30 days..."},
    {"role": "human",     "content": "And what about physical products?"},  ← follow-up
]
```

**LangChain handles this automatically.**

When you use `ChatPromptTemplate.from_template()`, LangChain wraps your
template in a `HumanMessage` object and passes it to the model in the correct
format for whichever model you're using:

```python
# src/prompts.py
RAG_PROMPT = ChatPromptTemplate.from_template("""
You are an assistant that answers questions based on provided documents.
Use ONLY the context below...

Context: {context}
Question: {question}
""")

# When invoked, this becomes:
# HumanMessage(content="You are an assistant... Context: [chunks] Question: [user question]")
```

Note: In our current project, we put everything in a single human message
(including the system instruction). For more control, you can use `from_messages()`:

```python
template = ChatPromptTemplate.from_messages([
    ("system", "You are an assistant that only answers from provided documents."),
    ("human",  "Context:\n{context}\n\nQuestion: {question}"),
])
```

---

### 2.4 What the LLM actually receives in our RAG system

Let's make this concrete. When a user asks "What does section 3.2 say?",
here is the complete text that gets sent to the LLM:

```
[Human message]:

You are an assistant that answers questions based on provided documents.
Use ONLY the context below to answer. Be direct and objective.
If the context does not contain the answer, say exactly:
"I could not find this information in the documents."

Context:
Section 3.2: Digital products purchased on or after January 1st are
non-refundable unless the product is proven defective. To initiate a
refund claim, customers must contact support within 30 days of purchase.

Refund requests for physical products follow a different process.
All physical product returns must include the original packaging and receipt.

Question: What does section 3.2 say?

Answer:
```

The model never "sees" the full original document — only the passages we retrieved.
The quality of the final answer depends entirely on whether we retrieved the right passages.

This is why retrieval quality is the most important factor in a RAG system.
A great LLM with bad retrieval produces bad answers.
A mediocre LLM with excellent retrieval produces good answers.

**Optimize retrieval first. Optimize the LLM second.**

---

### 2.5 Non-determinism and why it matters for testing

Even with temperature = 0, LLMs are not perfectly deterministic in practice.

Sources of non-determinism:
- **Floating-point arithmetic:** GPU/CPU floating-point operations can vary
  slightly between hardware and library versions
- **Batching:** if multiple requests are batched together, results can differ
- **Model updates:** if the underlying model is updated, outputs change

**The practical implication for testing:**

You cannot write tests like:
```python
assert llm.invoke("What is 2+2?") == "2+2 equals 4."
```

Because the exact phrasing of the output may change.

Instead, test the things that ARE deterministic — your Python logic:
```python
# routing logic (pure Python, always deterministic)
assert edge_after_retrieval({"documents": [doc], "attempts": 1}) == "generator"

# prompt template rendering (pure Python, always deterministic)
rendered = RAG_PROMPT.invoke({"context": "x", "question": "y"})
assert "ONLY" in rendered.messages[0].content
```

This is the testing strategy we follow in `tests/` — covered in detail in Part 7.

---

### 2.6 Local vs API-based LLMs — the Ollama choice

Our project uses Ollama to run LLMs locally. This is a deliberate architectural choice.

**API-based LLMs (OpenAI, Anthropic, Google):**
- No setup required
- Best model quality available
- Pay per token (cost at scale)
- Data leaves your machine (privacy implications)
- Latency depends on network + API queue

**Local LLMs via Ollama:**
- Requires download + local compute
- Lower quality than frontier models
- Free to run (hardware cost only)
- Data never leaves your machine (privacy by default)
- Latency depends only on your hardware

For a portfolio RAG project, local inference is ideal:
- Zero ongoing cost
- No API key management
- Privacy — you can index sensitive documents safely
- Works offline

The trade-off: local models (llama3.2 3B, mistral 7B) are significantly
less capable than GPT-4 or Claude for complex reasoning. For the retrieval
and synthesis tasks in RAG, this is usually acceptable.

**The Ollama abstraction in LangChain:**

```python
# src/chain.py and src/agent.py
from langchain_ollama import ChatOllama, OllamaEmbeddings

# LLM for text generation
llm = ChatOllama(model="mistral", temperature=0)

# Separate model for generating embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")
```

`ChatOllama` and `OllamaEmbeddings` implement the same LangChain interfaces
as `ChatOpenAI` and `OpenAIEmbeddings`. Swapping between local and API
models requires changing one line in `src/config.py`.

---

### Chapter 1 & 2 — Summary

**The core insight of RAG:**
An LLM is a reasoning engine, not a knowledge store.
Separate what the model *knows how to do* (reason, synthesize, write)
from what it *needs to know* (your documents).

**The three problems RAG solves:**
1. Knowledge cutoff → retrieve fresh documents at query time
2. Private data blindness → index your documents, retrieve from them
3. Hallucination → ground the LLM in retrieved context, add explicit fallback

**The two RAG phases:**
- Indexing (offline): load → split → embed → store
- Querying (real-time): embed question → retrieve → build prompt → LLM → answer

**Key LLM properties that affect RAG design:**
- Tokens determine context limits → reason for chunking documents
- Temperature = 0 for deterministic, factual RAG responses
- Chat format (system/human/assistant) for structured prompting
- The LLM only sees what you give it → retrieval quality is everything
- Non-determinism means test your logic, not the model's output

---

*End of Part 1.*
*Next: Part 2 — Embeddings and Vector Stores*
*The mathematical foundation of how semantic search actually works.*
