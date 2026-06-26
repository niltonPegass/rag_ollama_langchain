# Building RAG Systems from Scratch
## Part 5 of 8 — LangGraph: From Chains to Agents

**Series:** Building RAG Systems from Scratch  
**Part:** 5 of 8  
**Covers:** Chapters 8 and 9  
**Previous:** Part 4 — Building the Basic RAG Chain (Phase 1)  
**Next:** Part 6 — Observability with LangSmith and Langflow

---

## Chapter 8 — LangGraph: From Chains to Agents

Phase 1 gave us a working RAG pipeline. But it has a fundamental flaw:
every question follows the exact same path, regardless of whether the
retrieval produced useful results or not.

This chapter explains why that matters, and introduces LangGraph —
the tool that lets us add decision-making, loops, and state to our pipeline.

---

### 8.1 The problem with fixed chains

A LangChain chain is a **Directed Acyclic Graph (DAG)** — data flows in one
direction only, with no loops and no branching:

```
A → B → C → D → done
```

This is perfectly fine for simple, predictable pipelines.
But real applications hit situations where a fixed path fails:

**Situation 1: Retrieval returns irrelevant chunks**

```
User:      "What is the CEO's salary?"
Retriever: returns 4 chunks about "company history" and "product roadmap"
           (nothing about salaries — they're not in the documents)
Chain:     passes irrelevant chunks to LLM anyway
LLM:       hallucinates a salary figure based on the irrelevant context
User:      receives confidently wrong information
```

A chain has no mechanism to detect this and respond differently.

**Situation 2: A first attempt fails but a second would succeed**

Some queries are ambiguous. The first retrieval might not find the right chunks
because the question phrasing doesn't match the document vocabulary well.
A second attempt with the same query might still fail, but with a rephrased
query, it might succeed.

A chain can't retry. Once it passes through the retriever, it moves on to the LLM.

**What we need:**

```
After retrieval → evaluate quality → branch based on result:
    good chunks  → generate answer
    no chunks, can retry → try again
    no chunks, exhausted → honest fallback
```

This requires **conditional branching** and **loops** — neither of which
a plain chain supports.

---

### 8.2 What LangGraph adds

LangGraph is an extension of LangChain that models your application as a
**state machine** — a mathematical model you may have encountered in
computer science, digital circuits, or game AI.

A state machine has:
- A finite set of **states** (situations your application can be in)
- **Transitions** between states (how you move from one to another)
- **Shared state data** (information that persists across the run)
- A **starting state** and one or more **terminal states**

In LangGraph terminology:
- States → **nodes** (functions that do work)
- Transitions → **edges** (fixed) and **conditional edges** (branching)
- Shared state data → **AgentState** (a TypedDict passed to every node)
- Starting state → **entry point**
- Terminal state → **END**

---

### 8.3 Finite state machines — the mental model

You already know finite state machines intuitively, even if not by name.

A traffic light is a finite state machine:

```
States:      RED, GREEN, YELLOW
Transitions: RED → GREEN (after timer)
             GREEN → YELLOW (after timer)
             YELLOW → RED (after timer)
```

A vending machine is a finite state machine:

```
States:      IDLE, COLLECTING_MONEY, DISPENSING, RETURNING_CHANGE
Transitions: IDLE → COLLECTING_MONEY (coin inserted)
             COLLECTING_MONEY → DISPENSING (enough money inserted, item selected)
             COLLECTING_MONEY → RETURNING_CHANGE (cancel pressed)
             DISPENSING → IDLE (item dispensed)
```

Our RAG agent is a finite state machine:

```
States:      RETRIEVING, GENERATING, FALLBACK, DONE
Transitions: RETRIEVING → GENERATING (chunks found)
             RETRIEVING → RETRIEVING (no chunks, retry allowed)
             RETRIEVING → FALLBACK (no chunks, retries exhausted)
             GENERATING → DONE
             FALLBACK → DONE
```

LangGraph lets you implement this directly in Python, where each state
is a function and each transition is an edge in the graph.

---

### 8.4 TypedDict — typed state dictionaries

LangGraph passes state between nodes as a Python dictionary.
`TypedDict` adds type information to that dictionary:

```python
from typing import TypedDict, List
from langchain_core.documents import Document

class AgentState(TypedDict):
    question:   str
    documents:  List[Document]
    generation: str
    attempts:   int
```

**What is `TypedDict`?**

`TypedDict` is a special class from Python's `typing` module.
It's NOT a regular class — you can't instantiate it with `AgentState()`.

Instead, you create a plain Python dict that matches its structure:

```python
# WRONG — TypedDict is not a regular class
state = AgentState(question="...", documents=[], generation="", attempts=0)
# TypeError: Cannot instantiate typing.TypedDict

# CORRECT — create a plain dict matching the structure
state = {
    "question":   "What is the refund policy?",
    "documents":  [],
    "generation": "",
    "attempts":   0,
}
```

**Why use TypedDict at all then?**

Three reasons:

1. **IDE autocomplete:** your editor knows `state["question"]` is a `str`
   and `state["documents"]` is a `List[Document]`. It can autocomplete and
   warn you if you typo a key name.

2. **Documentation:** the TypedDict definition is the single authoritative
   description of what the state contains. Any developer reading the code
   immediately knows all fields and their types.

3. **LangGraph validation:** LangGraph uses the TypedDict schema to validate
   that nodes return the expected fields.

**`List[Document]` — generic type annotations:**

```python
from typing import List
from langchain_core.documents import Document

# List[Document] means: a list where every element is a Document object
documents: List[Document] = []

# Python doesn't enforce this at runtime — it's a hint, not a contract
# But your IDE will warn you if you do:
documents.append("not a document")  # IDE warns: expected Document, got str
```

In Python 3.9+, you can write `list[Document]` (lowercase) instead of `List[Document]`.
We use `List[Document]` (from `typing`) for compatibility with Python 3.8+.

---

### 8.5 The node contract

A LangGraph node is a plain Python function with one rule:

```
Input:  AgentState (or a dict matching its structure)
Output: AgentState (or a partial dict with updated fields)
```

```python
def node_retriever(state: AgentState) -> AgentState:
    documents = retriever.invoke(state["question"])
    return {
        **state,           # copy all existing fields unchanged
        "documents": documents,    # update this field
        "attempts":  state.get("attempts", 0) + 1,  # update this field
    }
```

**The `{**state, "key": value}` pattern:**

`**state` is Python's dictionary unpacking operator.
`{**state, "documents": documents}` means:
"create a new dict with all key-value pairs from `state`,
then add/override with `"documents": documents`."

```python
state = {"question": "hello", "documents": [], "generation": "", "attempts": 0}

# Update documents and attempts:
new_state = {**state, "documents": [doc1, doc2], "attempts": 1}
# → {"question": "hello", "documents": [doc1, doc2], "generation": "", "attempts": 1}
```

**Why not just modify `state` directly?**

```python
# Mutating approach (works, but risky):
state["documents"] = documents
state["attempts"] += 1
return state

# Immutable approach (safer):
return {**state, "documents": documents, "attempts": state["attempts"] + 1}
```

Mutating the input dict can cause subtle bugs if another part of the code
still holds a reference to the original state object. The immutable pattern
creates a new dict, leaving the original unchanged. This is a standard
functional programming principle — important in concurrent or complex systems.

---

### 8.6 Closures — nodes that capture their dependencies

In `src/agent.py`, nodes are defined inside `build_nodes()`:

```python
def build_nodes(retriever: VectorStoreRetriever, llm: ChatOllama) -> dict:

    generator_chain = RAG_PROMPT | llm | StrOutputParser()

    def node_retriever(state: AgentState) -> AgentState:
        documents = retriever.invoke(state["question"])  # uses `retriever` from outer scope
        ...

    def node_generator(state: AgentState) -> AgentState:
        answer = generator_chain.invoke(...)  # uses `generator_chain` from outer scope
        ...

    def node_fallback(state: AgentState) -> AgentState:
        ...

    return {
        "retriever": node_retriever,
        "generator": node_generator,
        "fallback":  node_fallback,
    }
```

`node_retriever`, `node_generator`, and `node_fallback` are defined inside
`build_nodes()`. They **close over** the variables from the outer scope —
`retriever`, `llm`, and `generator_chain` — capturing them even after
`build_nodes()` returns.

This is called a **closure**:

```python
# Demonstration of closure:
def make_multiplier(factor):
    def multiply(x):
        return x * factor   # factor is captured from outer scope
    return multiply         # returns the inner function

double = make_multiplier(2)
triple = make_multiplier(3)

double(5)   # → 10   (factor=2 is "remembered")
triple(5)   # → 15   (factor=3 is "remembered")
```

**Why use closures instead of global variables?**

```python
# BAD — global variables (hidden dependencies):
retriever = None   # global
llm = None         # global

def node_retriever(state):
    docs = retriever.invoke(...)   # uses global — not clear where it came from

# GOOD — closure (explicit dependencies):
def build_nodes(retriever, llm):
    def node_retriever(state):
        docs = retriever.invoke(...)  # clearly captured from build_nodes argument
```

With closures:
- Dependencies are explicit (passed as arguments to `build_nodes`)
- Tests can pass mock retrievers and LLMs without touching global state
- Multiple agents with different retrievers can coexist without conflict

---

### 8.7 Conditional edges — the routing function

After `node_retriever` runs, LangGraph calls `edge_after_retrieval` to decide
which node runs next:

```python
def edge_after_retrieval(state: AgentState) -> str:
    if len(state["documents"]) > 0:
        return "generator"

    if state.get("attempts", 0) >= MAX_RETRIES:
        log.warning(f"[router] no chunks after {MAX_RETRIES} attempts → fallback")
        return "fallback"

    log.info("[router] no chunks found → retrying")
    return "retriever"
```

**This function returns a string — not a node function.**

LangGraph uses string names to identify nodes for two reasons:
1. Strings can be serialized (saved to JSON, logged, traced)
2. The graph structure (which node name maps to which function) is defined
   separately in `add_conditional_edges()`

The mapping in `build_agent()` connects the return string to the actual node:

```python
graph.add_conditional_edges(
    "retriever",           # this node triggers the routing
    edge_after_retrieval,  # called after "retriever" runs
    {
        "generator": "generator",   # if routing_fn returns "generator" → run node_generator
        "retriever": "retriever",   # if routing_fn returns "retriever" → run node_retriever (loop)
        "fallback":  "fallback",    # if routing_fn returns "fallback"  → run node_fallback
    },
)
```

**The self-loop — `"retriever": "retriever"`:**

This is what creates the retry behavior. If `edge_after_retrieval` returns
`"retriever"`, the graph runs `node_retriever` again with the updated state
(which now has `attempts=1` instead of `attempts=0`).

On the second run, if still no chunks: `attempts=2 >= MAX_RETRIES=2` → fallback.

This is a loop in the graph — something impossible with a plain chain.

---

### 8.8 The END sentinel

```python
from langgraph.graph import END

graph.add_edge("generator", END)
graph.add_edge("fallback",  END)
```

`END` is a special constant defined by LangGraph.
When a node transitions to `END`, the graph stops running and returns
the final `AgentState` dict to the caller.

The caller (`main_agent.py`) extracts the answer:

```python
result = agent.invoke({...})
print(result["generation"])  # the final answer
```

---

### 8.9 Compiling the graph

```python
graph = StateGraph(AgentState)

for name, fn in nodes.items():
    graph.add_node(name, fn)

graph.set_entry_point("retriever")
graph.add_conditional_edges(...)
graph.add_edge("generator", END)
graph.add_edge("fallback",  END)

agent = graph.compile()
```

`StateGraph(AgentState)` creates a graph builder.
`AgentState` tells LangGraph what the state structure looks like.

`graph.compile()` does several things:
1. **Validates** the graph (no unreachable nodes, entry point defined, all
   conditional edge return values are mapped)
2. **Returns** a `CompiledGraph` — a Runnable with `.invoke()`, `.stream()`, etc.
3. **Enables tracing** — LangSmith can now trace every node execution

After compilation, the graph is immutable — you can't add more nodes or edges.

**The `for name, fn in nodes.items():` loop:**

`nodes` is a dict returned by `build_nodes()`:
```python
nodes = {
    "retriever": node_retriever,
    "generator": node_generator,
    "fallback":  node_fallback,
}
```

`.items()` returns key-value pairs:
```python
for name, fn in nodes.items():
    graph.add_node(name, fn)
# equivalent to:
# graph.add_node("retriever", node_retriever)
# graph.add_node("generator", node_generator)
# graph.add_node("fallback",  node_fallback)
```

The loop avoids repeating `add_node()` for each node manually.
If you add a new node to `build_nodes()`, you don't need to touch `build_agent()`.

---

## Chapter 9 — Building the Agentic RAG (Phase 2)

Now let's trace through complete executions of the agent —
both the happy path (chunks found) and the failure path (no chunks).

---

### 9.1 The initial state

Every agent invocation starts with a fresh state dict:

```python
result = agent.invoke({
    "question":   "What is the refund policy?",
    "documents":  [],   # empty — retriever hasn't run yet
    "generation": "",   # empty — generator hasn't run yet
    "attempts":   0,    # zero — no retrieval attempts yet
})
```

This is the "input" to the state machine.
Each node will update fields and pass the updated state forward.

---

### 9.2 Happy path — chunks found on first attempt

Let's trace a question that exists in the documents:

**Entry → node_retriever (attempt 1)**

```python
# state coming in:
{
    "question":   "What is the refund policy?",
    "documents":  [],
    "generation": "",
    "attempts":   0,
}

# Inside node_retriever:
documents = retriever.invoke("What is the refund policy?")
# ChromaDB returns 4 relevant chunks (cosine distance < 0.2)
# [Document("Section 3.2: Refunds..."), Document("Returns must be..."), ...]

# state going out:
{
    "question":   "What is the refund policy?",
    "documents":  [Document(...), Document(...), Document(...), Document(...)],
    "generation": "",
    "attempts":   1,    # incremented from 0 to 1
}
```

**Routing → edge_after_retrieval**

```python
state["documents"] = [Document(...), ...]  # not empty
len(state["documents"]) > 0               # True
→ return "generator"
```

**→ node_generator**

```python
# state coming in:
{
    "question":   "What is the refund policy?",
    "documents":  [Document("Section 3.2..."), Document("Returns must be...")],
    "generation": "",
    "attempts":   1,
}

# Inside node_generator:
context = "Section 3.2: Refunds are non-refundable within 30 days...\n\nReturns must be..."

answer = generator_chain.invoke({
    "context":  context,
    "question": "What is the refund policy?",
})
# LLM reads the context and generates:
# "According to the documents, digital products are non-refundable within 30 days..."

# state going out:
{
    "question":   "What is the refund policy?",
    "documents":  [Document(...), Document(...)],
    "generation": "According to the documents, digital products are non-refundable...",
    "attempts":   1,
}
```

**→ END**

```python
result = {
    "question":   "What is the refund policy?",
    "documents":  [Document(...), Document(...)],
    "generation": "According to the documents, digital products are non-refundable...",
    "attempts":   1,
}

# main_agent.py extracts:
print(result["generation"])
# → "According to the documents, digital products are non-refundable within 30 days..."
```

Total nodes executed: 2 (retriever → generator)
Total LLM calls: 1 (generator)
No hallucination: the answer is grounded in the retrieved chunks.

---

### 9.3 Failure path — no chunks found, fallback triggered

Now a question that is NOT in the documents:

**Entry → node_retriever (attempt 1)**

```python
# state coming in:
{"question": "What is the CEO's home address?", "documents": [], "generation": "", "attempts": 0}

# Inside node_retriever:
documents = retriever.invoke("What is the CEO's home address?")
# ChromaDB finds no chunks with cosine distance < 0.2 (nothing relevant)
# → []

# state going out:
{"question": "What is the CEO's home address?", "documents": [], "generation": "", "attempts": 1}
```

**Routing → edge_after_retrieval (after attempt 1)**

```python
len(state["documents"]) > 0      # False — documents is []
state.get("attempts", 0) >= MAX_RETRIES  # 1 >= 2 → False
→ return "retriever"             # retry
```

**→ node_retriever (attempt 2 — the retry)**

```python
# Same question, same ChromaDB — same result
documents = retriever.invoke("What is the CEO's home address?")
# → []

# state going out:
{"question": "...", "documents": [], "generation": "", "attempts": 2}
```

**Routing → edge_after_retrieval (after attempt 2)**

```python
len(state["documents"]) > 0      # False
state.get("attempts", 0) >= MAX_RETRIES  # 2 >= 2 → True
→ return "fallback"
```

**→ node_fallback**

```python
# No LLM call here — just a deterministic string
return {
    **state,
    "generation": "I could not find relevant information in the documents to answer this question.",
}
```

**→ END**

```python
result["generation"]
# → "I could not find relevant information in the documents to answer this question."
```

Total nodes executed: 3 (retriever → retriever → fallback)
Total LLM calls: 0 (fallback doesn't call the LLM)
No hallucination: explicit, honest failure message.

---

### 9.4 Why `state.get("attempts", 0)` instead of `state["attempts"]`

```python
# state["attempts"] — raises KeyError if "attempts" key doesn't exist
state["attempts"]

# state.get("attempts", 0) — returns 0 if "attempts" key doesn't exist
state.get("attempts", 0)
```

Even though we always initialize `"attempts": 0` before invoking the agent,
`.get()` with a default is defensive programming:

- If someone calls a node function directly in a test without providing all fields
- If a future refactor changes the initialization
- If LangGraph internally calls the routing function before state is fully set up

`.get("key", default)` prevents a `KeyError` in any of these edge cases.
It costs nothing and prevents potential bugs.

---

### 9.5 main_agent.py — the entry point for Phase 2

```python
def select_model() -> str:
    options = ", ".join(AVAILABLE_LLMS)
    choice  = input(f"Choose model ({options}) [default: {DEFAULT_LLM}]: ").strip()

    if choice not in AVAILABLE_LLMS:
        if choice:
            print(f"Unknown model '{choice}' — using default: {DEFAULT_LLM}")
        return DEFAULT_LLM

    return choice
```

**`", ".join(AVAILABLE_LLMS)`:**

`str.join(iterable)` concatenates elements of an iterable with the string
as separator:

```python
", ".join(["llama3.2", "mistral"])
# → "llama3.2, mistral"

" | ".join(["a", "b", "c"])
# → "a | b | c"
```

This is more idiomatic than:
```python
AVAILABLE_LLMS[0] + ", " + AVAILABLE_LLMS[1]  # fragile, breaks with 3+ models
```

**`if choice not in AVAILABLE_LLMS:`:**

`in` tests membership in a list:
```python
"mistral" in ["llama3.2", "mistral"]   # → True
"gpt-4"   in ["llama3.2", "mistral"]   # → False
```

`not in` is the negation. This validates user input: if they type something
not in the list, we fall back to the default.

**`if choice:` (inner check):**

An empty string is falsy in Python:
```python
bool("")    # → False
bool("x")   # → True
```

`if choice:` means "if the user typed something (not just pressed Enter)."
If they pressed Enter without typing, `choice = ""`, which is falsy —
we silently use the default without printing a warning.
If they typed "gpt-4" (not in the list), `choice = "gpt-4"` is truthy —
we print a warning and use the default.

---

**The main loop:**

```python
def main():
    model = select_model()
    log.info(f"Starting RAG agent (Phase 2) — model={model}")

    vectorstore = build_vectorstore()
    retriever   = build_retriever(vectorstore)
    agent       = build_agent(retriever, model=model)

    print("\n[RAG Agent] Type your question (or /bye to exit)\n")

    while True:
        question = input("Question: ").strip()

        if not question:
            continue

        if question.lower() == "/bye":
            print("Shutting down...")
            subprocess.run(["ollama", "stop", model])
            subprocess.run(["ollama", "stop", EMBEDDING_MODEL])
            break

        result = agent.invoke({
            "question":   question,
            "documents":  [],
            "generation": "",
            "attempts":   0,
        })

        print(f"\nAnswer: {result['generation']}\n")
```

Notice the **dependency chain** in `main()`:

```
build_vectorstore()              → Chroma instance
    ↓
build_retriever(vectorstore)     → VectorStoreRetriever instance
    ↓
build_agent(retriever, model)    → CompiledGraph (the agent)
```

Each function takes the output of the previous one.
This is the **explicit dependency injection** pattern:
- `build_vectorstore()` doesn't know about retrievers or agents
- `build_retriever()` doesn't know about agents or models
- `build_agent()` doesn't know about ChromaDB or document loading

Each module has a clearly bounded responsibility.
`main()` is the only place that knows about the full dependency chain.

**f-strings:**

```python
print(f"\nAnswer: {result['generation']}\n")
```

f-strings (formatted string literals) embed expressions inside `{}`:

```python
name = "world"
f"Hello, {name}!"          # → "Hello, world!"
f"2 + 2 = {2 + 2}"        # → "2 + 2 = 4"
f"{result['generation']}"  # → the value of result["generation"]
```

Note: inside an f-string, use single quotes `'generation'` if the f-string
itself uses double quotes `f"..."`. Mixing avoids the need to escape quotes.

---

### 9.6 Phase 1 vs Phase 2 — a direct comparison

| Aspect | Phase 1 (chain.py) | Phase 2 (agent.py) |
|---|---|---|
| Structure | Linear chain (fixed path) | State machine (branching) |
| Retrieval | Always runs, always returns k results | Threshold-gated, may return 0 |
| Failed retrieval | LLM receives empty/irrelevant context | Detected, retried, then fallback |
| Retry logic | None | Up to MAX_RETRIES |
| Fallback | None (LLM hallucinates) | Explicit "not found" message |
| LLM calls | Always 1 | 0 (fallback) to 1 (success) |
| Observability | Single trace | Per-node traces in LangSmith |
| Extensibility | Add step = rewrite chain | Add step = add node + edge |

**When to use Phase 1 vs Phase 2:**

Use the chain (Phase 1) when:
- You're prototyping and want minimal code
- Your documents cover all possible questions (retrieval never fails)
- You need maximum speed (one fewer abstraction layer)

Use the agent (Phase 2) when:
- Retrieval might fail for some questions
- You need honest "not found" behavior (no hallucination)
- You want per-node visibility in LangSmith
- You'll add more complexity later (query rewriting, tool use, memory)

---

### 9.7 The complete startup sequence for Phase 2

When you run `python main_agent.py`:

```
1. warnings.filterwarnings() — suppress deprecation warnings
2. load_dotenv() — activate LangSmith tracing
3. select_model() — prompt user: "Choose model (llama3.2, mistral):"
4. build_vectorstore()
   → Open ChromaDB connection
   → count == 0? → index documents (slow, first run only)
   → count > 0?  → load from disk (fast)
5. build_retriever(vectorstore)
   → vectorstore.as_retriever(search_type="similarity_score_threshold", ...)
   → returns VectorStoreRetriever
6. build_agent(retriever, model)
   → ChatOllama(model=model, temperature=0)
   → build_nodes(retriever, llm) → {node functions as closures}
   → StateGraph(AgentState)
   → add_node × 3, set_entry_point, add_conditional_edges, add_edge × 2
   → graph.compile() → CompiledGraph
7. "[RAG Agent] ready" — everything is initialized
8. while True → input("Question: ")
9. For each question:
   → agent.invoke({question, documents=[], generation="", attempts=0})
   → LangGraph runs the state machine:
      node_retriever → edge_after_retrieval → node_generator or node_fallback → END
   → print(result["generation"])
10. /bye → ollama stop → break → exit
```

---

### Chapter 8 & 9 — Summary

**Why chains aren't enough:**
- Fixed path → no branching, no retry, no fallback
- Empty context → LLM hallucinates
- No way to detect retrieval quality inside the pipeline

**LangGraph concepts:**
- State machine: states (nodes) + transitions (edges) + shared data (AgentState)
- `TypedDict`: typed dict schema — not a regular class, creates plain dicts
- `List[Document]`: type hint for lists of Document objects
- Nodes: plain Python functions, `AgentState → AgentState`
- `{**state, "key": value}`: immutable state update pattern
- Closures: inner functions capture outer variables (retriever, llm) — explicit dependencies
- Conditional edges: routing function returns a string key → mapped to next node
- Self-loop: `"retriever": "retriever"` in the mapping creates retry behavior
- `END`: sentinel that terminates the graph and returns final state
- `graph.compile()`: validates + builds the runnable CompiledGraph

**Execution traces:**
- Happy path: retriever → generator → END (1 LLM call)
- Failure path: retriever → retriever → fallback → END (0 LLM calls)

**main_agent.py patterns:**
- `", ".join(list)` for readable list formatting
- `choice not in list` for membership validation
- `if choice:` for truthy/falsy empty string check
- Explicit dependency injection: `vectorstore → retriever → agent`
- f-strings with `{}` for inline expression evaluation

**The key architectural win:**
The fallback node makes failure explicit.
Instead of silently hallucinating, the agent says "I couldn't find this."
That's the difference between a prototype and a trustworthy system.

---

*End of Part 5.*  
*Next: Part 6 — LangSmith (Observability) and Langflow (Visual Pipelines)*  
*How to see inside your RAG agent and how to represent it visually.*
