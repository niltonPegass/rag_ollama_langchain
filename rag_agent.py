import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv
load_dotenv()  # carrega o .env antes de tudo

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama

###

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from langchain_core.documents import Document
import glob, subprocess

models = ['llama3.2', 'mistral']
MODEL = input('Escolha o modelo que será utilizado' + f' ({", ".join(models)}): ').strip()

# ── 1. Vector store ──────────────────────────────────────────────────
def load_vectorstore():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vs = Chroma(persist_directory="vectorstore/", embedding_function=embeddings)
    print(f"[vectorstore] {vs._collection.count()} chunks carregados")
    return vs

# ── 2. Estado ────────────────────────────────────────────────────────
class AgentState(TypedDict):
    question:   str
    documents:  List[Document]
    generation: str
    attempts:   int

# ── 3. LLM e retriever ───────────────────────────────────────────────
llm = ChatOllama(model=f"{MODEL}", temperature=0)
vs  = load_vectorstore()

# retriever com score — retorna (doc, score) onde score é distância (menor = mais similar)
retriever = vs.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 6, "score_threshold": 0.2},
)

# ── 4. Nós ───────────────────────────────────────────────────────────

def node_retriever(state: AgentState) -> AgentState:
    docs = retriever.invoke(state["question"])
    print(f"[retriever] {len(docs)} chunks acima do threshold")
    state["documents"] = docs
    state["attempts"]  = state.get("attempts", 0) + 1
    return state

generator_prompt = ChatPromptTemplate.from_template("""
Você é um assistente que responde perguntas com base em documentos fornecidos.
Use APENAS o contexto abaixo para responder. Seja direto e objetivo.
Se o contexto não contiver a resposta, diga exatamente: "Não encontrei essa informação nos documentos."

Contexto:
{context}

Pergunta: {question}

Resposta:""")

generator_chain = generator_prompt | llm | StrOutputParser()

def node_generator(state: AgentState) -> AgentState:
    context = "\n\n".join(d.page_content for d in state["documents"])
    answer  = generator_chain.invoke({
        "context":  context,
        "question": state["question"]
    })
    state["generation"] = answer
    return state

def node_fallback(state: AgentState) -> AgentState:
    state["generation"] = "Não encontrei informações relevantes nos documentos para responder essa pergunta."
    return state

# ── 5. Edges condicionais ────────────────────────────────────────────

def edge_after_retrieval(state: AgentState) -> str:
    if len(state["documents"]) > 0:
        return "generator"
    if state.get("attempts", 0) >= 2:
        print("[retriever] 2 tentativas sem resultado — fallback")
        return "fallback"
    print("[retriever] sem chunks — tentando novamente")
    return "retriever"

# ── 6. Grafo ─────────────────────────────────────────────────────────
graph = StateGraph(AgentState)

graph.add_node("retriever", node_retriever)
graph.add_node("generator", node_generator)
graph.add_node("fallback",  node_fallback)

graph.set_entry_point("retriever")

graph.add_conditional_edges("retriever", edge_after_retrieval, {
    "generator": "generator",
    "retriever": "retriever",
    "fallback":  "fallback",
})

graph.add_edge("generator", END)
graph.add_edge("fallback",  END)

agent = graph.compile()

# ── 7. Main ──────────────────────────────────────────────────────────
if __name__ == "__main__":

    print("\n[Agente RAG] Digite sua pergunta (ou /bye para sair)\n")
    while True:
        question = input("Pergunta: ")
        if question.strip().lower() == "/bye":
            print("Encerrando...")
            subprocess.run(["ollama", "stop", f"{MODEL}"])
            subprocess.run(["ollama", "stop", "nomic-embed-text"])
            break
        result = agent.invoke({
            "question":   question,
            "documents":  [],
            "generation": "",
            "attempts":   0,
        })
        print(f"\nResposta: {result['generation']}\n")