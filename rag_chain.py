import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import glob

# ── 1. Carregar documentos
def load_docs(folder="docs/"):
    docs = []
    for path in glob.glob(f"{folder}**/*", recursive=True):
        if path.endswith(".pdf"):
            docs.extend(PyPDFLoader(path).load())
        elif path.endswith(".txt"):
            docs.extend(TextLoader(path).load())
    print(f"[load] {len(docs)} páginas/documentos carregados")
    return docs

# ── 2. Chunking
def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = splitter.split_documents(docs)
    print(f"[split] {len(chunks)} chunks gerados")
    return chunks

# # ── 3. Vector store
# def build_vectorstore(chunks):
#     embeddings = OllamaEmbeddings(model="nomic-embed-text")
#     vectorstore = Chroma.from_documents(
#         documents=chunks,
#         embedding=embeddings,
#         persist_directory="vectorstore/",
#     )
#     print("[vectorstore] ChromaDB criado e persistido")
#     return vectorstore

# ── 3. Vector store
def build_vectorstore():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma(
        persist_directory="vectorstore/",
        embedding_function=embeddings,
    )
    if vectorstore._collection.count() == 0:
        print("[vectorstore] Vazio — indexando documentos...")
        docs   = load_docs()
        chunks = split_docs(docs)
        vectorstore.add_documents(chunks)
        print(f"[vectorstore] {len(chunks)} chunks indexados e persistidos")
    else:
        print(f"[vectorstore] Carregado do disco ({vectorstore._collection.count()} chunks)")
    return vectorstore

# ── 4. Chain RAG
def build_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    prompt = ChatPromptTemplate.from_template("""
Você é um assistente prestativo. Responda à pergunta com base APENAS no contexto abaixo.
Se a resposta não estiver no contexto, diga que não sabe.

Contexto:
{context}

Pergunta: {question}
""")

    llm = ChatOllama(model="llama3.2", temperature=0)

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# ── 5. Main
if __name__ == "__main__":
    vectorstore = build_vectorstore()
    chain = build_chain(vectorstore)

    print("\n[RAG pronto] Digite sua pergunta (ou /bye para sair)\n")
    while True:
        question = input("Pergunta: ")
        if question.strip().lower() == "/bye":
            print("Encerrando...")
            import subprocess
            subprocess.run(["ollama", "stop", "llama3.2"])
            subprocess.run(["ollama", "stop", "nomic-embed-text"])
            break
        answer = chain.invoke(question)
        print(f"\nResposta: {answer}\n")