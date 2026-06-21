# diagnostico.py
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


def get_vectorstore():
    return Chroma(
        persist_directory="vectorstore/",
        embedding_function=OllamaEmbeddings(model="nomic-embed-text")
    )


def chunk_score(term: str):
    vs = get_vectorstore()
    print(f"\n--- Busca semântica: '{term}' ---")
    results = vs.similarity_search_with_score(term, k=6)
    if not results:
        print("Nenhum resultado encontrado.")
        return
    for doc, score in results:
        source = doc.metadata.get("source", "desconhecido")
        print(f"score: {score:.4f} | fonte: {source}")
        print(f"  {doc.page_content[:200]}")
        print()


def chunk_contains_term(term: str):
    vs = get_vectorstore()
    all_docs = vs.get()
    total = len(all_docs["documents"])
    encontrados = [
        (pc, meta)
        for pc, meta in zip(all_docs["documents"], all_docs["metadatas"])
        if term.lower() in pc.lower()
    ]
    print(f"\n--- Busca léxica: '{term}' ---")
    print(f"{len(encontrados)} de {total} chunks contêm o termo\n")
    for pc, meta in encontrados:
        source = meta.get("source", "desconhecido")
        print(f"fonte: {source}")
        print(f"  {pc[:300]}")
        print()


def list_sources():
    vs = get_vectorstore()
    all_docs = vs.get()
    total = len(all_docs["documents"])
    sources = {}
    for meta in all_docs["metadatas"]:
        src = meta.get("source", "desconhecido")
        sources[src] = sources.get(src, 0) + 1
    print(f"\n--- Arquivos indexados ({total} chunks no total) ---")
    for src, count in sorted(sources.items()):
        print(f"  {count:>4} chunks | {src}")


MENU = {
    "1": ("Busca semântica (score)",        chunk_score),
    "2": ("Busca léxica (termo exato)",     chunk_contains_term),
    "3": ("Listar arquivos indexados",      list_sources),
}

if __name__ == "__main__":
    print("\nDiagnóstico do vectorstore")
    for key, (label, _) in MENU.items():
        print(f"  {key} — {label}")

    escolha = input("\nEscolha: ").strip()
    func_entry = MENU.get(escolha)

    if not func_entry:
        print("Opção inválida.")
    else:
        label, func = func_entry
        if escolha in ("1", "2"):
            term = input("Termo: ").strip()
            func(term)
        else:
            func()