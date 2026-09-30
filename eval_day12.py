from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

emb = OllamaEmbeddings(model="nomic-embed-text")
vs = Chroma(collection_name="day7_full", embedding_function=emb, host="localhost", port=8000)
print(f"Chunks: {vs._collection.count()}")

tests = ["Explain Byte-Pair Encoding steps", "Explain Flash Attention tiling", "Difference char vs BPE"]
for q in tests:
    docs = vs.similarity_search(q, k=5)
    print(f"\nQ: {q} -> {len(docs)} docs OK")
