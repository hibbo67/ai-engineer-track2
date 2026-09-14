# Day 5: First RAG System - Local, no API
from sentence_transformers import SentenceTransformer
import numpy as np

# 1. Your knowledge base (tomorrow we use real files)
docs = [
    "AI Engineer Track 2 focuses on building LLM applications.",
    "RAG means Retrieval Augmented Generation - search then generate.",
    "Embeddings convert text to vectors for semantic search.",
    "Vector databases store embeddings for fast similarity search.",
]

# 2. Embed docs (same as Day 4)
model = SentenceTransformer('all-MiniLM-L6-v2')
doc_embeddings = model.encode(docs)

def rag_query(question: str):
    q_emb = model.encode([question])
    scores = np.dot(doc_embeddings, q_emb.T).flatten()
    best_idx = np.argmax(scores)

    print(f"Q: {question}")
    print(f"Retrieved: {docs[best_idx]}")
    print(f"Score: {scores[best_idx]:.3f}")
    print("-" * 50)
    return docs[best_idx]

# Test
rag_query("What is RAG?")
rag_query("What do AI Engineers build?")
