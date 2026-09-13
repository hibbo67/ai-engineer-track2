from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load small fast model
print("Loading model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Your AI Engineer knowledge base
docs = [
    "AI Engineer builds LLM applications",
    "Machine learning model training and evaluation",
    "Vector database for semantic search",
    "RAG pipeline with embeddings",
    "I love pizza and football"
]

print("\nEncoding docs...")
embeddings = model.encode(docs)

query = "How to build LLM apps?"
query_emb = model.encode([query])

# Compute similarity
sims = cosine_similarity(query_emb, embeddings)[0]

print(f"\nQuery: {query}\n")
for i, (doc, score) in enumerate(zip(docs, sims)):
    print(f"{score:.3f} -> {doc}")

print(f"\nMost relevant: '{docs[np.argmax(sims)]}' with score {np.max(sims):.3f}")
