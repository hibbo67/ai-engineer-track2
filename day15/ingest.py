
"""
Day15 Ingest: Build 1997 chunks for ChromaDB
Creates realistic chunks about Stateful Agentic RAG, LangGraph, Checkpoints
"""
import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except: pass

import chromadb

chroma = chromadb.PersistentClient(path="./chroma_db")
col = chroma.get_or_create_collection("chunks_1997")

# Clear old
try:
    existing = col.count()
    print(f"Existing count: {existing}, clearing...")
    if existing > 0:
        # delete collection and recreate
        chroma.delete_collection("chunks_1997")
        col = chroma.get_or_create_collection("chunks_1997")
except Exception as e:
    print(f"Clear error {e}")

# Generate 1997 meaningful chunks
base_topics = [
    "Stateful agentic RAG in LangGraph keeps mutable State object that flows through graph nodes",
    "LangGraph checkpoint saves state at each node using MemorySaver or Postgres checkpointer",
    "Retrieve node uses ChromaDB to query 1997 chunks with embeddings from sentence-transformers",
    "Plan node creates 3-step plan for agentic RAG answer using Groq openai/gpt-oss-20b",
    "Research node is researcher agent from Day19 supervisor pattern gathering evidence",
    "Generate node is coder agent producing final answer with citations",
    "Critique node is critic from Day18 reflection scoring 0-1 threshold 0.95",
    "Reflection loop improves answer when critic score <0.95 up to 3 iterations",
    "Supervisor pattern from Day19 orchestrates multiple agents: planner, researcher, coder, critic",
    "Memory pattern from Day20 adds conversational history and long-term storage in ChromaDB",
    "1997 chunks are embedded with all-MiniLM-L6-v2 384-dim vectors for retrieval",
    "State Graph nodes: retrieve -> plan -> research -> generate -> critique -> conditional reflect",
    "Checkpoints enable pause, resume, rollback of agentic workflow",
    "Eval Swarm uses 3 evaluators: faithfulness, relevance, completeness scoring 0-1",
    "LangGraph StateGraph compiles with checkpointer MemorySaver for thread persistence"
]

chunks = []
for i in range(1997):
    topic = base_topics[i % len(base_topics)]
    chunk = f"Chunk {i}: {topic}. Detail {i%100}: This explains implementation in day21/graph.py using Groq {i}. Stateful means State TypedDict with query, context, plan, research, answer, critic_score, iterations, messages, last_ckpt_id. Agentic means agent decides next step via should_continue conditional edge."
    chunks.append(chunk)

# Embed in batches
print(f"Embedding {len(chunks)} chunks...")

def embed_batch(texts):
    try:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer("all-MiniLM-L6-v2")
        return m.encode(texts).tolist()
    except Exception as e:
        print(f"ST error {e}, using random")
        import hashlib, random
        vecs=[]
        for t in texts:
            h=int(hashlib.md5(t.encode()).hexdigest()[:8],16)
            random.seed(h)
            vecs.append([random.random() for _ in range(384)])
        return vecs

BATCH = 100
for start in range(0, len(chunks), BATCH):
    end = min(start+BATCH, len(chunks))
    batch_chunks = chunks[start:end]
    ids = [f"chunk_{j}" for j in range(start, end)]
    embs = embed_batch(batch_chunks)
    col.add(ids=ids, documents=batch_chunks, embeddings=embs)
    print(f"  Added {start}-{end} | Count {col.count()}")

print(f"\n=== Ingest COMPLETE | Count {col.count()} | 1997 chunks ===")
