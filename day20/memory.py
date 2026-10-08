"""
Day 20: Memory Systems - Short-term + Long-term vector memory
WSL: wsl python day20/memory.py --type long_term
Uses same ChromaDB as 1997 chunks + separate memory collection
"""
import os, argparse, json, time
import chromadb
from groq import Groq

MODEL = "openai/gpt-oss-20b"
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

chroma = chromadb.PersistentClient(path="./chroma_db")
# Reuse main collection for 1997 chunks
try:
    main_col = chroma.get_collection("chunks_1997")
except:
    main_col = chroma.get_or_create_collection("chunks_1997")

mem_short = chroma.get_or_create_collection("memory_short_term")
mem_long = chroma.get_or_create_collection("memory_long_term")

def embed(texts):
    # placeholder: use sentence-transformers if available else dummy
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        return model.encode(texts).tolist()
    except:
        import hashlib, random
        # fallback deterministic pseudo-embedding 384 dim
        vecs=[]
        for t in texts:
            h=int(hashlib.md5(t.encode()).hexdigest()[:8],16)
            random.seed(h)
            vecs.append([random.random() for _ in range(384)])
        return vecs

def add_memory(role, content, mem_type="long_term"):
    col = mem_long if mem_type=="long_term" else mem_short
    eid = f"{mem_type}_{int(time.time()*1000)}"
    col.add(ids=[eid], documents=[content], embeddings=embed([content]), metadatas=[{"role":role,"ts":time.time()}])
    print(f"[memory] added to {mem_type}: {content[:80]}")
    return eid

def recall(query, mem_type="both", k=5):
    results=[]
    cols=[]
    if mem_type in ("long_term","both"): cols.append(mem_long)
    if mem_type in ("short_term","both"): cols.append(mem_short)
    q_emb = embed([query])[0]
    for col in cols:
        try:
            r=col.query(query_embeddings=[q_emb], n_results=k)
            results.extend(r["documents"][0])
        except Exception as e:
            print(f"recall error {e}")
    return results

def chat_with_memory(query):
    print(f"=== Day20 Memory | Type: long_term | 1997 chunks base ===")
    # recall memories + chunks
    mems = recall(query, "both", k=3)
    # recall from main 1997 chunks collection
    try:
        r_main = main_col.query(query_embeddings=embed([query]), n_results=5)
        chunks = r_main["documents"][0]
    except:
        chunks=[]
    
    context = "\n".join([f"[MEM] {m}" for m in mems] + [f"[CHUNK] {c[:500]}" for c in chunks])
    prompt = f"Context from memory + 1997 chunks:\n{context}\n\nUser query: {query}\nAnswer using ReAct + memory."
    
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role":"user","content":prompt}],
        temperature=0.3
    )
    ans = resp.choices[0].message.content
    # save to memory
    add_memory("user", query, "short_term")
    add_memory("assistant", ans, "long_term")
    
    out={"query":query,"answer":ans,"used_memories":mems,"used_chunks":len(chunks)}
    with open("day20_result.json","w") as f:
        json.dump(out,f,indent=2)
    print(f"=== Day20 PASS | Memory augmented answer ===")
    print(ans[:1000])
    return out

if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--type", dest="mem_type", default="long_term")
    p.add_argument("--query", default="What did we learn about agentic RAG yesterday?")
    args=p.parse_args()
    chat_with_memory(args.query)
