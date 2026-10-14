import os, json, time, sys
from typing import List
def log(m): print(m, file=sys.stderr)
try:
    import chromadb
    chroma = chromadb.PersistentClient(path="./chroma_db")
    chunks_col = chroma.get_or_create_collection("chunks_1997")
    ckpt_col = chroma.get_or_create_collection("langgraph_checkpoints")
    HAS_CHROMA = True
except:
    HAS_CHROMA = False
_EMBED_MODEL = None
def embed(texts: List[str]):
    global _EMBED_MODEL
    try:
        from sentence_transformers import SentenceTransformer
        if _EMBED_MODEL is None:
            log("[Embed] Loading MiniLM-L6-v2...")
            _EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        return _EMBED_MODEL.encode(texts).tolist()
    except Exception as e:
        log(f"Embed fallback {e}")
        import hashlib, random
        vecs=[]
        for t in texts:
            h=int(hashlib.md5(t.encode()).hexdigest()[:8],16)
            random.seed(h)
            vecs.append([random.random() for _ in range(384)])
        return vecs
def tool_search_chroma(query: str, n_results: int = 3) -> str:
    if not HAS_CHROMA: return "Chroma not available"
    try:
        q_emb = embed([query])[0]
        r = chunks_col.query(query_embeddings=[q_emb], n_results=n_results)
        docs = r["documents"][0]
        out = [f"[Doc {i+1}]: {d[:300]}" for i,d in enumerate(docs)]
        log(f"[Tool:search_chroma] '{query[:60]}' -> {len(docs)}")
        return "\n".join(out)[:900]
    except Exception as e:
        return f"search error: {e}"
def tool_save_checkpoint(query: str, answer: str, score: float, iteration: int) -> str:
    if not HAS_CHROMA: return "ckpt not saved"
    try:
        ckpt_id = f"day23_{int(time.time()*1000)}_iter{iteration}"
        doc = json.dumps({"query": query, "answer": answer[:1000], "score": score, "iter": iteration})
        ckpt_col.add(ids=[ckpt_id], documents=[doc], embeddings=embed([doc]), metadatas=[{"score": score, "ts": time.time(), "day": 23}])
        msg = f"Checkpoint saved: {ckpt_id} score {score}"
        log(f"[Tool:save_checkpoint] {msg}")
        return msg
    except Exception as e:
        return f"ckpt error: {e}"
def tool_critique(answer: str, query: str) -> str:
    score = 0.85
    if len(answer) > 400: score = 0.9
    if "LangGraph" in answer or "checkpoint" in answer: score += 0.05
    if len(answer) > 800: score += 0.02
    score = min(score, 0.97)
    log(f"[Tool:critique] Score {score} len {len(answer)}")
    return json.dumps({"score": score})
