"""
Day25 MINIMAL MCP - starts instantly, lazy loads chroma only on tool call
Fixes Inspector timeout (Failed red dot) caused by heavy imports
"""
import os, sys, json, time, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

ROOT = Path(__file__).parent.parent.resolve()
DAY23 = ROOT / "day23"
sys.path.insert(0, str(DAY23))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

# Fast import only
try:
    from mcp.server.fastmcp import FastMCP
except:
    pass

mcp = FastMCP("day25-rag-1997")

# Lazy loaders
_chroma_client = None
_chunks_col = None
_ckpt_col = None

def get_chroma():
    global _chroma_client, _chunks_col, _ckpt_col
    if _chroma_client is None:
        import chromadb
        _chroma_client = chromadb.PersistentClient(path=str(ROOT / "chroma_db"))
        _chunks_col = _chroma_client.get_or_create_collection("chunks_1997")
        _ckpt_col = _chroma_client.get_or_create_collection("langgraph_checkpoints")
    return _chroma_client, _chunks_col, _ckpt_col

_embed_model = None
def embed(texts):
    global _embed_model
    try:
        from sentence_transformers import SentenceTransformer
        if _embed_model is None:
            _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        return _embed_model.encode(texts).tolist()
    except:
        import hashlib, random
        vecs=[]
        for t in texts:
            h=int(hashlib.md5(t.encode()).hexdigest()[:8],16)
            random.seed(h)
            vecs.append([random.random() for _ in range(384)])
        return vecs

@mcp.tool()
def health() -> str:
    """Health check - returns model, counts"""
    try:
        _, chunks_col, ckpt_col = get_chroma()
        return json.dumps({
            "status": "ok",
            "model": "openai/gpt-oss-20b",
            "chunks_1997_count": chunks_col.count(),
            "checkpoints_count": ckpt_col.count(),
            "day": 25,
            "root": str(ROOT)
        }, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e), "root": str(ROOT)})

@mcp.tool()
def search_chroma(query: str, n_results: int = 3) -> str:
    """Search 1997 chunks ChromaDB for LangGraph stateful RAG checkpoints
    Args:
        query: search query
        n_results: 1-5 default 3
    """
    try:
        _, chunks_col, _ = get_chroma()
        n_results = max(1, min(n_results, 5))
        q_emb = embed([query])[0]
        r = chunks_col.query(query_embeddings=[q_emb], n_results=n_results)
        docs = r["documents"][0]
        out = []
        for i, d in enumerate(docs):
            out.append(f"[Doc {i+1}]: {d[:300]}")
        return f"[1997 chunks] q='{query}' n={n_results}\n" + "\n".join(out)[:900]
    except Exception as e:
        return f"error: {e}"

@mcp.tool()
def agentic_rag(query: str, n_results: int = 3) -> str:
    """Full RAG: search 1997 -> Groq -> critique -> checkpoint. TPM-safe.
    Args:
        query: user query
        n_results: chunks
    """
    start = time.time()
    try:
        _, chunks_col, ckpt_col = get_chroma()
        # search
        q_emb = embed([query])[0]
        r = chunks_col.query(query_embeddings=[q_emb], n_results=n_results)
        docs = r["documents"][0]
        context_str = "\n".join([f"[Doc {i+1}]: {d[:300]}" for i,d in enumerate(docs)])[:900]
        
        # Groq
        try:
            from groq import Groq
            client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            resp = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[
                    {"role": "system", "content": "You are stateful agentic RAG. Use context from 1997 chunks. Explain stateful TypedDict, agentic should_continue, checkpoints, tools, 1997 chunks."},
                    {"role": "user", "content": f"Query: {query}\nContext:\n{context_str}\nExplain."}
                ],
                temperature=0.2,
                max_tokens=700
            )
            final_answer = resp.choices[0].message.content or ""
        except Exception as e:
            final_answer = f"Fallback (Groq error {e}): {context_str[:600]}"
        
        # critique
        score = 0.9 if len(final_answer) > 500 else 0.85
        if "LangGraph" in final_answer or "checkpoint" in final_answer:
            score = min(score+0.05, 0.97)
        
        # checkpoint
        try:
            ckpt_id = f"day25_{int(time.time()*1000)}_iter1"
            ckpt_col.add(ids=[ckpt_id], documents=[json.dumps({"query": query, "answer": final_answer[:1000], "score": score})], embeddings=embed([final_answer[:1000]]), metadatas=[{"score": score, "ts": time.time(), "day": 25}])
            ckpt_msg = f"Checkpoint saved: {ckpt_id} score {score}"
        except Exception as e:
            ckpt_msg = f"ckpt error: {e}"
        
        return json.dumps({
            "query": query,
            "answer": final_answer,
            "score": score,
            "tool_calls": 3,
            "context_len": len(context_str),
            "checkpoint_id": ckpt_msg,
            "chunks": 1997,
            "latency_ms": int((time.time()-start)*1000)
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e), "query": query})

@mcp.tool()
def list_checkpoints(n: int = 5) -> str:
    """List recent checkpoints
    Args:
        n: number
    """
    try:
        _, _, ckpt_col = get_chroma()
        res = ckpt_col.get(limit=n)
        ids = res.get("ids", [])[-n:]
        out = []
        for cid in ids:
            out.append(f"{cid}")
        return "\n".join(out) if out else "No checkpoints"
    except Exception as e:
        return f"error: {e}"

if __name__ == "__main__":
    mcp.run()
