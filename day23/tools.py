"""
Day23: Tool Definitions - MCP-SAFE + TPM-SAFE
All logs to stderr, no stdout prints
"""
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
        log(f"[Tool:search_chroma] '{query[:60]}' -> {len(docs)} docs")
        return "\n".join(out)[:900]
    except Exception as e:
        return f"search error: {e}"

def tool_save_checkpoint(query: str, answer: str, score: float, iteration: int) -> str:
    if not HAS_CHROMA: return "ckpt not saved"
    try:
        ckpt_id = f"day23_{int(time.time()*1000)}_iter{iteration}"
        doc = json.dumps({"query": query, "answer": answer[:1000], "score": score, "iter": iteration})
        ckpt_col.add(ids=[ckpt_id], documents=[doc], embeddings=embed([doc]), metadatas=[{"score": score, "ts": time.time(), "day": 23}])
        msg = f"Checkpoint saved: {ckpt_id} with score {score}"
        log(f"[Tool:save_checkpoint] {msg}")
        return msg
    except Exception as e:
        return f"ckpt error: {e}"

def tool_critique(answer: str, query: str) -> str:
    score = 0.85
    if len(answer) > 400: score = 0.9
    if "LangGraph" in answer or "checkpoint" in answer or "stateful" in answer.lower(): score += 0.05
    if len(answer) > 800: score += 0.02
    score = min(score, 0.97)
    feedback = f"Score {score}: len {len(answer)}"
    log(f"[Tool:critique] {feedback}")
    return json.dumps({"score": score, "feedback": feedback})

def tool_calculate(expression: str) -> str:
    try:
        allowed = set("0123456789.+-*/() ")
        if not all(c in allowed for c in expression): return "Invalid chars"
        result = eval(expression)
        log(f"[Tool:calculate] {expression} = {result}")
        return f"{expression} = {result}"
    except Exception as e:
        return f"calc error: {e}"

TOOLS_SCHEMA = [
    {"type": "function", "function": {"name": "search_chroma", "description": "Search 1997 chunks", "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "n_results": {"type": "integer", "default": 2}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "save_checkpoint", "description": "Save checkpoint", "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "answer": {"type": "string"}, "score": {"type": "number"}, "iteration": {"type": "integer"}}, "required": ["query", "answer", "score", "iteration"]}}},
    {"type": "function", "function": {"name": "calculate", "description": "Calculate", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
    {"type": "function", "function": {"name": "critique", "description": "Score answer", "parameters": {"type": "object", "properties": {"answer": {"type": "string"}, "query": {"type": "string"}}, "required": ["answer", "query"]}}}
]
TOOL_MAP = {"search_chroma": tool_search_chroma, "save_checkpoint": tool_save_checkpoint, "calculate": tool_calculate, "critique": tool_critique}
def execute_tool(name: str, args: dict) -> str:
    fn = TOOL_MAP.get(name)
    if not fn: return f"Tool {name} not found"
    try: return fn(**args)
    except Exception as e: return f"Tool {name} error: {e}"
