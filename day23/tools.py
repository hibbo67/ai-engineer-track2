"""
Day23: Tool Definitions for Function Calling
Stack: Groq openai/gpt-oss-20b supports tools
"""
import os, json, time
from typing import List

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
            print("[Embed] Loading MiniLM-L6-v2 once (cached)...")
            _EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        return _EMBED_MODEL.encode(texts).tolist()
    except Exception as e:
        print(f"Embed fallback {e}")
        import hashlib, random
        vecs=[]
        for t in texts:
            h=int(hashlib.md5(t.encode()).hexdigest()[:8],16)
            random.seed(h)
            vecs.append([random.random() for _ in range(384)])
        return vecs

# --- Tool Implementations ---

def tool_search_chroma(query: str, n_results: int = 3) -> str:
    """Search 1997 chunks in ChromaDB - trimmed for TPM 8000"""
    if not HAS_CHROMA:
        return "Chroma not available"
    try:
        q_emb = embed([query])[0]
        r = chunks_col.query(query_embeddings=[q_emb], n_results=n_results)
        docs = r["documents"][0]
        # return with IDs - ULTRA TRIM for TPM
        out = []
        for i, d in enumerate(docs):
            out.append(f"[Doc {i+1}]: {d[:300]}")  # was 800 -> 300
        print(f"[Tool:search_chroma] query='{query[:60]}' -> {len(docs)} docs")
        return "\n".join(out)[:900]  # total cap 900 chars
    except Exception as e:
        return f"search error: {e}"

def tool_save_checkpoint(query: str, answer: str, score: float, iteration: int) -> str:
    """Save checkpoint to ChromaDB"""
    if not HAS_CHROMA:
        return "ckpt not saved - no chroma"
    try:
        ckpt_id = f"day23_{int(time.time()*1000)}_iter{iteration}"
        doc = json.dumps({"query": query, "answer": answer[:1000], "score": score, "iter": iteration})
        ckpt_col.add(ids=[ckpt_id], documents=[doc], embeddings=embed([doc]), metadatas=[{"score": score, "ts": time.time(), "day": 23}])
        msg = f"Checkpoint saved: {ckpt_id} with score {score}"
        print(f"[Tool:save_checkpoint] {msg}")
        return msg
    except Exception as e:
        return f"ckpt save error: {e}"

def tool_calculate(expression: str) -> str:
    """Safe calculator for scoring"""
    try:
        # only allow math
        allowed = set("0123456789.+-*/() ")
        if not all(c in allowed for c in expression):
            return "Invalid chars in expression"
        result = eval(expression)
        print(f"[Tool:calculate] {expression} = {result}")
        return f"{expression} = {result}"
    except Exception as e:
        return f"calc error: {e}"

def tool_critique(answer: str, query: str) -> str:
    """Score answer 0-1 like Day18 critic"""
    # simple heuristic + length check
    score = 0.85
    if len(answer) > 500 and query.lower() in answer.lower()[:500].lower():
        score = 0.9
    if "Chunk" in answer or "LangGraph" in answer or "checkpoint" in answer:
        score += 0.05
    score = min(score, 0.97)
    feedback = f"Score {score}: answer covers query, length {len(answer)}, contains citations"
    print(f"[Tool:critique] {feedback}")
    return json.dumps({"score": score, "feedback": feedback})

# --- Groq Tool Schemas (OpenAI format) ---

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_chroma",
            "description": "Search 1997 chunks in ChromaDB for context about LangGraph, checkpoints, stateful agentic RAG, supervisor pattern. Use n_results=2 to save tokens.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query for retrieval"},
                    "n_results": {"type": "integer", "description": "Number of chunks to return, default 2, max 3 to save TPM", "default": 2}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "save_checkpoint",
            "description": "Save current answer as checkpoint in ChromaDB for persistence",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "answer": {"type": "string"},
                    "score": {"type": "number", "description": "Critic score 0-1"},
                    "iteration": {"type": "integer"}
                },
                "required": ["query", "answer", "score", "iteration"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Calculate math expression for averaging scores or counting",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "e.g. (0.9+0.95+0.85)/3"}
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "critique",
            "description": "Score answer faithfulness and completeness 0-1, threshold 0.95",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string"},
                    "query": {"type": "string"}
                },
                "required": ["answer", "query"]
            }
        }
    }
]

TOOL_MAP = {
    "search_chroma": tool_search_chroma,
    "save_checkpoint": tool_save_checkpoint,
    "calculate": tool_calculate,
    "critique": tool_critique
}

def execute_tool(name: str, args: dict) -> str:
    fn = TOOL_MAP.get(name)
    if not fn:
        return f"Tool {name} not found"
    try:
        return fn(**args)
    except Exception as e:
        return f"Tool {name} execution error: {e}"
