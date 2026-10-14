"""
Day25: MCP Server - 1997 chunks as MCP tools - FINAL MINIMAL (stdio-safe)
No prints to stdout!
"""
import os, sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "day23"))

from mcp.server.fastmcp import FastMCP
from tools import tool_search_chroma, tool_save_checkpoint, tool_critique
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "openai/gpt-oss-20b"
mcp = FastMCP("day25-rag-1997")

SYSTEM_PROMPT = "You are stateful agentic RAG. Use context from 1997 chunks to answer. Explain stateful TypedDict State, agentic should_continue edge, checkpoints, tool calling, 1997 chunks."

@mcp.tool()
def search_chroma(query: str, n_results: int = 3) -> str:
    """Search 1997 chunks ChromaDB. Args: query, n_results 1-5"""
    n_results = max(1, min(n_results, 5))
    try:
        result = tool_search_chroma(query, n_results=n_results)
        return f"[1997 chunks] query='{query}' n={n_results}\n{result}"
    except Exception as e:
        return f"search error: {e}"

@mcp.tool()
def critique_answer(answer: str, query: str) -> str:
    """Score answer 0-1 threshold 0.95"""
    try:
        return tool_critique(answer, query)
    except Exception as e:
        return json.dumps({"score": 0.0, "error": str(e)})

@mcp.tool()
def save_checkpoint(query: str, answer: str, score: float) -> str:
    """Save checkpoint to ChromaDB"""
    try:
        return tool_save_checkpoint(query, answer, score, 1)
    except Exception as e:
        return f"checkpoint error: {e}"

@mcp.tool()
def agentic_rag(query: str, n_results: int = 3) -> str:
    """Full pipeline: search 1997 -> Groq 700 tokens -> critique -> checkpoint. TPM-safe 3 calls."""
    start = time.time()
    context_str = tool_search_chroma(query, n_results=n_results)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Query: {query}\n\nContext:\n{context_str}\n\nExplain."}
    ]
    try:
        resp = groq_client.chat.completions.create(model=MODEL, messages=messages, temperature=0.2, max_tokens=700)
        final_answer = resp.choices[0].message.content or ""
    except Exception as e:
        final_answer = f"Groq error {e} Context: {context_str[:600]}"
    crit_res = tool_critique(final_answer, query)
    try:
        score = float(json.loads(crit_res).get("score", 0.9))
    except:
        score = 0.9
    ckpt_id = tool_save_checkpoint(query, final_answer, score, 1)
    result = {"query": query, "answer": final_answer, "score": score, "tool_calls": 3, "checkpoint_id": ckpt_id, "model": MODEL, "chunks": 1997, "latency_ms": int((time.time()-start)*1000)}
    return json.dumps(result, indent=2)

@mcp.tool()
def health() -> str:
    """Health check"""
    try:
        import chromadb
        client = chromadb.PersistentClient(path="./chroma_db")
        chunks_col = client.get_collection("chunks_1997")
        count = chunks_col.count()
        ckpt_col = client.get_or_create_collection("langgraph_checkpoints")
        ckpt_count = ckpt_col.count()
        return json.dumps({"status": "ok", "model": MODEL, "chunks_1997_count": count, "checkpoints_count": ckpt_count, "tools": ["search_chroma","agentic_rag","critique_answer","save_checkpoint","health","list_checkpoints"], "day": 25}, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})

@mcp.tool()
def list_checkpoints(n: int = 5) -> str:
    """List recent checkpoints"""
    try:
        import chromadb
        client = chromadb.PersistentClient(path="./chroma_db")
        col = client.get_or_create_collection("langgraph_checkpoints")
        res = col.get(limit=n)
        out = [f"{i+1}. {cid} score={meta.get('score')} doc={doc[:150]}" for i,(cid,doc,meta) in enumerate(zip(res.get("ids",[]), res.get("documents",[]), res.get("metadatas",[])))]
        return "\n".join(out) if out else "No checkpoints"
    except Exception as e:
        return f"list error: {e}"

if __name__ == "__main__":
    mcp.run()
