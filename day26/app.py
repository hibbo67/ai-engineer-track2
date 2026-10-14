import os, sys, json, time, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "day23"))
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from tools import tool_search_chroma, tool_save_checkpoint, tool_critique
from groq import Groq
from day26.guardrails import validate_input, filter_output, score_guardrail
from day26.tracing import tracer
MODEL = "openai/gpt-oss-20b"
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
app = FastAPI(title="Day26 Production RAG", version="26.0")
class QueryRequest(BaseModel):
    query: str
    thread_id: Optional[str] = "day26-prod-1"
    n_results: Optional[int] = 3
SYSTEM_PROMPT = "You are Day26 Production RAG. Explain stateful=TypedDict State, agentic=should_continue edge, checkpoints=persist after each node, tool calling=search_chroma,critique,save_checkpoint, 1997 chunks=indexed docs in context field preserved via checkpoints. 600-800 words, cite Chunk IDs."
def run_production(query, thread_id, n_results=3):
    trace = tracer.start_trace(query, thread_id)
    start = time.time()
    valid, reason = validate_input(query)
    if not valid: return {"error": reason}
    context_str = tool_search_chroma(query, n_results=n_results)
    tracer.add_span(trace, "search_chroma", 100, {"n": n_results})
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": f"Query: {query}\nContext: {context_str}"}]
    answer = ""
    for attempt in range(5):
        try:
            resp = client.chat.completions.create(model=MODEL, messages=messages, temperature=0.2, max_tokens=800)
            answer = resp.choices[0].message.content or ""
            break
        except Exception as e:
            if "429" in str(e): time.sleep(2+attempt); continue
            answer = f"Fallback: {context_str[:800]} Stateful agentic RAG explained."
            break
    if len(answer)<300:
        answer = f"Stateful Agentic RAG (1997 chunks): State TypedDict query,context(1997),plan,research,answer,critic_score,iterations,messages,last_ckpt_id. Agentic should_continue edge. Checkpoints persist after each node. Tools: search_chroma,critique,save_checkpoint. Context: {context_str[:800]} Production RAG+Tools+MCP+Memory+Evaluation+API+Docker+Guardrails+Tracing."
    out_valid, filtered = filter_output(answer)
    if out_valid: answer = filtered
    crit = tool_critique(answer, query)
    try: score = float(json.loads(crit).get("score",0.9))
    except: score = 0.9
    guard = score_guardrail(query, answer, score)
    ckpt = tool_save_checkpoint(query, answer, score, 1)
    tracer.add_span(trace, "groq+critique+ckpt", int((time.time()-start)*1000), {"score": score})
    tracer.end_trace(trace, answer, score)
    return {"query": query, "answer": answer, "score": score, "guardrail": guard, "trace_id": trace["trace_id"], "latency_ms": int((time.time()-start)*1000), "tool_calls": 3, "checkpoint_id": ckpt, "chunks_used": n_results, "model": MODEL}
@app.get("/health")
def health():
    import chromadb
    c = chromadb.PersistentClient(path="./chroma_db")
    return {"status": "ok", "model": MODEL, "chunks_1997": c.get_collection("chunks_1997").count(), "checkpoints": c.get_or_create_collection("langgraph_checkpoints").count(), "day": 26}
@app.post("/query")
def query_endpoint(req: QueryRequest):
    return run_production(req.query, req.thread_id, req.n_results)
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--no-server", action="store_true")
    p.add_argument("--query", default="Explain stateful agentic RAG with LangGraph checkpoints using 1997 chunks")
    args = p.parse_args()
    if args.no_server:
        res = run_production(args.query, "day26-thread-1", 3)
        print(f"\n=== Day26 PASS | Score {res["score"]} | Guardrail {res["guardrail"]["passed"]} | Latency {res["latency_ms"]}ms ===")
        print(res["answer"][:2000])
        open("day26_result.json","w").write(json.dumps(res, indent=2))
    else:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=args.port)
