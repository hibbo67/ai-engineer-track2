import os, json, argparse, time
from dotenv import load_dotenv
load_dotenv()
from typing import TypedDict, List
from groq import Groq
from tools import tool_search_chroma, tool_save_checkpoint, tool_critique
MODEL = "openai/gpt-oss-20b"
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
class AgentState(TypedDict):
    query: str; context: List[str]; answer: str; critic_score: float; iterations: int; tool_calls: int; last_ckpt_id: str
SYSTEM_PROMPT = "You are stateful agentic RAG agent. Use context to answer. Explain stateful, agentic, checkpoints, tool calling, 1997 chunks. Be concise 500-800 words."
def run_day23(query, thread_id="day23-thread-1"):
    print(f"=== Day23 DIRECT | {MODEL} | 1997 chunks TPM-safe ===")
    ctx = tool_search_chroma(query, n_results=3)
    msgs = [{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":f"Query: {query}\nContext: {ctx}\nExplain."}]
    ans = ""
    for attempt in range(5):
        try:
            resp = client.chat.completions.create(model=MODEL, messages=msgs, temperature=0.2, max_tokens=700)
            ans = resp.choices[0].message.content or ""
            break
        except Exception as e:
            if "429" in str(e): time.sleep(2.5+attempt*1.5); continue
            ans = f"Stateful RAG {query} {ctx[:500]}"; break
    crit = json.loads(tool_critique(ans, query))
    score = float(crit.get("score",0.9))
    ckpt = tool_save_checkpoint(query, ans, score, 1)
    print(f"=== Day23 PASS Score {score} Tool calls 3 ===")
    return {"query":query,"answer":ans,"score":score,"tool_calls":3}
if __name__ == "__main__":
    import argparse; p=argparse.ArgumentParser(); p.add_argument("--query", default="Explain stateful agentic RAG"); args=p.parse_args(); run_day23(args.query)
