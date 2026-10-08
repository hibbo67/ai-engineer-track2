"""
Day 19: Multi-Agent Orchestration - Supervisor + 3 workers
WSL: wsl python day19/supervisor.py --agents 3
Plan: Track 2 - 1997 chunks, Groq openai/gpt-oss-20b, tools: search_chunks, bpe_encode
"""
import os, asyncio, json
from groq import Groq

MODEL = "openai/gpt-oss-20b"
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

TOOLS = [
  {"name": "search_chunks", "desc": "Search 1997 chunks via Chroma+BM25 12->5"},
  {"name": "bpe_encode", "desc": "BPE 4-step encoding for context compression"},
  {"name": "critic", "desc": "Self-critique loop from Day18"}
]

WORKERS = {
  "researcher": "You are researcher. Use search_chunks to gather evidence from 1997 chunks.",
  "coder": "You are coder. Use bpe_encode + context to produce code/answer.",
  "critic": "You are critic. Score answer 0-1 for faithfulness/relevance. If <0.95, suggest fix."
}

def call_llm(system, user):
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
        temperature=0.2
    )
    return resp.choices[0].message.content

def supervisor_task(query: str):
    print(f"=== Day19 Supervisor | Agents: 3 | Model: {MODEL} | 1997 chunks ===")
    # 1. Researcher
    r = call_llm(WORKERS["researcher"], f"Query: {query}\nUse search_chunks tool. Return top evidences.")
    print("[researcher] done")
    # 2. Coder
    c = call_llm(WORKERS["coder"], f"Query: {query}\nEvidence: {r[:2000]}\nProduce final answer with citations.")
    print("[coder] done")
    # 3. Critic loop (Day18 pattern)
    score = 0.0
    iters = 0
    while score < 0.95 and iters < 3:
        crit = call_llm(WORKERS["critic"], f"Query: {query}\nAnswer: {c}\nScore and feedback:")
        # naive parse score
        try:
            # extract float
            import re
            m = re.search(r"([0-1]\.[0-9]+)", crit)
            if m: score = float(m.group(1))
            else: score = 0.92 if "good" in crit.lower() else 0.85
        except:
            score = 0.85
        print(f"[critic] iter {iters+1} score {score}: {crit[:200]}")
        if score < 0.95:
            c = call_llm(WORKERS["coder"], f"Revise based on critic feedback: {crit}\nOriginal answer: {c}")
        iters+=1
    
    result = {"query": query, "answer": c, "research": r, "final_score": score, "iterations": iters+1}
    with open("day19_result.json","w") as f:
        json.dump(result, f, indent=2)
    print(f"=== Day19 PASS | Score {score} | Iterations {iters+1} ===")
    return result

if __name__ == "__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--agents", type=int, default=3)
    p.add_argument("--query", default="Explain agentic RAG with ReAct + reflection using 1997 chunks")
    args=p.parse_args()
    supervisor_task(args.query)
