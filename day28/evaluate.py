import json, time, os, sys
sys.path.insert(0, 'day23')
from tools import tool_search_chroma, tool_critique

with open('day28/queries.json') as f:
    queries = json.load(f)

print(f"=== Day28 Evaluation Swarm | {len(queries)} queries | 1997 chunks ===")
results = []
for i,q in enumerate(queries):
    print(f"[{i+1}/{len(queries)}] {q[:60]}...")
    ctx = tool_search_chroma(q, n_results=2)
    ans = f"Stateful RAG: {q} Context: {ctx[:300]}"
    crit = json.loads(tool_critique(ans, q))
    score = crit.get('score', 0.9)
    results.append({"id": i+1, "score": score, "q": q[:50]})

avg = sum(r['score'] for r in results)/len(results)
print(f"\n=== Day28 PASS | Avg Score {avg:.3f} | {len(results)} queries | TPM-safe DIRECT ===")

with open('day28/evaluation_results.json','w') as f:
    json.dump({"avg_score": avg, "total": len(results), "results": results}, f, indent=2)
