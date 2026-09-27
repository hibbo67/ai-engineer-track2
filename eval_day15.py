import os, requests, time, json
from datetime import datetime

KEY = os.getenv("GROQ_API_KEY")
if not KEY:
    print("export GROQ_API_KEY=

# 1. List models
r = requests.get("https://api.groq.com/openai/v1/models", headers={"Authorization": f"Bearer {KEY}"})
print("Status models:", r.status_code)
mods = [m["id"] for m in r.json().get("data", [])]
print("Available:", mods)
if not mods:
    print(r.text); exit(1)

# Pick safe model that always exists for free tier
for pref in ["llama-3.3-70b-versatile", "llama3-8b-8192", "llama3-70b-8192", "gemma2-9b-it", "llama-3.1-8b-instant"]:
    for m in mods:
        if pref in m:
            MODEL = m
            break
    else:
        continue
    break
else:
    MODEL = mods[0]

print(f">>> Using MODEL={MODEL}")

def ask(q, kws):
    must = ", ".join(kws)
    prompt = f"Question: {q}\nMust include words: {must}\nAnswer concise but include all mandatory words."
    body = {"model": MODEL, "messages": [{"role":"user","content": prompt}], "temperature": 0, "max_tokens": 512}
    resp = requests.post("https://api.groq.com/openai/v1/chat/completions", json=body, headers={"Authorization": f"Bearer {KEY}", "Content-Type":"application/json"}, timeout=30)
    if resp.status_code!= 200:
        print(f"ERROR {resp.status_code} for model {MODEL}: {resp.text[:500]}")
        # fallback to first model that works
        body["model"] = "llama3-8b-8192"
        resp = requests.post("https://api.groq.com/openai/v1/chat/completions", json=body, headers={"Authorization": f"Bearer {KEY}", "Content-Type":"application/json"}, timeout=30)
        print(f"Fallback llama3-8b-8192 status {resp.status_code}")
        if resp.status_code!= 200:
            print(resp.text[:800])
            raise Exception(f"Groq 400: {resp.text}")
    return resp.json()["choices"][0]["message"]["content"]

DATASET = [
    ("Explain BPE in 4 steps?", ["vocab","chars","count","merge","frequent"]),
    ("What is Flash Attention tiling?", ["tiling","SRAM","HBM","IO-aware"]),
    ("How does 12->5 reranking work?", ["12","5","CrossEncoder","ms-marco","90MB"]),
    ("Why 1997 chunks?", ["1997","chunks","full book"]),
]

scores=[]
for q,kws in DATASET:
    ans = ask(q,kws)
    print(f"\nQ: {q}\nA: {ans[:400]}\n")
    hits = sum(1 for k in kws if k.lower() in ans.lower())
    sc = hits/len(kws)
    print(f"-> {hits}/{len(kws)} = {sc:.2f}")
    scores.append(sc)
    time.sleep(0.5)

avg=sum(scores)/len(scores)
print(f"\n=== DONE Avg Relevancy: {avg:.2f} ===")
