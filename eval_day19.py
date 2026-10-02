import os
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI
client = OpenAI(api_key=os.getenv("GROQ_API_KEY"), base_url="https://api.groq.com/openai/v1")
MODEL="llama-3.3-70b-versatile"

questions = {
 "Explain BPE in 4 steps?": ["vocab","chars","count","merge","frequent"],
 "What is Flash Attention tiling?": ["tiling","SRAM","HBM","IO-aware"],
 "How does 12->5 reranking work?": ["12","5","rerank","CrossEncoder","ms-marco","Chroma","90MB"],
 "Why 1997 chunks?": ["1997","chunks","full","book"]
}

def ask(q):
    sys="You are Agentic RAG v3.1 - 1997 chunks 512 overlap 50, 12->5 rerank CrossEncoder ms-marco-MiniLM-L6-v2 90MB Chroma, BPE 4 steps vocab chars count merge frequent, Flash Attention tiling SRAM HBM IO-aware."
    r=client.chat.completions.create(model=MODEL, messages=[{"role":"system","content":sys},{"role":"user","content":q}], temperature=0.1, max_tokens=400)
    return r.choices[0].message.content

total=0
for q,kws in questions.items():
    a=ask(q)
    hit=sum(1 for k in kws if k.lower() in a.lower())
    score=hit/len(kws)
    total+=score
    print(f"Q: {q}\nA: {a[:200]}...\nKeywords {hit}/{len(kws)} -> {score:.2f}\n")
print(f"=== FINAL Avg: {total/len(questions):.2f} | Target 0.80 | {'PASS' if total/len(questions)>=0.8 else 'FAIL'} | {MODEL} ===")
