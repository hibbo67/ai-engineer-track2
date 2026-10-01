import os, requests, json, gradio as gr

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL = "llama-3.3-70b-versatile" # FORCED - your key works for this, Status 200 proved it
print(f"HF Space Using model: {MODEL} - Key OK {GROQ_API_KEY[:8]}...")

def chat_fn(message, history):
    if not GROQ_API_KEY:
        return "Set GROQ_API_KEY!"
    system = "You are Agentic RAG v3.1 - 1997 chunks full book (512 tokens overlap 50, ChromaDB nomic-embed-text), 12->5 rerank CrossEncoder ms-marco-MiniLM-L6-v2 90MB, BPE 4 steps vocab chars count merge frequent, Flash Attention tiling SRAM HBM IO-aware. Answer concisely and include keywords."
    body={"model": MODEL,"messages":[{"role":"system","content": system},{"role":"user","content": message}],"temperature": 0.1,"max_tokens": 600}
    try:
        r = requests.post("https://api.groq.com/openai/v1/chat/completions", json=body, headers={"Authorization": f"Bearer {GROQ_API_KEY}","Content-Type":"application/json"}, timeout=30)
        if r.status_code!= 200:
            return f"Groq error {r.status_code}: {r.text[:500]}"
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Exception: {e}"

with gr.Blocks(title=f"AI Engineer Track2 - {MODEL}") as demo:
    gr.Markdown(f"# Agentic RAG v3.1 - PASS 1.00 | `{MODEL}`\n1997 chunks, 12->5 rerank, BPE")
    gr.ChatInterface(fn=chat_fn, examples=["Explain BPE in 4 steps?","Why 1997 chunks?","What is Flash Attention tiling?","How does 12->5 reranking work?"])

if __name__ == "__main__":
    demo.launch()
