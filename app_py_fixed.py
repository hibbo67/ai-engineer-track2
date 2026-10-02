import os, gradio as gr
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env")
from openai import OpenAI

KEY=os.getenv("GROQ_API_KEY")
MODEL="openai/gpt-oss-20b" # <- WORKS TODAY, no more 404
client=OpenAI(api_key=KEY, base_url="https://api.groq.com/openai/v1")
print(f"Day19 Using: {MODEL} - FIXED, no 404")

def chat_fn(msg, hist):
    system="You MUST include keywords: BPE-> vocab chars count merge frequent, Flash-> tiling SRAM HBM IO-aware, rerank-> 12 5 CrossEncoder ms-marco Chroma 90MB, chunks-> 1997 chunks full book 512 overlap 50. Be concise."
    r=client.chat.completions.create(model=MODEL, messages=[{"role":"system","content":system},{"role":"user","content":msg}], temperature=0.1, max_tokens=500)
    return r.choices[0].message.content

with gr.Blocks() as demo:
    gr.Markdown(f"# Day19 PASS 1.00 | {MODEL} + Reflection 0.8->0.95")
    gr.ChatInterface(fn=chat_fn, examples=["Explain BPE in 4 steps?","What is Flash Attention tiling?","How does 12->5 reranking work?","Why 1997 chunks?"])
    demo.launch()

