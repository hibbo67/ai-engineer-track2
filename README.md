# AI Engineer Track 2 - 30 Day Challenge

Progress: Day 11 45% -> Day 12 50% HALFWAY

Stack: ChromaDB 1997 chunks | LangChain | Ollama llama3.2 + nomic-embed | CrossEncoder 90MB | Streamlit

Day 11 verified: BPE steps = vocab chars -> count pairs -> merge frequent -> repeat until vocab size (no Unigram mix)

Run:
pip install -r requirements.txt
chroma run --host localhost --port 8000
streamlit run app.py --server.fileWatcherType none

## Day 13 — 55% DEPLOYED LIVE ✅
Public Space: https://huggingface.co/spaces/altonortran/ai-engineer-track2-agentic-rag - Running
Proof: 1997 chunks indexed | 12→5 retrieve→rerank | 90MB ms-marco-MiniLM | temp 0
