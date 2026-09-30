<<<<<<< HEAD
---
title: Ai Engineer Track2 Agentic
emoji: 🌍
colorFrom: pink
colorTo: green
sdk: static
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
=======
# AI Engineer Track 2 - 30 Day Challenge

Progress: Day 11 45% -> Day 12 50% HALFWAY

Stack: ChromaDB 1997 chunks | LangChain | Ollama llama3.2 + nomic-embed | CrossEncoder 90MB | Streamlit

Day 11 verified: BPE steps = vocab chars -> count pairs -> merge frequent -> repeat until vocab size (no Unigram mix)

Run:
pip install -r requirements.txt
chroma run --host localhost --port 8000
streamlit run app.py --server.fileWatcherType none
>>>>>>> 0f21fb5 (Day 12: Flash Attention tiling clean + 1997 chunks Agentic RAG verified ✅ 50% HALFWAY)
