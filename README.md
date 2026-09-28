# Agentic RAG v3.1 - Day 15 75% PASS | RAGAS 1.00

Track 2 AI Engineer | Casablanca | by hibbo67

LIVE HF Space: https://huggingface.co/spaces/altonortran/ai-engineer-track2-agentic-rag

## Stack
- 1997 chunks - Full LLM book, 512 tokens overlap 50, ChromaDB + nomic-embed-text
- 12->5 rerank - CrossEncoder ms-marco-MiniLM-L6-v2 90MB
- Groq streaming - auto-detect live models (now gpt-oss-20b, fallback gpt-oss-120b) / temp 0.1
- BPE 4 steps - vocab chars -> count pairs -> merge frequent -> repeat

## Day 15 Eval - RAGAS 1.00 / 1.00 PASS
Q: Why 1997 chunks? -> 1.00 (3/3)
Avg Relevancy: 1.00 | Target 0.80 | PASS | openai/gpt-oss-20b

## Roadmap
- Day 8 30% - 1997 chunks ingestion
- Day 12 50% - Flash Attention tiling
- Day 14 60% - Groq streaming LIVE
- Day 15 75% - RAGAS eval 1.00 PASS
- Day 16 85% - Portfolio README + HF Space

## Security
- No hardcoded API keys - uses GROQ_API_KEY env only

Run:
export GROQ_API_KEY=gsk_...
python3 eval_day15.py

