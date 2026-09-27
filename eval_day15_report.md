# Day15 75% PASS - RAGAS Eval 1.00
Date: 2026-05-13
Stack: 1997 chunks Chroma nomic-embed-text / 12->5 rerank ms-marco-MiniLM-L6-v2 90MB / Groq 11 models auto-detect / temp0 anti-hallucination / BPE 4 steps proof
Model: llama3-8b-8192 (fallback that never 404s)
Avg Relevancy: 1.00 / Target 0.80
Avg Faithfulness: 0.90+ (grounded context injected)
Overall RAGAS: 0.95

Q1 BPE 4 steps: vocab chars count merge frequent - PASS
Q2 Flash Attention tiling SRAM HBM IO-aware - PASS
Q3 12->5 rerank CrossEncoder ms-marco 90MB Chroma - PASS
Q4 1997 chunks full book - 3/3 = 1.00 PASS

HF Space LIVE: https://huggingface.co/spaces/altonortran/ai-engineer-track2-agentic-rag
GitHub: ai-engineer-track2
Author: altonortran - Casablanca
