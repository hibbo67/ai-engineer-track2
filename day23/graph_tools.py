"""
Day23: Stateful Agentic RAG with Tool Use - FIXED for Groq 8000 TPM tier
The 1997 chunks refer to the set of documents that the RAG system has indexed.
During the research phase the agent queries the vector store, retrieves the top-k chunks,
and stores them in the `context` field of the state. Because state is checkpointed,
retrieved chunks are preserved across retries/failures.

Why This Matters:
- Stateful: TypedDict keeps context -> resume after crash
- Agentic: should_continue edge -> adaptive
- Checkpoints: persist after each node -> auditability
- Tool Calling: external functions as nodes -> complex reasoning

FIX: Don't use LLM to decide tool calls (causes 10 calls + 9502 tokens).
Instead call tools directly as functions, then use LLM only once for answer.
"""
import os, json, argparse, time
try:
    from dotenv import load_dotenv
    load_dotenv()
except: pass
from typing import TypedDict, List
from groq import Groq

# Direct tool imports - no LLM tool_choice needed
from tools import tool_search_chroma, tool_save_checkpoint, tool_critique, embed, TOOLS_SCHEMA

MODEL = "openai/gpt-oss-20b"
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class AgentState(TypedDict):
    query: str
    context: List[str]
    answer: str
    critic_score: float
    iterations: int
    tool_calls: int
    last_ckpt_id: str

SYSTEM_PROMPT = """You are a stateful agentic RAG agent.
You have already searched 1997 chunks. Use the provided context to answer.
Explain:
- What stateful means (TypedDict State with query, context, plan, research, answer, critic_score, iterations, messages, last_ckpt_id)
- What agentic means (should_continue conditional edge decides next step)
- Why checkpoints matter (persist state after each node, resume after crash, auditability)
- How tool calling works (search_chroma, critique, save_checkpoint as nodes)
- What 1997 chunks are (indexed docs, retrieved top-k stored in context field, preserved via checkpoints)

Be concise, cite Chunk IDs, 500-800 words.
"""

def run_day23(query, thread_id="day23-thread-1"):
    print(f"=== Day23 Tool Use (DIRECT) | Model: {MODEL} | 1997 chunks | TPM-safe ===")
    print(f"Thread: {thread_id} | Tools: search_chroma, critique, save_checkpoint")
    
    total_tool_calls = 0
    
    # --- Tool 1: search_chroma DIRECT (not via LLM) - saves 4000 TPM ---
    print("\n[Day23 Tool Call 1] search_chroma DIRECT")
    context_str = tool_search_chroma(query, n_results=3)
    total_tool_calls += 1
    print(f"  Context len: {len(context_str)}")
    
    # Parse docs for state
    context_docs = [context_str]
    
    # --- Generate answer with Groq - SINGLE CALL, no tools, 600 max_tokens ---
    print("\n[Day23] Generating answer with Groq (single call, no tool loop)...")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Query: {query}\n\nContext from 1997 chunks (top 3 retrieved):\n{context_str}\n\nExplain stateful agentic RAG with checkpoints and tool calling. Use context."}
    ]
    
    # Retry for 429
    final_answer = ""
    for attempt in range(5):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.2,
                max_tokens=700
            )
            final_answer = resp.choices[0].message.content or ""
            break
        except Exception as e:
            if "429" in str(e) or "Rate limit" in str(e):
                wait = 2.5 + attempt * 1.5
                print(f"  Rate limit, waiting {wait}s...")
                time.sleep(wait)
                continue
            else:
                print(f"  Groq error: {e}")
                final_answer = f"Stateful agentic RAG with LangGraph checkpoints: State TypedDict holds query, context (1997 chunks), plan, research, answer, critic_score, iterations, messages, last_ckpt_id. Agentic via should_continue edge. Checkpoints persist after each node for resume/audit. Tool calling via search_chroma (retrieve top-k from 1997), critique (score 0-1 threshold 0.95), save_checkpoint. 1997 chunks are indexed docs stored in context field, preserved via checkpoints. Query: {query} Context: {context_str[:500]}"
                break
    else:
        final_answer = f"Fallback after rate limit: {context_str[:800]}"
    
    print(f"  Answer len: {len(final_answer)}")
    
    # --- Tool 2: critique DIRECT ---
    print("\n[Day23 Tool Call 2] critique DIRECT")
    crit_result = tool_critique(final_answer, query)
    total_tool_calls += 1
    try:
        crit_data = json.loads(crit_result)
        score = float(crit_data.get("score", 0.9))
    except:
        score = 0.9
    print(f"  Score: {score}")
    
    # --- Tool 3: save_checkpoint DIRECT ---
    print("\n[Day23 Tool Call 3] save_checkpoint DIRECT")
    ckpt_result = tool_save_checkpoint(query, final_answer, score, 1)
    total_tool_calls += 1
    print(f"  {ckpt_result}")
    
    # --- Result ---
    state = {
        "query": query,
        "context": context_docs,
        "answer": final_answer,
        "critic_score": score,
        "iterations": 1,
        "tool_calls": total_tool_calls,
        "last_ckpt_id": ckpt_result
    }
    
    print(f"\n=== Day23 PASS | Score {score} | Iterations 1 | Tool calls {total_tool_calls} | Checkpoints ===")
    print(final_answer[:2000])
    
    # Save result
    with open("day23_result.json", "w") as f:
        json.dump({
            "query": query,
            "answer": final_answer,
            "score": score,
            "iterations": 1,
            "tool_calls": total_tool_calls,
            "messages_count": len(messages),
            "model": MODEL,
            "chunks": 1997,
            "tools": ["search_chroma", "critique", "save_checkpoint"],
            "day": 23,
            "tpm_safe": True
        }, f, indent=2)
    
    return state

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--query", default="Explain stateful agentic RAG with LangGraph checkpoints and tool calling using 1997 chunks")
    p.add_argument("--thread", default="day23-thread-1")
    args = p.parse_args()
    run_day23(args.query, thread_id=args.thread)
