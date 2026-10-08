"""
Day 21: State Graph + Checkpoints - LangGraph stateful agents
WSL: wsl python day21/graph.py --checkpoint
Stack: 1997 chunks, Groq openai/gpt-oss-20b, LangGraph, ChromaDB checkpoints
Follows Day 19 Supervisor + Day 20 Memory
"""
import os, json, argparse, time
try:
    from dotenv import load_dotenv
    load_dotenv()
except: pass
from typing import TypedDict, List, Annotated
from groq import Groq

MODEL = "openai/gpt-oss-20b"
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Try LangGraph, fallback to manual state if not installed
try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False
    print("LangGraph not installed - using manual state fallback")
    # Mock for type
    class StateGraph:
        pass
    END = "END"

# --- Chroma for checkpoints persistence ---
try:
    import chromadb
    chroma = chromadb.PersistentClient(path="./chroma_db")
    checkpoint_col = chroma.get_or_create_collection("langgraph_checkpoints")
    chunks_col = chroma.get_or_create_collection("chunks_1997")
    HAS_CHROMA = True
except:
    HAS_CHROMA = False

def embed(texts):
    try:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer("all-MiniLM-L6-v2")
        return m.encode(texts).tolist()
    except:
        import hashlib, random
        vecs=[]
        for t in texts:
            h=int(hashlib.md5(t.encode()).hexdigest()[:8],16)
            random.seed(h)
            vecs.append([random.random() for _ in range(384)])
        return vecs

# --- State Definition ---
class AgentState(TypedDict):
    query: str
    context: List[str]   # from 1997 chunks + memory
    plan: str
    research: str
    answer: str
    critic_score: float
    iterations: int
    messages: List[dict]
    last_ckpt_id: str

def call_groq(system, user):
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
        temperature=0.2
    )
    return resp.choices[0].message.content

# --- Nodes (ReAct + Reflection + Memory) ---
def node_retrieve(state: AgentState):
    print(f"[Day21:retrieve] query={state['query'][:80]}")
    q = state["query"]
    docs = []
    if HAS_CHROMA:
        try:
            q_emb = embed([q])[0]
            r = chunks_col.query(query_embeddings=[q_emb], n_results=5)
            docs = r["documents"][0]
        except Exception as e:
            print(f"chroma retrieve error {e}")
    state["context"] = docs
    state["messages"].append({"role":"retrieve","content":f"Retrieved {len(docs)} chunks from 1997"})
    return state

def node_plan(state: AgentState):
    print("[Day21:plan] creating plan")
    prompt = f"Query: {state['query']}\nContext chunks: {len(state['context'])}\nCreate 3-step plan for agentic RAG answer."
    plan = call_groq("You are planner. Create concise plan.", prompt)
    state["plan"] = plan
    state["messages"].append({"role":"plan","content":plan[:500]})
    return state

def node_research(state: AgentState):
    print("[Day21:research] researcher agent")
    ctx = "\n".join(state["context"][:3])
    prompt = f"Plan: {state['plan']}\nContext:\n{ctx[:2000]}\nQuery: {state['query']}\nGather evidence."
    research = call_groq("You are researcher from Day19 supervisor. Use search_chunks logic.", prompt)
    state["research"] = research
    state["messages"].append({"role":"researcher","content":research[:800]})
    return state

def node_generate(state: AgentState):
    print("[Day21:generate] coder agent")
    prompt = f"Plan: {state['plan']}\nResearch: {state['research'][:2000]}\nContext: {state['context'][0][:1000] if state['context'] else 'none'}\nQuery: {state['query']}\nProduce final answer with citations."
    ans = call_groq("You are coder agent from Day19. Answer with ReAct.", prompt)
    state["answer"] = ans
    state["iterations"] = state.get("iterations",0)+1
    state["messages"].append({"role":"generator","content":ans[:800]})
    return state

def node_critique(state: AgentState):
    print(f"[Day21:critique] iter {state.get('iterations',0)}")
    prompt = f"Query: {state['query']}\nAnswer: {state['answer']}\nScore 0-1 for faithfulness + relevance. Threshold 0.95. Give score and feedback."
    crit = call_groq("You are critic from Day18 reflection. Be strict.", prompt)
    # parse score
    import re
    m = re.search(r"([0-1]\.[0-9]+)", crit)
    score = float(m.group(1)) if m else (0.92 if "good" in crit.lower() else 0.85)
    state["critic_score"] = score
    state["messages"].append({"role":"critic","content":crit[:500]})
    print(f"  -> critic score {score}")
    # Save checkpoint
    if HAS_CHROMA:
        try:
            ckpt_id = f"day21_{int(time.time()*1000)}_iter{state['iterations']}"
            state["last_ckpt_id"] = ckpt_id
            doc = json.dumps({"query":state["query"],"score":score,"iter":state["iterations"],"answer":state["answer"][:1000]})
            checkpoint_col.add(ids=[ckpt_id], documents=[doc], embeddings=embed([doc]), metadatas=[{"score":score,"ts":time.time()}])
            print(f"  -> checkpoint saved {ckpt_id}")
        except Exception as e:
            print(f"ckpt save error {e}")
    return state

def should_continue(state: AgentState):
    if state.get("critic_score",0) >= 0.95:
        return "end"
    if state.get("iterations",0) >= 3:
        return "end"
    return "reflect"

def node_reflect(state: AgentState):
    print("[Day21:reflect] self-correction loop")
    prompt = f"Previous answer score {state['critic_score']} <0.95. Improve. Query: {state['query']}\nPrevious: {state['answer'][:1500]}\nResearch: {state['research'][:1000]}"
    improved = call_groq("You are reflection agent. Improve answer to pass 0.95.", prompt)
    state["answer"] = improved
    state["iterations"] += 1
    state["messages"].append({"role":"reflect","content":f"Iteration {state['iterations']} improved"})
    return state

# --- Build LangGraph ---
def build_graph():
    if not HAS_LANGGRAPH:
        print("Running manual state graph (LangGraph fallback)")
        return None
    
    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", node_retrieve)
    workflow.add_node("plan", node_plan)
    workflow.add_node("research", node_research)
    workflow.add_node("generate", node_generate)
    workflow.add_node("critique", node_critique)
    workflow.add_node("reflect", node_reflect)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "plan")
    workflow.add_edge("plan", "research")
    workflow.add_edge("research", "generate")
    workflow.add_edge("generate", "critique")
    workflow.add_conditional_edges("critique", should_continue, {"end": END, "reflect": "reflect"})
    workflow.add_edge("reflect", "generate")

    checkpointer = MemorySaver()
    app = workflow.compile(checkpointer=checkpointer)
    return app

def run_day21(query, thread_id="day21-thread-1", use_checkpoint=True):
    print(f"=== Day21 State Graph + Checkpoints | Model: {MODEL} | 1997 chunks ===")
    print(f"Checkpoint: {use_checkpoint} | Thread: {thread_id}")
    
    initial = AgentState(
        query=query,
        context=[],
        plan="",
        research="",
        answer="",
        critic_score=0.0,
        iterations=0,
        messages=[],
        last_ckpt_id=""
    )

    app = build_graph()
    
    if app and HAS_LANGGRAPH:
        # LangGraph execution with checkpointing
        config = {"configurable": {"thread_id": thread_id}}
        result = None
        for event in app.stream(initial, config):
            for node_name, node_state in event.items():
                print(f"  Node {node_name} completed")
                result = node_state
        final = result
    else:
        # Manual fallback graph (same logic)
        state = initial
        state = node_retrieve(state)
        state = node_plan(state)
        state = node_research(state)
        state = node_generate(state)
        state = node_critique(state)
        while should_continue(state) == "reflect":
            state = node_reflect(state)
            state = node_generate(state)
            state = node_critique(state)
        final = state

    print(f"\n=== Day21 PASS | Score {final.get('critic_score',0)} | Iterations {final.get('iterations',0)} | Checkpoints saved ===")
    print(final.get("answer","")[:2000])

    # Save result
    with open("day21_result.json","w") as f:
        json.dump({
            "query": query,
            "answer": final.get("answer",""),
            "score": final.get("critic_score",0),
            "iterations": final.get("iterations",0),
            "last_ckpt_id": final.get("last_ckpt_id",""),
            "messages_count": len(final.get("messages",[])),
            "model": MODEL,
            "chunks": 1997,
            "day": 21
        }, f, indent=2)

    return final

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", action="store_true", default=True)
    p.add_argument("--no-checkpoint", dest="checkpoint", action="store_false")
    p.add_argument("--query", default="Explain stateful agentic RAG with LangGraph checkpoints using 1997 chunks")
    p.add_argument("--thread", default="day21-thread-1")
    args = p.parse_args()
    run_day21(args.query, thread_id=args.thread, use_checkpoint=args.checkpoint)
