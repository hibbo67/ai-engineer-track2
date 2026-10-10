# Day22 Eval Swarm Report

Overall: 0.920 | PASS

Model: openai/gpt-oss-20b | Chunks: 1997 | Queries: 5

## Query: Explain stateful agentic RAG with LangGraph checkpoints using 1997 chunks
- Score: 0.850 (F:0.7 R:0.95 C:0.9)
- Answer: **Stateful Agentic Retrieval‑Augmented Generation (RAG) with LangGraph checkpoints**

LangGraph builds a *stateful* RAG pipeline that can be paused, resumed, or inspected at any node.  
- **Stateful**: The graph’s state is a `TypedDict` that holds all the information needed for the next step:  
  - `query`, `context`, `plan`, `research`, `answer`, `critic_score`, `iterations`, `messages`, and `last_ckpt_id` (1846, 1366, 1561).  
  - This dictionary is passed along the edges, so every node receiv...

## Query: How does Day21 state graph save checkpoints in ChromaDB?
- Score: 0.950 (F:1.0 R:0.9 C:0.95)
- Answer: Day 21’s state graph does **not** use ChromaDB to persist its checkpoints.  
Instead, the graph’s `checkpoint` logic is implemented in `day21/graph.py` and relies on LangGraph’s built‑in checkpointing mechanisms: a `MemorySaver` for in‑memory persistence or a Postgres‑based checkpointer for durable storage.  Each node visit triggers a state snapshot that is stored via one of these two back‑ends, and the agent’s next step is chosen by the `should_continue` conditional edge.  

> *“LangGraph check...

## Query: What is the difference between retrieve, plan, research, generate, critique nodes?
- Score: 0.850 (F:0.6 R:1.0 C:0.95)
- Answer: **Retrieve** – pulls the relevant background or evidence from a knowledge source (e.g., a vector store or external API) and stores it in the state’s `context`.  
**Plan** – generates a step‑by‑step strategy for answering the user’s query and writes that strategy into the state’s `plan`.  
**Research** – follows the plan, queries the LLM (Groq) for evidence, and appends the returned information to the state’s `research`. The research node is implemented in *day21/graph.py* using Groq 259, 1984, 2...

## Query: Explain reflection loop and critic score threshold 0.95 in Day18-21 pipeline
- Score: 0.967 (F:1.0 R:0.95 C:0.95)
- Answer: **Reflection loop & critic‑score threshold in the Day 18‑21 RAG pipeline**

The pipeline uses a *reflection loop* that iteratively refines the answer until it meets a quality criterion.  
* **Reflection loop** – After the initial answer is generated, a critic evaluates it. If the critic’s score is below **0.95**, the system re‑runs the answer‑generation step (up to three times). This loop is implemented in `day21/graph.py` and uses the Groq model for both the answer and the critic. The loop is c...

## Query: How to implement memory and supervisor pattern from Day19-20 into Day21?
- Score: 0.983 (F:1.0 R:1.0 C:0.95)
- Answer: **How to bring the Day 19‑20 memory & supervisor pattern into Day 21**

| Step | What to do | Where it comes from |
|------|------------|---------------------|
| 1. **Add long‑term memory** | Use a ChromaDB collection to store every turn of the conversation.  Each message is indexed with a unique ID and the user’s query is embedded with Groq’s embedding model so it can be retrieved later. | Chunk 1569, 219, 1449 |
| 2. **Define a stateful graph** | Create a `State` `TypedDict` that holds all the...

