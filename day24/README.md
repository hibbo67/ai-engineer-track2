# Day24: API + Docker - Enterprise RAG

Enterprise Stack: **RAG + Tools + MCP-ready + Memory + Evaluation + Tracing + Guardrails + API + Docker**

Day23 was CLI. Day24 wraps it as FastAPI.

## Architecture

```
Client -> FastAPI /ask -> tool_search_chroma (1997 chunks) -> Groq single call -> tool_critique -> tool_save_checkpoint -> response
```

- **RAG**: 1997 chunks Chroma `chunks_1997`
- **Tools**: 3 direct calls (no LLM loop, TPM-safe for Groq 8000 tier)
- **Memory**: AgentState + `langgraph_checkpoints` collection
- **Evaluation**: critique 0.9 score
- **Tracing**: logs + result.json + latency_ms
- **Guardrails**: query length 5-2000 chars validation
- **API**: FastAPI `/health`, `/ask`, `/docs`
- **Docker**: Dockerfile with healthcheck

## Run locally

```bash
source venv/bin/activate
pip install -r day24/requirements.txt

# Dev
uvicorn day24.api:app --host 0.0.0.0 --port 8000 --reload

# Test
curl http://localhost:8000/health
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d '{"query":"Explain stateful agentic RAG with LangGraph checkpoints using 1997 chunks"}'

# Docs
open http://localhost:8000/docs
```

## Run Docker

```bash
docker build -f day24/Dockerfile -t day24-rag-api .
docker run -p 8000:8000 -e GROQ_API_KEY=$GROQ_API_KEY -v $(pwd)/chroma_db:/app/chroma_db day24-rag-api

curl http://localhost:8000/health
```

## Why DIRECT tools?

Day23 original used Groq `tool_choice=auto` loop -> 14 calls, 9502 tokens, 429 rate limit, 400 JSON parse error.

Day24 uses DIRECT function calls:
- Tool 1: search_chroma() -> 900 chars
- LLM: single call 700 tokens
- Tool 2: critique() -> score
- Tool 3: save_checkpoint() -> id

Total ~1800 tokens, fits 8000 TPM tier `org_01m3fz7sf8ex8tdx895a8mddyq`.

## Next: Day25 MCP

Expose `search_chroma` as MCP server so Claude Desktop can call it.
