# Day25: MCP Server - 1997 chunks as Model Context Protocol tools

Your RAG is now callable by **Claude Desktop, Cursor, Windsurf, any MCP client**.

## Stack

`RAG (1997) + Tools (4) + MCP + Memory (checkpoints) + Evaluation (0.9) + API (Day24) + Docker`

## Tools exposed via MCP

| Tool | Description |
|---|---|
| `search_chroma` | Search 1997 chunks ChromaDB, returns top-k trimmed 900 chars TPM-safe |
| `agentic_rag` | Full pipeline: search → Groq single call (700 tokens) → critique → checkpoint. Returns JSON with answer, score, checkpoint_id |
| `critique_answer` | Score 0-1 like Day18 critic, threshold 0.95 |
| `save_checkpoint` | Persist answer to `langgraph_checkpoints` |
| `health` | Server health + counts |
| `list_checkpoints` | Last 5 checkpoints |

All tools use **DIRECT** calls (no LLM tool_choice loop) → 3 calls, ~1800 tokens, fits Groq 8000 TPM.

## Run locally

```bash
pip install -r day25/requirements.txt

# Test stdio (Claude Desktop style)
python day25/mcp_server.py
# then send JSON-RPC, or use mcp inspector

# Inspector (web UI to test tools)
npx @modelcontextprotocol/inspector python day25/mcp_server.py
# Open http://localhost:6274
# Click "List Tools" -> you see 6 tools
# Call search_chroma: query="stateful agentic RAG checkpoints"
```

## Claude Desktop config

Edit `claude_desktop_config.json`:

Windows: `%APPDATA%\Claude\claude_desktop_config.json`
WSL: `~/.config/Claude/claude_desktop_config.json` or check Claude docs

```json
{
  "mcpServers": {
    "day25-rag-1997": {
      "command": "python",
      "args": ["/home/nortran/ai-engineer-track2/day25/mcp_server.py"],
      "env": {
        "GROQ_API_KEY": "your_groq_key"
      }
    }
  }
}
```

Restart Claude Desktop. You will see 🔨 hammer icon with 6 tools. Ask:

> "Use search_chroma to explain stateful agentic RAG with 1997 chunks"

Claude will call your local ChromaDB!

## Test with Python client

```python
# test_mcp_client.py
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    params = StdioServerParameters(
        command="python",
        args=["day25/mcp_server.py"],
        env={"GROQ_API_KEY": os.getenv("GROQ_API_KEY")}
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print([t.name for t in tools.tools])
            
            # Call search_chroma
            result = await session.call_tool("search_chroma", {"query": "LangGraph checkpoints", "n_results": 2})
            print(result.content[0].text)

asyncio.run(main())
```

## Why MCP matters?

- Day23: Tools = Python functions only you can call
- Day24: API = HTTP endpoint anyone can curl
- Day25: MCP = **standard protocol** - any LLM (Claude, GPT, Cursor) can discover and call your 1997 chunks automatically

This is the enterprise vision you listed: **RAG + Tools + MCP + Memory + Evaluation + Tracing + Guardrails + API + Docker**

Day25 completes MCP ✅
