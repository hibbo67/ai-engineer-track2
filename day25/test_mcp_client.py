"""
Test Day25 MCP server with stdio client (without Claude Desktop)
"""
import asyncio, os, sys, json
from pathlib import Path

# Add day23 to path
sys.path.insert(0, str(Path(__file__).parent.parent / "day23"))

async def main():
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except ImportError:
        print("pip install mcp")
        return

    params = StdioServerParameters(
        command="python",
        args=["day25/mcp_server.py"],
        env={"GROQ_API_KEY": os.getenv("GROQ_API_KEY", "")}
    )
    
    print("=== Day25 MCP Test Client ===")
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("Initialized")
            
            tools_res = await session.list_tools()
            print(f"Tools: {[t.name for t in tools_res.tools]}")
            
            # 1. health
            print("\n--- health ---")
            res = await session.call_tool("health", {})
            print(res.content[0].text[:500])
            
            # 2. search_chroma
            print("\n--- search_chroma ---")
            res = await session.call_tool("search_chroma", {"query": "stateful agentic RAG checkpoints", "n_results": 2})
            print(res.content[0].text[:800])
            
            # 3. agentic_rag (full)
            print("\n--- agentic_rag ---")
            res = await session.call_tool("agentic_rag", {"query": "Explain stateful agentic RAG with LangGraph checkpoints using 1997 chunks", "n_results": 3})
            text = res.content[0].text
            print(text[:2000])
            
            # Parse score
            try:
                data = json.loads(text)
                print(f"\nScore: {data.get('score')} | Tool calls: {data.get('tool_calls')} | Latency: {data.get('latency_ms')}ms")
            except:
                pass

if __name__ == "__main__":
    asyncio.run(main())
