# MAFA Agents - Copilot Instructions

## Project Overview

MAFA (Multi-Agent Financial Advisor) is a TRUE MCP-based multi-agent system for financial trading and investment advice. The system uses official MCP protocol with stdio transport for agent communication.

## Architecture

### TRUE MCP Servers (`mcp_servers/`)

Each server exposes tools via official MCP stdio protocol:

- **market_research_server.py**: LSTM predictions, live news search
  - Tools: `predict_next_day`, `get_live_news`, `get_market_analysis`
- **execution_server.py**: Trade execution, balance/holdings
  - Tools: `execute_trade`, `check_balance`, `check_holdings`, `get_stock_price`, `validate_trade`
- **portfolio_server.py**: Portfolio analysis and risk metrics
  - Tools: `get_portfolio_snapshot`, `get_portfolio_allocation`, `analyze_portfolio_risk`, `check_position`
- **strategy_server.py**: Investment recommendations and rebalancing
  - Tools: `generate_investment_recommendation`, `get_risk_profile`, `portfolio_rebalancing_proposal`

### Core Components

- **mcp_orchestrator.py**: TRUE MCP orchestrator that coordinates all MCP servers
- **event_bus.py**: Redis pub/sub for agent communication (MCPTopics, MCPEvent)
- **API.py**: FastAPI backend with `/mcp/query` endpoint and WebSocket streaming
- **vectordbsupabase.py**: Supabase-backed vector memory for all agents

## Key Patterns

### MCP Server Pattern
```python
from mcp.server.fastmcp import FastMCP
mcp = FastMCP("server-name")

@mcp.tool()
def my_tool(arg: str) -> str:
    """Tool description."""
    return json.dumps({"result": "..."})

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

### Memory: ALL agents use SupabaseVectorDB
```python
from vectordbsupabase import SupabaseVectorDB
vector_db = SupabaseVectorDB()
```

### Event Bus Pattern
```python
from event_bus import get_event_bus, MCPTopics
event_bus = get_event_bus()
await event_bus.publish_raw(MCPTopics.MCP_RESULTS, user_id, payload)
```

## Supported Tickers

LSTM models are available for: `AAPL, AMZN, ADBE, GOOGL, IBM, JPM, META, MSFT, NVDA, ORCL, TSLA`

## Environment Variables

Required in `.env`:
- `GOOGLE_API_KEY`: Google Generative AI API key
- `SUPABASE_URL`: Supabase project URL
- `SUPABASE_KEY`: Supabase service key
- Redis connection (default: `redis://localhost:6379`)

## Running

```bash
# Install dependencies
pip install -r requirements.txt

# Start API server
uvicorn API:app --host 0.0.0.0 --port 5000

# Test MCP server directly
python mcp_servers/market_research_server.py
```

## API Endpoints

- `POST /mcp/query` - TRUE MCP orchestration (primary endpoint)
- `GET /health` - Health check with MCP server status
- `GET /mcp/servers` - List available MCP servers and tools
- `WS /ws/mcp-stream` - WebSocket for real-time event streaming
- Individual agent endpoints (legacy): `/execute-agent`, `/market-research-agent`, etc.

## Event Topics (MCPTopics)

- `market.raw`, `market.predictions`, `market.news`
- `strategy.recommendations`, `strategy.alerts`
- `portfolio.snapshots`, `portfolio.updates`
- `execution.orders`, `execution.results`, `execution.errors`
- `mcp.query`, `mcp.results`, `mcp.errors`
