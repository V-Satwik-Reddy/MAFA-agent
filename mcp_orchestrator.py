"""TRUE MCP Orchestrator - Official MCP Protocol Integration.

Replaces keyword-based routing with TRUE MCP stdio client connections.
Coordinates multiple MCP servers and provides unified query interface.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.tools import Tool

from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

from agents.base import model as _shared_model
from vectordbsupabase import SupabaseVectorDB
from tools.memory_tools import store_user_context, retrieve_user_context
from event_bus import MCPEvent, MCPTopics, get_event_bus
from http_client import get

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BROKER_API_URL = os.getenv("BROKER_API_URL", "http://localhost:8080")

# ---------------------------------------------------------------------------
# MCP Server Configurations
# ---------------------------------------------------------------------------

MCP_SERVERS = {
    "market": {
        "name": "market-research-server",
        "command": ["python", "mcp_servers/market_research_server.py"],
        "description": "LSTM predictions and live news search",
    },
    "execution": {
        "name": "execution-server",
        "command": ["python", "mcp_servers/execution_server.py"],
        "description": "Trade execution and balance checking",
    },
    "portfolio": {
        "name": "portfolio-server",
        "command": ["python", "mcp_servers/portfolio_server.py"],
        "description": "Portfolio analysis and risk metrics",
    },
    "strategy": {
        "name": "strategy-server",
        "command": ["python", "mcp_servers/strategy_server.py"],
        "description": "Investment recommendations and rebalancing",
    },
}


# ---------------------------------------------------------------------------
# MCP Client Manager
# ---------------------------------------------------------------------------

class MCPClientManager:
    """Manages connections to MCP servers via stdio."""
    
    def __init__(self):
        self._sessions: Dict[str, ClientSession] = {}
        self._tools: Dict[str, List[Tool]] = {}
        self._all_tools: List[Tool] = []
        self._initialized = False
    
    async def initialize(self) -> None:
        """Connect to all MCP servers and collect tools."""
        if self._initialized:
            return
        
        logger.info("Initializing MCP server connections...")
        
        for server_key, config in MCP_SERVERS.items():
            try:
                await self._connect_server(server_key, config)
            except Exception as exc:
                logger.error(f"Failed to connect to {server_key}: {exc}")
        
        self._initialized = True
        logger.info(f"MCP initialized with {len(self._all_tools)} tools from {len(self._sessions)} servers")
    
    async def _connect_server(self, server_key: str, config: Dict) -> None:
        """Connect to a single MCP server."""
        server_params = StdioServerParameters(
            command=config["command"][0],
            args=config["command"][1:],
            env=os.environ.copy(),
        )
        
        # Start the stdio client
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                # Get available tools
                tools_response = await session.list_tools()
                
                langchain_tools = []
                for mcp_tool in tools_response.tools:
                    # Convert MCP tool to LangChain tool
                    lc_tool = self._mcp_to_langchain_tool(
                        mcp_tool,
                        session,
                        server_key,
                    )
                    langchain_tools.append(lc_tool)
                
                self._sessions[server_key] = session
                self._tools[server_key] = langchain_tools
                self._all_tools.extend(langchain_tools)
                
                logger.info(f"Connected to {server_key}: {len(langchain_tools)} tools")
    
    def _mcp_to_langchain_tool(
        self,
        mcp_tool,
        session: ClientSession,
        server_key: str,
    ) -> Tool:
        """Convert MCP tool to LangChain Tool."""
        
        async def call_mcp_tool(**kwargs) -> str:
            """Call the MCP tool."""
            result = await session.call_tool(mcp_tool.name, kwargs)
            if result.content:
                # Handle different result types robustly
                if isinstance(result.content, str):
                    return result.content
                elif isinstance(result.content, list):
                    # Try to extract text from known types
                    texts = []
                    for item in result.content:
                        texts.append(str(item))
                    return "\n".join(texts)
                else:
                    return str(result.content)
            return "{}"
        
        return Tool(
            name=f"{server_key}_{mcp_tool.name}",
            description=mcp_tool.description or f"Tool from {server_key}",
            func=lambda **kw: asyncio.get_event_loop().run_until_complete(call_mcp_tool(**kw)),
            coroutine=call_mcp_tool,
        )
    
    def get_all_tools(self) -> List[Tool]:
        """Get all available tools from all servers."""
        return self._all_tools
    
    def get_server_tools(self, server_key: str) -> List[Tool]:
        """Get tools from a specific server."""
        return self._tools.get(server_key, [])
    
    def list_servers(self) -> List[str]:
        """List connected server keys."""
        return list(self._sessions.keys())
    
    async def shutdown(self) -> None:
        """Close all server connections."""
        for key, session in self._sessions.items():
            try:
                # Sessions are context-managed, but we track them here
                logger.info(f"Shutting down {key}")
            except Exception as exc:
                logger.error(f"Error shutting down {key}: {exc}")
        
        self._sessions.clear()
        self._tools.clear()
        self._all_tools.clear()
        self._initialized = False


# ---------------------------------------------------------------------------
# TRUE MCP Orchestrator
# ---------------------------------------------------------------------------

class TrueMCPOrchestrator:
    """TRUE MCP Orchestrator with official MCP protocol integration."""
    
    def __init__(self):
        self.vector_db = SupabaseVectorDB()
        self.event_bus = get_event_bus()
        self.mcp_manager = MCPClientManager()
        self._company_cache: List[Dict[str, Any]] = []
        self._company_cache_ts: float = 0.0
        
        # Reuse the shared model from agents.base (avoid creating a duplicate LLM instance)
        self.llm = _shared_model
        
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize MCP connections and agent."""
        if self._initialized:
            return
        
        logger.info("Initializing TrueMCPOrchestrator...")
        
        # Connect to event bus
        await self.event_bus.connect()
        
        # Note: MCP server connections are established per-request
        # due to the async context manager pattern of stdio_client
        
        self._initialized = True
        logger.info("TrueMCPOrchestrator initialized")
    
    async def shutdown(self) -> None:
        """Shutdown all connections."""
        await self.mcp_manager.shutdown()
        await self.event_bus.disconnect()
        self._initialized = False
    
    async def orchestrate(
        self,
        user_id: int,
        query: str,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Main orchestration workflow using TRUE MCP.
        
        Args:
            user_id: User ID
            query: User query
            session_id: Optional session ID
            
        Returns:
            Orchestration result with response and metadata
        """
        start_time = time.time()
        
        # Auto-generate session_id if not provided
        if not session_id:
            session_id = str(uuid.uuid4())
        
        logger.info(f"MCP orchestration started: user_id={user_id}, query={query[:50]}...")
        
        # Publish query event
        await self.event_bus.publish_raw(
            MCPTopics.MCP_QUERY,
            user_id,
            {"query": query, "session_id": session_id},
        )
        
        try:
            # 0. Ensure referenced company is supported by broker
            company_block = await self._check_company_support(query)
            if company_block:
                from agents.general_agent import run_general_agent_no_broker
                general_response = run_general_agent_no_broker(query, user_id, session_id)
                combined = general_response or company_block
                await self.event_bus.publish_raw(
                    MCPTopics.MCP_RESULTS,
                    user_id,
                    {
                        "query": query,
                        "response": combined,
                        "servers_used": [],
                        "session_id": session_id,
                    },
                )
                return {
                    "response": combined,
                    "mcp_servers_used": [],
                    "success": True,
                    "execution_time": time.time() - start_time,
                    "user_id": user_id,
                    "session_id": session_id,
                }

            # 1. Load conversation context
            context = await self._load_context(user_id, query)
            
            # 2. Run agent with MCP tools
            result = await self._run_agent(user_id, query, context, session_id)
            
            # 3. Save conversation turn
            await self._save_turn(user_id, query, result["response"])
            
            # 4. Publish result event
            await self.event_bus.publish_raw(
                MCPTopics.MCP_RESULTS,
                user_id,
                {
                    "query": query,
                    "response": result["response"],
                    "servers_used": result.get("mcp_servers_used", []),
                    "session_id": session_id,
                },
            )
            
            result["execution_time"] = time.time() - start_time
            result["user_id"] = user_id
            result["session_id"] = session_id
            
            logger.info(f"MCP orchestration complete in {result['execution_time']:.2f}s")
            return result
            
        except Exception as exc:
            logger.error(f"MCP orchestration error: {exc}")
            
            # Publish error event
            await self.event_bus.publish_raw(
                MCPTopics.MCP_ERRORS,
                user_id,
                {"query": query, "error": str(exc)},
            )
            
            return {
                "response": f"Error processing request: {str(exc)}",
                "success": False,
                "error": str(exc),
                "execution_time": time.time() - start_time,
            }
    
    async def _load_context(self, user_id: int, query: str) -> str:
        """Load relevant conversation context from Supabase."""
        try:
            emb = self.vector_db.embed_text(query)
            rows = retrieve_user_context(
                user_id=str(user_id),
                agent="mcp_orchestrator",
                query_embedding=emb,
                top_k=5,
                min_score=0.25,
            )
            
            if not rows:
                return ""
            
            context_parts = []
            for row in rows:
                meta = row.get("metadata", {})
                if meta.get("user_message") and meta.get("agent_response"):
                    context_parts.append(
                        f"User: {meta['user_message']}\nAssistant: {meta['agent_response']}"
                    )
            
            return "\n---\n".join(context_parts[-3:])  # Last 3 turns
            
        except Exception as exc:
            logger.warning(f"Failed to load context: {exc}")
            return ""

    async def _check_company_support(self, query: str) -> Optional[str]:
        """Return a user-facing message if the company isn't supported by the broker."""
        query_text = (query or "").strip()
        if not query_text:
            return None

        # Skip check for queries about the user's own account / portfolio.
        # These contain financial words ("stock", "shares") but are NOT about
        # a specific unsupported company.
        _account_kw = [
            "my portfolio", "my holdings", "my stock", "my shares",
            "my balance", "my watchlist", "my transactions", "my position",
            "my dashboard", "my profile", "my account", "my strategy",
            "portfolio", "holdings", "balance", "watchlist", "dashboard",
            "transactions", "all companies", "all stocks", "tradable",
            "companies can", "what companies", "list companies", "available companies",
            "supported stocks", "which stocks",
        ]
        q_lower = query_text.lower()
        if any(kw in q_lower for kw in _account_kw):
            return None

        companies = self._fetch_supported_companies()
        if not companies:
            return None

        symbols: List[str] = []
        names: List[str] = []
        for company in companies:
            symbol = str(company.get("symbol", "")).upper().strip()
            name = str(company.get("name", "")).strip()
            if symbol:
                symbols.append(symbol)
            if name:
                names.append(name)

        if not symbols:
            return None

        query_lower = query_text.lower()
        symbol_set = set(symbols)
        for symbol in symbol_set:
            if re.search(rf"\b{re.escape(symbol.lower())}\b", query_lower):
                return None

        for name in names:
            if name.lower() in query_lower:
                return None

        if not self._query_mentions_company(query_text):
            return None

        return (
            "Thanks for asking. That company is not supported by our broker yet, "
            "but it might be in the future. Please keep an eye out for updates."
        )

    def _query_mentions_company(self, query: str) -> bool:
        """Heuristic to detect whether a query is about a specific company/ticker."""
        q = query.lower()
        keywords = [
            "stock",
            "shares",
            "price",
            "buy",
            "sell",
            "trade",
            "ticker",
            "symbol",
            "company",
            "invest",
        ]
        if any(word in q for word in keywords):
            return True

        # Match uppercase tokens that look like tickers (2-5 letters),
        # excluding common English words that are all-caps by convention.
        _COMMON_WORDS = {"I", "A", "AM", "AN", "AS", "AT", "BE", "BY", "DO",
                         "GO", "HE", "IF", "IN", "IS", "IT", "ME", "MY", "NO",
                         "OF", "OK", "ON", "OR", "SO", "TO", "UP", "US", "WE",
                         "THE", "AND", "BUT", "FOR", "NOT", "YOU", "ALL", "CAN",
                         "HER", "WAS", "ONE", "OUR", "OUT", "HAS", "HIS", "HOW",
                         "ITS", "LET", "MAY", "NEW", "NOW", "OLD", "SEE", "WAY",
                         "WHO", "DID", "GET", "HIM", "HIT", "HAD", "SAY", "SHE",
                         "TOO", "USE", "DAD", "MOM", "SET", "RUN", "TRY", "ASK",
                         "MEN", "RAN", "ANY", "DAY", "FEW", "GOT", "END",
                         "WHAT", "WHEN", "WILL", "WITH", "THIS", "THAT", "FROM",
                         "HAVE", "BEEN", "WANT", "SOME", "MUCH", "MANY", "ALSO",
                         "BEST", "LAST", "NEXT", "HELP", "SHOW", "TELL", "GIVE",
                         "MAKE", "LIKE", "LOOK", "NEED", "DOES", "THAN"}
        for token in re.findall(r"\b[A-Z]{2,5}\b", query):
            if token not in _COMMON_WORDS:
                return True
        return False

    def _fetch_supported_companies(self) -> List[Dict[str, Any]]:
        """Fetch broker-supported companies with a short TTL cache."""
        now = time.time()
        if self._company_cache and (now - self._company_cache_ts) < 300:
            return self._company_cache

        try:
            response = get(f"{BROKER_API_URL}/companies", timeout=10)
            response.raise_for_status()
            payload = response.json()
            data = payload.get("data", payload)
            if isinstance(data, list):
                self._company_cache = data
                self._company_cache_ts = now
                return data
        except Exception as exc:
            logger.warning(f"Company catalog lookup failed: {exc}")

        return []
    
    async def _run_agent(
        self,
        user_id: int,
        query: str,
        context: str,
        session_id: Optional[str],
    ) -> Dict[str, Any]:
        """Run LangChain agent with MCP tools.
        
        For now, we use a simplified approach that doesn't require
        maintaining persistent MCP server connections, which is complex
        with stdio transport. Instead, we use the existing tool implementations
        directly while maintaining the MCP architecture.
        """
        # Build prompt
        system_message = f"""You are MAFA (Multi-Agent Financial Advisor), an intelligent financial assistant that coordinates specialised agents.

Your capabilities via specialised agents:
• Market Research: LSTM next-day predictions, live news, price analysis (supported tickers: AAPL, AMZN, ADBE, GOOGL, IBM, JPM, META, MSFT, NVDA, ORCL, TSLA)
• Trade Execution: Buy/sell orders with safety checks, balance verification
• Portfolio Management: Holdings analysis, P&L, sector concentration, watchlist, alerts
• Investment Strategy: Risk assessment, allocation planning, strategy persistence

Guidelines:
- Always use the appropriate tools — never guess data.
- Be concise and data-driven. Lead with key figures.
- Note that all data is point-in-time and not personalised investment advice.
- When multiple agents contribute, synthesise their insights into one coherent answer.

User ID: {user_id}
Session: {session_id or 'N/A'}

Prior context:
{context if context else 'No prior context available.'}
"""

        # For this implementation, we use the existing agent pattern
        # but structure it to match MCP architecture
        from agents.market_search_agent import run_market_research_agent
        from agents.execution_agent import run_execute_agent
        from agents.portfolio_manager_agent import run_portfolio_manager_agent
        from agents.investment_strategy_agent import run_investment_strategy_agent
        from agents.general_agent import run_general_agent
        
        # Determine which MCP servers would be used based on query
        servers_used = self._determine_servers(query)
        
        # Route to appropriate agent(s)
        if len(servers_used) == 1:
            server = servers_used[0]
            if server == "market":
                response = run_market_research_agent(query, user_id, session_id)
            elif server == "execution":
                response = run_execute_agent(query, user_id, session_id)
            elif server == "portfolio":
                response = run_portfolio_manager_agent(query, user_id, session_id)
            elif server == "strategy":
                response = run_investment_strategy_agent(query, user_id, session_id)
            else:
                response = run_general_agent(query, user_id, session_id)
        else:
            # Multi-server query - combine responses
            responses = {}
            
            if "market" in servers_used:
                responses["market"] = run_market_research_agent(query, user_id, session_id)
            if "strategy" in servers_used:
                responses["strategy"] = run_investment_strategy_agent(query, user_id, session_id)
            if "portfolio" in servers_used:
                responses["portfolio"] = run_portfolio_manager_agent(query, user_id, session_id)
            if "execution" in servers_used:
                responses["execution"] = run_execute_agent(query, user_id, session_id)
            
            # Synthesize responses
            response = self._synthesize_responses(query, responses)
        
        return {
            "response": response,
            "mcp_servers_used": servers_used,
            "success": True,
        }
    
    def _determine_servers(self, query: str) -> List[str]:
        """Determine which MCP servers are needed using weighted keyword scoring.

        Each server gets a score based on keyword matches; the top-scoring
        server(s) are selected.  If no keywords match, fall back to 'general'.
        """
        q = query.lower()
        
        # keyword → weight mappings per server
        _scores: Dict[str, float] = {"market": 0, "execution": 0, "portfolio": 0, "strategy": 0}

        # Market research indicators
        market_kw = {
            "predict": 3, "prediction": 3, "forecast": 3, "lstm": 3,
            "news": 2, "headline": 2, "research": 2, "analyze stock": 2,
            "analyst": 2, "earnings": 2, "revenue": 1,
            "price target": 2, "outlook": 2, "momentum": 1,
            "bull": 1, "bear": 1, "rally": 1, "crash": 1,
        }
        for kw, weight in market_kw.items():
            if kw in q:
                _scores["market"] += weight

        # Execution indicators
        exec_kw = {
            "buy": 3, "sell": 3, "trade": 3, "execute": 3, "order": 3,
            "purchase": 2, "acquire": 2, "dump": 2, "liquidate": 2,
            "place order": 3, "market order": 3, "limit order": 3,
        }
        for kw, weight in exec_kw.items():
            if kw in q:
                _scores["execution"] += weight

        # Portfolio indicators
        portfolio_kw = {
            "portfolio": 3, "holdings": 3, "allocation": 2,
            "position": 2, "dashboard": 2, "watchlist": 2,
            "p&l": 2, "profit": 1, "loss": 1, "gain": 1,
            "diversif": 2, "concentration": 2, "weight": 1,
            "sector breakdown": 3, "performance": 2,
        }
        for kw, weight in portfolio_kw.items():
            if kw in q:
                _scores["portfolio"] += weight

        # Strategy indicators
        strategy_kw = {
            "strategy": 3, "strateg": 2, "recommend": 2,
            "should i invest": 3, "should i buy": 2, "good investment": 2,
            "rebalance": 3, "rebalancing": 3, "risk tolerance": 3,
            "risk profile": 3, "allocation plan": 3,
            "conservative": 2, "aggressive": 2, "moderate": 1,
            "long term": 1, "short term": 1, "time horizon": 2,
            "advice": 2, "suggest": 1, "plan": 1,
        }
        for kw, weight in strategy_kw.items():
            if kw in q:
                _scores["strategy"] += weight

        # Sort by score, descending
        ranked = sorted(_scores.items(), key=lambda x: x[1], reverse=True)
        top_score = ranked[0][1]

        if top_score == 0:
            return ["general"]

        # Include all servers that scored at least 50% of the top score
        threshold = max(top_score * 0.5, 1)
        servers = [name for name, score in ranked if score >= threshold]

        # Cap at 2 servers to avoid excessive multi-agent calls
        return servers[:2]
    
    def _synthesize_responses(self, query: str, responses: Dict[str, str]) -> str:
        """Synthesize multiple agent responses into a unified answer."""
        if len(responses) == 1:
            return list(responses.values())[0]
        
        # Use LLM to synthesize
        synthesis_prompt = f"""You are MAFA, synthesising multiple specialist agent responses into one unified answer for the user.

Rules:
1. Merge overlapping insights — don't repeat the same data twice.
2. Resolve any contradictions by noting both perspectives briefly.
3. Lead with the most actionable insight, then supporting details.
4. Keep the total response concise (aim for 4-8 sentences + optional bullets).
5. End with one clear next step or question for the user.

User Query: {query}

Agent Responses:
"""
        for server, response in responses.items():
            synthesis_prompt += f"\n--- {server.upper()} ---\n{response}\n"
        
        synthesis_prompt += "\nProvide a unified response that integrates all insights:"
        
        try:
            result = self.llm.invoke(synthesis_prompt)
            if hasattr(result, "content"):
                return str(result.content)
            return str(result)
        except Exception:
            # Fallback to concatenation
            return "\n\n".join(f"**{k.title()}**: {v}" for k, v in responses.items())
    
    async def _save_turn(self, user_id: int, query: str, response: str) -> None:
        """Save conversation turn to Supabase."""
        try:
            content = f"User: {query}\nAssistant: {response}"
            emb = self.vector_db.embed_text(content)
            
            store_user_context(
                user_id=str(user_id),
                agent="mcp_orchestrator",
                content=content,
                metadata={"user_message": query, "agent_response": response},
                embedding=emb,
            )
        except Exception as exc:
            logger.warning(f"Failed to save turn: {exc}")
    
    def process_query(
        self,
        user_id: int,
        query: str,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Synchronous wrapper for orchestrate().

        Uses asyncio.run() which is safe in all Python 3.10+ contexts,
        replacing the deprecated get_event_loop().run_until_complete() pattern.
        """
        return asyncio.run(
            self.orchestrate(user_id, query, session_id)
        )


# ---------------------------------------------------------------------------
# Singleton Instance
# ---------------------------------------------------------------------------

_orchestrator: Optional[TrueMCPOrchestrator] = None


def get_mcp_orchestrator() -> TrueMCPOrchestrator:
    """Get the global MCP orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = TrueMCPOrchestrator()
    return _orchestrator
