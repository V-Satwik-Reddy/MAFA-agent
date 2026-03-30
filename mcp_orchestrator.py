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
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

from agents.base import model as _shared_model, sanitize_user_response
from vectordbsupabase import SupabaseVectorDB
from tools.memory_tools import store_user_context
from event_bus import MCPTopics, get_event_bus
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


@dataclass
class IntentRouteDecision:
    """Structured router decision returned by semantic intent classification."""

    primary_agents: List[str]
    secondary_agents: List[str]
    confidence: float
    requires_clarification: bool
    fallback_route: str
    rationale_short: str


ALLOWED_SERVER_KEYS = {"market", "execution", "portfolio", "strategy", "general"}
_TICKER_PATTERNS = (
    re.compile(r"\b(?:for|of|in|about)\s+([A-Za-z]{1,5})\b"),
    re.compile(r"\b([A-Z]{1,5})\b"),
)
_TICKER_COMMON_WORDS = {
    "I", "A", "AN", "THE", "AND", "OR", "FOR", "TO", "OF", "IN", "ON", "AT",
    "MY", "ME", "YOU", "ALL", "CAN", "WILL", "THIS", "THAT", "WITH", "FROM",
    "WHAT", "WHEN", "SHOW", "GIVE", "MOST", "BEST", "NEXT", "STOCK", "PRICE",
    "BUY", "SELL", "QUOTE", "NEWS", "PLAN", "RISK", "GUIDE",
}


# ---------------------------------------------------------------------------
# TRUE MCP Orchestrator
# ---------------------------------------------------------------------------

class TrueMCPOrchestrator:
    """TRUE MCP Orchestrator with official MCP protocol integration."""
    
    def __init__(self):
        self.vector_db = SupabaseVectorDB()
        self.event_bus = get_event_bus()
        self._company_cache: List[Dict[str, Any]] = []
        self._company_cache_ts: float = 0.0
        
        # Reuse the shared model from agents.base (avoid creating a duplicate LLM instance)
        self.llm = _shared_model
        self.semantic_router_enabled = os.getenv("SEMANTIC_ROUTER_ENABLED", "true").strip().lower() != "false"
        self.semantic_router_min_confidence = self._read_float_env("SEMANTIC_ROUTER_MIN_CONFIDENCE", 0.35)
        self.semantic_router_secondary_min_confidence = self._read_float_env("SEMANTIC_ROUTER_SECONDARY_MIN_CONFIDENCE", 0.55)
        self.mcp_transport_enabled = os.getenv("MCP_TRANSPORT_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
        
        self._initialized = False

    @staticmethod
    def _read_float_env(name: str, default: float) -> float:
        """Read a float env var with a safe default."""
        value = os.getenv(name)
        if value is None:
            return default
        try:
            return float(value)
        except ValueError:
            return default
    
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

            # 1. Run agent with MCP tools
            result = await self._run_agent(user_id, query, session_id)
            
            # 2. Save conversation turn
            await self._save_turn(user_id, query, result["response"])
            
            # 3. Publish result event
            await self.event_bus.publish_raw(
                MCPTopics.MCP_RESULTS,
                user_id,
                {
                    "query": query,
                    "response": result["response"],
                    "servers_used": result.get("mcp_servers_used", []),
                    "routing": result.get("routing", {}),
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
    
    async def _check_company_support(self, query: str) -> Optional[str]:
        """Return a user-facing message if the company isn't supported by the broker."""
        query_text = (query or "").strip()
        if not query_text:
            return None

        # Prediction/forecast queries are handled by market agent logic,
        # including supported/unavailable ticker responses.
        q_pred = query_text.lower()
        if any(k in q_pred for k in ("predict", "prediction", "forecast", "next-day", "next day")):
            return None

        # Price lookup queries should be handled by market/general paths,
        # including non-broker fallback data sources.
        if any(k in q_pred for k in ("stock price", "price of", "current price", "quote")):
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
        session_id: Optional[str],
    ) -> Dict[str, Any]:
        """Run LangChain agent with MCP tools.
        
        For now, we use a simplified approach that doesn't require
        maintaining persistent MCP server connections, which is complex
        with stdio transport. Instead, we use the existing tool implementations
        directly while maintaining the MCP architecture.
        """
        # For this implementation, we use the existing agent pattern
        # but structure it to match MCP architecture
        from agents.market_search_agent import run_market_research_agent
        from agents.execution_agent import run_execute_agent
        from agents.portfolio_manager_agent import run_portfolio_manager_agent
        from agents.investment_strategy_agent import run_investment_strategy_agent
        from agents.general_agent import run_general_agent

        in_process_handlers = {
            "market": run_market_research_agent,
            "execution": run_execute_agent,
            "portfolio": run_portfolio_manager_agent,
            "strategy": run_investment_strategy_agent,
            "general": run_general_agent,
        }
        
        # Determine which MCP servers would be used based on query
        route_info = self._resolve_route(query)
        servers_used = route_info["servers"]

        # Optional Stage-2 bridge: invoke MCP servers over stdio transport.
        # Keep execution on existing path to preserve confirmation safeguards.
        mcp_responses: Dict[str, str] = {}
        if self.mcp_transport_enabled:
            for server in servers_used:
                if server in {"market", "portfolio", "strategy"}:
                    try:
                        mcp_text = await self._run_server_via_mcp(server, query, user_id)
                        if mcp_text:
                            mcp_responses[server] = mcp_text
                    except Exception as exc:
                        logger.warning("MCP transport failed for %s: %s", server, exc)

            if mcp_responses:
                if len(mcp_responses) == 1:
                    response = list(mcp_responses.values())[0]
                else:
                    response = self._synthesize_responses(query, mcp_responses)

                response = sanitize_user_response(str(response), user_message=query)
                error_message = self._extract_response_error(response)
                success = error_message is None
                route_info = {
                    **route_info,
                    "transport": "mcp_stdio",
                    "mcp_tools_used": list(mcp_responses.keys()),
                }
                return {
                    "response": response,
                    "mcp_servers_used": servers_used,
                    "routing": route_info,
                    "success": success,
                    "error": error_message,
                }
        
        # Route to appropriate in-process agent(s)
        if len(servers_used) == 1:
            server = servers_used[0]
            handler = in_process_handlers.get(server, run_general_agent)
            response = handler(query, user_id, session_id)
        else:
            responses: Dict[str, str] = {}
            for server in servers_used:
                handler = in_process_handlers.get(server)
                if handler:
                    responses[server] = handler(query, user_id, session_id)

            response = self._synthesize_responses(query, responses)

        response = sanitize_user_response(str(response), user_message=query)

        error_message = self._extract_response_error(response)
        success = error_message is None
        
        return {
            "response": response,
            "mcp_servers_used": servers_used,
            "routing": {**route_info, "transport": "in_process_agents"},
            "success": success,
            "error": error_message,
        }

    async def _run_server_via_mcp(self, server: str, query: str, user_id: int) -> str:
        """Invoke a server through MCP stdio with best-effort tool selection."""
        config = MCP_SERVERS.get(server)
        if not config:
            raise RuntimeError(f"Unknown MCP server key: {server}")

        server_params = StdioServerParameters(
            command=config["command"][0],
            args=config["command"][1:],
            env=os.environ.copy(),
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await self._invoke_server_tool(session, server, query, user_id)

    async def _invoke_server_tool(self, session: ClientSession, server: str, query: str, user_id: int) -> str:
        """Map high-level intent to MCP tool calls for each server."""
        q = (query or "").lower()
        ticker = self._extract_ticker_from_query(query)

        if server == "market":
            if any(k in q for k in ("news", "headline", "trend")):
                return await self._call_mcp_tool_text(session, "get_live_news", {"query": query, "num_results": 5})
            if ticker and any(k in q for k in ("predict", "prediction", "forecast")):
                return await self._call_mcp_tool_text(session, "predict_next_day", {"symbol": ticker})
            if ticker:
                return await self._call_mcp_tool_text(session, "get_market_analysis", {"symbol": ticker})
            return await self._call_mcp_tool_text(session, "get_live_news", {"query": query, "num_results": 5})

        if server == "portfolio":
            if any(k in q for k in ("allocation", "weight", "breakdown")):
                return await self._call_mcp_tool_text(session, "get_portfolio_allocation", {})
            if any(k in q for k in ("risk", "concentration", "exposure")):
                return await self._call_mcp_tool_text(session, "analyze_portfolio_risk", {})
            if ticker and any(k in q for k in ("position", "holding", "own")):
                return await self._call_mcp_tool_text(session, "check_position", {"symbol": ticker})
            return await self._call_mcp_tool_text(session, "get_portfolio_snapshot", {})

        if server == "strategy":
            if any(k in q for k in ("rebalance", "rebalancing")):
                return await self._call_mcp_tool_text(session, "portfolio_rebalancing_proposal", {"user_id": user_id})
            if any(k in q for k in ("adherence", "following strategy", "on track")):
                return await self._call_mcp_tool_text(session, "check_strategy_adherence", {"user_id": user_id})
            if ticker and any(k in q for k in ("recommend", "should i", "buy", "sell", "invest")):
                return await self._call_mcp_tool_text(
                    session,
                    "generate_investment_recommendation",
                    {"symbol": ticker, "user_id": user_id},
                )
            return await self._call_mcp_tool_text(session, "get_risk_profile", {"user_id": user_id})

        raise RuntimeError(f"No MCP tool mapping for server: {server}")

    async def _call_mcp_tool_text(self, session: ClientSession, tool_name: str, kwargs: Dict[str, Any]) -> str:
        """Call an MCP tool and normalize text payloads."""
        result = await session.call_tool(tool_name, kwargs)
        content = getattr(result, "content", None)
        if content is None:
            return "{}"
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                text = getattr(item, "text", None)
                if isinstance(text, str):
                    parts.append(text)
                else:
                    parts.append(str(item))
            return "\n".join(parts) if parts else "{}"
        return str(content)

    @staticmethod
    def _extract_response_error(response: str) -> Optional[str]:
        """Return a normalized error message when agent output signals failure."""
        if not isinstance(response, str):
            return None

        text = response.strip()
        lower = text.lower()

        if lower.startswith("error processing request"):
            return text

        if "quota exceeded" in lower:
            return "Upstream LLM quota exceeded"

        if "429" in lower and "quota" in lower:
            return "Upstream LLM quota exceeded"

        if "rate limit" in lower and "openrouter" in lower:
            return "Upstream LLM rate limit exceeded"

        return None
    
    def _determine_servers(self, query: str) -> List[str]:
        """Compatibility helper that returns only resolved server keys."""
        return self._resolve_route(query)["servers"]

    def _resolve_route(self, query: str) -> Dict[str, Any]:
        """Determine which MCP servers are needed using semantic routing first.

        Strategy:
        1) deterministic overrides for high-safety/high-precision intents
        2) LLM intent routing with confidence gating
        3) keyword-weight fallback when LLM route is unavailable/low confidence
        """
        q = (query or "").lower()

        deterministic = self._determine_servers_override(q)
        if deterministic:
            return {
                "servers": deterministic,
                "source": "deterministic_override",
                "confidence": 1.0,
                "rationale": "matched deterministic high-priority route",
                "requires_clarification": False,
            }

        if self.semantic_router_enabled:
            decision = self._classify_intent_route(query)
        else:
            decision = None

        if decision:
            candidates: List[str] = []
            candidates.extend(decision.primary_agents)

            # Medium/high confidence can include supporting agents.
            if decision.confidence >= self.semantic_router_secondary_min_confidence:
                candidates.extend(decision.secondary_agents)

            deduped = self._normalize_servers(candidates)
            if deduped and decision.confidence >= self.semantic_router_min_confidence:
                logger.info(
                    "Semantic route selected: agents=%s confidence=%.2f reason=%s",
                    deduped,
                    decision.confidence,
                    decision.rationale_short,
                )
                return {
                    "servers": deduped[:2],
                    "source": "semantic_router",
                    "confidence": decision.confidence,
                    "rationale": decision.rationale_short,
                    "requires_clarification": decision.requires_clarification,
                }

            logger.info(
                "Semantic route fallback triggered: confidence=%.2f fallback=%s",
                decision.confidence,
                decision.fallback_route,
            )

        fallback_servers = self._determine_servers_keyword_fallback(q)
        return {
            "servers": fallback_servers,
            "source": "keyword_fallback",
            "confidence": 0.0,
            "rationale": "semantic route unavailable or below confidence threshold",
            "requires_clarification": False,
        }

    def _determine_servers_override(self, q: str) -> Optional[List[str]]:
        """Deterministic rules for requests where routing must be predictable."""
        if (
            any(k in q for k in ("most promising stocks", "promising stocks", "latest market trends"))
            and any(k in q for k in ("invest", "investment", "trends", "trend"))
        ):
            return ["market", "strategy"]

        if (
            any(k in q for k in ("based on my portfolio", "custom guide", "future investments", "investment guide"))
            and any(x in q for x in ("portfolio", "porfolio"))
        ):
            return ["portfolio", "strategy"]

        if any(k in q for k in ("future investments", "investment guide", "custom guide")):
            return ["strategy"]

        if any(k in q for k in ("stock price", "price of", "current price", "quote")):
            return ["market"]

        if any(k in q for k in ("next-day prediction", "next day prediction", "prediction", "forecast", "predict")):
            return ["market"]

        if "buying power" in q and not any(x in q for x in ("buy ", "sell ", "place order", "execute")):
            return ["portfolio"]

        if (
            ("holdings table" in q or "strict holdings" in q)
            and any(x in q for x in ("quantity", "avg", "average buy", "current price", "market value"))
        ):
            return ["portfolio"]

        if (
            ("concentration" in q or "mismatch" in q)
            and ("current holdings" in q or "factual" in q or "derived from current holdings" in q)
        ):
            return ["portfolio"]

        has_portfolio_context = any(k in q for k in ("portfolio", "holding", "holdings", "position", "allocation"))
        has_benchmark_compare = any(k in q for k in ("benchmark", "moderate investor", "mismatch", "compare"))
        has_risk_context = any(k in q for k in ("risk", "profile", "concentration"))
        if has_portfolio_context and has_benchmark_compare and has_risk_context:
            return ["portfolio", "strategy"]

        return None

    def _classify_intent_route(self, query: str) -> Optional[IntentRouteDecision]:
        """Semantic router that asks the LLM for structured multi-agent routing."""
        prompt = f"""You are an intent router for a financial multi-agent system.

Available agents:
- market: predictions, quotes, market news/trends, company research
- execution: buy/sell orders and trade execution flows
- portfolio: current holdings, balances, transactions, factual account state
- strategy: investment planning, allocation guidance, rebalancing, risk posture
- general: fallback for non-financial or ambiguous requests

Classify the user's intent and return JSON only with keys:
primary_agents (array of up to 2 agents),
secondary_agents (array of up to 2 agents),
confidence (0 to 1),
requires_clarification (boolean),
fallback_route (one of market|execution|portfolio|strategy|general),
rationale_short (max 20 words).

Rules:
1) Prefer precise routing over broad routing.
2) Use execution only for explicit order intent (buy/sell/execute).
3) For portfolio-derived recommendation requests, combine portfolio + strategy.
4) For simple price/quote/prediction requests, route to market.
5) If uncertain, set low confidence and fallback_route to general.

User query: {query}
"""
        try:
            result = self.llm.invoke(prompt)
            text = str(result.content) if hasattr(result, "content") else str(result)
            raw = self._extract_json_payload(text)
            if not raw:
                return None

            data = json.loads(raw)
            primary = self._normalize_servers(data.get("primary_agents", []))
            secondary = self._normalize_servers(data.get("secondary_agents", []))
            confidence = self._coerce_confidence(data.get("confidence"))
            requires_clarification = bool(data.get("requires_clarification", False))
            fallback_route = str(data.get("fallback_route", "general")).strip().lower()
            if fallback_route not in ALLOWED_SERVER_KEYS:
                fallback_route = "general"
            rationale = str(data.get("rationale_short", "")).strip()

            if not primary:
                primary = [fallback_route]

            return IntentRouteDecision(
                primary_agents=primary,
                secondary_agents=secondary,
                confidence=confidence,
                requires_clarification=requires_clarification,
                fallback_route=fallback_route,
                rationale_short=rationale,
            )
        except Exception as exc:
            logger.warning("Semantic router unavailable, using fallback: %s", exc)
            return None

    @staticmethod
    def _coerce_confidence(value: Any) -> float:
        """Normalize LLM confidence payload to [0, 1]."""
        try:
            conf = float(value)
        except (TypeError, ValueError):
            return 0.0
        if conf < 0:
            return 0.0
        if conf > 1:
            return 1.0
        return conf

    @staticmethod
    def _extract_json_payload(text: str) -> Optional[str]:
        """Extract the first JSON object from model output."""
        if not text:
            return None
        cleaned = text.strip()
        if cleaned.startswith("{") and cleaned.endswith("}"):
            return cleaned

        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        return cleaned[start : end + 1]

    @staticmethod
    def _normalize_servers(servers: Any) -> List[str]:
        """Return valid server keys in stable order with duplicates removed."""
        if not isinstance(servers, list):
            return []
        normalized: List[str] = []
        for server in servers:
            key = str(server).strip().lower()
            if key in ALLOWED_SERVER_KEYS and key not in normalized:
                normalized.append(key)
        return normalized

    def _determine_servers_keyword_fallback(self, q: str) -> List[str]:
        """Legacy weighted keyword fallback used when semantic routing is uncertain."""
        _scores: Dict[str, float] = {"market": 0, "execution": 0, "portfolio": 0, "strategy": 0}

        market_kw = {
            "predict": 3, "prediction": 3, "forecast": 3, "lstm": 3,
            "news": 2, "headline": 2, "research": 2, "analyze stock": 2,
            "analyst": 2, "earnings": 2, "revenue": 1,
            "price target": 2, "outlook": 2, "momentum": 1,
            "stock price": 3, "price of": 2, "current price": 3, "quote": 2,
            "bull": 1, "bear": 1, "rally": 1, "crash": 1,
        }
        for kw, weight in market_kw.items():
            if kw in q:
                _scores["market"] += weight

        exec_kw = {
            "buy": 3, "sell": 3, "trade": 3, "execute": 3, "order": 3,
            "purchase": 2, "acquire": 2, "dump": 2, "liquidate": 2,
            "place order": 3, "market order": 3, "limit order": 3,
        }
        for kw, weight in exec_kw.items():
            if kw in ("buy", "sell", "trade", "order"):
                if re.search(rf"\b{re.escape(kw)}\b", q):
                    _scores["execution"] += weight
            elif kw in q:
                _scores["execution"] += weight

        portfolio_kw = {
            "portfolio": 3, "holdings": 3, "allocation": 2,
            "position": 2, "dashboard": 2, "watchlist": 2,
            "p&l": 2, "profit": 1, "loss": 1, "gain": 1,
            "diversif": 2, "concentration": 2, "weight": 1,
            "sector breakdown": 3, "performance": 2,
            "balance": 3, "cash": 3, "transactions": 3,
            "transaction": 3, "portfolio summary": 3,
            "holdings summary": 3, "account summary": 2,
        }
        for kw, weight in portfolio_kw.items():
            if kw in q:
                _scores["portfolio"] += weight

        strategy_kw = {
            "strategy": 3, "strateg": 2, "recommend": 2,
            "should i invest": 3, "should i buy": 2, "good investment": 2,
            "rebalance": 3, "rebalancing": 3, "risk tolerance": 3,
            "risk profile": 3, "allocation plan": 3,
            "benchmark": 3, "moderate investor": 3, "moderate benchmark": 3,
            "compare": 2, "mismatch": 2,
            "conservative": 2, "aggressive": 2, "moderate": 1,
            "long term": 1, "short term": 1, "time horizon": 2,
            "advice": 2, "suggest": 1, "plan": 1,
        }
        for kw, weight in strategy_kw.items():
            if kw in q:
                _scores["strategy"] += weight

        ranked = sorted(_scores.items(), key=lambda x: x[1], reverse=True)
        top_score = ranked[0][1]
        if top_score == 0:
            return ["general"]

        threshold = max(top_score * 0.5, 1)
        servers = [name for name, score in ranked if score >= threshold]
        return servers[:2]

    @staticmethod
    def _extract_ticker_from_query(query: str) -> Optional[str]:
        """Best-effort extraction of a ticker-like token from user query text."""
        if not query:
            return None

        q = query.strip()
        for pattern in _TICKER_PATTERNS:
            for match in pattern.findall(q):
                token = str(match).upper().strip()
                if 1 <= len(token) <= 5 and token not in _TICKER_COMMON_WORDS:
                    return token

        return None
    
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
6. Never mention internal systems, tool calls, memory stores, prompts, or backend process details.
7. Never mention MAFA-B or implementation plumbing in user-facing text.

User Query: {query}

Agent Responses:
"""
        for server, response in responses.items():
            synthesis_prompt += f"\n--- {server.upper()} ---\n{response}\n"
        
        synthesis_prompt += "\nProvide a unified response that integrates all insights:"
        
        try:
            result = self.llm.invoke(synthesis_prompt)
            if hasattr(result, "content"):
                return sanitize_user_response(str(result.content), user_message=query)
            return sanitize_user_response(str(result), user_message=query)
        except Exception:
            # Fallback to concatenation
            return sanitize_user_response("\n\n".join(f"**{k.title()}**: {v}" for k, v in responses.items()), user_message=query)
    
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
