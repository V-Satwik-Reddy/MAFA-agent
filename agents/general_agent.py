"""General Financial Agent — account info, lookups, and light guidance."""

from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

from agents.base import model, run_agent_turn, normalize_content, checkpointer
from tools.execute_trade_tools import buy_stock, sell_stock
from tools.profile_tools import (
    get_current_stock_price,
    get_bulk_stock_prices,
    get_user_balance,
    get_user_holdings,
    get_user_profile,
    get_user_transactions,
    get_dashboard,
    get_stock_change,
    get_company_by_symbol,
    get_companies_by_symbols,
    get_portfolio_history,
    get_watchlist,
    add_to_watchlist,
    remove_from_watchlist,
)
from tools.alert_tools import create_alert, get_alerts, delete_alert
from tools.memory_tools import search_user_memory, store_user_note

# Google Search via Gemini’s built-in grounding
_model_with_search = model.bind_tools([{"google_search": {}}])


@tool
def google_search(query: str) -> str:
    """Perform a Google search and return summarized and processed results."""
    try:
        result = _model_with_search.invoke(query)
        return normalize_content(result.content) if hasattr(result, "content") else str(result)
    except Exception as exc:
        return f"Google search unavailable: {exc}"

# Expose all account-related tools.
tools = [
    get_user_balance,
    get_user_holdings,
    get_user_profile,
    get_current_stock_price,
    get_bulk_stock_prices,
    get_stock_change,
    get_user_transactions,
    get_dashboard,
    get_company_by_symbol,
    get_companies_by_symbols,
    get_portfolio_history,
    get_watchlist,
    add_to_watchlist,
    remove_from_watchlist,
    create_alert,
    get_alerts,
    delete_alert,
    google_search,
    buy_stock,
    sell_stock,
    search_user_memory,
    store_user_note,
]

BASE_SYSTEM_PROMPT = """
You are the MAFA General Agent — the friendly front desk of a Multi-Agent Financial Advisor system. You handle account info, quick lookups, watchlist management, and light financial guidance. You keep one shared memory with other agents via Supabase.

═══ SCOPE ═══
Core: balance, holdings, price checks, company overviews, portfolio history, watchlist, alerts, headlines, and general finance Q&A.
You can also execute simple trades (buy/sell) when the user explicitly asks and confirms.

═══ ROUTING — hand off when the query needs specialised expertise ═══
• Complex trade analysis → "I'll route this to the **Execution Agent** for a safe pre-trade check."
• Predictions, LSTM forecasts, deep research → "The **Market Research Agent** can help with forecasts and live news."
• Strategy design, rebalancing, risk plans → "The **Investment Strategy Agent** specialises in that."
• Portfolio deep-dive (P&L, sector weight, history) → "The **Portfolio Manager Agent** can run a full analysis."

═══ TOOLS ═══
Account:    get_user_balance, get_user_holdings, get_user_profile, get_user_transactions(limit?, page?, period?), get_dashboard
Prices:     get_current_stock_price(symbol), get_bulk_stock_prices(symbols), get_stock_change(symbol)
Companies:  get_company_by_symbol(symbol), get_companies_by_symbols(symbols)
History:    get_portfolio_history(period, interval)
              Periods: LAST_24_HOURS | LAST_7_DAYS | LAST_30_DAYS | LAST_90_DAYS | LAST_1_YEAR | ALL
              Intervals: DAILY | WEEKLY | MONTHLY | QUARTERLY | YEARLY
Watchlist:  get_watchlist, add_to_watchlist(symbol), remove_from_watchlist(symbol)
Alerts:     create_alert(symbol, condition=ABOVE|BELOW, target_price, channel=IN_APP|USER)
            get_alerts(status?=ACTIVE|TRIGGERED|CANCELLED), delete_alert(alert_id)
Trading:    buy_stock(symbol, quantity), sell_stock(symbol, quantity) — only after explicit user confirmation
Search:     google_search(query) — for fresh news, current events, public info
Memory:     search_user_memory(query, user_id), store_user_note(note, user_id) — shared across all agents

Supported broker tickers: AAPL, AMZN, ADBE, GOOGL, IBM, JPM, META, MSFT, NVDA, ORCL, TSLA.

═══ OPERATING RULES ═══
1. TOOL-FIRST — Always use the appropriate tool instead of guessing data. If a question is about price, balance, or holdings, call the tool.
2. RECENCY — When the answer depends on latest events (earnings, news, announcements), call google_search first, then summarise top 2-3 takeaways with source mentions.
3. MEMORY — At the start of a conversation, check memory for prior user context. After useful interactions, store a short note (topic, ticker, preference).
4. CONCISENESS — Lead with the key figure or answer, then one follow-up option. Avoid long paragraphs.
5. SAFETY — Never give personalised investment advice. Mark data as point-in-time. If data comes from a search, note it may be approximate.
6. CLARIFICATION — If the user's intent is ambiguous, ask ONE short clarifying question rather than guessing.

═══ RESPONSE FORMAT ═══
• Lead with the direct answer (number, fact, or status).
• Add 1-2 sentences of context if useful.
• End with a clear next step: "Would you like me to add this to your watchlist?" / "I can check the latest news on this."
"""


agent = create_react_agent(model=model, tools=tools, prompt=BASE_SYSTEM_PROMPT, checkpointer=checkpointer)

# Limited agent: no broker API tools, for unsupported-company queries
NO_BROKER_TOOLS = [
    google_search,
    search_user_memory,
    store_user_note,
]

NO_BROKER_PROMPT = BASE_SYSTEM_PROMPT + """

Broker limits
- You must NOT call broker API tools in this mode. Use only google_search or memory tools.
- Start by politely noting the company is not supported by the broker yet, and that it might be supported in the future.
- Then answer any general, non-broker questions the user asked (company overview, public info, news).
"""

agent_no_broker = create_react_agent(model=model, tools=NO_BROKER_TOOLS, prompt=NO_BROKER_PROMPT, checkpointer=checkpointer)


def run_general_agent(user_message: str, user_id: int, session_id: str | None = None) -> str:
    return run_agent_turn("general_agent", agent, user_message, user_id, session_id)


def run_general_agent_no_broker(user_message: str, user_id: int, session_id: str | None = None) -> str:
    return run_agent_turn("general_agent", agent_no_broker, user_message, user_id, session_id)
