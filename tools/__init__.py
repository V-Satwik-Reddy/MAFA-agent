"""Tool modules for LangGraph agents.

Modules
-------
profile_tools              – User balance, holdings, profile, preferences, transactions,
                             dashboard, stock prices (single + bulk), company lookups,
                             portfolio history, watchlist CRUD
execute_trade_tools        – buy_stock / sell_stock via MAFA-B
market_research_tools      – LSTM predictions, live news, OHLCV, companies, sectors
investment_strategy_tools  – Risk assessment, portfolio alignment, strategy generation,
                             adherence tracking (with saved-strategy + sector integration)
strategy_tools             – Strategy CRUD persistence via MAFA-B StrategyController
alert_tools                – Price alerts CRUD via MAFA-B AlertController
memory_tools               – Supabase vector-memory search & store
"""
