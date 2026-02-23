"""Profile & account tools – wired to real MAFA-B endpoints.

MAFA-B response shapes  (verified against actual Java DTOs / Controllers)
─────────────────────────────────────────────────────────────────────────
GET  /stockprice?symbol=X                → raw Double (NOT wrapped)
POST /bulkstockprice  body:{symbols:[]}  → ApiResponse { data: List<StockPriceDto> }
                                            StockPriceDto {symbol, close, date, open, high, low, volume}
GET  /stockchange?symbol=X               → raw StockChange {symbol, price, change, changePercent}
GET  /balance                            → ApiResponse { data: Double }
GET  /holdings                           → ApiResponse { data: List<Share> }
                                            Share {symbol, quantity, price, id}
GET  /profile/me                         → ApiResponse { data: Profile }
GET  /profile/preferences                → ApiResponse { data: PreferenceResponse }
GET  /transactions?limit=&page=&period=  → raw List<TransactionDto>
                                            TransactionDto {id, type, asset, assetQuantity, amount, createdAt}
                                            period enum: LAST_24_HOURS|LAST_7_DAYS|LAST_30_DAYS|LAST_90_DAYS|LAST_1_YEAR|ALL
GET  /dashboard                          → raw List<StockDto>
                                            StockDto {symbol, shares, totalAmount, currentPrice, avgBuyPrice, gainLoss}
GET  /companies                          → ApiResponse { data: List<CompanyDto> }
GET  /companies/{symbol}                 → ApiResponse { data: CompanyDto }  (cached)
POST /companies/by-symbols body:{symbols:[]} → ApiResponse { data: List<CompanyDto> }
GET  /sectors                            → ApiResponse { data: List<SectorDto> }
GET  /portfolio/history?period=X&interval=Y → ApiResponse { data: List<PortfolioDailySnapshotDTO> }
                                            PortfolioDailySnapshotDTO {date, totalValue, cashBalance, investedValue}
                                            period enum: LAST_24_HOURS|LAST_7_DAYS|LAST_30_DAYS|LAST_90_DAYS|LAST_1_YEAR|ALL
                                            interval enum: DAILY|WEEKLY|MONTHLY|QUARTERLY|YEARLY
GET  /watchlist                          → ApiResponse { data: List<WatchlistDto> }
                                            WatchlistDto {company: CompanyDto, addedAt}
POST /watchlist   body:{symbol:}         → ApiResponse { data: {symbol, addedAt} }
DELETE /watchlist/{symbol}               → ApiResponse { data: {symbol, removed} }
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool
from requests import RequestException

from http_client import get, post, delete

logger = logging.getLogger(__name__)

API_BASE = os.getenv("BROKER_API_URL", "http://localhost:8080")


# ── helpers ─────────────────────────────────────────────────────────────────

def _fetch_json(url: str, timeout: int = 10) -> Any:
    response = get(url, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _post_json(url: str, body: Dict[str, Any], timeout: int = 15) -> Any:
    response = post(url, json=body, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _delete_json(url: str, timeout: int = 10) -> Any:
    response = delete(url, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _unwrap(payload: Any) -> Any:
    """Extract .data from an ApiResponse wrapper when present."""
    if isinstance(payload, dict) and "data" in payload:
        return payload["data"]
    return payload


# ═══════════════════════════════════════════════════════════════════════════
#  PRICE TOOLS
# ═══════════════════════════════════════════════════════════════════════════

@tool
def get_current_stock_price(symbol: str) -> float:
    """Get the current (latest close) price for a single stock symbol.

    Calls MAFA-B  GET /stockprice?symbol=<symbol>
    Returns: raw Double
    """
    symbol = symbol.upper().strip()
    try:
        payload = _fetch_json(f"{API_BASE}/stockprice?symbol={symbol}")
        return float(payload)
    except (RequestException, ValueError, TypeError) as exc:
        logger.warning(f"Error fetching price for {symbol}: {exc}")
        return 0.0


@tool
def get_bulk_stock_prices(symbols: str) -> str:
    """Get current prices for multiple symbols in one call.

    *symbols*: comma-separated tickers, e.g. "AAPL,MSFT,GOOGL"

    Calls MAFA-B  POST /bulkstockprice
    Body: { "symbols": ["AAPL","MSFT","GOOGL"] }
    Returns: ApiResponse { data: List<StockPriceDto> }
        StockPriceDto: {symbol, close, date, open, high, low, volume}
    """
    sym_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    if not sym_list:
        return json.dumps({"error": "No symbols provided"})
    try:
        payload = _post_json(f"{API_BASE}/bulkstockprice", {"symbols": sym_list})
        data = _unwrap(payload)
        return json.dumps(data, default=str) if data else "[]"
    except (RequestException, ValueError, TypeError) as exc:
        logger.warning(f"Error fetching bulk prices: {exc}")
        return json.dumps({"error": str(exc)})


@tool
def get_stock_change(symbol: str) -> str:
    """Get price-change info for a symbol (price, change, changePercent).

    Calls MAFA-B  GET /stockchange?symbol=<symbol>
    Returns: raw StockChange {symbol, price, change, changePercent}
    """
    symbol = symbol.upper().strip()
    try:
        payload = _fetch_json(f"{API_BASE}/stockchange?symbol={symbol}")
        return json.dumps(payload)
    except (RequestException, ValueError, TypeError) as exc:
        logger.warning(f"Error fetching stock change for {symbol}: {exc}")
        return json.dumps({"symbol": symbol, "error": str(exc)})


# ═══════════════════════════════════════════════════════════════════════════
#  USER ACCOUNT TOOLS
# ═══════════════════════════════════════════════════════════════════════════

@tool
def get_user_balance() -> float:
    """Fetch the user's current cash balance.

    Calls MAFA-B  GET /balance → ApiResponse { data: Double }
    """
    try:
        payload = _fetch_json(f"{API_BASE}/balance")
        balance = _unwrap(payload)
        return float(balance) if balance is not None else 0.0
    except (RequestException, ValueError, TypeError) as exc:
        logger.warning(f"Error fetching user balance: {exc}")
        return 0.0


@tool
def get_user_holdings() -> str:
    """Fetch the user's current stock holdings.

    Calls MAFA-B  GET /holdings
    Returns JSON: List<Share> [{symbol, quantity, price, id}, ...]
    """
    try:
        payload = _fetch_json(f"{API_BASE}/holdings")
        holdings = _unwrap(payload)
        return json.dumps(holdings) if isinstance(holdings, list) else json.dumps([])
    except (RequestException, ValueError, TypeError) as exc:
        logger.warning(f"Error fetching user holdings: {exc}")
        return json.dumps([])


@tool
def get_user_profile() -> str:
    """Fetch the user's profile (name, balance, job, salary range, etc.).

    Calls MAFA-B  GET /profile/me → ApiResponse { data: Profile }
    """
    try:
        payload = _fetch_json(f"{API_BASE}/profile/me")
        profile = _unwrap(payload)
        return json.dumps(profile, default=str) if profile else "{}"
    except (RequestException, ValueError, TypeError) as exc:
        logger.warning(f"Error fetching user profile: {exc}")
        return "{}"


@tool
def get_user_preferences() -> str:
    """Fetch the user's investment preferences (risk tolerance, goals, sectors, companies).

    Calls MAFA-B  GET /profile/preferences
    Returns JSON: PreferenceResponse {investmentGoals, riskTolerance, preferredAsset,
                                      sectorIds: List<SectorDto>, companyIds: List<CompanyDto>}
    """
    try:
        payload = _fetch_json(f"{API_BASE}/profile/preferences")
        prefs = _unwrap(payload)
        return json.dumps(prefs, default=str) if prefs else "{}"
    except (RequestException, ValueError, TypeError) as exc:
        logger.warning(f"Error fetching user preferences: {exc}")
        return "{}"


@tool
def get_user_transactions(limit: Optional[int] = None, page: Optional[int] = None, period: Optional[str] = None) -> str:
    """Fetch the user's transaction history with optional filtering.

    Params:
        limit:  page size (optional)
        page:   1-based page number (optional)
        period: time-range filter (optional)
                Allowed values: LAST_24_HOURS | LAST_7_DAYS | LAST_30_DAYS |
                                LAST_90_DAYS | LAST_1_YEAR | ALL

    Calls MAFA-B  GET /transactions?limit=&page=&period=
    Returns JSON: List<TransactionDto> [{id, type, asset, assetQuantity, amount, createdAt}, ...]
    """
    try:
        params = []
        if limit:
            params.append(f"limit={limit}")
        if page:
            params.append(f"page={page}")
        if period:
            params.append(f"period={period.upper()}")
        qs = f"?{'&'.join(params)}" if params else ""
        payload = _fetch_json(f"{API_BASE}/transactions{qs}")
        if isinstance(payload, list):
            return json.dumps(payload, default=str)
        return json.dumps([])
    except (RequestException, ValueError, TypeError) as exc:
        logger.warning(f"Error fetching transactions: {exc}")
        return json.dumps([])


@tool
def get_dashboard() -> str:
    """Fetch portfolio dashboard with per-stock P&L details.

    Calls MAFA-B  GET /dashboard
    Returns JSON: List<StockDto> [{symbol, shares, totalAmount, currentPrice, avgBuyPrice, gainLoss}, ...]
    """
    try:
        payload = _fetch_json(f"{API_BASE}/dashboard")
        return json.dumps(payload) if isinstance(payload, list) else json.dumps([])
    except (RequestException, ValueError, TypeError) as exc:
        logger.warning(f"Error fetching dashboard: {exc}")
        return json.dumps([])


# ═══════════════════════════════════════════════════════════════════════════
#  COMPANY & SECTOR TOOLS
# ═══════════════════════════════════════════════════════════════════════════

@tool
def get_company_by_symbol(symbol: str) -> str:
    """Look up a single company by ticker symbol (includes sector info).

    Calls MAFA-B  GET /companies/{symbol}  (cached)
    Returns JSON: CompanyDto {id, symbol, name, sector: {id, name}}
    """
    symbol = symbol.upper().strip()
    try:
        payload = _fetch_json(f"{API_BASE}/companies/{symbol}")
        data = _unwrap(payload)
        return json.dumps(data) if data else "{}"
    except (RequestException, ValueError, TypeError) as exc:
        logger.warning(f"Error fetching company {symbol}: {exc}")
        return "{}"


@tool
def get_companies_by_symbols(symbols: str) -> str:
    """Look up multiple companies by ticker (bulk). Returns sector info for each.

    *symbols*: comma-separated tickers, e.g. "AAPL,MSFT,JPM"

    Calls MAFA-B  POST /companies/by-symbols
    Body: { "symbols": ["AAPL","MSFT","JPM"] }
    Returns JSON: List<CompanyDto> [{id, symbol, name, sector: {id, name}}, ...]
    """
    sym_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    if not sym_list:
        return json.dumps([])
    try:
        payload = _post_json(f"{API_BASE}/companies/by-symbols", {"symbols": sym_list})
        data = _unwrap(payload)
        return json.dumps(data) if isinstance(data, list) else json.dumps([])
    except (RequestException, ValueError, TypeError) as exc:
        logger.warning(f"Error fetching companies bulk: {exc}")
        return json.dumps([])


# ═══════════════════════════════════════════════════════════════════════════
#  PORTFOLIO HISTORY
# ═══════════════════════════════════════════════════════════════════════════

@tool
def get_portfolio_history(period: str = "LAST_30_DAYS", interval: str = "DAILY") -> str:
    """Fetch portfolio value history over time.

    Params:
        period:   LAST_24_HOURS | LAST_7_DAYS | LAST_30_DAYS | LAST_90_DAYS | LAST_1_YEAR | ALL
        interval: DAILY | WEEKLY | MONTHLY | QUARTERLY | YEARLY  (default DAILY)

    Calls MAFA-B  GET /portfolio/history?period=X&interval=Y
    Returns JSON: List<PortfolioDailySnapshotDTO> [{date, totalValue, cashBalance, investedValue}, ...]
    """
    try:
        url = f"{API_BASE}/portfolio/history?period={period.upper()}"
        if interval:
            url += f"&interval={interval.upper()}"
        payload = _fetch_json(url)
        data = _unwrap(payload)
        return json.dumps(data, default=str) if isinstance(data, list) else json.dumps([])
    except (RequestException, ValueError, TypeError) as exc:
        logger.warning(f"Error fetching portfolio history: {exc}")
        return json.dumps([])


# ═══════════════════════════════════════════════════════════════════════════
#  WATCHLIST TOOLS
# ═══════════════════════════════════════════════════════════════════════════

@tool
def get_watchlist() -> str:
    """Fetch the user's stock watchlist.

    Calls MAFA-B  GET /watchlist
    Returns JSON: List<WatchlistDto> [{company: {id, symbol, name, sector}, addedAt}, ...]
    """
    try:
        payload = _fetch_json(f"{API_BASE}/watchlist")
        data = _unwrap(payload)
        return json.dumps(data, default=str) if isinstance(data, list) else json.dumps([])
    except (RequestException, ValueError, TypeError) as exc:
        logger.warning(f"Error fetching watchlist: {exc}")
        return json.dumps([])


@tool
def add_to_watchlist(symbol: str) -> str:
    """Add a stock to the user's watchlist.

    Calls MAFA-B  POST /watchlist  body: { "symbol": "<TICKER>" }
    Returns JSON confirmation or error if already exists (409 Conflict).
    """
    symbol = symbol.upper().strip()
    try:
        payload = _post_json(f"{API_BASE}/watchlist", {"symbol": symbol})
        data = _unwrap(payload)
        return json.dumps(data, default=str) if data else json.dumps({"symbol": symbol, "added": True})
    except RequestException as exc:
        if hasattr(exc, 'response') and exc.response is not None and exc.response.status_code == 409:
            return json.dumps({"symbol": symbol, "error": "Already in watchlist"})
        logger.warning(f"Error adding {symbol} to watchlist: {exc}")
        return json.dumps({"error": str(exc)})


@tool
def remove_from_watchlist(symbol: str) -> str:
    """Remove a stock from the user's watchlist.

    Calls MAFA-B  DELETE /watchlist/{symbol}
    Returns confirmation or error if not found (404).
    """
    symbol = symbol.upper().strip()
    try:
        payload = _delete_json(f"{API_BASE}/watchlist/{symbol}")
        data = _unwrap(payload)
        return json.dumps(data) if data else json.dumps({"symbol": symbol, "removed": True})
    except RequestException as exc:
        if hasattr(exc, 'response') and exc.response is not None and exc.response.status_code == 404:
            return json.dumps({"symbol": symbol, "error": "Not in watchlist"})
        logger.warning(f"Error removing {symbol} from watchlist: {exc}")
        return json.dumps({"error": str(exc)})
