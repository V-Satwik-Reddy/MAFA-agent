"""Trade-execution tools – wired to real MAFA-B endpoints.

MAFA-B contract
────────────────
POST /execute/buy   body: { "quantity": long, "symbol": String }
POST /execute/sell  body: { "quantity": long, "symbol": String }

Both return  TransactionDto { id, type, asset, assetQuantity, amount, createdAt }
"""

import json
import logging
import os
from typing import Any, Dict

from langchain_core.tools import tool
from requests import RequestException

from http_client import post

logger = logging.getLogger(__name__)

API_BASE = os.getenv("BROKER_API_URL", "http://localhost:8080")


# ── helpers ─────────────────────────────────────────────────────────────────

def _post_json(url: str, body: Dict[str, Any], timeout: int = 30) -> Any:
    """POST JSON to MAFA-B and return parsed response."""
    response = post(url, json=body, timeout=timeout)
    response.raise_for_status()
    return response.json()


# ── tools ───────────────────────────────────────────────────────────────────

@tool
def buy_stock(symbol: str, quantity: int) -> str:
    """Buy shares of a stock.

    Calls MAFA-B  POST /execute/buy
    Body: { "quantity": <int>, "symbol": "<TICKER>" }
    Returns a TransactionDto JSON string on success.
    """
    symbol = symbol.upper().strip()
    if quantity <= 0:
        return json.dumps({"error": "Quantity must be a positive integer."})
    try:
        body = {"quantity": quantity, "symbol": symbol}
        result = _post_json(f"{API_BASE}/execute/buy", body)
        return json.dumps(result)
    except RequestException as exc:
        logger.warning(f"Buy order failed for {quantity}x {symbol}: {exc}")
        return json.dumps({"error": f"Buy order failed: {exc}"})


@tool
def sell_stock(symbol: str, quantity: int) -> str:
    """Sell shares of a stock.

    Calls MAFA-B  POST /execute/sell
    Body: { "quantity": <int>, "symbol": "<TICKER>" }
    Returns a TransactionDto JSON string on success.
    """
    symbol = symbol.upper().strip()
    if quantity <= 0:
        return json.dumps({"error": "Quantity must be a positive integer."})
    try:
        body = {"quantity": quantity, "symbol": symbol}
        result = _post_json(f"{API_BASE}/execute/sell", body)
        return json.dumps(result)
    except RequestException as exc:
        logger.warning(f"Sell order failed for {quantity}x {symbol}: {exc}")
        return json.dumps({"error": f"Sell order failed: {exc}"})
