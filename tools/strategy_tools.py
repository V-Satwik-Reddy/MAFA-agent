"""Strategy persistence tools – wired to MAFA-B StrategyController.

MAFA-B contract
────────────────
GET  /strategy            → ApiResponse {data: StrategyDto | null}  (active strategy; 404 if none)
GET  /strategy/history    → ApiResponse {data: List<StrategyDto>}
POST /strategy            body: StrategyRequestDto → ApiResponse {data: StrategyDto}
PUT  /strategy/{id}       body: StrategyRequestDto → ApiResponse {data: StrategyDto}

StrategyRequestDto:
    strategyType: String, goal: String, timeHorizonMonths: Integer,
    riskProfile: enum (CONSERVATIVE, MODERATE, AGGRESSIVE),
    targetAllocation: Map<String, Integer>,  sectorLimits: Map<String, Integer>,
    rebalancingFrequency: enum (MONTHLY, QUARTERLY, ANNUALLY, NONE)

StrategyDto:
    id, strategyType, goal, timeHorizonMonths, riskProfile,
    targetAllocation, sectorLimits, rebalancingFrequency,
    active (boolean), createdAt, updatedAt
"""

import json
import logging
import os
from typing import Any, Dict, Optional

from langchain_core.tools import tool
from requests import RequestException

from http_client import get, post, put

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


def _put_json(url: str, body: Dict[str, Any], timeout: int = 15) -> Any:
    response = put(url, json=body, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _unwrap(payload: Any) -> Any:
    if isinstance(payload, dict) and "data" in payload:
        return payload["data"]
    return payload


# ── tools ───────────────────────────────────────────────────────────────────

@tool
def get_active_strategy() -> str:
    """Fetch the user's currently active investment strategy.

    Calls MAFA-B  GET /strategy
    Returns JSON: StrategyDto or {} if none saved.
    """
    try:
        payload = _fetch_json(f"{API_BASE}/strategy")
        data = _unwrap(payload)
        return json.dumps(data) if data else "{}"
    except RequestException as exc:
        # 404 means no active strategy — that's fine
        if hasattr(exc, 'response') and exc.response is not None and exc.response.status_code == 404:
            return "{}"
        logger.warning(f"Error fetching active strategy: {exc}")
        return "{}"


@tool
def get_strategy_history() -> str:
    """Fetch all of the user's previously saved strategies.

    Calls MAFA-B  GET /strategy/history
    Returns JSON: List<StrategyDto>
    """
    try:
        payload = _fetch_json(f"{API_BASE}/strategy/history")
        data = _unwrap(payload)
        return json.dumps(data) if isinstance(data, list) else json.dumps([])
    except (RequestException, ValueError, TypeError) as exc:
        logger.warning(f"Error fetching strategy history: {exc}")
        return json.dumps([])


@tool
def save_strategy(strategy_json: str) -> str:
    """Save a new investment strategy (marks any previous strategy as inactive).

    *strategy_json* must be a JSON string with these fields:
        strategyType:         e.g. "moderate_growth", "aggressive_growth"
        goal:                 e.g. "Long-term wealth growth"
        timeHorizonMonths:    e.g. 60
        riskProfile:          "CONSERVATIVE" | "MODERATE" | "AGGRESSIVE"
        targetAllocation:     e.g. {"equity": 70, "debt": 20, "cash": 10}
        sectorLimits:         e.g. {"Technology": 35, "Financials": 20}  (optional)
        rebalancingFrequency: "MONTHLY" | "QUARTERLY" | "ANNUALLY" | "NONE"

    Calls MAFA-B  POST /strategy
    Returns JSON: the saved StrategyDto (with id, timestamps, active=true).
    """
    try:
        body = json.loads(strategy_json)
    except json.JSONDecodeError:
        return json.dumps({"error": "Invalid JSON for strategy"})

    try:
        payload = _post_json(f"{API_BASE}/strategy", body)
        data = _unwrap(payload)
        return json.dumps(data) if data else json.dumps({"error": "Save failed"})
    except RequestException as exc:
        logger.warning(f"Error saving strategy: {exc}")
        return json.dumps({"error": str(exc)})


@tool
def update_strategy(strategy_id: int, updates_json: str) -> str:
    """Partially update an existing strategy by ID.

    *updates_json*: JSON string with any subset of StrategyRequest fields to update.
    Fields set to null or omitted will not be changed.

    Calls MAFA-B  PUT /strategy/{id}
    Returns JSON: the updated StrategyDto.
    """
    try:
        body = json.loads(updates_json)
    except json.JSONDecodeError:
        return json.dumps({"error": "Invalid JSON for strategy update"})

    try:
        payload = _put_json(f"{API_BASE}/strategy/{strategy_id}", body)
        data = _unwrap(payload)
        return json.dumps(data) if data else json.dumps({"error": "Update failed"})
    except RequestException as exc:
        logger.warning(f"Error updating strategy {strategy_id}: {exc}")
        return json.dumps({"error": str(exc)})
