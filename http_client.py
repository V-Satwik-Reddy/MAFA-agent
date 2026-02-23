"""Thread-safe HTTP client with request-scoped JWT injection, automatic retries,
connection pooling, and per-request GET caching.

Optimisations over the previous version
────────────────────────────────────────
1. **Retry strategy** – auto-retries on 502/503/504 with exponential back-off
   (3 attempts, 0.5 s base).  Prevents transient gateway errors from bubbling.
2. **Connection pooling** – 10 pools × 20 connections each via HTTPAdapter.
   Keeps TCP connections alive across tool calls within the same agent turn.
3. **Default timeout** – 15 s applied automatically when callers omit one.
4. **Request-scoped GET cache** – avoids duplicate HTTP GETs when multiple
   tools query the same MAFA-B endpoint in a single agent turn (e.g. two
   tools both calling /balance).  Enabled/cleared by API.py around each call.
"""

import contextvars
from typing import Any, Dict, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ── Retry + Connection Pooling ────────────────────────────────────────────

_retry_strategy = Retry(
    total=3,
    backoff_factor=0.5,                    # 0.5 s → 1 s → 2 s
    status_forcelist=[502, 503, 504],      # only gateway / unavailable
    allowed_methods=["GET", "POST", "PUT", "DELETE"],
    raise_on_status=False,
)

_adapter = HTTPAdapter(
    max_retries=_retry_strategy,
    pool_connections=10,
    pool_maxsize=20,
)

_session = requests.Session()
_session.headers.update({
    "Content-Type": "application/json",
    "User-Agent": "MCP-FinancialAgent/1.0",
})
_session.mount("http://", _adapter)
_session.mount("https://", _adapter)

DEFAULT_TIMEOUT = 15  # seconds – fallback when caller doesn't specify


# ── Request-scoped Auth Token ─────────────────────────────────────────────

_request_token: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "_request_token", default=None,
)


def set_request_token(token: Optional[str]) -> None:
    """Persist the user JWT for the current request context only."""
    _request_token.set(token)


def get_auth_headers(token: Optional[str] = None) -> Dict[str, str]:
    """Return merged headers with Authorization when a token is available."""
    use_token = token or _request_token.get()
    headers = dict(_session.headers)
    if use_token:
        headers["Authorization"] = (
            use_token if use_token.lower().startswith("bearer ") else f"Bearer {use_token}"
        )
    return headers


# ── Request-scoped GET Cache ──────────────────────────────────────────────
# Prevents the same GET URL being fetched twice in a single agent turn.
# Enabled by init_request_cache() and cleared by clear_request_cache().

_request_cache: contextvars.ContextVar[Optional[Dict[str, requests.Response]]] = (
    contextvars.ContextVar("_request_cache", default=None)
)


def init_request_cache() -> None:
    """Enable per-request GET caching (call before agent invocation)."""
    _request_cache.set({})


def clear_request_cache() -> None:
    """Clear and disable the GET cache (call after agent invocation)."""
    _request_cache.set(None)


# ── Internal helpers ──────────────────────────────────────────────────────

def _merge_headers(headers: Optional[Dict[str, str]]) -> Dict[str, str]:
    merged = get_auth_headers()
    if headers:
        merged.update(headers)
    return merged


def _apply_timeout(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Inject DEFAULT_TIMEOUT when the caller didn't supply one."""
    if "timeout" not in kwargs:
        kwargs["timeout"] = DEFAULT_TIMEOUT
    return kwargs


# ── Public HTTP Methods ───────────────────────────────────────────────────

def get(url: str, *, headers: Optional[Dict[str, str]] = None, **kwargs: Any):
    """GET with per-request caching, retries, and connection pooling."""
    cache = _request_cache.get()
    if cache is not None and url in cache:
        return cache[url]
    response = _session.get(url, headers=_merge_headers(headers), **_apply_timeout(kwargs))
    if cache is not None and response.ok:
        cache[url] = response
    return response


def post(url: str, *, headers: Optional[Dict[str, str]] = None, **kwargs: Any):
    return _session.post(url, headers=_merge_headers(headers), **_apply_timeout(kwargs))


def put(url: str, *, headers: Optional[Dict[str, str]] = None, **kwargs: Any):
    return _session.put(url, headers=_merge_headers(headers), **_apply_timeout(kwargs))


def delete(url: str, *, headers: Optional[Dict[str, str]] = None, **kwargs: Any):
    return _session.delete(url, headers=_merge_headers(headers), **_apply_timeout(kwargs))
