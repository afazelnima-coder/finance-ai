"""
Cached wrappers for MCP tool functions.

TTLs by tool:
  get_market_data      60s   — price ticks frequently; 1-min staleness is acceptable
  get_market_overview  60s   — same rationale as above
  analyze_portfolio   300s   — LLM call; same description → same result
  lookup_expense_ratio 3600s — expense ratios change quarterly at most
  extract_ticker      86400s — company→ticker mapping is essentially static

Note: TTLCache is not thread-safe. This is fine for a single-worker uvicorn
process (all async I/O runs on one thread). If you add multiple uvicorn
workers, wrap cache access with a threading.Lock.
"""

import logging

from cachetools import TTLCache
from cachetools.keys import hashkey

from agents.market_agent import getMarketData, getMarketOverview
from agents.portfolio_agent import analyzePortfolio, lookupExpenseRatio
from utils.ticker_utils import extract_ticker

logger = logging.getLogger(__name__)

# ── Per-tool caches ────────────────────────────────────────────────────────

_market_data_cache:    TTLCache = TTLCache(maxsize=128, ttl=60)
_market_overview_cache: TTLCache = TTLCache(maxsize=1,   ttl=60)
_portfolio_cache:      TTLCache = TTLCache(maxsize=64,  ttl=300)
_expense_cache:        TTLCache = TTLCache(maxsize=128, ttl=3600)
_ticker_cache:         TTLCache = TTLCache(maxsize=256, ttl=86400)


# ── Internal helper ────────────────────────────────────────────────────────

def _get_or_set(cache: TTLCache, key, tool_name: str, fn):
    """Return cached value if present (HIT), otherwise call fn, store, and return (MISS)."""
    if key in cache:
        logger.info("CACHE HIT  | %s", tool_name)
        return cache[key]
    logger.info("CACHE MISS | %s", tool_name)
    result = fn()
    cache[key] = result
    return result


# ── Cached wrappers ────────────────────────────────────────────────────────

def cached_get_market_data(symbol: str) -> str:
    key = hashkey(symbol.upper())
    return _get_or_set(
        _market_data_cache, key,
        f"get_market_data({symbol.upper()})",
        lambda: getMarketData.func(symbol),
    )


def cached_get_market_overview() -> str:
    key = hashkey("overview")
    return _get_or_set(
        _market_overview_cache, key,
        "get_market_overview()",
        getMarketOverview.func,
    )


def cached_analyze_portfolio(portfolio_description: str) -> str:
    key = hashkey(portfolio_description.strip().lower())
    return _get_or_set(
        _portfolio_cache, key,
        "analyze_portfolio(...)",
        lambda: analyzePortfolio.func(portfolio_description),
    )


def cached_lookup_expense_ratio(fund_identifier: str) -> str:
    key = hashkey(fund_identifier.upper().strip())
    return _get_or_set(
        _expense_cache, key,
        f"lookup_expense_ratio({fund_identifier.upper().strip()})",
        lambda: lookupExpenseRatio.func(fund_identifier),
    )


def cached_extract_ticker(query: str) -> str | None:
    key = hashkey(query.strip().lower())
    return _get_or_set(
        _ticker_cache, key,
        f"extract_ticker({query.strip()!r})",
        lambda: extract_ticker(query),
    )


# ── Cache stats ────────────────────────────────────────────────────────────

def cache_info() -> dict:
    return {
        "get_market_data":      {"size": len(_market_data_cache),      "maxsize": _market_data_cache.maxsize,      "ttl": _market_data_cache.ttl},
        "get_market_overview":  {"size": len(_market_overview_cache),  "maxsize": _market_overview_cache.maxsize,  "ttl": _market_overview_cache.ttl},
        "analyze_portfolio":    {"size": len(_portfolio_cache),        "maxsize": _portfolio_cache.maxsize,        "ttl": _portfolio_cache.ttl},
        "lookup_expense_ratio": {"size": len(_expense_cache),          "maxsize": _expense_cache.maxsize,          "ttl": _expense_cache.ttl},
        "extract_ticker":       {"size": len(_ticker_cache),           "maxsize": _ticker_cache.maxsize,           "ttl": _ticker_cache.ttl},
    }
