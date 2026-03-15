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

from cachetools import TTLCache, cached
from cachetools.keys import hashkey

from agents.market_agent import getMarketData, getMarketOverview
from agents.portfolio_agent import analyzePortfolio, lookupExpenseRatio
from utils.ticker_utils import extract_ticker

# ── Per-tool caches ────────────────────────────────────────────────────────

_market_data_cache: TTLCache = TTLCache(maxsize=128, ttl=60)
_market_overview_cache: TTLCache = TTLCache(maxsize=1, ttl=60)
_portfolio_cache: TTLCache = TTLCache(maxsize=64, ttl=300)
_expense_cache: TTLCache = TTLCache(maxsize=128, ttl=3600)
_ticker_cache: TTLCache = TTLCache(maxsize=256, ttl=86400)


# ── Cached wrappers ────────────────────────────────────────────────────────

@cached(_market_data_cache, key=lambda symbol: hashkey(symbol.upper()))
def cached_get_market_data(symbol: str) -> str:
    return getMarketData.func(symbol)


@cached(_market_overview_cache, key=lambda: hashkey("overview"))
def cached_get_market_overview() -> str:
    return getMarketOverview.func()


@cached(_portfolio_cache, key=lambda desc: hashkey(desc.strip().lower()))
def cached_analyze_portfolio(portfolio_description: str) -> str:
    return analyzePortfolio.func(portfolio_description)


@cached(_expense_cache, key=lambda fund: hashkey(fund.upper().strip()))
def cached_lookup_expense_ratio(fund_identifier: str) -> str:
    return lookupExpenseRatio.func(fund_identifier)


@cached(_ticker_cache, key=lambda query: hashkey(query.strip().lower()))
def cached_extract_ticker(query: str) -> str | None:
    return extract_ticker(query)


# ── Cache stats (useful for debugging / monitoring) ────────────────────────

def cache_info() -> dict:
    return {
        "get_market_data":      {"size": len(_market_data_cache),     "maxsize": _market_data_cache.maxsize,     "ttl": _market_data_cache.ttl},
        "get_market_overview":  {"size": len(_market_overview_cache),  "maxsize": _market_overview_cache.maxsize,  "ttl": _market_overview_cache.ttl},
        "analyze_portfolio":    {"size": len(_portfolio_cache),        "maxsize": _portfolio_cache.maxsize,        "ttl": _portfolio_cache.ttl},
        "lookup_expense_ratio": {"size": len(_expense_cache),          "maxsize": _expense_cache.maxsize,          "ttl": _expense_cache.ttl},
        "extract_ticker":       {"size": len(_ticker_cache),           "maxsize": _ticker_cache.maxsize,           "ttl": _ticker_cache.ttl},
    }
