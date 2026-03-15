"""
Unit tests for utils/mcp_cache.py.

Covers:
- Cache miss: underlying function is called and result stored
- Cache hit: underlying function is NOT called again within TTL
- Key normalisation: different casings/whitespace map to the same entry
- Cache isolation: each tool has its own independent cache
- cache_info(): returns correct structure and size tracking
- Logging: CACHE HIT / CACHE MISS lines are emitted
"""
import logging
import pytest
from unittest.mock import MagicMock, patch

import utils.mcp_cache as mcp_cache
from utils.mcp_cache import (
    _market_data_cache,
    _market_overview_cache,
    _portfolio_cache,
    _expense_cache,
    _ticker_cache,
    cached_get_market_data,
    cached_get_market_overview,
    cached_analyze_portfolio,
    cached_lookup_expense_ratio,
    cached_extract_ticker,
    cache_info,
)


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def clear_all_caches():
    """Clear every cache before (and after) each test for isolation."""
    _market_data_cache.clear()
    _market_overview_cache.clear()
    _portfolio_cache.clear()
    _expense_cache.clear()
    _ticker_cache.clear()
    yield
    _market_data_cache.clear()
    _market_overview_cache.clear()
    _portfolio_cache.clear()
    _expense_cache.clear()
    _ticker_cache.clear()


def _mock_tool(return_value: str) -> MagicMock:
    """Return a mock LangChain @tool object whose .func returns *return_value*."""
    tool = MagicMock()
    tool.func.return_value = return_value
    return tool


# ── Cache miss / hit ───────────────────────────────────────────────────────

class TestGetMarketDataCache:

    def test_miss_calls_underlying_function(self):
        mock_tool = _mock_tool("price data")
        with patch.object(mcp_cache, "getMarketData", mock_tool):
            result = cached_get_market_data("AAPL")
        assert result == "price data"
        mock_tool.func.assert_called_once_with("AAPL")

    def test_hit_skips_underlying_function(self):
        mock_tool = _mock_tool("price data")
        with patch.object(mcp_cache, "getMarketData", mock_tool):
            cached_get_market_data("AAPL")
            cached_get_market_data("AAPL")
        mock_tool.func.assert_called_once()  # only the first call

    def test_different_symbols_each_miss(self):
        mock_tool = _mock_tool("data")
        with patch.object(mcp_cache, "getMarketData", mock_tool):
            cached_get_market_data("AAPL")
            cached_get_market_data("MSFT")
        assert mock_tool.func.call_count == 2


class TestGetMarketOverviewCache:

    def test_miss_calls_underlying_function(self):
        mock_tool = _mock_tool("overview data")
        with patch.object(mcp_cache, "getMarketOverview", mock_tool):
            result = cached_get_market_overview()
        assert result == "overview data"
        mock_tool.func.assert_called_once()

    def test_hit_skips_underlying_function(self):
        mock_tool = _mock_tool("overview data")
        with patch.object(mcp_cache, "getMarketOverview", mock_tool):
            cached_get_market_overview()
            cached_get_market_overview()
        mock_tool.func.assert_called_once()


class TestAnalyzePortfolioCache:

    def test_miss_calls_underlying_function(self):
        mock_tool = _mock_tool("portfolio report")
        with patch.object(mcp_cache, "analyzePortfolio", mock_tool):
            result = cached_analyze_portfolio("50% VOO, 50% BND")
        assert result == "portfolio report"
        mock_tool.func.assert_called_once_with("50% VOO, 50% BND")

    def test_hit_skips_underlying_function(self):
        mock_tool = _mock_tool("portfolio report")
        with patch.object(mcp_cache, "analyzePortfolio", mock_tool):
            cached_analyze_portfolio("50% VOO, 50% BND")
            cached_analyze_portfolio("50% VOO, 50% BND")
        mock_tool.func.assert_called_once()


class TestLookupExpenseRatioCache:

    def test_miss_calls_underlying_function(self):
        mock_tool = _mock_tool("0.03% expense ratio")
        with patch.object(mcp_cache, "lookupExpenseRatio", mock_tool):
            result = cached_lookup_expense_ratio("VOO")
        assert result == "0.03% expense ratio"
        mock_tool.func.assert_called_once_with("VOO")

    def test_hit_skips_underlying_function(self):
        mock_tool = _mock_tool("0.03% expense ratio")
        with patch.object(mcp_cache, "lookupExpenseRatio", mock_tool):
            cached_lookup_expense_ratio("VOO")
            cached_lookup_expense_ratio("VOO")
        mock_tool.func.assert_called_once()


class TestExtractTickerCache:

    def test_miss_calls_underlying_function(self):
        with patch.object(mcp_cache, "extract_ticker", return_value="AAPL") as mock_fn:
            result = cached_extract_ticker("How is Apple doing?")
        assert result == "AAPL"
        mock_fn.assert_called_once_with("How is Apple doing?")

    def test_hit_skips_underlying_function(self):
        with patch.object(mcp_cache, "extract_ticker", return_value="AAPL") as mock_fn:
            cached_extract_ticker("How is Apple doing?")
            cached_extract_ticker("How is Apple doing?")
        mock_fn.assert_called_once()

    def test_returns_none_when_no_ticker(self):
        with patch.object(mcp_cache, "extract_ticker", return_value=None):
            result = cached_extract_ticker("What is the market doing?")
        assert result is None


# ── Key normalisation ──────────────────────────────────────────────────────

class TestKeyNormalisation:

    def test_market_data_symbol_is_uppercased(self):
        """'aapl' and 'AAPL' resolve to the same cache entry."""
        mock_tool = _mock_tool("data")
        with patch.object(mcp_cache, "getMarketData", mock_tool):
            cached_get_market_data("aapl")
            cached_get_market_data("AAPL")
        mock_tool.func.assert_called_once()

    def test_expense_ratio_fund_is_normalised(self):
        """'voo', 'VOO', and ' VOO ' all map to the same entry."""
        mock_tool = _mock_tool("0.03%")
        with patch.object(mcp_cache, "lookupExpenseRatio", mock_tool):
            cached_lookup_expense_ratio("voo")
            cached_lookup_expense_ratio("VOO")
            cached_lookup_expense_ratio(" VOO ")
        mock_tool.func.assert_called_once()

    def test_portfolio_description_is_lowercased_and_stripped(self):
        """Whitespace and case differences don't create duplicate entries."""
        mock_tool = _mock_tool("report")
        with patch.object(mcp_cache, "analyzePortfolio", mock_tool):
            cached_analyze_portfolio("50% VOO")
            cached_analyze_portfolio("  50% voo  ")
        mock_tool.func.assert_called_once()

    def test_extract_ticker_query_is_normalised(self):
        """Trailing spaces and case differences don't create duplicate entries."""
        with patch.object(mcp_cache, "extract_ticker", return_value="AAPL") as mock_fn:
            cached_extract_ticker("apple stock")
            cached_extract_ticker("Apple Stock ")
        mock_fn.assert_called_once()


# ── Cache isolation ────────────────────────────────────────────────────────

class TestCacheIsolation:

    def test_caches_are_independent(self):
        """A miss on one tool does not count as a miss on another."""
        mock_market = _mock_tool("market data")
        mock_overview = _mock_tool("overview")
        with patch.object(mcp_cache, "getMarketData", mock_market), \
             patch.object(mcp_cache, "getMarketOverview", mock_overview):
            cached_get_market_data("AAPL")
            cached_get_market_overview()
            # Second calls should be hits for their own caches
            cached_get_market_data("AAPL")
            cached_get_market_overview()
        mock_market.func.assert_called_once()
        mock_overview.func.assert_called_once()


# ── cache_info() ───────────────────────────────────────────────────────────

class TestCacheInfo:

    def test_returns_all_five_tools(self):
        info = cache_info()
        expected_keys = {
            "get_market_data",
            "get_market_overview",
            "analyze_portfolio",
            "lookup_expense_ratio",
            "extract_ticker",
        }
        assert set(info.keys()) == expected_keys

    def test_each_entry_has_required_fields(self):
        info = cache_info()
        for tool_name, stats in info.items():
            assert "size" in stats, f"{tool_name} missing 'size'"
            assert "maxsize" in stats, f"{tool_name} missing 'maxsize'"
            assert "ttl" in stats, f"{tool_name} missing 'ttl'"

    def test_size_is_zero_when_empty(self):
        info = cache_info()
        for stats in info.values():
            assert stats["size"] == 0

    def test_size_increments_after_cache_miss(self):
        mock_tool = _mock_tool("data")
        with patch.object(mcp_cache, "getMarketData", mock_tool):
            cached_get_market_data("AAPL")
        assert cache_info()["get_market_data"]["size"] == 1

    def test_size_unchanged_after_cache_hit(self):
        mock_tool = _mock_tool("data")
        with patch.object(mcp_cache, "getMarketData", mock_tool):
            cached_get_market_data("AAPL")
            cached_get_market_data("AAPL")  # hit — no new entry
        assert cache_info()["get_market_data"]["size"] == 1

    def test_size_increments_for_each_distinct_key(self):
        mock_tool = _mock_tool("data")
        with patch.object(mcp_cache, "getMarketData", mock_tool):
            cached_get_market_data("AAPL")
            cached_get_market_data("MSFT")
            cached_get_market_data("TSLA")
        assert cache_info()["get_market_data"]["size"] == 3

    def test_correct_ttls(self):
        info = cache_info()
        assert info["get_market_data"]["ttl"] == 60
        assert info["get_market_overview"]["ttl"] == 60
        assert info["analyze_portfolio"]["ttl"] == 300
        assert info["lookup_expense_ratio"]["ttl"] == 3600
        assert info["extract_ticker"]["ttl"] == 86400


# ── Logging ────────────────────────────────────────────────────────────────

class TestCacheLogging:

    def test_logs_cache_miss(self, caplog):
        mock_tool = _mock_tool("data")
        with patch.object(mcp_cache, "getMarketData", mock_tool):
            with caplog.at_level(logging.INFO, logger="utils.mcp_cache"):
                cached_get_market_data("AAPL")
        assert any("CACHE MISS" in r.message for r in caplog.records)

    def test_logs_cache_hit(self, caplog):
        mock_tool = _mock_tool("data")
        with patch.object(mcp_cache, "getMarketData", mock_tool):
            with caplog.at_level(logging.INFO, logger="utils.mcp_cache"):
                cached_get_market_data("AAPL")  # miss
                cached_get_market_data("AAPL")  # hit
        assert any("CACHE HIT" in r.message for r in caplog.records)

    def test_log_includes_tool_name(self, caplog):
        mock_tool = _mock_tool("data")
        with patch.object(mcp_cache, "getMarketData", mock_tool):
            with caplog.at_level(logging.INFO, logger="utils.mcp_cache"):
                cached_get_market_data("AAPL")
        assert any("get_market_data" in r.message for r in caplog.records)

    def test_miss_log_includes_symbol(self, caplog):
        mock_tool = _mock_tool("data")
        with patch.object(mcp_cache, "getMarketData", mock_tool):
            with caplog.at_level(logging.INFO, logger="utils.mcp_cache"):
                cached_get_market_data("AAPL")
        assert any("AAPL" in r.message for r in caplog.records)
