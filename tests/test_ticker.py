"""
Tests for ticker extraction functionality.
"""
import pytest
from unittest.mock import patch, MagicMock


class TestExtractTicker:
    """Tests for the extract_ticker function."""

    def test_extracts_apple_ticker(self, mock_openai_llm):
        """Test extraction of Apple ticker."""
        from utils.ticker_utils import extract_ticker

        mock_llm = mock_openai_llm("AAPL")
        result = extract_ticker("How is Apple doing?", llm=mock_llm)

        assert result == "AAPL"

    def test_extracts_tesla_ticker(self, mock_openai_llm):
        """Test extraction of Tesla ticker."""
        from utils.ticker_utils import extract_ticker

        mock_llm = mock_openai_llm("TSLA")
        result = extract_ticker("Tesla stock price", llm=mock_llm)

        assert result == "TSLA"

    def test_returns_none_for_general_query(self, mock_openai_llm):
        """Test that general queries return None."""
        from utils.ticker_utils import extract_ticker

        mock_llm = mock_openai_llm("NONE")
        result = extract_ticker("What's driving tech stocks?", llm=mock_llm)

        assert result is None

    def test_extracts_index_ticker(self, mock_openai_llm):
        """Test extraction of market index ticker."""
        from utils.ticker_utils import extract_ticker

        mock_llm = mock_openai_llm("^GSPC")
        result = extract_ticker("How is the S&P 500 doing?", llm=mock_llm)

        assert result == "^GSPC"

    def test_extracts_etf_ticker(self, mock_openai_llm):
        """Test extraction of ETF ticker."""
        from utils.ticker_utils import extract_ticker

        mock_llm = mock_openai_llm("QQQ")
        result = extract_ticker("QQQ performance", llm=mock_llm)

        assert result == "QQQ"

    def test_extracts_international_adr(self, mock_openai_llm):
        """Test extraction of international ADR ticker."""
        from utils.ticker_utils import extract_ticker

        mock_llm = mock_openai_llm("SONY")
        result = extract_ticker("Sony stock price", llm=mock_llm)

        assert result == "SONY"

    def test_extracts_nintendo_adr(self, mock_openai_llm):
        """Test extraction of Nintendo ADR ticker."""
        from utils.ticker_utils import extract_ticker

        mock_llm = mock_openai_llm("NTDOY")
        result = extract_ticker("How is Nintendo doing?", llm=mock_llm)

        assert result == "NTDOY"

    def test_handles_lowercase_response(self, mock_openai_llm):
        """Test that lowercase responses are uppercased."""
        from utils.ticker_utils import extract_ticker

        mock_llm = mock_openai_llm("aapl")
        result = extract_ticker("Apple stock", llm=mock_llm)

        assert result == "AAPL"

    def test_handles_whitespace_response(self, mock_openai_llm):
        """Test that responses with whitespace are trimmed."""
        from utils.ticker_utils import extract_ticker

        mock_llm = mock_openai_llm("  MSFT  \n")
        result = extract_ticker("Microsoft stock", llm=mock_llm)

        assert result == "MSFT"

    def test_handles_none_case_insensitive(self, mock_openai_llm):
        """Test that 'none' in any case returns None."""
        from utils.ticker_utils import extract_ticker

        for none_value in ["NONE", "none", "None", "  none  "]:
            mock_llm = mock_openai_llm(none_value)
            result = extract_ticker("General market question", llm=mock_llm)
            assert result is None, f"Failed for: {none_value}"


class TestQuickTickerLookup:
    """Tests for the quick_ticker_lookup function."""

    def test_finds_apple(self):
        """Test quick lookup for Apple."""
        from utils.ticker_utils import quick_ticker_lookup

        result = quick_ticker_lookup("How is Apple doing?")
        assert result == "AAPL"

    def test_finds_microsoft(self):
        """Test quick lookup for Microsoft."""
        from utils.ticker_utils import quick_ticker_lookup

        result = quick_ticker_lookup("Microsoft stock price")
        assert result == "MSFT"

    def test_finds_google(self):
        """Test quick lookup for Google."""
        from utils.ticker_utils import quick_ticker_lookup

        result = quick_ticker_lookup("Google earnings")
        assert result == "GOOGL"

    def test_finds_sp500(self):
        """Test quick lookup for S&P 500."""
        from utils.ticker_utils import quick_ticker_lookup

        result = quick_ticker_lookup("How is the S&P 500 doing?")
        assert result == "^GSPC"

    def test_finds_dow_jones(self):
        """Test quick lookup for Dow Jones."""
        from utils.ticker_utils import quick_ticker_lookup

        result = quick_ticker_lookup("Dow Jones today")
        assert result == "^DJI"

    def test_finds_spy_etf(self):
        """Test quick lookup for SPY ETF."""
        from utils.ticker_utils import quick_ticker_lookup

        result = quick_ticker_lookup("SPY price")
        assert result == "SPY"

    def test_returns_none_for_unknown(self):
        """Test that unknown companies return None."""
        from utils.ticker_utils import quick_ticker_lookup

        result = quick_ticker_lookup("Some unknown company XYZ")
        assert result is None

    def test_case_insensitive(self):
        """Test that lookup is case insensitive."""
        from utils.ticker_utils import quick_ticker_lookup

        assert quick_ticker_lookup("APPLE stock") == "AAPL"
        assert quick_ticker_lookup("apple stock") == "AAPL"
        assert quick_ticker_lookup("Apple stock") == "AAPL"


class TestCommonTickers:
    """Tests for the COMMON_TICKERS dictionary."""

    def test_has_tech_giants(self):
        """Test that common tech giants are in the dictionary."""
        from utils.ticker_utils import COMMON_TICKERS

        assert "apple" in COMMON_TICKERS
        assert "microsoft" in COMMON_TICKERS
        assert "google" in COMMON_TICKERS
        assert "amazon" in COMMON_TICKERS
        assert "meta" in COMMON_TICKERS
        assert "tesla" in COMMON_TICKERS
        assert "nvidia" in COMMON_TICKERS

    def test_has_market_indices(self):
        """Test that market indices are in the dictionary."""
        from utils.ticker_utils import COMMON_TICKERS

        assert "s&p 500" in COMMON_TICKERS
        assert "dow jones" in COMMON_TICKERS
        assert "nasdaq" in COMMON_TICKERS

    def test_has_popular_etfs(self):
        """Test that popular ETFs are in the dictionary."""
        from utils.ticker_utils import COMMON_TICKERS

        assert "spy" in COMMON_TICKERS
        assert "qqq" in COMMON_TICKERS
        assert "voo" in COMMON_TICKERS
        assert "vti" in COMMON_TICKERS

    def test_ticker_values_are_uppercase(self):
        """Test that all ticker values are uppercase."""
        from utils.ticker_utils import COMMON_TICKERS

        for company, ticker in COMMON_TICKERS.items():
            assert ticker == ticker.upper(), f"Ticker for {company} is not uppercase: {ticker}"


class TestTickerWithFixtures:
    """Tests using pytest fixtures for common test data."""

    def test_extracts_tickers_from_fixture_queries(self, ticker_queries, mock_openai_llm):
        """Test extraction for all fixture queries."""
        from utils.ticker_utils import extract_ticker

        for query, expected_ticker in ticker_queries.items():
            mock_llm = mock_openai_llm(expected_ticker)
            result = extract_ticker(query, llm=mock_llm)
            assert result == expected_ticker, f"Failed for query: {query}"

    def test_general_queries_return_none(self, general_market_queries, mock_openai_llm):
        """Test that general market queries return None."""
        from utils.ticker_utils import extract_ticker

        mock_llm = mock_openai_llm("NONE")
        for query in general_market_queries:
            result = extract_ticker(query, llm=mock_llm)
            assert result is None, f"Expected None for query: {query}"
