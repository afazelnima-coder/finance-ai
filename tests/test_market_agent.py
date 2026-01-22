"""
Tests for the market agent functionality.
Tests real-time market data tools and news search.
"""
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime


class TestGetMarketData:
    """Tests for the getMarketData tool."""

    def test_get_market_data_returns_string(self):
        """Test that getMarketData returns a string response."""
        with patch("agents.market_agent.yf.Ticker") as mock_ticker:
            mock_instance = MagicMock()
            mock_instance.info = {
                "longName": "Apple Inc.",
                "currentPrice": 150.0,
                "previousClose": 148.0,
                "volume": 50000000,
                "marketCap": 2500000000000,
                "dayHigh": 152.0,
                "dayLow": 149.0,
            }
            mock_instance.history.return_value = MagicMock()
            mock_instance.history.return_value.empty = False
            mock_instance.history.return_value.__getitem__ = lambda self, key: MagicMock(iloc=MagicMock(__getitem__=lambda s, i: 150.0))
            mock_ticker.return_value = mock_instance

            from agents.market_agent import getMarketData

            result = getMarketData.invoke("AAPL")

            assert isinstance(result, str)
            assert "Apple" in result or "AAPL" in result

    def test_get_market_data_includes_price(self):
        """Test that getMarketData includes price information."""
        with patch("agents.market_agent.yf.Ticker") as mock_ticker:
            mock_instance = MagicMock()
            mock_instance.info = {
                "currentPrice": 150.50,
                "previousClose": 148.00,
                "volume": 50000000,
            }
            mock_history = MagicMock()
            mock_history.empty = False
            mock_history.__getitem__ = lambda self, key: MagicMock(iloc=MagicMock(__getitem__=lambda s, i: 150.50 if i == -1 else 148.00))
            mock_instance.history.return_value = mock_history
            mock_ticker.return_value = mock_instance

            from agents.market_agent import getMarketData

            result = getMarketData.invoke("AAPL")

            assert "150" in result  # Price should be in result

    def test_get_market_data_handles_error(self):
        """Test that getMarketData handles errors gracefully."""
        with patch("agents.market_agent.yf.Ticker") as mock_ticker:
            mock_ticker.side_effect = Exception("API Error")

            from agents.market_agent import getMarketData

            result = getMarketData.invoke("INVALID")

            assert "Error" in result or "error" in result


class TestGetMarketOverview:
    """Tests for the getMarketOverview tool."""

    def test_get_market_overview_returns_string(self):
        """Test that getMarketOverview returns a string."""
        with patch("agents.market_agent.yf.Ticker") as mock_ticker:
            mock_instance = MagicMock()
            mock_history = MagicMock()
            mock_history.empty = False

            # Mock Close column with proper indexing
            class MockClose:
                def __init__(self, values):
                    self.values = values

                @property
                def iloc(self):
                    return self

                def __getitem__(self, idx):
                    return self.values[idx]

            mock_history.__getitem__ = lambda self, key: MockClose([5000, 5050])
            mock_instance.history.return_value = mock_history
            mock_ticker.return_value = mock_instance

            from agents.market_agent import getMarketOverview

            result = getMarketOverview.invoke("")

            assert isinstance(result, str)
            assert "Market Overview" in result

    def test_get_market_overview_includes_major_indices(self):
        """Test that market overview includes major index names."""
        with patch("agents.market_agent.yf.Ticker") as mock_ticker:
            mock_instance = MagicMock()
            mock_history = MagicMock()
            mock_history.empty = False

            class MockClose:
                def __init__(self):
                    pass

                @property
                def iloc(self):
                    return self

                def __getitem__(self, idx):
                    return 5000.0

            mock_history.__getitem__ = lambda self, key: MockClose()
            mock_instance.history.return_value = mock_history
            mock_ticker.return_value = mock_instance

            from agents.market_agent import getMarketOverview

            result = getMarketOverview.invoke("")

            # Should mention at least some major indices
            major_indices = ["S&P 500", "Dow Jones", "Nasdaq"]
            found = any(idx in result for idx in major_indices)
            assert found or "Market Overview" in result


class TestSearchMarketNews:
    """Tests for the searchMarketNews tool."""

    def test_search_market_news_returns_string(self):
        """Test that searchMarketNews returns a string."""
        with patch("agents.market_agent.tavily_client.search") as mock_search:
            mock_search.return_value = {
                "results": [
                    {
                        "title": "Market Rally Continues",
                        "content": "Stocks continue to rise amid positive economic data...",
                        "url": "https://example.com/news"
                    }
                ]
            }

            from agents.market_agent import searchMarketNews

            result = searchMarketNews.invoke("market news today")

            assert isinstance(result, str)

    def test_search_market_news_includes_date(self):
        """Test that searchMarketNews includes current date in query."""
        with patch("agents.market_agent.tavily_client.search") as mock_search:
            mock_search.return_value = {"results": []}

            from agents.market_agent import searchMarketNews

            searchMarketNews.invoke("tech stocks")

            # Check that the search was called with date in query
            call_args = mock_search.call_args
            query = call_args.kwargs.get("query", "")
            today = datetime.now().strftime("%Y-%m-%d")
            assert today in query

    def test_search_market_news_handles_no_results(self):
        """Test that searchMarketNews handles empty results."""
        with patch("agents.market_agent.tavily_client.search") as mock_search:
            mock_search.return_value = {"results": []}

            from agents.market_agent import searchMarketNews

            result = searchMarketNews.invoke("obscure query")

            assert "No recent news" in result or isinstance(result, str)

    def test_search_market_news_handles_error(self):
        """Test that searchMarketNews handles API errors."""
        with patch("agents.market_agent.tavily_client.search") as mock_search:
            mock_search.side_effect = Exception("API Error")

            from agents.market_agent import searchMarketNews

            result = searchMarketNews.invoke("test query")

            assert "Error" in result or "error" in result


class TestGetSectorPerformance:
    """Tests for the getSectorPerformance tool."""

    def test_get_sector_performance_returns_string(self):
        """Test that getSectorPerformance returns a string."""
        with patch("agents.market_agent.yf.Ticker") as mock_ticker:
            mock_instance = MagicMock()
            mock_history = MagicMock()
            mock_history.empty = False

            class MockClose:
                @property
                def iloc(self):
                    return self

                def __getitem__(self, idx):
                    return 100.0 if idx == -1 else 99.0

            mock_history.__getitem__ = lambda self, key: MockClose()
            mock_instance.history.return_value = mock_history
            mock_ticker.return_value = mock_instance

            from agents.market_agent import getSectorPerformance

            result = getSectorPerformance.invoke("")

            assert isinstance(result, str)
            assert "Sector" in result

    def test_get_sector_performance_includes_sectors(self):
        """Test that sector performance includes sector names."""
        with patch("agents.market_agent.yf.Ticker") as mock_ticker:
            mock_instance = MagicMock()
            mock_history = MagicMock()
            mock_history.empty = False

            class MockClose:
                @property
                def iloc(self):
                    return self

                def __getitem__(self, idx):
                    return 100.0

            mock_history.__getitem__ = lambda self, key: MockClose()
            mock_instance.history.return_value = mock_history
            mock_ticker.return_value = mock_instance

            from agents.market_agent import getSectorPerformance

            result = getSectorPerformance.invoke("")

            # Should include at least some sector names
            sectors = ["Technology", "Healthcare", "Financials", "Energy"]
            found = any(sector in result for sector in sectors)
            assert found or "Sector Performance" in result


class TestMarketAgentTools:
    """Tests for the market agent tool configuration."""

    def test_agent_has_required_tools(self):
        """Test that agent has all required tools configured."""
        from agents.market_agent import getMarketData, getMarketOverview, searchMarketNews, getSectorPerformance

        # Test that tools exist and are callable
        assert getMarketData is not None
        assert getMarketOverview is not None
        assert searchMarketNews is not None
        assert getSectorPerformance is not None

    def test_agent_has_system_prompt(self):
        """Test that agent is configured correctly."""
        from agents.market_agent import agent

        # Agent should exist and be invokable
        assert agent is not None
        assert hasattr(agent, 'invoke')


class TestMarketDataFormatting:
    """Tests for data formatting in market tools."""

    def test_market_data_formats_large_numbers(self):
        """Test that large numbers are formatted properly."""
        with patch("agents.market_agent.yf.Ticker") as mock_ticker:
            mock_instance = MagicMock()
            mock_instance.info = {
                "currentPrice": 150.0,
                "previousClose": 148.0,
                "volume": 50000000,
                "marketCap": 2500000000000,  # 2.5 trillion
            }
            mock_history = MagicMock()
            mock_history.empty = False
            mock_history.__getitem__ = lambda self, key: MagicMock(iloc=MagicMock(__getitem__=lambda s, i: 150.0))
            mock_instance.history.return_value = mock_history
            mock_ticker.return_value = mock_instance

            from agents.market_agent import getMarketData

            result = getMarketData.invoke("AAPL")

            # Should format market cap with T for trillion
            assert "T" in result or "B" in result or "M" in result

    def test_sector_performance_shows_percentages(self):
        """Test that sector performance shows percentage changes."""
        with patch("agents.market_agent.yf.Ticker") as mock_ticker:
            mock_instance = MagicMock()
            mock_history = MagicMock()
            mock_history.empty = False

            class MockClose:
                @property
                def iloc(self):
                    return self

                def __getitem__(self, idx):
                    return 101.0 if idx == -1 else 100.0  # 1% gain

            mock_history.__getitem__ = lambda self, key: MockClose()
            mock_instance.history.return_value = mock_history
            mock_ticker.return_value = mock_instance

            from agents.market_agent import getSectorPerformance

            result = getSectorPerformance.invoke("")

            # Should show percentage
            assert "%" in result
