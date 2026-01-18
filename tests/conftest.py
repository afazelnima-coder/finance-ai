"""
Shared pytest fixtures for Finance Assistant tests.
"""
import os
import sys
import pytest
from unittest.mock import MagicMock, patch

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def mock_llm_response():
    """Factory fixture to create mock LLM responses."""
    def _create_response(content: str):
        mock_response = MagicMock()
        mock_response.content = content
        return mock_response
    return _create_response


@pytest.fixture
def mock_openai_llm(mock_llm_response):
    """Mock ChatOpenAI that returns configurable responses."""
    def _create_mock(response_content: str):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_llm_response(response_content)
        return mock_llm
    return _create_mock


@pytest.fixture
def finance_queries():
    """Sample finance-related queries for testing."""
    return [
        "What is a stock?",
        "How do I invest in ETFs?",
        "Explain compound interest",
        "What are the tax implications of selling stocks?",
        "How is Apple stock doing today?",
        "What's the current S&P 500 price?",
        "Explain dollar cost averaging",
        "What is a 401k?",
        "How do dividends work?",
        "What is inflation?",
    ]


@pytest.fixture
def non_finance_queries():
    """Sample non-finance queries for testing guardrails."""
    return [
        "What's the best pizza recipe?",
        "How do I train for a marathon?",
        "What's the capital of France?",
        "How do I fix a leaky faucet?",
        "What's the best movie of 2024?",
        "How do I bake a chocolate cake?",
        "What's the weather like today?",
        "How do I learn guitar?",
        "What's the fastest animal?",
        "How do I meditate?",
    ]


@pytest.fixture
def ticker_queries():
    """Sample queries with company names for ticker extraction."""
    return {
        "How is Apple doing?": "AAPL",
        "Tesla stock price": "TSLA",
        "Show me Microsoft chart": "MSFT",
        "Amazon earnings": "AMZN",
        "What's happening with Meta?": "META",
        "Google stock": "GOOGL",
        "S&P 500 performance": "^GSPC",
        "Dow Jones today": "^DJI",
        "QQQ ETF": "QQQ",
        "Sony stock": "SONY",
    }


@pytest.fixture
def general_market_queries():
    """Queries without specific tickers (should return NONE)."""
    return [
        "What's driving tech stocks?",
        "Is the market up today?",
        "Market trends this week",
        "Should I invest now?",
        "What sectors are performing well?",
    ]
