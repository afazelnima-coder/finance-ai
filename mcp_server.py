"""
MCP Server (stdio transport) — exposes finance assistant tools for local use.

Usage:
    python mcp_server.py

Register with Claude Code CLI:
    claude mcp add finance-assistant python /path/to/mcp_server.py
"""
import asyncio

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

from agents.market_agent import getMarketData, getMarketOverview
from agents.portfolio_agent import analyzePortfolio, lookupExpenseRatio
from utils.ticker_utils import extract_ticker

server = Server("finance-assistant")


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="get_market_data",
            description=(
                "Get real-time market data for a stock, ETF, or index. "
                "Returns current price, daily change, volume, market cap, and 52-week range."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": 'Ticker symbol, e.g. "AAPL", "SPY", "^GSPC" (S&P 500), "^DJI" (Dow Jones)',
                    }
                },
                "required": ["symbol"],
            },
        ),
        types.Tool(
            name="get_market_overview",
            description=(
                "Get a real-time snapshot of major market indices: "
                "S&P 500, Dow Jones, Nasdaq, Russell 2000, and VIX."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
        types.Tool(
            name="analyze_portfolio",
            description=(
                "Analyze an investment portfolio described in natural language. "
                "Returns asset allocation, weighted expense ratio, annual fee cost, "
                "diversification score (0-100), and risk assessment."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "portfolio_description": {
                        "type": "string",
                        "description": (
                            "Natural language description of the portfolio, e.g. "
                            '"I have $200K: 50% VOO, 30% BND, 20% cash" or '
                            '"AAPL $10K, MSFT $15K, VOO $50K"'
                        ),
                    }
                },
                "required": ["portfolio_description"],
            },
        ),
        types.Tool(
            name="lookup_expense_ratio",
            description=(
                "Look up the expense ratio for a mutual fund or ETF. "
                "Covers 50+ popular funds; falls back to web search for others."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "fund_identifier": {
                        "type": "string",
                        "description": 'Ticker symbol or fund name, e.g. "VOO", "Vanguard S&P 500", "FZROX"',
                    }
                },
                "required": ["fund_identifier"],
            },
        ),
        types.Tool(
            name="extract_ticker",
            description=(
                "Extract a stock ticker symbol from a natural language query. "
                "Handles US stocks, ADRs, and international exchanges. "
                'Returns the ticker string or "No specific ticker found".'
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": 'Natural language query mentioning a company, e.g. "What is Apple\'s price?"',
                    }
                },
                "required": ["query"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name == "get_market_data":
        result = getMarketData.func(arguments["symbol"])
    elif name == "get_market_overview":
        result = getMarketOverview.func()
    elif name == "analyze_portfolio":
        result = analyzePortfolio.func(arguments["portfolio_description"])
    elif name == "lookup_expense_ratio":
        result = lookupExpenseRatio.func(arguments["fund_identifier"])
    elif name == "extract_ticker":
        ticker = extract_ticker(arguments["query"])
        result = ticker if ticker else "No specific ticker found"
    else:
        result = f"Unknown tool: {name}"

    return [types.TextContent(type="text", text=str(result))]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
