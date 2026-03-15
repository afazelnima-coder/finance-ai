"""
MCP Server (SSE/HTTP transport) — exposes finance assistant tools for remote access.

Usage:
    uvicorn mcp_http_server:app --host 0.0.0.0 --port 8000

Register with Claude Code CLI:
    claude mcp add --transport sse finance-assistant http://localhost:8000/sse
"""
import asyncio

from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp import types

from utils.mcp_cache import (
    cached_get_market_data,
    cached_get_market_overview,
    cached_analyze_portfolio,
    cached_lookup_expense_ratio,
    cached_extract_ticker,
)

# ── MCP server definition ──────────────────────────────────────────────────

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
                        "description": 'Ticker symbol, e.g. "AAPL", "SPY", "^GSPC" (S&P 500)',
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
                            "Natural language description, e.g. "
                            '"I have $200K: 50% VOO, 30% BND, 20% cash"'
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
                        "description": 'Ticker symbol or fund name, e.g. "VOO", "Vanguard S&P 500"',
                    }
                },
                "required": ["fund_identifier"],
            },
        ),
        types.Tool(
            name="extract_ticker",
            description=(
                "Extract a stock ticker symbol from a natural language query. "
                "Handles US stocks, ADRs, and international exchanges."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": 'Query mentioning a company, e.g. "What is Apple\'s price?"',
                    }
                },
                "required": ["query"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name == "get_market_data":
        result = cached_get_market_data(arguments["symbol"])
    elif name == "get_market_overview":
        result = cached_get_market_overview()
    elif name == "analyze_portfolio":
        result = cached_analyze_portfolio(arguments["portfolio_description"])
    elif name == "lookup_expense_ratio":
        result = cached_lookup_expense_ratio(arguments["fund_identifier"])
    elif name == "extract_ticker":
        ticker = cached_extract_ticker(arguments["query"])
        result = ticker if ticker else "No specific ticker found"
    else:
        result = f"Unknown tool: {name}"

    return [types.TextContent(type="text", text=str(result))]


# ── Pure ASGI app with SSE transport ──────────────────────────────────────
# We avoid Starlette's Route/endpoint layer because Route expects the handler
# to return a Response object. The SSE transport takes over the raw connection
# directly and returns None, causing a TypeError if routed via Route.

sse = SseServerTransport("/messages/")


async def _lifespan(receive, send) -> None:
    while True:
        event = await receive()
        if event["type"] == "lifespan.startup":
            await send({"type": "lifespan.startup.complete"})
        elif event["type"] == "lifespan.shutdown":
            await send({"type": "lifespan.shutdown.complete"})
            return


async def app(scope, receive, send) -> None:
    if scope["type"] == "lifespan":
        await _lifespan(receive, send)
    elif scope["type"] == "http":
        path: str = scope.get("path", "")
        if path == "/sse" and scope.get("method", "").upper() == "GET":
            async with sse.connect_sse(scope, receive, send) as streams:
                await server.run(
                    streams[0],
                    streams[1],
                    server.create_initialization_options(),
                )
        elif path.startswith("/messages"):
            await sse.handle_post_message(scope, receive, send)
        else:
            from starlette.responses import Response
            await Response("Not Found", status_code=404)(scope, receive, send)
