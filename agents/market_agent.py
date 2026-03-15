from dotenv import load_dotenv
from datetime import datetime

from langchain.agents import create_agent
from langchain.tools import tool
from langchain.messages import HumanMessage
from langchain_openai import ChatOpenAI
from tavily import TavilyClient
import yfinance as yf

load_dotenv()

tavily_client = TavilyClient()


@tool
def getMarketData(symbol: str) -> str:
    """
    Get real-time market data for a stock, ETF, or index.

    Args:
        symbol: Ticker symbol (e.g., "AAPL", "SPY", "^GSPC" for S&P 500, "^DJI" for Dow Jones)

    Returns:
        Current price, daily change, volume, and key metrics.
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        hist = ticker.history(period="5d")

        if hist.empty:
            return f"Could not find data for symbol: {symbol}"

        # Get current/latest data
        current_price = info.get("currentPrice") or info.get("regularMarketPrice") or hist["Close"].iloc[-1]
        prev_close = info.get("previousClose") or (hist["Close"].iloc[-2] if len(hist) > 1 else current_price)

        change = current_price - prev_close
        change_pct = (change / prev_close) * 100 if prev_close else 0

        # Format response
        name = info.get("longName") or info.get("shortName") or symbol
        volume = info.get("volume") or info.get("regularMarketVolume") or 0
        market_cap = info.get("marketCap", 0)

        result = f"""**{name} ({symbol})**
- Current Price: ${current_price:.2f}
- Change: {change:+.2f} ({change_pct:+.2f}%)
- Volume: {volume:,}"""

        if market_cap:
            if market_cap >= 1e12:
                result += f"\n- Market Cap: ${market_cap/1e12:.2f}T"
            elif market_cap >= 1e9:
                result += f"\n- Market Cap: ${market_cap/1e9:.2f}B"

        # Add day range
        day_high = info.get("dayHigh") or hist["High"].iloc[-1]
        day_low = info.get("dayLow") or hist["Low"].iloc[-1]
        result += f"\n- Day Range: ${day_low:.2f} - ${day_high:.2f}"

        # Add 52-week range if available
        week52_high = info.get("fiftyTwoWeekHigh")
        week52_low = info.get("fiftyTwoWeekLow")
        if week52_high and week52_low:
            result += f"\n- 52-Week Range: ${week52_low:.2f} - ${week52_high:.2f}"

        return result

    except Exception as e:
        return f"Error fetching data for {symbol}: {str(e)}"


@tool
def getMarketOverview() -> str:
    """
    Get a real-time overview of major market indices.
    Use this for questions about "how is the market doing today" or general market status.

    Returns:
        Current status of S&P 500, Dow Jones, Nasdaq, and other major indices.
    """
    indices = {
        "^GSPC": "S&P 500",
        "^DJI": "Dow Jones",
        "^IXIC": "Nasdaq",
        "^RUT": "Russell 2000",
        "^VIX": "VIX (Volatility)"
    }

    results = ["**Market Overview** (Real-time)\n"]

    for symbol, name in indices.items():
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="2d")

            if not hist.empty:
                current = hist["Close"].iloc[-1]
                prev = hist["Close"].iloc[-2] if len(hist) > 1 else current
                change = current - prev
                change_pct = (change / prev) * 100 if prev else 0

                # Add emoji indicator
                emoji = "🟢" if change >= 0 else "🔴"

                if symbol == "^VIX":
                    results.append(f"{emoji} **{name}**: {current:.2f} ({change:+.2f})")
                else:
                    results.append(f"{emoji} **{name}**: {current:,.2f} ({change_pct:+.2f}%)")
        except:
            continue

    results.append(f"\n_Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_")
    return "\n".join(results)


@tool
def searchMarketNews(query: str) -> str:
    """
    Search for recent market news and events.
    Use this for questions about market events, news, or what's happening today.

    Args:
        query: Search query about market news or events

    Returns:
        Recent news articles and market updates.
    """
    # Add "today" or current date to improve recency
    today = datetime.now().strftime("%Y-%m-%d")
    enhanced_query = f"{query} {today}"

    try:
        result = tavily_client.search(
            query=enhanced_query,
            search_depth="advanced",
            include_domains=[
                "marketwatch.com",
                "finance.yahoo.com",
                "cnbc.com",
                "bloomberg.com",
                "reuters.com",
                "wsj.com"
            ],
            num_results=5,
            region="us",
            language="en"
        )

        if not result or not result.get("results"):
            return "No recent news found for this query."

        news_items = []
        for item in result["results"][:5]:
            title = item.get("title", "No title")
            content = item.get("content", "")[:200]
            url = item.get("url", "")
            news_items.append(f"**{title}**\n{content}...\n[Read more]({url})\n")

        return f"**Recent Market News** (as of {today}):\n\n" + "\n".join(news_items)

    except Exception as e:
        return f"Error searching news: {str(e)}"


@tool
def getSectorPerformance() -> str:
    """
    Get today's performance of major market sectors.
    Use this for questions about sector performance or which sectors are up/down.

    Returns:
        Performance of major sector ETFs.
    """
    sectors = {
        "XLK": "Technology",
        "XLF": "Financials",
        "XLV": "Healthcare",
        "XLE": "Energy",
        "XLI": "Industrials",
        "XLC": "Communication",
        "XLY": "Consumer Discretionary",
        "XLP": "Consumer Staples",
        "XLRE": "Real Estate",
        "XLU": "Utilities",
        "XLB": "Materials"
    }

    results = ["**Sector Performance** (Today)\n"]
    sector_data = []

    for symbol, name in sectors.items():
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="2d")

            if not hist.empty:
                current = hist["Close"].iloc[-1]
                prev = hist["Close"].iloc[-2] if len(hist) > 1 else current
                change_pct = ((current - prev) / prev) * 100 if prev else 0
                sector_data.append((name, change_pct, symbol))
        except:
            continue

    # Sort by performance
    sector_data.sort(key=lambda x: x[1], reverse=True)

    for name, change_pct, symbol in sector_data:
        emoji = "🟢" if change_pct >= 0 else "🔴"
        results.append(f"{emoji} **{name}** ({symbol}): {change_pct:+.2f}%")

    results.append(f"\n_Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_")
    return "\n".join(results)


# Create the agent with streaming enabled
llm = ChatOpenAI(model="gpt-4o-mini", streaming=True)

agent = create_agent(
    llm,
    tools=[getMarketData, getMarketOverview, searchMarketNews, getSectorPerformance],
    system_prompt="""You are a helpful market analyst assistant that provides accurate, real-time market information.

Your tools provide LIVE data:
- getMarketOverview: Use this FIRST for general "how is the market" questions
- getMarketData: Use for specific stock/ETF quotes (e.g., "AAPL price", "how is Tesla doing")
- searchMarketNews: Use for news, events, or "what's happening today" questions
- getSectorPerformance: Use for sector-related questions

IMPORTANT:
- Always use your tools to get current data - never rely on training data for prices or market status
- For "today's market" questions, use getMarketOverview first, then searchMarketNews if needed
- When asked about a specific stock, use getMarketData with the ticker symbol
- Present data clearly with the prices and percentages from your tools

Politely decline non-finance questions."""
)

if __name__ == "__main__":
    # Example interaction
    print("Test 1: Market Overview")
    print("-" * 40)
    user_input = "How is the market doing today?"
    response = agent.invoke({"messages": [HumanMessage(content=user_input)]})
    print(response["messages"][-1].content)

    print("\n" + "=" * 50 + "\n")

    print("Test 2: Specific Stock")
    print("-" * 40)
    user_input = "What's Apple's stock price?"
    response = agent.invoke({"messages": [HumanMessage(content=user_input)]})
    print(response["messages"][-1].content)
