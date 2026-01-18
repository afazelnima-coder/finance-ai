"""
Utility functions for ticker symbol extraction and processing.
"""
from langchain_openai import ChatOpenAI
from langchain.messages import HumanMessage


def get_ticker_llm():
    """Get the LLM for ticker extraction."""
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)


def extract_ticker(query: str, llm=None) -> str | None:
    """
    Use LLM to extract ticker symbol if a specific company is mentioned.

    Args:
        query: The user's query about a company or stock
        llm: Optional LLM instance (for testing). If None, creates a new one.

    Returns:
        Ticker symbol string if found, None if no specific company mentioned
    """
    if llm is None:
        llm = get_ticker_llm()

    prompt = f"""Analyze this market question and determine if it mentions a specific publicly traded company.

If a specific company is mentioned, return its stock ticker symbol.
- For US companies, return the US ticker (e.g., AAPL, MSFT)
- For international companies trading as ADRs in the US, return the ADR ticker
- For international companies not on US exchanges, use the format: TICKER.EXCHANGE
  (e.g., .T for Tokyo, .L for London, .HK for Hong Kong, .DE for Germany)

If NO specific company is mentioned (general market questions), return "NONE".

Examples:
- "What's the price of Apple?" → AAPL
- "How is Tesla doing today?" → TSLA
- "Show me Microsoft's chart" → MSFT
- "What's driving tech stocks?" → NONE
- "Is the market up today?" → NONE
- "Amazon earnings report" → AMZN
- "Compare nvidia to AMD" → NVDA
- "What's happening with meta?" → META
- "Sony stock price" → SONY
- "How is Nintendo doing?" → NTDOY
- "Toyota stock" → TM
- "Samsung electronics" → 005930.KS
- "Alibaba stock" → BABA
- "TSMC performance" → TSM
- "BMW stock" → BMW.DE
- "Honda" → HMC
- "Nestle" → NSRGY
- "SAP" → SAP
- "Spotify" → SPOT
- "Shopify" → SHOP
- "NIO stock" → NIO
- "BYD company" → BYDDY
- "Tencent" → TCEHY
- "SoftBank" → SFTBY
- "ASML" → ASML
- "Novo Nordisk" → NVO

Index funds, ETFs, and market indices:
- "S&P 500" → ^GSPC
- "SPY ETF" → SPY
- "How is the S&P doing?" → ^GSPC
- "Nasdaq" → ^IXIC
- "QQQ" → QQQ
- "Nasdaq 100 ETF" → QQQ
- "Dow Jones" → ^DJI
- "the Dow" → ^DJI
- "DIA ETF" → DIA
- "Russell 2000" → ^RUT
- "IWM" → IWM
- "VTI total market" → VTI
- "VOO" → VOO
- "Vanguard S&P 500" → VOO
- "VIX" → ^VIX
- "volatility index" → ^VIX
- "fear index" → ^VIX
- "bond ETF" → BND
- "AGG bonds" → AGG
- "tech sector ETF" → XLK
- "XLF financials" → XLF
- "energy sector" → XLE
- "gold ETF" → GLD
- "silver ETF" → SLV
- "emerging markets" → EEM
- "international stocks ETF" → VXUS
- "ARK Innovation" → ARKK
- "Bitcoin ETF" → IBIT
- "Ethereum ETF" → ETHA

User question: "{query}"

Respond with ONLY the ticker symbol or "NONE", nothing else."""

    response = llm.invoke([HumanMessage(content=prompt)])
    result = response.content.strip().upper()
    return None if result == "NONE" else result


# Common ticker mappings for quick lookup (no API call needed)
COMMON_TICKERS = {
    # US Tech Giants
    "apple": "AAPL",
    "microsoft": "MSFT",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "amazon": "AMZN",
    "meta": "META",
    "facebook": "META",
    "tesla": "TSLA",
    "nvidia": "NVDA",
    "netflix": "NFLX",
    # US Market Indices
    "s&p 500": "^GSPC",
    "s&p": "^GSPC",
    "dow jones": "^DJI",
    "dow": "^DJI",
    "nasdaq": "^IXIC",
    "russell 2000": "^RUT",
    # Popular ETFs
    "spy": "SPY",
    "qqq": "QQQ",
    "voo": "VOO",
    "vti": "VTI",
    "iwm": "IWM",
}


def quick_ticker_lookup(query: str) -> str | None:
    """
    Quick ticker lookup without LLM call for common companies.

    Args:
        query: The user's query

    Returns:
        Ticker if found in common mappings, None otherwise
    """
    query_lower = query.lower()
    for company, ticker in COMMON_TICKERS.items():
        if company in query_lower:
            return ticker
    return None
