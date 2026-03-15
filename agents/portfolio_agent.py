from dotenv import load_dotenv
import json
import re
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from enum import Enum

from langchain.agents import create_agent
from langchain.tools import tool
from langchain.messages import HumanMessage
from langchain_openai import ChatOpenAI
from tavily import TavilyClient

load_dotenv()

tavily_client = TavilyClient()

# Asset class enumeration
class AssetClass(Enum):
    US_STOCK = "US Stocks"
    INTL_STOCK = "International Stocks"
    BOND = "Bonds"
    CASH = "Cash"
    REAL_ESTATE = "Real Estate"
    COMMODITY = "Commodities"
    CRYPTO = "Cryptocurrency"
    MIXED = "Mixed/Target Date"

# Reference data for common fund expense ratios
EXPENSE_RATIOS = {
    # Vanguard ETFs/Funds
    "VOO": 0.03, "VTI": 0.03, "VXUS": 0.07, "BND": 0.03,
    "VTSAX": 0.04, "VFIAX": 0.04, "VBTLX": 0.05, "VTIAX": 0.11,
    "VIG": 0.06, "VYM": 0.06, "VNQ": 0.12, "VGSLX": 0.12,
    "VT": 0.07, "VEA": 0.05, "VWO": 0.08,
    # iShares ETFs
    "IVV": 0.03, "ITOT": 0.03, "AGG": 0.03, "IEMG": 0.09,
    "IEFA": 0.07, "IWM": 0.19, "IWF": 0.19, "IWD": 0.19,
    "IJH": 0.05, "IJR": 0.06,
    # SPDR ETFs
    "SPY": 0.09, "GLD": 0.40, "SLV": 0.50,
    # Schwab
    "SCHB": 0.03, "SCHX": 0.03, "SCHF": 0.06, "SCHA": 0.04,
    "SCHD": 0.06, "SCHG": 0.04,
    # Fidelity
    "FXAIX": 0.015, "FSKAX": 0.015, "FZROX": 0.00, "FZILX": 0.00,
    "FTIHX": 0.06, "FXNAX": 0.025,
    # Other popular
    "QQQ": 0.20, "ARKK": 0.75, "ARKW": 0.75,
    "DIA": 0.16, "XLK": 0.09, "XLF": 0.09, "XLE": 0.09,
    # Target date defaults
    "TARGET_DATE": 0.14,
}

# Asset class mappings for common tickers
ASSET_CLASS_MAP = {
    # US Stock ETFs
    "VOO": AssetClass.US_STOCK, "VTI": AssetClass.US_STOCK, "SPY": AssetClass.US_STOCK,
    "IVV": AssetClass.US_STOCK, "QQQ": AssetClass.US_STOCK, "FXAIX": AssetClass.US_STOCK,
    "FSKAX": AssetClass.US_STOCK, "FZROX": AssetClass.US_STOCK, "ITOT": AssetClass.US_STOCK,
    "SCHB": AssetClass.US_STOCK, "SCHX": AssetClass.US_STOCK, "IWM": AssetClass.US_STOCK,
    "VIG": AssetClass.US_STOCK, "VYM": AssetClass.US_STOCK, "SCHD": AssetClass.US_STOCK,
    "DIA": AssetClass.US_STOCK, "XLK": AssetClass.US_STOCK, "XLF": AssetClass.US_STOCK,
    # International Stock
    "VXUS": AssetClass.INTL_STOCK, "IEFA": AssetClass.INTL_STOCK, "IEMG": AssetClass.INTL_STOCK,
    "VTIAX": AssetClass.INTL_STOCK, "VEA": AssetClass.INTL_STOCK, "VWO": AssetClass.INTL_STOCK,
    "FZILX": AssetClass.INTL_STOCK, "SCHF": AssetClass.INTL_STOCK,
    # Bonds
    "BND": AssetClass.BOND, "AGG": AssetClass.BOND, "VBTLX": AssetClass.BOND,
    "FXNAX": AssetClass.BOND,
    # Real Estate
    "VNQ": AssetClass.REAL_ESTATE, "VGSLX": AssetClass.REAL_ESTATE,
    # Commodities
    "GLD": AssetClass.COMMODITY, "SLV": AssetClass.COMMODITY,
}

# Risk weights by asset class (higher = more volatile)
RISK_WEIGHTS = {
    AssetClass.CRYPTO: 5,
    AssetClass.COMMODITY: 3,
    AssetClass.INTL_STOCK: 2.5,
    AssetClass.US_STOCK: 2,
    AssetClass.REAL_ESTATE: 2,
    AssetClass.MIXED: 1.5,
    AssetClass.BOND: 1,
    AssetClass.CASH: 0,
}

@dataclass
class Holding:
    name: str
    ticker: Optional[str]
    value: float
    asset_class: AssetClass
    expense_ratio: Optional[float]
    is_index_fund: bool

@dataclass
class PortfolioMetrics:
    total_value: float
    holdings: List[Holding]
    allocations: Dict[str, float]
    weighted_expense_ratio: float
    annual_fee_cost: float
    diversification_score: int
    risk_level: str
    risk_factors: List[str]

# LLM for parsing
parsing_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def parse_portfolio(description: str) -> tuple[float, List[dict]]:
    """Use LLM to parse natural language portfolio description."""
    parse_prompt = f"""Extract portfolio holdings from this description.

For each holding, identify:
- name: The investment name (e.g., "Apple", "S&P 500 index fund", "Vanguard Total Stock")
- ticker: Stock/ETF ticker if identifiable (e.g., "AAPL", "VOO", "VTI"). Use null if unclear.
- value: Dollar amount as number (no symbols)
- is_percentage: true if originally given as percentage, false if dollar amount
- asset_type: one of "us_stock", "intl_stock", "bond", "cash", "real_estate", "commodity", "crypto", "mixed", "individual_stock"
- is_index_fund: true if it's an index fund or broad market ETF

Input: "{description}"

Also extract:
- total_value: The total portfolio value if mentioned (null if not mentioned)

Return ONLY valid JSON in this exact format:
{{
  "total_value": 200000,
  "holdings": [
    {{"name": "S&P 500 Index Fund", "ticker": "VOO", "value": 100000, "is_percentage": false, "asset_type": "us_stock", "is_index_fund": true}},
    {{"name": "Apple", "ticker": "AAPL", "value": 50000, "is_percentage": false, "asset_type": "individual_stock", "is_index_fund": false}}
  ]
}}

If percentages are given, convert them to dollar amounts using total_value.
If total_value is not given but percentages are used, set total_value to 100000 as default."""

    response = parsing_llm.invoke([HumanMessage(content=parse_prompt)])

    # Clean up the response - remove markdown code blocks if present
    content = response.content.strip()
    if content.startswith("```"):
        content = re.sub(r'^```(?:json)?\n?', '', content)
        content = re.sub(r'\n?```$', '', content)

    data = json.loads(content)
    return data.get("total_value", 100000), data.get("holdings", [])


def get_asset_class(asset_type: str, ticker: Optional[str]) -> AssetClass:
    """Determine asset class from type string and ticker."""
    if ticker and ticker.upper() in ASSET_CLASS_MAP:
        return ASSET_CLASS_MAP[ticker.upper()]

    type_map = {
        "us_stock": AssetClass.US_STOCK,
        "individual_stock": AssetClass.US_STOCK,
        "intl_stock": AssetClass.INTL_STOCK,
        "international_stock": AssetClass.INTL_STOCK,
        "bond": AssetClass.BOND,
        "bonds": AssetClass.BOND,
        "cash": AssetClass.CASH,
        "real_estate": AssetClass.REAL_ESTATE,
        "commodity": AssetClass.COMMODITY,
        "crypto": AssetClass.CRYPTO,
        "mixed": AssetClass.MIXED,
        "target_date": AssetClass.MIXED,
    }
    return type_map.get(asset_type.lower(), AssetClass.US_STOCK)


def get_expense_ratio(ticker: Optional[str], name: str) -> Optional[float]:
    """Look up expense ratio from reference data."""
    if ticker and ticker.upper() in EXPENSE_RATIOS:
        return EXPENSE_RATIOS[ticker.upper()]

    # Check for target date funds
    name_lower = name.lower()
    if "target" in name_lower and "date" in name_lower:
        return EXPENSE_RATIOS["TARGET_DATE"]

    # Individual stocks have no expense ratio
    return None


def calculate_diversification_score(holdings: List[Holding], allocations: Dict[str, float]) -> int:
    """Calculate diversification score (0-100)."""
    score = 0

    # Asset Class Diversity (40 points max)
    # More asset classes = better diversification
    num_asset_classes = len([a for a, pct in allocations.items() if pct > 5])
    score += min(num_asset_classes * 10, 40)

    # Holdings Count (20 points max)
    # More holdings = less concentration risk
    num_holdings = len(holdings)
    if num_holdings >= 10:
        score += 20
    elif num_holdings >= 5:
        score += 15
    elif num_holdings >= 3:
        score += 10
    else:
        score += 5

    # Index Fund Usage (20 points max)
    # Index funds provide built-in diversification
    index_fund_pct = sum(h.value for h in holdings if h.is_index_fund) / sum(h.value for h in holdings) * 100
    if index_fund_pct >= 80:
        score += 20
    elif index_fund_pct >= 50:
        score += 15
    elif index_fund_pct >= 25:
        score += 10
    else:
        score += 5

    # International Exposure (20 points max)
    intl_pct = allocations.get(AssetClass.INTL_STOCK.value, 0)
    if 15 <= intl_pct <= 40:
        score += 20  # Ideal range
    elif intl_pct > 0:
        score += 10
    else:
        score += 0

    return min(score, 100)


def assess_risk(holdings: List[Holding], allocations: Dict[str, float]) -> tuple[str, List[str]]:
    """Assess portfolio risk level and identify risk factors."""
    risk_factors = []
    total_value = sum(h.value for h in holdings)

    # Calculate weighted risk score
    weighted_risk = 0
    for holding in holdings:
        weight = holding.value / total_value
        risk_weight = RISK_WEIGHTS.get(holding.asset_class, 2)
        weighted_risk += weight * risk_weight

    # Calculate stock percentage
    stock_pct = allocations.get(AssetClass.US_STOCK.value, 0) + allocations.get(AssetClass.INTL_STOCK.value, 0)
    bond_pct = allocations.get(AssetClass.BOND.value, 0) + allocations.get(AssetClass.CASH.value, 0)

    # Determine risk level
    if weighted_risk >= 2.5 or stock_pct >= 90:
        risk_level = "Aggressive"
        risk_factors.append(f"High equity allocation ({stock_pct:.0f}%)")
    elif weighted_risk >= 2.0 or stock_pct >= 75:
        risk_level = "Moderate-Aggressive"
        risk_factors.append(f"Growth-oriented allocation ({stock_pct:.0f}% stocks)")
    elif weighted_risk >= 1.5 or stock_pct >= 50:
        risk_level = "Moderate"
        risk_factors.append("Balanced stock/bond allocation")
    elif weighted_risk >= 1.0 or stock_pct >= 30:
        risk_level = "Moderate-Conservative"
        risk_factors.append("Conservative allocation with some growth exposure")
    else:
        risk_level = "Conservative"
        risk_factors.append(f"Low equity allocation ({stock_pct:.0f}%)")

    # Check for concentration risk
    if holdings:
        max_holding_pct = max(h.value for h in holdings) / total_value * 100
        if max_holding_pct > 50:
            risk_factors.append(f"High concentration: largest position is {max_holding_pct:.0f}%")
        elif max_holding_pct > 30:
            risk_factors.append(f"Moderate concentration: largest position is {max_holding_pct:.0f}%")

    # Check for individual stock risk
    individual_stocks = [h for h in holdings if not h.is_index_fund and h.asset_class == AssetClass.US_STOCK]
    if individual_stocks:
        individual_pct = sum(s.value for s in individual_stocks) / total_value * 100
        if individual_pct > 30:
            risk_factors.append(f"High individual stock exposure ({individual_pct:.0f}%)")

    # Check for crypto exposure
    crypto_pct = allocations.get(AssetClass.CRYPTO.value, 0)
    if crypto_pct > 10:
        risk_factors.append(f"High cryptocurrency allocation ({crypto_pct:.0f}%)")

    # Check for lack of bonds
    if bond_pct < 10 and stock_pct > 60:
        risk_factors.append("Limited fixed income allocation")

    return risk_level, risk_factors


def format_portfolio_report(metrics: PortfolioMetrics) -> str:
    """Format the portfolio metrics into a readable report."""
    report = []
    report.append("=" * 50)
    report.append("PORTFOLIO ANALYSIS")
    report.append("=" * 50)

    # Total Value
    report.append(f"\nTOTAL VALUE: ${metrics.total_value:,.0f}")

    # Asset Allocation
    report.append("\n--- ASSET ALLOCATION ---")
    for asset_class, pct in sorted(metrics.allocations.items(), key=lambda x: x[1], reverse=True):
        value = metrics.total_value * pct / 100
        report.append(f"  {asset_class}: {pct:.1f}% (${value:,.0f})")

    # Holdings Detail
    report.append("\n--- HOLDINGS ---")
    for i, h in enumerate(sorted(metrics.holdings, key=lambda x: x.value, reverse=True), 1):
        pct = (h.value / metrics.total_value) * 100
        ticker_str = f" ({h.ticker})" if h.ticker else ""
        expense_str = f" | ER: {h.expense_ratio:.2f}%" if h.expense_ratio is not None else ""
        report.append(f"  {i}. {h.name}{ticker_str}: ${h.value:,.0f} ({pct:.1f}%){expense_str}")

    # Expense Analysis
    report.append("\n--- EXPENSE ANALYSIS ---")
    report.append(f"  Weighted Expense Ratio: {metrics.weighted_expense_ratio:.3f}%")
    report.append(f"  Estimated Annual Cost: ${metrics.annual_fee_cost:,.0f}")

    # Diversification Score
    report.append("\n--- DIVERSIFICATION SCORE ---")
    score = metrics.diversification_score
    if score >= 80:
        assessment = "Excellent"
    elif score >= 60:
        assessment = "Good"
    elif score >= 40:
        assessment = "Fair"
    else:
        assessment = "Needs Improvement"
    report.append(f"  Score: {score}/100 ({assessment})")

    # Risk Assessment
    report.append("\n--- RISK ASSESSMENT ---")
    report.append(f"  Risk Level: {metrics.risk_level}")
    report.append("  Factors:")
    for factor in metrics.risk_factors:
        report.append(f"    - {factor}")

    report.append("\n" + "=" * 50)

    return "\n".join(report)


@tool
def analyzePortfolio(portfolio_description: str) -> str:
    """
    Analyzes a user's investment portfolio and provides comprehensive metrics.

    Args:
        portfolio_description: Natural language description of the portfolio, e.g.,
            "I have $200K: 50% in S&P 500 index fund, 30% bonds, 20% target date 2045"
            or "AAPL $10K, MSFT $15K, VOO $50K, BND $25K"

    Returns comprehensive analysis including:
    - Total portfolio value
    - Asset allocation percentages
    - Expense ratios for each holding
    - Diversification score (0-100)
    - Risk assessment with specific factors
    """
    try:
        # Parse the portfolio description
        total_value, parsed_holdings = parse_portfolio(portfolio_description)

        if not parsed_holdings:
            return "I couldn't parse any holdings from your description. Please describe your portfolio with values, e.g., 'I have $100K: 60% in VOO, 40% in BND'"

        # Build Holding objects
        holdings = []
        for h in parsed_holdings:
            ticker = h.get("ticker")
            name = h.get("name", "Unknown")
            value = float(h.get("value", 0))
            asset_type = h.get("asset_type", "us_stock")
            is_index = h.get("is_index_fund", False)

            asset_class = get_asset_class(asset_type, ticker)
            expense_ratio = get_expense_ratio(ticker, name)

            holdings.append(Holding(
                name=name,
                ticker=ticker.upper() if ticker else None,
                value=value,
                asset_class=asset_class,
                expense_ratio=expense_ratio,
                is_index_fund=is_index
            ))

        # Recalculate total if needed
        calculated_total = sum(h.value for h in holdings)
        if calculated_total > 0:
            total_value = calculated_total

        # Calculate allocations
        allocations = {}
        for holding in holdings:
            ac = holding.asset_class.value
            if ac not in allocations:
                allocations[ac] = 0
            allocations[ac] += (holding.value / total_value) * 100

        # Calculate weighted expense ratio
        weighted_expense = 0
        for holding in holdings:
            if holding.expense_ratio is not None:
                weight = holding.value / total_value
                weighted_expense += holding.expense_ratio * weight

        annual_fee_cost = total_value * weighted_expense / 100

        # Calculate diversification score
        div_score = calculate_diversification_score(holdings, allocations)

        # Assess risk
        risk_level, risk_factors = assess_risk(holdings, allocations)

        # Build metrics object
        metrics = PortfolioMetrics(
            total_value=total_value,
            holdings=holdings,
            allocations=allocations,
            weighted_expense_ratio=weighted_expense,
            annual_fee_cost=annual_fee_cost,
            diversification_score=div_score,
            risk_level=risk_level,
            risk_factors=risk_factors
        )

        return format_portfolio_report(metrics)

    except json.JSONDecodeError as e:
        return f"I had trouble understanding your portfolio description. Please try describing it more clearly, e.g., 'I have $200K total: 50% in VOO, 30% in BND, 20% in a target date fund'"
    except Exception as e:
        return f"An error occurred while analyzing your portfolio: {str(e)}"


@tool
def lookupExpenseRatio(fund_identifier: str) -> str:
    """
    Looks up the expense ratio for a mutual fund or ETF.

    Args:
        fund_identifier: Ticker symbol or fund name (e.g., "VOO", "Vanguard S&P 500")

    Returns: Expense ratio information for the fund.
    """
    # Normalize input
    ticker = fund_identifier.upper().strip()

    # Check reference data first
    if ticker in EXPENSE_RATIOS:
        return f"{ticker}: {EXPENSE_RATIOS[ticker]:.3f}% expense ratio"

    # Try to find partial matches
    for key, ratio in EXPENSE_RATIOS.items():
        if key in ticker or ticker in key:
            return f"{key}: {ratio:.3f}% expense ratio"

    # Fallback to web search
    try:
        result = tavily_client.search(
            query=f"{fund_identifier} expense ratio fee",
            include_domains=["morningstar.com", "finance.yahoo.com", "etf.com"],
            num_results=2,
            region="us",
            language="en"
        )

        if result and result.get("results"):
            return f"Search results for {fund_identifier} expense ratio:\n" + "\n".join(
                [f"- {r.get('content', '')[:200]}..." for r in result["results"][:2]]
            )
        return f"Could not find expense ratio for {fund_identifier}. Try searching with the ticker symbol."
    except Exception as e:
        return f"Could not look up expense ratio for {fund_identifier}: {str(e)}"


@tool
def searchFinance(query: str) -> str:
    """Searches for financial information on the web for portfolio-related questions."""
    return tavily_client.search(
        query=query,
        include_domains=["investopedia.com", "morningstar.com", "finance.yahoo.com", "bogleheads.org"],
        num_results=2,
        region="us",
        language="en"
    )


# Create the agent with streaming enabled
llm = ChatOpenAI(model="gpt-4o-mini", streaming=True)

agent = create_agent(
    llm,
    tools=[analyzePortfolio, lookupExpenseRatio, searchFinance],
    system_prompt="""You are a helpful finance assistant specialized in portfolio analysis and management.

Your capabilities:
1. Analyze investment portfolios to calculate total value, allocation percentages, and weighted expense ratios
2. Assess portfolio diversification across asset classes and geographies
3. Evaluate portfolio risk based on asset allocation and concentration
4. Look up expense ratios for mutual funds and ETFs
5. Provide actionable insights for portfolio improvement

When a user describes their portfolio:
1. Use the analyzePortfolio tool to get comprehensive metrics
2. Explain the results in clear, accessible language
3. Highlight any concerns (high fees, concentration risk, lack of diversification)
4. Suggest improvements where appropriate

When a user asks about a specific fund's expense ratio:
1. Use the lookupExpenseRatio tool

For general portfolio strategy questions, use the searchFinance tool if needed.

Always be helpful, accurate, and focused on the user's financial wellbeing.
Politely decline questions that are not related to finance or investing."""
)

if __name__ == "__main__":
    # Example interaction
    print("Test 1: Comprehensive portfolio analysis")
    print("-" * 50)
    user_input = "I have $200K in my 401K: 50% in S&P 500 index fund (VOO), 30% in a bond fund (BND), and 20% in a target date 2045 fund"
    response = agent.invoke({"messages": [HumanMessage(content=user_input)]})
    print(response["messages"][-1].content)

    print("\n" + "=" * 50 + "\n")

    print("Test 2: Individual stocks portfolio")
    print("-" * 50)
    user_input = "My portfolio: AAPL $25,000, MSFT $20,000, VOO $40,000, BND $15,000"
    response = agent.invoke({"messages": [HumanMessage(content=user_input)]})
    print(response["messages"][-1].content)
