# Usage Guide

This guide provides comprehensive examples and tutorials for using all features of the Finance Assistant.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Chat Tab](#chat-tab)
3. [News Tab](#news-tab)
4. [Market Tab](#market-tab)
5. [Agent Routing Examples](#agent-routing-examples)
6. [CLI Usage](#cli-usage)

---

## Getting Started

### Launching the Application

```bash
uv run streamlit run streamlit_app.py
# Opens at http://localhost:8501
```

---

## Chat Tab

### Example Queries by Agent

| Query | Routed To | Description |
|-------|-----------|-------------|
| "What is a mutual fund?" | QA Agent | General finance concepts |
| "What is Apple's stock price?" | Market Agent | Real-time market data |
| "Latest Tesla news" | News Agent | Financial news |
| "How do capital gains taxes work?" | Tax Agent | Tax questions |
| "How should I save for retirement?" | Goal Agent | Financial planning |
| "How to diversify my portfolio?" | Portfolio Agent | Investment advice |
| "Best pizza recipe?" | **BLOCKED** | Off-topic (guardrail) |

### Conversation Memory

The chat maintains conversation history within a session:

```
User: What is a stock?
Assistant: A stock represents ownership in a company...

User: How do I buy one?
Assistant: [Understands "one" refers to stocks from context]
```

### Starting Fresh

Click "New Conversation" in the sidebar to clear history.

---

## News Tab

### Quick Actions

| Button | Action |
|--------|--------|
| 📰 Latest Headlines | Fetches top financial news |
| 📈 Market News | Stock market specific news |
| 🔄 Clear Chat | Clears news conversation |

### Example Queries

```
"What are the latest financial headlines?"
"News about Apple earnings"
"What's happening in the crypto market?"
"Latest Fed interest rate news"
```

---

## Market Tab

### Smart Ticker Recognition

Type company names naturally - no need to know ticker symbols:

| You Type | Recognized As |
|----------|---------------|
| Apple | AAPL |
| Tesla stock | TSLA |
| Microsoft price | MSFT |
| S&P 500 | ^GSPC |
| Nasdaq | ^IXIC |
| Nintendo | NTDOY |
| Sony | SONY |
| Bitcoin ETF | IBIT |

### Supported Securities

**US Stocks**: AAPL, MSFT, GOOGL, AMZN, TSLA, etc.

**International ADRs**: SONY, NTDOY, TM, BABA, TSM

**Market Indices**:
- S&P 500: ^GSPC
- Dow Jones: ^DJI  
- Nasdaq: ^IXIC
- VIX: ^VIX

**ETFs**: SPY, QQQ, VOO, VTI, IWM, GLD, SLV

### Chart Features

- **Candlestick chart**: OHLC price data
- **Moving averages**: MA20 (orange), MA50 (blue)
- **Time periods**: 1mo, 3mo, 6mo, 1y

### Metrics Displayed

| Metric | Description |
|--------|-------------|
| Price | Current price with daily change |
| Volume | Trading volume |
| Market Cap | Total market value |
| P/E Ratio | Price-to-earnings ratio |

---

## Agent Routing Examples

### QA Agent (RAG)
```
"What is compound interest?"
"Explain dollar-cost averaging"
"Define market capitalization"
"What are ETFs?"
```

### Market Agent
```
"Current price of AAPL"
"How is the market doing today?"
"Tesla stock performance"
"What's Amazon trading at?"
```

### News Agent
```
"Latest financial news"
"Apple earnings news"
"Fed interest rate decision"
"Crypto market updates"
```

### Tax Agent
```
"What are capital gains taxes?"
"How do I calculate my tax bracket?"
"What deductions can I claim?"
"401k tax benefits"
```

### Goal Agent
```
"How to save for retirement?"
"Setting up an emergency fund"
"How much should I save monthly?"
"FIRE movement explained"
```

### Portfolio Agent
```
"How to diversify my portfolio?"
"Asset allocation strategies"
"Rebalancing frequency"
"Growth vs value investing"
```

---

## CLI Usage

### Test Router Agent Directly

```bash
uv run python -m agents.router_agent_v2
```

This runs two test cases:
1. On-topic: "What is a stock option?"
2. Off-topic: "What is the best recipe for chocolate cake?"

### Test Individual Agents

```python
from agents import qa_agent, market_agent
from langchain_core.messages import HumanMessage

# QA Agent
response = qa_agent.agent.invoke({
    "messages": [HumanMessage(content="What is a bond?")]
})
print(response["messages"][-1].content)

# Market Agent
response = market_agent.agent.invoke({
    "messages": [HumanMessage(content="AAPL price")]
})
print(response["messages"][-1].content)
```

---

## Tips & Best Practices

1. **Be Specific**: "What is Apple's current P/E ratio?" is better than "Tell me about Apple"

2. **Use Natural Language**: The system understands context, no need for formal queries

3. **Check the Routing**: The UI shows which agent handled your query

4. **Market Tab for Charts**: Use the Market tab when you want visual data

5. **Conversation Context**: Follow-up questions work within the same session
