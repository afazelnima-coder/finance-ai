# API Documentation

This document provides detailed API documentation for all agents, functions, and components in the Finance Assistant system.

## Table of Contents

1. [Router Agent API](#router-agent-api)
2. [Specialized Agents](#specialized-agents)
3. [Guardrails API](#guardrails-api)
4. [Utility Functions](#utility-functions)
5. [State Management](#state-management)
6. [External API Integration](#external-api-integration)

---

## Router Agent API

### Module: `agents/router_agent_v2.py`

The main orchestrator that routes queries to specialized agents.

### State Definition

```python
class State(TypedDict):
    """
    State object passed through the LangGraph workflow.

    Attributes:
        messages: List of conversation messages (HumanMessage, AIMessage)
        next_agent: Name of the selected agent for routing
        is_on_topic: Boolean indicating if query passed guardrail check
    """
    messages: Annotated[list, add_messages]
    next_agent: str
    is_on_topic: bool
```

### Functions

#### `guardrail_node(state: State) -> dict`

Validates if the user query is finance-related.

**Parameters:**
| Name | Type | Description |
|------|------|-------------|
| `state` | `State` | Current workflow state |

**Returns:**
```python
{"is_on_topic": bool}  # True if finance-related, False otherwise
```

**Example:**
```python
state = {"messages": [HumanMessage(content="What is a stock?")]}
result = guardrail_node(state)
# Returns: {"is_on_topic": True}
```

---

#### `router_node(state: State) -> dict`

Determines which specialized agent should handle the query.

**Parameters:**
| Name | Type | Description |
|------|------|-------------|
| `state` | `State` | Current workflow state |

**Returns:**
```python
{"next_agent": str}  # One of: "qa", "market", "news", "tax", "goal", "portfolio"
```

**Routing Logic:**
| Query Pattern | Target Agent |
|--------------|--------------|
| "what is", "explain", "define" | `qa` |
| "price", "stock", "market" | `market` |
| "news", "headlines", "update" | `news` |
| "tax", "deduction", "IRS" | `tax` |
| "goal", "retire", "save" | `goal` |
| "portfolio", "allocate", "diversify" | `portfolio` |

---

#### `off_topic_node(state: State) -> dict`

Returns a polite decline message for off-topic queries.

**Parameters:**
| Name | Type | Description |
|------|------|-------------|
| `state` | `State` | Current workflow state |

**Returns:**
```python
{
    "messages": [AIMessage(content="I'm sorry, but I can only help with finance-related questions...")]
}
```

---

#### Agent Node Functions

Each agent has a corresponding node function:

```python
def qa_node(state: State) -> dict:
    """Invokes the QA Agent."""

def market_node(state: State) -> dict:
    """Invokes the Market Agent."""

def news_node(state: State) -> dict:
    """Invokes the News Agent."""

def tax_node(state: State) -> dict:
    """Invokes the Tax Agent."""

def goal_node(state: State) -> dict:
    """Invokes the Goal Agent."""

def portfolio_node(state: State) -> dict:
    """Invokes the Portfolio Agent."""
```

**Returns (all):**
```python
{"messages": [AIMessage(content="...")]}  # Agent's response
```

---

### Compiled Agent

```python
# Agent instance
agent = workflow.compile(checkpointer=checkpointer)
```

#### `agent.invoke(input: dict, config: dict) -> dict`

Synchronously invoke the router agent.

**Parameters:**
| Name | Type | Description |
|------|------|-------------|
| `input` | `dict` | Must contain `messages` key with list of messages |
| `config` | `dict` | Must contain `configurable.thread_id` for memory |

**Returns:**
```python
{
    "messages": [...],  # All messages including response
    "next_agent": str,  # Selected agent
    "is_on_topic": bool # Guardrail result
}
```

**Example:**
```python
from langchain_core.messages import HumanMessage

response = agent.invoke(
    {"messages": [HumanMessage(content="What is a dividend?")]},
    config={"configurable": {"thread_id": "user-123"}}
)
print(response["messages"][-1].content)
```

---

#### `agent.astream_events(input: dict, config: dict, version: str)`

Asynchronously stream events from the agent.

**Parameters:**
| Name | Type | Description |
|------|------|-------------|
| `input` | `dict` | Must contain `messages` key |
| `config` | `dict` | Must contain `configurable.thread_id` |
| `version` | `str` | API version, use `"v2"` |

**Yields:**
```python
{
    "event": str,           # Event type
    "name": str,            # Component name
    "data": dict,           # Event data
    "metadata": {
        "langgraph_node": str  # Current node name
    }
}
```

**Event Types:**
| Event | Description |
|-------|-------------|
| `on_chat_model_start` | LLM invocation started |
| `on_chat_model_stream` | Token received |
| `on_chat_model_end` | LLM invocation completed |
| `on_chain_start` | Chain/node started |
| `on_chain_end` | Chain/node completed |

**Example:**
```python
async for event in agent.astream_events(
    {"messages": [HumanMessage(content="What is inflation?")]},
    config={"configurable": {"thread_id": "user-123"}},
    version="v2"
):
    if event["event"] == "on_chat_model_stream":
        chunk = event["data"]["chunk"]
        if hasattr(chunk, "content"):
            print(chunk.content, end="", flush=True)
```

---

## Specialized Agents

### QA Agent

**Module:** `agents/qa_agent.py`

RAG-based agent for answering general finance questions.

#### Configuration

```python
# Vector store retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# System prompt
system_prompt = """
You are a helpful financial assistant. Use the provided context
to answer questions about finance concepts, terms, and strategies.
If you don't know the answer, say so honestly.
"""
```

#### Usage

```python
from agents import qa_agent
from langchain_core.messages import HumanMessage

response = qa_agent.agent.invoke({
    "messages": [HumanMessage(content="What is compound interest?")]
})
print(response["messages"][-1].content)
```

---

### Market Agent

**Module:** `agents/market_agent.py`

Agent for real-time market data and analysis.

#### Tools

| Tool | Function | Description |
|------|----------|-------------|
| `get_stock_price` | Fetch current price | Returns price, change, volume |
| `get_market_overview` | Market summary | Major indices performance |

#### Usage

```python
from agents import market_agent
from langchain_core.messages import HumanMessage

response = market_agent.agent.invoke({
    "messages": [HumanMessage(content="What's Apple's current stock price?")]
})
```

---

### News Agent

**Module:** `agents/news_agent.py`

Agent for fetching financial news.

#### Tools

| Tool | Function | Description |
|------|----------|-------------|
| `search_news` | Tavily search | Search recent news articles |
| `get_market_news` | Alpha Vantage | Get market-specific news |

#### Usage

```python
from agents import news_agent
from langchain_core.messages import HumanMessage

response = news_agent.agent.invoke({
    "messages": [HumanMessage(content="What's the latest news about Tesla?")]
})
```

---

### Tax Agent

**Module:** `agents/tax_agent.py`

Agent for tax-related questions.

#### Usage

```python
from agents import tax_agent
from langchain_core.messages import HumanMessage

response = tax_agent.agent.invoke({
    "messages": [HumanMessage(content="What are capital gains taxes?")]
})
```

---

### Goal Agent

**Module:** `agents/goal_agent.py`

Agent for financial planning and goals.

#### Usage

```python
from agents import goal_agent
from langchain_core.messages import HumanMessage

response = goal_agent.agent.invoke({
    "messages": [HumanMessage(content="How should I plan for retirement?")]
})
```

---

### Portfolio Agent

**Module:** `agents/portfolio_agent.py`

Agent for portfolio management advice.

#### Usage

```python
from agents import portfolio_agent
from langchain_core.messages import HumanMessage

response = portfolio_agent.agent.invoke({
    "messages": [HumanMessage(content="How should I diversify my portfolio?")]
})
```

---

## Guardrails API

### Module: `agents/router_agent_v2.py`

### FinanceTopicValidator

```python
@register_validator(name="finance_topic_validator", data_type="string")
class FinanceTopicValidator(Validator):
    """
    Custom Guardrails AI validator that checks if input is finance-related.

    Uses GPT-4o-mini for classification.
    """
```

#### Methods

##### `validate(value: str, metadata: Dict = {}) -> ValidationResult`

Validates if the input string is finance-related.

**Parameters:**
| Name | Type | Description |
|------|------|-------------|
| `value` | `str` | The text to validate |
| `metadata` | `Dict` | Optional metadata (unused) |

**Returns:**
```python
PassResult()  # If finance-related
FailResult(error_message="...")  # If not finance-related
```

**Example:**
```python
validator = FinanceTopicValidator()

# Finance-related
result = validator.validate("What is a mutual fund?")
# Returns: PassResult()

# Not finance-related
result = validator.validate("What's a good pizza recipe?")
# Returns: FailResult(error_message="The query is not related to finance topics.")
```

---

### Guard Instance

```python
finance_guard = Guard().use(
    FinanceTopicValidator(on_fail="noop")
)
```

#### `finance_guard.validate(text: str) -> ValidationOutcome`

Validate text against the finance topic guardrail.

**Returns:**
```python
ValidationOutcome(
    validation_passed: bool,
    validated_output: str,
    ...
)
```

---

## Utility Functions

### Module: `streamlit_app.py`

### `is_finance_related(query: str) -> bool`

Global function to check if a query is finance-related.

**Parameters:**
| Name | Type | Description |
|------|------|-------------|
| `query` | `str` | User query to validate |

**Returns:**
`bool` - True if finance-related, False otherwise

**Example:**
```python
is_finance_related("What is a stock?")  # True
is_finance_related("Best pizza in NYC?")  # False
```

---

### `extract_ticker(query: str) -> str | None`

Extract stock ticker symbol from natural language query.

**Parameters:**
| Name | Type | Description |
|------|------|-------------|
| `query` | `str` | User query mentioning a company |

**Returns:**
`str | None` - Ticker symbol or None if no company mentioned

**Examples:**
```python
extract_ticker("How is Apple doing?")  # "AAPL"
extract_ticker("Tesla stock price")     # "TSLA"
extract_ticker("S&P 500 performance")   # "^GSPC"
extract_ticker("Market trends today")   # None
```

---

### `display_stock_chart(ticker: str, period: str = "6mo")`

Display interactive stock chart with metrics.

**Parameters:**
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `ticker` | `str` | - | Stock ticker symbol |
| `period` | `str` | `"6mo"` | Time period: "1mo", "3mo", "6mo", "1y" |

**Displays:**
- Company name and ticker
- Key metrics (price, volume, market cap, P/E)
- Candlestick chart with MA20/MA50
- Powered by yfinance and Plotly

---

## State Management

### Streamlit Session State

| Key | Type | Description |
|-----|------|-------------|
| `chat_history` | `list[dict]` | Chat tab message history |
| `thread_id` | `str` | Unique conversation thread ID |
| `news_chat_history` | `list[dict]` | News tab message history |
| `news_thread_id` | `str` | News conversation thread ID |
| `pending_prompt` | `str | None` | Pending chat message |
| `news_pending_prompt` | `str | None` | Pending news message |
| `conversation_threads` | `dict` | Thread ID mapping |

### Message Format

```python
{
    "role": "user" | "assistant",
    "content": str
}
```

---

## External API Integration

### OpenAI API

**Used for:** LLM inference, embeddings

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    streaming=True
)

# Embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)
```

---

### Alpha Vantage API

**Used for:** Stock prices, market data, news

```python
import requests

API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
BASE_URL = "https://www.alphavantage.co/query"

# Stock quote
response = requests.get(BASE_URL, params={
    "function": "GLOBAL_QUOTE",
    "symbol": "AAPL",
    "apikey": API_KEY
})

# Market news
response = requests.get(BASE_URL, params={
    "function": "NEWS_SENTIMENT",
    "tickers": "AAPL",
    "apikey": API_KEY
})
```

---

### Tavily API

**Used for:** News search

```python
from tavily import TavilyClient

client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

results = client.search(
    query="latest financial news",
    search_depth="advanced",
    max_results=5
)
```

---

### yfinance

**Used for:** Stock data, charts (Market Tab)

```python
import yfinance as yf

# Get stock info
stock = yf.Ticker("AAPL")
info = stock.info
history = stock.history(period="6mo")

# Available data
info["currentPrice"]      # Current price
info["marketCap"]         # Market capitalization
info["trailingPE"]        # P/E ratio
info["dividendYield"]     # Dividend yield
info["fiftyTwoWeekHigh"]  # 52-week high
info["fiftyTwoWeekLow"]   # 52-week low
```

---

## Error Handling

### Standard Error Response

All agents follow this error handling pattern:

```python
try:
    response = agent.invoke({"messages": messages})
    return response["messages"][-1].content
except Exception as e:
    return f"Error: {str(e)}"
```

### Error Types

| Error | Cause | Handling |
|-------|-------|----------|
| `OpenAIError` | API issues | Retry with backoff |
| `RateLimitError` | Too many requests | Wait and retry |
| `InvalidAPIKey` | Bad API key | Check configuration |
| `NetworkError` | Connection issues | Retry |
| `ValidationError` | Invalid input | Return error message |

---

## Rate Limits

### OpenAI

| Model | Requests/min | Tokens/min |
|-------|--------------|------------|
| gpt-4o-mini | 500 | 200,000 |
| text-embedding-3-small | 3,000 | 1,000,000 |

### Alpha Vantage

| Plan | Requests/min | Requests/day |
|------|--------------|--------------|
| Free | 5 | 500 |
| Premium | 75 | Unlimited |

### Tavily

| Plan | Requests/month |
|------|----------------|
| Free | 1,000 |
| Pro | 10,000 |
