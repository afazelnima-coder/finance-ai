# Finance Assistant

A multi-agent AI-powered finance assistant built with LangChain, LangGraph, and Streamlit. The application uses specialized agents to answer finance questions, provide market data, fetch news, and more.

## Features

- **Multi-Agent Architecture**: 6 specialized agents for different financial domains
- **Smart Routing**: LangGraph-based router that directs queries to the appropriate agent
- **AI Guardrails**: Off-topic question filtering using custom LLM-based validators
- **Interactive Charts**: Real-time stock charts with candlestick patterns and moving averages
- **Smart Ticker Recognition**: Natural language company lookup (e.g., "Apple" → AAPL)
- **RAG-Powered Q&A**: 500+ Investopedia articles for accurate finance answers
- **Streaming Responses**: Real-time response streaming for better UX
- **Observability**: LangSmith integration for tracing and debugging

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Streamlit UI                                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐                     │
│  │   Chat   │  │   News   │  │  Market  │  │  About   │                     │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └──────────┘                     │
└───────┼─────────────┼─────────────┼─────────────────────────────────────────┘
        │             │             │
        ▼             │             ▼
┌───────────────┐     │     ┌───────────────────────────────────┐
│   Guardrail   │     │     │         Market Tab                │
│    (LLM)      │     │     │  ┌─────────────┐ ┌─────────────┐  │
└───────┬───────┘     │     │  │  Guardrail  │ │   Ticker    │  │
        │             │     │  │    Check    │ │  Extractor  │  │
        ▼             │     │  └──────┬──────┘ └──────┬──────┘  │
┌───────────────┐     │     │         │              │          │
│ Router Agent  │     │     │         ▼              ▼          │
│  (LangGraph)  │     │     │  ┌─────────────┐ ┌─────────────┐  │
└───────┬───────┘     │     │  │   Market    │ │  yfinance   │  │
        │             │     │  │   Agent     │ │   + Plotly  │  │
        ▼             │     │  └─────────────┘ └─────────────┘  │
┌───────────────────────────┴───────────────────────────────────┘
│                    Specialized Agents
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌───────────┐
│  │   QA    │ │ Market  │ │  News   │ │   Tax   │ │  Goal   │ │ Portfolio │
│  │ (RAG)   │ │         │ │         │ │         │ │         │ │           │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └─────┬─────┘
│       │           │           │           │           │            │
│       ▼           ▼           ▼           ▼           ▼            ▼
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
│  │  FAISS  │ │ Alpha   │ │ Tavily/ │ │   LLM   │ │   LLM   │ │   LLM   │
│  │ VectorDB│ │Vantage  │ │Alpha V. │ │         │ │         │ │         │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘
└─────────────────────────────────────────────────────────────────────────────
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| **Frontend** | Streamlit |
| **AI Framework** | LangChain, LangGraph |
| **LLM** | OpenAI GPT-4o-mini |
| **Guardrails** | Guardrails AI (Custom Validators) |
| **Vector Database** | FAISS |
| **Embeddings** | OpenAI text-embedding-3-small |
| **Stock Data** | yfinance, Alpha Vantage API |
| **News** | Tavily API, Alpha Vantage News |
| **Charts** | Plotly |
| **Observability** | LangSmith |

## Project Structure

```
cap-proj/
├── agents/
│   ├── __init__.py
│   ├── router_agent_v2.py    # Main router with guardrails (LangGraph)
│   ├── qa_agent.py           # RAG-based Q&A agent
│   ├── market_agent.py       # Market data agent
│   ├── news_agent.py         # Financial news agent
│   ├── tax_agent.py          # Tax questions agent
│   ├── goal_agent.py         # Financial goals agent
│   └── portfolio_agent.py    # Portfolio management agent
├── rag/
│   └── vector_db_loader.py   # FAISS vector store loader
├── streamlit_app.py          # Main Streamlit application
├── main.py                   # CLI entry point
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables (not in repo)
└── README.md
```

## Setup

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd cap-proj
   ```

2. **Create virtual environment and install dependencies**

   Using uv (recommended):
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv pip install -r requirements.txt
   ```

   Or using pip:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Set up environment variables**

   Create a `.env` file in the project root:
   ```env
   # Required
   OPENAI_API_KEY='your-openai-api-key'
   TAVILY_API_KEY='your-tavily-api-key'
   ALPHA_VANTAGE_API_KEY='your-alpha-vantage-api-key'

   # LangSmith (optional but recommended)
   LANGSMITH_TRACING='true'
   LANGSMITH_API_KEY='your-langsmith-api-key'
   LANGSMITH_PROJECT='finance-assistant'
   ```

4. **Run the application**
   ```bash
   uv run streamlit run streamlit_app.py
   ```

   Or:
   ```bash
   streamlit run streamlit_app.py
   ```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key for LLM and embeddings |
| `TAVILY_API_KEY` | Yes | Tavily API key for news search |
| `ALPHA_VANTAGE_API_KEY` | Yes | Alpha Vantage API for market data and news |
| `LANGSMITH_TRACING` | No | Enable LangSmith tracing (`true`/`false`) |
| `LANGSMITH_API_KEY` | No | LangSmith API key for observability |
| `LANGSMITH_PROJECT` | No | Project name in LangSmith dashboard |

## Agents

### Router Agent (`router_agent_v2.py`)

The main orchestrator built with LangGraph. It:
1. Receives user queries
2. Runs guardrail validation (finance topic check)
3. Routes to the appropriate specialized agent
4. Returns the response

**Graph Structure:**
```
START → guardrail → [on-topic?] → router → [agent] → END
                  → [off-topic] → off_topic → END
```

### QA Agent (`qa_agent.py`)

- Uses RAG (Retrieval-Augmented Generation)
- Queries FAISS vector store with 500+ Investopedia articles
- Best for: "What is...", "Explain...", "Define..." questions

### Market Agent (`market_agent.py`)

- Fetches real-time market data
- Uses Alpha Vantage API
- Best for: Stock prices, market trends, company data

### News Agent (`news_agent.py`)

- Fetches latest financial news
- Uses Tavily and Alpha Vantage News APIs
- Best for: Current events, market news, company announcements

### Tax Agent (`tax_agent.py`)

- Answers tax-related questions
- Best for: Tax calculations, deductions, tax planning

### Goal Agent (`goal_agent.py`)

- Financial planning assistance
- Best for: Retirement planning, savings goals, budgeting

### Portfolio Agent (`portfolio_agent.py`)

- Portfolio management advice
- Best for: Asset allocation, diversification, investment strategies

## Guardrails

The application uses AI-powered guardrails to ensure all queries are finance-related:

### Implementation

```python
# Custom validator using LLM
@register_validator(name="finance_topic_validator", data_type="string")
class FinanceTopicValidator(Validator):
    def validate(self, value: str, metadata: Dict = {}) -> ValidationResult:
        # Uses GPT-4o-mini to classify if topic is finance-related
        ...
```

### Valid Topics

- Stocks, bonds, ETFs, mutual funds
- Investing, trading, portfolio management
- Taxes, retirement planning, budgeting
- Cryptocurrency, forex, commodities
- Financial news, market trends, economics

### Blocked Topics

- Cooking, recipes, food
- Sports, entertainment, movies
- Health, medical advice
- Travel, fashion, gaming
- General knowledge unrelated to finance

## Market Tab Features

### Smart Ticker Recognition

The Market tab uses an LLM to interpret natural language queries:

```
"How is Apple doing?" → AAPL
"Tesla stock price" → TSLA
"Show me S&P 500" → ^GSPC
"Nintendo" → NTDOY
```

### Supported Securities

- **US Stocks**: AAPL, MSFT, GOOGL, etc.
- **International ADRs**: SONY, TM, NTDOY, BABA, TSM
- **Market Indices**: ^GSPC (S&P 500), ^DJI (Dow), ^IXIC (Nasdaq)
- **ETFs**: SPY, QQQ, VOO, VTI, IWM
- **Sector ETFs**: XLK, XLF, XLE
- **Commodity ETFs**: GLD, SLV
- **Crypto ETFs**: IBIT, ETHA

### Chart Features

- Interactive candlestick charts
- 20-day and 50-day moving averages
- Configurable time periods (1mo, 3mo, 6mo, 1y)
- Key metrics: Price, Volume, Market Cap, P/E Ratio

## LangSmith Integration

When enabled, LangSmith provides:

- **Tracing**: See the full flow of agent calls
- **Latency**: Monitor response times
- **Token Usage**: Track LLM token consumption
- **Debugging**: Inspect inputs/outputs at each step

Access your traces at [smith.langchain.com](https://smith.langchain.com)

## Development

### Running Tests

```bash
# Test router agent directly
uv run python -m agents.router_agent_v2
```

### Project Commands

```bash
# Run Streamlit app
uv run streamlit run streamlit_app.py

# Run with Gradio UI (alternative)
uv run python agents/router_agent_gradio_ui.py

# Add new dependency
uv add <package-name>
```

### Adding a New Agent

1. Create `agents/new_agent.py`:
   ```python
   from langchain.agents import create_agent
   from langgraph.checkpoint.memory import InMemorySaver

   agent = create_agent(
       "gpt-4o-mini",
       tools=[...],
       checkpointer=InMemorySaver(),
       system_prompt="..."
   )
   ```

2. Register in `agents/__init__.py`

3. Add to router in `router_agent_v2.py`:
   - Add import
   - Create node function
   - Add to valid agents list
   - Add node and edges to graph

## API Keys

### OpenAI
Get your API key at [platform.openai.com](https://platform.openai.com/api-keys)

### Tavily
Sign up at [tavily.com](https://tavily.com) for news search API

### Alpha Vantage
Get a free API key at [alphavantage.co](https://www.alphavantage.co/support/#api-key)

### LangSmith
Sign up at [smith.langchain.com](https://smith.langchain.com) for observability
