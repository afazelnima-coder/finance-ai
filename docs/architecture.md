# Architecture Documentation

## Overview

Finance Assistant is a multi-agent AI system designed to provide comprehensive financial information and analysis. The system uses a microservices-inspired architecture where specialized agents handle specific domains of financial queries.

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                   CLIENT LAYER                                   │
│  ┌─────────────────────────────────────────────────────────────────────────────┐│
│  │                         Streamlit Web Application                           ││
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐       ││
│  │  │  Chat Tab    │ │  News Tab    │ │ Market Tab   │ │  About Tab   │       ││
│  │  │              │ │              │ │              │ │              │       ││
│  │  │ - Streaming  │ │ - Quick      │ │ - Ticker     │ │ - Info       │       ││
│  │  │ - History    │ │   Actions    │ │   Lookup     │ │ - Tech Stack │       ││
│  │  │ - Routing    │ │ - Chat UI    │ │ - Charts     │ │              │       ││
│  │  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘       ││
│  └─────────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                               GUARDRAIL LAYER                                    │
│  ┌─────────────────────────────────────────────────────────────────────────────┐│
│  │                        Finance Topic Validator                               ││
│  │                                                                              ││
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                      ││
│  │  │   Input     │───▶│  LLM-based  │───▶│  Pass/Fail  │                      ││
│  │  │   Query     │    │  Classifier │    │  Decision   │                      ││
│  │  └─────────────┘    └─────────────┘    └─────────────┘                      ││
│  │                                              │                               ││
│  │                     ┌────────────────────────┴────────────────────┐         ││
│  │                     ▼                                             ▼         ││
│  │            ┌─────────────┐                              ┌─────────────┐     ││
│  │            │  On-Topic   │                              │ Off-Topic   │     ││
│  │            │  Continue   │                              │  Decline    │     ││
│  │            └─────────────┘                              └─────────────┘     ││
│  └─────────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                ROUTING LAYER                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────┐│
│  │                     LangGraph Router Agent                                   ││
│  │                                                                              ││
│  │  ┌─────────────────────────────────────────────────────────────────────┐    ││
│  │  │                        State Machine                                 │    ││
│  │  │                                                                      │    ││
│  │  │   START ──▶ guardrail ──▶ router ──▶ [agent_node] ──▶ END           │    ││
│  │  │                  │                                                   │    ││
│  │  │                  └──────▶ off_topic ──────────────────▶ END         │    ││
│  │  │                                                                      │    ││
│  │  └─────────────────────────────────────────────────────────────────────┘    ││
│  │                                                                              ││
│  │  Router Decision Logic:                                                      ││
│  │  ┌─────────────┐                                                            ││
│  │  │ "what is"   │ ──▶ QA Agent                                               ││
│  │  │ "explain"   │                                                            ││
│  │  ├─────────────┤                                                            ││
│  │  │ "price"     │ ──▶ Market Agent                                           ││
│  │  │ "stock"     │                                                            ││
│  │  ├─────────────┤                                                            ││
│  │  │ "news"      │ ──▶ News Agent                                             ││
│  │  │ "headlines" │                                                            ││
│  │  ├─────────────┤                                                            ││
│  │  │ "tax"       │ ──▶ Tax Agent                                              ││
│  │  │ "deduction" │                                                            ││
│  │  ├─────────────┤                                                            ││
│  │  │ "goal"      │ ──▶ Goal Agent                                             ││
│  │  │ "retire"    │                                                            ││
│  │  ├─────────────┤                                                            ││
│  │  │ "portfolio" │ ──▶ Portfolio Agent                                        ││
│  │  │ "allocate"  │                                                            ││
│  │  └─────────────┘                                                            ││
│  └─────────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                 AGENT LAYER                                      │
│                                                                                  │
│  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐ ┌───────────────┐        │
│  │   QA Agent    │ │ Market Agent  │ │  News Agent   │ │   Tax Agent   │        │
│  │               │ │               │ │               │ │               │        │
│  │ ┌───────────┐ │ │ ┌───────────┐ │ │ ┌───────────┐ │ │ ┌───────────┐ │        │
│  │ │    RAG    │ │ │ │   Tools   │ │ │ │   Tools   │ │ │ │    LLM    │ │        │
│  │ │  Pipeline │ │ │ │           │ │ │ │           │ │ │ │  Direct   │ │        │
│  │ └─────┬─────┘ │ │ └─────┬─────┘ │ │ └─────┬─────┘ │ │ └───────────┘ │        │
│  │       │       │ │       │       │ │       │       │ │               │        │
│  │       ▼       │ │       ▼       │ │       ▼       │ │               │        │
│  │ ┌───────────┐ │ │ ┌───────────┐ │ │ ┌───────────┐ │ │               │        │
│  │ │   FAISS   │ │ │ │  yFinance │ │ │ │  Tavily   │ │ │               │        │
│  │ │  Vector   │ │ │ │           │ │ │ │ Alpha V   │ │ │               │        │
│  │ │    DB     │ │ │ │    API    │ │ │ │   APIs    │ │ │               │        │
│  │ └───────────┘ │ │ └───────────┘ │ │ └───────────┘ │ │               │        │
│  └───────────────┘ └───────────────┘ └───────────────┘ └───────────────┘        │
│                                                                                  │
│  ┌───────────────┐ ┌───────────────┐                                            │
│  │  Goal Agent   │ │Portfolio Agent│                                            │
│  │               │ │               │                                            │
│  │ ┌───────────┐ │ │ ┌───────────┐ │                                            │
│  │ │    LLM    │ │ │ │    LLM    │ │                                            │
│  │ │  Direct   │ │ │ │  Direct   │ │                                            │
│  │ └───────────┘ │ │ └───────────┘ │                                            │
│  └───────────────┘ └───────────────┘                                            │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              EXTERNAL SERVICES                                   │
│                                                                                  │
│  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐ ┌───────────────┐        │
│  │    OpenAI     │ │ Alpha Vantage │ │    Tavily     │ │   yfinance    │        │
│  │               │ │               │ │               │ │               │        │
│  │ - GPT-4o-mini │ │ - Stock Data  │ │ - News Search │ │ - Stock Data  │        │
│  │ - Embeddings  │ │ - Market News │ │ - Web Search  │ │ - Charts      │        │
│  └───────────────┘ └───────────────┘ └───────────────┘ └───────────────┘        │
│                                                                                  │
│  ┌───────────────┐ ┌───────────────┐                                            │
│  │  HuggingFace  │ │   LangSmith   │                                            │
│  │               │ │               │                                            │
│  │ - Investopedia│ │ - Tracing     │                                            │
│  │   Dataset     │ │ - Monitoring  │                                            │
│  └───────────────┘ └───────────────┘                                            │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Client Layer (Streamlit)

The frontend is built with Streamlit and provides four main tabs:

| Tab | Purpose | Key Features |
|-----|---------|--------------|
| **Chat** | Main conversation interface | Streaming responses, chat history, agent routing indicators |
| **News** | Financial news queries | Quick action buttons, dedicated news chat, streaming |
| **Market** | Stock lookup and analysis | Smart ticker recognition, interactive charts, metrics |
| **About** | System information | Architecture overview, tech stack, feature list |

### 2. Guardrail Layer

The guardrail layer ensures all queries are finance-related before processing.

```python
# Guardrail Flow
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Query     │────▶│  Validator  │────▶│  Decision   │
│   Input     │     │   (LLM)     │     │             │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
                    ┌──────────────────────────┴──────────────────────────┐
                    │                                                      │
                    ▼                                                      ▼
           ┌─────────────┐                                        ┌─────────────┐
           │   PASS      │                                        │   FAIL      │
           │  (Finance)  │                                        │ (Off-topic) │
           └──────┬──────┘                                        └──────┬──────┘
                  │                                                      │
                  ▼                                                      ▼
           ┌─────────────┐                                        ┌─────────────┐
           │  Continue   │                                        │   Return    │
           │  to Router  │                                        │   Warning   │
           └─────────────┘                                        └─────────────┘
```

**Implementation:**
- Uses GPT-4o-mini for classification
- Validates against predefined topic lists
- Cached LLM instance for performance

### 3. Routing Layer (LangGraph)

The router uses LangGraph's StateGraph for deterministic flow control.

```python
# State Definition
class State(TypedDict):
    messages: Annotated[list, add_messages]
    next_agent: str
    is_on_topic: bool
```

**Graph Nodes:**
| Node | Function | Description |
|------|----------|-------------|
| `guardrail` | `guardrail_node()` | Validates topic relevance |
| `off_topic` | `off_topic_node()` | Returns decline message |
| `router` | `router_node()` | Determines target agent |
| `qa` | `qa_node()` | Invokes QA agent |
| `market` | `market_node()` | Invokes Market agent |
| `news` | `news_node()` | Invokes News agent |
| `tax` | `tax_node()` | Invokes Tax agent |
| `goal` | `goal_node()` | Invokes Goal agent |
| `portfolio` | `portfolio_node()` | Invokes Portfolio agent |

**Graph Edges:**
```
START ──────────────▶ guardrail
guardrail ──(on_topic)──▶ router
guardrail ──(off_topic)─▶ off_topic
router ──(qa)──────────▶ qa ─────────▶ END
router ──(market)──────▶ market ─────▶ END
router ──(news)────────▶ news ───────▶ END
router ──(tax)─────────▶ tax ────────▶ END
router ──(goal)────────▶ goal ───────▶ END
router ──(portfolio)───▶ portfolio ──▶ END
off_topic ─────────────────────────────▶ END
```

### 4. Agent Layer

Each agent is specialized for a specific domain:

#### QA Agent (RAG-based)
```
┌─────────────────────────────────────────────────────┐
│                     QA Agent                         │
│                                                      │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐         │
│  │  Query  │───▶│Retriever│───▶│   LLM   │         │
│  └─────────┘    └────┬────┘    └────┬────┘         │
│                      │              │               │
│                      ▼              ▼               │
│                ┌─────────┐    ┌─────────┐          │
│                │  FAISS  │    │ Answer  │          │
│                │ VectorDB│    │         │          │
│                └─────────┘    └─────────┘          │
└─────────────────────────────────────────────────────┘
```

- **Data Source**: 500 Investopedia articles
- **Embedding Model**: OpenAI text-embedding-3-small
- **Vector Store**: FAISS (local)
- **Retrieval**: Top-k similarity search (k=3)

#### Market Agent (API-based)
```
┌─────────────────────────────────────────────────────┐
│                   Market Agent                       │
│                                                      │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐         │
│  │  Query  │───▶│  Tools  │───▶│   LLM   │         │
│  └─────────┘    └────┬────┘    └────┬────┘         │
│                      │              │               │
│                      ▼              ▼               │
│                ┌─────────┐    ┌─────────┐          │
│                │  Alpha  │    │Response │          │
│                │ Vantage │    │         │          │
│                └─────────┘    └─────────┘          │
└─────────────────────────────────────────────────────┘
```

- **Tools**: Stock quote, market overview
- **Data Provider**: Alpha Vantage API
- **Capabilities**: Real-time prices, company data

#### News Agent (Search-based)
```
┌─────────────────────────────────────────────────────┐
│                    News Agent                        │
│                                                      │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐         │
│  │  Query  │───▶│  Tools  │───▶│   LLM   │         │
│  └─────────┘    └────┬────┘    └────┬────┘         │
│                      │              │               │
│               ┌──────┴──────┐       │               │
│               ▼             ▼       ▼               │
│         ┌─────────┐   ┌─────────┐ ┌─────────┐      │
│         │ Tavily  │   │  Alpha  │ │Response │      │
│         │   API   │   │Vantage  │ │         │      │
│         └─────────┘   └─────────┘ └─────────┘      │
└─────────────────────────────────────────────────────┘
```

- **Primary**: Tavily for news search
- **Fallback**: Alpha Vantage news API
- **Capabilities**: Real-time news, topic-specific search

### 5. Data Flow

#### Request Flow
```
User Input
    │
    ▼
┌─────────────────┐
│   Streamlit     │
│   Frontend      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Guardrail     │──────▶ Off-topic? ──▶ Return Warning
│   Validator     │
└────────┬────────┘
         │ On-topic
         ▼
┌─────────────────┐
│     Router      │
│   (LangGraph)   │
└────────┬────────┘
         │
    ┌────┴────┬────────┬────────┬────────┬────────┐
    ▼         ▼        ▼        ▼        ▼        ▼
┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐
│  QA  │ │Market│ │ News │ │ Tax  │ │ Goal │ │Port- │
│Agent │ │Agent │ │Agent │ │Agent │ │Agent │ │folio │
└──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘
   │        │        │        │        │        │
   └────────┴────────┴────────┴────────┴────────┘
                          │
                          ▼
                    ┌───────────┐
                    │  Response │
                    │ (Streamed)│
                    └───────────┘
```

#### Streaming Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                      Streaming Pipeline                          │
│                                                                  │
│  Agent ──▶ astream_events() ──▶ Filter ──▶ UI Update            │
│                                    │                             │
│                                    ▼                             │
│                            ┌─────────────┐                       │
│                            │   Filter:   │                       │
│                            │ - Skip      │                       │
│                            │   router    │                       │
│                            │ - Skip      │                       │
│                            │   guardrail │                       │
│                            │ - Pass      │                       │
│                            │   agent     │                       │
│                            │   tokens    │                       │
│                            └─────────────┘                       │
└─────────────────────────────────────────────────────────────────┘
```

## Memory Management

### Conversation Memory

Each session maintains conversation history using LangGraph's checkpointer:

```python
checkpointer = InMemorySaver()
agent = workflow.compile(checkpointer=checkpointer)

# Thread-based memory
config = {"configurable": {"thread_id": "unique-thread-id"}}
```

### Session State (Streamlit)

```python
# Chat Tab
st.session_state.chat_history      # List of messages
st.session_state.thread_id         # Unique conversation ID

# News Tab
st.session_state.news_chat_history # News-specific history
st.session_state.news_thread_id    # News conversation ID

# Shared
st.session_state.conversation_threads  # Thread mapping
```

## Caching Strategy

### Streamlit App Cache

```python
# LLM Instance Caching
@st.cache_resource
def get_guardrail_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Vector Store Caching
@lru_cache
def get_vectorstore():
    return FAISS.load_local("investopedia_faiss_index", embeddings)
```

### MCP Server Cache (`utils/mcp_cache.py`)

All five MCP tools are wrapped with per-tool `TTLCache` instances (from `cachetools`).
Results are keyed on normalised inputs so identical queries never hit the external API twice
within the TTL window.

```
MCP Client request
       │
       ▼
┌──────────────────────────────────────────────┐
│              call_tool()                      │
│                                               │
│  cache key = normalise(arguments)             │
│       │                                       │
│       ▼                                       │
│  ┌─────────┐   HIT   ┌──────────────────┐    │
│  │ TTLCache│────────▶│  Cached result   │    │
│  └────┬────┘         └──────────────────┘    │
│       │ MISS                                  │
│       ▼                                       │
│  ┌─────────────────┐                          │
│  │  Agent function │  (yfinance / OpenAI /    │
│  │  (live call)    │   Tavily / hardcoded)    │
│  └────────┬────────┘                          │
│           │                                   │
│           ▼                                   │
│  ┌─────────────────┐                          │
│  │  Store in cache │                          │
│  │  with TTL       │                          │
│  └────────┬────────┘                          │
│           │                                   │
│           ▼                                   │
│  ┌──────────────────┐                         │
│  │  Return result   │                         │
│  └──────────────────┘                         │
└──────────────────────────────────────────────┘
```

| Tool | Cache key | TTL | Max entries | Rationale |
|---|---|---|---|---|
| `get_market_data` | `symbol.upper()` | 60s | 128 | Price ticks per second; 1-min staleness acceptable |
| `get_market_overview` | `"overview"` (fixed) | 60s | 1 | Single no-arg call |
| `analyze_portfolio` | `description.strip().lower()` | 300s | 64 | Expensive LLM call; same input → same output |
| `lookup_expense_ratio` | `fund.upper().strip()` | 3600s | 128 | Expense ratios change quarterly |
| `extract_ticker` | `query.strip().lower()` | 86400s | 256 | Company→ticker mapping is static |

**Thread safety note:** `TTLCache` is not thread-safe. This is safe for a single-worker
uvicorn process (all async I/O runs on one event-loop thread). Add a `threading.Lock`
if you run multiple uvicorn workers.

## Error Handling

```
┌─────────────────────────────────────────────────────────────────┐
│                     Error Handling Flow                          │
│                                                                  │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐      │
│  │  Try    │───▶│ Execute │───▶│ Success │───▶│ Return  │      │
│  │         │    │         │    │         │    │ Result  │      │
│  └─────────┘    └────┬────┘    └─────────┘    └─────────┘      │
│                      │                                          │
│                      │ Exception                                │
│                      ▼                                          │
│                ┌─────────┐    ┌─────────┐                       │
│                │  Log    │───▶│ Return  │                       │
│                │  Error  │    │ Message │                       │
│                └─────────┘    └─────────┘                       │
└─────────────────────────────────────────────────────────────────┘
```

## Security Considerations

1. **API Key Management**: Environment variables, never committed
2. **Input Validation**: Guardrails prevent off-topic/malicious queries
3. **Rate Limiting**: Dependent on external API limits
4. **No PII Storage**: Conversations are session-only (InMemorySaver)

## MCP Server Layer

The MCP server exposes the finance assistant's core tools over HTTP/SSE, allowing Claude
and other MCP clients to call them directly without going through the Streamlit UI.

```
┌─────────────────────────────────────────────────────────────────┐
│                        MCP Clients                               │
│   Claude Code CLI  │  Claude Desktop  │  Any MCP-compatible LLM │
└────────────────────┴──────────────────┴─────────────────────────┘
                                │
                    SSE / HTTP (port 8001)
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                  mcp_http_server.py (uvicorn)                    │
│                                                                  │
│  GET /sse  ──▶  open SSE stream, send session_id                │
│  POST /messages/?session_id=…  ──▶  JSON-RPC dispatch           │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    call_tool()                            │   │
│  │                                                          │   │
│  │   get_market_data  │  get_market_overview                │   │
│  │   analyze_portfolio│  lookup_expense_ratio               │   │
│  │   extract_ticker                                         │   │
│  │            │                                             │   │
│  │            ▼                                             │   │
│  │   utils/mcp_cache.py  (TTLCache per tool)                │   │
│  │            │                                             │   │
│  │            ▼                                             │   │
│  │   agents/ + utils/  (live call on cache miss)            │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

A stdio variant (`mcp_server.py`) is also available for local Claude Desktop integration.

---

## Scalability Considerations

| Component | Current | Scalable Alternative |
|-----------|---------|---------------------|
| Vector Store | FAISS (local) | Pinecone, Weaviate |
| Memory | InMemorySaver | Redis, PostgreSQL |
| LLM | OpenAI API | Self-hosted, Load balanced |
| Frontend | Single Streamlit | Multiple instances + LB |
| MCP cache | `cachetools` TTLCache (in-process) | Redis (shared across replicas) |
