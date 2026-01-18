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

## Scalability Considerations

| Component | Current | Scalable Alternative |
|-----------|---------|---------------------|
| Vector Store | FAISS (local) | Pinecone, Weaviate |
| Memory | InMemorySaver | Redis, PostgreSQL |
| LLM | OpenAI API | Self-hosted, Load balanced |
| Frontend | Single Streamlit | Multiple instances + LB |
