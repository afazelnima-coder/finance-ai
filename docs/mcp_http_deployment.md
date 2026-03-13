# MCP HTTP Server — Deployment Guide

This guide covers deploying the finance assistant's MCP HTTP server, either locally or via Docker Compose.

---

## Prerequisites

- Python 3.13+ (for local) or Docker + Docker Compose (for containerised)
- A `.env` file in the project root with all required API keys:

```
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
ALPHA_VANTAGE_API_KEY=...
LANGCHAIN_API_KEY=...          # optional, for LangSmith tracing
LANGCHAIN_TRACING_V2=true      # optional
```

---

## Option A — Local (no Docker)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the MCP HTTP server

```bash
uvicorn mcp_http_server:app --host 0.0.0.0 --port 8000
```

The server listens at `http://localhost:8000/sse`.

### 3. (Optional) Start the Streamlit app in a separate terminal

```bash
streamlit run streamlit_app.py
```

---

## Option B — Docker Compose

### 1. Build and start both services

```bash
docker-compose -f docker-compose.http.yml up --build
```

This starts two containers:

| Service | URL |
|---|---|
| Streamlit web app | http://localhost:8501 |
| MCP HTTP server | http://localhost:8000/sse |

### 2. Stop services

```bash
docker-compose -f docker-compose.http.yml down
```

---

## Registering with Claude Code CLI

```bash
claude mcp add --transport sse finance-assistant http://localhost:8000/sse
```

To verify registration:

```bash
claude mcp list
```

Claude will now have access to these tools:

| Tool | Description |
|---|---|
| `get_market_data` | Real-time price, change, volume for any ticker |
| `get_market_overview` | Live snapshot of S&P 500, Dow, Nasdaq, VIX |
| `analyze_portfolio` | Full portfolio analysis from natural language |
| `lookup_expense_ratio` | Fund expense ratio lookup |
| `extract_ticker` | Natural language → ticker symbol |

---

## Testing the MCP server

```bash
# With the server running on port 8000:
python test_mcp_tools.py
```

Expected output:

```
Connecting to SSE stream …
  Session endpoint: /messages/?session_id=<uuid>

Sending initialize …
  Server info: {'name': 'finance-assistant', 'version': '1.0.0'}

Sending tools/list …
==================================================
  Found 5 MCP tool(s):
==================================================
  Tool: get_market_data
  ...
All tests passed!
```

---

## Local stdio server (for Claude Desktop)

For local use without a running server process, use the stdio transport instead:

```bash
# .mcp.json entry for Claude Desktop:
{
  "mcpServers": {
    "finance-assistant-stdio": {
      "command": "python",
      "args": ["/path/to/cap-proj/mcp_server.py"]
    }
  }
}
```

Or register directly:

```bash
claude mcp add finance-assistant-stdio python /path/to/cap-proj/mcp_server.py
```
