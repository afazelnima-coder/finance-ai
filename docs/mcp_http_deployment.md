# MCP HTTP Server — Deployment Guide

This guide covers three deployment scenarios:
- **Local development** — run directly with `uv`
- **EC2 (shared instance)** — Docker Compose alongside an existing project
- **EC2 (fresh instance)** — full setup from scratch

---

## Required API Keys

All deployments need a `.env` file. Copy the template and fill in your values:

```bash
cp .env.example .env
```

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes | Powers all LLM calls |
| `TAVILY_API_KEY` | Yes | Web search for news and expense ratio fallback |
| `ALPHA_VANTAGE_API_KEY` | Yes | Market data (news sentiment) |
| `LANGSMITH_TRACING` | No | Set to `true` to enable LangSmith tracing |
| `LANGSMITH_API_KEY` | No | LangSmith project API key |
| `LANGSMITH_PROJECT` | No | LangSmith project name |

---

## Option A — Local Development (uv)

### 1. Install dependencies

```bash
uv sync
```

### 2. Start the MCP HTTP server

```bash
uv run uvicorn mcp_http_server:app --host 0.0.0.0 --port 8001
```

### 3. Start the Streamlit app (separate terminal)

```bash
uv run streamlit run streamlit_app.py --server.port 8502
```

### 4. Test the MCP server

```bash
uv run python test_mcp_tools.py
```

### 5. Register with Claude Code CLI

```bash
claude mcp add --transport sse finance-assistant http://localhost:8001/sse
```

---

## Option B — EC2 Shared Instance (existing Docker Compose project on same server)

This project uses ports **8001** (MCP) and **8502** (Streamlit) to avoid conflicting
with the existing project on ports 8000/8501.

### 1. SSH into the instance

```bash
ssh -i ~/.ssh/your-key.pem ubuntu@<ec2-ip>
```

### 2. Clone the repo

```bash
git clone <repo-url> cap-proj
cd cap-proj
```

### 3. Create the `.env` file

```bash
cp .env.example .env
nano .env   # fill in OPENAI_API_KEY, TAVILY_API_KEY, ALPHA_VANTAGE_API_KEY
```

### 4. Open ports in the EC2 security group

In the AWS Console → EC2 → Security Groups → Inbound rules, add:

| Type | Port | Source |
|---|---|---|
| Custom TCP | 8001 | 0.0.0.0/0 (or your IP only) |
| Custom TCP | 8502 | 0.0.0.0/0 (or your IP only) |

### 5. Start the services

```bash
docker-compose -f docker-compose.http.yml up -d --build
```

Both containers start independently of the existing project's containers. Verify:

```bash
docker-compose -f docker-compose.http.yml ps
```

Expected output:
```
NAME                    STATUS    PORTS
cap-proj-web-1          running   0.0.0.0:8502->8502/tcp
cap-proj-mcp-server-1   running   0.0.0.0:8001->8001/tcp
```

### 6. Register with Claude Code CLI (from your local machine)

```bash
claude mcp add --transport sse finance-assistant http://<ec2-ip>:8001/sse
```

Verify:
```bash
claude mcp list
```

### Updating after a code change

```bash
ssh -i ~/.ssh/your-key.pem ubuntu@<ec2-ip>
cd cap-proj
git pull
docker-compose -f docker-compose.http.yml up -d --build
```

### Stopping the services

```bash
docker-compose -f docker-compose.http.yml down
```

This only stops this project's containers — the existing project is unaffected.

---

## Option C — EC2 Fresh Instance (Docker not installed)

### 1. Install Docker

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
  https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" \
  -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Allow running docker without sudo
sudo usermod -aG docker $USER
newgrp docker
```

### 2. Continue from Option B, step 2

---

## Available MCP Tools

Once registered, Claude has access to these tools:

| Tool | Description |
|---|---|
| `get_market_data` | Real-time price, change, volume, and 52-week range for any ticker |
| `get_market_overview` | Live snapshot of S&P 500, Dow Jones, Nasdaq, Russell 2000, VIX |
| `analyze_portfolio` | Full portfolio analysis from a natural language description |
| `lookup_expense_ratio` | Expense ratio lookup for 50+ funds, with web search fallback |
| `extract_ticker` | Resolve a company name to its ticker symbol |

---

## Local stdio Transport (Claude Desktop)

For local use without a server process, use the stdio server instead.

Register with Claude Code CLI:
```bash
claude mcp add finance-assistant-stdio python /path/to/cap-proj/mcp_server.py
```

Or add to `~/.claude/claude_desktop_config.json` manually:
```json
{
  "mcpServers": {
    "finance-assistant": {
      "command": "python",
      "args": ["/path/to/cap-proj/mcp_server.py"]
    }
  }
}
```

---

## Caching

All five MCP tools are wrapped with in-memory TTL caches (`cachetools.TTLCache`) defined
in `utils/mcp_cache.py`. On a cache **hit** the result is returned instantly with no
external API call. On a **miss** the live function runs and the result is stored for the
TTL duration.

| Tool | TTL | Cache key |
|---|---|---|
| `get_market_data` | 60s | ticker symbol (uppercased) |
| `get_market_overview` | 60s | fixed key `"overview"` |
| `analyze_portfolio` | 5 min | portfolio description (lowercased + stripped) |
| `lookup_expense_ratio` | 1 hour | fund identifier (uppercased + stripped) |
| `extract_ticker` | 24 hours | query (lowercased + stripped) |

### Tuning TTLs

Edit `utils/mcp_cache.py` and change the `ttl=` parameter for the relevant cache:

```python
# Example: shorten market data freshness to 30 seconds
_market_data_cache: TTLCache = TTLCache(maxsize=128, ttl=30)
```

Rebuild the container after any change:
```bash
docker-compose -f docker-compose.http.yml up -d --build mcp-server
```

### Cache limits

Each cache has a `maxsize` that caps the number of entries. When full, the
least-recently-used entry is evicted regardless of TTL. Increase `maxsize` if you
serve many distinct tickers or portfolio descriptions:

```python
_market_data_cache: TTLCache = TTLCache(maxsize=512, ttl=60)
```

### Scaling beyond a single process

The TTL cache lives in-process. If you scale the MCP server to multiple uvicorn workers
or replicas, each process has its own independent cache. To share cache state across
processes, replace `TTLCache` with a Redis backend (`redis-py` + `setex`/`get`), and add
a Redis service to `docker-compose.http.yml`.

---

## Troubleshooting

**`No module named X` when starting the server**
The Docker image installs from `requirements.txt`. Make sure any new dependency is listed there before rebuilding.

**`port is already allocated`**
Another container or process is using the port. Check with:
```bash
sudo lsof -i :8001
sudo lsof -i :8502
```

**MCP tools not showing in Claude**
Confirm the server is reachable first:
```bash
curl http://<ec2-ip>:8001/sse
# Should start streaming an SSE response
```
Then re-run `claude mcp list` to confirm registration.

**Container exits immediately**
Check logs:
```bash
docker-compose -f docker-compose.http.yml logs mcp-server
docker-compose -f docker-compose.http.yml logs web
```
Most common cause: missing or malformed `.env` file.
