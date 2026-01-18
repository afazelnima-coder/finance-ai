# Setup Guide

This guide provides comprehensive instructions for setting up the Finance Assistant application in development and production environments.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Detailed Installation](#detailed-installation)
4. [Environment Configuration](#environment-configuration)
5. [Database Setup](#database-setup)
6. [Running the Application](#running-the-application)
7. [Production Deployment](#production-deployment)
8. [Docker Setup](#docker-setup)

---

## Prerequisites

### System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| **Python** | 3.11 | 3.12+ |
| **RAM** | 4 GB | 8 GB+ |
| **Storage** | 2 GB | 5 GB+ |
| **OS** | macOS, Linux, Windows | macOS, Linux |

### Required Software

1. **Python 3.11+**
   ```bash
   # Check Python version
   python --version
   # or
   python3 --version
   ```

2. **uv (Recommended)** - Fast Python package manager
   ```bash
   # Install uv
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Verify installation
   uv --version
   ```

3. **Git**
   ```bash
   git --version
   ```

### Required API Keys

| Service | Purpose | Sign Up URL |
|---------|---------|-------------|
| **OpenAI** | LLM & Embeddings | [platform.openai.com](https://platform.openai.com/api-keys) |
| **Tavily** | News Search | [tavily.com](https://tavily.com) |
| **Alpha Vantage** | Market Data | [alphavantage.co](https://www.alphavantage.co/support/#api-key) |
| **LangSmith** (Optional) | Observability | [smith.langchain.com](https://smith.langchain.com) |

---

## Quick Start

For experienced developers, here's the fastest way to get running:

```bash
# Clone and enter directory
git clone <repository-url>
cd cap-proj

# Create environment and install dependencies
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

# Configure environment
cat > .env << 'EOF'
OPENAI_API_KEY='your-openai-key'
TAVILY_API_KEY='your-tavily-key'
ALPHA_VANTAGE_API_KEY='your-alphavantage-key'
EOF

# Create vector database
uv run python rag/vector_db_loader.py

# Run application
uv run streamlit run streamlit_app.py
```

---

## Detailed Installation

### Step 1: Clone the Repository

```bash
# HTTPS
git clone https://github.com/your-org/cap-proj.git

# SSH
git clone git@github.com:your-org/cap-proj.git

# Navigate to project
cd cap-proj
```

### Step 2: Create Virtual Environment

#### Using uv (Recommended)

```bash
# Create virtual environment
uv venv

# Activate (macOS/Linux)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate

# Verify activation
which python
# Should show: /path/to/cap-proj/.venv/bin/python
```

#### Using pip

```bash
# Create virtual environment
python -m venv .venv

# Activate (macOS/Linux)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate
```

### Step 3: Install Dependencies

#### Using uv

```bash
# Install from requirements.txt
uv pip install -r requirements.txt

# Or install with sync
uv pip sync requirements.txt
```

#### Using pip

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
# Check key packages
python -c "import langchain; print(f'LangChain: {langchain.__version__}')"
python -c "import streamlit; print(f'Streamlit: {streamlit.__version__}')"
python -c "import guardrails; print(f'Guardrails: {guardrails.__version__}')"
```

---

## Environment Configuration

### Create Environment File

Create a `.env` file in the project root:

```bash
touch .env
```

### Required Variables

```env
# =============================================================================
# REQUIRED CONFIGURATION
# =============================================================================

# OpenAI API Key
# Used for: LLM (GPT-4o-mini), Embeddings (text-embedding-3-small)
# Get from: https://platform.openai.com/api-keys
OPENAI_API_KEY='sk-proj-...'

# Tavily API Key
# Used for: News search, web search
# Get from: https://tavily.com
TAVILY_API_KEY='tvly-...'

# Alpha Vantage API Key
# Used for: Stock market data, financial news
# Get from: https://www.alphavantage.co/support/#api-key
ALPHA_VANTAGE_API_KEY='...'
```

### Optional Variables

```env
# =============================================================================
# OPTIONAL CONFIGURATION
# =============================================================================

# LangSmith (Observability & Tracing)
# Get from: https://smith.langchain.com
LANGSMITH_TRACING='true'
LANGSMITH_API_KEY='lsv2_pt_...'
LANGSMITH_PROJECT='finance-assistant'

# Advanced OpenAI Configuration
OPENAI_API_BASE='https://api.openai.com/v1'  # Custom endpoint
OPENAI_ORGANIZATION='org-...'                  # Organization ID
```

### Environment Variable Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes | - | OpenAI API authentication |
| `TAVILY_API_KEY` | Yes | - | Tavily news search API |
| `ALPHA_VANTAGE_API_KEY` | Yes | - | Market data API |
| `LANGSMITH_TRACING` | No | `false` | Enable LangSmith tracing |
| `LANGSMITH_API_KEY` | No | - | LangSmith authentication |
| `LANGSMITH_PROJECT` | No | `default` | LangSmith project name |

### Validate Configuration

```bash
# Check environment variables are loaded
python -c "
from dotenv import load_dotenv
import os
load_dotenv()

required = ['OPENAI_API_KEY', 'TAVILY_API_KEY', 'ALPHA_VANTAGE_API_KEY']
for var in required:
    value = os.getenv(var)
    if value:
        print(f'✓ {var}: {value[:10]}...')
    else:
        print(f'✗ {var}: NOT SET')
"
```

---

## Database Setup

### Vector Database (FAISS)

The QA Agent requires a FAISS vector database populated with Investopedia articles.

#### Create Vector Database

```bash
# Run the loader script
uv run python rag/vector_db_loader.py
```

#### Expected Output

```
Loaded 500 documents from Investopedia dataset
Split into 847 chunks
Creating embeddings with OpenAI...
Estimated cost: ~$0.008 (very low for 500 documents)
Creating vector store...
✓ Vector store created
✓ Vector store saved to 'investopedia_faiss_index'

============================================================
Testing RAG system with financial queries
============================================================

📊 Query: What is a stock option?
------------------------------------------------------------
[Result 1]
Topic: Options & Derivatives
Title: Stock Option
...
```

#### Verify Database Creation

```bash
# Check database files exist
ls -la investopedia_faiss_index/
# Expected output:
# index.faiss
# index.pkl
```

#### Troubleshooting Database Creation

| Issue | Cause | Solution |
|-------|-------|----------|
| `No module 'datasets'` | Missing dependency | `uv pip install datasets` |
| `OpenAI API error` | Invalid API key | Check `OPENAI_API_KEY` in `.env` |
| `Rate limit exceeded` | Too many requests | Wait and retry, or use different key |
| `Out of memory` | Large dataset | Reduce `MAX_DOCS` in script |

---

## Running the Application

### Development Mode

```bash
# Standard run
uv run streamlit run streamlit_app.py

# With specific port
uv run streamlit run streamlit_app.py --server.port 8080

# With hot reload disabled
uv run streamlit run streamlit_app.py --server.runOnSave false
```

### Access the Application

Open your browser and navigate to:
- **Local**: http://localhost:8501
- **Network**: http://[your-ip]:8501

### Verify All Components

1. **Chat Tab**: Ask "What is a stock?"
   - Should route to QA Agent
   - Should return RAG-based answer

2. **News Tab**: Click "Latest Headlines"
   - Should fetch recent financial news

3. **Market Tab**: Type "Apple"
   - Should show AAPL chart and metrics

4. **Off-topic Test**: Ask "What's a good recipe?"
   - Should be blocked by guardrails

---

## Production Deployment

### Streamlit Cloud

1. Push code to GitHub
2. Connect repository to [share.streamlit.io](https://share.streamlit.io)
3. Add secrets in Streamlit Cloud dashboard:
   ```toml
   OPENAI_API_KEY = "sk-..."
   TAVILY_API_KEY = "tvly-..."
   ALPHA_VANTAGE_API_KEY = "..."
   ```

### Self-Hosted (Systemd)

Create a systemd service file:

```bash
sudo nano /etc/systemd/system/finance-assistant.service
```

```ini
[Unit]
Description=Finance Assistant Streamlit App
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/finance-assistant
Environment="PATH=/opt/finance-assistant/.venv/bin"
ExecStart=/opt/finance-assistant/.venv/bin/streamlit run streamlit_app.py --server.port 8501 --server.headless true
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable finance-assistant
sudo systemctl start finance-assistant
sudo systemctl status finance-assistant
```

### Nginx Reverse Proxy

```nginx
server {
    listen 80;
    server_name finance.yourdomain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 86400;
    }

    location /_stcore/stream {
        proxy_pass http://localhost:8501/_stcore/stream;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 86400;
    }
}
```

---

## Docker Setup

### Dockerfile

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create vector database (requires OPENAI_API_KEY at build time)
ARG OPENAI_API_KEY
RUN python rag/vector_db_loader.py

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run application
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.headless=true"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  finance-assistant:
    build:
      context: .
      args:
        OPENAI_API_KEY: ${OPENAI_API_KEY}
    ports:
      - "8501:8501"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - TAVILY_API_KEY=${TAVILY_API_KEY}
      - ALPHA_VANTAGE_API_KEY=${ALPHA_VANTAGE_API_KEY}
      - LANGSMITH_TRACING=${LANGSMITH_TRACING:-false}
      - LANGSMITH_API_KEY=${LANGSMITH_API_KEY:-}
      - LANGSMITH_PROJECT=${LANGSMITH_PROJECT:-finance-assistant}
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### Build and Run

```bash
# Build image
docker-compose build

# Run container
docker-compose up -d

# View logs
docker-compose logs -f

# Stop container
docker-compose down
```

---

## Next Steps

- [API Documentation](./api.md) - Detailed API reference
- [Usage Guide](./usage.md) - How to use each feature
- [Troubleshooting](./troubleshooting.md) - Common issues and solutions
