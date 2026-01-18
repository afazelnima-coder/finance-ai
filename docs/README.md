# Documentation Index

Welcome to the Finance Assistant documentation.

## Quick Links

| Document | Description |
|----------|-------------|
| [Architecture](./architecture.md) | System design, data flow, component details |
| [Setup Guide](./setup.md) | Installation, configuration, deployment |
| [API Reference](./api.md) | Agent APIs, functions, parameters |
| [Usage Guide](./usage.md) | Examples, tutorials, best practices |
| [Troubleshooting](./troubleshooting.md) | Common issues and solutions |
| [Development](./development.md) | Contributing, adding agents, testing |

## Overview

Finance Assistant is a multi-agent AI system built with:

- **LangChain & LangGraph** - Agent orchestration
- **Guardrails AI** - Input validation
- **Streamlit** - Web interface
- **OpenAI** - LLM & embeddings
- **FAISS** - Vector search
- **yfinance & Alpha Vantage** - Market data

## Getting Started

1. **New users**: Start with the [Setup Guide](./setup.md)
2. **Using the app**: See the [Usage Guide](./usage.md)
3. **Having issues**: Check [Troubleshooting](./troubleshooting.md)
4. **Contributing**: Read the [Development Guide](./development.md)

## Architecture at a Glance

```
User Query → Guardrail → Router → Specialized Agent → Response
                ↓
           Off-topic? → Decline
```

## Support

- GitHub Issues: Report bugs and request features
- Documentation: This docs folder
- LangSmith: Enable tracing for debugging
