# Troubleshooting Guide

This guide helps you diagnose and resolve common issues with the Finance Assistant.

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [API Key Problems](#api-key-problems)
3. [Vector Database Issues](#vector-database-issues)
4. [Runtime Errors](#runtime-errors)
5. [Agent-Specific Issues](#agent-specific-issues)
6. [Performance Issues](#performance-issues)
7. [UI Issues](#ui-issues)

---

## Installation Issues

### Issue: `ModuleNotFoundError`

**Symptom:**
```
ModuleNotFoundError: No module named 'langchain'
```

**Solutions:**

1. Ensure virtual environment is activated:
   ```bash
   source .venv/bin/activate
   ```

2. Reinstall dependencies:
   ```bash
   uv pip install -r requirements.txt
   ```

3. Check Python version:
   ```bash
   python --version  # Should be 3.11+
   ```

---

### Issue: Dependency Conflicts

**Symptom:**
```
ERROR: Cannot install package-a and package-b because of conflicting dependencies
```

**Solutions:**

1. Clear cache and reinstall:
   ```bash
   uv cache clean
   uv pip install -r requirements.txt --force-reinstall
   ```

2. Create fresh environment:
   ```bash
   rm -rf .venv
   uv venv
   source .venv/bin/activate
   uv pip install -r requirements.txt
   ```

---

## API Key Problems

### Issue: Invalid OpenAI API Key

**Symptom:**
```
openai.AuthenticationError: Invalid API key
```

**Solutions:**

1. Verify key format (should start with `sk-`):
   ```bash
   grep OPENAI_API_KEY .env
   ```

2. Check for extra spaces or quotes:
   ```env
   # Correct
   OPENAI_API_KEY='sk-proj-...'
   
   # Wrong (extra space)
   OPENAI_API_KEY= 'sk-proj-...'
   ```

3. Test key directly:
   ```python
   import openai
   openai.api_key = "your-key"
   openai.models.list()
   ```

---

### Issue: Rate Limit Exceeded

**Symptom:**
```
openai.RateLimitError: Rate limit reached
```

**Solutions:**

1. Wait a few minutes and retry
2. Check your OpenAI usage dashboard
3. Upgrade your OpenAI plan if needed
4. Implement exponential backoff (already in agents)

---

### Issue: Alpha Vantage API Limit

**Symptom:**
```
Note: API call frequency limit reached (5 calls/minute)
```

**Solutions:**

1. Wait 60 seconds between requests
2. Upgrade to premium API key
3. Use yfinance as fallback (Market tab does this)

---

## Vector Database Issues

### Issue: FAISS Index Not Found

**Symptom:**
```
RuntimeError: No such file or directory: 'investopedia_faiss_index'
```

**Solution:**

Create the vector database:
```bash
uv run python rag/vector_db_loader.py
```

---

### Issue: FAISS Index Corrupted

**Symptom:**
```
RuntimeError: Error loading FAISS index
```

**Solutions:**

1. Delete and recreate:
   ```bash
   rm -rf investopedia_faiss_index/
   uv run python rag/vector_db_loader.py
   ```

2. Check disk space:
   ```bash
   df -h
   ```

---

### Issue: allow_dangerous_deserialization Error

**Symptom:**
```
ValueError: The de-serialization relies on loading a pickle file...
```

**Solution:**

Update the FAISS loading code:
```python
vectorstore = FAISS.load_local(
    "investopedia_faiss_index",
    embeddings,
    allow_dangerous_deserialization=True  # Add this
)
```

---

## Runtime Errors

### Issue: Guardrail Validation Failed

**Symptom:**
All queries return "off-topic" message.

**Solutions:**

1. Check LLM is responding:
   ```python
   from langchain_openai import ChatOpenAI
   llm = ChatOpenAI(model="gpt-4o-mini")
   print(llm.invoke("Hello"))
   ```

2. Verify guardrail function:
   ```python
   from streamlit_app import is_finance_related
   print(is_finance_related("What is a stock?"))  # Should be True
   ```

---

### Issue: Agent Timeout

**Symptom:**
```
TimeoutError: Agent did not respond within timeout period
```

**Solutions:**

1. Check network connectivity
2. Verify API services are operational
3. Increase timeout in agent configuration

---

### Issue: Memory Error

**Symptom:**
```
MemoryError: Unable to allocate array
```

**Solutions:**

1. Reduce MAX_DOCS in vector_db_loader.py:
   ```python
   MAX_DOCS = 250  # Reduce from 500
   ```

2. Increase system swap space
3. Use a machine with more RAM

---

## Agent-Specific Issues

### QA Agent Not Finding Answers

**Symptom:**
Agent returns "I don't have information about that."

**Solutions:**

1. Verify vector database exists and has content:
   ```bash
   ls -la investopedia_faiss_index/
   ```

2. Check retrieval is working:
   ```python
   from langchain_community.vectorstores import FAISS
   from langchain_openai import OpenAIEmbeddings
   
   vectorstore = FAISS.load_local(
       "investopedia_faiss_index",
       OpenAIEmbeddings(),
       allow_dangerous_deserialization=True
   )
   results = vectorstore.similarity_search("stock", k=3)
   print(len(results))  # Should be 3
   ```

---

### Market Agent No Data

**Symptom:**
Market agent returns empty or error responses.

**Solutions:**

1. Check Alpha Vantage API key:
   ```bash
   curl "https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=AAPL&apikey=YOUR_KEY"
   ```

2. Verify yfinance works:
   ```python
   import yfinance as yf
   stock = yf.Ticker("AAPL")
   print(stock.info.get("currentPrice"))
   ```

---

### News Agent Empty Results

**Symptom:**
News agent returns no news articles.

**Solutions:**

1. Test Tavily API:
   ```python
   from tavily import TavilyClient
   client = TavilyClient()
   results = client.search("financial news")
   print(len(results["results"]))
   ```

2. Check Alpha Vantage news:
   ```bash
   curl "https://www.alphavantage.co/query?function=NEWS_SENTIMENT&apikey=YOUR_KEY"
   ```

---

## Performance Issues

### Issue: Slow Response Times

**Possible Causes & Solutions:**

| Cause | Solution |
|-------|----------|
| Cold start | First request is always slower; wait for warmup |
| Large context | Keep conversations concise |
| Network latency | Check internet connection |
| API rate limits | Space out requests |

### Issue: High Memory Usage

**Solutions:**

1. Clear Streamlit cache:
   ```python
   st.cache_resource.clear()
   ```

2. Restart the application

3. Use `InMemorySaver` with shorter history

---

## UI Issues

### Issue: Streamlit Won't Start

**Symptom:**
```
Error: Could not find a suitable config file
```

**Solutions:**

1. Ensure you're in the project directory:
   ```bash
   cd /path/to/cap-proj
   ```

2. Check streamlit is installed:
   ```bash
   uv pip show streamlit
   ```

---

### Issue: Charts Not Displaying

**Symptom:**
Plotly charts show blank or error.

**Solutions:**

1. Reinstall plotly:
   ```bash
   uv pip install --force-reinstall plotly
   ```

2. Check browser console for JavaScript errors

3. Try a different browser

---

### Issue: Streaming Not Working

**Symptom:**
Responses appear all at once instead of streaming.

**Solutions:**

1. Ensure streaming is enabled in LLM config:
   ```python
   llm = ChatOpenAI(model="gpt-4o-mini", streaming=True)
   ```

2. Check WebSocket connection (browser dev tools)

---

## Getting Help

### Collect Diagnostic Information

Before seeking help, collect:

```bash
# Python version
python --version

# Package versions
uv pip freeze | grep -E "langchain|streamlit|guardrails|openai"

# Environment check
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('OpenAI:', bool(os.getenv('OPENAI_API_KEY')))"

# Test basic functionality
uv run python -c "from langchain_openai import ChatOpenAI; print(ChatOpenAI().invoke('test'))"
```

### Log Files

Enable verbose logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### LangSmith Traces

If LangSmith is enabled, check traces at:
https://smith.langchain.com/projects/finance-assistant
