# Development Guide

This guide is for developers who want to contribute to or extend the Finance Assistant.

## Table of Contents

1. [Development Setup](#development-setup)
2. [Project Structure](#project-structure)
3. [Adding New Agents](#adding-new-agents)
4. [Modifying Guardrails](#modifying-guardrails)
5. [Testing](#testing)
6. [Code Style](#code-style)
7. [Contributing](#contributing)

---

## Development Setup

### Clone and Setup

```bash
git clone <repository-url>
cd cap-proj

# Create development environment
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Setup pre-commit hooks (optional)
uv pip install pre-commit
pre-commit install
```

### Development Dependencies

```bash
# Add dev dependencies
uv pip install pytest pytest-asyncio black isort mypy
```

---

## Project Structure

```
cap-proj/
├── agents/                    # Agent modules
│   ├── __init__.py           # Agent exports
│   ├── router_agent_v2.py    # Main router (LangGraph)
│   ├── qa_agent.py           # RAG agent
│   ├── market_agent.py       # Market data agent
│   ├── news_agent.py         # News agent
│   ├── tax_agent.py          # Tax agent
│   ├── goal_agent.py         # Goal planning agent
│   └── portfolio_agent.py    # Portfolio agent
├── rag/                       # RAG components
│   └── vector_db_loader.py   # FAISS loader
├── docs/                      # Documentation
│   ├── architecture.md
│   ├── setup.md
│   ├── api.md
│   ├── usage.md
│   └── troubleshooting.md
├── streamlit_app.py          # Main application
├── requirements.txt          # Dependencies
├── .env                      # Environment variables
└── README.md                 # Project overview
```

---

## Adding New Agents

### Step 1: Create Agent File

Create `agents/new_agent.py`:

```python
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

# Define tools (optional)
@tool
def my_tool(query: str) -> str:
    """Description of what this tool does."""
    # Implementation
    return result

# Create agent
agent = create_agent(
    "gpt-4o-mini",
    tools=[my_tool],  # or [] if no tools
    checkpointer=InMemorySaver(),
    system_prompt="""
    You are a specialized assistant for [domain].
    [Additional instructions]
    """
)

if __name__ == "__main__":
    from langchain_core.messages import HumanMessage
    response = agent.invoke({
        "messages": [HumanMessage(content="Test query")]
    })
    print(response["messages"][-1].content)
```

### Step 2: Register in __init__.py

Edit `agents/__init__.py`:

```python
from . import (
    qa_agent,
    market_agent,
    news_agent,
    tax_agent,
    goal_agent,
    portfolio_agent,
    new_agent,  # Add this
)
```

### Step 3: Add to Router

Edit `agents/router_agent_v2.py`:

```python
# Add import
from . import new_agent

# Add node function
def new_agent_node(state: State):
    """New agent node."""
    print("🤖 Executing New Agent")
    messages = state["messages"]
    response = new_agent.agent.invoke({"messages": messages})
    return {"messages": [response["messages"][-1]]}

# Update route_to_agent return type
def route_to_agent(state: State) -> Literal[
    "qa", "market", "news", "tax", "goal", "portfolio", "new_agent"
]:
    return state["next_agent"]

# Add node to workflow
workflow.add_node("new_agent", new_agent_node)

# Add conditional edge
# In the router's routing logic, add handling for new_agent

# Add edge to END
workflow.add_edge("new_agent", END)
```

### Step 4: Update Router Prompt

Update the routing prompt in `router_node()`:

```python
routing_prompt = f"""...
Available agents:
- qa: General finance concepts
- market: Stock prices, trends
- news: Financial news
- tax: Tax questions
- goal: Financial planning
- portfolio: Portfolio management
- new_agent: [Description of new agent]
...
"""
```

---

## Modifying Guardrails

### Adding New Valid Topics

Edit `router_agent_v2.py`:

```python
class FinanceTopicValidator(Validator):
    def __init__(self, on_fail: str = "noop"):
        super().__init__(on_fail=on_fail)
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.valid_topics = [
            # Existing topics...
            "new_topic_1",
            "new_topic_2",
        ]
```

### Customizing Rejection Message

Edit `off_topic_node()`:

```python
def off_topic_node(state: State):
    off_topic_message = AIMessage(
        content="Your custom rejection message here..."
    )
    return {"messages": [off_topic_message]}
```

### Adding Guardrails to New Tabs

```python
# In streamlit_app.py, use the shared function:
if submit_button and user_query:
    if not is_finance_related(user_query):
        st.warning("Off-topic question detected...")
    else:
        # Process query
```

---

## Testing

### Unit Tests

Create `tests/test_agents.py`:

```python
import pytest
from langchain_core.messages import HumanMessage

def test_qa_agent():
    from agents import qa_agent
    
    response = qa_agent.agent.invoke({
        "messages": [HumanMessage(content="What is a stock?")]
    })
    
    assert response["messages"]
    assert len(response["messages"][-1].content) > 0

def test_guardrail_blocks_offtopic():
    from streamlit_app import is_finance_related
    
    assert is_finance_related("What is a bond?") == True
    assert is_finance_related("Best pizza recipe?") == False

def test_ticker_extraction():
    from streamlit_app import extract_ticker
    
    assert extract_ticker("Apple stock") == "AAPL"
    assert extract_ticker("How is Tesla?") == "TSLA"
    assert extract_ticker("Market trends") is None
```

### Run Tests

```bash
# Run all tests
uv run pytest tests/

# Run with coverage
uv run pytest tests/ --cov=agents --cov-report=html

# Run specific test
uv run pytest tests/test_agents.py::test_qa_agent -v
```

### Integration Tests

```python
def test_full_routing_flow():
    from agents import router_agent_v2
    from langchain_core.messages import HumanMessage
    
    config = {"configurable": {"thread_id": "test"}}
    
    # Test QA routing
    response = router_agent_v2.agent.invoke(
        {"messages": [HumanMessage(content="What is inflation?")]},
        config=config
    )
    assert "QA" in str(response) or len(response["messages"]) > 1
```

---

## Code Style

### Formatting

```bash
# Format with black
black agents/ streamlit_app.py

# Sort imports
isort agents/ streamlit_app.py
```

### Type Checking

```bash
# Run mypy
mypy agents/ --ignore-missing-imports
```

### Linting

```bash
# Run ruff
ruff check agents/ streamlit_app.py
```

---

## Contributing

### Workflow

1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-feature`
3. Make changes
4. Run tests: `uv run pytest`
5. Format code: `black . && isort .`
6. Commit: `git commit -m "Add new feature"`
7. Push: `git push origin feature/new-feature`
8. Create Pull Request

### Commit Messages

Follow conventional commits:

```
feat: add new crypto agent
fix: resolve API timeout issue  
docs: update troubleshooting guide
refactor: simplify router logic
test: add guardrail unit tests
```

### PR Checklist

- [ ] Tests pass locally
- [ ] Code is formatted
- [ ] Documentation updated
- [ ] No sensitive data committed
- [ ] Changelog updated (if applicable)
