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
├── tests/                     # Unit tests
│   ├── __init__.py
│   ├── conftest.py           # Pytest fixtures
│   ├── test_guardrails.py    # Guardrail validation tests
│   ├── test_router.py        # Router agent tests
│   └── test_ticker.py        # Ticker extraction tests
├── utils/                     # Utility modules
│   ├── __init__.py
│   └── ticker_utils.py       # Ticker extraction utilities
├── rag/                       # RAG components
│   └── vector_db_loader.py   # FAISS loader
├── docs/                      # Documentation
│   ├── architecture.md
│   ├── setup.md
│   ├── api.md
│   ├── usage.md
│   └── troubleshooting.md
├── streamlit_app.py          # Main application
├── pytest.ini                # Pytest configuration
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

The project uses **pytest** for unit testing with mock-based tests to avoid API calls during testing.

### Test Structure

```
tests/
├── __init__.py
├── conftest.py           # Shared fixtures for all tests
├── test_guardrails.py    # Tests for guardrail validation
├── test_router.py        # Tests for router agent logic
└── test_ticker.py        # Tests for ticker extraction
```

### Test Categories

#### 1. Guardrail Tests (`test_guardrails.py`)

Tests the finance topic validation system:

| Test Class | Description |
|------------|-------------|
| `TestFinanceTopicValidator` | Tests the custom LLM-based validator |
| `TestGuardrailNode` | Tests the guardrail node in the graph |
| `TestCheckTopic` | Tests the conditional routing function |
| `TestOffTopicNode` | Tests the off-topic response handler |

```python
# Example: Testing guardrail validation
def test_validator_passes_finance_topic(self, mock_openai_llm):
    from agents.router_agent_v2 import FinanceTopicValidator
    validator = FinanceTopicValidator()
    validator.llm = mock_openai_llm("yes")

    result = validator.validate("What is a stock?")
    assert result.outcome == "pass"
```

#### 2. Router Tests (`test_router.py`)

Tests the agent routing logic:

| Test Class | Description |
|------------|-------------|
| `TestRouterNode` | Tests routing decisions for different query types |
| `TestRouteToAgent` | Tests the conditional edge function |
| `TestAgentNodes` | Tests individual agent node invocations |
| `TestStateGraphStructure` | Tests the LangGraph workflow structure |

```python
# Example: Testing routing to market agent
def test_routes_to_market_for_prices(self, mock_openai_llm):
    with patch("agents.router_agent_v2.llm", mock_openai_llm("market")):
        from agents.router_agent_v2 import router_node

        state = {"messages": [HumanMessage(content="What's AAPL's price?")]}
        result = router_node(state)

        assert result["next_agent"] == "market"
```

#### 3. Ticker Tests (`test_ticker.py`)

Tests ticker symbol extraction:

| Test Class | Description |
|------------|-------------|
| `TestExtractTicker` | Tests LLM-based ticker extraction |
| `TestQuickTickerLookup` | Tests fast dictionary-based lookup |
| `TestCommonTickers` | Tests the common tickers dictionary |
| `TestTickerWithFixtures` | Tests using pytest fixtures |

```python
# Example: Testing ticker extraction
def test_extracts_apple_ticker(self, mock_openai_llm):
    from utils.ticker_utils import extract_ticker

    mock_llm = mock_openai_llm("AAPL")
    result = extract_ticker("How is Apple doing?", llm=mock_llm)

    assert result == "AAPL"
```

### Shared Fixtures (`conftest.py`)

The `conftest.py` file provides reusable fixtures:

```python
@pytest.fixture
def mock_llm_response():
    """Factory to create mock LLM responses."""
    def _create_response(content: str):
        mock_response = MagicMock()
        mock_response.content = content
        return mock_response
    return _create_response

@pytest.fixture
def mock_openai_llm(mock_llm_response):
    """Mock ChatOpenAI that returns configurable responses."""
    def _create_mock(response_content: str):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_llm_response(response_content)
        return mock_llm
    return _create_mock

@pytest.fixture
def finance_queries():
    """Sample finance-related queries for testing."""
    return ["What is a stock?", "How do I invest in ETFs?", ...]

@pytest.fixture
def non_finance_queries():
    """Sample non-finance queries for testing guardrails."""
    return ["What's the best pizza recipe?", "How do I train for a marathon?", ...]
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run specific test file
uv run pytest tests/test_guardrails.py

# Run specific test class
uv run pytest tests/test_router.py::TestRouterNode

# Run specific test method
uv run pytest tests/test_router.py::TestRouterNode::test_routes_to_qa_for_definitions

# Run with short traceback on failures
uv run pytest --tb=short

# Run and stop on first failure
uv run pytest -x

# Run tests matching a pattern
uv run pytest -k "guardrail"

# Run with coverage report
uv run pytest --cov=agents --cov=utils --cov-report=html
```

### Pytest Command Options

| Option | Description |
|--------|-------------|
| `-v` | Verbose output - shows each test name |
| `-vv` | More verbose - shows full diff on failures |
| `--tb=short` | Short traceback format |
| `--tb=no` | No traceback on failures |
| `-x` | Stop on first failure |
| `-k "pattern"` | Run tests matching pattern |
| `--cov=module` | Generate coverage for module |
| `--cov-report=html` | HTML coverage report |
| `-n auto` | Run tests in parallel (requires pytest-xdist) |

### Writing New Tests

#### 1. Create a test file

```python
# tests/test_new_feature.py
import pytest
from unittest.mock import patch, MagicMock

class TestNewFeature:
    """Tests for the new feature."""

    def test_basic_functionality(self):
        """Test that basic functionality works."""
        # Arrange
        input_data = "test input"

        # Act
        result = my_function(input_data)

        # Assert
        assert result == expected_output

    def test_with_mock(self, mock_openai_llm):
        """Test with mocked LLM."""
        with patch("module.llm", mock_openai_llm("expected response")):
            result = function_using_llm("query")
            assert result == "expected response"
```

#### 2. Use fixtures for common data

```python
# In conftest.py
@pytest.fixture
def sample_data():
    return {"key": "value"}

# In test file
def test_with_fixture(sample_data):
    assert sample_data["key"] == "value"
```

#### 3. Mock external dependencies

```python
# Mock API calls to avoid real requests during tests
with patch("requests.get") as mock_get:
    mock_get.return_value.json.return_value = {"data": "test"}
    result = fetch_data()
    assert result == {"data": "test"}
```

### Test Configuration (`pytest.ini`)

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short
filterwarnings =
    ignore::DeprecationWarning
    ignore::PendingDeprecationWarning
```

### Integration Tests

For testing the full agent flow (requires API keys):

```python
@pytest.mark.integration
def test_full_routing_flow():
    """Integration test for complete routing flow."""
    from agents import router_agent_v2
    from langchain_core.messages import HumanMessage

    config = {"configurable": {"thread_id": "test"}}

    response = router_agent_v2.agent.invoke(
        {"messages": [HumanMessage(content="What is inflation?")]},
        config=config
    )

    assert len(response["messages"]) > 1
    assert response["messages"][-1].content  # Has response content
```

Run integration tests separately:

```bash
# Skip integration tests (default)
uv run pytest -m "not integration"

# Run only integration tests
uv run pytest -m integration
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
