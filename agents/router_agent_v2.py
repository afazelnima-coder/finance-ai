from dotenv import load_dotenv
from typing import Annotated, Literal, Dict, Any
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from guardrails import Guard
from guardrails.validators import Validator, register_validator, ValidationResult, PassResult, FailResult

try:
    from . import market_agent, goal_agent, news_agent, tax_agent, qa_agent, portfolio_agent
except ImportError:
    import market_agent, goal_agent, news_agent, tax_agent, qa_agent, portfolio_agent

load_dotenv()

# Custom finance topic validator using LLM
@register_validator(name="finance_topic_validator", data_type="string")
class FinanceTopicValidator(Validator):
    """Validates that the input is related to finance topics using an LLM."""

    def __init__(self, on_fail: str = "noop"):
        super().__init__(on_fail=on_fail)
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.valid_topics = [
            "finance", "investing", "stocks", "bonds", "mutual funds", "ETFs",
            "retirement planning", "401k", "IRA", "taxes", "tax planning",
            "budgeting", "savings", "banking", "credit", "loans", "mortgages",
            "insurance", "real estate investment", "cryptocurrency", "forex",
            "commodities", "portfolio management", "asset allocation",
            "financial planning", "wealth management", "financial markets",
            "economics", "inflation", "interest rates", "financial news",
            "company earnings", "stock market", "dividends", "capital gains"
        ]

    def validate(self, value: str, metadata: Dict[str, Any] = {}) -> ValidationResult:
        """Check if the input is finance-related using LLM."""
        # Get conversation context from metadata if available
        conversation_context = metadata.get("conversation_context", "")

        if conversation_context:
            prompt = f"""You are a topic classifier. Determine if the following user message is related to finance topics.
Consider the conversation context to understand follow-up questions.

Finance topics include: {', '.join(self.valid_topics)}

Recent conversation:
{conversation_context}

Current user message: "{value}"

IMPORTANT: If the current message is a follow-up question (like "what are the most common?", "tell me more", "which one is best?")
that refers to a previous finance-related topic, it should be considered finance-related.

Respond with ONLY "yes" if the message is related to finance (either directly or as a follow-up), or "no" if it is not related to finance.
Do not explain your reasoning, just respond with "yes" or "no"."""
        else:
            prompt = f"""You are a topic classifier. Determine if the following user message is related to finance topics.

Finance topics include: {', '.join(self.valid_topics)}

User message: "{value}"

Respond with ONLY "yes" if the message is related to finance, or "no" if it is not related to finance.
Do not explain your reasoning, just respond with "yes" or "no"."""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        is_finance = response.content.strip().lower() == "yes"

        if is_finance:
            return PassResult()
        else:
            return FailResult(error_message="The query is not related to finance topics.")

# Initialize the finance topic guardrail
finance_guard = Guard().use(
    FinanceTopicValidator(on_fail="noop")
)

# Define the state
class State(TypedDict):
    messages: Annotated[list, add_messages]
    next_agent: str
    is_on_topic: bool

# Initialize the LLM for routing (with streaming)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)

def guardrail_node(state: State):
    """Guardrail node that checks if the query is finance-related."""
    messages = state["messages"]
    last_message = messages[-1].content if messages else ""

    print("🛡️ Checking topic relevance...")

    # Build conversation context from recent messages (last 4 messages for context)
    conversation_context = ""
    if len(messages) > 1:
        # Get up to 4 recent messages before the current one for context
        context_messages = messages[-5:-1] if len(messages) > 5 else messages[:-1]
        context_parts = []
        for msg in context_messages:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            # Truncate long messages to avoid token bloat
            content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
            context_parts.append(f"{role}: {content}")
        conversation_context = "\n".join(context_parts)

    try:
        # Validate the message against the finance topic guardrail with context
        result = finance_guard.validate(
            last_message,
            metadata={"conversation_context": conversation_context}
        )
        is_valid = result.validation_passed

        if is_valid:
            print("✅ Topic is finance-related, proceeding with routing")
        else:
            print("⚠️ Topic is off-topic, will decline to answer")

        return {"is_on_topic": is_valid}
    except Exception as e:
        # If guardrail check fails, default to allowing the message
        print(f"⚠️ Guardrail check error: {e}, defaulting to on-topic")
        return {"is_on_topic": True}

def check_topic(state: State) -> Literal["router", "off_topic"]:
    """Conditional edge that checks if topic is on-topic."""
    if state.get("is_on_topic", True):
        return "router"
    return "off_topic"

def off_topic_node(state: State):
    """Node that handles off-topic questions."""
    print("🚫 Handling off-topic question")
    off_topic_message = AIMessage(
        content="I'm sorry, but I can only help with finance-related questions. "
        "This includes topics like investing, stocks, bonds, retirement planning, "
        "taxes, budgeting, savings, loans, mortgages, and other financial matters. "
        "Please feel free to ask me anything about these topics!"
    )
    return {"messages": [off_topic_message]}

def router_node(state: State):
    """Router node that decides which agent to call."""
    messages = state["messages"]
    last_message = messages[-1].content if messages else ""

    # Build conversation context from recent messages for better routing of follow-ups
    conversation_context = ""
    if len(messages) > 1:
        context_messages = messages[-5:-1] if len(messages) > 5 else messages[:-1]
        context_parts = []
        for msg in context_messages:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
            context_parts.append(f"{role}: {content}")
        conversation_context = "\n".join(context_parts)

    # Use LLM to classify the query
    if conversation_context:
        routing_prompt = f"""You are a routing assistant for a financial advice system.
    Analyze the user's question and decide which specialized agent should handle it.
    Consider the conversation context to understand follow-up questions.

    Available agents:
    - qa: General finance concepts and definitions (use for "what is", "explain", "define")
    - market: Current market data, stock prices, trends
    - news: Latest financial news and updates
    - tax: Tax-related questions and calculations
    - goal: Financial planning, retirement, savings goals
    - portfolio: Investment portfolio management and allocation, analyzing holdings

    Recent conversation:
    {conversation_context}

    Current user question: {last_message}

    IMPORTANT: If this is a follow-up question (like "what are the most common?", "tell me more", "which is best?"),
    route to the SAME agent that handled the previous topic.

    Respond with ONLY the agent name (qa, market, news, tax, goal, or portfolio)."""
    else:
        routing_prompt = f"""You are a routing assistant for a financial advice system.
    Analyze the user's question and decide which specialized agent should handle it.

    Available agents:
    - qa: General finance concepts and definitions (use for "what is", "explain", "define")
    - market: Current market data, stock prices, trends
    - news: Latest financial news and updates
    - tax: Tax-related questions and calculations
    - goal: Financial planning, retirement, savings goals
    - portfolio: Investment portfolio management and allocation, analyzing holdings

    User question: {last_message}

    Respond with ONLY the agent name (qa, market, news, tax, goal, or portfolio)."""

    response = llm.invoke([HumanMessage(content=routing_prompt)])
    agent_choice = response.content.strip().lower()

    # Validate and default to qa if invalid
    valid_agents = ["qa", "market", "news", "tax", "goal", "portfolio"]
    if agent_choice not in valid_agents:
        agent_choice = "qa"

    print(f"🤖 Router decided: {agent_choice.upper()} Agent")

    return {"next_agent": agent_choice}

def qa_node(state: State):
    """QA agent node."""
    print("🤖 Executing QA Agent")
    messages = state["messages"]
    response = qa_agent.agent.invoke({"messages": messages})
    return {"messages": [response["messages"][-1]]}

def market_node(state: State):
    """Market agent node."""
    print("🤖 Executing Market Agent")
    messages = state["messages"]
    response = market_agent.agent.invoke({"messages": messages})
    return {"messages": [response["messages"][-1]]}

def news_node(state: State):
    """News agent node."""
    print("🤖 Executing News Agent")
    messages = state["messages"]
    response = news_agent.agent.invoke({"messages": messages})
    return {"messages": [response["messages"][-1]]}

def tax_node(state: State):
    """Tax agent node."""
    print("🤖 Executing Tax Agent")
    messages = state["messages"]
    response = tax_agent.agent.invoke({"messages": messages})
    return {"messages": [response["messages"][-1]]}

def goal_node(state: State):
    """Goal agent node."""
    print("🤖 Executing Goal Agent")
    messages = state["messages"]
    response = goal_agent.agent.invoke({"messages": messages})
    return {"messages": [response["messages"][-1]]}

def portfolio_node(state: State):
    """Portfolio agent node."""
    print("🤖 Executing Portfolio Agent")
    messages = state["messages"]
    response = portfolio_agent.agent.invoke({"messages": messages})
    return {"messages": [response["messages"][-1]]}

def route_to_agent(state: State) -> Literal["qa", "market", "news", "tax", "goal", "portfolio"]:
    """Conditional edge that routes to the appropriate agent."""
    return state["next_agent"]

# Build the graph
workflow = StateGraph(State)

# Add nodes
workflow.add_node("guardrail", guardrail_node)
workflow.add_node("off_topic", off_topic_node)
workflow.add_node("router", router_node)
workflow.add_node("qa", qa_node)
workflow.add_node("market", market_node)
workflow.add_node("news", news_node)
workflow.add_node("tax", tax_node)
workflow.add_node("goal", goal_node)
workflow.add_node("portfolio", portfolio_node)

# Add edges - start with guardrail check
workflow.add_edge(START, "guardrail")

# Conditional edge from guardrail: if on-topic go to router, otherwise go to off_topic
workflow.add_conditional_edges(
    "guardrail",
    check_topic,
    {
        "router": "router",
        "off_topic": "off_topic"
    }
)

# Router routes to appropriate agent
workflow.add_conditional_edges(
    "router",
    route_to_agent,
    {
        "qa": "qa",
        "market": "market",
        "news": "news",
        "tax": "tax",
        "goal": "goal",
        "portfolio": "portfolio"
    }
)

# All agent nodes and off_topic go to END
workflow.add_edge("off_topic", END)
workflow.add_edge("qa", END)
workflow.add_edge("market", END)
workflow.add_edge("news", END)
workflow.add_edge("tax", END)
workflow.add_edge("goal", END)
workflow.add_edge("portfolio", END)

# Compile with checkpointer for memory
checkpointer = InMemorySaver()
agent = workflow.compile(checkpointer=checkpointer)

if __name__ == "__main__":
    # Example interaction - On-topic question
    print("=" * 50)
    print("Test 1: On-topic finance question")
    print("=" * 50)
    user_input = "What is a stock option?"
    config = {"configurable": {"thread_id": "test-thread-1"}}

    response = agent.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config=config
    )

    print("\nResponse:")
    print(response["messages"][-1].content)

    # Example interaction - Off-topic question
    print("\n" + "=" * 50)
    print("Test 2: Off-topic question")
    print("=" * 50)
    user_input = "What is the best recipe for chocolate cake?"
    config = {"configurable": {"thread_id": "test-thread-2"}}

    response = agent.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config=config
    )

    print("\nResponse:")
    print(response["messages"][-1].content)
