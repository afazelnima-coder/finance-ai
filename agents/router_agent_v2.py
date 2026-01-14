from dotenv import load_dotenv
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

from . import market_agent, goal_agent, news_agent, tax_agent, qa_agent, portfolio_agent

load_dotenv()

# Define the state
class State(TypedDict):
    messages: Annotated[list, add_messages]
    next_agent: str

# Initialize the LLM for routing (with streaming)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)

def router_node(state: State):
    """Router node that decides which agent to call."""
    messages = state["messages"]
    last_message = messages[-1].content if messages else ""

    # Use LLM to classify the query
    routing_prompt = f"""You are a routing assistant for a financial advice system.
    Analyze the user's question and decide which specialized agent should handle it.

    Available agents:
    - qa: General finance concepts and definitions (use for "what is", "explain", "define")
    - market: Current market data, stock prices, trends
    - news: Latest financial news and updates
    - tax: Tax-related questions and calculations
    - goal: Financial planning, retirement, savings goals
    - portfolio: Investment portfolio management and allocation

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
workflow.add_node("router", router_node)
workflow.add_node("qa", qa_node)
workflow.add_node("market", market_node)
workflow.add_node("news", news_node)
workflow.add_node("tax", tax_node)
workflow.add_node("goal", goal_node)
workflow.add_node("portfolio", portfolio_node)

# Add edges
workflow.add_edge(START, "router")
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

# All agent nodes go to END
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
    # Example interaction
    user_input = "What is a stock option?"
    config = {"configurable": {"thread_id": "test-thread"}}

    response = agent.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config=config
    )

    print("\nResponse:")
    print(response["messages"][-1].content)
