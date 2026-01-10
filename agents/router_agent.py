from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langchain.messages import HumanMessage
from tavily import TavilyClient
from . import market_agent, goal_agent, news_agent, tax_agent, qa_agent, portfolio_agent

load_dotenv()

tavily_client = TavilyClient()

@tool
def callQA(query: str) -> str:
    """Calls the QA agent to answer finance questions."""
    response = qa_agent.agent.invoke({"messages": [HumanMessage(content=query)]})
    return response["messages"][-1].content

@tool
def callMarket(query: str) -> str:
    """Calls the Market agent to answer current market state and trends questions."""
    response = market_agent.agent.invoke({"messages": [HumanMessage(content=query)]})
    return response["messages"][-1].content

@tool
def callNews(query: str) -> str:
    """Calls the News agent to get the latest financial news."""
    response = news_agent.agent.invoke({"messages": [HumanMessage(content=query)]})
    return response["messages"][-1].content

@tool
def callTax(query: str) -> str:
    """Calls the Tax agent to answer tax-related questions."""
    response = tax_agent.agent.invoke({"messages": [HumanMessage(content=query)]})
    return response["messages"][-1].content

@tool
def callGoal(query: str) -> str:
    """Calls the Goal agent to answer questions about financial goals."""
    response = goal_agent.agent.invoke({"messages": [HumanMessage(content=query)]})
    return response["messages"][-1].content

@tool
def callPortfolio(query: str) -> str:
    """Calls the Portfolio agent to answer questions about investment portfolios."""
    response = portfolio_agent.agent.invoke({"messages": [HumanMessage(content=query)]})
    return response["messages"][-1].content

# Create the agent
agent = create_agent(
    "gpt-5-nano",
    tools=[callQA, callMarket, callNews, callTax, callGoal, callPortfolio],
    checkpointer=InMemorySaver(),
    system_prompt="""
    You are a helpful assistant specialized in finance questions. 
    Take the query and decide which specialized agent to call among QA, Market, News, Tax, Goal, and Portfolio agents.
    Answer only financial concepts. Decline to answer any non-financial questions with a short message.
    """
)

if __name__ == "__main__":
    # Example interaction
    user_input = "What is the current Apple price?"
    response = agent.invoke({"messages": [HumanMessage(content=user_input)]})

    print(response["messages"][-1].content)