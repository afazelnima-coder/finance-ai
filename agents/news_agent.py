from dotenv import load_dotenv  

from langchain.agents import create_agent
from langchain.tools import tool
from langchain.messages import HumanMessage
from tavily import TavilyClient

load_dotenv()

tavily_client = TavilyClient()

@tool
def searchFinance(query: str) -> str:
    """Searches for financial information on the web."""
    return tavily_client.search(
        query=query,
        include_domains=["marketwatch.com", "finance.yahoo.com", "nasdaq.com"],
        num_results=1,
        region="us",
        language="en"
    )

# Create the agent
agent = create_agent(
    "gpt-5-nano",
    tools=[searchFinance],
    system_prompt="""
    You are a helpful finance expert that gets the top 5 latest news about the stock market and provides accurate and up-to-date information. 
    Only provide the latest top 5 news articles about the user's query.
    Answer only financial concepts and decline any unrelated questions.
    """
)

if __name__ == "__main__":
    # Example interaction
    user_input = "What is going on with oranges?"
    response = agent.invoke({"messages": [HumanMessage(content=user_input)]})

    print(response["messages"][-1].content)