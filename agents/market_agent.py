from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.tools import tool
from langchain.messages import HumanMessage
from langchain_openai import ChatOpenAI
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

# Create the agent with streaming enabled
llm = ChatOpenAI(model="gpt-4o-mini", streaming=True)

agent = create_agent(
    llm,
    tools=[searchFinance],
    system_prompt="You are a helpful finance assistant that always provides accurate and up-to-date information. Answer only financial concepts."
)

if __name__ == "__main__":
    # Example interaction
    user_input = "What is the current state of the stock market and what are the top performing sectors?"
    response = agent.invoke({"messages": [HumanMessage(content=user_input)]})

    print(response["messages"][-1].content)