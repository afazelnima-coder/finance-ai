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
        include_domains=["investopedia.com", "morningstar.com", "yahoo finance"],
        num_results=1,
        region="us",
        language="en"
    )

# Create the agent with streaming enabled
llm = ChatOpenAI(model="gpt-4o-mini", streaming=True)

agent = create_agent(
    llm,
    tools=[searchFinance],
    system_prompt="You are a helpful finance assistant specialized in reviewing a user portfolio. Answer only financial concepts."
)

if __name__ == "__main__":
    # Example interaction
    user_input = "I have 200K in my 401K. 50% in S&P 500 index fund, and 50% in a target date fund. How should I rebalance my portfolio to retire comfortably in 20 years from now?"
    response = agent.invoke({"messages": [HumanMessage(content=user_input)]})

    print(response["messages"][-1].content)