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
        include_domains=["investopedia.com"],
        num_results=1,
        region="us",
        language="en"
    )

# Create the agent
agent = create_agent(
    "gpt-5-nano",
    tools=[searchFinance],
    system_prompt="You are a helpful assistant specialized in finance questions. Answer only financial concepts."
)

if __name__ == "__main__":
    # Example interaction
    user_input = "I have pain in my lower back. What could be the cause?"
    response = agent.invoke({"messages": [HumanMessage(content=user_input)]})

    print(response["messages"][-1].content)