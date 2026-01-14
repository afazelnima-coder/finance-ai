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
        include_domains=["investopedia.com"],
        num_results=1,
        region="us",
        language="en"
    )

# Create the agent with streaming enabled
llm = ChatOpenAI(model="gpt-5-nano", streaming=True)

agent = create_agent(
    llm,
    tools=[searchFinance],
    system_prompt=""""
    You are a helpful finance assistant specialized in assisting with setting and planning goals. 
    Answer only financial concepts. 
    Any unrelated questions should be politely declined and specify that they are outside the scope of financial advice.
    """
)

if __name__ == "__main__":
    # Example interaction
    user_input = "I want to save for a house down payment. What are some good savings strategies?"
    response = agent.invoke({"messages": [HumanMessage(content=user_input)]})

    print(response["messages"][-1].content)