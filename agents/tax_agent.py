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
        include_domains=["irs.gov", "taxfoundation.org", "turbotax.intuit.com"],
        num_results=1,
        region="us",
        language="en"
    )

# Create the agent with streaming enabled
llm = ChatOpenAI(model="gpt-4o-mini", streaming=True)

agent = create_agent(
    llm,
    tools=[searchFinance],
    system_prompt="""
    You are a helpful tax expert assistant and educator.
    Provide accurate and up-to-date information about tax concepts, regulations, and filing procedures.
    Answer only financial concepts and decline any unrelated questions.
    """
)

if __name__ == "__main__":
    # Example interaction
    user_input = "What is the difference between a tax credit and a tax deduction?"
    response = agent.invoke({"messages": [HumanMessage(content=user_input)]})

    print(response["messages"][-1].content)