from dotenv import load_dotenv  

from langchain.agents import create_agent
from langchain.tools import tool
from langchain.messages import HumanMessage
from tavily import TavilyClient

load_dotenv()

tavily_client = TavilyClient()

@tool
def searchFinance(query: str) -> str:
    """Searches for financial news articles on the web and returns headlines with summaries."""
    results = tavily_client.search(
        query=query,
        include_domains=["marketwatch.com", "finance.yahoo.com", "nasdaq.com", "cnbc.com", "reuters.com"],
        max_results=5,
        search_depth="advanced",
        include_answer=True,
        include_raw_content=False
    )

    # Format the results with bold title, italic summary, and link
    formatted_results = []
    if "results" in results:
        for idx, result in enumerate(results["results"], 1):
            title = result.get("title", "No title")
            content = result.get("content", "No content available")
            url = result.get("url", "")

            # Format: Bold title, italic summary, link at bottom
            article = f"**{title}**\n\n*{content}*\n\n{url}"
            formatted_results.append(article)

    return "\n\n---\n\n".join(formatted_results) if formatted_results else "No news articles found."

# Create the agent
agent = create_agent(
    "gpt-5-nano",
    tools=[searchFinance],
    system_prompt="""
    You are a helpful finance news assistant that provides the latest financial news with actual headlines and summaries.

    When a user asks for news:
    1. Use the searchFinance tool to get real news articles
    2. Present each article with its headline, summary, and link
    3. Format the response clearly with numbered articles
    4. Include 3-5 relevant news articles

    For general news requests, search for "latest financial news today"
    For specific topics, search for that specific topic (e.g., "Tesla stock news", "Federal Reserve news")

    Always present the actual article titles and summaries from the search results.
    Bring back only financial news requests on any topic.
    """
)

if __name__ == "__main__":
    # Example interaction
    user_input = "What is going on with oranges?"
    response = agent.invoke({"messages": [HumanMessage(content=user_input)]})

    print(response["messages"][-1].content)