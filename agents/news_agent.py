from dotenv import load_dotenv  

from langchain.agents import create_agent
from langchain.tools import tool
from langchain.messages import HumanMessage
from tavily import TavilyClient
import requests
import os

load_dotenv()
# get the api key from environment variable from dotenv
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')

tavily_client = TavilyClient()

@tool
def searchFinance(query: str) -> str:
    """Searches for financial news articles on the web and returns headlines with summaries."""

    ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
    results = []

    # Alpha Vantage topics parameter only accepts predefined topics
    VALID_TOPICS = [
        "blockchain", "earnings", "ipo", "mergers_and_acquisitions",
        "financial_markets", "economy_fiscal", "economy_monetary",
        "economy_macro", "energy_transportation", "finance",
        "life_sciences", "manufacturing", "real_estate",
        "retail_wholesale", "technology"
    ]

    # Check if query matches a valid Alpha Vantage topic
    query_lower = query.lower().replace(" ", "_")
    matching_topic = next((t for t in VALID_TOPICS if t in query_lower or query_lower in t), None)

    if matching_topic:
        # Use Alpha Vantage with valid topic
        url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&topics={matching_topic}&apikey={ALPHA_VANTAGE_API_KEY}'
        try:
            r = requests.get(url)
            data = r.json()
            results = data.get("feed", [])[:5]
        except Exception as e:
            print(f"Alpha Vantage error: {e}")

    # Fall back to Tavily for arbitrary topics or if Alpha Vantage returned no results
    if not results:
        try:
            tavily_results = tavily_client.search(
                query=f"{query} financial news",
                include_domains=["marketwatch.com", "finance.yahoo.com", "nasdaq.com", "cnbc.com", "reuters.com", "bloomberg.com", "wsj.com"],
                max_results=5,
                search_depth="advanced",
                include_answer=True,
                include_raw_content=False
            )
            results = tavily_results.get("results", [])
        except Exception as e:
            print(f"Tavily error: {e}")

    # Format the results with bold title, italic summary, and link
    formatted_results = []

    for result in results:
        title = result.get("title", "No title")
        # Handle both Alpha Vantage ("summary") and Tavily ("content") formats
        summary = result.get("summary") or result.get("content", "No content available")
        url = result.get("url", "")

        # Format: Bold title, italic summary, link at bottom
        article = f"**{title}**\n\n*{summary}*\n\n{url}"
        print(article, "\n")
        formatted_results.append(article)

    return "\n\n---\n\n".join(formatted_results) if formatted_results else "No news articles found."

# Create the agent with memory support
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    "gpt-4o-mini",
    tools=[searchFinance],
    checkpointer=InMemorySaver(),
    system_prompt="""
    You are a conversational financial news assistant that helps users stay informed about financial markets and news.

    Your capabilities:
    1. Search for and present the latest financial news articles with headlines, summaries, and links
    2. Answer follow-up questions about the news articles you've shared
    3. Provide context and explanations about news events
    4. Help users understand the implications of financial news
    5. Search for news on specific topics, companies, or market sectors

    When a user asks for news:
    - Use the searchFinance tool to get real, current news articles
    - Present articles with their full titles (in bold), summaries (in italics), and links
    - For general news requests, search for "latest financial news today"
    - For specific topics, search for that exact topic
    - For any topic, focus your answer only on financial news

    When users ask follow-up questions:
    - Provide thoughtful analysis and context based on the news you've already shared
    - If you need more current information, use the searchFinance tool again
    - Reference specific articles from your previous responses when relevant
    - Help users understand market implications, trends, and connections between news events

    Be conversational, helpful, and insightful. Go beyond just listing articles - help users understand what the news means.
    """
)

if __name__ == "__main__":
    # Example interaction
    user_input = "What is going on with oranges?"
    response = agent.invoke({"messages": [HumanMessage(content=user_input)]})

    print(response["messages"][-1].content)