import streamlit as st
from dotenv import load_dotenv
from langchain.messages import HumanMessage
import sys
import os
import uuid

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents import router_agent_v2 as router_agent, news_agent

load_dotenv()

# Configure page
st.set_page_config(
    page_title="Finance Assistant",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "conversation_threads" not in st.session_state:
    st.session_state.conversation_threads = {}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "news_content" not in st.session_state:
    st.session_state.news_content = None

# Sidebar
with st.sidebar:
    st.title("💼 Finance Assistant")
    st.markdown("---")

    st.markdown("""
    ### Available Services
    - 🎓 General Finance Q&A (RAG)
    - 📈 Market Analysis
    - 📰 Financial News
    - 💰 Tax Information
    - 🎯 Financial Goals
    - 📊 Portfolio Management
    """)

    st.markdown("---")

    if st.button("🔄 New Conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

# Main content - Tabs
tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "📰 News", "📊 Market", "ℹ️ About"])

# Tab 1: Chat Interface
with tab1:
    st.header("Chat with Finance Assistant")

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a finance question..."):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get bot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Ensure thread_id exists
                    if st.session_state.thread_id not in st.session_state.conversation_threads:
                        st.session_state.conversation_threads[st.session_state.thread_id] = str(uuid.uuid4())

                    # Configure the agent with the thread
                    config = {"configurable": {"thread_id": st.session_state.conversation_threads[st.session_state.thread_id]}}

                    # Capture print output to detect which agent was called
                    import io
                    from contextlib import redirect_stdout

                    f = io.StringIO()
                    with redirect_stdout(f):
                        response = router_agent.agent.invoke(
                            {"messages": [HumanMessage(content=prompt)]},
                            config=config
                        )

                    # Check captured output for agent routing
                    output = f.getvalue()
                    called_agent = None
                    if "Executing QA Agent" in output:
                        called_agent = "QA Agent (RAG)"
                    elif "Executing Market Agent" in output:
                        called_agent = "Market Agent"
                    elif "Executing News Agent" in output:
                        called_agent = "News Agent"
                    elif "Executing Tax Agent" in output:
                        called_agent = "Tax Agent"
                    elif "Executing Goal Agent" in output:
                        called_agent = "Goal Agent"
                    elif "Executing Portfolio Agent" in output:
                        called_agent = "Portfolio Agent"

                    # Extract response
                    last_message = response["messages"][-1]
                    if hasattr(last_message, 'content'):
                        bot_message = last_message.content
                    elif isinstance(last_message, dict):
                        bot_message = last_message.get('content', str(last_message))
                    else:
                        bot_message = str(last_message)

                    # Add agent info
                    if called_agent:
                        full_message = f"*[Routed to: {called_agent}]*\n\n{bot_message}"
                    else:
                        full_message = bot_message

                    st.markdown(full_message)

                    # Add to chat history
                    st.session_state.chat_history.append({"role": "assistant", "content": full_message})

                except Exception as e:
                    error_message = f"Sorry, I encountered an error: {str(e)}"
                    st.error(error_message)
                    st.session_state.chat_history.append({"role": "assistant", "content": error_message})

# Tab 2: News
with tab2:
    st.header("📰 Latest Financial News")

    # Initialize search results state
    if "search_results" not in st.session_state:
        st.session_state.search_results = None
    if "search_query" not in st.session_state:
        st.session_state.search_query = None

    # Search section
    st.markdown("### Search for Specific News")

    # Create a form to handle Enter key
    with st.form(key="news_search_form", clear_on_submit=False):
        col1, col2 = st.columns([4, 1])

        with col1:
            news_search = st.text_input(
                "Search topic:",
                placeholder="e.g., Tesla stock, Federal Reserve, cryptocurrency...",
                label_visibility="collapsed",
                key="news_search_input"
            )

        with col2:
            search_button = st.form_submit_button("🔍 Search", use_container_width=True)

    # Handle search
    if search_button and news_search:
        with st.spinner(f"Searching news about '{news_search}'..."):
            try:
                response = news_agent.agent.invoke(
                    {"messages": [HumanMessage(content=f"What are the latest news about {news_search}?")]}
                )
                st.session_state.search_results = response["messages"][-1].content
                st.session_state.search_query = news_search

            except Exception as e:
                st.session_state.search_results = f"Error searching news: {str(e)}"
                st.session_state.search_query = news_search

    # Display search results if available
    if st.session_state.search_results:
        st.success(f"Search Results for: {st.session_state.search_query}")
        st.markdown(st.session_state.search_results)

        # Add a button to clear search and go back to general news
        if st.button("← Back to General News", key="back_to_general"):
            st.session_state.search_results = None
            st.session_state.search_query = None
            st.rerun()
    else:
        # Only show general news when not displaying search results
        st.markdown("---")

        # General news section
        st.markdown("### General Financial News")

        col3, col4 = st.columns([3, 1])

        with col4:
            if st.button("🔄 Refresh", use_container_width=True, key="news_refresh_button"):
                st.session_state.news_content = None
                st.rerun()

        # Fetch general news if not already fetched
        if st.session_state.news_content is None:
            with st.spinner("Fetching latest financial news..."):
                try:
                    response = news_agent.agent.invoke(
                        {"messages": [HumanMessage(content="What are the latest financial news headlines?")]}
                    )
                    st.session_state.news_content = response["messages"][-1].content
                except Exception as e:
                    st.session_state.news_content = f"Error fetching news: {str(e)}"

        # Display general news
        st.markdown(st.session_state.news_content)

# Tab 3: Market
with tab3:
    st.header("📊 Market Overview")

    st.markdown("""
    ### Quick Market Lookup
    Enter a stock symbol or ask about market trends.
    """)

    market_query = st.text_input("Enter your market question:", placeholder="e.g., What's the current price of AAPL?")

    if st.button("Get Market Data") and market_query:
        with st.spinner("Fetching market data..."):
            try:
                from agents import market_agent
                response = market_agent.agent.invoke(
                    {"messages": [HumanMessage(content=market_query)]}
                )
                market_response = response["messages"][-1].content
                st.success("Market Data")
                st.markdown(market_response)
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Tab 4: About
with tab4:
    st.header("ℹ️ About Finance Assistant")

    st.markdown("""
    ## Welcome to Your AI-Powered Finance Assistant!

    This application uses multiple specialized AI agents to help you with various financial tasks:

    ### 🎓 QA Agent (RAG)
    - Answers general finance questions using a knowledge base of 500+ Investopedia articles
    - Powered by Retrieval-Augmented Generation (RAG) for accurate, source-based answers

    ### 📈 Market Agent
    - Provides real-time stock prices and market trends
    - Analyzes market conditions and movements

    ### 📰 News Agent
    - Fetches the latest financial news
    - Keeps you updated on market-moving events

    ### 💰 Tax Agent
    - Answers tax-related questions
    - Helps with tax calculations and planning

    ### 🎯 Goal Agent
    - Assists with financial planning
    - Helps set and track financial goals

    ### 📊 Portfolio Agent
    - Provides portfolio management advice
    - Analyzes investment allocations

    ---

    ### How It Works

    1. **Ask a Question**: Type your finance question in the chat
    2. **Smart Routing**: Our router agent analyzes your question and routes it to the most appropriate specialist
    3. **Get Expert Answer**: The specialist agent processes your question and provides a detailed response

    ---

    ### Technology Stack

    - **Frontend**: Streamlit
    - **AI Framework**: LangChain & LangGraph
    - **LLM**: OpenAI GPT-4o-mini
    - **Vector Database**: FAISS
    - **Embeddings**: OpenAI text-embedding-3-small

    ---

    **Built with ❤️ using LangChain and Streamlit**
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>Finance Assistant v2.0 | Powered by LangChain & Streamlit</div>",
    unsafe_allow_html=True
)
