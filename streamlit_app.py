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
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        # Get bot response with streaming
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            called_agent = None
            has_ai_response = False

            try:
                # Ensure thread_id exists
                if st.session_state.thread_id not in st.session_state.conversation_threads:
                    st.session_state.conversation_threads[st.session_state.thread_id] = str(uuid.uuid4())

                # Configure the agent with the thread
                config = {"configurable": {"thread_id": st.session_state.conversation_threads[st.session_state.thread_id]}}

                # Capture print output to detect which agent was called
                import io
                from contextlib import redirect_stdout
                from langchain.messages import AIMessage

                f = io.StringIO()

                # Build message history for context (including current message)
                messages = []
                for msg in st.session_state.chat_history:
                    if msg["role"] == "user":
                        messages.append(HumanMessage(content=msg["content"]))
                    else:
                        messages.append(AIMessage(content=msg["content"]))

                # Stream the response with spinner
                with st.spinner(""):
                    with redirect_stdout(f):
                        for event in router_agent.agent.stream(
                            {"messages": messages},
                            config=config,
                            stream_mode="values"
                        ):
                            # Get the last message from the event
                            if "messages" in event and event["messages"]:
                                last_msg = event["messages"][-1]
                                # Only show AI messages (not tool calls or echoed questions)
                                if hasattr(last_msg, 'content') and last_msg.content:
                                    # Filter out tool calls and only show actual AI responses
                                    if not hasattr(last_msg, 'tool_calls') or not last_msg.tool_calls:
                                        # Don't show if it's just echoing the user's question
                                        if last_msg.content != prompt and len(last_msg.content.strip()) > 0:
                                            full_response = last_msg.content
                                            has_ai_response = True
                                            # Show streaming with cursor
                                            message_placeholder.markdown(full_response + "▌")

                # Check captured output for agent routing
                output = f.getvalue()
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

                # Add agent info and display final response
                if has_ai_response and full_response:
                    if called_agent:
                        full_message = f"*[Routed to: {called_agent}]*\n\n{full_response}"
                    else:
                        full_message = full_response

                    # Display final response without cursor
                    message_placeholder.markdown(full_message)

                    # Add to chat history
                    st.session_state.chat_history.append({"role": "assistant", "content": full_message})
                else:
                    error_message = "Sorry, I couldn't generate a response."
                    message_placeholder.error(error_message)
                    st.session_state.chat_history.append({"role": "assistant", "content": error_message})

            except Exception as e:
                error_message = f"Sorry, I encountered an error: {str(e)}"
                message_placeholder.error(error_message)
                st.session_state.chat_history.append({"role": "assistant", "content": error_message})

# Tab 2: News
with tab2:
    st.header("📰 Financial News Assistant")

    # Initialize news chat history and thread
    if "news_chat_history" not in st.session_state:
        st.session_state.news_chat_history = []
    if "news_thread_id" not in st.session_state:
        st.session_state.news_thread_id = str(uuid.uuid4())

    # Quick actions
    st.markdown("### Quick Actions")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📰 Latest Headlines", use_container_width=True, key="latest_headlines"):
            prompt = "What are the latest financial news headlines?"
            st.session_state.news_chat_history.append({"role": "user", "content": prompt})

            with st.spinner("Fetching latest headlines..."):
                try:
                    from langchain.messages import AIMessage
                    config = {"configurable": {"thread_id": st.session_state.news_thread_id}}

                    # Build full message history
                    messages = []
                    for msg in st.session_state.news_chat_history:
                        if msg["role"] == "user":
                            messages.append(HumanMessage(content=msg["content"]))
                        else:
                            messages.append(AIMessage(content=msg["content"]))

                    response = news_agent.agent.invoke(
                        {"messages": messages},
                        config=config
                    )
                    bot_message = response["messages"][-1].content
                    st.session_state.news_chat_history.append({"role": "assistant", "content": bot_message})
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.session_state.news_chat_history.append({"role": "assistant", "content": error_msg})
            st.rerun()

    with col2:
        if st.button("📈 Market News", use_container_width=True, key="market_news"):
            prompt = "What are the latest stock market news?"
            st.session_state.news_chat_history.append({"role": "user", "content": prompt})

            with st.spinner("Fetching market news..."):
                try:
                    from langchain.messages import AIMessage
                    config = {"configurable": {"thread_id": st.session_state.news_thread_id}}

                    # Build full message history
                    messages = []
                    for msg in st.session_state.news_chat_history:
                        if msg["role"] == "user":
                            messages.append(HumanMessage(content=msg["content"]))
                        else:
                            messages.append(AIMessage(content=msg["content"]))

                    response = news_agent.agent.invoke(
                        {"messages": messages},
                        config=config
                    )
                    bot_message = response["messages"][-1].content
                    st.session_state.news_chat_history.append({"role": "assistant", "content": bot_message})
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.session_state.news_chat_history.append({"role": "assistant", "content": error_msg})
            st.rerun()

    with col3:
        if st.button("🔄 Clear Chat", use_container_width=True, key="clear_news"):
            st.session_state.news_chat_history = []
            st.session_state.news_thread_id = str(uuid.uuid4())
            st.rerun()

    st.markdown("---")

    # Display chat history in a scrollable container
    chat_container = st.container(height=400)
    with chat_container:
        for message in st.session_state.news_chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Chat input (stays at bottom, outside the scrollable container)
    if prompt := st.chat_input("Ask about financial news or search for specific topics...", key="news_chat_input"):
        # Add user message to history
        st.session_state.news_chat_history.append({"role": "user", "content": prompt})

        # Get bot response
        with st.spinner("🔍 Searching for news..."):
            try:
                from langchain.messages import AIMessage
                config = {"configurable": {"thread_id": st.session_state.news_thread_id}}

                # Build message history for context
                messages = []
                for msg in st.session_state.news_chat_history:
                    if msg["role"] == "user":
                        messages.append(HumanMessage(content=msg["content"]))
                    else:
                        messages.append(AIMessage(content=msg["content"]))

                response = news_agent.agent.invoke(
                    {"messages": messages},
                    config=config
                )
                bot_message = response["messages"][-1].content
                st.session_state.news_chat_history.append({"role": "assistant", "content": bot_message})

            except Exception as e:
                error_message = f"Sorry, I encountered an error: {str(e)}"
                st.session_state.news_chat_history.append({"role": "assistant", "content": error_message})

        st.rerun()

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
