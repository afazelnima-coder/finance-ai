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

    # Check for pending prompt
    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None

    # Display chat history in a scrollable container
    chat_container = st.container(height=450)
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Handle streaming if there's a pending prompt
        if st.session_state.pending_prompt:
            prompt = st.session_state.pending_prompt

            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                message_placeholder.markdown("▌")

                try:
                    import asyncio
                    import io
                    from contextlib import redirect_stdout
                    from langchain.messages import AIMessage

                    if st.session_state.thread_id not in st.session_state.conversation_threads:
                        st.session_state.conversation_threads[st.session_state.thread_id] = str(uuid.uuid4())

                    config = {"configurable": {"thread_id": st.session_state.conversation_threads[st.session_state.thread_id]}}

                    messages = []
                    for msg in st.session_state.chat_history:
                        if msg["role"] == "user":
                            messages.append(HumanMessage(content=msg["content"]))
                        else:
                            messages.append(AIMessage(content=msg["content"]))
                    messages.append(HumanMessage(content=prompt))

                    # Use a dict to share state with async function
                    state = {"full_response": "", "final_response": None}
                    f = io.StringIO()

                    # Async streaming function
                    async def stream_response():
                        with redirect_stdout(f):
                            async for event in router_agent.agent.astream_events(
                                {"messages": messages},
                                config=config,
                                version="v2"
                            ):
                                # Look for chat model stream events (token chunks)
                                if event["event"] == "on_chat_model_stream":
                                    # Skip router and guardrail node responses - only stream from sub-agent nodes
                                    langgraph_node = event.get("metadata", {}).get("langgraph_node", "")
                                    if langgraph_node in ["router", "guardrail"]:
                                        continue

                                    chunk = event["data"]["chunk"]
                                    if hasattr(chunk, "content") and chunk.content:
                                        state["full_response"] += chunk.content
                                        message_placeholder.markdown(state["full_response"] + "▌")

                                # Capture final state for off-topic responses
                                if event["event"] == "on_chain_end" and event.get("name") == "LangGraph":
                                    state["final_response"] = event.get("data", {}).get("output", {})

                    # Run async streaming
                    asyncio.run(stream_response())
                    full_response = state["full_response"]

                    # Check which agent was called or if off-topic
                    output = f.getvalue()
                    called_agent = None
                    is_off_topic = "Handling off-topic question" in output

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

                    # Handle off-topic responses (not streamed, get from final state)
                    if is_off_topic and not full_response and state["final_response"]:
                        final_messages = state["final_response"].get("messages", [])
                        if final_messages:
                            full_response = final_messages[-1].content

                    if full_response:
                        if is_off_topic:
                            full_message = f"*[Off-topic query detected]*\n\n{full_response}"
                        elif called_agent:
                            full_message = f"*[Routed to: {called_agent}]*\n\n{full_response}"
                        else:
                            full_message = full_response
                        message_placeholder.markdown(full_message)
                        st.session_state.chat_history.append({"role": "user", "content": prompt})
                        st.session_state.chat_history.append({"role": "assistant", "content": full_message})
                    else:
                        error_message = "Sorry, I couldn't generate a response."
                        message_placeholder.error(error_message)
                        st.session_state.chat_history.append({"role": "user", "content": prompt})
                        st.session_state.chat_history.append({"role": "assistant", "content": error_message})

                except Exception as e:
                    error_message = f"Sorry, I encountered an error: {str(e)}"
                    message_placeholder.error(error_message)
                    st.session_state.chat_history.append({"role": "user", "content": prompt})
                    st.session_state.chat_history.append({"role": "assistant", "content": error_message})

                st.session_state.pending_prompt = None
                st.rerun()

    # Chat input (stays at bottom)
    if prompt := st.chat_input("Ask a finance question..."):
        st.session_state.pending_prompt = prompt
        st.rerun()

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

    # Check for pending news prompt
    if "news_pending_prompt" not in st.session_state:
        st.session_state.news_pending_prompt = None

    # Display chat history and handle streaming in the same container
    news_chat_container = st.container(height=400)
    with news_chat_container:
        for message in st.session_state.news_chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # If there's a pending prompt, show streaming response
        if st.session_state.news_pending_prompt:
            prompt = st.session_state.news_pending_prompt

            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                message_placeholder.markdown("🔍 Searching for news...▌")

                try:
                    from langchain.messages import AIMessage
                    config = {"configurable": {"thread_id": st.session_state.news_thread_id}}

                    messages = []
                    for msg in st.session_state.news_chat_history:
                        if msg["role"] == "user":
                            messages.append(HumanMessage(content=msg["content"]))
                        else:
                            messages.append(AIMessage(content=msg["content"]))
                    messages.append(HumanMessage(content=prompt))

                    full_response = ""

                    for msg, metadata in news_agent.agent.stream(
                        {"messages": messages},
                        config=config,
                        stream_mode="messages"
                    ):
                        # Stream AI message chunks
                        if hasattr(msg, 'content') and msg.content:
                            if msg.__class__.__name__ in ('AIMessageChunk', 'AIMessage'):
                                if not hasattr(msg, 'tool_calls') or not msg.tool_calls:
                                    full_response += msg.content
                                    message_placeholder.markdown(full_response + "▌")

                    if full_response:
                        message_placeholder.markdown(full_response)
                        st.session_state.news_chat_history.append({"role": "user", "content": prompt})
                        st.session_state.news_chat_history.append({"role": "assistant", "content": full_response})
                    else:
                        error_message = "Sorry, I couldn't find any news."
                        message_placeholder.error(error_message)
                        st.session_state.news_chat_history.append({"role": "user", "content": prompt})
                        st.session_state.news_chat_history.append({"role": "assistant", "content": error_message})

                except Exception as e:
                    error_message = f"Sorry, I encountered an error: {str(e)}"
                    message_placeholder.error(error_message)
                    st.session_state.news_chat_history.append({"role": "user", "content": prompt})
                    st.session_state.news_chat_history.append({"role": "assistant", "content": error_message})

                st.session_state.news_pending_prompt = None
                st.rerun()

    # Chat input (stays at bottom)
    if prompt := st.chat_input("Ask about financial news or search for specific topics...", key="news_chat_input"):
        st.session_state.news_pending_prompt = prompt
        st.rerun()

# Tab 3: Market
with tab3:
    st.header("📊 Market Overview")

    st.markdown("""
    ### Quick Market Lookup
    Enter a stock symbol or ask about market trends.
    """)

    with st.form(key="market_form"):
        market_query = st.text_input("Enter your market question:", placeholder="e.g., What's the current price of AAPL?")
        submit_button = st.form_submit_button("Get Market Data")

    if submit_button and market_query:
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
