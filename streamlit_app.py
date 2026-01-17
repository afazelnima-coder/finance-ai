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

    import yfinance as yf
    import plotly.graph_objects as go
    from datetime import datetime, timedelta
    from langchain_openai import ChatOpenAI

    # LLM for ticker extraction
    @st.cache_resource
    def get_ticker_llm():
        return ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def extract_ticker(query: str) -> str | None:
        """Use LLM to extract ticker symbol if a specific company is mentioned."""
        llm = get_ticker_llm()
        prompt = f"""Analyze this market question and determine if it mentions a specific publicly traded company.

If a specific company is mentioned, return its stock ticker symbol.
- For US companies, return the US ticker (e.g., AAPL, MSFT)
- For international companies trading as ADRs in the US, return the ADR ticker
- For international companies not on US exchanges, use the format: TICKER.EXCHANGE
  (e.g., .T for Tokyo, .L for London, .HK for Hong Kong, .DE for Germany)

If NO specific company is mentioned (general market questions), return "NONE".

Examples:
- "What's the price of Apple?" → AAPL
- "How is Tesla doing today?" → TSLA
- "Show me Microsoft's chart" → MSFT
- "What's driving tech stocks?" → NONE
- "Is the market up today?" → NONE
- "Amazon earnings report" → AMZN
- "Compare nvidia to AMD" → NVDA
- "What's happening with meta?" → META
- "Sony stock price" → SONY
- "How is Nintendo doing?" → NTDOY
- "Toyota stock" → TM
- "Samsung electronics" → 005930.KS
- "Alibaba stock" → BABA
- "TSMC performance" → TSM
- "BMW stock" → BMW.DE
- "Honda" → HMC
- "Nestle" → NSRGY
- "SAP" → SAP
- "Spotify" → SPOT
- "Shopify" → SHOP
- "NIO stock" → NIO
- "BYD company" → BYDDY
- "Tencent" → TCEHY
- "SoftBank" → SFTBY
- "ASML" → ASML
- "Novo Nordisk" → NVO

Index funds, ETFs, and market indices:
- "S&P 500" → ^GSPC
- "SPY ETF" → SPY
- "How is the S&P doing?" → ^GSPC
- "Nasdaq" → ^IXIC
- "QQQ" → QQQ
- "Nasdaq 100 ETF" → QQQ
- "Dow Jones" → ^DJI
- "the Dow" → ^DJI
- "DIA ETF" → DIA
- "Russell 2000" → ^RUT
- "IWM" → IWM
- "VTI total market" → VTI
- "VOO" → VOO
- "Vanguard S&P 500" → VOO
- "VIX" → ^VIX
- "volatility index" → ^VIX
- "fear index" → ^VIX
- "bond ETF" → BND
- "AGG bonds" → AGG
- "tech sector ETF" → XLK
- "XLF financials" → XLF
- "energy sector" → XLE
- "gold ETF" → GLD
- "silver ETF" → SLV
- "emerging markets" → EEM
- "international stocks ETF" → VXUS
- "ARK Innovation" → ARKK
- "Bitcoin ETF" → IBIT
- "Ethereum ETF" → ETHA

User question: "{query}"

Respond with ONLY the ticker symbol or "NONE", nothing else."""

        response = llm.invoke([HumanMessage(content=prompt)])
        result = response.content.strip().upper()
        return None if result == "NONE" else result

    def display_stock_chart(ticker: str, period: str = "6mo"):
        """Display stock chart and metrics for a ticker."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            hist = stock.history(period=period)

            if hist.empty:
                st.warning(f"No data found for ticker: {ticker}")
                return

            # Company header
            company_name = info.get("longName", ticker)
            st.subheader(f"📈 {company_name} ({ticker})")

            # Key metrics row
            col1, col2, col3, col4 = st.columns(4)

            current_price = info.get("currentPrice") or info.get("regularMarketPrice") or hist["Close"].iloc[-1]
            prev_close = info.get("previousClose") or (hist["Close"].iloc[-2] if len(hist) > 1 else current_price)
            price_change = current_price - prev_close
            price_change_pct = (price_change / prev_close) * 100 if prev_close else 0

            with col1:
                st.metric("Price", f"${current_price:.2f}", f"{price_change:+.2f} ({price_change_pct:+.2f}%)")
            with col2:
                volume = info.get("volume") or info.get("regularMarketVolume") or 0
                st.metric("Volume", f"{volume:,.0f}")
            with col3:
                market_cap = info.get("marketCap", 0)
                if market_cap >= 1e12:
                    st.metric("Market Cap", f"${market_cap/1e12:.2f}T")
                elif market_cap >= 1e9:
                    st.metric("Market Cap", f"${market_cap/1e9:.2f}B")
                else:
                    st.metric("Market Cap", f"${market_cap/1e6:.2f}M")
            with col4:
                pe_ratio = info.get("trailingPE", "N/A")
                st.metric("P/E Ratio", f"{pe_ratio:.2f}" if isinstance(pe_ratio, (int, float)) else pe_ratio)

            # Price chart
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=hist.index,
                open=hist['Open'],
                high=hist['High'],
                low=hist['Low'],
                close=hist['Close'],
                name='Price'
            ))

            if len(hist) >= 20:
                hist['MA20'] = hist['Close'].rolling(window=20).mean()
                fig.add_trace(go.Scatter(x=hist.index, y=hist['MA20'], mode='lines', name='MA20', line=dict(color='orange', width=1)))
            if len(hist) >= 50:
                hist['MA50'] = hist['Close'].rolling(window=50).mean()
                fig.add_trace(go.Scatter(x=hist.index, y=hist['MA50'], mode='lines', name='MA50', line=dict(color='blue', width=1)))

            fig.update_layout(
                xaxis_rangeslider_visible=False,
                height=350,
                margin=dict(l=0, r=0, t=30, b=0),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error fetching stock data: {str(e)}")

    # Single unified input
    st.markdown("Ask any market question. If you mention a company, I'll show you the chart too!")

    with st.form(key="market_form"):
        market_query = st.text_input("Your question:", placeholder="e.g., How is Apple doing? / What's driving the market today?")
        col1, col2 = st.columns([3, 1])
        with col1:
            submit_button = st.form_submit_button("Ask", use_container_width=True)
        with col2:
            chart_period = st.selectbox("Chart period:", ["1mo", "3mo", "6mo", "1y"], index=2, label_visibility="collapsed")

    if submit_button and market_query:
        # Extract ticker if company mentioned
        with st.spinner("Analyzing question..."):
            ticker = extract_ticker(market_query)

        # Show chart if specific company detected
        if ticker:
            display_stock_chart(ticker, chart_period)
            st.markdown("---")

        # Get AI response
        st.markdown("### 💬 Market Analysis")
        with st.spinner("Getting market insights..."):
            try:
                from agents import market_agent
                response = market_agent.agent.invoke(
                    {"messages": [HumanMessage(content=market_query)]}
                )
                market_response = response["messages"][-1].content
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
