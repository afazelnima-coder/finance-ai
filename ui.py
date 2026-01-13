import gradio as gr
from dotenv import load_dotenv
from langchain.messages import HumanMessage
import sys
import os
import uuid

# Add the parent directory to sys.path so Python can find the agents package
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents import router_agent_v2 as router_agent

load_dotenv()

# Store conversation threads
conversation_threads = {}

def chat(message, history, thread_id):
    """
    Process user message and return response.

    Args:
        message: User's input message
        history: Chat history (list of [user_msg, bot_msg] pairs)
        thread_id: Unique identifier for this conversation thread

    Returns:
        Updated history and empty message box
    """
    if not message.strip():
        return history, ""

    # Capture print output to detect which agent was called
    import io
    from contextlib import redirect_stdout

    f = io.StringIO()
    called_agent = None

    try:
        # Ensure thread_id exists
        if thread_id not in conversation_threads:
            conversation_threads[thread_id] = str(uuid.uuid4())

        # Configure the agent with the thread
        config = {"configurable": {"thread_id": conversation_threads[thread_id]}}

        # Invoke the router agent while capturing stdout
        with redirect_stdout(f):
            response = router_agent.agent.invoke(
                {"messages": [HumanMessage(content=message)]},
                config=config
            )

        # Check captured output for agent routing
        output = f.getvalue()
        if "Router -> QA Agent" in output:
            called_agent = "QA Agent (RAG)"
        elif "Router -> Market Agent" in output:
            called_agent = "Market Agent"
        elif "Router -> News Agent" in output:
            called_agent = "News Agent"
        elif "Router -> Tax Agent" in output:
            called_agent = "Tax Agent"
        elif "Router -> Goal Agent" in output:
            called_agent = "Goal Agent"
        elif "Router -> Portfolio Agent" in output:
            called_agent = "Portfolio Agent"

        # Print to console for debugging
        if called_agent:
            print(f"🤖 Routed to: {called_agent}")

        # Extract the response text - handle both dict and object formats
        last_message = response["messages"][-1]
        if hasattr(last_message, 'content'):
            bot_message = last_message.content
        elif isinstance(last_message, dict):
            bot_message = last_message.get('content', str(last_message))
        else:
            bot_message = str(last_message)

        # Add agent info to the response
        if called_agent:
            bot_message = f"*[Routed to: {called_agent}]*\n\n{bot_message}"

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        bot_message = f"Sorry, I encountered an error: {str(e)}"
        print("Full error traceback:")
        print(error_details)

    # Append to history - Gradio 6.0 expects dict format with role and content
    if history is None:
        history = []

    # Add user message
    history.append({"role": "user", "content": message})
    # Add assistant response
    history.append({"role": "assistant", "content": bot_message})

    return history, ""

def clear_conversation():
    """Clear the conversation and start fresh."""
    new_thread_id = str(uuid.uuid4())
    return [], new_thread_id

# Create the Gradio interface
with gr.Blocks(title="Finance Assistant") as demo:
    gr.Markdown(
        """
        # 💼 Finance Assistant

        Ask me anything about finance! I can help with:
        - **General Finance Questions** (QA Agent with RAG knowledge base)
        - **Market Analysis** (Current prices & trends)
        - **Financial News** (Latest updates)
        - **Tax Information** (Tax-related queries)
        - **Financial Goals** (Planning & advice)
        - **Portfolio Management** (Investment portfolios)
        """
    )

    # Hidden state for thread_id
    thread_id = gr.State(value=str(uuid.uuid4()))

    # Chatbot interface
    chatbot = gr.Chatbot(
        label="Chat",
        height=500
    )

    with gr.Row():
        msg = gr.Textbox(
            label="Your message",
            placeholder="Ask a finance question...",
            scale=4,
            show_label=False
        )
        submit_btn = gr.Button("Send", variant="primary", scale=1)

    with gr.Row():
        clear_btn = gr.Button("🔄 New Conversation", variant="secondary")

    gr.Markdown(
        """
        ### Example Questions:
        - What is a stock option?
        - What's the current price of Apple stock?
        - What are the latest financial news?
        - How do I calculate capital gains tax?
        - How should I plan for retirement?
        - What's a good portfolio allocation strategy?
        """
    )

    # Event handlers
    msg.submit(chat, inputs=[msg, chatbot, thread_id], outputs=[chatbot, msg])
    submit_btn.click(chat, inputs=[msg, chatbot, thread_id], outputs=[chatbot, msg])
    clear_btn.click(clear_conversation, outputs=[chatbot, thread_id])

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft()
    )
