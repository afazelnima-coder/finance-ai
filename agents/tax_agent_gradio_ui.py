from langchain_openai import ChatOpenAI  
from langchain.messages import AIMessage, HumanMessage  
import gradio as gr
from dotenv import load_dotenv  
from tax_agent import agent

load_dotenv()

# On every call, we get the full chat history and the new message
def predict(message, history):
    history_langchain_format = []
    for pair in history:
        if len(pair) == 2:
            user, bot = pair
            if user:
                history_langchain_format.append(HumanMessage(content=user))
            if bot:
                history_langchain_format.append(AIMessage(content=bot))
    history_langchain_format.append(HumanMessage(content=message))

    # Call the agent
    agent_response = agent.invoke({"messages": history_langchain_format})

    # Prepare new history for Gradio using ChatMessage
    new_history = []
    for pair in history:
        if len(pair) == 2:
            user, bot = pair
            if user:
                new_history.append(gr.ChatMessage(role="user", content=user))
            if bot:
                new_history.append(gr.ChatMessage(role="assistant", content=bot))

    # Add the latest user message
    new_history.append(gr.ChatMessage(role="user", content=message))

    # Show all new messages from agent (including tool calls)
    if isinstance(agent_response, dict) and "messages" in agent_response:
        prev_len = len(history_langchain_format)
        new_msgs = agent_response["messages"][prev_len:]
        for msg in new_msgs:
            # Check for tool_calls attribute
            tool_calls = getattr(msg, "tool_calls", None)
            content = getattr(msg, "content", str(msg))
            if tool_calls:
                # Display each tool call
                for tool_call in tool_calls:
                    tool_name = tool_call.get("name", "unknown tool")
                    tool_args = tool_call.get("args", {})
                    new_history.append(
                        gr.ChatMessage(
                            role="assistant",
                            content=f"🔧 Tool used: {tool_name} with args {tool_args}"
                        )
                    )
            if content:
                new_history.append(gr.ChatMessage(role="assistant", content=content))
    elif hasattr(agent_response, "content"):
        new_history.append(gr.ChatMessage(role="assistant", content=agent_response.content))
    else:
        new_history.append(gr.ChatMessage(role="assistant", content=str(agent_response)))

    return new_history

demo = gr.ChatInterface(
    predict,
    api_name="chat",
)

demo.launch()