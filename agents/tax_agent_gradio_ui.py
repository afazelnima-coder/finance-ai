# This is a simple general-purpose chatbot built on top of LangChain and Gradio.
# Before running this, make sure you have exported your OpenAI API key as an environment variable:
# export OPENAI_API_KEY="your-openai-api-key"

from langchain_openai import ChatOpenAI  
from langchain.messages import AIMessage, HumanMessage  
import gradio as gr
from dotenv import load_dotenv  
from tax_agent import agent

load_dotenv()

# model = ChatOpenAI(model="gpt-4o-mini")

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
    agent_response = agent.invoke({"messages": history_langchain_format})
    # Extract the assistant's reply
    if isinstance(agent_response, dict) and "messages" in agent_response:
        last_msg = agent_response["messages"][-1]
        answer = getattr(last_msg, "content", str(last_msg))
    elif hasattr(agent_response, "content"):
        answer = agent_response.content
    else:
        answer = str(agent_response)
    return answer

demo = gr.ChatInterface(
    predict,
    api_name="chat",
)

demo.launch()