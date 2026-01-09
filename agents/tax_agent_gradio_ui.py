from langchain_openai import ChatOpenAI  
from langchain.messages import AIMessage, HumanMessage  
import gradio as gr
from dotenv import load_dotenv  
from tax_agent import agent

load_dotenv()

# On every call, we get the full chat history and the new message
def predict(message, history):
    history_langchain_format = []

    # we convert the history into LangChain message format
    for pair in history:
        if len(pair) == 2:
            user, bot = pair
            if user:
                history_langchain_format.append(HumanMessage(content=user))
            if bot:
                history_langchain_format.append(AIMessage(content=bot))

    # then we add the latest user message
    history_langchain_format.append(HumanMessage(content=message))
    
    # call the agent with the formatted history
    agent_response = agent.invoke({"messages": history_langchain_format})
    
    # Extract the assistant's reply
    # agent_response should be a dict with "messages" key
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