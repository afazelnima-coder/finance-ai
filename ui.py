import gradio as gr
from langchain.messages import HumanMessage
import sys
import os
# Add the parent directory to sys.path so Python can find the agents package
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.tax_agent import agent  
# Make sure the agents directory is in your PYTHONPATH or is a package (contains __init__.py)
# If your project structure is:
# /Users/Nima/Documents/IK/cap-proj/
# ├── agents/
# │   ├── __init__.py
# │   └── tax_agent.py
# └── ui.py
# Then this import is correct:
# from agents.tax_agent import agent

# If you don't have __init__.py in agents/, add an empty one to make it a package.
def chat_fn(message, history):
    # history: list of [user, agent] pairs
    response = agent.invoke({"messages": [HumanMessage(content=message)]})
    answer = response["messages"][-1].content
    history = history or []
    history.append((message, answer))
    return history, history

with gr.Blocks() as demo:
    gr.Markdown("# Tax Expert Chatbot")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Ask a tax question")
    clear = gr.Button("Clear")

    def clear_fn():
        return [], []

    msg.submit(chat_fn, [msg, chatbot], [chatbot, chatbot])
    clear.click(clear_fn, None, [chatbot, chatbot])

demo.launch()