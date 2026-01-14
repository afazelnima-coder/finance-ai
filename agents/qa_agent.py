from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.tools import tool
from langchain.messages import HumanMessage
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

load_dotenv()

# Load the RAG vector store (lazy loading - only loaded when needed)
_vectorstore = None

def get_vectorstore():
    """Lazy load the vector store."""
    global _vectorstore
    if _vectorstore is None:
        embeddings = OpenAIEmbeddings()
        _vectorstore = FAISS.load_local(
            "investopedia_faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )
    return _vectorstore

@tool
def searchFinance(query: str) -> str:
    """Searches for financial information in the Investopedia knowledge base using RAG.

    Args:
        query: The financial question or topic to search for

    Returns:
        Relevant financial information from Investopedia articles
    """
    try:
        vectorstore = get_vectorstore()
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        results = retriever.invoke(query)

        # Format the results
        if not results:
            return "No relevant information found in the knowledge base."

        formatted_results = []
        for i, doc in enumerate(results, 1):
            formatted_results.append(f"Source {i}:\n{doc.page_content}\n")

        return "\n".join(formatted_results)

    except FileNotFoundError:
        return "Error: RAG knowledge base not found. Please run 'python rag/vector_db_loader.py' first to build the index."
    except Exception as e:
        return f"Error searching knowledge base: {str(e)}"

# Create the agent with streaming enabled
llm = ChatOpenAI(model="gpt-4o-mini", streaming=True)

agent = create_agent(
    llm,
    tools=[searchFinance],
    system_prompt="""You are a helpful financial advisor assistant specialized in finance questions.

When answering questions:
1. Use the searchFinance tool to retrieve relevant information from the Investopedia knowledge base
2. Synthesize the information from the retrieved sources into a clear, concise answer
3. Only answer questions related to finance, investments, economics, and business
4. If asked about non-financial topics (like medical, legal, etc.), politely decline and remind the user you specialize in finance

Always base your answers on the retrieved information from the knowledge base."""
)

if __name__ == "__main__":
    # Example interaction
    user_input = "What is the difference between a 401k and a Roth IRA?"
    response = agent.invoke({"messages": [HumanMessage(content=user_input)]})

    print(response["messages"][-1].content)