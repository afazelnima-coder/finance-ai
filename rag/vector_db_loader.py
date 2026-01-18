from datasets import load_dataset
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import os

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

# 1. Load a dataset (using a subset for faster testing)
dataset = load_dataset("FinLang/investopedia-embedding-dataset")

# Use first 500 items for a solid knowledge base
# This covers a wide range of finance topics while keeping costs reasonable
MAX_DOCS = 500
docs = []
for i, item in enumerate(dataset['train']):
    if i >= MAX_DOCS:
        break
    # Combine Topic, Title, Question, and Answer for rich context
    doc_text = f"Topic: {item['Topic']}\nTitle: {item['Title']}\n\nQuestion: {item['Question']}\n\nAnswer: {item['Answer']}"
    docs.append(doc_text)

print(f"Loaded {len(docs)} documents from Investopedia dataset")

# 2. Split documents
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs_split = splitter.create_documents(docs)  # create_documents expects list of strings
print(f"Split into {len(docs_split)} chunks")

# 3. Create embeddings and FAISS vector store
# Using OpenAI embeddings (small-scale, should be quick and cheap for 75 docs)
print("\nCreating embeddings with OpenAI...")
print(f"Estimated cost: ~${len(docs_split) * 0.0001:.3f} (very low for 75 documents)")
embeddings = OpenAIEmbeddings()

print("\nCreating vector store...")
vectorstore = FAISS.from_documents(docs_split, embeddings)
print("✓ Vector store created")

# 4. Save the vector store for later use
vectorstore.save_local("investopedia_faiss_index")
print("✓ Vector store saved to 'investopedia_faiss_index'")

# 5. Test retrieval with sample queries
print("\n" + "="*60)
print("Testing RAG system with financial queries")
print("="*60)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

test_queries = [
    "What is a stock option?",
    "How do dividends work?",
    "What is the difference between stocks and bonds?"
]

for query in test_queries:
    print(f"\n📊 Query: {query}")
    print("-" * 60)
    results = retriever.invoke(query)

    for i, doc in enumerate(results, 1):
        print(f"\n[Result {i}]")
        # Show first 300 characters of the result
        content = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
        print(content)
    print("\n" + "="*60)