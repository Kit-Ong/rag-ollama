import os
import chromadb
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

DATA_PATH = "data/md"
embedding_model = OllamaEmbeddings(model="mxbai-embed-large")
generative_model = OllamaLLM(model="codellama")

def load_and_chunk_documents():
    print("Loading documents...")
    loader = DirectoryLoader(path=DATA_PATH, recursive=True)
    documents = loader.load()
    
    print("Chunking documents...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    return chunks

def setup_vectorstore():
    print("Setting up vector store...")
    documents = load_and_chunk_documents()
    
    # Create vector store
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory="chroma_db"
    )
    print("Vector store ready!")
    return vectorstore

def query_documents(vectorstore, query):
    print(f"Searching for: {query}")
    
    # Search for relevant documents
    docs = vectorstore.similarity_search(query, k=3)
    
    if not docs:
        return "No relevant documents found."
    
    # Use the most relevant document
    context = docs[0].page_content
    
    # Create prompt
    template = """
    Using this data: {context}
    Answer this question: {question}
    Only use the provided data to answer.
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | generative_model
    
    print("Generating response...")
    response = chain.invoke({"context": context, "question": query})
    
    return response

def main():
    print("Initializing RAG system...")
    vectorstore = setup_vectorstore()
    
    print("\nRAG system ready!")
    
    while True:
        user_input = input("\nEnter your question (or 'quit' to exit): ")
        
        if user_input.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break
            
        try:
            answer = query_documents(vectorstore, user_input)
            print(f"\nAnswer: {answer}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()