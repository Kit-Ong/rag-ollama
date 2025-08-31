# Import required libraries
from langchain_ollama import OllamaEmbeddings, OllamaLLM
import chromadb
import os
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate

# Define the LLM model to be used
llm_model = "llama3.2"

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

# Configure ChromaDB
# Initialize the ChromaDB client with persistent storage in the current directory
chroma_client = chromadb.PersistentClient(path=os.path.join(os.getcwd(), "chroma"))

class ChromaDBEmbeddingFunction:
    """
    Custom embedding function for ChromaDB using embeddings from Ollama.
    """
    def __init__(self, langchain_embeddings):
        self.langchain_embeddings = langchain_embeddings

    def __call__(self, input):
        # Ensure the input is in a list format for processing
        if isinstance(input, str):
            input = [input]
        return self.langchain_embeddings.embed_documents(input)
        
# Define a custom embedding function for ChromaDB using Ollama
class ChromaDBEmbeddingFunction:
    """
    Custom embedding function for ChromaDB using embeddings from Ollama.
    """
    def __init__(self, langchain_embeddings):
        self.langchain_embeddings = langchain_embeddings

    def __call__(self, input):
        # Ensure the input is in a list format for processing
        if isinstance(input, str):
            input = [input]
        return self.langchain_embeddings.embed_documents(input)

# Initialize the embedding function with Ollama embeddings
embedding = ChromaDBEmbeddingFunction(
    OllamaEmbeddings(
        model=llm_model,
        base_url="http://localhost:11434"  # Adjust the base URL as per your Ollama server configuration
    )
)

# Define a collection for the RAG workflow
collection_name = "rag_collection_demo_1"
collection = chroma_client.get_or_create_collection(
    name=collection_name,
    metadata={"description": "A collection for RAG with Ollama - Demo1"},
    embedding_function=embedding  # Use the custom embedding function
)

# Function to add documents to the ChromaDB collection
def add_documents_to_collection(documents, ids):
    """
    Add documents to the ChromaDB collection.
    
    Args:
        documents (list of str): The documents to add.
        ids (list of str): Unique IDs for the documents.
    """
    collection.add(
        documents=documents,
        ids=ids
    )

# Example: Add sample documents to the collection
documents = [
    "Vision of maybank Indonesia is To be the leading financial services provider in Indonesia, driven by passionately committed and innovative people, creating value and serving communities",
    "Make financial services simple, intuitive, and accessible",
    "Build trusted partnerships for a sustainable future together"
]
doc_ids = ["doc1", "doc2", "doc3"]

# Documents only need to be added once or whenever an update is required. 
# This line of code is included for demonstration purposes:
# add_documents_to_collection(documents, doc_ids)

# Function to query the ChromaDB collection
def query_chromadb(query_text, n_results=1):
    """
    Query the ChromaDB collection for relevant documents.
    
    Args:
        query_text (str): The input query.
        n_results (int): The number of top results to return.
    
    Returns:
        list of dict: The top matching documents and their metadata.
    """
    # results = collection.query(
    #     query_texts=[query_text],
    #     n_results=n_results
    # )
    # return results["documents"], results["metadatas"]

    embedding = OllamaEmbeddings(
            model=llm_model,
            base_url="http://localhost:11434"  # Adjust the base URL as per your Ollama server configuration
    )
    
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding)

    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    print(results[0][1])
    if len(results) == 0 or results[0][1] < 0.3:
        print(f"Unable to find matching results.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)
    llm = OllamaLLM(model=llm_model)
    return llm.invoke(prompt)

# Function to interact with the Ollama LLM
def query_ollama(prompt):
    """
    Send a query to Ollama and retrieve the response.
    
    Args:
        prompt (str): The input prompt for Ollama.
    
    Returns:
        str: The response from Ollama.
    """
    llm = OllamaLLM(model=llm_model)
    return llm.invoke(prompt)

# RAG pipeline: Combine ChromaDB and Ollama for Retrieval-Augmented Generation
def rag_pipeline(query_text):
    
    """
    Perform Retrieval-Augmented Generation (RAG) by combining ChromaDB and Ollama.
    
    Args:
        query_text (str): The input query.
    
    Returns:
        str: The generated response from Ollama augmented with retrieved context.
    """
    # Step 1: Retrieve relevant documents from ChromaDB
    # retrieved_docs, metadata = query_chromadb(query_text)
    response = query_chromadb(query_text)
    # context = " ".join(retrieved_docs[0]) if retrieved_docs else "No relevant documents found."

    # print("######## Retrieved Context ########")
    # print(context)

    # # Step 2: Send the query along with the context to Ollama
    # augmented_prompt = f"Context: {context}\n\nQuestion: {query_text}\nAnswer:"
    # print("######## Augmented Prompt ########")
    # print(augmented_prompt)

    # response = query_ollama(augmented_prompt)
    return response

# Example usage
# Define a query to test the RAG pipeline
query = "How does Alice meet the Mad Hatter?"  # Change the query as needed
response = rag_pipeline(query)
# print("######## Response from LLM ########\n", response)