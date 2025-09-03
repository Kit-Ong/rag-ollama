import os
import chromadb
from typing import List
from langchain.schema import Document
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

DATA_PATH = "data/md"
# DATA_PATH = "[PATH_TO_YOUR_FOLDER]"
embedding_model = OllamaEmbeddings(model="mxbai-embed-large")
generative_model = OllamaLLM(model="codellama")

client = chromadb.PersistentClient(path="chroma_db")
collection_name = "docs"

###
# Generate data store by chunking documents
# @return: List of text chunks
###
def generate_data_store():
    documents = load_documents()
    chunks = smart_chunk_with_llm(documents)
    return chunks

###
# Load documents from the specified directory
# @return: List of Document objects
###
def load_documents():
    loader = DirectoryLoader(
        path=DATA_PATH,
        recursive=True
    )
    documents = loader.load()
    return documents

###
# This use conventional chunking without LLM using RecursiveCharacterTextSplitter by langchain
# @param documents: list of Document objects to be chunked
###
def conventional_chunk(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    document = chunks[10]
    print(document.page_content)
    print(document.metadata)

    return chunks

###
# This use LLM to find natural semantic boundaries for chunking
# @param documents: list of Document objects to be chunked
###
def smart_chunk_with_llm(documents: list[Document], chunk_size: int = 50) -> List[str]:
    """
    Chunks text using LLM to find natural semantic boundaries
    
    Args:
        text: Input text to chunk
        chunk_size: Target chunk size (approximate)
    
    Returns:
        List of text chunks
    """
    llm = OllamaLLM(model="codellama")
    
    # Initial rough splits to handle large texts
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
    )
    rough_chunks = text_splitter.split_documents(documents)
    
    final_chunks = []
    
    for chunk in rough_chunks:
        prompt = f"""
        Analyze this text and find the best place to split it into smaller chunks.
        Return only the character position number where the split should occur.
        The split should be at a natural break point (end of sentence, paragraph, etc).
        
        Text: {chunk}
        """
        
        try:
            # Get split point from LLM
            split_point = int(llm.predict(prompt).strip())
            
            if 0 < split_point < len(chunk):
                final_chunks.append(chunk[:split_point])
                final_chunks.append(chunk[split_point:])
            else:
                final_chunks.append(chunk)
                
        except:
            # Fallback: just use the rough chunk
            final_chunks.append(chunk)
    
    return [c for c in final_chunks if c]

###
# Process documents to create or get existing embedding collection
# @return: ChromaDB collection object
###
def process_to_embedding():
    # Get or create collection
    try:
        collection = client.get_collection(collection_name)
    except:
        collection = client.create_collection(collection_name)


    if collection.count() > 0:
        collection = client.get_collection("docs")
    else:
        # # store each document in a vector embedding database
        documents = generate_data_store()

        Chroma.from_documents(
            documents=documents,
            embedding=embedding_model,
            collection_name=collection_name,
            persist_directory="chroma_db"  # saves locally
        )
        for i, doc in enumerate(documents):
            embedding = embedding_model.embed_query(doc.page_content)
            collection.add(
                documents=[doc.page_content],
                metadatas=[doc.metadata],
                ids=[f"doc_{i}"],
                embeddings=[embedding]
            )
    return collection

###
# Query the local vector database for relevant documents
# @param user_input: the question to be answered
# @return: top 3 relevant documents
###
def query_to_local(user_input):
    collection2 = process_to_embedding() # ensure collection is ready
    response = embedding_model.embed_query(text=user_input) # embedding query so we can search on vectors

    # query the collection for relevant documents
    results = collection2.query(
            query_embeddings = response,
            n_results=3
        )
    
    return results


###
# Main prompting function to handle user input and generate answers
###
def prompting():
    print("\nHello! This is a simple question-answering application using Ollama and ChromaDB.")

    while True:
        user_input = input("Please enter your question ('exit' to quit, 'cls' to clear): ")
        if user_input.lower() == 'exit':
            print("Exiting the application. Goodbye!")
            break

        elif user_input.lower() == 'cls':
            os.system('cls' if os.name == 'nt' else 'clear')
            continue


        ###
        #  1. RETRIEVE (supporting data)
        #  How -> query the local vector database for relevant documents
        #  will return top 3 relevant documents
        ###
        results = query_to_local(user_input)

        ###
        #  if no relevant documents found, inform the user
        #  relevant result only if distance is less than 1 (tweak as needed) and greater than 0.1
        #  else use the relevant document to answer the question
        ###
        if results['distances'][0][0] > 1:
            data = "No relevant data found."
            output = "I'm sorry, I don't have enough information to answer that question."
        else:
            data = results['documents'][0][0]

            ###
            #  2. AUGMENT (using the retrieved data)
            #  How -> use a prompt template to combine the user question and the retrieved data
            ###
            template = """
                Using this data: {data}. 
                Respond to this prompt: {question}.
                Dont use any data outside of this text to answer the question.
                """
            prompt = ChatPromptTemplate.from_template(template)
            
            ###
            #  3. GENERATE (the final answer)
            #  How -> use the generative LLM to answer the question based on the augmented prompt
            ###
            chain = prompt | generative_model
            output = chain.invoke({"question": user_input, "data": data})

        print("\n")
        print("#### Your Question ####")
        print(user_input)
        print("\n")
        print("\n#### Response RAG ####")
        print(data)
        print("\n")

        print("#### Response LLM ####")
        print(output)
        print("\n")

if __name__ == "__main__":
    prompting()
