import os
import chromadb
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM

# Test simple functionality
print("Testing document loading...")
loader = DirectoryLoader(path="data/md", recursive=True)
documents = loader.load()
print(f"Loaded {len(documents)} documents")

print("Testing text splitting...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
chunks = text_splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")

print("Testing embedding model...")
embedding_model = OllamaEmbeddings(model="mxbai-embed-large")
test_embedding = embedding_model.embed_query("test query")
print(f"Embedding created with dimension: {len(test_embedding)}")

print("Testing LLM...")
llm = OllamaLLM(model="codellama")
response = llm.invoke("Say hello")
print(f"LLM response: {response}")

print("All tests passed!")