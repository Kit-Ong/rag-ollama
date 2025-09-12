from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

print("Step 1: Loading documents...")
loader = DirectoryLoader(path="data/md", recursive=True)
documents = loader.load()
print(f"Loaded {len(documents)} documents")

print("Step 2: Splitting documents...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks")

print("Step 3: Creating embeddings (this may take a while)...")
embedding_model = OllamaEmbeddings(model="mxbai-embed-large")

print("Step 4: Creating vector store...")
vectorstore = Chroma.from_documents(
    documents=chunks[:5],  # Only use first 5 chunks for testing
    embedding=embedding_model,
    persist_directory="test_chroma_db"
)

print("Done! Vector store created successfully.")