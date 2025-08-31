import ollama
import chromadb
from langchain.schema import Document
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

DATA_PATH = "data/books"
embedding_model = OllamaEmbeddings(model="mxbai-embed-large")

def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    return chunks

def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    documents = loader.load()
    return documents


def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    document = chunks[10]
    print(document.page_content)
    print(document.metadata)

    return chunks


client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection(name="docs")
collection_name = "docs"

# Get or create collection
try:
    collection = client.get_collection(collection_name)
    print("Collection"+": "+str(collection.count()))
    print(f"Collection '{collection_name}' exists ✅")
except:
    collection = client.create_collection(collection_name)
    print(f"Created new collection '{collection_name}'")


if collection.count() > 0:
    collection = client.get_collection("docs")
else:
    # # store each document in a vector embedding database
    documents = generate_data_store()

    vectorstore = Chroma.from_documents(
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


# an example input
input = "What is kind of integrity should concerns in android?"

# generate an embedding for the input and retrieve the most relevant doc
response = ollama.embed(
  model="mxbai-embed-large",
  input=input
)
results = collection.query(
  query_embeddings=response["embeddings"],
  n_results=1
)
data = results['documents'][0][0]

# generate a response combining the prompt and data we retrieved in step 2
output = ollama.generate(
  model="llama3.2",
  prompt=f"Using this data: {data}. Respond to this prompt: {input}"
)

print("#### Response RAG ####")
print(data)
print("\n")

print("#### Response LLM ####")
print(output['response'])