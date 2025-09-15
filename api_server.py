from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import asyncio
import shutil
from pathlib import Path

# Import existing functionality
from query_ollama import (
    query_to_local, 
    process_to_embedding, 
    generate_data_store,
    embedding_model,
    generative_model,
    client,
    collection_name
)
from langchain_core.prompts import ChatPromptTemplate

app = FastAPI(title="RAG Ollama API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # NextJS default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class QueryRequest(BaseModel):
    question: str
    use_rag: bool = True

class QueryResponse(BaseModel):
    question: str
    answer: str
    retrieved_data: str
    distance: float
    processing_time: float

class StatusResponse(BaseModel):
    collection_exists: bool
    document_count: int
    total_chunks: int
    status_message: str

class ConfigRequest(BaseModel):
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    embedding_model_name: Optional[str] = None
    generative_model_name: Optional[str] = None

# Global configuration
config = {
    "chunk_size": 300,
    "chunk_overlap": 100,
    "embedding_model": "mxbai-embed-large",
    "generative_model": "codellama"
}

@app.get("/")
async def root():
    return {"message": "RAG Ollama API is running!"}

@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get the current status of the vector database and document processing."""
    try:
        # Check if collection exists and get document count
        try:
            collection = client.get_collection(collection_name)
            doc_count = collection.count()
            collection_exists = True
            status_msg = f"Collection '{collection_name}' is ready with {doc_count} documents"
        except:
            doc_count = 0
            collection_exists = False
            status_msg = "Collection not found. Please process documents first."
        
        return StatusResponse(
            collection_exists=collection_exists,
            document_count=doc_count,
            total_chunks=doc_count,  # In this case, each document is a chunk
            status_message=status_msg
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting status: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query the RAG system with a question."""
    import time
    
    start_time = time.time()
    
    try:
        if request.use_rag:
            # Use RAG pipeline
            results = query_to_local(request.question)
            
            # Check if relevant documents found
            if results['distances'][0][0] > 1:
                retrieved_data = "No relevant data found."
                answer = "I'm sorry, I don't have enough information to answer that question."
                distance = results['distances'][0][0]
            else:
                retrieved_data = results['documents'][0][0]
                distance = results['distances'][0][0]
                
                # Generate answer using the retrieved data
                template = """
                Using this data: {data}. 
                Respond to this prompt: {question}.
                Don't use any data outside of this text to answer the question.
                """
                prompt = ChatPromptTemplate.from_template(template)
                chain = prompt | generative_model
                answer = chain.invoke({"question": request.question, "data": retrieved_data})
        else:
            # Direct LLM query without RAG
            answer = generative_model.invoke(request.question)
            retrieved_data = "No RAG data used - direct LLM response"
            distance = 0.0
        
        processing_time = time.time() - start_time
        
        return QueryResponse(
            question=request.question,
            answer=answer,
            retrieved_data=retrieved_data,
            distance=distance,
            processing_time=processing_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/process-documents")
async def process_documents():
    """Process documents and create embeddings."""
    try:
        # This will create embeddings if they don't exist
        collection = process_to_embedding()
        doc_count = collection.count()
        
        return {
            "message": f"Documents processed successfully. Created embeddings for {doc_count} chunks.",
            "document_count": doc_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing documents: {str(e)}")

@app.post("/upload-document")
async def upload_document(file: UploadFile = File(...)):
    """Upload a new document to the data directory."""
    try:
        # Ensure the upload directory exists
        data_dir = Path("data/md")
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the uploaded file
        file_path = data_dir / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        return {
            "message": f"File '{file.filename}' uploaded successfully",
            "file_path": str(file_path),
            "note": "Run /process-documents to update embeddings"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")

@app.delete("/clear-database")
async def clear_database():
    """Clear the vector database."""
    try:
        # Delete the collection if it exists
        try:
            client.delete_collection(collection_name)
            message = f"Collection '{collection_name}' deleted successfully"
        except:
            message = f"Collection '{collection_name}' was not found"
        
        return {"message": message}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing database: {str(e)}")

@app.get("/config")
async def get_config():
    """Get current configuration."""
    return {
        "config": config,
        "available_models": {
            "embedding": ["mxbai-embed-large", "nomic-embed-text"],
            "generative": ["codellama", "llama2", "mistral", "phi"]
        }
    }

@app.post("/config")
async def update_config(request: ConfigRequest):
    """Update configuration settings."""
    try:
        updated_fields = []
        
        if request.chunk_size is not None:
            config["chunk_size"] = request.chunk_size
            updated_fields.append(f"chunk_size: {request.chunk_size}")
        
        if request.chunk_overlap is not None:
            config["chunk_overlap"] = request.chunk_overlap
            updated_fields.append(f"chunk_overlap: {request.chunk_overlap}")
        
        if request.embedding_model_name is not None:
            config["embedding_model"] = request.embedding_model_name
            updated_fields.append(f"embedding_model: {request.embedding_model_name}")
        
        if request.generative_model_name is not None:
            config["generative_model"] = request.generative_model_name
            updated_fields.append(f"generative_model: {request.generative_model_name}")
        
        return {
            "message": f"Configuration updated: {', '.join(updated_fields)}",
            "config": config,
            "note": "Restart server and reprocess documents for model changes to take effect"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating config: {str(e)}")

@app.get("/documents")
async def list_documents():
    """List all documents in the data directory."""
    try:
        data_dir = Path("data/md")
        if not data_dir.exists():
            return {"documents": [], "message": "Data directory not found"}
        
        documents = []
        for file_path in data_dir.rglob("*"):
            if file_path.is_file():
                stat = file_path.stat()
                documents.append({
                    "name": file_path.name,
                    "path": str(file_path.relative_to(data_dir)),
                    "size": stat.st_size,
                    "modified": stat.st_mtime
                })
        
        return {
            "documents": documents,
            "total_count": len(documents)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)