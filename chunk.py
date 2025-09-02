from typing import List
from langchain.schema import Document
from langchain_ollama import OllamaLLM
from langchain.schema import Document
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

DATA_PATH = "data/md"

def smart_chunk_with_llm(text: str, chunk_size: int = 50) -> List[str]:
    print("asdfasdfasdf")
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

def recursive_chunk_without_llm(documents: List[Document], chunk_size: int = 100) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=25
    )
    chunks = text_splitter.split_documents(documents)

    document = chunks[10]
    print(document.page_content)
    print(document.metadata)

    return chunks

if __name__ == "__main__":
    
    loader = DirectoryLoader(
        path=DATA_PATH,
        recursive=True
    )
    documents = loader.load()
    # print(smart_chunk_with_llm(documents))
    print(recursive_chunk_without_llm(documents))