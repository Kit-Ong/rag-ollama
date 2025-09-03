from typing import List
from langchain.schema import Document
from langchain_ollama import OllamaLLM
from langchain.schema import Document
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

DATA_PATH = "data/md"

class ChunkMethod:
    LLM = "llm"
    RECURSIVE = "recursive"

class Chunker:
    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 100, chunk_method: str = ChunkMethod.LLM):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        if chunk_method not in (ChunkMethod.LLM, ChunkMethod.RECURSIVE):
            raise ValueError(f"Invalid chunk_method: {chunk_method}. Must be 'llm' or 'recursive'.")
        self.chunk_method = chunk_method

    def __smart_chunk_with_llm(self, documents: List[Document]) -> List[str]:
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
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        rough_chunks = text_splitter.split_documents(documents)
        
        
        final_chunks = []
        
        for chunk in rough_chunks:
            prompt = f"""
            Analyze this text in List[Document] format and find the best place to split it into smaller chunks.
            Return only the character position number where the split should occur.
            The split should be at a natural break point (end of sentence, paragraph, etc).
            the response should be in List[str] format.
            
            Text: {documents}
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

    def __recursive_chunk_without_llm(self, documents: List[Document]) -> List[Document]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        chunks = text_splitter.split_documents(documents)

        document = chunks[10]
        print(document.page_content)
        print(document.metadata)

        return chunks
    
    def chunk(self, documents: List[Document]) -> List[str]:
        if self.chunk_method == ChunkMethod.LLM:
            return self.__smart_chunk_with_llm(documents)
        elif self.chunk_method == ChunkMethod.RECURSIVE:
            return self.__recursive_chunk_without_llm(documents)
        else:
            return self.__recursive_chunk_without_llm(documents)

if __name__ == "__main__":
    
    loader = DirectoryLoader(
        path=DATA_PATH,
        recursive=True
    )
    documents = loader.load()

    chunker = Chunker(chunk_size=100, chunk_overlap=100, chunk_method=ChunkMethod.LLM)
    print(chunker.chunk(documents))