from typing import List, Dict
import logging
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from shared.utils.config import settings
from langchain_core.embeddings import Embeddings
from langchain_experimental.text_splitter import SemanticChunker
from etl_service.embeddings import Embedder

logger = logging.getLogger(__name__)

class LangChainEmbeddingsWrapper(Embeddings):
    """Wrapper for our etl_service.embeddings.Embedder to be compatible with LangChain."""
    def __init__(self, embedder: Embedder):
        self.embedder = embedder

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embedder.generate(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.embedder.generate([text])[0]

class Chunker:
    def __init__(self, chunk_size=None, overlap=None):
        self.chunk_size = chunk_size or settings.ETL_CHUNK_SIZE
        self.overlap = overlap or settings.ETL_CHUNK_OVERLAP

        # 1. Split by headers first
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        self.md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        
        # 2. Semantic Chunker for paragraphs
        self.embedder = Embedder()
        self.lc_embeddings = LangChainEmbeddingsWrapper(self.embedder)
        
        # We use a semantic chunker which looks at embedding similarity to find breakpoints
        self.semantic_splitter = SemanticChunker(
            self.lc_embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=90.0
        )
        
        # 3. Recursive fallback for very large blocks or if semantic fails
        self.char_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def chunk_document(self, pages: List[Dict]) -> List[Dict]:
        """
        Splits text into chunks while preserving page metadata and Markdown structure (especially tables).
        """
        chunks = []
        
        for page in pages:
            text = page["text"]
            page_num = page["page"]
            
            if not text or not text.strip():
                continue
                
            try:
                # 1. Semantic split by Headers
                md_sections = self.md_splitter.split_text(text)
                
                for section in md_sections:
                    # Append header context
                    header_context = " > ".join([v for k, v in section.metadata.items() if k.startswith("Header")])
                    
                    # Split logic blocks: Paragraphs vs Tables
                    content = section.page_content
                    # Avoid split by double newline if it's already a single block
                    blocks = [content] 
                    if "\n\n" in content:
                        blocks = content.split("\n\n")
                    
                    for block in blocks:
                        block = block.strip()
                        if not block:
                            continue
                            
                        # If it's a table, keep it atomic
                        if "|" in block and "---" in block:
                            final_chunks = [block]
                        else:
                            # Use semantic chunker for regular text blocks
                            try:
                                final_chunks = self.semantic_splitter.split_text(block)
                            except Exception as sem_e:
                                logger.debug(f"Semantic split failed for block, falling back to recursive: {sem_e}")
                                final_chunks = self.char_splitter.split_text(block)
                        
                        for chunk_text in final_chunks:
                            chunk_text = chunk_text.strip()
                            if not chunk_text:
                                continue
                                
                            full_text = chunk_text
                            if header_context:
                                full_text = f"Section: {header_context}\n\n{chunk_text}"
                                
                            chunks.append({
                                "text": full_text,
                                "meta": {"page": page_num}
                            })
                    
            except Exception as e:
                logger.warning(f"Markdown chunking failed on page {page_num}: {e}. Falling back to simple chunking.")
                # Fallback to pure length-based chunking
                doc_splits = self.char_splitter.split_text(text)
                for chunk_text in doc_splits:
                    chunks.append({
                        "text": chunk_text,
                        "meta": {"page": page_num}
                    })
                
        return chunks
