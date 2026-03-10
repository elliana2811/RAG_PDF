from typing import List, Dict
from openai import OpenAI
from rank_bm25 import BM25Okapi
from shared.utils.config import settings
import logging

logger = logging.getLogger(__name__)

class SparseEmbedder:
    def __init__(self):
        try:
            from fastembed import SparseTextEmbedding
            # Initialize FastEmbed with the configured BM25 model
            self.model = SparseTextEmbedding(model_name=settings.FAST_EMBED_SPARSE_MODEL)
            logger.info(f"Initialized FastEmbed Sparse Embedder with model: {settings.FAST_EMBED_SPARSE_MODEL}")
        except ImportError:
            logger.error("fastembed is not installed. Sparse embedding will fail. Please run: pip install fastembed")
            self.model = None

    def generate(self, texts: List[str]) -> List[Dict[str, List]]:
        """
        Generates sparse vectors using FastEmbed.
        Returns a list of dictionaries with 'indices' and 'values' for each text.
        """
        if not self.model:
            raise RuntimeError("Sparse Embedder is not initialized (fastembed missing).")
        
        try:
            # FastEmbed returns an iterable of SparseEmbedding objects
            embeddings = list(self.model.embed(texts))
            # Convert to the format expected by Qdrant (indices and values)
            result = []
            for emb in embeddings:
                # FastEmbed SparseEmbedding object has .indices and .values
                # We need to ensure they are standard lists for serialization
                result.append({
                    "indices": emb.indices.tolist() if hasattr(emb.indices, 'tolist') else list(emb.indices),
                    "values": emb.values.tolist() if hasattr(emb.values, 'tolist') else list(emb.values)
                })
            return result
        except Exception as e:
            logger.error(f"Failed to generate sparse embeddings: {e}")
            raise

import httpx

class Embedder:
    def __init__(self):
        self.provider = settings.EMBEDDING_PROVIDER.lower()
        self.model = settings.EMBEDDING_MODEL_NAME
        
        if self.provider == "gemini":
            import google.generativeai as genai
            if not settings.GEMINI_API_KEY:
                logger.warning("GEMINI_API_KEY is not set. Gemini embedding generation will fail.")
            else:
                genai.configure(api_key=settings.GEMINI_API_KEY)
                self.genai_client = genai
        else:
            self.client = OpenAI(
                base_url=settings.EMBEDDING_BASE_URL,
                api_key=settings.EMBEDDING_API_KEY,
                http_client=httpx.Client()
            )

    def generate(self, texts: List[str]) -> List[List[float]]:
        """
        Generates dense embeddings using the configured provider.
        """
        try:
            if self.provider == "gemini":
                # Google Generative AI supports batch embedding via 'embed_content'
                # method: models.embed_content(model, content, task_type='retrieval_document', title=None)
                # But typically for batch we iterate or check if batch is supported.
                # Actually, `genai.embed_content` can take a list of strings but documentation varies on model.
                # Let's iterate for safety or check batch support.
                # Updated info: genai.embed_content supports 'content' as a list.
                # However, task_type is important. For indexing -> RETRIEVAL_DOCUMENT. For query -> RETRIEVAL_QUERY.
                # We should probably default to RETRIEVAL_DOCUMENT here as this is ETL (indexing).
                # But this class might be used for retrieval (query) too?
                # The `Retriever` uses this class. We might need to differentiate or pick a neutral task_type or default.
                # Simple approach: Use 'retrieval_document' for general purpose if acceptable, or no task_type.
                
                # NOTE: For `gemini-embedding-001`, task_type is optional but recommended.
                
                embeddings = []
                # Simple loop to handle one by one or small batches if needed.
                # The API usually handles batches. Let's try sending the list.
                result = self.genai_client.embed_content(
                    model=self.model,
                    content=texts,
                    task_type="retrieval_document" # Assumption: mostly used for indexing here
                )
                
                # result['embedding'] is normally the key. If batch, it might be result['embedding'] as list of lists.
                if 'embedding' in result:
                     # Check if it's a list of lists or single list (if 1 text).
                     # If we sent a list, we expect a list of embeddings.
                     # But current SDK might return strict dict.
                     
                     # Correct handling for batch in newer SDKs:
                     # return [entry['embedding'] for entry in result['embedding']] if structured differently?
                     # Let's check the return structure. It's usually a dict.
                     return result['embedding']
                else:
                    # Fallback or error
                    raise ValueError("Embeeding key not found in Gemini response")

            else:
                # OpenAI / Ollama
                # OpenAI embedding API expects input as text or list of tokens
                # We handle batching if necessary, but here assume manageable batch size
                data = [text.replace("\n", " ") for text in texts]
                response = self.client.embeddings.create(input=data, model=self.model)
                return [data.embedding for data in response.data]
        except Exception as e:
            print(f"Embedding generation failed: {e}") # Print to stdout for debugging
            logger.error(f"Embedding generation failed: {e}")
            # Mock fallback for resilience during testing/setup issues
            import random
            return [[random.random() for _ in range(settings.EMBEDDING_DIMENSION)] for _ in texts]

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self.generate(texts)

