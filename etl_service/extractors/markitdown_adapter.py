from markitdown import MarkItDown
from openai import OpenAI
from shared.utils.config import settings
import logging
import os
import tempfile

logger = logging.getLogger(__name__)

class MarkItDownAdapter:
    _instance = None
    _client = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            # Configure OpenAI client for MarkItDown
            # MarkItDown typically accepts an 'llm_client' or 'llm_model' argument during conversion
            # or in constructor depending on version. 
            # Assuming standard usage: md = MarkItDown(llm_client=..., llm_model=...)
            
            cls._client = OpenAI(
                base_url=settings.LLM_BASE_URL,
                api_key=settings.LLM_API_KEY
            )
            
            try:
                cls._instance = MarkItDown(
                    llm_client=cls._client,
                    llm_model=settings.LLM_MODEL_NAME
                )
            except Exception as e:
                logger.warning(f"Failed to init MarkItDown with LLM: {e}. Fallback to basic mode.")
                cls._instance = MarkItDown()
                
        return cls._instance, cls._client

    @staticmethod
    def extract_text(file_content: bytes, filename: str) -> str:
        """
        Uses MarkItDown to convert file content to Markdown text.
        Handles temporary file creation since libraries often need file paths.
        """
        md, client = MarkItDownAdapter.get_instance()
        
        # Determine extension
        _, ext = os.path.splitext(filename)
        if not ext:
            ext = ".tmp"
            
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(file_content)
            tmp_path = tmp.name
            
        try:
            # Perform conversion
            # Note: MarkItDown might use 'convert' or similar method
            result = md.convert(tmp_path)
            
            # Result usually has .text_content or similar
            if hasattr(result, "text_content"):
                return result.text_content
            return str(result)
            
        except Exception as e:
            logger.error(f"MarkItDown conversion failed for {filename}: {e}")
            raise e
        finally:
            # Cleanup
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
