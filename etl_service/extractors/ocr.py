import base64
import httpx
import logging
import io
from PIL import Image
from typing import List, Dict
from shared.utils.config import settings

logger = logging.getLogger(__name__)


class OCRProcessor:
    """OCR processor using Ollama glm-ocr vision model."""

    @staticmethod
    def process_pdf(file_content: bytes) -> List[Dict]:
        """
        Extracts text from a scanned PDF using Ollama's glm-ocr vision model.
        Converts PDF pages to images, then sends each to the vision model for OCR.
        """
        if not settings.OCR_ENABLED:
            logger.warning("OCR is disabled.")
            return []

        try:
            from pdf2image import convert_from_bytes
        except ImportError:
            logger.error("pdf2image is required for OCR. Install with: pip install pdf2image")
            return []

        try:
            images = convert_from_bytes(file_content, dpi=200)
        except Exception as e:
            logger.error(f"Failed to convert PDF to images: {e}")
            return []

        pages = []
        for i, img in enumerate(images):
            page_num = i + 1
            logger.info(f"Running OCR on page {page_num}/{len(images)}...")

            try:
                text = OCRProcessor._ocr_image(img)
                pages.append({"page": page_num, "text": text})
            except Exception as e:
                logger.error(f"OCR failed for page {page_num}: {e}")
                pages.append({"page": page_num, "text": ""})

        return pages

    @staticmethod
    def process_image(file_content: bytes) -> List[Dict]:
        """
        Extracts text from an image using Zai OCR API.
        """
        if not settings.OCR_ENABLED:
            logger.warning("OCR is disabled.")
            return []

        try:
            img = Image.open(io.BytesIO(file_content))
            text = OCRProcessor._ocr_image(img)
            return [{"page": 1, "text": text}]
        except Exception as e:
            logger.error(f"Failed to process image: {e}")
            return []

    @staticmethod
    def _ocr_image(image) -> str:
        """Send an image to Mistral OCR API for text extraction.
        Uses configuration from settings: MISTRAL_API_KEY, MISTRAL_BASE_URL, MISTRAL_MODEL_NAME.
        """
        # Convert image to base64
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        img_b64 = f"data:image/png;base64,{img_b64}"

        # Prepare request payload for Mistral
        payload = {
            "model": getattr(settings, "MISTRAL_MODEL_NAME", "mistral-ocr-latest"),
            "document": {
                "type": "image_url",
                "image_url": img_b64
            }
        }

        headers = {
            "Authorization": f"Bearer {getattr(settings, 'MISTRAL_API_KEY', '')}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        # Mistral OCR endpoint
        endpoint = getattr(settings, "MISTRAL_BASE_URL", "https://api.mistral.ai")
        # Ensure no trailing slash
        endpoint = endpoint.rstrip('/')
        url = f"{endpoint}/v1/ocr"

        with httpx.Client(timeout=120.0) as client:
            response = client.post(url, json=payload, headers=headers)
            if response.status_code != 200:
                logger.error(f"OCR API Error: {response.status_code} - {response.text}")
            response.raise_for_status()
            data = response.json()
            logger.info(f"Mistral OCR Response: {data}")

        # The Mistral API typically returns markdown in the pages -> markdown array
        result = ""
        if "pages" in data and isinstance(data["pages"], list):
            for page in data["pages"]:
                if "markdown" in page:
                    result += page["markdown"] + "\n\n"
        
        # Fallback if the format is slightly different 
        if not result and "text" in data:
             result = data["text"]

        return result.strip()

