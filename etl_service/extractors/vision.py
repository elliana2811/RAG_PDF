import base64
import logging
import io
from PIL import Image
from openai import OpenAI
from shared.utils.config import settings

logger = logging.getLogger(__name__)

class VisionProcessor:
    """Uses OpenAI GPT-4o Vision API to describe charts and tables from images."""

    @staticmethod
    def get_client():
        return OpenAI(api_key=settings.OPENAI_API_KEY)

    @staticmethod
    def summarize_image(image_bytes: bytes) -> str:
        """
        Sends image bytes to GPT-4o vision model to get a detailed textual summary.
        """
        if not settings.ENABLE_VISION_FALLBACK or not settings.OPENAI_API_KEY:
            logger.warning("Vision fallback is disabled or missing OPENAI_API_KEY.")
            return ""

        try:
            # Re-encode just to be safe and ensure format is correct
            img = Image.open(io.BytesIO(image_bytes))
            buffer = io.BytesIO()
            # Convert to RGB to avoid alpha channel issues with JPEG or PNG compatibility
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img.save(buffer, format="JPEG")
            img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            
            client = VisionProcessor.get_client()
            
            prompt = (
                "Bạn là một chuyên gia phân tích dữ liệu cao cấp. Hãy phân tích hình ảnh này, vốn là một trang từ tài liệu PDF có thể chứa biểu đồ, đồ thị, hình ảnh minh họa hoặc bảng biểu. "
                "Nhiệm vụ của bạn là cung cấp một bản tóm tắt chi tiết bằng văn bản. "
                "1. Nếu có biểu đồ: Mô tả loại biểu đồ, các trục, các đại lượng, xu hướng chính, các điểm dữ liệu quan trọng và kết luận rút ra. "
                "2. Nếu có bảng biểu: Mô tả cấu trúc bảng, các cột/hàng chính và tóm tắt các con số quan trọng (không cần vẽ lại bảng bằng Markdown). "
                "3. Nếu có hình ảnh minh họa: Mô tả nội dung hình ảnh đóng góp gì cho ngữ cảnh tài liệu. "
                "Mục tiêu là giúp một hệ thống RAG có thể hiểu và trả lời câu hỏi về nội dung trực quan này mà không cần nhìn thấy hình ảnh gốc. "
                "YÊU CẦU BẮT BUỘC: Viết toàn bộ câu trả lời bằng TIẾNG VIỆT (Tiếng Việt)."
            )

            response = client.chat.completions.create(
                model=settings.VISION_MODEL_NAME,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_b64}",
                                    "detail": "high"
                                },
                            },
                        ],
                    }
                ],
                max_tokens=1000,
            )
            
            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Failed to summarize image with Vision model: {e}")
            return ""
