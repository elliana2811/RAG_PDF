# FiinGroup RAG - ETL Service

Hệ thống ETL (Extract, Transform, Load) chuyên dụng để xử lý tài liệu PDF và hình ảnh cho đường ống RAG (Retrieval-Augmented Generation).

## Tính năng chính

- **Xử lý PDF & Image**: Hỗ trợ cả PDF văn bản (Text-based) và PDF quét (Scanned).
- **OCR Mạnh mẽ**: Sử dụng **Mistral OCR** (`mistral-ocr-latest`) để trích xuất nội dung từ hình ảnh và trang quét.
- **Xử lý bất đồng bộ**: Sử dụng **Redis Queue** để xử lý các tác vụ nặng ngầm, đảm bảo API luôn phản hồi nhanh.
- **Tìm kiếm Vector**: Tích hợp **Qdrant** để lưu trữ dense và sparse embeddings (Splade).
- **Quản lý Metadata**: Sử dụng **MongoDB** để lưu trữ thông tin tài liệu và theo dõi trạng thái xử lý.

## Kiến trúc hệ thống

1.  **API Service (`main.py`)**: Tiếp nhận yêu cầu tải lên, đăng ký tài liệu và trả về kết quả.
2.  **Worker Service (`worker.py`)**: Chạy ngầm, lấy nhiệm vụ từ Redis để thực hiện OCR, Chunking và Indexing.
3.  **Core Components**:
    *   `extractors/`: Chứa các module trích xuất văn bản (OCR, PDF, Vision).
    *   `chunking.py`: Chia nhỏ văn bản thành các đoạn (chunks) phù hợp cho LLM.
    *   `embeddings.py`: Tạo vector representation cho văn bản.

## Hướng dẫn cài đặt

### 1. Cấu hình môi trường
Sao chép file `.env.example` thành `.env` và điền các thông tin cần thiết:
```bash
cp .env.example .env
```

### 2. Cài đặt Python Dependencies
Khuyến nghị sử dụng môi trường ảo:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Yêu cầu hạ tầng
Hệ thống cần các dịch vụ sau đang chạy:
- **MongoDB**: Lưu trữ metadata.
- **Redis**: Hàng đợi công việc.
- **MinIO**: Lưu trữ file vật lý.
- **Qdrant**: Vector Database.

## Sử dụng

### Khởi chạy API
```bash
uvicorn etl_service.main:app --host 0.0.0.0 --port 8000
```

### Khởi chạy Worker
```bash
python -m etl_service.worker
```

## Ghi chú quan trọng
- Hệ thống hiện tại **chỉ tập trung xử lý file PDF và hình ảnh**.
- Các bảng biểu trong PDF sẽ được trích xuất dưới dạng Markdown và lưu vào Vector DB để phục vụ tìm kiếm ngữ nghĩa.
