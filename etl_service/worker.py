import asyncio
import json
import logging
import ulid
from shared.utils.config import settings
from shared.db.redis import RedisClient
from shared.db.mongo import MongoDB
from shared.db.minio import MinIOClient
from shared.db.qdrant import QdrantDB
from etl_service.extractors.pdf import PDFExtractor
from etl_service.extractors.ocr import OCRProcessor
from etl_service.extractors.markitdown_adapter import MarkItDownAdapter
from etl_service.extractors.vision import VisionProcessor
import os
import pandas as pd
import io
from etl_service.chunking import Chunker
from etl_service.embeddings import Embedder, SparseEmbedder
from qdrant_client.models import PointStruct, VectorParams, Distance
from qdrant_client import models

logger = logging.getLogger("etl_worker")

class IngestionWorker:
    def __init__(self):
        self.redis = None
        self.mongo = None
        self.minio = None
        self.qdrant = None
        self.chunker = Chunker()
        self.embedder = Embedder()
        self.sparse_embedder = SparseEmbedder()

    async def setup(self):
        await MongoDB.connect()
        self.mongo = MongoDB.get_db()
        self.redis = await RedisClient.get_client()
        self.minio = MinIOClient.get_client()
        self.qdrant = QdrantDB.get_async_client()
        
        # Ensure Qdrant collection exists
        try:
            await self.qdrant.get_collection(settings.QDRANT_COLLECTION)
        except:
            await self.qdrant.create_collection(
                collection_name=settings.QDRANT_COLLECTION,
                vectors_config={
                    "dense": VectorParams(size=settings.EMBEDDING_DIMENSION, distance=Distance.COSINE)
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(
                        index=models.SparseIndexParams(
                            on_disk=False,
                        ),
                        modifier=models.Modifier.IDF, # BM25 requires IDF modifier
                    )
                }
            )

    async def process_job(self, job_data: dict):
        doc_id = job_data["doc_id"]
        logger.info(f"Processing job for doc_id: {doc_id}")

        try:
            # 1. Update status to processing
            await self.mongo.documents.update_one(
                {"doc_id": doc_id},
                {"$set": {"status": "processing"}}
            )

            # 2. Download file
            bucket, object_name = job_data["raw_uri"].replace("minio://", "").split("/", 1)
            response = self.minio.get_object(bucket, object_name)
            file_content = response.read()
            response.close()
            response.release_conn()
            # 3. Determine File Type and Extract Text
            _, ext = os.path.splitext(object_name)
            ext = ext.lower()
            
            # Use Zai OCR for PDF and Images per user request
            ocr_formats = [".pdf", ".jpg", ".jpeg", ".png"]
            # Formats supported by MarkItDown (Office/Text/Web/etc) - REMOVED
            markitdown_formats = []
            
            pages = []
            
            if ext in ocr_formats:
                if ext == ".pdf":
                    # Check if PDF is scanned or has a text layer
                    logger.info(f"Checking if PDF is scanned or text-based: {doc_id}")
                    temp_extracted = PDFExtractor.extract_text(file_content)
                    
                    if PDFExtractor.is_scanned(temp_extracted):
                        logger.info(f"PDF detected as SCANNED. Using Mistral OCR: {doc_id}")
                        pages = OCRProcessor.process_pdf(file_content)
                    else:
                        logger.info(f"PDF detected as TEXT-BASED. Using MarkItDown: {doc_id}")
                        try:
                            text_content = MarkItDownAdapter.extract_text(file_content, object_name)
                            if text_content and text_content.strip():
                                pages = [{"page": 1, "text": text_content}]
                        except Exception as e:
                            logger.warning(f"MarkItDown failed for text PDF {doc_id}, falling back to pypdf layers: {e}")
                            pages = temp_extracted
                else:
                    # Generic image processing
                    logger.info(f"Using Mistral OCR for image: {doc_id}")
                    pages = OCRProcessor.process_image(file_content)

                # Shared Vision Fallback Logic for Charts/Images via GPT-4o Vision
                if getattr(settings, "ENABLE_VISION_FALLBACK", False) and pages:
                    logger.info("Running GPT-4o Vision Fallback for extra visual context...")
                    try:
                        from pdf2image import convert_from_bytes
                        pdf_images = convert_from_bytes(file_content, dpi=200)
                        
                        for p in pages:
                            img_index = p["page"] - 1
                            if img_index < len(pdf_images):
                                logger.info(f"Sending page {p['page']} to GPT-4o Vision Fallback...")
                                img_buffer = io.BytesIO()
                                pdf_images[img_index].save(img_buffer, format="JPEG")
                                summary = VisionProcessor.summarize_image(img_buffer.getvalue())
                                if summary:
                                    p["text"] += f"\n\n### Chart/Image Summary (GPT-4o Vision Fallback):\n{summary}\n\n"
                    except Exception as e:
                        logger.error(f"Failed during Vision Fallback: {e}")
            elif ext in markitdown_formats:
                logger.info(f"Using MarkItDown for {ext} file: {doc_id}")
                try:
                    text_content = MarkItDownAdapter.extract_text(file_content, object_name)
                    if text_content and text_content.strip():
                        pages = [{"page": 1, "text": text_content}]
                except Exception as e:
                    logger.warning(f"MarkItDown failed for {doc_id}: {e}")
            
            if not pages:
                # Fallback for other formats or if specific extraction failed
                if ext == ".pdf":
                     # This should not normally be reached if OCR is enabled, 
                     # but keeping as safety or for when OCR_ENABLED=False
                     pages = PDFExtractor.extract_text(file_content)
                else:
                    try:
                        text = file_content.decode('utf-8', errors='ignore')
                        pages = [{"page": 1, "text": text}]
                    except:
                         raise Exception("No text extracted from document")
            
            if not pages:
                # Last resort: Try generic text decode if it's potentially a text file
                try:
                    text = file_content.decode('utf-8')
                    pages = [{"page": 1, "text": text}]
                except:
                     raise Exception("No text extracted from document")

            # 3.1 Structured data extraction (REMOVED)

            # 4. Chunk
            chunks = self.chunker.chunk_document(pages)
            logger.info(f"Generated {len(chunks)} chunks for {doc_id}")

            # 5. Embed & Index
            points = []
            
            chunk_texts = [c["text"] for c in chunks]
            dense_vectors = self.embedder.generate_embeddings(chunk_texts)
            sparse_vectors = self.sparse_embedder.generate(chunk_texts)

            for i, chunk in enumerate(chunks):
                chunk_id = str(ulid.ULID().to_uuid())
                
                point = PointStruct(
                    id=chunk_id,
                    vector={
                        "dense": dense_vectors[i],
                        "sparse": models.SparseVector(
                            indices=sparse_vectors[i]["indices"],
                            values=sparse_vectors[i]["values"]
                        )
                    },
                    payload={
                        "doc_id": doc_id,
                        "notebook_id": job_data["notebook_id"],
                        "workspace_id": job_data["workspace_id"],
                        "text": chunk["text"],
                        "page": chunk["meta"]["page"],
                        "created_at": str(job_data.get("created_at", ""))
                    }
                )
                points.append(point)

            # Batch upsert to Qdrant
            if points:
                await self.qdrant.upsert(
                    collection_name=settings.QDRANT_COLLECTION,
                    points=points
                )
            
            # 6. Save chunks to Mongo - DISABLED as per user request
            # "only structured_data should be search/added to mongodb, do not add chunks"
            # mongo_chunks = [
            #     {
            #         "chunk_id": p.id,
            #         "doc_id": doc_id,
            #         "notebook_id": job_data["notebook_id"],
            #         "text": p.payload["text"],
            #         "meta": {"page": p.payload["page"]}
            #     }
            #     for p in points
            # ]
            # if mongo_chunks:
            #     await self.mongo.chunks.insert_many(mongo_chunks)

            # 7. Update status to indexed
            await self.mongo.documents.update_one(
                {"doc_id": doc_id},
                {"$set": {
                    "status": "indexed",
                    "stats.pages": len(pages),
                    "stats.chunks": len(chunks)
                }}
            )
            logger.info(f"Successfully indexed {doc_id}")

        except Exception as e:
            logger.error(f"Failed to process {doc_id}: {str(e)}")

            attempt = job_data.get("attempt", 1)
            max_retries = settings.WORKER_MAX_RETRIES

            if attempt < max_retries:
                delay = settings.WORKER_RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logger.warning(f"Retrying {doc_id} (attempt {attempt + 1}/{max_retries}) after {delay}s...")

                await self.mongo.documents.update_one(
                    {"doc_id": doc_id},
                    {"$set": {
                        "status": "queued",
                        "failure": {"message": str(e), "attempt": attempt}
                    }}
                )

                await asyncio.sleep(delay)
                retry_job = {**job_data, "attempt": attempt + 1}
                await self.redis.lpush("ingestion:queue", json.dumps(retry_job))
            else:
                logger.error(f"Max retries reached for {doc_id}. Marking as failed.")
                await self.mongo.documents.update_one(
                    {"doc_id": doc_id},
                    {"$set": {
                        "status": "failed",
                        "failure": {"message": str(e), "attempts": attempt}
                    }}
                )

    async def run(self):
        await self.setup()
        logger.info("Worker started. Waiting for jobs...")
        
        while True:
            # Blocking pop from Redis
            result = await self.redis.brpop("ingestion:queue", timeout=5)
            if result:
                _, job_json = result
                job_data = json.loads(job_json)
                await self.process_job(job_data)
            await asyncio.sleep(0.1)

if __name__ == "__main__":
    worker = IngestionWorker()
    asyncio.run(worker.run())
