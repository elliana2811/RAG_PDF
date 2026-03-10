from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi import Request
from shared.utils.config import settings
from shared.db.minio import MinIOClient
from shared.db.mongo import MongoDB
from shared.db.redis import RedisClient
from etl_service.models import PresignedUrlRequest, PresignedUrlResponse, DocumentRegisterRequest
from datetime import datetime, timedelta
import ulid
import json
from shared.utils.auth import verify_api_key
from shared.db.qdrant import QdrantDB
from fastapi import Depends, Query
from typing import List, Optional
import boto3
from botocore.client import Config
from qdrant_client import models # Import models for Qdrant filtering

app = FastAPI(
    title="FiinLM ETL Service",
    dependencies=[Depends(verify_api_key)]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4117", "http://localhost:3001", "http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import logging

logger = logging.getLogger("uvicorn.error")

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Validation Error: {exc.errors()}")
    body = await request.body()
    logger.error(f"Request Body: {body.decode()}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": body.decode()},
    )


@app.on_event("startup")
async def startup_db_client():
    await MongoDB.connect()
    MinIOClient.ensure_buckets()
    await RedisClient.get_client()

@app.on_event("shutdown")
async def shutdown_db_client():
    await MongoDB.close()
    await RedisClient.close()

@app.post("/v1/uploads/presign", response_model=PresignedUrlResponse)
async def generate_presigned_url(request: PresignedUrlRequest):
    doc_id = str(ulid.ULID())
    
    # Construct object name: raw/workspace_id/notebook_id/doc_id.ext
    ext = request.filename.split('.')[-1] if '.' in request.filename else "bin"
    object_name = f"{settings.MINIO_BUCKET_RAW}/{request.workspace_id}/{request.notebook_id}/{doc_id}.{ext}"
    
    try:
        # Use boto3 for offline presigning.
        # This avoids the issue where minio-py tries to connect to 'localhost' inside the container.
        # We explicitly set the endpoint to the public endpoint (localhost:9000) so the signature matches the browser's Host header.
        s3_client = boto3.client(
            's3',
            endpoint_url=f"http://{settings.MINIO_PUBLIC_ENDPOINT}",
            aws_access_key_id=settings.MINIO_ACCESS_KEY,
            aws_secret_access_key=settings.MINIO_SECRET_KEY,
            config=Config(signature_version='s3v4'),
            region_name='us-east-1',
            use_ssl=False,
            verify=False
        )
        
        url = s3_client.generate_presigned_url(
            'put_object',
            Params={
                'Bucket': settings.MINIO_BUCKET_RAW,
                'Key': object_name,
                'ContentType': request.mime_type
            },
            ExpiresIn=900
        )
        print(f"DEBUG: Boto3 Generated URL: {url}")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"Error generating presigned URL: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    return PresignedUrlResponse(
        upload_url=url,
        doc_id=doc_id,
        raw_uri=f"minio://{settings.MINIO_BUCKET_RAW}/{object_name}"
    )

@app.post("/v1/documents/register")
async def register_document(request: DocumentRegisterRequest):
    db = MongoDB.get_db()
    redis_client = await RedisClient.get_client()
    
    # Check if document already exists (idempotency check by doc_id)
    existing_doc = await db.documents.find_one({"doc_id": request.doc_id})
    if existing_doc:
        return {"message": "Document already registered", "doc_id": request.doc_id}

    document = {
        "doc_id": request.doc_id,
        "notebook_id": request.notebook_id,
        "workspace_id": request.workspace_id,
        "source": {
            "type": "upload",
            "original_name": request.original_name
        },
        "storage": {
            "raw_uri": request.raw_uri,
            "processed_uri": None,
            "ocr_pages_prefix": None
        },
        "mime_type": request.mime_type,
        "status": "queued",
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
        "stats": {
            "pages": 0,
            "chunks": 0
        }
    }
    
    await db.documents.insert_one(document)
    
    # Enqueue job to Redis
    job = {
        "task": "process_upload",
        "job_id": str(ulid.ULID()),
        "doc_id": request.doc_id,
        "workspace_id": request.workspace_id,
        "notebook_id": request.notebook_id,
        "raw_uri": request.raw_uri,
        "attempt": 1
    }
    
    await redis_client.lpush("ingestion:queue", json.dumps(job))
    
    return {"message": "Document registered and queued", "job_id": job["job_id"]}

@app.get("/v1/documents")
async def list_documents(
    workspace_id: str = Query(...),
    notebook_id: str = Query(...)
):
    db = MongoDB.get_db()
    cursor = db.documents.find(
        {"workspace_id": workspace_id, "notebook_id": notebook_id},
        {"_id": 0} # Exclude internal ID
    ).sort("created_at", -1)
    
    documents = await cursor.to_list(length=100)
    return documents

@app.get("/v1/documents/{doc_id}/chunks")
async def get_document_chunks(doc_id: str):
    db = MongoDB.get_db()
    doc = await db.documents.find_one({"doc_id": doc_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    cursor = db.chunks.find(
        {"doc_id": doc_id},
        {"_id": 0}
    ).sort("meta.page", 1)
    chunks = await cursor.to_list(length=500)

    return {
        "doc_id": doc_id,
        "filename": doc.get("source", {}).get("original_name", ""),
        "status": doc.get("status", "unknown"),
        "total_chunks": len(chunks),
        "chunks": chunks,
    }

@app.get("/v1/documents/{doc_id}/download-url")
async def download_document_url(doc_id: str):
    db = MongoDB.get_db()
    doc = await db.documents.find_one({"doc_id": doc_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
        
    raw_uri = doc.get("storage", {}).get("raw_uri", "")
    original_name = doc.get("source", {}).get("original_name", f"{doc_id}.bin")
    
    if not raw_uri or not raw_uri.startswith("minio://"):
        raise HTTPException(status_code=400, detail="Invalid storage URI")
        
    try:
        path_part = raw_uri.replace("minio://", "")
        bucket, key = path_part.split("/", 1)
    except ValueError:
        raise HTTPException(status_code=500, detail="Malformed storage URI")

    try:
        s3_client = boto3.client(
            's3',
            endpoint_url=f"http://{settings.MINIO_PUBLIC_ENDPOINT}",
            aws_access_key_id=settings.MINIO_ACCESS_KEY,
            aws_secret_access_key=settings.MINIO_SECRET_KEY,
            config=Config(signature_version='s3v4'),
            region_name='us-east-1',
            use_ssl=False,
            verify=False
        )
        
        url = s3_client.generate_presigned_url(
            'get_object',
            Params={
                'Bucket': bucket,
                'Key': key,
                'ResponseContentDisposition': f'attachment; filename="{original_name}"'
            },
            ExpiresIn=300
        )
        
    except Exception as e:
        logger.error(f"Error generating download URL: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        
    return {"download_url": url, "filename": original_name}

@app.get("/v1/documents/{doc_id}/preview-url")
async def preview_document_url(doc_id: str):
    db = MongoDB.get_db()
    doc = await db.documents.find_one({"doc_id": doc_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
        
    raw_uri = doc.get("storage", {}).get("raw_uri", "")
    mime_type = doc.get("mime_type", "application/octet-stream")
    
    if not raw_uri or not raw_uri.startswith("minio://"):
        raise HTTPException(status_code=400, detail="Invalid storage URI")
        
    try:
        path_part = raw_uri.replace("minio://", "")
        bucket, key = path_part.split("/", 1)
    except ValueError:
        raise HTTPException(status_code=500, detail="Malformed storage URI")

    try:
        s3_client = boto3.client(
            's3',
            endpoint_url=f"http://{settings.MINIO_PUBLIC_ENDPOINT}",
            aws_access_key_id=settings.MINIO_ACCESS_KEY,
            aws_secret_access_key=settings.MINIO_SECRET_KEY,
            config=Config(signature_version='s3v4'),
            region_name='us-east-1',
            use_ssl=False,
            verify=False
        )
        
        # NOTE: Content-Disposition 'inline' allows browser preview
        url = s3_client.generate_presigned_url(
            'get_object',
            Params={
                'Bucket': bucket,
                'Key': key,
                'ResponseContentType': mime_type,
                'ResponseContentDisposition': 'inline'
            },
            ExpiresIn=300
        )
        
    except Exception as e:
        logger.error(f"Error generating preview URL: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        
    return {
        "preview_url": url, 
        "mime_type": mime_type, 
        "filename": doc.get("source", {}).get("original_name", "")
    }

@app.delete("/v1/documents/{doc_id}")
async def delete_document(doc_id: str):
    db = MongoDB.get_db()
    minio_client = MinIOClient.get_client()
    qdrant_client = QdrantDB.get_async_client()
    
    # 1. Fetch document metadata
    doc = await db.documents.find_one({"doc_id": doc_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # 2. Delete from Qdrant
    try:
        logger.info(f"Attempting to delete doc_id {doc_id} from Qdrant collection {settings.QDRANT_COLLECTION}")
        
        # Verify count before delete
        count_before = await qdrant_client.count(
            collection_name=settings.QDRANT_COLLECTION,
            count_filter=models.Filter(
                must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
            )
        )
        logger.info(f"Qdrant points found for {doc_id} before delete: {count_before.count}")

        await qdrant_client.delete(
            collection_name=settings.QDRANT_COLLECTION,
            points_selector=models.Filter(
                must=[
                    models.FieldCondition(
                        key="doc_id",
                        match=models.MatchValue(value=doc_id)
                    )
                ]
            ),
            wait=True # Force wait for operation to persist
        )
        
        # Verify count after delete
        count_after = await qdrant_client.count(
            collection_name=settings.QDRANT_COLLECTION,
            count_filter=models.Filter(
                must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
            )
        )
        logger.info(f"Qdrant points found for {doc_id} after delete: {count_after.count}")
        
        if count_after.count > 0:
            logger.error(f"FAILED to delete key {doc_id} from Qdrant!")
        else:
            logger.info(f"Successfully deleted {doc_id} from Qdrant")
            
    except Exception as e:
        logger.error(f"Error deleting from Qdrant: {e}")
        import traceback
        traceback.print_exc()
    
    # 3. Delete from MinIO
    from urllib.parse import unquote
    try:
        # Extract object name from URI
        # minio://bucket/raw/ws_id/nb_id/doc_id.ext
        raw_uri = doc.get("storage", {}).get("raw_uri")
        if raw_uri and raw_uri.startswith("minio://"):
            _, path = raw_uri.replace("minio://", "").split("/", 1)
            path = unquote(path)  # Just in case URL encoded
            bucket = settings.MINIO_BUCKET_RAW
            minio_client.remove_object(bucket, path)
            print(f"Deleted from MinIO (Raw): {bucket}/{path}")
            
        # Also cleanup processed if exists
        processed_uri = doc.get("storage", {}).get("processed_uri")
        if processed_uri and processed_uri.startswith("minio://"):
            _, path = processed_uri.replace("minio://", "").split("/", 1)
            path = unquote(path)
            bucket = settings.MINIO_BUCKET_PROCESSED
            minio_client.remove_object(bucket, path)
            print(f"Deleted from MinIO (Processed): {bucket}/{path}")
    except Exception as e:
        print(f"Error deleting from MinIO: {e}")
    
    # 4. Delete from MongoDB (Chunks + Document)
    await db.chunks.delete_many({"doc_id": doc_id})
    await db.documents.delete_one({"doc_id": doc_id})
    
    return {"message": "Document deleted successfully"}

from qdrant_client import models # Import models for Qdrant filtering
