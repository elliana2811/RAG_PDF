from pydantic import BaseModel
from typing import Optional, Dict
from datetime import datetime

class DocumentRegisterRequest(BaseModel):
    notebook_id: str
    workspace_id: str
    doc_id: str
    raw_uri: str
    original_name: str
    mime_type: str

class PresignedUrlRequest(BaseModel):
    notebook_id: str
    workspace_id: str
    filename: str
    mime_type: str

class PresignedUrlResponse(BaseModel):
    upload_url: str
    doc_id: str
    raw_uri: str
