import logging
import mimetypes
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.document_processor import SUPPORTED_EXTENSIONS, chunk_document, extract_document
from src.rag_pipeline import RAGPipeline
from src.retriever import get_embedding_model
from src.vector_store import (
    delete_document_chunks,
    get_vector_store,
    store_chunks,
)

logger = logging.getLogger("docuchat_api")
logging.basicConfig(level=logging.INFO)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = PROJECT_ROOT / "src" / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

try:
    UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
except (OSError, PermissionError):
    UPLOAD_DIR = Path(tempfile.gettempdir()) / "docuchat_uploads"
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB

# Registry to track metadata of uploaded files (size, timestamp, type, chunks)
DOCUMENT_REGISTRY: dict[str, dict[str, Any]] = {}

app = FastAPI(
    title="DocuChat API",
    description="Enterprise Multi-Document RAG System with Citations & Context Isolation",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str = Field(..., description="User query against indexed documents")
    backend: str = Field(default="supabase", description="Vector store backend: 'supabase' or 'faiss'")
    match_threshold: float = Field(default=0.20, description="Minimum cosine similarity threshold")
    match_count: int = Field(default=3, description="Maximum number of chunks to retrieve")
    document_ids: list[str] | None = Field(
        default=None,
        description="Optional list of active document IDs / filenames to isolate search context strictly to",
    )
    session_id: str | None = Field(default=None, description="Optional conversation / session ID")


class ChatResponse(BaseModel):
    question: str
    answer: str
    backend: str
    validation: dict[str, Any]
    retrieved_chunks: list[dict[str, Any]]
    document_ids: list[str] | None = None


@app.get("/api/status")
def get_system_status():
    default_backend = os.getenv("VECTOR_STORE_BACKEND", "supabase")
    supabase_configured = bool(os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_SECRET_KEY"))
    openai_configured = bool(os.getenv("OPENAI_API_KEY"))

    return {
        "status": "online",
        "default_backend": default_backend,
        "supabase_configured": supabase_configured,
        "openai_configured": openai_configured,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "supported_formats": sorted(list(SUPPORTED_EXTENSIONS)),
        "max_upload_size_mb": 25,
    }


@app.get("/api/documents")
def list_documents(
    backend: str = "supabase",
    session_id: str | None = None,
):
    """
    List all indexed documents available for chatting, including file size,
    chunk counts, and preview capabilities.
    """
    store = get_vector_store(backend)
    store_docs = store.list_documents(session_id=session_id)

    # Merge vector store records with in-memory upload registry
    docs_map: dict[str, dict[str, Any]] = {}
    for item in store_docs:
        doc_id = item["document_id"]
        docs_map[doc_id] = {
            "document_id": doc_id,
            "filename": item.get("filename", doc_id),
            "chunk_count": item.get("chunk_count", 0),
            "pages_count": item.get("pages_count", 1),
            "file_size": 0,
            "file_type": Path(item.get("filename", doc_id)).suffix.lower(),
            "uploaded_at": None,
            "has_preview": False,
        }

    for doc_id, meta in DOCUMENT_REGISTRY.items():
        if doc_id in docs_map:
            docs_map[doc_id].update({
                "file_size": meta.get("file_size", 0),
                "file_type": meta.get("file_type", ""),
                "uploaded_at": meta.get("uploaded_at"),
                "has_preview": True,
            })
        else:
            docs_map[doc_id] = meta

    # Check for sample policy if present in data folder
    sample_pdf = PROJECT_ROOT / "data" / "sample_company_policy.pdf"
    if sample_pdf.exists() and "sample_company_policy.pdf" not in docs_map:
        docs_map["sample_company_policy.pdf"] = {
            "document_id": "sample_company_policy.pdf",
            "filename": "sample_company_policy.pdf",
            "chunk_count": 3,
            "pages_count": 1,
            "file_size": sample_pdf.stat().st_size,
            "file_type": ".pdf",
            "uploaded_at": None,
            "has_preview": True,
        }

    return {"documents": list(docs_map.values())}


@app.post(
    "/api/upload",
    openapi_extra={
        "requestBody": {
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "file": {
                                "type": "string",
                                "format": "binary",
                                "description": "Single document file to upload (PDF, DOCX, PNG, JPG, JPEG)",
                            },
                            "files": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "format": "binary",
                                },
                                "description": "Multiple document files to upload",
                            },
                        },
                    }
                }
            }
        }
    },
)
async def upload_documents(
    file: UploadFile | str | None = File(default=None),
    files: list[UploadFile | str] | str | None = File(default=None),
    backend: str = "supabase",
    session_id: str | None = None,
):
    """
    Universal Document Ingestion endpoint supporting single and multiple file uploads.
    Supports PDF, DOCX, PNG, JPG, and JPEG documents.
    """
    upload_list: list[UploadFile] = []
    if hasattr(file, "filename") and file.filename:
        upload_list.append(file)
    if files:
        if isinstance(files, list):
            for f in files:
                if hasattr(f, "filename") and f.filename:
                    upload_list.append(f)
        elif hasattr(files, "filename") and files.filename:
            upload_list.append(files)

    if not upload_list:
        raise HTTPException(
            status_code=400,
            detail="No files selected. Please select one or more documents to upload.",
        )

    processed_results = []
    total_chunks_stored = 0

    for upload_file in upload_list:
        filename = upload_file.filename
        suffix = Path(filename).suffix.lower()

        if suffix not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format '{suffix}'. Please upload a .pdf, .docx, .png, .jpg, or .jpeg file.",
            )

        saved_path = UPLOAD_DIR / filename
        file_bytes = await upload_file.read()
        file_size = len(file_bytes)

        if file_size > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=400,
                detail=f"File '{filename}' ({file_size // (1024*1024)}MB) exceeds maximum limit of 25MB.",
            )

        # Write to disk
        with open(saved_path, "wb") as buffer:
            buffer.write(file_bytes)

        try:
            # 1. Parse document / OCR
            records = extract_document(saved_path)

            # 2. Chunk document
            chunks = chunk_document(records, chunk_size=200, overlap=40)

            # 3. Generate embeddings
            model = get_embedding_model()
            embeddings = model.encode(
                [c["text"] for c in chunks],
                normalize_embeddings=True,
            ).tolist()

            # 4. Store in vector database
            stored_count = store_chunks(
                chunks=chunks,
                embeddings=embeddings,
                document_id=filename,
                backend=backend,
                session_id=session_id,
            )
            total_chunks_stored += stored_count

            # 5. Record metadata
            doc_meta = {
                "document_id": filename,
                "filename": filename,
                "file_type": suffix,
                "file_size": file_size,
                "chunk_count": len(chunks),
                "pages_count": len(set(r.get("page") for r in records if r.get("page"))),
                "uploaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "has_preview": True,
            }
            DOCUMENT_REGISTRY[filename] = doc_meta

            processed_results.append({
                "filename": filename,
                "records_extracted": len(records),
                "chunks_stored": len(chunks),
                "file_size": file_size,
                "file_type": suffix,
            })

        except Exception as e:
            logger.error("Processing failed for %s: %s", filename, e)
            raise HTTPException(status_code=500, detail=f"Failed processing '{filename}': {str(e)}")

    return {
        "success": True,
        "filename": upload_list[0].filename if len(upload_list) == 1 else f"{len(upload_list)} files",
        "files_uploaded": len(upload_list),
        "results": processed_results,
        "records_extracted": sum(r["records_extracted"] for r in processed_results),
        "chunks_stored": total_chunks_stored,
        "backend": backend,
        "message": f"Successfully indexed {len(upload_list)} document(s) ({total_chunks_stored} chunks stored into {backend.upper()}).",
    }


@app.delete("/api/documents/{document_id}")
def delete_document(
    document_id: str,
    backend: str = "supabase",
    session_id: str | None = None,
):
    """
    Delete a document and completely remove its chunks from the vector database
    so it can never be retrieved or contaminate subsequent answers.
    """
    deleted_chunks = delete_document_chunks(
        document_id=document_id,
        backend=backend,
        session_id=session_id,
    )

    DOCUMENT_REGISTRY.pop(document_id, None)

    # Clean local file if present
    file_path = UPLOAD_DIR / document_id
    try:
        if file_path.exists():
            file_path.unlink()
    except OSError:
        pass

    return {
        "success": True,
        "document_id": document_id,
        "deleted_chunks": deleted_chunks,
        "message": f"Document '{document_id}' and all associated searchable vectors were removed.",
    }


@app.get("/api/documents/{document_id}/preview")
def preview_document(document_id: str):
    """
    Serve uploaded documents for in-browser visual previews (PDFs, Images).
    """
    # Check UPLOAD_DIR first, then data directory
    candidate_paths = [
        UPLOAD_DIR / document_id,
        PROJECT_ROOT / "data" / document_id,
        PROJECT_ROOT / "data" / "uploads" / document_id,
    ]

    target_file = None
    for p in candidate_paths:
        if p.exists() and p.is_file():
            target_file = p
            break

    if not target_file:
        raise HTTPException(
            status_code=404,
            detail=f"Preview file '{document_id}' not found.",
        )

    mime_type, _ = mimetypes.guess_type(str(target_file))
    if not mime_type:
        suffix = target_file.suffix.lower()
        if suffix == ".pdf":
            mime_type = "application/pdf"
        elif suffix in {".png", ".jpg", ".jpeg"}:
            mime_type = f"image/{suffix.lstrip('.')}"
        else:
            mime_type = "application/octet-stream"

    return FileResponse(
        str(target_file),
        media_type=mime_type,
        filename=target_file.name,
    )


@app.post("/api/chat", response_model=ChatResponse)
def chat_with_documents(request: ChatRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    pipeline = RAGPipeline(vector_backend=request.backend)
    result = pipeline.answer_question(
        question=request.question,
        match_count=request.match_count,
        match_threshold=request.match_threshold,
        backend=request.backend,
        document_ids=request.document_ids,
        session_id=request.session_id,
    )

    return ChatResponse(
        question=result["question"],
        answer=result["answer"],
        backend=result["backend"],
        validation=result["validation"],
        retrieved_chunks=result["retrieved_chunks"],
        document_ids=result.get("document_ids"),
    )


# Serve static web interface
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def serve_home():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"message": "DocuChat API is running. Access /docs for Swagger API documentation."}
