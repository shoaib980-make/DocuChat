import logging
import os
import shutil
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.document_processor import chunk_document, extract_document
from src.rag_pipeline import RAGPipeline
from src.retriever import get_embedding_model
from src.vector_store import get_vector_store, store_chunks

logger = logging.getLogger("docuchat_api")
logging.basicConfig(level=logging.INFO)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = PROJECT_ROOT / "src" / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

try:
    UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
except (OSError, PermissionError):
    import tempfile
    UPLOAD_DIR = Path(tempfile.gettempdir()) / "docuchat_uploads"
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="DocuChat API",
    description="Enterprise RAG System with Citations & Hallucination Control",
    version="1.0.0",
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


class ChatResponse(BaseModel):
    question: str
    answer: str
    backend: str
    validation: dict[str, Any]
    retrieved_chunks: list[dict[str, Any]]


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
        "supported_formats": [".pdf", ".docx"],
    }


@app.post("/api/upload")
async def upload_document(
    file: UploadFile = File(...),
    backend: str = "supabase",
):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".pdf", ".docx"}:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{suffix}'. Please upload a .pdf or .docx file.",
        )

    temp_path = UPLOAD_DIR / file.filename
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 1. Parse document
        records = extract_document(temp_path)

        # 2. Chunk document
        chunks = chunk_document(records, chunk_size=200, overlap=40)

        # 3. Generate embeddings
        model = get_embedding_model()
        embeddings = model.encode(
            [c["text"] for c in chunks],
            normalize_embeddings=True,
        ).tolist()

        # 4. Store in vector database
        store_chunks(
            chunks=chunks,
            embeddings=embeddings,
            document_id=file.filename,
            backend=backend,
        )

        return {
            "success": True,
            "filename": file.filename,
            "records_extracted": len(records),
            "chunks_stored": len(chunks),
            "backend": backend,
            "message": f"Successfully indexed '{file.filename}' into {backend.upper()}.",
        }

    except Exception as e:
        logger.error("Upload error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            if temp_path.exists():
                temp_path.unlink()
        except OSError:
            pass


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
    )

    return ChatResponse(
        question=result["question"],
        answer=result["answer"],
        backend=result["backend"],
        validation=result["validation"],
        retrieved_chunks=result["retrieved_chunks"],
    )


# Serve static web interface
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def serve_home():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"message": "DocuChat API is running. Access /docs for Swagger API documentation."}
