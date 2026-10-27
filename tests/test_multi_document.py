import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.api import app
from src.rag_pipeline import REFUSAL_MESSAGE, RAGPipeline
from src.vector_store import FaissVectorStore

client = TestClient(app)


def test_universal_upload_images_and_multi_file(tmp_path):
    # 1. Create a dummy PNG image
    img = Image.new("RGB", (200, 100), color=(73, 109, 137))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")
    img_bytes = img_byte_arr.getvalue()

    # 2. Create a dummy JPEG image
    jpg_img = Image.new("RGB", (150, 150), color=(255, 255, 255))
    jpg_byte_arr = io.BytesIO()
    jpg_img.save(jpg_byte_arr, format="JPEG")
    jpg_bytes = jpg_byte_arr.getvalue()

    # Multi-file upload
    files = [
        ("files", ("receipt_test.png", img_bytes, "image/png")),
        ("files", ("invoice_test.jpg", jpg_bytes, "image/jpeg")),
    ]

    response = client.post("/api/upload?backend=faiss", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["files_uploaded"] == 2
    assert data["chunks_stored"] >= 2


def test_document_context_isolation_prevents_mixing(tmp_path):
    """
    CRITICAL TEST:
    Verify that when Document A and Document B are indexed,
    querying Document B will NEVER return or leak information from Document A.
    """
    index_file = tmp_path / "iso_faiss.bin"
    meta_file = tmp_path / "iso_meta.json"
    store = FaissVectorStore(dimension=3, index_file=index_file, metadata_file=meta_file)

    # Ingest Document A: Alpha Policy
    chunks_a = [
        {
            "chunk_id": 0,
            "text": "Alpha Corp employees receive 35 days paid annual leave.",
            "source": "Alpha_Policy.pdf",
            "page": 1,
        }
    ]
    emb_a = [[1.0, 0.0, 0.0]]
    store.store(chunks_a, emb_a, document_id="Alpha_Policy.pdf")

    # Ingest Document B: Beta Transcript
    chunks_b = [
        {
            "chunk_id": 0,
            "text": "Student John Doe achieved CGPA 3.90 in Computer Science.",
            "source": "Beta_Transcript.pdf",
            "page": 1,
        }
    ]
    emb_b = [[0.0, 1.0, 0.0]]
    store.store(chunks_b, emb_b, document_id="Beta_Transcript.pdf")

    # 1. Search with document_ids=["Beta_Transcript.pdf"]
    results_beta_only = store.search(
        query_embedding=[1.0, 0.0, 0.0],  # Vector similar to Alpha_Policy chunk
        match_count=5,
        match_threshold=0.10,
        document_ids=["Beta_Transcript.pdf"],
    )
    # MUST NOT contain Alpha_Policy chunks!
    assert all(r["document_id"] == "Beta_Transcript.pdf" for r in results_beta_only)
    assert not any("Alpha Corp" in r["content"] for r in results_beta_only)

    # 2. Search with document_ids=["Alpha_Policy.pdf"]
    results_alpha_only = store.search(
        query_embedding=[1.0, 0.0, 0.0],
        match_count=5,
        match_threshold=0.10,
        document_ids=["Alpha_Policy.pdf"],
    )
    assert len(results_alpha_only) == 1
    assert results_alpha_only[0]["document_id"] == "Alpha_Policy.pdf"
    assert "Alpha Corp" in results_alpha_only[0]["content"]


def test_document_deletion_erases_vectors(tmp_path):
    index_file = tmp_path / "del_faiss.bin"
    meta_file = tmp_path / "del_meta.json"
    store = FaissVectorStore(dimension=3, index_file=index_file, metadata_file=meta_file)

    chunks = [
        {"chunk_id": 0, "text": "Secret financial report 2026.", "source": "Finance.pdf", "page": 1},
        {"chunk_id": 1, "text": "Public handbook guide.", "source": "Public.pdf", "page": 1},
    ]
    embeddings = [
        [0.707, 0.707, 0.0],
        [0.0, 0.707, 0.707],
    ]
    store.store(chunks, embeddings, document_id="Finance.pdf")

    # Delete Finance.pdf
    deleted_count = store.delete_document("Finance.pdf")
    assert deleted_count >= 1

    # Search should find nothing from Finance.pdf
    res = store.search([0.707, 0.707, 0.0], match_threshold=0.1, document_ids=["Finance.pdf"])
    assert len(res) == 0


def test_api_chat_with_document_isolation():
    # Chat with explicit document_ids targeting sample_company_policy.pdf
    payload = {
        "question": "How many vacation days do employees receive annually?",
        "backend": "supabase",
        "match_threshold": 0.15,
        "match_count": 3,
        "document_ids": ["sample_company_policy.pdf"],
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["validation"]["is_grounded"] is True
    assert "sample_company_policy.pdf" in data["answer"]


def test_api_list_and_preview_documents():
    # Test GET /api/documents
    res = client.get("/api/documents")
    assert res.status_code == 200
    data = res.json()
    assert "documents" in data

    # Test GET preview for existing sample policy
    preview_res = client.get("/api/documents/sample_company_policy.pdf/preview")
    assert preview_res.status_code in {200, 404}  # 200 if sample present, 404 otherwise


def test_api_upload_rejects_executable():
    files = {"file": ("malicious.exe", b"binarycontent", "application/x-msdownload")}
    response = client.post("/api/upload", files=files)
    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]
