import pytest
from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)


def test_api_status_endpoint():
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "default_backend" in data
    assert ".pdf" in data["supported_formats"]


def test_api_serve_home():
    response = client.get("/")
    assert response.status_code == 200
    assert "DocuChat" in response.text


def test_api_chat_valid_query():
    payload = {
        "question": "How many vacation days do employees receive?",
        "backend": "supabase",
        "match_threshold": 0.20,
        "match_count": 2,
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "validation" in data
    assert data["validation"]["is_grounded"] is True


def test_api_chat_empty_query_rejected():
    response = client.post("/api/chat", json={"question": "   "})
    assert response.status_code == 400


def test_api_upload_unsupported_extension():
    files = {"file": ("unsupported.txt", b"plain text content", "text/plain")}
    response = client.post("/api/upload", files=files)
    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]
