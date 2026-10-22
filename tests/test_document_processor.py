
import pytest

from src.document_processor import chunk_text, chunk_document


def test_chunk_text_returns_overlapping_chunks():
    text = "abcdefghijklmnopqrstuvwxyz"

    chunks = chunk_text(text, chunk_size=10, overlap=3)

    assert chunks == [
        "abcdefghij",
        "hijklmnopq",
        "opqrstuvwx",
        "vwxyz",
    ]


def test_chunk_text_empty_input():
    assert chunk_text("   ") == []


def test_chunk_text_rejects_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_text("Some text", chunk_size=10, overlap=10)


def test_chunk_document_preserves_metadata():
    records = [
        {
            "text": "A" * 20,
            "source": "policy.pdf",
            "page": 2,
        }
    ]

    chunks = chunk_document(records, chunk_size=10, overlap=2)

    assert len(chunks) == 3
    assert chunks[0]["source"] == "policy.pdf"
    assert chunks[0]["page"] == 2
    assert chunks[0]["chunk_id"] == 0
