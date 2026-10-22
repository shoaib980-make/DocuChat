
from pathlib import Path
from typing import Any

from docx import Document
from pypdf import PdfReader


SUPPORTED_EXTENSIONS = {".pdf", ".docx"}


def extract_document(file_path: str | Path) -> list[dict[str, Any]]:
    """
    Extract text from a PDF or DOCX document.

    Each returned record contains:
    - text: extracted text
    - source: original filename
    - page: PDF page number, or None for DOCX
    """
    path = Path(file_path)

    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Document not found: {path}")

    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported document type: {extension}")

    records = []

    if extension == ".pdf":
        reader = PdfReader(str(path))

        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()

            if text:
                records.append(
                    {
                        "text": text,
                        "source": path.name,
                        "page": page_number,
                    }
                )

    elif extension == ".docx":
        document = Document(str(path))

        paragraphs = [
            paragraph.text.strip()
            for paragraph in document.paragraphs
            if paragraph.text.strip()
        ]

        text = "\n".join(paragraphs)

        if text:
            records.append(
                {
                    "text": text,
                    "source": path.name,
                    "page": None,
                }
            )

    if not records:
        raise ValueError(
            f"No extractable text found in {path.name}. "
            "The file may be empty, scanned, or image-based."
        )

    return records


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[str]:
    """
    Split text into overlapping character-based chunks.

    chunk_size and overlap are measured in characters, not tokens.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size.")

    text = text.strip()

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - overlap

    return chunks


def chunk_document(
    records: list[dict[str, Any]],
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[dict[str, Any]]:
    """Chunk extracted document records and preserve their metadata."""
    chunks = []

    for record in records:
        text_chunks = chunk_text(
            record["text"],
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for chunk in text_chunks:
            chunks.append(
                {
                    "chunk_id": len(chunks),
                    "text": chunk,
                    "source": record["source"],
                    "page": record["page"],
                }
            )

    return chunks
