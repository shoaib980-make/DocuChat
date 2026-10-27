import base64
import io
import logging
import os
from pathlib import Path
from typing import Any

from docx import Document
from PIL import Image, ExifTags
from pypdf import PdfReader

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".png", ".jpg", ".jpeg"}


def _ocr_image_with_tesseract(image: Image.Image) -> str:
    """Attempt OCR using pytesseract if available."""
    try:
        import pytesseract
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        logger.debug("pytesseract OCR not available or failed: %s", e)
        return ""


def _ocr_image_with_openai(image: Image.Image, api_key: str) -> str:
    """Attempt OCR via OpenAI GPT-4o-mini vision if API key is provided."""
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)

        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        b64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Extract and transcribe all readable text, tables, and labels from this image verbatim. If no text exists, reply with 'NO_TEXT'.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64_image}"},
                        },
                    ],
                }
            ],
            max_tokens=1000,
            temperature=0.0,
        )
        content = response.choices[0].message.content or ""
        if "NO_TEXT" in content.strip():
            return ""
        return content.strip()
    except Exception as e:
        logger.debug("OpenAI vision OCR failed: %s", e)
        return ""


def _extract_image_metadata_text(image: Image.Image, filename: str) -> str:
    """Extract descriptive text and EXIF metadata from an image."""
    parts = [f"Image Document: {filename}"]
    parts.append(f"Format: {image.format or 'Image'}, Dimensions: {image.width}x{image.height}px, Mode: {image.mode}")

    # Check for text in PNG info
    if hasattr(image, "info") and image.info:
        for k, v in image.info.items():
            if isinstance(v, str) and len(v.strip()) > 0 and k.lower() not in {"dpi", "icc_profile"}:
                parts.append(f"{k}: {v.strip()}")

    # Check EXIF
    try:
        exif = image.getexif()
        if exif:
            for tag_id, value in exif.items():
                tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                if isinstance(value, str) and value.strip() and tag_name.lower() in {"imagedescription", "usercomment", "software", "artist"}:
                    parts.append(f"{tag_name}: {value.strip()}")
    except Exception:
        pass

    return "\n".join(parts)


def extract_document(file_path: str | Path) -> list[dict[str, Any]]:
    """
    Extract text and metadata from a PDF, DOCX, PNG, JPG, or JPEG document.
    Performs OCR on scanned PDF pages and image files when available.

    Each returned record contains:
    - text: extracted text
    - source: original filename
    - page: page number (int) or None
    """
    path = Path(file_path)

    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Document not found: {path}")

    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file format '{extension}'. Please upload a .pdf, .docx, .png, .jpg, or .jpeg file."
        )

    records: list[dict[str, Any]] = []
    openai_key = os.getenv("OPENAI_API_KEY")

    if extension == ".pdf":
        reader = PdfReader(str(path))

        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()

            # If page text extraction is blank, attempt OCR on embedded images
            if not text and hasattr(page, "images") and page.images:
                ocr_texts = []
                for img_obj in page.images:
                    try:
                        pil_img = Image.open(io.BytesIO(img_obj.data))
                        ocr_res = _ocr_image_with_tesseract(pil_img)
                        if not ocr_res and openai_key:
                            ocr_res = _ocr_image_with_openai(pil_img, openai_key)
                        if ocr_res:
                            ocr_texts.append(ocr_res)
                    except Exception as err:
                        logger.debug("Failed extracting image from PDF page %d: %s", page_number, err)
                if ocr_texts:
                    text = "\n".join(ocr_texts).strip()

            if text:
                records.append({
                    "text": text,
                    "source": path.name,
                    "page": page_number,
                })

    elif extension == ".docx":
        document = Document(str(path))
        paragraphs = [
            paragraph.text.strip()
            for paragraph in document.paragraphs
            if paragraph.text.strip()
        ]
        text = "\n".join(paragraphs).strip()
        if text:
            records.append({
                "text": text,
                "source": path.name,
                "page": None,
            })

    elif extension in {".png", ".jpg", ".jpeg"}:
        try:
            with Image.open(str(path)) as img:
                # 1. Try Tesseract OCR
                text = _ocr_image_with_tesseract(img)

                # 2. Try OpenAI Vision OCR if API key present
                if not text and openai_key:
                    text = _ocr_image_with_openai(img, openai_key)

                # 3. Fallback: image metadata / description
                if not text:
                    text = _extract_image_metadata_text(img, path.name)

                if text:
                    records.append({
                        "text": text,
                        "source": path.name,
                        "page": 1,
                    })
        except Exception as e:
            logger.error("Failed to read image %s: %s", path.name, e)
            raise ValueError(f"Could not read image file '{path.name}': {e}")

    if not records:
        raise ValueError(
            f"No extractable text found in '{path.name}'. "
            "The file may be empty or contain unscannable content."
        )

    return records


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[str]:
    """
    Create overlapping character-based chunks.
    Prefer whitespace boundaries to avoid splitting words.
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
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)

        if end < text_length:
            boundary = text.rfind(" ", start, end)
            if boundary > start:
                end = boundary

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        next_start = max(start + 1, end - overlap)

        if next_start > 0 and not text[next_start - 1].isspace():
            whitespace = text.find(" ", next_start, min(next_start + 100, text_length))
            if whitespace != -1:
                next_start = whitespace + 1

        start = next_start

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
            chunks.append({
                "chunk_id": len(chunks),
                "text": chunk,
                "source": record["source"],
                "page": record["page"],
            })

    return chunks
