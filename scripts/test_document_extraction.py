import sys
from pathlib import Path

# Add the project root to Python's import path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.document_processor import extract_document, chunk_document

DATA_DIR = PROJECT_ROOT / "data"

for filename in ["sample_company_policy.pdf", "sample_company_policy.docx"]:
    path = DATA_DIR / filename

    records = extract_document(path)
    chunks = chunk_document(records, chunk_size=200, overlap=40)

    print(f"\nDocument: {filename}")
    print(f"Extracted records: {len(records)}")
    print(f"Generated chunks: {len(chunks)}")

    for chunk in chunks:
        print(
            f"\nChunk {chunk['chunk_id']} | "
            f"Page: {chunk['page']} | "
            f"Source: {chunk['source']}"
        )
        print(chunk["text"][:150])