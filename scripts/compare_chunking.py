import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.document_processor import extract_document, chunk_document

DATA_DIR = PROJECT_ROOT / "data"
document_path = DATA_DIR / "sample_company_policy.pdf"

records = extract_document(document_path)

configs = [
    (200, 40),
    (500, 100),
    (1000, 200),
]

for chunk_size, overlap in configs:
    chunks = chunk_document(
        records,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    print(f"\n{'=' * 50}")
    print(f"Chunk size: {chunk_size} | Overlap: {overlap}")
    print(f"Total chunks: {len(chunks)}")

    for chunk in chunks:
        print(f"\nChunk {chunk['chunk_id']} | Page: {chunk['page']}")
        print(chunk["text"])