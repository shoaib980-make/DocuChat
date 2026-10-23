import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sentence_transformers import SentenceTransformer
from src.document_processor import extract_document, chunk_document
from src.vector_store import store_chunks

DOCUMENT_PATH = PROJECT_ROOT / "data" / "sample_company_policy.pdf"


def main():
    if not DOCUMENT_PATH.exists():
        raise FileNotFoundError(
            f"Sample PDF not found: {DOCUMENT_PATH}"
        )

    print("Extracting document...")
    records = extract_document(DOCUMENT_PATH)
    chunks = chunk_document(records, chunk_size=200, overlap=40)

    print(f"Extracted records: {len(records)}")
    print(f"Created chunks: {len(chunks)}")

    if not chunks:
        raise RuntimeError("No chunks were created from the PDF.")

    print("Loading embedding model...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("Generating embeddings...")
    embeddings = model.encode(
        [chunk["text"] for chunk in chunks],
        normalize_embeddings=True,
    ).tolist()

    print("Storing chunks in Supabase...")
    stored_count = store_chunks(
        chunks=chunks,
        embeddings=embeddings,
        document_id=DOCUMENT_PATH.name,
    )

    print(f"Supabase rows returned by upsert: {stored_count}")


if __name__ == "__main__":
    main()