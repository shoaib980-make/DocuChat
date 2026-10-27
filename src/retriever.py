from typing import Any
from sentence_transformers import SentenceTransformer

from src.vector_store import get_vector_store

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_model = None


def get_embedding_model() -> SentenceTransformer:
    global _model

    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)

    return _model


def search_documents(
    question: str,
    match_count: int = 5,
    match_threshold: float = 0.20,
    backend: str | None = None,
    document_ids: list[str] | None = None,
    session_id: str | None = None,
) -> list[dict[str, Any]]:
    """
    Search indexed documents for semantic matches to the user question.
    Supports swapping seamlessly between Supabase pgvector and FAISS,
    and isolates context strictly to specified document_ids and session_id.
    """
    question = question.strip()

    if not question:
        raise ValueError("Question must not be empty.")

    if not 1 <= match_count <= 20:
        raise ValueError("match_count must be between 1 and 20.")

    if not -1.0 <= match_threshold <= 1.0:
        raise ValueError("match_threshold must be between -1 and 1.")

    model = get_embedding_model()

    query_embedding = model.encode(
        question,
        normalize_embeddings=True,
    ).tolist()

    store = get_vector_store(backend)
    return store.search(
        query_embedding=query_embedding,
        match_count=match_count,
        match_threshold=match_threshold,
        document_ids=document_ids,
        session_id=session_id,
    )