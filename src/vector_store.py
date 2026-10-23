import json
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from dotenv import load_dotenv
from supabase import Client, create_client

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

logger = logging.getLogger(__name__)

FAISS_DEFAULT_INDEX_PATH = PROJECT_ROOT / "data" / "faiss_index.bin"
FAISS_DEFAULT_META_PATH = PROJECT_ROOT / "data" / "faiss_metadata.json"


def get_supabase_client() -> Client:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SECRET_KEY")

    if not url or not key:
        raise RuntimeError(
            "SUPABASE_URL or SUPABASE_SECRET_KEY is missing from .env"
        )

    return create_client(url, key)


class BaseVectorStore(ABC):
    """
    Abstract Vector Store interface ensuring interchangeable backends:
    FAISS (local / in-memory / file fallback) and Supabase (hosted pgvector).
    """

    @abstractmethod
    def store(
        self,
        chunks: list[dict[str, Any]],
        embeddings: list[list[float]],
        document_id: str,
    ) -> int:
        """Store chunk records with their vector embeddings."""
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        match_count: int = 5,
        match_threshold: float = 0.20,
    ) -> list[dict[str, Any]]:
        """Search for top similar chunks given a query embedding."""
        pass


class SupabaseVectorStore(BaseVectorStore):
    """
    Hosted vector store backed by PostgreSQL pgvector on Supabase.
    Ideal for production, concurrent multi-user environments with ACID guarantees.
    """

    def __init__(self, client: Client | None = None):
        self.client = client or get_supabase_client()

    def store(
        self,
        chunks: list[dict[str, Any]],
        embeddings: list[list[float]],
        document_id: str,
    ) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError("Each chunk must have exactly one embedding.")

        if not chunks:
            return 0

        rows = []
        for chunk, embedding in zip(chunks, embeddings):
            if len(embedding) != 384:
                raise ValueError("Expected 384-dimensional embeddings.")

            vector_value = "[" + ",".join(
                str(float(value)) for value in embedding
            ) + "]"

            rows.append({
                "document_id": document_id,
                "chunk_index": chunk.get("chunk_id", 0),
                "content": chunk.get("text", ""),
                "source": chunk.get("source", document_id),
                "page": chunk.get("page"),
                "embedding": vector_value,
            })

        response = (
            self.client.table("document_chunks")
            .upsert(rows, on_conflict="document_id,chunk_index")
            .execute()
        )
        return len(response.data) if response.data else len(rows)

    def search(
        self,
        query_embedding: list[float],
        match_count: int = 5,
        match_threshold: float = 0.20,
    ) -> list[dict[str, Any]]:
        response = self.client.rpc(
            "match_document_chunks",
            {
                "query_embedding": query_embedding,
                "match_threshold": match_threshold,
                "match_count": match_count,
            },
        ).execute()

        results = response.data or []
        formatted = []
        for row in results:
            formatted.append({
                "id": row.get("id"),
                "document_id": row.get("document_id"),
                "chunk_index": row.get("chunk_index"),
                "content": row.get("content", ""),
                "source": row.get("source", ""),
                "page": row.get("page"),
                "similarity": float(row.get("similarity", 0.0)),
            })
        return formatted


class FaissVectorStore(BaseVectorStore):
    """
    Local vector store backed by FAISS IndexFlatIP (cosine similarity on normalized vectors).
    Ideal for local development, air-gapped environments, edge devices, and fast local fallback.
    """

    def __init__(
        self,
        dimension: int = 384,
        index_file: Path | None = FAISS_DEFAULT_INDEX_PATH,
        metadata_file: Path | None = FAISS_DEFAULT_META_PATH,
    ):
        self.dimension = dimension
        self.index_file = index_file
        self.metadata_file = metadata_file
        self.metadata: list[dict[str, Any]] = []
        self.index: faiss.IndexFlatIP = faiss.IndexFlatIP(dimension)

        self._load_if_exists()

    def _load_if_exists(self) -> None:
        if (
            self.index_file
            and self.index_file.exists()
            and self.metadata_file
            and self.metadata_file.exists()
        ):
            try:
                self.index = faiss.read_index(str(self.index_file))
                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
                logger.info(
                    "Loaded existing FAISS index with %d vectors",
                    self.index.ntotal,
                )
            except Exception as e:
                logger.warning("Could not load existing FAISS index: %s", e)
                self.index = faiss.IndexFlatIP(self.dimension)
                self.metadata = []

    def _save_to_disk(self) -> None:
        if self.index_file and self.metadata_file:
            self.index_file.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self.index, str(self.index_file))
            with open(self.metadata_file, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, indent=2)

    def store(
        self,
        chunks: list[dict[str, Any]],
        embeddings: list[list[float]],
        document_id: str,
    ) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError("Each chunk must have exactly one embedding.")

        if not chunks:
            return 0

        vectors_np = np.asarray(embeddings, dtype=np.float32)
        if vectors_np.shape[1] != self.dimension:
            raise ValueError(f"Expected {self.dimension}-dimensional embeddings.")

        # Ensure L2 normalization for cosine similarity with IndexFlatIP
        faiss.normalize_L2(vectors_np)
        self.index.add(vectors_np)

        start_idx = len(self.metadata)
        for i, chunk in enumerate(chunks):
            self.metadata.append({
                "id": start_idx + i + 1,
                "document_id": document_id,
                "chunk_index": chunk.get("chunk_id", i),
                "content": chunk.get("text", ""),
                "source": chunk.get("source", document_id),
                "page": chunk.get("page"),
            })

        self._save_to_disk()
        return len(chunks)

    def search(
        self,
        query_embedding: list[float],
        match_count: int = 5,
        match_threshold: float = 0.20,
    ) -> list[dict[str, Any]]:
        if self.index.ntotal == 0:
            return []

        query_np = np.asarray([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_np)

        k = min(match_count, self.index.ntotal)
        scores, indices = self.index.search(query_np, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            sim = float(score)
            if sim >= match_threshold:
                item = dict(self.metadata[int(idx)])
                item["similarity"] = sim
                results.append(item)

        # Sort descending by similarity
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results


def get_vector_store(backend: str | None = None) -> BaseVectorStore:
    """
    Vector store factory.
    Enables swapping FAISS and Supabase pgvector behind the same interface.
    Controlled via backend parameter or VECTOR_STORE_BACKEND environment variable.
    """
    selected = (backend or os.getenv("VECTOR_STORE_BACKEND", "supabase")).lower()

    if selected == "faiss":
        return FaissVectorStore()
    elif selected == "supabase":
        try:
            return SupabaseVectorStore()
        except Exception as e:
            logger.warning(
                "Failed to initialize Supabase store (%s). Falling back to FAISS.",
                e,
            )
            return FaissVectorStore()
    else:
        raise ValueError(
            f"Unknown vector store backend: '{selected}'. Supported: 'supabase', 'faiss'"
        )


def store_chunks(
    chunks: list[dict[str, Any]],
    embeddings: list[list[float]],
    document_id: str,
    backend: str | None = None,
) -> int:
    """Convenience wrapper for storing chunks in the configured vector store."""
    store = get_vector_store(backend)
    return store.store(chunks, embeddings, document_id)