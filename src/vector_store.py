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
        session_id: str | None = None,
    ) -> int:
        """Store chunk records with their vector embeddings."""
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        match_count: int = 5,
        match_threshold: float = 0.20,
        document_ids: list[str] | None = None,
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search for top similar chunks given a query embedding and optional document filters."""
        pass

    @abstractmethod
    def delete_document(
        self,
        document_id: str,
        session_id: str | None = None,
    ) -> int:
        """Delete all chunks and vectors associated with a document."""
        pass

    @abstractmethod
    def list_documents(
        self,
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """List summary of indexed documents."""
        pass


class SupabaseVectorStore(BaseVectorStore):
    """
    Hosted vector store backed by PostgreSQL pgvector on Supabase.
    Ideal for production, concurrent multi-user environments with ACID guarantees.
    """

    _rpc_has_filter_param: bool = True

    def __init__(self, client: Client | None = None):
        self.client = client or get_supabase_client()

    def store(
        self,
        chunks: list[dict[str, Any]],
        embeddings: list[list[float]],
        document_id: str,
        session_id: str | None = None,
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
        document_ids: list[str] | None = None,
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        target_docs = [d.strip() for d in document_ids if d and d.strip()] if document_ids else None

        rpc_params: dict[str, Any] = {
            "query_embedding": query_embedding,
            "match_threshold": match_threshold,
            "match_count": max(match_count * 5, 25) if target_docs else match_count,
        }
        if target_docs and SupabaseVectorStore._rpc_has_filter_param:
            rpc_params["filter_document_ids"] = target_docs

        try:
            response = self.client.rpc("match_document_chunks", rpc_params).execute()
            results = response.data or []
        except Exception as e:
            # Fallback for legacy RPC signature without filter_document_ids
            SupabaseVectorStore._rpc_has_filter_param = False
            logger.debug("RPC search fallback without filter_document_ids: %s", e)
            rpc_params.pop("filter_document_ids", None)
            response = self.client.rpc("match_document_chunks", rpc_params).execute()
            results = response.data or []

        formatted = []
        doc_set = set(target_docs) if target_docs else None

        for row in results:
            doc_id = str(row.get("document_id") or "")
            source = str(row.get("source") or "")

            # Strict document isolation filter:
            if doc_set is not None:
                if doc_id not in doc_set and source not in doc_set:
                    continue

            formatted.append({
                "id": row.get("id"),
                "document_id": doc_id,
                "chunk_index": row.get("chunk_index"),
                "content": row.get("content", ""),
                "source": source or doc_id,
                "page": row.get("page"),
                "similarity": float(row.get("similarity", 0.0)),
            })
            if len(formatted) >= match_count:
                break

        return formatted

    def delete_document(
        self,
        document_id: str,
        session_id: str | None = None,
    ) -> int:
        try:
            res1 = self.client.table("document_chunks").delete().eq("document_id", document_id).execute()
            count = len(res1.data) if res1.data else 0
            if count == 0:
                res2 = self.client.table("document_chunks").delete().eq("source", document_id).execute()
                count += len(res2.data) if res2.data else 0
            return count
        except Exception as e:
            logger.error("Failed to delete document '%s' from Supabase: %s", document_id, e)
            return 0

    def list_documents(
        self,
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        try:
            res = self.client.table("document_chunks").select("document_id, source, page").execute()
            docs: dict[str, dict[str, Any]] = {}
            for row in (res.data or []):
                doc_id = row.get("document_id") or row.get("source") or "unknown"
                if doc_id not in docs:
                    docs[doc_id] = {
                        "document_id": doc_id,
                        "filename": row.get("source") or doc_id,
                        "chunk_count": 0,
                        "pages": set(),
                    }
                docs[doc_id]["chunk_count"] += 1
                if row.get("page") is not None:
                    docs[doc_id]["pages"].add(row.get("page"))

            return [
                {
                    "document_id": d["document_id"],
                    "filename": d["filename"],
                    "chunk_count": d["chunk_count"],
                    "pages_count": len(d["pages"]) if d["pages"] else 1,
                }
                for d in docs.values()
            ]
        except Exception as e:
            logger.error("Failed listing documents from Supabase: %s", e)
            return []


class FaissVectorStore(BaseVectorStore):
    """
    Local vector store backed by FAISS IndexFlatIP (cosine similarity on normalized vectors).
    Ideal for local development, air-gapped environments, edge devices, and fast local fallback.
    """

    def __init__(
        self,
        dimension: int = 384,
        index_file: Path | str | None = None,
        metadata_file: Path | str | None = None,
    ):
        self.dimension = dimension
        self.index_file = Path(index_file) if index_file else FAISS_DEFAULT_INDEX_PATH
        self.metadata_file = Path(metadata_file) if metadata_file else FAISS_DEFAULT_META_PATH
        self.vectors_file = self.index_file.with_suffix(".npy") if self.index_file else None

        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata: list[dict[str, Any]] = []
        self.vectors = np.empty((0, self.dimension), dtype=np.float32)

        self._load_from_disk()

    def _load_from_disk(self) -> None:
        if self.index_file and self.metadata_file:
            if self.index_file.exists() and self.metadata_file.exists():
                try:
                    self.index = faiss.read_index(str(self.index_file))
                    with open(self.metadata_file, "r", encoding="utf-8") as f:
                        self.metadata = json.load(f)

                    if self.vectors_file and self.vectors_file.exists():
                        self.vectors = np.load(str(self.vectors_file))
                    else:
                        # Reconstruct if supported or initialize empty
                        self.vectors = np.empty((0, self.dimension), dtype=np.float32)

                    logger.info(
                        "Loaded FAISS index (%d vectors) from disk.",
                        self.index.ntotal,
                    )
                except Exception as e:
                    logger.warning("Could not load existing FAISS index: %s", e)
                    self.index = faiss.IndexFlatIP(self.dimension)
                    self.metadata = []
                    self.vectors = np.empty((0, self.dimension), dtype=np.float32)

    def _save_to_disk(self) -> None:
        if self.index_file and self.metadata_file:
            try:
                self.index_file.parent.mkdir(parents=True, exist_ok=True)
                faiss.write_index(self.index, str(self.index_file))
                with open(self.metadata_file, "w", encoding="utf-8") as f:
                    json.dump(self.metadata, f, indent=2)
                if self.vectors_file and len(self.vectors) > 0:
                    np.save(str(self.vectors_file), self.vectors)
            except (OSError, PermissionError) as e:
                logger.warning("Could not persist FAISS index to disk (%s). Keeping in memory.", e)

    def store(
        self,
        chunks: list[dict[str, Any]],
        embeddings: list[list[float]],
        document_id: str,
        session_id: str | None = None,
    ) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError("Each chunk must have exactly one embedding.")

        if not chunks:
            return 0

        vectors_np = np.asarray(embeddings, dtype=np.float32)
        if vectors_np.ndim != 2 or vectors_np.shape[1] != self.dimension:
            raise ValueError(f"Expected {self.dimension}-dimensional embeddings.")

        # Ensure L2 normalization for cosine similarity with IndexFlatIP
        faiss.normalize_L2(vectors_np)
        self.index.add(vectors_np)

        if len(self.vectors) == 0:
            self.vectors = vectors_np
        else:
            self.vectors = np.vstack([self.vectors, vectors_np])

        start_idx = len(self.metadata)
        for i, chunk in enumerate(chunks):
            self.metadata.append({
                "id": start_idx + i + 1,
                "document_id": document_id,
                "chunk_index": chunk.get("chunk_id", i),
                "content": chunk.get("text", ""),
                "source": chunk.get("source", document_id),
                "page": chunk.get("page"),
                "session_id": session_id,
            })

        self._save_to_disk()
        return len(chunks)

    def search(
        self,
        query_embedding: list[float],
        match_count: int = 5,
        match_threshold: float = 0.20,
        document_ids: list[str] | None = None,
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        if self.index.ntotal == 0:
            return []

        query_np = np.asarray([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_np)

        k = self.index.ntotal
        scores, indices = self.index.search(query_np, k)

        target_docs = [d.strip() for d in document_ids if d and d.strip()] if document_ids else None
        doc_filter_set = set(target_docs) if target_docs else None

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or idx >= len(self.metadata):
                continue
            sim = float(score)
            if sim < match_threshold:
                continue

            item = dict(self.metadata[int(idx)])
            doc_id = str(item.get("document_id") or "")
            source = str(item.get("source") or "")

            # Strict document isolation filter
            if doc_filter_set is not None:
                if doc_id not in doc_filter_set and source not in doc_filter_set:
                    continue

            if session_id and item.get("session_id") and item.get("session_id") != session_id:
                continue

            item["similarity"] = sim
            results.append(item)
            if len(results) >= match_count:
                break

        return results

    def delete_document(
        self,
        document_id: str,
        session_id: str | None = None,
    ) -> int:
        if not self.metadata:
            return 0

        keep_indices = []
        keep_metadata = []
        deleted_count = 0

        for idx, item in enumerate(self.metadata):
            matches_doc = (item.get("document_id") == document_id or item.get("source") == document_id)
            matches_session = (session_id is None or item.get("session_id") == session_id)

            if matches_doc and matches_session:
                deleted_count += 1
            else:
                keep_indices.append(idx)
                keep_metadata.append(item)

        if deleted_count > 0:
            self.metadata = keep_metadata
            self.index = faiss.IndexFlatIP(self.dimension)

            if len(self.vectors) == len(self.metadata) + deleted_count and keep_indices:
                self.vectors = self.vectors[keep_indices]
                self.index.add(self.vectors)
            else:
                self.vectors = np.empty((0, self.dimension), dtype=np.float32)

            self._save_to_disk()

        return deleted_count

    def list_documents(
        self,
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        docs: dict[str, dict[str, Any]] = {}
        for item in self.metadata:
            if session_id and item.get("session_id") and item.get("session_id") != session_id:
                continue

            doc_id = item.get("document_id") or item.get("source") or "unknown"
            if doc_id not in docs:
                docs[doc_id] = {
                    "document_id": doc_id,
                    "filename": item.get("source") or doc_id,
                    "chunk_count": 0,
                    "pages": set(),
                }
            docs[doc_id]["chunk_count"] += 1
            if item.get("page") is not None:
                docs[doc_id]["pages"].add(item.get("page"))

        return [
            {
                "document_id": d["document_id"],
                "filename": d["filename"],
                "chunk_count": d["chunk_count"],
                "pages_count": len(d["pages"]) if d["pages"] else 1,
            }
            for d in docs.values()
        ]


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
    session_id: str | None = None,
) -> int:
    """Convenience wrapper for storing chunks in the configured vector store."""
    store = get_vector_store(backend)
    return store.store(chunks, embeddings, document_id, session_id=session_id)


def delete_document_chunks(
    document_id: str,
    backend: str | None = None,
    session_id: str | None = None,
) -> int:
    """Convenience wrapper for deleting all chunks of a document."""
    store = get_vector_store(backend)
    return store.delete_document(document_id, session_id=session_id)