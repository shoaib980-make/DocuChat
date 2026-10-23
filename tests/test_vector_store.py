import numpy as np
import pytest

from src.vector_store import (
    BaseVectorStore,
    FaissVectorStore,
    SupabaseVectorStore,
    get_vector_store,
)


def test_faiss_vector_store_store_and_search(tmp_path):
    index_file = tmp_path / "test_faiss.bin"
    meta_file = tmp_path / "test_meta.json"

    store = FaissVectorStore(
        dimension=3,
        index_file=index_file,
        metadata_file=meta_file,
    )

    chunks = [
        {"chunk_id": 0, "text": "Vacation policy allows 20 days off.", "source": "policy.pdf", "page": 1},
        {"chunk_id": 1, "text": "Remote work is permitted upon manager approval.", "source": "policy.pdf", "page": 2},
    ]

    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]

    stored = store.store(chunks, embeddings, document_id="policy.pdf")
    assert stored == 2
    assert store.index.ntotal == 2

    # Query matching chunk 0
    results = store.search(query_embedding=[1.0, 0.0, 0.0], match_count=2, match_threshold=0.5)
    assert len(results) == 1
    assert results[0]["chunk_index"] == 0
    assert results[0]["source"] == "policy.pdf"
    assert results[0]["similarity"] > 0.99

    # Check persistence reload
    reloaded_store = FaissVectorStore(
        dimension=3,
        index_file=index_file,
        metadata_file=meta_file,
    )
    assert reloaded_store.index.ntotal == 2
    reload_results = reloaded_store.search(query_embedding=[0.0, 1.0, 0.0], match_count=1)
    assert len(reload_results) == 1
    assert reload_results[0]["chunk_index"] == 1


def test_get_vector_store_factory():
    faiss_store = get_vector_store(backend="faiss")
    assert isinstance(faiss_store, BaseVectorStore)
    assert isinstance(faiss_store, FaissVectorStore)

    supabase_store = get_vector_store(backend="supabase")
    assert isinstance(supabase_store, BaseVectorStore)


def test_faiss_vector_store_rejects_dimension_mismatch(tmp_path):
    store = FaissVectorStore(
        dimension=3,
        index_file=tmp_path / "idx.bin",
        metadata_file=tmp_path / "meta.json",
    )
    chunks = [{"chunk_id": 0, "text": "test", "source": "doc", "page": 1}]
    # 2 dimensions instead of 3
    embeddings = [[1.0, 0.0]]

    with pytest.raises(ValueError, match="Expected 3-dimensional embeddings"):
        store.store(chunks, embeddings, "doc")
