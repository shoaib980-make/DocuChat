
import numpy as np
import pytest
import faiss

from src.semantic_search import (
    build_index,
    create_embeddings,
    load_snippets,
    search,
)


class FakeEmbeddingModel:
    """Small deterministic model for fast unit tests."""

    def encode(
        self,
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ):
        vectors = {
            "vacation policy": [1.0, 0.0, 0.0],
            "employees receive paid leave": [0.99, 0.01, 0.0],
            "database technology": [0.0, 1.0, 0.0],
            "postgresql is our database": [0.01, 0.99, 0.0],
            "unrelated topic": [0.0, 0.0, 1.0],
        }

        result = np.array(
            [vectors[text] for text in texts],
            dtype=np.float32,
        )

        if normalize_embeddings:
            norms = np.linalg.norm(result, axis=1, keepdims=True)
            result = result / norms

        return result


def test_load_snippets():
    snippets = load_snippets()

    assert len(snippets) == 50
    assert all(snippet.strip() for snippet in snippets)


def test_create_embeddings_shape():
    model = FakeEmbeddingModel()
    snippets = ["vacation policy", "database technology"]

    embeddings = create_embeddings(snippets, model)

    assert embeddings.shape == (2, 3)
    assert embeddings.dtype == np.float32
    assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0)


def test_build_index():
    embeddings = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )

    index = build_index(embeddings)

    assert isinstance(index, faiss.Index)
    assert index.ntotal == 2
    assert index.d == 3


def test_search_returns_relevant_snippet():
    model = FakeEmbeddingModel()
    snippets = [
        "employees receive paid leave",
        "postgresql is our database",
        "unrelated topic",
    ]

    embeddings = create_embeddings(snippets, model)
    index = build_index(embeddings)

    # This query uses the same deterministic representation
    # as the vacation policy in the fake model.
    results = search(
        "vacation policy",
        model,
        index,
        snippets,
        top_k=2,
    )

    assert len(results) == 2
    assert results[0]["snippet"] == "employees receive paid leave"
    assert results[0]["score"] > 0.9


def test_search_rejects_empty_question():
    model = FakeEmbeddingModel()
    snippets = ["employees receive paid leave"]

    embeddings = create_embeddings(snippets, model)
    index = build_index(embeddings)

    with pytest.raises(ValueError, match="Question cannot be empty"):
        search("   ", model, index, snippets)


def test_search_rejects_invalid_top_k():
    model = FakeEmbeddingModel()
    snippets = ["employees receive paid leave"]

    embeddings = create_embeddings(snippets, model)
    index = build_index(embeddings)

    with pytest.raises(ValueError, match="top_k must be at least 1"):
        search("vacation policy", model, index, snippets, top_k=0)
