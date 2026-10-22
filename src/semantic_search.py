
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "snippets.txt"

# Local embedding model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"



def load_snippets() -> list[str]:
    """Load snippets separated by blank lines or individual lines."""
    text = DATA_PATH.read_text(encoding="utf-8")
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()

    # First try splitting on blank lines.
    snippets = [
        item.strip()
        for item in text.split("\n\n")
        if item.strip()
    ]

    # If the file contains one paragraph with all 50 sentences,
    # split on individual lines instead.
    if len(snippets) == 1:
        snippets = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

    if not snippets:
        raise ValueError("No snippets found in the dataset.")

    return snippets



def create_embeddings(
    snippets: list[str],
    model: SentenceTransformer,
) -> np.ndarray:
    """Convert text snippets into normalized embedding vectors."""
    embeddings = model.encode(
        snippets,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    return np.asarray(embeddings, dtype=np.float32)


def build_index(embeddings: np.ndarray) -> faiss.Index:
    """Create a FAISS index for cosine similarity search."""
    dimension = embeddings.shape[1]

    # Inner product equals cosine similarity for normalized vectors.
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index


def search(
    question: str,
    model: SentenceTransformer,
    index: faiss.Index,
    snippets: list[str],
    top_k: int = 3,
) -> list[dict]:
    """Return the most semantically similar snippets."""
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    query_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    query_embedding = np.asarray(query_embedding, dtype=np.float32)

    k = min(top_k, index.ntotal)
    scores, indices = index.search(query_embedding, k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        results.append(
            {
                "snippet": snippets[int(idx)],
                "score": float(score),
                "index": int(idx),
            }
        )

    return results


def main():
    snippets = load_snippets()

    print(f"Loaded {len(snippets)} snippets.")
    print(f"Loading embedding model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    embeddings = create_embeddings(snippets, model)
    index = build_index(embeddings)

    print(f"Embedding shape: {embeddings.shape}")
    print(f"FAISS vectors stored: {index.ntotal}")

    questions = [
        "How many vacation days do staff get?",
        "How does the company protect user accounts?",
        "Which database does the backend use?",
        "What happens before code goes into production?",
        "How can a customer get their money back?",
    ]

    for question in questions:
        print(f"\nQuestion: {question}")

        results = search(question, model, index, snippets, top_k=3)

        for rank, result in enumerate(results, start=1):
            print(
                f"{rank}. Score: {result['score']:.4f} | "
                f"{result['snippet']}"
            )


if __name__ == "__main__":
    main()
