import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.retriever import search_documents


def main():
    questions = [
        "How many paid vacation days do employees receive?",
        "What are the rules for working remotely?",
        "How should production credentials be protected?",
    ]

    for question in questions:
        print(f"\nQuestion: {question}")

        results = search_documents(
            question,
            match_count=3,
            match_threshold=0.20,
        )

        if not results:
            print("No relevant chunks found.")
            continue

        for rank, result in enumerate(results, start=1):
            print(
                f"\n{rank}. Similarity: "
                f"{result['similarity']:.3f}"
            )
            print(f"Source: {result['source']}")
            print(f"Page: {result['page']}")
            print(result["content"])


if __name__ == "__main__":
    main()