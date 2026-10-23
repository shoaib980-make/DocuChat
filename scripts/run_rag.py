import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag_pipeline import RAGPipeline


def main():
    parser = argparse.ArgumentParser(description="DocuChat RAG Query Runner")
    parser.add_argument(
        "--question",
        "-q",
        type=str,
        default="How many vacation days do employees receive annually?",
        help="Question to query against the documents",
    )
    parser.add_argument(
        "--backend",
        "-b",
        type=str,
        choices=["supabase", "faiss"],
        default="supabase",
        help="Vector store backend to use",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.20,
        help="Minimum similarity threshold for retrieval",
    )
    args = parser.parse_args()

    print("=" * 70)
    print(f" DOCUCHAT RAG QUERY (Backend: {args.backend.upper()})")
    print("=" * 70)
    print(f"Question: {args.question}\n")

    pipeline = RAGPipeline(vector_backend=args.backend)
    result = pipeline.answer_question(
        question=args.question,
        match_threshold=args.threshold,
    )

    print("--- GENERATED ANSWER ---")
    print(result["answer"])
    print("\n--- VALIDATION & GROUNDEDNESS REPORT ---")
    val = result["validation"]
    print(f"Status:             {val['status']}")
    print(f"Is Grounded:        {val['is_grounded']}")
    print(f"Citations Valid:    {val['citations_valid']}")
    print(f"Groundedness Score: {val['groundedness_score'] * 100:.1f}%")
    print(f"Found Citations:    {val['found_citations']}")
    if val["flags"]:
        print(f"Diagnostic Flags:   {val['flags']}")

    print(f"\n--- RETRIEVED CHUNKS ({len(result['retrieved_chunks'])}) ---")
    for i, chunk in enumerate(result["retrieved_chunks"], 1):
        print(
            f"[{i}] Sim: {chunk.get('similarity', 0.0):.3f} | "
            f"Doc: {chunk.get('source')} (Page {chunk.get('page')}, Chunk #{chunk.get('chunk_index')})"
        )
        print(f"    \"{chunk.get('content')[:90]}...\"\n")


if __name__ == "__main__":
    main()
