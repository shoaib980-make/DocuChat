import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag_pipeline import (
    RAGPromptBuilder,
    DeterministicGroundedGenerator,
    AnswerValidator,
    REFUSAL_MESSAGE,
)
from src.retriever import search_documents

DEBUG_CASES = [
    {
        "case_id": "CASE-1",
        "title": "Strict Threshold Filtering Out Relevant Knowledge",
        "question": "How many vacation days do employees receive?",
        "test_threshold": 0.85,  # Artificially strict threshold
        "prompt_type": "standard_grounded",
        "description": "User asks a valid question, but similarity threshold is set unreasonably high (0.85).",
    },
    {
        "case_id": "CASE-2",
        "title": "Corpus Coverage Gap / Out-of-Domain Query",
        "question": "Where is the cafeteria menu and daily specials posted?",
        "test_threshold": 0.20,
        "prompt_type": "standard_grounded",
        "description": "Document corpus does not contain any food or cafeteria policy.",
    },
    {
        "case_id": "CASE-3",
        "title": "Unconstrained Hallucinatory Prompt (Weak Grounding)",
        "question": "How should production credentials be protected?",
        "test_threshold": 0.20,
        "prompt_type": "unconstrained_prompt",  # Injects hallucinated outside claims
        "description": "Context is retrieved successfully, but prompt encourages outside speculation.",
    },
    {
        "case_id": "CASE-4",
        "title": "Missing Citation Formatting Instruction",
        "question": "What is the policy for working remotely?",
        "test_threshold": 0.20,
        "prompt_type": "missing_citations",  # Generates answer without [Doc:... Chunk:...]
        "description": "Context is retrieved and answer is accurate, but citation syntax is omitted.",
    },
    {
        "case_id": "CASE-5",
        "title": "Fully Grounded Baseline (Optimal Retrieval + Prompt)",
        "question": "What are the rules for working remotely?",
        "test_threshold": 0.20,
        "prompt_type": "standard_grounded",
        "description": "Optimal retrieval threshold and strict grounding prompt.",
    },
]


def diagnose_case(case: dict, backend: str = "supabase") -> dict:
    question = case["question"]
    threshold = case["test_threshold"]
    prompt_type = case["prompt_type"]

    # 1. Retrieve chunks
    retrieved = search_documents(
        question=question,
        match_count=3,
        match_threshold=threshold,
        backend=backend,
    )

    max_sim = max((c.get("similarity", 0.0) for c in retrieved), default=0.0)

    # Check unconstrained retrieval (threshold=0.10) to see if knowledge actually exists in DB
    full_search = search_documents(
        question=question,
        match_count=3,
        match_threshold=0.10,
        backend=backend,
    )
    unconstrained_max_sim = max((c.get("similarity", 0.0) for c in full_search), default=0.0)

    # 2. Simulate or generate answer based on prompt_type
    generator = DeterministicGroundedGenerator()
    if prompt_type == "standard_grounded":
        raw_answer = generator.generate(question, retrieved)
    elif prompt_type == "unconstrained_prompt":
        # Simulate LLM adding ungrounded outside advice (hallucination)
        raw_answer = (
            "Production credentials should be stored in AWS Secrets Manager or HashiCorp Vault. "
            "You should also rotate keys every 90 days and use RSA 4096-bit encryption."
        )
    elif prompt_type == "missing_citations":
        # Accurate answer but stripped of citations
        raw_answer = (
            "Employees may request remote work with manager approval. Teams must maintain normal "
            "communication and availability during working hours."
        )
    else:
        raw_answer = generator.generate(question, retrieved)

    # 3. Validate
    val = AnswerValidator.validate(raw_answer, retrieved, match_threshold=threshold)

    # 4. Diagnostic Root-Cause Analysis
    if len(retrieved) == 0:
        if unconstrained_max_sim >= 0.50:
            fault = "RETRIEVAL_FAULT"
            root_cause = (
                f"Threshold too aggressive: chunks exist at sim={unconstrained_max_sim:.3f}, "
                f"but filtered by test threshold {threshold:.2f}."
            )
            recommendation = f"Lower match_threshold from {threshold:.2f} to 0.25-0.35, or tune chunking size."
        else:
            fault = "CORPUS_GAP"
            root_cause = f"Corpus does not contain this information (max similarity only {unconstrained_max_sim:.3f})."
            recommendation = "Ingest the missing company documents into the knowledge base."
    elif not val["citations_valid"] and val["groundedness_score"] > 0.8:
        fault = "PROMPTING_FAULT"
        root_cause = "Retrieved chunks are accurate, but prompt failed to enforce mandatory inline [Doc:..., Chunk:...] tags."
        recommendation = "Strengthen system prompt Rule 3 with few-shot citation examples."
    elif not val["is_grounded"] and val["status"] == "POTENTIAL_HALLUCINATION":
        fault = "PROMPTING_FAULT"
        root_cause = "LLM introduced external knowledge not present in the retrieved chunks."
        recommendation = "Set temperature=0.0 and enforce: 'Rely strictly on provided context; refuse if not present.'"
    else:
        fault = "NONE (SUCCESS)"
        root_cause = "Optimal retrieval and grounded response with proper citations."
        recommendation = "Maintain current retrieval and prompt configuration."

    return {
        "case_id": case["case_id"],
        "title": case["title"],
        "fault": fault,
        "root_cause": root_cause,
        "recommendation": recommendation,
        "retrieved_count": len(retrieved),
        "max_similarity": max_sim,
        "answer_preview": raw_answer[:80],
        "is_grounded": val["is_grounded"],
        "status": val["status"],
    }


def run_debugging_lab(backend: str = "supabase"):
    print("=" * 80)
    print(" DAY 4 HANDS-ON LAB: RETRIEVAL vs PROMPTING ROOT-CAUSE DIAGNOSTIC")
    print(f" Execution Vector Backend: {backend.upper()}")
    print("=" * 80)

    diagnoses = []
    for case in DEBUG_CASES:
        diag = diagnose_case(case, backend=backend)
        diagnoses.append(diag)

        print(f"\n--- [{diag['case_id']}] {diag['title']} ---")
        print(f"  Fault Diagnosis: {diag['fault']}")
        print(f"  Root Cause:      {diag['root_cause']}")
        print(f"  Recommendation:  {diag['recommendation']}")
        print(f"  Retrieved:       {diag['retrieved_count']} chunks (Max Sim: {diag['max_similarity']:.3f})")
        print(f"  Answer Sample:   \"{diag['answer_preview']}...\"")

    print("\n" + "=" * 80)
    print(" SUMMARY MATRIX: RETRIEVAL VS PROMPTING FAILURES")
    print("=" * 80)
    print(f"{'Case ID':<8} | {'Classification':<18} | {'Root Cause Summary':<48}")
    print("-" * 80)
    for d in diagnoses:
        print(f"{d['case_id']:<8} | {d['fault']:<18} | {d['root_cause'][:48]}")
    print("=" * 80)


if __name__ == "__main__":
    b = sys.argv[1] if len(sys.argv) > 1 else "supabase"
    run_debugging_lab(backend=b)
