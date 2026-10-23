import pytest

from src.rag_pipeline import (
    RAGPromptBuilder,
    DeterministicGroundedGenerator,
    AnswerValidator,
    REFUSAL_MESSAGE,
    SYSTEM_PROMPT,
)


def test_rag_prompt_builder_structure():
    chunks = [
        {
            "chunk_index": 0,
            "source": "leave_policy.pdf",
            "page": 1,
            "content": "Full-time employees receive 20 paid vacation days annually.",
        }
    ]
    messages = RAGPromptBuilder.build_messages(
        question="How many vacation days do I get?",
        chunks=chunks,
    )

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "STRICT GROUNDING RULES" in messages[0]["content"]
    assert "[Doc: leave_policy.pdf, Page: 1, Chunk: 0]" in messages[1]["content"]
    assert "[QUESTION]\nHow many vacation days do I get?" in messages[1]["content"]


def test_rag_prompt_builder_empty_chunks():
    messages = RAGPromptBuilder.build_messages("Any query", chunks=[])
    assert "(No relevant document chunks found)" in messages[1]["content"]


def test_deterministic_generator_grounded_answer():
    generator = DeterministicGroundedGenerator()
    chunks = [
        {
            "chunk_index": 0,
            "source": "security.pdf",
            "page": 1,
            "content": "Employees must enable multi-factor authentication. Production credentials must be stored securely.",
        }
    ]

    answer = generator.generate(
        question="What are the rules for production credentials?",
        chunks=chunks,
    )

    assert "Production credentials must be stored securely." in answer
    assert "[Doc: security.pdf, Page: 1, Chunk: 0]" in answer


def test_deterministic_generator_refusal_when_unrelated():
    generator = DeterministicGroundedGenerator()
    chunks = [
        {
            "chunk_index": 0,
            "source": "security.pdf",
            "page": 1,
            "content": "Employees must enable multi-factor authentication.",
        }
    ]

    # Question on topic completely outside the chunk
    answer = generator.generate(
        question="What is the CEO bonus payout schedule?",
        chunks=chunks,
    )

    assert answer == REFUSAL_MESSAGE


def test_answer_validator_refusal():
    val = AnswerValidator.validate(REFUSAL_MESSAGE, retrieved_chunks=[])
    assert val["is_grounded"] is True
    assert val["status"] == "REFUSED"
    assert val["groundedness_score"] == 1.0


def test_answer_validator_detects_hallucinated_citation():
    retrieved = [
        {"chunk_index": 0, "source": "policy.pdf", "page": 1, "content": "Sample content"}
    ]
    # Answer cites Chunk 99 which does NOT exist in retrieved context
    fake_cited_answer = (
        "Employees get 20 days off. [Doc: policy.pdf, Page: 1, Chunk: 99]"
    )

    val = AnswerValidator.validate(fake_cited_answer, retrieved_chunks=retrieved)
    assert val["citations_valid"] is False
    assert val["is_grounded"] is False
    assert any("Hallucinated citations detected" in f for f in val["flags"])


def test_answer_validator_detects_missing_citations():
    retrieved = [
        {"chunk_index": 0, "source": "policy.pdf", "page": 1, "content": "Sample content"}
    ]
    uncited_answer = "Employees get 20 days off with manager approval."

    val = AnswerValidator.validate(uncited_answer, retrieved_chunks=retrieved)
    assert val["citations_valid"] is False
    assert any("Missing inline citations" in f for f in val["flags"])
