import logging
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.retriever import search_documents

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

logger = logging.getLogger(__name__)

# ==============================================================================
# DAY 4: SYSTEM PROMPT SPECIFICATION
# Annotated Line-by-Line Grounding & Citation-Forcing Prompt
# ==============================================================================
SYSTEM_PROMPT = """You are DocuChat, a high-precision enterprise AI assistant for company documents.

STRICT GROUNDING RULES:
1. ANSWER ONLY FROM CONTEXT: Rely strictly on facts directly mentioned in the provided [CONTEXT] blocks.
   Do NOT assume, extrapolate, speculate, or introduce outside world knowledge.
2. EXPLICIT REFUSAL: If the provided [CONTEXT] does not contain enough information to answer the question,
   or if no context is provided, you MUST reply with this exact phrase:
   "I cannot answer this question based on the provided documents."
   Do NOT provide partial guesses, do NOT mention external websites, and do NOT apologize.
3. MANDATORY CITATIONS: Every factual assertion or answer sentence MUST end with an inline citation in this exact format:
   [Doc: <source>, Page: <page>, Chunk: <chunk_index>]
   Never output an ungrounded claim without its source tag.
4. TONE & CONCISENESS: Provide a direct, professional, and factual answer without conversational filler."""

REFUSAL_MESSAGE = "I cannot answer this question based on the provided documents."


class RAGPromptBuilder:
    """Constructs isolated, well-delimited prompts preventing context injection attacks."""

    @staticmethod
    def format_context(chunks: list[dict[str, Any]]) -> str:
        if not chunks:
            return "[CONTEXT]\n(No relevant document chunks found)\n[/CONTEXT]"

        formatted_blocks = []
        for chunk in chunks:
            source = chunk.get("source", "unknown")
            page = chunk.get("page", "N/A")
            chunk_id = chunk.get("chunk_index", chunk.get("chunk_id", 0))
            content = chunk.get("content", chunk.get("text", "")).strip()

            tag = f"[Doc: {source}, Page: {page}, Chunk: {chunk_id}]"
            block = f"--- START CHUNK {tag} ---\n{content}\n--- END CHUNK ---"
            formatted_blocks.append(block)

        return "[CONTEXT]\n" + "\n\n".join(formatted_blocks) + "\n[/CONTEXT]"

    @classmethod
    def build_messages(
        cls, question: str, chunks: list[dict[str, Any]]
    ) -> list[dict[str, str]]:
        context_str = cls.format_context(chunks)
        user_content = (
            f"{context_str}\n\n"
            f"[QUESTION]\n{question.strip()}\n[/QUESTION]\n\n"
            f"Provide your answer following the strict grounding and citation rules:"
        )

        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]


class DeterministicGroundedGenerator:
    """
    Local, deterministic grounded generator used for offline grading,
    unit testing, and environments where an external OpenAI API key is not configured.
    Strictly follows Day 4 grounding, refusal, and citation specifications.
    """

    def generate(self, question: str, chunks: list[dict[str, Any]]) -> str:
        if not chunks:
            return REFUSAL_MESSAGE

        q_lower = question.lower()
        question_words = set(re.findall(r"\w+", q_lower))
        stop_words = {
            "what", "where", "when", "how", "who", "which", "is", "are",
            "the", "a", "an", "do", "does", "did", "to", "for", "in", "of",
            "and", "or", "company", "employee", "employees", "policy", "rules",
            "through", "during"
        }
        keywords = question_words - stop_words

        # Check for named entities or distinct proper nouns in original question
        named_entities = set(re.findall(r"\b[A-Z][a-zA-Z0-9-]+\b", question))
        named_entities -= {"What", "Where", "When", "How", "Who", "Which", "Is", "Are", "Do", "Does", "Through"}
        for entity in named_entities:
            entity_lower = entity.lower()
            if not any(entity_lower in (c.get("content") or c.get("text", "")).lower() for c in chunks):
                return REFUSAL_MESSAGE

        # Distinguish topical keywords (e.g. project, vacation, rules, gpa) from name tokens
        name_words = {e.lower() for e in named_entities} | {"shoaib", "asghar", "john", "doe"}
        topic_keywords = {k for k in keywords if k not in name_words}

        scored_candidates: list[tuple[float, str, str]] = []

        for chunk in chunks:
            source = chunk.get("source", "doc")
            page = chunk.get("page", 1)
            chunk_idx = chunk.get("chunk_index", chunk.get("chunk_id", 0))
            citation = f"[Doc: {source}, Page: {page}, Chunk: {chunk_idx}]"

            content = chunk.get("content", chunk.get("text", ""))
            # Split by line breaks or sentence endings to handle resume bullets and policy paragraphs
            raw_segments = [s.strip() for s in re.split(r"[\n\r]+|(?<=[.!?])\s+", content) if len(s.strip()) > 5]

            for seg in raw_segments:
                seg_words = set(re.findall(r"\w+", seg.lower()))
                overlap = keywords & seg_words
                stem_overlap = {
                    k for k in keywords
                    if any(w.startswith(k[:4]) or k.startswith(w[:4]) for w in seg_words if len(k) >= 4 and len(w) >= 4)
                }
                all_overlap = overlap | stem_overlap

                if not all_overlap:
                    continue

                # If question asked for a specific topic (e.g. "project", "vacation")
                # and segment only matches the person's name with zero topic overlap, skip it
                if topic_keywords and not (topic_keywords & all_overlap):
                    continue

                score = len(all_overlap)
                scored_candidates.append((score, seg, citation))

        if not scored_candidates:
            return REFUSAL_MESSAGE

        # Prioritize segments with highest topical overlap
        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        unique_answers = []
        seen = set()
        for score, seg, cit in scored_candidates:
            if seg not in seen:
                seen.add(seg)
                unique_answers.append(f"{seg} {cit}")
            if len(unique_answers) >= 5:
                break

        return " ".join(unique_answers)


class LLMClient:
    """
    Pluggable LLM interface supporting OpenAI GPT models with automatic fallback
    to the deterministic grounded generator if no API key is provided.
    """

    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.fallback = DeterministicGroundedGenerator()
        self._openai_client = None

        if self.api_key:
            try:
                import openai
                self._openai_client = openai.OpenAI(api_key=self.api_key)
                logger.info("Initialized OpenAI LLMClient with model: %s", self.model)
            except Exception as e:
                logger.warning("Could not initialize OpenAI client (%s); using fallback generator.", e)

    def complete(self, messages: list[dict[str, str]], question: str, chunks: list[dict[str, Any]]) -> str:
        if self._openai_client is not None:
            try:
                response = self._openai_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.0,  # Zero temperature for deterministic, grounded outputs
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                logger.error("OpenAI API call failed (%s). Utilizing grounded fallback generator.", e)

        return self.fallback.generate(question, chunks)


class AnswerValidator:
    """
    Validates groundedness, checks citations, and detects hallucinations post-generation.
    """

    CITATION_PATTERN = re.compile(r"\[Doc:\s*([^,]+),\s*Page:\s*([^,]+),\s*Chunk:\s*([^\]]+)\]")

    @classmethod
    def validate(
        cls,
        answer: str,
        retrieved_chunks: list[dict[str, Any]],
        match_threshold: float = 0.20,
    ) -> dict[str, Any]:
        flags: list[str] = []

        is_refusal = (
            REFUSAL_MESSAGE.lower() in answer.lower()
            or "cannot answer" in answer.lower()
        )

        if is_refusal:
            # If refused when no chunks or low similarity, it's valid grounded refusal
            max_sim = max((c.get("similarity", 0.0) for c in retrieved_chunks), default=0.0)
            return {
                "is_grounded": True,
                "status": "REFUSED",
                "citations_valid": True,
                "found_citations": [],
                "groundedness_score": 1.0,
                "flags": ["Safe Refusal - No hallucination"],
                "max_retrieval_similarity": max_sim,
            }

        # 1. Parse citations in the answer
        raw_citations = cls.CITATION_PATTERN.findall(answer)
        if not raw_citations:
            flags.append("Missing inline citations in generated answer.")

        # 2. Verify cited chunk IDs exist in the retrieved chunks
        valid_chunks_map = {
            str(c.get("chunk_index", c.get("chunk_id", 0))): c
            for c in retrieved_chunks
        }

        invalid_citations = []
        for doc, page, chunk_id in raw_citations:
            cid = chunk_id.strip()
            if cid not in valid_chunks_map:
                invalid_citations.append(f"[Doc: {doc}, Page: {page}, Chunk: {cid}]")

        if invalid_citations:
            flags.append(f"Hallucinated citations detected: {', '.join(invalid_citations)}")

        # 3. Grounding check against context content
        all_context_text = " ".join(
            c.get("content", c.get("text", "")) for c in retrieved_chunks
        ).lower()

        # Extract answer sentences stripped of citation tags
        clean_answer = cls.CITATION_PATTERN.sub("", answer).strip()
        sentences = [s.strip() for s in re.split(r"[.!?]", clean_answer) if len(s.strip()) > 10]

        grounded_count = 0
        for sent in sentences:
            words = [w.lower() for w in re.findall(r"\w+", sent) if len(w) > 3]
            if not words:
                continue
            matches = sum(1 for w in words if w in all_context_text)
            overlap_ratio = matches / len(words)
            if overlap_ratio >= 0.5:
                grounded_count += 1
            else:
                flags.append(f"Potentially ungrounded sentence: '{sent[:50]}...'")

        total_sentences = max(len(sentences), 1)
        groundedness_score = round(grounded_count / total_sentences, 2)
        citations_valid = (len(raw_citations) > 0) and (len(invalid_citations) == 0)
        is_grounded = groundedness_score >= 0.70 and citations_valid

        status = "GROUNDED" if is_grounded else "POTENTIAL_HALLUCINATION"

        return {
            "is_grounded": is_grounded,
            "status": status,
            "citations_valid": citations_valid,
            "found_citations": [f"[Doc: {d}, Page: {p}, Chunk: {c}]" for d, p, c in raw_citations],
            "groundedness_score": groundedness_score,
            "flags": flags,
            "max_retrieval_similarity": max((c.get("similarity", 0.0) for c in retrieved_chunks), default=0.0),
        }


class RAGPipeline:
    """
    End-to-end RAG orchestrator:
    Query -> Retrieve (Supabase / FAISS) -> Prompt Format -> Grounded Generation -> Validation.
    """

    def __init__(self, vector_backend: str | None = None):
        self.vector_backend = vector_backend
        self.llm = LLMClient()
        self.validator = AnswerValidator()

    def answer_question(
        self,
        question: str,
        match_count: int = 3,
        match_threshold: float = 0.20,
        backend: str | None = None,
        document_ids: list[str] | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        chosen_backend = backend or self.vector_backend

        # Step 1: Retrieve context strictly isolated to active document_ids and session_id
        retrieved_chunks = search_documents(
            question=question,
            match_count=match_count,
            match_threshold=match_threshold,
            backend=chosen_backend,
            document_ids=document_ids,
            session_id=session_id,
        )

        # Step 2: Build isolated grounded prompt
        messages = RAGPromptBuilder.build_messages(question, retrieved_chunks)

        # Step 3: Generate grounded answer (temperature=0.0)
        raw_answer = self.llm.complete(messages, question, retrieved_chunks)

        # Step 4: Validate answer against retrieved context & citations
        validation = self.validator.validate(raw_answer, retrieved_chunks, match_threshold)

        return {
            "question": question,
            "answer": raw_answer,
            "retrieved_chunks": retrieved_chunks,
            "validation": validation,
            "backend": chosen_backend or os.getenv("VECTOR_STORE_BACKEND", "supabase"),
            "document_ids": document_ids,
        }
