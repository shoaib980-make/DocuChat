# Complete Walkthrough: Day 3 & Day 4 Implementation Guide

## Executive Summary

This document explains everything built for **Day 3 (Supabase & pgvector + Vector Store Swapping)** and **Day 4 (RAG Prompting, Citation Enforcement & Hallucination Control)** in the DocuChat project.

---

## 1. Day 3: Supabase & pgvector (Architecture & Code)

### The Problem
Previously:
- Day 1 used **FAISS** in standalone memory (`src/semantic_search.py`).
- Day 3 introduced **Supabase** via direct queries (`src/vector_store.py` and `src/retriever.py`).
- However, there was **no unified interface** allowing seamless swapping between FAISS (local fallback) and Supabase (hosted production), which is a key requirement of the weekly project and curriculum.

### The Solution: `BaseVectorStore` Pattern
In `src/vector_store.py`, we created an Object-Oriented Vector Store abstraction:

```python
class BaseVectorStore(ABC):
    @abstractmethod
    def store(self, chunks: list[dict], embeddings: list[list[float]], document_id: str) -> int: ...

    @abstractmethod
    def search(self, query_embedding: list[float], match_count: int = 5, match_threshold: float = 0.20) -> list[dict]: ...
```

Two concrete implementations share this interface:
1. **`SupabaseVectorStore`**:
   - Executes upserts into the PostgreSQL `document_chunks` table.
   - Calls the `match_document_chunks` RPC function using cosine distance (`<=>`).
2. **`FaissVectorStore`**:
   - Maintains an in-memory or persisted on-disk `faiss.IndexFlatIP` index (`data/faiss_index.bin` + `data/faiss_metadata.json`).
   - Normalizes vectors via L2 norm so inner product matches cosine similarity.
   - Returns identical dictionary output formats.

### Factory Function & Environment Control
```python
def get_vector_store(backend: str | None = None) -> BaseVectorStore:
    selected = (backend or os.getenv("VECTOR_STORE_BACKEND", "supabase")).lower()
    ...
```
You can switch the entire application between Supabase and FAISS simply by changing `VECTOR_STORE_BACKEND=faiss` or passing `backend="faiss"`.

### Database Schema (`sql/schema.sql`)
The PostgreSQL schema enables:
- `CREATE EXTENSION IF NOT EXISTS vector;`
- `document_chunks` table storing `document_id`, `chunk_index`, `content`, `source`, `page`, and `embedding VECTOR(384)`.
- HNSW index: `CREATE INDEX ... USING hnsw (embedding vector_cosine_ops);` for lightning-fast nearest-neighbor search.
- Stored procedure `match_document_chunks` returning chunks where `1 - (embedding <=> query_embedding) >= match_threshold`.

---

## 2. Day 4: RAG Prompting & Hallucination Control

### Core RAG Flow (`src/rag_pipeline.py`)

```
User Query
    │
    ▼
[Retriever] ───► Vector Search in Supabase or FAISS (Threshold Filtering)
    │
    ▼
[Prompt Builder] ──► Isolated [CONTEXT] blocks with [Doc, Page, Chunk] tags
    │
    ▼
[LLM Client] ───► Zero-Temperature Generation (OpenAI / Local Grounded Generator)
    │
    ▼
[Answer Validator] ─► Post-Answer Grounding Check & Citation Audit
    │
    ▼
Validated Grounded Answer with Inline Citations
```

### The 4 Pillars of Hallucination Control
1. **Isolated Context Ingestion:**
   Chunks are wrapped in structured delimiters (`--- START CHUNK [Doc: ..., Page: ..., Chunk: ...] --- ... --- END CHUNK ---`). This eliminates prompt injection and prevents context bleeding.
2. **Explicit Refusal Contract:**
   The prompt explicitly mandates:
   *"If the provided [CONTEXT] does not contain enough information to answer the question, you MUST reply with this exact phrase: 'I cannot answer this question based on the provided documents.'"*
   This eliminates vague guesses or external hallucinations.
3. **Mandatory Inline Citations:**
   Every generated claim must end with `[Doc: <source>, Page: <page>, Chunk: <chunk_index>]`.
4. **Post-Answer Validator (`AnswerValidator`):**
   - Audits generated citations against retrieved chunk IDs.
   - Computes a lexical/semantic groundedness score.
   - Detects fabricated or hallucinated citations.

---

## 3. Hands-On Verification Labs Built

| Lab Script | Description | How to Run |
| :--- | :--- | :--- |
| `scripts/test_vector_store_swap.py` | Indexes into both Supabase & FAISS, runs identical queries, and proves 100% Top-1 chunk alignment. | `.venv\Scripts\python scripts/test_vector_store_swap.py` |
| `scripts/adversarial_testing.py` | Tests 5 unanswerable questions (401k, gym stipend, Tokyo office) and verifies 100% safe refusal. | `.venv\Scripts\python scripts/adversarial_testing.py` |
| `scripts/retrieval_debugging_lab.py` | Diagnoses 5 failure cases to distinguish **Retrieval Faults** (thresholds, corpus gaps) from **Prompting Faults** (missing citations, weak grounding). | `.venv\Scripts\python scripts/retrieval_debugging_lab.py` |
| `scripts/run_rag.py` | CLI tool to ask any question and view generated answer, parsed citations, and validation metrics. | `.venv\Scripts\python scripts/run_rag.py -q "What are the rules for remote work?"` |

---

## 4. Key Learnings & Concepts to Master

1. **Why Normalize Vectors?**
   Cosine similarity is defined as $\frac{A \cdot B}{\|A\| \|B\|}$. When vectors are normalized to unit length ($\|A\| = 1$), cosine similarity simplifies to the dot product ($A \cdot B$). In FAISS, `IndexFlatIP` computes dot products at hardware speed.
2. **Why HNSW over IVFFlat in Supabase?**
   - IVFFlat requires periodic re-training and re-clustering as new chunks are inserted.
   - HNSW constructs a multi-layer hierarchical graph that supports incremental inserts without re-training and delivers higher query recall.
3. **Retrieval Failure vs. Prompting Failure:**
   - **Retrieval Failure:** The correct information was never passed to the LLM (e.g., threshold filtered it out, chunk boundaries severed the sentence, or the document wasn't ingested).
   - **Prompting Failure:** The correct information was present in the context, but the LLM hallucinated, failed to cite, or ignored the facts due to ambiguous instructions or high temperature.
