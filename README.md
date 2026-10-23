# DocuChat: Enterprise Document RAG with Citations & Hallucination Control

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Supabase](https://img.shields.io/badge/Supabase-pgvector-3ECF8E?logo=supabase&logoColor=white)](https://supabase.com/)
[![FAISS](https://img.shields.io/badge/FAISS-CPU-0081CB?logo=meta&logoColor=white)](https://github.com/facebookresearch/faiss)
[![Tests](https://img.shields.io/badge/Tests-25%20Passed-10B981)](#-automated-tests)
[![Evaluation](https://img.shields.io/badge/Eval%20Score-100%25-6366F1)](#-evaluation-scorecard)

**DocuChat** is an end-to-end production Retrieval-Augmented Generation (RAG) system built to chat with company policies, compliance documents, and corporate knowledge bases. It features grounded prompt engineering, verifiable inline source citations, strict refusal on out-of-domain queries, and seamless swapping between **Supabase pgvector** (hosted cloud) and **FAISS** (local fallback).

---

## 🌟 Key Features

1. **Multi-Format Document Ingestion:**
   - Ingests **PDF** and **DOCX** files with page-number tracking and metadata preservation.
   - Whitespace-aware chunking with sliding overlap (default: 200 characters, 40-character overlap) avoiding severed word boundaries.
2. **Dual Vector Database Architecture (`BaseVectorStore`):**
   - **Primary Backend:** Hosted **Supabase PostgreSQL + pgvector** with an optimized HNSW index (`vector_cosine_ops`).
   - **Local Fallback Backend:** High-speed **FAISS** `IndexFlatIP` with disk persistence (`data/faiss_index.bin`).
   - Swap backends at runtime via the web UI toggle, query parameters, or `VECTOR_STORE_BACKEND`.
3. **Grounded Prompt Engineering & Hallucination Control:**
   - Isolated delimited context injection (`--- START CHUNK [Doc: ..., Page: ..., Chunk: ...] ---`).
   - Zero temperature (`temperature=0.0`) for deterministic, grounded outputs.
   - Mandatory refusal behavior: Outputs `"I cannot answer this question based on the provided documents."` for ungrounded queries.
   - Mandatory inline citation forcing: `[Doc: <source>, Page: <page>, Chunk: <index>]`.
4. **Post-Answer Audit & Validation (`AnswerValidator`):**
   - Programmatically parses citations and verifies they exist in retrieved chunks (detects citation fabrication).
   - Computes a groundedness score (0.0 to 1.0) and status flag (`GROUNDED`, `REFUSED`, or `POTENTIAL_HALLUCINATION`).
5. **Interactive Web Chat Application:**
   - Built-in single-page application with dark glassmorphism styling.
   - Document upload dropzone, interactive clickable citation tags, source drawer, and quick test chips.
6. **Empirical 15-Question Evaluation Suite:**
   - Automated evaluation script testing 10 in-domain answerable questions + 5 adversarial unanswerable questions.

---

## 🏗️ System Architecture

```
User Query / Document Upload (.pdf, .docx)
               │
               ▼
       [FastAPI Application] (src/api.py)
        ├── POST /api/upload ──► Document Processor (src/document_processor.py)
        │                         └── Chunking (200 chars / 40 overlap)
        │                         └── Embeddings (all-MiniLM-L6-v2)
        │
        └── POST /api/chat   ──► Retriever (src/retriever.py)
                                  │
                                  ├──► [Supabase pgvector] (Hosted RPC: match_document_chunks)
                                  └──► [FAISS IndexFlatIP] (Local Fallback Index)
                                  │
                                  ▼
                          [RAG Pipeline] (src/rag_pipeline.py)
                           ├── RAGPromptBuilder (Delimited Context + Strict Rules)
                           ├── LLMClient (OpenAI gpt-4o-mini or Local Deterministic Generator)
                           └── AnswerValidator (Citation Audit & Groundedness Verification)
                                  │
                                  ▼
                         Grounded Answer + Citations + Source Inspector
```

---

## 📊 Evaluation Scorecard (15-Question Test Set)

Evaluated on the benchmark dataset [`data/eval_dataset.json`](file:///c:/Users/Admin/projects/DocuChat/data/eval_dataset.json):

| Metric | Target | Measured Score | Status |
| :--- | :---: | :---: | :---: |
| **Retrieval Hit Rate (Top-3)** | >= 80% | **100.0%** | ✅ PASS |
| **Groundedness Score** | >= 80% | **100.0%** | ✅ PASS |
| **Citation Accuracy** | >= 85% | **100.0%** | ✅ PASS |
| **Adversarial Refusal Rate** | 100% | **100.0%** | ✅ PASS |
| **Overall Accuracy** | >= 85% | **100.0%** | ✅ PASS |

Full report available in [`docs/evaluation_report.md`](file:///c:/Users/Admin/projects/DocuChat/docs/evaluation_report.md).

---

## 🚀 Quickstart Guide

### 1. Installation

Clone the repository and install dependencies:
```bash
git clone https://github.com/shoaib980-make/DocuChat.git
cd DocuChat

# Create virtual environment
python -m venv .venv
.\.venv\Scripts\activate   # On Windows
# source .venv/bin/activate  # On Linux/macOS

# Install requirements
pip install -r requirements.txt
```

### 2. Environment Configuration

Copy the example environment file:
```bash
cp .env.example .env
```
Fill in your Supabase credentials:
```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_PUBLISHABLE_KEY=your-anon-key
SUPABASE_SECRET_KEY=your-service-role-secret-key
VECTOR_STORE_BACKEND=supabase

# Optional (Defaults to local grounded generator if omitted)
OPENAI_API_KEY=your-openai-api-key
OPENAI_MODEL=gpt-4o-mini
```

### 3. Database Schema Setup

Run the SQL script located in [`sql/schema.sql`](file:///c:/Users/Admin/projects/DocuChat/sql/schema.sql) in your Supabase SQL Editor to enable `pgvector`, create the `document_chunks` table, configure the HNSW cosine index, and create the `match_document_chunks` RPC function.

### 4. Index Sample Document
```bash
python scripts/index_sample_document.py
```

### 5. Launch the Web Application
```bash
uvicorn src.api:app --host 127.0.0.1 --port 8000 --reload
```
Open your browser at `http://127.0.0.1:8000` to interact with DocuChat!

---

## 🧪 Automated Tests & Labs

Execute all unit tests and hands-on diagnostic labs:

```bash
# Run all 25 unit tests
pytest

# Test swapping FAISS and Supabase pgvector behind identical interface
python scripts/test_vector_store_swap.py

# Run adversarial test suite (5 unanswerable questions)
python scripts/adversarial_testing.py

# Run retrieval vs prompting root-cause diagnostic lab
python scripts/retrieval_debugging_lab.py

# Run the 15-question evaluation suite
python scripts/evaluate_rag.py
```

---

## 📁 Repository Structure

```
DocuChat/
├── data/
│   ├── eval_dataset.json          # 15-question evaluation benchmark
│   ├── sample_company_policy.pdf  # Sample corporate policy document
│   ├── sample_company_policy.docx # Word doc version
│   ├── snippets.txt               # Day 1 test snippets
│   ├── faiss_index.bin            # Local persistent FAISS vector index
│   └── faiss_metadata.json        # FAISS chunk metadata store
├── docs/
│   ├── chunking_strategy.md       # 1-page write-up of chunking trade-offs
│   ├── faiss_vs_supabase.md       # Vector database architectural decision matrix
│   ├── rag_prompt_spec.md         # Line-by-line annotated RAG system prompt
│   ├── evaluation_report.md       # Automated 15-question evaluation report
│   └── day3_day4_summary.md       # Technical walkthrough of Day 3 and 4
├── sql/
│   └── schema.sql                 # Supabase pgvector schema, HNSW index & RPC
├── src/
│   ├── api.py                     # FastAPI application & endpoints
│   ├── document_processor.py      # PDF/DOCX parser & boundary-aware chunker
│   ├── rag_pipeline.py            # RAG prompt builder, LLM client & validator
│   ├── retriever.py               # Semantic search query interface
│   ├── vector_store.py            # BaseVectorStore, Supabase & FAISS stores
│   ├── semantic_search.py         # Day 1 baseline FAISS search
│   └── static/
│       └── index.html             # DocuChat single-page chat UI
├── scripts/
│   ├── adversarial_testing.py     # 5 adversarial unanswerable tests
│   ├── compare_chunking.py        # Day 2 chunking strategy experiment
│   ├── evaluate_rag.py            # Automated 15-question evaluator
│   ├── index_sample_document.py   # Document ingestion script
│   ├── retrieval_debugging_lab.py # Root-cause diagnostic lab
│   ├── run_rag.py                 # CLI query runner
│   └── test_vector_store_swap.py  # Backend interchangeability test
├── tests/
│   ├── test_api.py                # FastAPI endpoint tests
│   ├── test_document_processor.py # Chunker and parser tests
│   ├── test_rag_pipeline.py       # Prompt, generator and validator tests
│   ├── test_semantic_search.py    # Embedding and index tests
│   └── test_vector_store.py       # FAISS and Supabase store tests
├── .env.example                   # Environment configuration template
├── requirements.txt               # Project dependencies
└── README.md                      # Documentation & instructions
```

---

## 📖 Deliverables Documentation
- [RAG Prompt Specification (Line-by-Line Annotated)](docs/rag_prompt_spec.md)
- [Chunking Strategy & Trade-Off Analysis](docs/chunking_strategy.md)
- [FAISS vs. Supabase pgvector Decision Matrix](docs/faiss_vs_supabase.md)
- [15-Question Evaluation Report](docs/evaluation_report.md)
- [Day 3 & Day 4 Summary](docs/day3_day4_summary.md)
