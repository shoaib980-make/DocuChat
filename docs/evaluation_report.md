# DocuChat: 15-Question RAG System Evaluation Report

**Backend Evaluated:** `SUPABASE`  
**Date:** Day 5 Milestone Checkpoint  
**Dataset:** [`data/eval_dataset.json`](file:///c:/Users/Admin/projects/DocuChat/data/eval_dataset.json) (10 Answerable + 5 Adversarial Out-of-Domain)  

---

## 1. Executive Summary & Evaluation Scorecard

| Evaluation Metric | Target Threshold | Measured Score | Status |
| :--- | :---: | :---: | :---: |
| **Retrieval Hit Rate** (Top-3) | >= 80% | **100.0%** | PASS |
| **Groundedness Score** | >= 80% | **100.0%** | PASS |
| **Citation Accuracy** | >= 85% | **100.0%** | PASS |
| **Adversarial Refusal Rate** | 100% | **100.0%** | PASS |
| **Overall Accuracy** | >= 85% | **100.0%** | PASS |

---

## 2. Key Findings & Performance Highlights

1. **Zero Hallucinations on Out-of-Domain Inquiries:**  
   All 5 adversarial questions (401k match, gym stipends, executive contacts, international travel allowances, satellite offices) resulted in strict, standardized refusals. No external hallucinations were produced.

2. **High-Precision Semantic Retrieval:**  
   The `all-MiniLM-L6-v2` 384-dimensional dense retriever achieved a **100.0%** Top-3 hit rate across corporate leave, remote work guidelines, and credential security policies.

3. **Verifiable Inline Citations:**  
   **100.0%** of answerable queries included verified inline citations linking answers directly back to specific document sources, page numbers, and chunk indices.

---

## 3. Detailed Per-Question Evaluation Matrix

| ID | Type | Question Preview | Max Sim | Retrieval Hit | Citation | Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `EVAL-01` | Answerable | How many vacation days do full-time employ... | 0.635 | YES | VALID | `GROUNDED` |
| `EVAL-02` | Answerable | Through which system must leave requests b... | 0.246 | YES | VALID | `GROUNDED` |
| `EVAL-03` | Answerable | Who must approve employee leave requests b... | 0.493 | YES | VALID | `GROUNDED` |
| `EVAL-04` | Answerable | Are employees allowed to request remote wo... | 0.776 | YES | VALID | `GROUNDED` |
| `EVAL-05` | Answerable | What communication standards must remote t... | 0.643 | YES | VALID | `GROUNDED` |
| `EVAL-06` | Answerable | What authentication security mechanism mus... | 0.597 | YES | VALID | `GROUNDED` |
| `EVAL-07` | Answerable | Where are production credentials forbidden... | 0.621 | YES | VALID | `GROUNDED` |
| `EVAL-08` | Answerable | How should production credentials and secr... | 0.601 | YES | VALID | `GROUNDED` |
| `EVAL-09` | Answerable | Does leave approval require HR signoff or ... | 0.440 | YES | VALID | `GROUNDED` |
| `EVAL-10` | Answerable | What are the rules regarding employee avai... | 0.639 | YES | VALID | `GROUNDED` |
| `EVAL-11` | Unanswerable | What is the company 401(k) matching percen... | 0.325 | N/A (Adversarial) | VALID | `REFUSED` |
| `EVAL-12` | Unanswerable | How much annual gym membership reimburseme... | 0.288 | N/A (Adversarial) | VALID | `REFUSED` |
| `EVAL-13` | Unanswerable | Who is the Chief Executive Officer (CEO) a... | 0.326 | N/A (Adversarial) | VALID | `REFUSED` |
| `EVAL-14` | Unanswerable | What is the maximum daily meal allowance d... | 0.324 | N/A (Adversarial) | VALID | `REFUSED` |
| `EVAL-15` | Unanswerable | What are the operating hours of the Tokyo,... | 0.281 | N/A (Adversarial) | VALID | `REFUSED` |

---

## 4. Methodology & Evaluation Criteria

- **Retrieval Relevance (20%):** Evaluates if the top retrieved chunks contain the ground-truth answer.
- **Grounded Answers with Correct Citations (25%):** Ensures claims in answers are directly attested by the source chunks and cite `[Doc: ..., Page: ..., Chunk: ...]`. 
- **Hallucination / No-Answer Handling (15%):** Validates that unanswerable queries trigger the mandatory refusal sentence without speculation.
- **Evaluation Script Reproducibility:** Automatically verifiable by running `.venv\Scripts\python scripts/evaluate_rag.py`.