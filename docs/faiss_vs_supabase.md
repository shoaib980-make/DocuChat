# FAISS vs Supabase pgvector: Architectural & Decision Criteria

When architecting production RAG applications for DevOrbis knowledge base clients, choosing the appropriate vector storage backend is one of the most critical infrastructure decisions.

---

## 1. Core Architecture Comparison

| Feature Dimension | FAISS (Facebook AI Similarity Search) | Supabase pgvector (PostgreSQL) |
| :--- | :--- | :--- |
| **Type** | C++ Vector Similarity Library (with Python bindings) | Relational Database Extension (`vector` on PostgreSQL) |
| **Hosting Model** | Embedded / Local (In-process memory or local disk file) | Managed Cloud Service or Self-Hosted PostgreSQL instance |
| **Data Storage** | Raw numeric arrays in RAM / mmap binary index | Tabular rows alongside transactional relational data |
| **Metadata Handling** | External (separate JSON, SQLite, or dict mapping) | Native SQL columns (source, page, author, created_at) |
| **Search Index Types** | `IndexFlatIP`, `IndexIVFFlat`, `IndexHNSW`, `IndexPQ` | `hnsw` (cosine, L2, dot product), `ivfflat` |
| **CRUD & Updates** | Expensive (rebuilding or manual vector IDs) | Native SQL `INSERT`, `UPDATE`, `UPSERT`, `DELETE` |
| **Concurrent Access** | Single-node / requires custom service wrapper | Production multi-user concurrency with ACID guarantees |

---

## 2. Deep Dive: Decision Criteria

### When to Choose FAISS (Local / Embedded)
1. **Ultra-Low Latency / High QPS:** 
   FAISS runs directly in RAM inside the Python runtime, eliminating network round-trip latency. Search operations execute in single-digit milliseconds or microseconds.
2. **Offline & Air-Gapped Environments:**
   Edge applications, local CLI tools, desktop apps, or sensitive internal networks where outbound database traffic is restricted.
3. **Static Corpora / Read-Heavy Batches:**
   Large historical archives that are indexed once and queried millions of times without frequent insertions or deletions.
4. **Development & Fallback:**
   Local unit testing, offline CI/CD pipelines, and zero-dependency local fallbacks when cloud infrastructure is unavailable.

### When to Choose Supabase pgvector (Hosted / Production)
1. **Relational + Vector Co-location:**
   Chunks reside in the same database as users, organizations, permissions, document upload records, and audit logs.
2. **Metadata Filtering (Hybrid Queries):**
   Execute SQL queries that combine vector similarity with complex filters in a single atomic pass:
   ```sql
   SELECT * FROM document_chunks 
   WHERE organization_id = 'org_123' 
     AND source = 'q3_financials.pdf'
   ORDER BY embedding <=> query_vector 
   LIMIT 5;
   ```
3. **Dynamic Documents (Frequent Updates & Deletions):**
   When users upload, edit, or delete documents, removing chunks is as simple as `DELETE FROM document_chunks WHERE document_id = 'doc_456'`.
4. **Enterprise Multi-Tenancy & Row-Level Security (RLS):**
   Enforce document isolation between different client tenants using native Postgres RLS policies.

---

## 3. Decision Matrix for Client Projects

```
                                  Client Project Vector Store Decision Flow
                                                     │
                                   Does the project require cloud multi-user
                                   access, user permissions, or live document
                                   upload/delete capabilities?
                                                     │
                                    ┌────────────────┴────────────────┐
                                   YES                                NO
                                    │                                 │
                         Use Supabase pgvector                 Are there strict local-only,
                         (Managed PostgreSQL +                offline, or sub-5ms latency
                          pgvector + HNSW index)              requirements?
                                                                      │
                                                           ┌──────────┴──────────┐
                                                          YES                    NO
                                                           │                     │
                                                    Use FAISS             Use Supabase
                                                    (Local Index)         (Default Standard)
```

---

## 4. Why We Implemented Both in DocuChat
In `src/vector_store.py`, DocuChat implements `BaseVectorStore` with two interchangeable backends:
- **`SupabaseVectorStore`**: The primary production engine for hosted, persistent storage.
- **`FaissVectorStore`**: The seamless local fallback ensuring that the system functions offline, in unit tests, and during development with zero cloud dependencies.
