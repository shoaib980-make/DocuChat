-- ==============================================================================
-- Day 3: Supabase & pgvector Schema Setup
-- DocuChat - Enterprise Document Q&A System
-- ==============================================================================

-- 1. Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Create table for storing document chunks, embeddings, and metadata
CREATE TABLE IF NOT EXISTS document_chunks (
    id BIGSERIAL PRIMARY KEY,
    document_id TEXT NOT NULL,
    chunk_index INT NOT NULL,
    content TEXT NOT NULL,
    source TEXT NOT NULL,
    page INT,
    embedding VECTOR(384) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT uq_document_chunk UNIQUE (document_id, chunk_index)
);

-- 3. Create HNSW index for fast cosine distance similarity searches
-- (HNSW provides superior query speed and recall compared to IVFFlat for production)
CREATE INDEX IF NOT EXISTS document_chunks_embedding_hnsw_idx 
ON document_chunks 
USING hnsw (embedding vector_cosine_ops);

-- Optional secondary index on metadata for fast filtering by document or source
CREATE INDEX IF NOT EXISTS idx_document_chunks_document_id 
ON document_chunks (document_id);

CREATE INDEX IF NOT EXISTS idx_document_chunks_source 
ON document_chunks (source);

-- 4. Stored Procedure for Cosine Similarity Search
-- Matches query vector against chunk embeddings and returns chunks above threshold
CREATE OR REPLACE FUNCTION match_document_chunks(
    query_embedding VECTOR(384),
    match_threshold FLOAT DEFAULT 0.20,
    match_count INT DEFAULT 5,
    filter_document_ids TEXT[] DEFAULT NULL
)
RETURNS TABLE (
    id BIGINT,
    document_id TEXT,
    chunk_index INT,
    content TEXT,
    source TEXT,
    page INT,
    similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        dc.id,
        dc.document_id,
        dc.chunk_index,
        dc.content,
        dc.source,
        dc.page,
        1 - (dc.embedding <=> query_embedding) AS similarity
    FROM document_chunks dc
    WHERE 1 - (dc.embedding <=> query_embedding) >= match_threshold
      AND (filter_document_ids IS NULL OR dc.document_id = ANY(filter_document_ids) OR dc.source = ANY(filter_document_ids))
    ORDER BY dc.embedding <=> query_embedding ASC
    LIMIT match_count;
END;
$$;
