-- ============================================================================
-- Supabase Migration: Initial Publication Tracking Schema
-- ============================================================================
-- This schema creates the publication_tracking table for the Growth Lab
-- Deep Search project. It tracks publications through all ETL pipeline stages:
-- discovery, download, processing (OCR), embedding, and ingestion.
--
-- Key Features:
-- - PostgreSQL-specific (JSONB, TIMESTAMPTZ)
-- - Row Level Security (RLS) for team access control
-- - Auto-updating timestamps
-- - Indexes for query performance
-- - Webhooks trigger for ETL pipeline integration
--
-- Usage:
-- 1. Copy this SQL into Supabase SQL Editor
-- 2. Run the entire script
-- 3. Verify tables and policies in Supabase Dashboard
-- ============================================================================

-- Create publication_tracking table
CREATE TABLE IF NOT EXISTS publication_tracking (
    -- ========================================================================
    -- Core Identification Fields
    -- ========================================================================
    publication_id VARCHAR(255) PRIMARY KEY,
    source_url TEXT NOT NULL,
    title TEXT,
    authors TEXT,
    year INTEGER,
    abstract TEXT,

    -- File URLs stored as JSONB array (GCS paths, not Supabase Storage)
    -- Example: ["gs://bucket/publications/file1.pdf", "gs://bucket/publications/file2.pdf"]
    file_urls JSONB,

    -- Content hash for change detection (SHA-256)
    content_hash VARCHAR(64),

    -- ========================================================================
    -- Timestamps
    -- ========================================================================
    -- When publication was first discovered by scraper
    discovery_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Last updated timestamp (auto-updated on any row change)
    last_updated TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- ========================================================================
    -- Download Stage Tracking
    -- ========================================================================
    download_status VARCHAR(20) NOT NULL DEFAULT 'Pending',
    download_timestamp TIMESTAMPTZ,
    download_attempt_count INTEGER NOT NULL DEFAULT 0,

    -- ========================================================================
    -- Processing Stage Tracking (OCR, text extraction, chunking)
    -- ========================================================================
    processing_status VARCHAR(20) NOT NULL DEFAULT 'Pending',
    processing_timestamp TIMESTAMPTZ,
    processing_attempt_count INTEGER NOT NULL DEFAULT 0,

    -- ========================================================================
    -- Embedding Stage Tracking (vector embeddings generation)
    -- ========================================================================
    embedding_status VARCHAR(20) NOT NULL DEFAULT 'Pending',
    embedding_timestamp TIMESTAMPTZ,
    embedding_attempt_count INTEGER NOT NULL DEFAULT 0,

    -- ========================================================================
    -- Ingestion Stage Tracking (loading into vector database)
    -- ========================================================================
    ingestion_status VARCHAR(20) NOT NULL DEFAULT 'Pending',
    ingestion_timestamp TIMESTAMPTZ,
    ingestion_attempt_count INTEGER NOT NULL DEFAULT 0,

    -- ========================================================================
    -- Error Tracking
    -- ========================================================================
    error_message TEXT,

    -- ========================================================================
    -- Data Validation Constraints
    -- ========================================================================
    CHECK (download_status IN ('Pending', 'In Progress', 'Downloaded', 'Failed')),
    CHECK (processing_status IN ('Pending', 'In Progress', 'Processed', 'OCR_Failed', 'Chunking_Failed', 'Failed')),
    CHECK (embedding_status IN ('Pending', 'In Progress', 'Embedded', 'Failed')),
    CHECK (ingestion_status IN ('Pending', 'In Progress', 'Ingested', 'Failed')),
    CHECK (year IS NULL OR (year >= 1900 AND year <= 2100))
);

-- ============================================================================
-- Indexes for Query Performance
-- ============================================================================
-- These indexes optimize common query patterns:
-- - Filtering by status (for ETL pipeline queries)
-- - Sorting by last_updated (for recent activity)
-- - Searching by title/year (for comms team searches)

CREATE INDEX IF NOT EXISTS idx_pub_download_status
    ON publication_tracking(download_status);

CREATE INDEX IF NOT EXISTS idx_pub_processing_status
    ON publication_tracking(processing_status);

CREATE INDEX IF NOT EXISTS idx_pub_embedding_status
    ON publication_tracking(embedding_status);

CREATE INDEX IF NOT EXISTS idx_pub_ingestion_status
    ON publication_tracking(ingestion_status);

CREATE INDEX IF NOT EXISTS idx_pub_last_updated
    ON publication_tracking(last_updated DESC);

CREATE INDEX IF NOT EXISTS idx_pub_title
    ON publication_tracking(title);

CREATE INDEX IF NOT EXISTS idx_pub_year
    ON publication_tracking(year);

-- Composite index for common ETL pipeline queries
CREATE INDEX IF NOT EXISTS idx_pub_pipeline_status
    ON publication_tracking(download_status, processing_status, embedding_status);

-- ============================================================================
-- Auto-Update Trigger for last_updated Timestamp
-- ============================================================================
-- Automatically updates last_updated field whenever any column changes

CREATE OR REPLACE FUNCTION update_publication_tracking_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.last_updated = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_publication_tracking_timestamp
    BEFORE UPDATE ON publication_tracking
    FOR EACH ROW
    EXECUTE FUNCTION update_publication_tracking_timestamp();

-- ============================================================================
-- Row Level Security (RLS) Configuration
-- ============================================================================
-- Enables fine-grained access control for different user roles

ALTER TABLE publication_tracking ENABLE ROW LEVEL SECURITY;

-- Policy: Service role (ETL pipeline, backend services) has full access
-- Used by: ETL containers, FastAPI service
CREATE POLICY "Service role has full access"
    ON publication_tracking
    FOR ALL
    TO service_role
    USING (true)
    WITH CHECK (true);

-- Policy: Authenticated users (Growth Lab comms team) can read and write
-- Used by: Team members accessing Supabase Studio
CREATE POLICY "Authenticated users can manage publications"
    ON publication_tracking
    FOR ALL
    TO authenticated
    USING (true)
    WITH CHECK (true);

-- Policy: Anonymous users can read (optional, disable if not needed)
-- Uncomment below if you want public read-only access
-- CREATE POLICY "Anonymous users can read"
--     ON publication_tracking
--     FOR SELECT
--     TO anon
--     USING (true);

-- ============================================================================
-- Database Functions for ETL Pipeline Integration
-- ============================================================================

-- Function: Get publications ready for specific ETL stage
-- Usage: SELECT * FROM get_publications_for_stage('processing');
CREATE OR REPLACE FUNCTION get_publications_for_stage(stage_name TEXT)
RETURNS SETOF publication_tracking AS $$
BEGIN
    RETURN QUERY
    CASE stage_name
        WHEN 'download' THEN
            SELECT * FROM publication_tracking
            WHERE download_status = 'Pending'
            ORDER BY discovery_timestamp ASC;

        WHEN 'processing' THEN
            SELECT * FROM publication_tracking
            WHERE download_status = 'Downloaded'
            AND processing_status = 'Pending'
            ORDER BY download_timestamp ASC;

        WHEN 'embedding' THEN
            SELECT * FROM publication_tracking
            WHERE processing_status = 'Processed'
            AND embedding_status = 'Pending'
            ORDER BY processing_timestamp ASC;

        WHEN 'ingestion' THEN
            SELECT * FROM publication_tracking
            WHERE embedding_status = 'Embedded'
            AND ingestion_status = 'Pending'
            ORDER BY embedding_timestamp ASC;

        ELSE
            RAISE EXCEPTION 'Invalid stage name: %. Must be one of: download, processing, embedding, ingestion', stage_name;
    END CASE;
END;
$$ LANGUAGE plpgsql;

-- Function: Update publication status for specific stage
-- Usage: SELECT update_stage_status('pub123', 'download', 'Downloaded');
CREATE OR REPLACE FUNCTION update_stage_status(
    pub_id VARCHAR(255),
    stage_name TEXT,
    new_status TEXT,
    error_msg TEXT DEFAULT NULL
)
RETURNS VOID AS $$
BEGIN
    CASE stage_name
        WHEN 'download' THEN
            UPDATE publication_tracking
            SET download_status = new_status,
                download_timestamp = NOW(),
                download_attempt_count = download_attempt_count + 1,
                error_message = COALESCE(error_msg, error_message)
            WHERE publication_id = pub_id;

        WHEN 'processing' THEN
            UPDATE publication_tracking
            SET processing_status = new_status,
                processing_timestamp = NOW(),
                processing_attempt_count = processing_attempt_count + 1,
                error_message = COALESCE(error_msg, error_message)
            WHERE publication_id = pub_id;

        WHEN 'embedding' THEN
            UPDATE publication_tracking
            SET embedding_status = new_status,
                embedding_timestamp = NOW(),
                embedding_attempt_count = embedding_attempt_count + 1,
                error_message = COALESCE(error_msg, error_message)
            WHERE publication_id = pub_id;

        WHEN 'ingestion' THEN
            UPDATE publication_tracking
            SET ingestion_status = new_status,
                ingestion_timestamp = NOW(),
                ingestion_attempt_count = ingestion_attempt_count + 1,
                error_message = COALESCE(error_msg, error_message)
            WHERE publication_id = pub_id;

        ELSE
            RAISE EXCEPTION 'Invalid stage name: %. Must be one of: download, processing, embedding, ingestion', stage_name;
    END CASE;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- Webhook Trigger for ETL Pipeline Integration
-- ============================================================================
-- This trigger notifies the ETL pipeline when publications are manually
-- updated by the comms team, so reprocessing can be triggered automatically.

-- Create notification function
CREATE OR REPLACE FUNCTION notify_publication_change()
RETURNS TRIGGER AS $$
DECLARE
    notification JSON;
BEGIN
    -- Build notification payload
    notification = json_build_object(
        'operation', TG_OP,
        'record', row_to_json(NEW),
        'old_record', CASE WHEN TG_OP = 'UPDATE' THEN row_to_json(OLD) ELSE NULL END,
        'table', TG_TABLE_NAME,
        'timestamp', NOW()
    );

    -- Send notification via pg_notify channel
    PERFORM pg_notify('publication_changed', notification::text);

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for INSERT, UPDATE operations
CREATE TRIGGER trigger_notify_publication_change
    AFTER INSERT OR UPDATE ON publication_tracking
    FOR EACH ROW
    EXECUTE FUNCTION notify_publication_change();

-- ============================================================================
-- Views for Common Queries
-- ============================================================================

-- View: Publications needing attention (failed or stuck)
CREATE OR REPLACE VIEW publications_needing_attention AS
SELECT
    publication_id,
    title,
    authors,
    year,
    download_status,
    processing_status,
    embedding_status,
    ingestion_status,
    error_message,
    last_updated
FROM publication_tracking
WHERE
    download_status = 'Failed' OR
    processing_status IN ('Failed', 'OCR_Failed', 'Chunking_Failed') OR
    embedding_status = 'Failed' OR
    ingestion_status = 'Failed' OR
    (last_updated < NOW() - INTERVAL '7 days' AND ingestion_status != 'Ingested')
ORDER BY last_updated DESC;

-- View: ETL Pipeline Progress Summary
CREATE OR REPLACE VIEW etl_pipeline_summary AS
SELECT
    COUNT(*) AS total_publications,
    COUNT(*) FILTER (WHERE download_status = 'Downloaded') AS downloaded,
    COUNT(*) FILTER (WHERE processing_status = 'Processed') AS processed,
    COUNT(*) FILTER (WHERE embedding_status = 'Embedded') AS embedded,
    COUNT(*) FILTER (WHERE ingestion_status = 'Ingested') AS ingested,
    COUNT(*) FILTER (WHERE download_status = 'Failed' OR processing_status LIKE '%Failed' OR embedding_status = 'Failed' OR ingestion_status = 'Failed') AS failed,
    COUNT(*) FILTER (WHERE download_status = 'Pending') AS pending_download,
    COUNT(*) FILTER (WHERE download_status = 'Downloaded' AND processing_status = 'Pending') AS pending_processing,
    COUNT(*) FILTER (WHERE processing_status = 'Processed' AND embedding_status = 'Pending') AS pending_embedding,
    COUNT(*) FILTER (WHERE embedding_status = 'Embedded' AND ingestion_status = 'Pending') AS pending_ingestion
FROM publication_tracking;

-- ============================================================================
-- Sample Data for Testing (Optional - Delete in Production)
-- ============================================================================
-- Uncomment below to insert test data for development

/*
INSERT INTO publication_tracking (
    publication_id,
    source_url,
    title,
    authors,
    year,
    abstract,
    file_urls,
    download_status
) VALUES
(
    'test_pub_001',
    'https://growthlab.hks.harvard.edu/publications/test-paper',
    'Test Publication: Economic Complexity and Development',
    'Ricardo Hausmann, Cesar Hidalgo',
    2023,
    'This is a test publication for verifying the system setup.',
    '["gs://gl-deep-search-data/publications/test_pub_001.pdf"]'::jsonb,
    'Pending'
),
(
    'test_pub_002',
    'https://growthlab.hks.harvard.edu/publications/another-test',
    'Test Publication 2: The Atlas of Economic Complexity',
    'Ricardo Hausmann, Cesar Hidalgo, Sebastian Bustos',
    2022,
    'Another test publication to verify multiple entries.',
    '["gs://gl-deep-search-data/publications/test_pub_002.pdf"]'::jsonb,
    'Downloaded'
);
*/

-- ============================================================================
-- Migration Complete
-- ============================================================================
-- Next Steps:
-- 1. Verify tables exist in Supabase Dashboard → Database → Tables
-- 2. Check RLS policies in Database → Policies
-- 3. Test queries in SQL Editor:
--    SELECT * FROM etl_pipeline_summary;
--    SELECT * FROM publications_needing_attention;
-- 4. Invite comms team members in Supabase Dashboard → Authentication → Users
-- 5. Test manual publication entry in Table Editor
-- ============================================================================
