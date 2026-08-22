-- ETL Tracking Metadata Schema
-- This schema defines the structure for tracking publications through the ETL pipeline

CREATE TABLE IF NOT EXISTS publication_tracking (
    -- Core identification fields
    publication_id VARCHAR(255) PRIMARY KEY,  -- Unique identifier for the publication (e.g., DOI, internal hash)
    source_url TEXT NOT NULL,                 -- URL where the publication was found
    title TEXT,                               -- Extracted title if available
    authors TEXT,                             -- Publication authors
    year INTEGER,                             -- Publication year
    abstract TEXT,                            -- Publication abstract
    file_urls TEXT,                           -- JSON array of associated file URLs
    content_hash VARCHAR(64),                 -- Hash of content to detect changes

    -- Discovery stage
    discovery_timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,  -- When publication was first identified

    -- Download stage
    download_status VARCHAR(20) NOT NULL DEFAULT 'Pending',  -- Pending, Downloaded, Failed
    download_timestamp TIMESTAMP,                            -- When download was completed or failed
    download_attempt_count INTEGER NOT NULL DEFAULT 0,       -- Number of download attempts

    -- Processing stage
    processing_status VARCHAR(20) NOT NULL DEFAULT 'Pending',  -- Pending, Processed, OCR_Failed, Chunking_Failed
    processing_timestamp TIMESTAMP,                            -- When processing was completed or failed
    processing_attempt_count INTEGER NOT NULL DEFAULT 0,       -- Number of processing attempts

    -- Embedding stage
    embedding_status VARCHAR(20) NOT NULL DEFAULT 'Pending',  -- Pending, Embedded, Failed
    embedding_timestamp TIMESTAMP,                            -- When embedding was completed or failed
    embedding_attempt_count INTEGER NOT NULL DEFAULT 0,       -- Number of embedding attempts

    -- Ingestion stage
    ingestion_status VARCHAR(20) NOT NULL DEFAULT 'Pending',  -- Pending, Ingested, Failed
    ingestion_timestamp TIMESTAMP,                            -- When ingestion was completed or failed
    ingestion_attempt_count INTEGER NOT NULL DEFAULT 0,       -- Number of ingestion attempts

    -- Tagging stage
    tagging_status VARCHAR(20) NOT NULL DEFAULT 'Pending',  -- Pending, Tagged, Failed
    tagging_timestamp TIMESTAMP,                            -- When tagging was completed or failed
    tagging_attempt_count INTEGER NOT NULL DEFAULT 0,       -- Number of tagging attempts
    tags TEXT,                                              -- JSON dict with taxonomy keys (regions, topics, etc.)

    -- General tracking
    last_updated TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,  -- Timestamp of last modification
    error_message TEXT,                                         -- Most recent error message (if any)

    -- Constraints
    CHECK (download_status IN ('Pending', 'In Progress', 'Downloaded', 'Failed')),
    CHECK (processing_status IN ('Pending', 'In Progress', 'Processed', 'OCR_Failed', 'Chunking_Failed', 'Failed')),
    CHECK (embedding_status IN ('Pending', 'In Progress', 'Embedded', 'Failed')),
    CHECK (ingestion_status IN ('Pending', 'In Progress', 'Ingested', 'Failed')),
    CHECK (tagging_status IN ('Pending', 'In Progress', 'Tagged', 'Failed'))
);

-- Create indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_pub_download_status ON publication_tracking(download_status);
CREATE INDEX IF NOT EXISTS idx_pub_processing_status ON publication_tracking(processing_status);
CREATE INDEX IF NOT EXISTS idx_pub_embedding_status ON publication_tracking(embedding_status);
CREATE INDEX IF NOT EXISTS idx_pub_ingestion_status ON publication_tracking(ingestion_status);
CREATE INDEX IF NOT EXISTS idx_pub_tagging_status ON publication_tracking(tagging_status);
CREATE INDEX IF NOT EXISTS idx_pub_last_updated ON publication_tracking(last_updated);
CREATE INDEX IF NOT EXISTS idx_pub_title ON publication_tracking(title);
CREATE INDEX IF NOT EXISTS idx_pub_year ON publication_tracking(year);
