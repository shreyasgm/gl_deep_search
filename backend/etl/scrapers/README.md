# Growth Lab Deep Search - Scrapers Documentation

This document provides a detailed explanation of the Growth Lab and OpenAlex scrapers implemented in this project.

## Table of Contents
- [Overview](#overview)
- [Growth Lab Scraper](#growth-lab-scraper)
  - [Scraping Process](#growthlab-scraping-process)
  - [Enrichment Process](#growthlab-enrichment-process)
  - [Publication ID Generation](#growthlab-id-generation)
  - [Data Storage](#growthlab-data-storage)
- [OpenAlex Scraper](#openalex-scraper)
  - [API Client](#openalex-api-client)
  - [Fetching Process](#openalex-fetching-process)
  - [Publication ID Generation](#openalex-id-generation)
  - [Data Storage](#openalex-data-storage)
- [Common Features](#common-features)
  - [Content Hash Generation](#content-hash-generation)
  - [Incremental Updates](#incremental-updates)

## Overview

The Growth Lab Deep Search project includes two different scrapers:

1. **Growth Lab Scraper**: Web scraper for publications on the Harvard Growth Lab website
2. **OpenAlex Scraper**: API client for retrieving publications from the OpenAlex academic database

Both scrapers convert source-specific data into standardized `Publication` objects that can be used throughout the system.

## Growth Lab Scraper

The Growth Lab scraper (`growthlab.py`) is responsible for extracting publications from the Harvard Growth Lab website.

### GrowthLab Scraping Process

1. **Configuration Loading**:
   - Loads scraper settings from `config.yaml` (or uses defaults)
   - Configures base URL, scrape delay, concurrency limits, and retry settings

2. **Page Discovery**:
   - Fetches the main publications page to discover the total number of pages
   - Uses BeautifulSoup to parse the pagination information

3. **Concurrent Page Processing**:
   - Creates tasks for all pages (using asyncio)
   - Uses a semaphore to control concurrency and avoid overwhelming the server
   - Implements retry logic with exponential backoff for reliability

4. **Publication Extraction**:
   - For each page, extracts all publication elements using BeautifulSoup
   - Extracts title, authors, year, abstract, URLs, and file links
   - Applies year corrections for publications with known incorrect dates

5. **Robust Error Handling**:
   - Implements comprehensive retry logic with configurable parameters
   - Uses exponential backoff with jitter to prevent the "thundering herd" problem
   - Gracefully handles network issues, timeouts, and server errors

### GrowthLab Enrichment Process

After the basic scraping is complete, the scraper performs an additional enrichment step:

1. **Endnote File Discovery**:
   - For each publication page, checks for available Endnote citation files
   - These files often contain additional metadata not visible on the main page

2. **Endnote Parsing**:
   - Downloads and parses Endnote files to extract structured data
   - Uses the Endnote data to fill in missing information (authors, abstracts, etc.)

3. **Concurrent Enrichment**:
   - Processes enrichment tasks concurrently with controlled concurrency
   - Uses the same robust retry mechanism as the main scraping process

### GrowthLab ID Generation

Publications need stable IDs across runs. The ID generation uses a tiered approach:

1. **URL-based ID** (preferred):
   - Extracts the publication slug from the URL
   - Creates a hash of the slug: `gl_url_{SHA256_HASH[:16]}`

2. **Metadata-based ID** (fallback):
   - Normalizes metadata fields (title, authors, year)
   - Combines normalized fields in a consistent format
   - Creates a hash: `gl_{YEAR}_{SHA256_HASH[:16]}`

3. **Random ID** (last resort):
   - If no stable data is available, creates a random timestamp-based ID
   - Format: `gl_unknown_{SHA256_HASH[:16]}`

The ID generation process includes text normalization (lowercasing, removing punctuation, standardizing whitespace) to ensure stability even with minor text variations.

### GrowthLab Data Storage

The scraper provides methods to:
- Save publications to CSV files
- Load publications from CSV files
- Perform incremental updates by comparing new and existing publications

## OpenAlex Scraper

The OpenAlex scraper (`openalex.py`) is an API client for retrieving publication data from the OpenAlex academic database.

### OpenAlex API Client

1. **Configuration Loading**:
   - Loads API settings from `config.yaml` (or uses defaults)
   - Configures author ID, email for API attribution, and retry settings

2. **API URL Construction**:
   - Builds API URLs with appropriate filters (author ID, publication type)
   - Includes pagination parameters and attribution email

### OpenAlex Fetching Process

1. **Pagination Handling**:
   - Uses OpenAlex cursor-based pagination to retrieve all results
   - Processes pages sequentially due to API cursor requirements

2. **Robust API Interaction**:
   - Implements retry logic for handling rate limits and server errors
   - Uses both per-page retries and overall fetch retries

3. **Abstract Reconstruction**:
   - OpenAlex provides abstracts as "inverted indexes" for copyright reasons
   - The scraper reconstructs full abstract text from these indexes

4. **Publication Processing**:
   - Extracts metadata from API response (title, authors, year, citations, etc.)
   - Converts to standardized Publication objects

### OpenAlex ID Generation

OpenAlex publications already have stable IDs, but the scraper implements a tiered approach for consistency:

1. **OpenAlex ID** (preferred):
   - Uses the native OpenAlex ID: `oa_{OPENALEX_ID}`

2. **DOI-based ID** (first fallback):
   - If a DOI is available, creates a DOI-based ID
   - Format: `oa_doi_{SHA256_HASH[:16]}`

3. **URL-based ID** (second fallback):
   - If a publication URL is available, creates a URL-based ID
   - Format: `oa_url_{SHA256_HASH[:16]}`

4. **Metadata-based ID** (third fallback):
   - Uses normalized metadata fields
   - Format: `oa_{YEAR}_{SHA256_HASH[:16]}`

### OpenAlex Data Storage

Like the Growth Lab scraper, the OpenAlex scraper provides methods to:
- Save publications to CSV files
- Load publications from CSV files
- Perform incremental updates

## Common Features

### Content Hash Generation

Both scrapers implement a content hash generation mechanism:
- Creates a SHA-256 hash of all publication data fields
- Used to detect changes in publications across scraping runs
- Enables efficient incremental updates

### Incremental Updates

Both scrapers support incremental updates:
1. Load existing publications
2. Fetch new publications
3. Compare content hashes to identify changed publications
4. Merge results, preferring new data when changes are detected
5. Retain publications that no longer appear in the source

This process ensures the database stays current while minimizing duplicate data.

## Integration Testing

Both scrapers include comprehensive integration tests that verify ID generation with real data:

### GrowthLab ID Testing
- Verifies that most publications (>50%) use URL-based IDs as expected
- Confirms that ID generation is stable when regenerating IDs from publication data
- Tests all fallback mechanisms to ensure proper ID generation in all cases
- Provides detailed statistics about ID distribution

### OpenAlex ID Testing
- Verifies that most publications (>90%) use native OpenAlex IDs
- Confirms that OpenAlex IDs are properly preserved and prefixed
- Tests all fallback mechanisms (DOI-based, URL-based, metadata-based)
- Provides detailed statistics about ID distribution

Run the integration tests with:
```bash
uv run pytest -m integration
```

## Usage

Both scrapers are typically run through the corresponding scripts in the `scripts` directory:
- `run_growthlab_scraper.py`: Runs the Growth Lab scraper
- `run_openalex_scraper.py`: Runs the OpenAlex scraper

These scripts handle command-line arguments, storage configuration, and logging setup.
