# Test Strategy Audit: Executive Summary

## Overview

Four detailed audits were conducted across the entire codebase:
- [Scrapers & Downloaders](audit_scrapers_downloaders.md) — scrapers (OpenAlex, GrowthLab) + file downloaders
- [Processing Pipeline](audit_processing_pipeline.md) — PDF backends, text chunker, embeddings generator
- [Orchestration & Models](audit_orchestration_models.md) — orchestrator, publication tracker, data models
- [Service Layer & Storage](audit_service_storage.md) — FastAPI app, LangGraph agent, Qdrant, storage layer

This document synthesizes the cross-cutting findings and prioritizes the most impactful changes.

---

## The Fundamental Problem: LLM-Authored Tests

The audits confirmed the core concern: tests written by the same LLM that wrote the production code exhibit several systematic patterns:

### 1. Self-Mocking (Tests That Test Their Own Mocks)
The most egregious pattern found across the codebase. In multiple test files, the test patches **the very function it claims to test**, then asserts on the mock's return value. This provides literally zero coverage while creating a false sense of safety.

**Confirmed instances:**
- `test_gl_file_downloader.py::test_validate_downloaded_file` — patches `_validate_downloaded_file` with a mock, asserts mock returns True
- `test_gl_file_downloader.py::test_download_file_impl` — patches `_download_file_impl` with a mock, asserts mock's return
- `test_pdf_backends.py::TestDoclingBackendUnit::test_extract_success` — patches `DoclingBackend.extract` with a mock, calls mock, verifies mock's return
- `test_growthlab.py::test_publication_enrichment` — manually sets fields on an object and asserts they are set (no production code called)

### 2. Skipped Test Files as Coverage Theater
`test_oa_file_downloader.py` contains **840 lines of tests that are all globally skipped** via `pytestmark = pytest.mark.skip`. This is the most complex downloader in the codebase (~1100 lines of production code) with zero effective coverage.

### 3. Dataclass/Enum Plumbing Tests
Multiple tests simply verify that Python dataclasses store values and enums have the expected `.value` strings. These test Python language features, not business logic:
- `TestComponentStatus.test_component_status_values`
- `TestComponentResult.test_component_result_creation`
- `TestOrchestrationConfig.test_orchestration_config_defaults`
- `TestProcessingPlan.test_processing_plan_creation`

### 4. Over-Mocking That Bypasses All Business Logic
The orchestrator tests mock out every component method (`_run_scraper`, `_run_file_downloader`, etc.), so they only test the for-loop that iterates over components. The actual business logic inside each `_run_*` method — CSV loading, publication filtering, download limits, error handling — is never exercised.

---

## Most Critical Gaps (Ranked by Risk)

### Tier 1: CRITICAL — High-risk code with zero or actively misleading coverage

| # | Gap | Risk | Effort |
|---|-----|------|--------|
| 1 | **`agent.py` — the entire LangGraph search agent has zero tests.** This is the most complex file in the codebase. Contains query analysis, hybrid retrieval with deduplication, LLM-based document grading, synthesis with citation enrichment (off-by-one risk on `source_number - 1`), and retry logic. Pure functions `_build_filter()` and `_should_retry()` are trivially testable. | CRITICAL | Low-Medium |
| 2 | **`main.py` — all 3 FastAPI endpoints are untested.** No TestClient tests. No contract validation between API request/response models and actual behavior. | CRITICAL | Low |
| 3 | **Self-mocking tests create false coverage** — `test_validate_downloaded_file`, `test_download_file_impl`, `test_extract_success` (Docling) all need to be rewritten from scratch. They are worse than having no tests because they provide false confidence. | CRITICAL | Low |
| 4 | **`retry.py` has zero tests** — a core utility used across the entire ETL pipeline for handling transient failures. Exponential backoff, jitter, max-retry logic all untested. | CRITICAL | Low |
| 5 | **`oa_file_downloader.py` — 1100 lines of production code, all tests skipped.** The most complex downloader (OA checking, scidownl fallback, URL resolution, file validation) has zero effective coverage. | CRITICAL | Medium |

### Tier 2: HIGH — Important business logic with inadequate coverage

| # | Gap | Risk | Effort |
|---|-----|------|--------|
| 6 | **Orchestrator critical-failure early-exit is untested.** When the Growth Lab Scraper fails, the pipeline should stop. The test uses the file downloader (non-critical), which does NOT trigger this branch. The most important error-handling behavior is unverified. | HIGH | Low |
| 7 | **`_enforce_token_limits()` / `_force_split_by_tokens()` in text chunker are untested.** These are the last-resort safety net preventing embedding failures from oversized chunks. | HIGH | Low |
| 8 | **Embeddings generator error paths are untested.** `generate_embeddings_for_document()` has 4 distinct failure modes (chunks not found, empty chunks, no embeddings, exception) — none tested. | HIGH | Low |
| 9 | **`ingest_to_qdrant.py` — data ingestion pipeline is untested.** `deterministic_uuid()` generates primary keys. `build_points_for_document()` merges parquet + JSON + DB records into Qdrant points. Bugs corrupt the search index. | HIGH | Medium |
| 10 | **Storage layer (factory, local, cloud) — zero tests.** `StorageFactory` auto-detection could silently route the entire pipeline to the wrong backend. `CloudStorage` bugs could cause data loss. | HIGH | Medium |
| 11 | **`_parse_author_string()` — pure function with complex regex, zero tests.** Handles multiple author name formats. Easy to test, easy to break. | HIGH | Low |
| 12 | **`process_results()` in OpenAlex — core data transformation is untested.** The function that extracts fields from raw API responses is only covered by a skipped test. | HIGH | Low |
| 13 | **GrowthLab `parse_publication()` with new HTML structure is untested.** Only the old `biblio-entry` format has tests. The current live website format (`cp-publication`) could break without detection. | HIGH | Low |

### Tier 3: MEDIUM — Significant gaps but lower blast radius

| # | Gap | Risk | Effort |
|---|-----|------|--------|
| 14 | **`_deep_merge()` has no direct tests.** Backbone of the `--dev` config overlay system. | MEDIUM | Low |
| 15 | **`error_message` persistence bug in `PublicationTracking`.** Status update methods set `error_message` on failure but never clear it on success. Stale errors persist. No test catches this. | MEDIUM | Low |
| 16 | **`_detect_sentences()` not tested with adversarial input.** OCR output without punctuation, abbreviations, decimal numbers, URLs. | MEDIUM | Low |
| 17 | **Frontend `api_client.py` — schema drift risk.** Response parsing manually maps JSON to dataclasses with no test coverage. | MEDIUM | Low |
| 18 | **MarkerBackend has no success-path unit test.** Only the failure path is tested. | MEDIUM | Low |
| 19 | **`_build_url()` in OpenAlex — `lstrip('A')` is likely a bug.** Strips ALL leading A's, so author ID "AAB123" becomes "B123". | MEDIUM | Low |
| 20 | **Duplicate GCS implementations** (`cloud.py` vs `gcs.py`). One may be dead code. | MEDIUM | Low |

---

## What's Actually Good

Not everything is broken. Some tests provide genuine confidence:

1. **Publication tracker DB tests** — exercise real SQLite operations with in-memory databases. Cover the state machine well.
2. **GrowthLab integration tests** — pagination uniqueness, FacetWP detection, real website scraping. These catch real regressions.
3. **`_correct_file_extension()` tests** — test real file system behavior with `tmp_path`. Model unit tests.
4. **ETL pipeline integration tests** (`test_etl_pipeline_integration.py`) — run real PDFs through processing, chunking, and embedding with dev config. Verify actual database state transitions.
5. **Embedding service truncation tests** — test real math (MRL truncation + renormalization), not just mock wiring.
6. **Text chunker test suite** — strongest test file overall. Good coverage of config validation, multiple strategies, error handling, token limits. Main weakness is reliance on clean English text.

---

## What to Remove or Consolidate

| Test | Reason | Action |
|------|--------|--------|
| `test_validate_downloaded_file` (gl_file_downloader) | Self-mocking: tests a mock, not the function | **Rewrite** |
| `test_download_file_impl` (gl_file_downloader) | Self-mocking: tests a mock, not the function | **Rewrite** |
| `test_download_growthlab_files` (gl_file_downloader) | Tests a locally-defined mock function, not production code | **Rewrite** |
| `TestDoclingBackendUnit.test_extract_success` | Self-mocking: patches `extract()` then calls mock | **Rewrite** |
| `test_publication_enrichment` (growthlab) | Manually sets fields, asserts they're set. No production code called | **Remove** |
| `TestComponentStatus.test_component_status_values` | Tests Python enum internals | **Remove** |
| `TestComponentResult.test_component_result_creation` | Tests Python dataclass initialization | **Remove** |
| `TestOrchestrationConfig.test_orchestration_config_defaults` | Tests Python dataclass defaults | **Remove** |
| `TestProcessingPlan.test_processing_plan_creation` | Tests Python dataclass initialization | **Remove** |
| `test_orchestrator_integration` (in test_text_chunker.py) | Tests orchestrator dispatch, not chunking. Wrong file | **Relocate** |
| `test_process_actual_sample_pdfs` (text_chunker) | Hardcoded absolute paths. Always skips in CI | **Remove or fixture-ize** |
| `test_sentence_transformer_save_format` (embeddings) | Redundant with `test_save_embeddings_format` | **Remove** |
| 4x `test_update_*_status` (publication tracking) | Near-identical. Could be 1 parameterized test | **Consolidate** |

---

## Structural Recommendations

### 1. Establish a Testing Taxonomy
Define clear categories and enforce them:
- **Unit tests**: No I/O, no network, no heavy model loading. Mock at the boundary. Run in < 1s each.
- **Integration tests** (`@pytest.mark.integration`): May load models, hit real APIs, use real files. Run separately.
- **Contract tests**: Verify schema compatibility between components (e.g., Qdrant point structure matches between ingestion and search).

### 2. Adopt a "Test the Seams" Strategy
Instead of testing every function in isolation (which leads to over-mocking), focus on the **boundaries**:
- Input parsing (HTML, API responses, CSV, JSON)
- Output serialization (Qdrant points, Parquet files, API responses)
- Error boundaries (what happens when external services fail)
- Pure business logic (filtering, deduplication, merge strategies, URL construction)

### 3. Address the `eval()` Security Vulnerability
Both `openalex.py` and `growthlab.py` use `eval()` on CSV data in `load_from_csv()`. This is arbitrary code execution. Replace with `ast.literal_eval()` or `json.loads()`.

### 4. Consider Property-Based Testing for Parsers
Author name parsing, URL construction, and text chunking are ideal candidates for hypothesis-based property testing. For example: "for any valid author string, `_parse_author_string()` should return a non-empty list of non-empty strings."

### 5. Prioritize Pure Function Tests
The highest-ROI tests are for pure functions that currently have zero coverage:
- `_build_filter()`, `_should_retry()` (agent.py)
- `_parse_author_string()` (growthlab.py)
- `_build_url()` (openalex.py)
- `_deep_merge()` (orchestrator.py)
- `deterministic_uuid()` (ingest_to_qdrant.py)
- `retry_with_backoff()` (retry.py)

These are the easiest to write, fastest to run, and most likely to catch real bugs.

---

## Detailed Audit Documents

For file-by-file analysis with specific line numbers and test names, see:
- [audit_scrapers_downloaders.md](audit_scrapers_downloaders.md)
- [audit_processing_pipeline.md](audit_processing_pipeline.md)
- [audit_orchestration_models.md](audit_orchestration_models.md)
- [audit_service_storage.md](audit_service_storage.md)
