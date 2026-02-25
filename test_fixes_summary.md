# Test Suite Rewrite: Complete Summary

This document describes every change made to the test suite and production code, organized by subsystem. Each section explains **what** was changed and **why** in plain language.

---

## Overview

| Metric | Before | After |
|--------|--------|-------|
| Total tests (unit, non-integration) | ~180 (many self-mocking or skipped) | 243 passing |
| Globally skipped test files | 1 (840 lines in `test_oa_file_downloader.py`) | 0 |
| Self-mocking tests (testing their own mocks) | 4 confirmed | 0 |
| Dataclass/enum plumbing tests | 6 | 0 |
| Service layer test files | 1 (`test_embedding_service.py`) | 6 |
| Production bugs found and fixed | — | 3 |
| Net lines of test code | — | -690 (removed more dead code than added) |
| Coverage | Unmeasured (inflated by self-mocking) | 68% (genuine) |

### Production Code Changes (3 files, 3 bugs fixed)

1. **`openalex.py`**: `lstrip('A')` → `removeprefix('A')` in URL construction
2. **`openalex.py` + `growthlab.py`**: `eval()` → `ast.literal_eval()` in CSV loading (security fix)
3. **`tracking.py`**: `error_message` now cleared on success status transitions (was persisting stale errors)

---

## Part 1: Scrapers & Downloaders

### Tests Deleted

#### `test_publication_enrichment` (from `test_growthlab.py`)
**What it did**: Created a `GrowthLabPublication` object, manually set fields like `abstract` and `endnote_url` on it, then asserted those fields had the values it just set.

**Why it was deleted**: No production code was ever called. This is like testing that Python's `obj.x = 5; assert obj.x == 5` works. It provided zero coverage of the actual enrichment pipeline (`enrich_publication_from_page`, `parse_endnote_content`, etc.) and gave false confidence that enrichment was tested.

#### `test_openalex_real_data_id_generation` (from `test_openalex.py`)
**What it did**: A skipped test that tried to load a CSV from a hardcoded absolute path on a specific developer's machine, then verified model ID generation.

**Why it was deleted**: The test could never run in CI (the file path doesn't exist). Even locally, it used `eval()` on CSV data and only tested Pydantic model behavior, not scraper logic.

### Tests Rewritten

#### `test_validate_downloaded_file` (in `test_gl_file_downloader.py`)
**What the old test did**: Created real PDF, DOC, and DOCX files on disk with correct magic bytes — then immediately threw all that work away by patching `Path.exists`, `Path.stat`, `builtins.open`, AND the `_validate_downloaded_file` function itself with mocks. It then called the mock and asserted the mock returned `{'is_valid': True}`. This was the most egregious self-mocking test in the codebase: it literally tested "if I tell the mock to return True, it returns True."

**What the new test does**: Creates real temporary files with correct magic bytes (PDF starts with `%PDF-`, DOCX with `PK\x03\x04`, DOC with `\xd0\xcf\x11\xe0`), plus invalid and too-small files. Calls the **real** `_validate_downloaded_file()` function against each file. Asserts that valid files pass and invalid/small files are rejected. This exercises the actual magic byte checking, size validation, and format detection logic.

#### `test_download_file_impl` (in `test_gl_file_downloader.py`, renamed to `test_download_file_impl_status_codes`)
**What the old test did**: Patched `_download_file_impl` itself with a mock that returned a pre-built `DownloadResult`, then called the mock and asserted on the mock's return value. Identical to the validation test problem — it tested the mock, not the production code.

**What the new test does**: Mocks the HTTP session (curl_cffi `AsyncSession`) at the boundary — not the function under test. Tests the real `_download_file_impl()` function with four HTTP status code scenarios:
- **200**: Full download succeeds, file is written
- **206**: Partial content (resume scenario), file is appended
- **416**: Range Not Satisfiable (file already complete), returns success without re-downloading
- **404**: Error status code, returns failure with error message

#### `test_download_growthlab_files` (in `test_gl_file_downloader.py`)
**What the old test did**: Defined a completely separate mock function that mimicked the production function's structure, then tested that mock function. The real `download_growthlab_files()` was never imported or called.

**What the new test does**: Creates a real CSV file using the production `save_to_csv()` function, then calls the real `download_growthlab_files()` with a mocked `FileDownloader.download_file` (to avoid actual HTTP calls). Verifies that the production function correctly loads publications from CSV, passes them to the downloader, and returns results.

### Tests Added

#### `_parse_author_string()` — 8 test cases
This is a pure function in `growthlab.py` (line ~36-58) with regex logic for splitting author name strings. It had zero tests despite handling multiple formats. New tests cover:
- Single author: `"Hausmann, R."` → `["Hausmann, R."]`
- Two authors with ampersand: `"Hausmann, R. & Klinger, B."` → `["Hausmann, R.", "Klinger, B."]`
- Multiple with comma+ampersand: `"Hausmann, R., Tyson, L.D. & Zahidi, S."` → three authors
- Empty string → empty list
- Whitespace only → empty list
- `None` input → empty list
- Single full name without comma: `"Ricardo Hausmann"` → `["Ricardo Hausmann"]`
- Parenthetical initials: `"Hausmann (R.) & Klinger (B.)"` → two authors

#### `process_results()` — 5 test cases
Core data transformation in `openalex.py` (line ~149-220) that converts raw OpenAlex API JSON responses into `OpenAlexPublication` objects. Previously untested despite being the most business-critical function. New tests:
- Complete API response with all fields (authorships, primary_location, concepts, etc.)
- Missing `authorships` field → empty authors list
- Missing `primary_location` → no file URLs
- Inverted abstract reconstruction with repeated word positions
- Empty results list → empty publications list

#### `_build_url()` — 5 test cases
URL construction in `openalex.py` (line ~54-65). Tests exposed a real bug:
- Normal author ID `"A12345"` → URL contains `A12345`
- Author ID without leading A → correctly prefixed
- **Bug-exposing test**: Author ID `"AAB123"` — `lstrip('A')` stripped ALL leading A's, producing `"B123"` instead of `"AB123"`. Test failed (RED), production code fixed with `removeprefix('A')`, test passed (GREEN).
- With cursor parameter → `&cursor=xxx` appended
- Without cursor → no cursor in URL

#### `parse_publication()` new HTML format — 1 test
The Growth Lab website changed its HTML structure from `biblio-entry` divs to `cp-publication` divs. The old test only covered the old format. New test creates sample HTML with the new structure (`publication-title`, `publication-authors`, `publication-year`, `publication-excerpt`, `publication-links`) and verifies the parser extracts all fields correctly.

#### `parse_endnote_content()` — 3 test cases
Pure parsing function for Endnote bibliography format. Tests:
- Valid Endnote with `%A` (author), `%T` (title), `%D` (year), `%X` (abstract) tags
- Missing fields → returns partial data without error
- HTML in abstract field → tags are preserved or handled appropriately

#### CSV safety tests (`TestLoadFromCsvSafety`) — 2 tests (one per scraper)
Verify that `load_from_csv()` correctly parses list-like string values from CSV files after the `eval()` → `ast.literal_eval()` security fix.

### `test_oa_file_downloader.py` — Complete Rewrite
**What the old file was**: 840 lines of tests, all globally skipped via `pytestmark = pytest.mark.skip(...)`. Every test class and function provided zero coverage. Many tests were self-mocking (e.g., `test_get_file_path_generation` reconstructed the expected hash using the same logic as production code — a mirror test).

**What was done**: Deleted all 840 lines. Wrote 16 focused new tests from scratch:

- **`TestGetFilePath`** (8 tests): Tests `_get_file_path()` deterministic path generation — publication ID appears in path, OpenAlex directory used, same URL produces same path, different URLs produce different paths, extension preserved, URL without extension handled, query parameters stripped, fallback ID for missing paper_id.
- **`TestValidateDownloadedFile`** (7 tests): Tests file validation with real temp files — valid PDF/DOCX/DOC magic bytes, invalid magic bytes rejected, file too small rejected, nonexistent file handled, unknown content type with valid content.
- **1 integration test**: Resolves a known arXiv open access paper URL and verifies detection.

### Production Bug Fixes

#### `lstrip('A')` → `removeprefix('A')` in `openalex.py`
`lstrip('A')` strips ALL leading A characters from a string, not just a prefix. So author ID `"AAB123"` would become `"B123"` (both A's stripped) instead of the correct `"AB123"` (only the prefix A removed). The TDD cycle was: write test asserting `"AAB123"` produces URL with `"AB123"` → test fails → fix with `removeprefix('A')` → test passes.

#### `eval()` → `ast.literal_eval()` in both scrapers
Both `openalex.py` and `growthlab.py` used `eval()` to parse list-like strings from CSV data (e.g., `"['url1', 'url2']"`). `eval()` executes arbitrary Python code, so a malicious or corrupted CSV could run arbitrary code. Replaced with `ast.literal_eval()`, which only parses Python literal expressions (strings, lists, dicts, numbers) without code execution.

---

## Part 2: Processing Pipeline

### Tests Deleted

#### `test_orchestrator_integration` (from `test_text_chunker.py`)
**What it did**: Patched `_run_text_chunker` with `AsyncMock` in the orchestrator and verified it was called. Also tested dry-run pipeline simulation.

**Why it was deleted**: It tested orchestrator dispatch behavior, not text chunking logic. It was in the wrong test file. The test content was preserved and relocated to `test_orchestrator.py` by Sub-agent 3 (see Part 3).

#### `test_process_actual_sample_pdfs` (from `test_text_chunker.py`)
**What it did**: Tried to process PDF files from hardcoded absolute paths on a specific developer's machine (`/Users/shg309/.../sample1.pdf`).

**Why it was deleted**: Always skipped in CI because the paths don't exist. Not portable across developers. The ETL integration tests (`test_etl_pipeline_integration.py`) already cover real PDF processing with proper fixtures.

#### `test_sentence_transformer_save_format` (from `test_embeddings_generator.py`)
**What it did**: Created `ChunkEmbedding` objects with hardcoded vectors, saved them via `_save_embeddings()`, and verified the Parquet + JSON output format.

**Why it was deleted**: Nearly identical to `test_save_embeddings_format`, which does the same thing. The only difference was the model name in metadata — not worth a separate test.

### Tests Rewritten

#### `TestDoclingBackendUnit.test_extract_success` (in `test_pdf_backends.py`)
**What the old test did**: Patched `DoclingBackend.extract()` itself with a mock that returned a pre-built `ExtractionResult(success=True, text="Extracted text", backend_name="docling")`, then called `backend.extract()` (which was now just the mock), and asserted on the mock's hardcoded return values. This tested absolutely nothing about the real extraction logic.

**What the new test does**: Mocks `_get_converter()` (the internal dependency that creates the Docling converter), NOT `extract()` itself. The mock converter returns a fake `DocumentConversionResult` with `ConversionStatus.SUCCESS`, mock pages (3), tables (1), and pictures (2). The **real** `extract()` method is then called, which exercises:
- Status checking (`ConversionStatus.SUCCESS` vs failure)
- Markdown export from the conversion result
- Page count extraction from document metadata
- Table and picture count from document attributes
- Backend name assignment
- Correct error handling for the success path

### Tests Added

#### `_enforce_token_limits()` and `_force_split_by_tokens()` — 4 tests
These are safety-net functions in the text chunker that force-split any chunk exceeding the embedding model's token limit. They're the last line of defense against embedding API failures from oversized input. Previously untested.
- **Oversized chunk gets split**: A chunk with 2x the max token count is split into multiple chunks, each respecting the limit, with correctly reindexed chunk IDs
- **Chunk at limit not split**: A chunk exactly at the max size passes through unchanged
- **Multiple oversized chunks**: A mix of oversized and normal chunks — only the oversized ones are split
- **Force split produces valid chunks**: Direct test of `_force_split_by_tokens()` on large text, verifying token counts

#### `_detect_sentences()` with adversarial input — 6 tests
The sentence detection regex `([.!?;]+\s+)` is fragile with non-standard text. These tests document its behavior:
- **OCR text without punctuation**: Continuous text with no periods — function returns one big "sentence" (documents the limitation)
- **Abbreviations**: `"Dr. Smith et al. found that..."` — tests that the regex doesn't incorrectly split on abbreviation periods
- **Decimal numbers**: `"The rate was 3.14 percent."` — tests behavior with periods inside numbers
- **URLs**: `"Visit https://example.com. Then continue."` — tests that periods in URLs don't cause incorrect splits
- **Empty string**: Returns empty list without crashing
- **Only punctuation**: Handles gracefully

#### Embeddings generator error paths — 4 tests
`generate_embeddings_for_document()` has 4 distinct failure modes that were all untested. These are the most common production failure modes:
- **Chunks file not found**: Document ID with no corresponding chunks file → returns FAILED status with "not found" error
- **Empty chunks list**: `chunks.json` contains `[]` → returns FAILED with "no chunks" error
- **API returns no embeddings**: Batch generation returns empty list → returns FAILED with "failed to generate" error
- **Unexpected exception**: Batch generation raises RuntimeError → returns FAILED with the exception message

#### Embeddings batching — 1 test
Verifies that 65 texts with `batch_size=32` produces exactly 3 API calls with batch sizes [32, 32, 1], yielding 65 total embeddings. The API mock tracks call counts and batch sizes.

#### MarkerBackend success path — 1 test
Previously only the failure path (model loading raises exception) was tested. The new test mocks `_load_models()` and pre-sets the converter, then calls the real `extract()` → `_extract_v1()` and verifies: success status, backend name "marker", text content from markdown output, metadata extraction, and correct converter invocation.

---

## Part 3: Orchestration & Models

### Tests Deleted

Six dataclass/enum plumbing tests were removed:

#### `TestComponentStatus.test_component_status_values` (from `test_orchestrator.py`)
Asserted `ComponentStatus.PENDING.value == "pending"`, `ComponentStatus.SUCCESS.value == "success"`, etc. This tests that Python's `Enum` class stores `.value` correctly — a language feature, not business logic.

#### `TestComponentResult` — 3 tests (from `test_orchestrator.py`)
- `test_component_result_creation`: Created a `ComponentResult` and asserted every field matched what was passed in. Tests Python's `@dataclass` constructor.
- `test_component_result_with_error`: Same thing with different field values.
- `test_component_result_with_output_files`: Same thing with output_files field.

#### `TestOrchestrationConfig.test_orchestration_config_defaults` (from `test_orchestrator.py`)
Verified that `OrchestrationConfig()` default values matched what was declared in the dataclass definition. Tests Python's default value mechanism.

#### `TestProcessingPlan.test_processing_plan_creation` (from `test_publication_tracking.py`)
Created a `ProcessingPlan` and asserted its fields matched input. Same pattern — tests Python dataclass behavior, not business logic.

### Tests Rewritten

#### `test_run_pipeline_with_component_failure` (in `test_orchestrator.py`)
**What the old test did**: Mocked the file downloader component to fail, then verified the pipeline continued running. This tested a NON-critical failure path (file downloader failure doesn't stop the pipeline by design). The most important error-handling branch — Growth Lab Scraper failure causing an early pipeline exit — was never tested.

**What the new test does**: Two separate tests replace the old one:
1. **`test_non_critical_failure_continues_pipeline`** — Renamed version of the old test, clarifying it tests the non-critical path (file downloader failure doesn't stop pipeline, remaining components still run).
2. **`test_scraper_failure_stops_pipeline`** (new) — Mocks `_run_scraper` to return a FAILED `ComponentResult`. Verifies that the pipeline exits early after the scraper failure (the `if component_name in ["Growth Lab Scraper"]` branch at `orchestrator.py:228-234`) and that subsequent components (file downloader, PDF processor, etc.) are NOT called. This is the critical safety behavior: if we can't get the publication list, there's no point running the rest of the pipeline.

### Tests Added

#### `_deep_merge()` — 7 test cases
Pure function at `orchestrator.py:102-114` that merges YAML config overlays. It's the backbone of the `--dev` flag system (merging `config.dev.yaml` over `config.yaml`). Previously had zero direct tests despite being used in every test run via the `dev_config_path` fixture.
- **Flat merge**: `{"a": 1}` + `{"b": 2}` → `{"a": 1, "b": 2}`
- **Nested merge**: Recursively merges nested dicts
- **Overlay wins**: When both have the same key, overlay's value takes precedence
- **Base preserved**: Keys only in base are retained
- **Empty overlay**: Returns base unchanged
- **Empty base**: Returns overlay
- **Does not mutate inputs**: Verifying the function creates a new dict rather than modifying the inputs

#### `retry_with_backoff()` — 6 tests (NEW FILE: `test_retry.py`)
Core utility at `retry.py:19-68` used throughout the ETL pipeline for handling transient failures. Had ZERO test coverage despite being critical infrastructure.
- **Success on first try**: Function returns immediately, no sleep called
- **Success after retries**: Function fails twice then succeeds — verifies retries happen and the successful result is returned
- **Max retries exceeded**: Function always fails — verifies the exception propagates after exhausting retries
- **Non-retriable exception**: Wrong exception type → propagates immediately without retrying (e.g., `TypeError` when only retrying on `ConnectionError`)
- **Backoff delay capped at max_delay**: Verifies exponential delays (base × 2^attempt) but capped at `max_delay`. Uses deterministic jitter (patched `random.uniform`) to verify exact delay values
- **Args/kwargs forwarded**: Positional and keyword arguments are correctly passed to the target function

#### Scraper failure stops pipeline — 1 test
(Described in the rewrite section above.) Verifies the critical-failure early-exit at `orchestrator.py:228-234`.

#### Text chunker dispatch — 1 test (relocated)
Adapted from the deleted `test_orchestrator_integration` in `test_text_chunker.py`. Now lives in `test_orchestrator.py` where it belongs. Verifies that the orchestrator's component list includes "Text Chunker" and dispatches to `_run_text_chunker`.

#### `error_message` persistence bug — 4 parametrized test cases
Tests the production bug fix (see below). For each of the 4 status update methods:
1. Set status to FAILED with `error="timeout"`
2. Assert `error_message == "timeout"` (error is stored)
3. Set status to a success value with no error
4. Assert `error_message is None` (error is cleared)

Before the fix, step 4 failed because `error_message` was only set when the `error` parameter was truthy, never cleared when it was `None`.

### Tests Consolidated

#### 4× `test_update_*_status` → 1 parametrized test
The old code had four nearly identical tests:
```python
def test_update_download_status(self): ...
def test_update_processing_status(self): ...
def test_update_embedding_status(self): ...
def test_update_ingestion_status(self): ...
```
Each created a `PublicationTracking` object, called the update method, and asserted the status field changed. Consolidated into a single `@pytest.mark.parametrize` test parameterized over `(status_field, update_method, status_value, timestamp_field)`.

### Production Bug Fix

#### `error_message` persistence in `tracking.py`
**The bug**: All four `update_*_status` methods had:
```python
if error:
    self.error_message = error
```
This meant `error_message` was only SET when an error was provided, but never CLEARED on success. So if a publication failed with `error="timeout"`, then later succeeded, the `error_message` field would still contain `"timeout"` — a stale error that could mislead operators or downstream code.

**The fix**: Changed to unconditional assignment:
```python
self.error_message = error
```
Now when `error=None` (the default for success transitions), `error_message` is correctly cleared to `None`. Applied consistently to all 4 update methods.

---

## Part 4: Service Layer & Storage

This area had **catastrophic test coverage gaps** — out of 12 production files, only `embedding_service.py` had any tests. Five new test files were created from scratch.

### `test_agent.py` (NEW FILE — 12 tests)

The search agent (`agent.py`) is the most complex file in the service layer: LangGraph state machine with query analysis, hybrid retrieval, document grading, LLM synthesis, and retry logic. It had zero tests.

#### `_build_filter()` — 5 tests
Static pure function that converts a filter dict to a Qdrant `Filter` object.
- Empty dict → returns `None` (no filter)
- Year only → `Filter` with year `MatchValue` condition
- Document ID only → `Filter` with document_id `MatchValue` condition
- Both → combined `Filter` with both conditions
- Falsy year (`year=0`) → skipped (walrus operator `if year := ...` is falsy for 0)

#### `_should_retry()` — 3 tests
Pure conditional that determines the next graph node after grading.
- No chunks + `retry_count=0` → returns `"retrieve"` (try retrieval again)
- No chunks + `retry_count=2` → returns `"synthesize"` (give up retrying, synthesize with what we have)
- Has chunks → returns `"synthesize"` regardless of retry count

#### Citation enrichment in `_synthesize()` — 3 tests
The synthesis step enriches LLM-generated citations with metadata from the retrieved chunks. The mapping `source_number - 1` is off-by-one-error-prone.
- `source_number=1` correctly maps to `chunks[0]` and enriches empty metadata fields (title, authors, year, url)
- Out-of-bounds `source_number` (e.g., 99 when only 5 chunks exist) → handled gracefully, no `IndexError`
- Pre-populated citation fields from the LLM are NOT overwritten by chunk metadata (only empty fields are enriched)

#### Integration test — 1 test (marked `@pytest.mark.integration`)
End-to-end `SearchAgent.run()` with mocked Qdrant and embedding services but real LangGraph graph execution. Verifies graph nodes connect correctly and execute in order (analyze_query → retrieve → grade_documents → synthesize).

### `test_main.py` (NEW FILE — 11 tests)

The FastAPI application (`main.py`) with 3 endpoints had zero test coverage.

#### `_build_qdrant_filter()` — 5 tests
Helper that converts `SearchFilters` to Qdrant `Filter` (note: duplicates `SearchAgent._build_filter()` — flagged for consolidation).
- `None` filters → `None`
- Empty `SearchFilters` → `None`
- Year only → correct Filter
- Document ID only → correct Filter
- Both → combined Filter

#### `_scored_point_to_chunk_result()` — 3 tests
Maps a Qdrant `ScoredPoint` to the API's `ChunkResult` response model.
- Full payload with all fields → all fields mapped correctly
- Missing optional fields → defaults used
- `None` payload → all defaults

#### FastAPI endpoints — 3 tests (using `TestClient`)
- **`GET /health`** — Returns 200 with expected structure (qdrant_connected, collection info, etc.)
- **`POST /search/chunks`** — Valid request with mocked QdrantService and EmbeddingService → returns correct `SearchResponse` schema with results
- **`POST /search`** — Valid request with mocked agent → returns correct `AgentSearchResponse` with answer, citations, and metadata

### `test_ingest_to_qdrant.py` (NEW FILE — 10 tests)

The data ingestion pipeline (`ingest_to_qdrant.py`) converts processed documents into Qdrant vector points. Had zero tests despite being the bridge between ETL and search.

#### `deterministic_uuid()` — 4 tests
Primary key generation for Qdrant points.
- Same input → same UUID (deterministic)
- Different inputs → different UUIDs (collision-free)
- Output is valid UUID format
- Confirms UUID version 5 (namespace-based)

#### `build_points_for_document()` — 6 tests
Complex function that merges parquet embeddings, JSON metadata, tracker records, and BM25 sparse vectors into Qdrant `PointStruct` objects.
- Correct number of points built (matches rows in parquet)
- Payload contains all expected fields (text_content, chunk_id, document_id, title, authors, year, url, chunk_index, total_chunks)
- Points have both `dense` and `bm25` named vectors
- Point IDs are deterministic UUIDs (based on chunk_id)
- Missing publication metadata → defaults to `None`/empty
- Sparse model called with correct text content from chunks

### `test_storage.py` (NEW FILE — 24 tests)

The storage abstraction layer (factory, local, cloud) had zero tests despite routing ALL file operations in the pipeline.

#### `StorageFactory.create_storage()` — 5 tests
- Explicit `"local"` → returns `LocalStorage` instance
- Explicit `"cloud"` → returns `CloudStorage` instance (mocked GCS)
- `CLOUD_ENVIRONMENT` env var set → auto-detects cloud
- No env var → auto-detects local
- Missing config file → uses fallback defaults (still works)

#### `LocalStorage` — 10 tests (using `tmp_path`, no mocks)
All tested against real filesystem:
- `get_path()` returns correct joined path
- `exists()` true for existing file, false for missing
- `glob()` finds matching files and returns relative paths
- `ensure_dir()` creates nested directories
- `upload()` is a no-op (local storage)
- `download()` returns the local path (no-op for local)
- `list_files()` with and without glob pattern

#### `CloudStorage` — 9 tests (mocked GCS client)
- `download()` three code paths: single file, directory prefix, nothing found
- `exists()` two-phase check: exact blob match, directory prefix match, nothing
- `glob()` with prefix optimization (extracts static prefix from glob pattern for efficient GCS listing)
- `get_path()` returns local cache path
- `ensure_dir()` creates local directory for cache

### `test_embedding_service.py` — No Changes
All 4 existing tests are well-structured (they mock at the API boundary and test real math). Kept as-is.

---

## Cross-Cutting Observations

### Patterns eliminated
1. **Self-mocking**: Patching the function under test and asserting on the mock's return value. Found in 4 tests, all rewritten.
2. **Dataclass plumbing**: Testing that Python dataclasses store values. Found in 6 tests, all deleted.
3. **Globally skipped files**: 840 lines of dead test code in `test_oa_file_downloader.py`. Replaced with 16 focused tests.
4. **Mirror tests**: Tests that reconstructed expected values using the same logic as production code. Found in OA file downloader, eliminated in rewrite.

### Where mocks are now used (correctly)
- HTTP sessions (curl_cffi, aiohttp) — mocked to avoid network calls
- LLM APIs (OpenAI, Anthropic) — mocked to avoid cost and flakiness
- GCS client — mocked to avoid cloud dependency
- Qdrant client — mocked for unit tests
- `asyncio.sleep` — mocked in retry tests to avoid delays
- File I/O — real `tmp_path` used wherever possible, mocked only for cloud storage

### What's still not covered (known gaps, lower priority)
- `profiling.py` — timing/resource utilities (low risk)
- `gcs.py` — possibly dead code alongside `cloud.py` (needs investigation)
- `database.py` — ORM wrapper with module-level side effects (medium risk)
- `frontend/api_client.py` — schema drift risk but low blast radius
- `qdrant_service.py` — vector search operations (tested indirectly through agent tests)
- Individual `_run_*` methods inside orchestrator — only tested via mock dispatch
