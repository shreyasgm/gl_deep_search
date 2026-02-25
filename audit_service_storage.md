# Test Audit: Service Layer & Storage

## Executive Summary

The service and storage subsystem has **catastrophic test coverage gaps**. Out of 12 production files audited, only 1 has any test coverage at all (`embedding_service.py`), and even that coverage is narrow -- limited to truncation logic and initialization wiring. The entire FastAPI application, LangGraph agent, Qdrant service, storage abstraction layer, database module, ingestion script, and frontend have zero test coverage. The existing tests are reasonable in isolation but insufficient to catch regressions in the system's most critical code paths: query routing, hybrid search, document grading, LLM synthesis, and data persistence.

## File-by-File Analysis

---

### `backend/service/embedding_service.py` <-> `backend/tests/service/test_embedding_service.py`

#### Coverage
- **Tested:**
  - `embed_query()` MRL truncation when API returns oversized vectors (4096 -> 1024)
  - `embed_query()` passthrough when dimensions already match
  - `embed_query()` verifies correct model ID is passed to API
  - `initialize()` verifies AsyncOpenAI is created with correct base_url and api_key
- **UNTESTED:**
  - `embed_query()` with zero vector (norm == 0 branch on line 66-67)
  - `embed_query()` when API returns *fewer* dims than target (edge: 512 returned, 1024 expected)
  - `sparse_embed_query()` -- no tests at all
  - `sparse_embed_documents()` -- no tests at all
  - `close()` method
  - `client` property RuntimeError when not initialized
  - `sparse_model` property RuntimeError when not initialized
  - Error handling when OpenAI API call fails (network error, rate limit, invalid response)

#### Tautological / Mirror Tests
- **None identified.** The tests are well-structured -- they mock the API at the HTTP boundary and test the actual truncation/normalization math. The `test_truncation_when_api_returns_full_dims` test genuinely validates the renormalization math rather than just asserting the mock returns what was set up.

#### Over-Mocked Tests
- **`test_initialize_creates_client_with_openrouter_base_url`**: This test patches `AsyncOpenAI` at the module level and asserts it was called with the right args. This is pure wiring verification -- it confirms `__init__` parameters are forwarded correctly but does not test that the resulting client actually works. **Severity: Low** -- this is a reasonable smoke test for configuration plumbing.
- **`test_embed_query_calls_correct_model`**: Verifies `embeddings.create` is called with the right model string. This is wiring verification, but it is useful because a model ID typo would silently produce wrong embeddings. **Severity: Acceptable.**

#### Missing Edge Cases
- **Zero-norm vector after truncation**: If the first 1024 dimensions are all zeros (unlikely but possible with pathological embeddings), the `norm > 0` guard prevents division-by-zero but the returned all-zeros vector is semantically meaningless. No test validates this path.
- **Empty string input**: What happens when `embed_query("")` is called? The OpenAI API may reject it, but the service has no guard.
- **BM25 sparse embedding with empty/whitespace-only text**: `sparse_embed_query("")` -- does fastembed return an empty sparse vector, or does it raise?
- **API returning empty `data` list**: If `response.data` is empty, `response.data[0]` will throw `IndexError`. No test for this.

#### Missing Critical Tests
- **`sparse_embed_query`**: This is a core business function used in every hybrid search. It converts fastembed output to Qdrant SparseVector format. Zero test coverage means any change to fastembed's output format would break production silently.
- **`sparse_embed_documents`**: Used in the ingestion pipeline. Same risk as above.

#### Unnecessary Tests
- None. All 4 existing tests serve a purpose.

---

### `backend/service/agent.py` <-> NO TESTS

#### Coverage
- **Tested:** Nothing.
- **UNTESTED:** Everything -- this is the most complex and highest-risk file in the service layer.

#### Missing Critical Tests

**`SearchAgent._build_filter()` (static method, lines 432-453):**
- Pure function, no external dependencies, trivially testable.
- Should test: empty dict returns None; year-only filter; document_id-only filter; both filters combined; filter with falsy values (year=0, empty string doc_id).

**`SearchAgent._should_retry()` (lines 335-341):**
- Pure function on state dict, trivially testable.
- Should test: empty chunks + retry_count < 2 returns "retrieve"; empty chunks + retry_count >= 2 returns "synthesize"; non-empty chunks returns "synthesize" regardless of retry_count.

**`SearchAgent._analyze_query()` (lines 193-221):**
- LLM-dependent but the fallback path (exception handling) is critical: when the LLM fails, it should fall back to raw query. This fallback should be tested with a mock that raises.
- The structured output parsing (QueryAnalysis -> search_queries + extracted_filters) should be tested with a mock returning a known QueryAnalysis object.

**`SearchAgent._retrieve()` (lines 227-283):**
- Complex deduplication logic: when multiple queries return the same chunk_id, the highest-scoring one should win. This is pure logic operating on dicts and is testable with mocked Qdrant responses.
- Retry broadening: on retry_count > 0, filters should be dropped and only raw query used. Testable with state manipulation.
- Filter merging: user filters should take precedence over extracted filters (line 233). This is a dict merge order that should be explicitly tested.

**`SearchAgent._grade_documents()` (lines 289-329):**
- Empty chunks -> returns empty list. Testable.
- LLM failure fallback -> keeps all chunks. Testable with a mock that raises.
- Index bounds validation (line 322): out-of-range indices from LLM should be filtered out. This is a critical safety check that should have explicit tests.

**`SearchAgent._synthesize()` (lines 347-426):**
- Empty chunks -> returns canned "I couldn't find..." message. Testable.
- Citation enrichment logic (lines 402-413): when LLM returns a citation with source_number=N, it should pull metadata from chunks[N-1]. This is off-by-one-error-prone and must be tested.
- Citation enrichment with out-of-bounds source_number: what if the LLM hallucinates source_number=99 when there are only 5 chunks?
- LLM failure -> returns error message. Testable.

**`SearchAgent.run()` (lines 145-163):**
- Integration test of the full graph pipeline with mocked services. Should verify the graph nodes execute in the correct order.

---

### `backend/service/qdrant_service.py` <-> NO TESTS

#### Coverage
- **Tested:** Nothing.
- **UNTESTED:** Every method.

#### Missing Critical Tests

**`QdrantService.connect()` / `close()`:**
- Lifecycle management. Should test that `close()` is safe to call when not connected (it is -- the `if self._client` guard handles it). Should test that `client` property raises RuntimeError before `connect()`.

**`QdrantService.ensure_collection()`:**
- Should test: collection already exists -> no-op; collection does not exist -> creates with correct named vector config (dense + bm25 sparse).
- The vector config structure (named vectors `"dense"` + sparse `"bm25"`) is a critical contract with the ingestion pipeline. If this drifts, all searches break.

**`QdrantService.upsert_points()`:**
- Batching logic: should test that 250 points with batch_size=100 results in 3 batches (100, 100, 50).

**`QdrantService.hybrid_search()`:**
- The prefetch_limit calculation (`top_k * 3`) and the RRF fusion configuration are critical. If the prefetch configuration is wrong, search quality degrades silently.
- This wraps `client.query_points()` with a specific prefetch + fusion structure. A test should verify the correct Qdrant API call structure is constructed.

**`QdrantService.search()`:**
- Simpler than hybrid_search but the `using="dense"` named vector reference must match the collection schema.

**`QdrantService.get_by_document_id()`:**
- The limit=1000 hardcap means documents with >1000 chunks silently lose data. This should be documented and tested.

**`QdrantService.health_check()`:**
- Should test: success returns True; any exception returns False.

---

### `backend/service/main.py` <-> NO TESTS

#### Coverage
- **Tested:** Nothing.
- **UNTESTED:** All 3 endpoints + lifespan + helper functions.

#### Missing Critical Tests

**`_build_qdrant_filter()` (lines 110-134):**
- Duplicates `SearchAgent._build_filter()` logic. Both should be tested, but this duplication itself is a code smell that should be flagged.
- Pure function, trivially testable.

**`_scored_point_to_chunk_result()` (lines 137-156):**
- Pure mapping function. Should be tested with a mock ScoredPoint including missing payload fields to verify defaults work.
- Edge case: `point.payload` is None (line 141 handles this). Should be tested.

**`POST /search/chunks` endpoint:**
- Should have a FastAPI TestClient test with mocked QdrantService and EmbeddingService.
- Should test: successful search, empty results, top_k clamping to max_top_k, filter construction, 500 error handling.

**`POST /search` endpoint (agent search):**
- Should test: successful agent search, 500 error handling, filter passthrough.
- The filter dict construction (lines 233-237) manually unpacks SearchFilters into a plain dict -- this is error-prone.

**`GET /health` endpoint:**
- Should test: healthy response, Qdrant disconnected response, collection_info failure.

**Lifespan:**
- The lifespan creates payload indexes and initializes all services. Testing this is harder but a smoke test with mocked dependencies would catch import errors and initialization ordering bugs.

---

### `backend/service/config.py` <-> NO TESTS

#### Coverage
- **Tested:** Indirectly tested via the embedding service test fixture.
- **UNTESTED:** No dedicated tests.

#### Missing Critical Tests
- **Default values**: The defaults (qdrant_url, collection name, embedding model, dimensions) are load-bearing. A test should verify that `ServiceSettings()` produces expected defaults.
- **Env var override**: Should test that environment variables correctly override defaults (this is pydantic-settings' core feature, but a smoke test prevents config regressions).
- Low priority -- pydantic-settings is well-tested upstream, and the embedding test already constructs `ServiceSettings` with overrides.

---

### `backend/service/models.py` <-> NO TESTS

#### Coverage
- **Tested:** Indirectly by any endpoint tests (which don't exist).
- **UNTESTED:** No dedicated tests.

#### Missing Critical Tests
- **Validation constraints**: `SearchRequest.query` has `min_length=1, max_length=2000`; `top_k` has `ge=1, le=50`. These should be tested to prevent accidental constraint changes.
- **Default values**: `top_k=10`, `SearchFilters` optional fields default to None.
- Low-medium priority -- these are Pydantic models and fairly self-documenting, but the validation constraints are business rules.

---

### `backend/service/scripts/ingest_to_qdrant.py` <-> NO TESTS

#### Coverage
- **Tested:** Nothing.
- **UNTESTED:** Everything.

#### Missing Critical Tests

**`deterministic_uuid()`:**
- Pure function. Should test that the same chunk_id always produces the same UUID, and different chunk_ids produce different UUIDs. This is the primary key generation logic for Qdrant.

**`discover_documents()`:**
- Pure function. Should test: root doesn't exist -> empty list; directories missing parquet or metadata -> skipped; valid directories -> included.

**`build_points_for_document()`:**
- Complex function that merges parquet embeddings, JSON metadata, publication tracking data, and BM25 sparse vectors into Qdrant PointStruct objects.
- The author parsing try/except block (lines 153-159) handles legacy `authors_json` fallback -- this is error-prone and should be tested.
- Should test: with and without publication metadata; with missing chunk metadata; vector/payload structure matches what QdrantService.ensure_collection() expects.

**`load_publication_metadata()`:**
- Database-dependent. Could be tested with an in-memory SQLite database.

**`ingest()`:**
- Full pipeline integration. At minimum, a smoke test with mocked Qdrant and filesystem.

---

### `backend/storage/base.py` <-> NO TESTS

#### Coverage
- **Tested:** Nothing (abstract base class).
- **Assessment:** This is an ABC defining the interface contract. It doesn't need direct tests, but the implementations do.

---

### `backend/storage/local.py` <-> NO TESTS

#### Coverage
- **Tested:** Nothing.
- **UNTESTED:** Everything.

#### Missing Critical Tests

**`LocalStorage.__init__()`:**
- Creates the base directory on initialization. Should test that the directory is created.

**`LocalStorage.get_path()`:**
- Joins base_path + filename. Trivial but worth a parametric test.

**`LocalStorage.list_files()`:**
- With pattern and without pattern. Should test: empty directory, files with glob pattern, base_path doesn't exist -> empty list.

**`LocalStorage.exists()`:**
- Should test for files and directories.

**`LocalStorage.download()`:**
- No-op for local storage. Should verify it returns the correct path.

**`LocalStorage.upload()`:**
- No-op for local storage. Smoke test.

**`LocalStorage.glob()`:**
- Should test: matching files, no matches, relative path calculation, ValueError catch on line 84-85.

Risk: **Medium**. This is actively used by the ETL pipeline. A bug here would silently produce wrong file paths.

---

### `backend/storage/cloud.py` <-> NO TESTS

#### Coverage
- **Tested:** Nothing.
- **UNTESTED:** Everything.

#### Missing Critical Tests

**`CloudStorage.exists()`:**
- Two-phase check: exact blob match, then directory prefix check. Both branches need testing.

**`CloudStorage.download()`:**
- Three paths: single file download, directory prefix download, nothing found. Each needs testing with mocked GCS client.

**`CloudStorage.upload()`:**
- Three paths: file doesn't exist (warning), single file upload, directory recursive upload.

**`CloudStorage.glob()`:**
- Static prefix optimization (lines 184-191) is complex. Should test with patterns like `"processed/chunks/**/*.json"` to verify the optimization works correctly.

Risk: **High**. Cloud storage is used in production. A bug here causes data loss or silent ingestion failures. The GCS client should be mocked for unit tests.

---

### `backend/storage/gcs.py` <-> NO TESTS

#### Coverage
- **Tested:** Nothing.
- **UNTESTED:** Everything.

#### Assessment
This appears to be an older/parallel GCS implementation alongside `cloud.py`. It has additional methods (`read_text`, `write_text`, `read_binary`, `write_binary`, `copy`, `move`, `remove`) that `cloud.py` does not have, but it does not implement the full `StorageBase` interface properly (missing `upload()`, `download()`, `glob()`). The `__exit__` method suggests it was intended as a context manager but `__enter__` is missing.

**Code smell**: Two separate GCS implementations (`cloud.py` and `gcs.py`) is confusing. Which one is actually used?

Risk: **Medium-High** depending on usage. At minimum, this module should either be tested or removed.

---

### `backend/storage/database.py` <-> NO TESTS

#### Coverage
- **Tested:** Nothing (though the database module is indirectly tested via ETL tests that use the tracking models).
- **UNTESTED:** Everything in the `Database` class, `init_db`, `ensure_db_initialized`, `get_db_session`.

#### Missing Critical Tests

**Module-level side effects (line 117):**
- `ensure_db_initialized()` runs on import if `AUTO_INIT_DB` is truthy. This means importing this module has side effects (creates/migrates a database). Tests that import this module indirectly will trigger database initialization. This is a testability hazard.

**`Database.get_or_create()`:**
- Should test: record exists -> return existing; record doesn't exist -> create new. The `defaults` merging logic should be tested.

**`Database.bulk_update()`:**
- Should test: updates existing records, skips non-existent ones.

**`Database.execute_raw_sql()`:**
- Rollback on error should be tested.

Risk: **Medium**. The `Database` class is a general-purpose ORM wrapper. The module-level `engine` and `ensure_db_initialized()` are more concerning because they have side effects on import.

---

### `backend/storage/factory.py` <-> NO TESTS

#### Coverage
- **Tested:** Nothing.
- **UNTESTED:** Everything.

#### Missing Critical Tests

**`StorageFactory.create_storage()` (lines 40-84):**
- Auto-detection logic: `CLOUD_ENVIRONMENT` env var -> cloud; absence -> local. Should be tested with mocked env vars.
- `detect_automatically: False` in config -> reads from `config.storage.type`. Should be tested.
- Local storage path resolution: relative paths should be resolved against project root. Should be tested.
- Default config fallback when YAML file is missing (lines 31-37). Should be tested.

Risk: **High**. The factory is the entry point for all storage operations. If auto-detection produces the wrong storage type, the entire ETL pipeline operates against the wrong backend (local vs cloud).

---

### `frontend/app.py` <-> NO TESTS

#### Coverage
- **Tested:** Nothing.
- **UNTESTED:** All rendering logic, session state management, search orchestration.

#### Assessment
Streamlit apps are notoriously hard to test. The rendering functions (`_render_agent_results`, `_render_chunk_results`) are tightly coupled to `st.markdown()` calls. However, `_run_search()` contains testable logic (mode switching, error handling).

Risk: **Low-Medium**. Frontend bugs are visible immediately in manual testing. Automated testing here would have low ROI given Streamlit's testing limitations.

---

### `frontend/api_client.py` <-> NO TESTS

#### Coverage
- **Tested:** Nothing.
- **UNTESTED:** All HTTP client methods.

#### Missing Critical Tests

**`SearchClient.health()`:**
- Should test: successful health check, connection error, timeout, HTTP error. All return appropriate `HealthStatus` or `APIError` objects.

**`SearchClient.agent_search()` and `SearchClient.chunk_search()`:**
- Should test response parsing, error handling for each exception type (ConnectionError, Timeout, HTTPError).
- The response parsing code manually maps JSON fields to dataclass fields. Any field name mismatch between backend models and client dataclasses would silently produce None/default values.

**`SearchClient._build_search_body()`:**
- Should test: with year filter, without year filter.

**`_extract_detail()`:**
- Should test: None response, valid JSON with "detail" key, invalid JSON response.

Risk: **Medium**. The API client is the contract boundary between frontend and backend. A test that constructs known JSON responses and verifies correct parsing would catch schema drift.

---

## Completely Untested Modules

| Module | Risk Level | Justification |
|--------|-----------|---------------|
| `backend/service/agent.py` | **CRITICAL** | Core business logic: LangGraph agent with query analysis, retrieval, grading, synthesis. Contains complex state management, LLM interactions with fallback paths, deduplication logic, citation enrichment with off-by-one potential. Any regression here breaks the primary user-facing feature. |
| `backend/service/qdrant_service.py` | **HIGH** | All vector search operations. Named vector configuration must match ingestion schema. Hybrid search prefetch/fusion setup is fragile. |
| `backend/service/main.py` | **HIGH** | FastAPI endpoints -- the only entry point for all API consumers. No endpoint tests means no contract validation. |
| `backend/service/scripts/ingest_to_qdrant.py` | **HIGH** | Data ingestion pipeline. `build_points_for_document()` merges multiple data sources into Qdrant points. `deterministic_uuid()` generates primary keys. Bugs here corrupt the search index. |
| `backend/storage/cloud.py` | **HIGH** | Production cloud storage with GCS. Download/upload/glob logic is non-trivial. Bugs cause data loss. |
| `backend/storage/factory.py` | **HIGH** | Storage type auto-detection. Wrong detection = entire pipeline runs against wrong storage backend. |
| `backend/storage/local.py` | **MEDIUM** | Simpler than cloud but still the ETL workhorse for local dev. `glob()` and `list_files()` have edge cases. |
| `backend/storage/database.py` | **MEDIUM** | Database class with CRUD operations. Module-level side effects on import are a testability concern. |
| `backend/storage/gcs.py` | **MEDIUM** | Possibly dead code / parallel GCS implementation. Needs investigation -- if unused, should be removed; if used, needs tests. |
| `backend/service/config.py` | **LOW** | Pydantic-settings does the heavy lifting. Defaults are load-bearing but low complexity. |
| `backend/service/models.py` | **LOW** | Pydantic models with validation constraints. Important for API contract but low logic complexity. |
| `frontend/api_client.py` | **MEDIUM** | Contract boundary between frontend and backend. Schema drift risk. |
| `frontend/app.py` | **LOW** | Streamlit app. Hard to test, bugs are immediately visible. |

---

## Structural Issues Identified

### 1. Duplicate Filter Construction Logic
`SearchAgent._build_filter()` (agent.py:432-453) and `_build_qdrant_filter()` (main.py:110-134) implement the same filter-to-Qdrant-filter conversion independently. This duplication means a filter bug fix must be applied in two places.

### 2. Duplicate GCS Implementations
`backend/storage/cloud.py` (CloudStorage) and `backend/storage/gcs.py` (GCSStorage) are two separate GCS implementations. `cloud.py` implements the full `StorageBase` interface; `gcs.py` has extra methods (read_text, write_text, etc.) but incomplete `StorageBase` coverage (no `upload`, `download`, `glob`). The factory only uses `CloudStorage`. `GCSStorage` may be dead code.

### 3. Module-Level Database Side Effects
`backend/storage/database.py` runs `ensure_db_initialized()` on import (line 117-118). This means any test that transitively imports this module will attempt to create/migrate a SQLite database. This complicates test isolation.

### 4. No `__init__.py` Test Directory for Service
`backend/tests/service/` has no `__init__.py`, relying on pytest's rootdir discovery. This is fine but worth noting for import consistency.

---

## Priority Recommendations

1. **[CRITICAL] Test `SearchAgent._build_filter()` and `SearchAgent._should_retry()`** -- These are pure functions with zero dependencies. They can be tested in under 30 minutes and validate the most dangerous business logic (retry policy and filter construction). Start here for immediate ROI.

2. **[CRITICAL] Test `SearchAgent._analyze_query()`, `_grade_documents()`, and `_synthesize()` fallback paths** -- Mock the LLM to raise exceptions and verify fallback behavior. These error paths are exercised in production when LLMs are flaky, and bugs here cause 500 errors instead of graceful degradation.

3. **[CRITICAL] Test citation enrichment in `_synthesize()` (lines 402-413)** -- The off-by-one mapping (`source_number - 1` to index into chunks) and the None-guarded field enrichment are high-risk. Test with out-of-bounds source_number, missing chunk metadata fields, and normal cases.

4. **[HIGH] Add FastAPI TestClient tests for all 3 endpoints** -- Use dependency injection overrides to mock QdrantService and EmbeddingService. Test happy path, error handling, and request validation (empty query, top_k out of range).

5. **[HIGH] Test `QdrantService.ensure_collection()` vector schema** -- The named vector configuration (`"dense"` + `"bm25"` sparse) is a contract between ingestion and search. A mock-based test should verify the exact `create_collection()` call structure.

6. **[HIGH] Test `ingest_to_qdrant.deterministic_uuid()` and `build_points_for_document()`** -- The UUID function is trivially testable. `build_points_for_document()` needs a test with fixture parquet/JSON files to verify the point payload structure matches what the search service expects.

7. **[HIGH] Test `StorageFactory.create_storage()` with environment variable manipulation** -- Test all branches: auto-detect local, auto-detect cloud, explicit override, config file missing fallback.

8. **[MEDIUM] Test `LocalStorage` with a temp directory** -- `glob()`, `list_files()`, `exists()` against a real (temporary) filesystem. Fast, no mocks needed, high confidence.

9. **[MEDIUM] Test `CloudStorage` with mocked GCS client** -- Focus on `download()` (3 code paths), `exists()` (2-phase check), and `glob()` (prefix optimization).

10. **[MEDIUM] Test `frontend/api_client.py` response parsing** -- Mock `requests.Session` to return known JSON payloads and verify dataclass construction. This catches schema drift between backend and frontend models.

11. **[LOW] Investigate and resolve `gcs.py` vs `cloud.py` duplication** -- If `GCSStorage` is unused, remove it. If it is used, it needs tests and should implement `StorageBase` fully (or the interface should be refactored).

12. **[LOW] Add contract tests for `backend/service/models.py`** -- Verify validation constraints (min_length, max_length, ge, le) to prevent accidental changes to API contracts.
