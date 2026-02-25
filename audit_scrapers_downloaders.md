# Test Audit: ETL Scrapers & File Downloaders

## Executive Summary

The test suite for the ETL Scrapers & File Downloaders subsystem has **significant structural problems** that undermine its value as a safety net. The most critical finding is that **all OpenAlex file downloader tests are globally skipped** via `pytestmark`, meaning the most complex downloader has zero effective test coverage. The Growth Lab file downloader's validation test is a textbook example of a tautological test -- it mocks the very method it claims to test, then asserts on the mock's return value. Several tests across the subsystem verify mock wiring rather than business logic, and important edge cases in HTML parsing, URL resolution, and merge logic are untested.

---

## File-by-File Analysis

### openalex.py <-> test_openalex.py

**Production file:** `/Users/shg309/Dropbox (Personal)/Education/hks_cid_growth_lab/gl_deep_search/backend/etl/scrapers/openalex.py`
**Test file:** `/Users/shg309/Dropbox (Personal)/Education/hks_cid_growth_lab/gl_deep_search/backend/tests/etl/test_openalex.py`

#### Coverage

**Tested functions/logic:**
- `_extract_abstract()` -- tested with normal, empty, and gap cases. This is the best-tested function in the file.
- `save_to_csv()` / `load_from_csv()` -- round-trip test exists and is meaningful.
- `OpenAlexPublication` model -- ID generation, content hash, various ID fallback paths.

**UNTESTED functions/logic:**
- `_build_url()` -- not tested at all. The URL construction logic (author ID stripping with `lstrip('A')`, cursor handling, per-page parameter) is completely uncovered.
- `fetch_page()` -- the only test is `@pytest.mark.skip`. Retry logic, rate limiting (429 handling), and error recovery are untested.
- `fetch_all_pages()` -- skipped. The pagination loop, overall retry counter, and empty-results handling are untested.
- `process_results()` -- skipped. This is the core data transformation function that extracts fields from raw API responses, and the only test for it (`test_fetch_publications`) is skipped.
- `update_publications()` -- skipped. The merge logic (new vs. updated vs. retained publications) is untested despite being business-critical.
- `_load_config()` -- not tested (fallback defaults, YAML loading failure).

#### Tautological / Mirror Tests

- **`test_publication_model`**: While not strictly tautological, this test is testing the `OpenAlexPublication` Pydantic model, not the `openalex.py` scraper. It validates model behavior (ID generation, validators) that belongs in a separate model test file. The scraper's own logic is untested.

#### Over-Mocked Tests

- **`test_fetch_page` (skipped)**: Patches `OpenAlexClient.fetch_page` with a completely new function, then calls the patched version. This tests the mock, not the production code. Even if it were not skipped, it would validate nothing about the actual HTTP request, retry logic, or response parsing.
- **`test_fetch_publications` (skipped)**: Mocks `fetch_all_pages` entirely, so `fetch_publications` only exercises `process_results()`. This is acceptable as a unit test of `process_results`, but calling it `test_fetch_publications` is misleading.
- **`test_update_publications` (skipped)**: Mocks `fetch_publications`, so it only tests the merge logic. While the approach is reasonable, the test is skipped.

#### Missing Edge Cases

- `_extract_abstract()` with overlapping positions (same position mapped to multiple words).
- `_extract_abstract()` with `None` passed instead of `{}`.
- `process_results()` with malformed results: missing `id`, missing `authorships`, `None` values for nested dicts like `primary_location`.
- `fetch_page()` with non-JSON responses, network timeouts, and partial responses.
- `_build_url()` with author IDs that already start with "A" vs. those that don't (the `lstrip('A')` is dangerous -- it strips ALL leading A's, so an ID like "AAB123" becomes "B123").
- `load_from_csv()` using `eval()` on CSV data -- this is a **security vulnerability** (arbitrary code execution) that has no test demonstrating the risk or guarding against it.

#### Missing Critical Tests

1. **`process_results()` is the core business logic** -- it transforms raw OpenAlex API responses into publication objects. This should be the most thoroughly tested function, but its only test is skipped.
2. **`update_publications()` merge logic** -- the three-way merge (new, updated, retained) is business-critical and completely untested.
3. **`_build_url()` URL construction** -- the `lstrip('A')` bug should be caught by a test.
4. **`fetch_page()` retry and rate-limit behavior** -- the 429 handler sleeps for 60 seconds; the general retry sleeps for 5 seconds. This logic is untested.

#### Unnecessary Tests

- **`test_openalex_real_data_id_generation` (skipped)**: This integration test reads from a real CSV file that won't exist in CI. Even when run manually, it uses `eval()` on CSV data and doesn't test any scraper logic -- it only tests model ID generation which should be in a model test file.

---

### growthlab.py <-> test_growthlab.py

**Production file:** `/Users/shg309/Dropbox (Personal)/Education/hks_cid_growth_lab/gl_deep_search/backend/etl/scrapers/growthlab.py`
**Test file:** `/Users/shg309/Dropbox (Personal)/Education/hks_cid_growth_lab/gl_deep_search/backend/tests/etl/test_growthlab.py`

#### Coverage

**Tested functions/logic:**
- `parse_publication()` -- tested with old HTML structure (`biblio-entry`). This is a meaningful test.
- `extract_publications()` -- tested with mocks (deduplication, limit parameter). Deduplication test is well-designed.
- `update_publications()` -- tested with mocked `extract_and_enrich_publications`.
- `save_to_csv()` / `load_from_csv()` -- round-trip test exists.
- `GrowthLabPublication` model -- ID generation, content hash, year validation.
- Integration tests for real website scraping (pagination, deduplication, FacetWP max page detection) -- these are **excellent** and represent the strongest tests in the entire subsystem.

**UNTESTED functions/logic:**
- `_parse_author_string()` -- This is a standalone function with complex regex logic for splitting author strings. Zero unit tests despite handling multiple formats ("Hausmann, R. & Klinger, B.", "Hausmann, R., Tyson, L.D. & Zahidi, S.", simple comma-separated names).
- `parse_publication()` with new HTML structure (`cp-publication`) -- the test only covers the old `biblio-entry` format. The new `cp-publication` structure with `publication-title`, `publication-authors`, `publication-year`, `publication-excerpt`, and `publication-links` is completely untested.
- `_build_facetwp_payload()` -- not tested.
- `_parse_publication_elements()` -- not tested. Handles three different HTML structures (cp-publication, wp-block-post > cp-publication, biblio-entry).
- `_fetch_page_impl()` -- not directly tested. The distinction between page 1 (GET) and pages 2+ (FacetWP AJAX POST) is only tested via integration tests.
- `_get_max_page_num_impl()` -- FWP_JSON extraction logic is only tested via integration tests.
- `get_endnote_file_url()` -- not tested.
- `parse_endnote_content()` -- not tested. Contains logic for parsing Endnote format (%A, %T, %D, %X tags).
- `_enrich_publication_impl()` -- not tested.
- `enrich_publication_from_page()` -- not tested. Contains JSON-LD extraction, abstract extraction from publication pages.
- `enrich_publication()` -- not tested. The fallback chain (Endnote -> page enrichment) is untested.
- `extract_and_enrich_publications()` -- only indirectly tested through `update_publications`.
- `_load_config()` -- not tested.
- `year_corrections` dictionary -- the hardcoded year corrections are not tested.

#### Tautological / Mirror Tests

- **`test_publication_enrichment`**: This test manually sets fields on a publication object and then asserts those fields are set. It does not call any production enrichment code. The comment says "Test the enrichment functionality without using async mocks" but the test simulates the enrichment by hand rather than testing it. This is a textbook tautological test.

#### Over-Mocked Tests

- **`test_extract_publications`**: Mocks `get_max_page_num`, `fetch_page`, AND `aiohttp.ClientSession`. The test creates a pre-built `GrowthLabPublication` and returns it from the mocked `fetch_page`. This tests that `extract_publications` calls `fetch_page` and collects results -- pure wiring verification.
- **`test_extract_publications_with_limit`**: Same over-mocking pattern. The limit logic is partially tested, but the mock returns a fixed set of publications per page, so the actual limit-based early stopping and page estimation logic is not meaningfully validated.
- **`test_update_publications_with_limit`**: Mocks `extract_and_enrich_publications` entirely. Only tests that the result is sliced to the limit -- trivial logic.

#### Missing Edge Cases

- `parse_publication()` with missing title, missing authors, missing abstract, missing year, missing file URLs.
- `parse_publication()` with relative URLs that need base domain construction.
- `parse_publication()` with new HTML structure (cp-publication format).
- `_parse_author_string()` with edge cases: single author, author with parenthetical initials, empty string, string with only whitespace, ampersand without spaces.
- `load_from_csv()` using `eval()` on CSV data (same security concern as OpenAlex).
- `update_publications()` merge logic: publication updated (different hash), publication unchanged, publication removed from website but retained, new publication added.
- FacetWP AJAX response with missing `template` key, malformed JSON, empty template.
- Rate limiting behavior (429 responses) in `_fetch_page_impl`.

#### Missing Critical Tests

1. **`_parse_author_string()`** -- This is pure business logic with no external dependencies, making it the ideal candidate for thorough unit testing. The regex-based splitting handles multiple formats and is likely to break on edge cases.
2. **`parse_publication()` with new HTML structure** -- The production code supports both old and new HTML formats, but only the old format is tested. If the new website structure changes, there's no test to catch the regression.
3. **`parse_endnote_content()`** -- Pure parsing logic with no dependencies, easy to test, but completely untested.
4. **`enrich_publication_from_page()`** -- JSON-LD extraction and abstract extraction from publication pages is complex logic with no test coverage.
5. **`update_publications()` merge logic** -- The three-way merge (same as OpenAlex) is business-critical and only superficially tested via mocks.

#### Unnecessary Tests

- **`test_publication_enrichment`**: As described above, this test manually sets fields and asserts on them. It tests nothing.
- **`test_scraper_works_without_tracker`**: Mocks `extract_and_enrich_publications`, only verifies no exception is raised. Marginal value.

---

### oa_file_downloader.py <-> test_oa_file_downloader.py

**Production file:** `/Users/shg309/Dropbox (Personal)/Education/hks_cid_growth_lab/gl_deep_search/backend/etl/utils/oa_file_downloader.py`
**Test file:** `/Users/shg309/Dropbox (Personal)/Education/hks_cid_growth_lab/gl_deep_search/backend/tests/etl/test_oa_file_downloader.py`

#### Coverage

**ALL TESTS ARE GLOBALLY SKIPPED** via `pytestmark = pytest.mark.skip(...)`. Every test class and function in this file provides zero coverage in any test run. The file is ~840 lines of dead test code.

Despite being skipped, the tests are well-structured and cover:
- `__init__` and config loading
- `_get_file_path()` -- path generation with extensions, without extensions, and missing pub ID
- `_check_open_access()` -- Unpaywall success, not-found, API error
- `_resolve_url_and_check_content()` -- redirect following, non-download content detection
- `_download_file_with_aiohttp()` -- success, resume, validation failure, HTTP errors
- `_download_file_with_scidownl()` -- success and failure
- `_validate_downloaded_file()` -- PDF validation, file too small, file not found
- `download_file()` -- cached file handling

#### Tautological / Mirror Tests

- **`test_get_file_path_generation`**: Constructs the expected URL hash and path string using the same logic as the production code (MD5 of cleaned URL, same path template). This is a mirror test -- if the production code changes the hashing scheme, the test would need the same change, making it unable to catch regressions.

#### Over-Mocked Tests

- Nearly every test in `TestDownloaderHttpDownload` mocks `_resolve_url_and_check_content`, `_validate_downloaded_file`, `storage.exists`, `storage.stat`, and `aiofiles.open`. The tests verify that these mocked components are called with the right arguments, but they don't test the actual download logic. This is wiring verification, not behavior testing.
- **`test_download_aiohttp_success`**: Mocks URL resolution, file validation, storage operations, and the HTTP response. The only thing being tested is that `_download_file_with_aiohttp` calls these components in the right order with the right arguments.
- **`MockStorageBase`** and **`MockOpenAlexPublication`**: Custom mock classes that duplicate the interface of production classes. If the production interface changes, these mocks won't break -- they'll silently become stale.

#### Missing Edge Cases

- `_check_open_access()` with CORE API responses (only Unpaywall is tested).
- `_resolve_url_and_check_content()` with relative redirect URLs.
- `_resolve_url_and_check_content()` with too many redirects.
- `download_file()` DOI path -- the complex branching logic (check OA -> try OA download -> fallback to scidownl) is not tested.
- `download_publications()` -- exception handling during download, progress tracking, session cleanup.
- `_download_file_with_scidownl()` with incorrect DOI format (the DOI normalization logic).
- File validation with Word document magic bytes.
- File validation with files exceeding `max_file_size`.

#### Missing Critical Tests

1. **The entire file is skipped.** Every test listed above provides zero value until the skip marker is removed.
2. **DOI download orchestration** (`download_file` with `is_doi=True`): The most complex flow -- OA check, OA download, scidownl fallback -- has no test.
3. **`download_publications()`**: The top-level orchestration method that processes multiple publications is untested.
4. **Session lifecycle**: `_get_session()` creates an aiohttp session with specific configuration (timeouts, connection pooling, user agents). No test verifies this configuration.

#### Unnecessary Tests

- All tests are currently unnecessary because they are skipped and provide no coverage. If unskipped:
  - **`test_init_defaults`**: Tests that `asyncio.Semaphore` is called with the right argument. This is testing Python's constructor mechanism, not business logic.

---

### gl_file_downloader.py <-> test_gl_file_downloader.py

**Production file:** `/Users/shg309/Dropbox (Personal)/Education/hks_cid_growth_lab/gl_deep_search/backend/etl/utils/gl_file_downloader.py`
**Test file:** `/Users/shg309/Dropbox (Personal)/Education/hks_cid_growth_lab/gl_deep_search/backend/tests/etl/test_gl_file_downloader.py`

#### Coverage

**Tested functions/logic:**
- `_get_file_path()` -- tested for PDF, URL without filename, and DOCX. Meaningful test.
- `_correct_file_extension()` -- tested for bin-to-pdf rename, no-op when correct, and charset stripping. **These are the best unit tests in the entire subsystem** -- they test real file system behavior with tmp_path.
- `download_file()` with cached file -- verifies that existing files skip download.
- `download_publications()` -- tested with mocked `download_file`.
- `download_growthlab_files()` -- tested with mocked scraper and downloader.
- Integration tests for actual file downloads (PDF URL, non-PDF URL, multiple publications) -- well-designed but depend on external URLs.

**UNTESTED functions/logic:**
- `_download_file_impl()` -- the actual download implementation with streaming, status code handling (200, 206, 416, errors), file writing, and validation. No real test.
- `download_file()` -- the retry logic wrapping `_download_file_impl` via `retry_with_backoff` is untested.
- `_validate_downloaded_file()` -- **appears tested but is not** (see Tautological section below).
- `_load_config()` -- not tested.
- `_get_session()` -- curl_cffi session configuration is not tested.
- `_log_download_summary()` -- not tested (low priority).
- `download_publications()` -- the publication tracker integration (updating `DownloadStatus` based on results) is untested.
- Resume logic in `_download_file_impl()` -- the Range header construction and partial content handling.

#### Tautological / Mirror Tests

- **`test_validate_downloaded_file`**: This is the most egregious tautological test in the codebase. The test:
  1. Creates real PDF, DOC, DOCX, invalid, and small files on disk.
  2. Patches `Path.exists` and `Path.stat` globally (overriding the real files it just created).
  3. Patches `builtins.open` (overriding the real files).
  4. Then **patches `file_downloader._validate_downloaded_file` itself** with a mock.
  5. Calls the mocked method and asserts on the mock's return value.

  This test literally tests: "if I tell the mock to return `{'is_valid': True}`, it returns `{'is_valid': True}`". It validates absolutely nothing about the production validation logic. The real validation logic (magic byte checking, size validation, format detection) has **zero test coverage**.

#### Over-Mocked Tests

- **`test_download_file_impl`**: Patches `_download_file_impl` itself with a mock that returns a pre-built `DownloadResult`, then calls the mock and asserts on the mock's return value. This is identical to the validation test problem -- it tests the mock, not the production code.
- **`test_download_publications`**: Mocks `download_file` to always return success. Only verifies that the method collects results from the right number of files and associates them with the right publication IDs. The actual download logic, error handling, and tracker integration are not tested.
- **`test_download_growthlab_files`**: Creates a completely separate mock function that mimics the production function's structure. This tests the mock function, not the real `download_growthlab_files()`. The test does not even import or call the real function.

#### Missing Edge Cases

- `_get_file_path()` with URLs containing special characters, unicode, or encoded paths.
- `_get_file_path()` when `publication.paper_id` is `None` or empty.
- `_correct_file_extension()` for Word document types (.doc, .docx), Excel types (.xls, .xlsx).
- `_correct_file_extension()` when the file doesn't exist.
- `_correct_file_extension()` when `content_type` is `None` or empty.
- `_download_file_impl()` with HTTP 416 (Range Not Satisfiable) response.
- `_download_file_impl()` with HTTP 206 (Partial Content) for resume.
- `_download_file_impl()` with non-200/206/416 status codes.
- `_download_file_impl()` when file writing fails.
- `_download_file_impl()` with `RequestsError` and `TimeoutError` exceptions.
- `_validate_downloaded_file()` with files that have incorrect magic bytes for their type.
- `_validate_downloaded_file()` with files exceeding `max_file_size`.
- `download_publications()` with publications that have no `file_urls`.
- `download_publications()` with mixed success/failure results and tracker status updates.

#### Missing Critical Tests

1. **`_validate_downloaded_file()` needs real tests** -- the existing test is tautological and provides zero coverage. This is the gatekeeper for file quality and currently has no real validation.
2. **`_download_file_impl()` needs at least basic tests** -- this is the core download method. Even with mocked HTTP sessions, testing the status code branching (200, 206, 416, error) and file writing logic would be valuable.
3. **Publication tracker integration in `download_publications()`** -- the code updates `DownloadStatus.DOWNLOADED` or `DownloadStatus.FAILED` based on results, but this is never tested.
4. **`download_file()` with retry behavior** -- the `retry_with_backoff` wrapping is a key resilience feature that is untested.

#### Unnecessary Tests

- **`test_download_file_impl`**: Tests a mock of itself. Zero value.
- **`test_validate_downloaded_file`**: Tests a mock of itself. Negative value -- it creates a false sense of coverage.
- **`test_download_growthlab_files`**: Tests a locally-defined mock function, not the production `download_growthlab_files()`. Zero value for regression detection.

---

## Priority Recommendations

1. **[CRITICAL] Unskip or rewrite `test_oa_file_downloader.py`**: The entire OpenAlex file downloader has zero test coverage. Either fix the underlying issues that caused the skip and re-enable the tests, or delete the file and write new tests that actually run. Currently 840 lines of dead test code.

2. **[CRITICAL] Rewrite `test_validate_downloaded_file` in `test_gl_file_downloader.py`**: The current test patches the method it's testing and asserts on mock return values. Replace with tests that create real temp files (valid PDF with magic bytes, invalid file, too-small file, too-large file) and call the real `_validate_downloaded_file()` method against them. The fixture already creates the test files; the test just needs to stop mocking everything.

3. **[CRITICAL] Rewrite `test_download_file_impl` in `test_gl_file_downloader.py`**: Same problem -- replace the self-mocking test with one that exercises the actual status-code branching logic, even if the HTTP session is mocked.

4. **[HIGH] Add unit tests for `_parse_author_string()`**: This is a pure function with complex regex logic and zero tests. Add tests for all documented formats plus edge cases (single author, empty string, unusual punctuation).

5. **[HIGH] Add unit tests for `process_results()` in OpenAlex**: This is the core transformation logic. Test with complete results, partial results (missing fields), and malformed results (None values, missing keys).

6. **[HIGH] Add tests for `parse_publication()` with new HTML structure**: The `cp-publication` format is the current website structure but has zero test coverage. Add BeautifulSoup-based tests with sample HTML.

7. **[HIGH] Add unit tests for `parse_endnote_content()`**: Pure parsing logic with no dependencies. Test with valid Endnote format, missing fields, HTML in abstract field.

8. **[MEDIUM] Add tests for `_build_url()` in OpenAlex**: The `lstrip('A')` on author IDs is likely a bug (strips all leading A's, not just a prefix). A test would catch this.

9. **[MEDIUM] Add tests for `update_publications()` merge logic** (both OpenAlex and GrowthLab): Test the three-way merge: new publication, updated publication (different hash), unchanged publication, removed publication.

10. **[MEDIUM] Replace `test_download_growthlab_files` with a real test**: The current test defines its own mock function instead of testing the production code. Rewrite to actually call `download_growthlab_files()` with mocked dependencies.

11. **[MEDIUM] Add tests for `enrich_publication_from_page()`**: This extracts metadata from JSON-LD and HTML, which is fragile parsing logic that should have tests.

12. **[LOW] Remove `test_publication_enrichment` from `test_growthlab.py`**: It tests nothing -- it manually sets fields and asserts on them.

13. **[LOW] Address `eval()` usage in `load_from_csv()`** (both scrapers): Using `eval()` on CSV data is a security risk. While not a test issue per se, tests should verify that malicious CSV data doesn't execute arbitrary code, or the code should be refactored to use `ast.literal_eval()` or `json.loads()`.

14. **[LOW] Consolidate model tests**: Both test files contain extensive tests for `OpenAlexPublication` and `GrowthLabPublication` model behavior (ID generation, content hash, validators). These belong in a dedicated `test_models.py` file, not mixed into scraper tests.
