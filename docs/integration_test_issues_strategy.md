# Plan: Restore and Add Integration Tests

## Context

The test suite rewrite deleted 26 tests from `test_oa_file_downloader.py` (previously globally skipped) and replaced them with 16 focused unit tests. While the unit tests are better quality (real file validation, no self-mocking), 16 of the old tests covering critical control flow paths were lost with no replacement — the entire OA check → download → scidownl fallback chain, HTTP download mechanics, download orchestration, retry logic, and the top-level entry point. Additionally, `test_agent.py` and `test_main.py` only test happy paths. The user values integration tests that exercise actual control flow paths over mock-heavy unit tests.

---

## Files to Modify

| File                                           | Tests to Add | Focus                                                                                                                     |
| ---------------------------------------------- | ------------ | ------------------------------------------------------------------------------------------------------------------------- |
| `backend/tests/etl/test_oa_file_downloader.py` | ~20          | All 4 `download_file` paths, URL resolution, OA checking, scidownl, batch orchestration, 2 real-network integration tests |
| `backend/tests/service/test_agent.py`          | 3            | Retry path, max retries exhausted, query analysis failure fallback                                                        |
| `backend/tests/service/test_main.py`           | 3            | Error handling paths (500s, degraded health)                                                                              |

No production code changes. No files deleted. Only test additions.

---

## `test_oa_file_downloader.py` — Primary Target

### Mocking Strategy

Mock at HTTP boundaries only (not internal methods). The key boundaries:

- `session.get()` / `session.head()` — aiohttp HTTP calls
- `scidownl.scihub_download` — external library
- File system — use real `tmp_path`, no mocking

Let internal methods (`_resolve_url_and_check_content`, `_download_file_with_aiohttp`, `_check_open_access`, `_validate_downloaded_file`) run for real so we test actual control flow.

### Shared Helper: `AsyncContextManagerMock`

aiohttp uses `async with session.get(url) as response:` pattern. Need a helper to mock this:

```python
class AsyncContextManagerMock:
    def __init__(self, return_value):
        self.return_value = return_value
    async def __aenter__(self):
        return self.return_value
    async def __aexit__(self, *args):
        pass
```

Plus a `_make_mock_response(status, headers, content, json_data)` factory.

---

### Tests to Add (no marker — run by default)

#### `TestResolveUrlAndCheckContent` (5 tests)

Mock `session.head` to control HTTP responses. Exercise the redirect-following loop and content-type detection.

1. **`test_resolve_follows_redirects`** — 302 → 200 with PDF content type. Assert returns `(final_url, "application/pdf", True)`.
2. **`test_resolve_max_redirects_exceeded`** — Always returns 302. Assert returns `(None, None, False)`.
3. **`test_resolve_detects_pdf_content_type`** — 200 with `application/pdf`. Assert `is_direct=True`.
4. **`test_resolve_detects_attachment_disposition`** — 200 with `Content-Disposition: attachment; filename=paper.pdf`. Assert `is_direct=True`.
5. **`test_resolve_html_not_direct`** — 200 with `text/html`. Assert `is_direct=False`.

#### `TestDownloadFileWithAiohttp` (6 tests)

Mock `session.head` + `session.get`. Let validation run against real temp files.

6. **`test_aiohttp_success_200`** — Mock head→PDF direct, mock get→200 streaming PDF bytes. Assert `success=True`, file on disk has PDF magic bytes.
7. **`test_aiohttp_url_resolution_failure`** — Mock head to raise `ClientError`. Assert `success=False`, error contains `"resolve"`.
8. **`test_aiohttp_landing_page_rejected`** — Mock head→`text/html`. Assert `success=False`, error mentions `"landing page"`.
9. **`test_aiohttp_http_error_status`** — Mock head→PDF direct, mock get→403. Assert `success=False`, error contains `"403"`.
10. **`test_aiohttp_validation_fails_cleans_up`** — Mock get→200 with non-PDF bytes despite PDF content-type. Assert `success=False`, destination file deleted.
11. **`test_aiohttp_resume_206`** — Write partial file, mock get→206 with remaining bytes. Assert final file = partial + remainder, `success=True`.

#### `TestCheckOpenAccess` (3 tests)

Mock `session.get` for Unpaywall and CORE API URLs.

12. **`test_unpaywall_returns_oa_url`** — Mock Unpaywall JSON with `is_oa=True`, `best_oa_location.url_for_pdf`. Assert returns `(True, pdf_url)`.
13. **`test_unpaywall_not_oa_core_fallback_success`** — Mock Unpaywall `is_oa=False`, mock CORE returning download URL. Assert returns `(True, core_url)`.
14. **`test_both_apis_fail_returns_false`** — Mock Unpaywall 404, CORE empty. Assert returns `(False, None)`.

#### `TestDownloadWithScidownl` (2 tests)

Mock `scidownl.scihub_download` at the library boundary.

15. **`test_scidownl_success`** — Mock to write valid PDF to destination. Assert `success=True`, `source="scidownl"`.
16. **`test_scidownl_failure`** — Mock to raise exception. Assert `success=False`.

#### `TestDownloadFileOrchestration` (5 tests)

Test the main `download_file()` routing logic. Mock `_check_open_access` (because it does HTTP) and mock session for aiohttp calls, but let the rest of the control flow run.

17. **`test_cached_file_skips_download`** — Pre-create valid PDF at destination. Call with `overwrite=False`. Assert `cached=True`, no HTTP calls, `download_stats["cached"]` incremented.
18. **`test_doi_open_access_path`** — DOI URL, mock OA check returns `(True, oa_url)`, mock aiohttp success. Assert `success=True`, `download_stats["open_access"]` incremented.
19. **`test_doi_oa_fails_scidownl_fallback`** — Mock OA returns `(False, None)`, mock scidownl success. Assert `success=True`, `download_stats["scidownl"]` incremented.
20. **`test_non_doi_direct_download`** — Non-DOI URL, mock aiohttp success. Assert `success=True`, OA check never called.
21. **`test_doi_oa_download_fails_falls_to_scidownl`** — Mock OA returns URL, mock aiohttp fails, mock scidownl succeeds. Assert scidownl result returned.

#### `TestDownloadPublications` (2 tests)

22. **`test_batch_orchestration_doi_detection`** — 2 pubs (one DOI, one direct URL). Mock `download_file`. Assert DOI pub called with `is_doi=True`, direct pub called with `is_doi=False`, results have correct publication IDs.
23. **`test_handles_download_exceptions_gracefully`** — Mock `download_file` to raise for first pub, succeed for second. Assert both results recorded.

#### `TestDownloadOpenalexFilesEntryPoint` (1 test)

24. **`test_entry_point_loads_csv_and_filters`** — Create CSV via `save_to_csv`, mock `download_publications`. Assert CSV loaded correctly, publications filtered to those with `file_urls`.

---

### Tests to Add (`@pytest.mark.integration` — real network)

25. **`test_integration_unpaywall_lookup`** — Real Unpaywall API hit for known DOI. `pytest.skip` on network failure.
26. **`test_integration_full_doi_download`** — Real arXiv DOI download end-to-end. Assert PDF exists on disk. `pytest.skip` on network failure.

---

## `test_agent.py` — 3 Tests

### `TestSearchAgentIntegration` (add to existing class)

27. **`test_retry_path_no_chunks_then_broader_search`** — Mock first `hybrid_search`→empty, `grading`→empty. Mock second `hybrid_search`→chunks. Assert `retry_count` incremented, second search uses broader query, pipeline completes through synthesize.
28. **`test_max_retries_exhausted_synthesizes_empty`** — Mock `hybrid_search` always empty, grading always empty. Assert after retries exhausted, `synthesize` runs with empty chunks, answer indicates no relevant information found.
29. **`test_query_analysis_failure_uses_raw_query`** — Mock LLM `ainvoke` to raise on first call (query analysis). Assert pipeline falls back to raw query, continues and completes.

---

## `test_main.py` — 3 Tests

### New class: `TestEndpointErrorHandling`

30. **`test_search_chunks_500_on_service_error`** — Mock qdrant service to raise. Assert 500 response.
31. **`test_agent_search_500_on_agent_error`** — Mock agent to raise. Assert 500 response.
32. **`test_health_degraded_on_collection_error`** — Mock collection info to raise. Assert health returns but shows degraded status.

---

## Execution

```bash
uv run pytest backend/tests/etl/test_oa_file_downloader.py backend/tests/service/test_agent.py backend/tests/service/test_main.py -v
uv run ruff check backend/tests/etl/test_oa_file_downloader.py backend/tests/service/test_agent.py backend/tests/service/test_main.py
uv run ruff format backend/tests/etl/test_oa_file_downloader.py backend/tests/service/test_agent.py backend/tests/service/test_main.py
uv run pytest backend/tests/ -q  # full suite regression check
```

Run tests after each class is added (single sub-agent in worktree isolation).

---

## What This Does NOT Change

- All existing tests remain untouched (no deletions, no modifications to passing tests)
- No production code changes
- Other test files (`test_gl_file_downloader.py`, `test_growthlab.py`, `test_orchestrator.py`, etc.) are not modified — they already have adequate integration coverage
