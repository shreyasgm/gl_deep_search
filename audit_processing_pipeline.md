# Test Audit: ETL Processing Pipeline

## Executive Summary

The ETL processing pipeline tests are reasonably comprehensive for the "happy path" but suffer from several classic LLM-authored-test problems: one critically tautological mock in the Docling backend tests (where the test mocks `extract()` itself and then calls the mock), significant gaps in adversarial and edge-case coverage for the text chunker (unicode, malformed input, tokenizer round-trip fidelity), and near-total absence of meaningful error-path testing for the embeddings generator (e.g., network timeouts, partial batch failures, malformed API responses). The text chunker tests are the strongest of the three subsystems but still rely on toy inputs that do not stress the sentence detection or structure parsing with realistic OCR output. The embeddings generator tests have the most critical gap: the OpenRouter API retry path is tested once, but batch-level partial failure, zero-length embedding vectors, dimension mismatches from the API, and norms-of-zero division errors are completely untested.

---

## File-by-File Analysis

### pdf_processor.py <-> test_pdf_processor.py

#### Coverage

**Tested functions/logic:**
- `process_pdf()` skip-when-already-processed path (via `test_skip_already_processed`)
- `process_pdf()` force-reprocess override (via `test_force_reprocess_ignores_existing`)
- `process_pdf()` page cap enforcement (via `test_page_cap_skips_long_pdf`)
- `process_pdf()` calls `storage.upload()` after writing output (via `test_upload_called_after_processing`)
- `find_growth_lab_pdfs()` discovers PDFs via glob and magic-byte detection (via `test_finds_pdfs_via_storage_glob`)
- `find_growth_lab_pdfs()` returns empty list when directory missing (via `test_returns_empty_when_no_dir`)
- `process_pdf()` end-to-end with unstructured backend (integration test)

**UNTESTED functions/logic:**
- `_load_config()` -- config loading from YAML, error fallback to defaults
- `_init_backend()` -- fallback from primary to secondary backend on ImportError
- `_get_processed_relative()` -- the `except ValueError` path that uses md5 hash for unknown paths
- `_extract_publication_id()` -- extraction of pub ID from path
- `process_pdf()` -- the `result.success == False` path from the backend (backend returns failure)
- `process_pdf()` -- the "extracted text too short" path (`len(full_text) < min_chars_per_page`)
- `process_pdf()` -- the general exception handler (lines 291-306)
- `process_pdf()` -- all `PublicationTracker` interaction paths (IN_PROGRESS, PROCESSED, FAILED status updates, and tracker exception handling)
- `process_pdfs()` -- batch processing, success rate calculation, the `show_progress=False` branch with its own exception handler
- `process_growth_lab_pdfs()` -- the top-level orchestration function

#### Tautological / Mirror Tests
- None identified. The unit tests properly mock the backend and verify behavioral outcomes.

#### Over-Mocked Tests
- `test_page_cap_skips_long_pdf`: The mock of `pypdfium2` via `sys.modules` patching is brittle. It patches the entire module namespace rather than just the `PdfDocument` class. This works but is fragile and could silently pass even if the production code changes how it imports pypdfium2.
- `processor_with_mock_backend` fixture: Uses `PDFProcessor.__new__()` to bypass `__init__`, then manually sets all attributes. This means the fixture tests nothing about actual initialization and could drift if new attributes are added. However, this is a reasonable tradeoff to avoid loading heavy backends in unit tests.

#### Missing Edge Cases
- **Non-existent PDF file**: `process_pdf()` checks `pdf_path.exists()` but this path is never tested
- **Permission-denied on PDF file**: No test for OS-level read errors
- **Backend returns empty string (success=True but text="")**: The `min_chars_per_page` filter is untested
- **Backend raises unexpected exception**: The outer `try/except` in `process_pdf()` is untested
- **Tracker raises exception during status update**: All tracker error paths are untested
- **Page cap with corrupted PDF**: pypdfium2 fails to open the file -- the `except Exception` on line 230 is untested
- **PDF with exactly `max_pages` pages**: Boundary condition (should process, not skip)
- **`find_growth_lab_pdfs` with non-PDF files that have non-readable permissions**: The `except (OSError, PermissionError)` is untested

#### Missing Critical Tests
1. **Backend fallback mechanism** (`_init_backend`): The production code has a fallback from marker to docling (or vice versa). This fallback logic is never tested, but it is a critical reliability feature.
2. **PublicationTracker integration**: All tracker interactions are untested. Since tracker failures are silently swallowed (by design), a test should verify this swallowing behavior.
3. **Batch processing (`process_pdfs`)**: The batch function has its own exception handling in the `show_progress=False` branch that differs from the `show_progress=True` branch. Neither is tested.

#### Unnecessary Tests
- None. All tests verify meaningful behavior.

---

### pdf_backends/base.py, __init__.py, docling_backend.py, marker_backend.py, unstructured_backend.py, device.py <-> test_pdf_backends.py, test_pdf_backends_integration.py

#### Coverage

**Tested functions/logic:**
- `ExtractionResult` dataclass defaults (success=True, error=None, metadata={})
- `ExtractionResult` failure construction
- `DeviceType` enum and `detect_device()` with env var overrides (cuda, mps, cpu)
- `detect_device()` with invalid env var
- `detect_device()` fallback when torch unavailable (partially -- see note below)
- Backend registry: `register_backend()` + `get_backend()` round-trip
- Backend registry: unknown backend raises ValueError
- Lazy import mechanism for docling and marker (with skip on ImportError)
- `DoclingBackend.extract()` success path (but see tautological note)
- `DoclingBackend.extract()` failure path (converter raises exception)
- `MarkerBackend.extract()` failure path (model loading raises exception)
- Unstructured backend integration test (real PDF extraction)
- Docling and Marker integration tests (behind `@pytest.mark.integration`)

**UNTESTED functions/logic:**
- `PDFBackend.extract_batch()` default sequential implementation
- `PDFBackend.cleanup()` no-op method
- `DoclingBackend._get_converter()` -- lazy initialization, device mapping, config propagation
- `DoclingBackend.extract_batch()` -- native batch API with convert_all
- `DoclingBackend.cleanup()` -- resource release
- `MarkerBackend._get_device_str()` -- device string mapping
- `MarkerBackend._load_models()` -- lazy model loading, version detection, v0 vs v1 path
- `MarkerBackend._extract_v0()` -- v0.3.x API path
- `MarkerBackend._extract_v1()` -- v1.x+ API path, config passthrough keys, converter caching
- `MarkerBackend.cleanup()` -- resource release
- `UnstructuredBackend.extract()` -- config propagation (strategy, languages, extract_images)
- `UnstructuredBackend.extract()` -- page number extraction from element metadata
- `_detect_marker_version()` -- version detection logic
- `is_slurm_environment()` -- SLURM detection utility
- Backend module loaded but did not register itself (line 63-67 of `__init__.py`)

#### Tautological / Mirror Tests
- **`TestDoclingBackendUnit.test_extract_success` is critically tautological**: This test patches `DoclingBackend.extract` itself (line 135-137) with a mock that returns a pre-built `ExtractionResult`, then calls `backend.extract()`. The test is literally calling a mock and verifying the mock's return value. It tests nothing about the actual Docling extraction logic. The assertions (`result.success`, `result.backend_name == "docling"`, `len(result.text) > 0`) are all verifying values that the test itself hardcoded. This is the most severe testing defect in the entire suite.

#### Over-Mocked Tests
- `test_no_torch_falls_back_to_cpu`: The nested `patch.dict` calls for environment variables and `sys.modules` make this test extremely fragile and hard to reason about. The comment even acknowledges "This is tricky to test" and weakens the assertion to just `isinstance(device, DeviceType)` -- which passes for any device, not specifically CPU. This test effectively validates nothing specific.

#### Missing Edge Cases
- **Docling `ConversionStatus.PARTIAL_SUCCESS`**: The production code accepts this status but it is never tested (only `SUCCESS` is mocked in the broken test)
- **Docling document without `pages`, `tables`, or `pictures` attributes**: The `hasattr` checks are untested
- **Marker v0 vs v1 version detection**: `_detect_marker_version()` is never tested
- **Marker v1 config passthrough**: The `_MARKER_PASSTHROUGH_KEYS` mechanism is untested
- **Unstructured elements without page metadata**: The `page_number is not None` check is untested
- **Unstructured with zero elements**: Returns empty text
- **Empty PDF (0 pages)**: Behavior with degenerate input
- **Backend registry: module loads but does not call register_backend()**: Line 63-67 error path

#### Missing Critical Tests
1. **A non-tautological unit test for `DoclingBackend.extract()`**: The current test is useless. Need to mock `_get_converter()` to return a mock converter, and mock `ConversionStatus` for the comparison, then verify the actual `extract()` method logic (markdown export, page counting, metadata extraction).
2. **MarkerBackend success path**: There is no unit test for a successful Marker extraction. Only the failure path is tested.
3. **`extract_batch()` for any backend**: Batch processing is completely untested at the unit level.
4. **Unstructured backend unit tests**: There are zero unit tests for UnstructuredBackend. Only an integration test exists.

#### Unnecessary Tests
- `TestDoclingBackendUnit.test_extract_success` should be rewritten, not removed. In its current form it provides zero value.

---

### text_chunker.py <-> test_text_chunker.py

#### Coverage

**Tested functions/logic:**
- Configuration loading from YAML (`test_configuration_loading`)
- Default configuration values (`test_default_configuration_values`)
- Invalid configuration handling -- negative sizes, min>max, overlap>size, invalid strategy (`test_invalid_configuration_handling`)
- End-to-end PDF document processing (`test_process_pdf_document_end_to_end`)
- End-to-end transcript processing (`test_process_transcript_document_end_to_end`)
- Batch processing with error isolation (`test_batch_processing_with_multiple_documents`)
- Fixed-size chunking strategy (`test_fixed_size_chunking_strategy`)
- Sentence-aware chunking strategy (`test_sentence_aware_chunking_strategy`)
- Structure-aware chunking strategy (`test_structure_aware_chunking_strategy`)
- Hybrid chunking strategy (`test_hybrid_chunking_strategy`)
- Strategy fallback chain (`test_graceful_strategy_fallback`)
- Required metadata fields (`test_metadata_handling_required_fields`)
- Optional metadata fields / page numbers (`test_metadata_handling_optional_fields`)
- Empty input handling (`test_error_handling_empty_input`)
- Short document as single chunk (`test_error_handling_empty_input`, `test_short_document_vs_short_chunk_distinction`)
- File not found error (`test_error_handling_file_not_found`)
- Chunk token limit enforcement (`test_chunk_size_validation`)
- Overlap continuity with token matching (`test_overlap_continuity`)
- Embedding model token limit enforcement (`test_embedding_model_token_limit_enforced`)
- Chunk size clamped when exceeds max after model limit (`test_chunk_size_clamped_when_exceeds_max_after_model_limit`)
- JSON output schema validation (`test_json_output_schema_validation`)
- Directory structure creation (`test_directory_structure_creation`)
- Resume capability / skip processed files (`test_resume_capability_skip_processed`)
- Works without tracker (`test_chunker_works_without_tracker`)
- ETL orchestrator integration (`test_orchestrator_integration`)
- Performance requirements (`test_performance_requirements`)
- Actual sample PDF processing (optional, skips if files absent) (`test_process_actual_sample_pdfs`)

**UNTESTED functions/logic:**
- `_HFTokenizerAdapter` -- the adapter class wrapping HuggingFace tokenizers
- `_load_tokenizer()` -- fallback chain: model_name -> sentence-transformers/model_name -> tiktoken
- `_split_text_by_tokens()` -- standalone token-based splitting utility (only used indirectly)
- `_enforce_token_limits()` -- the safety net that force-splits oversized chunks
- `_force_split_by_tokens()` -- last-resort token splitting
- `_clean_text_for_chunking()` -- page marker removal, whitespace normalization
- `_extract_page_info()` -- page marker parsing
- `_find_pages_for_position()` -- page range overlap calculation
- `_find_pages_for_text()` -- fallback page detection
- `_detect_sentences()` -- regex-based sentence detection
- `_detect_structure()` -- regex-based structure detection
- `_split_large_section()` -- splitting oversized sections with sentence boundaries
- `_get_overlap_text()` -- overlap extraction by token count
- `_save_chunks()` -- resume capability (file exists check), JSON serialization
- `_chunks_relative_for()` -- path transformation logic
- `_resolve_output_dir()` -- output directory resolution with multiple fallback paths
- `_resolve_processed_documents_dir()` -- processed documents path resolution
- `process_all_documents()` -- tracker status update logic (both success and failure paths), upload to remote storage
- `chunk_by_sentences()` -- the overlap logic within sentence chunking specifically
- `_chunk_section_with_constraints()` -- the core hybrid chunking logic for individual sections

#### Tautological / Mirror Tests
- `test_short_document_vs_short_chunk_distinction` (Test 2): The assertion on line 798 checks `len(chunk.text_content) >= simple_config["min_chunk_size"]` which compares character length against a token-based config value. The `min_chunk_size` is 20 tokens, but the assertion checks character length >= 20. For English text, this accidentally passes because character counts are always larger than token counts. This is a latent bug in the test -- it validates the wrong metric and would pass even if the production code stopped enforcing the minimum.

#### Over-Mocked Tests
- `test_graceful_strategy_fallback`: Patches both `chunk_by_structure` and `chunk_by_sentences` to raise exceptions, then verifies `chunk_fixed_size` is used as fallback. This is a reasonable test design, but it completely sidesteps whether the fallback chain logic in `create_chunks()` handles the strategy ordering correctly. A more useful test would verify the specific order of strategy attempts and that the fallback only triggers after the previous strategy fails (not just that fixed-size eventually runs).
- `test_orchestrator_integration`: Patches `_run_text_chunker` with `AsyncMock` and then asserts it was awaited. This only tests that the orchestrator calls the method -- it tests zero chunking logic. The `dry_run=True` path before it also tests nothing about chunking.

#### Missing Edge Cases
- **Unicode text with mixed scripts**: Arabic, Chinese, emoji -- these can break sentence detection regex and affect token counts dramatically
- **Text with no sentence terminators**: OCR output often lacks punctuation entirely. The `_detect_sentences()` regex `([.!?;]+\s+)` would return the entire text as one "sentence," which could then exceed chunk limits. This specific failure mode (OCR without punctuation) is mentioned in the `_enforce_token_limits` docstring as a known risk but is never tested.
- **Text consisting entirely of page markers**: Would result in empty `clean_text` after cleaning
- **Text with Windows-style line endings (\r\n)**: The `_extract_page_info()` and structure detection split on `\n` only
- **Text with extremely long lines (no newlines at all)**: Structure detection relies on `text.split("\n")`
- **Nested markdown headers** (e.g., `## Introduction` followed by `### Background`): Structure detection behavior with hierarchy
- **Tokenizer returning different decoded text than encoded**: Round-trip fidelity of `encode()` -> `decode()` is assumed but never tested
- **Config file that does not exist or is not valid YAML**: `_load_config()` raises on error, but this is never tested
- **`max_chunk_size` exactly equal to `embedding_max_tokens`**: Boundary condition for the clamping logic
- **All strategies returning empty chunk lists**: The "All chunking strategies failed" error path
- **`_chunks_relative_for()` with paths that don't start with "processed/documents"**: The else branch

#### Missing Critical Tests
1. **`_detect_sentences()` with adversarial input**: The sentence regex is the most fragile part of the pipeline. It should be tested with abbreviations ("Dr. Smith"), decimal numbers ("3.14"), URLs, and text with no punctuation.
2. **`_enforce_token_limits()` / `_force_split_by_tokens()`**: These safety-net functions are the last line of defense against embedding failures. They should be directly tested with oversized input.
3. **`_clean_text_for_chunking()` with realistic OCR artifacts**: OCR output often has header/footer repetition, page number artifacts, and garbled characters.
4. **Overlap correctness in `chunk_by_sentences()`**: The overlap logic (`_get_overlap_text`) is tested only for `chunk_fixed_size`. The sentence-based overlap logic (which uses a different approach -- taking last sentences within token budget) is untested.
5. **`_detect_structure()` with no matching markers**: When no headers are found, the function returns one big section. This interacts with the size-splitting logic in `chunk_by_structure()`.

#### Unnecessary Tests
- `test_orchestrator_integration`: This test lives in the chunker test file but tests the orchestrator's dry-run and method dispatch. It validates nothing about chunking and belongs in an orchestrator test file.
- `test_process_actual_sample_pdfs`: Tests with hardcoded absolute paths to a specific developer's machine. Will always skip in CI. Should either use a fixture mechanism or be removed.

---

### embeddings_generator.py <-> test_embeddings_generator.py

#### Coverage

**Tested functions/logic:**
- Retry mechanism with RateLimitError and eventual success (`test_retry_mechanism_with_eventual_success`)
- Save embeddings format (Parquet + JSON) (`test_save_embeddings_format`)
- Resume capability / idempotent saves (`test_resume_capability`)
- SentenceTransformer MRL truncation and re-normalization (`test_sentence_transformer_truncation_and_renormalization`)
- SentenceTransformer no-truncation path (`test_sentence_transformer_no_truncation_when_dims_match`)
- SentenceTransformer save format (`test_sentence_transformer_save_format`)
- Real SentenceTransformer inference with all-MiniLM-L6-v2 (`test_real_sentence_transformer_inference`)
- End-to-end embedding generation with OpenRouter (integration) (`test_end_to_end_embedding_generation`)
- PublicationTracker integration (integration) (`test_publication_tracker_integration`)
- Batch processing of multiple documents (integration) (`test_batch_processing_multiple_documents`)

**UNTESTED functions/logic:**
- `__init__()` -- unsupported model provider ValueError
- `__init__()` -- fallback when EMBEDDING_API_KEY env var is missing (api_key=None)
- `_load_config()` -- config loading failure
- `generate_embeddings_for_document()` -- chunks file not found path
- `generate_embeddings_for_document()` -- empty chunks data path
- `generate_embeddings_for_document()` -- dict-with-"chunks"-key format vs array format
- `generate_embeddings_for_document()` -- embeddings list is empty after batch generation
- `generate_embeddings_for_document()` -- general exception handler
- `_generate_embeddings_batch()` -- OpenRouter path: OpenAIError (non-rate-limit) retry
- `_generate_embeddings_batch()` -- OpenRouter path: unexpected exception (non-OpenAI error)
- `_generate_embeddings_batch()` -- OpenRouter path: max retries exhausted
- `_generate_embeddings_batch()` -- OpenRouter path: MRL truncation for API-returned embeddings
- `_generate_embeddings_batch()` -- rate_limit_delay sleep between batches
- `_generate_embeddings_batch()` -- batch boundary behavior (texts split correctly across batches)
- `_save_embeddings()` -- remote storage existence check path
- `_save_embeddings()` -- storage.upload() call
- `_save_embeddings()` -- exception during save
- `_resolve_chunks_path()` -- multiple matches warning
- `_resolve_chunks_path()` -- fallback to local data directory
- `_resolve_chunks_path()` -- storage.glob() exception handling
- `_resolve_output_dir()` -- all fallback paths
- `_resolve_output_relative()` -- path transformation logic
- `process_all_documents()` -- no publications found path
- `process_all_documents()` -- document_ids filtering with missing IDs
- `process_all_documents()` -- per-document exception handling
- `process_all_documents()` -- general exception handler
- `run_embeddings_generator()` -- synchronous wrapper

#### Tautological / Mirror Tests
- `test_save_embeddings_format` and `test_sentence_transformer_save_format`: These tests create `ChunkEmbedding` objects with hardcoded `[0.1] * 1024` vectors, pass them to `_save_embeddings()`, then read the Parquet and verify the vectors are `[0.1] * 1024`. They test serialization/deserialization but never test that the embedding generation itself produces correct dimensionality. The actual `_generate_embeddings_batch()` -> `_save_embeddings()` pipeline is never tested end-to-end in unit tests.
- `test_resume_capability`: Calls `_save_embeddings()` twice and verifies the file still exists. This passes trivially because the second call returns early. A more meaningful test would verify the file contents are unchanged (not just that the file exists), or that the second call does not overwrite with different data.

#### Over-Mocked Tests
- `test_retry_mechanism_with_eventual_success`: This test correctly validates retry behavior but only for `RateLimitError`. The production code also handles `OpenAIError` with identical retry logic (lines 388-398) and a catch-all `Exception` that raises immediately (lines 400-402). Only one of three error paths is tested. Additionally, the mock response returns `[0.1] * 1024` for a single text, but the test never verifies that the returned embedding matches -- it only checks `len(embeddings) == 1`.

#### Missing Edge Cases
- **API returns embeddings with wrong dimensionality**: No test verifies behavior when the API returns vectors of unexpected length
- **API returns empty embedding list**: `response.data = []`
- **API returns NaN or Inf values in embedding vectors**: The MRL truncation includes `np.linalg.norm()` which would produce `nan` for division-by-zero with zero-norm vectors
- **Division by zero in MRL re-normalization**: If a truncated vector happens to be all zeros (extremely unlikely but possible), `norms` would be 0 and division would produce `inf`/`nan`
- **Very large batch (thousands of texts)**: Batching boundary behavior
- **Single text that exceeds API token limit**: No test for what happens when a single chunk is too large for the embedding API
- **Network timeout**: The `timeout` parameter is set but timeout errors are never tested
- **Malformed chunks.json**: Invalid JSON, missing required fields (`text_content`, `chunk_id`)
- **Chunks with empty text_content**: `""` passed to embedding API
- **Concurrent calls to `process_all_documents()`**: Race conditions on file writes
- **Storage without `glob`, `get_path`, or `upload` methods**: Various `hasattr` checks in path resolution

#### Missing Critical Tests
1. **OpenRouter MRL truncation path**: The `_generate_embeddings_batch()` method has separate MRL truncation code for OpenRouter responses (lines 404-409) that is never tested. This is a copy of the SentenceTransformer truncation logic but applied to API responses.
2. **Max retries exhausted**: When all retry attempts fail, the code re-raises the exception. This path is never tested. It is critical for understanding failure propagation.
3. **Batch splitting behavior**: For the OpenRouter path, texts are split into batches of `batch_size`. No test verifies that batching works correctly with, say, 65 texts at batch_size=32 (should produce 3 API calls with correct batch sizes).
4. **`generate_embeddings_for_document()` error paths**: The chunks-not-found, empty-chunks, and no-embeddings-generated failure paths are all untested. These are the most common failure modes in production.
5. **`run_embeddings_generator()` synchronous wrapper**: Never tested at all.

#### Unnecessary Tests
- `test_sentence_transformer_save_format`: This test is largely redundant with `test_save_embeddings_format`. Both test `_save_embeddings()` with the same pattern (create ChunkEmbedding objects, call save, verify Parquet + JSON). The only difference is the model name in metadata. These could be combined or the second removed.

---

## Priority Recommendations

1. **[CRITICAL] Fix the tautological Docling test**: `TestDoclingBackendUnit.test_extract_success` mocks the method it is testing. It provides zero coverage. Rewrite it to mock `_get_converter()` and `ConversionStatus`, then exercise the actual `extract()` logic including status checking, markdown export, page counting, and metadata extraction.

2. **[CRITICAL] Add unit tests for `_enforce_token_limits()` and `_force_split_by_tokens()`**: These are the last-resort safety net preventing embedding failures from oversized chunks. Test with a chunk that is 2x the max token limit and verify it is correctly split and reindexed.

3. **[CRITICAL] Add tests for embeddings generator error paths**: Test `generate_embeddings_for_document()` when chunks file is missing, when chunks JSON is empty, when batch generation returns no embeddings, and when an unexpected exception occurs. These are the most common production failure modes.

4. **[HIGH] Add a MarkerBackend success-path unit test**: Currently there is zero unit test coverage for a successful Marker extraction. Mock `_load_models()` and `_detect_marker_version()` to test both v0 and v1 code paths.

5. **[HIGH] Test OpenRouter MRL truncation and batching**: The API-side truncation code (lines 404-409 of embeddings_generator.py) is duplicated from the SentenceTransformer path but completely untested. Also test that a list of 65 texts produces the correct number of API calls with correct batch sizes.

6. **[HIGH] Test `_detect_sentences()` with adversarial input**: Feed it text without any punctuation (common in OCR output), text with abbreviations ("Dr.", "U.S.A."), decimal numbers ("3.14"), and URLs. The regex `([.!?;]+\s+)` will fail on many of these. Even if the behavior is "known bad," the test documents the limitation.

7. **[HIGH] Test PDF processor backend fallback mechanism**: `_init_backend()` tries the configured backend and falls back on ImportError. This is a critical reliability feature that is completely untested.

8. **[MEDIUM] Add embeddings generator retry exhaustion test**: Verify that when all retries fail, the exception propagates correctly and the document is marked as FAILED in the tracker.

9. **[MEDIUM] Test text chunker with unicode and non-Latin scripts**: Token counts for CJK characters, Arabic text, and emoji differ dramatically from English. The chunking logic assumes roughly consistent token-to-character ratios in several places (e.g., character position tracking).

10. **[MEDIUM] Test `_clean_text_for_chunking()` directly**: This function is called by every strategy but is never directly tested. Test with realistic OCR artifacts: repeated headers, garbled characters, mixed page marker formats.

11. **[MEDIUM] Remove or relocate `test_orchestrator_integration`**: This test in `test_text_chunker.py` tests the orchestrator, not the chunker. It belongs in an orchestrator test file.

12. **[LOW] Fix `test_short_document_vs_short_chunk_distinction`**: The assertion on line 798 compares character length against a token-based config value. This should use `chunk.token_count >= simple_config["min_chunk_size"]`.

13. **[LOW] Add `is_slurm_environment()` test**: Trivial function but completely untested. One-line test with `patch.dict(os.environ, {"SLURM_JOB_ID": "12345"})`.

14. **[LOW] Clean up `test_process_actual_sample_pdfs`**: Hardcoded absolute paths to a specific developer's machine. Should use a conftest fixture or be moved to a manual test directory.
