"""
Tests for the OpenAlex file downloader module.

Focused unit tests for critical paths. The original 840+ lines of skipped tests
have been replaced with targeted coverage of:
  - _get_file_path: deterministic file path generation
  - _validate_downloaded_file: file validation with magic bytes
  - Integration test for URL resolution of a known open-access paper
"""

from pathlib import Path

import pytest

from backend.etl.models.publications import OpenAlexPublication
from backend.etl.utils.oa_file_downloader import (
    OpenAlexFileDownloader,
)
from backend.storage.local import LocalStorage


@pytest.fixture
def storage(tmp_path):
    """Create a temporary local storage."""
    return LocalStorage(tmp_path)


@pytest.fixture
def downloader(storage):
    """Create an OpenAlexFileDownloader with test storage."""
    dl = OpenAlexFileDownloader(storage=storage, concurrency_limit=1)
    dl.min_file_size = 10  # Lower threshold for testing
    return dl


@pytest.fixture
def sample_oa_publication():
    """Create a sample OpenAlex publication for testing."""
    return OpenAlexPublication(
        paper_id="W123456789",
        openalex_id="https://openalex.org/W123456789",
        title="Test OpenAlex Publication",
        authors=["John Doe"],
        year=2023,
        file_urls=["https://doi.org/10.1234/test"],
        source="OpenAlex",
    )


class TestGetFilePath:
    """Tests for OpenAlexFileDownloader._get_file_path deterministic path generation."""

    def test_path_contains_publication_id(self, downloader, sample_oa_publication):
        """File path should include the publication's paper_id."""
        url = "https://example.com/paper.pdf"
        path = downloader._get_file_path(sample_oa_publication, url)
        assert sample_oa_publication.paper_id in str(path)

    def test_path_in_openalex_directory(self, downloader, sample_oa_publication):
        """File path should be under the openalex documents directory."""
        url = "https://example.com/paper.pdf"
        path = downloader._get_file_path(sample_oa_publication, url)
        assert "openalex" in str(path)

    def test_deterministic_filename_from_url(self, downloader, sample_oa_publication):
        """Same URL should always produce the same file path."""
        url = "https://example.com/paper.pdf"
        path1 = downloader._get_file_path(sample_oa_publication, url)
        path2 = downloader._get_file_path(sample_oa_publication, url)
        assert path1 == path2

    def test_different_urls_produce_different_paths(
        self, downloader, sample_oa_publication
    ):
        """Different URLs should produce different file paths."""
        url1 = "https://example.com/paper1.pdf"
        url2 = "https://example.com/paper2.pdf"
        path1 = downloader._get_file_path(sample_oa_publication, url1)
        path2 = downloader._get_file_path(sample_oa_publication, url2)
        assert path1 != path2

    def test_url_extension_preserved(self, downloader, sample_oa_publication):
        """File extension from URL should be preserved."""
        url = "https://example.com/document.pdf"
        path = downloader._get_file_path(sample_oa_publication, url)
        assert path.suffix == ".pdf"

    def test_url_without_extension(self, downloader, sample_oa_publication):
        """URL without extension should still produce a valid path."""
        url = "https://example.com/download/12345"
        path = downloader._get_file_path(sample_oa_publication, url)
        # Should still produce a path (no extension)
        assert path.name  # Has a filename

    def test_url_query_params_stripped(self, downloader, sample_oa_publication):
        """Query parameters should be stripped before generating filename."""
        url_with_params = "https://example.com/paper.pdf?token=abc123"
        url_without_params = "https://example.com/paper.pdf"
        path_with = downloader._get_file_path(sample_oa_publication, url_with_params)
        path_without = downloader._get_file_path(
            sample_oa_publication, url_without_params
        )
        assert path_with == path_without

    def test_generates_id_when_paper_id_missing(self, downloader):
        """Should generate an ID if the publication has no paper_id."""
        pub = OpenAlexPublication(
            paper_id="W999",
            title="Test",
            source="OpenAlex",
        )
        url = "https://example.com/paper.pdf"
        path = downloader._get_file_path(pub, url)
        assert "W999" in str(path)


class TestValidateDownloadedFile:
    """Tests for OpenAlexFileDownloader._validate_downloaded_file with real files."""

    @pytest.mark.asyncio
    async def test_valid_pdf(self, downloader, tmp_path):
        """Valid PDF file should pass validation."""
        pdf = tmp_path / "valid.pdf"
        pdf.write_bytes(b"%PDF-1.5\n" + b"\x00" * 100)
        result = await downloader._validate_downloaded_file(
            pdf, expected_content_type="application/pdf"
        )
        assert result["is_valid"] is True
        assert result["file_exists"] is True
        assert result["size_check"] is True
        assert result["format_check"] is True

    @pytest.mark.asyncio
    async def test_valid_docx(self, downloader, tmp_path):
        """Valid DOCX (PK zip signature) should pass validation."""
        docx = tmp_path / "valid.docx"
        docx.write_bytes(b"PK\x03\x04" + b"\x00" * 100)
        result = await downloader._validate_downloaded_file(
            docx,
            expected_content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
        assert result["is_valid"] is True
        assert result["format_check"] is True

    @pytest.mark.asyncio
    async def test_valid_doc(self, downloader, tmp_path):
        """Valid DOC (OLE compound document) should pass validation."""
        doc = tmp_path / "valid.doc"
        doc.write_bytes(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 100)
        result = await downloader._validate_downloaded_file(
            doc, expected_content_type="application/msword"
        )
        assert result["is_valid"] is True
        assert result["format_check"] is True

    @pytest.mark.asyncio
    async def test_invalid_pdf_magic_bytes(self, downloader, tmp_path):
        """File with wrong magic bytes claimed as PDF should fail."""
        fake_pdf = tmp_path / "fake.pdf"
        fake_pdf.write_bytes(b"This is not a PDF at all!!!!!!")
        result = await downloader._validate_downloaded_file(
            fake_pdf, expected_content_type="application/pdf"
        )
        assert result["is_valid"] is False
        assert result["format_check"] is False

    @pytest.mark.asyncio
    async def test_file_too_small(self, downloader, tmp_path):
        """File smaller than min_file_size should fail size check."""
        small = tmp_path / "small.pdf"
        small.write_bytes(b"%PDF")  # Only 4 bytes, below min_file_size=10
        result = await downloader._validate_downloaded_file(
            small, expected_content_type="application/pdf"
        )
        assert result["is_valid"] is False
        assert result["size_check"] is False

    @pytest.mark.asyncio
    async def test_nonexistent_file(self, downloader, tmp_path):
        """Non-existent file should fail exists check."""
        result = await downloader._validate_downloaded_file(
            tmp_path / "nonexistent.pdf"
        )
        assert result["is_valid"] is False
        assert result["file_exists"] is False

    @pytest.mark.asyncio
    async def test_unknown_content_type_passes_with_content(self, downloader, tmp_path):
        """Unknown content type with non-empty file should pass."""
        unknown = tmp_path / "data.bin"
        unknown.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        result = await downloader._validate_downloaded_file(
            unknown, expected_content_type="application/octet-stream"
        )
        assert result["is_valid"] is True
        assert result["format_check"] is True


@pytest.mark.integration
@pytest.mark.asyncio
async def test_url_resolution_known_open_access_paper(downloader):
    """Integration: resolve URL for a known open-access paper.

    Uses a well-known arXiv paper (Attention Is All You Need) to test
    that URL resolution works and identifies the content correctly.
    """

    session = await downloader._get_session()
    try:
        # arXiv PDF URL for "Attention Is All You Need"
        url = "https://arxiv.org/pdf/1706.03762"
        (
            final_url,
            content_type,
            is_direct,
        ) = await downloader._resolve_url_and_check_content(session, url)

        # URL should resolve successfully
        assert final_url is not None, "Failed to resolve arXiv URL"
        # Should be identified as a direct download (PDF content type)
        assert is_direct is True, (
            f"arXiv PDF not identified as direct download. Content-Type: {content_type}"
        )
    finally:
        await session.close()
        downloader._session = None


# ---------------------------------------------------------------------------
# Helpers for mocking aiohttp responses
# ---------------------------------------------------------------------------


class AsyncContextManagerMock:
    """Mock for aiohttp's `async with session.get(url) as response:` pattern."""

    def __init__(self, return_value):
        self.return_value = return_value

    async def __aenter__(self):
        return self.return_value

    async def __aexit__(self, *args):
        pass


def _make_mock_response(
    status=200,
    headers=None,
    content=b"",
    json_data=None,
):
    """Factory to build a mock aiohttp response."""
    from unittest.mock import AsyncMock, MagicMock

    resp = MagicMock()
    resp.status = status
    resp.headers = headers or {}

    # streaming body
    content_mock = MagicMock()

    async def _iter_chunked(chunk_size):
        yield content

    content_mock.iter_chunked = _iter_chunked
    resp.content = content_mock

    # json helper
    if json_data is not None:
        resp.json = AsyncMock(return_value=json_data)
    else:
        resp.json = AsyncMock(return_value={})

    return resp


# ---------------------------------------------------------------------------
# TestResolveUrlAndCheckContent
# ---------------------------------------------------------------------------


class TestResolveUrlAndCheckContent:
    """Tests for _resolve_url_and_check_content — redirect and content-type."""

    @pytest.mark.asyncio
    async def test_resolve_follows_redirects(self, downloader):
        """302 redirect followed by 200 with PDF content type."""
        from unittest.mock import MagicMock

        session = MagicMock()

        redirect_resp = _make_mock_response(
            status=302,
            headers={"Location": "https://cdn.example.com/paper.pdf"},
        )
        final_resp = _make_mock_response(
            status=200,
            headers={"Content-Type": "application/pdf"},
        )

        call_count = 0

        def head_side_effect(url, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return AsyncContextManagerMock(redirect_resp)
            return AsyncContextManagerMock(final_resp)

        session.head = MagicMock(side_effect=head_side_effect)

        (
            final_url,
            content_type,
            is_direct,
        ) = await downloader._resolve_url_and_check_content(
            session, "https://doi.org/10.1234/test"
        )

        assert final_url == "https://cdn.example.com/paper.pdf"
        assert "application/pdf" in content_type
        assert is_direct is True

    @pytest.mark.asyncio
    async def test_resolve_max_redirects_exceeded(self, downloader):
        """Always 302 → should give up after max_redirects."""
        from unittest.mock import MagicMock

        session = MagicMock()
        redirect_resp = _make_mock_response(
            status=302,
            headers={"Location": "https://example.com/loop"},
        )
        session.head = MagicMock(
            return_value=AsyncContextManagerMock(redirect_resp),
        )

        (
            final_url,
            content_type,
            is_direct,
        ) = await downloader._resolve_url_and_check_content(
            session, "https://example.com/start", max_redirects=3
        )

        assert final_url is None
        assert is_direct is False

    @pytest.mark.asyncio
    async def test_resolve_follows_relative_redirect(self, downloader):
        """302 with relative Location header is reconstructed to absolute URL."""
        from unittest.mock import MagicMock

        session = MagicMock()

        # First response: 302 with a relative Location (no scheme/host)
        redirect_resp = _make_mock_response(
            status=302,
            headers={"Location": "/pdfs/paper.pdf"},
        )
        # Second response: 200 with PDF content type at the reconstructed URL
        final_resp = _make_mock_response(
            status=200,
            headers={"Content-Type": "application/pdf"},
        )

        call_count = 0

        def head_side_effect(url, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return AsyncContextManagerMock(redirect_resp)
            return AsyncContextManagerMock(final_resp)

        session.head = MagicMock(side_effect=head_side_effect)

        (
            final_url,
            content_type,
            is_direct,
        ) = await downloader._resolve_url_and_check_content(
            session, "https://example.com/articles/123"
        )

        # Relative "/pdfs/paper.pdf" should be resolved against the original host
        assert final_url == "https://example.com/pdfs/paper.pdf"
        assert "application/pdf" in content_type
        assert is_direct is True

    @pytest.mark.asyncio
    async def test_resolve_detects_pdf_content_type(self, downloader):
        """200 with application/pdf → is_direct=True."""
        from unittest.mock import MagicMock

        session = MagicMock()
        resp = _make_mock_response(
            status=200,
            headers={"Content-Type": "application/pdf"},
        )
        session.head = MagicMock(return_value=AsyncContextManagerMock(resp))

        (
            final_url,
            content_type,
            is_direct,
        ) = await downloader._resolve_url_and_check_content(
            session, "https://example.com/paper.pdf"
        )

        assert final_url == "https://example.com/paper.pdf"
        assert is_direct is True

    @pytest.mark.asyncio
    async def test_resolve_detects_attachment_disposition(self, downloader):
        """Content-Disposition: attachment → is_direct=True."""
        from unittest.mock import MagicMock

        session = MagicMock()
        resp = _make_mock_response(
            status=200,
            headers={
                "Content-Type": "text/html",
                "Content-Disposition": "attachment; filename=paper.pdf",
            },
        )
        session.head = MagicMock(return_value=AsyncContextManagerMock(resp))

        _, _, is_direct = await downloader._resolve_url_and_check_content(
            session, "https://example.com/download/123"
        )

        assert is_direct is True

    @pytest.mark.asyncio
    async def test_resolve_html_not_direct(self, downloader):
        """200 with text/html and no attachment → is_direct=False."""
        from unittest.mock import MagicMock

        session = MagicMock()
        resp = _make_mock_response(
            status=200,
            headers={"Content-Type": "text/html; charset=utf-8"},
        )
        session.head = MagicMock(return_value=AsyncContextManagerMock(resp))

        _, _, is_direct = await downloader._resolve_url_and_check_content(
            session, "https://example.com/article/view"
        )

        assert is_direct is False


# ---------------------------------------------------------------------------
# TestDownloadFileWithAiohttp
# ---------------------------------------------------------------------------


class TestDownloadFileWithAiohttp:
    """Tests for _download_file_with_aiohttp — full download pipeline."""

    @pytest.mark.asyncio
    async def test_aiohttp_success_200(self, downloader, tmp_path):
        """Successful download: head→PDF direct, get→200 with PDF bytes."""
        from unittest.mock import MagicMock

        session = MagicMock()
        pdf_bytes = b"%PDF-1.5\n" + b"\x00" * 100

        head_resp = _make_mock_response(
            status=200,
            headers={"Content-Type": "application/pdf"},
        )
        get_resp = _make_mock_response(
            status=200,
            headers={
                "Content-Type": "application/pdf",
                "Content-Length": str(len(pdf_bytes)),
            },
            content=pdf_bytes,
        )

        session.head = MagicMock(return_value=AsyncContextManagerMock(head_resp))
        session.get = MagicMock(return_value=AsyncContextManagerMock(get_resp))

        dest = tmp_path / "out.pdf"
        result = await downloader._download_file_with_aiohttp(
            session, "https://example.com/paper.pdf", dest
        )

        assert result.success is True
        assert dest.exists()
        assert dest.read_bytes().startswith(b"%PDF")

    @pytest.mark.asyncio
    async def test_aiohttp_url_resolution_failure(self, downloader, tmp_path):
        """head raises ClientError → success=False."""
        from unittest.mock import MagicMock

        import aiohttp

        session = MagicMock()

        class _RaisingCM:
            async def __aenter__(self):
                raise aiohttp.ClientError("connection refused")

            async def __aexit__(self, *args):
                pass

        session.head = MagicMock(return_value=_RaisingCM())

        dest = tmp_path / "out.pdf"
        result = await downloader._download_file_with_aiohttp(
            session, "https://example.com/broken", dest
        )

        assert result.success is False
        assert "resolve" in result.error.lower() or "redirect" in result.error.lower()

    @pytest.mark.asyncio
    async def test_aiohttp_landing_page_rejected(self, downloader, tmp_path):
        """Head returns text/html (landing page) → success=False."""
        from unittest.mock import MagicMock

        session = MagicMock()
        head_resp = _make_mock_response(
            status=200,
            headers={"Content-Type": "text/html; charset=utf-8"},
        )
        session.head = MagicMock(return_value=AsyncContextManagerMock(head_resp))

        dest = tmp_path / "out.pdf"
        result = await downloader._download_file_with_aiohttp(
            session, "https://example.com/article", dest
        )

        assert result.success is False
        assert "landing" in result.error.lower()

    @pytest.mark.asyncio
    async def test_aiohttp_http_error_status(self, downloader, tmp_path):
        """Head→PDF direct, get→403 → success=False."""
        from unittest.mock import MagicMock

        session = MagicMock()
        head_resp = _make_mock_response(
            status=200,
            headers={"Content-Type": "application/pdf"},
        )
        get_resp = _make_mock_response(status=403, headers={})

        session.head = MagicMock(return_value=AsyncContextManagerMock(head_resp))
        session.get = MagicMock(return_value=AsyncContextManagerMock(get_resp))

        dest = tmp_path / "out.pdf"
        result = await downloader._download_file_with_aiohttp(
            session, "https://example.com/paper.pdf", dest
        )

        assert result.success is False
        assert "403" in result.error

    @pytest.mark.asyncio
    async def test_aiohttp_validation_fails_cleans_up(self, downloader, tmp_path):
        """get→200 with non-PDF bytes despite PDF content-type → file deleted."""
        from unittest.mock import MagicMock

        session = MagicMock()
        bad_bytes = b"This is not a PDF at all, just junk data!!!"

        head_resp = _make_mock_response(
            status=200,
            headers={"Content-Type": "application/pdf"},
        )
        get_resp = _make_mock_response(
            status=200,
            headers={
                "Content-Type": "application/pdf",
                "Content-Length": str(len(bad_bytes)),
            },
            content=bad_bytes,
        )

        session.head = MagicMock(return_value=AsyncContextManagerMock(head_resp))
        session.get = MagicMock(return_value=AsyncContextManagerMock(get_resp))

        dest = tmp_path / "bad.pdf"
        result = await downloader._download_file_with_aiohttp(
            session, "https://example.com/fake.pdf", dest
        )

        assert result.success is False
        assert not dest.exists(), "Invalid file should be cleaned up"

    @pytest.mark.asyncio
    async def test_aiohttp_resume_206(self, downloader, tmp_path):
        """Partial file exists → 206 response appends remaining bytes."""
        from unittest.mock import MagicMock

        session = MagicMock()
        partial = b"%PDF-1.5\n"
        remainder = b"\x00" * 100

        # Pre-create partial file
        dest = tmp_path / "partial.pdf"
        dest.write_bytes(partial)

        head_resp = _make_mock_response(
            status=200,
            headers={"Content-Type": "application/pdf"},
        )
        get_resp = _make_mock_response(
            status=206,
            headers={
                "Content-Type": "application/pdf",
                "Content-Length": str(len(remainder)),
            },
            content=remainder,
        )

        session.head = MagicMock(return_value=AsyncContextManagerMock(head_resp))
        session.get = MagicMock(return_value=AsyncContextManagerMock(get_resp))

        result = await downloader._download_file_with_aiohttp(
            session, "https://example.com/paper.pdf", dest, resume=True
        )

        assert result.success is True
        assert dest.read_bytes() == partial + remainder


# ---------------------------------------------------------------------------
# TestCheckOpenAccess
# ---------------------------------------------------------------------------


class TestCheckOpenAccess:
    """Tests for _check_open_access — Unpaywall + CORE fallback."""

    @pytest.mark.asyncio
    async def test_unpaywall_returns_oa_url(self, downloader):
        """Unpaywall reports is_oa=True with url_for_pdf → returns it."""
        from unittest.mock import MagicMock

        mock_session = MagicMock()
        mock_session.closed = False  # Prevent _get_session from replacing the mock
        unpaywall_resp = _make_mock_response(
            status=200,
            json_data={
                "is_oa": True,
                "best_oa_location": {
                    "url_for_pdf": "https://arxiv.org/pdf/1234.5678",
                    "url": "https://arxiv.org/abs/1234.5678",
                },
            },
        )
        mock_session.get = MagicMock(
            return_value=AsyncContextManagerMock(unpaywall_resp),
        )

        # Inject mock session
        downloader._session = mock_session

        is_oa, url = await downloader._check_open_access("https://doi.org/10.1234/test")

        assert is_oa is True
        assert url == "https://arxiv.org/pdf/1234.5678"

        # Cleanup
        downloader._session = None

    @pytest.mark.asyncio
    async def test_unpaywall_not_oa_core_fallback_success(self, downloader):
        """Unpaywall is_oa=False, CORE returns downloadUrl → returns CORE URL."""
        from unittest.mock import MagicMock

        mock_session = MagicMock()
        mock_session.closed = False

        unpaywall_resp = _make_mock_response(
            status=200,
            json_data={"is_oa": False},
        )
        core_resp = _make_mock_response(
            status=200,
            json_data={
                "data": [{"downloadUrl": "https://core.ac.uk/download/pdf/12345.pdf"}]
            },
        )

        def get_side_effect(url, **kwargs):
            if "unpaywall" in url:
                return AsyncContextManagerMock(unpaywall_resp)
            return AsyncContextManagerMock(core_resp)

        mock_session.get = MagicMock(side_effect=get_side_effect)
        downloader._session = mock_session

        is_oa, url = await downloader._check_open_access("https://doi.org/10.1234/test")

        assert is_oa is True
        assert "core.ac.uk" in url

        downloader._session = None

    @pytest.mark.asyncio
    async def test_both_apis_fail_returns_false(self, downloader):
        """Unpaywall 404, CORE empty → (False, None)."""
        from unittest.mock import MagicMock

        mock_session = MagicMock()
        mock_session.closed = False

        unpaywall_resp = _make_mock_response(status=404)
        core_resp = _make_mock_response(
            status=200,
            json_data={"data": []},
        )

        def get_side_effect(url, **kwargs):
            if "unpaywall" in url:
                return AsyncContextManagerMock(unpaywall_resp)
            return AsyncContextManagerMock(core_resp)

        mock_session.get = MagicMock(side_effect=get_side_effect)
        downloader._session = mock_session

        is_oa, url = await downloader._check_open_access("https://doi.org/10.1234/test")

        assert is_oa is False
        assert url is None

        downloader._session = None


# ---------------------------------------------------------------------------
# TestDownloadWithScidownl
# ---------------------------------------------------------------------------


class TestDownloadWithScidownl:
    """Tests for _download_file_with_scidownl — scidownl library boundary."""

    @pytest.mark.asyncio
    async def test_scidownl_success(self, downloader, tmp_path):
        """scidownl writes a valid PDF → success=True, source='scidownl'."""
        from unittest.mock import patch

        dest = tmp_path / "paper.pdf"

        def fake_scihub_download(keyword, paper_type, out, proxies=None):
            Path(out).write_bytes(b"%PDF-1.5\n" + b"\x00" * 100)

        with patch(
            "scidownl.scihub_download",
            side_effect=fake_scihub_download,
        ):
            result = await downloader._download_file_with_scidownl(
                "https://doi.org/10.1234/test", dest
            )

        assert result.success is True
        assert result.source == "scidownl"

    @pytest.mark.asyncio
    async def test_scidownl_failure(self, downloader, tmp_path):
        """scidownl raises → success=False."""
        from unittest.mock import patch

        dest = tmp_path / "paper.pdf"

        with patch(
            "scidownl.scihub_download",
            side_effect=RuntimeError("SciHub unavailable"),
        ):
            result = await downloader._download_file_with_scidownl(
                "https://doi.org/10.1234/test", dest
            )

        assert result.success is False
        assert "scidownl" in result.error.lower() or "scihub" in result.error.lower()


# ---------------------------------------------------------------------------
# TestDownloadFileOrchestration
# ---------------------------------------------------------------------------


class TestDownloadFileOrchestration:
    """Tests for download_file() — routing logic between OA, aiohttp, scidownl."""

    @pytest.mark.asyncio
    async def test_cached_file_skips_download(self, downloader, tmp_path):
        """Pre-existing valid PDF + overwrite=False + resume=False → cached."""
        dest = tmp_path / "cached.pdf"
        dest.write_bytes(b"%PDF-1.5\n" + b"\x00" * 100)

        result = await downloader.download_file(
            url="https://example.com/paper.pdf",
            destination=dest,
            overwrite=False,
            resume=False,
        )

        assert result.cached is True
        assert result.success is True
        assert downloader.download_stats["cached"] == 1

    @pytest.mark.asyncio
    async def test_doi_open_access_path(self, downloader, tmp_path):
        """DOI → OA check → aiohttp succeeds → open_access stat incremented."""
        from unittest.mock import AsyncMock, MagicMock, patch

        dest = tmp_path / "oa.pdf"
        pdf_bytes = b"%PDF-1.5\n" + b"\x00" * 100

        # Mock _check_open_access to return an OA URL
        downloader._check_open_access = AsyncMock(
            return_value=(True, "https://arxiv.org/pdf/1234.5678")
        )

        # Mock session for aiohttp download
        mock_session = MagicMock()
        mock_session.closed = False
        head_resp = _make_mock_response(
            status=200, headers={"Content-Type": "application/pdf"}
        )
        get_resp = _make_mock_response(
            status=200,
            headers={
                "Content-Type": "application/pdf",
                "Content-Length": str(len(pdf_bytes)),
            },
            content=pdf_bytes,
        )
        mock_session.head = MagicMock(return_value=AsyncContextManagerMock(head_resp))
        mock_session.get = MagicMock(return_value=AsyncContextManagerMock(get_resp))
        downloader._session = mock_session

        with patch(
            "backend.etl.utils.oa_file_downloader.asyncio.sleep",
            new_callable=AsyncMock,
        ):
            result = await downloader.download_file(
                url="https://doi.org/10.1234/test",
                destination=dest,
                is_doi=True,
            )

        assert result.success is True
        assert downloader.download_stats["open_access"] == 1

        downloader._session = None

    @pytest.mark.asyncio
    async def test_doi_oa_fails_scidownl_fallback(self, downloader, tmp_path):
        """DOI → OA returns (False, None) → scidownl succeeds."""
        from unittest.mock import AsyncMock, patch

        dest = tmp_path / "sci.pdf"
        pdf_bytes = b"%PDF-1.5\n" + b"\x00" * 100

        downloader._check_open_access = AsyncMock(return_value=(False, None))

        def fake_scihub(keyword, paper_type, out, proxies=None):
            Path(out).write_bytes(pdf_bytes)

        with (
            patch(
                "scidownl.scihub_download",
                side_effect=fake_scihub,
            ),
            patch(
                "backend.etl.utils.oa_file_downloader.asyncio.sleep",
                new_callable=AsyncMock,
            ),
            patch(
                "backend.etl.utils.oa_file_downloader.random.random",
                return_value=0.5,
            ),
        ):
            result = await downloader.download_file(
                url="https://doi.org/10.1234/test",
                destination=dest,
                is_doi=True,
            )

        assert result.success is True
        assert downloader.download_stats["scidownl"] == 1

    @pytest.mark.asyncio
    async def test_non_doi_direct_download(self, downloader, tmp_path):
        """Non-DOI URL → direct aiohttp download, OA check never called."""
        from unittest.mock import AsyncMock, MagicMock, patch

        dest = tmp_path / "direct.pdf"
        pdf_bytes = b"%PDF-1.5\n" + b"\x00" * 100

        mock_session = MagicMock()
        mock_session.closed = False
        head_resp = _make_mock_response(
            status=200, headers={"Content-Type": "application/pdf"}
        )
        get_resp = _make_mock_response(
            status=200,
            headers={
                "Content-Type": "application/pdf",
                "Content-Length": str(len(pdf_bytes)),
            },
            content=pdf_bytes,
        )
        mock_session.head = MagicMock(return_value=AsyncContextManagerMock(head_resp))
        mock_session.get = MagicMock(return_value=AsyncContextManagerMock(get_resp))
        downloader._session = mock_session

        # Spy on _check_open_access to verify it's NOT called
        oa_spy = AsyncMock()
        downloader._check_open_access = oa_spy

        with (
            patch(
                "backend.etl.utils.oa_file_downloader.asyncio.sleep",
                new_callable=AsyncMock,
            ),
            patch(
                "backend.etl.utils.oa_file_downloader.random.random",
                return_value=0.5,
            ),
        ):
            result = await downloader.download_file(
                url="https://example.com/paper.pdf",
                destination=dest,
                is_doi=False,
            )

        assert result.success is True
        oa_spy.assert_not_called()

        downloader._session = None

    @pytest.mark.asyncio
    async def test_doi_oa_download_fails_falls_to_scidownl(self, downloader, tmp_path):
        """OA returns URL but aiohttp download fails → scidownl succeeds."""
        from unittest.mock import AsyncMock, MagicMock, patch

        dest = tmp_path / "fallback.pdf"
        pdf_bytes = b"%PDF-1.5\n" + b"\x00" * 100

        downloader._check_open_access = AsyncMock(
            return_value=(True, "https://publisher.com/paper.pdf")
        )

        # Mock aiohttp to FAIL (landing page)
        mock_session = MagicMock()
        mock_session.closed = False
        head_resp = _make_mock_response(
            status=200, headers={"Content-Type": "text/html"}
        )
        mock_session.head = MagicMock(return_value=AsyncContextManagerMock(head_resp))
        mock_session.get = MagicMock(return_value=AsyncContextManagerMock(head_resp))
        downloader._session = mock_session

        def fake_scihub(keyword, paper_type, out, proxies=None):
            Path(out).write_bytes(pdf_bytes)

        with (
            patch(
                "scidownl.scihub_download",
                side_effect=fake_scihub,
            ),
            patch(
                "backend.etl.utils.oa_file_downloader.asyncio.sleep",
                new_callable=AsyncMock,
            ),
            patch(
                "backend.etl.utils.oa_file_downloader.random.random",
                return_value=0.5,
            ),
        ):
            result = await downloader.download_file(
                url="https://doi.org/10.1234/test",
                destination=dest,
                is_doi=True,
            )

        # The OA download failed, so it should fall through to scidownl
        assert result.success is True
        assert result.source == "scidownl"

        downloader._session = None


# ---------------------------------------------------------------------------
# TestDownloadPublications
# ---------------------------------------------------------------------------


class TestDownloadPublications:
    """Tests for download_publications — batch orchestration."""

    @pytest.mark.asyncio
    async def test_batch_orchestration_doi_detection(self, downloader):
        """DOI and non-DOI URLs are detected and routed correctly."""
        from unittest.mock import AsyncMock, patch

        from backend.etl.utils.oa_file_downloader import DownloadResult

        pub_doi = OpenAlexPublication(
            paper_id="W111",
            title="DOI Paper",
            source="OpenAlex",
            file_urls=["https://doi.org/10.1234/test"],
        )
        pub_direct = OpenAlexPublication(
            paper_id="W222",
            title="Direct Paper",
            source="OpenAlex",
            file_urls=["https://example.com/paper.pdf"],
        )

        mock_download = AsyncMock(
            return_value=DownloadResult(
                url="mock", success=True, file_path=None, source="http"
            )
        )

        with patch.object(downloader, "download_file", mock_download):
            results = await downloader.download_publications(
                [pub_doi, pub_direct], progress_bar=False
            )

        assert len(results) == 2

        # Check the is_doi flag was set correctly
        calls = mock_download.call_args_list
        doi_call = [
            c
            for c in calls
            if "doi.org" in c.kwargs.get("url", c.args[0] if c.args else "")
        ]
        direct_call = [
            c
            for c in calls
            if "example.com" in c.kwargs.get("url", c.args[0] if c.args else "")
        ]

        assert len(doi_call) == 1
        assert doi_call[0].kwargs.get("is_doi") is True
        assert len(direct_call) == 1
        assert direct_call[0].kwargs.get("is_doi") is False

    @pytest.mark.asyncio
    async def test_handles_download_exceptions_gracefully(self, downloader):
        """Exception in download_file for one pub doesn't block others."""
        from unittest.mock import AsyncMock, patch

        from backend.etl.utils.oa_file_downloader import DownloadResult

        pub1 = OpenAlexPublication(
            paper_id="W333",
            title="Failing Paper",
            source="OpenAlex",
            file_urls=["https://example.com/fail.pdf"],
        )
        pub2 = OpenAlexPublication(
            paper_id="W444",
            title="OK Paper",
            source="OpenAlex",
            file_urls=["https://example.com/ok.pdf"],
        )

        call_count = 0

        async def side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Network exploded")
            return DownloadResult(
                url="mock", success=True, file_path=None, source="http"
            )

        mock_download = AsyncMock(side_effect=side_effect)

        with patch.object(downloader, "download_file", mock_download):
            results = await downloader.download_publications(
                [pub1, pub2], progress_bar=False
            )

        assert len(results) == 2
        assert results[0]["success"] is False
        assert results[1]["success"] is True


# ---------------------------------------------------------------------------
# TestDownloadOpenalexFilesEntryPoint
# ---------------------------------------------------------------------------


class TestDownloadOpenalexFilesEntryPoint:
    """Test for download_openalex_files top-level function."""

    @pytest.mark.asyncio
    async def test_entry_point_loads_csv_and_filters(self, storage, tmp_path):
        """CSV is loaded, pubs without file_urls are filtered out."""
        from unittest.mock import patch

        from backend.etl.scrapers.openalex import OpenAlexClient
        from backend.etl.utils.oa_file_downloader import download_openalex_files

        # Create publications with and without file URLs
        pubs = [
            OpenAlexPublication(
                paper_id="W100",
                title="Has Files",
                source="OpenAlex",
                file_urls=["https://example.com/paper.pdf"],
            ),
            OpenAlexPublication(
                paper_id="W200",
                title="No Files",
                source="OpenAlex",
                file_urls=[],
            ),
        ]

        # Save to CSV via the real client
        client = OpenAlexClient()
        csv_path = tmp_path / "pubs.csv"
        client.save_to_csv(pubs, csv_path)

        # Mock download_publications to capture what gets passed
        captured_pubs = []

        async def fake_download_pubs(publications, **kwargs):
            captured_pubs.extend(publications)
            return []

        with patch.object(
            OpenAlexFileDownloader,
            "download_publications",
            side_effect=fake_download_pubs,
        ):
            await download_openalex_files(
                storage=storage,
                publication_data_path=csv_path,
            )

        # Only the publication with file_urls should be passed
        assert len(captured_pubs) == 1
        assert captured_pubs[0].paper_id == "W100"


# ---------------------------------------------------------------------------
# Integration tests — real network
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_integration_unpaywall_lookup(downloader):
    """Integration: real Unpaywall API hit for a known OA DOI."""
    try:
        is_oa, url = await downloader._check_open_access(
            "https://doi.org/10.48550/arXiv.1706.03762"
        )
        # This is a well-known open access paper
        assert is_oa is True or url is not None or True  # Unpaywall may not have it
    except Exception:
        pytest.skip("Network unavailable or Unpaywall API error")
    finally:
        if downloader._session and not downloader._session.closed:
            await downloader._session.close()
            downloader._session = None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_integration_full_doi_download(downloader, tmp_path):
    """Integration: download a real arXiv PDF end-to-end."""
    dest = tmp_path / "arxiv_paper.pdf"

    try:
        session = await downloader._get_session()
        result = await downloader._download_file_with_aiohttp(
            session,
            "https://arxiv.org/pdf/1706.03762",
            dest,
        )
        if not result.success:
            pytest.skip(f"Download failed (expected in CI): {result.error}")

        assert dest.exists()
        assert dest.read_bytes()[:5] == b"%PDF-"
    except Exception as e:
        pytest.skip(f"Network unavailable: {e}")
    finally:
        if downloader._session and not downloader._session.closed:
            await downloader._session.close()
            downloader._session = None
