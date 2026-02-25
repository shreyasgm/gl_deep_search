"""Tests for the storage layer: factory, local, and cloud implementations."""

from unittest.mock import MagicMock, patch

import pytest

from backend.storage.cloud import CloudStorage
from backend.storage.local import LocalStorage

# ===========================================================================
# LocalStorage tests — no mocks needed, uses tmp_path
# ===========================================================================


class TestLocalStorage:
    """Tests for LocalStorage using real filesystem via tmp_path."""

    def test_get_path(self, tmp_path):
        storage = LocalStorage(tmp_path)
        result = storage.get_path("subdir/file.txt")
        assert result == tmp_path / "subdir/file.txt"

    def test_exists_true_for_existing_file(self, tmp_path):
        storage = LocalStorage(tmp_path)
        (tmp_path / "existing.txt").write_text("hello")
        assert storage.exists("existing.txt") is True

    def test_exists_false_for_missing_file(self, tmp_path):
        storage = LocalStorage(tmp_path)
        assert storage.exists("nonexistent.txt") is False

    def test_glob_finds_matching_files(self, tmp_path):
        storage = LocalStorage(tmp_path)
        subdir = tmp_path / "data"
        subdir.mkdir()
        (subdir / "file1.json").write_text("{}")
        (subdir / "file2.json").write_text("{}")
        (subdir / "file3.txt").write_text("text")

        results = storage.glob("data/*.json")
        assert len(results) == 2
        assert all(r.endswith(".json") for r in results)

    def test_glob_returns_relative_paths(self, tmp_path):
        storage = LocalStorage(tmp_path)
        (tmp_path / "readme.md").write_text("# Readme")

        results = storage.glob("*.md")
        assert results == ["readme.md"]

    def test_ensure_dir_creates_nested_directory(self, tmp_path):
        storage = LocalStorage(tmp_path)
        new_dir = tmp_path / "a" / "b" / "c"
        storage.ensure_dir(new_dir)
        assert new_dir.exists()
        assert new_dir.is_dir()

    def test_upload_is_noop(self, tmp_path):
        storage = LocalStorage(tmp_path)
        # Should not raise
        storage.upload("anything.txt")

    def test_download_returns_local_path(self, tmp_path):
        storage = LocalStorage(tmp_path)
        (tmp_path / "data.csv").write_text("a,b,c")
        result = storage.download("data.csv")
        assert result == tmp_path / "data.csv"

    def test_list_files_without_pattern(self, tmp_path):
        storage = LocalStorage(tmp_path)
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.txt").write_text("b")
        subdir = tmp_path / "subdir"
        subdir.mkdir()

        files = storage.list_files()
        # Should return only files, not directories
        assert len(files) == 2
        assert all(f.is_file() for f in files)

    def test_list_files_with_pattern(self, tmp_path):
        storage = LocalStorage(tmp_path)
        (tmp_path / "a.json").write_text("{}")
        (tmp_path / "b.txt").write_text("text")

        files = storage.list_files("*.json")
        assert len(files) == 1
        assert files[0].name == "a.json"


# ===========================================================================
# CloudStorage tests — mock GCS client
# ===========================================================================


class TestCloudStorage:
    """Tests for CloudStorage with mocked GCS client."""

    @pytest.fixture
    def cloud_storage(self):
        """Create a CloudStorage instance with mocked GCS client."""
        with patch("backend.storage.cloud.storage.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_bucket = MagicMock()
            mock_client.bucket.return_value = mock_bucket

            cs = CloudStorage("test-bucket", base_prefix="prefix")
            # Expose mocks for assertions
            cs._mock_client = mock_client
            cs._mock_bucket = mock_bucket
            return cs

    # -- download() tests --

    def test_download_single_file(self, cloud_storage):
        """Download an exact blob match (single file)."""
        mock_blob = MagicMock()
        mock_blob.exists.return_value = True
        cloud_storage._mock_bucket.blob.return_value = mock_blob

        result = cloud_storage.download("data/file.txt")

        assert result == cloud_storage.local_cache / "data/file.txt"
        mock_blob.download_to_filename.assert_called_once()

    def test_download_directory_prefix(self, cloud_storage):
        """Download a directory prefix (multiple blobs)."""
        # First blob.exists() returns False (not a single file)
        mock_blob = MagicMock()
        mock_blob.exists.return_value = False
        cloud_storage._mock_bucket.blob.return_value = mock_blob

        # list_blobs returns some blobs under the prefix
        blob1 = MagicMock()
        blob1.name = "prefix/data/dir/file1.txt"
        blob1.exists.return_value = True
        # Not a directory marker
        type(blob1).name = property(lambda self: "prefix/data/dir/file1.txt")

        blob2 = MagicMock()
        type(blob2).name = property(lambda self: "prefix/data/dir/file2.txt")

        cloud_storage._mock_bucket.list_blobs.return_value = [blob1, blob2]

        result = cloud_storage.download("data/dir")

        assert result == cloud_storage.local_cache / "data/dir"
        # Both blobs should be downloaded
        assert blob1.download_to_filename.called
        assert blob2.download_to_filename.called

    def test_download_nothing_found(self, cloud_storage):
        """When neither exact blob nor prefix blobs exist."""
        mock_blob = MagicMock()
        mock_blob.exists.return_value = False
        cloud_storage._mock_bucket.blob.return_value = mock_blob

        cloud_storage._mock_bucket.list_blobs.return_value = []

        result = cloud_storage.download("nonexistent/path")
        # Returns the local path even though nothing was downloaded
        assert result == cloud_storage.local_cache / "nonexistent/path"

    # -- exists() tests --

    def test_exists_exact_blob(self, cloud_storage):
        """Exact blob exists returns True."""
        mock_blob = MagicMock()
        mock_blob.exists.return_value = True
        cloud_storage._mock_bucket.blob.return_value = mock_blob

        assert cloud_storage.exists("some/file.txt") is True

    def test_exists_directory_prefix(self, cloud_storage):
        """When exact blob doesn't exist, but blobs exist under prefix."""
        mock_blob = MagicMock()
        mock_blob.exists.return_value = False
        cloud_storage._mock_bucket.blob.return_value = mock_blob

        # list_blobs returns at least one result
        sub_blob = MagicMock()
        cloud_storage._mock_bucket.list_blobs.return_value = iter([sub_blob])

        assert cloud_storage.exists("some/dir") is True

    def test_exists_nothing(self, cloud_storage):
        """Neither blob nor prefix match."""
        mock_blob = MagicMock()
        mock_blob.exists.return_value = False
        cloud_storage._mock_bucket.blob.return_value = mock_blob

        cloud_storage._mock_bucket.list_blobs.return_value = iter([])

        assert cloud_storage.exists("ghost/path") is False

    # -- glob() tests --

    def test_glob_prefix_optimization(self, cloud_storage):
        """Glob extracts static prefix from the pattern to narrow listing."""
        blob1 = MagicMock()
        type(blob1).name = property(lambda self: "prefix/data/docs/file1.json")

        blob2 = MagicMock()
        type(blob2).name = property(lambda self: "prefix/data/docs/file2.json")

        blob3 = MagicMock()
        type(blob3).name = property(lambda self: "prefix/data/docs/subdir/")

        cloud_storage._mock_bucket.list_blobs.return_value = [blob1, blob2, blob3]

        results = cloud_storage.glob("data/docs/*.json")

        # Should have called list_blobs with optimized prefix
        call_kwargs = cloud_storage._mock_bucket.list_blobs.call_args
        listing_prefix = call_kwargs.kwargs.get("prefix") or call_kwargs[1].get(
            "prefix"
        )
        assert "data/docs/" in listing_prefix

        # Should return matching files (not directory markers)
        assert len(results) == 2
        assert "data/docs/file1.json" in results
        assert "data/docs/file2.json" in results

    def test_get_path_returns_local_cache_path(self, cloud_storage):
        result = cloud_storage.get_path("some/file.txt")
        assert result == cloud_storage.local_cache / "some/file.txt"

    def test_ensure_dir_creates_local_directory(self, cloud_storage):
        new_dir = cloud_storage.local_cache / "new" / "dir"
        cloud_storage.ensure_dir(new_dir)
        assert new_dir.exists()


# ===========================================================================
# StorageFactory tests
# ===========================================================================


class TestStorageFactory:
    """Tests for StorageFactory.create_storage()."""

    def test_explicit_local_returns_local_storage(self, tmp_path):
        """Explicitly requesting 'local' returns LocalStorage."""
        from backend.storage.factory import StorageFactory

        config_path = tmp_path / "config.yaml"
        config_path.write_text("runtime:\n  local_storage_path: /tmp/test_storage\n")

        storage = StorageFactory.create_storage(
            config_path=config_path, storage_type="local"
        )
        assert isinstance(storage, LocalStorage)

    def test_explicit_cloud_returns_cloud_storage(self, tmp_path):
        """Explicitly requesting 'cloud' returns CloudStorage."""
        from backend.storage.factory import StorageFactory

        config_path = tmp_path / "config.yaml"
        config_path.write_text("runtime:\n  gcs_bucket: my-bucket\n")

        with patch("backend.storage.cloud.storage.Client"):
            storage = StorageFactory.create_storage(
                config_path=config_path, storage_type="cloud"
            )
            assert isinstance(storage, CloudStorage)

    def test_auto_detect_cloud_from_env(self, tmp_path):
        """When CLOUD_ENVIRONMENT is set, auto-detect returns cloud."""
        from backend.storage.factory import StorageFactory

        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            "runtime:\n  detect_automatically: true\n  gcs_bucket: test-bucket\n"
        )

        with (
            patch.dict("os.environ", {"CLOUD_ENVIRONMENT": "gcp"}),
            patch("backend.storage.cloud.storage.Client"),
        ):
            storage = StorageFactory.create_storage(config_path=config_path)
            assert isinstance(storage, CloudStorage)

    def test_auto_detect_local_without_env(self, tmp_path):
        """Without CLOUD_ENVIRONMENT, auto-detect returns local."""
        from backend.storage.factory import StorageFactory

        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            "runtime:\n  detect_automatically: true\n  local_storage_path: /tmp/test\n"
        )

        with patch.dict("os.environ", {}, clear=True):
            storage = StorageFactory.create_storage(config_path=config_path)
            assert isinstance(storage, LocalStorage)

    def test_missing_config_uses_fallback_defaults(self):
        """When config file doesn't exist, use fallback defaults."""
        from backend.storage.factory import StorageFactory

        storage = StorageFactory.create_storage(
            config_path="/nonexistent/path/config.yaml", storage_type="local"
        )
        assert isinstance(storage, LocalStorage)
