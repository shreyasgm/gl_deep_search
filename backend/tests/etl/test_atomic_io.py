"""
Tests for the atomic write helpers.

The pipeline treats "the artifact exists" as "the artifact is complete", so the
guarantee that matters is that a writer which dies mid-write leaves either the
previous file or nothing at all -- never a truncated file at the target path.
"""

import json

import pytest

from backend.etl.utils.atomic_io import (
    PARTIAL_SUFFIX,
    atomic_write,
    atomic_writer,
    partial_path_for,
    remove_partial,
)


def test_atomic_write_creates_file(tmp_path):
    """Text is written to the target path."""
    target = tmp_path / "out.txt"

    atomic_write(target, "hello")

    assert target.read_text(encoding="utf-8") == "hello"
    assert list(tmp_path.iterdir()) == [target]


def test_atomic_write_bytes(tmp_path):
    """Bytes select binary mode."""
    target = tmp_path / "out.bin"

    atomic_write(target, b"%PDF-1.5\n")

    assert target.read_bytes() == b"%PDF-1.5\n"


def test_atomic_write_creates_parent_directories(tmp_path):
    """Missing parent directories are created."""
    target = tmp_path / "a" / "b" / "out.txt"

    atomic_write(target, "nested")

    assert target.read_text(encoding="utf-8") == "nested"


def test_atomic_write_replaces_existing_file(tmp_path):
    """An existing file is replaced, leaving no temporary files behind."""
    target = tmp_path / "out.txt"
    target.write_text("old", encoding="utf-8")

    atomic_write(target, "new")

    assert target.read_text(encoding="utf-8") == "new"
    assert list(tmp_path.iterdir()) == [target]


def test_writer_failure_leaves_no_partial_file(tmp_path):
    """A writer that raises mid-write must not create the target file."""
    target = tmp_path / "out.txt"

    with pytest.raises(RuntimeError, match="boom"):
        with atomic_writer(target, "w") as handle:
            handle.write("half a document")
            raise RuntimeError("boom")

    assert not target.exists()
    assert list(tmp_path.iterdir()) == [], "temporary file was not cleaned up"


def test_writer_failure_leaves_existing_file_intact(tmp_path):
    """A failed rewrite must leave the previous complete file untouched."""
    target = tmp_path / "out.txt"
    target.write_text("original contents", encoding="utf-8")

    with pytest.raises(RuntimeError, match="boom"):
        with atomic_writer(target, "w") as handle:
            handle.write("clobbered")
            raise RuntimeError("boom")

    assert target.read_text(encoding="utf-8") == "original contents"
    assert list(tmp_path.iterdir()) == [target]


def test_writer_supports_json_dump(tmp_path):
    """json.dump works against the yielded handle."""
    target = tmp_path / "chunks.json"
    payload = [{"chunk_id": "a", "text": "héllo"}]

    with atomic_writer(target, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    assert json.loads(target.read_text(encoding="utf-8")) == payload


def test_writer_supports_parquet(tmp_path):
    """pyarrow can write through the yielded binary handle."""
    pd = pytest.importorskip("pandas")
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")

    target = tmp_path / "embeddings.parquet"
    table = pa.Table.from_pandas(
        pd.DataFrame({"chunk_id": ["a", "b"], "embedding": [[0.1, 0.2], [0.3, 0.4]]})
    )

    with atomic_writer(target, "wb") as handle:
        pq.write_table(table, handle, compression="snappy")

    round_tripped = pq.read_table(target).to_pandas()
    assert list(round_tripped["chunk_id"]) == ["a", "b"]
    assert list(tmp_path.iterdir()) == [target]


def test_writer_rejects_non_write_modes(tmp_path):
    """Read, append and update modes cannot be made atomic."""
    target = tmp_path / "out.txt"

    for mode in ("r", "rb", "a", "ab", "w+", "x"):
        with pytest.raises(ValueError):
            with atomic_writer(target, mode):
                pass


def test_writer_rejects_encoding_in_binary_mode(tmp_path):
    """Binary mode takes no encoding."""
    with pytest.raises(ValueError):
        with atomic_writer(tmp_path / "out.bin", "wb", encoding="utf-8"):
            pass


def test_partial_path_helpers(tmp_path):
    """The .part sidecar sits next to its destination and can be removed."""
    destination = tmp_path / "doc.pdf"
    part = partial_path_for(destination)

    assert part.name == f"doc.pdf{PARTIAL_SUFFIX}"
    assert part.parent == destination.parent

    part.write_bytes(b"partial")
    remove_partial(part)
    assert not part.exists()

    # Removing a file that is already gone is not an error
    remove_partial(part)
