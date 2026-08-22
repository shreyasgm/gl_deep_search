"""Atomic filesystem writes for ETL artifacts.

The ETL pipeline runs as a long batch job that is expected to be killed and
resumed.  Resume checks treat "the file exists" as "the file is complete", so a
process that dies part-way through a write leaves a truncated artifact that is
never revisited.

Every helper here writes to a temporary file *in the same directory* as the
target, flushes it to disk, and then :func:`os.replace` it into place.  Same
directory matters: :func:`os.replace` is only atomic within a single
filesystem, so a temp file under ``/tmp`` would silently degrade to a
copy-then-rename.  Readers therefore only ever see the old file or the complete
new one, never a half-written one.
"""

import os
import stat
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import IO, Any

# Permission bits applied to newly created artifacts.  ``tempfile.mkstemp``
# creates files as 0600; without this, atomically written files would be less
# readable than files written with a plain ``open()``.
DEFAULT_FILE_MODE = 0o644

# Suffix for the sidecar file a streaming download writes into.  Downloads
# cannot buffer in memory, so they stream to ``<destination>.part`` and rename
# once the transfer is complete and verified.  Nothing downstream may treat a
# ``.part`` file as finished data.
PARTIAL_SUFFIX = ".part"


def partial_path_for(destination: str | Path) -> Path:
    """Return the sidecar path a download streams into before it completes.

    Args:
        destination: Final path the download will end up at.

    Returns:
        ``destination`` with :data:`PARTIAL_SUFFIX` appended, in the same
        directory so the eventual rename stays atomic.
    """
    target = Path(destination)
    return target.with_name(target.name + PARTIAL_SUFFIX)


def remove_partial(path: str | Path) -> None:
    """Delete a partial-download sidecar, ignoring a missing file.

    Args:
        path: Path to the ``.part`` file to remove.
    """
    with suppress(OSError):
        Path(path).unlink()


def _resolve_file_mode(target: Path) -> int:
    """Choose the permission bits for the file that will replace ``target``.

    Args:
        target: Final path the temporary file will be renamed to.

    Returns:
        Permission bits of the existing target when it exists, otherwise
        :data:`DEFAULT_FILE_MODE`.
    """
    try:
        return stat.S_IMODE(target.stat().st_mode)
    except OSError:
        return DEFAULT_FILE_MODE


def _fsync_directory(directory: Path) -> None:
    """Flush a directory entry so the rename survives a power loss.

    Best effort only: some platforms and filesystems do not allow opening a
    directory for this purpose, and a failure here does not make the written
    data any less complete.

    Args:
        directory: Directory whose entries should be flushed.
    """
    with suppress(OSError):
        dir_fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)


@contextmanager
def atomic_writer(
    path: str | Path,
    mode: str = "w",
    *,
    encoding: str | None = None,
    newline: str | None = None,
) -> Iterator[IO[Any]]:
    """Open a file handle whose contents land at ``path`` atomically.

    The handle points at a temporary file in the same directory as ``path``.
    On clean exit the handle is flushed, ``fsync``-ed and renamed over ``path``.
    If the body raises, the temporary file is removed and any file already at
    ``path`` is left untouched.

    Use this for writers that need a file object, such as ``json.dump``,
    ``DataFrame.to_csv`` or ``pyarrow.parquet.write_table``.

    Args:
        path: Final destination of the written file.
        mode: Write mode, either text (``"w"``) or binary (``"wb"``).  Read,
            append and update modes are rejected because the temporary file
            always starts empty.
        encoding: Text encoding.  Defaults to ``"utf-8"`` in text mode and must
            be ``None`` in binary mode.
        newline: Newline handling, passed through to :func:`open`.  Pass ``""``
            for csv-style writers that emit their own line terminators.

    Yields:
        An open file handle to write to.

    Raises:
        ValueError: If ``mode`` is not a write mode, or if ``encoding`` is set
            for a binary mode.
    """
    if any(flag in mode for flag in ("r", "a", "+", "x")):
        raise ValueError(f"atomic_writer requires a write mode, got {mode!r}")
    if "w" not in mode:
        raise ValueError(f"atomic_writer requires a write mode, got {mode!r}")

    binary = "b" in mode
    if binary and encoding is not None:
        raise ValueError("encoding cannot be used with a binary mode")
    if not binary and encoding is None:
        encoding = "utf-8"

    target = Path(path)
    directory = target.parent
    directory.mkdir(parents=True, exist_ok=True)

    file_mode = _resolve_file_mode(target)
    fd, tmp_name = tempfile.mkstemp(
        dir=directory, prefix=f".{target.name}.", suffix=".tmp"
    )
    tmp_path = Path(tmp_name)

    try:
        with os.fdopen(fd, mode, encoding=encoding, newline=newline) as handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp_path, file_mode)
        os.replace(tmp_path, target)
    except BaseException:
        with suppress(OSError):
            tmp_path.unlink()
        raise

    _fsync_directory(directory)


def atomic_write(
    path: str | Path,
    data: str | bytes,
    *,
    encoding: str = "utf-8",
) -> Path:
    """Write ``data`` to ``path`` atomically.

    Convenience wrapper around :func:`atomic_writer` for callers that already
    hold the full contents in memory.

    Args:
        path: Final destination of the written file.
        data: Text or bytes to write.  ``bytes`` selects binary mode.
        encoding: Text encoding, ignored when ``data`` is ``bytes``.

    Returns:
        The path that was written.
    """
    target = Path(path)
    if isinstance(data, bytes):
        with atomic_writer(target, "wb") as handle:
            handle.write(data)
    else:
        with atomic_writer(target, "w", encoding=encoding) as handle:
            handle.write(data)
    return target
