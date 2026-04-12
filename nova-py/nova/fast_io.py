"""High-performance binary I/O for NOVA arrays.

This module provides a *fast path* for writing and reading single arrays in
NOVA's native binary format.  It bypasses Zarr overhead entirely, writing a
compact header followed by optionally zstd-compressed numpy data.

For uncompressed data ``numpy.ndarray.tofile`` / ``numpy.fromfile`` are used
for zero-copy I/O.  For compressed data the ``zstandard`` C library provides
multi-threaded zstd compression.

File layout (nova_fast format)::

    [8 bytes]  magic number  b"NOVAFAST"
    [2 bytes]  format version (uint16 LE) = 1
    [2 bytes]  header length in bytes (uint16 LE)
    [N bytes]  JSON header (UTF-8) containing dtype, shape, compression info
    [...]      zstd-compressed raw array data  (or raw data if level == 0)

The format is intentionally simple so that readers in any language (C, Rust,
Julia, ...) can parse it without Zarr or HDF5 libraries.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import numpy as np

_MAGIC = b"NOVAFAST"
_VERSION = 1


def fast_write(
    path: str | Path,
    data: np.ndarray,
    compression_level: int = 1,
    threads: int = -1,
    metadata: dict[str, Any] | None = None,
) -> int:
    """Write a numpy array to a NOVA fast-format file.

    Parameters
    ----------
    path : str or Path
        Output file path.
    data : numpy.ndarray
        Array to write.  Must be C-contiguous; will be made contiguous if not.
    compression_level : int
        Zstd compression level (0 = no compression, 1?22 = zstd levels).
        Level 1 gives the best speed/ratio tradeoff.
    threads : int
        Number of zstd compression threads (-1 = all cores).
    metadata : dict, optional
        Additional JSON-serialisable metadata stored in the header.

    Returns
    -------
    int
        Number of bytes written.
    """
    path = Path(path)
    data = np.ascontiguousarray(data)

    # Build header
    header: dict[str, Any] = {
        "dtype": str(data.dtype),
        "shape": list(data.shape),
        "byte_order": "little" if data.dtype.byteorder in ("<", "=") else "big",
        "compression": "zstd" if compression_level > 0 else "none",
        "compression_level": compression_level,
    }
    if metadata:
        header["metadata"] = metadata
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")

    with open(path, "wb") as f:
        f.write(_MAGIC)
        f.write(struct.pack("<HH", _VERSION, len(header_bytes)))
        f.write(header_bytes)

        if compression_level > 0:
            import zstandard as zstd

            cctx = zstd.ZstdCompressor(level=compression_level, threads=threads)
            payload = cctx.compress(data.tobytes())
            f.write(payload)
        else:
            # Zero-copy write -- fastest path
            data.tofile(f)

    return path.stat().st_size


def fast_read(path: str | Path) -> np.ndarray:
    """Read a numpy array from a NOVA fast-format file.

    Parameters
    ----------
    path : str or Path
        Input file path.

    Returns
    -------
    numpy.ndarray
        The reconstructed array.

    Raises
    ------
    ValueError
        If the file is not a valid NOVA fast-format file.
    """
    path = Path(path)

    with open(path, "rb") as f:
        magic = f.read(8)
        if magic != _MAGIC:
            raise ValueError(f"Not a NOVA fast file (bad magic: {magic!r})")

        version, header_len = struct.unpack("<HH", f.read(4))
        if version != _VERSION:
            raise ValueError(f"Unsupported NOVA fast version: {version}")

        header = json.loads(f.read(header_len).decode("utf-8"))
        dtype = np.dtype(header["dtype"])
        shape = tuple(header["shape"])

        if header.get("compression", "none") == "zstd":
            import zstandard as zstd

            dctx = zstd.ZstdDecompressor()
            raw = dctx.decompress(f.read())
            return np.frombuffer(raw, dtype=dtype).reshape(shape)
        else:
            # Zero-copy read -- fastest path
            return np.fromfile(f, dtype=dtype).reshape(shape)


def fast_read_slice(
    path: str | Path,
    slices: tuple[slice, ...],
) -> np.ndarray:
    """Read a partial slice from a NOVA fast-format file.

    For uncompressed files this could use mmap; for compressed files the full
    array is decompressed then sliced (same as FITS for compressed data).

    Parameters
    ----------
    path : str or Path
        Input file path.
    slices : tuple of slice
        Region to extract.

    Returns
    -------
    numpy.ndarray
        The sliced array.
    """
    arr = fast_read(path)
    return arr[slices]
