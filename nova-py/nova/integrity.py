"""SHA-256 chunk integrity verification module.

Provides functions for computing and verifying SHA-256 hashes of data chunks,
implementing INV-4 (INTEGRITY_BY_DEFAULT).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


def compute_sha256(data: bytes) -> str:
    """Compute SHA-256 hash of binary data.

    Parameters
    ----------
    data : bytes
        Binary data to hash.

    Returns
    -------
    str
        Hex-encoded SHA-256 hash string.
    """
    return hashlib.sha256(data).hexdigest()


def compute_file_sha256(path: str | Path) -> str:
    """Compute SHA-256 hash of a file.

    Parameters
    ----------
    path : str or Path
        Path to the file.

    Returns
    -------
    str
        Hex-encoded SHA-256 hash string.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def verify_chunk(data: bytes, expected_hash: str) -> bool:
    """Verify the integrity of a data chunk.

    Parameters
    ----------
    data : bytes
        Chunk binary data.
    expected_hash : str
        Expected SHA-256 hash (hex-encoded).

    Returns
    -------
    bool
        True if the chunk integrity is verified, False otherwise.
    """
    actual_hash = compute_sha256(data)
    return actual_hash == expected_hash


def verify_store_integrity(
    store_path: str | Path,
    chunk_index: dict[str, Any],
) -> list[str]:
    """Verify integrity of all chunks in a NOVA store.

    Parameters
    ----------
    store_path : str or Path
        Path to the NOVA store directory.
    chunk_index : dict
        The chunk index dictionary (nova_index.json content).

    Returns
    -------
    list of str
        List of integrity errors. Empty list means all chunks are valid.
    """
    store_path = Path(store_path)
    errors: list[str] = []

    for chunk_info in chunk_index.get("nova:chunks", []):
        chunk_path = store_path / chunk_info["nova:path"]
        expected_hash = chunk_info["nova:sha256"]

        if not chunk_path.exists():
            errors.append(f"Missing chunk: {chunk_info['nova:path']}")
            continue

        actual_hash = compute_file_sha256(chunk_path)
        if actual_hash != expected_hash:
            errors.append(
                f"Integrity check failed for {chunk_info['nova:path']}: "
                f"expected {expected_hash[:16]}..., got {actual_hash[:16]}..."
            )

    return errors
