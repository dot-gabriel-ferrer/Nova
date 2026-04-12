"""Tests for the NOVA integrity module."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from nova.integrity import compute_sha256, compute_file_sha256, verify_chunk


class TestComputeSha256:
    """Tests for SHA-256 computation."""

    def test_empty_data(self) -> None:
        result = compute_sha256(b"")
        # Known SHA-256 of empty string
        assert result == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

    def test_known_data(self) -> None:
        result = compute_sha256(b"NOVA format")
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_deterministic(self) -> None:
        data = b"test data for NOVA integrity"
        assert compute_sha256(data) == compute_sha256(data)

    def test_different_data_different_hash(self) -> None:
        assert compute_sha256(b"data1") != compute_sha256(b"data2")


class TestComputeFileSha256:
    """Tests for file SHA-256 computation."""

    def test_file_hash(self) -> None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
            f.write(b"test file content")
            f.flush()
            path = f.name

        file_hash = compute_file_sha256(path)
        data_hash = compute_sha256(b"test file content")
        assert file_hash == data_hash
        Path(path).unlink()


class TestVerifyChunk:
    """Tests for chunk verification."""

    def test_valid_chunk(self) -> None:
        data = b"valid chunk data"
        expected = compute_sha256(data)
        assert verify_chunk(data, expected) is True

    def test_invalid_chunk(self) -> None:
        data = b"corrupted chunk data"
        wrong_hash = "0" * 64
        assert verify_chunk(data, wrong_hash) is False

    def test_empty_chunk(self) -> None:
        data = b""
        expected = compute_sha256(data)
        assert verify_chunk(data, expected) is True
