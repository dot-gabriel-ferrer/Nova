"""Tests for the NOVA fast binary I/O module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from nova.fast_io import fast_write, fast_read, fast_read_slice


class TestFastWrite:
    """Tests for fast_write."""

    def test_write_creates_file(self, tmp_path: Path) -> None:
        data = np.ones((64, 64), dtype=np.float64)
        path = tmp_path / "test.nova"
        sz = fast_write(path, data)
        assert path.exists()
        assert sz > 0

    def test_write_uncompressed(self, tmp_path: Path) -> None:
        data = np.ones((32, 32), dtype=np.float64)
        path = tmp_path / "test.nova"
        fast_write(path, data, compression_level=0)
        loaded = fast_read(path)
        np.testing.assert_array_equal(loaded, data)

    def test_write_compressed(self, tmp_path: Path) -> None:
        data = np.arange(1024, dtype=np.float64).reshape(32, 32)
        path = tmp_path / "test.nova"
        fast_write(path, data, compression_level=1)
        loaded = fast_read(path)
        np.testing.assert_array_equal(loaded, data)

    def test_compressed_smaller_for_compressible_data(self, tmp_path: Path) -> None:
        # Gradient data is highly compressible
        data = np.zeros((256, 256), dtype=np.float64)
        path_raw = tmp_path / "raw.nova"
        path_zst = tmp_path / "zst.nova"
        sz_raw = fast_write(path_raw, data, compression_level=0)
        sz_zst = fast_write(path_zst, data, compression_level=1)
        assert sz_zst < sz_raw

    def test_roundtrip_float32(self, tmp_path: Path) -> None:
        data = np.random.default_rng(42).standard_normal((128, 128)).astype(np.float32)
        path = tmp_path / "test.nova"
        fast_write(path, data, compression_level=1)
        loaded = fast_read(path)
        np.testing.assert_array_equal(loaded, data)
        assert loaded.dtype == np.float32

    def test_roundtrip_int16(self, tmp_path: Path) -> None:
        data = np.arange(100, dtype=np.int16).reshape(10, 10)
        path = tmp_path / "test.nova"
        fast_write(path, data, compression_level=0)
        loaded = fast_read(path)
        np.testing.assert_array_equal(loaded, data)
        assert loaded.dtype == np.int16

    def test_roundtrip_3d(self, tmp_path: Path) -> None:
        data = np.ones((4, 32, 32), dtype=np.float64)
        path = tmp_path / "test.nova"
        fast_write(path, data, compression_level=1)
        loaded = fast_read(path)
        np.testing.assert_array_equal(loaded, data)
        assert loaded.shape == (4, 32, 32)

    def test_with_metadata(self, tmp_path: Path) -> None:
        data = np.ones((16, 16), dtype=np.float64)
        path = tmp_path / "test.nova"
        fast_write(path, data, metadata={"telescope": "HST", "exposure": 300.0})
        loaded = fast_read(path)
        np.testing.assert_array_equal(loaded, data)


class TestFastRead:
    """Tests for fast_read."""

    def test_invalid_magic(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.nova"
        path.write_bytes(b"BADMAGIC" + b"\x00" * 100)
        with pytest.raises(ValueError, match="Not a NOVA fast file"):
            fast_read(path)

    def test_read_back_large_array(self, tmp_path: Path) -> None:
        data = np.random.default_rng(99).standard_normal((512, 512))
        path = tmp_path / "large.nova"
        fast_write(path, data, compression_level=0)
        loaded = fast_read(path)
        np.testing.assert_array_equal(loaded, data)


class TestFastReadSlice:
    """Tests for fast_read_slice."""

    def test_partial_read(self, tmp_path: Path) -> None:
        data = np.arange(256 * 256, dtype=np.float64).reshape(256, 256)
        path = tmp_path / "test.nova"
        fast_write(path, data, compression_level=0)

        slices = (slice(50, 100), slice(50, 100))
        partial = fast_read_slice(path, slices)
        np.testing.assert_array_equal(partial, data[slices])
        assert partial.shape == (50, 50)

    def test_partial_read_compressed(self, tmp_path: Path) -> None:
        data = np.arange(256 * 256, dtype=np.float64).reshape(256, 256)
        path = tmp_path / "test.nova"
        fast_write(path, data, compression_level=1)

        slices = (slice(0, 128), slice(128, 256))
        partial = fast_read_slice(path, slices)
        np.testing.assert_array_equal(partial, data[slices])
