"""Post-development audit tests.

Covers: constants centralisation, input validation added during audit,
URL scheme validation, and type-safety improvements.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

# -----------------------------------------------------------------------
#  constants.py -- all shared values importable
# -----------------------------------------------------------------------

class TestConstants:
    """Verify that the constants module exposes the expected values."""

    def test_nova_version_is_string(self) -> None:
        from nova.constants import NOVA_VERSION
        assert isinstance(NOVA_VERSION, str)
        assert "." in NOVA_VERSION

    def test_nova_context_is_url(self) -> None:
        from nova.constants import NOVA_CONTEXT
        assert NOVA_CONTEXT.startswith("https://")

    def test_supported_dtypes_frozen(self) -> None:
        from nova.constants import SUPPORTED_DTYPES
        assert isinstance(SUPPORTED_DTYPES, frozenset)
        assert "float64" in SUPPORTED_DTYPES

    def test_supported_codecs_frozen(self) -> None:
        from nova.constants import SUPPORTED_CODECS
        assert isinstance(SUPPORTED_CODECS, frozenset)
        assert "zstd" in SUPPORTED_CODECS

    def test_mad_to_std_value(self) -> None:
        from nova.constants import MAD_TO_STD
        assert abs(MAD_TO_STD - 1.4826) < 1e-4

    def test_hash_read_size(self) -> None:
        from nova.constants import HASH_READ_SIZE
        assert HASH_READ_SIZE == 65_536

    def test_fits_extensions_frozen(self) -> None:
        from nova.constants import FITS_EXTENSIONS
        assert isinstance(FITS_EXTENSIONS, frozenset)
        assert ".fits" in FITS_EXTENSIONS

    def test_supported_remote_schemes(self) -> None:
        from nova.constants import SUPPORTED_REMOTE_SCHEMES
        assert "https" in SUPPORTED_REMOTE_SCHEMES
        assert "s3" in SUPPORTED_REMOTE_SCHEMES


# -----------------------------------------------------------------------
#  container.py -- input validation
# -----------------------------------------------------------------------

class TestContainerInputValidation:
    """New validation checks added during the audit."""

    def test_set_science_data_rejects_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            from nova.container import NovaDataset
            ds = NovaDataset(Path(tmp) / "test.nova.zarr", mode="w")
            with pytest.raises(TypeError, match="numpy.ndarray"):
                ds.set_science_data([[1, 2], [3, 4]])  # type: ignore[arg-type]
            ds.close()

    def test_set_uncertainty_rejects_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            from nova.container import NovaDataset
            ds = NovaDataset(Path(tmp) / "test.nova.zarr", mode="w")
            with pytest.raises(TypeError, match="numpy.ndarray"):
                ds.set_uncertainty([[1, 2], [3, 4]])  # type: ignore[arg-type]
            ds.close()

    def test_set_mask_rejects_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            from nova.container import NovaDataset
            ds = NovaDataset(Path(tmp) / "test.nova.zarr", mode="w")
            with pytest.raises(TypeError, match="numpy.ndarray"):
                ds.set_mask([[0, 1], [1, 0]])  # type: ignore[arg-type]
            ds.close()

    def test_set_science_data_accepts_ndarray(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            from nova.container import NovaDataset
            ds = NovaDataset(Path(tmp) / "test.nova.zarr", mode="w")
            ds.set_science_data(np.ones((32, 32), dtype=np.float64))
            ds.close()


# -----------------------------------------------------------------------
#  remote.py -- URL scheme validation
# -----------------------------------------------------------------------

class TestRemoteSchemeValidation:
    """Verify that open_remote rejects unsupported URI schemes."""

    def test_unsupported_scheme_raises(self) -> None:
        from nova.remote import RemoteNovaDataset
        with pytest.raises(ValueError, match="Unsupported URL scheme"):
            RemoteNovaDataset("ftp://example.org/obs.nova.zarr")

    def test_missing_scheme_raises(self) -> None:
        from nova.remote import RemoteNovaDataset
        with pytest.raises(ValueError, match="Unsupported URL scheme"):
            RemoteNovaDataset("/local/path/obs.nova.zarr")


# -----------------------------------------------------------------------
#  streaming.py -- parameter validation
# -----------------------------------------------------------------------

class TestStreamingValidation:
    """Verify buffer_size and compression_level bounds."""

    def test_negative_buffer_size_raises(self) -> None:
        from nova.streaming import StreamWriter
        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(ValueError, match="buffer_size"):
                StreamWriter(
                    Path(tmp) / "ts.nova.zarr",
                    frame_shape=(8, 8),
                    buffer_size=-1,
                )

    def test_compression_level_too_high_raises(self) -> None:
        from nova.streaming import StreamWriter
        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(ValueError, match="compression_level"):
                StreamWriter(
                    Path(tmp) / "ts.nova.zarr",
                    frame_shape=(8, 8),
                    compression_level=30,
                )


# -----------------------------------------------------------------------
#  ml.py -- method validation
# -----------------------------------------------------------------------

class TestMLValidation:
    """Verify normalization method checking."""

    def test_unknown_method_raises(self) -> None:
        from nova.ml import compute_normalization
        with pytest.raises(ValueError, match="Unknown normalization"):
            compute_normalization(np.ones(10), method="bogus")


# -----------------------------------------------------------------------
#  Module imports from constants
# -----------------------------------------------------------------------

class TestConstantsIntegration:
    """Verify that modules re-export constants from the central module."""

    def test_container_uses_central_version(self) -> None:
        from nova.container import NOVA_VERSION as cv
        from nova.constants import NOVA_VERSION as kv
        assert cv == kv

    def test_container_uses_central_context(self) -> None:
        from nova.container import NOVA_CONTEXT as cc
        from nova.constants import NOVA_CONTEXT as kc
        assert cc == kc

    def test_integrity_uses_central_hash_size(self) -> None:
        """integrity.py must use HASH_READ_SIZE from constants."""
        import inspect
        from nova import integrity
        src = inspect.getsource(integrity.compute_file_sha256)
        assert "HASH_READ_SIZE" in src

    def test_ml_uses_central_mad_to_std(self) -> None:
        """ml.py must reference MAD_TO_STD instead of the literal 1.4826."""
        import inspect
        from nova import ml
        src = inspect.getsource(ml.normalize)
        assert "MAD_TO_STD" in src
        assert "1.4826" not in src

    def test_math_uses_central_mad_to_std(self) -> None:
        """math.py must reference MAD_TO_STD instead of the literal 1.4826."""
        import inspect
        from nova import math as nm
        src = inspect.getsource(nm.robust_statistics)
        assert "MAD_TO_STD" in src
        assert "1.4826" not in src
