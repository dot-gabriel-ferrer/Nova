"""Shared constants for the NOVA library.

Centralises magic numbers, URLs, and format-wide defaults so every module
imports from one place.  See also the NOVA specification (section 2) for
the design invariants these constants support.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
#  Format identity
# ---------------------------------------------------------------------------

NOVA_VERSION: str = "0.3.0"
"""Current NOVA format + library version string."""

NOVA_CONTEXT: str = "https://nova-astro.org/v0.1/context.jsonld"
"""JSON-LD context URL used in all NOVA metadata documents."""

# ---------------------------------------------------------------------------
#  I/O defaults
# ---------------------------------------------------------------------------

DEFAULT_CHUNK_SHAPE_2D: tuple[int, int] = (512, 512)
"""Default chunk shape for 2-D images (512x512 float64 = 2 MB per chunk)."""

DEFAULT_COMPRESSION_CODEC: str = "zstd"
"""Default compression codec name."""

DEFAULT_COMPRESSION_LEVEL: int = 1
"""ZSTD compression level (1 = best speed/ratio tradeoff)."""

SUPPORTED_CODECS: frozenset[str] = frozenset({"zstd", "none"})
"""Set of compression codec names accepted by the container."""

HASH_READ_SIZE: int = 65_536
"""Byte-size of read chunks when computing SHA-256 file hashes (64 KiB)."""

TABLE_CHUNK_SIZE: int = 65_536
"""Default chunk size for table column arrays (rows per chunk)."""

# ---------------------------------------------------------------------------
#  Data types
# ---------------------------------------------------------------------------

SUPPORTED_DTYPES: frozenset[str] = frozenset({
    "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32",
    "float16", "float32", "float64",
    "complex64", "complex128",
})
"""NumPy dtype names accepted as NOVA science array data types."""

# ---------------------------------------------------------------------------
#  Mathematical constants used across modules
# ---------------------------------------------------------------------------

MAD_TO_STD: float = 1.4826
"""Scale factor that converts the Median Absolute Deviation (MAD) to an
estimate of the standard deviation for a normal distribution.
Derivation: 1 / Phi^{-1}(3/4) where Phi is the normal CDF."""

ARCSEC_PER_DEG: float = 3600.0
"""Number of arcseconds in one degree."""

# ---------------------------------------------------------------------------
#  FITS migration helpers
# ---------------------------------------------------------------------------

FITS_EXTENSIONS: frozenset[str] = frozenset({
    ".fits", ".fit", ".fts",
    ".fits.gz", ".fit.gz", ".fts.gz",
})
"""File extensions recognised as FITS files by the migration tool."""

# ---------------------------------------------------------------------------
#  Remote / cloud protocols
# ---------------------------------------------------------------------------

SUPPORTED_REMOTE_SCHEMES: frozenset[str] = frozenset({
    "http", "https", "s3", "gs", "gcs", "az", "abfs",
})
"""URI schemes accepted by ``open_remote()``."""
