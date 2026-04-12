"""NOVA container management using Zarr v3 as the base store.

This module provides the core data model for creating, reading, and writing
NOVA datasets backed by Zarr stores.
"""

from __future__ import annotations

import json
import datetime
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from nova.wcs import NovaWCS
from nova.provenance import ProvenanceBundle
from nova.integrity import compute_sha256


# NOVA format version
NOVA_VERSION = "0.1.0"

# Default chunk shape for 2D images (512x512 float64 = 2 MB per chunk)
DEFAULT_CHUNK_SHAPE_2D: tuple[int, int] = (512, 512)

# Default compression settings
DEFAULT_COMPRESSOR = "zstd"
DEFAULT_COMPRESSION_LEVEL = 3

# NOVA JSON-LD context URL
NOVA_CONTEXT = "https://nova-astro.org/v0.1/context.jsonld"


class NovaDataset:
    """A NOVA dataset backed by a Zarr store.

    Parameters
    ----------
    store_path : str or Path
        Path to the Zarr store directory (`.nova.zarr`).
    mode : str
        Open mode: 'r' (read-only), 'w' (write), 'a' (append).

    Attributes
    ----------
    store_path : Path
        Path to the Zarr store.
    wcs : NovaWCS or None
        World Coordinate System metadata.
    provenance : ProvenanceBundle or None
        W3C PROV-DM provenance metadata.
    metadata : dict
        Root metadata dictionary.
    """

    def __init__(self, store_path: str | Path, mode: str = "r") -> None:
        self.store_path = Path(store_path)
        self.mode = mode
        self._root: zarr.Group | None = None
        self._metadata: dict[str, Any] = {}
        self._wcs: NovaWCS | None = None
        self._provenance: ProvenanceBundle | None = None

        if mode == "r" and self.store_path.exists():
            self._open_existing()
        elif mode in ("w", "a"):
            self._ensure_store()

    def _open_existing(self) -> None:
        """Open an existing NOVA store for reading."""
        self._root = zarr.open_group(str(self.store_path), mode="r")

        # Load metadata
        metadata_path = self.store_path / "nova_metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                self._metadata = json.load(f)

        # Load WCS
        wcs_path = self.store_path / "wcs.json"
        if wcs_path.exists():
            with open(wcs_path) as f:
                wcs_data = json.load(f)
            self._wcs = NovaWCS.from_dict(wcs_data)

        # Load provenance
        prov_path = self.store_path / "provenance.json"
        if prov_path.exists():
            with open(prov_path) as f:
                prov_data = json.load(f)
            self._provenance = ProvenanceBundle.from_dict(prov_data)

    def _ensure_store(self) -> None:
        """Create or open a Zarr store for writing."""
        self.store_path.mkdir(parents=True, exist_ok=True)
        zarr_mode = "w" if self.mode == "w" else "a"
        self._root = zarr.open_group(str(self.store_path), mode=zarr_mode)

    @property
    def wcs(self) -> NovaWCS | None:
        """World Coordinate System metadata."""
        return self._wcs

    @wcs.setter
    def wcs(self, value: NovaWCS) -> None:
        self._wcs = value

    @property
    def provenance(self) -> ProvenanceBundle | None:
        """W3C PROV-DM provenance metadata."""
        return self._provenance

    @provenance.setter
    def provenance(self, value: ProvenanceBundle) -> None:
        self._provenance = value

    @property
    def metadata(self) -> dict[str, Any]:
        """Root metadata dictionary."""
        return self._metadata

    @property
    def data(self) -> zarr.Array | None:
        """Primary science data array."""
        if self._root is None:
            return None
        try:
            return self._root["data"]["science"]
        except KeyError:
            return None

    @property
    def uncertainty(self) -> zarr.Array | None:
        """Uncertainty plane array."""
        if self._root is None:
            return None
        try:
            return self._root["data"]["uncertainty"]
        except KeyError:
            return None

    @property
    def mask(self) -> zarr.Array | None:
        """Data quality mask array."""
        if self._root is None:
            return None
        try:
            return self._root["data"]["mask"]
        except KeyError:
            return None

    def set_science_data(
        self,
        data: np.ndarray,
        chunks: tuple[int, ...] | None = None,
        compression_level: int = DEFAULT_COMPRESSION_LEVEL,
    ) -> None:
        """Store the primary science data array.

        Parameters
        ----------
        data : numpy.ndarray
            Science data array.
        chunks : tuple of int, optional
            Chunk shape. Defaults to (512, 512) for 2D data.
        compression_level : int
            ZSTD compression level (default: 3).
        """
        if self._root is None:
            raise RuntimeError("Store not initialized. Open in 'w' or 'a' mode.")

        if chunks is None:
            if data.ndim == 2:
                chunks = DEFAULT_CHUNK_SHAPE_2D
            else:
                # Auto-chunk: ~2MB per chunk
                chunks = tuple(min(s, 512) for s in data.shape)

        try:
            from numcodecs import Zstd
            compressor = Zstd(level=compression_level)
        except ImportError:
            compressor = None

        data_group = self._root.require_group("data")
        arr = data_group.create_array(
            "science",
            shape=data.shape,
            chunks=chunks,
            dtype=data.dtype,
            overwrite=True,
        )
        arr[:] = data

    def set_uncertainty(
        self,
        data: np.ndarray,
        chunks: tuple[int, ...] | None = None,
    ) -> None:
        """Store the uncertainty plane.

        Parameters
        ----------
        data : numpy.ndarray
            Uncertainty data array (same shape as science data).
        chunks : tuple of int, optional
            Chunk shape.
        """
        if self._root is None:
            raise RuntimeError("Store not initialized. Open in 'w' or 'a' mode.")

        if chunks is None:
            if data.ndim == 2:
                chunks = DEFAULT_CHUNK_SHAPE_2D
            else:
                chunks = tuple(min(s, 512) for s in data.shape)

        data_group = self._root.require_group("data")
        arr = data_group.create_array(
            "uncertainty",
            shape=data.shape,
            chunks=chunks,
            dtype=data.dtype,
            overwrite=True,
        )
        arr[:] = data

    def set_mask(
        self,
        data: np.ndarray,
        chunks: tuple[int, ...] | None = None,
    ) -> None:
        """Store the data quality mask.

        Parameters
        ----------
        data : numpy.ndarray
            Mask data array (uint8 or uint16, same shape as science data).
        chunks : tuple of int, optional
            Chunk shape.
        """
        if self._root is None:
            raise RuntimeError("Store not initialized. Open in 'w' or 'a' mode.")

        if chunks is None:
            if data.ndim == 2:
                chunks = DEFAULT_CHUNK_SHAPE_2D
            else:
                chunks = tuple(min(s, 512) for s in data.shape)

        data_group = self._root.require_group("data")
        arr = data_group.create_array(
            "mask",
            shape=data.shape,
            chunks=chunks,
            dtype=data.dtype,
            overwrite=True,
        )
        arr[:] = data

    def _build_metadata(self) -> dict[str, Any]:
        """Build the root metadata dictionary."""
        meta: dict[str, Any] = {
            "@context": NOVA_CONTEXT,
            "@type": "nova:Observation",
            "nova:version": NOVA_VERSION,
            "nova:created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        meta.update(self._metadata)
        return meta

    def _build_chunk_index(self) -> dict[str, Any]:
        """Build the chunk index for cloud-native access."""
        index: dict[str, Any] = {
            "@context": NOVA_CONTEXT,
            "@type": "nova:ChunkIndex",
            "nova:version": NOVA_VERSION,
            "nova:created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "nova:chunks": [],
        }

        # Walk the store directory to find chunk files
        if self.store_path.exists():
            for chunk_path in sorted(self.store_path.rglob("*")):
                if chunk_path.is_file() and chunk_path.name not in (
                    "nova_metadata.json",
                    "nova_index.json",
                    "wcs.json",
                    "provenance.json",
                    ".zattrs",
                    ".zgroup",
                    ".zarray",
                    "zarr.json",
                ):
                    rel_path = str(chunk_path.relative_to(self.store_path))
                    # Skip hidden files and metadata
                    if rel_path.startswith("."):
                        continue
                    chunk_data = chunk_path.read_bytes()
                    chunk_hash = compute_sha256(chunk_data)
                    index["nova:chunks"].append({
                        "nova:path": rel_path,
                        "nova:offset": 0,
                        "nova:size": len(chunk_data),
                        "nova:sha256": chunk_hash,
                    })

        return index

    def save(self) -> None:
        """Write all metadata files to the store.

        Writes nova_metadata.json, nova_index.json, wcs.json, and
        provenance.json to the store root.
        """
        if self.mode == "r":
            raise RuntimeError("Cannot save in read-only mode.")

        # Write root metadata
        meta = self._build_metadata()
        meta_path = self.store_path / "nova_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        # Write WCS
        if self._wcs is not None:
            wcs_path = self.store_path / "wcs.json"
            with open(wcs_path, "w") as f:
                json.dump(self._wcs.to_dict(), f, indent=2)

        # Write provenance
        if self._provenance is not None:
            prov_path = self.store_path / "provenance.json"
            with open(prov_path, "w") as f:
                json.dump(self._provenance.to_dict(), f, indent=2)

        # Write chunk index
        index = self._build_chunk_index()
        index_path = self.store_path / "nova_index.json"
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)

    def close(self) -> None:
        """Close the dataset."""
        self._root = None

    def __enter__(self) -> NovaDataset:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def open_dataset(path: str | Path, mode: str = "r") -> NovaDataset:
    """Open a NOVA dataset.

    Parameters
    ----------
    path : str or Path
        Path to the NOVA store (`.nova.zarr` directory).
    mode : str
        Open mode: 'r' (read-only), 'w' (write), 'a' (append).

    Returns
    -------
    NovaDataset
        The opened NOVA dataset.
    """
    return NovaDataset(path, mode=mode)


def create_dataset(
    path: str | Path,
    data: np.ndarray,
    wcs: NovaWCS | None = None,
    metadata: dict[str, Any] | None = None,
) -> NovaDataset:
    """Create a new NOVA dataset.

    Parameters
    ----------
    path : str or Path
        Path for the new NOVA store.
    data : numpy.ndarray
        Primary science data array.
    wcs : NovaWCS, optional
        WCS metadata.
    metadata : dict, optional
        Additional metadata key-value pairs.

    Returns
    -------
    NovaDataset
        The created NOVA dataset.
    """
    ds = NovaDataset(path, mode="w")
    ds.set_science_data(data)
    if wcs is not None:
        ds.wcs = wcs
    if metadata is not None:
        ds._metadata.update(metadata)
    ds.save()
    return ds
