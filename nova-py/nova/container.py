"""NOVA container management using Zarr v3 as the base store.

This module provides the core data model for creating, reading, and writing
NOVA datasets backed by Zarr stores.  Supports multi-extension datasets
(multiple HDU groups), table data (structured arrays), and multiple
compression codecs.
"""

from __future__ import annotations

import json
import datetime
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from zarr.codecs import ZstdCodec

from nova.wcs import NovaWCS
from nova.provenance import ProvenanceBundle
from nova.integrity import compute_sha256


# NOVA format version
NOVA_VERSION = "0.3.0"

# Default chunk shape for 2D images (512x512 float64 = 2 MB per chunk)
DEFAULT_CHUNK_SHAPE_2D: tuple[int, int] = (512, 512)

# Default compression settings
# Level 1 gives the best speed/ratio tradeoff for typical astronomical data
# while keeping writes nearly as fast as raw (uncompressed) I/O.
DEFAULT_COMPRESSOR = "zstd"
DEFAULT_COMPRESSION_LEVEL = 1

# NOVA JSON-LD context URL
NOVA_CONTEXT = "https://nova-astro.org/v0.1/context.jsonld"

# Supported compression codecs
SUPPORTED_CODECS = {"zstd", "none"}

# All data types supported by NOVA (section 4.3 of the specification)
SUPPORTED_DTYPES = {
    "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32",
    "float16", "float32", "float64",
    "complex64", "complex128",
}


class NovaTable:
    """A table stored inside a NOVA dataset.

    Tables are represented as a collection of Zarr arrays (one per column)
    inside a Zarr group, together with column metadata stored as JSON.

    Parameters
    ----------
    name : str
        Table name / group path inside the store.
    columns : dict[str, np.ndarray]
        Column name -> 1-D array mapping.
    column_meta : dict[str, dict] | None
        Per-column metadata (units, UCDs, descriptions).
    """

    def __init__(
        self,
        name: str,
        columns: dict[str, np.ndarray] | None = None,
        column_meta: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.name = name
        self.columns: dict[str, np.ndarray] = columns or {}
        self.column_meta: dict[str, dict[str, Any]] = column_meta or {}
        self._nrows: int | None = None
        if self.columns:
            first = next(iter(self.columns.values()))
            self._nrows = len(first)

    @property
    def nrows(self) -> int:
        if self._nrows is None:
            return 0
        return self._nrows

    @property
    def colnames(self) -> list[str]:
        return list(self.columns.keys())

    def add_column(
        self,
        name: str,
        data: np.ndarray,
        unit: str | None = None,
        ucd: str | None = None,
        description: str | None = None,
    ) -> None:
        """Add a column to the table."""
        if self._nrows is not None and len(data) != self._nrows:
            raise ValueError(
                f"Column '{name}' has {len(data)} rows, expected {self._nrows}."
            )
        self.columns[name] = data
        if self._nrows is None:
            self._nrows = len(data)
        meta: dict[str, Any] = {}
        if unit is not None:
            meta["nova:unit"] = unit
        if ucd is not None:
            meta["nova:ucd"] = ucd
        if description is not None:
            meta["nova:description"] = description
        if meta:
            self.column_meta[name] = meta

    def to_structured_array(self) -> np.ndarray:
        """Convert to a NumPy structured array."""
        dtype_list = [(name, col.dtype) for name, col in self.columns.items()]
        result = np.empty(self.nrows, dtype=dtype_list)
        for name, col in self.columns.items():
            result[name] = col
        return result

    @classmethod
    def from_structured_array(
        cls,
        name: str,
        arr: np.ndarray,
        column_meta: dict[str, dict[str, Any]] | None = None,
    ) -> NovaTable:
        """Create a NovaTable from a NumPy structured array."""
        columns = {field: arr[field] for field in arr.dtype.names}
        return cls(name=name, columns=columns, column_meta=column_meta)

    def to_dict(self) -> dict[str, Any]:
        """Serialise table metadata to a JSON-serialisable dictionary."""
        col_descriptors = []
        for name, data in self.columns.items():
            desc: dict[str, Any] = {
                "nova:name": name,
                "nova:dtype": str(data.dtype),
                "nova:length": len(data),
            }
            if name in self.column_meta:
                desc.update(self.column_meta[name])
            col_descriptors.append(desc)
        return {
            "@type": "nova:Table",
            "nova:name": self.name,
            "nova:nrows": self.nrows,
            "nova:columns": col_descriptors,
        }


class NovaExtension:
    """Represents one extension (HDU equivalent) inside a multi-extension
    NOVA dataset.

    Parameters
    ----------
    name : str
        Extension name (e.g. ``'SCI'``, ``'ERR'``, ``'DQ'``).
    data : np.ndarray | None
        Image data for this extension.
    header : dict | None
        Header keyword dict for this extension.
    wcs : NovaWCS | None
        Extension-specific WCS.
    extver : int
        Extension version number (like FITS EXTVER).
    """

    def __init__(
        self,
        name: str,
        data: np.ndarray | None = None,
        header: dict[str, Any] | None = None,
        wcs: NovaWCS | None = None,
        extver: int = 1,
    ) -> None:
        self.name = name
        self.data = data
        self.header = header or {}
        self.wcs = wcs
        self.extver = extver


class NovaDataset:
    """A NOVA dataset backed by a Zarr store.

    Supports single-extension (classic) and multi-extension datasets (MEF
    equivalent).  Each extension is a Zarr group with its own data array,
    optional WCS, and header keywords.

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
        World Coordinate System metadata (primary extension).
    provenance : ProvenanceBundle or None
        W3C PROV-DM provenance metadata.
    metadata : dict
        Root metadata dictionary.
    extensions : list[NovaExtension]
        Multi-extension data (populated when reading MEF datasets).
    tables : dict[str, NovaTable]
        Named tables stored in the dataset.
    """

    def __init__(self, store_path: str | Path, mode: str = "r") -> None:
        self.store_path = Path(store_path)
        self.mode = mode
        self._root: zarr.Group | None = None
        self._metadata: dict[str, Any] = {}
        self._wcs: NovaWCS | None = None
        self._provenance: ProvenanceBundle | None = None
        self._extensions: list[NovaExtension] = []
        self._tables: dict[str, NovaTable] = {}

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
            try:
                with open(metadata_path) as f:
                    self._metadata = json.load(f)
            except (json.JSONDecodeError, ValueError):
                self._metadata = {}

        # Load WCS
        wcs_path = self.store_path / "wcs.json"
        if wcs_path.exists():
            try:
                with open(wcs_path) as f:
                    wcs_data = json.load(f)
                self._wcs = NovaWCS.from_dict(wcs_data)
            except (json.JSONDecodeError, ValueError, KeyError):
                pass

        # Load provenance
        prov_path = self.store_path / "provenance.json"
        if prov_path.exists():
            try:
                with open(prov_path) as f:
                    prov_data = json.load(f)
                self._provenance = ProvenanceBundle.from_dict(prov_data)
            except (json.JSONDecodeError, ValueError, KeyError):
                pass

        # Load multi-extension data
        extensions_path = self.store_path / "extensions.json"
        if extensions_path.exists():
            with open(extensions_path) as f:
                ext_meta = json.load(f)
            for ext_info in ext_meta.get("nova:extensions", []):
                ext_name = ext_info.get("nova:name", "UNKNOWN")
                ext_ver = ext_info.get("nova:extver", 1)
                ext_header = ext_info.get("nova:header", {})
                ext_wcs = None
                ext_wcs_path = self.store_path / "extensions" / ext_name / "wcs.json"
                if ext_wcs_path.exists():
                    with open(ext_wcs_path) as f:
                        ext_wcs = NovaWCS.from_dict(json.load(f))
                ext_data = None
                try:
                    ext_data_arr = self._root["extensions"][ext_name]["data"]
                    ext_data = np.array(ext_data_arr)
                except (KeyError, TypeError):
                    pass
                self._extensions.append(
                    NovaExtension(
                        name=ext_name,
                        data=ext_data,
                        header=ext_header,
                        wcs=ext_wcs,
                        extver=ext_ver,
                    )
                )

        # Load tables
        tables_path = self.store_path / "tables.json"
        if tables_path.exists():
            with open(tables_path) as f:
                tables_meta = json.load(f)
            for tbl_info in tables_meta.get("nova:tables", []):
                tbl_name = tbl_info.get("nova:name", "unknown")
                columns: dict[str, np.ndarray] = {}
                col_meta: dict[str, dict[str, Any]] = {}
                for col_desc in tbl_info.get("nova:columns", []):
                    col_name = col_desc["nova:name"]
                    try:
                        col_arr = self._root["tables"][tbl_name][col_name]
                        columns[col_name] = np.array(col_arr)
                    except (KeyError, TypeError):
                        pass
                    meta_keys = {}
                    for k in ("nova:unit", "nova:ucd", "nova:description"):
                        if k in col_desc:
                            meta_keys[k] = col_desc[k]
                    if meta_keys:
                        col_meta[col_name] = meta_keys
                self._tables[tbl_name] = NovaTable(
                    name=tbl_name, columns=columns, column_meta=col_meta,
                )

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

    # -- Multi-extension support --------------------------------------

    @property
    def extensions(self) -> list[NovaExtension]:
        """List of extensions in a multi-extension dataset."""
        return self._extensions

    def add_extension(self, ext: NovaExtension) -> None:
        """Add an extension to the dataset."""
        self._extensions.append(ext)

    def get_extension(
        self, name: str, extver: int = 1,
    ) -> NovaExtension | None:
        """Retrieve an extension by name and version."""
        for ext in self._extensions:
            if ext.name == name and ext.extver == extver:
                return ext
        return None

    # -- Table support ------------------------------------------------

    @property
    def tables(self) -> dict[str, NovaTable]:
        """Named tables in the dataset."""
        return self._tables

    def add_table(self, table: NovaTable) -> None:
        """Add a table to the dataset."""
        self._tables[table.name] = table

    def get_table(self, name: str) -> NovaTable | None:
        """Retrieve a table by name."""
        return self._tables.get(name)

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
            Chunk shape. Auto-selected for best performance if not provided.
        compression_level : int
            ZSTD compression level (default: 1 for speed).
        """
        if self._root is None:
            raise RuntimeError("Store not initialized. Open in 'w' or 'a' mode.")

        if chunks is None:
            chunks = _optimal_chunks(data.shape)

        compressors: list[ZstdCodec] | None = (
            [ZstdCodec(level=compression_level)] if compression_level > 0 else None
        )

        data_group = self._root.require_group("data")
        data_group.create_array(
            "science",
            chunks=chunks,
            compressors=compressors,
            data=data,
            write_data=True,
            overwrite=True,
        )

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
            chunks = _optimal_chunks(data.shape)

        data_group = self._root.require_group("data")
        data_group.create_array(
            "uncertainty",
            chunks=chunks,
            compressors=[ZstdCodec(level=DEFAULT_COMPRESSION_LEVEL)],
            data=data,
            write_data=True,
            overwrite=True,
        )

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
            chunks = _optimal_chunks(data.shape)

        data_group = self._root.require_group("data")
        data_group.create_array(
            "mask",
            chunks=chunks,
            compressors=[ZstdCodec(level=DEFAULT_COMPRESSION_LEVEL)],
            data=data,
            write_data=True,
            overwrite=True,
        )

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

    def save(self, build_index: bool = False) -> None:
        """Write all metadata files to the store.

        Writes nova_metadata.json, wcs.json, provenance.json, extensions, and
        tables to the store root.  Optionally builds the chunk integrity index
        (nova_index.json) which requires reading back every chunk and computing
        SHA-256 hashes.

        Parameters
        ----------
        build_index : bool
            If True, build the chunk integrity index.  This is expensive for
            large datasets and should only be used when integrity verification
            is required (e.g. before archiving or distribution).  Default is
            False for performance.
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

        # Write extensions (multi-extension support)
        if self._extensions:
            self._save_extensions()

        # Write tables
        if self._tables:
            self._save_tables()

        # Write chunk index (expensive -- reads all chunks + SHA-256)
        if build_index:
            index = self._build_chunk_index()
            index_path = self.store_path / "nova_index.json"
            with open(index_path, "w") as f:
                json.dump(index, f, indent=2)

    def _save_extensions(self) -> None:
        """Persist multi-extension data and metadata."""
        if self._root is None:
            raise RuntimeError("Store not initialized.")

        ext_meta_list: list[dict[str, Any]] = []
        ext_group = self._root.require_group("extensions")

        for ext in self._extensions:
            sub = ext_group.require_group(ext.name)
            if ext.data is not None:
                chunks = _optimal_chunks(ext.data.shape)
                sub.create_array(
                    "data",
                    chunks=chunks,
                    compressors=[ZstdCodec(level=DEFAULT_COMPRESSION_LEVEL)],
                    data=ext.data,
                    write_data=True,
                    overwrite=True,
                )
            # Extension-specific WCS
            if ext.wcs is not None:
                ext_dir = self.store_path / "extensions" / ext.name
                ext_dir.mkdir(parents=True, exist_ok=True)
                with open(ext_dir / "wcs.json", "w") as f:
                    json.dump(ext.wcs.to_dict(), f, indent=2)

            ext_meta_list.append({
                "nova:name": ext.name,
                "nova:extver": ext.extver,
                "nova:header": ext.header,
                "nova:has_data": ext.data is not None,
                "nova:shape": list(ext.data.shape) if ext.data is not None else None,
                "nova:dtype": str(ext.data.dtype) if ext.data is not None else None,
            })

        ext_doc = {
            "@context": NOVA_CONTEXT,
            "@type": "nova:MultiExtensionDataset",
            "nova:extensions": ext_meta_list,
        }
        with open(self.store_path / "extensions.json", "w") as f:
            json.dump(ext_doc, f, indent=2)

    def _save_tables(self) -> None:
        """Persist table data and metadata."""
        if self._root is None:
            raise RuntimeError("Store not initialized.")

        tables_group = self._root.require_group("tables")
        tbl_meta_list: list[dict[str, Any]] = []

        for tbl_name, table in self._tables.items():
            tbl_group = tables_group.require_group(tbl_name)
            for col_name, col_data in table.columns.items():
                tbl_group.create_array(
                    col_name,
                    chunks=(min(len(col_data), 65536),),
                    data=col_data,
                    write_data=True,
                    overwrite=True,
                )
            tbl_meta_list.append(table.to_dict())

        tables_doc = {
            "@context": NOVA_CONTEXT,
            "@type": "nova:TableCollection",
            "nova:tables": tbl_meta_list,
        }
        with open(self.store_path / "tables.json", "w") as f:
            json.dump(tables_doc, f, indent=2)

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


def _optimal_chunks(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Choose chunk sizes that balance I/O and partial-read efficiency.

    For small-to-medium arrays the entire array is stored as a single chunk
    to minimise filesystem overhead.  For very large arrays, chunks are sized
    to ~4 MB each, keeping the number of chunk files manageable.

    Parameters
    ----------
    shape : tuple of int
        Shape of the data array.

    Returns
    -------
    tuple of int
        Optimal chunk shape.
    """
    total_elements = 1
    for s in shape:
        total_elements *= s

    # For arrays up to ~32 MB (4M float64 elements), use a single chunk
    if total_elements <= 4_194_304:
        return shape

    # For larger arrays, target ~4 MB per chunk (512k float64 elements)
    target_elements = 524_288
    if len(shape) == 2:
        # For 2D: square-ish chunks
        side = max(1, int(target_elements ** 0.5))
        return (min(shape[0], side), min(shape[1], side))

    # For N-D: cap each dimension at 512
    return tuple(min(s, 512) for s in shape)
