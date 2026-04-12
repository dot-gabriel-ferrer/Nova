"""Remote store access for NOVA datasets.

Provides transparent access to NOVA datasets stored on remote servers
(HTTP/HTTPS, S3, Google Cloud Storage, Azure Blob Storage) using the
fsspec and zarr backends.  Data is fetched lazily -- only the chunks
actually accessed are downloaded.

Usage examples::

    import nova
    # HTTP(S)
    ds = nova.open_remote("https://archive.example.org/obs/12345.nova.zarr")
    cutout = ds.data[1000:1100, 2000:2100]  # only needed chunks fetched

    # S3
    ds = nova.open_remote("s3://my-bucket/obs/12345.nova.zarr")

    # With authentication
    ds = nova.open_remote(
        "s3://private-bucket/obs.nova.zarr",
        storage_options={"key": "...", "secret": "..."},
    )

Requires ``fsspec`` to be installed.  Protocol-specific backends
(``s3fs`` for S3, ``gcsfs`` for GCS, ``adlfs`` for Azure) are only
needed when accessing the corresponding storage service.
"""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np


def open_remote(
    url: str,
    *,
    mode: str = "r",
    storage_options: dict[str, Any] | None = None,
) -> "RemoteNovaDataset":
    """Open a NOVA dataset from a remote URL.

    Parameters
    ----------
    url : str
        URL to the NOVA Zarr store.  Supported schemes include
        ``http://``, ``https://``, ``s3://``, ``gs://``, ``az://``.
    mode : str
        ``"r"`` for read-only (default).  Remote write is supported
        for S3/GCS/Azure when the backend allows it.
    storage_options : dict, optional
        Extra keyword arguments forwarded to the *fsspec* filesystem
        constructor (e.g. authentication credentials).

    Returns
    -------
    RemoteNovaDataset
        A dataset-like object whose ``.data`` property returns a lazy
        Zarr array -- only the accessed slices trigger network I/O.

    Raises
    ------
    ImportError
        If ``fsspec`` or a protocol-specific backend is not installed.
    """
    return RemoteNovaDataset(url, mode=mode, storage_options=storage_options)


class RemoteNovaDataset:
    """A NOVA dataset backed by a remote Zarr store.

    Acts like :class:`nova.container.NovaDataset` but all I/O goes
    through *fsspec* so data is fetched lazily over the network.

    Parameters
    ----------
    url : str
        Remote URL to the ``.nova.zarr`` store.
    mode : str
        ``"r"`` for read-only, ``"w"`` for write, ``"a"`` for append.
    storage_options : dict or None
        Credentials / configuration forwarded to the filesystem.
    """

    def __init__(
        self,
        url: str,
        mode: str = "r",
        storage_options: dict[str, Any] | None = None,
    ) -> None:
        self.url = url.rstrip("/")
        self.mode = mode
        self.storage_options = storage_options or {}
        self._store = None
        self._root = None
        self._metadata: dict[str, Any] = {}
        self._wcs = None
        self._provenance = None

        self._open()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _open(self) -> None:
        """Initialise the remote Zarr group."""
        try:
            import fsspec
        except ImportError:
            raise ImportError(
                "fsspec is required for remote access.  "
                "Install it with:  pip install fsspec"
            )
        import zarr

        fs, _, paths = fsspec.core.url_to_fs(
            self.url, **self.storage_options,
        )
        store = fsspec.implementations.reference.ReferenceFileSystem
        # Use the generic zarr-compatible fsspec store
        mapper = fs.get_mapper(paths[0] if isinstance(paths, list) else self.url)
        self._store = mapper
        self._root = zarr.open_group(mapper, mode=self.mode)

        # Load metadata files if they exist in the store
        self._load_metadata(fs, paths[0] if isinstance(paths, list) else self.url)

    def _load_metadata(self, fs: Any, root_path: str) -> None:
        """Attempt to load JSON metadata files from the remote store."""
        # Metadata
        meta_path = root_path.rstrip("/") + "/nova_metadata.json"
        try:
            with fs.open(meta_path, "r") as f:
                self._metadata = json.load(f)
        except (FileNotFoundError, OSError):
            pass

        # WCS
        wcs_path = root_path.rstrip("/") + "/wcs.json"
        try:
            with fs.open(wcs_path, "r") as f:
                wcs_data = json.load(f)
            from nova.wcs import NovaWCS
            self._wcs = NovaWCS.from_dict(wcs_data)
        except (FileNotFoundError, OSError, ImportError):
            pass

        # Provenance
        prov_path = root_path.rstrip("/") + "/provenance.json"
        try:
            with fs.open(prov_path, "r") as f:
                prov_data = json.load(f)
            from nova.provenance import ProvenanceBundle
            self._provenance = ProvenanceBundle.from_dict(prov_data)
        except (FileNotFoundError, OSError, ImportError):
            pass

    # ------------------------------------------------------------------
    # Public properties (mirror NovaDataset API)
    # ------------------------------------------------------------------

    @property
    def metadata(self) -> dict[str, Any]:
        """Root metadata dictionary."""
        return self._metadata

    @property
    def wcs(self):
        """WCS metadata (None if not available remotely)."""
        return self._wcs

    @property
    def provenance(self):
        """Provenance bundle (None if not available)."""
        return self._provenance

    @property
    def data(self):
        """Primary science data as a lazy Zarr array.

        Only the slices you actually index are fetched from the remote
        server, making it efficient for cutouts and partial reads.
        """
        if self._root is None:
            return None
        try:
            return self._root["data"]["science"]
        except KeyError:
            return None

    @property
    def uncertainty(self):
        """Uncertainty plane (lazy remote access)."""
        if self._root is None:
            return None
        try:
            return self._root["data"]["uncertainty"]
        except KeyError:
            return None

    @property
    def mask(self):
        """Data quality mask (lazy remote access)."""
        if self._root is None:
            return None
        try:
            return self._root["data"]["mask"]
        except KeyError:
            return None

    def close(self) -> None:
        """Release resources."""
        self._root = None
        self._store = None

    def __enter__(self) -> "RemoteNovaDataset":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"RemoteNovaDataset(url={self.url!r})"


def is_remote_url(path: str) -> bool:
    """Check whether *path* looks like a remote URL.

    Returns True for http(s), s3, gs, az, abfs schemes.
    """
    lower = str(path).lower()
    return any(
        lower.startswith(scheme)
        for scheme in ("http://", "https://", "s3://", "gs://", "az://", "abfs://")
    )
