"""Streaming and append support for NOVA datasets.

Enables appending new data to existing NOVA arrays -- useful for time-series
observations, monitoring pipelines, and data that arrives incrementally.

Usage::

    from nova.streaming import open_appendable, append_frame

    ds = open_appendable("timeseries.nova.zarr", frame_shape=(256, 256))
    for frame in camera_stream():
        append_frame(ds, frame)
    ds.close()

Also provides a context-manager based ``StreamWriter`` for more control
over buffering and flush intervals.
"""

from __future__ import annotations

import json
import datetime
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from zarr.codecs import ZstdCodec

from nova.constants import DEFAULT_COMPRESSION_LEVEL, NOVA_CONTEXT, NOVA_VERSION


class StreamWriter:
    """Buffered writer that appends frames to a NOVA dataset.

    Parameters
    ----------
    store_path : str or Path
        Path to the ``.nova.zarr`` store.
    frame_shape : tuple[int, ...]
        Shape of each individual frame (e.g. ``(256, 256)``).
    dtype : str or numpy dtype
        Data type for the frames (default ``float64``).
    axis : int
        Axis along which new frames are appended (default ``0``).
    buffer_size : int
        Number of frames to buffer in memory before flushing to disk.
    compression_level : int
        ZSTD compression level (0 for uncompressed).
    metadata : dict or None
        Additional metadata stored in ``nova_metadata.json``.
    """

    def __init__(
        self,
        store_path: str | Path,
        frame_shape: tuple[int, ...],
        *,
        dtype: str | np.dtype = "float64",
        axis: int = 0,
        buffer_size: int = 10,
        compression_level: int = DEFAULT_COMPRESSION_LEVEL,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.store_path = Path(store_path)
        self.frame_shape = frame_shape
        self.dtype = np.dtype(dtype)
        self.axis = axis
        if buffer_size < 1:
            raise ValueError(f"buffer_size must be >= 1, got {buffer_size}")
        self.buffer_size = buffer_size
        if not (0 <= compression_level <= 22):
            raise ValueError(
                f"compression_level must be 0-22, got {compression_level}"
            )
        self.compression_level = compression_level
        self._extra_metadata = metadata or {}
        self._buffer: list[np.ndarray] = []
        self._total_frames: int = 0

        self._init_store()

    def _init_store(self) -> None:
        """Create or open the Zarr store."""
        self.store_path.mkdir(parents=True, exist_ok=True)
        self._root = zarr.open_group(str(self.store_path), mode="a")

        # Build the full array shape: (0, *frame_shape) initially
        full_shape = list(self.frame_shape)
        full_shape.insert(self.axis, 0)

        # Chunk shape: one frame per chunk along the append axis
        chunk_shape = list(self.frame_shape)
        chunk_shape.insert(self.axis, 1)

        data_group = self._root.require_group("data")

        # Check if science array already exists (resuming)
        try:
            self._array = data_group["science"]
            # Recover total frame count from existing array
            self._total_frames = self._array.shape[self.axis]
        except KeyError:
            compressors = (
                [ZstdCodec(level=self.compression_level)]
                if self.compression_level > 0
                else None
            )
            self._array = data_group.create_array(
                "science",
                shape=tuple(full_shape),
                chunks=tuple(chunk_shape),
                dtype=self.dtype,
                compressors=compressors,
            )

    def append(self, frame: np.ndarray) -> None:
        """Add a single frame to the buffer.

        The frame is written to disk immediately (no internal buffering)
        so that data is not lost if the process crashes.

        Parameters
        ----------
        frame : numpy.ndarray
            Array with shape matching ``frame_shape``.
        """
        if frame.shape != self.frame_shape:
            raise ValueError(
                f"Frame shape {frame.shape} does not match "
                f"expected {self.frame_shape}."
            )
        frame = np.asarray(frame, dtype=self.dtype)
        self._buffer.append(frame)
        if len(self._buffer) >= self.buffer_size:
            self.flush()

    def flush(self) -> None:
        """Write buffered frames to disk."""
        if not self._buffer:
            return

        batch = np.stack(self._buffer, axis=self.axis)
        n_new = batch.shape[self.axis]

        # Resize the array to accommodate new frames
        new_shape = list(self._array.shape)
        new_shape[self.axis] += n_new
        self._array.resize(tuple(new_shape))

        # Write the new data
        slices: list[Any] = [slice(None)] * len(new_shape)
        slices[self.axis] = slice(
            self._total_frames, self._total_frames + n_new,
        )
        self._array[tuple(slices)] = batch

        self._total_frames += n_new
        self._buffer.clear()

    @property
    def total_frames(self) -> int:
        """Number of frames written so far (including buffered)."""
        return self._total_frames + len(self._buffer)

    def save_metadata(self) -> None:
        """Write NOVA metadata to the store root."""
        meta: dict[str, Any] = {
            "@context": NOVA_CONTEXT,
            "@type": "nova:TimeSeries",
            "nova:version": NOVA_VERSION,
            "nova:created": datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat(),
            "nova:total_frames": self._total_frames,
            "nova:frame_shape": list(self.frame_shape),
            "nova:dtype": str(self.dtype),
        }
        meta.update(self._extra_metadata)
        meta_path = self.store_path / "nova_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    def close(self) -> None:
        """Flush remaining buffer, write metadata, and release resources."""
        self.flush()
        self.save_metadata()
        self._root = None

    def __enter__(self) -> "StreamWriter":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def open_appendable(
    store_path: str | Path,
    frame_shape: tuple[int, ...],
    *,
    dtype: str | np.dtype = "float64",
    buffer_size: int = 10,
    compression_level: int = 1,
) -> StreamWriter:
    """Open (or create) a NOVA store for append-mode streaming writes.

    Parameters
    ----------
    store_path : str or Path
        Path to the ``.nova.zarr`` store.
    frame_shape : tuple of int
        Shape of each frame.
    dtype : str or numpy.dtype
        Data type for frames.
    buffer_size : int
        Frames buffered before auto-flush.
    compression_level : int
        ZSTD compression level.

    Returns
    -------
    StreamWriter
    """
    return StreamWriter(
        store_path,
        frame_shape,
        dtype=dtype,
        buffer_size=buffer_size,
        compression_level=compression_level,
    )


def append_frame(writer: StreamWriter, frame: np.ndarray) -> None:
    """Convenience function: append *frame* and flush immediately.

    Parameters
    ----------
    writer : StreamWriter
        An open ``StreamWriter``.
    frame : numpy.ndarray
        Array matching the writer's ``frame_shape``.
    """
    writer.append(frame)
    writer.flush()
