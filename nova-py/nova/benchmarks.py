"""Performance benchmarking module for NOVA format.

Provides utilities to benchmark NOVA against FITS, HDF5, and raw NumPy for:
- Read/write speed comparison
- Compression ratio analysis
- Chunk-based (cloud-simulated) partial read performance
- Multi-format throughput comparison
"""

from __future__ import annotations

import time
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run.

    Parameters
    ----------
    name : str
        Name of the benchmark.
    operation : str
        Operation type ('write', 'read', 'partial_read', 'compress').
    format_name : str
        Format tested ('NOVA', 'FITS', 'HDF5', 'NumPy').
    elapsed_seconds : float
        Wall-clock time in seconds.
    data_shape : tuple of int
        Shape of the data array.
    data_dtype : str
        Data type string.
    file_size_bytes : int
        Size of the output file in bytes.
    raw_data_bytes : int
        Size of the raw (uncompressed) data in bytes.
    extra : dict
        Additional benchmark-specific metrics.
    """

    name: str
    operation: str
    format_name: str
    elapsed_seconds: float
    data_shape: tuple[int, ...]
    data_dtype: str
    file_size_bytes: int = 0
    raw_data_bytes: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def compression_ratio(self) -> float:
        """Compression ratio (raw / compressed). Higher is better."""
        if self.file_size_bytes == 0:
            return 0.0
        return self.raw_data_bytes / self.file_size_bytes

    @property
    def throughput_mbps(self) -> float:
        """Throughput in MB/s."""
        if self.elapsed_seconds == 0:
            return 0.0
        return (self.raw_data_bytes / (1024 * 1024)) / self.elapsed_seconds

    def summary(self) -> str:
        """Return a human-readable summary string."""
        lines = [
            f"Benchmark: {self.name}",
            f"  Format:      {self.format_name}",
            f"  Operation:   {self.operation}",
            f"  Data shape:  {self.data_shape}",
            f"  Data dtype:  {self.data_dtype}",
            f"  Time:        {self.elapsed_seconds:.4f} s",
            f"  Throughput:  {self.throughput_mbps:.2f} MB/s",
            f"  File size:   {self.file_size_bytes / (1024*1024):.2f} MB",
            f"  Raw size:    {self.raw_data_bytes / (1024*1024):.2f} MB",
            f"  Compression: {self.compression_ratio:.2f}x",
        ]
        for key, value in self.extra.items():
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)


@dataclass
class BenchmarkComparison:
    """Comparison of NOVA vs FITS benchmark results.

    Parameters
    ----------
    nova_result : BenchmarkResult
        NOVA benchmark result.
    fits_result : BenchmarkResult
        FITS benchmark result.
    """

    nova_result: BenchmarkResult
    fits_result: BenchmarkResult

    @property
    def speedup(self) -> float:
        """Speed improvement factor (FITS_time / NOVA_time). >1 means NOVA is faster."""
        if self.nova_result.elapsed_seconds == 0:
            return float("inf")
        return self.fits_result.elapsed_seconds / self.nova_result.elapsed_seconds

    @property
    def compression_improvement(self) -> float:
        """Compression improvement (NOVA_ratio / FITS_ratio). >1 means NOVA compresses better."""
        fits_ratio = self.fits_result.compression_ratio
        if fits_ratio == 0:
            return float("inf")
        return self.nova_result.compression_ratio / fits_ratio

    def summary(self) -> str:
        """Return a human-readable comparison summary."""
        lines = [
            "=" * 60,
            f"Performance Comparison: {self.nova_result.name}",
            "=" * 60,
            "",
            "--- NOVA ---",
            self.nova_result.summary(),
            "",
            "--- FITS ---",
            self.fits_result.summary(),
            "",
            "--- Comparison ---",
            f"  Speed:       {self.speedup:.2f}x {'(NOVA faster)' if self.speedup > 1 else '(FITS faster)'}",
            f"  Compression: {self.compression_improvement:.2f}x {'(NOVA better)' if self.compression_improvement > 1 else '(FITS better)'}",
            f"  NOVA file:   {self.nova_result.file_size_bytes / (1024*1024):.2f} MB",
            f"  FITS file:   {self.fits_result.file_size_bytes / (1024*1024):.2f} MB",
        ]
        return "\n".join(lines)


def generate_test_data(
    shape: tuple[int, ...] = (4096, 4096),
    dtype: str = "float64",
    pattern: str = "gaussian_noise",
    seed: int = 42,
) -> np.ndarray:
    """Generate synthetic astronomical test data.

    Parameters
    ----------
    shape : tuple of int
        Shape of the data array.
    dtype : str
        Numpy dtype string.
    pattern : str
        Data pattern: 'gaussian_noise', 'gradient', 'sparse', 'realistic_sky'.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    numpy.ndarray
        Generated test data.
    """
    rng = np.random.default_rng(seed)

    if pattern == "gaussian_noise":
        data = rng.standard_normal(shape).astype(dtype)
    elif pattern == "gradient":
        # Smooth gradient (highly compressible)
        coords = np.meshgrid(
            *[np.linspace(0, 1, s) for s in shape], indexing="ij"
        )
        data = sum(c for c in coords).astype(dtype)  # type: ignore[arg-type]
    elif pattern == "sparse":
        # Mostly zeros with some signal (typical sky)
        data = np.zeros(shape, dtype=dtype)
        n_sources = max(1, int(np.prod(shape) * 0.001))
        indices = tuple(rng.integers(0, s, size=n_sources) for s in shape)
        data[indices] = rng.uniform(100, 10000, size=n_sources).astype(dtype)
    elif pattern == "realistic_sky":
        # Background + Gaussian sources (simulated astronomical image)
        background = rng.normal(1000.0, 30.0, shape).astype("float64")
        sources = np.zeros(shape, dtype="float64")
        n_sources = max(10, int(np.prod(shape) * 0.0001))
        for _ in range(n_sources):
            center = tuple(rng.integers(0, s) for s in shape)
            flux = rng.uniform(500, 50000)
            sigma = rng.uniform(1.5, 5.0)
            coords = np.meshgrid(
                *[np.arange(s) for s in shape], indexing="ij"
            )
            r2 = sum((c - cx) ** 2 for c, cx in zip(coords, center))
            sources += flux * np.exp(-r2 / (2 * sigma**2))
        data = (background + sources).astype(dtype)
    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    return data


# ---------------------------------------------------------------------------
#  NOVA benchmarks (fast binary I/O — primary performance path)
# ---------------------------------------------------------------------------

def benchmark_nova_write(
    data: np.ndarray,
    output_dir: str | Path | None = None,
    compression_level: int = 0,
) -> BenchmarkResult:
    """Benchmark NOVA write performance using the fast binary I/O path.

    Parameters
    ----------
    data : numpy.ndarray
        Data to write.
    output_dir : str or Path, optional
        Output directory. Uses a temporary directory if not provided.
    compression_level : int
        ZSTD compression level.  Defaults to 0 (no compression) for an
        apples-to-apples comparison with FITS, which is also uncompressed.
        The container default (``DEFAULT_COMPRESSION_LEVEL = 1``) uses light
        compression for storage efficiency.

    Returns
    -------
    BenchmarkResult
        Write benchmark result.
    """
    from nova.fast_io import fast_write

    if output_dir is None:
        output_dir = tempfile.mkdtemp()
    file_path = Path(output_dir) / "bench_write.nova"

    raw_bytes = data.nbytes

    start = time.perf_counter()
    fast_write(file_path, data, compression_level=compression_level)
    elapsed = time.perf_counter() - start

    file_size = file_path.stat().st_size

    return BenchmarkResult(
        name="NOVA Write",
        operation="write",
        format_name="NOVA",
        elapsed_seconds=elapsed,
        data_shape=data.shape,
        data_dtype=str(data.dtype),
        file_size_bytes=file_size,
        raw_data_bytes=raw_bytes,
        extra={"compression_level": compression_level},
    )


def benchmark_nova_read(
    store_path: str | Path,
    data_shape: tuple[int, ...] | None = None,
) -> BenchmarkResult:
    """Benchmark NOVA read performance using the fast binary I/O path.

    Parameters
    ----------
    store_path : str or Path
        Path to the NOVA file.
    data_shape : tuple of int, optional
        Expected data shape (for metadata).

    Returns
    -------
    BenchmarkResult
        Read benchmark result.
    """
    from nova.fast_io import fast_read

    store_path = Path(store_path)
    file_size = store_path.stat().st_size

    start = time.perf_counter()
    full_data = fast_read(store_path)
    elapsed = time.perf_counter() - start

    return BenchmarkResult(
        name="NOVA Read",
        operation="read",
        format_name="NOVA",
        elapsed_seconds=elapsed,
        data_shape=full_data.shape,
        data_dtype=str(full_data.dtype),
        file_size_bytes=file_size,
        raw_data_bytes=full_data.nbytes,
    )


def benchmark_nova_partial_read(
    store_path: str | Path,
    slices: tuple[slice, ...],
) -> BenchmarkResult:
    """Benchmark NOVA partial read.

    Parameters
    ----------
    store_path : str or Path
        Path to the NOVA file.
    slices : tuple of slice
        Region to read.

    Returns
    -------
    BenchmarkResult
        Partial read benchmark result.
    """
    from nova.fast_io import fast_read_slice

    store_path = Path(store_path)
    file_size = store_path.stat().st_size

    start = time.perf_counter()
    partial = fast_read_slice(store_path, slices)
    elapsed = time.perf_counter() - start

    region_str = ", ".join(f"{s.start}:{s.stop}" for s in slices)

    return BenchmarkResult(
        name="NOVA Partial Read",
        operation="partial_read",
        format_name="NOVA",
        elapsed_seconds=elapsed,
        data_shape=partial.shape,
        data_dtype=str(partial.dtype),
        file_size_bytes=file_size,
        raw_data_bytes=partial.nbytes,
        extra={"region": f"[{region_str}]"},
    )


# ---------------------------------------------------------------------------
#  NOVA Zarr benchmarks (cloud-native chunked path)
# ---------------------------------------------------------------------------

def benchmark_nova_zarr_write(
    data: np.ndarray,
    output_dir: str | Path | None = None,
    compression_level: int = 1,
) -> BenchmarkResult:
    """Benchmark NOVA Zarr-backed write performance.

    Parameters
    ----------
    data : numpy.ndarray
        Data to write.
    output_dir : str or Path, optional
        Output directory.
    compression_level : int
        ZSTD compression level.

    Returns
    -------
    BenchmarkResult
        Write benchmark result.
    """
    from nova.container import NovaDataset

    if output_dir is None:
        output_dir = tempfile.mkdtemp()
    store_path = Path(output_dir) / "bench_write.nova.zarr"

    raw_bytes = data.nbytes

    start = time.perf_counter()
    ds = NovaDataset(store_path, mode="w")
    ds.set_science_data(data, compression_level=compression_level)
    ds.save()
    ds.close()
    elapsed = time.perf_counter() - start

    file_size = _dir_size(store_path)

    return BenchmarkResult(
        name="NOVA Zarr Write",
        operation="write",
        format_name="NOVA (Zarr)",
        elapsed_seconds=elapsed,
        data_shape=data.shape,
        data_dtype=str(data.dtype),
        file_size_bytes=file_size,
        raw_data_bytes=raw_bytes,
        extra={"compression_level": compression_level},
    )


def benchmark_nova_zarr_read(
    store_path: str | Path,
    data_shape: tuple[int, ...] | None = None,
) -> BenchmarkResult:
    """Benchmark NOVA Zarr-backed read performance.

    Parameters
    ----------
    store_path : str or Path
        Path to the NOVA store.
    data_shape : tuple of int, optional
        Expected data shape.

    Returns
    -------
    BenchmarkResult
        Read benchmark result.
    """
    from nova.container import NovaDataset

    store_path = Path(store_path)
    file_size = _dir_size(store_path)

    start = time.perf_counter()
    ds = NovaDataset(store_path, mode="r")
    arr = ds.data
    if arr is not None:
        full_data = np.array(arr)
        shape = full_data.shape
        dtype = str(full_data.dtype)
        raw_bytes = full_data.nbytes
    else:
        shape = data_shape or (0,)
        dtype = "unknown"
        raw_bytes = 0
    ds.close()
    elapsed = time.perf_counter() - start

    return BenchmarkResult(
        name="NOVA Zarr Read",
        operation="read",
        format_name="NOVA (Zarr)",
        elapsed_seconds=elapsed,
        data_shape=shape,
        data_dtype=dtype,
        file_size_bytes=file_size,
        raw_data_bytes=raw_bytes,
    )


def benchmark_nova_zarr_partial_read(
    store_path: str | Path,
    slices: tuple[slice, ...],
) -> BenchmarkResult:
    """Benchmark NOVA Zarr partial (chunk-based) read — cloud-native access.

    Parameters
    ----------
    store_path : str or Path
        Path to the NOVA store.
    slices : tuple of slice
        Region to read.

    Returns
    -------
    BenchmarkResult
        Partial read benchmark result.
    """
    from nova.container import NovaDataset

    store_path = Path(store_path)
    file_size = _dir_size(store_path)

    start = time.perf_counter()
    ds = NovaDataset(store_path, mode="r")
    arr = ds.data
    if arr is not None:
        partial = np.array(arr[slices])
        shape = partial.shape
        dtype = str(partial.dtype)
        raw_bytes = partial.nbytes
    else:
        shape = (0,)
        dtype = "unknown"
        raw_bytes = 0
    ds.close()
    elapsed = time.perf_counter() - start

    region_str = ", ".join(f"{s.start}:{s.stop}" for s in slices)

    return BenchmarkResult(
        name="NOVA Zarr Partial Read (cloud-native)",
        operation="partial_read",
        format_name="NOVA (Zarr)",
        elapsed_seconds=elapsed,
        data_shape=shape,
        data_dtype=dtype,
        file_size_bytes=file_size,
        raw_data_bytes=raw_bytes,
        extra={"region": f"[{region_str}]"},
    )


# ---------------------------------------------------------------------------
#  FITS benchmarks
# ---------------------------------------------------------------------------

def benchmark_fits_write(
    data: np.ndarray,
    output_dir: str | Path | None = None,
) -> BenchmarkResult:
    """Benchmark FITS write performance.

    Timing includes byte-order conversion (FITS mandates big-endian) for a
    fair comparison with NOVA which stores native byte order.

    Parameters
    ----------
    data : numpy.ndarray
        Data to write.
    output_dir : str or Path, optional
        Output directory.

    Returns
    -------
    BenchmarkResult
        Write benchmark result.
    """
    try:
        from astropy.io import fits as astropy_fits
    except ImportError:
        raise ImportError("astropy is required for FITS benchmarks.")

    if output_dir is None:
        output_dir = tempfile.mkdtemp()
    fits_path = Path(output_dir) / "bench_write.fits"

    raw_bytes = data.nbytes

    # Include byte-order conversion in timing (FITS mandates big-endian)
    start = time.perf_counter()
    if data.dtype.byteorder == "<" or (
        data.dtype.byteorder == "=" and np.little_endian
    ):
        fits_data = data.astype(data.dtype.newbyteorder(">"))
    else:
        fits_data = data
    hdu = astropy_fits.PrimaryHDU(data=fits_data)
    hdul = astropy_fits.HDUList([hdu])
    hdul.writeto(str(fits_path), overwrite=True)
    elapsed = time.perf_counter() - start

    file_size = fits_path.stat().st_size

    return BenchmarkResult(
        name="FITS Write",
        operation="write",
        format_name="FITS",
        elapsed_seconds=elapsed,
        data_shape=data.shape,
        data_dtype=str(data.dtype),
        file_size_bytes=file_size,
        raw_data_bytes=raw_bytes,
    )


def benchmark_fits_read(
    fits_path: str | Path,
) -> BenchmarkResult:
    """Benchmark FITS full read performance (no memory mapping).

    Memory mapping is disabled for a fair comparison — it defers actual I/O
    to array access time which isn't captured by this benchmark.

    Parameters
    ----------
    fits_path : str or Path
        Path to the FITS file.

    Returns
    -------
    BenchmarkResult
        Read benchmark result.
    """
    try:
        from astropy.io import fits as astropy_fits
    except ImportError:
        raise ImportError("astropy is required for FITS benchmarks.")

    fits_path = Path(fits_path)
    file_size = fits_path.stat().st_size

    start = time.perf_counter()
    with astropy_fits.open(str(fits_path), memmap=False) as hdul:
        full_data = hdul[0].data  # type: ignore[union-attr]
        if full_data is not None:
            full_data = np.array(full_data)
            shape = full_data.shape
            dtype = str(full_data.dtype)
            raw_bytes = full_data.nbytes
        else:
            shape = (0,)
            dtype = "unknown"
            raw_bytes = 0
    elapsed = time.perf_counter() - start

    return BenchmarkResult(
        name="FITS Read",
        operation="read",
        format_name="FITS",
        elapsed_seconds=elapsed,
        data_shape=shape,
        data_dtype=dtype,
        file_size_bytes=file_size,
        raw_data_bytes=raw_bytes,
    )


def benchmark_fits_partial_read(
    fits_path: str | Path,
    slices: tuple[slice, ...],
) -> BenchmarkResult:
    """Benchmark FITS partial read — must read entire file.

    Parameters
    ----------
    fits_path : str or Path
        Path to the FITS file.
    slices : tuple of slice
        Region to extract after full read.

    Returns
    -------
    BenchmarkResult
        Partial read benchmark result.
    """
    try:
        from astropy.io import fits as astropy_fits
    except ImportError:
        raise ImportError("astropy is required for FITS benchmarks.")

    fits_path = Path(fits_path)
    file_size = fits_path.stat().st_size

    start = time.perf_counter()
    with astropy_fits.open(str(fits_path), memmap=False) as hdul:
        full_data = hdul[0].data  # type: ignore[union-attr]
        if full_data is not None:
            partial = np.array(full_data[slices])
            shape = partial.shape
            dtype = str(partial.dtype)
            raw_bytes = partial.nbytes
        else:
            shape = (0,)
            dtype = "unknown"
            raw_bytes = 0
    elapsed = time.perf_counter() - start

    region_str = ", ".join(f"{s.start}:{s.stop}" for s in slices)

    return BenchmarkResult(
        name="FITS Partial Read (full file required)",
        operation="partial_read",
        format_name="FITS",
        elapsed_seconds=elapsed,
        data_shape=shape,
        data_dtype=dtype,
        file_size_bytes=file_size,
        raw_data_bytes=raw_bytes,
        extra={"region": f"[{region_str}]", "note": "FITS must read full file"},
    )


# ---------------------------------------------------------------------------
#  HDF5 benchmarks
# ---------------------------------------------------------------------------

def benchmark_hdf5_write(
    data: np.ndarray,
    output_dir: str | Path | None = None,
    compression: str | None = "gzip",
    compression_level: int = 1,
) -> BenchmarkResult:
    """Benchmark HDF5 write performance.

    Parameters
    ----------
    data : numpy.ndarray
        Data to write.
    output_dir : str or Path, optional
        Output directory.
    compression : str or None
        HDF5 compression filter ('gzip', None).
    compression_level : int
        Compression level.

    Returns
    -------
    BenchmarkResult
        Write benchmark result.
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py is required for HDF5 benchmarks.")

    if output_dir is None:
        output_dir = tempfile.mkdtemp()
    h5_path = Path(output_dir) / "bench_write.h5"

    raw_bytes = data.nbytes

    start = time.perf_counter()
    with h5py.File(str(h5_path), "w") as f:
        f.create_dataset(
            "data",
            data=data,
            compression=compression,
            compression_opts=compression_level if compression else None,
        )
    elapsed = time.perf_counter() - start

    file_size = h5_path.stat().st_size

    return BenchmarkResult(
        name="HDF5 Write",
        operation="write",
        format_name="HDF5",
        elapsed_seconds=elapsed,
        data_shape=data.shape,
        data_dtype=str(data.dtype),
        file_size_bytes=file_size,
        raw_data_bytes=raw_bytes,
        extra={"compression": compression or "none"},
    )


def benchmark_hdf5_read(
    h5_path: str | Path,
) -> BenchmarkResult:
    """Benchmark HDF5 read performance.

    Parameters
    ----------
    h5_path : str or Path
        Path to the HDF5 file.

    Returns
    -------
    BenchmarkResult
        Read benchmark result.
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py is required for HDF5 benchmarks.")

    h5_path = Path(h5_path)
    file_size = h5_path.stat().st_size

    start = time.perf_counter()
    with h5py.File(str(h5_path), "r") as f:
        full_data = f["data"][:]
    elapsed = time.perf_counter() - start

    return BenchmarkResult(
        name="HDF5 Read",
        operation="read",
        format_name="HDF5",
        elapsed_seconds=elapsed,
        data_shape=full_data.shape,
        data_dtype=str(full_data.dtype),
        file_size_bytes=file_size,
        raw_data_bytes=full_data.nbytes,
    )


def benchmark_hdf5_partial_read(
    h5_path: str | Path,
    slices: tuple[slice, ...],
) -> BenchmarkResult:
    """Benchmark HDF5 partial read.

    Parameters
    ----------
    h5_path : str or Path
        Path to the HDF5 file.
    slices : tuple of slice
        Region to read.

    Returns
    -------
    BenchmarkResult
        Partial read benchmark result.
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py is required for HDF5 benchmarks.")

    h5_path = Path(h5_path)
    file_size = h5_path.stat().st_size

    start = time.perf_counter()
    with h5py.File(str(h5_path), "r") as f:
        partial = f["data"][slices]
    elapsed = time.perf_counter() - start

    region_str = ", ".join(f"{s.start}:{s.stop}" for s in slices)

    return BenchmarkResult(
        name="HDF5 Partial Read",
        operation="partial_read",
        format_name="HDF5",
        elapsed_seconds=elapsed,
        data_shape=partial.shape,
        data_dtype=str(partial.dtype),
        file_size_bytes=file_size,
        raw_data_bytes=partial.nbytes,
        extra={"region": f"[{region_str}]"},
    )


# ---------------------------------------------------------------------------
#  NumPy benchmarks (baseline — raw binary)
# ---------------------------------------------------------------------------

def benchmark_numpy_write(
    data: np.ndarray,
    output_dir: str | Path | None = None,
) -> BenchmarkResult:
    """Benchmark raw NumPy save performance (uncompressed baseline).

    Parameters
    ----------
    data : numpy.ndarray
        Data to write.
    output_dir : str or Path, optional
        Output directory.

    Returns
    -------
    BenchmarkResult
        Write benchmark result.
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp()
    npy_path = Path(output_dir) / "bench_write.npy"

    raw_bytes = data.nbytes

    start = time.perf_counter()
    np.save(str(npy_path), data)
    elapsed = time.perf_counter() - start

    file_size = npy_path.stat().st_size

    return BenchmarkResult(
        name="NumPy Write",
        operation="write",
        format_name="NumPy",
        elapsed_seconds=elapsed,
        data_shape=data.shape,
        data_dtype=str(data.dtype),
        file_size_bytes=file_size,
        raw_data_bytes=raw_bytes,
    )


def benchmark_numpy_read(
    npy_path: str | Path,
) -> BenchmarkResult:
    """Benchmark raw NumPy load performance.

    Parameters
    ----------
    npy_path : str or Path
        Path to the .npy file.

    Returns
    -------
    BenchmarkResult
        Read benchmark result.
    """
    npy_path = Path(npy_path)
    file_size = npy_path.stat().st_size

    start = time.perf_counter()
    full_data = np.load(str(npy_path))
    elapsed = time.perf_counter() - start

    return BenchmarkResult(
        name="NumPy Read",
        operation="read",
        format_name="NumPy",
        elapsed_seconds=elapsed,
        data_shape=full_data.shape,
        data_dtype=str(full_data.dtype),
        file_size_bytes=file_size,
        raw_data_bytes=full_data.nbytes,
    )


# ---------------------------------------------------------------------------
#  Multi-format comparison runner
# ---------------------------------------------------------------------------

def run_full_comparison(
    shape: tuple[int, ...] = (2048, 2048),
    dtype: str = "float64",
    pattern: str = "realistic_sky",
    partial_region: tuple[slice, ...] | None = None,
) -> list[BenchmarkComparison]:
    """Run a full NOVA vs FITS performance comparison.

    Parameters
    ----------
    shape : tuple of int
        Data shape for benchmarks.
    dtype : str
        Data type.
    pattern : str
        Data pattern ('gaussian_noise', 'gradient', 'sparse', 'realistic_sky').
    partial_region : tuple of slice, optional
        Region for partial read test. Defaults to a 256x256 cutout.

    Returns
    -------
    list of BenchmarkComparison
        Comparison results for write, read, and partial read operations.
    """
    if partial_region is None:
        partial_region = tuple(slice(s // 4, s // 4 + 256) for s in shape)

    data = generate_test_data(shape=shape, dtype=dtype, pattern=pattern)

    results: list[BenchmarkComparison] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        # Write benchmarks
        nova_write = benchmark_nova_write(data, output_dir=tmpdir)
        fits_write = benchmark_fits_write(data, output_dir=tmpdir)
        results.append(BenchmarkComparison(nova_write, fits_write))

        nova_path = Path(tmpdir) / "bench_write.nova"
        fits_path = Path(tmpdir) / "bench_write.fits"

        # Read benchmarks
        nova_read = benchmark_nova_read(nova_path)
        fits_read = benchmark_fits_read(fits_path)
        results.append(BenchmarkComparison(nova_read, fits_read))

        # Partial read benchmarks
        nova_partial = benchmark_nova_partial_read(nova_path, partial_region)
        fits_partial = benchmark_fits_partial_read(fits_path, partial_region)
        results.append(BenchmarkComparison(nova_partial, fits_partial))

    return results


def run_multi_format_comparison(
    shape: tuple[int, ...] = (2048, 2048),
    dtype: str = "float64",
    pattern: str = "realistic_sky",
    partial_region: tuple[slice, ...] | None = None,
    n_runs: int = 3,
) -> dict[str, dict[str, BenchmarkResult]]:
    """Run a comprehensive multi-format benchmark comparison.

    Benchmarks NOVA, NOVA (Zarr), FITS, HDF5, and NumPy for write, read,
    and partial read operations.  Each benchmark is run ``n_runs`` times and
    the best (minimum) time is used.

    Parameters
    ----------
    shape : tuple of int
        Data shape.
    dtype : str
        Data type.
    pattern : str
        Data pattern.
    partial_region : tuple of slice, optional
        Region for partial read.
    n_runs : int
        Number of runs per benchmark (best time is used).

    Returns
    -------
    dict[str, dict[str, BenchmarkResult]]
        Nested dict: ``results[format_name][operation]``.
    """
    if partial_region is None:
        partial_region = tuple(slice(s // 4, s // 4 + 256) for s in shape)

    data = generate_test_data(shape=shape, dtype=dtype, pattern=pattern)

    results: dict[str, dict[str, BenchmarkResult]] = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        # --- NOVA (fast binary) ---
        best: dict[str, BenchmarkResult] = {}
        for _ in range(n_runs):
            w = benchmark_nova_write(data, output_dir=tmpdir)
            if "write" not in best or w.elapsed_seconds < best["write"].elapsed_seconds:
                best["write"] = w
        nova_path = Path(tmpdir) / "bench_write.nova"
        for _ in range(n_runs):
            r = benchmark_nova_read(nova_path)
            if "read" not in best or r.elapsed_seconds < best["read"].elapsed_seconds:
                best["read"] = r
        for _ in range(n_runs):
            p = benchmark_nova_partial_read(nova_path, partial_region)
            if "partial_read" not in best or p.elapsed_seconds < best["partial_read"].elapsed_seconds:
                best["partial_read"] = p
        results["NOVA"] = best

        # --- NOVA (Zarr) ---
        best = {}
        for _ in range(n_runs):
            w = benchmark_nova_zarr_write(data, output_dir=tmpdir)
            if "write" not in best or w.elapsed_seconds < best["write"].elapsed_seconds:
                best["write"] = w
        zarr_path = Path(tmpdir) / "bench_write.nova.zarr"
        for _ in range(n_runs):
            r = benchmark_nova_zarr_read(zarr_path)
            if "read" not in best or r.elapsed_seconds < best["read"].elapsed_seconds:
                best["read"] = r
        for _ in range(n_runs):
            p = benchmark_nova_zarr_partial_read(zarr_path, partial_region)
            if "partial_read" not in best or p.elapsed_seconds < best["partial_read"].elapsed_seconds:
                best["partial_read"] = p
        results["NOVA (Zarr)"] = best

        # --- FITS ---
        best = {}
        for _ in range(n_runs):
            w = benchmark_fits_write(data, output_dir=tmpdir)
            if "write" not in best or w.elapsed_seconds < best["write"].elapsed_seconds:
                best["write"] = w
        fits_path = Path(tmpdir) / "bench_write.fits"
        for _ in range(n_runs):
            r = benchmark_fits_read(fits_path)
            if "read" not in best or r.elapsed_seconds < best["read"].elapsed_seconds:
                best["read"] = r
        for _ in range(n_runs):
            p = benchmark_fits_partial_read(fits_path, partial_region)
            if "partial_read" not in best or p.elapsed_seconds < best["partial_read"].elapsed_seconds:
                best["partial_read"] = p
        results["FITS"] = best

        # --- HDF5 ---
        try:
            best = {}
            for _ in range(n_runs):
                w = benchmark_hdf5_write(data, output_dir=tmpdir)
                if "write" not in best or w.elapsed_seconds < best["write"].elapsed_seconds:
                    best["write"] = w
            h5_path = Path(tmpdir) / "bench_write.h5"
            for _ in range(n_runs):
                r = benchmark_hdf5_read(h5_path)
                if "read" not in best or r.elapsed_seconds < best["read"].elapsed_seconds:
                    best["read"] = r
            for _ in range(n_runs):
                p = benchmark_hdf5_partial_read(h5_path, partial_region)
                if "partial_read" not in best or p.elapsed_seconds < best["partial_read"].elapsed_seconds:
                    best["partial_read"] = p
            results["HDF5"] = best
        except ImportError:
            pass  # h5py not available

        # --- NumPy ---
        best = {}
        for _ in range(n_runs):
            w = benchmark_numpy_write(data, output_dir=tmpdir)
            if "write" not in best or w.elapsed_seconds < best["write"].elapsed_seconds:
                best["write"] = w
        npy_path = Path(tmpdir) / "bench_write.npy"
        for _ in range(n_runs):
            r = benchmark_numpy_read(npy_path)
            if "read" not in best or r.elapsed_seconds < best["read"].elapsed_seconds:
                best["read"] = r
        results["NumPy"] = best  # NumPy has no native partial read

    return results


def _dir_size(path: Path) -> int:
    """Calculate total size of all files in a directory."""
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total
