"""Tests for the NOVA benchmarking module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from nova.benchmarks import (
    BenchmarkResult,
    BenchmarkComparison,
    generate_test_data,
    benchmark_nova_write,
    benchmark_nova_read,
    benchmark_nova_partial_read,
    benchmark_fits_write,
    benchmark_fits_read,
    benchmark_fits_partial_read,
    run_full_comparison,
)


class TestBenchmarkResult:
    """Tests for BenchmarkResult dataclass."""

    def test_compression_ratio(self) -> None:
        result = BenchmarkResult(
            name="test",
            operation="write",
            format_name="NOVA",
            elapsed_seconds=1.0,
            data_shape=(100, 100),
            data_dtype="float64",
            file_size_bytes=40000,
            raw_data_bytes=80000,
        )
        assert result.compression_ratio == pytest.approx(2.0)

    def test_throughput(self) -> None:
        result = BenchmarkResult(
            name="test",
            operation="write",
            format_name="NOVA",
            elapsed_seconds=1.0,
            data_shape=(100, 100),
            data_dtype="float64",
            file_size_bytes=40000,
            raw_data_bytes=1024 * 1024,  # 1 MB
        )
        assert result.throughput_mbps == pytest.approx(1.0)

    def test_summary_contains_info(self) -> None:
        result = BenchmarkResult(
            name="test_bench",
            operation="write",
            format_name="NOVA",
            elapsed_seconds=0.5,
            data_shape=(512, 512),
            data_dtype="float64",
            file_size_bytes=1000000,
            raw_data_bytes=2097152,
        )
        summary = result.summary()
        assert "test_bench" in summary
        assert "NOVA" in summary
        assert "write" in summary

    def test_zero_file_size(self) -> None:
        result = BenchmarkResult(
            name="test",
            operation="write",
            format_name="NOVA",
            elapsed_seconds=1.0,
            data_shape=(10,),
            data_dtype="float64",
            file_size_bytes=0,
            raw_data_bytes=80,
        )
        assert result.compression_ratio == 0.0


class TestBenchmarkComparison:
    """Tests for BenchmarkComparison dataclass."""

    def test_speedup(self) -> None:
        nova = BenchmarkResult(
            name="test", operation="write", format_name="NOVA",
            elapsed_seconds=1.0, data_shape=(100,), data_dtype="f8",
        )
        fits = BenchmarkResult(
            name="test", operation="write", format_name="FITS",
            elapsed_seconds=2.0, data_shape=(100,), data_dtype="f8",
        )
        comp = BenchmarkComparison(nova, fits)
        assert comp.speedup == pytest.approx(2.0)

    def test_summary(self) -> None:
        nova = BenchmarkResult(
            name="test", operation="write", format_name="NOVA",
            elapsed_seconds=1.0, data_shape=(100,), data_dtype="f8",
            file_size_bytes=100, raw_data_bytes=200,
        )
        fits = BenchmarkResult(
            name="test", operation="write", format_name="FITS",
            elapsed_seconds=2.0, data_shape=(100,), data_dtype="f8",
            file_size_bytes=200, raw_data_bytes=200,
        )
        comp = BenchmarkComparison(nova, fits)
        summary = comp.summary()
        assert "NOVA" in summary
        assert "FITS" in summary


class TestGenerateTestData:
    """Tests for test data generation."""

    def test_gaussian_noise(self) -> None:
        data = generate_test_data(shape=(64, 64), pattern="gaussian_noise")
        assert data.shape == (64, 64)
        assert data.dtype == np.float64

    def test_gradient(self) -> None:
        data = generate_test_data(shape=(32, 32), pattern="gradient")
        assert data.shape == (32, 32)

    def test_sparse(self) -> None:
        data = generate_test_data(shape=(64, 64), pattern="sparse")
        assert data.shape == (64, 64)
        # Most values should be zero
        assert np.count_nonzero(data) < data.size * 0.01

    def test_realistic_sky(self) -> None:
        data = generate_test_data(shape=(128, 128), pattern="realistic_sky")
        assert data.shape == (128, 128)
        assert data.mean() > 0  # Should have positive background

    def test_custom_dtype(self) -> None:
        data = generate_test_data(shape=(32, 32), dtype="float32")
        assert data.dtype == np.float32

    def test_reproducible(self) -> None:
        d1 = generate_test_data(shape=(32, 32), seed=123)
        d2 = generate_test_data(shape=(32, 32), seed=123)
        np.testing.assert_array_equal(d1, d2)

    def test_unknown_pattern_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown pattern"):
            generate_test_data(pattern="nonexistent")


class TestNovaBenchmarks:
    """Tests for NOVA benchmark functions."""

    def test_write_and_read(self) -> None:
        data = generate_test_data(shape=(128, 128))
        with tempfile.TemporaryDirectory() as tmpdir:
            write_result = benchmark_nova_write(data, output_dir=tmpdir)
            assert write_result.format_name == "NOVA"
            assert write_result.operation == "write"
            assert write_result.elapsed_seconds > 0
            assert write_result.file_size_bytes > 0

            nova_path = Path(tmpdir) / "bench_write.nova"
            read_result = benchmark_nova_read(nova_path)
            assert read_result.format_name == "NOVA"
            assert read_result.operation == "read"
            assert read_result.elapsed_seconds > 0

    def test_partial_read(self) -> None:
        data = generate_test_data(shape=(256, 256))
        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark_nova_write(data, output_dir=tmpdir)
            nova_path = Path(tmpdir) / "bench_write.nova"

            result = benchmark_nova_partial_read(
                nova_path, (slice(50, 100), slice(50, 100))
            )
            assert result.operation == "partial_read"
            assert result.data_shape == (50, 50)


class TestFitsBenchmarks:
    """Tests for FITS benchmark functions."""

    def test_write_and_read(self) -> None:
        data = generate_test_data(shape=(128, 128))
        with tempfile.TemporaryDirectory() as tmpdir:
            write_result = benchmark_fits_write(data, output_dir=tmpdir)
            assert write_result.format_name == "FITS"
            assert write_result.operation == "write"
            assert write_result.file_size_bytes > 0

            fits_path = Path(tmpdir) / "bench_write.fits"
            read_result = benchmark_fits_read(fits_path)
            assert read_result.format_name == "FITS"
            assert read_result.operation == "read"

    def test_partial_read(self) -> None:
        data = generate_test_data(shape=(256, 256))
        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark_fits_write(data, output_dir=tmpdir)
            fits_path = Path(tmpdir) / "bench_write.fits"

            result = benchmark_fits_partial_read(
                fits_path, (slice(50, 100), slice(50, 100))
            )
            assert result.operation == "partial_read"
            assert result.data_shape == (50, 50)


class TestRunFullComparison:
    """Tests for the full comparison runner."""

    def test_full_comparison(self) -> None:
        results = run_full_comparison(
            shape=(128, 128), dtype="float64", pattern="gaussian_noise"
        )
        assert len(results) == 3  # write, read, partial_read

        # Check write comparison
        assert results[0].nova_result.operation == "write"
        assert results[0].fits_result.operation == "write"

        # Check read comparison
        assert results[1].nova_result.operation == "read"
        assert results[1].fits_result.operation == "read"

        # Check partial read comparison
        assert results[2].nova_result.operation == "partial_read"
        assert results[2].fits_result.operation == "partial_read"
