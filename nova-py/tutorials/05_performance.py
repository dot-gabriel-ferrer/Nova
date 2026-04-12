"""Tutorial 05: Performance Comparison -- NOVA vs FITS benchmarks.

This tutorial runs concrete performance benchmarks comparing NOVA and FITS:
  1. Write speed comparison
  2. Read speed comparison
  3. Partial read (cloud access simulation)
  4. Compression ratio comparison
  5. Results summary with different data patterns

Run:
    cd nova-py
    python tutorials/05_performance.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np


def main() -> None:
    """Run the performance comparison tutorial."""
    from nova.benchmarks import (
        generate_test_data,
        run_full_comparison,
        benchmark_nova_write,
        benchmark_fits_write,
    )

    print("=" * 70)
    print("  Performance Comparison: NOVA vs FITS")
    print("  Benchmarking write, read, compression, and partial access")
    print("=" * 70)
    print()

    # -- Step 1: Full comparison with realistic sky data -------------------
    print("Step 1: Full comparison -- 2048x2048 realistic sky (float64)")
    print("-" * 70)

    results = run_full_comparison(
        shape=(2048, 2048),
        dtype="float64",
        pattern="realistic_sky",
    )

    for comp in results:
        print(comp.summary())
        print()

    # -- Step 2: Compression comparison with different patterns ------------
    print("Step 2: Compression ratio with different data patterns")
    print("-" * 70)

    patterns = ["gaussian_noise", "gradient", "sparse", "realistic_sky"]
    shape = (1024, 1024)

    print(f"  {'Pattern':<20s} {'NOVA size':>12s} {'FITS size':>12s} "
          f"{'NOVA ratio':>12s} {'Savings':>10s}")
    print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*12} {'-'*10}")

    for pattern in patterns:
        data = generate_test_data(shape=shape, dtype="float64", pattern=pattern)

        with tempfile.TemporaryDirectory() as tmpdir:
            nova_res = benchmark_nova_write(data, output_dir=tmpdir)
            fits_res = benchmark_fits_write(data, output_dir=tmpdir)

        nova_mb = nova_res.file_size_bytes / (1024 * 1024)
        fits_mb = fits_res.file_size_bytes / (1024 * 1024)
        savings = (1 - nova_res.file_size_bytes / fits_res.file_size_bytes) * 100

        print(f"  {pattern:<20s} {nova_mb:>10.2f} MB {fits_mb:>10.2f} MB "
              f"{nova_res.compression_ratio:>10.2f}x {savings:>8.1f}%")

    print()
    print("  -> NOVA's ZSTD compression is most effective on structured data")
    print("  -> Gradient data compresses extremely well (high redundancy)")
    print("  -> Random noise is least compressible (as expected)")
    print()

    # -- Step 3: Scaling benchmark -----------------------------------------
    print("Step 3: Performance scaling with data size")
    print("-" * 70)

    sizes = [(512, 512), (1024, 1024), (2048, 2048)]

    print(f"  {'Shape':<12s} {'Raw MB':>8s} {'NOVA write':>12s} "
          f"{'FITS write':>12s} {'NOVA read':>11s} {'FITS read':>11s}")
    print(f"  {'-'*12} {'-'*8} {'-'*12} {'-'*12} {'-'*11} {'-'*11}")

    for shape in sizes:
        comps = run_full_comparison(
            shape=shape, dtype="float64", pattern="realistic_sky"
        )
        write_comp = comps[0]
        read_comp = comps[1]
        raw_mb = write_comp.nova_result.raw_data_bytes / (1024 * 1024)

        print(
            f"  {str(shape):<12s} {raw_mb:>6.1f}  "
            f"{write_comp.nova_result.throughput_mbps:>9.1f} MB/s "
            f"{write_comp.fits_result.throughput_mbps:>9.1f} MB/s "
            f"{read_comp.nova_result.throughput_mbps:>8.1f} MB/s "
            f"{read_comp.fits_result.throughput_mbps:>8.1f} MB/s"
        )

    print()

    # -- Step 4: Partial read advantage ------------------------------------
    print("Step 4: Partial read advantage (cloud-access simulation)")
    print("-" * 70)

    data_large = generate_test_data(
        shape=(2048, 2048), dtype="float64", pattern="realistic_sky"
    )

    cutout_sizes = [
        ("64x64", (slice(900, 964), slice(900, 964))),
        ("256x256", (slice(900, 1156), slice(900, 1156))),
        ("512x512", (slice(768, 1280), slice(768, 1280))),
    ]

    from nova.benchmarks import (
        benchmark_nova_read,
        benchmark_nova_partial_read,
        benchmark_fits_read,
        benchmark_fits_partial_read,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        # Write both formats
        nova_write = benchmark_nova_write(data_large, output_dir=tmpdir)
        fits_write = benchmark_fits_write(data_large, output_dir=tmpdir)
        nova_path = Path(tmpdir) / "bench_write.nova.zarr"
        fits_path = Path(tmpdir) / "bench_write.fits"

        print(f"  Full image:  2048x2048 ({data_large.nbytes / (1024*1024):.0f} MB raw)")
        print()
        print(f"  {'Cutout':<12s} {'NOVA time':>11s} {'FITS time':>11s} {'Speedup':>10s}")
        print(f"  {'-'*12} {'-'*11} {'-'*11} {'-'*10}")

        for name, slices in cutout_sizes:
            nova_partial = benchmark_nova_partial_read(nova_path, slices)
            fits_partial = benchmark_fits_partial_read(fits_path, slices)

            speedup = fits_partial.elapsed_seconds / max(nova_partial.elapsed_seconds, 1e-9)

            print(
                f"  {name:<12s} "
                f"{nova_partial.elapsed_seconds*1000:>8.2f} ms "
                f"{fits_partial.elapsed_seconds*1000:>8.2f} ms "
                f"{speedup:>8.1f}x"
            )

    print()
    print("  -> NOVA's chunk-based access reads only what's needed")
    print("  -> FITS must load the entire file for any access pattern")
    print("  -> The advantage grows with larger files and smaller cutouts")
    print()

    # -- Summary ----------------------------------------------------------
    print("=" * 70)
    print("  Summary: NOVA Advantages")
    print("=" * 70)
    print()
    print("  OK ZSTD compression -> smaller files (30-90% savings on structured data)")
    print("  OK Chunk-based access -> read only what you need")
    print("  OK Cloud-native -> <=2 HTTP requests for any region")
    print("  OK Little-endian native -> no byte-swapping on modern hardware")
    print("  OK Parallel writes -> lock-free concurrent chunk updates")
    print()
    print("=" * 70)
    print("  OK Tutorial complete! Performance benchmarks demonstrated.")
    print("=" * 70)


if __name__ == "__main__":
    main()
