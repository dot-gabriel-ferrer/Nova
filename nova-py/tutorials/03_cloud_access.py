"""Tutorial 03: Cloud-Native Access — Efficient chunk-based data retrieval.

This tutorial demonstrates NOVA's cloud-native architecture:
  1. How Zarr chunks enable partial data access
  2. How the chunk index works
  3. Simulating cloud-like access patterns (read only what you need)
  4. Comparing full-file vs partial reads

Run:
    cd nova-py
    python tutorials/03_cloud_access.py
"""

from __future__ import annotations

import json
import time
import tempfile
from pathlib import Path

import numpy as np


def main() -> None:
    """Run the cloud access tutorial."""
    from nova.container import NovaDataset
    from nova.wcs import NovaWCS, WCSAxis, AffineTransform, CelestialFrame, Projection

    print("=" * 70)
    print("  Cloud-Native Access Tutorial")
    print("  Efficient chunk-based data retrieval with NOVA")
    print("=" * 70)
    print()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # ── Step 1: Create a large dataset ────────────────────────────────
        print("Step 1: Create a large 4096×4096 dataset (128 MB raw)")
        print("-" * 70)

        rng = np.random.default_rng(42)
        data = rng.normal(1000.0, 30.0, (4096, 4096)).astype(np.float64)

        store_path = tmpdir / "large_survey.nova.zarr"
        ds = NovaDataset(store_path, mode="w")
        ds.set_science_data(data, compression_level=3)

        wcs = NovaWCS(
            naxes=2,
            axes=[
                WCSAxis(index=0, ctype="RA---TAN", crpix=2048.0, crval=180.0, unit="deg"),
                WCSAxis(index=1, ctype="DEC--TAN", crpix=2048.0, crval=0.0, unit="deg"),
            ],
            transform=AffineTransform(cd_matrix=[
                [-2.778e-04, 0.0],
                [0.0, 2.778e-04],
            ]),
            celestial_frame=CelestialFrame(system="ICRS"),
            projection=Projection(code="TAN"),
        )
        ds.wcs = wcs
        ds.save()
        ds.close()

        total_size = sum(f.stat().st_size for f in store_path.rglob("*") if f.is_file())
        print(f"  Data shape:   {data.shape}")
        print(f"  Raw size:     {data.nbytes / (1024*1024):.1f} MB")
        print(f"  NOVA size:    {total_size / (1024*1024):.1f} MB")
        print(f"  Compression:  {data.nbytes / total_size:.1f}x")
        print()

        # ── Step 2: Understand the chunk structure ────────────────────────
        print("Step 2: Understand the Zarr chunk structure")
        print("-" * 70)

        chunk_size = (512, 512)
        n_chunks_x = 4096 // chunk_size[0]
        n_chunks_y = 4096 // chunk_size[1]
        print(f"  Chunk shape:  {chunk_size}")
        print(f"  Grid:         {n_chunks_x} × {n_chunks_y} = {n_chunks_x * n_chunks_y} chunks")
        print(f"  Chunk size:   {chunk_size[0] * chunk_size[1] * 8 / (1024*1024):.1f} MB (raw)")
        print()
        print("  Chunk layout:")
        print("  ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐")
        for row in range(n_chunks_x):
            cells = "│".join(f" {row},{c} " for c in range(n_chunks_y))
            print(f"  │{cells}│")
            if row < n_chunks_x - 1:
                print("  ├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤")
        print("  └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘")
        print()

        # ── Step 3: Inspect the chunk index ───────────────────────────────
        print("Step 3: Inspect the NOVA chunk index")
        print("-" * 70)

        index_path = store_path / "nova_index.json"
        with open(index_path) as f:
            index = json.load(f)

        chunks = index.get("nova:chunks", [])
        print(f"  Total chunks indexed: {len(chunks)}")
        if chunks:
            print(f"  First chunk:")
            first = chunks[0]
            print(f"    Path:   {first['nova:path']}")
            print(f"    Size:   {first['nova:size']} bytes")
            print(f"    SHA256: {first['nova:sha256'][:32]}...")
        print()
        print("  → In cloud mode, the client reads this index first (1 HTTP request)")
        print("  → Then fetches only the needed chunks (1 more HTTP request each)")
        print()

        # ── Step 4: Demonstrate partial vs full reads ─────────────────────
        print("Step 4: Compare full read vs partial read performance")
        print("-" * 70)

        ds = NovaDataset(store_path, mode="r")

        # Full read
        t0 = time.perf_counter()
        full_data = np.array(ds.data)  # type: ignore[arg-type]
        t_full = time.perf_counter() - t0

        # Small cutout (256×256 = 0.5 MB vs 128 MB)
        t0 = time.perf_counter()
        cutout_small = np.array(ds.data[1000:1256, 2000:2256])  # type: ignore[index]
        t_small = time.perf_counter() - t0

        # Medium cutout (1024×1024 = 8 MB vs 128 MB)
        t0 = time.perf_counter()
        cutout_medium = np.array(ds.data[1000:2024, 1000:2024])  # type: ignore[index]
        t_medium = time.perf_counter() - t0

        # Tiny cutout (64×64 = 32 KB vs 128 MB)
        t0 = time.perf_counter()
        cutout_tiny = np.array(ds.data[2000:2064, 2000:2064])  # type: ignore[index]
        t_tiny = time.perf_counter() - t0

        ds.close()

        print(f"  {'Operation':<30s} {'Shape':>12s} {'Size':>10s} {'Time':>10s}")
        print(f"  {'-'*30} {'-'*12} {'-'*10} {'-'*10}")
        print(f"  {'Full read':<30s} {'4096×4096':>12s} "
              f"{'128 MB':>10s} {t_full*1000:>8.1f} ms")
        print(f"  {'Medium cutout':<30s} {'1024×1024':>12s} "
              f"{'8 MB':>10s} {t_medium*1000:>8.1f} ms")
        print(f"  {'Small cutout':<30s} {'256×256':>12s} "
              f"{'0.5 MB':>10s} {t_small*1000:>8.1f} ms")
        print(f"  {'Tiny cutout':<30s} {'64×64':>12s} "
              f"{'32 KB':>10s} {t_tiny*1000:>8.1f} ms")
        print()
        print("  Key insight: NOVA reads ONLY the chunks that overlap your region.")
        print("  In cloud storage, this means:")
        print("    • 1 HTTP request for the chunk index")
        print("    • 1 HTTP Range request per needed chunk")
        print("    • No need to download the full file!")
        print()

        # ── Step 5: Cloud access pattern simulation ───────────────────────
        print("Step 5: Simulated cloud access pattern")
        print("-" * 70)

        # Simulate what happens in the cloud
        region = (slice(1000, 1256), slice(2000, 2256))
        chunk_rows = range(region[0].start // 512, (region[0].stop - 1) // 512 + 1)
        chunk_cols = range(region[1].start // 512, (region[1].stop - 1) // 512 + 1)
        needed_chunks = [(r, c) for r in chunk_rows for c in chunk_cols]

        print(f"  Requested region: [{region[0].start}:{region[0].stop}, "
              f"{region[1].start}:{region[1].stop}]")
        print(f"  Chunks needed:    {len(needed_chunks)}")
        for r, c in needed_chunks:
            print(f"    Chunk ({r}, {c}): rows [{r*512}:{(r+1)*512}], "
                  f"cols [{c*512}:{(c+1)*512}]")
        print()
        print(f"  HTTP requests needed: 1 (index) + {len(needed_chunks)} (chunks) "
              f"= {1 + len(needed_chunks)} total")
        print(f"  Data transferred:     ~{len(needed_chunks) * 2:.0f} MB "
              f"(vs 128 MB full file)")
        print(f"  Savings:              {(1 - len(needed_chunks) * 2 / 128) * 100:.0f}%")

    print()
    print("=" * 70)
    print("  ✓ Tutorial complete! Cloud-native access patterns demonstrated.")
    print("=" * 70)


if __name__ == "__main__":
    main()
