"""Tutorial 07: Migration and Streaming -- Batch conversion and time-series.

This tutorial demonstrates two Phase 2 features:

1. Batch migration: converting an entire directory of FITS files to NOVA
2. Streaming: appending time-series frames to a NOVA dataset

These features are designed for production pipelines where data arrives
continuously or needs to be converted in bulk.
"""

from __future__ import annotations

import tempfile
import shutil
from pathlib import Path

import numpy as np


def run_tutorial():
    tmpdir = Path(tempfile.mkdtemp())

    try:
        # ---- Part 1: Batch Migration ----
        print("=" * 60)
        print("Part 1: Batch FITS -> NOVA Migration")
        print("=" * 60)

        # Create some sample FITS files to migrate
        fits_dir = tmpdir / "fits_archive"
        fits_dir.mkdir()
        nova_dir = tmpdir / "nova_archive"

        from astropy.io import fits

        rng = np.random.default_rng(42)
        for i in range(5):
            data = rng.normal(100, 10, (64, 64)).astype(np.float32)
            hdu = fits.PrimaryHDU(data=data)
            hdu.header["OBJECT"] = f"Star Field {i+1}"
            hdu.header["EXPTIME"] = 30.0 + i * 10
            hdu.writeto(str(fits_dir / f"obs_{i:03d}.fits"), overwrite=True)
        print(f"Created {len(list(fits_dir.glob('*.fits')))} FITS files")

        # Dry-run first to see what would be converted
        from nova.migrate import migrate_directory

        report = migrate_directory(fits_dir, nova_dir, dry_run=True)
        print(f"\nDry-run: {report.total_files} files would be converted")

        # Actual conversion with verification
        report = migrate_directory(fits_dir, nova_dir, verify=True)
        print(f"\nMigration complete:")
        print(f"  Converted: {report.converted}")
        print(f"  Failed:    {report.failed}")
        print(f"  Verified:  {report.verified}")
        print(f"  Time:      {report.elapsed_seconds:.2f}s")

        # Incremental -- running again should skip all files
        report2 = migrate_directory(fits_dir, nova_dir, incremental=True)
        print(f"\nIncremental re-run:")
        print(f"  Converted: {report2.converted}")
        print(f"  Skipped:   {report2.skipped}")

        # ---- Part 2: Streaming / Time-Series ----
        print("\n" + "=" * 60)
        print("Part 2: Streaming Time-Series Ingest")
        print("=" * 60)

        from nova.streaming import StreamWriter, open_appendable

        store_path = tmpdir / "timeseries.nova.zarr"

        # Simulate a camera producing 16x16 frames
        print("\nStreaming 20 frames from a simulated camera...")
        with StreamWriter(store_path, frame_shape=(16, 16), buffer_size=5) as writer:
            for i in range(20):
                frame = rng.poisson(100, (16, 16)).astype(float)
                writer.append(frame)
            print(f"  Total frames written: {writer.total_frames}")

        # Read back the time-series
        from nova.container import NovaDataset

        ds = NovaDataset(store_path, mode="r")
        cube = np.array(ds.data)
        print(f"  Data cube shape: {cube.shape}")
        print(f"  Frame 0 mean: {cube[0].mean():.1f}")
        print(f"  Frame 19 mean: {cube[19].mean():.1f}")
        ds.close()

        # ---- Part 3: Pipeline Adapters ----
        print("\n" + "=" * 60)
        print("Part 3: Pipeline Adapters (CCDData)")
        print("=" * 60)

        from nova.adapters import to_ccddata, from_ccddata

        # Convert a NOVA file to CCDData for use with ccdproc
        nova_stores = list(nova_dir.rglob("*.nova.zarr"))
        if nova_stores:
            ccd = to_ccddata(nova_stores[0], unit="adu")
            print(f"\nCCDData from NOVA:")
            print(f"  Shape: {ccd.data.shape}")
            print(f"  Unit:  {ccd.unit}")
            print(f"  Mean:  {ccd.data.mean():.2f}")

            # Convert back to NOVA
            roundtrip_path = tmpdir / "roundtrip.nova.zarr"
            from_ccddata(ccd, roundtrip_path)
            ds2 = NovaDataset(roundtrip_path, mode="r")
            print(f"  Round-trip OK: {np.allclose(ccd.data, np.array(ds2.data))}")
            ds2.close()

        print("\nTutorial complete.")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    run_tutorial()
