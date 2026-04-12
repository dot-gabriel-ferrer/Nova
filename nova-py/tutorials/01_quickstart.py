"""Tutorial 01: NOVA Quickstart — Creating your first NOVA dataset.

This step-by-step tutorial demonstrates how to create, write, and read
a NOVA dataset from scratch. No FITS file needed.

Steps:
  1. Create synthetic astronomical data (a simulated sky image)
  2. Create a NOVA dataset with WCS and metadata
  3. Save the dataset to disk
  4. Re-open and inspect the dataset
  5. Read a partial region (cloud-native chunk access)

Run:
    cd nova-py
    python tutorials/01_quickstart.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np


def main() -> None:
    """Run the quickstart tutorial."""
    from nova.container import NovaDataset, create_dataset
    from nova.wcs import NovaWCS, WCSAxis, AffineTransform, CelestialFrame, Projection

    print("=" * 70)
    print("  NOVA Quickstart Tutorial")
    print("  Creating your first NOVA dataset from scratch")
    print("=" * 70)
    print()

    # ── Step 1: Create synthetic data ─────────────────────────────────────
    print("Step 1: Generate a synthetic 1024×1024 astronomical image")
    print("-" * 70)

    rng = np.random.default_rng(42)
    # Background sky with Gaussian noise
    sky = rng.normal(loc=1000.0, scale=30.0, size=(1024, 1024)).astype(np.float64)
    # Add a few "stars" (Gaussian point sources)
    for _ in range(50):
        cx, cy = rng.integers(0, 1024, size=2)
        flux = rng.uniform(500, 20000)
        sigma = rng.uniform(1.5, 4.0)
        y, x = np.ogrid[-cx:1024 - cx, -cy:1024 - cy]
        sky += flux * np.exp(-(x**2 + y**2) / (2 * sigma**2))

    print(f"  Data shape: {sky.shape}")
    print(f"  Data dtype: {sky.dtype}")
    print(f"  Data range: [{sky.min():.1f}, {sky.max():.1f}]")
    print(f"  Raw size:   {sky.nbytes / (1024*1024):.1f} MB")
    print()

    # ── Step 2: Define WCS metadata ───────────────────────────────────────
    print("Step 2: Define World Coordinate System (WCS) metadata")
    print("-" * 70)

    wcs = NovaWCS(
        naxes=2,
        axes=[
            WCSAxis(index=0, ctype="RA---TAN", crpix=512.0, crval=150.12, unit="deg"),
            WCSAxis(index=1, ctype="DEC--TAN", crpix=512.0, crval=2.21, unit="deg"),
        ],
        transform=AffineTransform(cd_matrix=[
            [-7.3056e-05, 0.0],
            [0.0, 7.3056e-05],
        ]),
        celestial_frame=CelestialFrame(system="ICRS"),
        projection=Projection(code="TAN"),
    )

    print(f"  Projection:  {wcs.projection.name}")  # type: ignore[union-attr]
    print(f"  Frame:       {wcs.celestial_frame.system}")  # type: ignore[union-attr]
    print(f"  Pixel scale: {wcs.transform.pixel_scale} arcsec/pixel")
    print()

    # ── Step 3: Create and save NOVA dataset ──────────────────────────────
    print("Step 3: Create and save NOVA dataset")
    print("-" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        store_path = Path(tmpdir) / "my_observation.nova.zarr"

        # Method A: Manual (fine-grained control)
        ds = NovaDataset(store_path, mode="w")
        ds.set_science_data(sky, compression_level=3)
        ds.wcs = wcs
        ds._metadata.update({
            "nova:observer": "Tutorial User",
            "nova:telescope": "Simulated 2m Telescope",
            "nova:instrument": "SimCam",
            "nova:filter": "r-band",
            "nova:data_level": "L0",
        })
        ds.save()
        ds.close()

        # Check file size
        total_size = sum(
            f.stat().st_size
            for f in store_path.rglob("*") if f.is_file()
        )
        print(f"  Saved to:    {store_path.name}")
        print(f"  File size:   {total_size / (1024*1024):.2f} MB")
        print(f"  Compression: {sky.nbytes / total_size:.1f}x")
        print()

        # ── Step 4: Re-open and inspect ───────────────────────────────────
        print("Step 4: Re-open and inspect the dataset")
        print("-" * 70)

        ds2 = NovaDataset(store_path, mode="r")
        print(f"  Data shape:  {ds2.data.shape}")  # type: ignore[union-attr]
        print(f"  Data dtype:  {ds2.data.dtype}")  # type: ignore[union-attr]
        print(f"  WCS type:    {type(ds2.wcs).__name__}")
        print(f"  Projection:  {ds2.wcs.projection.code}")  # type: ignore[union-attr]

        # Check metadata
        meta_path = store_path / "nova_metadata.json"
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"  Observer:    {meta.get('nova:observer', 'N/A')}")
        print(f"  Telescope:   {meta.get('nova:telescope', 'N/A')}")
        print()

        # ── Step 5: Partial read (cloud-native access) ────────────────────
        print("Step 5: Read a 256×256 cutout (cloud-native chunk access)")
        print("-" * 70)

        import time
        t0 = time.perf_counter()
        cutout = np.array(ds2.data[400:656, 400:656])  # type: ignore[index]
        t1 = time.perf_counter()

        print(f"  Cutout shape: {cutout.shape}")
        print(f"  Cutout range: [{cutout.min():.1f}, {cutout.max():.1f}]")
        print(f"  Read time:    {(t1 - t0)*1000:.2f} ms")
        print(f"  Data read:    {cutout.nbytes / 1024:.1f} KB (not {sky.nbytes / (1024*1024):.1f} MB)")
        print()
        print("  → NOVA only reads the chunks needed, not the full file!")
        print("  → In cloud storage, this means ≤2 HTTP requests.")

        ds2.close()

    print()
    print("=" * 70)
    print("  ✓ Tutorial complete! You've created your first NOVA dataset.")
    print("=" * 70)


if __name__ == "__main__":
    main()
