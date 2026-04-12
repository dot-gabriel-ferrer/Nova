"""Tests for the NOVA container module."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from nova.container import NovaDataset, create_dataset, open_dataset
from nova.wcs import NovaWCS, WCSAxis, AffineTransform, CelestialFrame


class TestNovaDataset:
    """Tests for NovaDataset."""

    def test_create_and_read(self, tmp_path: Path) -> None:
        """Test creating a NOVA dataset and reading it back."""
        store_path = tmp_path / "test.nova.zarr"
        data = np.random.default_rng(42).standard_normal((256, 256))

        # Create
        ds = NovaDataset(store_path, mode="w")
        ds.set_science_data(data)
        ds.save()
        ds.close()

        # Read
        ds2 = NovaDataset(store_path, mode="r")
        assert ds2.data is not None
        np.testing.assert_array_almost_equal(np.array(ds2.data), data)
        ds2.close()

    def test_create_dataset_helper(self, tmp_path: Path) -> None:
        """Test the create_dataset convenience function."""
        store_path = tmp_path / "test2.nova.zarr"
        data = np.ones((128, 128), dtype=np.float32)

        ds = create_dataset(store_path, data)
        assert ds.data is not None
        assert (tmp_path / "test2.nova.zarr" / "nova_metadata.json").exists()
        assert (tmp_path / "test2.nova.zarr" / "nova_index.json").exists()
        ds.close()

    def test_with_wcs(self, tmp_path: Path) -> None:
        """Test creating a dataset with WCS metadata."""
        store_path = tmp_path / "wcs.nova.zarr"
        data = np.zeros((512, 512), dtype=np.float64)

        wcs = NovaWCS(
            naxes=2,
            axes=[
                WCSAxis(0, "RA---TAN", 256.0, 180.0, "deg"),
                WCSAxis(1, "DEC--TAN", 256.0, 45.0, "deg"),
            ],
            transform=AffineTransform(
                cd_matrix=[[-1e-4, 0.0], [0.0, 1e-4]]
            ),
            celestial_frame=CelestialFrame("ICRS", 2000.0),
        )

        ds = create_dataset(store_path, data, wcs=wcs)
        assert (store_path / "wcs.json").exists()

        # Read back
        ds2 = open_dataset(store_path)
        assert ds2.wcs is not None
        assert ds2.wcs.naxes == 2
        assert ds2.wcs.celestial_frame is not None
        assert ds2.wcs.celestial_frame.system == "ICRS"
        ds.close()
        ds2.close()

    def test_metadata_json_valid(self, tmp_path: Path) -> None:
        """Test that nova_metadata.json is valid JSON."""
        store_path = tmp_path / "meta.nova.zarr"
        data = np.ones((64, 64))

        ds = create_dataset(store_path, data)
        meta_path = store_path / "nova_metadata.json"
        with open(meta_path) as f:
            meta = json.load(f)

        assert meta["@context"] == "https://nova-astro.org/v0.1/context.jsonld"
        assert meta["@type"] == "nova:Observation"
        assert "nova:version" in meta
        ds.close()

    def test_chunk_index_created(self, tmp_path: Path) -> None:
        """Test that nova_index.json is created with chunk hashes."""
        store_path = tmp_path / "index.nova.zarr"
        data = np.ones((64, 64))

        ds = create_dataset(store_path, data)
        index_path = store_path / "nova_index.json"
        assert index_path.exists()

        with open(index_path) as f:
            index = json.load(f)

        assert index["@type"] == "nova:ChunkIndex"
        assert "nova:chunks" in index
        ds.close()

    def test_context_manager(self, tmp_path: Path) -> None:
        """Test using NovaDataset as a context manager."""
        store_path = tmp_path / "ctx.nova.zarr"
        data = np.ones((64, 64))

        with NovaDataset(store_path, mode="w") as ds:
            ds.set_science_data(data)
            ds.save()

        with NovaDataset(store_path, mode="r") as ds:
            assert ds.data is not None

    def test_uncertainty_and_mask(self, tmp_path: Path) -> None:
        """Test storing uncertainty and mask planes."""
        store_path = tmp_path / "full.nova.zarr"
        data = np.ones((128, 128), dtype=np.float64)
        unc = np.ones((128, 128), dtype=np.float32) * 0.1
        mask = np.zeros((128, 128), dtype=np.uint8)

        ds = NovaDataset(store_path, mode="w")
        ds.set_science_data(data)
        ds.set_uncertainty(unc)
        ds.set_mask(mask)
        ds.save()
        ds.close()

        ds2 = NovaDataset(store_path, mode="r")
        assert ds2.data is not None
        assert ds2.uncertainty is not None
        assert ds2.mask is not None
        np.testing.assert_array_almost_equal(
            np.array(ds2.uncertainty), unc
        )
        ds2.close()
