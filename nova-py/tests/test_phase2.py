"""Tests for Phase 2 features: remote access, batch migration, streaming,
and pipeline adapters.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from nova.container import NovaDataset, NOVA_VERSION


# -----------------------------------------------------------------------
#  Remote access helpers
# -----------------------------------------------------------------------

class TestRemoteHelpers:
    """Tests for the remote URL detection utility."""

    def test_http_detected(self) -> None:
        from nova.remote import is_remote_url
        assert is_remote_url("http://example.org/obs.nova.zarr") is True

    def test_https_detected(self) -> None:
        from nova.remote import is_remote_url
        assert is_remote_url("https://archive.org/obs.nova.zarr") is True

    def test_s3_detected(self) -> None:
        from nova.remote import is_remote_url
        assert is_remote_url("s3://bucket/obs.nova.zarr") is True

    def test_gs_detected(self) -> None:
        from nova.remote import is_remote_url
        assert is_remote_url("gs://bucket/obs.nova.zarr") is True

    def test_local_not_remote(self) -> None:
        from nova.remote import is_remote_url
        assert is_remote_url("/data/obs.nova.zarr") is False

    def test_relative_not_remote(self) -> None:
        from nova.remote import is_remote_url
        assert is_remote_url("obs.nova.zarr") is False

    def test_azure_detected(self) -> None:
        from nova.remote import is_remote_url
        assert is_remote_url("az://container/obs.nova.zarr") is True
        assert is_remote_url("abfs://container/obs.nova.zarr") is True


# -----------------------------------------------------------------------
#  Batch migration
# -----------------------------------------------------------------------

class TestMigration:
    """Tests for the batch migration tool."""

    def setup_method(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp())
        self.src = self.tmpdir / "fits_src"
        self.dst = self.tmpdir / "nova_dst"
        self.src.mkdir()

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _create_fits(self, name: str, shape: tuple[int, int] = (32, 32)) -> Path:
        from astropy.io import fits
        data = np.random.default_rng(42).normal(100, 10, shape).astype(np.float32)
        hdu = fits.PrimaryHDU(data=data)
        path = self.src / name
        hdu.writeto(str(path), overwrite=True)
        return path

    def test_discover_fits_files(self) -> None:
        from nova.migrate import discover_fits_files
        self._create_fits("a.fits")
        self._create_fits("b.fit")
        (self.src / "subdir").mkdir()
        self._create_fits("subdir/c.fits")
        # Write out the last one properly
        from astropy.io import fits
        fits.PrimaryHDU(data=np.zeros((4, 4))).writeto(
            str(self.src / "subdir" / "c.fits"), overwrite=True,
        )

        found = discover_fits_files(self.src)
        names = [p.name for p in found]
        assert "a.fits" in names
        assert "b.fit" in names
        assert "c.fits" in names
        assert len(found) >= 3

    def test_migrate_directory_dry_run(self) -> None:
        from nova.migrate import migrate_directory
        self._create_fits("test1.fits")
        self._create_fits("test2.fits")

        report = migrate_directory(self.src, self.dst, dry_run=True)
        assert report.total_files == 2
        assert report.converted == 2
        assert report.failed == 0
        # Dry-run should not create any output
        assert not self.dst.exists()

    def test_migrate_directory_single(self) -> None:
        from nova.migrate import migrate_directory
        self._create_fits("obs.fits")

        report = migrate_directory(self.src, self.dst, parallel=1)
        assert report.total_files == 1
        assert report.converted == 1
        assert report.failed == 0
        # Check output exists
        nova_stores = list(self.dst.rglob("*.nova.zarr"))
        assert len(nova_stores) == 1

    def test_migrate_with_verify(self) -> None:
        from nova.migrate import migrate_directory
        self._create_fits("verified.fits")

        report = migrate_directory(self.src, self.dst, verify=True)
        assert report.converted == 1
        assert report.verified == 1

    def test_migrate_incremental(self) -> None:
        from nova.migrate import migrate_directory
        self._create_fits("inc.fits")

        # First run
        report1 = migrate_directory(self.src, self.dst, incremental=True)
        assert report1.converted == 1
        assert report1.skipped == 0

        # Second run -- should skip
        report2 = migrate_directory(self.src, self.dst, incremental=True)
        assert report2.converted == 0
        assert report2.skipped == 1

    def test_migration_report_summary(self) -> None:
        from nova.migrate import MigrationReport
        report = MigrationReport(
            source_dir="/src", dest_dir="/dst",
            total_files=5, converted=4, failed=1,
        )
        text = report.summary()
        assert "5" in text
        assert "4" in text
        assert "1" in text


# -----------------------------------------------------------------------
#  Streaming / append support
# -----------------------------------------------------------------------

class TestStreaming:
    """Tests for append-mode streaming writes."""

    def setup_method(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp())

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_stream_writer_basic(self) -> None:
        from nova.streaming import StreamWriter
        store = self.tmpdir / "stream.nova.zarr"
        writer = StreamWriter(store, frame_shape=(16, 16), buffer_size=2)

        frame = np.ones((16, 16))
        writer.append(frame)
        writer.append(frame * 2)  # triggers flush at buffer_size=2
        writer.append(frame * 3)
        writer.close()

        # Read back
        ds = NovaDataset(store, mode="r")
        data = np.array(ds.data)
        assert data.shape == (3, 16, 16)
        np.testing.assert_array_almost_equal(data[0], 1.0)
        np.testing.assert_array_almost_equal(data[1], 2.0)
        np.testing.assert_array_almost_equal(data[2], 3.0)
        ds.close()

    def test_stream_writer_context_manager(self) -> None:
        from nova.streaming import StreamWriter
        store = self.tmpdir / "ctx.nova.zarr"

        with StreamWriter(store, frame_shape=(8,), buffer_size=5) as w:
            for i in range(7):
                w.append(np.full((8,), float(i)))

        ds = NovaDataset(store, mode="r")
        data = np.array(ds.data)
        assert data.shape == (7, 8)
        ds.close()

    def test_open_appendable(self) -> None:
        from nova.streaming import open_appendable
        store = self.tmpdir / "append.nova.zarr"

        writer = open_appendable(store, frame_shape=(4, 4), buffer_size=3)
        for i in range(5):
            writer.append(np.full((4, 4), float(i)))
        writer.close()

        ds = NovaDataset(store, mode="r")
        assert np.array(ds.data).shape == (5, 4, 4)
        ds.close()

    def test_append_frame_convenience(self) -> None:
        from nova.streaming import open_appendable, append_frame
        store = self.tmpdir / "conv.nova.zarr"

        w = open_appendable(store, frame_shape=(4,), buffer_size=100)
        append_frame(w, np.array([1.0, 2.0, 3.0, 4.0]))
        append_frame(w, np.array([5.0, 6.0, 7.0, 8.0]))
        w.close()

        ds = NovaDataset(store, mode="r")
        data = np.array(ds.data)
        assert data.shape == (2, 4)
        np.testing.assert_array_equal(data[0], [1, 2, 3, 4])
        ds.close()

    def test_stream_metadata_written(self) -> None:
        from nova.streaming import StreamWriter
        store = self.tmpdir / "meta.nova.zarr"

        with StreamWriter(store, frame_shape=(8,), buffer_size=1) as w:
            w.append(np.ones(8))

        meta_path = store / "nova_metadata.json"
        assert meta_path.exists()
        with open(meta_path) as f:
            meta = json.load(f)
        assert meta["@type"] == "nova:TimeSeries"
        assert meta["nova:total_frames"] == 1

    def test_stream_wrong_shape_raises(self) -> None:
        from nova.streaming import StreamWriter
        store = self.tmpdir / "bad.nova.zarr"
        w = StreamWriter(store, frame_shape=(8,), buffer_size=1)
        with pytest.raises(ValueError, match="does not match"):
            w.append(np.ones(16))
        w.close()


# -----------------------------------------------------------------------
#  Pipeline adapters
# -----------------------------------------------------------------------

class TestAdapters:
    """Tests for pipeline adapter helpers."""

    def setup_method(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp())

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_dataset(self) -> Path:
        store = self.tmpdir / "adapt.nova.zarr"
        ds = NovaDataset(store, mode="w")
        ds.set_science_data(np.random.default_rng(0).normal(0, 1, (64, 64)))
        ds.set_uncertainty(np.ones((64, 64)) * 0.1)
        ds.set_mask(np.zeros((64, 64), dtype=np.uint8))
        ds.save()
        ds.close()
        return store

    def test_to_ccddata(self) -> None:
        from nova.adapters import to_ccddata
        store = self._make_dataset()
        ccd = to_ccddata(store, unit="adu")
        assert ccd.data.shape == (64, 64)
        assert ccd.uncertainty is not None
        assert ccd.mask is not None

    def test_from_ccddata(self) -> None:
        from astropy.nddata import CCDData, StdDevUncertainty
        import astropy.units as u
        from nova.adapters import from_ccddata

        data = np.ones((32, 32))
        ccd = CCDData(data, unit=u.adu, uncertainty=StdDevUncertainty(data * 0.05))
        out = self.tmpdir / "from_ccd.nova.zarr"
        from_ccddata(ccd, out)

        ds = NovaDataset(out, mode="r")
        assert ds.data is not None
        np.testing.assert_array_almost_equal(np.array(ds.data), 1.0)
        ds.close()

    def test_to_nddata(self) -> None:
        from nova.adapters import to_nddata
        store = self._make_dataset()
        nd = to_nddata(store)
        assert nd.data.shape == (64, 64)

    def test_nova_to_hdulist(self) -> None:
        from nova.adapters import nova_to_hdulist
        store = self._make_dataset()
        hdul = nova_to_hdulist(store)
        assert len(hdul) >= 1
        assert hdul[0].data.shape == (64, 64)

    def test_roundtrip_ccddata(self) -> None:
        from astropy.nddata import CCDData
        import astropy.units as u
        from nova.adapters import to_ccddata, from_ccddata

        original = CCDData(
            np.random.default_rng(1).normal(0, 1, (48, 48)),
            unit=u.electron,
        )
        nova_path = self.tmpdir / "rt.nova.zarr"
        from_ccddata(original, nova_path)
        recovered = to_ccddata(nova_path, unit="electron")
        np.testing.assert_array_almost_equal(recovered.data, original.data)


# -----------------------------------------------------------------------
#  CLI migrate subcommand
# -----------------------------------------------------------------------

class TestCLIMigrate:
    """Tests for the ``nova migrate`` CLI subcommand."""

    def setup_method(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp())
        self.src = self.tmpdir / "fits"
        self.dst = self.tmpdir / "nova"
        self.src.mkdir()

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _create_fits(self) -> None:
        from astropy.io import fits
        fits.PrimaryHDU(data=np.zeros((8, 8))).writeto(
            str(self.src / "test.fits"), overwrite=True,
        )

    def test_migrate_dry_run(self) -> None:
        from nova.cli import main
        self._create_fits()
        rc = main(["migrate", str(self.src), str(self.dst), "--dry-run"])
        assert rc == 0

    def test_migrate_real(self) -> None:
        from nova.cli import main
        self._create_fits()
        rc = main(["migrate", str(self.src), str(self.dst)])
        assert rc == 0
        nova_stores = list(self.dst.rglob("*.nova.zarr"))
        assert len(nova_stores) == 1


# -----------------------------------------------------------------------
#  Version check
# -----------------------------------------------------------------------

class TestVersion:
    def test_version_is_current(self) -> None:
        import nova
        assert nova.__version__ == "1.0.0"
        assert NOVA_VERSION == "1.0.0"
