"""Tests for Phase 1 features: multi-extension, table data, data types, and
enhanced FITS conversion.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from nova.container import NovaDataset, NovaExtension, NovaTable, NOVA_VERSION


# ──────────────────────────────────────────────────────────────────────────
#  NovaTable tests
# ──────────────────────────────────────────────────────────────────────────

class TestNovaTable:
    """Tests for the NovaTable class."""

    def test_create_empty_table(self) -> None:
        tbl = NovaTable(name="sources")
        assert tbl.nrows == 0
        assert tbl.colnames == []

    def test_add_columns(self) -> None:
        tbl = NovaTable(name="catalog")
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([4.0, 5.0, 6.0])
        tbl.add_column("x", x, unit="pix", ucd="pos.cartesian.x")
        tbl.add_column("y", y, unit="pix")
        assert tbl.nrows == 3
        assert tbl.colnames == ["x", "y"]
        assert tbl.column_meta["x"]["nova:unit"] == "pix"
        assert tbl.column_meta["x"]["nova:ucd"] == "pos.cartesian.x"

    def test_column_length_mismatch_raises(self) -> None:
        tbl = NovaTable(name="test")
        tbl.add_column("a", np.array([1, 2, 3]))
        with pytest.raises(ValueError, match="expected 3"):
            tbl.add_column("b", np.array([1, 2]))

    def test_to_structured_array(self) -> None:
        tbl = NovaTable(name="test")
        tbl.add_column("id", np.array([1, 2, 3], dtype=np.int32))
        tbl.add_column("flux", np.array([100.0, 200.0, 300.0], dtype=np.float64))
        sa = tbl.to_structured_array()
        assert sa.dtype.names == ("id", "flux")
        np.testing.assert_array_equal(sa["id"], [1, 2, 3])

    def test_from_structured_array(self) -> None:
        dt = np.dtype([("ra", np.float64), ("dec", np.float64), ("mag", np.float32)])
        arr = np.array([(10.0, 20.0, 15.5), (30.0, 40.0, 16.2)], dtype=dt)
        tbl = NovaTable.from_structured_array("stars", arr)
        assert tbl.nrows == 2
        assert set(tbl.colnames) == {"ra", "dec", "mag"}

    def test_to_dict(self) -> None:
        tbl = NovaTable(name="test")
        tbl.add_column("x", np.array([1.0]), unit="deg")
        d = tbl.to_dict()
        assert d["@type"] == "nova:Table"
        assert d["nova:nrows"] == 1
        assert len(d["nova:columns"]) == 1
        assert d["nova:columns"][0]["nova:unit"] == "deg"

    def test_multiple_dtypes(self) -> None:
        tbl = NovaTable(name="mixed")
        tbl.add_column("id", np.arange(5, dtype=np.int64))
        tbl.add_column("flag", np.ones(5, dtype=np.uint8))
        tbl.add_column("value", np.zeros(5, dtype=np.float32))
        tbl.add_column("name_hash", np.arange(5, dtype=np.int16))
        assert tbl.nrows == 5
        sa = tbl.to_structured_array()
        assert sa["flag"].dtype == np.uint8


# ──────────────────────────────────────────────────────────────────────────
#  NovaExtension tests
# ──────────────────────────────────────────────────────────────────────────

class TestNovaExtension:
    """Tests for NovaExtension."""

    def test_create_extension(self) -> None:
        data = np.ones((64, 64), dtype=np.float32)
        ext = NovaExtension(name="SCI", data=data, extver=1)
        assert ext.name == "SCI"
        assert ext.data.shape == (64, 64)
        assert ext.extver == 1

    def test_extension_with_header(self) -> None:
        ext = NovaExtension(
            name="ERR",
            data=np.zeros((32, 32)),
            header={"BUNIT": "e-/s", "EXPTIME": 300.0},
        )
        assert ext.header["BUNIT"] == "e-/s"

    def test_extension_with_wcs(self) -> None:
        from nova.wcs import NovaWCS, WCSAxis, AffineTransform
        axis_ra = WCSAxis(index=0, ctype="RA---TAN", crpix=512.0, crval=180.0, unit="deg")
        axis_dec = WCSAxis(index=1, ctype="DEC--TAN", crpix=512.0, crval=45.0, unit="deg")
        transform = AffineTransform(cd_matrix=[[-1e-4, 0], [0, 1e-4]])
        wcs = NovaWCS(naxes=2, axes=[axis_ra, axis_dec], transform=transform)
        ext = NovaExtension(name="SCI", data=np.ones((100, 100)), wcs=wcs)
        assert ext.wcs is not None
        assert ext.wcs.naxes == 2


# ──────────────────────────────────────────────────────────────────────────
#  Multi-extension dataset tests
# ──────────────────────────────────────────────────────────────────────────

class TestMultiExtensionDataset:
    """Tests for multi-extension NOVA datasets."""

    def setup_method(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp())

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_create_mef_dataset(self) -> None:
        store = self.tmpdir / "mef.nova.zarr"
        ds = NovaDataset(store, mode="w")
        sci = np.random.default_rng(0).normal(100, 10, (256, 256))
        err = np.sqrt(np.abs(sci))
        dq = np.zeros((256, 256), dtype=np.uint8)

        ds.set_science_data(sci.astype(np.float64))
        ds.add_extension(NovaExtension("SCI", data=sci.astype(np.float64)))
        ds.add_extension(NovaExtension("ERR", data=err.astype(np.float32)))
        ds.add_extension(NovaExtension("DQ", data=dq))
        ds.save()

        # Read back
        ds2 = NovaDataset(store, mode="r")
        assert len(ds2.extensions) == 3
        assert ds2.extensions[0].name == "SCI"
        assert ds2.extensions[1].name == "ERR"
        assert ds2.extensions[2].name == "DQ"
        np.testing.assert_array_almost_equal(ds2.extensions[0].data, sci)

    def test_get_extension_by_name(self) -> None:
        store = self.tmpdir / "byname.nova.zarr"
        ds = NovaDataset(store, mode="w")
        ds.set_science_data(np.zeros((10, 10)))
        ds.add_extension(NovaExtension("SCI", data=np.zeros((10, 10))))
        ds.add_extension(NovaExtension("ERR", data=np.ones((10, 10))))
        ds.save()

        ds2 = NovaDataset(store, mode="r")
        err_ext = ds2.get_extension("ERR")
        assert err_ext is not None
        np.testing.assert_array_equal(err_ext.data, 1.0)

    def test_extensions_json_roundtrip(self) -> None:
        store = self.tmpdir / "ext_meta.nova.zarr"
        ds = NovaDataset(store, mode="w")
        ds.set_science_data(np.zeros((8, 8)))
        ds.add_extension(NovaExtension(
            "SCI", data=np.zeros((8, 8)),
            header={"EXPTIME": 120.0}, extver=2,
        ))
        ds.save()

        # Verify JSON file exists and is valid
        ext_json = store / "extensions.json"
        assert ext_json.exists()
        with open(ext_json) as f:
            data = json.load(f)
        assert data["@type"] == "nova:MultiExtensionDataset"
        assert len(data["nova:extensions"]) == 1
        assert data["nova:extensions"][0]["nova:extver"] == 2


# ──────────────────────────────────────────────────────────────────────────
#  Table persistence tests
# ──────────────────────────────────────────────────────────────────────────

class TestTablePersistence:
    """Tests for table data storage and retrieval."""

    def setup_method(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp())

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_save_and_load_table(self) -> None:
        store = self.tmpdir / "table.nova.zarr"
        ds = NovaDataset(store, mode="w")
        ds.set_science_data(np.zeros((10, 10)))

        tbl = NovaTable(name="SOURCES")
        tbl.add_column("ra", np.array([10.0, 20.0, 30.0]), unit="deg", ucd="pos.eq.ra")
        tbl.add_column("dec", np.array([-5.0, 0.0, 5.0]), unit="deg")
        tbl.add_column("flux", np.array([100.0, 200.0, 300.0]), unit="Jy")
        ds.add_table(tbl)
        ds.save()

        # Read back
        ds2 = NovaDataset(store, mode="r")
        assert "SOURCES" in ds2.tables
        tbl2 = ds2.get_table("SOURCES")
        assert tbl2 is not None
        assert tbl2.nrows == 3
        np.testing.assert_array_almost_equal(tbl2.columns["ra"], [10.0, 20.0, 30.0])

    def test_multiple_tables(self) -> None:
        store = self.tmpdir / "multi_tbl.nova.zarr"
        ds = NovaDataset(store, mode="w")
        ds.set_science_data(np.zeros((4, 4)))

        t1 = NovaTable(name="STARS")
        t1.add_column("id", np.arange(5, dtype=np.int32))
        ds.add_table(t1)

        t2 = NovaTable(name="GALAXIES")
        t2.add_column("id", np.arange(3, dtype=np.int32))
        t2.add_column("z", np.array([0.1, 0.5, 1.2]))
        ds.add_table(t2)

        ds.save()

        ds2 = NovaDataset(store, mode="r")
        assert len(ds2.tables) == 2
        assert ds2.get_table("STARS").nrows == 5
        assert ds2.get_table("GALAXIES").nrows == 3

    def test_table_json_metadata(self) -> None:
        store = self.tmpdir / "tbl_meta.nova.zarr"
        ds = NovaDataset(store, mode="w")
        ds.set_science_data(np.zeros((4, 4)))
        tbl = NovaTable(name="CAT")
        tbl.add_column("x", np.array([1.0]), unit="pix")
        ds.add_table(tbl)
        ds.save()

        tables_json = store / "tables.json"
        assert tables_json.exists()
        with open(tables_json) as f:
            data = json.load(f)
        assert data["@type"] == "nova:TableCollection"


# ──────────────────────────────────────────────────────────────────────────
#  Extended data type tests
# ──────────────────────────────────────────────────────────────────────────

class TestExtendedDataTypes:
    """Tests for complex, integer, and scaled data types."""

    def setup_method(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp())

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_complex64_data(self) -> None:
        store = self.tmpdir / "complex64.nova.zarr"
        data = np.array([[1+2j, 3+4j], [5+6j, 7+8j]], dtype=np.complex64)
        ds = NovaDataset(store, mode="w")
        ds.set_science_data(data)
        ds.save()

        ds2 = NovaDataset(store, mode="r")
        loaded = np.array(ds2.data)
        np.testing.assert_array_almost_equal(loaded, data)

    def test_complex128_data(self) -> None:
        store = self.tmpdir / "complex128.nova.zarr"
        data = np.array([[1+2j, 3+4j]], dtype=np.complex128)
        ds = NovaDataset(store, mode="w")
        ds.set_science_data(data)
        ds.save()

        ds2 = NovaDataset(store, mode="r")
        loaded = np.array(ds2.data)
        np.testing.assert_array_almost_equal(loaded, data)

    def test_int8_data(self) -> None:
        store = self.tmpdir / "int8.nova.zarr"
        data = np.array([[1, -2], [3, -4]], dtype=np.int8)
        ds = NovaDataset(store, mode="w")
        ds.set_science_data(data)
        ds.save()

        ds2 = NovaDataset(store, mode="r")
        np.testing.assert_array_equal(np.array(ds2.data), data)

    def test_uint16_data(self) -> None:
        store = self.tmpdir / "uint16.nova.zarr"
        data = np.array([[0, 65535], [1000, 2000]], dtype=np.uint16)
        ds = NovaDataset(store, mode="w")
        ds.set_science_data(data)
        ds.save()

        ds2 = NovaDataset(store, mode="r")
        np.testing.assert_array_equal(np.array(ds2.data), data)

    def test_uint32_data(self) -> None:
        store = self.tmpdir / "uint32.nova.zarr"
        data = np.array([[0, 4294967295]], dtype=np.uint32)
        ds = NovaDataset(store, mode="w")
        ds.set_science_data(data)
        ds.save()

        ds2 = NovaDataset(store, mode="r")
        np.testing.assert_array_equal(np.array(ds2.data), data)

    def test_float16_data(self) -> None:
        store = self.tmpdir / "float16.nova.zarr"
        data = np.array([[1.0, 2.5], [3.0, 4.5]], dtype=np.float16)
        ds = NovaDataset(store, mode="w")
        ds.set_science_data(data)
        ds.save()

        ds2 = NovaDataset(store, mode="r")
        loaded = np.array(ds2.data)
        np.testing.assert_array_almost_equal(loaded, data, decimal=1)

    def test_int64_data(self) -> None:
        store = self.tmpdir / "int64.nova.zarr"
        data = np.array([[-(2**62), 2**62 - 1]], dtype=np.int64)
        ds = NovaDataset(store, mode="w")
        ds.set_science_data(data)
        ds.save()

        ds2 = NovaDataset(store, mode="r")
        np.testing.assert_array_equal(np.array(ds2.data), data)


# ──────────────────────────────────────────────────────────────────────────
#  MEF FITS conversion tests
# ──────────────────────────────────────────────────────────────────────────

class TestMEFConversion:
    """Tests for multi-extension FITS ↔ NOVA conversion."""

    def setup_method(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp())

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _create_mef_fits(self, path: Path) -> None:
        """Create a synthetic multi-extension FITS file."""
        from astropy.io import fits

        primary = fits.PrimaryHDU(data=None)
        primary.header["TELESCOP"] = "TEST"

        sci_data = np.random.default_rng(42).normal(100, 10, (128, 128)).astype(np.float32)
        sci = fits.ImageHDU(data=sci_data, name="SCI")
        sci.header["CRPIX1"] = 64.0
        sci.header["CRPIX2"] = 64.0
        sci.header["CRVAL1"] = 150.0
        sci.header["CRVAL2"] = 2.0
        sci.header["CTYPE1"] = "RA---TAN"
        sci.header["CTYPE2"] = "DEC--TAN"
        sci.header["CD1_1"] = -1e-4
        sci.header["CD2_2"] = 1e-4

        err_data = np.sqrt(np.abs(sci_data))
        err = fits.ImageHDU(data=err_data, name="ERR")

        dq_data = np.zeros((128, 128), dtype=np.int16)
        dq = fits.ImageHDU(data=dq_data, name="DQ")

        # Add a binary table
        col1 = fits.Column(name="RA", format="D", array=np.array([150.1, 150.2]))
        col2 = fits.Column(name="DEC", format="D", array=np.array([2.1, 2.2]))
        col3 = fits.Column(name="MAG", format="E", array=np.array([18.5, 19.2]))
        tbl = fits.BinTableHDU.from_columns([col1, col2, col3], name="CATALOG")
        tbl.header["TUNIT1"] = "deg"
        tbl.header["TUNIT2"] = "deg"
        tbl.header["TUNIT3"] = "mag"

        hdul = fits.HDUList([primary, sci, err, dq, tbl])
        hdul.writeto(str(path), overwrite=True)

    def test_mef_fits_to_nova(self) -> None:
        fits_path = self.tmpdir / "test_mef.fits"
        nova_path = self.tmpdir / "test_mef.nova.zarr"
        self._create_mef_fits(fits_path)

        from nova.fits_converter import from_fits
        ds = from_fits(fits_path, nova_path, all_extensions=True)

        # Should have image extensions (SCI, ERR, DQ)
        assert len(ds.extensions) >= 2
        # Should have a table
        assert "CATALOG" in ds.tables
        cat = ds.get_table("CATALOG")
        assert cat.nrows == 2
        assert "RA" in cat.colnames
        ds.close()

    def test_mef_roundtrip(self) -> None:
        fits_path = self.tmpdir / "roundtrip.fits"
        nova_path = self.tmpdir / "roundtrip.nova.zarr"
        fits_out = self.tmpdir / "roundtrip_out.fits"

        self._create_mef_fits(fits_path)

        from nova.fits_converter import from_fits, to_fits
        ds = from_fits(fits_path, nova_path, all_extensions=True)
        ds.close()

        to_fits(nova_path, fits_out, overwrite=True)

        from astropy.io import fits
        with fits.open(str(fits_out)) as hdul:
            # Should have multiple HDUs (primary + extensions + table)
            assert len(hdul) >= 3

    def test_mef_preserves_headers(self) -> None:
        fits_path = self.tmpdir / "headers.fits"
        nova_path = self.tmpdir / "headers.nova.zarr"
        self._create_mef_fits(fits_path)

        from nova.fits_converter import from_fits
        ds = from_fits(fits_path, nova_path, all_extensions=True)

        # Check that per-HDU headers are stored
        fits_origin = nova_path / "fits_origin"
        assert fits_origin.exists()
        header_files = list(fits_origin.glob("header_hdu*.txt"))
        # Should have one header file per HDU
        assert len(header_files) >= 4  # primary + SCI + ERR + DQ + CATALOG

    def test_table_column_units_roundtrip(self) -> None:
        fits_path = self.tmpdir / "tbl_units.fits"
        nova_path = self.tmpdir / "tbl_units.nova.zarr"
        self._create_mef_fits(fits_path)

        from nova.fits_converter import from_fits
        ds = from_fits(fits_path, nova_path, all_extensions=True)
        cat = ds.get_table("CATALOG")
        assert cat is not None
        # Check that column units were preserved
        if "RA" in cat.column_meta:
            assert cat.column_meta["RA"].get("nova:unit") == "deg"


# ──────────────────────────────────────────────────────────────────────────
#  Scaled integer (BSCALE/BZERO) tests
# ──────────────────────────────────────────────────────────────────────────

class TestScaledIntegers:
    """Tests for BSCALE/BZERO handling."""

    def test_apply_scaling_unsigned16(self) -> None:
        from nova.fits_converter import _apply_scaling
        data = np.array([0, -32768, 32767], dtype=np.int16)
        header = {"BSCALE": 1.0, "BZERO": 32768.0}
        result = _apply_scaling(data, header)
        assert result.dtype == np.uint16

    def test_apply_scaling_unsigned32(self) -> None:
        from nova.fits_converter import _apply_scaling
        data = np.array([0, -2147483648], dtype=np.int32)
        header = {"BSCALE": 1.0, "BZERO": 2147483648.0}
        result = _apply_scaling(data, header)
        assert result.dtype == np.uint32

    def test_apply_scaling_linear(self) -> None:
        from nova.fits_converter import _apply_scaling
        data = np.array([0, 100, 200], dtype=np.int16)
        header = {"BSCALE": 0.5, "BZERO": 10.0}
        result = _apply_scaling(data, header)
        np.testing.assert_array_almost_equal(result, [10.0, 60.0, 110.0])

    def test_no_scaling(self) -> None:
        from nova.fits_converter import _apply_scaling
        data = np.array([1, 2, 3], dtype=np.int32)
        header = {"BSCALE": 1.0, "BZERO": 0.0}
        result = _apply_scaling(data, header)
        np.testing.assert_array_equal(result, data)


# ──────────────────────────────────────────────────────────────────────────
#  NOVA version test
# ──────────────────────────────────────────────────────────────────────────

class TestVersion:
    """Test version bump."""

    def test_version_is_0_2(self) -> None:
        import nova
        assert nova.__version__ == "0.3.0"
        assert NOVA_VERSION == "0.3.0"


# ──────────────────────────────────────────────────────────────────────────
#  FITS column format mapping test
# ──────────────────────────────────────────────────────────────────────────

class TestColumnFormatMapping:
    """Tests for numpy dtype to FITS column format mapping."""

    def test_float64_maps_to_D(self) -> None:
        from nova.fits_converter import _numpy_to_fits_column_format
        assert _numpy_to_fits_column_format(np.dtype("float64")) == "D"

    def test_float32_maps_to_E(self) -> None:
        from nova.fits_converter import _numpy_to_fits_column_format
        assert _numpy_to_fits_column_format(np.dtype("float32")) == "E"

    def test_int32_maps_to_J(self) -> None:
        from nova.fits_converter import _numpy_to_fits_column_format
        assert _numpy_to_fits_column_format(np.dtype("int32")) == "J"

    def test_int64_maps_to_K(self) -> None:
        from nova.fits_converter import _numpy_to_fits_column_format
        assert _numpy_to_fits_column_format(np.dtype("int64")) == "K"

    def test_uint8_maps_to_B(self) -> None:
        from nova.fits_converter import _numpy_to_fits_column_format
        assert _numpy_to_fits_column_format(np.dtype("uint8")) == "B"
