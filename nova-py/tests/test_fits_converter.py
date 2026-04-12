"""Tests for the NOVA FITS converter module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from nova.wcs import NovaWCS


class TestFITSConverter:
    """Tests for FITS<->NOVA conversion.

    These tests only run if astropy is available.
    """

    @pytest.fixture(autouse=True)
    def check_astropy(self) -> None:
        """Skip tests if astropy is not installed."""
        pytest.importorskip("astropy")

    def _create_test_fits(self, path: Path) -> None:
        """Create a minimal test FITS file."""
        from astropy.io import fits

        data = np.random.default_rng(42).standard_normal((256, 256)).astype(
            np.float64
        )
        header = fits.Header()
        header["WCSAXES"] = 2
        header["CRPIX1"] = 128.0
        header["CRPIX2"] = 128.0
        header["CRVAL1"] = 150.0
        header["CRVAL2"] = 2.0
        header["CTYPE1"] = "RA---TAN"
        header["CTYPE2"] = "DEC--TAN"
        header["CD1_1"] = -7.3e-05
        header["CD1_2"] = 0.0
        header["CD2_1"] = 0.0
        header["CD2_2"] = 7.3e-05
        header["RADESYS"] = "ICRS"
        header["EQUINOX"] = 2000.0
        header["OBJECT"] = "Test Field"
        header["EXPTIME"] = 300.0

        hdu = fits.PrimaryHDU(data=data, header=header)
        hdul = fits.HDUList([hdu])
        hdul.writeto(str(path), overwrite=True)

    def test_fits_to_nova(self, tmp_path: Path) -> None:
        """Test FITS -> NOVA conversion."""
        from nova.fits_converter import from_fits

        fits_path = tmp_path / "test.fits"
        nova_path = tmp_path / "test.nova.zarr"
        self._create_test_fits(fits_path)

        ds = from_fits(fits_path, nova_path)

        assert ds.data is not None
        assert ds.wcs is not None
        assert ds.wcs.naxes == 2
        assert ds.wcs.axes[0].ctype == "RA---TAN"
        assert ds.provenance is not None
        assert (nova_path / "nova_metadata.json").exists()
        assert (nova_path / "wcs.json").exists()
        assert (nova_path / "provenance.json").exists()
        assert (nova_path / "fits_origin" / "header.txt").exists()
        ds.close()

    def test_nova_to_fits(self, tmp_path: Path) -> None:
        """Test NOVA -> FITS round-trip conversion."""
        from nova.fits_converter import from_fits, to_fits

        fits_path = tmp_path / "original.fits"
        nova_path = tmp_path / "converted.nova.zarr"
        fits_output = tmp_path / "roundtrip.fits"

        self._create_test_fits(fits_path)
        ds = from_fits(fits_path, nova_path)
        ds.close()

        to_fits(nova_path, fits_output)
        assert fits_output.exists()

        # Verify the round-trip data
        from astropy.io import fits

        with fits.open(str(fits_output)) as hdul:
            header = hdul[0].header
            assert header["CTYPE1"] == "RA---TAN"
            assert header["RADESYS"] == "ICRS"
            assert "HISTORY" in header

    def test_fits_not_found(self, tmp_path: Path) -> None:
        """Test that FileNotFoundError is raised for missing FITS files."""
        from nova.fits_converter import from_fits

        with pytest.raises(FileNotFoundError):
            from_fits(tmp_path / "nonexistent.fits", tmp_path / "out.nova.zarr")

    def test_fits_keywords_preserved(self, tmp_path: Path) -> None:
        """Test that non-WCS FITS keywords are preserved."""
        from nova.fits_converter import from_fits

        fits_path = tmp_path / "keywords.fits"
        nova_path = tmp_path / "keywords.nova.zarr"
        self._create_test_fits(fits_path)

        ds = from_fits(fits_path, nova_path)
        keywords = ds.metadata.get("nova:keywords", {})
        assert keywords.get("OBJECT") == "Test Field"
        assert keywords.get("EXPTIME") == 300.0
        ds.close()
