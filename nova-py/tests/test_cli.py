"""Tests for the NOVA CLI module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from nova.cli import main
from nova.container import create_dataset
from nova.wcs import NovaWCS, WCSAxis, AffineTransform


class TestCLI:
    """Tests for NOVA CLI commands."""

    def test_no_args(self) -> None:
        """Running without arguments should show help and return 0."""
        result = main([])
        assert result == 0

    def test_info_command(self, tmp_path: Path) -> None:
        """Test the info command."""
        store_path = tmp_path / "test.nova.zarr"
        data = np.ones((64, 64))
        wcs = NovaWCS(
            naxes=2,
            axes=[
                WCSAxis(0, "RA---TAN", 32.0, 150.0, "deg"),
                WCSAxis(1, "DEC--TAN", 32.0, 2.0, "deg"),
            ],
            transform=AffineTransform(
                cd_matrix=[[-1e-4, 0.0], [0.0, 1e-4]]
            ),
        )
        create_dataset(store_path, data, wcs=wcs)

        result = main(["info", str(store_path)])
        assert result == 0

    def test_info_nonexistent(self) -> None:
        """Test info command with nonexistent store."""
        result = main(["info", "/nonexistent/path"])
        assert result == 1

    def test_validate_command(self, tmp_path: Path) -> None:
        """Test the validate command."""
        store_path = tmp_path / "valid.nova.zarr"
        data = np.ones((64, 64))
        create_dataset(store_path, data)

        result = main(["validate", str(store_path)])
        assert result == 0

    def test_validate_missing_store(self) -> None:
        """Test validate command with nonexistent path."""
        result = main(["validate", "/nonexistent/path"])
        assert result == 1

    def test_convert_fits_to_nova(self, tmp_path: Path) -> None:
        """Test FITS -> NOVA conversion via CLI."""
        pytest.importorskip("astropy")
        from astropy.io import fits

        # Create test FITS file
        fits_path = tmp_path / "test.fits"
        data = np.ones((64, 64), dtype=np.float64)
        header = fits.Header()
        header["CTYPE1"] = "RA---TAN"
        header["CTYPE2"] = "DEC--TAN"
        header["CRPIX1"] = 32.0
        header["CRPIX2"] = 32.0
        header["CRVAL1"] = 150.0
        header["CRVAL2"] = 2.0
        header["CD1_1"] = -1e-4
        header["CD1_2"] = 0.0
        header["CD2_1"] = 0.0
        header["CD2_2"] = 1e-4
        hdu = fits.PrimaryHDU(data=data, header=header)
        fits.HDUList([hdu]).writeto(str(fits_path), overwrite=True)

        nova_path = tmp_path / "output.nova.zarr"
        result = main(["convert", str(fits_path), str(nova_path)])
        assert result == 0
        assert nova_path.exists()

    def test_convert_nova_to_fits(self, tmp_path: Path) -> None:
        """Test NOVA -> FITS conversion via CLI."""
        pytest.importorskip("astropy")

        store_path = tmp_path / "input.nova.zarr"
        data = np.ones((64, 64))
        create_dataset(store_path, data)

        fits_path = tmp_path / "output.fits"
        result = main(["convert", str(store_path), str(fits_path)])
        assert result == 0
        assert fits_path.exists()

    def test_convert_nonexistent_input(self) -> None:
        """Test convert with nonexistent input."""
        result = main(["convert", "/nonexistent.fits", "/tmp/output.nova.zarr"])
        assert result == 1

    def test_benchmark_command(self) -> None:
        """Test the benchmark command with small size."""
        result = main(["benchmark", "--size", "64", "--pattern", "gaussian_noise"])
        assert result == 0
