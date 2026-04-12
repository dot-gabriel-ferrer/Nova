"""Tests for the NOVA WCS module."""

from __future__ import annotations

import json
import math

import pytest

from nova.wcs import (
    NovaWCS,
    WCSAxis,
    AffineTransform,
    CelestialFrame,
    Projection,
    _classify_axis,
    _extract_projection_code,
    _parse_cd_matrix,
)


class TestWCSAxis:
    """Tests for WCSAxis."""

    def test_create_celestial_axis(self) -> None:
        axis = WCSAxis(
            index=0,
            ctype="RA---TAN",
            crpix=2048.0,
            crval=150.0,
            unit="deg",
        )
        assert axis.axis_type == "nova:CelestialAxis"
        assert axis.ucd == "pos.eq.ra"

    def test_create_spectral_axis(self) -> None:
        axis = WCSAxis(
            index=2,
            ctype="FREQ",
            crpix=1.0,
            crval=1.4e9,
            unit="Hz",
        )
        assert axis.axis_type == "nova:SpectralAxis"
        assert axis.ucd == "em.freq"

    def test_roundtrip_dict(self) -> None:
        axis = WCSAxis(
            index=0,
            ctype="RA---TAN",
            crpix=2048.0,
            crval=150.0,
            unit="deg",
        )
        d = axis.to_dict()
        axis2 = WCSAxis.from_dict(d)
        assert axis2.index == axis.index
        assert axis2.ctype == axis.ctype
        assert axis2.crpix == axis.crpix
        assert axis2.crval == axis.crval
        assert axis2.unit == axis.unit
        assert axis2.axis_type == axis.axis_type


class TestAffineTransform:
    """Tests for AffineTransform."""

    def test_pixel_scale_computation(self) -> None:
        cd = [[-7.305555555556e-05, 0.0], [0.0, 7.305555555556e-05]]
        transform = AffineTransform(cd_matrix=cd)
        # Pixel scale should be ~0.263 arcsec/pixel
        assert transform.pixel_scale is not None
        assert abs(transform.pixel_scale - 0.263) < 0.001

    def test_roundtrip_dict(self) -> None:
        cd = [[-1e-4, 0.0], [0.0, 1e-4]]
        transform = AffineTransform(cd_matrix=cd)
        d = transform.to_dict()
        transform2 = AffineTransform.from_dict(d)
        assert transform2.cd_matrix == transform.cd_matrix


class TestNovaWCS:
    """Tests for the full WCS object."""

    def test_from_fits_header_cd_matrix(self) -> None:
        """Test conversion from a standard FITS header with CD matrix."""
        header = {
            "WCSAXES": 2,
            "CRPIX1": 2048.0,
            "CRPIX2": 2048.0,
            "CRVAL1": 150.1191666667,
            "CRVAL2": 2.2058333333,
            "CTYPE1": "RA---TAN",
            "CTYPE2": "DEC--TAN",
            "CD1_1": -7.305555555556e-05,
            "CD1_2": 0.0,
            "CD2_1": 0.0,
            "CD2_2": 7.305555555556e-05,
            "RADESYS": "ICRS",
            "EQUINOX": 2000.0,
        }
        wcs = NovaWCS.from_fits_header(header)

        assert wcs.naxes == 2
        assert len(wcs.axes) == 2
        assert wcs.axes[0].ctype == "RA---TAN"
        assert wcs.axes[1].ctype == "DEC--TAN"
        assert wcs.axes[0].crpix == 2048.0
        assert wcs.axes[0].crval == 150.1191666667
        assert wcs.celestial_frame is not None
        assert wcs.celestial_frame.system == "ICRS"
        assert wcs.celestial_frame.equinox == 2000.0
        assert wcs.projection is not None
        assert wcs.projection.code == "TAN"

    def test_from_fits_header_pc_matrix(self) -> None:
        """Test conversion from a FITS header with PC matrix + CDELT."""
        header = {
            "WCSAXES": 2,
            "CRPIX1": 512.0,
            "CRPIX2": 512.0,
            "CRVAL1": 180.0,
            "CRVAL2": -30.0,
            "CTYPE1": "RA---SIN",
            "CTYPE2": "DEC--SIN",
            "CDELT1": -1e-4,
            "CDELT2": 1e-4,
            "PC1_1": 1.0,
            "PC1_2": 0.0,
            "PC2_1": 0.0,
            "PC2_2": 1.0,
            "RADESYS": "FK5",
            "EQUINOX": 2000.0,
        }
        wcs = NovaWCS.from_fits_header(header)

        assert wcs.naxes == 2
        assert wcs.transform.cd_matrix[0][0] == pytest.approx(-1e-4)
        assert wcs.transform.cd_matrix[1][1] == pytest.approx(1e-4)
        assert wcs.projection is not None
        assert wcs.projection.code == "SIN"

    def test_from_fits_header_cdelt_crota(self) -> None:
        """Test conversion from legacy CDELT + CROTA header."""
        header = {
            "NAXIS": 2,
            "CRPIX1": 256.0,
            "CRPIX2": 256.0,
            "CRVAL1": 45.0,
            "CRVAL2": 60.0,
            "CTYPE1": "RA---TAN",
            "CTYPE2": "DEC--TAN",
            "CDELT1": -0.001,
            "CDELT2": 0.001,
            "CROTA2": 30.0,
            "RADESYS": "ICRS",
        }
        wcs = NovaWCS.from_fits_header(header)

        # Check that CD matrix was computed from CDELT + CROTA
        cd = wcs.transform.cd_matrix
        cos30 = math.cos(math.radians(30))
        sin30 = math.sin(math.radians(30))
        assert cd[0][0] == pytest.approx(-0.001 * cos30)
        assert cd[0][1] == pytest.approx(-0.001 * sin30)
        assert cd[1][0] == pytest.approx(-0.001 * sin30)
        assert cd[1][1] == pytest.approx(0.001 * cos30)

    def test_roundtrip_json(self) -> None:
        """Test JSON serialization round-trip."""
        header = {
            "WCSAXES": 2,
            "CRPIX1": 2048.0,
            "CRPIX2": 2048.0,
            "CRVAL1": 150.1191666667,
            "CRVAL2": 2.2058333333,
            "CTYPE1": "RA---TAN",
            "CTYPE2": "DEC--TAN",
            "CD1_1": -7.305555555556e-05,
            "CD1_2": 0.0,
            "CD2_1": 0.0,
            "CD2_2": 7.305555555556e-05,
            "RADESYS": "ICRS",
            "EQUINOX": 2000.0,
        }
        wcs = NovaWCS.from_fits_header(header)
        d = wcs.to_dict()

        # Verify JSON serializable
        json_str = json.dumps(d, indent=2)
        parsed = json.loads(json_str)

        wcs2 = NovaWCS.from_dict(parsed)
        assert wcs2.naxes == wcs.naxes
        assert wcs2.axes[0].crval == wcs.axes[0].crval
        assert wcs2.transform.cd_matrix == wcs.transform.cd_matrix
        assert wcs2.celestial_frame is not None
        assert wcs2.celestial_frame.system == "ICRS"

    def test_to_fits_header(self) -> None:
        """Test conversion back to FITS keywords."""
        wcs = NovaWCS(
            naxes=2,
            axes=[
                WCSAxis(0, "RA---TAN", 2048.0, 150.0, "deg"),
                WCSAxis(1, "DEC--TAN", 2048.0, 2.0, "deg"),
            ],
            transform=AffineTransform(
                cd_matrix=[[-7.3e-05, 0.0], [0.0, 7.3e-05]]
            ),
            celestial_frame=CelestialFrame("ICRS", 2000.0),
        )
        header = wcs.to_fits_header()

        assert header["WCSAXES"] == 2
        assert header["CTYPE1"] == "RA---TAN"
        assert header["CRPIX1"] == 2048.0
        assert header["CRVAL1"] == 150.0
        assert header["CD1_1"] == -7.3e-05
        assert header["RADESYS"] == "ICRS"

    def test_json_ld_context(self) -> None:
        """Test that JSON-LD output includes proper context."""
        wcs = NovaWCS(
            naxes=2,
            axes=[
                WCSAxis(0, "RA---TAN", 1024.0, 180.0, "deg"),
                WCSAxis(1, "DEC--TAN", 1024.0, 45.0, "deg"),
            ],
            transform=AffineTransform(
                cd_matrix=[[-1e-4, 0.0], [0.0, 1e-4]]
            ),
        )
        d = wcs.to_dict()
        assert d["@context"] == "https://nova-astro.org/v0.1/context.jsonld"
        assert d["@type"] == "nova:WCS"


class TestHelperFunctions:
    """Tests for WCS helper functions."""

    def test_classify_axis(self) -> None:
        assert _classify_axis("RA---TAN") == "nova:CelestialAxis"
        assert _classify_axis("DEC--TAN") == "nova:CelestialAxis"
        assert _classify_axis("GLON-AIT") == "nova:CelestialAxis"
        assert _classify_axis("FREQ") == "nova:SpectralAxis"
        assert _classify_axis("WAVE-TAB") == "nova:SpectralAxis"
        assert _classify_axis("STOKES") == "nova:StokesAxis"
        assert _classify_axis("UNKNOWN") == "nova:LinearAxis"
        assert _classify_axis("") == "nova:LinearAxis"

    def test_extract_projection_code(self) -> None:
        assert _extract_projection_code("RA---TAN") == "TAN"
        assert _extract_projection_code("DEC--SIN") == "SIN"
        assert _extract_projection_code("GLON-AIT") == "AIT"
        assert _extract_projection_code("FREQ") == ""
        assert _extract_projection_code("") == ""

    def test_parse_cd_matrix_cd(self) -> None:
        header = {
            "CD1_1": -1e-4,
            "CD1_2": 0.0,
            "CD2_1": 0.0,
            "CD2_2": 1e-4,
        }
        cd = _parse_cd_matrix(header, 2)
        assert cd[0][0] == -1e-4
        assert cd[1][1] == 1e-4

    def test_parse_cd_matrix_pc(self) -> None:
        header = {
            "PC1_1": 1.0,
            "PC1_2": 0.0,
            "PC2_1": 0.0,
            "PC2_2": 1.0,
            "CDELT1": -2e-4,
            "CDELT2": 2e-4,
        }
        cd = _parse_cd_matrix(header, 2)
        assert cd[0][0] == pytest.approx(-2e-4)
        assert cd[1][1] == pytest.approx(2e-4)
