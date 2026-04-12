"""NOVA World Coordinate System (WCS) module.

Provides a structured, typed WCS representation using JSON-LD,
replacing the keyword-based WCS of FITS.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


# NOVA JSON-LD context
NOVA_CONTEXT = "https://nova-astro.org/v0.1/context.jsonld"

# FITS projection code to human-readable name mapping
PROJECTION_NAMES: dict[str, str] = {
    "TAN": "Gnomonic (Tangent Plane)",
    "SIN": "Slant Orthographic",
    "AIT": "Hammer-Aitoff",
    "MOL": "Mollweide",
    "CAR": "Plate Carrée",
    "MER": "Mercator",
    "STG": "Stereographic",
    "ARC": "Zenithal Equidistant",
    "ZEA": "Zenithal Equal-Area",
    "CEA": "Cylindrical Equal-Area",
    "SFL": "Sanson-Flamsteed",
    "NCP": "North Celestial Pole (deprecated, use SIN)",
    "GLS": "Global Sinusoidal (deprecated, use SFL)",
    "TPN": "Polynomial TAN",
    "ZPN": "Zenithal Polynomial",
    "HPX": "HEALPix",
}

# Axis type classification based on CTYPE
AXIS_TYPE_MAP: dict[str, str] = {
    "RA": "nova:CelestialAxis",
    "DEC": "nova:CelestialAxis",
    "GLON": "nova:CelestialAxis",
    "GLAT": "nova:CelestialAxis",
    "ELON": "nova:CelestialAxis",
    "ELAT": "nova:CelestialAxis",
    "SLON": "nova:CelestialAxis",
    "SLAT": "nova:CelestialAxis",
    "FREQ": "nova:SpectralAxis",
    "WAVE": "nova:SpectralAxis",
    "VELO": "nova:SpectralAxis",
    "VOPT": "nova:SpectralAxis",
    "VRAD": "nova:SpectralAxis",
    "ENER": "nova:SpectralAxis",
    "WAVN": "nova:SpectralAxis",
    "AWAV": "nova:SpectralAxis",
    "TIME": "nova:TemporalAxis",
    "UTC": "nova:TemporalAxis",
    "TAI": "nova:TemporalAxis",
    "STOKES": "nova:StokesAxis",
}

# UCD mapping for common axis types
UCD_MAP: dict[str, str] = {
    "RA": "pos.eq.ra",
    "DEC": "pos.eq.dec",
    "GLON": "pos.galactic.lon",
    "GLAT": "pos.galactic.lat",
    "FREQ": "em.freq",
    "WAVE": "em.wl",
    "ENER": "em.energy",
    "VELO": "spect.dopplerVeloc",
    "VRAD": "spect.dopplerVeloc.radio",
    "VOPT": "spect.dopplerVeloc.opt",
    "TIME": "time.epoch",
    "STOKES": "phys.polarization.stokes",
}


@dataclass
class WCSAxis:
    """A single WCS axis definition.

    Parameters
    ----------
    index : int
        Zero-based axis index.
    ctype : str
        FITS CTYPE string (e.g., 'RA---TAN').
    crpix : float
        Reference pixel (1-based).
    crval : float
        World coordinate at reference pixel.
    unit : str
        Physical unit string.
    axis_type : str
        NOVA axis type (e.g., 'nova:CelestialAxis').
    ucd : str
        IVOA Unified Content Descriptor.
    name : str
        Human-readable axis name.
    """

    index: int
    ctype: str
    crpix: float
    crval: float
    unit: str
    axis_type: str = ""
    ucd: str = ""
    name: str = ""

    def __post_init__(self) -> None:
        if not self.axis_type:
            self.axis_type = _classify_axis(self.ctype)
        if not self.ucd:
            self.ucd = _get_ucd(self.ctype)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-LD dictionary."""
        d: dict[str, Any] = {
            "@type": self.axis_type,
            "nova:index": self.index,
            "nova:ctype": self.ctype,
            "nova:crpix": self.crpix,
            "nova:crval": self.crval,
            "nova:unit": self.unit,
        }
        if self.ucd:
            d["nova:ucd"] = self.ucd
        if self.name:
            d["nova:name"] = self.name
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WCSAxis:
        """Create from JSON-LD dictionary."""
        return cls(
            index=data["nova:index"],
            ctype=data["nova:ctype"],
            crpix=data["nova:crpix"],
            crval=data["nova:crval"],
            unit=data["nova:unit"],
            axis_type=data.get("@type", ""),
            ucd=data.get("nova:ucd", ""),
            name=data.get("nova:name", ""),
        )


@dataclass
class AffineTransform:
    """Affine coordinate transform using a CD matrix.

    Parameters
    ----------
    cd_matrix : list of list of float
        CD matrix (row-major). cd_matrix[i][j] = CDi_j in FITS.
    pixel_scale : float or None
        Derived pixel scale in arcsec/pixel.
    """

    cd_matrix: list[list[float]]
    pixel_scale: float | None = None

    def __post_init__(self) -> None:
        if self.pixel_scale is None and len(self.cd_matrix) >= 2:
            # Compute pixel scale from CD matrix
            # Scale = sqrt(CD1_1^2 + CD2_1^2) * 3600 for arcsec/pixel
            cd11 = self.cd_matrix[0][0]
            cd21 = self.cd_matrix[1][0]
            self.pixel_scale = round(math.sqrt(cd11**2 + cd21**2) * 3600, 6)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-LD dictionary."""
        d: dict[str, Any] = {
            "@type": "nova:AffineTransform",
            "nova:cd_matrix": self.cd_matrix,
        }
        if self.pixel_scale is not None:
            d["nova:pixel_scale"] = {
                "nova:value": self.pixel_scale,
                "nova:unit": "arcsec/pixel",
            }
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AffineTransform:
        """Create from JSON-LD dictionary."""
        pixel_scale = None
        if "nova:pixel_scale" in data:
            pixel_scale = data["nova:pixel_scale"]["nova:value"]
        return cls(
            cd_matrix=data["nova:cd_matrix"],
            pixel_scale=pixel_scale,
        )


@dataclass
class CelestialFrame:
    """Celestial reference frame.

    Parameters
    ----------
    system : str
        Reference system ('ICRS', 'FK5', 'FK4', etc.).
    equinox : float or None
        Equinox (Julian year). Required for FK4/FK5, ignored for ICRS.
    """

    system: str = "ICRS"
    equinox: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-LD dictionary."""
        d: dict[str, Any] = {
            "@type": "nova:CelestialFrame",
            "nova:system": self.system,
        }
        if self.equinox is not None:
            d["nova:equinox"] = self.equinox
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CelestialFrame:
        """Create from JSON-LD dictionary."""
        return cls(
            system=data["nova:system"],
            equinox=data.get("nova:equinox"),
        )


@dataclass
class Projection:
    """Map projection for celestial coordinates.

    Parameters
    ----------
    code : str
        Three-letter FITS projection code (e.g., 'TAN', 'SIN').
    name : str
        Human-readable projection name.
    """

    code: str
    name: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            self.name = PROJECTION_NAMES.get(self.code, self.code)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-LD dictionary."""
        # Determine the @type from the code
        type_name = f"nova:{self.code.title()}Projection"
        return {
            "@type": type_name,
            "nova:code": self.code,
            "nova:name": self.name,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Projection:
        """Create from JSON-LD dictionary."""
        return cls(
            code=data["nova:code"],
            name=data.get("nova:name", ""),
        )


@dataclass
class NovaWCS:
    """NOVA World Coordinate System.

    A structured, typed WCS representation that replaces FITS keyword-based WCS.

    Parameters
    ----------
    naxes : int
        Number of WCS axes.
    axes : list of WCSAxis
        Axis definitions.
    transform : AffineTransform
        Coordinate transform.
    celestial_frame : CelestialFrame or None
        Celestial reference frame.
    projection : Projection or None
        Map projection.
    fits_origin : dict or None
        Original FITS keywords for round-trip fidelity.
    """

    naxes: int
    axes: list[WCSAxis]
    transform: AffineTransform
    celestial_frame: CelestialFrame | None = None
    projection: Projection | None = None
    fits_origin: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-LD dictionary.

        Returns
        -------
        dict
            JSON-LD representation of the WCS.
        """
        d: dict[str, Any] = {
            "@context": NOVA_CONTEXT,
            "@type": "nova:WCS",
            "nova:naxes": self.naxes,
            "nova:axes": [ax.to_dict() for ax in self.axes],
            "nova:transform": self.transform.to_dict(),
        }
        if self.celestial_frame is not None:
            d["nova:celestial_frame"] = self.celestial_frame.to_dict()
        if self.projection is not None:
            d["nova:projection"] = self.projection.to_dict()
        if self.fits_origin is not None:
            d["nova:fits_origin"] = self.fits_origin
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NovaWCS:
        """Create from JSON-LD dictionary.

        Parameters
        ----------
        data : dict
            JSON-LD dictionary.

        Returns
        -------
        NovaWCS
            Parsed WCS object.
        """
        axes = [WCSAxis.from_dict(ax) for ax in data["nova:axes"]]
        transform = AffineTransform.from_dict(data["nova:transform"])

        celestial_frame = None
        if "nova:celestial_frame" in data:
            celestial_frame = CelestialFrame.from_dict(data["nova:celestial_frame"])

        projection = None
        if "nova:projection" in data:
            projection = Projection.from_dict(data["nova:projection"])

        return cls(
            naxes=data["nova:naxes"],
            axes=axes,
            transform=transform,
            celestial_frame=celestial_frame,
            projection=projection,
            fits_origin=data.get("nova:fits_origin"),
        )

    @classmethod
    def from_fits_header(cls, header: dict[str, Any]) -> NovaWCS:
        """Create a NovaWCS from a FITS header dictionary.

        Supports CD matrix, PC matrix + CDELT, and CDELT + CROTA conventions.

        Parameters
        ----------
        header : dict
            FITS header as a dictionary of keyword-value pairs.

        Returns
        -------
        NovaWCS
            Structured WCS object.
        """
        naxes = int(header.get("WCSAXES", header.get("NAXIS", 2)))

        # Parse axes
        axes: list[WCSAxis] = []
        for i in range(naxes):
            n = i + 1  # FITS is 1-based
            ctype = str(header.get(f"CTYPE{n}", "")).strip()
            crpix = float(header.get(f"CRPIX{n}", 0.0))
            crval = float(header.get(f"CRVAL{n}", 0.0))
            cunit = str(header.get(f"CUNIT{n}", _default_unit(ctype))).strip()

            axes.append(WCSAxis(
                index=i,
                ctype=ctype,
                crpix=crpix,
                crval=crval,
                unit=cunit,
            ))

        # Parse transform (CD matrix, PC matrix, or CDELT+CROTA)
        cd_matrix = _parse_cd_matrix(header, naxes)
        transform = AffineTransform(cd_matrix=cd_matrix)

        # Parse celestial frame
        celestial_frame = None
        radesys = header.get("RADESYS", header.get("RADECSYS"))
        if radesys:
            celestial_frame = CelestialFrame(
                system=str(radesys).strip(),
                equinox=header.get("EQUINOX"),
            )

        # Parse projection from CTYPE
        projection = None
        if naxes >= 2 and axes[0].ctype:
            proj_code = _extract_projection_code(axes[0].ctype)
            if proj_code:
                projection = Projection(code=proj_code)

        # Preserve original FITS keywords
        fits_origin: dict[str, Any] = {}
        wcs_keywords = {
            "WCSAXES", "NAXIS", "RADESYS", "RADECSYS", "EQUINOX",
        }
        for n in range(1, naxes + 1):
            wcs_keywords.update({
                f"CRPIX{n}", f"CRVAL{n}", f"CTYPE{n}", f"CUNIT{n}",
                f"CDELT{n}", f"CROTA{n}",
            })
            for m in range(1, naxes + 1):
                wcs_keywords.update({f"CD{n}_{m}", f"PC{n}_{m}"})

        for key in wcs_keywords:
            if key in header:
                fits_origin[key] = header[key]

        return cls(
            naxes=naxes,
            axes=axes,
            transform=transform,
            celestial_frame=celestial_frame,
            projection=projection,
            fits_origin=fits_origin,
        )

    def to_fits_header(self) -> dict[str, Any]:
        """Convert back to FITS header keywords.

        Returns
        -------
        dict
            FITS header keywords as a dictionary.
        """
        header: dict[str, Any] = {}
        header["WCSAXES"] = self.naxes

        for axis in self.axes:
            n = axis.index + 1
            header[f"CTYPE{n}"] = axis.ctype
            header[f"CRPIX{n}"] = axis.crpix
            header[f"CRVAL{n}"] = axis.crval
            if axis.unit:
                header[f"CUNIT{n}"] = axis.unit

        # Write CD matrix
        cd = self.transform.cd_matrix
        for i in range(self.naxes):
            for j in range(self.naxes):
                if i < len(cd) and j < len(cd[i]):
                    header[f"CD{i + 1}_{j + 1}"] = cd[i][j]

        if self.celestial_frame is not None:
            header["RADESYS"] = self.celestial_frame.system
            if self.celestial_frame.equinox is not None:
                header["EQUINOX"] = self.celestial_frame.equinox

        return header


def _classify_axis(ctype: str) -> str:
    """Classify a FITS CTYPE into a NOVA axis type."""
    if not ctype:
        return "nova:LinearAxis"
    # Extract the coordinate name (before the projection separator)
    coord = ctype.split("-")[0].strip()
    return AXIS_TYPE_MAP.get(coord, "nova:LinearAxis")


def _get_ucd(ctype: str) -> str:
    """Get the IVOA UCD for a FITS CTYPE."""
    if not ctype:
        return ""
    coord = ctype.split("-")[0].strip()
    return UCD_MAP.get(coord, "")


def _extract_projection_code(ctype: str) -> str:
    """Extract the 3-letter projection code from a FITS CTYPE.

    For example, 'RA---TAN' returns 'TAN', 'DEC--SIN' returns 'SIN'.
    """
    if not ctype or len(ctype) < 5:
        return ""
    # The projection code is the last 3 characters after removing dashes
    parts = ctype.replace(" ", "").split("-")
    # Filter out empty strings
    parts = [p for p in parts if p]
    if len(parts) >= 2:
        return parts[-1]
    return ""


def _default_unit(ctype: str) -> str:
    """Return a sensible default unit for a FITS CTYPE."""
    if not ctype:
        return ""
    coord = ctype.split("-")[0].strip()
    if coord in ("RA", "DEC", "GLON", "GLAT", "ELON", "ELAT", "SLON", "SLAT"):
        return "deg"
    if coord == "FREQ":
        return "Hz"
    if coord in ("WAVE", "AWAV"):
        return "m"
    if coord in ("VELO", "VOPT", "VRAD"):
        return "m/s"
    if coord == "ENER":
        return "J"
    if coord in ("TIME", "UTC", "TAI"):
        return "s"
    return ""


def _parse_cd_matrix(header: dict[str, Any], naxes: int) -> list[list[float]]:
    """Parse the coordinate transform matrix from FITS header.

    Handles three conventions in order of precedence:
    1. CDi_j matrix
    2. PCi_j matrix + CDELTn
    3. CDELTn + CROTAn (legacy)
    """
    # Try CD matrix first
    has_cd = any(f"CD{i + 1}_{j + 1}" in header
                 for i in range(naxes) for j in range(naxes))
    if has_cd:
        matrix = []
        for i in range(naxes):
            row = []
            for j in range(naxes):
                row.append(float(header.get(f"CD{i + 1}_{j + 1}", 0.0)))
            matrix.append(row)
        return matrix

    # Try PC matrix + CDELT
    has_pc = any(f"PC{i + 1}_{j + 1}" in header
                 for i in range(naxes) for j in range(naxes))
    if has_pc:
        matrix = []
        for i in range(naxes):
            row = []
            cdelt = float(header.get(f"CDELT{i + 1}", 1.0))
            for j in range(naxes):
                default_pc = 1.0 if i == j else 0.0
                pc = float(header.get(f"PC{i + 1}_{j + 1}", default_pc))
                row.append(pc * cdelt)
            matrix.append(row)
        return matrix

    # Legacy CDELT + CROTA
    cdelts = [float(header.get(f"CDELT{i + 1}", 1.0)) for i in range(naxes)]

    if naxes >= 2:
        crota = float(header.get("CROTA2", header.get("CROTA1", 0.0)))
        cos_r = math.cos(math.radians(crota))
        sin_r = math.sin(math.radians(crota))
        matrix = [[0.0] * naxes for _ in range(naxes)]
        matrix[0][0] = cdelts[0] * cos_r
        matrix[0][1] = -cdelts[1] * sin_r
        matrix[1][0] = cdelts[0] * sin_r
        matrix[1][1] = cdelts[1] * cos_r
        # Fill remaining diagonal elements
        for i in range(2, naxes):
            matrix[i][i] = cdelts[i]
        return matrix

    # 1D: just CDELT
    return [[cdelts[0]]]
