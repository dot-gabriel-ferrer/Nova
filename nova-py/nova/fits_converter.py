"""FITS↔NOVA converter module.

Provides bidirectional conversion between FITS and NOVA formats,
ensuring lossless round-trip fidelity (INV-1: BACKWARD_COMPAT).
"""

from __future__ import annotations

import json
import datetime
from pathlib import Path
from typing import Any

import numpy as np

from nova.wcs import NovaWCS
from nova.container import NovaDataset, NOVA_CONTEXT, NOVA_VERSION
from nova.integrity import compute_sha256


# FITS BITPIX to numpy dtype mapping
BITPIX_TO_DTYPE: dict[int, str] = {
    8: "uint8",
    16: "int16",
    32: "int32",
    64: "int64",
    -32: "float32",
    -64: "float64",
}

# Numpy dtype to FITS BITPIX mapping
DTYPE_TO_BITPIX: dict[str, int] = {v: k for k, v in BITPIX_TO_DTYPE.items()}

# Additional dtype mappings for NOVA-specific types
DTYPE_TO_BITPIX["float16"] = -32  # Upscale to float32 in FITS
DTYPE_TO_BITPIX["bfloat16"] = -32


def from_fits(
    fits_path: str | Path,
    nova_path: str | Path,
    *,
    hdu_index: int = 0,
    include_provenance: bool = True,
) -> NovaDataset:
    """Convert a FITS file to NOVA format.

    Parameters
    ----------
    fits_path : str or Path
        Path to the input FITS file.
    nova_path : str or Path
        Path for the output NOVA store (`.nova.zarr`).
    hdu_index : int
        Index of the HDU to convert (default: 0 for primary).
    include_provenance : bool
        Whether to generate provenance metadata for the conversion.

    Returns
    -------
    NovaDataset
        The created NOVA dataset.

    Raises
    ------
    ImportError
        If astropy is not installed.
    FileNotFoundError
        If the FITS file does not exist.
    """
    try:
        from astropy.io import fits as astropy_fits
    except ImportError:
        raise ImportError(
            "astropy is required for FITS conversion. "
            "Install it with: pip install astropy"
        )

    fits_path = Path(fits_path)
    nova_path = Path(nova_path)

    if not fits_path.exists():
        raise FileNotFoundError(f"FITS file not found: {fits_path}")

    with astropy_fits.open(str(fits_path)) as hdul:
        hdu = hdul[hdu_index]
        header = dict(hdu.header)
        data = hdu.data

        if data is None:
            raise ValueError(
                f"HDU {hdu_index} contains no data. "
                f"Try a different HDU index."
            )

        # Convert data to little-endian if needed (INV: CLOUD_FIRST / L-E native)
        if data.dtype.byteorder == ">":
            data = data.astype(data.dtype.newbyteorder("<"))

        # Parse WCS from FITS header
        wcs = NovaWCS.from_fits_header(header)

        # Create NOVA dataset
        ds = NovaDataset(nova_path, mode="w")
        ds.set_science_data(data)
        ds.wcs = wcs

        # Build metadata
        ds._metadata.update({
            "nova:data_level": "L0",
            "nova:fits_origin": {
                "nova:filename": fits_path.name,
                "nova:header_cards": len(hdu.header),
                "nova:extensions": len(hdul),
                "nova:bitpix": int(header.get("BITPIX", 0)),
            },
        })

        # Preserve all FITS keywords
        keywords: dict[str, Any] = {}
        wcs_keys = _get_wcs_keywords(header)
        for key, value in header.items():
            if key and key not in wcs_keys and key not in ("", "COMMENT", "HISTORY"):
                # Skip non-serializable values
                try:
                    json.dumps(value)
                    keywords[key] = value
                except (TypeError, ValueError):
                    keywords[key] = str(value)

        if keywords:
            ds._metadata["nova:keywords"] = keywords

        # Preserve COMMENT and HISTORY
        comments = [str(v) for v in header.get("COMMENT", []) if v]
        if comments and isinstance(comments, list):
            ds._metadata["nova:comments"] = comments

        history = [str(v) for v in header.get("HISTORY", []) if v]
        if history and isinstance(history, list):
            ds._metadata["nova:history"] = history

        # Store original FITS header as text
        fits_origin_dir = nova_path / "fits_origin"
        fits_origin_dir.mkdir(parents=True, exist_ok=True)
        header_text = str(hdu.header)
        with open(fits_origin_dir / "header.txt", "w") as f:
            f.write(header_text)

        # Generate conversion provenance
        if include_provenance:
            from nova.provenance import ProvenanceBundle
            prov = _build_conversion_provenance(fits_path, nova_path)
            ds.provenance = prov

        ds.save()
        return ds


def to_fits(
    nova_path: str | Path,
    fits_path: str | Path,
    *,
    overwrite: bool = False,
) -> None:
    """Convert a NOVA dataset back to FITS format.

    Parameters
    ----------
    nova_path : str or Path
        Path to the NOVA store.
    fits_path : str or Path
        Path for the output FITS file.
    overwrite : bool
        Whether to overwrite an existing FITS file.

    Raises
    ------
    ImportError
        If astropy is not installed.
    FileExistsError
        If the output file exists and overwrite is False.
    """
    try:
        from astropy.io import fits as astropy_fits
    except ImportError:
        raise ImportError(
            "astropy is required for FITS conversion. "
            "Install it with: pip install astropy"
        )

    fits_path = Path(fits_path)
    if fits_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output file already exists: {fits_path}. "
            f"Use overwrite=True to replace."
        )

    ds = NovaDataset(nova_path, mode="r")

    if ds.data is None:
        raise ValueError("NOVA dataset contains no science data.")

    # Read data
    data = np.array(ds.data)

    # Handle float16/bfloat16 → float32 upscaling
    if data.dtype in (np.float16,):
        import warnings
        warnings.warn(
            "Upscaling float16 data to float32 for FITS compatibility.",
            UserWarning,
            stacklevel=2,
        )
        data = data.astype(np.float32)

    # Convert to big-endian for FITS
    if data.dtype.byteorder == "<" or (
        data.dtype.byteorder == "=" and np.little_endian
    ):
        data = data.astype(data.dtype.newbyteorder(">"))

    # Create FITS header
    header = astropy_fits.Header()

    # Add WCS keywords
    if ds.wcs is not None:
        wcs_header = ds.wcs.to_fits_header()
        for key, value in wcs_header.items():
            header[key] = value

    # Add preserved keywords
    keywords = ds.metadata.get("nova:keywords", {})
    for key, value in keywords.items():
        if key not in header and len(key) <= 8:
            try:
                header[key] = value
            except ValueError:
                pass

    # Add HISTORY noting NOVA conversion
    header.add_history("Converted from NOVA format")
    header.add_history(f"NOVA version: {NOVA_VERSION}")

    # Add provenance as HISTORY cards
    if ds.provenance is not None:
        header.add_history("--- NOVA Provenance ---")
        prov_dict = ds.provenance.to_dict()
        # Add concise provenance summary
        for activity in prov_dict.get("prov:activity", []):
            activity_id = activity.get("@id", "unknown")
            software = activity.get("nova:software", "unknown")
            header.add_history(f"Activity: {activity_id} ({software})")

    # Write FITS file
    hdu = astropy_fits.PrimaryHDU(data=data, header=header)
    hdul = astropy_fits.HDUList([hdu])
    hdul.writeto(str(fits_path), overwrite=overwrite)


def _get_wcs_keywords(header: dict[str, Any]) -> set[str]:
    """Get the set of WCS-related keywords from a FITS header."""
    wcs_keys: set[str] = {
        "WCSAXES", "RADESYS", "RADECSYS", "EQUINOX",
        "LONPOLE", "LATPOLE", "RESTFRQ", "RESTWAV",
        "SPECSYS", "SSYSOBS", "SSYSSRC", "VELREF",
        "VELOSYS", "ZSOURCE", "TIMESYS", "MJDREF",
        "DATEREF", "OBSGEO-X", "OBSGEO-Y", "OBSGEO-Z",
    }
    naxis = int(header.get("NAXIS", 0))
    wcsaxes = int(header.get("WCSAXES", naxis))
    for n in range(1, max(naxis, wcsaxes) + 1):
        wcs_keys.update({
            f"CTYPE{n}", f"CRPIX{n}", f"CRVAL{n}", f"CDELT{n}",
            f"CUNIT{n}", f"CROTA{n}", f"NAXIS{n}",
        })
        for m in range(1, max(naxis, wcsaxes) + 1):
            wcs_keys.update({f"CD{n}_{m}", f"PC{n}_{m}"})
    # SIP keywords
    for prefix in ("A", "B", "AP", "BP"):
        wcs_keys.add(f"{prefix}_ORDER")
        for i in range(10):
            for j in range(10):
                wcs_keys.add(f"{prefix}_{i}_{j}")
    return wcs_keys


def _build_conversion_provenance(
    fits_path: Path,
    nova_path: Path,
) -> "ProvenanceBundle":
    """Build a provenance bundle for FITS→NOVA conversion."""
    from nova.provenance import ProvenanceBundle

    now = datetime.datetime.now(datetime.timezone.utc).isoformat()

    prov_data = {
        "@context": [
            NOVA_CONTEXT,
            "https://www.w3.org/ns/prov",
        ],
        "@type": "prov:Bundle",
        "prov:entity": [
            {
                "@id": f"nova:entity/{fits_path.stem}",
                "@type": "prov:Entity",
                "nova:filename": fits_path.name,
                "nova:format": "FITS",
                "nova:data_level": "L0",
            },
            {
                "@id": f"nova:entity/{nova_path.stem}",
                "@type": "prov:Entity",
                "nova:filename": nova_path.name,
                "nova:format": "NOVA",
                "prov:wasDerivedFrom": {
                    "@id": f"nova:entity/{fits_path.stem}",
                },
                "prov:wasGeneratedBy": {
                    "@id": "nova:activity/fits-to-nova-conversion",
                },
            },
        ],
        "prov:activity": [
            {
                "@id": "nova:activity/fits-to-nova-conversion",
                "@type": "prov:Activity",
                "prov:startedAtTime": now,
                "prov:endedAtTime": now,
                "prov:used": {
                    "@id": f"nova:entity/{fits_path.stem}",
                },
                "prov:generated": {
                    "@id": f"nova:entity/{nova_path.stem}",
                },
                "nova:software": f"nova-py v{NOVA_VERSION}",
                "nova:algorithm": "fits_to_nova_conversion",
            },
        ],
        "prov:agent": [
            {
                "@id": "nova:agent/nova-py",
                "@type": "prov:SoftwareAgent",
                "nova:name": "nova-py",
                "nova:version": NOVA_VERSION,
            },
        ],
    }

    return ProvenanceBundle.from_dict(prov_data)
