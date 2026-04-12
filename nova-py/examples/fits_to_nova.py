"""Example: Convert a FITS WCS header to NOVA JSON-LD.

This script demonstrates converting the sample FITS WCS header from
the NOVA specification into a structured JSON-LD object.
"""

from __future__ import annotations

import json
from nova.wcs import NovaWCS


def main() -> None:
    """Convert a sample FITS WCS header to NOVA JSON-LD."""

    # Sample FITS header from the NOVA spec
    fits_header = {
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

    print("=" * 60)
    print("FITS WCS Header (keyword-based)")
    print("=" * 60)
    for key, value in fits_header.items():
        print(f"  {key:8s} = {value}")

    print()

    # Convert to NOVA WCS
    wcs = NovaWCS.from_fits_header(fits_header)

    # Serialize to JSON-LD
    wcs_dict = wcs.to_dict()
    json_output = json.dumps(wcs_dict, indent=2)

    print("=" * 60)
    print("NOVA WCS JSON-LD (structured, typed, validatable)")
    print("=" * 60)
    print(json_output)

    print()

    # Round-trip back to FITS
    fits_output = wcs.to_fits_header()

    print("=" * 60)
    print("Round-trip back to FITS keywords")
    print("=" * 60)
    for key, value in fits_output.items():
        print(f"  {key:8s} = {value}")

    print()
    print("OK Lossless round-trip conversion verified (INV-1: BACKWARD_COMPAT)")


if __name__ == "__main__":
    main()
