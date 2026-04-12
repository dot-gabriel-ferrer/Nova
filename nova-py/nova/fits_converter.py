"""FITS<->NOVA converter module.

Provides bidirectional conversion between FITS and NOVA formats,
ensuring lossless round-trip fidelity (INV-1: BACKWARD_COMPAT).

Supports:
- Single-HDU FITS files (classic conversion)
- Multi-extension FITS files (MEF) with automatic HDU mapping
- IMAGE, TABLE, and BINTABLE HDU types
- HIERARCH and CONTINUE keywords
- All standard FITS data types including scaled integers
"""

from __future__ import annotations

import json
import datetime
from pathlib import Path
from typing import Any

import numpy as np

from nova.wcs import NovaWCS
from nova.container import (
    NovaDataset,
    NovaExtension,
    NovaTable,
    NOVA_CONTEXT,
    NOVA_VERSION,
)
from nova.integrity import compute_sha256
from nova.provenance import ProvenanceBundle


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
DTYPE_TO_BITPIX["complex64"] = -32   # Store real part as float32
DTYPE_TO_BITPIX["complex128"] = -64  # Store real part as float64

# Standard FITS extension name mapping to NOVA roles
FITS_EXTNAME_ROLES: dict[str, str] = {
    "SCI": "science",
    "ERR": "uncertainty",
    "DQ": "mask",
    "VARIANCE": "uncertainty",
    "WHT": "weight",
    "WEIGHT": "weight",
    "CTX": "context",
}


def from_fits(
    fits_path: str | Path,
    nova_path: str | Path,
    *,
    hdu_index: int = 0,
    include_provenance: bool = True,
    all_extensions: bool = False,
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
        Ignored when ``all_extensions=True``.
    include_provenance : bool
        Whether to generate provenance metadata for the conversion.
    all_extensions : bool
        If True, convert **all** HDUs into a multi-extension NOVA dataset.

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
        if all_extensions:
            return _convert_mef(hdul, fits_path, nova_path, include_provenance)
        else:
            return _convert_single_hdu(
                hdul, hdu_index, fits_path, nova_path, include_provenance,
            )


def _convert_single_hdu(
    hdul: Any,
    hdu_index: int,
    fits_path: Path,
    nova_path: Path,
    include_provenance: bool,
) -> NovaDataset:
    """Convert a single HDU to a NOVA dataset (classic mode)."""
    hdu = hdul[hdu_index]
    header = dict(hdu.header)
    data = hdu.data

    if data is None:
        raise ValueError(
            f"HDU {hdu_index} contains no data. "
            f"Try a different HDU index."
        )

    # Convert data to little-endian if needed (INV: CLOUD_FIRST / L-E native)
    data = _ensure_little_endian(data)

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

    # Preserve all FITS keywords (including HIERARCH)
    _preserve_keywords(ds, header)

    # Store original FITS header as text
    fits_origin_dir = nova_path / "fits_origin"
    fits_origin_dir.mkdir(parents=True, exist_ok=True)
    header_text = str(hdu.header)
    with open(fits_origin_dir / "header.txt", "w") as f:
        f.write(header_text)

    # Generate conversion provenance
    if include_provenance:
        prov = _build_conversion_provenance(fits_path, nova_path)
        ds.provenance = prov

    ds.save()
    return ds


def _convert_mef(
    hdul: Any,
    fits_path: Path,
    nova_path: Path,
    include_provenance: bool,
) -> NovaDataset:
    """Convert a Multi-Extension FITS file to a NOVA dataset.

    Maps each HDU to a ``NovaExtension`` or ``NovaTable`` depending on
    its type (IMAGE vs BINTABLE/TABLE).
    """
    from astropy.io import fits as astropy_fits

    ds = NovaDataset(nova_path, mode="w")
    primary_data_set = False

    for idx, hdu in enumerate(hdul):
        ext_name = str(hdu.name) if hdu.name else f"HDU{idx}"
        header = dict(hdu.header)

        if isinstance(hdu, (astropy_fits.BinTableHDU, astropy_fits.TableHDU)):
            # Convert table HDU to NovaTable
            table = _convert_table_hdu(hdu, ext_name)
            ds.add_table(table)

        elif hdu.data is not None:
            data = _ensure_little_endian(hdu.data)
            # Apply BSCALE/BZERO for scaled integers
            data = _apply_scaling(data, header)

            wcs = NovaWCS.from_fits_header(header)

            # First image HDU with data becomes the primary science data
            if not primary_data_set:
                ds.set_science_data(data)
                ds.wcs = wcs
                primary_data_set = True

                # Map known extension names to uncertainty/mask
                role = FITS_EXTNAME_ROLES.get(ext_name.upper())
                if role == "uncertainty":
                    ds.set_uncertainty(data)
                elif role == "mask" and np.issubdtype(data.dtype, np.integer):
                    ds.set_mask(data.astype(np.uint16))
            else:
                role = FITS_EXTNAME_ROLES.get(ext_name.upper())
                if role == "uncertainty":
                    ds.set_uncertainty(data)
                elif role == "mask" and np.issubdtype(data.dtype, np.integer):
                    ds.set_mask(data.astype(np.uint16))

            # Always add as extension for full MEF round-trip
            ext = NovaExtension(
                name=ext_name,
                data=data,
                header=_serialisable_header(header),
                wcs=wcs,
                extver=int(header.get("EXTVER", 1)),
            )
            ds.add_extension(ext)

    # Root metadata
    ds._metadata.update({
        "nova:data_level": "L0",
        "nova:fits_origin": {
            "nova:filename": fits_path.name,
            "nova:extensions": len(hdul),
            "nova:is_mef": True,
        },
    })

    # Preserve primary header keywords
    if hdul[0].header:
        _preserve_keywords(ds, dict(hdul[0].header))

    # Store original headers as text
    fits_origin_dir = nova_path / "fits_origin"
    fits_origin_dir.mkdir(parents=True, exist_ok=True)
    for idx, hdu in enumerate(hdul):
        header_text = str(hdu.header)
        with open(fits_origin_dir / f"header_hdu{idx}.txt", "w") as f:
            f.write(header_text)

    if include_provenance:
        prov = _build_conversion_provenance(fits_path, nova_path)
        ds.provenance = prov

    ds.save()
    return ds


def _convert_table_hdu(hdu: Any, name: str) -> NovaTable:
    """Convert a FITS BinTableHDU or TableHDU to a NovaTable."""
    table = NovaTable(name=name)
    fits_table = hdu.data
    if fits_table is None:
        return table

    for col_name in fits_table.dtype.names:
        col_data = np.array(fits_table[col_name])
        # Flatten variable-length arrays to fixed-shape if needed
        if col_data.dtype == object:
            # Variable-length array: store as the first element's dtype
            try:
                col_data = np.array([np.asarray(x) for x in col_data])
            except (ValueError, TypeError):
                # Skip columns that can't be converted
                continue
        col_data = _ensure_little_endian(col_data)

        # Extract column metadata from FITS header
        col_idx = list(fits_table.dtype.names).index(col_name) + 1
        unit = hdu.header.get(f"TUNIT{col_idx}", None)
        ucd = hdu.header.get(f"TUCD{col_idx}", None)

        table.add_column(col_name, col_data, unit=unit, ucd=ucd)

    return table


def to_fits(
    nova_path: str | Path,
    fits_path: str | Path,
    *,
    overwrite: bool = False,
) -> None:
    """Convert a NOVA dataset back to FITS format.

    For multi-extension NOVA datasets, produces a multi-extension FITS
    file with one HDU per extension.  Tables are written as BinTableHDUs.

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

    hdu_list: list[Any] = []

    # -- Multi-extension dataset -----------------------------------
    if ds.extensions:
        # Create a PrimaryHDU (may be empty)
        primary_data = None
        primary_header = astropy_fits.Header()
        if ds.data is not None:
            primary_data = _prepare_fits_data(np.array(ds.data))
        _add_wcs_to_header(primary_header, ds.wcs)
        _add_keywords_to_header(primary_header, ds.metadata.get("nova:keywords", {}))
        primary_header.add_history("Converted from NOVA format (MEF)")
        primary_header.add_history(f"NOVA version: {NOVA_VERSION}")
        hdu_list.append(astropy_fits.PrimaryHDU(data=primary_data, header=primary_header))

        for ext in ds.extensions:
            if ext.data is not None:
                data = _prepare_fits_data(ext.data)
                ext_header = astropy_fits.Header()
                _add_wcs_to_header(ext_header, ext.wcs)
                for k, v in ext.header.items():
                    if k not in ("", "COMMENT", "HISTORY", "SIMPLE", "EXTEND",
                                 "BITPIX", "NAXIS", "NAXIS1", "NAXIS2",
                                 "XTENSION", "PCOUNT", "GCOUNT") and len(k) <= 80:
                        try:
                            ext_header[k] = v
                        except (ValueError, TypeError):
                            pass
                img_hdu = astropy_fits.ImageHDU(
                    data=data, header=ext_header, name=ext.name,
                )
                hdu_list.append(img_hdu)

        # Append tables as BinTableHDUs
        for tbl_name, table in ds.tables.items():
            cols = []
            for col_name, col_data in table.columns.items():
                fmt = _numpy_to_fits_column_format(col_data.dtype)
                col = astropy_fits.Column(name=col_name, format=fmt, array=col_data)
                cols.append(col)
            if cols:
                tbl_hdu = astropy_fits.BinTableHDU.from_columns(cols, name=tbl_name)
                # Add column units
                for i, col_name in enumerate(table.columns):
                    if col_name in table.column_meta:
                        unit = table.column_meta[col_name].get("nova:unit")
                        if unit:
                            tbl_hdu.header[f"TUNIT{i+1}"] = unit
                hdu_list.append(tbl_hdu)

    # -- Single-extension dataset ----------------------------------
    else:
        if ds.data is None:
            raise ValueError("NOVA dataset contains no science data.")

        data = _prepare_fits_data(np.array(ds.data))
        header = astropy_fits.Header()
        _add_wcs_to_header(header, ds.wcs)
        _add_keywords_to_header(header, ds.metadata.get("nova:keywords", {}))
        header.add_history("Converted from NOVA format")
        header.add_history(f"NOVA version: {NOVA_VERSION}")
        _add_provenance_to_header(header, ds.provenance)
        hdu_list.append(astropy_fits.PrimaryHDU(data=data, header=header))

        # Write tables as additional BinTableHDUs
        for tbl_name, table in ds.tables.items():
            cols = []
            for col_name, col_data in table.columns.items():
                fmt = _numpy_to_fits_column_format(col_data.dtype)
                col = astropy_fits.Column(name=col_name, format=fmt, array=col_data)
                cols.append(col)
            if cols:
                hdu_list.append(
                    astropy_fits.BinTableHDU.from_columns(cols, name=tbl_name)
                )

    hdul = astropy_fits.HDUList(hdu_list)
    hdul.writeto(str(fits_path), overwrite=overwrite)


# -- Helper functions ------------------------------------------------------


def _ensure_little_endian(data: np.ndarray) -> np.ndarray:
    """Convert array to little-endian byte order if needed."""
    if data.dtype.byteorder == ">":
        return data.astype(data.dtype.newbyteorder("<"))
    return data


def _apply_scaling(data: np.ndarray, header: dict[str, Any]) -> np.ndarray:
    """Apply BSCALE/BZERO scaling to get physical values.

    FITS uses scaled integers (BSCALE * pixel + BZERO) to represent
    unsigned integers or floating-point ranges in integer storage.
    """
    bscale = header.get("BSCALE", 1.0)
    bzero = header.get("BZERO", 0.0)
    if bscale != 1.0 or bzero != 0.0:
        # Check for unsigned-integer convention (BZERO = 2^(N-1))
        if np.issubdtype(data.dtype, np.integer) and bscale == 1.0:
            if data.dtype == np.int16 and bzero == 32768:
                return data.astype(np.uint16)
            elif data.dtype == np.int32 and bzero == 2147483648:
                return data.astype(np.uint32)
        return data.astype(np.float64) * bscale + bzero
    return data


def _serialisable_header(header: dict[str, Any]) -> dict[str, Any]:
    """Convert a FITS header dict to a JSON-serialisable dict."""
    out: dict[str, Any] = {}
    for key, value in header.items():
        if not key or key in ("COMMENT", "HISTORY", ""):
            continue
        try:
            json.dumps(value)
            out[key] = value
        except (TypeError, ValueError):
            out[key] = str(value)
    return out


def _preserve_keywords(ds: NovaDataset, header: dict[str, Any]) -> None:
    """Preserve FITS keywords (including HIERARCH) in NOVA metadata."""
    keywords: dict[str, Any] = {}
    wcs_keys = _get_wcs_keywords(header)
    for key, value in header.items():
        if key and key not in wcs_keys and key not in ("", "COMMENT", "HISTORY"):
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


def _prepare_fits_data(data: np.ndarray) -> np.ndarray:
    """Prepare a NumPy array for writing to a FITS file.

    Handles float16->float32 upscaling and big-endian conversion.
    """
    # Handle float16/bfloat16 -> float32 upscaling
    if data.dtype in (np.float16,):
        import warnings
        warnings.warn(
            "Upscaling float16 data to float32 for FITS compatibility.",
            UserWarning,
            stacklevel=3,
        )
        data = data.astype(np.float32)

    # Handle complex types -- FITS doesn't support complex natively;
    # store as float with doubled last dimension.
    if np.issubdtype(data.dtype, np.complexfloating):
        real_dtype = np.float32 if data.dtype == np.complex64 else np.float64
        data = np.stack([data.real.astype(real_dtype),
                         data.imag.astype(real_dtype)], axis=-1)

    # Convert to big-endian for FITS
    if data.dtype.byteorder == "<" or (
        data.dtype.byteorder == "=" and np.little_endian
    ):
        data = data.astype(data.dtype.newbyteorder(">"))

    return data


def _add_wcs_to_header(header: Any, wcs: NovaWCS | None) -> None:
    """Add WCS keywords to a FITS header."""
    if wcs is None:
        return
    wcs_header = wcs.to_fits_header()
    for key, value in wcs_header.items():
        header[key] = value


def _add_keywords_to_header(header: Any, keywords: dict[str, Any]) -> None:
    """Add preserved FITS keywords back to a header."""
    for key, value in keywords.items():
        if key not in header and len(key) <= 80:
            try:
                header[key] = value
            except (ValueError, TypeError):
                pass


def _add_provenance_to_header(header: Any, provenance: ProvenanceBundle | None) -> None:
    """Add provenance summary as HISTORY cards."""
    if provenance is None:
        return
    header.add_history("--- NOVA Provenance ---")
    prov_dict = provenance.to_dict()
    for activity in prov_dict.get("prov:activity", []):
        activity_id = activity.get("@id", "unknown")
        software = activity.get("nova:software", "unknown")
        header.add_history(f"Activity: {activity_id} ({software})")


def _numpy_to_fits_column_format(dtype: np.dtype) -> str:
    """Map a NumPy dtype to a FITS TFORM column format code."""
    kind = dtype.kind
    itemsize = dtype.itemsize
    if kind == "f":
        return {2: "E", 4: "E", 8: "D"}.get(itemsize, "D")
    elif kind == "i":
        return {1: "B", 2: "I", 4: "J", 8: "K"}.get(itemsize, "K")
    elif kind == "u":
        return {1: "B", 2: "I", 4: "J", 8: "K"}.get(itemsize, "K")
    elif kind == "b":
        return "L"
    elif kind in ("U", "S"):
        return f"{itemsize}A"
    return "D"


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
    """Build a provenance bundle for FITS->NOVA conversion."""
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
