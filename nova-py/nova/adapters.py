"""Pipeline adapter helpers for integrating NOVA with existing tools.

Provides lightweight adapter classes and helpers that allow NOVA datasets
to be used where astropy CCDData, NDData, or similar objects are expected.

These adapters do *not* depend on the external packages at import time;
the dependency is only resolved when the adapter is actually used.

Usage::

    from nova.adapters import to_ccddata, from_ccddata

    # NOVA -> astropy CCDData
    ccd = to_ccddata("observation.nova.zarr")

    # astropy CCDData -> NOVA
    from_ccddata(ccd, "observation.nova.zarr")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def to_ccddata(
    nova_path: str | Path,
    *,
    unit: str = "adu",
) -> Any:
    """Convert a NOVA dataset to an astropy ``CCDData`` object.

    Parameters
    ----------
    nova_path : str or Path
        Path to a ``.nova.zarr`` store.
    unit : str
        Astropy unit string for the data (default ``"adu"``).

    Returns
    -------
    astropy.nddata.CCDData
        CCDData object with data, uncertainty, mask, and WCS from NOVA.

    Raises
    ------
    ImportError
        If ``astropy`` is not installed.
    """
    try:
        from astropy.nddata import CCDData, StdDevUncertainty
        from astropy.wcs import WCS as AstropyWCS
        import astropy.units as u
    except ImportError:
        raise ImportError(
            "astropy is required for CCDData conversion.  "
            "Install it with:  pip install astropy"
        )

    from nova.container import NovaDataset

    ds = NovaDataset(nova_path, mode="r")

    data = np.array(ds.data) if ds.data is not None else np.empty((0, 0))

    uncertainty = None
    if ds.uncertainty is not None:
        uncertainty = StdDevUncertainty(np.array(ds.uncertainty))

    mask = None
    if ds.mask is not None:
        mask = np.array(ds.mask).astype(bool)

    # Build astropy WCS from NOVA WCS
    wcs = None
    if ds.wcs is not None:
        header = ds.wcs.to_fits_header()
        wcs = AstropyWCS(header)

    ccd = CCDData(
        data,
        unit=u.Unit(unit),
        uncertainty=uncertainty,
        mask=mask,
        wcs=wcs,
    )
    ds.close()
    return ccd


def from_ccddata(
    ccd: Any,
    nova_path: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Write an astropy ``CCDData`` object as a NOVA dataset.

    Parameters
    ----------
    ccd : astropy.nddata.CCDData
        Source CCDData object.
    nova_path : str or Path
        Path for the output ``.nova.zarr`` store.
    metadata : dict, optional
        Additional metadata key-value pairs.
    """
    from nova.container import NovaDataset

    ds = NovaDataset(nova_path, mode="w")
    ds.set_science_data(np.asarray(ccd.data, dtype=float))

    if ccd.uncertainty is not None:
        ds.set_uncertainty(np.asarray(ccd.uncertainty.array, dtype=float))

    if ccd.mask is not None:
        ds.set_mask(np.asarray(ccd.mask, dtype=np.uint8))

    # Convert astropy WCS -> NOVA WCS
    if ccd.wcs is not None:
        from nova.wcs import NovaWCS
        try:
            header = dict(ccd.wcs.to_header())
            ds.wcs = NovaWCS.from_fits_header(header)
        except (ValueError, KeyError, TypeError):
            pass  # WCS conversion can fail for exotic projections

    if metadata:
        ds._metadata.update(metadata)

    ds.save()
    ds.close()


def to_nddata(nova_path: str | Path) -> Any:
    """Convert a NOVA dataset to an astropy ``NDData`` object.

    This is a lighter-weight alternative to ``to_ccddata`` that only
    requires ``astropy.nddata.NDData`` (no unit requirement).

    Parameters
    ----------
    nova_path : str or Path
        Path to a ``.nova.zarr`` store.

    Returns
    -------
    astropy.nddata.NDData
    """
    try:
        from astropy.nddata import NDData, StdDevUncertainty
    except ImportError:
        raise ImportError("astropy is required.  pip install astropy")

    from nova.container import NovaDataset

    ds = NovaDataset(nova_path, mode="r")
    data = np.array(ds.data) if ds.data is not None else np.empty((0, 0))
    unc = None
    if ds.uncertainty is not None:
        unc = StdDevUncertainty(np.array(ds.uncertainty))
    mask = None
    if ds.mask is not None:
        mask = np.array(ds.mask).astype(bool)

    nd = NDData(data, uncertainty=unc, mask=mask)
    ds.close()
    return nd


def nova_to_hdulist(nova_path: str | Path) -> Any:
    """Convert a NOVA dataset to an astropy ``HDUList``.

    Equivalent to calling ``nova.to_fits`` but returns the HDUList
    object in memory rather than writing to a file.

    Parameters
    ----------
    nova_path : str or Path
        Path to the ``.nova.zarr`` store.

    Returns
    -------
    astropy.io.fits.HDUList
    """
    try:
        from astropy.io import fits
    except ImportError:
        raise ImportError("astropy is required.  pip install astropy")

    from nova.container import NovaDataset

    ds = NovaDataset(nova_path, mode="r")
    hdus: list[Any] = []

    # Primary HDU
    if ds.data is not None:
        primary = fits.PrimaryHDU(data=np.array(ds.data))
    else:
        primary = fits.PrimaryHDU()

    if ds.wcs is not None:
        header = ds.wcs.to_fits_header()
        for key, val in header.items():
            if key not in ("", "SIMPLE", "BITPIX", "NAXIS", "EXTEND"):
                try:
                    primary.header[key] = val
                except (ValueError, TypeError):
                    pass
    hdus.append(primary)

    # Extensions
    for ext in ds.extensions:
        if ext.data is not None:
            hdu = fits.ImageHDU(data=ext.data, name=ext.name)
            for k, v in ext.header.items():
                try:
                    hdu.header[k] = v
                except (ValueError, TypeError):
                    pass
            hdus.append(hdu)

    ds.close()
    return fits.HDUList(hdus)
