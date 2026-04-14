"""Photometry pipeline tools for NOVA.

Provides a complete photometric calibration and measurement pipeline
including aperture photometry with error propagation, zero-point
determination, extinction correction, limiting magnitude estimation,
and differential photometry.  Designed to replace external tools like
SExtractor + SCAMP + DAOPHOT for common photometric workflows.

All functions operate on plain NumPy arrays and optionally log their
actions to an OperationHistory.

Key capabilities
----------------
- Multi-aperture photometry with proper Poisson + read-noise errors.
- Automatic zero-point calibration against a reference catalog.
- Atmospheric extinction correction using Bouguer's law.
- Color-term correction from multi-band observations.
- Growth-curve analysis and aperture corrections.
- Limiting magnitude estimation.
- Differential photometry for time-series / transit work.
- Magnitude system conversions (AB, Vega, instrumental).

Example
-------
>>> from nova.photometry_pipeline import multi_aperture_photometry
>>> from nova.photometry_pipeline import calibrate_zeropoint
>>> phot = multi_aperture_photometry(image, sources, radii=[3,5,7],
...                                   gain=1.5, readnoise=10.0)
>>> zp = calibrate_zeropoint(phot['mag_5'], catalog_mag)
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from nova.constants import MAD_TO_STD


# -------------------------------------------------------------------
#  Multi-aperture photometry
# -------------------------------------------------------------------

def multi_aperture_photometry(
    data: np.ndarray,
    sources: np.ndarray,
    radii: list[float],
    *,
    sky_inner: float | None = None,
    sky_outer: float | None = None,
    gain: float = 1.0,
    readnoise: float = 0.0,
) -> dict[str, Any]:
    """Perform aperture photometry at multiple radii with error estimates.

    Parameters
    ----------
    data : np.ndarray
        2-D image.
    sources : np.ndarray
        Source positions, shape ``(N, 2)`` with ``[x, y]``.
    radii : list[float]
        List of aperture radii in pixels.
    sky_inner : float or None
        Inner radius for sky annulus (default: ``max(radii) + 5``).
    sky_outer : float or None
        Outer radius for sky annulus (default: ``sky_inner + 10``).
    gain : float
        Detector gain (e-/ADU).
    readnoise : float
        Read noise in electrons.

    Returns
    -------
    dict
        Keys: ``'x'``, ``'y'`` (source positions),
        ``'flux_<r>'``, ``'fluxerr_<r>'``, ``'mag_<r>'``, ``'magerr_<r>'``
        for each radius *r*, ``'sky'``, ``'sky_rms'``.
    """
    if not isinstance(data, np.ndarray) or data.ndim != 2:
        raise TypeError("data must be a 2-D numpy array.")
    if not isinstance(sources, np.ndarray) or sources.ndim != 2:
        raise TypeError("sources must be a 2-D array with [x, y] columns.")
    if not radii:
        raise ValueError("At least one aperture radius is required.")

    h, w = data.shape
    max_r = max(radii)
    if sky_inner is None:
        sky_inner = max_r + 5.0
    if sky_outer is None:
        sky_outer = sky_inner + 10.0

    nsrc = sources.shape[0]
    result: dict[str, Any] = {
        "x": sources[:, 0].copy(),
        "y": sources[:, 1].copy(),
        "sky": np.zeros(nsrc, dtype=np.float64),
        "sky_rms": np.zeros(nsrc, dtype=np.float64),
    }
    for r in radii:
        rk = "{:.0f}".format(r)
        result["flux_" + rk] = np.zeros(nsrc, dtype=np.float64)
        result["fluxerr_" + rk] = np.zeros(nsrc, dtype=np.float64)
        result["mag_" + rk] = np.full(nsrc, np.nan, dtype=np.float64)
        result["magerr_" + rk] = np.full(nsrc, np.nan, dtype=np.float64)

    box = int(math.ceil(sky_outer)) + 1
    yy_template, xx_template = np.mgrid[-box:box + 1, -box:box + 1]
    rr_template = np.sqrt(xx_template ** 2.0 + yy_template ** 2.0)

    for i in range(nsrc):
        cx = float(sources[i, 0])
        cy = float(sources[i, 1])
        ix, iy = int(round(cx)), int(round(cy))

        # Bounds check
        if ix - box < 0 or ix + box >= w or iy - box < 0 or iy + box >= h:
            continue

        stamp = data[iy - box:iy + box + 1, ix - box:ix + box + 1].astype(np.float64)

        # Sky estimation from annulus
        sky_mask = (rr_template >= sky_inner) & (rr_template <= sky_outer)
        sky_pixels = stamp[sky_mask]
        if len(sky_pixels) > 0:
            sky_med = float(np.median(sky_pixels))
            sky_mad = float(np.median(np.abs(sky_pixels - sky_med)))
            sky_rms = sky_mad * MAD_TO_STD
        else:
            sky_med = 0.0
            sky_rms = 0.0
        result["sky"][i] = sky_med
        result["sky_rms"][i] = sky_rms

        for r in radii:
            rk = "{:.0f}".format(r)
            ap_mask = rr_template <= r
            n_pix = int(np.sum(ap_mask))
            if n_pix == 0:
                continue
            ap_sum = float(np.sum(stamp[ap_mask])) - sky_med * n_pix

            # Poisson + readnoise + sky noise error
            if gain > 0:
                var_src = max(ap_sum * gain, 0.0) / (gain ** 2)
            else:
                var_src = 0.0
            var_sky = (sky_rms ** 2) * n_pix
            var_read = (readnoise ** 2) * n_pix / (gain ** 2 if gain > 0 else 1.0)
            flux_err = math.sqrt(var_src + var_sky + var_read)

            result["flux_" + rk][i] = ap_sum
            result["fluxerr_" + rk][i] = flux_err

            if ap_sum > 0:
                result["mag_" + rk][i] = -2.5 * math.log10(ap_sum)
                result["magerr_" + rk][i] = 2.5 / math.log(10) * flux_err / ap_sum

    return result


# -------------------------------------------------------------------
#  Zero-point calibration
# -------------------------------------------------------------------

def calibrate_zeropoint(
    inst_mag: np.ndarray,
    catalog_mag: np.ndarray,
    *,
    sigma_clip: float = 3.0,
    max_iter: int = 5,
) -> dict[str, float]:
    """Determine the photometric zero-point.

    ``catalog_mag = inst_mag + zeropoint``

    Parameters
    ----------
    inst_mag : np.ndarray
        Instrumental magnitudes.
    catalog_mag : np.ndarray
        Reference catalog magnitudes.
    sigma_clip : float
        Clipping threshold in sigma.
    max_iter : int
        Maximum sigma-clipping iterations.

    Returns
    -------
    dict
        ``{'zeropoint': float, 'zp_err': float, 'n_used': int,
           'n_rejected': int}``
    """
    inst_mag = np.asarray(inst_mag, dtype=np.float64)
    catalog_mag = np.asarray(catalog_mag, dtype=np.float64)
    valid = np.isfinite(inst_mag) & np.isfinite(catalog_mag)
    diff = catalog_mag - inst_mag

    mask = valid.copy()
    for _ in range(max_iter):
        d = diff[mask]
        if len(d) < 3:
            break
        med = float(np.median(d))
        mad = float(np.median(np.abs(d - med)))
        sig = mad * MAD_TO_STD
        if sig == 0:
            break
        mask = valid & (np.abs(diff - med) < sigma_clip * sig)

    d = diff[mask]
    zp = float(np.median(d))
    zp_err = float(np.std(d) / max(1, math.sqrt(len(d))))

    return {
        "zeropoint": zp,
        "zp_err": zp_err,
        "n_used": int(np.sum(mask)),
        "n_rejected": int(np.sum(valid) - np.sum(mask)),
    }


# -------------------------------------------------------------------
#  Extinction correction
# -------------------------------------------------------------------

def extinction_correct(
    mag: np.ndarray,
    airmass: float,
    k_lambda: float,
) -> np.ndarray:
    """Apply atmospheric extinction correction.

    ``mag_corrected = mag - k_lambda * airmass``

    Parameters
    ----------
    mag : np.ndarray
        Observed magnitudes.
    airmass : float
        Airmass of the observation.
    k_lambda : float
        Extinction coefficient (mag / airmass) for the filter band.

    Returns
    -------
    np.ndarray
        Extinction-corrected magnitudes.
    """
    mag = np.asarray(mag, dtype=np.float64)
    if airmass < 1.0:
        raise ValueError("Airmass must be >= 1.0, got {}.".format(airmass))
    return mag - k_lambda * airmass


def color_term_correct(
    mag: np.ndarray,
    color: np.ndarray,
    color_term: float,
) -> np.ndarray:
    """Apply color-term correction.

    ``mag_corrected = mag + color_term * color``

    Parameters
    ----------
    mag : np.ndarray
        Magnitudes in the target band.
    color : np.ndarray
        Color index (e.g. B-V) for each source.
    color_term : float
        Color coefficient.

    Returns
    -------
    np.ndarray
        Color-corrected magnitudes.
    """
    mag = np.asarray(mag, dtype=np.float64)
    color = np.asarray(color, dtype=np.float64)
    return mag + color_term * color


# -------------------------------------------------------------------
#  Limiting magnitude
# -------------------------------------------------------------------

def limiting_magnitude(
    sky_rms: float,
    zeropoint: float,
    aperture_npix: int,
    *,
    nsigma: float = 5.0,
    gain: float = 1.0,
    readnoise: float = 0.0,
) -> float:
    """Estimate the limiting magnitude for a given detection threshold.

    Parameters
    ----------
    sky_rms : float
        RMS of the sky background (ADU).
    zeropoint : float
        Photometric zero-point (mag).
    aperture_npix : int
        Number of pixels in the aperture.
    nsigma : float
        Detection significance threshold.
    gain : float
        Detector gain (e-/ADU).
    readnoise : float
        Read noise (electrons).

    Returns
    -------
    float
        Limiting magnitude (AB or Vega, depending on zeropoint system).
    """
    sky_noise = sky_rms * math.sqrt(aperture_npix)
    read_noise = readnoise * math.sqrt(aperture_npix) / gain if gain > 0 else 0.0
    total_noise = math.sqrt(sky_noise ** 2 + read_noise ** 2)
    flux_limit = nsigma * total_noise
    if flux_limit <= 0:
        return float("nan")
    return -2.5 * math.log10(flux_limit) + zeropoint


# -------------------------------------------------------------------
#  Growth curve / aperture correction
# -------------------------------------------------------------------

def growth_curve(
    data: np.ndarray,
    x: float,
    y: float,
    radii: list[float],
) -> dict[str, np.ndarray]:
    """Compute the curve of growth for a single source.

    Parameters
    ----------
    data : np.ndarray
        2-D image.
    x, y : float
        Source centroid (pixel coords).
    radii : list[float]
        Radii at which to measure enclosed flux.

    Returns
    -------
    dict
        ``{'radii': np.ndarray, 'flux': np.ndarray,
           'normalized_flux': np.ndarray}``
    """
    if not isinstance(data, np.ndarray) or data.ndim != 2:
        raise TypeError("data must be a 2-D numpy array.")

    h, w = data.shape
    radii = sorted(radii)
    fluxes = []
    max_r = max(radii)
    box = int(math.ceil(max_r)) + 1
    ix, iy = int(round(x)), int(round(y))

    if ix - box < 0 or ix + box >= w or iy - box < 0 or iy + box >= h:
        return {
            "radii": np.array(radii),
            "flux": np.full(len(radii), np.nan),
            "normalized_flux": np.full(len(radii), np.nan),
        }

    yy, xx = np.mgrid[iy - box:iy + box + 1, ix - box:ix + box + 1]
    rr = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
    stamp = data[iy - box:iy + box + 1, ix - box:ix + box + 1].astype(np.float64)

    for r in radii:
        mask = rr <= r
        fluxes.append(float(np.sum(stamp[mask])))

    flux_arr = np.array(fluxes)
    total = flux_arr[-1] if flux_arr[-1] != 0 else 1.0

    return {
        "radii": np.array(radii),
        "flux": flux_arr,
        "normalized_flux": flux_arr / total,
    }


def aperture_correction(
    phot_small: np.ndarray,
    phot_large: np.ndarray,
    *,
    sigma_clip: float = 3.0,
) -> dict[str, float]:
    """Compute aperture correction from bright, isolated stars.

    ``mag_corrected = mag_small + apcor``

    Parameters
    ----------
    phot_small : np.ndarray
        Instrumental magnitudes through the small aperture.
    phot_large : np.ndarray
        Instrumental magnitudes through the large aperture.
    sigma_clip : float
        Sigma-clipping threshold.

    Returns
    -------
    dict
        ``{'apcor': float, 'apcor_err': float, 'n_used': int}``
    """
    diff = np.asarray(phot_large, dtype=np.float64) - np.asarray(phot_small, dtype=np.float64)
    valid = np.isfinite(diff)
    d = diff[valid]
    if len(d) < 3:
        return {"apcor": 0.0, "apcor_err": float("nan"), "n_used": len(d)}

    med = float(np.median(d))
    mad = float(np.median(np.abs(d - med)))
    sig = mad * MAD_TO_STD
    if sig > 0:
        mask = np.abs(d - med) < sigma_clip * sig
        d = d[mask]

    apcor = float(np.median(d))
    apcor_err = float(np.std(d) / max(1, math.sqrt(len(d))))
    return {"apcor": apcor, "apcor_err": apcor_err, "n_used": len(d)}


# -------------------------------------------------------------------
#  Differential photometry
# -------------------------------------------------------------------

def differential_photometry(
    target_flux: np.ndarray,
    comparison_flux: np.ndarray,
    *,
    comparison_mag: float | None = None,
) -> dict[str, np.ndarray]:
    """Compute differential photometry relative to comparison star(s).

    Parameters
    ----------
    target_flux : np.ndarray
        1-D array of target flux measurements over time.
    comparison_flux : np.ndarray
        1-D or 2-D array of comparison star flux (if 2-D, rows = stars,
        cols = time epochs; ensemble average is used).
    comparison_mag : float or None
        If given, calibrate to absolute magnitudes using the known
        magnitude of the comparison star / ensemble.

    Returns
    -------
    dict
        ``{'delta_mag': np.ndarray, 'delta_mag_err': np.ndarray,
           'rel_flux': np.ndarray}``
    """
    target_flux = np.asarray(target_flux, dtype=np.float64)
    comparison_flux = np.asarray(comparison_flux, dtype=np.float64)

    if comparison_flux.ndim == 2:
        comp = np.median(comparison_flux, axis=0)
    else:
        comp = comparison_flux

    if comp.shape != target_flux.shape:
        raise ValueError("target and comparison flux must have the same length.")

    rel_flux = np.where(comp > 0, target_flux / comp, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        delta_mag = -2.5 * np.log10(rel_flux)

    # Propagate Poisson errors
    with np.errstate(divide="ignore", invalid="ignore"):
        t_err = np.where(target_flux > 0, 1.0 / np.sqrt(target_flux), 0.0)
        c_err = np.where(comp > 0, 1.0 / np.sqrt(comp), 0.0)
        frac_err = np.sqrt(t_err ** 2 + c_err ** 2)
        delta_mag_err = 2.5 / math.log(10) * frac_err

    if comparison_mag is not None:
        delta_mag = delta_mag + comparison_mag

    return {
        "delta_mag": delta_mag,
        "delta_mag_err": delta_mag_err,
        "rel_flux": rel_flux,
    }


# -------------------------------------------------------------------
#  Magnitude system conversions
# -------------------------------------------------------------------

def ab_to_vega(
    mag_ab: np.ndarray,
    ab_vega_offset: float,
) -> np.ndarray:
    """Convert AB magnitudes to Vega magnitudes.

    ``mag_vega = mag_ab - ab_vega_offset``

    Parameters
    ----------
    mag_ab : np.ndarray
        AB magnitudes.
    ab_vega_offset : float
        AB - Vega offset for the band (e.g. 0.02 for V band).

    Returns
    -------
    np.ndarray
        Vega magnitudes.
    """
    return np.asarray(mag_ab, dtype=np.float64) - ab_vega_offset


def vega_to_ab(
    mag_vega: np.ndarray,
    ab_vega_offset: float,
) -> np.ndarray:
    """Convert Vega magnitudes to AB magnitudes.

    Parameters
    ----------
    mag_vega : np.ndarray
        Vega magnitudes.
    ab_vega_offset : float
        AB - Vega offset for the band.

    Returns
    -------
    np.ndarray
        AB magnitudes.
    """
    return np.asarray(mag_vega, dtype=np.float64) + ab_vega_offset


def flux_to_mag(
    flux: np.ndarray,
    zeropoint: float = 0.0,
) -> np.ndarray:
    """Convert flux to magnitude.

    ``mag = -2.5 * log10(flux) + zeropoint``

    Parameters
    ----------
    flux : np.ndarray
        Flux values (must be positive).
    zeropoint : float
        Zero-point offset.

    Returns
    -------
    np.ndarray
        Magnitudes (NaN where flux <= 0).
    """
    flux = np.asarray(flux, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        mag = np.where(flux > 0, -2.5 * np.log10(flux) + zeropoint, np.nan)
    return mag


def mag_to_flux(
    mag: np.ndarray,
    zeropoint: float = 0.0,
) -> np.ndarray:
    """Convert magnitude to flux.

    ``flux = 10^(-0.4 * (mag - zeropoint))``

    Parameters
    ----------
    mag : np.ndarray
        Magnitudes.
    zeropoint : float
        Zero-point offset.

    Returns
    -------
    np.ndarray
        Flux values.
    """
    mag = np.asarray(mag, dtype=np.float64)
    return np.power(10.0, -0.4 * (mag - zeropoint))
