"""Advanced photometry tools for NOVA.

Production-quality routines for point-source and extended-source photometry,
photometric calibration, aperture corrections, and crowded-field analysis.
All functions operate on plain ``numpy.ndarray`` objects and require only
NumPy at import time; ``scipy.optimize`` and ``scipy.ndimage`` are imported
lazily where needed.

Categories
----------
- **PSF photometry**: simultaneous PSF fitting and iterative subtraction.
- **Extended-source photometry**: Petrosian radius, Kron radius, isophotal
  photometry, and radial surface-brightness profiles.
- **Photometric calibration**: zero-point, extinction, and colour-term
  corrections.
- **Aperture corrections**: small-to-large aperture offsets and curves of
  growth.
- **Crowded-field photometry**: neighbour finding, source deblending, and
  completeness testing via artificial-star injection.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from nova.constants import MAD_TO_STD


# ---------------------------------------------------------------------------
# Lazy optional imports
# ---------------------------------------------------------------------------

def _import_scipy_optimize():  # noqa: D401
    """Lazily import *scipy.optimize* with a clear error on failure."""
    try:
        import scipy.optimize as opt
        return opt
    except ImportError:
        raise ImportError(
            "scipy is required for this function.  "
            "Install it with:  pip install scipy"
        ) from None


def _import_scipy_ndimage():  # noqa: D401
    """Lazily import *scipy.ndimage* with a clear error on failure."""
    try:
        import scipy.ndimage as ndi
        return ndi
    except ImportError:
        raise ImportError(
            "scipy is required for this function.  "
            "Install it with:  pip install scipy"
        ) from None


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _cutout(
    image: np.ndarray,
    x0: float,
    y0: float,
    box_size: int,
) -> tuple[np.ndarray, int, int, int, int]:
    """Extract a square cutout centred near *(x0, y0)*.

    Parameters
    ----------
    image : ndarray
        2-D image.
    x0, y0 : float
        Centre position (x = column, y = row).
    box_size : int
        Side length of the square cutout in pixels.

    Returns
    -------
    cutout : ndarray
        2-D sub-image (may be smaller than *box_size* at edges).
    y_lo, y_hi, x_lo, x_hi : int
        Slice boundaries in the parent image.
    """
    half = box_size // 2
    ny, nx = image.shape[:2]

    y_lo = max(0, int(round(y0)) - half)
    y_hi = min(ny, int(round(y0)) + half + 1)
    x_lo = max(0, int(round(x0)) - half)
    x_hi = min(nx, int(round(x0)) + half + 1)

    return image[y_lo:y_hi, x_lo:x_hi].copy(), y_lo, y_hi, x_lo, x_hi


def _stamp_at(
    psf_model: np.ndarray,
    x0: float,
    y0: float,
    box_size: int,
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, int, int, int, int]:
    """Return a PSF stamp aligned with the image grid at *(x0, y0)*.

    The PSF model is resampled to match the cutout footprint using
    simple nearest-neighbour indexing offset by the sub-pixel shift.

    Parameters
    ----------
    psf_model : ndarray
        Normalised 2-D PSF stamp (odd side lengths recommended).
    x0, y0 : float
        Source centre in image coordinates.
    box_size : int
        Side length of the output stamp.
    image_shape : tuple of int
        (ny, nx) of the parent image.

    Returns
    -------
    stamp : ndarray
        PSF stamp matching the cutout footprint.
    y_lo, y_hi, x_lo, x_hi : int
        Slice boundaries in the parent image.
    """
    half = box_size // 2
    ny, nx = image_shape

    y_lo = max(0, int(round(y0)) - half)
    y_hi = min(ny, int(round(y0)) + half + 1)
    x_lo = max(0, int(round(x0)) - half)
    x_hi = min(nx, int(round(x0)) + half + 1)

    out_h = y_hi - y_lo
    out_w = x_hi - x_lo

    # Build coordinate grids relative to source centre
    yy = np.arange(y_lo, y_hi) - y0
    xx = np.arange(x_lo, x_hi) - x0

    # Map to PSF-model pixel coordinates
    psf_cy = (psf_model.shape[0] - 1) / 2.0
    psf_cx = (psf_model.shape[1] - 1) / 2.0

    py = np.round(yy + psf_cy).astype(int)
    px = np.round(xx + psf_cx).astype(int)

    stamp = np.zeros((out_h, out_w), dtype=np.float64)
    for i, iy in enumerate(py):
        for j, ix in enumerate(px):
            if 0 <= iy < psf_model.shape[0] and 0 <= ix < psf_model.shape[1]:
                stamp[i, j] = psf_model[iy, ix]

    return stamp, y_lo, y_hi, x_lo, x_hi


def _robust_std(arr: np.ndarray) -> float:
    """MAD-based robust standard-deviation estimate.

    Parameters
    ----------
    arr : ndarray
        Input array (NaNs ignored).

    Returns
    -------
    float
        Robust standard deviation.
    """
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0
    return float(MAD_TO_STD * np.median(np.abs(finite - np.median(finite))))


def _circular_mask(radius: float, shape: tuple[int, int],
                   x0: float, y0: float) -> np.ndarray:
    """Boolean mask for a circle of given radius centred at *(x0, y0)*.

    Parameters
    ----------
    radius : float
        Circle radius in pixels.
    shape : tuple of int
        (ny, nx) of the output mask.
    x0, y0 : float
        Centre of the circle (x = column, y = row).

    Returns
    -------
    ndarray of bool
        True inside the circle.
    """
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    return ((xx - x0) ** 2 + (yy - y0) ** 2) <= radius ** 2


# =========================================================================
# 1. PSF Photometry
# =========================================================================

def psf_photometry(
    image: np.ndarray,
    positions: np.ndarray,
    psf_model: np.ndarray,
    box_size: int = 21,
    n_iters: int = 3,
) -> list[dict[str, float]]:
    """Simultaneous PSF-fitting photometry at given positions.

    For each source the amplitude is determined by linear least-squares
    minimisation of ``||cutout - amplitude * psf||^2`` within a square
    box.  The fit is iterated *n_iters* times, subtracting neighbouring
    source contributions between iterations.

    Parameters
    ----------
    image : ndarray
        2-D science image.
    positions : ndarray, shape (N, 2)
        Source positions as ``[[x0, y0], ...]``.
    psf_model : ndarray
        Normalised 2-D PSF stamp (should sum to ~1).
    box_size : int
        Side length of the fitting box in pixels.
    n_iters : int
        Number of fitting iterations.

    Returns
    -------
    list of dict
        One dictionary per source with keys ``x``, ``y``, ``flux``,
        ``flux_err``, ``chi2``.
    """
    image = np.asarray(image, dtype=np.float64)
    positions = np.atleast_2d(np.asarray(positions, dtype=np.float64))
    psf_model = np.asarray(psf_model, dtype=np.float64)

    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions must have shape (N, 2)")
    if image.ndim != 2:
        raise ValueError("image must be 2-D")

    n_sources = positions.shape[0]
    fluxes = np.zeros(n_sources, dtype=np.float64)
    flux_errs = np.zeros(n_sources, dtype=np.float64)
    chi2s = np.zeros(n_sources, dtype=np.float64)

    residual = image.copy()

    for _iteration in range(n_iters):
        # Add back current model before re-fitting
        for k in range(n_sources):
            x0, y0 = positions[k]
            stamp, yl, yh, xl, xh = _stamp_at(
                psf_model, x0, y0, box_size, image.shape,
            )
            residual[yl:yh, xl:xh] += fluxes[k] * stamp

        new_fluxes = np.zeros(n_sources, dtype=np.float64)
        new_flux_errs = np.zeros(n_sources, dtype=np.float64)
        new_chi2s = np.zeros(n_sources, dtype=np.float64)

        for k in range(n_sources):
            x0, y0 = positions[k]
            stamp, yl, yh, xl, xh = _stamp_at(
                psf_model, x0, y0, box_size, image.shape,
            )

            cutout = residual[yl:yh, xl:xh].copy()

            # Subtract contributions of other sources
            for j in range(n_sources):
                if j == k:
                    continue
                xj, yj = positions[j]
                sj, yjl, yjh, xjl, xjh = _stamp_at(
                    psf_model, xj, yj, box_size, image.shape,
                )
                # Overlap region
                o_yl = max(yl, yjl)
                o_yh = min(yh, yjh)
                o_xl = max(xl, xjl)
                o_xh = min(xh, xjh)
                if o_yl >= o_yh or o_xl >= o_xh:
                    continue
                cutout[
                    o_yl - yl:o_yh - yl,
                    o_xl - xl:o_xh - xl,
                ] -= fluxes[j] * sj[
                    o_yl - yjl:o_yh - yjl,
                    o_xl - xjl:o_xh - xjl,
                ]

            # Linear least-squares: flux = sum(psf * data) / sum(psf^2)
            psf_flat = stamp.ravel()
            data_flat = cutout.ravel()

            mask = np.isfinite(data_flat) & np.isfinite(psf_flat)
            psf_v = psf_flat[mask]
            data_v = data_flat[mask]

            denom = np.dot(psf_v, psf_v)
            if denom == 0.0:
                new_fluxes[k] = 0.0
                new_flux_errs[k] = np.nan
                new_chi2s[k] = np.nan
                continue

            amp = np.dot(psf_v, data_v) / denom
            resid = data_v - amp * psf_v
            n_pix = psf_v.size
            if n_pix > 1:
                sigma2 = np.sum(resid ** 2) / (n_pix - 1)
                amp_err = np.sqrt(sigma2 / denom) if sigma2 > 0 else 0.0
                chi2_val = float(np.sum(resid ** 2) / max(sigma2, 1e-30))
            else:
                amp_err = np.nan
                chi2_val = np.nan

            new_fluxes[k] = amp
            new_flux_errs[k] = amp_err
            new_chi2s[k] = chi2_val

        # Subtract updated model from residual
        for k in range(n_sources):
            x0, y0 = positions[k]
            stamp, yl, yh, xl, xh = _stamp_at(
                psf_model, x0, y0, box_size, image.shape,
            )
            residual[yl:yh, xl:xh] -= new_fluxes[k] * stamp

        fluxes = new_fluxes
        flux_errs = new_flux_errs
        chi2s = new_chi2s

    results: list[dict[str, float]] = []
    for k in range(n_sources):
        results.append({
            "x": float(positions[k, 0]),
            "y": float(positions[k, 1]),
            "flux": float(fluxes[k]),
            "flux_err": float(flux_errs[k]),
            "chi2": float(chi2s[k]),
        })
    return results


def iterative_psf_subtract(
    image: np.ndarray,
    positions: np.ndarray,
    psf_model: np.ndarray,
    box_size: int = 21,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Iteratively subtract fitted PSFs from an image.

    Each source is fitted and subtracted in order of decreasing
    brightness, which is particularly useful for crowded fields.

    Parameters
    ----------
    image : ndarray
        2-D science image.
    positions : ndarray, shape (N, 2)
        Source positions as ``[[x0, y0], ...]``.
    psf_model : ndarray
        Normalised 2-D PSF stamp.
    box_size : int
        Side length of the fitting box in pixels.

    Returns
    -------
    residual : ndarray
        Image after all PSFs have been subtracted.
    source_table : list of dict
        One dictionary per source with keys ``x``, ``y``, ``flux``,
        ``flux_err``, ``chi2``.
    """
    image = np.asarray(image, dtype=np.float64)
    positions = np.atleast_2d(np.asarray(positions, dtype=np.float64))
    psf_model = np.asarray(psf_model, dtype=np.float64)

    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions must have shape (N, 2)")
    if image.ndim != 2:
        raise ValueError("image must be 2-D")

    residual = image.copy()
    n_sources = positions.shape[0]

    # Initial rough flux estimates (sum in cutout)
    rough_flux = np.zeros(n_sources)
    for k in range(n_sources):
        x0, y0 = positions[k]
        cut, *_ = _cutout(residual, x0, y0, box_size)
        rough_flux[k] = np.nansum(cut)

    # Process brightest first
    order = np.argsort(-np.abs(rough_flux))

    source_table: list[dict[str, float]] = [{} for _ in range(n_sources)]

    for k in order:
        x0, y0 = positions[k]
        stamp, yl, yh, xl, xh = _stamp_at(
            psf_model, x0, y0, box_size, image.shape,
        )
        cutout = residual[yl:yh, xl:xh].copy()

        psf_flat = stamp.ravel()
        data_flat = cutout.ravel()
        mask = np.isfinite(data_flat) & np.isfinite(psf_flat)
        psf_v = psf_flat[mask]
        data_v = data_flat[mask]

        denom = np.dot(psf_v, psf_v)
        if denom == 0.0:
            source_table[k] = {
                "x": float(x0), "y": float(y0),
                "flux": 0.0, "flux_err": np.nan, "chi2": np.nan,
            }
            continue

        amp = np.dot(psf_v, data_v) / denom
        resid_vec = data_v - amp * psf_v
        n_pix = psf_v.size
        if n_pix > 1:
            sigma2 = np.sum(resid_vec ** 2) / (n_pix - 1)
            amp_err = np.sqrt(sigma2 / denom) if sigma2 > 0 else 0.0
            chi2_val = float(np.sum(resid_vec ** 2) / max(sigma2, 1e-30))
        else:
            amp_err = np.nan
            chi2_val = np.nan

        # Subtract model from residual
        residual[yl:yh, xl:xh] -= amp * stamp

        source_table[k] = {
            "x": float(x0),
            "y": float(y0),
            "flux": float(amp),
            "flux_err": float(amp_err),
            "chi2": float(chi2_val),
        }

    return residual, source_table


# =========================================================================
# 2. Extended Source Photometry
# =========================================================================

def petrosian_radius(
    radial_profile: np.ndarray,
    radii: np.ndarray,
    eta: float = 0.2,
) -> float:
    """Compute the Petrosian radius.

    The Petrosian radius is defined as the radius where the ratio of the
    local surface brightness to the mean surface brightness within that
    radius equals *eta*.

    Parameters
    ----------
    radial_profile : ndarray
        1-D azimuthally-averaged surface-brightness profile.
    radii : ndarray
        1-D array of radii corresponding to *radial_profile*.
    eta : float
        Petrosian ratio threshold (default 0.2).

    Returns
    -------
    float
        Petrosian radius.  Returns ``np.nan`` if the criterion is never
        satisfied.
    """
    radial_profile = np.asarray(radial_profile, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)

    if radial_profile.size == 0 or radii.size == 0:
        return np.nan

    # Cumulative mean surface brightness: <SB>(<r) = cumsum(SB*dr) / r
    dr = np.gradient(radii)
    cumulative = np.cumsum(radial_profile * dr)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_sb = np.where(radii > 0, cumulative / radii, 0.0)

    # Petrosian ratio: SB(r) / <SB>(<r)
    with np.errstate(divide="ignore", invalid="ignore"):
        pet_ratio = np.where(mean_sb > 0, radial_profile / mean_sb, np.inf)

    # Find first crossing below eta
    below = np.where(pet_ratio <= eta)[0]
    if below.size == 0:
        return np.nan

    idx = below[0]
    if idx == 0:
        return float(radii[0])

    # Linear interpolation between bracketing points
    r_lo, r_hi = radii[idx - 1], radii[idx]
    p_lo, p_hi = pet_ratio[idx - 1], pet_ratio[idx]
    if p_lo == p_hi:
        return float(r_lo)
    frac = (eta - p_lo) / (p_hi - p_lo)
    return float(r_lo + frac * (r_hi - r_lo))


def kron_radius(
    image: np.ndarray,
    x0: float,
    y0: float,
    initial_radius: float = 5.0,
    k: float = 2.5,
) -> float:
    """Compute the Kron radius for an extended source.

    The first-moment (Kron) radius is defined as

    .. math::

        r_{\\text{kron}} = \\frac{\\sum r\\,I(r)}{\\sum I(r)}

    evaluated within *initial_radius*.  The returned value is
    ``k * r_kron``.

    Parameters
    ----------
    image : ndarray
        2-D science image.
    x0, y0 : float
        Source centre in pixel coordinates (x = column, y = row).
    initial_radius : float
        Radius within which the first moment is computed.
    k : float
        Kron scaling factor (default 2.5).

    Returns
    -------
    float
        Scaled Kron radius ``k * r_kron``.  Returns ``np.nan`` if the
        flux sum is non-positive.
    """
    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 2:
        raise ValueError("image must be 2-D")

    ny, nx = image.shape
    yy, xx = np.mgrid[:ny, :nx]
    rr = np.sqrt((xx - x0) ** 2 + (yy - y0) ** 2)

    inside = rr <= initial_radius
    intensities = image[inside]
    distances = rr[inside]

    # Only consider positive pixels
    pos = intensities > 0
    if not np.any(pos):
        return np.nan

    sum_rI = np.sum(distances[pos] * intensities[pos])
    sum_I = np.sum(intensities[pos])

    if sum_I <= 0:
        return np.nan

    r_kron = sum_rI / sum_I
    return float(k * r_kron)


def isophotal_photometry(
    image: np.ndarray,
    x0: float,
    y0: float,
    threshold: float,
    max_radius: int = 100,
) -> dict[str, float]:
    """Compute flux within an isophote connected to the source centre.

    Pixels above *threshold* that are contiguously connected to the
    centre pixel are summed.

    Parameters
    ----------
    image : ndarray
        2-D science image.
    x0, y0 : float
        Source centre in pixel coordinates (x = column, y = row).
    threshold : float
        Surface-brightness threshold defining the isophote.
    max_radius : int
        Maximum radius to consider (limits search area).

    Returns
    -------
    dict
        Keys: ``flux``, ``area`` (pixels), ``mean_sb`` (mean surface
        brightness), ``equivalent_radius`` (radius of equal-area circle).

    Raises
    ------
    ValueError
        If image is not 2-D or centre is outside the image.
    """
    ndi = _import_scipy_ndimage()

    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 2:
        raise ValueError("image must be 2-D")

    ny, nx = image.shape
    ix0, iy0 = int(round(x0)), int(round(y0))

    if not (0 <= iy0 < ny and 0 <= ix0 < nx):
        raise ValueError(
            f"Centre ({x0}, {y0}) is outside image of shape {image.shape}"
        )

    # Restrict to sub-image around centre for efficiency
    yl = max(0, iy0 - max_radius)
    yh = min(ny, iy0 + max_radius + 1)
    xl = max(0, ix0 - max_radius)
    xh = min(nx, ix0 + max_radius + 1)

    sub = image[yl:yh, xl:xh]
    above = sub >= threshold

    # Label connected components
    labels, n_labels = ndi.label(above)

    # Identify label at centre
    cy, cx = iy0 - yl, ix0 - xl
    centre_label = labels[cy, cx]

    if centre_label == 0:
        # Centre pixel below threshold
        return {
            "flux": 0.0,
            "area": 0.0,
            "mean_sb": 0.0,
            "equivalent_radius": 0.0,
        }

    component = labels == centre_label
    area = float(np.sum(component))
    flux = float(np.sum(sub[component]))
    mean_sb = flux / area if area > 0 else 0.0
    equivalent_radius = float(np.sqrt(area / np.pi))

    return {
        "flux": flux,
        "area": area,
        "mean_sb": mean_sb,
        "equivalent_radius": equivalent_radius,
    }


def radial_profile(
    image: np.ndarray,
    x0: float,
    y0: float,
    max_radius: float = 50.0,
    n_bins: int = 50,
) -> dict[str, np.ndarray]:
    """Compute azimuthally-averaged radial surface-brightness profile.

    Parameters
    ----------
    image : ndarray
        2-D science image.
    x0, y0 : float
        Centre in pixel coordinates (x = column, y = row).
    max_radius : float
        Maximum radius in pixels.
    n_bins : int
        Number of radial bins.

    Returns
    -------
    dict
        Keys: ``radii`` (bin centres), ``profile`` (mean surface
        brightness per bin), ``profile_err`` (standard error of the mean
        per bin).
    """
    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 2:
        raise ValueError("image must be 2-D")

    ny, nx = image.shape
    yy, xx = np.mgrid[:ny, :nx]
    rr = np.sqrt((xx - x0) ** 2 + (yy - y0) ** 2)

    bin_edges = np.linspace(0, max_radius, n_bins + 1)
    radii_out = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    profile_out = np.full(n_bins, np.nan)
    profile_err_out = np.full(n_bins, np.nan)

    for i in range(n_bins):
        mask = (rr >= bin_edges[i]) & (rr < bin_edges[i + 1])
        vals = image[mask]
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            continue
        profile_out[i] = np.mean(finite)
        if finite.size > 1:
            profile_err_out[i] = np.std(finite, ddof=1) / np.sqrt(finite.size)
        else:
            profile_err_out[i] = 0.0

    return {
        "radii": radii_out,
        "profile": profile_out,
        "profile_err": profile_err_out,
    }


# =========================================================================
# 3. Photometric Calibration
# =========================================================================

def compute_zeropoint(
    inst_mags: np.ndarray,
    catalog_mags: np.ndarray,
    weights: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute photometric zero-point from matched star lists.

    The zero-point is the robust (median) offset between catalogue and
    instrumental magnitudes:

    .. math::

        \\text{ZP} = \\mathrm{median}(m_{\\text{cat}} - m_{\\text{inst}})

    Parameters
    ----------
    inst_mags : ndarray
        Instrumental magnitudes.
    catalog_mags : ndarray
        Catalogue (true) magnitudes.
    weights : ndarray, optional
        Per-star weights.  If provided, a weighted median is
        approximated by repeating values.

    Returns
    -------
    dict
        Keys: ``zeropoint``, ``zeropoint_err`` (MAD-based),
        ``n_stars``, ``residual_rms``.

    Raises
    ------
    ValueError
        If input arrays differ in length or are empty.
    """
    inst_mags = np.asarray(inst_mags, dtype=np.float64).ravel()
    catalog_mags = np.asarray(catalog_mags, dtype=np.float64).ravel()

    if inst_mags.size != catalog_mags.size:
        raise ValueError(
            "inst_mags and catalog_mags must have the same length"
        )

    # Mask non-finite values
    diff = catalog_mags - inst_mags
    valid = np.isfinite(diff)

    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64).ravel()
        if weights.size != inst_mags.size:
            raise ValueError("weights must match input array length")
        valid &= np.isfinite(weights) & (weights > 0)

    diff = diff[valid]
    n_stars = diff.size

    if n_stars == 0:
        return {
            "zeropoint": np.nan,
            "zeropoint_err": np.nan,
            "n_stars": 0,
            "residual_rms": np.nan,
        }

    if weights is not None:
        w = weights[valid]
        # Weighted median via sorted cumulative weights
        order = np.argsort(diff)
        sorted_diff = diff[order]
        cum_w = np.cumsum(w[order])
        mid = cum_w[-1] / 2.0
        idx = np.searchsorted(cum_w, mid)
        idx = min(idx, len(sorted_diff) - 1)
        zp = float(sorted_diff[idx])
    else:
        zp = float(np.median(diff))

    residuals = diff - zp
    zp_err = float(MAD_TO_STD * np.median(np.abs(residuals)))
    rms = float(np.sqrt(np.mean(residuals ** 2)))

    return {
        "zeropoint": zp,
        "zeropoint_err": zp_err,
        "n_stars": n_stars,
        "residual_rms": rms,
    }


def apply_zeropoint(
    inst_mags: np.ndarray,
    zeropoint: float,
    inst_mag_errs: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Apply a photometric zero-point to instrumental magnitudes.

    .. math::

        m_{\\text{cal}} = m_{\\text{inst}} + \\text{ZP}

    Parameters
    ----------
    inst_mags : ndarray
        Instrumental magnitudes.
    zeropoint : float
        Photometric zero-point.
    inst_mag_errs : ndarray, optional
        Uncertainties on instrumental magnitudes.  If provided, the same
        values are returned (zero-point is a constant offset).

    Returns
    -------
    dict
        Keys: ``cal_mags`` and optionally ``cal_mag_errs``.
    """
    inst_mags = np.asarray(inst_mags, dtype=np.float64)
    cal_mags = inst_mags + zeropoint

    result: dict[str, Any] = {"cal_mags": cal_mags}

    if inst_mag_errs is not None:
        result["cal_mag_errs"] = np.asarray(inst_mag_errs, dtype=np.float64).copy()

    return result


def extinction_correct(
    mags: np.ndarray,
    airmass: float | np.ndarray,
    extinction_coeff: float,
) -> np.ndarray:
    """Correct magnitudes for atmospheric extinction.

    .. math::

        m_{\\text{corrected}} = m - k \\cdot X

    where *k* is the extinction coefficient and *X* is the airmass.

    Parameters
    ----------
    mags : ndarray
        Observed magnitudes.
    airmass : float or ndarray
        Airmass value(s).
    extinction_coeff : float
        Extinction coefficient *k* (magnitudes per airmass).

    Returns
    -------
    ndarray
        Extinction-corrected magnitudes.
    """
    mags = np.asarray(mags, dtype=np.float64)
    airmass = np.asarray(airmass, dtype=np.float64)
    return mags - extinction_coeff * airmass


def color_term_correct(
    mags: np.ndarray,
    colors: np.ndarray,
    color_coeff: float,
) -> np.ndarray:
    """Apply a colour-term correction to magnitudes.

    .. math::

        m_{\\text{corrected}} = m + c \\cdot \\text{colour}

    Parameters
    ----------
    mags : ndarray
        Magnitudes to correct.
    colors : ndarray
        Colour indices (e.g. B-V) for each source.
    color_coeff : float
        Colour coefficient *c*.

    Returns
    -------
    ndarray
        Colour-corrected magnitudes.
    """
    mags = np.asarray(mags, dtype=np.float64)
    colors = np.asarray(colors, dtype=np.float64)
    return mags + color_coeff * colors


# =========================================================================
# 4. Aperture Corrections
# =========================================================================

def aperture_correction(
    image: np.ndarray,
    positions: np.ndarray,
    radii_small: float,
    radii_large: float,
    background: float | None = None,
) -> dict[str, float]:
    """Compute the aperture correction between two aperture sizes.

    Measures the flux in small and large circular apertures for a set of
    sources and returns the median magnitude offset.

    Parameters
    ----------
    image : ndarray
        2-D science image.
    positions : ndarray, shape (N, 2)
        Source positions as ``[[x0, y0], ...]``.
    radii_small : float
        Radius of the small aperture in pixels.
    radii_large : float
        Radius of the large aperture in pixels.
    background : float, optional
        Per-pixel background level to subtract.  If ``None`` the
        background is assumed already subtracted.

    Returns
    -------
    dict
        Keys: ``correction`` (median magnitude offset, negative means
        the small aperture loses flux), ``correction_err``
        (MAD-based uncertainty), ``n_sources`` (number of usable
        sources).

    Raises
    ------
    ValueError
        If ``radii_small >= radii_large``.
    """
    image = np.asarray(image, dtype=np.float64)
    positions = np.atleast_2d(np.asarray(positions, dtype=np.float64))

    if radii_small >= radii_large:
        raise ValueError("radii_small must be less than radii_large")
    if image.ndim != 2:
        raise ValueError("image must be 2-D")

    bg = background if background is not None else 0.0

    offsets: list[float] = []

    for k in range(positions.shape[0]):
        x0, y0 = positions[k]

        # Ensure apertures fit inside image
        ny, nx = image.shape
        if (x0 - radii_large < -0.5 or x0 + radii_large > nx - 0.5 or
                y0 - radii_large < -0.5 or y0 + radii_large > ny - 0.5):
            continue

        mask_s = _circular_mask(radii_small, image.shape, x0, y0)
        mask_l = _circular_mask(radii_large, image.shape, x0, y0)

        flux_s = np.sum(image[mask_s]) - bg * np.sum(mask_s)
        flux_l = np.sum(image[mask_l]) - bg * np.sum(mask_l)

        if flux_s > 0 and flux_l > 0:
            offsets.append(-2.5 * np.log10(flux_s / flux_l))

    n = len(offsets)
    if n == 0:
        return {
            "correction": np.nan,
            "correction_err": np.nan,
            "n_sources": 0,
        }

    offsets_arr = np.array(offsets)
    corr = float(np.median(offsets_arr))
    corr_err = float(
        MAD_TO_STD * np.median(np.abs(offsets_arr - corr))
    )

    return {
        "correction": corr,
        "correction_err": corr_err,
        "n_sources": n,
    }


def curve_of_growth(
    image: np.ndarray,
    x0: float,
    y0: float,
    max_radius: float = 30.0,
    step: float = 1.0,
    background: float = 0.0,
) -> dict[str, np.ndarray]:
    """Compute cumulative flux as a function of aperture radius.

    Parameters
    ----------
    image : ndarray
        2-D science image.
    x0, y0 : float
        Source centre in pixel coordinates (x = column, y = row).
    max_radius : float
        Maximum aperture radius in pixels.
    step : float
        Radial step size in pixels.
    background : float
        Per-pixel background level to subtract.

    Returns
    -------
    dict
        Keys: ``radii``, ``enclosed_flux``, ``normalized_flux`` (flux
        normalised to the value at *max_radius*).
    """
    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 2:
        raise ValueError("image must be 2-D")

    radii = np.arange(step, max_radius + step / 2.0, step)
    enclosed = np.zeros_like(radii)

    ny, nx = image.shape
    yy, xx = np.mgrid[:ny, :nx]
    rr = np.sqrt((xx - x0) ** 2 + (yy - y0) ** 2)

    for i, r in enumerate(radii):
        mask = rr <= r
        enclosed[i] = np.sum(image[mask]) - background * np.sum(mask)

    total = enclosed[-1] if enclosed.size > 0 and enclosed[-1] != 0 else 1.0
    normalized = enclosed / total

    return {
        "radii": radii,
        "enclosed_flux": enclosed,
        "normalized_flux": normalized,
    }


# =========================================================================
# 5. Crowded-Field Photometry
# =========================================================================

def find_neighbors(
    positions: np.ndarray,
    radius: float,
) -> list[tuple[int, int, float]]:
    """Find all pairs of sources within a given radius.

    Parameters
    ----------
    positions : ndarray, shape (N, 2)
        Source positions as ``[[x0, y0], ...]``.
    radius : float
        Search radius in pixels.

    Returns
    -------
    list of (int, int, float)
        Each tuple is ``(i, j, distance)`` with ``i < j``.
    """
    positions = np.atleast_2d(np.asarray(positions, dtype=np.float64))
    n = positions.shape[0]

    if n == 0:
        return []

    pairs: list[tuple[int, int, float]] = []

    for i in range(n):
        # Vectorised distance from source i to all later sources
        dx = positions[i + 1:, 0] - positions[i, 0]
        dy = positions[i + 1:, 1] - positions[i, 1]
        dists = np.sqrt(dx ** 2 + dy ** 2)

        close = np.where(dists <= radius)[0]
        for idx in close:
            j = i + 1 + idx
            pairs.append((i, j, float(dists[idx])))

    return pairs


def deblend_sources(
    image: np.ndarray,
    positions: np.ndarray,
    psf_model: np.ndarray,
    box_size: int = 31,
    n_iters: int = 10,
) -> list[dict[str, float]]:
    """Simultaneous multi-source PSF fitting for blended sources.

    All sources within the fitting region are modelled simultaneously.
    The algorithm iteratively fits each source after subtracting the
    current model of all neighbours, converging to the best-fit
    amplitudes.

    Parameters
    ----------
    image : ndarray
        2-D science image.
    positions : ndarray, shape (N, 2)
        Source positions as ``[[x0, y0], ...]``.
    psf_model : ndarray
        Normalised 2-D PSF stamp.
    box_size : int
        Side length of the fitting box centred on each source.
    n_iters : int
        Number of iterations for the simultaneous fit.

    Returns
    -------
    list of dict
        One dictionary per source with keys ``x``, ``y``, ``flux``,
        ``flux_err``.
    """
    image = np.asarray(image, dtype=np.float64)
    positions = np.atleast_2d(np.asarray(positions, dtype=np.float64))
    psf_model = np.asarray(psf_model, dtype=np.float64)

    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions must have shape (N, 2)")
    if image.ndim != 2:
        raise ValueError("image must be 2-D")

    n_sources = positions.shape[0]
    fluxes = np.zeros(n_sources, dtype=np.float64)
    flux_errs = np.full(n_sources, np.nan)

    # Pre-compute stamps
    stamps: list[tuple[np.ndarray, int, int, int, int]] = []
    for k in range(n_sources):
        x0, y0 = positions[k]
        stamps.append(
            _stamp_at(psf_model, x0, y0, box_size, image.shape)
        )

    for _iteration in range(n_iters):
        for k in range(n_sources):
            stamp_k, yl, yh, xl, xh = stamps[k]
            cutout = image[yl:yh, xl:xh].copy()

            # Subtract contributions of all other sources
            for j in range(n_sources):
                if j == k:
                    continue
                stamp_j, jyl, jyh, jxl, jxh = stamps[j]
                # Overlap region
                o_yl = max(yl, jyl)
                o_yh = min(yh, jyh)
                o_xl = max(xl, jxl)
                o_xh = min(xh, jxh)
                if o_yl >= o_yh or o_xl >= o_xh:
                    continue
                cutout[
                    o_yl - yl:o_yh - yl,
                    o_xl - xl:o_xh - xl,
                ] -= fluxes[j] * stamp_j[
                    o_yl - jyl:o_yh - jyl,
                    o_xl - jxl:o_xh - jxl,
                ]

            psf_flat = stamp_k.ravel()
            data_flat = cutout.ravel()
            mask = np.isfinite(data_flat) & np.isfinite(psf_flat)
            psf_v = psf_flat[mask]
            data_v = data_flat[mask]

            denom = np.dot(psf_v, psf_v)
            if denom == 0.0:
                fluxes[k] = 0.0
                flux_errs[k] = np.nan
                continue

            fluxes[k] = np.dot(psf_v, data_v) / denom

            resid = data_v - fluxes[k] * psf_v
            n_pix = psf_v.size
            if n_pix > 1:
                sigma2 = np.sum(resid ** 2) / (n_pix - 1)
                flux_errs[k] = np.sqrt(sigma2 / denom) if sigma2 > 0 else 0.0
            else:
                flux_errs[k] = np.nan

    results: list[dict[str, float]] = []
    for k in range(n_sources):
        results.append({
            "x": float(positions[k, 0]),
            "y": float(positions[k, 1]),
            "flux": float(fluxes[k]),
            "flux_err": float(flux_errs[k]),
        })
    return results


def completeness_test(
    image: np.ndarray,
    psf_model: np.ndarray,
    positions: np.ndarray,
    fluxes: np.ndarray,
    n_trials: int = 100,
) -> dict[str, np.ndarray]:
    """Estimate photometric completeness via artificial-star injection.

    For each trial, artificial sources are injected at random positions
    with the given fluxes and recovery is attempted using PSF fitting.
    A source is considered recovered if its measured flux is within a
    factor of two of the injected flux and within 2 pixels of the
    injected position.

    Parameters
    ----------
    image : ndarray
        2-D science image (used as the background into which stars are
        injected).
    psf_model : ndarray
        Normalised 2-D PSF stamp.
    positions : ndarray, shape (N, 2)
        Candidate injection positions as ``[[x0, y0], ...]``.  Each
        trial selects a random subset.
    fluxes : ndarray
        1-D array of flux levels to test.  Each flux level is tested
        independently.
    n_trials : int
        Number of injection/recovery trials per flux level.

    Returns
    -------
    dict
        Keys: ``injected_fluxes`` (the input flux levels),
        ``recovery_fraction`` (fraction recovered per flux level),
        ``magnitude_bins`` (approximate instrumental magnitude for each
        flux level, ``-2.5 * log10(flux)``).
    """
    image = np.asarray(image, dtype=np.float64)
    psf_model = np.asarray(psf_model, dtype=np.float64)
    positions = np.atleast_2d(np.asarray(positions, dtype=np.float64))
    fluxes = np.asarray(fluxes, dtype=np.float64).ravel()

    if image.ndim != 2:
        raise ValueError("image must be 2-D")
    if positions.shape[0] == 0:
        return {
            "injected_fluxes": fluxes,
            "recovery_fraction": np.zeros_like(fluxes),
            "magnitude_bins": np.full_like(fluxes, np.nan),
        }

    rng = np.random.default_rng(42)
    n_positions = positions.shape[0]
    recovery = np.zeros(fluxes.size, dtype=np.float64)
    box_size = max(psf_model.shape) + 4

    for fi, flux_val in enumerate(fluxes):
        n_recovered = 0
        n_total = 0

        for _trial in range(n_trials):
            # Pick a random position
            idx = rng.integers(0, n_positions)
            x0, y0 = positions[idx]

            # Inject artificial star
            test_image = image.copy()
            stamp, yl, yh, xl, xh = _stamp_at(
                psf_model, x0, y0, box_size, image.shape,
            )
            test_image[yl:yh, xl:xh] += flux_val * stamp

            # Attempt recovery
            pos_arr = np.array([[x0, y0]])
            result = psf_photometry(
                test_image, pos_arr, psf_model,
                box_size=box_size, n_iters=1,
            )
            measured = result[0]["flux"]

            # Recovery criteria: flux within factor 2, position is exact
            if (flux_val > 0
                    and 0.5 * flux_val <= measured <= 2.0 * flux_val):
                n_recovered += 1
            n_total += 1

        recovery[fi] = n_recovered / max(n_total, 1)

    with np.errstate(divide="ignore", invalid="ignore"):
        mag_bins = np.where(
            fluxes > 0,
            -2.5 * np.log10(fluxes),
            np.nan,
        )

    return {
        "injected_fluxes": fluxes,
        "recovery_fraction": recovery,
        "magnitude_bins": mag_bins,
    }
