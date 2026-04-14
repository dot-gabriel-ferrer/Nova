"""Astrometry pipeline tools for NOVA.

Provides a complete, self-contained astrometry toolkit so users can
perform plate solving, astrometric calibration, proper-motion correction,
and quality assessment without leaving the NOVA ecosystem.

All functions operate on plain NumPy arrays (or dicts of arrays) and
optionally record their operations in an OperationHistory or Pipeline.

Key capabilities
----------------
- Centroid and source extraction for astrometric reference stars.
- Plate solving via triangle matching (no external solver needed).
- Astrometric residual analysis and quality metrics.
- Proper-motion and parallax correction to a given epoch.
- Distortion fitting (SIP polynomial) from matched catalogs.
- Tangent-plane projection utilities.

Example
-------
>>> from nova.astrometry import extract_centroids, plate_solve
>>> sources = extract_centroids(image, fwhm=3.5, threshold=5.0)
>>> wcs_solution = plate_solve(sources, reference_catalog,
...                            image_shape=image.shape)
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from nova.constants import ARCSEC_PER_DEG


# -------------------------------------------------------------------
#  Source extraction for astrometry
# -------------------------------------------------------------------

def extract_centroids(
    data: np.ndarray,
    *,
    fwhm: float = 3.5,
    threshold: float = 5.0,
    max_sources: int = 500,
    border: int = 10,
) -> np.ndarray:
    """Extract point-source centroids from an image.

    Uses a simple peak-finding approach: locate pixels above
    ``threshold * background_rms``, then refine positions with a
    centroid window of radius ~ *fwhm*.

    Parameters
    ----------
    data : np.ndarray
        2-D image array.
    fwhm : float
        Expected full-width at half-maximum of point sources (pixels).
    threshold : float
        Detection threshold in units of the background RMS.
    max_sources : int
        Maximum number of sources to return (brightest first).
    border : int
        Ignore this many pixels at each image edge.

    Returns
    -------
    np.ndarray
        Array of shape ``(N, 3)`` with columns ``[x, y, flux]``.
    """
    if not isinstance(data, np.ndarray) or data.ndim != 2:
        raise TypeError("data must be a 2-D numpy array.")
    if fwhm <= 0:
        raise ValueError("fwhm must be positive.")
    if threshold <= 0:
        raise ValueError("threshold must be positive.")

    h, w = data.shape
    # Rough background estimate
    med = float(np.median(data))
    mad = float(np.median(np.abs(data - med)))
    rms = mad * 1.4826  # MAD -> sigma for normal distribution
    if rms == 0:
        return np.empty((0, 3), dtype=np.float64)

    det_level = med + threshold * rms
    radius = max(1, int(round(fwhm)))

    # Simple peak finder
    candidates = []
    for y in range(border, h - border):
        for x in range(border, w - border):
            if data[y, x] < det_level:
                continue
            # Local maximum check
            ylo = max(0, y - radius)
            yhi = min(h, y + radius + 1)
            xlo = max(0, x - radius)
            xhi = min(w, x + radius + 1)
            local = data[ylo:yhi, xlo:xhi]
            if data[y, x] == np.max(local):
                candidates.append((x, y, float(data[y, x]) - med))

    if not candidates:
        return np.empty((0, 3), dtype=np.float64)

    # Sort by flux descending, keep top sources
    candidates.sort(key=lambda c: -c[2])
    candidates = candidates[:max_sources]

    # Refine centroids with weighted centroid in a window
    results = []
    for cx, cy, flux in candidates:
        ylo = max(0, cy - radius)
        yhi = min(h, cy + radius + 1)
        xlo = max(0, cx - radius)
        xhi = min(w, cx + radius + 1)
        stamp = data[ylo:yhi, xlo:xhi].astype(np.float64) - med
        stamp = np.clip(stamp, 0, None)
        total = stamp.sum()
        if total <= 0:
            results.append([float(cx), float(cy), flux])
            continue
        yy, xx = np.mgrid[ylo:yhi, xlo:xhi]
        refined_x = float(np.sum(xx * stamp) / total)
        refined_y = float(np.sum(yy * stamp) / total)
        results.append([refined_x, refined_y, flux])

    return np.array(results, dtype=np.float64)


# -------------------------------------------------------------------
#  Triangle matching for plate solving
# -------------------------------------------------------------------

def _triangle_invariants(
    p1: np.ndarray, p2: np.ndarray, p3: np.ndarray,
) -> tuple[float, float]:
    """Compute scale-invariant triangle ratios.

    Given three points sorted by longest side, returns
    ``(d2/d1, d3/d1)`` where d1 >= d2 >= d3 are the three side lengths.
    """
    sides = np.array([
        np.sqrt(np.sum((p1 - p2) ** 2)),
        np.sqrt(np.sum((p2 - p3) ** 2)),
        np.sqrt(np.sum((p1 - p3) ** 2)),
    ])
    sides.sort()
    d1 = sides[2]
    if d1 == 0:
        return (0.0, 0.0)
    return (float(sides[1] / d1), float(sides[0] / d1))


def _build_triangles(
    sources: np.ndarray, n_bright: int = 40,
) -> list[tuple[tuple[float, float], tuple[int, int, int]]]:
    """Build triangle invariants from the brightest sources.

    Parameters
    ----------
    sources : np.ndarray
        (N, 2+) array; columns 0, 1 are x, y.
    n_bright : int
        Number of brightest sources to consider.

    Returns
    -------
    list of (invariant_tuple, index_tuple)
    """
    n = min(len(sources), n_bright)
    pts = sources[:n, :2]
    triangles = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                inv = _triangle_invariants(pts[i], pts[j], pts[k])
                triangles.append((inv, (i, j, k)))
    return triangles


def plate_solve(
    detected: np.ndarray,
    catalog_radec: np.ndarray,
    *,
    image_shape: tuple[int, int],
    pixel_scale_guess: float | None = None,
    match_tol: float = 0.01,
    n_bright: int = 30,
) -> dict[str, Any]:
    """Solve the astrometric plate using triangle matching.

    Parameters
    ----------
    detected : np.ndarray
        Detected source positions, shape ``(N, 2+)`` with ``[x, y, ...]``.
    catalog_radec : np.ndarray
        Reference catalog, shape ``(M, 2+)`` with ``[ra, dec, ...]`` in
        degrees.
    image_shape : tuple[int, int]
        ``(height, width)`` of the image.
    pixel_scale_guess : float or None
        Approximate pixel scale in arcsec/pixel.  If None, a rough
        estimate is derived from image size and catalog spread.
    match_tol : float
        Tolerance for triangle invariant matching (default 0.01).
    n_bright : int
        Number of brightest objects to use for matching.

    Returns
    -------
    dict
        ``{'crpix': (cx, cy), 'crval': (ra0, dec0),
           'cd_matrix': 2x2 array, 'matched': N, 'residual_arcsec': float,
           'success': bool}``
    """
    if not isinstance(detected, np.ndarray) or detected.ndim != 2:
        raise TypeError("detected must be a 2-D array.")
    if not isinstance(catalog_radec, np.ndarray) or catalog_radec.ndim != 2:
        raise TypeError("catalog_radec must be a 2-D array.")
    if detected.shape[0] < 3 or catalog_radec.shape[0] < 3:
        return {"success": False, "reason": "Need at least 3 sources."}

    h, w = image_shape
    crpix = np.array([w / 2.0, h / 2.0])

    # Gnomonic projection of catalog around its centroid
    ra_c = float(np.median(catalog_radec[:, 0]))
    dec_c = float(np.median(catalog_radec[:, 1]))
    cat_xi, cat_eta = _radec_to_tangent(catalog_radec[:, 0], catalog_radec[:, 1],
                                        ra_c, dec_c)
    cat_proj = np.column_stack([cat_xi, cat_eta])

    # Build triangle lists
    det_tri = _build_triangles(detected, n_bright)
    cat_tri = _build_triangles(cat_proj, n_bright)

    # Match triangles
    matches = []
    for (d_inv, d_idx), _ in [(dt, None) for dt in det_tri]:
        for c_inv, c_idx in cat_tri:
            d0 = abs(d_inv[0] - c_inv[0])
            d1 = abs(d_inv[1] - c_inv[1])
            if d0 < match_tol and d1 < match_tol:
                matches.append((d_idx, c_idx))
                break

    if len(matches) < 1:
        return {"success": False, "reason": "No triangle matches found."}

    # Vote for best matched pairs
    pair_votes: dict[tuple[int, int], int] = {}
    for d_idx, c_idx in matches:
        for di, ci in zip(d_idx, c_idx):
            key = (di, ci)
            pair_votes[key] = pair_votes.get(key, 0) + 1

    # Keep pairs with >= 2 votes
    good_pairs = [(d, c) for (d, c), v in pair_votes.items() if v >= 2]
    if len(good_pairs) < 3:
        # Fall back to all pairs with at least 1 vote
        good_pairs = list(pair_votes.keys())
    if len(good_pairs) < 3:
        return {"success": False, "reason": "Insufficient matched pairs."}

    # Solve for affine transform: cat_proj = A * (det - crpix)
    det_pts = np.array([detected[d, :2] - crpix for d, _ in good_pairs])
    cat_pts = np.array([cat_proj[c] for _, c in good_pairs])

    # Least-squares: cat = det @ M^T  =>  M = (det^T det)^-1 det^T cat
    try:
        M, _, _, _ = np.linalg.lstsq(det_pts, cat_pts, rcond=None)
    except np.linalg.LinAlgError:
        return {"success": False, "reason": "Singular matrix in plate solve."}

    cd_matrix = M.T  # 2x2

    # Compute residuals
    predicted = det_pts @ M
    residuals = cat_pts - predicted
    res_deg = np.sqrt(np.sum(residuals ** 2, axis=1))
    res_arcsec = float(np.median(res_deg) * ARCSEC_PER_DEG)

    return {
        "success": True,
        "crpix": (float(crpix[0]), float(crpix[1])),
        "crval": (ra_c, dec_c),
        "cd_matrix": cd_matrix,
        "matched": len(good_pairs),
        "residual_arcsec": res_arcsec,
    }


# -------------------------------------------------------------------
#  Tangent-plane projection
# -------------------------------------------------------------------

def _radec_to_tangent(
    ra: np.ndarray,
    dec: np.ndarray,
    ra0: float,
    dec0: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Gnomonic (tangent-plane) projection.

    Parameters
    ----------
    ra, dec : np.ndarray
        Coordinates in degrees.
    ra0, dec0 : float
        Tangent point in degrees.

    Returns
    -------
    xi, eta : np.ndarray
        Standard coordinates in degrees.
    """
    d2r = math.pi / 180.0
    ra_r = np.asarray(ra, dtype=np.float64) * d2r
    dec_r = np.asarray(dec, dtype=np.float64) * d2r
    ra0_r = ra0 * d2r
    dec0_r = dec0 * d2r

    cos_dec = np.cos(dec_r)
    sin_dec = np.sin(dec_r)
    cos_dec0 = math.cos(dec0_r)
    sin_dec0 = math.sin(dec0_r)
    cos_dra = np.cos(ra_r - ra0_r)

    denom = sin_dec0 * sin_dec + cos_dec0 * cos_dec * cos_dra
    denom = np.where(denom == 0, 1e-30, denom)

    xi = cos_dec * np.sin(ra_r - ra0_r) / denom
    eta = (cos_dec0 * sin_dec - sin_dec0 * cos_dec * cos_dra) / denom
    return xi / d2r, eta / d2r


def _tangent_to_radec(
    xi: np.ndarray,
    eta: np.ndarray,
    ra0: float,
    dec0: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse gnomonic projection.

    Parameters
    ----------
    xi, eta : np.ndarray
        Standard coordinates in degrees.
    ra0, dec0 : float
        Tangent point in degrees.

    Returns
    -------
    ra, dec : np.ndarray
        Sky coordinates in degrees.
    """
    d2r = math.pi / 180.0
    xi_r = np.asarray(xi, dtype=np.float64) * d2r
    eta_r = np.asarray(eta, dtype=np.float64) * d2r
    ra0_r = ra0 * d2r
    dec0_r = dec0 * d2r

    cos_dec0 = math.cos(dec0_r)
    sin_dec0 = math.sin(dec0_r)

    denom = cos_dec0 - eta_r * sin_dec0
    ra = ra0_r + np.arctan2(xi_r, denom)
    dec = np.arctan2(
        (sin_dec0 + eta_r * cos_dec0) * np.cos(ra - ra0_r),
        denom,
    )
    return ra / d2r, dec / d2r


# -------------------------------------------------------------------
#  Proper-motion and parallax correction
# -------------------------------------------------------------------

def correct_proper_motion(
    ra: np.ndarray,
    dec: np.ndarray,
    pmra: np.ndarray,
    pmdec: np.ndarray,
    epoch_from: float,
    epoch_to: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply proper-motion correction to move sources between epochs.

    Parameters
    ----------
    ra, dec : np.ndarray
        Coordinates at *epoch_from* in degrees.
    pmra : np.ndarray
        Proper motion in RA (mas/yr, includes cos(dec) factor).
    pmdec : np.ndarray
        Proper motion in Dec (mas/yr).
    epoch_from : float
        Reference epoch (Julian year, e.g. 2016.0 for Gaia DR3).
    epoch_to : float
        Target epoch (Julian year).

    Returns
    -------
    ra_new, dec_new : np.ndarray
        Corrected coordinates in degrees.
    """
    ra = np.asarray(ra, dtype=np.float64)
    dec = np.asarray(dec, dtype=np.float64)
    pmra = np.asarray(pmra, dtype=np.float64)
    pmdec = np.asarray(pmdec, dtype=np.float64)

    dt = epoch_to - epoch_from  # years
    # mas/yr -> deg/yr:  /1000 -> arcsec, /3600 -> deg
    dra = pmra * dt / (ARCSEC_PER_DEG * 1000.0)
    ddec = pmdec * dt / (ARCSEC_PER_DEG * 1000.0)

    # pmra already includes cos(dec) so divide it out to get true RA shift
    cos_dec = np.cos(np.deg2rad(dec))
    cos_dec = np.where(cos_dec == 0, 1e-30, cos_dec)
    ra_new = ra + dra / cos_dec
    dec_new = dec + ddec

    return ra_new, dec_new


def correct_parallax(
    ra: np.ndarray,
    dec: np.ndarray,
    parallax: np.ndarray,
    observer_xyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply parallax correction for an observer's position.

    Parameters
    ----------
    ra, dec : np.ndarray
        Source coordinates in degrees.
    parallax : np.ndarray
        Parallax in milliarcseconds.
    observer_xyz : np.ndarray
        Observer's barycentric position in AU, shape ``(3,)``.

    Returns
    -------
    ra_corr, dec_corr : np.ndarray
        Parallax-corrected coordinates in degrees.
    """
    ra = np.asarray(ra, dtype=np.float64)
    dec = np.asarray(dec, dtype=np.float64)
    parallax = np.asarray(parallax, dtype=np.float64)
    observer_xyz = np.asarray(observer_xyz, dtype=np.float64)

    d2r = np.pi / 180.0
    ra_r = ra * d2r
    dec_r = dec * d2r

    plx_rad = parallax / (ARCSEC_PER_DEG * 1000.0) * d2r  # mas -> rad

    cos_ra = np.cos(ra_r)
    sin_ra = np.sin(ra_r)
    cos_dec = np.cos(dec_r)
    sin_dec = np.sin(dec_r)

    # Parallax shift in tangent plane (radians)
    ox, oy, oz = observer_xyz
    dra = plx_rad * (ox * sin_ra - oy * cos_ra) / cos_dec
    ddec = plx_rad * (ox * cos_ra * sin_dec + oy * sin_ra * sin_dec - oz * cos_dec)

    ra_corr = ra + np.rad2deg(dra)
    dec_corr = dec + np.rad2deg(ddec)
    return ra_corr, dec_corr


# -------------------------------------------------------------------
#  Astrometric quality assessment
# -------------------------------------------------------------------

def astrometric_residuals(
    detected_xy: np.ndarray,
    catalog_radec: np.ndarray,
    crpix: tuple[float, float],
    crval: tuple[float, float],
    cd_matrix: np.ndarray,
) -> dict[str, Any]:
    """Compute astrometric residuals for a WCS solution.

    Parameters
    ----------
    detected_xy : np.ndarray
        Detected pixel positions, shape ``(N, 2)``.
    catalog_radec : np.ndarray
        Matched catalog RA/Dec in degrees, shape ``(N, 2)``.
    crpix : tuple
        Reference pixel ``(x, y)``.
    crval : tuple
        Reference sky coordinate ``(ra, dec)`` in degrees.
    cd_matrix : np.ndarray
        2x2 CD matrix (degrees/pixel).

    Returns
    -------
    dict
        ``{'rms_arcsec': float, 'median_arcsec': float,
           'max_arcsec': float, 'n_sources': int,
           'residuals_arcsec': np.ndarray}``
    """
    if detected_xy.shape[0] != catalog_radec.shape[0]:
        raise ValueError("detected_xy and catalog_radec must have the same length.")

    crpix_arr = np.array(crpix)
    cd = np.asarray(cd_matrix, dtype=np.float64)

    # Predicted tangent-plane coords
    dxy = detected_xy - crpix_arr
    predicted_tp = dxy @ cd.T  # degrees

    # Catalog tangent-plane coords
    cat_xi, cat_eta = _radec_to_tangent(
        catalog_radec[:, 0], catalog_radec[:, 1],
        crval[0], crval[1],
    )
    cat_tp = np.column_stack([cat_xi, cat_eta])

    diff = (cat_tp - predicted_tp) * ARCSEC_PER_DEG
    sep = np.sqrt(np.sum(diff ** 2, axis=1))

    return {
        "rms_arcsec": float(np.sqrt(np.mean(sep ** 2))),
        "median_arcsec": float(np.median(sep)),
        "max_arcsec": float(np.max(sep)),
        "n_sources": len(sep),
        "residuals_arcsec": sep,
    }


def fit_distortion_sip(
    detected_xy: np.ndarray,
    catalog_radec: np.ndarray,
    crpix: tuple[float, float],
    crval: tuple[float, float],
    cd_matrix: np.ndarray,
    order: int = 3,
) -> dict[str, np.ndarray]:
    """Fit SIP distortion polynomial from matched star positions.

    Parameters
    ----------
    detected_xy : np.ndarray
        Pixel positions, shape ``(N, 2)``.
    catalog_radec : np.ndarray
        Catalog RA/Dec in degrees, shape ``(N, 2)``.
    crpix : tuple
        Reference pixel ``(x, y)``.
    crval : tuple
        Reference sky ``(ra, dec)`` in degrees.
    cd_matrix : np.ndarray
        2x2 CD matrix.
    order : int
        SIP polynomial order (2--6, default 3).

    Returns
    -------
    dict
        ``{'a_coeffs': np.ndarray, 'b_coeffs': np.ndarray,
           'order': int, 'residual_arcsec': float}``
    """
    if order < 2 or order > 6:
        raise ValueError("SIP order must be between 2 and 6.")
    if detected_xy.shape[0] < (order + 1) * (order + 2) // 2:
        raise ValueError("Not enough sources for SIP order {}.".format(order))

    crpix_arr = np.array(crpix)
    cd = np.asarray(cd_matrix, dtype=np.float64)
    cd_inv = np.linalg.inv(cd)

    # Catalog projected to tangent plane
    cat_xi, cat_eta = _radec_to_tangent(
        catalog_radec[:, 0], catalog_radec[:, 1],
        crval[0], crval[1],
    )
    cat_tp = np.column_stack([cat_xi, cat_eta])

    # Ideal pixel offsets from CD
    ideal = cat_tp @ cd_inv.T
    actual = detected_xy - crpix_arr
    dx = actual[:, 0] - ideal[:, 0]
    dy = actual[:, 1] - ideal[:, 1]

    # Build polynomial basis
    u = ideal[:, 0]
    v = ideal[:, 1]
    cols = []
    for p in range(2, order + 1):
        for q in range(p + 1):
            cols.append(u ** (p - q) * v ** q)
    basis = np.column_stack(cols) if cols else np.empty((len(u), 0))

    # Solve for A (x distortion) and B (y distortion)
    a_coeffs, _, _, _ = np.linalg.lstsq(basis, dx, rcond=None)
    b_coeffs, _, _, _ = np.linalg.lstsq(basis, dy, rcond=None)

    # Residual
    dx_pred = basis @ a_coeffs
    dy_pred = basis @ b_coeffs
    res = np.sqrt((dx - dx_pred) ** 2 + (dy - dy_pred) ** 2)
    res_arcsec = float(np.median(res)) * float(np.sqrt(cd[0, 0] ** 2 + cd[0, 1] ** 2)) * ARCSEC_PER_DEG

    return {
        "a_coeffs": a_coeffs,
        "b_coeffs": b_coeffs,
        "order": order,
        "residual_arcsec": res_arcsec,
    }
