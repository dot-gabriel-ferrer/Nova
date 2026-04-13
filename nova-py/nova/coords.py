"""Advanced coordinate transforms and distortion support for NOVA.

Extends the WCS representation layer in :mod:`nova.wcs` with concrete
coordinate math: SIP and TPV distortion corrections, lookup-table
distortions, celestial frame transforms (ICRS/Galactic/Ecliptic),
precession, and astrometric verification utilities.

All public functions accept both scalar and array inputs (converted
internally via ``np.asarray``) and require only NumPy at import time.
``scipy.interpolate`` and ``scipy.ndimage`` are imported lazily where
needed.

Categories
----------
- **SIP distortion**: Simple Imaging Polynomial forward / inverse / fit.
- **TPV distortion**: Tangent Polynomial (PV coefficients).
- **Lookup-table distortion**: IMAGE-type 2-D correction tables.
- **Frame transforms**: ICRS <-> Galactic, Ecliptic, FK5, precession.
- **TAN projection helpers**: gnomonic projection / deprojection.
- **Astrometric verification**: cross-matching and residual analysis.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from nova.constants import ARCSEC_PER_DEG


# ---------------------------------------------------------------------------
# Lazy optional imports
# ---------------------------------------------------------------------------

def _import_scipy_interpolate():  # noqa: D401
    """Lazily import *scipy.interpolate* with a clear error on failure."""
    try:
        import scipy.interpolate as interp
        return interp
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
# Internal constants (degrees <-> radians helpers)
# ---------------------------------------------------------------------------

_DEG2RAD: float = np.pi / 180.0
_RAD2DEG: float = 180.0 / np.pi


# ---------------------------------------------------------------------------
# TAN projection helpers
# ---------------------------------------------------------------------------

def tan_project(
    ra: float | np.ndarray,
    dec: float | np.ndarray,
    crval_ra: float,
    crval_dec: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Gnomonic (TAN) projection: sky to tangent-plane.

    Maps ``(ra, dec)`` to intermediate world coordinates ``(xi, eta)``
    using the standard FITS TAN projection centred at ``(crval_ra,
    crval_dec)``.

    Parameters
    ----------
    ra, dec : float or ndarray
        Sky coordinates in degrees.
    crval_ra, crval_dec : float
        Projection centre (CRVAL) in degrees.

    Returns
    -------
    xi, eta : ndarray
        Tangent-plane coordinates in degrees.
    """
    ra = np.asarray(ra, dtype=np.float64)
    dec = np.asarray(dec, dtype=np.float64)

    ra_r = ra * _DEG2RAD
    dec_r = dec * _DEG2RAD
    ra0_r = crval_ra * _DEG2RAD
    dec0_r = crval_dec * _DEG2RAD

    cos_dec = np.cos(dec_r)
    sin_dec = np.sin(dec_r)
    cos_dec0 = np.cos(dec0_r)
    sin_dec0 = np.sin(dec0_r)
    cos_dra = np.cos(ra_r - ra0_r)
    sin_dra = np.sin(ra_r - ra0_r)

    denom = sin_dec * sin_dec0 + cos_dec * cos_dec0 * cos_dra

    xi = (cos_dec * sin_dra / denom) * _RAD2DEG
    eta = ((sin_dec * cos_dec0 - cos_dec * sin_dec0 * cos_dra)
           / denom) * _RAD2DEG

    return xi, eta


def tan_deproject(
    xi: float | np.ndarray,
    eta: float | np.ndarray,
    crval_ra: float,
    crval_dec: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse gnomonic (TAN) deprojection: tangent-plane to sky.

    Parameters
    ----------
    xi, eta : float or ndarray
        Tangent-plane coordinates in degrees.
    crval_ra, crval_dec : float
        Projection centre (CRVAL) in degrees.

    Returns
    -------
    ra, dec : ndarray
        Sky coordinates in degrees.  ``ra`` is normalised to [0, 360).
    """
    xi = np.asarray(xi, dtype=np.float64)
    eta = np.asarray(eta, dtype=np.float64)

    xi_r = xi * _DEG2RAD
    eta_r = eta * _DEG2RAD
    ra0_r = crval_ra * _DEG2RAD
    dec0_r = crval_dec * _DEG2RAD

    cos_dec0 = np.cos(dec0_r)
    sin_dec0 = np.sin(dec0_r)

    rho = np.sqrt(xi_r**2 + eta_r**2)
    c = np.arctan(rho)

    cos_c = np.cos(c)
    sin_c = np.sin(c)

    # Handle the special case rho == 0
    safe_rho = np.where(rho == 0.0, 1.0, rho)

    dec_r = np.arcsin(cos_c * sin_dec0
                      + eta_r * sin_c * cos_dec0 / safe_rho)
    ra_r = ra0_r + np.arctan2(
        xi_r * sin_c,
        safe_rho * cos_dec0 * cos_c - eta_r * sin_dec0 * sin_c,
    )

    # When rho == 0, the point is the projection centre itself
    dec_r = np.where(rho == 0.0, dec0_r, dec_r)
    ra_r = np.where(rho == 0.0, ra0_r, ra_r)

    ra_deg = ra_r * _RAD2DEG % 360.0
    dec_deg = dec_r * _RAD2DEG

    return ra_deg, dec_deg


# ---------------------------------------------------------------------------
# SIP distortion
# ---------------------------------------------------------------------------

def _sip_polynomial(
    u: np.ndarray,
    v: np.ndarray,
    coeffs: dict[tuple[int, int], float],
    order: int,
) -> np.ndarray:
    """Evaluate a SIP polynomial sum_{p+q<=order} C[p,q] * u^p * v^q.

    Parameters
    ----------
    u, v : ndarray
        Intermediate pixel coordinates (relative to CRPIX).
    coeffs : dict
        Mapping ``(p, q) -> value``.
    order : int
        Maximum polynomial order.

    Returns
    -------
    ndarray
        Polynomial value at each position.
    """
    result = np.zeros_like(u, dtype=np.float64)
    for (p, q), value in coeffs.items():
        if p + q > order or p + q < 2:
            continue
        result = result + value * (u ** p) * (v ** q)
    return result


def sip_forward(
    u: float | np.ndarray,
    v: float | np.ndarray,
    a_coeffs: dict[tuple[int, int], float],
    b_coeffs: dict[tuple[int, int], float],
    a_order: int,
    b_order: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply SIP forward distortion correction.

    Transforms undistorted intermediate pixel coordinates ``(u, v)``
    (relative to CRPIX) to distorted coordinates::

        u' = u + f(u, v)
        v' = v + g(u, v)

    where *f* and *g* are defined by the A and B SIP coefficient matrices.

    Parameters
    ----------
    u, v : float or ndarray
        Undistorted pixel offsets from CRPIX.
    a_coeffs, b_coeffs : dict
        SIP polynomial coefficients mapping ``(p, q) -> value``.
    a_order, b_order : int
        Maximum polynomial orders for A and B polynomials.

    Returns
    -------
    u_corrected, v_corrected : ndarray
        Distorted pixel offsets.
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    du = _sip_polynomial(u, v, a_coeffs, a_order)
    dv = _sip_polynomial(u, v, b_coeffs, b_order)

    return u + du, v + dv


def sip_inverse(
    u_dist: float | np.ndarray,
    v_dist: float | np.ndarray,
    ap_coeffs: dict[tuple[int, int], float],
    bp_coeffs: dict[tuple[int, int], float],
    ap_order: int,
    bp_order: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply SIP inverse distortion (AP, BP polynomials).

    Transforms distorted pixel offsets back to undistorted offsets::

        u = u_dist + f'(u_dist, v_dist)
        v = v_dist + g'(u_dist, v_dist)

    Parameters
    ----------
    u_dist, v_dist : float or ndarray
        Distorted pixel offsets from CRPIX.
    ap_coeffs, bp_coeffs : dict
        SIP inverse polynomial coefficients ``(p, q) -> value``.
    ap_order, bp_order : int
        Maximum polynomial orders for AP and BP polynomials.

    Returns
    -------
    u_undistorted, v_undistorted : ndarray
        Undistorted pixel offsets.
    """
    u_dist = np.asarray(u_dist, dtype=np.float64)
    v_dist = np.asarray(v_dist, dtype=np.float64)

    du = _sip_polynomial(u_dist, v_dist, ap_coeffs, ap_order)
    dv = _sip_polynomial(u_dist, v_dist, bp_coeffs, bp_order)

    return u_dist + du, v_dist + dv


def fit_sip(
    pixel_coords: np.ndarray,
    world_coords: np.ndarray,
    crpix: tuple[float, float],
    cd_matrix: np.ndarray | list[list[float]],
    order: int = 3,
    n_iters: int = 5,
) -> dict[str, Any]:
    """Fit SIP distortion coefficients from matched coordinates.

    Uses iterative least-squares fitting to determine the A and B
    coefficient matrices from matched pixel and world coordinate pairs.

    Parameters
    ----------
    pixel_coords : ndarray, shape (N, 2)
        Pixel coordinates (x, y), 0-based.
    world_coords : ndarray, shape (N, 2)
        Corresponding world coordinates (ra, dec) in degrees.
    crpix : tuple of float
        Reference pixel (1-based FITS convention).
    cd_matrix : ndarray or list, shape (2, 2)
        CD matrix (degrees per pixel).
    order : int
        SIP polynomial order (default 3).
    n_iters : int
        Number of refinement iterations.

    Returns
    -------
    dict
        ``a_coeffs``, ``b_coeffs`` (dict ``(p,q)->value``),
        ``a_order``, ``b_order`` (int), and ``residual_rms`` (float,
        in pixels).
    """
    pixel_coords = np.asarray(pixel_coords, dtype=np.float64)
    world_coords = np.asarray(world_coords, dtype=np.float64)
    cd = np.asarray(cd_matrix, dtype=np.float64).reshape(2, 2)

    crpix_x, crpix_y = crpix

    # Pixel offsets from CRPIX (0-based pixel -> 1-based CRPIX)
    u0 = pixel_coords[:, 0] - (crpix_x - 1.0)
    v0 = pixel_coords[:, 1] - (crpix_y - 1.0)

    # Estimate CRVAL: the world coordinate at the reference pixel.
    # Use the linear WCS to extrapolate from data points back to CRPIX.
    xi_lin = cd[0, 0] * u0 + cd[0, 1] * v0
    eta_lin = cd[1, 0] * u0 + cd[1, 1] * v0
    crval_dec = float(np.median(world_coords[:, 1] - eta_lin))
    cos_crval_dec = np.cos(crval_dec * _DEG2RAD)
    if abs(cos_crval_dec) > 1e-12:
        crval_ra = float(np.median(
            world_coords[:, 0] - xi_lin / cos_crval_dec
        ))
    else:
        crval_ra = float(np.median(world_coords[:, 0]))

    # Project world coords to tangent plane
    xi, eta = tan_project(
        world_coords[:, 0], world_coords[:, 1], crval_ra, crval_dec,
    )

    # Invert CD matrix to get ideal pixel offsets from world offsets
    cd_inv = np.linalg.inv(cd)
    u_ideal = cd_inv[0, 0] * xi + cd_inv[0, 1] * eta
    v_ideal = cd_inv[1, 0] * xi + cd_inv[1, 1] * eta

    # Build design matrix including linear terms (p+q >= 0) so that
    # constant/linear offsets from slight CRVAL/CD errors are absorbed.
    # Only p+q >= 2 terms are retained as SIP coefficients.
    def _build_design_matrix(
        u: np.ndarray, v: np.ndarray,
    ) -> tuple[np.ndarray, list[tuple[int, int]]]:
        cols: list[np.ndarray] = []
        col_indices: list[tuple[int, int]] = []
        for p in range(order + 1):
            for q in range(order + 1 - p):
                cols.append((u ** p) * (v ** q))
                col_indices.append((p, q))
        if not cols:
            return np.empty((len(u), 0)), col_indices
        return np.column_stack(cols), col_indices

    # Target: u_ideal = u0 + SIP(u0), so SIP(u0) = u_ideal - u0
    du_target = u_ideal - u0
    dv_target = v_ideal - v0

    design, col_idx = _build_design_matrix(u0, v0)

    a_coeffs = {}
    b_coeffs = {}
    residual_rms = float("inf")

    if design.shape[1] > 0:
        a_sol, _, _, _ = np.linalg.lstsq(design, du_target, rcond=None)
        b_sol, _, _, _ = np.linalg.lstsq(design, dv_target, rcond=None)

        # Residual of the full model (including constant/linear terms)
        a_residual = du_target - design @ a_sol
        b_residual = dv_target - design @ b_sol
        residual_rms = float(np.sqrt(
            np.mean(a_residual**2 + b_residual**2)
        ))

        # Extract only SIP terms (p+q >= 2)
        for i, (p, q) in enumerate(col_idx):
            if p + q < 2:
                continue
            if abs(a_sol[i]) > 1e-16:
                a_coeffs[(p, q)] = float(a_sol[i])
            if abs(b_sol[i]) > 1e-16:
                b_coeffs[(p, q)] = float(b_sol[i])

    return {
        "a_coeffs": a_coeffs,
        "b_coeffs": b_coeffs,
        "a_order": order,
        "b_order": order,
        "residual_rms": residual_rms,
    }


def sip_pixel_to_world(
    pixel_x: float | np.ndarray,
    pixel_y: float | np.ndarray,
    crpix: tuple[float, float],
    cd_matrix: np.ndarray | list[list[float]],
    a_coeffs: dict[tuple[int, int], float],
    b_coeffs: dict[tuple[int, int], float],
    crval: tuple[float, float],
    projection: str = "TAN",
) -> tuple[np.ndarray, np.ndarray]:
    """Full pixel-to-world transform with SIP distortion.

    Pipeline: pixel offsets -> SIP forward -> CD matrix -> TAN
    deprojection -> sky coordinates.

    Parameters
    ----------
    pixel_x, pixel_y : float or ndarray
        Pixel coordinates (0-based).
    crpix : tuple of float
        Reference pixel (1-based FITS convention).
    cd_matrix : ndarray or list, shape (2, 2)
        CD matrix (degrees per pixel).
    a_coeffs, b_coeffs : dict
        SIP A and B polynomial coefficients.
    crval : tuple of float
        (CRVAL1, CRVAL2) in degrees.
    projection : str
        Projection type (only ``"TAN"`` is supported).

    Returns
    -------
    ra, dec : ndarray
        Sky coordinates in degrees.

    Raises
    ------
    ValueError
        If an unsupported projection is requested.
    """
    if projection != "TAN":
        raise ValueError(f"Unsupported projection: {projection!r}")

    pixel_x = np.asarray(pixel_x, dtype=np.float64)
    pixel_y = np.asarray(pixel_y, dtype=np.float64)
    cd = np.asarray(cd_matrix, dtype=np.float64).reshape(2, 2)

    # Pixel offsets from CRPIX (0-based pixel, 1-based CRPIX)
    u = pixel_x - (crpix[0] - 1.0)
    v = pixel_y - (crpix[1] - 1.0)

    # Apply SIP forward distortion
    a_order = max((p + q for (p, q) in a_coeffs), default=0)
    b_order = max((p + q for (p, q) in b_coeffs), default=0)
    u_sip, v_sip = sip_forward(u, v, a_coeffs, b_coeffs, a_order, b_order)

    # CD matrix: pixel offsets -> intermediate world coords (degrees)
    xi = cd[0, 0] * u_sip + cd[0, 1] * v_sip
    eta = cd[1, 0] * u_sip + cd[1, 1] * v_sip

    # TAN deprojection
    ra, dec = tan_deproject(xi, eta, crval[0], crval[1])
    return ra, dec


def sip_world_to_pixel(
    ra: float | np.ndarray,
    dec: float | np.ndarray,
    crpix: tuple[float, float],
    cd_matrix: np.ndarray | list[list[float]],
    ap_coeffs: dict[tuple[int, int], float],
    bp_coeffs: dict[tuple[int, int], float],
    crval: tuple[float, float],
    projection: str = "TAN",
) -> tuple[np.ndarray, np.ndarray]:
    """Full world-to-pixel transform with SIP inverse distortion.

    Pipeline: sky -> TAN projection -> CD inverse -> SIP inverse ->
    pixel coordinates.

    Parameters
    ----------
    ra, dec : float or ndarray
        Sky coordinates in degrees.
    crpix : tuple of float
        Reference pixel (1-based FITS convention).
    cd_matrix : ndarray or list, shape (2, 2)
        CD matrix (degrees per pixel).
    ap_coeffs, bp_coeffs : dict
        SIP inverse (AP, BP) polynomial coefficients.
    crval : tuple of float
        (CRVAL1, CRVAL2) in degrees.
    projection : str
        Projection type (only ``"TAN"`` is supported).

    Returns
    -------
    pixel_x, pixel_y : ndarray
        Pixel coordinates (0-based).

    Raises
    ------
    ValueError
        If an unsupported projection is requested.
    """
    if projection != "TAN":
        raise ValueError(f"Unsupported projection: {projection!r}")

    cd = np.asarray(cd_matrix, dtype=np.float64).reshape(2, 2)

    # TAN projection: sky -> intermediate world coords
    xi, eta = tan_project(ra, dec, crval[0], crval[1])

    # Invert CD matrix: world -> distorted pixel offsets
    cd_inv = np.linalg.inv(cd)
    u_dist = cd_inv[0, 0] * xi + cd_inv[0, 1] * eta
    v_dist = cd_inv[1, 0] * xi + cd_inv[1, 1] * eta

    # Apply SIP inverse (AP, BP)
    ap_order = max((p + q for (p, q) in ap_coeffs), default=0)
    bp_order = max((p + q for (p, q) in bp_coeffs), default=0)
    u, v = sip_inverse(u_dist, v_dist, ap_coeffs, bp_coeffs,
                       ap_order, bp_order)

    # Convert back to pixel coordinates (0-based)
    pixel_x = u + (crpix[0] - 1.0)
    pixel_y = v + (crpix[1] - 1.0)
    return pixel_x, pixel_y


# ---------------------------------------------------------------------------
# TPV distortion
# ---------------------------------------------------------------------------

# Standard TPV polynomial term expansion: index -> (powers of xi, eta).
# Follows the FITS convention for PVi_j (j = 0..39).
_TPV_TERMS: dict[int, tuple[int, int]] = {
    0: (0, 0),
    1: (1, 0),
    2: (0, 1),
    3: (2, 0),
    4: (1, 1),
    5: (0, 2),
    6: (3, 0),
    7: (2, 1),
    8: (1, 2),
    9: (0, 3),
    10: (4, 0),
    11: (3, 1),
    12: (2, 2),
    13: (1, 3),
    14: (0, 4),
    15: (5, 0),
    16: (4, 1),
    17: (3, 2),
    18: (2, 3),
    19: (1, 4),
    20: (0, 5),
    21: (6, 0),
    22: (5, 1),
    23: (4, 2),
    24: (3, 3),
    25: (2, 4),
    26: (1, 5),
    27: (0, 6),
    28: (7, 0),
    29: (6, 1),
    30: (5, 2),
    31: (4, 3),
    32: (3, 4),
    33: (2, 5),
    34: (1, 6),
    35: (0, 7),
    36: (8, 0),
    37: (7, 1),
    38: (6, 2),
    39: (5, 3),
}


def _tpv_evaluate(
    x: np.ndarray,
    y: np.ndarray,
    pv_coeffs: dict[int, float],
) -> np.ndarray:
    """Evaluate a TPV polynomial.

    Parameters
    ----------
    x, y : ndarray
        Input coordinates (xi or eta direction).
    pv_coeffs : dict
        Mapping ``{index: coefficient}`` for PV terms.

    Returns
    -------
    ndarray
        Evaluated polynomial.
    """
    result = np.zeros_like(x, dtype=np.float64)
    for idx, coeff in pv_coeffs.items():
        if idx not in _TPV_TERMS:
            continue
        px, py = _TPV_TERMS[idx]
        result = result + coeff * (x ** px) * (y ** py)
    return result


def tpv_forward(
    xi: float | np.ndarray,
    eta: float | np.ndarray,
    pv1_coeffs: dict[int, float],
    pv2_coeffs: dict[int, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Apply TPV polynomial distortion.

    The TPV convention applies polynomial corrections to intermediate
    pixel coordinates ``(xi, eta)`` using ``PV1_j`` and ``PV2_j``
    coefficients.

    Parameters
    ----------
    xi, eta : float or ndarray
        Intermediate pixel coordinates (CD-matrix-applied offsets from
        CRPIX, in degrees).
    pv1_coeffs, pv2_coeffs : dict
        TPV polynomial coefficients ``{index: value}`` for axes 1 and 2.

    Returns
    -------
    xi_corr, eta_corr : ndarray
        Distortion-corrected intermediate coordinates (degrees).
    """
    xi = np.asarray(xi, dtype=np.float64)
    eta = np.asarray(eta, dtype=np.float64)

    xi_corr = _tpv_evaluate(xi, eta, pv1_coeffs)
    eta_corr = _tpv_evaluate(eta, xi, pv2_coeffs)

    return xi_corr, eta_corr


def tpv_pixel_to_world(
    pixel_x: float | np.ndarray,
    pixel_y: float | np.ndarray,
    crpix: tuple[float, float],
    cd_matrix: np.ndarray | list[list[float]],
    pv1_coeffs: dict[int, float],
    pv2_coeffs: dict[int, float],
    crval: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Full pixel-to-world transform with TPV distortion.

    Pipeline: pixel offsets -> CD matrix -> TPV polynomial -> TAN
    deprojection.

    Parameters
    ----------
    pixel_x, pixel_y : float or ndarray
        Pixel coordinates (0-based).
    crpix : tuple of float
        Reference pixel (1-based FITS convention).
    cd_matrix : ndarray or list, shape (2, 2)
        CD matrix (degrees per pixel).
    pv1_coeffs, pv2_coeffs : dict
        TPV polynomial coefficients for axes 1 and 2.
    crval : tuple of float
        (CRVAL1, CRVAL2) in degrees.

    Returns
    -------
    ra, dec : ndarray
        Sky coordinates in degrees.
    """
    pixel_x = np.asarray(pixel_x, dtype=np.float64)
    pixel_y = np.asarray(pixel_y, dtype=np.float64)
    cd = np.asarray(cd_matrix, dtype=np.float64).reshape(2, 2)

    # Pixel offsets from CRPIX
    u = pixel_x - (crpix[0] - 1.0)
    v = pixel_y - (crpix[1] - 1.0)

    # CD matrix: pixel offsets -> intermediate world coordinates
    xi_lin = cd[0, 0] * u + cd[0, 1] * v
    eta_lin = cd[1, 0] * u + cd[1, 1] * v

    # Apply TPV polynomial distortion
    xi_corr, eta_corr = tpv_forward(xi_lin, eta_lin, pv1_coeffs, pv2_coeffs)

    # TAN deprojection
    return tan_deproject(xi_corr, eta_corr, crval[0], crval[1])


def tpv_world_to_pixel(
    ra: float | np.ndarray,
    dec: float | np.ndarray,
    crpix: tuple[float, float],
    cd_matrix: np.ndarray | list[list[float]],
    pv1_coeffs: dict[int, float],
    pv2_coeffs: dict[int, float],
    crval: tuple[float, float],
    n_iters: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Full world-to-pixel transform with TPV (iterative inversion).

    Since TPV distortion is not analytically invertible, this function
    uses fixed-point iteration starting from the linear (no-distortion)
    solution.

    Parameters
    ----------
    ra, dec : float or ndarray
        Sky coordinates in degrees.
    crpix : tuple of float
        Reference pixel (1-based FITS convention).
    cd_matrix : ndarray or list, shape (2, 2)
        CD matrix (degrees per pixel).
    pv1_coeffs, pv2_coeffs : dict
        TPV polynomial coefficients for axes 1 and 2.
    crval : tuple of float
        (CRVAL1, CRVAL2) in degrees.
    n_iters : int
        Number of iterations for convergence (default 20).

    Returns
    -------
    pixel_x, pixel_y : ndarray
        Pixel coordinates (0-based).
    """
    cd = np.asarray(cd_matrix, dtype=np.float64).reshape(2, 2)
    cd_inv = np.linalg.inv(cd)

    # TAN projection: sky -> target intermediate world coordinates
    xi_target, eta_target = tan_project(ra, dec, crval[0], crval[1])

    # Initial guess: linear (no distortion) solution
    u = cd_inv[0, 0] * xi_target + cd_inv[0, 1] * eta_target
    v = cd_inv[1, 0] * xi_target + cd_inv[1, 1] * eta_target

    # Fixed-point iteration
    for _ in range(n_iters):
        xi_lin = cd[0, 0] * u + cd[0, 1] * v
        eta_lin = cd[1, 0] * u + cd[1, 1] * v

        xi_corr, eta_corr = tpv_forward(xi_lin, eta_lin,
                                         pv1_coeffs, pv2_coeffs)

        # Residual in intermediate world coordinates
        dxi = xi_target - xi_corr
        deta = eta_target - eta_corr

        # Map residual back to pixel space and correct
        u = u + cd_inv[0, 0] * dxi + cd_inv[0, 1] * deta
        v = v + cd_inv[1, 0] * dxi + cd_inv[1, 1] * deta

    pixel_x = u + (crpix[0] - 1.0)
    pixel_y = v + (crpix[1] - 1.0)
    return pixel_x, pixel_y


# ---------------------------------------------------------------------------
# Lookup-table distortion
# ---------------------------------------------------------------------------

def lookup_distortion(
    pixel_x: float | np.ndarray,
    pixel_y: float | np.ndarray,
    distortion_table_x: np.ndarray,
    distortion_table_y: np.ndarray,
    crpix_table: tuple[float, float] = (1.0, 1.0),
    cdelt_table: tuple[float, float] = (1.0, 1.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Look up IMAGE-type distortion corrections from 2-D tables.

    Bilinearly interpolates the correction tables at the given pixel
    positions to return ``(dx, dy)`` offsets.

    Parameters
    ----------
    pixel_x, pixel_y : float or ndarray
        Pixel coordinates (0-based).
    distortion_table_x, distortion_table_y : ndarray, 2-D
        Distortion lookup tables for x and y corrections.
    crpix_table : tuple of float
        Reference pixel of the distortion table grid (1-based).
    cdelt_table : tuple of float
        Pixel step size for the distortion table grid.

    Returns
    -------
    dx, dy : ndarray
        Pixel corrections at each input position.
    """
    ndi = _import_scipy_ndimage()

    pixel_x = np.asarray(pixel_x, dtype=np.float64)
    pixel_y = np.asarray(pixel_y, dtype=np.float64)

    # Map pixel coordinates to table grid indices
    # Table grid index = (pixel - crpix_table + 1) / cdelt_table
    tab_x = (pixel_x - (crpix_table[0] - 1.0)) / cdelt_table[0]
    tab_y = (pixel_y - (crpix_table[1] - 1.0)) / cdelt_table[1]

    # Flatten for map_coordinates then reshape back
    orig_shape = pixel_x.shape
    coords = np.array([tab_y.ravel(), tab_x.ravel()])

    dx = ndi.map_coordinates(
        np.asarray(distortion_table_x, dtype=np.float64),
        coords, order=1, mode="nearest",
    ).reshape(orig_shape)

    dy = ndi.map_coordinates(
        np.asarray(distortion_table_y, dtype=np.float64),
        coords, order=1, mode="nearest",
    ).reshape(orig_shape)

    return dx, dy


def apply_lookup_correction(
    pixel_x: float | np.ndarray,
    pixel_y: float | np.ndarray,
    distortion_table_x: np.ndarray,
    distortion_table_y: np.ndarray,
    crpix_table: tuple[float, float] = (1.0, 1.0),
    cdelt_table: tuple[float, float] = (1.0, 1.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Apply lookup-table distortion correction to pixel positions.

    Parameters
    ----------
    pixel_x, pixel_y : float or ndarray
        Original pixel coordinates (0-based).
    distortion_table_x, distortion_table_y : ndarray, 2-D
        Distortion lookup tables.
    crpix_table : tuple of float
        Reference pixel of the table grid (1-based).
    cdelt_table : tuple of float
        Pixel step size for the table grid.

    Returns
    -------
    corrected_x, corrected_y : ndarray
        Corrected pixel positions.
    """
    pixel_x = np.asarray(pixel_x, dtype=np.float64)
    pixel_y = np.asarray(pixel_y, dtype=np.float64)

    dx, dy = lookup_distortion(
        pixel_x, pixel_y,
        distortion_table_x, distortion_table_y,
        crpix_table, cdelt_table,
    )
    return pixel_x + dx, pixel_y + dy


# ---------------------------------------------------------------------------
# Frame transforms -- rotation matrix helpers
# ---------------------------------------------------------------------------

def _rotation_matrix_z(angle_rad: float) -> np.ndarray:
    """Rotation matrix around the z-axis."""
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([
        [c, s, 0.0],
        [-s, c, 0.0],
        [0.0, 0.0, 1.0],
    ])


def _rotation_matrix_y(angle_rad: float) -> np.ndarray:
    """Rotation matrix around the y-axis."""
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([
        [c, 0.0, -s],
        [0.0, 1.0, 0.0],
        [s, 0.0, c],
    ])


def _rotation_matrix_x(angle_rad: float) -> np.ndarray:
    """Rotation matrix around the x-axis."""
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, c, s],
        [0.0, -s, c],
    ])


def _radec_to_xyz(
    ra: np.ndarray,
    dec: np.ndarray,
) -> np.ndarray:
    """Convert (ra, dec) in radians to unit Cartesian vectors.

    Returns
    -------
    ndarray, shape (..., 3)
        Cartesian unit vectors.
    """
    cos_dec = np.cos(dec)
    return np.stack([
        cos_dec * np.cos(ra),
        cos_dec * np.sin(ra),
        np.sin(dec),
    ], axis=-1)


def _xyz_to_radec(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert Cartesian vectors to (ra, dec) in radians.

    Returns
    -------
    ra, dec : ndarray
        Spherical coordinates in radians.  ``ra`` is in [0, 2*pi).
    """
    ra = np.arctan2(xyz[..., 1], xyz[..., 0]) % (2.0 * np.pi)
    dec = np.arcsin(np.clip(xyz[..., 2], -1.0, 1.0))
    return ra, dec


# ---------------------------------------------------------------------------
# Frame transforms -- Galactic <-> Equatorial (ICRS)
# ---------------------------------------------------------------------------

# J2000 North Galactic Pole in ICRS (Hipparcos values, IAU 1958 definition
# refined by Reid & Brunthaler 2004):
#   alpha_NGP = 192.85948 deg,  delta_NGP = 27.12825 deg
#   l_NCP     = 122.93192 deg  (galactic longitude of celestial pole)
_ALPHA_NGP_RAD: float = 192.85948 * _DEG2RAD
_DELTA_NGP_RAD: float = 27.12825 * _DEG2RAD
_L_NCP_RAD: float = 122.93192 * _DEG2RAD

# Pre-compute the equatorial-to-galactic rotation matrix.
# Uses the well-known Hipparcos values (ESA, 1997, Vol. 1, Eq. 1.5.11).
_R_EQ_TO_GAL: np.ndarray = np.array([
    [-0.0548755604, -0.8734370902, -0.4838350155],
    [+0.4941094279, -0.4448296300, +0.7469822445],
    [-0.8676661490, -0.1980763734, +0.4559837762],
])

_R_GAL_TO_EQ: np.ndarray = _R_EQ_TO_GAL.T


def equatorial_to_galactic(
    ra: float | np.ndarray,
    dec: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert equatorial (ICRS) to Galactic coordinates.

    Parameters
    ----------
    ra, dec : float or ndarray
        Equatorial coordinates in degrees.

    Returns
    -------
    l, b : ndarray
        Galactic longitude and latitude in degrees.
    """
    ra_r = np.asarray(ra, dtype=np.float64) * _DEG2RAD
    dec_r = np.asarray(dec, dtype=np.float64) * _DEG2RAD

    xyz_eq = _radec_to_xyz(ra_r, dec_r)
    xyz_gal = xyz_eq @ _R_EQ_TO_GAL.T

    l_rad, b_rad = _xyz_to_radec(xyz_gal)
    return l_rad * _RAD2DEG, b_rad * _RAD2DEG


def galactic_to_equatorial(
    l: float | np.ndarray,
    b: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert Galactic to equatorial (ICRS) coordinates.

    Parameters
    ----------
    l, b : float or ndarray
        Galactic longitude and latitude in degrees.

    Returns
    -------
    ra, dec : ndarray
        Equatorial coordinates in degrees.  ``ra`` is in [0, 360).
    """
    l_rad = np.asarray(l, dtype=np.float64) * _DEG2RAD
    b_rad = np.asarray(b, dtype=np.float64) * _DEG2RAD

    xyz_gal = _radec_to_xyz(l_rad, b_rad)
    xyz_eq = xyz_gal @ _R_GAL_TO_EQ.T

    ra_rad, dec_rad = _xyz_to_radec(xyz_eq)
    return ra_rad * _RAD2DEG, dec_rad * _RAD2DEG


# ---------------------------------------------------------------------------
# Frame transforms -- Ecliptic <-> Equatorial
# ---------------------------------------------------------------------------

def equatorial_to_ecliptic(
    ra: float | np.ndarray,
    dec: float | np.ndarray,
    obliquity: float = 23.439281,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert equatorial to ecliptic coordinates.

    Parameters
    ----------
    ra, dec : float or ndarray
        Equatorial coordinates in degrees.
    obliquity : float
        Obliquity of the ecliptic in degrees (default: J2000 mean
        obliquity 23.439281 deg).

    Returns
    -------
    lon, lat : ndarray
        Ecliptic longitude and latitude in degrees.
    """
    ra_r = np.asarray(ra, dtype=np.float64) * _DEG2RAD
    dec_r = np.asarray(dec, dtype=np.float64) * _DEG2RAD
    eps = obliquity * _DEG2RAD

    cos_eps = np.cos(eps)
    sin_eps = np.sin(eps)

    sin_dec = np.sin(dec_r)
    cos_dec = np.cos(dec_r)
    sin_ra = np.sin(ra_r)
    cos_ra = np.cos(ra_r)

    # Ecliptic latitude
    sin_lat = sin_dec * cos_eps - cos_dec * sin_eps * sin_ra
    lat = np.arcsin(np.clip(sin_lat, -1.0, 1.0))

    # Ecliptic longitude
    lon = np.arctan2(
        sin_ra * cos_eps + np.tan(dec_r) * sin_eps,
        cos_ra,
    ) % (2.0 * np.pi)

    return lon * _RAD2DEG, lat * _RAD2DEG


def ecliptic_to_equatorial(
    lon: float | np.ndarray,
    lat: float | np.ndarray,
    obliquity: float = 23.439281,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert ecliptic to equatorial coordinates.

    Parameters
    ----------
    lon, lat : float or ndarray
        Ecliptic longitude and latitude in degrees.
    obliquity : float
        Obliquity of the ecliptic in degrees.

    Returns
    -------
    ra, dec : ndarray
        Equatorial coordinates in degrees.  ``ra`` is in [0, 360).
    """
    lon_r = np.asarray(lon, dtype=np.float64) * _DEG2RAD
    lat_r = np.asarray(lat, dtype=np.float64) * _DEG2RAD
    eps = obliquity * _DEG2RAD

    cos_eps = np.cos(eps)
    sin_eps = np.sin(eps)

    sin_lat = np.sin(lat_r)
    cos_lat = np.cos(lat_r)
    sin_lon = np.sin(lon_r)
    cos_lon = np.cos(lon_r)

    sin_dec = sin_lat * cos_eps + cos_lat * sin_eps * sin_lon
    dec = np.arcsin(np.clip(sin_dec, -1.0, 1.0))

    ra = np.arctan2(
        sin_lon * cos_eps - np.tan(lat_r) * sin_eps,
        cos_lon,
    ) % (2.0 * np.pi)

    return ra * _RAD2DEG, dec * _RAD2DEG


# ---------------------------------------------------------------------------
# Frame transforms -- FK5 <-> ICRS
# ---------------------------------------------------------------------------

def fk5_to_icrs(
    ra: float | np.ndarray,
    dec: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert FK5 (J2000) to ICRS coordinates.

    For J2000 the FK5 and ICRS frames are aligned to better than
    0.1 arcsec (the residual frame rotation is at the milli-arcsecond
    level).  This implementation returns the input unchanged, which is
    suitable for all practical wide-field work.

    Parameters
    ----------
    ra, dec : float or ndarray
        FK5 J2000 coordinates in degrees.

    Returns
    -------
    ra_icrs, dec_icrs : ndarray
        ICRS coordinates in degrees.
    """
    return (
        np.asarray(ra, dtype=np.float64).copy(),
        np.asarray(dec, dtype=np.float64).copy(),
    )


# ---------------------------------------------------------------------------
# Frame transforms -- Precession
# ---------------------------------------------------------------------------

def _precession_matrix_from_j2000(epoch: float) -> np.ndarray:
    """Build the precession rotation matrix from J2000.0 to *epoch*.

    Uses the IAU 2006 precession angles (Lieske parameterization).

    Parameters
    ----------
    epoch : float
        Target Julian year.

    Returns
    -------
    ndarray, shape (3, 3)
        Rotation matrix.
    """
    t = (epoch - 2000.0) / 100.0  # Julian centuries from J2000

    # IAU 2006 precession angles in arcseconds
    zeta_a = 2.650545 * t + 2306.083227 * t**2 + 1.0946788 * t**3
    z_a = -2.650545 * t + 2306.077181 * t**2 + 1.0927348 * t**3
    theta_a = 2004.191903 * t**2 - 0.4294934 * t**3

    zeta = zeta_a / ARCSEC_PER_DEG * _DEG2RAD
    z = z_a / ARCSEC_PER_DEG * _DEG2RAD
    theta = theta_a / ARCSEC_PER_DEG * _DEG2RAD

    cos_z = np.cos(z)
    sin_z = np.sin(z)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    cos_ze = np.cos(zeta)
    sin_ze = np.sin(zeta)

    return np.array([
        [cos_z * cos_t * cos_ze - sin_z * sin_ze,
         -cos_z * cos_t * sin_ze - sin_z * cos_ze,
         -cos_z * sin_t],
        [sin_z * cos_t * cos_ze + cos_z * sin_ze,
         -sin_z * cos_t * sin_ze + cos_z * cos_ze,
         -sin_z * sin_t],
        [sin_t * cos_ze,
         -sin_t * sin_ze,
         cos_t],
    ])


def precess(
    ra: float | np.ndarray,
    dec: float | np.ndarray,
    epoch_from: float,
    epoch_to: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Precess coordinates between Julian epochs.

    Uses the IAU 2006 precession model (Lieske parameterization).
    Internally composes precession matrices through J2000.0 so that
    round-tripping between arbitrary epochs is exact to numerical
    precision.

    Parameters
    ----------
    ra, dec : float or ndarray
        Input coordinates in degrees.
    epoch_from : float
        Source epoch as a Julian year (e.g. 2000.0).
    epoch_to : float
        Target epoch as a Julian year (e.g. 2050.0).

    Returns
    -------
    ra_new, dec_new : ndarray
        Precessed coordinates in degrees.
    """
    ra_r = np.asarray(ra, dtype=np.float64) * _DEG2RAD
    dec_r = np.asarray(dec, dtype=np.float64) * _DEG2RAD

    # Combined matrix: epoch_from -> J2000 -> epoch_to
    #   M_from takes J2000 -> epoch_from, so M_from^T goes back.
    #   M_to   takes J2000 -> epoch_to.
    m_from = _precession_matrix_from_j2000(epoch_from)
    m_to = _precession_matrix_from_j2000(epoch_to)
    rot = m_to @ m_from.T

    cos_dec = np.cos(dec_r)
    x = cos_dec * np.cos(ra_r)
    y = cos_dec * np.sin(ra_r)
    z_vec = np.sin(dec_r)

    x2 = rot[0, 0] * x + rot[0, 1] * y + rot[0, 2] * z_vec
    y2 = rot[1, 0] * x + rot[1, 1] * y + rot[1, 2] * z_vec
    z2 = rot[2, 0] * x + rot[2, 1] * y + rot[2, 2] * z_vec

    ra_new = np.arctan2(y2, x2) % (2.0 * np.pi)
    dec_new = np.arcsin(np.clip(z2, -1.0, 1.0))

    return ra_new * _RAD2DEG, dec_new * _RAD2DEG


# ---------------------------------------------------------------------------
# Angular separation (Vincenty formula)
# ---------------------------------------------------------------------------

def angular_separation(
    ra1: float | np.ndarray,
    dec1: float | np.ndarray,
    ra2: float | np.ndarray,
    dec2: float | np.ndarray,
) -> np.ndarray:
    """Compute angular separation using the Vincenty formula.

    The Vincenty formula is numerically stable for both small and large
    separations, unlike the simpler cosine formula.

    Parameters
    ----------
    ra1, dec1 : float or ndarray
        Coordinates of first position(s) in degrees.
    ra2, dec2 : float or ndarray
        Coordinates of second position(s) in degrees.

    Returns
    -------
    ndarray
        Angular separation in degrees.
    """
    ra1_r = np.asarray(ra1, dtype=np.float64) * _DEG2RAD
    dec1_r = np.asarray(dec1, dtype=np.float64) * _DEG2RAD
    ra2_r = np.asarray(ra2, dtype=np.float64) * _DEG2RAD
    dec2_r = np.asarray(dec2, dtype=np.float64) * _DEG2RAD

    dra = ra2_r - ra1_r
    cos_dec1 = np.cos(dec1_r)
    sin_dec1 = np.sin(dec1_r)
    cos_dec2 = np.cos(dec2_r)
    sin_dec2 = np.sin(dec2_r)

    # Vincenty numerator
    term1 = cos_dec2 * np.sin(dra)
    term2 = cos_dec1 * sin_dec2 - sin_dec1 * cos_dec2 * np.cos(dra)
    num = np.sqrt(term1**2 + term2**2)

    # Vincenty denominator
    den = sin_dec1 * sin_dec2 + cos_dec1 * cos_dec2 * np.cos(dra)

    return np.arctan2(num, den) * _RAD2DEG


# ---------------------------------------------------------------------------
# Astrometric verification
# ---------------------------------------------------------------------------

def cross_match(
    ra1: np.ndarray,
    dec1: np.ndarray,
    ra2: np.ndarray,
    dec2: np.ndarray,
    radius: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cross-match two sky catalogues within a search radius.

    For each source in catalogue 1, finds the nearest source in catalogue
    2 within *radius* arcseconds.  Uses a brute-force angular separation
    search which is practical for catalogues up to ~10**4 sources.

    Parameters
    ----------
    ra1, dec1 : ndarray, shape (N,)
        Coordinates of catalogue 1 in degrees.
    ra2, dec2 : ndarray, shape (M,)
        Coordinates of catalogue 2 in degrees.
    radius : float
        Maximum match distance in arcseconds.

    Returns
    -------
    idx1 : ndarray of int
        Indices into catalogue 1 for matched pairs.
    idx2 : ndarray of int
        Indices into catalogue 2 for matched pairs.
    separations : ndarray
        Angular separations of matched pairs in arcseconds.
    """
    ra1 = np.asarray(ra1, dtype=np.float64)
    dec1 = np.asarray(dec1, dtype=np.float64)
    ra2 = np.asarray(ra2, dtype=np.float64)
    dec2 = np.asarray(dec2, dtype=np.float64)

    radius_deg = radius / ARCSEC_PER_DEG

    idx1_list: list[int] = []
    idx2_list: list[int] = []
    sep_list: list[float] = []

    for i in range(ra1.size):
        seps = angular_separation(ra1[i], dec1[i], ra2, dec2)
        best = np.argmin(seps)
        if seps[best] <= radius_deg:
            idx1_list.append(i)
            idx2_list.append(int(best))
            sep_list.append(float(seps[best]) * ARCSEC_PER_DEG)

    return (
        np.array(idx1_list, dtype=np.intp),
        np.array(idx2_list, dtype=np.intp),
        np.array(sep_list, dtype=np.float64),
    )


def compute_astrometric_residuals(
    measured_ra: np.ndarray,
    measured_dec: np.ndarray,
    catalog_ra: np.ndarray,
    catalog_dec: np.ndarray,
    match_radius: float = 2.0,
) -> dict[str, Any]:
    """Match sources and compute astrometric residuals.

    Cross-matches measured positions against a reference catalogue and
    returns statistical summaries of the positional offsets.

    Parameters
    ----------
    measured_ra, measured_dec : ndarray
        Measured source positions in degrees.
    catalog_ra, catalog_dec : ndarray
        Reference catalogue positions in degrees.
    match_radius : float
        Maximum match distance in arcseconds.

    Returns
    -------
    dict
        Keys:

        - ``n_matched`` : int -- number of matched sources.
        - ``median_offset_ra`` : float -- median RA offset (arcsec).
        - ``median_offset_dec`` : float -- median Dec offset (arcsec).
        - ``rms_ra`` : float -- RMS of RA offsets (arcsec).
        - ``rms_dec`` : float -- RMS of Dec offsets (arcsec).
        - ``rms_total`` : float -- combined RMS (arcsec).
    """
    idx1, idx2, _ = cross_match(
        measured_ra, measured_dec,
        catalog_ra, catalog_dec,
        radius=match_radius,
    )

    if idx1.size == 0:
        return {
            "n_matched": 0,
            "median_offset_ra": float("nan"),
            "median_offset_dec": float("nan"),
            "rms_ra": float("nan"),
            "rms_dec": float("nan"),
            "rms_total": float("nan"),
        }

    m_ra = np.asarray(measured_ra, dtype=np.float64)[idx1]
    m_dec = np.asarray(measured_dec, dtype=np.float64)[idx1]
    c_ra = np.asarray(catalog_ra, dtype=np.float64)[idx2]
    c_dec = np.asarray(catalog_dec, dtype=np.float64)[idx2]

    # RA offset includes cos(dec) correction
    dra = (m_ra - c_ra) * np.cos(np.radians(m_dec)) * ARCSEC_PER_DEG
    ddec = (m_dec - c_dec) * ARCSEC_PER_DEG

    rms_ra = float(np.sqrt(np.mean(dra**2)))
    rms_dec = float(np.sqrt(np.mean(ddec**2)))
    rms_total = float(np.sqrt(np.mean(dra**2 + ddec**2)))

    return {
        "n_matched": int(idx1.size),
        "median_offset_ra": float(np.median(dra)),
        "median_offset_dec": float(np.median(ddec)),
        "rms_ra": rms_ra,
        "rms_dec": rms_dec,
        "rms_total": rms_total,
    }


def verify_wcs_solution(
    image: np.ndarray,
    wcs_params: dict[str, Any],
    catalog_ra: np.ndarray,
    catalog_dec: np.ndarray,
    catalog_mag: np.ndarray | None = None,
    detection_sigma: float = 5.0,
) -> dict[str, Any]:
    """Full WCS verification pipeline.

    Detects sources in *image*, computes centroids, transforms to sky
    coordinates using the supplied WCS parameters, cross-matches against
    the reference catalogue, and reports astrometric residuals.

    Parameters
    ----------
    image : ndarray, 2-D
        Science image.
    wcs_params : dict
        WCS parameters with keys ``crpix`` (tuple), ``cd_matrix``
        (2x2 list/array), ``crval`` (tuple), and optionally
        ``a_coeffs``, ``b_coeffs`` for SIP distortion.
    catalog_ra, catalog_dec : ndarray
        Reference catalogue positions in degrees.
    catalog_mag : ndarray or None
        Reference magnitudes (used for bright-source filtering when
        provided).
    detection_sigma : float
        Detection threshold in units of the background standard
        deviation.

    Returns
    -------
    dict
        Keys include all fields from
        :func:`compute_astrometric_residuals` plus:

        - ``n_detected`` : int -- number of sources detected.
        - ``n_catalog`` : int -- number of reference sources.
        - ``detection_sigma`` : float -- sigma threshold used.
    """
    image = np.asarray(image, dtype=np.float64)

    # Simple source detection: threshold above median + sigma * MAD-based std
    med = np.median(image)
    mad = np.median(np.abs(image - med))
    std_est = mad * 1.4826  # MAD_TO_STD
    threshold = med + detection_sigma * std_est

    ndi = _import_scipy_ndimage()

    binary = image > threshold
    labelled, n_sources = ndi.label(binary)

    if n_sources == 0:
        return {
            "n_detected": 0,
            "n_catalog": int(np.asarray(catalog_ra).size),
            "detection_sigma": detection_sigma,
            "n_matched": 0,
            "median_offset_ra": float("nan"),
            "median_offset_dec": float("nan"),
            "rms_ra": float("nan"),
            "rms_dec": float("nan"),
            "rms_total": float("nan"),
        }

    # Compute flux-weighted centroids
    centroids_y = np.empty(n_sources)
    centroids_x = np.empty(n_sources)
    yy, xx = np.mgrid[0:image.shape[0], 0:image.shape[1]]

    for i in range(n_sources):
        mask = labelled == (i + 1)
        flux = image[mask] - med
        total = flux.sum()
        if total <= 0:
            total = 1.0
        centroids_x[i] = (xx[mask] * flux).sum() / total
        centroids_y[i] = (yy[mask] * flux).sum() / total

    # Transform detected pixel positions to sky coordinates
    crpix = wcs_params["crpix"]
    cd_matrix = wcs_params["cd_matrix"]
    crval = wcs_params["crval"]
    a_coeffs = wcs_params.get("a_coeffs", {})
    b_coeffs = wcs_params.get("b_coeffs", {})

    if a_coeffs or b_coeffs:
        det_ra, det_dec = sip_pixel_to_world(
            centroids_x, centroids_y, crpix, cd_matrix,
            a_coeffs, b_coeffs, crval,
        )
    else:
        # Linear WCS (no SIP)
        cd = np.asarray(cd_matrix, dtype=np.float64).reshape(2, 2)
        u = centroids_x - (crpix[0] - 1.0)
        v = centroids_y - (crpix[1] - 1.0)
        xi = cd[0, 0] * u + cd[0, 1] * v
        eta = cd[1, 0] * u + cd[1, 1] * v
        det_ra, det_dec = tan_deproject(xi, eta, crval[0], crval[1])

    # Cross-match and compute residuals
    residuals = compute_astrometric_residuals(
        det_ra, det_dec, catalog_ra, catalog_dec,
    )

    residuals["n_detected"] = n_sources
    residuals["n_catalog"] = int(np.asarray(catalog_ra).size)
    residuals["detection_sigma"] = detection_sigma
    return residuals
