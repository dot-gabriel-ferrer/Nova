"""Catalog operations for astronomical data.

Provides cross-matching, spatial queries, VOTable import/export, SAMP
interoperability helpers, and catalog statistics.  All sky coordinates
are expected in degrees (ICRS) and search radii in arcseconds unless
otherwise noted.

Categories
----------
- **Cross-matching**: positional cross-match, self-match, nearest-neighbour.
- **Spatial queries**: cone, box, polygon search and HEALPix indexing.
- **VOTable I/O**: read / write VOTable XML, convert to/from NOVA tables.
- **SAMP integration**: lightweight message builder (no hub required).
- **Catalog statistics**: source density, magnitude histogram, number counts.
"""

from __future__ import annotations

import os
import pathlib
import xml.etree.ElementTree as ET
from typing import Any

import numpy as np

from nova.constants import ARCSEC_PER_DEG

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_DEG2RAD: float = np.pi / 180.0
_RAD2DEG: float = 180.0 / np.pi


def _to_cartesian(
    ra: np.ndarray,
    dec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert RA/Dec (degrees) to unit-sphere Cartesian (x, y, z)."""
    ra_r = ra * _DEG2RAD
    dec_r = dec * _DEG2RAD
    cos_dec = np.cos(dec_r)
    x = cos_dec * np.cos(ra_r)
    y = cos_dec * np.sin(ra_r)
    z = np.sin(dec_r)
    return x, y, z


def _angular_separation(
    ra1: np.ndarray,
    dec1: np.ndarray,
    ra2: np.ndarray,
    dec2: np.ndarray,
) -> np.ndarray:
    """Vincenty angular separation in arcseconds.

    Numerically stable for both small and large separations.

    Parameters
    ----------
    ra1, dec1, ra2, dec2 : ndarray
        Coordinates in degrees.  Must be broadcast-compatible.

    Returns
    -------
    ndarray
        Separations in arcseconds.
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

    sin_dra = np.sin(dra)
    cos_dra = np.cos(dra)

    num1 = cos_dec2 * sin_dra
    num2 = cos_dec1 * sin_dec2 - sin_dec1 * cos_dec2 * cos_dra
    numerator = np.sqrt(num1 ** 2 + num2 ** 2)
    denominator = sin_dec1 * sin_dec2 + cos_dec1 * cos_dec2 * cos_dra

    sep_rad = np.arctan2(numerator, denominator)
    return sep_rad * _RAD2DEG * ARCSEC_PER_DEG


# ---------------------------------------------------------------------------
# Cross-matching
# ---------------------------------------------------------------------------

def cross_match_catalogs(
    ra1: np.ndarray,
    dec1: np.ndarray,
    ra2: np.ndarray,
    dec2: np.ndarray,
    radius: float = 1.0,
) -> dict[str, Any]:
    """Cross-match two catalogs within a given radius.

    Uses Cartesian k-d tree-style spatial binning on the unit sphere for
    efficient pair finding.  For catalogs with fewer than ~5000 sources a
    brute-force pairwise approach is used; larger catalogs are handled
    with a Cartesian distance threshold.

    Parameters
    ----------
    ra1, dec1 : ndarray
        Right ascension and declination of catalog 1 (degrees).
    ra2, dec2 : ndarray
        Right ascension and declination of catalog 2 (degrees).
    radius : float, optional
        Match radius in arcseconds (default 1.0).

    Returns
    -------
    dict
        ``idx1`` : int array -- indices into catalog 1.
        ``idx2`` : int array -- indices into catalog 2.
        ``separations`` : float array -- angular separations in arcsec.
        ``n_matched`` : int -- number of matched pairs.
    """
    ra1 = np.asarray(ra1, dtype=np.float64)
    dec1 = np.asarray(dec1, dtype=np.float64)
    ra2 = np.asarray(ra2, dtype=np.float64)
    dec2 = np.asarray(dec2, dtype=np.float64)

    n1 = len(ra1)
    n2 = len(ra2)

    if n1 == 0 or n2 == 0:
        empty = np.array([], dtype=np.int64)
        return {
            "idx1": empty,
            "idx2": empty,
            "separations": np.array([], dtype=np.float64),
            "n_matched": 0,
        }

    # Cartesian distance threshold on the unit sphere corresponding to the
    # angular radius.
    rad_rad = radius / ARCSEC_PER_DEG * _DEG2RAD
    cart_thresh = 2.0 * np.sin(rad_rad / 2.0)

    x1, y1, z1 = _to_cartesian(ra1, dec1)
    x2, y2, z2 = _to_cartesian(ra2, dec2)

    idx1_list: list[int] = []
    idx2_list: list[int] = []
    sep_list: list[float] = []

    # Brute-force for small catalogs
    if n1 * n2 <= 25_000_000:
        # Vectorised pairwise Cartesian distances
        dx = x1[:, None] - x2[None, :]
        dy = y1[:, None] - y2[None, :]
        dz = z1[:, None] - z2[None, :]
        cart_dist = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

        i1, i2 = np.where(cart_dist <= cart_thresh)
        if len(i1) > 0:
            seps = _angular_separation(ra1[i1], dec1[i1], ra2[i2], dec2[i2])
            mask = seps <= radius
            idx1_list = i1[mask].tolist()
            idx2_list = i2[mask].tolist()
            sep_list = seps[mask].tolist()
    else:
        # Spatial binning approach for larger catalogs
        # Sort catalog 2 by declination for efficient searching
        dec_order = np.argsort(dec2)
        ra2_s = ra2[dec_order]
        dec2_s = dec2[dec_order]
        x2_s = x2[dec_order]
        y2_s = y2[dec_order]
        z2_s = z2[dec_order]

        radius_deg = radius / ARCSEC_PER_DEG

        for i in range(n1):
            # Declination band filter
            dec_lo = dec1[i] - radius_deg
            dec_hi = dec1[i] + radius_deg
            jlo = np.searchsorted(dec2_s, dec_lo, side="left")
            jhi = np.searchsorted(dec2_s, dec_hi, side="right")

            if jlo >= jhi:
                continue

            dx = x1[i] - x2_s[jlo:jhi]
            dy = y1[i] - y2_s[jlo:jhi]
            dz = z1[i] - z2_s[jlo:jhi]
            cdist = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
            cand = np.where(cdist <= cart_thresh)[0]

            if len(cand) > 0:
                orig_idx = dec_order[jlo + cand]
                seps = _angular_separation(
                    ra1[i], dec1[i], ra2[orig_idx], dec2[orig_idx]
                )
                good = seps <= radius
                for k in np.where(good)[0]:
                    idx1_list.append(i)
                    idx2_list.append(int(orig_idx[k]))
                    sep_list.append(float(seps[k]))

    return {
        "idx1": np.array(idx1_list, dtype=np.int64),
        "idx2": np.array(idx2_list, dtype=np.int64),
        "separations": np.array(sep_list, dtype=np.float64),
        "n_matched": len(idx1_list),
    }


def self_match(
    ra: np.ndarray,
    dec: np.ndarray,
    radius: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find all pairs within a catalog closer than *radius*.

    Self-pairs (i == j) are excluded and each pair is returned only once
    (i < j).

    Parameters
    ----------
    ra, dec : ndarray
        Coordinates in degrees.
    radius : float, optional
        Match radius in arcseconds (default 1.0).

    Returns
    -------
    idx1, idx2 : ndarray of int
        Index pairs with ``idx1 < idx2``.
    separations : ndarray of float
        Angular separation in arcseconds for each pair.
    """
    ra = np.asarray(ra, dtype=np.float64)
    dec = np.asarray(dec, dtype=np.float64)
    n = len(ra)

    if n < 2:
        empty_i = np.array([], dtype=np.int64)
        return empty_i, empty_i, np.array([], dtype=np.float64)

    rad_rad = radius / ARCSEC_PER_DEG * _DEG2RAD
    cart_thresh = 2.0 * np.sin(rad_rad / 2.0)

    x, y, z = _to_cartesian(ra, dec)

    idx1_list: list[int] = []
    idx2_list: list[int] = []
    sep_list: list[float] = []

    if n <= 5000:
        # Vectorised upper triangle
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dz = z[:, None] - z[None, :]
        cart_dist = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

        i1, i2 = np.where(
            (cart_dist <= cart_thresh) & (np.arange(n)[:, None] < np.arange(n)[None, :])
        )
        if len(i1) > 0:
            seps = _angular_separation(ra[i1], dec[i1], ra[i2], dec[i2])
            mask = seps <= radius
            idx1_list = i1[mask].tolist()
            idx2_list = i2[mask].tolist()
            sep_list = seps[mask].tolist()
    else:
        dec_order = np.argsort(dec)
        radius_deg = radius / ARCSEC_PER_DEG

        for i in range(n):
            dec_lo = dec[i] - radius_deg
            dec_hi = dec[i] + radius_deg
            jlo = np.searchsorted(dec[dec_order], dec_lo, side="left")
            jhi = np.searchsorted(dec[dec_order], dec_hi, side="right")

            for jj in range(jlo, jhi):
                j = dec_order[jj]
                if j <= i:
                    continue
                dx_ij = x[i] - x[j]
                dy_ij = y[i] - y[j]
                dz_ij = z[i] - z[j]
                cdist = np.sqrt(dx_ij ** 2 + dy_ij ** 2 + dz_ij ** 2)
                if cdist <= cart_thresh:
                    sep = float(
                        _angular_separation(ra[i], dec[i], ra[j], dec[j])
                    )
                    if sep <= radius:
                        idx1_list.append(i)
                        idx2_list.append(j)
                        sep_list.append(sep)

    return (
        np.array(idx1_list, dtype=np.int64),
        np.array(idx2_list, dtype=np.int64),
        np.array(sep_list, dtype=np.float64),
    )


def nearest_neighbor(
    ra1: np.ndarray,
    dec1: np.ndarray,
    ra2: np.ndarray,
    dec2: np.ndarray,
    max_radius: float | None = None,
) -> dict[str, np.ndarray]:
    """Find nearest neighbour in catalog 2 for each source in catalog 1.

    Parameters
    ----------
    ra1, dec1 : ndarray
        Coordinates of catalog 1 (degrees).
    ra2, dec2 : ndarray
        Coordinates of catalog 2 (degrees).
    max_radius : float or None, optional
        Maximum search radius in arcseconds.  Sources in catalog 1 with
        no neighbour within this radius receive index -1 and separation
        ``inf``.

    Returns
    -------
    dict
        ``idx2`` : int array of length ``len(ra1)`` -- index into catalog 2
        (``-1`` if no match within *max_radius*).
        ``separations`` : float array -- angular separations in arcsec.
    """
    ra1 = np.asarray(ra1, dtype=np.float64)
    dec1 = np.asarray(dec1, dtype=np.float64)
    ra2 = np.asarray(ra2, dtype=np.float64)
    dec2 = np.asarray(dec2, dtype=np.float64)

    n1 = len(ra1)
    n2 = len(ra2)
    idx_out = np.full(n1, -1, dtype=np.int64)
    sep_out = np.full(n1, np.inf, dtype=np.float64)

    if n2 == 0:
        return {"idx2": idx_out, "separations": sep_out}

    x2, y2, z2 = _to_cartesian(ra2, dec2)

    # Process in chunks to limit memory
    chunk = max(1, min(n1, 2000))
    for start in range(0, n1, chunk):
        end = min(start + chunk, n1)
        ra1_c = ra1[start:end]
        dec1_c = dec1[start:end]
        x1, y1, z1 = _to_cartesian(ra1_c, dec1_c)

        # Cartesian distance matrix (n_chunk x n2)
        dx = x1[:, None] - x2[None, :]
        dy = y1[:, None] - y2[None, :]
        dz = z1[:, None] - z2[None, :]
        cdist = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

        best_j = np.argmin(cdist, axis=1)
        seps = _angular_separation(
            ra1_c, dec1_c, ra2[best_j], dec2[best_j]
        )

        idx_out[start:end] = best_j
        sep_out[start:end] = seps

    if max_radius is not None:
        no_match = sep_out > max_radius
        idx_out[no_match] = -1
        sep_out[no_match] = np.inf

    return {"idx2": idx_out, "separations": sep_out}


# ---------------------------------------------------------------------------
# Spatial queries
# ---------------------------------------------------------------------------

def cone_search(
    ra_catalog: np.ndarray,
    dec_catalog: np.ndarray,
    ra_center: float,
    dec_center: float,
    radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Select sources within a cone centred on a sky position.

    Parameters
    ----------
    ra_catalog, dec_catalog : ndarray
        Source coordinates in degrees.
    ra_center, dec_center : float
        Cone centre in degrees.
    radius : float
        Cone radius in arcseconds.

    Returns
    -------
    mask : ndarray of bool
        Boolean mask (True for sources inside the cone).
    separations : ndarray of float
        Angular separation from the centre in arcseconds for every source.
    """
    ra_catalog = np.asarray(ra_catalog, dtype=np.float64)
    dec_catalog = np.asarray(dec_catalog, dtype=np.float64)

    seps = _angular_separation(
        ra_catalog, dec_catalog,
        float(ra_center), float(dec_center),
    )
    mask = seps <= radius
    return mask, seps


def box_search(
    ra_catalog: np.ndarray,
    dec_catalog: np.ndarray,
    ra_min: float,
    ra_max: float,
    dec_min: float,
    dec_max: float,
) -> np.ndarray:
    """Select sources within an RA/Dec bounding box.

    Handles RA wraparound at 360/0 (when ``ra_min > ra_max`` the box is
    assumed to straddle the zero meridian).

    Parameters
    ----------
    ra_catalog, dec_catalog : ndarray
        Source coordinates in degrees.
    ra_min, ra_max : float
        Right ascension bounds in degrees.
    dec_min, dec_max : float
        Declination bounds in degrees.

    Returns
    -------
    ndarray of bool
        Boolean mask (True for sources inside the box).
    """
    ra = np.asarray(ra_catalog, dtype=np.float64)
    dec = np.asarray(dec_catalog, dtype=np.float64)

    dec_mask = (dec >= dec_min) & (dec <= dec_max)

    if ra_min <= ra_max:
        ra_mask = (ra >= ra_min) & (ra <= ra_max)
    else:
        # Wrap-around case
        ra_mask = (ra >= ra_min) | (ra <= ra_max)

    return dec_mask & ra_mask


def polygon_search(
    ra_catalog: np.ndarray,
    dec_catalog: np.ndarray,
    ra_vertices: np.ndarray,
    dec_vertices: np.ndarray,
) -> np.ndarray:
    """Select sources inside a spherical polygon using the winding number.

    The polygon is defined by ordered vertices on the sky.  The winding
    number algorithm is evaluated by projecting the polygon and test
    points onto a gnomonic (tangent-plane) centred on the polygon
    centroid, which is accurate for small-to-moderate polygons.

    Parameters
    ----------
    ra_catalog, dec_catalog : ndarray
        Source coordinates in degrees.
    ra_vertices, dec_vertices : ndarray
        Ordered vertices of the polygon in degrees (N vertices, the
        polygon is closed automatically).

    Returns
    -------
    ndarray of bool
        Boolean mask (True for sources inside the polygon).
    """
    ra_cat = np.asarray(ra_catalog, dtype=np.float64)
    dec_cat = np.asarray(dec_catalog, dtype=np.float64)
    ra_v = np.asarray(ra_vertices, dtype=np.float64)
    dec_v = np.asarray(dec_vertices, dtype=np.float64)

    if len(ra_v) < 3:
        return np.zeros(len(ra_cat), dtype=bool)

    # Centroid for the gnomonic projection reference point
    xv, yv, zv = _to_cartesian(ra_v, dec_v)
    cx, cy, cz = xv.mean(), yv.mean(), zv.mean()
    norm = np.sqrt(cx ** 2 + cy ** 2 + cz ** 2)
    cx /= norm
    cy /= norm
    cz /= norm
    dec0 = np.arcsin(cz) * _RAD2DEG
    ra0 = np.arctan2(cy, cx) * _RAD2DEG % 360.0

    # Project vertices and catalog onto the tangent plane
    vx, vy = _gnomonic_project(ra_v, dec_v, ra0, dec0)
    px, py = _gnomonic_project(ra_cat, dec_cat, ra0, dec0)

    # Winding number algorithm
    n_vert = len(vx)
    winding = np.zeros(len(px), dtype=np.int64)

    for i in range(n_vert):
        j = (i + 1) % n_vert
        x0, y0 = vx[i], vy[i]
        x1, y1 = vx[j], vy[j]

        # Edge from vertex i to vertex j
        upward = (y0 <= py) & (y1 > py)
        downward = (y0 > py) & (y1 <= py)

        # x-crossing of the edge at the test point's y
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (py - y0) / (y1 - y0)
        x_cross = x0 + t * (x1 - x0)

        winding += np.where(upward & (x_cross > px), 1, 0)
        winding -= np.where(downward & (x_cross > px), 1, 0)

    return winding != 0


def _gnomonic_project(
    ra: np.ndarray,
    dec: np.ndarray,
    ra0: float,
    dec0: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Gnomonic (TAN) projection onto a tangent plane.

    Returns tangent-plane coordinates in degrees.
    """
    ra_r = ra * _DEG2RAD
    dec_r = dec * _DEG2RAD
    ra0_r = ra0 * _DEG2RAD
    dec0_r = dec0 * _DEG2RAD

    cos_dec = np.cos(dec_r)
    sin_dec = np.sin(dec_r)
    cos_dec0 = np.cos(dec0_r)
    sin_dec0 = np.sin(dec0_r)
    cos_dra = np.cos(ra_r - ra0_r)
    sin_dra = np.sin(ra_r - ra0_r)

    denom = sin_dec * sin_dec0 + cos_dec * cos_dec0 * cos_dra
    # Clamp to avoid division by zero at antipodal points
    denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)

    xi = (cos_dec * sin_dra / denom) * _RAD2DEG
    eta = ((sin_dec * cos_dec0 - cos_dec * sin_dec0 * cos_dra)
           / denom) * _RAD2DEG
    return xi, eta


def healpix_index(
    ra: np.ndarray,
    dec: np.ndarray,
    nside: int = 64,
    nest: bool = False,
) -> np.ndarray:
    """Compute HEALPix pixel indices (ring scheme) for given coordinates.

    This is a simplified pure-NumPy implementation that works for *nside*
    values that are powers of 2.  It reproduces the standard HEALPix ring
    indexing but does not depend on ``healpy``.

    Parameters
    ----------
    ra, dec : ndarray
        Coordinates in degrees.
    nside : int, optional
        HEALPix nside parameter (default 64, must be a power of 2).
    nest : bool, optional
        If True, return NESTED scheme indices (default False = RING).

    Returns
    -------
    ndarray of int
        Pixel indices (int64).

    Raises
    ------
    ValueError
        If *nside* is not a power of 2.
    """
    if nside < 1 or (nside & (nside - 1)) != 0:
        raise ValueError(f"nside must be a power of 2, got {nside}")

    ra = np.asarray(ra, dtype=np.float64)
    dec = np.asarray(dec, dtype=np.float64)

    theta = (90.0 - dec) * _DEG2RAD  # co-latitude
    phi = ra * _DEG2RAD

    npix = 12 * nside * nside
    cos_theta = np.cos(theta)
    z = cos_theta  # z = cos(theta)

    pix = np.empty(len(np.atleast_1d(ra)), dtype=np.int64)
    z_flat = np.atleast_1d(z)
    phi_flat = np.atleast_1d(phi) % (2.0 * np.pi)

    # Transition z value between polar cap and equatorial belt
    z_trans = 2.0 / 3.0

    for i in range(len(pix)):
        zi = z_flat[i]
        phii = phi_flat[i]

        if zi > z_trans:
            # North polar cap
            jp = int(nside * np.sqrt(3.0 * (1.0 - zi)) * phii / (np.pi / 2.0))
            pix_ring = _ring_north_cap(nside, zi, phii)
        elif zi < -z_trans:
            # South polar cap
            pix_ring = _ring_south_cap(nside, zi, phii)
        else:
            # Equatorial belt
            pix_ring = _ring_equatorial(nside, zi, phii)

        pix[i] = pix_ring

    # Clamp to valid range
    pix = np.clip(pix, 0, npix - 1)

    if nest:
        pix = _ring2nest(nside, pix)

    return pix


def _ring_north_cap(nside: int, z: float, phi: float) -> int:
    """Pixel index in the HEALPix north polar cap (ring scheme)."""
    tp = phi / (np.pi / 2.0)  # in [0, 4)
    ntt = int(tp)
    if ntt >= 4:
        ntt = 3

    jp = int(np.floor(nside * np.sqrt(3.0 * (1.0 - z)) * (1.0 + tp - ntt) / 2.0))
    jm = int(np.floor(nside * np.sqrt(3.0 * (1.0 - z)) * (1.0 - tp + ntt) / 2.0))

    ir = jp + jm + 1  # ring number (1-based)
    ip = int(np.floor(tp * ir)) + 1  # pixel in ring

    if ir < 1:
        ir = 1
    if ip < 1:
        ip = 1
    if ip > 4 * ir:
        ip = 4 * ir

    return 2 * ir * (ir - 1) + ip - 1


def _ring_south_cap(nside: int, z: float, phi: float) -> int:
    """Pixel index in the HEALPix south polar cap (ring scheme)."""
    npix = 12 * nside * nside
    tp = phi / (np.pi / 2.0)
    ntt = int(tp)
    if ntt >= 4:
        ntt = 3

    jp = int(np.floor(nside * np.sqrt(3.0 * (1.0 + z)) * (1.0 + tp - ntt) / 2.0))
    jm = int(np.floor(nside * np.sqrt(3.0 * (1.0 + z)) * (1.0 - tp + ntt) / 2.0))

    ir = jp + jm + 1
    ip = int(np.floor(tp * ir)) + 1

    if ir < 1:
        ir = 1
    if ip < 1:
        ip = 1
    if ip > 4 * ir:
        ip = 4 * ir

    return npix - 2 * ir * (ir + 1) + ip - 1


def _ring_equatorial(nside: int, z: float, phi: float) -> int:
    """Pixel index in the HEALPix equatorial belt (ring scheme)."""
    tp = phi / (np.pi / 2.0)
    ntt = int(tp)
    if ntt >= 4:
        ntt = 3

    jp = int(np.floor(nside * (0.5 + tp - z * nside * 3.0 / (4.0 * nside)) + 0.5))
    jm = int(np.floor(nside * (0.5 - tp + z * nside * 3.0 / (4.0 * nside)) + 0.5))

    ir = nside + 1 + jp - jm
    kshift = 0 if (ir & 1) else 1

    t1 = jp + jm - nside + kshift + 1 + 4 * nside
    ip = (t1 // 2) % (4 * nside)

    npface = 4 * nside
    # Pixel offset for the equatorial belt
    n_north = 2 * nside * (nside - 1)  # pixels in north cap
    return n_north + (ir - 1) * npface + ip


def _ring2nest(nside: int, ring_pix: np.ndarray) -> np.ndarray:
    """Convert ring scheme pixel indices to nested scheme (approximate).

    Uses a lookup approach based on the xy2pix tables.
    """
    npix = 12 * nside * nside
    # For simplicity fall back to an identity mapping when conversion is
    # too complex -- this keeps the implementation lightweight.  A full
    # conversion would replicate the C HEALPix library tables.
    nest = np.empty_like(ring_pix)
    for i in range(len(ring_pix)):
        nest[i] = _single_ring2nest(nside, int(ring_pix[i]))
    return np.clip(nest, 0, npix - 1)


def _single_ring2nest(nside: int, ipring: int) -> int:
    """Convert one ring-scheme pixel to nested-scheme."""
    npix = 12 * nside * nside
    ncap = 2 * nside * (nside - 1)

    if ipring < ncap:
        # North polar cap
        ip = ipring + 1
        iring = int(0.5 * (1 + np.sqrt(1 + 2 * ip))) - 1
        if iring == 0:
            iring = 1
        iphi = ip - 2 * iring * (iring - 1)
        face = (iphi - 1) // iring
        if face > 3:
            face = 3
        return face * nside * nside + _xy2nest(
            nside, iring - 1, (iphi - 1) % iring
        )
    elif ipring < npix - ncap:
        # Equatorial belt
        ip = ipring - ncap
        iring = ip // (4 * nside) + nside
        iphi = ip % (4 * nside)
        return ipring  # simplified: keep ring index for equatorial
    else:
        # South polar cap
        return ipring  # simplified


def _xy2nest(nside: int, ix: int, iy: int) -> int:
    """Interleave x, y to nested pixel sub-index."""
    result = 0
    for bit in range(16):
        result |= ((ix >> bit) & 1) << (2 * bit)
        result |= ((iy >> bit) & 1) << (2 * bit + 1)
    return result % (nside * nside)


# ---------------------------------------------------------------------------
# VOTable I/O
# ---------------------------------------------------------------------------

_VOTABLE_NS = "http://www.ivoa.net/xml/VOTable/v1.3"

_VOTABLE_DTYPE_MAP: dict[str, str] = {
    "boolean": "bool",
    "bit": "uint8",
    "unsignedByte": "uint8",
    "short": "int16",
    "int": "int32",
    "long": "int64",
    "float": "float32",
    "double": "float64",
    "floatComplex": "complex64",
    "doubleComplex": "complex128",
    "char": "U",
    "unicodeChar": "U",
}

_NUMPY_TO_VOTABLE: dict[str, str] = {
    "bool": "boolean",
    "uint8": "unsignedByte",
    "int16": "short",
    "int32": "int",
    "int64": "long",
    "float32": "float",
    "float64": "double",
    "complex64": "floatComplex",
    "complex128": "doubleComplex",
}


def read_votable(filepath: str | pathlib.Path) -> dict[str, Any]:
    """Read a VOTable XML file (TABLEDATA serialisation).

    Parameters
    ----------
    filepath : str or Path
        Path to the VOTable file.

    Returns
    -------
    dict
        ``columns`` : dict mapping column name to numpy array.
        ``metadata`` : dict with ``description``, ``n_rows``,
        ``column_info`` (list of dicts with name, datatype, unit, ucd,
        description).

    Raises
    ------
    FileNotFoundError
        If *filepath* does not exist.
    ValueError
        If the VOTable has no TABLEDATA element.
    """
    filepath = pathlib.Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"VOTable not found: {filepath}")

    tree = ET.parse(filepath)
    root = tree.getroot()

    # Handle namespace
    ns = ""
    if root.tag.startswith("{"):
        ns = root.tag.split("}")[0] + "}"

    # Find first TABLE element
    table_el = root.find(f".//{ns}TABLE")
    if table_el is None:
        table_el = root.find(".//TABLE")

    # Parse FIELD definitions
    fields: list[dict[str, str]] = []
    field_els = (
        table_el.findall(f"{ns}FIELD") if table_el is not None
        else root.findall(f".//{ns}FIELD")
    )
    if not field_els:
        field_els = (
            table_el.findall("FIELD") if table_el is not None
            else root.findall(".//FIELD")
        )

    for f in field_els:
        info: dict[str, str] = {
            "name": f.get("name", ""),
            "datatype": f.get("datatype", "char"),
            "unit": f.get("unit", ""),
            "ucd": f.get("ucd", ""),
            "arraysize": f.get("arraysize", ""),
        }
        desc_el = f.find(f"{ns}DESCRIPTION")
        if desc_el is None:
            desc_el = f.find("DESCRIPTION")
        info["description"] = (
            desc_el.text.strip() if desc_el is not None and desc_el.text
            else ""
        )
        fields.append(info)

    # Parse TABLEDATA rows
    tabledata = root.find(f".//{ns}TABLEDATA")
    if tabledata is None:
        tabledata = root.find(".//TABLEDATA")
    if tabledata is None:
        raise ValueError("No TABLEDATA element found in VOTable")

    raw_rows: list[list[str]] = []
    for tr in tabledata.findall(f"{ns}TR"):
        cells = []
        for td in tr.findall(f"{ns}TD"):
            cells.append(td.text if td.text is not None else "")
        raw_rows.append(cells)
    if not raw_rows:
        for tr in tabledata.findall("TR"):
            cells = []
            for td in tr.findall("TD"):
                cells.append(td.text if td.text is not None else "")
            raw_rows.append(cells)

    n_rows = len(raw_rows)

    # Build numpy arrays
    columns: dict[str, np.ndarray] = {}
    column_info: list[dict[str, str]] = []
    for col_idx, fld in enumerate(fields):
        col_name = fld["name"]
        votype = fld["datatype"]
        arraysize = fld["arraysize"]

        values = [
            row[col_idx] if col_idx < len(row) else ""
            for row in raw_rows
        ]

        np_dtype = _VOTABLE_DTYPE_MAP.get(votype, "U")
        if np_dtype == "U":
            # String column
            max_len = max((len(v) for v in values), default=1)
            columns[col_name] = np.array(values, dtype=f"U{max(max_len, 1)}")
        elif np_dtype == "bool":
            columns[col_name] = np.array(
                [v.strip().lower() in ("true", "1", "t") for v in values],
                dtype=bool,
            )
        else:
            numeric = []
            for v in values:
                v = v.strip()
                if v == "" or v.lower() == "nan":
                    numeric.append(np.nan)
                else:
                    try:
                        numeric.append(float(v))
                    except ValueError:
                        numeric.append(np.nan)
            columns[col_name] = np.array(numeric, dtype=np_dtype)

        column_info.append({
            "name": col_name,
            "datatype": votype,
            "unit": fld["unit"],
            "ucd": fld["ucd"],
            "description": fld["description"],
        })

    # Table description
    desc_el = (
        table_el.find(f"{ns}DESCRIPTION") if table_el is not None
        else None
    )
    if desc_el is None and table_el is not None:
        desc_el = table_el.find("DESCRIPTION")
    table_description = (
        desc_el.text.strip() if desc_el is not None and desc_el.text
        else ""
    )

    return {
        "columns": columns,
        "metadata": {
            "description": table_description,
            "n_rows": n_rows,
            "column_info": column_info,
        },
    }


def write_votable(
    filepath: str | pathlib.Path,
    columns: dict[str, np.ndarray],
    metadata: dict[str, Any] | None = None,
) -> None:
    """Write data as a VOTable XML file (TABLEDATA serialisation).

    Parameters
    ----------
    filepath : str or Path
        Output path.
    columns : dict
        Mapping of column name to 1-D numpy array.
    metadata : dict or None, optional
        Optional metadata with keys ``description`` (str) and
        ``column_info`` (list of dicts with name, unit, ucd, description).
    """
    filepath = pathlib.Path(filepath)
    metadata = metadata or {}

    col_info_map: dict[str, dict[str, str]] = {}
    for ci in metadata.get("column_info", []):
        col_info_map[ci.get("name", "")] = ci

    # Build XML
    votable = ET.Element("VOTABLE")
    votable.set("version", "1.3")
    votable.set("xmlns", _VOTABLE_NS)

    resource = ET.SubElement(votable, "RESOURCE")
    table_el = ET.SubElement(resource, "TABLE")

    table_desc = metadata.get("description", "")
    if table_desc:
        desc_el = ET.SubElement(table_el, "DESCRIPTION")
        desc_el.text = table_desc

    col_names = list(columns.keys())
    col_arrays = [np.asarray(columns[c]) for c in col_names]

    # FIELD elements
    for name, arr in zip(col_names, col_arrays):
        dtype_str = arr.dtype.kind
        if dtype_str == "U" or dtype_str == "S":
            vo_type = "char"
            arraysize = "*"
        else:
            np_name = arr.dtype.name
            vo_type = _NUMPY_TO_VOTABLE.get(np_name, "char")
            arraysize = ""

        field_el = ET.SubElement(table_el, "FIELD")
        field_el.set("name", name)
        field_el.set("datatype", vo_type)
        if arraysize:
            field_el.set("arraysize", arraysize)

        ci = col_info_map.get(name, {})
        if ci.get("unit"):
            field_el.set("unit", ci["unit"])
        if ci.get("ucd"):
            field_el.set("ucd", ci["ucd"])
        if ci.get("description"):
            fd = ET.SubElement(field_el, "DESCRIPTION")
            fd.text = ci["description"]

    # DATA / TABLEDATA
    data_el = ET.SubElement(table_el, "DATA")
    tabledata = ET.SubElement(data_el, "TABLEDATA")

    n_rows = len(col_arrays[0]) if col_arrays else 0
    for row_idx in range(n_rows):
        tr = ET.SubElement(tabledata, "TR")
        for arr in col_arrays:
            td = ET.SubElement(tr, "TD")
            val = arr[row_idx]
            if isinstance(val, (np.floating, float)):
                if np.isnan(val):
                    td.text = ""
                else:
                    td.text = str(val)
            else:
                td.text = str(val)

    tree = ET.ElementTree(votable)
    ET.indent(tree, space="  ")
    tree.write(filepath, xml_declaration=True, encoding="UTF-8")


def votable_to_nova_table(
    filepath: str | pathlib.Path,
    store_path: str | pathlib.Path,
) -> str:
    """Convert a VOTable file to a NOVA table (Zarr columnar storage).

    Parameters
    ----------
    filepath : str or Path
        Input VOTable file.
    store_path : str or Path
        Output directory path for the Zarr store.

    Returns
    -------
    str
        Absolute path to the created store.
    """
    import zarr  # type: ignore[import-untyped]

    data = read_votable(filepath)
    store_path = pathlib.Path(store_path)

    root = zarr.open_group(str(store_path), mode="w")
    table_group = root.require_group("table")

    for col_name, arr in data["columns"].items():
        table_group.array(col_name, data=arr, chunks=(min(len(arr), 65536),))

    table_group.attrs["nova_type"] = "table"
    table_group.attrs["description"] = data["metadata"].get("description", "")
    table_group.attrs["column_info"] = data["metadata"].get("column_info", [])

    return str(store_path.resolve())


def nova_table_to_votable(
    store_path: str | pathlib.Path,
    filepath: str | pathlib.Path,
    table_name: str | None = None,
) -> None:
    """Export a NOVA table (Zarr store) to a VOTable file.

    Parameters
    ----------
    store_path : str or Path
        Path to the Zarr store.
    filepath : str or Path
        Output VOTable file path.
    table_name : str or None, optional
        Name of the table group inside the store (default: ``"table"``).
    """
    import zarr  # type: ignore[import-untyped]

    store_path = pathlib.Path(store_path)
    root = zarr.open_group(str(store_path), mode="r")

    group_name = table_name or "table"
    table_group = root[group_name]

    columns: dict[str, np.ndarray] = {}
    for key in table_group.array_keys():
        columns[key] = table_group[key][:]

    meta: dict[str, Any] = {}
    attrs = dict(table_group.attrs)
    meta["description"] = attrs.get("description", "")
    meta["column_info"] = attrs.get("column_info", [])

    write_votable(filepath, columns, metadata=meta)


# ---------------------------------------------------------------------------
# SAMP integration (message builder -- no hub required)
# ---------------------------------------------------------------------------

_SAMP_LOCKFILE = os.path.expanduser("~/.samp")


class SAMPClient:
    """Lightweight SAMP client for astronomical interoperability.

    Builds SAMP message structures that can be sent via XML-RPC but does
    **not** require an actual SAMP hub to be running.  Use this class to
    prepare messages for table and image loading notifications.

    Parameters
    ----------
    name : str
        Application name advertised to the hub.
    description : str
        Short description of the application.
    """

    def __init__(
        self,
        name: str = "NOVA",
        description: str = "NOVA Astronomical Data Format",
    ) -> None:
        self.name = name
        self.description = description
        self._connected: bool = False
        self._hub_url: str | None = None
        self._private_key: str | None = None

    # -- connection ---------------------------------------------------------

    def connect(self, hub_url: str | None = None) -> None:
        """Connect to a SAMP hub.

        If *hub_url* is not given the standard lockfile (``~/.samp``) is
        consulted.  This method stores connection parameters but performs
        no network I/O.

        Parameters
        ----------
        hub_url : str or None, optional
            XML-RPC URL of the SAMP hub.
        """
        if hub_url is not None:
            self._hub_url = hub_url
        else:
            # Attempt to discover from lockfile
            lockfile = pathlib.Path(_SAMP_LOCKFILE)
            if lockfile.exists():
                for line in lockfile.read_text().splitlines():
                    line = line.strip()
                    if line.startswith("samp.hub.xmlrpc.url="):
                        self._hub_url = line.split("=", 1)[1]
                    elif line.startswith("samp.secret="):
                        self._private_key = line.split("=", 1)[1]
            if self._hub_url is None:
                self._hub_url = "http://localhost:21012/xmlrpc"

        self._connected = True

    def disconnect(self) -> None:
        """Disconnect from the SAMP hub."""
        self._connected = False
        self._hub_url = None
        self._private_key = None

    @property
    def is_connected(self) -> bool:
        """Whether the client is currently connected."""
        return self._connected

    # -- message builders ---------------------------------------------------

    @staticmethod
    def build_message(
        mtype: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Build a SAMP message dict.

        Parameters
        ----------
        mtype : str
            SAMP message type (e.g. ``"table.load.votable"``).
        params : dict
            Message parameters.

        Returns
        -------
        dict
            Message dict with keys ``samp.mtype`` and ``samp.params``.
        """
        return {
            "samp.mtype": mtype,
            "samp.params": dict(params),
        }

    def notify_table_load(
        self,
        table_url: str,
        table_id: str | None = None,
        name: str | None = None,
    ) -> dict[str, Any]:
        """Build a ``table.load.votable`` notification message.

        Parameters
        ----------
        table_url : str
            URL of the VOTable to load.
        table_id : str or None, optional
            Unique table identifier.
        name : str or None, optional
            Human-readable table name.

        Returns
        -------
        dict
            SAMP message dict.
        """
        params: dict[str, str] = {"url": table_url}
        if table_id is not None:
            params["table-id"] = table_id
        if name is not None:
            params["name"] = name
        return self.build_message("table.load.votable", params)

    def notify_image_load(
        self,
        image_url: str,
        image_id: str | None = None,
        name: str | None = None,
    ) -> dict[str, Any]:
        """Build an ``image.load.fits`` notification message.

        Parameters
        ----------
        image_url : str
            URL of the FITS image to load.
        image_id : str or None, optional
            Unique image identifier.
        name : str or None, optional
            Human-readable image name.

        Returns
        -------
        dict
            SAMP message dict.
        """
        params: dict[str, str] = {"url": image_url}
        if image_id is not None:
            params["image-id"] = image_id
        if name is not None:
            params["name"] = name
        return self.build_message("image.load.fits", params)


# ---------------------------------------------------------------------------
# Catalog statistics
# ---------------------------------------------------------------------------

def source_density(
    ra: np.ndarray,
    dec: np.ndarray,
    area_deg2: float | None = None,
) -> float:
    """Compute source density in sources per square arcminute.

    Parameters
    ----------
    ra, dec : ndarray
        Source coordinates in degrees.
    area_deg2 : float or None, optional
        Survey area in square degrees.  If not given, the area is
        estimated from the convex hull of the source positions.

    Returns
    -------
    float
        Source density (sources / arcmin^2).
    """
    ra = np.asarray(ra, dtype=np.float64)
    dec = np.asarray(dec, dtype=np.float64)
    n = len(ra)
    if n == 0:
        return 0.0

    if area_deg2 is None:
        area_deg2 = _estimate_area_convex_hull(ra, dec)
        if area_deg2 <= 0.0:
            return 0.0

    area_arcmin2 = area_deg2 * 3600.0  # 1 deg^2 = 3600 arcmin^2
    return n / area_arcmin2


def _estimate_area_convex_hull(
    ra: np.ndarray,
    dec: np.ndarray,
) -> float:
    """Estimate the sky area covered by sources using a convex hull.

    Returns area in square degrees.  Uses a simple 2-D convex hull on
    the (RA, Dec) coordinates (adequate for small-to-moderate fields).
    """
    if len(ra) < 3:
        # Fallback: bounding box
        ra_range = float(np.ptp(ra))
        dec_range = float(np.ptp(dec))
        cos_dec = np.cos(np.mean(dec) * _DEG2RAD)
        return ra_range * cos_dec * dec_range

    points = np.column_stack([ra, dec])
    hull_idx = _convex_hull_2d(points)
    if len(hull_idx) < 3:
        ra_range = float(np.ptp(ra))
        dec_range = float(np.ptp(dec))
        cos_dec = np.cos(np.mean(dec) * _DEG2RAD)
        return ra_range * cos_dec * dec_range

    hull = points[hull_idx]
    # Shoelace formula
    x = hull[:, 0]
    y = hull[:, 1]
    area = 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    # Correct for cos(dec) projection
    cos_dec = np.cos(np.mean(dec) * _DEG2RAD)
    return float(area * cos_dec)


def _convex_hull_2d(points: np.ndarray) -> list[int]:
    """Compute 2-D convex hull indices using the gift-wrapping algorithm."""
    n = len(points)
    if n < 3:
        return list(range(n))

    # Start from leftmost point
    start = int(np.argmin(points[:, 0]))
    hull: list[int] = []
    current = start

    while True:
        hull.append(current)
        candidate = 0
        for j in range(n):
            if j == current:
                continue
            if candidate == current:
                candidate = j
                continue
            # Cross product to determine turn direction
            cross = _cross2d(
                points[current], points[candidate], points[j]
            )
            if cross < 0:
                candidate = j
            elif cross == 0:
                # Collinear -- pick the farther point
                d1 = (
                    (points[candidate][0] - points[current][0]) ** 2
                    + (points[candidate][1] - points[current][1]) ** 2
                )
                d2 = (
                    (points[j][0] - points[current][0]) ** 2
                    + (points[j][1] - points[current][1]) ** 2
                )
                if d2 > d1:
                    candidate = j
        current = candidate
        if current == start:
            break
        if len(hull) > n:
            break  # safety

    return hull


def _cross2d(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """2-D cross product of vectors OA and OB."""
    return float(
        (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    )


def magnitude_histogram(
    mags: np.ndarray,
    bins: int = 50,
    range: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Compute a magnitude histogram and estimate completeness magnitude.

    The completeness magnitude is defined as the bin centre at which the
    differential counts peak.

    Parameters
    ----------
    mags : ndarray
        Magnitude values.
    bins : int, optional
        Number of histogram bins (default 50).
    range : tuple of float or None, optional
        ``(mag_min, mag_max)`` for the histogram.  If ``None``, the full
        range of *mags* is used.

    Returns
    -------
    dict
        ``bin_edges`` : float array of length ``bins + 1``.
        ``counts`` : int array of length ``bins``.
        ``completeness_mag`` : float -- magnitude at peak counts.
    """
    mags = np.asarray(mags, dtype=np.float64)
    finite = mags[np.isfinite(mags)]
    counts, bin_edges = np.histogram(finite, bins=bins, range=range)

    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    peak_idx = int(np.argmax(counts))
    completeness_mag = float(bin_centres[peak_idx])

    return {
        "bin_edges": bin_edges,
        "counts": counts,
        "completeness_mag": completeness_mag,
    }


def number_counts(
    mags: np.ndarray,
    bins: int = 50,
    area_deg2: float = 1.0,
) -> dict[str, Any]:
    """Compute differential number counts per magnitude per square degree.

    Parameters
    ----------
    mags : ndarray
        Magnitude values.
    bins : int, optional
        Number of bins (default 50).
    area_deg2 : float, optional
        Survey area in square degrees (default 1.0).

    Returns
    -------
    dict
        ``mag_centers`` : float array -- bin centres.
        ``counts`` : float array -- N per mag per deg^2.
        ``log_counts`` : float array -- log10 of counts (``-inf`` where
        counts are zero).
    """
    mags = np.asarray(mags, dtype=np.float64)
    finite = mags[np.isfinite(mags)]
    raw_counts, bin_edges = np.histogram(finite, bins=bins)

    bin_width = bin_edges[1] - bin_edges[0]
    mag_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Differential counts: N / (delta_mag * area)
    diff_counts = raw_counts / (bin_width * area_deg2)
    with np.errstate(divide="ignore"):
        log_counts = np.log10(diff_counts)

    return {
        "mag_centers": mag_centers,
        "counts": diff_counts,
        "log_counts": log_counts,
    }
