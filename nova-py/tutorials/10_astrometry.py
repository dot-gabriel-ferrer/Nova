"""Tutorial 10: NOVA Astrometry Tools.

Demonstrates the astrometry module: source extraction, plate solving,
proper-motion correction, residual analysis, and SIP distortion fitting.

Steps:
  1. Generate a synthetic star field image
  2. Extract centroids from the image
  3. Plate-solve to find the WCS solution
  4. Compute astrometric residuals
  5. Apply proper-motion correction between epochs
  6. Fit SIP distortion polynomials

Run:
    cd nova-py
    python tutorials/10_astrometry.py
"""

from __future__ import annotations

import numpy as np


def _make_star_field(
    n_stars: int, shape: tuple[int, int], rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Return (image, true_positions) with Gaussian stars.

    true_positions has shape (n_stars, 3): [x, y, flux].
    """
    img = rng.normal(loc=200.0, scale=10.0, size=shape).astype(np.float64)
    positions = np.zeros((n_stars, 3))
    border = 20
    for i in range(n_stars):
        x = rng.uniform(border, shape[1] - border)
        y = rng.uniform(border, shape[0] - border)
        flux = rng.uniform(2000.0, 50000.0)
        sigma = rng.uniform(1.8, 3.0)
        yy, xx = np.ogrid[0:shape[0], 0:shape[1]]
        img += flux * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2))
        positions[i] = [x, y, flux]
    return img, positions


def _pixel_to_radec(
    xy: np.ndarray, crpix: tuple[float, float],
    crval: tuple[float, float], cd: np.ndarray,
) -> np.ndarray:
    """Simple tangent-plane pixel-to-sky projection (small field)."""
    dx = xy[:, 0] - crpix[0]
    dy = xy[:, 1] - crpix[1]
    ra = crval[0] + cd[0, 0] * dx + cd[0, 1] * dy
    dec = crval[1] + cd[1, 0] * dx + cd[1, 1] * dy
    return np.column_stack([ra, dec])


def main() -> None:
    from nova.astrometry import (
        astrometric_residuals,
        correct_proper_motion,
        extract_centroids,
        fit_distortion_sip,
        plate_solve,
    )

    print("=" * 70)
    print("  NOVA Tutorial 10: Astrometry Tools")
    print("=" * 70)
    print()

    rng = np.random.default_rng(12345)
    shape = (512, 512)

    # -- Reference WCS (ground truth) --------------------------------------
    crpix = (256.0, 256.0)
    crval = (150.0, 2.0)  # RA, Dec in degrees
    pixel_scale_deg = 0.3 / 3600.0  # 0.3 arcsec/pixel
    cd = np.array([[-pixel_scale_deg, 0.0],
                   [0.0, pixel_scale_deg]])

    # -- Step 1: Generate synthetic star field -----------------------------
    print("Step 1: Generate a synthetic star field (80 stars)")
    print("-" * 70)

    image, true_pos = _make_star_field(80, shape, rng)
    print(f"  Image shape: {image.shape}")
    print(f"  Pixel range: [{image.min():.1f}, {image.max():.1f}]")
    print(f"  True stars:  {len(true_pos)}")
    print()

    # -- Step 2: Extract centroids -----------------------------------------
    print("Step 2: Extract centroids from the image")
    print("-" * 70)

    centroids = extract_centroids(image, fwhm=4.0, threshold=5.0, max_sources=100)
    print(f"  Detected sources: {len(centroids)}")
    print(f"  Columns: [x, y, flux]  shape={centroids.shape}")
    print(f"  Brightest source flux: {centroids[0, 2]:.1f}")
    print(f"  Faintest  source flux: {centroids[-1, 2]:.1f}")
    print()

    # -- Step 3: Plate solve -----------------------------------------------
    print("Step 3: Plate solve to recover WCS")
    print("-" * 70)

    # Build a catalog from true positions (with small scatter)
    catalog_radec = _pixel_to_radec(true_pos[:, :2], crpix, crval, cd)
    catalog_radec += rng.normal(0, 0.1 / 3600.0, catalog_radec.shape)

    solution = plate_solve(
        centroids,
        catalog_radec,
        image_shape=shape,
        pixel_scale_guess=0.3,
        match_tol=0.02,
        n_bright=30,
    )

    print(f"  Success:      {solution['success']}")
    print(f"  Matched:      {solution['matched']} stars")
    print(f"  CRPIX:        ({solution['crpix'][0]:.1f}, {solution['crpix'][1]:.1f})")
    print(f"  CRVAL:        ({solution['crval'][0]:.6f}, {solution['crval'][1]:.6f})")
    print(f"  Residual:     {solution['residual_arcsec']:.4f} arcsec")
    print()

    # -- Step 4: Astrometric residuals -------------------------------------
    print("Step 4: Compute astrometric residuals")
    print("-" * 70)

    n_match = min(len(centroids), len(catalog_radec))
    resid = astrometric_residuals(
        centroids[:n_match, :2],
        catalog_radec[:n_match],
        crpix=tuple(solution["crpix"]),
        crval=tuple(solution["crval"]),
        cd_matrix=np.array(solution["cd_matrix"]),
    )

    print(f"  Sources used: {resid['n_sources']}")
    print(f"  RMS:          {resid['rms_arcsec']:.4f} arcsec")
    print(f"  Median:       {resid['median_arcsec']:.4f} arcsec")
    print(f"  Max:          {resid['max_arcsec']:.4f} arcsec")
    print()

    # -- Step 5: Proper-motion correction ----------------------------------
    print("Step 5: Proper-motion correction (epoch 2016 -> 2024)")
    print("-" * 70)

    n_pm = 20
    ra_cat = catalog_radec[:n_pm, 0]
    dec_cat = catalog_radec[:n_pm, 1]
    pmra = rng.normal(0.0, 15.0, n_pm)    # mas/yr
    pmdec = rng.normal(0.0, 15.0, n_pm)   # mas/yr

    ra_new, dec_new = correct_proper_motion(
        ra_cat, dec_cat, pmra, pmdec, epoch_from=2016.0, epoch_to=2024.0,
    )

    delta_ra_mas = (ra_new - ra_cat) * 3600.0 * 1000.0
    delta_dec_mas = (dec_new - dec_cat) * 3600.0 * 1000.0
    print(f"  Stars corrected: {n_pm}")
    print(f"  RA  shift (mas): mean={delta_ra_mas.mean():.2f}  "
          f"std={delta_ra_mas.std():.2f}")
    print(f"  Dec shift (mas): mean={delta_dec_mas.mean():.2f}  "
          f"std={delta_dec_mas.std():.2f}")
    print()

    # -- Step 6: Fit SIP distortion ----------------------------------------
    print("Step 6: Fit SIP distortion polynomials")
    print("-" * 70)

    sip = fit_distortion_sip(
        centroids[:n_match, :2],
        catalog_radec[:n_match],
        crpix=tuple(solution["crpix"]),
        crval=tuple(solution["crval"]),
        cd_matrix=np.array(solution["cd_matrix"]),
        order=3,
    )

    print(f"  SIP order:        {sip['order']}")
    print(f"  A coefficients:   shape={sip['a_coeffs'].shape}")
    print(f"  B coefficients:   shape={sip['b_coeffs'].shape}")
    print(f"  Residual:         {sip['residual_arcsec']:.4f} arcsec")
    print()

    print("=" * 70)
    print("  OK  Tutorial complete -- astrometry tools explored.")
    print("=" * 70)


if __name__ == "__main__":
    main()
