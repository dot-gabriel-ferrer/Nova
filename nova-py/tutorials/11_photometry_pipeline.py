"""Tutorial 11: NOVA Photometry Pipeline.

Demonstrates multi-aperture photometry, zero-point calibration,
extinction correction, differential photometry, growth-curve analysis,
aperture corrections, and magnitude conversions.

Steps:
  1. Create a synthetic star-field image with known fluxes
  2. Multi-aperture photometry at several radii
  3. Zero-point calibration against catalog magnitudes
  4. Atmospheric extinction correction
  5. Differential photometry for a variable star
  6. Growth-curve analysis and aperture corrections
  7. Magnitude system conversions (AB <-> Vega, flux <-> mag)

Run:
    cd nova-py
    python tutorials/11_photometry_pipeline.py
"""

from __future__ import annotations

import numpy as np


def _synthetic_image(
    n_stars: int, shape: tuple[int, int], rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (image, sources) where sources is (N, 3): [x, y, flux]."""
    img = rng.normal(loc=300.0, scale=8.0, size=shape).astype(np.float64)
    sources = np.zeros((n_stars, 3))
    border = 30
    for i in range(n_stars):
        x = rng.uniform(border, shape[1] - border)
        y = rng.uniform(border, shape[0] - border)
        flux = rng.uniform(3000.0, 80000.0)
        sigma = 2.5
        yy, xx = np.ogrid[0:shape[0], 0:shape[1]]
        img += flux * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2))
        sources[i] = [x, y, flux]
    return img, sources


def main() -> None:
    from nova.photometry_pipeline import (
        ab_to_vega,
        aperture_correction,
        calibrate_zeropoint,
        differential_photometry,
        extinction_correct,
        flux_to_mag,
        growth_curve,
        mag_to_flux,
        multi_aperture_photometry,
        vega_to_ab,
    )

    print("=" * 70)
    print("  NOVA Tutorial 11: Photometry Pipeline")
    print("=" * 70)
    print()

    rng = np.random.default_rng(42)
    shape = (512, 512)
    n_stars = 40

    # -- Step 1: Synthetic image -------------------------------------------
    print("Step 1: Create synthetic image with %d stars" % n_stars)
    print("-" * 70)

    image, true_sources = _synthetic_image(n_stars, shape, rng)
    print(f"  Image shape: {image.shape}")
    print(f"  Pixel range: [{image.min():.1f}, {image.max():.1f}]")
    print()

    # -- Step 2: Multi-aperture photometry ---------------------------------
    print("Step 2: Multi-aperture photometry (radii = 4, 6, 8, 10 px)")
    print("-" * 70)

    positions = true_sources[:, :2]
    radii = [4.0, 6.0, 8.0, 10.0]
    phot = multi_aperture_photometry(
        image, positions, radii, gain=1.5, readnoise=5.0,
    )

    print(f"  Sources measured: {len(phot['x'])}")
    print(f"  Columns: {sorted(phot.keys())[:8]}...")
    # Show results for smallest and largest aperture
    r_small, r_large = radii[0], radii[-1]
    ks = f"flux_{r_small:.0f}" if f"flux_{r_small:.0f}" in phot else f"flux_{r_small}"
    kl = f"flux_{r_large:.0f}" if f"flux_{r_large:.0f}" in phot else f"flux_{r_large}"
    print(f"  Median flux (r={r_small}): {np.median(phot[ks]):.1f}")
    print(f"  Median flux (r={r_large}): {np.median(phot[kl]):.1f}")
    print(f"  Median sky:  {np.median(phot['sky']):.2f}")
    print()

    # -- Step 3: Zero-point calibration ------------------------------------
    print("Step 3: Zero-point calibration from catalog stars")
    print("-" * 70)

    mag_key = f"mag_{r_large:.0f}" if f"mag_{r_large:.0f}" in phot else f"mag_{r_large}"
    inst_mag = np.array(phot[mag_key])
    valid = np.isfinite(inst_mag)
    inst_mag_valid = inst_mag[valid]

    # Simulate catalog magnitudes: inst_mag + ZP + scatter
    true_zp = 25.0
    catalog_mag = inst_mag_valid + true_zp + rng.normal(0, 0.02, inst_mag_valid.shape)

    zp_result = calibrate_zeropoint(inst_mag_valid, catalog_mag, sigma_clip=3.0)
    print(f"  True zero-point:       {true_zp:.3f}")
    print(f"  Measured zero-point:   {zp_result['zeropoint']:.3f}")
    print(f"  Zero-point error:      {zp_result['zp_err']:.4f}")
    print(f"  Stars used / rejected: {zp_result['n_used']} / {zp_result['n_rejected']}")
    print()

    # -- Step 4: Extinction correction -------------------------------------
    print("Step 4: Atmospheric extinction correction")
    print("-" * 70)

    airmass = 1.3
    k_r = 0.12  # extinction coefficient for r-band (mag/airmass)
    cal_mag = inst_mag_valid + zp_result["zeropoint"]
    corrected_mag = extinction_correct(cal_mag, airmass=airmass, k_lambda=k_r)

    shift = np.median(cal_mag - corrected_mag)
    print(f"  Airmass:              {airmass}")
    print(f"  k_lambda (r-band):    {k_r} mag/airmass")
    print(f"  Correction applied:   {shift:.3f} mag")
    print(f"  Example: {cal_mag[0]:.3f} -> {corrected_mag[0]:.3f}")
    print()

    # -- Step 5: Differential photometry -----------------------------------
    print("Step 5: Differential photometry for a variable star")
    print("-" * 70)

    n_epochs = 50
    base_flux = 10000.0
    # Simulate sinusoidal variability
    time = np.linspace(0, 10, n_epochs)
    target_flux = base_flux * (1.0 + 0.05 * np.sin(2 * np.pi * time / 3.0))
    target_flux += rng.normal(0, 100, n_epochs)

    # Two comparison stars (steady)
    comp_flux = np.vstack([
        rng.normal(15000, 120, n_epochs),
        rng.normal(12000, 100, n_epochs),
    ])

    diff = differential_photometry(target_flux, comp_flux, comparison_mag=12.5)
    print(f"  Epochs:            {n_epochs}")
    print(f"  Comparison stars:  2 (ensemble)")
    print(f"  delta_mag range:   [{diff['delta_mag'].min():.4f}, "
          f"{diff['delta_mag'].max():.4f}]")
    print(f"  delta_mag std:     {diff['delta_mag'].std():.4f}")
    print(f"  Mean rel_flux:     {diff['rel_flux'].mean():.4f}")
    print()

    # -- Step 6: Growth curve and aperture correction ----------------------
    print("Step 6: Growth-curve analysis and aperture corrections")
    print("-" * 70)

    # Pick brightest source
    idx_bright = np.argmax(true_sources[:, 2])
    xb, yb = true_sources[idx_bright, 0], true_sources[idx_bright, 1]
    gc_radii = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0]
    gc = growth_curve(image, xb, yb, gc_radii)

    print(f"  Source at ({xb:.1f}, {yb:.1f})")
    print(f"  Radii: {gc_radii}")
    print(f"  Normalized flux at r=4:  {gc['normalized_flux'][2]:.4f}")
    print(f"  Normalized flux at r=15: {gc['normalized_flux'][-1]:.4f}")

    # Aperture correction between small and large apertures
    ms = f"mag_{r_small:.0f}" if f"mag_{r_small:.0f}" in phot else f"mag_{r_small}"
    ml = f"mag_{r_large:.0f}" if f"mag_{r_large:.0f}" in phot else f"mag_{r_large}"
    mag_s = np.array(phot[ms])
    mag_l = np.array(phot[ml])
    both_valid = np.isfinite(mag_s) & np.isfinite(mag_l)
    apcor = aperture_correction(mag_s[both_valid], mag_l[both_valid])
    print(f"  Aperture correction (r={r_small} -> r={r_large}): "
          f"{apcor['apcor']:.4f} +/- {apcor['apcor_err']:.4f} mag")
    print(f"  Stars used: {apcor['n_used']}")
    print()

    # -- Step 7: Magnitude conversions -------------------------------------
    print("Step 7: Magnitude conversions (AB <-> Vega, flux <-> mag)")
    print("-" * 70)

    test_mag = np.array([18.0, 19.0, 20.0, 21.0])
    zp = 25.0

    flux_vals = mag_to_flux(test_mag, zeropoint=zp)
    roundtrip_mag = flux_to_mag(flux_vals, zeropoint=zp)
    print(f"  mag -> flux -> mag round-trip:")
    for m, f, m2 in zip(test_mag, flux_vals, roundtrip_mag):
        print(f"    mag={m:.1f}  flux={f:.1f}  mag_back={m2:.4f}")

    ab_offset = 0.02  # approximate for V-band
    mag_vega = ab_to_vega(test_mag, ab_vega_offset=ab_offset)
    mag_ab_back = vega_to_ab(mag_vega, ab_vega_offset=ab_offset)
    print(f"  AB -> Vega -> AB (offset={ab_offset}):")
    print(f"    AB:   {test_mag}")
    print(f"    Vega: {mag_vega}")
    print(f"    AB:   {mag_ab_back}  (round-trip)")
    print()

    print("=" * 70)
    print("  OK  Tutorial complete -- photometry pipeline explored.")
    print("=" * 70)


if __name__ == "__main__":
    main()
