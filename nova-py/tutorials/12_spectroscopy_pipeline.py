"""Tutorial 12: NOVA Spectroscopy Pipeline.

Demonstrates 1-D spectral extraction, continuum fitting, telluric
correction, spectral stacking, redshift measurement, S/N estimation,
and equivalent-width measurement.

Steps:
  1. Generate a synthetic 2-D spectral image
  2. Optimal extraction of a 1-D spectrum
  3. Fit and subtract the continuum
  4. Telluric correction for atmospheric absorption
  5. Stack multiple spectra
  6. Measure redshift via cross-correlation
  7. Estimate signal-to-noise ratio
  8. Measure equivalent width of a spectral line

Run:
    cd nova-py
    python tutorials/12_spectroscopy_pipeline.py
"""

from __future__ import annotations

import numpy as np


def _make_2d_spectrum(
    n_spatial: int, n_spectral: int, rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a synthetic 2-D spectral image and trace.

    Returns (data_2d, trace) where trace gives the spatial centre at
    each column.
    """
    data = rng.normal(loc=10.0, scale=2.0,
                      size=(n_spatial, n_spectral)).astype(np.float64)
    trace = np.full(n_spectral, n_spatial / 2.0)
    # Add a slight curvature
    cols = np.arange(n_spectral)
    trace = trace + 2.0 * np.sin(2.0 * np.pi * cols / n_spectral)

    # Gaussian spatial profile with absorption and emission lines
    wavelength = np.linspace(4000, 7000, n_spectral)
    continuum_flux = 500.0 * np.ones(n_spectral)
    # Add an absorption line at 5000 A
    continuum_flux -= 200.0 * np.exp(-0.5 * ((wavelength - 5000) / 5) ** 2)
    # Add an emission line at 6563 A (H-alpha)
    continuum_flux += 300.0 * np.exp(-0.5 * ((wavelength - 6563) / 4) ** 2)

    for j in range(n_spectral):
        profile = np.exp(-0.5 * ((np.arange(n_spatial) - trace[j]) / 2.0) ** 2)
        data[:, j] += continuum_flux[j] * profile

    return data, trace


def main() -> None:
    from nova.spectroscopy_pipeline import (
        correct_telluric,
        equivalent_width,
        estimate_snr,
        fit_continuum,
        measure_redshift,
        optimal_extract,
        stack_spectra,
    )

    print("=" * 70)
    print("  NOVA Tutorial 12: Spectroscopy Pipeline")
    print("=" * 70)
    print()

    rng = np.random.default_rng(99)
    n_spatial, n_spectral = 40, 1000
    wavelength = np.linspace(4000.0, 7000.0, n_spectral)

    # -- Step 1: Synthetic 2-D spectrum ------------------------------------
    print("Step 1: Generate synthetic 2-D spectral image")
    print("-" * 70)

    data_2d, trace = _make_2d_spectrum(n_spatial, n_spectral, rng)
    print(f"  Image shape:     {data_2d.shape}  (spatial x spectral)")
    print(f"  Trace centre:    {trace[0]:.1f} .. {trace[n_spectral//2]:.1f} "
          f".. {trace[-1]:.1f} pixels")
    print()

    # -- Step 2: Optimal extraction ----------------------------------------
    print("Step 2: Optimal extraction (Horne 1986)")
    print("-" * 70)

    ext = optimal_extract(
        data_2d, trace, aperture_half=5, gain=1.5, readnoise=4.0,
    )

    flux_1d = ext["flux"]
    var_1d = ext["variance"]
    snr_1d = ext["snr"]
    print(f"  Extracted 1-D length: {len(flux_1d)}")
    print(f"  Flux range:           [{flux_1d.min():.1f}, {flux_1d.max():.1f}]")
    print(f"  Median S/N per pixel: {np.median(snr_1d):.1f}")
    print()

    # -- Step 3: Continuum fitting -----------------------------------------
    print("Step 3: Fit the continuum (polynomial, order=5)")
    print("-" * 70)

    cont = fit_continuum(wavelength, flux_1d, method="polynomial", order=5,
                         sigma_clip=3.0)
    residual = flux_1d - cont
    print(f"  Continuum range:  [{cont.min():.1f}, {cont.max():.1f}]")
    print(f"  Residual std:     {residual.std():.2f}")
    print(f"  Residual mean:    {residual.mean():.2f}")
    print()

    # -- Step 4: Telluric correction ---------------------------------------
    print("Step 4: Telluric correction for atmospheric absorption")
    print("-" * 70)

    # Simulate telluric transmission with absorption bands
    telluric = np.ones(n_spectral)
    # O2 band near 6870 A (approximate)
    telluric -= 0.4 * np.exp(-0.5 * ((wavelength - 6870) / 20) ** 2)
    # Water band near 5900 A
    telluric -= 0.25 * np.exp(-0.5 * ((wavelength - 5900) / 30) ** 2)
    telluric = np.clip(telluric, 0.05, 1.0)

    flux_observed = flux_1d * telluric
    flux_corrected = correct_telluric(flux_observed, telluric, min_transmission=0.1)

    max_diff = np.max(np.abs(flux_corrected - flux_1d))
    print(f"  Telluric min transmission: {telluric.min():.3f}")
    print(f"  Max correction applied:    {(1.0 / telluric.min()):.2f}x")
    print(f"  Max residual vs original:  {max_diff:.2f}")
    print()

    # -- Step 5: Stack multiple spectra ------------------------------------
    print("Step 5: Stack 5 noisy spectra (median combine)")
    print("-" * 70)

    wavelengths_list = [wavelength.copy() for _ in range(5)]
    fluxes_list = [flux_1d + rng.normal(0, 15, n_spectral) for _ in range(5)]

    stacked = stack_spectra(wavelengths_list, fluxes_list, method="median")
    single_noise = np.std(fluxes_list[0] - flux_1d)
    stacked_noise = np.std(stacked["flux"] - flux_1d)
    print(f"  Spectra combined: {stacked['n_combined']}")
    print(f"  Single noise:     {single_noise:.2f}")
    print(f"  Stacked noise:    {stacked_noise:.2f}")
    print(f"  Improvement:      {single_noise / stacked_noise:.2f}x")
    print()

    # -- Step 6: Measure redshift ------------------------------------------
    print("Step 6: Measure redshift via cross-correlation")
    print("-" * 70)

    z_true = 0.05
    template_wave = wavelength.copy()
    template_flux = flux_1d / cont  # continuum-normalized

    # Simulate an observed spectrum at z=0.05
    obs_wave = wavelength * (1.0 + z_true)
    obs_flux = np.interp(wavelength, obs_wave, template_flux,
                         left=1.0, right=1.0)
    obs_flux += rng.normal(0, 0.02, n_spectral)

    z_result = measure_redshift(
        wavelength, obs_flux,
        template_wave, template_flux,
        z_min=0.0, z_max=0.15, z_step=0.0005,
    )

    print(f"  True redshift:     z = {z_true:.4f}")
    print(f"  Measured redshift: z = {z_result['z_best']:.4f}")
    print(f"  Redshift error:    +/- {z_result['z_err']:.4f}")
    print(f"  CC peak value:     {z_result['cc_peak']:.4f}")
    print()

    # -- Step 7: Estimate S/N ----------------------------------------------
    print("Step 7: Estimate signal-to-noise ratio")
    print("-" * 70)

    snr_der = estimate_snr(flux_1d, method="der_snr")
    snr_var = estimate_snr(flux_1d, variance=var_1d, method="variance")
    print(f"  DER_SNR method:  median S/N = {snr_der['snr_median']:.1f}")
    print(f"  Variance method: median S/N = {snr_var['snr_median']:.1f}")
    print()

    # -- Step 8: Equivalent width ------------------------------------------
    print("Step 8: Measure equivalent width of absorption line at 5000 A")
    print("-" * 70)

    ew_result = equivalent_width(
        wavelength, flux_1d, cont,
        line_range=(4970.0, 5030.0),
        variance=var_1d,
    )

    print(f"  Equivalent width: {ew_result['ew']:.3f} A")
    print(f"  EW error:         {ew_result['ew_err']:.3f} A")
    print(f"  Line flux:        {ew_result['line_flux']:.2f}")
    print(f"  Line flux error:  {ew_result['line_flux_err']:.2f}")
    print()

    print("=" * 70)
    print("  OK  Tutorial complete -- spectroscopy pipeline explored.")
    print("=" * 70)


if __name__ == "__main__":
    main()
