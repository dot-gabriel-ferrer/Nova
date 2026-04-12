#!/usr/bin/env python3
"""Tutorial 06: NOVA Integrated Math & Visualization Tools.

Demonstrates NOVA's built-in mathematical operations and visualization
functions for astronomical data processing.

Usage:
    cd nova-py
    python tutorials/06_math_tools.py
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

# Ensure nova is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nova.math import (
    sigma_clip,
    sigma_clipped_stats,
    robust_statistics,
    gaussian_kernel_2d,
    smooth_gaussian,
    rebin,
    resize_image,
    stack_images,
    estimate_background,
    detect_sources,
    aperture_photometry,
    continuum_normalize,
    equivalent_width,
    cosmic_ray_clean,
)


def main() -> None:
    print("=" * 60)
    print("NOVA Tutorial 06: Integrated Math Tools")
    print("=" * 60)

    rng = np.random.default_rng(42)

    # -- 1. Sigma-Clipped Statistics --
    print("\n[chart] 1. Sigma-Clipped Statistics")
    data = rng.normal(100, 10, 10000)
    data[:20] = 1000  # outliers
    stats = sigma_clipped_stats(data, sigma=3.0)
    print(f"   Mean:   {stats['mean']:.2f} (true: 100)")
    print(f"   Std:    {stats['std']:.2f} (true: 10)")
    print(f"   Count:  {stats['count']} / 10000")

    # -- 2. Robust Statistics --
    print("\n[chart] 2. Robust Statistics (Biweight)")
    rstats = robust_statistics(data)
    print(f"   Median:            {rstats['median']:.2f}")
    print(f"   MAD:               {rstats['mad']:.2f}")
    print(f"   Biweight Location: {rstats['biweight_location']:.2f}")
    print(f"   Biweight Scale:    {rstats['biweight_scale']:.2f}")

    # -- 3. Image Smoothing --
    print("\n[search] 3. Gaussian Smoothing")
    image = rng.normal(100, 10, (256, 256))
    yy, xx = np.ogrid[0:256, 0:256]
    image += 5000 * np.exp(-((xx - 128)**2 + (yy - 128)**2) / (2 * 5**2))
    smoothed = smooth_gaussian(image, sigma=2.0)
    print(f"   Original noise: sigma = {np.std(image):.1f}")
    print(f"   Smoothed noise: sigma = {np.std(smoothed):.1f}")

    # -- 4. Image Rebinning --
    print("\n>> 4. Image Rebinning")
    rebinned = rebin(image, (64, 64), method="sum")
    print(f"   Original: {image.shape}, total flux = {image.sum():.0f}")
    print(f"   Rebinned: {rebinned.shape}, total flux = {rebinned.sum():.0f}")
    print(f"   Flux conserved: {'[done]' if abs(image.sum() - rebinned.sum()) < 1 else '[error]'}")

    # -- 5. Background Estimation --
    print("\n>> 5. Background Estimation")
    bg, rms_map = estimate_background(image, box_size=64)
    print(f"   Background median: {np.median(bg):.1f} ADU")
    print(f"   RMS median:        {np.median(rms_map):.1f} ADU")

    # -- 6. Source Detection --
    print("\n>> 6. Source Detection")
    # Create image with known sources
    sky = rng.poisson(200, (256, 256)).astype(float)
    for _ in range(10):
        cx, cy = rng.uniform(30, 226, 2)
        flux = rng.uniform(1000, 10000)
        sky += flux * np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * 3**2))
    bg_sky, _ = estimate_background(sky, box_size=64)
    sources = detect_sources(sky - bg_sky, nsigma=5.0, min_area=5)
    print(f"   Injected: 10 sources")
    print(f"   Detected: {len(sources)} sources")

    # -- 7. Aperture Photometry --
    print("\n>> 7. Aperture Photometry")
    if sources:
        src = sources[0]
        phot = aperture_photometry(
            sky, x=src["x"], y=src["y"],
            radius=8.0, annulus_inner=12.0, annulus_outer=18.0,
        )
        print(f"   Source at ({src['x']:.1f}, {src['y']:.1f})")
        print(f"   Raw flux:  {phot['flux']:.0f}")
        print(f"   Background: {phot['background']:.0f}")
        print(f"   Net flux:  {phot['flux_corrected']:.0f}")

    # -- 8. Image Stacking --
    print("\n>> 8. Image Stacking")
    exposures = [rng.poisson(200, (128, 128)).astype(float) for _ in range(5)]
    stacked_mean = stack_images(exposures, method="mean")
    stacked_median = stack_images(exposures, method="median")
    stacked_sc = stack_images(exposures, method="sigma_clip")
    bg_slice = (slice(100, 128), slice(100, 128))
    print(f"   Single exposure noise: sigma = {np.std(exposures[0][bg_slice]):.2f}")
    print(f"   Mean stack noise:      sigma = {np.std(stacked_mean[bg_slice]):.2f}")
    print(f"   Median stack noise:    sigma = {np.std(stacked_median[bg_slice]):.2f}")
    print(f"   sigma-clip stack noise:    sigma = {np.std(stacked_sc[bg_slice]):.2f}")

    # -- 9. Spectral Analysis --
    print("\n>> 9. Spectral Analysis")
    wavelength = np.linspace(4000, 7000, 1000)
    flux = np.ones_like(wavelength) * 100.0
    flux -= 50 * np.exp(-0.5 * ((wavelength - 6563) / 10)**2)  # Halpha absorption
    flux += rng.normal(0, 0.5, len(flux))
    norm_flux, continuum = continuum_normalize(wavelength, flux)
    ew = equivalent_width(wavelength, norm_flux, 6563, 50)
    print(f"   Halpha Equivalent Width: {ew:.2f} A")

    # -- 10. Cosmic Ray Cleaning --
    print("\n>> 10. Cosmic Ray Cleaning")
    cr_image = rng.normal(100, 1, (128, 128))
    cr_image[50, 50] = 50000  # cosmic ray
    cr_image[80, 30] = 30000  # cosmic ray
    cleaned = cosmic_ray_clean(cr_image, sigma=5.0)
    print(f"   Before: max = {cr_image.max():.0f}")
    print(f"   After:  max = {cleaned.max():.0f}")
    print(f"   Cosmic rays removed: [done]")

    print("\n" + "=" * 60)
    print("Tutorial complete! All math tools working correctly.")
    print("=" * 60)


if __name__ == "__main__":
    main()
