"""Tests for Phase 3: Advanced Math and Analysis (v0.4.0).

Covers image_processing, photometry, spectral, coords, and catalog modules.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest


# =====================================================================
#  3.1 -- Image Processing
# =====================================================================

class TestPSFModelling:
    """Tests for PSF model functions."""

    def test_moffat_2d_peak(self):
        from nova.image_processing import moffat_2d
        val = moffat_2d(0.0, 0.0, amplitude=100.0, x0=0.0, y0=0.0,
                        alpha=3.0, beta=2.5)
        assert abs(val - 100.0) < 1e-10

    def test_moffat_2d_off_center(self):
        from nova.image_processing import moffat_2d
        val = moffat_2d(5.0, 5.0, amplitude=100.0, x0=0.0, y0=0.0,
                        alpha=3.0, beta=2.5)
        assert val < 100.0
        assert val > 0.0

    def test_gaussian_2d_peak(self):
        from nova.image_processing import gaussian_2d
        val = gaussian_2d(5.0, 5.0, amplitude=200.0, x0=5.0, y0=5.0,
                          sigma_x=2.0, sigma_y=2.0)
        assert abs(val - 200.0) < 1e-10

    def test_gaussian_2d_array(self):
        from nova.image_processing import gaussian_2d
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0, 1.0, 2.0])
        vals = gaussian_2d(x, y, 1.0, 1.0, 1.0, 1.0, 1.0)
        assert vals.shape == (3,)

    def test_multi_gaussian_psf(self):
        from nova.image_processing import multi_gaussian_psf
        val = multi_gaussian_psf(0.0, 0.0, amplitudes=[1.0, 0.5],
                                 x0=0.0, y0=0.0, sigmas=[1.0, 3.0])
        assert val > 1.0  # sum of both components at center

    def test_fit_moffat_psf(self):
        from nova.image_processing import moffat_2d, fit_moffat_psf
        y, x = np.mgrid[0:41, 0:41]
        img = moffat_2d(x.astype(float), y.astype(float),
                        amplitude=500.0, x0=20.0, y0=20.0,
                        alpha=3.0, beta=2.5)
        img += np.random.default_rng(42).normal(0, 1.0, img.shape)
        result = fit_moffat_psf(img, 20.0, 20.0, box_size=21)
        assert abs(result["x0"] - 20.0) < 1.0
        assert abs(result["y0"] - 20.0) < 1.0
        assert result["fwhm"] > 0

    def test_fit_gaussian_psf(self):
        from nova.image_processing import gaussian_2d, fit_gaussian_psf
        y, x = np.mgrid[0:41, 0:41]
        img = gaussian_2d(x.astype(float), y.astype(float),
                          amplitude=300.0, x0=20.0, y0=20.0,
                          sigma_x=3.0, sigma_y=3.0)
        img += np.random.default_rng(42).normal(0, 0.5, img.shape)
        result = fit_gaussian_psf(img, 20.0, 20.0, box_size=21)
        assert abs(result["x0"] - 20.0) < 1.0
        assert abs(result["y0"] - 20.0) < 1.0

    def test_generate_psf_image(self):
        from nova.image_processing import generate_psf_image
        params = {"amplitude": 1.0, "x0": 16.0, "y0": 16.0,
                  "alpha": 3.0, "beta": 2.5}
        img = generate_psf_image((33, 33), params, model_type="moffat")
        assert img.shape == (33, 33)
        assert img[16, 16] == img.max()


class TestImageRegistration:
    """Tests for image alignment functions."""

    def test_compute_shift_zero(self):
        from nova.image_processing import compute_shift
        rng = np.random.default_rng(42)
        img = rng.normal(100, 10, (64, 64))
        dy, dx = compute_shift(img, img)
        assert abs(dx) < 0.5
        assert abs(dy) < 0.5

    def test_compute_shift_known(self):
        from nova.image_processing import compute_shift
        rng = np.random.default_rng(42)
        ref = np.zeros((64, 64))
        ref[25:35, 25:35] = 100.0
        target = np.zeros((64, 64))
        target[27:37, 28:38] = 100.0  # shifted by (2,3)
        dy, dx = compute_shift(ref, target, upsample_factor=1)
        # The shift direction convention may vary; check magnitude
        assert abs(abs(dy) - 2) < 1.5
        assert abs(abs(dx) - 3) < 1.5

    def test_feature_align(self):
        from nova.image_processing import feature_align
        ref = np.zeros((64, 64))
        ref[30, 30] = 100.0
        ref[30, 50] = 100.0
        ref[50, 30] = 100.0
        target = ref.copy()
        aligned, shift = feature_align(ref, target)
        assert aligned.shape == ref.shape


class TestImageSubtraction:
    """Tests for image subtraction."""

    def test_subtract_scaled_identical(self):
        from nova.image_processing import subtract_scaled
        rng = np.random.default_rng(42)
        img = rng.normal(100, 10, (32, 32))
        diff = subtract_scaled(img, img)
        assert np.allclose(diff, 0, atol=1e-10)

    def test_subtract_scaled_with_factor(self):
        from nova.image_processing import subtract_scaled
        rng = np.random.default_rng(42)
        ref = rng.normal(100, 10, (32, 32))
        science = ref * 2.0
        diff = subtract_scaled(science, ref, scale=2.0)
        assert np.allclose(diff, 0, atol=1e-10)


class TestCalibration:
    """Tests for calibration helpers."""

    def test_subtract_bias(self):
        from nova.image_processing import subtract_bias
        data = np.full((32, 32), 1100.0)
        bias = np.full((32, 32), 100.0)
        result = subtract_bias(data, bias)
        assert np.allclose(result, 1000.0)

    def test_apply_flat(self):
        from nova.image_processing import apply_flat
        data = np.full((32, 32), 1000.0)
        flat = np.full((32, 32), 0.5)
        result = apply_flat(data, flat)
        assert np.allclose(result, 2000.0)

    def test_apply_flat_min_clamp(self):
        from nova.image_processing import apply_flat
        data = np.full((4, 4), 100.0)
        flat = np.zeros((4, 4))
        result = apply_flat(data, flat, min_flat=0.01)
        assert np.all(np.isfinite(result))

    def test_subtract_dark(self):
        from nova.image_processing import subtract_dark
        data = np.full((8, 8), 500.0)
        dark = np.full((8, 8), 100.0)
        result = subtract_dark(data, dark, exposure_data=120.0,
                               exposure_dark=60.0)
        assert np.allclose(result, 300.0)

    def test_correct_overscan(self):
        from nova.image_processing import correct_overscan
        data = np.full((32, 40), 1000.0)
        data[:, 32:] = 100.0  # overscan region
        result = correct_overscan(data, overscan_region=(32, 40), axis=1)
        assert result.shape == data.shape


class TestBadPixels:
    """Tests for bad pixel handling."""

    def test_interpolate_bad_pixels(self):
        from nova.image_processing import interpolate_bad_pixels
        data = np.ones((16, 16)) * 100.0
        mask = np.zeros((16, 16), dtype=bool)
        mask[8, 8] = True
        data[8, 8] = 0.0
        result = interpolate_bad_pixels(data, mask)
        assert abs(result[8, 8] - 100.0) < 1.0

    def test_build_bad_pixel_mask(self):
        from nova.image_processing import build_bad_pixel_mask
        rng = np.random.default_rng(42)
        darks = rng.normal(100, 5, (3, 32, 32))
        darks[0, 10, 10] = 10000  # hot pixel
        flats = np.ones((3, 32, 32))
        flats[0, 15, 15] = 0.0  # dead pixel
        mask = build_bad_pixel_mask(darks, flats)
        assert mask.dtype == bool
        assert mask.shape == (32, 32)


class TestFringeCorrection:
    """Tests for fringe correction."""

    def test_build_fringe_map(self):
        from nova.image_processing import build_fringe_map
        rng = np.random.default_rng(42)
        fringe = np.sin(np.linspace(0, 4 * np.pi, 32))[None, :] * np.ones((32, 1))
        images = [rng.normal(100, 5, (32, 32)) + fringe for _ in range(5)]
        fmap = build_fringe_map(images)
        assert fmap.shape == (32, 32)

    def test_subtract_fringe(self):
        from nova.image_processing import subtract_fringe
        data = np.ones((32, 32)) * 100.0
        fringe = np.ones((32, 32)) * 10.0
        result = subtract_fringe(data, fringe, scale=1.0)
        assert np.allclose(result, 90.0)


# =====================================================================
#  3.2 -- Photometry
# =====================================================================

class TestPSFPhotometry:
    """Tests for PSF photometry."""

    def test_psf_photometry_single(self):
        from nova.photometry import psf_photometry
        rng = np.random.default_rng(42)
        psf = np.zeros((11, 11))
        y, x = np.mgrid[0:11, 0:11]
        psf = np.exp(-0.5 * ((x - 5)**2 + (y - 5)**2) / 2.0**2)
        psf /= psf.sum()
        img = np.zeros((64, 64))
        flux_true = 1000.0
        img[27:38, 27:38] += flux_true * psf
        img += rng.normal(0, 1, img.shape)
        result = psf_photometry(img, [(32.0, 32.0)], psf, box_size=11)
        assert len(result) == 1
        assert "flux" in result[0]
        assert result[0]["flux"] > 0

    def test_iterative_psf_subtract(self):
        from nova.photometry import iterative_psf_subtract
        psf = np.zeros((11, 11))
        y, x = np.mgrid[0:11, 0:11]
        psf = np.exp(-0.5 * ((x - 5)**2 + (y - 5)**2) / 2.0**2)
        psf /= psf.sum()
        img = np.zeros((64, 64))
        img[27:38, 27:38] += 500 * psf
        residual, table = iterative_psf_subtract(img, [(32.0, 32.0)], psf)
        assert residual.shape == img.shape
        assert len(table) == 1


class TestExtendedPhotometry:
    """Tests for extended source photometry."""

    def test_petrosian_radius(self):
        from nova.photometry import petrosian_radius
        radii = np.arange(1, 30, dtype=float)
        profile = 100.0 * np.exp(-radii / 5.0)
        r_p = petrosian_radius(profile, radii)
        assert r_p > 0

    def test_kron_radius(self):
        from nova.photometry import kron_radius
        y, x = np.mgrid[0:64, 0:64]
        img = 1000.0 * np.exp(-0.5 * ((x - 32)**2 + (y - 32)**2) / 5.0**2)
        r_k = kron_radius(img, 32.0, 32.0)
        assert r_k > 0

    def test_radial_profile(self):
        from nova.photometry import radial_profile
        y, x = np.mgrid[0:64, 0:64]
        img = 1000.0 * np.exp(-0.5 * ((x - 32)**2 + (y - 32)**2) / 5.0**2)
        result = radial_profile(img, 32.0, 32.0, max_radius=20)
        assert "radii" in result
        assert "profile" in result
        assert len(result["radii"]) > 0

    def test_isophotal_photometry(self):
        from nova.photometry import isophotal_photometry
        y, x = np.mgrid[0:64, 0:64]
        img = 1000.0 * np.exp(-0.5 * ((x - 32)**2 + (y - 32)**2) / 5.0**2)
        result = isophotal_photometry(img, 32.0, 32.0, threshold=10.0)
        assert result["flux"] > 0
        assert result["area"] > 0


class TestPhotometricCalibration:
    """Tests for zero-point and calibration."""

    def test_compute_zeropoint(self):
        from nova.photometry import compute_zeropoint
        inst = np.array([-10.0, -11.0, -9.5, -10.5])
        cat = np.array([15.0, 14.0, 15.5, 14.5])
        result = compute_zeropoint(inst, cat)
        assert "zeropoint" in result
        assert abs(result["zeropoint"] - 25.0) < 0.5

    def test_apply_zeropoint(self):
        from nova.photometry import apply_zeropoint
        inst = np.array([-10.0, -11.0])
        cal = apply_zeropoint(inst, 25.0)
        assert np.allclose(cal, [15.0, 14.0])

    def test_extinction_correct(self):
        from nova.photometry import extinction_correct
        mags = np.array([15.0, 16.0])
        corrected = extinction_correct(mags, airmass=1.5,
                                        extinction_coeff=0.15)
        assert np.all(corrected < mags)

    def test_color_term(self):
        from nova.photometry import color_term_correct
        mags = np.array([15.0, 16.0])
        colors = np.array([0.5, 1.0])
        corrected = color_term_correct(mags, colors, color_coeff=0.05)
        assert corrected.shape == mags.shape


class TestApertureCorrections:
    """Tests for aperture corrections."""

    def test_curve_of_growth(self):
        from nova.photometry import curve_of_growth
        y, x = np.mgrid[0:64, 0:64]
        img = 1000.0 * np.exp(-0.5 * ((x - 32)**2 + (y - 32)**2) / 3.0**2)
        result = curve_of_growth(img, 32.0, 32.0, max_radius=20)
        assert "radii" in result
        assert "enclosed_flux" in result
        # flux should be monotonically increasing
        assert np.all(np.diff(result["enclosed_flux"]) >= -1e-10)


class TestCrowdedField:
    """Tests for crowded field photometry."""

    def test_find_neighbors(self):
        from nova.photometry import find_neighbors
        positions = [(0.0, 0.0), (1.0, 0.0), (10.0, 10.0)]
        pairs = find_neighbors(positions, radius=2.0)
        assert len(pairs) >= 1  # (0,1) pair should match

    def test_completeness_test(self):
        from nova.photometry import completeness_test
        rng = np.random.default_rng(42)
        img = rng.normal(100, 5, (64, 64))
        psf = np.zeros((11, 11))
        y, x = np.mgrid[0:11, 0:11]
        psf = np.exp(-0.5 * ((x - 5)**2 + (y - 5)**2) / 2.0**2)
        psf /= psf.sum()
        result = completeness_test(img, psf, [(32.0, 32.0)],
                                    fluxes=[1000.0], n_trials=5)
        assert "recovery_fraction" in result


# =====================================================================
#  3.3 -- Spectral
# =====================================================================

class TestWavelengthCalibration:
    """Tests for wavelength calibration."""

    def test_identify_lines(self):
        from nova.spectral import identify_lines
        rng = np.random.default_rng(42)
        spec = rng.normal(0, 1, 200)
        spec[50] = 20.0
        spec[100] = 25.0
        spec[150] = 18.0
        positions = identify_lines(spec, threshold=5.0, min_distance=10)
        assert len(positions) >= 2

    def test_fit_wavelength_solution(self):
        from nova.spectral import fit_wavelength_solution
        pixels = np.array([50.0, 100.0, 150.0, 200.0])
        waves = np.array([4000.0, 5000.0, 6000.0, 7000.0])
        result = fit_wavelength_solution(pixels, waves, order=1)
        assert "coefficients" in result
        assert result["residual_rms"] < 1.0

    def test_apply_wavelength_solution(self):
        from nova.spectral import (fit_wavelength_solution,
                                    apply_wavelength_solution)
        pixels = np.array([0.0, 100.0, 200.0])
        waves = np.array([4000.0, 5000.0, 6000.0])
        result = fit_wavelength_solution(pixels, waves, order=1)
        w = apply_wavelength_solution(np.array([50.0]),
                                       result["coefficients"])
        assert abs(w[0] - 4500.0) < 10.0

    def test_calibrate_spectrum(self):
        from nova.spectral import calibrate_spectrum
        rng = np.random.default_rng(42)
        flux = rng.normal(100, 5, 300)
        flux[50] = 200.0
        flux[150] = 220.0
        flux[250] = 210.0
        pixel_pos = np.array([50.0, 150.0, 250.0])
        known_waves = np.array([4000.0, 5000.0, 6000.0])
        wavelengths, cal_flux = calibrate_spectrum(flux, pixel_pos,
                                                    known_waves, order=1)
        assert len(wavelengths) == len(flux)


class TestSkySubtraction:
    """Tests for sky subtraction."""

    def test_extract_sky(self):
        from nova.spectral import extract_sky
        rng = np.random.default_rng(42)
        image = rng.normal(100, 5, (50, 200))
        sky = extract_sky(image, [(0, 10), (40, 50)])
        assert sky.shape == (200,)

    def test_subtract_sky_1d(self):
        from nova.spectral import subtract_sky_1d
        spec = np.ones(100) * 150.0
        sky = np.ones(100) * 50.0
        result = subtract_sky_1d(spec, sky)
        assert np.allclose(result, 100.0)

    def test_subtract_sky_2d(self):
        from nova.spectral import subtract_sky_2d
        rng = np.random.default_rng(42)
        img = rng.normal(100, 2, (50, 200))
        img[20:30, :] += 500  # source
        result = subtract_sky_2d(img, [(0, 10), (40, 50)])
        assert result.shape == img.shape


class TestRadialVelocity:
    """Tests for radial velocity measurement."""

    def test_cross_correlate_rv(self):
        from nova.spectral import cross_correlate_rv
        wavelengths = np.linspace(4000, 7000, 1000)
        template = np.ones(1000)
        template[400] = 0.5  # absorption line at 5200A
        template[500] = 0.3
        observed = template.copy()  # zero velocity
        result = cross_correlate_rv(observed, template, wavelengths,
                                     v_range=(-200, 200), v_step=5.0)
        assert "velocity" in result
        assert abs(result["velocity"]) < 50  # should be near zero

    def test_doppler_shift(self):
        from nova.spectral import doppler_shift
        w = np.array([5000.0, 6000.0])
        shifted = doppler_shift(w, 100.0)  # 100 km/s
        assert np.all(shifted > w)

    def test_barycentric_correction(self):
        from nova.spectral import barycentric_correction
        w = np.array([5000.0, 6000.0])
        corrected = barycentric_correction(w, 15.0)
        assert corrected.shape == w.shape


class TestEmissionLineFitting:
    """Tests for emission line fitting."""

    def test_gaussian_line(self):
        from nova.spectral import gaussian_line
        w = np.linspace(4990, 5010, 100)
        profile = gaussian_line(w, amplitude=10.0, center=5000.0,
                                 sigma=2.0)
        peak_idx = np.argmax(profile)
        assert abs(w[peak_idx] - 5000.0) < 0.5

    def test_voigt_line(self):
        from nova.spectral import voigt_line
        w = np.linspace(4990, 5010, 100)
        profile = voigt_line(w, amplitude=10.0, center=5000.0,
                              sigma=1.0, gamma=1.0)
        assert np.max(profile) > 0

    def test_fit_emission_line(self):
        from nova.spectral import gaussian_line, fit_emission_line
        w = np.linspace(4980, 5020, 200)
        flux = gaussian_line(w, 50.0, 5000.0, 2.0, continuum=100.0)
        rng = np.random.default_rng(42)
        flux += rng.normal(0, 1, len(flux))
        result = fit_emission_line(w, flux, center_guess=5000.0)
        assert abs(result["center"] - 5000.0) < 1.0
        assert result["amplitude"] > 0

    def test_fit_multi_lines(self):
        from nova.spectral import gaussian_line, fit_multi_lines
        w = np.linspace(4900, 5100, 500)
        flux = (gaussian_line(w, 30.0, 4960.0, 2.0) +
                gaussian_line(w, 50.0, 5007.0, 2.0) + 100.0)
        rng = np.random.default_rng(42)
        flux += rng.normal(0, 0.5, len(flux))
        result = fit_multi_lines(w, flux, [4960.0, 5007.0])
        assert len(result) == 2


class TestEchelleSupport:
    """Tests for echelle spectrum support."""

    def test_trace_order(self):
        from nova.spectral import trace_order
        img = np.zeros((100, 200))
        for col in range(200):
            row = 50 + int(2 * np.sin(col / 50.0))
            img[max(0, row - 1):min(100, row + 2), col] = 100.0
        coeffs = trace_order(img, start_col=100, start_row=50)
        assert len(coeffs) >= 2  # at least linear

    def test_merge_orders(self):
        from nova.spectral import merge_orders
        w1 = np.linspace(4000, 4600, 100)
        w2 = np.linspace(4500, 5100, 100)
        f1 = np.ones(100) * 100.0
        f2 = np.ones(100) * 100.0
        mw, mf = merge_orders([f1, f2], [w1, w2])
        assert len(mw) > 0
        assert len(mf) == len(mw)


# =====================================================================
#  3.4 -- Coordinate Transforms
# =====================================================================

class TestSIPDistortion:
    """Tests for SIP distortion support."""

    def test_sip_forward_no_distortion(self):
        from nova.coords import sip_forward
        u, v = sip_forward(10.0, 20.0, {}, {}, 0, 0)
        assert abs(u - 10.0) < 1e-10
        assert abs(v - 20.0) < 1e-10

    def test_sip_forward_with_coeffs(self):
        from nova.coords import sip_forward
        a = {(2, 0): 1e-5, (0, 2): 2e-5}
        b = {(1, 1): 1e-5}
        u, v = sip_forward(100.0, 100.0, a, b, 2, 2)
        assert u != 100.0  # distortion applied
        assert v != 100.0

    def test_sip_roundtrip(self):
        from nova.coords import sip_forward, sip_inverse
        a = {(2, 0): 1e-6}
        b = {(0, 2): 1e-6}
        ap = {(2, 0): -1e-6}
        bp = {(0, 2): -1e-6}
        u, v = sip_forward(50.0, 50.0, a, b, 2, 2)
        u2, v2 = sip_inverse(u, v, ap, bp, 2, 2)
        # Approximate roundtrip (inverse coeffs are approximate)
        assert abs(u2 - 50.0) < 1.0
        assert abs(v2 - 50.0) < 1.0


class TestTPVDistortion:
    """Tests for TPV distortion."""

    def test_tpv_forward_identity(self):
        from nova.coords import tpv_forward
        # PV with only linear terms (identity-like)
        pv1 = {1: 1.0}  # xi term
        pv2 = {1: 1.0}  # eta term
        xi_out, eta_out = tpv_forward(0.01, 0.02, pv1, pv2)
        assert abs(xi_out - 0.01) < 1e-6
        assert abs(eta_out - 0.02) < 1e-6


class TestFrameTransforms:
    """Tests for coordinate frame transforms."""

    def test_equatorial_to_galactic_center(self):
        from nova.coords import equatorial_to_galactic
        # Galactic center: roughly RA=266.405, Dec=-28.936
        l, b = equatorial_to_galactic(266.405, -28.936)
        assert abs(l) < 2.0 or abs(l - 360) < 2.0
        assert abs(b) < 2.0

    def test_galactic_roundtrip(self):
        from nova.coords import equatorial_to_galactic, galactic_to_equatorial
        ra, dec = 120.0, 45.0
        l, b = equatorial_to_galactic(ra, dec)
        ra2, dec2 = galactic_to_equatorial(l, b)
        assert abs(ra2 - ra) < 0.01
        assert abs(dec2 - dec) < 0.01

    def test_ecliptic_roundtrip(self):
        from nova.coords import equatorial_to_ecliptic, ecliptic_to_equatorial
        ra, dec = 180.0, 30.0
        lon, lat = equatorial_to_ecliptic(ra, dec)
        ra2, dec2 = ecliptic_to_equatorial(lon, lat)
        assert abs(ra2 - ra) < 0.01
        assert abs(dec2 - dec) < 0.01

    def test_precess(self):
        from nova.coords import precess
        ra, dec = precess(180.0, 45.0, 2000.0, 2050.0)
        assert abs(ra - 180.0) < 2.0
        assert abs(dec - 45.0) < 2.0


class TestAngularSeparation:
    """Tests for angular separation and cross-matching."""

    def test_angular_separation_zero(self):
        from nova.coords import angular_separation
        sep = angular_separation(10.0, 20.0, 10.0, 20.0)
        assert abs(sep) < 1e-10

    def test_angular_separation_known(self):
        from nova.coords import angular_separation
        # 1 degree apart along RA at dec=0
        sep = angular_separation(10.0, 0.0, 11.0, 0.0)
        assert abs(sep - 1.0) < 0.01

    def test_angular_separation_poles(self):
        from nova.coords import angular_separation
        sep = angular_separation(0.0, 90.0, 180.0, 89.0)
        assert abs(sep - 1.0) < 0.01

    def test_cross_match(self):
        from nova.coords import cross_match
        ra1 = np.array([10.0, 20.0, 30.0])
        dec1 = np.array([10.0, 20.0, 30.0])
        ra2 = np.array([10.001, 20.001, 50.0])
        dec2 = np.array([10.001, 20.001, 50.0])
        idx1, idx2, seps = cross_match(ra1, dec1, ra2, dec2, radius=10.0)
        assert len(idx1) >= 2  # first two should match

    def test_compute_astrometric_residuals(self):
        from nova.coords import compute_astrometric_residuals
        rng = np.random.default_rng(42)
        ra = rng.uniform(10, 11, 20)
        dec = rng.uniform(20, 21, 20)
        offset = 0.5 / 3600.0  # 0.5 arcsec offset
        result = compute_astrometric_residuals(
            ra + offset, dec + offset, ra, dec, match_radius=5.0)
        assert result["n_matched"] > 0


class TestTANProjection:
    """Tests for TAN projection helpers."""

    def test_tan_roundtrip(self):
        from nova.coords import tan_project, tan_deproject
        crval_ra, crval_dec = 180.0, 45.0
        ra, dec = 180.1, 45.05
        xi, eta = tan_project(ra, dec, crval_ra, crval_dec)
        ra2, dec2 = tan_deproject(xi, eta, crval_ra, crval_dec)
        assert abs(ra2 - ra) < 1e-8
        assert abs(dec2 - dec) < 1e-8


# =====================================================================
#  3.5 -- Catalog Operations
# =====================================================================

class TestCatalogCrossMatch:
    """Tests for catalog cross-matching."""

    def test_cross_match_catalogs(self):
        from nova.catalog import cross_match_catalogs
        ra1 = np.array([10.0, 20.0, 30.0])
        dec1 = np.array([10.0, 20.0, 30.0])
        ra2 = np.array([10.0001, 20.0001, 50.0])
        dec2 = np.array([10.0001, 20.0001, 50.0])
        result = cross_match_catalogs(ra1, dec1, ra2, dec2, radius=5.0)
        assert result["n_matched"] >= 2

    def test_self_match(self):
        from nova.catalog import self_match
        ra = np.array([10.0, 10.0003, 20.0])
        dec = np.array([10.0, 10.0003, 20.0])
        idx1, idx2, seps = self_match(ra, dec, radius=5.0)
        assert len(idx1) >= 1

    def test_nearest_neighbor(self):
        from nova.catalog import nearest_neighbor
        ra1 = np.array([10.0, 20.0])
        dec1 = np.array([10.0, 20.0])
        ra2 = np.array([10.001, 20.001, 50.0])
        dec2 = np.array([10.001, 20.001, 50.0])
        result = nearest_neighbor(ra1, dec1, ra2, dec2)
        assert result["idx2"][0] == 0
        assert result["idx2"][1] == 1


class TestSpatialQueries:
    """Tests for spatial queries."""

    def test_cone_search(self):
        from nova.catalog import cone_search
        ra = np.array([10.0, 10.001, 50.0])
        dec = np.array([10.0, 10.001, 50.0])
        mask, seps = cone_search(ra, dec, 10.0, 10.0, radius=10.0)
        assert mask[0] and mask[1]
        assert not mask[2]

    def test_box_search(self):
        from nova.catalog import box_search
        ra = np.array([5.0, 15.0, 355.0])
        dec = np.array([10.0, 10.0, 10.0])
        mask = box_search(ra, dec, 0.0, 20.0, 5.0, 15.0)
        assert mask[0] and mask[1]

    def test_box_search_wraparound(self):
        from nova.catalog import box_search
        ra = np.array([1.0, 359.0, 180.0])
        dec = np.array([0.0, 0.0, 0.0])
        mask = box_search(ra, dec, 350.0, 10.0, -5.0, 5.0)
        assert mask[0] and mask[1]
        assert not mask[2]


class TestVOTable:
    """Tests for VOTable I/O."""

    def test_write_read_votable(self):
        from nova.catalog import write_votable, read_votable
        with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as f:
            path = f.name
        try:
            columns = {
                "ra": np.array([10.0, 20.0, 30.0]),
                "dec": np.array([10.0, 20.0, 30.0]),
                "mag": np.array([15.0, 16.0, 17.0]),
            }
            write_votable(path, columns, metadata={"description": "Test"})
            result = read_votable(path)
            assert "columns" in result
            assert len(result["columns"]["ra"]) == 3
        finally:
            os.unlink(path)


class TestSAMPClient:
    """Tests for SAMP client."""

    def test_samp_client_init(self):
        from nova.catalog import SAMPClient
        client = SAMPClient(name="TestNOVA")
        assert not client.is_connected

    def test_build_message(self):
        from nova.catalog import SAMPClient
        client = SAMPClient()
        msg = client.build_message("table.load.votable",
                                    {"url": "http://example.com/t.xml"})
        assert msg["samp.mtype"] == "table.load.votable"

    def test_notify_table_load(self):
        from nova.catalog import SAMPClient
        client = SAMPClient()
        msg = client.notify_table_load("http://example.com/table.xml",
                                        name="test")
        assert "samp.mtype" in msg


class TestCatalogStatistics:
    """Tests for catalog statistics."""

    def test_source_density(self):
        from nova.catalog import source_density
        rng = np.random.default_rng(42)
        ra = rng.uniform(10, 11, 100)
        dec = rng.uniform(20, 21, 100)
        density = source_density(ra, dec)
        assert density > 0

    def test_magnitude_histogram(self):
        from nova.catalog import magnitude_histogram
        rng = np.random.default_rng(42)
        mags = rng.normal(20, 2, 500)
        result = magnitude_histogram(mags)
        assert "counts" in result
        assert "completeness_mag" in result

    def test_number_counts(self):
        from nova.catalog import number_counts
        rng = np.random.default_rng(42)
        mags = rng.normal(20, 2, 500)
        result = number_counts(mags, area_deg2=1.0)
        assert "mag_centers" in result
        assert "counts" in result
