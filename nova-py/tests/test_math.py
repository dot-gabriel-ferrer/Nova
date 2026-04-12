"""Tests for nova.math — integrated astronomical mathematical tools."""

import numpy as np
import numpy.testing as npt
import pytest

from nova.math import (
    sigma_clip,
    sigma_clipped_stats,
    robust_statistics,
    histogram,
    gaussian_kernel_2d,
    convolve_fft,
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


# ──────────────────────────────────────────────────────────────────────────
#  Statistics
# ──────────────────────────────────────────────────────────────────────────

class TestSigmaClip:
    def test_basic_clipping(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 1000)
        data[0] = 100  # outlier
        data[1] = -100  # outlier
        clipped = sigma_clip(data, sigma=3.0)
        assert clipped.mask[0]  # outlier masked
        assert clipped.mask[1]  # outlier masked
        assert clipped.compressed().size < 1000

    def test_no_outliers(self):
        data = np.ones(100)
        clipped = sigma_clip(data)
        assert not np.any(clipped.mask)

    def test_nan_handling(self):
        data = np.array([1.0, 2.0, np.nan, 3.0, 4.0])
        clipped = sigma_clip(data)
        assert clipped.mask[2]  # NaN masked

    def test_mean_center(self):
        rng = np.random.default_rng(42)
        data = rng.normal(10, 1, 500)
        data[0] = 200
        clipped = sigma_clip(data, center_func="mean")
        assert clipped.mask[0]


class TestSigmaClippedStats:
    def test_basic_stats(self):
        rng = np.random.default_rng(42)
        data = rng.normal(100, 10, 10000)
        stats = sigma_clipped_stats(data)
        assert abs(stats["mean"] - 100) < 1.0
        assert abs(stats["std"] - 10) < 1.0
        assert stats["count"] > 9000

    def test_with_outliers(self):
        data = np.concatenate([np.ones(100), [1000.0]])
        stats = sigma_clipped_stats(data)
        assert abs(stats["mean"] - 1.0) < 0.1

    def test_empty_after_clip(self):
        data = np.array([np.nan, np.nan])
        stats = sigma_clipped_stats(data)
        assert np.isnan(stats["mean"])
        assert stats["count"] == 0


class TestRobustStatistics:
    def test_gaussian_data(self):
        rng = np.random.default_rng(42)
        data = rng.normal(50, 5, 10000)
        stats = robust_statistics(data)
        assert abs(stats["median"] - 50) < 0.5
        assert stats["mad"] > 0
        assert abs(stats["biweight_location"] - 50) < 1.0

    def test_empty_data(self):
        stats = robust_statistics(np.array([]))
        assert np.isnan(stats["median"])

    def test_constant_data(self):
        data = np.ones(100) * 42.0
        stats = robust_statistics(data)
        assert stats["median"] == 42.0
        assert stats["mad"] == 0.0


class TestHistogram:
    def test_basic_histogram(self):
        data = np.arange(100, dtype=float)
        counts, edges = histogram(data, bins=10)
        assert len(counts) == 10
        assert len(edges) == 11
        assert counts.sum() == 100

    def test_nan_ignored(self):
        data = np.array([1.0, 2.0, np.nan, 3.0])
        counts, edges = histogram(data, bins=3)
        assert counts.sum() == 3


# ──────────────────────────────────────────────────────────────────────────
#  Convolution
# ──────────────────────────────────────────────────────────────────────────

class TestGaussianKernel:
    def test_normalization(self):
        kernel = gaussian_kernel_2d(2.0)
        npt.assert_almost_equal(kernel.sum(), 1.0, decimal=10)

    def test_shape_odd(self):
        kernel = gaussian_kernel_2d(3.0)
        assert kernel.shape[0] % 2 == 1
        assert kernel.shape[1] % 2 == 1

    def test_custom_size(self):
        kernel = gaussian_kernel_2d(1.0, size=11)
        assert kernel.shape == (11, 11)

    def test_even_size_corrected(self):
        kernel = gaussian_kernel_2d(1.0, size=10)
        assert kernel.shape[0] % 2 == 1  # auto-corrected to odd


class TestConvolveFft:
    def test_identity_convolution(self):
        data = np.ones((32, 32))
        kernel = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        result = convolve_fft(data, kernel)
        npt.assert_allclose(result, 1.0, atol=1e-10)

    def test_smoothing_reduces_noise(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, (64, 64))
        kernel = gaussian_kernel_2d(2.0)
        result = convolve_fft(data, kernel)
        assert np.std(result) < np.std(data)

    def test_nan_interpolation(self):
        data = np.ones((32, 32))
        data[16, 16] = np.nan
        kernel = gaussian_kernel_2d(2.0)
        result = convolve_fft(data, kernel, nan_treatment="interpolate")
        assert np.isfinite(result[16, 16])


class TestSmoothGaussian:
    def test_smooth_reduces_std(self):
        rng = np.random.default_rng(42)
        data = rng.normal(100, 10, (64, 64))
        smoothed = smooth_gaussian(data, sigma=3.0)
        assert np.std(smoothed) < np.std(data)


# ──────────────────────────────────────────────────────────────────────────
#  Resampling
# ──────────────────────────────────────────────────────────────────────────

class TestRebin:
    def test_sum_rebin(self):
        data = np.ones((100, 100))
        rebinned = rebin(data, (50, 50), method="sum")
        assert rebinned.shape == (50, 50)
        npt.assert_allclose(rebinned, 4.0)

    def test_mean_rebin(self):
        data = np.ones((100, 100))
        rebinned = rebin(data, (50, 50), method="mean")
        npt.assert_allclose(rebinned, 1.0)

    def test_non_divisible_raises(self):
        with pytest.raises(ValueError, match="integer factors"):
            rebin(np.ones((100, 100)), (30, 30))


class TestResizeImage:
    def test_upsample(self):
        data = np.ones((10, 10))
        resized = resize_image(data, (20, 20))
        assert resized.shape == (20, 20)
        npt.assert_allclose(resized, 1.0)

    def test_downsample(self):
        data = np.ones((20, 20)) * 5.0
        resized = resize_image(data, (10, 10))
        npt.assert_allclose(resized, 5.0, atol=0.1)

    def test_nearest_neighbor(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        resized = resize_image(data, (4, 4), order=0)
        assert resized.shape == (4, 4)


# ──────────────────────────────────────────────────────────────────────────
#  Image Stacking
# ──────────────────────────────────────────────────────────────────────────

class TestStackImages:
    def test_mean_stacking(self):
        images = [np.ones((32, 32)) * i for i in range(5)]
        result = stack_images(images, method="mean")
        npt.assert_allclose(result, 2.0)

    def test_median_stacking(self):
        images = [np.ones((32, 32)) * i for i in [1, 2, 100]]
        result = stack_images(images, method="median")
        npt.assert_allclose(result, 2.0)

    def test_sigma_clip_stacking(self):
        rng = np.random.default_rng(42)
        images = [rng.normal(10, 0.1, (32, 32)) for _ in range(10)]
        images[0][:] = 1000  # outlier frame
        result = stack_images(images, method="sigma_clip", sigma=3.0)
        assert abs(np.mean(result) - 10.0) < 1.0

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            stack_images([])

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape"):
            stack_images([np.ones((10, 10)), np.ones((20, 20))])


# ──────────────────────────────────────────────────────────────────────────
#  Background Estimation
# ──────────────────────────────────────────────────────────────────────────

class TestEstimateBackground:
    def test_flat_background(self):
        data = np.ones((128, 128)) * 100.0
        bg, rms = estimate_background(data, box_size=32)
        assert bg.shape == (128, 128)
        npt.assert_allclose(bg, 100.0, atol=1.0)

    def test_background_shape(self):
        data = np.ones((256, 256))
        bg, rms = estimate_background(data, box_size=64)
        assert bg.shape == data.shape
        assert rms.shape == data.shape


# ──────────────────────────────────────────────────────────────────────────
#  Source Detection
# ──────────────────────────────────────────────────────────────────────────

class TestDetectSources:
    def test_detect_single_source(self):
        data = np.zeros((64, 64))
        # Add a bright source
        yy, xx = np.ogrid[0:64, 0:64]
        data += 100 * np.exp(-((xx - 32)**2 + (yy - 32)**2) / (2 * 3**2))
        sources = detect_sources(data, nsigma=3.0, min_area=3)
        assert len(sources) >= 1
        # Centroid should be near (32, 32)
        s = sources[0]
        assert abs(s["x"] - 32) < 3
        assert abs(s["y"] - 32) < 3

    def test_no_sources_in_noise(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 0.01, (64, 64))
        sources = detect_sources(data, nsigma=100.0)
        assert len(sources) == 0

    def test_with_explicit_threshold(self):
        data = np.zeros((32, 32))
        data[15:18, 15:18] = 100.0
        sources = detect_sources(data, threshold=50.0, min_area=1)
        assert len(sources) >= 1


# ──────────────────────────────────────────────────────────────────────────
#  Aperture Photometry
# ──────────────────────────────────────────────────────────────────────────

class TestAperturePhotometry:
    def test_known_flux(self):
        data = np.zeros((64, 64))
        data[30:34, 30:34] = 10.0  # 4x4 = 160 total flux
        result = aperture_photometry(data, x=32.0, y=32.0, radius=5.0)
        assert result["flux"] > 0
        assert result["area"] > 0

    def test_background_subtraction(self):
        data = np.ones((64, 64)) * 10.0  # flat background
        data[32, 32] = 110.0  # bright pixel
        result = aperture_photometry(
            data, x=32, y=32, radius=3.0,
            annulus_inner=10.0, annulus_outer=15.0,
        )
        assert result["background"] > 0
        assert result["flux_corrected"] < result["flux"]


# ──────────────────────────────────────────────────────────────────────────
#  Spectral
# ──────────────────────────────────────────────────────────────────────────

class TestContinuumNormalize:
    def test_flat_continuum(self):
        wavelength = np.linspace(4000, 7000, 1000)
        flux = np.ones_like(wavelength) * 100.0
        norm, cont = continuum_normalize(wavelength, flux)
        npt.assert_allclose(norm, 1.0, atol=0.01)

    def test_with_absorption_line(self):
        wavelength = np.linspace(4000, 7000, 1000)
        flux = np.ones_like(wavelength) * 100.0
        # Add absorption line
        line_center = 5500
        flux -= 50 * np.exp(-0.5 * ((wavelength - line_center) / 10) ** 2)
        norm, cont = continuum_normalize(wavelength, flux)
        # Normalized continuum should be ~1, line should be < 1
        assert np.min(norm) < 0.7


class TestEquivalentWidth:
    def test_absorption_line(self):
        wavelength = np.linspace(4000, 7000, 1000)
        norm_flux = np.ones_like(wavelength)
        # Deep absorption line at 5500
        line_center = 5500
        norm_flux -= 0.5 * np.exp(-0.5 * ((wavelength - line_center) / 10) ** 2)
        ew = equivalent_width(wavelength, norm_flux, 5500, 50)
        assert ew > 0  # Positive for absorption

    def test_no_line(self):
        wavelength = np.linspace(4000, 7000, 100)
        flux = np.ones_like(wavelength)
        ew = equivalent_width(wavelength, flux, 5500, 50)
        npt.assert_almost_equal(ew, 0.0, decimal=5)


# ──────────────────────────────────────────────────────────────────────────
#  Cosmic Ray Cleaning
# ──────────────────────────────────────────────────────────────────────────

class TestCosmicRayClean:
    def test_cleans_hot_pixels(self):
        rng = np.random.default_rng(42)
        data = rng.normal(100, 1, (64, 64))
        data[30, 30] = 10000  # cosmic ray
        cleaned = cosmic_ray_clean(data, sigma=5.0)
        assert cleaned[30, 30] < 200  # should be replaced

    def test_preserves_normal_data(self):
        data = np.ones((32, 32)) * 50.0
        cleaned = cosmic_ray_clean(data)
        npt.assert_allclose(cleaned, 50.0, atol=1e-10)
