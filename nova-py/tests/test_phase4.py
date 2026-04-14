"""Tests for the NOVA pipeline framework, operations, and pipeline modules.

Covers: pipeline.py, operations.py, astrometry.py,
        photometry_pipeline.py, spectroscopy_pipeline.py
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

# -------------------------------------------------------------------
#  Pipeline framework tests
# -------------------------------------------------------------------

from nova.pipeline import Pipeline, Step, StepLog, PipelineLog


class TestStep:
    """Tests for the Step dataclass."""

    def test_execute_returns_array(self):
        def double(data):
            return data * 2
        step = Step(name="double", func=double)
        result = step.execute(np.array([1.0, 2.0]))
        np.testing.assert_array_equal(result, [2.0, 4.0])

    def test_execute_rejects_non_array_input(self):
        step = Step(name="noop", func=lambda d: d)
        with pytest.raises(TypeError, match="numpy ndarray"):
            step.execute([1, 2, 3])

    def test_execute_rejects_non_array_return(self):
        step = Step(name="bad", func=lambda d: 42)
        with pytest.raises(TypeError, match="must return"):
            step.execute(np.array([1.0]))

    def test_params_forwarded(self):
        def add_val(data, value=0):
            return data + value
        step = Step(name="add", func=add_val, params={"value": 10})
        result = step.execute(np.array([1.0]))
        np.testing.assert_array_equal(result, [11.0])


class TestPipeline:
    """Tests for the Pipeline class."""

    def _make_pipeline(self):
        p = Pipeline("test_pipe")
        p.add_step("double", lambda d: d * 2)
        p.add_step("add_one", lambda d: d + 1)
        return p

    def test_add_and_list_steps(self):
        p = self._make_pipeline()
        assert p.step_names == ["double", "add_one"]
        assert len(p) == 2

    def test_insert_step(self):
        p = self._make_pipeline()
        p.insert_step(1, "half", lambda d: d / 2)
        assert p.step_names == ["double", "half", "add_one"]

    def test_remove_step(self):
        p = self._make_pipeline()
        p.remove_step("double")
        assert p.step_names == ["add_one"]

    def test_remove_nonexistent_step_raises(self):
        p = self._make_pipeline()
        with pytest.raises(KeyError, match="nope"):
            p.remove_step("nope")

    def test_run_produces_correct_result(self):
        p = self._make_pipeline()
        data = np.array([1.0, 2.0, 3.0])
        result = p.run(data)
        expected = data * 2 + 1
        np.testing.assert_array_equal(result, expected)

    def test_run_produces_log(self):
        p = self._make_pipeline()
        p.run(np.array([1.0]))
        log = p.log
        assert log is not None
        assert log.pipeline_name == "test_pipe"
        assert len(log.steps) == 2
        assert log.steps[0].step_name == "double"
        assert log.steps[1].step_name == "add_one"
        assert log.total_duration_seconds >= 0

    def test_log_has_checksums(self):
        p = self._make_pipeline()
        p.run(np.array([1.0, 2.0]))
        for step_log in p.log.steps:
            assert len(step_log.input_sha256) == 64
            assert len(step_log.output_sha256) == 64

    def test_run_rejects_non_array(self):
        p = self._make_pipeline()
        with pytest.raises(TypeError):
            p.run([1, 2, 3])

    def test_run_empty_pipeline_raises(self):
        p = Pipeline("empty")
        with pytest.raises(RuntimeError, match="no steps"):
            p.run(np.array([1.0]))

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="not be empty"):
            Pipeline("")

    def test_empty_step_name_raises(self):
        p = Pipeline("test")
        with pytest.raises(ValueError, match="not be empty"):
            p.add_step("", lambda d: d)


class TestPipelineSerialization:
    """Tests for Pipeline/Log JSON serialization."""

    def test_log_roundtrip_json(self):
        p = Pipeline("serial_test")
        p.add_step("inc", lambda d: d + 1)
        p.run(np.array([0.0]))
        text = p.log.to_json()
        loaded = PipelineLog.from_json(text)
        assert loaded.pipeline_name == "serial_test"
        assert len(loaded.steps) == 1

    def test_pipeline_definition_json(self):
        p = Pipeline("def_test", version="2.0")
        p.add_step("scale", lambda d: d * 2, description="double it")
        text = p.to_json()
        d = json.loads(text)
        assert d["pipeline_name"] == "def_test"
        assert d["version"] == "2.0"
        assert len(d["steps"]) == 1
        assert d["steps"][0]["description"] == "double it"

    def test_pipeline_from_json_with_registry(self):
        p = Pipeline("round")
        p.add_step("inc", lambda d: d + 1)
        text = p.to_json()
        # Reconstruct with a registry
        registry = {"nova.pipeline.TestPipelineSerialization.test_pipeline_from_json_with_registry.<locals>.<lambda>": lambda d: d + 1}
        p2 = Pipeline.from_json(text, func_registry=registry)
        assert p2.step_names == ["inc"]

    def test_step_log_roundtrip(self):
        sl = StepLog(
            step_name="test", func_name="f", params_summary={},
            input_sha256="a" * 64, output_sha256="b" * 64,
            input_shape=(10,), output_shape=(10,),
            input_dtype="float64", output_dtype="float64",
            started_at="2026-01-01", ended_at="2026-01-01",
            duration_seconds=0.1,
        )
        d = sl.to_dict()
        sl2 = StepLog.from_dict(d)
        assert sl2.step_name == "test"
        assert sl2.duration_seconds == 0.1


# -------------------------------------------------------------------
#  Operations tests
# -------------------------------------------------------------------

from nova.operations import (
    OperationHistory, OperationRecord,
    op_add, op_subtract, op_multiply, op_divide,
    op_clip, op_mask_replace, op_normalize, op_rebin, op_combine,
)


class TestOperationHistory:
    """Tests for OperationHistory tracking."""

    def test_empty_history(self):
        h = OperationHistory()
        assert len(h) == 0

    def test_operations_recorded(self):
        h = OperationHistory()
        data = np.ones((4, 4))
        op_add(data, 1.0, history=h, label="add1")
        op_subtract(data, 0.5, history=h, label="sub1")
        assert len(h) == 2
        assert h.records[0].label == "add1"
        assert h.records[1].label == "sub1"

    def test_json_roundtrip(self):
        h = OperationHistory()
        data = np.ones(10)
        op_multiply(data, 2.0, history=h, label="scale")
        text = h.to_json()
        h2 = OperationHistory.from_json(text)
        assert len(h2) == 1
        assert h2.records[0].label == "scale"
        assert h2.records[0].operation == "multiply"


class TestArithmeticOperations:
    """Tests for tracked arithmetic operations."""

    def test_op_add(self):
        data = np.array([1.0, 2.0, 3.0])
        result = op_add(data, 10.0)
        np.testing.assert_array_equal(result, [11.0, 12.0, 13.0])

    def test_op_add_array(self):
        data = np.array([1.0, 2.0])
        operand = np.array([10.0, 20.0])
        result = op_add(data, operand)
        np.testing.assert_array_equal(result, [11.0, 22.0])

    def test_op_subtract(self):
        data = np.array([10.0, 20.0])
        result = op_subtract(data, 5.0)
        np.testing.assert_array_equal(result, [5.0, 15.0])

    def test_op_multiply(self):
        data = np.array([2.0, 3.0])
        result = op_multiply(data, 4.0)
        np.testing.assert_array_equal(result, [8.0, 12.0])

    def test_op_divide(self):
        data = np.array([10.0, 20.0])
        result = op_divide(data, 5.0)
        np.testing.assert_array_equal(result, [2.0, 4.0])

    def test_op_divide_zero_safe(self):
        data = np.array([10.0, 20.0])
        result = op_divide(data, 0.0)
        np.testing.assert_array_equal(result, [0.0, 0.0])

    def test_op_clip(self):
        data = np.array([-5.0, 0.0, 5.0, 10.0])
        result = op_clip(data, lower=0.0, upper=7.0)
        np.testing.assert_array_equal(result, [0.0, 0.0, 5.0, 7.0])

    def test_op_mask_replace(self):
        data = np.array([1.0, 2.0, 3.0, 4.0])
        mask = np.array([False, True, False, True])
        result = op_mask_replace(data, mask, fill_value=-1.0)
        np.testing.assert_array_equal(result, [1.0, -1.0, 3.0, -1.0])

    def test_op_normalize_minmax(self):
        data = np.array([0.0, 5.0, 10.0])
        result = op_normalize(data, method="minmax")
        np.testing.assert_allclose(result, [0.0, 0.5, 1.0])

    def test_op_normalize_zscore(self):
        data = np.array([10.0, 20.0, 30.0])
        result = op_normalize(data, method="zscore")
        assert abs(np.mean(result)) < 1e-10

    def test_op_normalize_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            op_normalize(np.array([1.0]), method="bogus")

    def test_op_rebin(self):
        data = np.ones((4, 4))
        result = op_rebin(data, factor=2)
        assert result.shape == (2, 2)
        np.testing.assert_array_equal(result, 4.0)

    def test_op_rebin_bad_factor_raises(self):
        data = np.ones((5, 5))
        with pytest.raises(ValueError, match="does not divide"):
            op_rebin(data, factor=2)

    def test_op_combine_median(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([3.0, 2.0, 1.0])
        c = np.array([2.0, 2.0, 2.0])
        result = op_combine([a, b, c], method="median")
        np.testing.assert_array_equal(result, [2.0, 2.0, 2.0])

    def test_op_combine_mean(self):
        a = np.array([1.0, 4.0])
        b = np.array([3.0, 2.0])
        result = op_combine([a, b], method="mean")
        np.testing.assert_array_equal(result, [2.0, 3.0])

    def test_op_combine_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            op_combine([], method="median")

    def test_op_combine_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="Shape mismatch"):
            op_combine([np.ones(3), np.ones(4)], method="mean")

    def test_type_validation(self):
        with pytest.raises(TypeError, match="numpy ndarray"):
            op_add([1, 2], 1.0)


# -------------------------------------------------------------------
#  Astrometry tests
# -------------------------------------------------------------------

from nova.astrometry import (
    extract_centroids, plate_solve, correct_proper_motion,
    correct_parallax, astrometric_residuals, fit_distortion_sip,
)


class TestExtractCentroids:
    """Tests for centroid extraction."""

    def _make_test_image(self, size=100, n_stars=5, fwhm=3.0):
        """Create a test image with Gaussian point sources."""
        img = np.random.normal(100, 10, (size, size))
        sigma = fwhm / 2.3548
        positions = []
        rng = np.random.RandomState(42)
        for _ in range(n_stars):
            x = rng.randint(20, size - 20)
            y = rng.randint(20, size - 20)
            flux = rng.uniform(500, 2000)
            yy, xx = np.mgrid[0:size, 0:size]
            img += flux * np.exp(-0.5 * ((xx - x) ** 2 + (yy - y) ** 2) / sigma ** 2)
            positions.append((x, y))
        return img, positions

    def test_finds_sources(self):
        img, positions = self._make_test_image()
        sources = extract_centroids(img, fwhm=3.5, threshold=5.0)
        assert sources.shape[1] == 3
        assert sources.shape[0] >= 1

    def test_empty_image(self):
        img = np.zeros((50, 50))
        sources = extract_centroids(img, fwhm=3.0, threshold=5.0)
        assert sources.shape[0] == 0

    def test_rejects_non_2d(self):
        with pytest.raises(TypeError):
            extract_centroids(np.ones(10), fwhm=3.0)

    def test_rejects_bad_fwhm(self):
        with pytest.raises(ValueError, match="fwhm"):
            extract_centroids(np.ones((10, 10)), fwhm=-1.0)


class TestCorrectProperMotion:
    """Tests for proper-motion correction."""

    def test_no_motion(self):
        ra = np.array([180.0])
        dec = np.array([45.0])
        ra2, dec2 = correct_proper_motion(ra, dec, np.array([0.0]), np.array([0.0]),
                                          2016.0, 2020.0)
        np.testing.assert_array_equal(ra2, ra)
        np.testing.assert_array_equal(dec2, dec)

    def test_positive_motion(self):
        ra = np.array([180.0])
        dec = np.array([0.0])
        # 1000 mas/yr in RA over 1 year = ~0.2778 arcsec = 7.716e-5 deg
        ra2, dec2 = correct_proper_motion(
            ra, dec, np.array([1000.0]), np.array([0.0]),
            2020.0, 2021.0,
        )
        assert ra2[0] > ra[0]  # should have moved east


class TestCorrectParallax:
    """Tests for parallax correction."""

    def test_zero_parallax(self):
        ra = np.array([180.0])
        dec = np.array([45.0])
        ra2, dec2 = correct_parallax(ra, dec, np.array([0.0]),
                                     np.array([1.0, 0.0, 0.0]))
        np.testing.assert_array_almost_equal(ra2, ra)
        np.testing.assert_array_almost_equal(dec2, dec)


class TestAstrometricResiduals:
    """Tests for residual computation."""

    def test_perfect_solution(self):
        # Generate matched pairs with a known CD matrix
        n = 20
        cd = np.array([[1e-4, 0], [0, 1e-4]])
        crpix = (50.0, 50.0)
        crval = (180.0, 45.0)
        rng = np.random.RandomState(123)
        xy = rng.uniform(10, 90, (n, 2))
        # Predict RA/Dec from the WCS (simplified: ignore projection)
        dxy = xy - np.array(crpix)
        tp = dxy @ cd.T
        # These are tangent-plane coords; use crval as center
        ra_cat = crval[0] + tp[:, 0]
        dec_cat = crval[1] + tp[:, 1]
        radec = np.column_stack([ra_cat, dec_cat])
        res = astrometric_residuals(xy, radec, crpix, crval, cd)
        assert res["rms_arcsec"] < 5.0  # should be small for a linear model
        assert res["n_sources"] == n


# -------------------------------------------------------------------
#  Photometry pipeline tests
# -------------------------------------------------------------------

from nova.photometry_pipeline import (
    multi_aperture_photometry, calibrate_zeropoint, extinction_correct,
    color_term_correct, limiting_magnitude, growth_curve,
    aperture_correction, differential_photometry,
    ab_to_vega, vega_to_ab, flux_to_mag, mag_to_flux,
)


class TestMultiAperturePhotometry:
    """Tests for multi-aperture photometry."""

    def _make_source_image(self, size=100, flux=1000.0, bg=100.0):
        """Create a test image with one source at the centre."""
        img = np.full((size, size), bg, dtype=np.float64)
        cx, cy = size // 2, size // 2
        yy, xx = np.mgrid[0:size, 0:size]
        sigma = 2.0
        img += flux * np.exp(-0.5 * ((xx - cx) ** 2 + (yy - cy) ** 2) / sigma ** 2)
        return img

    def test_single_source(self):
        img = self._make_source_image()
        sources = np.array([[50.0, 50.0]])
        result = multi_aperture_photometry(img, sources, radii=[5, 10],
                                           gain=1.0, readnoise=0.0)
        assert "flux_5" in result
        assert "flux_10" in result
        assert result["flux_10"][0] > result["flux_5"][0]

    def test_rejects_non_2d(self):
        with pytest.raises(TypeError):
            multi_aperture_photometry(np.ones(10), np.array([[5, 5]]), radii=[3])

    def test_empty_radii_raises(self):
        with pytest.raises(ValueError, match="radius"):
            multi_aperture_photometry(np.ones((10, 10)), np.array([[5, 5]]),
                                      radii=[])


class TestCalibrateZeropoint:
    """Tests for zero-point calibration."""

    def test_known_offset(self):
        rng = np.random.RandomState(99)
        n = 50
        inst = rng.uniform(-15, -10, n)
        zp_true = 25.0
        cat = inst + zp_true + rng.normal(0, 0.05, n)
        result = calibrate_zeropoint(inst, cat)
        assert abs(result["zeropoint"] - zp_true) < 0.2
        assert result["n_used"] > 0

    def test_rejects_nans(self):
        inst = np.array([1.0, np.nan, 3.0])
        cat = np.array([26.0, 27.0, 28.0])
        result = calibrate_zeropoint(inst, cat)
        assert result["n_used"] <= 2


class TestExtinctionCorrect:
    """Tests for extinction correction."""

    def test_correction(self):
        mag = np.array([20.0, 21.0])
        result = extinction_correct(mag, airmass=1.5, k_lambda=0.15)
        expected = mag - 0.15 * 1.5
        np.testing.assert_allclose(result, expected)

    def test_bad_airmass_raises(self):
        with pytest.raises(ValueError, match="Airmass"):
            extinction_correct(np.array([20.0]), airmass=0.5, k_lambda=0.1)


class TestMagnitudeConversions:
    """Tests for magnitude system conversions."""

    def test_ab_vega_roundtrip(self):
        mag_ab = np.array([20.0, 21.0])
        offset = 0.02
        mag_vega = ab_to_vega(mag_ab, offset)
        mag_ab2 = vega_to_ab(mag_vega, offset)
        np.testing.assert_allclose(mag_ab2, mag_ab)

    def test_flux_mag_roundtrip(self):
        flux = np.array([100.0, 1000.0, 10000.0])
        mag = flux_to_mag(flux, zeropoint=25.0)
        flux2 = mag_to_flux(mag, zeropoint=25.0)
        np.testing.assert_allclose(flux2, flux, rtol=1e-10)

    def test_flux_to_mag_negative(self):
        mag = flux_to_mag(np.array([-1.0]))
        assert np.isnan(mag[0])


class TestDifferentialPhotometry:
    """Tests for differential photometry."""

    def test_constant_ratio(self):
        target = np.array([100.0, 100.0, 100.0])
        comp = np.array([200.0, 200.0, 200.0])
        result = differential_photometry(target, comp)
        # delta_mag should be constant
        dm = result["delta_mag"]
        assert np.std(dm) < 1e-10

    def test_with_comparison_mag(self):
        target = np.array([100.0])
        comp = np.array([100.0])
        result = differential_photometry(target, comp, comparison_mag=15.0)
        np.testing.assert_allclose(result["delta_mag"], [15.0])


class TestLimitingMagnitude:
    """Tests for limiting magnitude."""

    def test_positive_result(self):
        lm = limiting_magnitude(sky_rms=10.0, zeropoint=25.0,
                                aperture_npix=50, nsigma=5.0, gain=1.0)
        assert np.isfinite(lm)
        assert lm > 0

    def test_zero_sky(self):
        lm = limiting_magnitude(sky_rms=0.0, zeropoint=25.0,
                                aperture_npix=50, nsigma=5.0, gain=1.0)
        assert np.isnan(lm)  # zero noise => infinite depth


class TestGrowthCurve:
    """Tests for growth curve analysis."""

    def test_increasing_flux(self):
        img = np.zeros((50, 50))
        img[25, 25] = 100.0
        gc = growth_curve(img, 25.0, 25.0, radii=[1, 3, 5, 10])
        assert len(gc["radii"]) == 4
        # Flux should be non-decreasing
        for i in range(1, len(gc["flux"])):
            assert gc["flux"][i] >= gc["flux"][i - 1]


# -------------------------------------------------------------------
#  Spectroscopy pipeline tests
# -------------------------------------------------------------------

from nova.spectroscopy_pipeline import (
    optimal_extract, fit_continuum, normalize_spectrum,
    model_telluric, correct_telluric, resample_spectrum,
    stack_spectra, measure_redshift, estimate_snr,
    equivalent_width, smooth_spectrum,
)


class TestOptimalExtract:
    """Tests for optimal extraction."""

    def test_extracts_signal(self):
        ny, nx = 20, 100
        data = np.random.normal(0, 1, (ny, nx))
        # Add a trace at row 10 with signal
        data[9:12, :] += 50.0
        trace = np.full(nx, 10.0)
        result = optimal_extract(data, trace, aperture_half=3)
        assert result["flux"].shape == (nx,)
        assert np.median(result["flux"]) > 0

    def test_rejects_non_2d(self):
        with pytest.raises(TypeError):
            optimal_extract(np.ones(10), np.ones(10))

    def test_trace_length_mismatch(self):
        with pytest.raises(ValueError, match="trace length"):
            optimal_extract(np.ones((10, 20)), np.ones(10))


class TestFitContinuum:
    """Tests for continuum fitting."""

    def test_polynomial(self):
        wave = np.linspace(4000, 7000, 200)
        cont_true = 1.0 + 0.001 * (wave - 5500)
        flux = cont_true + np.random.normal(0, 0.01, 200)
        cont_fit = fit_continuum(wave, flux, method="polynomial", order=3)
        np.testing.assert_allclose(cont_fit, cont_true, atol=0.1)

    def test_median(self):
        wave = np.linspace(4000, 7000, 200)
        flux = np.ones(200)
        flux[100] = 5.0  # emission line
        cont = fit_continuum(wave, flux, method="median", window=11)
        # Continuum should be close to 1 everywhere
        assert abs(np.median(cont) - 1.0) < 0.1

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            fit_continuum(np.arange(10.0), np.arange(10.0), method="bogus")


class TestNormalizeSpectrum:
    """Tests for spectrum normalization."""

    def test_normalized_near_one(self):
        wave = np.linspace(4000, 7000, 200)
        flux = np.ones(200) * 100.0
        result = normalize_spectrum(wave, flux)
        np.testing.assert_allclose(result["flux_normalized"], 1.0, atol=0.1)


class TestTelluric:
    """Tests for telluric correction."""

    def test_model_creates_absorption(self):
        wave = np.linspace(6000, 8000, 500)
        trans = model_telluric(wave, bands=[(6870, 30), (7600, 50)], depth=0.8)
        assert trans.min() < 1.0
        assert trans.max() == 1.0

    def test_correction_restores_flux(self):
        wave = np.linspace(6000, 8000, 500)
        flux_true = np.ones(500) * 100.0
        trans = model_telluric(wave, bands=[(7000, 30)], depth=0.5)
        flux_obs = flux_true * trans
        flux_corrected = correct_telluric(flux_obs, trans)
        np.testing.assert_allclose(flux_corrected, flux_true, atol=1.0)


class TestResampleSpectrum:
    """Tests for spectral resampling."""

    def test_identity_resample(self):
        wave = np.arange(100.0)
        flux = np.sin(wave / 10.0)
        result = resample_spectrum(wave, flux, wave)
        np.testing.assert_allclose(result["flux"], flux)

    def test_out_of_range_nan(self):
        wave = np.arange(10.0)
        flux = np.ones(10)
        new_wave = np.array([-1.0, 5.0, 15.0])
        result = resample_spectrum(wave, flux, new_wave)
        assert np.isnan(result["flux"][0])
        assert result["flux"][1] == 1.0
        assert np.isnan(result["flux"][2])


class TestStackSpectra:
    """Tests for spectral stacking."""

    def test_median_stack(self):
        w = np.arange(100.0)
        f1 = np.ones(100) * 10
        f2 = np.ones(100) * 20
        f3 = np.ones(100) * 15
        result = stack_spectra([w, w, w], [f1, f2, f3], method="median")
        np.testing.assert_allclose(result["flux"], 15.0)
        assert result["n_combined"] == 3

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            stack_spectra([], [])


class TestMeasureRedshift:
    """Tests for redshift measurement."""

    def test_known_redshift(self):
        # Template: emission line at 5000 A
        t_wave = np.linspace(4500, 5500, 500)
        t_flux = np.exp(-0.5 * ((t_wave - 5000) / 5) ** 2)
        # Observed: same line at z=0.1 -> 5500 A
        z_true = 0.1
        o_wave = np.linspace(4500, 6500, 500)
        o_flux = np.exp(-0.5 * ((o_wave - 5000 * (1 + z_true)) / 5) ** 2)
        result = measure_redshift(o_wave, o_flux, t_wave, t_flux,
                                  z_min=0.0, z_max=0.3, z_step=0.001)
        assert abs(result["z_best"] - z_true) < 0.01


class TestEstimateSNR:
    """Tests for S/N estimation."""

    def test_high_snr(self):
        flux = np.ones(200) * 1000 + np.random.normal(0, 1, 200)
        result = estimate_snr(flux, method="der_snr")
        assert result["snr_median"] > 50

    def test_variance_method(self):
        flux = np.ones(100) * 100
        var = np.ones(100) * 4.0
        result = estimate_snr(flux, variance=var, method="variance")
        np.testing.assert_allclose(result["snr_median"], 50.0)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            estimate_snr(np.ones(10), method="bogus")


class TestEquivalentWidth:
    """Tests for equivalent width measurement."""

    def test_absorption_line(self):
        wave = np.linspace(4800, 5200, 200)
        cont = np.ones(200) * 100.0
        # Add an absorption line centered at 5000
        flux = cont.copy()
        flux -= 50 * np.exp(-0.5 * ((wave - 5000) / 5) ** 2)
        result = equivalent_width(wave, flux, cont, line_range=(4980, 5020))
        assert result["ew"] > 0  # absorption -> positive EW
        assert result["line_flux"] > 0  # absorbed flux


class TestSmoothSpectrum:
    """Tests for spectral smoothing."""

    def test_boxcar_preserves_shape(self):
        flux = np.random.normal(0, 1, 100)
        smoothed = smooth_spectrum(flux, kernel_size=5, method="boxcar")
        assert smoothed.shape == flux.shape

    def test_gaussian_reduces_noise(self):
        flux = np.ones(100) + np.random.normal(0, 1, 100)
        smoothed = smooth_spectrum(flux, kernel_size=11, method="gaussian")
        assert np.std(smoothed) < np.std(flux)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            smooth_spectrum(np.ones(10), method="bogus")
