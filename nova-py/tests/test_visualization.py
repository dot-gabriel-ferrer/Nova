"""Tests for nova.visualization — astronomical display functions."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from nova.visualization import (
    display_image,
    display_rgb,
    display_spectrum,
    display_histogram,
    display_cutout,
    display_comparison,
    display_mosaic,
    display_provenance,
    _apply_stretch,
    _percentile_interval,
)


@pytest.fixture
def sample_image():
    """Generate a sample astronomical image."""
    rng = np.random.default_rng(42)
    data = rng.normal(100, 10, (128, 128))
    # Add some sources
    yy, xx = np.ogrid[0:128, 0:128]
    data += 500 * np.exp(-((xx - 64)**2 + (yy - 64)**2) / (2 * 5**2))
    data += 300 * np.exp(-((xx - 30)**2 + (yy - 90)**2) / (2 * 3**2))
    return data


class TestApplyStretch:
    def test_linear(self):
        data = np.linspace(0, 100, 100)
        result = _apply_stretch(data, "linear")
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_log(self):
        data = np.linspace(1, 100, 100)
        result = _apply_stretch(data, "log")
        assert np.all(np.isfinite(result))

    def test_sqrt(self):
        data = np.linspace(0, 100, 100)
        result = _apply_stretch(data, "sqrt")
        assert np.all(result >= 0)

    def test_asinh(self):
        data = np.linspace(0, 1000, 100)
        result = _apply_stretch(data, "asinh")
        assert np.all(np.isfinite(result))

    def test_power(self):
        data = np.linspace(0, 100, 100)
        result = _apply_stretch(data, "power", power=2.0)
        assert np.all(np.isfinite(result))

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown stretch"):
            _apply_stretch(np.array([1, 2, 3]), "invalid")


class TestPercentileInterval:
    def test_basic(self):
        data = np.arange(1000, dtype=float)
        low, high = _percentile_interval(data, 99.0)
        assert low < high
        assert low >= 0
        assert high <= 999


class TestDisplayImage:
    def test_returns_figure(self, sample_image):
        fig = display_image(sample_image, title="Test Image")
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_save_to_file(self, sample_image):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.png"
            result = display_image(sample_image, output_path=path)
            assert result is None
            assert path.exists()
            assert path.stat().st_size > 0

    def test_with_wcs_info(self, sample_image):
        wcs_info = {
            "nova:axes": [
                {"nova:ctype": "RA---TAN"},
                {"nova:ctype": "DEC--TAN"},
            ]
        }
        fig = display_image(sample_image, wcs_info=wcs_info)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_stretch_options(self, sample_image):
        for stretch in ["linear", "log", "sqrt", "asinh"]:
            fig = display_image(sample_image, stretch=stretch)
            assert fig is not None
            import matplotlib.pyplot as plt
            plt.close(fig)


class TestDisplayRGB:
    def test_basic_rgb(self, sample_image):
        fig = display_rgb(sample_image, sample_image * 0.8, sample_image * 0.6)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_save_to_file(self, sample_image):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rgb.png"
            result = display_rgb(
                sample_image, sample_image, sample_image,
                output_path=path,
            )
            assert result is None
            assert path.exists()


class TestDisplaySpectrum:
    def test_basic_spectrum(self):
        wavelength = np.linspace(4000, 7000, 500)
        flux = np.sin(wavelength / 100) + 5.0
        fig = display_spectrum(wavelength, flux)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_with_line_markers(self):
        wavelength = np.linspace(4000, 7000, 500)
        flux = np.ones_like(wavelength)
        markers = [
            {"wavelength": 4861, "name": "Hβ"},
            {"wavelength": 6563, "name": "Hα"},
        ]
        fig = display_spectrum(wavelength, flux, line_markers=markers)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestDisplayHistogram:
    def test_basic_histogram(self, sample_image):
        fig = display_histogram(sample_image)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_save_to_file(self, sample_image):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "hist.png"
            result = display_histogram(sample_image, output_path=path)
            assert result is None
            assert path.exists()


class TestDisplayCutout:
    def test_basic_cutout(self, sample_image):
        fig = display_cutout(sample_image, center=(64, 64), size=20)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestDisplayComparison:
    def test_two_images(self, sample_image):
        fig = display_comparison(sample_image, sample_image * 1.1)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_without_difference(self, sample_image):
        fig = display_comparison(
            sample_image, sample_image,
            show_difference=False,
        )
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestDisplayMosaic:
    def test_grid_of_images(self, sample_image):
        images = [sample_image * (i + 1) for i in range(4)]
        fig = display_mosaic(images, ncols=2)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_with_titles(self, sample_image):
        images = [sample_image for _ in range(3)]
        titles = ["A", "B", "C"]
        fig = display_mosaic(images, titles=titles, ncols=3)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestDisplayProvenance:
    def test_basic_provenance(self):
        prov = {
            "@type": "prov:Bundle",
            "prov:entity": [
                {"@id": "raw_data", "nova:entity_type": "RawObservation"},
                {"@id": "cal_data", "nova:entity_type": "CalibratedImage"},
            ],
            "prov:activity": [
                {"@id": "calibration"},
            ],
            "prov:agent": [
                {"@id": "pipeline", "nova:name": "CalPipe v1.0"},
            ],
        }
        fig = display_provenance(prov)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)
