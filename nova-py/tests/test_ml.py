"""Tests for the NOVA ML module."""

from __future__ import annotations

import numpy as np
import pytest

from nova.ml import (
    NormalizationMetadata,
    compute_normalization,
    normalize,
    denormalize,
    to_tensor,
    NORMALIZATION_METHODS,
)


class TestNormalizationMetadata:
    """Tests for NormalizationMetadata dataclass."""

    def test_to_dict(self) -> None:
        meta = NormalizationMetadata(
            method="min_max",
            min_val=0.0,
            max_val=1.0,
            mean=0.5,
            std=0.3,
        )
        d = meta.to_dict()
        assert d["@type"] == "nova:MLNormalization"
        assert d["nova:method"] == "min_max"
        assert d["nova:min"] == 0.0
        assert d["nova:max"] == 1.0

    def test_from_dict(self) -> None:
        d = {
            "@type": "nova:MLNormalization",
            "nova:method": "z_score",
            "nova:mean": 100.0,
            "nova:std": 15.0,
        }
        meta = NormalizationMetadata.from_dict(d)
        assert meta.method == "z_score"
        assert meta.mean == 100.0
        assert meta.std == 15.0

    def test_roundtrip(self) -> None:
        original = NormalizationMetadata(
            method="robust",
            min_val=-5.0,
            max_val=50000.0,
            mean=1000.0,
            std=300.0,
            median=990.0,
            mad=200.0,
        )
        d = original.to_dict()
        restored = NormalizationMetadata.from_dict(d)
        assert restored.method == original.method
        assert restored.min_val == original.min_val
        assert restored.max_val == original.max_val
        assert restored.mean == original.mean
        assert restored.median == original.median
        assert restored.mad == original.mad


class TestComputeNormalization:
    """Tests for compute_normalization."""

    def test_min_max(self) -> None:
        data = np.array([0.0, 5.0, 10.0])
        meta = compute_normalization(data, method="min_max")
        assert meta.min_val == pytest.approx(0.0)
        assert meta.max_val == pytest.approx(10.0)
        assert meta.mean == pytest.approx(5.0)

    def test_z_score(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.normal(100.0, 15.0, size=10000)
        meta = compute_normalization(data, method="z_score")
        assert meta.mean == pytest.approx(100.0, abs=1.0)
        assert meta.std == pytest.approx(15.0, abs=1.0)

    def test_robust(self) -> None:
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])  # outlier
        meta = compute_normalization(data, method="robust")
        assert meta.median is not None
        assert meta.mad is not None

    def test_unknown_method_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown normalization"):
            compute_normalization(np.array([1.0]), method="unknown")

    def test_handles_nan(self) -> None:
        data = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
        meta = compute_normalization(data, method="min_max")
        assert meta.min_val == pytest.approx(1.0)
        assert meta.max_val == pytest.approx(5.0)

    def test_all_methods(self) -> None:
        data = np.random.default_rng(42).standard_normal((64, 64))
        for method in NORMALIZATION_METHODS:
            meta = compute_normalization(data, method=method)
            assert meta.method == method
            assert meta.min_val is not None
            assert meta.max_val is not None


class TestNormalize:
    """Tests for normalize/denormalize."""

    def test_min_max_range(self) -> None:
        data = np.array([0.0, 5.0, 10.0])
        meta = compute_normalization(data, method="min_max")
        normed = normalize(data, meta)
        assert normed[0] == pytest.approx(0.0)
        assert normed[2] == pytest.approx(1.0)

    def test_z_score_properties(self) -> None:
        data = np.random.default_rng(42).standard_normal(1000) * 15 + 100
        meta = compute_normalization(data, method="z_score")
        normed = normalize(data, meta)
        assert np.mean(normed) == pytest.approx(0.0, abs=0.1)
        assert np.std(normed) == pytest.approx(1.0, abs=0.1)

    def test_min_max_roundtrip(self) -> None:
        data = np.random.default_rng(42).uniform(10, 500, size=(32, 32))
        meta = compute_normalization(data, method="min_max")
        normed = normalize(data, meta)
        recovered = denormalize(normed, meta)
        np.testing.assert_array_almost_equal(recovered, data, decimal=10)

    def test_z_score_roundtrip(self) -> None:
        data = np.random.default_rng(42).standard_normal((32, 32)) * 100
        meta = compute_normalization(data, method="z_score")
        normed = normalize(data, meta)
        recovered = denormalize(normed, meta)
        np.testing.assert_array_almost_equal(recovered, data, decimal=8)

    def test_asinh_roundtrip(self) -> None:
        data = np.random.default_rng(42).uniform(0, 10000, size=(32, 32))
        meta = compute_normalization(data, method="asinh")
        normed = normalize(data, meta)
        recovered = denormalize(normed, meta)
        np.testing.assert_array_almost_equal(recovered, data, decimal=5)


class TestToTensor:
    """Tests for tensor preparation."""

    def test_basic_conversion(self) -> None:
        data = np.ones((64, 64), dtype=np.float64)
        tensor, meta = to_tensor(data)
        assert tensor.dtype == np.float32
        assert tensor.shape == (64, 64)
        assert meta is None

    def test_with_normalization(self) -> None:
        data = np.random.default_rng(42).uniform(0, 1000, size=(64, 64))
        tensor, meta = to_tensor(data, normalize_method="min_max")
        assert meta is not None
        assert meta.method == "min_max"
        assert tensor.min() >= -0.01  # roughly [0, 1]
        assert tensor.max() <= 1.01

    def test_batch_and_channel_dims(self) -> None:
        data = np.ones((64, 64), dtype=np.float64)
        tensor, _ = to_tensor(
            data, add_batch_dim=True, add_channel_dim=True
        )
        assert tensor.shape == (1, 1, 64, 64)

    def test_channel_dim_only(self) -> None:
        data = np.ones((64, 64), dtype=np.float64)
        tensor, _ = to_tensor(data, add_channel_dim=True)
        assert tensor.shape == (1, 64, 64)

    def test_dtype_conversion(self) -> None:
        data = np.ones((32, 32), dtype=np.float64)
        tensor_f16, _ = to_tensor(data, dtype="float16")
        assert tensor_f16.dtype == np.float16

        tensor_f32, _ = to_tensor(data, dtype="float32")
        assert tensor_f32.dtype == np.float32

    def test_bfloat16_fallback(self) -> None:
        data = np.ones((32, 32), dtype=np.float64)
        tensor, _ = to_tensor(data, dtype="bfloat16")
        # bfloat16 not native in NumPy, should fall back to float32
        assert tensor.dtype == np.float32
