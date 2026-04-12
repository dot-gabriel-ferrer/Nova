"""NOVA ML-native module (INV-7: ML_NATIVE).

Provides tools for machine learning interoperability:
- Normalization metadata (min-max, z-score, robust, log, asinh)
- Tensor export to NumPy, PyTorch, and JAX
- Float16/BFloat16 native support with standardized metadata
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


# Supported normalization methods
NORMALIZATION_METHODS = ("min_max", "z_score", "robust", "log", "asinh", "custom")


@dataclass
class NormalizationMetadata:
    """Normalization metadata for ML tensor compatibility.

    Parameters
    ----------
    method : str
        Normalization method.
    min_val : float or None
        Minimum value of the data.
    max_val : float or None
        Maximum value of the data.
    mean : float or None
        Mean of the data.
    std : float or None
        Standard deviation of the data.
    median : float or None
        Median of the data.
    mad : float or None
        Median absolute deviation.
    """

    method: str = "min_max"
    min_val: float | None = None
    max_val: float | None = None
    mean: float | None = None
    std: float | None = None
    median: float | None = None
    mad: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-LD dictionary."""
        d: dict[str, Any] = {
            "@type": "nova:MLNormalization",
            "nova:method": self.method,
        }
        if self.min_val is not None:
            d["nova:min"] = self.min_val
        if self.max_val is not None:
            d["nova:max"] = self.max_val
        if self.mean is not None:
            d["nova:mean"] = self.mean
        if self.std is not None:
            d["nova:std"] = self.std
        if self.median is not None:
            d["nova:median"] = self.median
        if self.mad is not None:
            d["nova:mad"] = self.mad
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NormalizationMetadata:
        """Create from JSON-LD dictionary."""
        return cls(
            method=data.get("nova:method", "min_max"),
            min_val=data.get("nova:min"),
            max_val=data.get("nova:max"),
            mean=data.get("nova:mean"),
            std=data.get("nova:std"),
            median=data.get("nova:median"),
            mad=data.get("nova:mad"),
        )


def compute_normalization(
    data: np.ndarray,
    method: str = "min_max",
) -> NormalizationMetadata:
    """Compute normalization metadata from a data array.

    Parameters
    ----------
    data : numpy.ndarray
        Input data array.
    method : str
        Normalization method. One of: 'min_max', 'z_score', 'robust',
        'log', 'asinh', 'custom'.

    Returns
    -------
    NormalizationMetadata
        Computed normalization metadata.

    Raises
    ------
    ValueError
        If method is not recognized.
    """
    if method not in NORMALIZATION_METHODS:
        raise ValueError(
            f"Unknown normalization method: '{method}'. "
            f"Must be one of: {NORMALIZATION_METHODS}"
        )

    # Handle NaN values
    finite_data = data[np.isfinite(data)] if np.issubdtype(data.dtype, np.floating) else data.ravel()

    meta = NormalizationMetadata(method=method)
    meta.min_val = float(np.min(finite_data)) if finite_data.size > 0 else 0.0
    meta.max_val = float(np.max(finite_data)) if finite_data.size > 0 else 0.0
    meta.mean = float(np.mean(finite_data)) if finite_data.size > 0 else 0.0
    meta.std = float(np.std(finite_data)) if finite_data.size > 0 else 0.0
    meta.median = float(np.median(finite_data)) if finite_data.size > 0 else 0.0

    if finite_data.size > 0:
        meta.mad = float(np.median(np.abs(finite_data - meta.median)))
    else:
        meta.mad = 0.0

    return meta


def normalize(
    data: np.ndarray,
    metadata: NormalizationMetadata,
) -> np.ndarray:
    """Apply normalization to a data array.

    Parameters
    ----------
    data : numpy.ndarray
        Input data array.
    metadata : NormalizationMetadata
        Normalization parameters.

    Returns
    -------
    numpy.ndarray
        Normalized data.
    """
    result = data.astype(np.float64)

    if metadata.method == "min_max":
        min_val = metadata.min_val if metadata.min_val is not None else 0.0
        max_val = metadata.max_val if metadata.max_val is not None else 1.0
        range_val = max_val - min_val
        if range_val > 0:
            result = (result - min_val) / range_val
        else:
            result = np.zeros_like(result)

    elif metadata.method == "z_score":
        mean = metadata.mean if metadata.mean is not None else 0.0
        std = metadata.std if metadata.std is not None else 1.0
        if std > 0:
            result = (result - mean) / std
        else:
            result = result - mean

    elif metadata.method == "robust":
        median = metadata.median if metadata.median is not None else 0.0
        mad = metadata.mad if metadata.mad is not None else 1.0
        if mad > 0:
            result = (result - median) / (1.4826 * mad)
        else:
            result = result - median

    elif metadata.method == "log":
        # Log normalization with offset to avoid log(0)
        min_val = metadata.min_val if metadata.min_val is not None else 0.0
        offset = abs(min_val) + 1.0 if min_val <= 0 else 0.0
        result = np.log1p(result + offset)

    elif metadata.method == "asinh":
        # Asinh normalization (common in astronomy for wide dynamic range)
        result = np.arcsinh(result)

    return result


def denormalize(
    data: np.ndarray,
    metadata: NormalizationMetadata,
) -> np.ndarray:
    """Reverse normalization to recover original values.

    Parameters
    ----------
    data : numpy.ndarray
        Normalized data array.
    metadata : NormalizationMetadata
        Normalization parameters used for the original normalization.

    Returns
    -------
    numpy.ndarray
        Denormalized data.
    """
    result = data.astype(np.float64)

    if metadata.method == "min_max":
        min_val = metadata.min_val if metadata.min_val is not None else 0.0
        max_val = metadata.max_val if metadata.max_val is not None else 1.0
        result = result * (max_val - min_val) + min_val

    elif metadata.method == "z_score":
        mean = metadata.mean if metadata.mean is not None else 0.0
        std = metadata.std if metadata.std is not None else 1.0
        result = result * std + mean

    elif metadata.method == "robust":
        median = metadata.median if metadata.median is not None else 0.0
        mad = metadata.mad if metadata.mad is not None else 1.0
        result = result * (1.4826 * mad) + median

    elif metadata.method == "log":
        min_val = metadata.min_val if metadata.min_val is not None else 0.0
        offset = abs(min_val) + 1.0 if min_val <= 0 else 0.0
        result = np.expm1(result) - offset

    elif metadata.method == "asinh":
        result = np.sinh(result)

    return result


def to_tensor(
    data: np.ndarray,
    dtype: str = "float32",
    normalize_method: str | None = None,
    add_batch_dim: bool = False,
    add_channel_dim: bool = False,
) -> tuple[np.ndarray, NormalizationMetadata | None]:
    """Prepare a NOVA data array for ML tensor consumption.

    Returns a NumPy array ready for conversion to PyTorch/JAX tensors,
    with optional normalization and dimension handling.

    Parameters
    ----------
    data : numpy.ndarray
        Input data array.
    dtype : str
        Target dtype ('float16', 'bfloat16', 'float32', 'float64').
    normalize_method : str, optional
        If provided, normalize the data using the specified method.
    add_batch_dim : bool
        If True, add a batch dimension (axis 0).
    add_channel_dim : bool
        If True, add a channel dimension (axis 0 for 2D, axis 1 for 3D+).

    Returns
    -------
    tuple of (numpy.ndarray, NormalizationMetadata or None)
        Tensor-ready array and normalization metadata (if normalization applied).
    """
    norm_meta = None

    # Apply normalization
    if normalize_method is not None:
        norm_meta = compute_normalization(data, method=normalize_method)
        result = normalize(data, norm_meta)
    else:
        result = data.astype(np.float64)

    # Add channel dimension
    if add_channel_dim:
        result = np.expand_dims(result, axis=0)

    # Add batch dimension
    if add_batch_dim:
        result = np.expand_dims(result, axis=0)

    # Convert to target dtype
    if dtype == "bfloat16":
        # NumPy doesn't natively support bfloat16; keep as float32
        result = result.astype(np.float32)
    else:
        result = result.astype(dtype)

    return result, norm_meta


def to_pytorch(
    data: np.ndarray,
    normalize_method: str | None = None,
    add_batch_dim: bool = True,
    add_channel_dim: bool = True,
    device: str = "cpu",
) -> Any:
    """Convert a NOVA data array to a PyTorch tensor.

    Parameters
    ----------
    data : numpy.ndarray
        Input data array.
    normalize_method : str, optional
        Normalization method to apply.
    add_batch_dim : bool
        Whether to add a batch dimension.
    add_channel_dim : bool
        Whether to add a channel dimension.
    device : str
        PyTorch device ('cpu', 'cuda', etc.).

    Returns
    -------
    torch.Tensor
        PyTorch tensor.

    Raises
    ------
    ImportError
        If PyTorch is not installed.
    """
    try:
        import torch
    except ImportError:
        raise ImportError(
            "PyTorch is required for tensor export. "
            "Install it with: pip install torch"
        )

    tensor_data, _ = to_tensor(
        data,
        dtype="float32",
        normalize_method=normalize_method,
        add_batch_dim=add_batch_dim,
        add_channel_dim=add_channel_dim,
    )

    return torch.from_numpy(tensor_data).to(device)


def to_jax(
    data: np.ndarray,
    normalize_method: str | None = None,
    add_batch_dim: bool = True,
    add_channel_dim: bool = True,
) -> Any:
    """Convert a NOVA data array to a JAX array.

    Parameters
    ----------
    data : numpy.ndarray
        Input data array.
    normalize_method : str, optional
        Normalization method to apply.
    add_batch_dim : bool
        Whether to add a batch dimension.
    add_channel_dim : bool
        Whether to add a channel dimension.

    Returns
    -------
    jax.numpy.ndarray
        JAX array.

    Raises
    ------
    ImportError
        If JAX is not installed.
    """
    try:
        import jax.numpy as jnp
    except ImportError:
        raise ImportError(
            "JAX is required for tensor export. "
            "Install it with: pip install jax jaxlib"
        )

    tensor_data, _ = to_tensor(
        data,
        dtype="float32",
        normalize_method=normalize_method,
        add_batch_dim=add_batch_dim,
        add_channel_dim=add_channel_dim,
    )

    return jnp.array(tensor_data)
