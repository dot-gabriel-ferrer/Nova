"""Native data operations for NOVA with change tracking.

Provides in-place and copy-on-write array operations that automatically
record every transformation applied to the data.  This module is the
bridge between the raw NumPy arrays stored in NOVA datasets and the
pipeline framework: each operation generates a log entry that can later
be embedded in provenance metadata.

Design principles
-----------------
- Every public function returns a new array (no silent in-place mutation).
- Every public function appends to an OperationHistory if one is provided.
- Histories can be serialized to JSON and stored inside NOVA metadata.
- Operations cover the most common astronomical data manipulations so
  users do not need to leave NOVA to process their data.

Example
-------
>>> from nova.operations import OperationHistory, op_subtract, op_divide
>>> hist = OperationHistory()
>>> calibrated = op_subtract(raw, bias, history=hist, label="bias")
>>> calibrated = op_divide(calibrated, flat, history=hist, label="flat")
>>> print(hist.to_json())
"""

from __future__ import annotations

import datetime
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# -------------------------------------------------------------------
#  Operation record
# -------------------------------------------------------------------

@dataclass
class OperationRecord:
    """Immutable record of a single data operation.

    Parameters
    ----------
    label : str
        Human-readable label (e.g. ``'bias_subtract'``).
    operation : str
        Canonical operation name (``'subtract'``, ``'divide'``, ...).
    input_sha256 : str
        SHA-256 hex digest of the input array.
    output_sha256 : str
        SHA-256 hex digest of the result array.
    input_shape : tuple[int, ...]
        Shape of the input array.
    output_shape : tuple[int, ...]
        Shape of the result.
    input_dtype : str
        Data type of the input.
    output_dtype : str
        Data type of the result.
    params : dict
        Extra parameters (scalar values, shapes of secondary operands, etc.).
    timestamp : str
        ISO-8601 timestamp.
    """

    label: str
    operation: str
    input_sha256: str
    output_sha256: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    input_dtype: str
    output_dtype: str
    params: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary."""
        return {
            "label": self.label,
            "operation": self.operation,
            "input_sha256": self.input_sha256,
            "output_sha256": self.output_sha256,
            "input_shape": list(self.input_shape),
            "output_shape": list(self.output_shape),
            "input_dtype": self.input_dtype,
            "output_dtype": self.output_dtype,
            "params": self.params,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> OperationRecord:
        """Deserialize from a dictionary."""
        return cls(
            label=d["label"],
            operation=d["operation"],
            input_sha256=d["input_sha256"],
            output_sha256=d["output_sha256"],
            input_shape=tuple(d["input_shape"]),
            output_shape=tuple(d["output_shape"]),
            input_dtype=d["input_dtype"],
            output_dtype=d["output_dtype"],
            params=d.get("params", {}),
            timestamp=d.get("timestamp", ""),
        )


# -------------------------------------------------------------------
#  Operation history
# -------------------------------------------------------------------

class OperationHistory:
    """Append-only log of data operations.

    Parameters
    ----------
    records : list[OperationRecord] or None
        Pre-existing records (e.g. loaded from a saved file).

    Examples
    --------
    >>> hist = OperationHistory()
    >>> result = op_subtract(raw, bias, history=hist, label="bias")
    >>> len(hist)
    1
    """

    def __init__(
        self, records: list[OperationRecord] | None = None,
    ) -> None:
        self._records: list[OperationRecord] = list(records) if records else []

    def append(self, record: OperationRecord) -> None:
        """Add a record to the history."""
        self._records.append(record)

    @property
    def records(self) -> list[OperationRecord]:
        """Ordered list of operation records (read-only copy)."""
        return list(self._records)

    def to_dict_list(self) -> list[dict[str, Any]]:
        """Serialize all records to a list of dicts."""
        return [r.to_dict() for r in self._records]

    def to_json(self, indent: int = 2) -> str:
        """Serialize all records to a JSON string."""
        return json.dumps(self.to_dict_list(), indent=indent)

    @classmethod
    def from_json(cls, text: str) -> OperationHistory:
        """Deserialize from a JSON string."""
        raw = json.loads(text)
        return cls(records=[OperationRecord.from_dict(d) for d in raw])

    def __len__(self) -> int:
        return len(self._records)

    def __repr__(self) -> str:
        return "OperationHistory({} records)".format(len(self._records))


# -------------------------------------------------------------------
#  Internal helpers
# -------------------------------------------------------------------

def _sha256(arr: np.ndarray) -> str:
    """Hex digest of an array's raw bytes."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def _now_iso() -> str:
    """Current UTC time as ISO-8601 string."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _record(
    label: str,
    operation: str,
    inp: np.ndarray,
    out: np.ndarray,
    history: OperationHistory | None,
    extra_params: dict[str, Any] | None = None,
) -> None:
    """Write an OperationRecord into *history* if it is not None."""
    if history is None:
        return
    history.append(OperationRecord(
        label=label,
        operation=operation,
        input_sha256=_sha256(inp),
        output_sha256=_sha256(out),
        input_shape=inp.shape,
        output_shape=out.shape,
        input_dtype=str(inp.dtype),
        output_dtype=str(out.dtype),
        params=extra_params or {},
        timestamp=_now_iso(),
    ))


def _validate_array(arr: Any, name: str = "input") -> np.ndarray:
    """Ensure *arr* is an ndarray."""
    if not isinstance(arr, np.ndarray):
        raise TypeError(
            "{} must be a numpy ndarray, got {}".format(name, type(arr).__name__)
        )
    return arr


# -------------------------------------------------------------------
#  Arithmetic operations
# -------------------------------------------------------------------

def op_add(
    data: np.ndarray,
    operand: np.ndarray | float,
    *,
    history: OperationHistory | None = None,
    label: str = "add",
) -> np.ndarray:
    """Element-wise addition with tracking.

    Parameters
    ----------
    data : np.ndarray
        Primary data array.
    operand : np.ndarray or float
        Value(s) to add.
    history : OperationHistory or None
        If provided, a record is appended.
    label : str
        Label for the operation record.

    Returns
    -------
    np.ndarray
        ``data + operand``.
    """
    data = _validate_array(data, "data")
    result = data + operand
    params: dict[str, Any] = {}
    if isinstance(operand, np.ndarray):
        params["operand_shape"] = list(operand.shape)
        params["operand_dtype"] = str(operand.dtype)
    else:
        params["operand_value"] = float(operand)
    _record(label, "add", data, result, history, params)
    return result


def op_subtract(
    data: np.ndarray,
    operand: np.ndarray | float,
    *,
    history: OperationHistory | None = None,
    label: str = "subtract",
) -> np.ndarray:
    """Element-wise subtraction with tracking.

    Parameters
    ----------
    data : np.ndarray
        Primary data array.
    operand : np.ndarray or float
        Value(s) to subtract.
    history : OperationHistory or None
        If provided, a record is appended.
    label : str
        Label for the operation record.

    Returns
    -------
    np.ndarray
        ``data - operand``.
    """
    data = _validate_array(data, "data")
    result = data - operand
    params: dict[str, Any] = {}
    if isinstance(operand, np.ndarray):
        params["operand_shape"] = list(operand.shape)
        params["operand_dtype"] = str(operand.dtype)
    else:
        params["operand_value"] = float(operand)
    _record(label, "subtract", data, result, history, params)
    return result


def op_multiply(
    data: np.ndarray,
    operand: np.ndarray | float,
    *,
    history: OperationHistory | None = None,
    label: str = "multiply",
) -> np.ndarray:
    """Element-wise multiplication with tracking.

    Parameters
    ----------
    data : np.ndarray
        Primary data array.
    operand : np.ndarray or float
        Value(s) to multiply by.
    history : OperationHistory or None
        If provided, a record is appended.
    label : str
        Label for the operation record.

    Returns
    -------
    np.ndarray
        ``data * operand``.
    """
    data = _validate_array(data, "data")
    result = data * operand
    params: dict[str, Any] = {}
    if isinstance(operand, np.ndarray):
        params["operand_shape"] = list(operand.shape)
        params["operand_dtype"] = str(operand.dtype)
    else:
        params["operand_value"] = float(operand)
    _record(label, "multiply", data, result, history, params)
    return result


def op_divide(
    data: np.ndarray,
    operand: np.ndarray | float,
    *,
    history: OperationHistory | None = None,
    label: str = "divide",
) -> np.ndarray:
    """Element-wise division with tracking.

    Zero-valued divisors produce zero in the output (not NaN/Inf).

    Parameters
    ----------
    data : np.ndarray
        Primary data array.
    operand : np.ndarray or float
        Value(s) to divide by.
    history : OperationHistory or None
        If provided, a record is appended.
    label : str
        Label for the operation record.

    Returns
    -------
    np.ndarray
        ``data / operand`` (safe division).
    """
    data = _validate_array(data, "data")
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(operand != 0, data / operand, 0.0)
    result = result.astype(data.dtype)
    params: dict[str, Any] = {}
    if isinstance(operand, np.ndarray):
        params["operand_shape"] = list(operand.shape)
        params["operand_dtype"] = str(operand.dtype)
    else:
        params["operand_value"] = float(operand)
    _record(label, "divide", data, result, history, params)
    return result


# -------------------------------------------------------------------
#  Clipping and masking
# -------------------------------------------------------------------

def op_clip(
    data: np.ndarray,
    *,
    lower: float | None = None,
    upper: float | None = None,
    history: OperationHistory | None = None,
    label: str = "clip",
) -> np.ndarray:
    """Clip array values to [lower, upper] with tracking.

    Parameters
    ----------
    data : np.ndarray
        Input data.
    lower : float or None
        Minimum value (None = no lower bound).
    upper : float or None
        Maximum value (None = no upper bound).
    history : OperationHistory or None
        If provided, a record is appended.
    label : str
        Label for the record.

    Returns
    -------
    np.ndarray
        Clipped array.
    """
    data = _validate_array(data, "data")
    result = np.clip(data, lower, upper)
    params = {"lower": lower, "upper": upper}
    _record(label, "clip", data, result, history, params)
    return result


def op_mask_replace(
    data: np.ndarray,
    mask: np.ndarray,
    fill_value: float = 0.0,
    *,
    history: OperationHistory | None = None,
    label: str = "mask_replace",
) -> np.ndarray:
    """Replace masked pixels with *fill_value*.

    Parameters
    ----------
    data : np.ndarray
        Input data.
    mask : np.ndarray
        Boolean mask (True = bad pixel).
    fill_value : float
        Replacement value.
    history : OperationHistory or None
        If provided, a record is appended.
    label : str
        Label for the record.

    Returns
    -------
    np.ndarray
        Array with masked pixels replaced.
    """
    data = _validate_array(data, "data")
    mask = _validate_array(mask, "mask")
    result = data.copy()
    result[mask.astype(bool)] = fill_value
    params = {"fill_value": fill_value, "masked_pixels": int(np.sum(mask.astype(bool)))}
    _record(label, "mask_replace", data, result, history, params)
    return result


# -------------------------------------------------------------------
#  Statistical operations
# -------------------------------------------------------------------

def op_normalize(
    data: np.ndarray,
    *,
    method: str = "minmax",
    history: OperationHistory | None = None,
    label: str = "normalize",
) -> np.ndarray:
    """Normalize array values with tracking.

    Parameters
    ----------
    data : np.ndarray
        Input data.
    method : str
        ``'minmax'`` scales to [0, 1]; ``'zscore'`` centres to mean=0, std=1.
    history : OperationHistory or None
        If provided, a record is appended.
    label : str
        Label for the record.

    Returns
    -------
    np.ndarray
        Normalized array (float64).

    Raises
    ------
    ValueError
        If *method* is not recognised.
    """
    data = _validate_array(data, "data")
    if method == "minmax":
        lo, hi = float(np.nanmin(data)), float(np.nanmax(data))
        rng = hi - lo
        if rng == 0:
            result = np.zeros_like(data, dtype=np.float64)
        else:
            result = (data.astype(np.float64) - lo) / rng
    elif method == "zscore":
        mean = float(np.nanmean(data))
        std = float(np.nanstd(data))
        if std == 0:
            result = np.zeros_like(data, dtype=np.float64)
        else:
            result = (data.astype(np.float64) - mean) / std
    else:
        raise ValueError("Unknown normalization method '{}'.".format(method))
    params = {"method": method}
    _record(label, "normalize", data, result, history, params)
    return result


def op_rebin(
    data: np.ndarray,
    factor: int,
    *,
    history: OperationHistory | None = None,
    label: str = "rebin",
) -> np.ndarray:
    """Rebin a 2-D array by an integer factor (sum of pixels).

    Parameters
    ----------
    data : np.ndarray
        2-D input array.
    factor : int
        Rebinning factor (must divide both dimensions evenly).
    history : OperationHistory or None
        If provided, a record is appended.
    label : str
        Label for the record.

    Returns
    -------
    np.ndarray
        Rebinned array with shape ``(H/factor, W/factor)``.

    Raises
    ------
    ValueError
        If *factor* does not divide the array dimensions.
    """
    data = _validate_array(data, "data")
    if data.ndim != 2:
        raise ValueError("op_rebin requires a 2-D array.")
    h, w = data.shape
    if h % factor != 0 or w % factor != 0:
        raise ValueError(
            "Rebin factor {} does not divide shape ({}, {}).".format(factor, h, w)
        )
    result = data.reshape(h // factor, factor, w // factor, factor).sum(axis=(1, 3))
    params = {"factor": factor}
    _record(label, "rebin", data, result, history, params)
    return result


# -------------------------------------------------------------------
#  Convenience: combine
# -------------------------------------------------------------------

def op_combine(
    arrays: list[np.ndarray],
    method: str = "median",
    *,
    history: OperationHistory | None = None,
    label: str = "combine",
) -> np.ndarray:
    """Combine a list of arrays into one using *method*.

    Parameters
    ----------
    arrays : list[np.ndarray]
        Arrays to combine (must share the same shape).
    method : str
        ``'median'``, ``'mean'``, or ``'sum'``.
    history : OperationHistory or None
        If provided, a record is appended.
    label : str
        Label for the record.

    Returns
    -------
    np.ndarray
        Combined array.

    Raises
    ------
    ValueError
        If *arrays* is empty or shapes mismatch, or *method* is unknown.
    """
    if not arrays:
        raise ValueError("Cannot combine an empty list of arrays.")
    for i, a in enumerate(arrays):
        _validate_array(a, "arrays[{}]".format(i))
    shape0 = arrays[0].shape
    for i, a in enumerate(arrays):
        if a.shape != shape0:
            raise ValueError(
                "Shape mismatch: arrays[0] has shape {} but arrays[{}] has {}.".format(
                    shape0, i, a.shape
                )
            )

    stack = np.stack(arrays, axis=0)
    if method == "median":
        result = np.median(stack, axis=0)
    elif method == "mean":
        result = np.mean(stack, axis=0)
    elif method == "sum":
        result = np.sum(stack, axis=0)
    else:
        raise ValueError("Unknown combine method '{}'.".format(method))
    result = result.astype(arrays[0].dtype)

    params = {"method": method, "n_arrays": len(arrays)}
    _record(label, "combine", arrays[0], result, history, params)
    return result
