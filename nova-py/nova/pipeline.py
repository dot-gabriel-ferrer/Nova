"""Pipeline framework for NOVA.

Provides a declarative pipeline system that records every processing step
applied to a dataset, including parameters, timestamps, and data checksums.
Pipelines can be saved, loaded, replayed on different datasets, and
inspected for full reproducibility.

A Pipeline is a sequence of Steps.  Each Step wraps a callable together
with its keyword arguments.  When a pipeline is executed, each step's
output becomes the next step's input, and a PipelineLog is written into
the NOVA dataset's metadata so every consumer can see exactly how the
data was produced.

Example
-------
>>> from nova.pipeline import Pipeline, Step
>>> p = Pipeline("ccd_reduction")
>>> p.add_step("bias", subtract_bias, master_bias=bias_array)
>>> p.add_step("flat", apply_flat, master_flat=flat_array)
>>> result = p.run(raw_image)
>>> p.log   # list of dicts with timestamps and checksums
"""

from __future__ import annotations

import copy
import datetime
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


# -------------------------------------------------------------------
#  Step
# -------------------------------------------------------------------

@dataclass
class Step:
    """A single processing step inside a pipeline.

    Parameters
    ----------
    name : str
        Human-readable step name (e.g. ``'bias_subtract'``).
    func : callable
        The function to execute.  Must accept a NumPy array as its first
        positional argument and return a NumPy array.
    params : dict
        Keyword arguments forwarded to *func*.
    description : str
        Optional longer description of what this step does.
    """

    name: str
    func: Callable[..., np.ndarray]
    params: dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def execute(self, data: np.ndarray) -> np.ndarray:
        """Run the step on *data* and return the result.

        Parameters
        ----------
        data : np.ndarray
            Input array.

        Returns
        -------
        np.ndarray
            Processed array.

        Raises
        ------
        TypeError
            If *data* is not an ndarray or *func* does not return one.
        """
        if not isinstance(data, np.ndarray):
            raise TypeError(
                "Step input must be a numpy ndarray, got "
                + type(data).__name__
            )
        result = self.func(data, **self.params)
        if not isinstance(result, np.ndarray):
            raise TypeError(
                "Step '{}' must return a numpy ndarray, got {}".format(
                    self.name, type(result).__name__
                )
            )
        return result


# -------------------------------------------------------------------
#  StepLog -- immutable record of a step execution
# -------------------------------------------------------------------

@dataclass
class StepLog:
    """Immutable record of one step execution.

    Parameters
    ----------
    step_name : str
        Name of the step.
    func_name : str
        Qualified name of the callable.
    params_summary : dict
        Serializable summary of the parameters used.
    input_sha256 : str
        SHA-256 hex digest of the input array bytes.
    output_sha256 : str
        SHA-256 hex digest of the output array bytes.
    input_shape : tuple[int, ...]
        Shape of the input array.
    output_shape : tuple[int, ...]
        Shape of the output array.
    input_dtype : str
        Data type of the input.
    output_dtype : str
        Data type of the output.
    started_at : str
        ISO-8601 timestamp when the step started.
    ended_at : str
        ISO-8601 timestamp when the step ended.
    duration_seconds : float
        Wall-clock duration in seconds.
    description : str
        Step description.
    """

    step_name: str
    func_name: str
    params_summary: dict[str, Any]
    input_sha256: str
    output_sha256: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    input_dtype: str
    output_dtype: str
    started_at: str
    ended_at: str
    duration_seconds: float
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary."""
        return {
            "step_name": self.step_name,
            "func_name": self.func_name,
            "params": self.params_summary,
            "input_sha256": self.input_sha256,
            "output_sha256": self.output_sha256,
            "input_shape": list(self.input_shape),
            "output_shape": list(self.output_shape),
            "input_dtype": self.input_dtype,
            "output_dtype": self.output_dtype,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_seconds": self.duration_seconds,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StepLog:
        """Deserialize from a dictionary."""
        return cls(
            step_name=d["step_name"],
            func_name=d["func_name"],
            params_summary=d.get("params", {}),
            input_sha256=d["input_sha256"],
            output_sha256=d["output_sha256"],
            input_shape=tuple(d["input_shape"]),
            output_shape=tuple(d["output_shape"]),
            input_dtype=d["input_dtype"],
            output_dtype=d["output_dtype"],
            started_at=d["started_at"],
            ended_at=d["ended_at"],
            duration_seconds=d["duration_seconds"],
            description=d.get("description", ""),
        )


# -------------------------------------------------------------------
#  PipelineLog -- full execution record
# -------------------------------------------------------------------

@dataclass
class PipelineLog:
    """Complete record of a pipeline execution.

    Parameters
    ----------
    pipeline_name : str
        Name of the pipeline.
    version : str
        Pipeline version string.
    executed_at : str
        ISO-8601 timestamp of execution start.
    total_duration_seconds : float
        Total wall-clock time.
    steps : list[StepLog]
        Ordered list of step logs.
    metadata : dict
        Extra user-supplied metadata.
    """

    pipeline_name: str
    version: str
    executed_at: str
    total_duration_seconds: float
    steps: list[StepLog] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary suitable for JSON."""
        return {
            "pipeline_name": self.pipeline_name,
            "version": self.version,
            "executed_at": self.executed_at,
            "total_duration_seconds": self.total_duration_seconds,
            "steps": [s.to_dict() for s in self.steps],
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PipelineLog:
        """Deserialize from a dictionary."""
        return cls(
            pipeline_name=d["pipeline_name"],
            version=d.get("version", ""),
            executed_at=d["executed_at"],
            total_duration_seconds=d["total_duration_seconds"],
            steps=[StepLog.from_dict(s) for s in d.get("steps", [])],
            metadata=d.get("metadata", {}),
        )

    @classmethod
    def from_json(cls, text: str) -> PipelineLog:
        """Deserialize from a JSON string."""
        return cls.from_dict(json.loads(text))


# -------------------------------------------------------------------
#  Helpers
# -------------------------------------------------------------------

def _array_sha256(arr: np.ndarray) -> str:
    """Compute SHA-256 hex digest of an array's raw bytes."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def _summarize_params(params: dict[str, Any]) -> dict[str, Any]:
    """Create a JSON-safe summary of step parameters.

    NumPy arrays are replaced by shape/dtype descriptions.
    Other non-serializable objects are converted to their repr.
    """
    summary: dict[str, Any] = {}
    for key, val in params.items():
        if isinstance(val, np.ndarray):
            summary[key] = {
                "type": "ndarray",
                "shape": list(val.shape),
                "dtype": str(val.dtype),
            }
        elif isinstance(val, (int, float, bool, str, type(None))):
            summary[key] = val
        elif isinstance(val, (list, tuple)):
            summary[key] = list(val)
        elif isinstance(val, dict):
            summary[key] = _summarize_params(val)
        else:
            summary[key] = repr(val)
    return summary


def _func_qualname(func: Callable) -> str:
    """Return a human-readable qualified name for a callable."""
    module = getattr(func, "__module__", "")
    qualname = getattr(func, "__qualname__", getattr(func, "__name__", repr(func)))
    if module:
        return "{}.{}".format(module, qualname)
    return qualname


# -------------------------------------------------------------------
#  Pipeline
# -------------------------------------------------------------------

class Pipeline:
    """Declarative, reproducible processing pipeline.

    A pipeline is a named, ordered sequence of processing steps.  Each
    step is a callable that transforms a NumPy array.  When the pipeline
    is executed, every step is logged with timestamps, checksums, and
    parameter summaries so the processing history is fully reproducible.

    Parameters
    ----------
    name : str
        Pipeline name (e.g. ``'ccd_reduction'``, ``'spectral_extraction'``).
    version : str
        Optional version tag.
    metadata : dict or None
        Arbitrary metadata attached to the pipeline log (instrument,
        observer, comments, etc.).

    Examples
    --------
    >>> p = Pipeline("quick_reduce")
    >>> p.add_step("bias", subtract_bias, master_bias=bias_frame)
    >>> p.add_step("flat", apply_flat, master_flat=flat_frame)
    >>> result = p.run(raw_data)
    >>> print(p.log.to_json())
    """

    def __init__(
        self,
        name: str,
        version: str = "1.0",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not name:
            raise ValueError("Pipeline name must not be empty.")
        self.name = name
        self.version = version
        self.metadata: dict[str, Any] = metadata or {}
        self._steps: list[Step] = []
        self._log: PipelineLog | None = None

    # -- step management ------------------------------------------------

    def add_step(
        self,
        name: str,
        func: Callable[..., np.ndarray],
        description: str = "",
        **params: Any,
    ) -> None:
        """Append a processing step.

        Parameters
        ----------
        name : str
            Step name.
        func : callable
            Processing function (first arg = ndarray, returns ndarray).
        description : str
            Optional description.
        **params
            Keyword arguments forwarded to *func* at execution time.
        """
        if not name:
            raise ValueError("Step name must not be empty.")
        self._steps.append(Step(
            name=name, func=func, params=params, description=description,
        ))

    def insert_step(
        self,
        index: int,
        name: str,
        func: Callable[..., np.ndarray],
        description: str = "",
        **params: Any,
    ) -> None:
        """Insert a step at *index*.

        Parameters
        ----------
        index : int
            Position (0-based) where the step is inserted.
        name : str
            Step name.
        func : callable
            Processing function.
        description : str
            Optional description.
        **params
            Keyword arguments forwarded to *func*.
        """
        if not name:
            raise ValueError("Step name must not be empty.")
        self._steps.insert(index, Step(
            name=name, func=func, params=params, description=description,
        ))

    def remove_step(self, name: str) -> None:
        """Remove the first step whose name matches *name*.

        Raises
        ------
        KeyError
            If no step with *name* exists.
        """
        for i, step in enumerate(self._steps):
            if step.name == name:
                del self._steps[i]
                return
        raise KeyError("No step named '{}'.".format(name))

    @property
    def steps(self) -> list[Step]:
        """Ordered list of pipeline steps (read-only copy)."""
        return list(self._steps)

    @property
    def step_names(self) -> list[str]:
        """List of step names in order."""
        return [s.name for s in self._steps]

    # -- execution ------------------------------------------------------

    def run(self, data: np.ndarray) -> np.ndarray:
        """Execute the full pipeline on *data*.

        Parameters
        ----------
        data : np.ndarray
            Input data array.

        Returns
        -------
        np.ndarray
            Processed array after all steps.

        Raises
        ------
        TypeError
            If *data* is not a NumPy array.
        RuntimeError
            If the pipeline has no steps.
        """
        if not isinstance(data, np.ndarray):
            raise TypeError("Pipeline input must be a numpy ndarray.")
        if not self._steps:
            raise RuntimeError("Pipeline '{}' has no steps.".format(self.name))

        pipe_start = datetime.datetime.now(datetime.timezone.utc)
        step_logs: list[StepLog] = []
        current = data

        for step in self._steps:
            t0 = datetime.datetime.now(datetime.timezone.utc)
            in_hash = _array_sha256(current)
            in_shape = current.shape
            in_dtype = str(current.dtype)

            current = step.execute(current)

            t1 = datetime.datetime.now(datetime.timezone.utc)
            out_hash = _array_sha256(current)
            dt = (t1 - t0).total_seconds()

            step_logs.append(StepLog(
                step_name=step.name,
                func_name=_func_qualname(step.func),
                params_summary=_summarize_params(step.params),
                input_sha256=in_hash,
                output_sha256=out_hash,
                input_shape=in_shape,
                output_shape=current.shape,
                input_dtype=in_dtype,
                output_dtype=str(current.dtype),
                started_at=t0.isoformat(),
                ended_at=t1.isoformat(),
                duration_seconds=dt,
                description=step.description,
            ))

        pipe_end = datetime.datetime.now(datetime.timezone.utc)
        self._log = PipelineLog(
            pipeline_name=self.name,
            version=self.version,
            executed_at=pipe_start.isoformat(),
            total_duration_seconds=(pipe_end - pipe_start).total_seconds(),
            steps=step_logs,
            metadata=self.metadata,
        )
        return current

    @property
    def log(self) -> PipelineLog | None:
        """The execution log from the most recent ``run()``, or None."""
        return self._log

    # -- serialization --------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize the pipeline definition (steps + metadata) to a dict.

        Note: step functions are stored by qualified name only; the caller
        must supply a function registry to reconstruct them.
        """
        return {
            "pipeline_name": self.name,
            "version": self.version,
            "metadata": self.metadata,
            "steps": [
                {
                    "name": s.name,
                    "func": _func_qualname(s.func),
                    "params": _summarize_params(s.params),
                    "description": s.description,
                }
                for s in self._steps
            ],
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize the pipeline definition to JSON."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(
        cls,
        d: dict[str, Any],
        func_registry: dict[str, Callable] | None = None,
    ) -> Pipeline:
        """Reconstruct a Pipeline from a dict.

        Parameters
        ----------
        d : dict
            Pipeline definition (as produced by ``to_dict()``).
        func_registry : dict or None
            Mapping of function qualified names to callables.  If a step
            references a function not in the registry, a placeholder is
            used that raises ``NotImplementedError`` when called.

        Returns
        -------
        Pipeline
        """
        registry = func_registry or {}
        p = cls(
            name=d["pipeline_name"],
            version=d.get("version", "1.0"),
            metadata=d.get("metadata", {}),
        )
        for sd in d.get("steps", []):
            fname = sd["func"]
            func = registry.get(fname)
            if func is None:
                # Build a placeholder so the pipeline structure is intact
                def _placeholder(data: np.ndarray, _n: str = fname, **kw: Any) -> np.ndarray:
                    raise NotImplementedError(
                        "Function '{}' is not in the registry.".format(_n)
                    )
                func = _placeholder
            p.add_step(
                name=sd["name"],
                func=func,
                description=sd.get("description", ""),
            )
        return p

    @classmethod
    def from_json(
        cls,
        text: str,
        func_registry: dict[str, Callable] | None = None,
    ) -> Pipeline:
        """Reconstruct a Pipeline from a JSON string."""
        return cls.from_dict(json.loads(text), func_registry)

    def __repr__(self) -> str:
        steps_str = ", ".join(s.name for s in self._steps)
        return "Pipeline('{}', steps=[{}])".format(self.name, steps_str)

    def __len__(self) -> int:
        return len(self._steps)
