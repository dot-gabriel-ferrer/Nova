"""Tutorial 08: NOVA Pipeline Framework.

Demonstrates how to build reproducible data-processing pipelines using
the Pipeline, Step, and PipelineLog classes.

Steps:
  1. Create a Pipeline with named Steps
  2. Run the pipeline on synthetic test data
  3. Inspect the PipelineLog (step names, timestamps, checksums)
  4. Serialize the pipeline definition to JSON and reload it
  5. Serialize the execution log to JSON

Run:
    cd nova-py
    python tutorials/08_pipeline_framework.py
"""

from __future__ import annotations

import json

import numpy as np


# ---- helper processing functions used as pipeline steps ----

def subtract_background(data: np.ndarray, *, percentile: float = 50.0) -> np.ndarray:
    """Estimate and subtract a flat background level."""
    bg = np.percentile(data, percentile)
    return data - bg


def clip_negatives(data: np.ndarray) -> np.ndarray:
    """Clip pixel values below zero."""
    return np.clip(data, 0.0, None)


def normalize_peak(data: np.ndarray) -> np.ndarray:
    """Normalize so that the peak value equals 1."""
    mx = data.max()
    if mx == 0:
        return data
    return data / mx


def main() -> None:
    from nova.pipeline import Pipeline, PipelineLog

    print("=" * 70)
    print("  NOVA Tutorial 08: Pipeline Framework")
    print("=" * 70)
    print()

    # -- Step 1: Build a pipeline ------------------------------------------
    print("Step 1: Create a Pipeline with named Steps")
    print("-" * 70)

    pipe = Pipeline("image_reduce", version="0.3", metadata={"band": "r"})
    pipe.add_step(
        "background_subtract",
        subtract_background,
        description="Subtract median background",
        percentile=50.0,
    )
    pipe.add_step(
        "clip_negatives",
        clip_negatives,
        description="Zero out negative pixels",
    )
    pipe.add_step(
        "normalize",
        normalize_peak,
        description="Normalize to peak = 1",
    )

    print(f"  Pipeline name:  {pipe.name}")
    print(f"  Version:        {pipe.version}")
    print(f"  Number of steps: {len(pipe.steps)}")
    print(f"  Step names:      {pipe.step_names}")
    print()

    # -- Step 2: Run the pipeline ------------------------------------------
    print("Step 2: Run the pipeline on a synthetic 512x512 image")
    print("-" * 70)

    rng = np.random.default_rng(42)
    image = rng.normal(loc=1000.0, scale=30.0, size=(512, 512)).astype(np.float64)
    # Plant a bright source
    yy, xx = np.ogrid[-256:256, -256:256]
    image += 5000.0 * np.exp(-(xx**2 + yy**2) / (2 * 3.0**2))

    result = pipe.run(image)

    print(f"  Input  range: [{image.min():.1f}, {image.max():.1f}]")
    print(f"  Output range: [{result.min():.4f}, {result.max():.4f}]")
    print(f"  Output shape: {result.shape}")
    print()

    # -- Step 3: Inspect the PipelineLog -----------------------------------
    print("Step 3: Inspect the PipelineLog")
    print("-" * 70)

    log: PipelineLog = pipe.log  # type: ignore[assignment]
    print(f"  Pipeline:       {log.pipeline_name} v{log.version}")
    print(f"  Executed at:    {log.executed_at}")
    print(f"  Total duration: {log.total_duration_seconds:.6f} s")
    print(f"  Steps executed: {len(log.steps)}")
    print()
    for sl in log.steps:
        print(f"    [{sl.step_name}]")
        print(f"      Function:     {sl.func_name}")
        print(f"      Input shape:  {sl.input_shape}  dtype={sl.input_dtype}")
        print(f"      Output shape: {sl.output_shape} dtype={sl.output_dtype}")
        print(f"      Duration:     {sl.duration_seconds:.6f} s")
        print(f"      Input SHA256: {sl.input_sha256[:16]}...")
        print(f"      Output SHA256:{sl.output_sha256[:16]}...")
    print()

    # -- Step 4: Save and reload pipeline definition -----------------------
    print("Step 4: Serialize pipeline definition to JSON and reload")
    print("-" * 70)

    pipe_dict = pipe.to_dict()
    pipe_json = json.dumps(pipe_dict, indent=2)
    print(f"  Serialized JSON length: {len(pipe_json)} chars")
    print(f"  Keys: {list(pipe_dict.keys())}")

    # Rebuild from dict using a function registry
    registry = {
        "subtract_background": subtract_background,
        "clip_negatives": clip_negatives,
        "normalize_peak": normalize_peak,
    }
    pipe2 = Pipeline.from_dict(pipe_dict, func_registry=registry)
    print(f"  Reloaded pipeline: {pipe2.name} v{pipe2.version}")
    print(f"  Steps match: {pipe2.step_names == pipe.step_names}")
    print()

    # -- Step 5: Serialize execution log -----------------------------------
    print("Step 5: Serialize the execution log to JSON")
    print("-" * 70)

    log_json = log.to_json(indent=2)
    print(f"  Log JSON length: {len(log_json)} chars")
    # Round-trip
    log2 = PipelineLog.from_json(log_json)
    print(f"  Round-trip pipeline: {log2.pipeline_name}")
    print(f"  Round-trip steps:    {len(log2.steps)}")
    print(f"  Round-trip match:    {log2.pipeline_name == log.pipeline_name}")
    print()

    print("=" * 70)
    print("  OK  Tutorial complete -- pipeline framework explored.")
    print("=" * 70)


if __name__ == "__main__":
    main()
