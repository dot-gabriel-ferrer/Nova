"""Tutorial 09: NOVA Native Operations with Tracking.

Demonstrates the tracked arithmetic and array operations provided by
nova.operations, including the OperationHistory audit trail.

Steps:
  1. Basic arithmetic: op_add, op_subtract, op_multiply, op_divide
  2. Clipping, masking, and normalization
  3. Rebinning a 2-D image
  4. Combining multiple images (median stack)
  5. Reviewing and serializing the OperationHistory

Run:
    cd nova-py
    python tutorials/09_native_operations.py
"""

from __future__ import annotations

import json

import numpy as np


def main() -> None:
    from nova.operations import (
        OperationHistory,
        op_add,
        op_clip,
        op_combine,
        op_divide,
        op_mask_replace,
        op_multiply,
        op_normalize,
        op_rebin,
        op_subtract,
    )

    print("=" * 70)
    print("  NOVA Tutorial 09: Native Operations with Tracking")
    print("=" * 70)
    print()

    rng = np.random.default_rng(42)
    hist = OperationHistory()

    # -- 1. Tracked arithmetic ---------------------------------------------
    print("Step 1: Tracked arithmetic operations")
    print("-" * 70)

    science = rng.normal(1000.0, 30.0, (256, 256)).astype(np.float64)
    dark = rng.normal(100.0, 5.0, (256, 256)).astype(np.float64)
    flat = rng.uniform(0.95, 1.05, (256, 256)).astype(np.float64)

    # Subtract dark current
    reduced = op_subtract(science, dark, history=hist, label="dark_subtract")
    print(f"  After dark subtract:  mean={reduced.mean():.2f}")

    # Divide by flat field
    reduced = op_divide(reduced, flat, history=hist, label="flat_divide")
    print(f"  After flat divide:    mean={reduced.mean():.2f}")

    # Add a constant pedestal
    reduced = op_add(reduced, 500.0, history=hist, label="add_pedestal")
    print(f"  After add pedestal:   mean={reduced.mean():.2f}")

    # Scale by gain
    reduced = op_multiply(reduced, 2.5, history=hist, label="gain_multiply")
    print(f"  After gain multiply:  mean={reduced.mean():.2f}")
    print()

    # -- 2. Clip, mask, normalize ------------------------------------------
    print("Step 2: Clipping, masking, and normalization")
    print("-" * 70)

    # Clip extreme values
    clipped = op_clip(reduced, lower=0.0, upper=5000.0, history=hist, label="clip")
    print(f"  After clip [0, 5000]: min={clipped.min():.2f}  max={clipped.max():.2f}")

    # Create a bad-pixel mask and replace
    mask = np.zeros((256, 256), dtype=bool)
    mask[100:110, 100:110] = True  # 10x10 dead region
    masked = op_mask_replace(clipped, mask, fill_value=0.0, history=hist,
                             label="bad_pixel_replace")
    print(f"  Replaced {mask.sum()} bad pixels with 0.0")
    print(f"  Value at (105,105): {masked[105, 105]:.2f}")

    # Normalize
    normed = op_normalize(masked, method="minmax", history=hist, label="minmax_norm")
    print(f"  After minmax normalize: min={normed.min():.4f}  max={normed.max():.4f}")

    normed_z = op_normalize(masked, method="zscore", history=hist, label="zscore_norm")
    print(f"  After zscore normalize: mean={normed_z.mean():.4f}  std={normed_z.std():.4f}")
    print()

    # -- 3. Rebinning ------------------------------------------------------
    print("Step 3: Rebin 256x256 image by factor 4")
    print("-" * 70)

    rebinned = op_rebin(clipped, factor=4, history=hist, label="rebin_4x")
    print(f"  Input shape:  {clipped.shape}")
    print(f"  Output shape: {rebinned.shape}")
    print(f"  Sum preserved: input={clipped.sum():.1f}  output={rebinned.sum():.1f}")
    print()

    # -- 4. Combine multiple images ----------------------------------------
    print("Step 4: Median-combine 5 noisy frames")
    print("-" * 70)

    frames = [rng.normal(1000.0, 30.0, (128, 128)) for _ in range(5)]
    combined = op_combine(frames, method="median", history=hist, label="median_stack")
    single_std = frames[0].std()
    combined_std = combined.std()
    print(f"  Single frame std: {single_std:.2f}")
    print(f"  Combined std:     {combined_std:.2f}")
    print(f"  Noise reduction:  {single_std / combined_std:.2f}x")
    print()

    # -- 5. Operation history ----------------------------------------------
    print("Step 5: Review OperationHistory")
    print("-" * 70)

    print(f"  Total recorded operations: {len(hist.records)}")
    print()
    for i, rec in enumerate(hist.records):
        print(f"    [{i}] {rec.label:20s}  "
              f"in={rec.input_shape}  out={rec.output_shape}  "
              f"ts={rec.timestamp}")

    print()
    print("  Serializing history to JSON...")
    history_json = hist.to_json(indent=2)
    print(f"  JSON length: {len(history_json)} chars")

    # Round-trip
    hist2 = OperationHistory.from_json(history_json)
    print(f"  Round-trip records: {len(hist2.records)}")
    print(f"  Labels match: "
          f"{[r.label for r in hist2.records] == [r.label for r in hist.records]}")
    print()

    print("=" * 70)
    print("  OK  Tutorial complete -- native operations explored.")
    print("=" * 70)


if __name__ == "__main__":
    main()
