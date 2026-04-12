"""NOVA command-line interface.

Provides CLI tools for NOVA format operations:
- nova convert: FITS↔NOVA conversion
- nova info: Display dataset information
- nova validate: Validate NOVA stores
- nova benchmark: Run performance benchmarks
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments. Uses sys.argv if not provided.

    Returns
    -------
    int
        Exit code (0 for success, 1 for failure).
    """
    parser = argparse.ArgumentParser(
        prog="nova",
        description="NOVA — Next-generation Open Volumetric Archive CLI",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # convert command
    convert_parser = subparsers.add_parser(
        "convert",
        help="Convert between FITS and NOVA formats",
    )
    convert_parser.add_argument("input", help="Input file path")
    convert_parser.add_argument("output", help="Output file path")
    convert_parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing output file",
    )

    # info command
    info_parser = subparsers.add_parser(
        "info",
        help="Display NOVA dataset information",
    )
    info_parser.add_argument("path", help="Path to NOVA store")

    # validate command
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate a NOVA store against the specification",
    )
    validate_parser.add_argument("path", help="Path to NOVA store")

    # benchmark command
    bench_parser = subparsers.add_parser(
        "benchmark",
        help="Run NOVA vs FITS performance benchmarks",
    )
    bench_parser.add_argument(
        "--size", type=int, default=1024,
        help="Image size (NxN pixels, default: 1024)",
    )
    bench_parser.add_argument(
        "--pattern", default="realistic_sky",
        choices=["gaussian_noise", "gradient", "sparse", "realistic_sky"],
        help="Data pattern (default: realistic_sky)",
    )

    # migrate command
    migrate_parser = subparsers.add_parser(
        "migrate",
        help="Batch convert a directory of FITS files to NOVA format",
    )
    migrate_parser.add_argument("source", help="Source directory containing FITS files")
    migrate_parser.add_argument("dest", help="Destination directory for NOVA stores")
    migrate_parser.add_argument(
        "--parallel", type=int, default=1,
        help="Number of parallel conversion workers (default: 1)",
    )
    migrate_parser.add_argument(
        "--verify", action="store_true",
        help="Verify round-trip fidelity for each converted file",
    )
    migrate_parser.add_argument(
        "--dry-run", action="store_true",
        help="List files that would be converted without writing anything",
    )
    migrate_parser.add_argument(
        "--incremental", action="store_true",
        help="Skip files whose NOVA output already exists",
    )

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "convert":
        return _cmd_convert(args)
    elif args.command == "info":
        return _cmd_info(args)
    elif args.command == "validate":
        return _cmd_validate(args)
    elif args.command == "benchmark":
        return _cmd_benchmark(args)
    elif args.command == "migrate":
        return _cmd_migrate(args)

    return 0


def _cmd_convert(args: argparse.Namespace) -> int:
    """Handle the convert command."""
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        return 1

    # Determine conversion direction
    input_ext = input_path.suffix.lower()
    is_fits_input = input_ext in (".fits", ".fit", ".fts")
    is_nova_input = input_path.name.endswith(".nova.zarr")

    if is_fits_input:
        # FITS → NOVA
        from nova.fits_converter import from_fits
        print(f"Converting FITS → NOVA: {input_path} → {output_path}")
        start = time.perf_counter()
        ds = from_fits(input_path, output_path)
        elapsed = time.perf_counter() - start
        shape = "unknown"
        if ds.data is not None:
            import numpy as np
            shape = str(np.array(ds.data).shape)
        ds.close()
        print(f"  Done in {elapsed:.2f}s (shape: {shape})")
        return 0

    elif is_nova_input:
        # NOVA → FITS
        from nova.fits_converter import to_fits
        print(f"Converting NOVA → FITS: {input_path} → {output_path}")
        start = time.perf_counter()
        to_fits(input_path, output_path, overwrite=args.overwrite)
        elapsed = time.perf_counter() - start
        print(f"  Done in {elapsed:.2f}s")
        return 0

    else:
        print(
            f"Error: Cannot determine conversion direction. "
            f"Input must be .fits or .nova.zarr",
            file=sys.stderr,
        )
        return 1


def _cmd_info(args: argparse.Namespace) -> int:
    """Handle the info command."""
    store_path = Path(args.path)
    if not store_path.exists():
        print(f"Error: Store not found: {store_path}", file=sys.stderr)
        return 1

    from nova.container import NovaDataset

    print(f"NOVA Dataset: {store_path}")
    print("=" * 60)

    ds = NovaDataset(store_path, mode="r")

    # Metadata
    meta_path = store_path / "nova_metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"  Version:    {meta.get('nova:version', 'unknown')}")
        print(f"  Created:    {meta.get('nova:created', 'unknown')}")
        print(f"  Data Level: {meta.get('nova:data_level', 'unknown')}")
        print(f"  Type:       {meta.get('@type', 'unknown')}")

    # Data arrays
    import numpy as np
    if ds.data is not None:
        arr = ds.data
        print(f"\n  Science Data:")
        print(f"    Shape: {arr.shape}")
        print(f"    Dtype: {arr.dtype}")
        print(f"    Chunks: {arr.chunks if hasattr(arr, 'chunks') else 'N/A'}")

    if ds.uncertainty is not None:
        arr = ds.uncertainty
        print(f"\n  Uncertainty:")
        print(f"    Shape: {arr.shape}")
        print(f"    Dtype: {arr.dtype}")

    if ds.mask is not None:
        arr = ds.mask
        print(f"\n  Mask:")
        print(f"    Shape: {arr.shape}")
        print(f"    Dtype: {arr.dtype}")

    # WCS
    if ds.wcs is not None:
        print(f"\n  WCS:")
        print(f"    Axes: {ds.wcs.naxes}")
        for ax in ds.wcs.axes:
            print(f"    Axis {ax.index}: {ax.ctype} ({ax.axis_type})")
        if ds.wcs.projection:
            print(f"    Projection: {ds.wcs.projection.name}")
        if ds.wcs.celestial_frame:
            print(f"    Frame: {ds.wcs.celestial_frame.system}")

    # Provenance
    if ds.provenance is not None:
        print(f"\n  Provenance:")
        print(f"    Entities:   {len(ds.provenance.entities)}")
        print(f"    Activities: {len(ds.provenance.activities)}")
        print(f"    Agents:     {len(ds.provenance.agents)}")

    # Store size
    total_size = sum(
        f.stat().st_size for f in store_path.rglob("*") if f.is_file()
    )
    print(f"\n  Total Store Size: {total_size / (1024*1024):.2f} MB")

    ds.close()
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    """Handle the validate command."""
    store_path = Path(args.path)

    from nova.validation import validate_store

    print(f"Validating NOVA store: {store_path}")
    print("=" * 60)

    results = validate_store(store_path)
    total_errors = 0

    for filename, errors in results.items():
        if errors:
            print(f"\n  ✗ {filename} ({len(errors)} errors)")
            for error in errors:
                print(f"    - {error}")
            total_errors += len(errors)
        else:
            print(f"  ✓ {filename}")

    print()
    if total_errors == 0:
        print("Validation passed ✓")
        return 0
    else:
        print(f"Validation failed: {total_errors} error(s) found")
        return 1


def _cmd_benchmark(args: argparse.Namespace) -> int:
    """Handle the benchmark command."""
    from nova.benchmarks import run_full_comparison

    shape = (args.size, args.size)
    print(f"Running NOVA vs FITS benchmarks ({args.size}x{args.size}, {args.pattern})")
    print("=" * 60)

    results = run_full_comparison(
        shape=shape,
        dtype="float64",
        pattern=args.pattern,
    )

    for comparison in results:
        print(comparison.summary())
        print()

    return 0


def _cmd_migrate(args: argparse.Namespace) -> int:
    """Handle the migrate command."""
    from nova.migrate import migrate_directory

    print(f"NOVA batch migration: {args.source} -> {args.dest}")
    if args.dry_run:
        print("  (dry-run mode -- nothing will be written)")
    print("=" * 60)

    report = migrate_directory(
        src=args.source,
        dst=args.dest,
        parallel=args.parallel,
        verify=args.verify,
        dry_run=args.dry_run,
        incremental=args.incremental,
    )

    print(report.summary())
    return 0 if report.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
