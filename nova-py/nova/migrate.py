"""Batch migration tool for FITS-to-NOVA archive conversion.

Provides both a library API and CLI integration for converting entire
directories of FITS files to NOVA format.  Supports parallel conversion,
dry-run analysis, incremental migration, and round-trip verification.

Library usage::

    from nova.migrate import migrate_directory, MigrationReport

    report = migrate_directory(
        src="raw_fits/",
        dst="nova_archive/",
        parallel=4,
        verify=True,
    )
    print(report.summary())

CLI usage (via ``nova migrate``)::

    nova migrate raw_fits/ nova_archive/ --parallel 4 --verify
    nova migrate raw_fits/ nova_archive/ --dry-run
"""

from __future__ import annotations

import os
import time
import hashlib
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


# Default FITS extensions to recognise
FITS_EXTENSIONS = {".fits", ".fit", ".fts", ".fits.gz", ".fit.gz", ".fts.gz"}


@dataclass
class FileResult:
    """Result of converting a single FITS file."""

    fits_path: str
    nova_path: str
    success: bool
    elapsed_seconds: float = 0.0
    error: str = ""
    verified: bool = False
    fits_size_bytes: int = 0
    nova_size_bytes: int = 0


@dataclass
class MigrationReport:
    """Aggregate report for a batch migration run."""

    source_dir: str
    dest_dir: str
    total_files: int = 0
    converted: int = 0
    skipped: int = 0
    failed: int = 0
    verified: int = 0
    elapsed_seconds: float = 0.0
    file_results: list[FileResult] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Migration report: {self.source_dir} -> {self.dest_dir}",
            f"  Total FITS files found: {self.total_files}",
            f"  Converted:  {self.converted}",
            f"  Skipped:    {self.skipped}",
            f"  Failed:     {self.failed}",
            f"  Verified:   {self.verified}",
            f"  Time:       {self.elapsed_seconds:.2f}s",
        ]
        if self.file_results:
            total_fits = sum(r.fits_size_bytes for r in self.file_results)
            total_nova = sum(r.nova_size_bytes for r in self.file_results if r.success)
            if total_fits > 0:
                ratio = total_nova / total_fits
                lines.append(
                    f"  Size ratio: {ratio:.2f}x "
                    f"({total_fits / 1e6:.1f} MB -> {total_nova / 1e6:.1f} MB)"
                )
        for r in self.file_results:
            if not r.success:
                lines.append(f"  FAILED: {r.fits_path} -- {r.error}")
        return "\n".join(lines)


def discover_fits_files(directory: str | Path) -> list[Path]:
    """Recursively find all FITS files under *directory*."""
    directory = Path(directory)
    results: list[Path] = []
    for root, _dirs, files in os.walk(directory):
        for name in sorted(files):
            lower = name.lower()
            if any(lower.endswith(ext) for ext in FITS_EXTENSIONS):
                results.append(Path(root) / name)
    return results


def _nova_dest_path(fits_path: Path, src_root: Path, dst_root: Path) -> Path:
    """Compute the destination NOVA path preserving directory structure."""
    rel = fits_path.relative_to(src_root)
    # Replace .fits (and variants) with .nova.zarr
    stem = rel.stem
    if stem.endswith(".fits"):
        stem = stem[: -len(".fits")]
    nova_name = stem + ".nova.zarr"
    return dst_root / rel.parent / nova_name


def _convert_one(
    fits_path: Path,
    nova_path: Path,
    *,
    all_extensions: bool = True,
    verify: bool = False,
) -> FileResult:
    """Convert a single FITS file and optionally verify round-trip."""
    result = FileResult(
        fits_path=str(fits_path),
        nova_path=str(nova_path),
        success=False,
    )
    result.fits_size_bytes = fits_path.stat().st_size

    start = time.perf_counter()
    try:
        from nova.fits_converter import from_fits

        nova_path.parent.mkdir(parents=True, exist_ok=True)
        ds = from_fits(fits_path, nova_path, all_extensions=all_extensions)
        ds.close()
        result.success = True

        # Compute NOVA store size
        result.nova_size_bytes = sum(
            f.stat().st_size for f in nova_path.rglob("*") if f.is_file()
        )

        # Optional round-trip verification
        if verify:
            result.verified = _verify_roundtrip(fits_path, nova_path)

    except Exception as exc:
        result.error = str(exc)

    result.elapsed_seconds = time.perf_counter() - start
    return result


def _verify_roundtrip(fits_path: Path, nova_path: Path) -> bool:
    """Verify that NOVA data matches the original FITS data."""
    try:
        from astropy.io import fits as pyfits
        from nova.container import NovaDataset

        with pyfits.open(str(fits_path)) as hdul:
            for hdu in hdul:
                if hdu.data is not None and hdu.data.ndim >= 2:
                    original = np.array(hdu.data, dtype=float)
                    break
            else:
                return True  # no image data to compare

        ds = NovaDataset(nova_path, mode="r")
        try:
            if ds.data is not None:
                converted = np.array(ds.data, dtype=float)
                if original.shape != converted.shape:
                    return False
                return np.allclose(original, converted, equal_nan=True)
            return True
        finally:
            ds.close()
    except (OSError, ValueError, TypeError, KeyError):
        return False


def migrate_directory(
    src: str | Path,
    dst: str | Path,
    *,
    parallel: int = 1,
    all_extensions: bool = True,
    verify: bool = False,
    dry_run: bool = False,
    incremental: bool = False,
) -> MigrationReport:
    """Convert all FITS files under *src* to NOVA format under *dst*.

    Parameters
    ----------
    src : str or Path
        Source directory containing FITS files.
    dst : str or Path
        Destination directory for NOVA stores.
    parallel : int
        Number of parallel conversion processes (default 1).
    all_extensions : bool
        Convert all FITS extensions (MEF support).
    verify : bool
        Verify round-trip fidelity for each file.
    dry_run : bool
        Only list files that would be converted, do not write anything.
    incremental : bool
        Skip files whose NOVA output already exists.

    Returns
    -------
    MigrationReport
        Summary of the conversion run.
    """
    src = Path(src).resolve()
    dst = Path(dst).resolve()

    if parallel < 1:
        raise ValueError(f"parallel must be >= 1, got {parallel}")

    fits_files = discover_fits_files(src)

    report = MigrationReport(
        source_dir=str(src),
        dest_dir=str(dst),
        total_files=len(fits_files),
    )

    if dry_run:
        for fp in fits_files:
            nova_path = _nova_dest_path(fp, src, dst)
            report.file_results.append(FileResult(
                fits_path=str(fp),
                nova_path=str(nova_path),
                success=True,
                fits_size_bytes=fp.stat().st_size,
            ))
        report.converted = len(fits_files)
        return report

    start_total = time.perf_counter()
    tasks: list[tuple[Path, Path]] = []

    for fp in fits_files:
        nova_path = _nova_dest_path(fp, src, dst)
        if incremental and nova_path.exists():
            report.skipped += 1
            continue
        tasks.append((fp, nova_path))

    if parallel <= 1:
        for fp, np_ in tasks:
            r = _convert_one(fp, np_, all_extensions=all_extensions, verify=verify)
            report.file_results.append(r)
            if r.success:
                report.converted += 1
                if r.verified:
                    report.verified += 1
            else:
                report.failed += 1
    else:
        with ProcessPoolExecutor(max_workers=parallel) as pool:
            futures = {
                pool.submit(
                    _convert_one, fp, np_,
                    all_extensions=all_extensions,
                    verify=verify,
                ): (fp, np_)
                for fp, np_ in tasks
            }
            for future in as_completed(futures):
                r = future.result()
                report.file_results.append(r)
                if r.success:
                    report.converted += 1
                    if r.verified:
                        report.verified += 1
                else:
                    report.failed += 1

    report.elapsed_seconds = time.perf_counter() - start_total
    return report
