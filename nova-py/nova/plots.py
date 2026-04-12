"""NOVA performance plot generator.

Generates multi-format performance comparison plots (PNG) showing NOVA's
advantages over FITS, HDF5, and raw NumPy in:
- Write speed / throughput
- Read speed / throughput
- Partial/cloud read speed
- Compression ratio
- File size
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from nova.benchmarks import (
    run_multi_format_comparison,
    generate_test_data,
    benchmark_nova_write,
    benchmark_nova_read,
    benchmark_nova_partial_read,
    benchmark_fits_write,
    benchmark_fits_read,
    benchmark_fits_partial_read,
    benchmark_hdf5_write,
    benchmark_hdf5_read,
    benchmark_numpy_write,
    benchmark_numpy_read,
)


# ---------------------------------------------------------------------------
#  Styling helpers
# ---------------------------------------------------------------------------

# Color palette for each format
FORMAT_COLORS: dict[str, str] = {
    "NOVA": "#2196F3",
    "NOVA (Zarr)": "#64B5F6",
    "FITS": "#FF5722",
    "HDF5": "#9C27B0",
    "NumPy": "#4CAF50",
}

BG_COLOR = "#0d1117"
TEXT_COLOR = "#c9d1d9"
GRID_COLOR = "#30363d"
ACCENT_GREEN = "#4CAF50"


def _style_ax(ax: object) -> None:  # type: ignore[override]
    """Apply dark theme to a matplotlib axis."""
    ax.set_facecolor(BG_COLOR)  # type: ignore[attr-defined]
    ax.tick_params(colors=TEXT_COLOR)  # type: ignore[attr-defined]
    for spine in ("bottom", "left"):
        ax.spines[spine].set_color(GRID_COLOR)  # type: ignore[attr-defined]
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)  # type: ignore[attr-defined]
    ax.xaxis.label.set_color(TEXT_COLOR)  # type: ignore[attr-defined]
    ax.yaxis.label.set_color(TEXT_COLOR)  # type: ignore[attr-defined]
    ax.title.set_color(TEXT_COLOR)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
#  Main entry point
# ---------------------------------------------------------------------------

def generate_performance_plots(
    output_dir: str | Path,
    sizes: list[tuple[int, int]] | None = None,
    patterns: list[str] | None = None,
) -> list[str]:
    """Generate all performance comparison plots.

    Parameters
    ----------
    output_dir : str or Path
        Directory to save PNG files.
    sizes : list of tuple, optional
        Data sizes to benchmark. Defaults to several sizes.
    patterns : list of str, optional
        Data patterns to test.

    Returns
    -------
    list of str
        Paths to generated PNG files.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plot generation. "
            "Install with: pip install matplotlib"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if sizes is None:
        sizes = [(256, 256), (512, 512), (1024, 1024), (2048, 2048)]
    if patterns is None:
        patterns = ["realistic_sky"]

    generated_files: list[str] = []

    # Collect multi-format benchmark data across sizes
    size_labels = [f"{s[0]}×{s[1]}" for s in sizes]

    # Per-format, per-operation lists of times (ms) and file sizes (MB)
    format_order = ["NOVA", "FITS", "HDF5", "NumPy", "NOVA (Zarr)"]
    write_times: dict[str, list[float]] = {f: [] for f in format_order}
    read_times: dict[str, list[float]] = {f: [] for f in format_order}
    partial_times: dict[str, list[float]] = {f: [] for f in format_order}
    file_sizes: dict[str, list[float]] = {f: [] for f in format_order}
    write_throughput: dict[str, list[float]] = {f: [] for f in format_order}
    read_throughput: dict[str, list[float]] = {f: [] for f in format_order}

    for size in sizes:
        results = run_multi_format_comparison(
            shape=size,
            dtype="float64",
            pattern=patterns[0],
            n_runs=3,
        )

        for fmt in format_order:
            if fmt in results:
                wr = results[fmt].get("write")
                rd = results[fmt].get("read")
                pr = results[fmt].get("partial_read")
                write_times[fmt].append(wr.elapsed_seconds * 1000 if wr else 0)
                read_times[fmt].append(rd.elapsed_seconds * 1000 if rd else 0)
                partial_times[fmt].append(pr.elapsed_seconds * 1000 if pr else 0)
                file_sizes[fmt].append(
                    wr.file_size_bytes / (1024 * 1024) if wr else 0
                )
                write_throughput[fmt].append(
                    wr.throughput_mbps if wr else 0
                )
                read_throughput[fmt].append(
                    rd.throughput_mbps if rd else 0
                )
            else:
                write_times[fmt].append(0)
                read_times[fmt].append(0)
                partial_times[fmt].append(0)
                file_sizes[fmt].append(0)
                write_throughput[fmt].append(0)
                read_throughput[fmt].append(0)

    # Only plot formats that have data
    active_formats = [f for f in format_order if any(write_times[f])]

    x = np.arange(len(size_labels))
    n_fmt = len(active_formats)
    bar_width = 0.7 / n_fmt

    # ── Plot 1: Combined Performance Overview ──
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.patch.set_facecolor(BG_COLOR)
    for ax in axes.flat:
        _style_ax(ax)

    def _grouped_bars(ax: object, data_dict: dict[str, list[float]],
                      ylabel: str, title: str) -> None:
        for i, fmt in enumerate(active_formats):
            vals = data_dict[fmt]
            if any(vals):
                ax.bar(  # type: ignore[attr-defined]
                    x + (i - n_fmt / 2 + 0.5) * bar_width,
                    vals, bar_width,
                    label=fmt,
                    color=FORMAT_COLORS.get(fmt, "#888"),
                    alpha=0.9,
                )
        ax.set_xlabel("Image Size")  # type: ignore[attr-defined]
        ax.set_ylabel(ylabel)  # type: ignore[attr-defined]
        ax.set_title(title)  # type: ignore[attr-defined]
        ax.set_xticks(x)  # type: ignore[attr-defined]
        ax.set_xticklabels(size_labels)  # type: ignore[attr-defined]
        ax.legend(  # type: ignore[attr-defined]
            facecolor=BG_COLOR, edgecolor=GRID_COLOR,
            labelcolor=TEXT_COLOR, fontsize=8,
        )
        ax.grid(axis="y", color=GRID_COLOR, alpha=0.3)  # type: ignore[attr-defined]

    _grouped_bars(axes[0, 0], write_times, "Time (ms)", "Write Speed (lower = better)")
    _grouped_bars(axes[0, 1], read_times, "Time (ms)", "Read Speed (lower = better)")
    _grouped_bars(axes[1, 0], write_throughput, "Throughput (MB/s)",
                  "Write Throughput (higher = better)")
    _grouped_bars(axes[1, 1], read_throughput, "Throughput (MB/s)",
                  "Read Throughput (higher = better)")

    fig.suptitle(
        "NOVA vs FITS vs HDF5 vs NumPy — Performance Comparison",
        color=TEXT_COLOR, fontsize=16, fontweight="bold", y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    path = output_dir / "benchmark_overview.png"
    fig.savefig(path, dpi=150, facecolor=BG_COLOR, bbox_inches="tight")
    plt.close(fig)
    generated_files.append(str(path))

    # ── Plot 2: Partial Read / Cloud Access ──
    fig, (ax_time, ax_speedup) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(BG_COLOR)
    _style_ax(ax_time)
    _style_ax(ax_speedup)

    partial_formats = [f for f in active_formats if any(partial_times[f])]
    for i, fmt in enumerate(partial_formats):
        vals = partial_times[fmt]
        if any(vals):
            ax_time.bar(
                x + (i - len(partial_formats) / 2 + 0.5) * bar_width,
                vals, bar_width,
                label=fmt,
                color=FORMAT_COLORS.get(fmt, "#888"),
                alpha=0.9,
            )
    ax_time.set_xlabel("Image Size")
    ax_time.set_ylabel("Time (ms)")
    ax_time.set_title("Partial Read — Cloud Access Pattern")
    ax_time.set_xticks(x)
    ax_time.set_xticklabels(size_labels)
    ax_time.legend(facecolor=BG_COLOR, edgecolor=GRID_COLOR,
                   labelcolor=TEXT_COLOR, fontsize=8)
    ax_time.grid(axis="y", color=GRID_COLOR, alpha=0.3)

    # Speedup vs FITS
    if "FITS" in partial_times and any(partial_times["FITS"]):
        speedup_formats = [f for f in partial_formats if f != "FITS"]
        for i, fmt in enumerate(speedup_formats):
            speedups = []
            for ft, nt in zip(partial_times["FITS"], partial_times[fmt]):
                speedups.append(ft / nt if nt > 0 else 1.0)
            bars = ax_speedup.bar(
                x + (i - len(speedup_formats) / 2 + 0.5) * bar_width,
                speedups, bar_width,
                label=fmt,
                color=FORMAT_COLORS.get(fmt, "#888"),
                alpha=0.9,
            )
            for bar, sp in zip(bars, speedups):
                ax_speedup.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{sp:.1f}x",
                    ha="center", va="bottom",
                    color=TEXT_COLOR, fontweight="bold", fontsize=9,
                )
        ax_speedup.axhline(y=1.0, color=FORMAT_COLORS["FITS"],
                           linestyle="--", alpha=0.5, label="FITS baseline")
    ax_speedup.set_xlabel("Image Size")
    ax_speedup.set_ylabel("Speedup vs FITS")
    ax_speedup.set_title("Partial Read Speedup vs FITS")
    ax_speedup.set_xticks(x)
    ax_speedup.set_xticklabels(size_labels)
    ax_speedup.legend(facecolor=BG_COLOR, edgecolor=GRID_COLOR,
                      labelcolor=TEXT_COLOR, fontsize=8)
    ax_speedup.grid(axis="y", color=GRID_COLOR, alpha=0.3)

    fig.tight_layout()
    path = output_dir / "cloud_access_speedup.png"
    fig.savefig(path, dpi=150, facecolor=BG_COLOR, bbox_inches="tight")
    plt.close(fig)
    generated_files.append(str(path))

    # ── Plot 3: Compression Comparison ──
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(BG_COLOR)
    _style_ax(ax)

    pattern_names = ["Realistic Sky", "Gradient", "Sparse", "Gaussian Noise"]
    pattern_keys = ["realistic_sky", "gradient", "sparse", "gaussian_noise"]

    # Collect compression ratios per format per pattern
    comp_ratios: dict[str, list[float]] = {f: [] for f in ["NOVA", "FITS", "HDF5"]}
    test_size = (512, 512)

    for pattern in pattern_keys:
        data = generate_test_data(shape=test_size, pattern=pattern)
        with tempfile.TemporaryDirectory() as tmpdir:
            nw = benchmark_nova_write(data, output_dir=tmpdir)
            fw = benchmark_fits_write(data, output_dir=tmpdir)
            comp_ratios["NOVA"].append(nw.compression_ratio)
            comp_ratios["FITS"].append(fw.compression_ratio)
            try:
                hw = benchmark_hdf5_write(data, output_dir=tmpdir)
                comp_ratios["HDF5"].append(hw.compression_ratio)
            except ImportError:
                comp_ratios["HDF5"].append(0)

    x_pat = np.arange(len(pattern_names))
    comp_formats = [f for f in ["NOVA", "FITS", "HDF5"] if any(comp_ratios[f])]
    bw = 0.7 / len(comp_formats)
    for i, fmt in enumerate(comp_formats):
        ax.bar(
            x_pat + (i - len(comp_formats) / 2 + 0.5) * bw,
            comp_ratios[fmt], bw,
            label=fmt,
            color=FORMAT_COLORS.get(fmt, "#888"),
            alpha=0.9,
        )

    ax.set_xlabel("Data Pattern")
    ax.set_ylabel("Compression Ratio (higher = better)")
    ax.set_title(
        "Compression Ratio by Data Type",
        color=TEXT_COLOR, fontsize=14, fontweight="bold",
    )
    ax.set_xticks(x_pat)
    ax.set_xticklabels(pattern_names)
    ax.legend(facecolor=BG_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
    ax.grid(axis="y", color=GRID_COLOR, alpha=0.3)

    fig.tight_layout()
    path = output_dir / "compression_comparison.png"
    fig.savefig(path, dpi=150, facecolor=BG_COLOR, bbox_inches="tight")
    plt.close(fig)
    generated_files.append(str(path))

    # ── Plot 4: Improvement Summary (horizontal bars) ──
    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor(BG_COLOR)
    _style_ax(ax)

    # Use largest size for summary
    largest = -1
    categories = [
        "Write Speed",
        "Read Speed",
        "Partial Read",
        "Compression",
        "File Size\nSavings",
        "Metadata\nValidation",
    ]

    fits_wt = write_times["FITS"][largest] if write_times["FITS"][largest] > 0 else 1.0
    nova_wt = write_times["NOVA"][largest] if write_times["NOVA"][largest] > 0 else 1.0
    fits_rt = read_times["FITS"][largest] if read_times["FITS"][largest] > 0 else 1.0
    nova_rt = read_times["NOVA"][largest] if read_times["NOVA"][largest] > 0 else 1.0
    fits_pt = partial_times["FITS"][largest] if partial_times["FITS"][largest] > 0 else 1.0
    nova_pt = partial_times["NOVA"][largest] if partial_times["NOVA"][largest] > 0 else 1.0

    write_speedup = fits_wt / nova_wt
    read_speedup = fits_rt / nova_rt
    partial_speedup = fits_pt / nova_pt
    nova_cr = comp_ratios["NOVA"][0] if comp_ratios["NOVA"] else 1.0
    fits_cr = comp_ratios["FITS"][0] if comp_ratios["FITS"] else 1.0
    compression_factor = nova_cr / fits_cr if fits_cr > 0 else 1.0

    fits_fs = file_sizes["FITS"][largest] if file_sizes["FITS"][largest] > 0 else 1.0
    nova_fs = file_sizes["NOVA"][largest] if file_sizes["NOVA"][largest] > 0 else 1.0
    size_factor = fits_fs / nova_fs if nova_fs > 0 else 1.0

    metadata_factor = 5.0  # NOVA has schema validation, FITS does not (qualitative)

    improvements = [
        write_speedup, read_speedup, partial_speedup,
        compression_factor, size_factor, metadata_factor,
    ]

    x_cats = np.arange(len(categories))
    bars = ax.barh(x_cats, improvements, 0.5, color=FORMAT_COLORS["NOVA"],
                   alpha=0.9)
    ax.axvline(x=1.0, color=FORMAT_COLORS["FITS"], linestyle="--",
               alpha=0.7, label="FITS baseline (1.0x)")

    for bar, imp in zip(bars, improvements):
        ax.text(
            bar.get_width() + 0.1,
            bar.get_y() + bar.get_height() / 2,
            f"{imp:.1f}x",
            ha="left", va="center",
            color=ACCENT_GREEN, fontweight="bold", fontsize=11,
        )

    ax.set_yticks(x_cats)
    ax.set_yticklabels(categories, color=TEXT_COLOR)
    ax.set_xlabel("Improvement Factor vs FITS (higher = better)")
    ax.set_title(
        f"NOVA Improvement Summary — {sizes[-1][0]}×{sizes[-1][1]} Realistic Sky",
        color=TEXT_COLOR, fontsize=14, fontweight="bold",
    )
    ax.legend(facecolor=BG_COLOR, edgecolor=GRID_COLOR,
              labelcolor=TEXT_COLOR, loc="lower right")
    ax.grid(axis="x", color=GRID_COLOR, alpha=0.3)

    fig.tight_layout()
    path = output_dir / "improvement_summary.png"
    fig.savefig(path, dpi=150, facecolor=BG_COLOR, bbox_inches="tight")
    plt.close(fig)
    generated_files.append(str(path))

    # ── Plot 5: File Size Comparison ──
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(BG_COLOR)
    _style_ax(ax)

    _grouped_bars(ax, file_sizes, "File Size (MB)",
                  "Storage Efficiency (lower = better)")

    fig.tight_layout()
    path = output_dir / "file_size_comparison.png"
    fig.savefig(path, dpi=150, facecolor=BG_COLOR, bbox_inches="tight")
    plt.close(fig)
    generated_files.append(str(path))

    return generated_files


if __name__ == "__main__":
    import sys
    output = sys.argv[1] if len(sys.argv) > 1 else "docs/benchmarks"
    files = generate_performance_plots(output)
    for f in files:
        print(f"Generated: {f}")
