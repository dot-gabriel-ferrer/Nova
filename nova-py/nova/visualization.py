"""NOVA visualization tools for astronomical data.

Easy-to-use display functions optimised for astronomical images, spectra,
and NOVA metadata inspection.  All functions produce publication-quality
``matplotlib`` figures with sensible defaults (colour maps, stretch, labels).

Functions
---------
- ``display_image`` — Quick-look image display with optional WCS grid.
- ``display_rgb`` — Compose and display a three-colour image.
- ``display_spectrum`` — 1-D spectral plot with optional line markers.
- ``display_histogram`` — Pixel-value histogram with statistics overlay.
- ``display_cutout`` — Zoom into a region with coordinate annotations.
- ``display_comparison`` — Side-by-side comparison of two images.
- ``display_mosaic`` — Grid of multiple images.
- ``display_provenance`` — Visual provenance chain diagram.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np


def _ensure_matplotlib():
    """Import and configure matplotlib or raise a helpful error."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install with: pip install 'nova-astro[plots]'"
        )


def _apply_stretch(
    data: np.ndarray,
    stretch: Literal["linear", "log", "sqrt", "asinh", "power"] = "linear",
    power: float = 2.0,
) -> np.ndarray:
    """Apply a pixel-value stretch for display.

    Parameters
    ----------
    data : ndarray
        Input image data.
    stretch : str
        Stretch type.
    power : float
        Exponent for power stretch.

    Returns
    -------
    ndarray
        Stretched data.
    """
    arr = np.asarray(data, dtype=np.float64)
    # Normalize to [0, 1]
    vmin = np.nanmin(arr)
    vmax = np.nanmax(arr)
    if vmax == vmin:
        return np.zeros_like(arr)
    norm = (arr - vmin) / (vmax - vmin)

    if stretch == "linear":
        return norm
    elif stretch == "log":
        return np.log1p(norm * 1000) / np.log1p(1000)
    elif stretch == "sqrt":
        return np.sqrt(norm)
    elif stretch == "asinh":
        return np.arcsinh(norm * 10) / np.arcsinh(10)
    elif stretch == "power":
        return np.power(norm, 1.0 / power)
    else:
        raise ValueError(f"Unknown stretch: {stretch!r}")


def _percentile_interval(
    data: np.ndarray,
    percentile: float = 99.5,
) -> tuple[float, float]:
    """Compute display limits from percentile interval."""
    arr = data[np.isfinite(data)]
    if arr.size == 0:
        return (0.0, 1.0)
    low = float(np.percentile(arr, 100 - percentile))
    high = float(np.percentile(arr, percentile))
    return (low, high)


# ──────────────────────────────────────────────────────────────────────────
#  Core Display Functions
# ──────────────────────────────────────────────────────────────────────────

def display_image(
    data: np.ndarray,
    title: str = "",
    cmap: str = "gray",
    stretch: Literal["linear", "log", "sqrt", "asinh", "power"] = "linear",
    percentile: float = 99.5,
    colorbar: bool = True,
    figsize: tuple[float, float] = (8, 8),
    output_path: str | Path | None = None,
    wcs_info: dict[str, Any] | None = None,
) -> Any:
    """Display a 2-D astronomical image.

    Parameters
    ----------
    data : ndarray
        2-D image data.
    title : str
        Figure title.
    cmap : str
        Matplotlib colormap (default: ``"gray"``).
    stretch : str
        Pixel stretch: ``"linear"``, ``"log"``, ``"sqrt"``, ``"asinh"``, ``"power"``.
    percentile : float
        Percentile interval for display limits.
    colorbar : bool
        Show colorbar.
    figsize : tuple
        Figure size in inches.
    output_path : str or Path, optional
        If provided, saves the figure to this path instead of returning it.
    wcs_info : dict, optional
        WCS metadata dictionary to add coordinate labels.

    Returns
    -------
    matplotlib.figure.Figure or None
        The figure object, or None if saved to file.
    """
    plt = _ensure_matplotlib()
    data = np.asarray(data, dtype=np.float64)

    fig, ax = plt.subplots(figsize=figsize)

    # Apply stretch and limits
    vmin, vmax = _percentile_interval(data, percentile)
    display_data = np.clip(data, vmin, vmax)
    display_data = _apply_stretch(display_data, stretch=stretch)

    im = ax.imshow(display_data, origin="lower", cmap=cmap, aspect="equal")

    if colorbar:
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("Pixel Value", color="#c9d1d9")
        cb.ax.tick_params(colors="#c9d1d9")

    # Add WCS info as axis labels if available
    if wcs_info and "nova:axes" in wcs_info:
        axes = wcs_info["nova:axes"]
        if len(axes) >= 2:
            ax.set_xlabel(axes[0].get("nova:ctype", "X (pixels)"))
            ax.set_ylabel(axes[1].get("nova:ctype", "Y (pixels)"))
    else:
        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")

    if title:
        ax.set_title(title, fontsize=14, fontweight="bold")

    fig.tight_layout()

    if output_path:
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        return None
    return fig


def display_rgb(
    red: np.ndarray,
    green: np.ndarray,
    blue: np.ndarray,
    title: str = "RGB Composite",
    stretch: Literal["linear", "log", "sqrt", "asinh"] = "asinh",
    percentile: float = 99.5,
    figsize: tuple[float, float] = (8, 8),
    output_path: str | Path | None = None,
) -> Any:
    """Create and display an RGB composite image.

    Parameters
    ----------
    red, green, blue : ndarray
        2-D arrays for each colour channel (same shape).
    title : str
        Figure title.
    stretch : str
        Pixel stretch.
    percentile : float
        Percentile interval for each channel.
    figsize : tuple
        Figure size.
    output_path : str or Path, optional
        Save path.

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    plt = _ensure_matplotlib()

    channels = []
    for ch in [red, green, blue]:
        ch = np.asarray(ch, dtype=np.float64)
        vmin, vmax = _percentile_interval(ch, percentile)
        ch_clipped = np.clip(ch, vmin, vmax)
        ch_stretched = _apply_stretch(ch_clipped, stretch=stretch)
        channels.append(ch_stretched)

    rgb = np.stack(channels, axis=-1)
    rgb = np.clip(rgb, 0, 1)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(rgb, origin="lower", aspect="equal")
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold")
    fig.tight_layout()

    if output_path:
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        return None
    return fig


def display_spectrum(
    wavelength: np.ndarray,
    flux: np.ndarray,
    title: str = "Spectrum",
    xlabel: str = "Wavelength",
    ylabel: str = "Flux",
    line_markers: list[dict[str, Any]] | None = None,
    figsize: tuple[float, float] = (12, 4),
    output_path: str | Path | None = None,
) -> Any:
    """Display a 1-D spectrum.

    Parameters
    ----------
    wavelength : ndarray
        Wavelength array.
    flux : ndarray
        Flux array.
    title : str
        Figure title.
    xlabel, ylabel : str
        Axis labels.
    line_markers : list of dict, optional
        List of spectral line markers: ``{"wavelength": float, "name": str}``.
    figsize : tuple
        Figure size.
    output_path : str or Path, optional
        Save path.

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    plt = _ensure_matplotlib()

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(wavelength, flux, color="#2196F3", linewidth=0.8, alpha=0.9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold")

    if line_markers:
        for marker in line_markers:
            wl = marker["wavelength"]
            name = marker.get("name", "")
            ax.axvline(wl, color="#FF5722", linestyle="--", alpha=0.5)
            ax.text(wl, ax.get_ylim()[1] * 0.95, name,
                    rotation=90, va="top", ha="right",
                    fontsize=8, color="#FF5722")

    ax.grid(alpha=0.3)
    fig.tight_layout()

    if output_path:
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        return None
    return fig


def display_histogram(
    data: np.ndarray,
    bins: int = 256,
    title: str = "Pixel Value Distribution",
    log_scale: bool = True,
    stats_overlay: bool = True,
    figsize: tuple[float, float] = (10, 4),
    output_path: str | Path | None = None,
) -> Any:
    """Display pixel-value histogram with optional statistics.

    Parameters
    ----------
    data : ndarray
        Image data.
    bins : int
        Number of histogram bins.
    title : str
        Figure title.
    log_scale : bool
        Use logarithmic y-axis.
    stats_overlay : bool
        Show statistics text overlay.
    figsize : tuple
        Figure size.
    output_path : str or Path, optional
        Save path.

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    plt = _ensure_matplotlib()

    arr = np.asarray(data, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(arr, bins=bins, color="#2196F3", alpha=0.8, edgecolor="none")
    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel("Pixel Value")
    ax.set_ylabel("Count")
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold")

    if stats_overlay and arr.size > 0:
        stats_text = (
            f"Mean: {np.mean(arr):.2f}\n"
            f"Median: {np.median(arr):.2f}\n"
            f"Std: {np.std(arr):.2f}\n"
            f"Min: {np.min(arr):.2f}\n"
            f"Max: {np.max(arr):.2f}"
        )
        ax.text(0.98, 0.95, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax.grid(alpha=0.3)
    fig.tight_layout()

    if output_path:
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        return None
    return fig


def display_cutout(
    data: np.ndarray,
    center: tuple[int, int],
    size: int,
    title: str = "Cutout",
    cmap: str = "gray",
    stretch: Literal["linear", "log", "sqrt", "asinh"] = "linear",
    figsize: tuple[float, float] = (6, 6),
    output_path: str | Path | None = None,
) -> Any:
    """Display a square cutout from a larger image.

    Parameters
    ----------
    data : ndarray
        Full 2-D image.
    center : tuple of int
        (y, x) center of the cutout.
    size : int
        Half-size of the cutout box.
    title : str
        Figure title.
    cmap : str
        Colormap.
    stretch : str
        Pixel stretch.
    figsize : tuple
        Figure size.
    output_path : str or Path, optional
        Save path.

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    plt = _ensure_matplotlib()

    data = np.asarray(data, dtype=np.float64)
    cy, cx = center
    y0 = max(0, cy - size)
    y1 = min(data.shape[0], cy + size)
    x0 = max(0, cx - size)
    x1 = min(data.shape[1], cx + size)
    cutout = data[y0:y1, x0:x1]

    return display_image(
        cutout, title=f"{title} [{y0}:{y1}, {x0}:{x1}]",
        cmap=cmap, stretch=stretch, figsize=figsize,
        output_path=output_path,
    )


def display_comparison(
    image1: np.ndarray,
    image2: np.ndarray,
    title1: str = "Image 1",
    title2: str = "Image 2",
    cmap: str = "gray",
    stretch: Literal["linear", "log", "sqrt", "asinh"] = "linear",
    percentile: float = 99.5,
    show_difference: bool = True,
    figsize: tuple[float, float] = (16, 5),
    output_path: str | Path | None = None,
) -> Any:
    """Side-by-side comparison of two images.

    Parameters
    ----------
    image1, image2 : ndarray
        Two 2-D images to compare.
    title1, title2 : str
        Titles for each panel.
    cmap : str
        Colormap.
    stretch : str
        Pixel stretch.
    percentile : float
        Display percentile interval.
    show_difference : bool
        Show a third panel with the difference image.
    figsize : tuple
        Figure size.
    output_path : str or Path, optional
        Save path.

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    plt = _ensure_matplotlib()

    ncols = 3 if show_difference else 2
    fig, axes = plt.subplots(1, ncols, figsize=figsize)

    for ax, img, title in [(axes[0], image1, title1), (axes[1], image2, title2)]:
        img = np.asarray(img, dtype=np.float64)
        vmin, vmax = _percentile_interval(img, percentile)
        display_data = np.clip(img, vmin, vmax)
        display_data = _apply_stretch(display_data, stretch=stretch)
        ax.imshow(display_data, origin="lower", cmap=cmap, aspect="equal")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")

    if show_difference:
        diff = np.asarray(image1, dtype=np.float64) - np.asarray(image2, dtype=np.float64)
        vmax_diff = np.nanmax(np.abs(diff))
        if vmax_diff == 0:
            vmax_diff = 1.0
        im = axes[2].imshow(diff, origin="lower", cmap="RdBu_r",
                            vmin=-vmax_diff, vmax=vmax_diff, aspect="equal")
        axes[2].set_title("Difference", fontsize=12, fontweight="bold")
        axes[2].set_xlabel("X (pixels)")
        axes[2].set_ylabel("Y (pixels)")
        fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    fig.tight_layout()

    if output_path:
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        return None
    return fig


def display_mosaic(
    images: list[np.ndarray],
    titles: list[str] | None = None,
    ncols: int = 3,
    cmap: str = "gray",
    stretch: Literal["linear", "log", "sqrt", "asinh"] = "linear",
    figsize: tuple[float, float] | None = None,
    output_path: str | Path | None = None,
) -> Any:
    """Display a grid of images.

    Parameters
    ----------
    images : list of ndarray
        List of 2-D images.
    titles : list of str, optional
        Title for each image.
    ncols : int
        Number of columns.
    cmap : str
        Colormap.
    stretch : str
        Pixel stretch.
    figsize : tuple, optional
        Figure size (auto-calculated if not provided).
    output_path : str or Path, optional
        Save path.

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    plt = _ensure_matplotlib()

    n = len(images)
    nrows = (n + ncols - 1) // ncols

    if figsize is None:
        figsize = (4 * ncols, 4 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for idx in range(nrows * ncols):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        if idx < n:
            img = np.asarray(images[idx], dtype=np.float64)
            vmin, vmax = _percentile_interval(img)
            display_data = np.clip(img, vmin, vmax)
            display_data = _apply_stretch(display_data, stretch=stretch)
            ax.imshow(display_data, origin="lower", cmap=cmap, aspect="equal")
            if titles and idx < len(titles):
                ax.set_title(titles[idx], fontsize=10)
        else:
            ax.set_visible(False)

    fig.tight_layout()

    if output_path:
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        return None
    return fig


def display_provenance(
    provenance_dict: dict[str, Any],
    title: str = "Provenance Chain",
    figsize: tuple[float, float] = (14, 6),
    output_path: str | Path | None = None,
) -> Any:
    """Visualise a NOVA provenance bundle as a flow diagram.

    Parameters
    ----------
    provenance_dict : dict
        W3C PROV-DM bundle dictionary.
    title : str
        Figure title.
    figsize : tuple
        Figure size.
    output_path : str or Path, optional
        Save path.

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    plt = _ensure_matplotlib()
    from matplotlib.patches import FancyBboxPatch

    entities = provenance_dict.get("prov:entity", [])
    activities = provenance_dict.get("prov:activity", [])
    agents = provenance_dict.get("prov:agent", [])

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(-1, 10)
    ax.set_ylim(-1, max(3, len(entities) + 1))
    ax.set_aspect("equal")
    ax.axis("off")

    # Draw entities
    for i, entity in enumerate(entities):
        eid = entity.get("@id", f"entity_{i}")
        etype = entity.get("nova:entity_type", "Unknown")
        y = len(entities) - i - 0.5
        box = FancyBboxPatch((0.5, y - 0.3), 3, 0.6,
                             boxstyle="round,pad=0.1",
                             facecolor="#2196F3", edgecolor="white",
                             alpha=0.8)
        ax.add_patch(box)
        ax.text(2.0, y, f"{eid}\n({etype})", ha="center", va="center",
                fontsize=8, color="white", fontweight="bold")

    # Draw activities
    for i, activity in enumerate(activities):
        aid = activity.get("@id", f"activity_{i}")
        y = len(entities) / 2
        x = 5.5 + i * 2
        box = FancyBboxPatch((x - 1, y - 0.3), 2, 0.6,
                             boxstyle="round,pad=0.1",
                             facecolor="#FF9800", edgecolor="white",
                             alpha=0.8)
        ax.add_patch(box)
        ax.text(x, y, aid, ha="center", va="center",
                fontsize=8, color="white", fontweight="bold")

    # Draw agents
    for i, agent in enumerate(agents):
        agid = agent.get("@id", f"agent_{i}")
        aname = agent.get("nova:name", agid)
        x = 5.5 + i * 2
        y = -0.5
        box = FancyBboxPatch((x - 1, y - 0.2), 2, 0.4,
                             boxstyle="round,pad=0.1",
                             facecolor="#4CAF50", edgecolor="white",
                             alpha=0.8)
        ax.add_patch(box)
        ax.text(x, y, aname, ha="center", va="center",
                fontsize=7, color="white")

    if title:
        ax.set_title(title, fontsize=14, fontweight="bold")

    fig.tight_layout()

    if output_path:
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        return None
    return fig
