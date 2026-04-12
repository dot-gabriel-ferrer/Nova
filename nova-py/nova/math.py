"""NOVA integrated mathematical tools for astronomical data.

High-performance, NumPy-optimized mathematical operations commonly needed
in astronomical data processing pipelines.  These are designed to work
directly on NOVA arrays (or plain ``numpy.ndarray``), eliminating the need
for external dependencies for the most common operations.

Categories
----------
- **Statistics**: sigma-clipped stats, robust estimators, histograms.
- **Convolution**: 2-D Gaussian & custom-kernel convolution (FFT-based).
- **Resampling**: rebinning, up/downscaling, drizzle-style combine.
- **Image stacking**: mean, median, sigma-clipped stacking with rejection.
- **Background estimation**: mesh-based background & RMS maps.
- **Source detection helpers**: simple threshold detection, aperture photometry.
- **Spectral**: continuum normalisation, line fitting helpers.
"""

from __future__ import annotations

from typing import Literal
import numpy as np

from nova.constants import MAD_TO_STD


# --------------------------------------------------------------------------
#  Statistics
# --------------------------------------------------------------------------

def sigma_clip(
    data: np.ndarray,
    sigma: float = 3.0,
    max_iters: int = 10,
    center_func: Literal["mean", "median"] = "median",
) -> np.ma.MaskedArray:
    """Iterative sigma-clipping of data.

    Parameters
    ----------
    data : ndarray
        Input data (any shape).
    sigma : float
        Number of standard deviations for clipping.
    max_iters : int
        Maximum number of iterations.
    center_func : {"mean", "median"}
        Function to compute the center value.

    Returns
    -------
    numpy.ma.MaskedArray
        Data with outliers masked.
    """
    arr = np.asarray(data, dtype=np.float64)
    mask = np.isnan(arr)
    for _ in range(max_iters):
        valid = arr[~mask]
        if valid.size == 0:
            break
        center = np.median(valid) if center_func == "median" else np.mean(valid)
        std = np.std(valid, ddof=1) if valid.size > 1 else 0.0
        if std == 0.0:
            break
        new_mask = mask | (np.abs(arr - center) > sigma * std)
        if np.array_equal(new_mask, mask):
            break
        mask = new_mask
    return np.ma.MaskedArray(arr, mask=mask)


def sigma_clipped_stats(
    data: np.ndarray,
    sigma: float = 3.0,
    max_iters: int = 10,
) -> dict[str, float]:
    """Compute sigma-clipped mean, median, and standard deviation.

    Parameters
    ----------
    data : ndarray
        Input data.
    sigma : float
        Clipping threshold in standard deviations.
    max_iters : int
        Maximum clipping iterations.

    Returns
    -------
    dict
        Keys: ``mean``, ``median``, ``std``, ``min``, ``max``, ``count``.
    """
    clipped = sigma_clip(data, sigma=sigma, max_iters=max_iters)
    valid = clipped.compressed()
    if valid.size == 0:
        return {"mean": np.nan, "median": np.nan, "std": np.nan,
                "min": np.nan, "max": np.nan, "count": 0}
    return {
        "mean": float(np.mean(valid)),
        "median": float(np.median(valid)),
        "std": float(np.std(valid, ddof=1)) if valid.size > 1 else 0.0,
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
        "count": int(valid.size),
    }


def robust_statistics(data: np.ndarray) -> dict[str, float]:
    """Compute robust statistics (median, MAD, biweight midvariance).

    Parameters
    ----------
    data : ndarray
        Input data.

    Returns
    -------
    dict
        Keys: ``median``, ``mad``, ``biweight_location``, ``biweight_scale``.
    """
    arr = np.asarray(data, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"median": np.nan, "mad": np.nan,
                "biweight_location": np.nan, "biweight_scale": np.nan}

    med = float(np.median(arr))
    deviations = np.abs(arr - med)
    mad = float(np.median(deviations))

    # Biweight estimators (Tukey, c=9 for location, c=9 for scale)
    c_loc = 9.0
    mad_scale = mad * MAD_TO_STD  # scale MAD to approximate std
    if mad_scale > 0:
        u = (arr - med) / (c_loc * mad_scale)
        mask = np.abs(u) < 1.0
        if np.any(mask):
            w = (1.0 - u**2) ** 2
            w[~mask] = 0.0
            bi_loc = float(med + np.sum((arr - med) * w) / np.sum(w))
        else:
            bi_loc = med

        # Biweight midvariance
        n = arr.size
        u_scale = (arr - med) / (c_loc * mad_scale)
        mask_s = np.abs(u_scale) < 1.0
        if np.sum(mask_s) > 1:
            numerator = np.sum(((arr[mask_s] - med) ** 2)
                               * (1 - u_scale[mask_s] ** 2) ** 4)
            d1 = np.sum((1 - u_scale[mask_s] ** 2)
                        * (1 - 5 * u_scale[mask_s] ** 2))
            bi_scale = float(np.sqrt(n * numerator / (d1 * (d1 - 1))))
        else:
            bi_scale = 0.0
    else:
        bi_loc = med
        bi_scale = 0.0

    return {
        "median": med,
        "mad": mad,
        "biweight_location": bi_loc,
        "biweight_scale": bi_scale,
    }


def histogram(
    data: np.ndarray,
    bins: int | str = "auto",
    range_: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute histogram, ignoring NaN values.

    Parameters
    ----------
    data : ndarray
        Input data.
    bins : int or str
        Number of bins or binning strategy (passed to ``numpy.histogram``).
    range_ : tuple of float, optional
        Range of values to include.

    Returns
    -------
    counts : ndarray
        Histogram bin counts.
    bin_edges : ndarray
        Bin edge values.
    """
    arr = np.asarray(data, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    return np.histogram(arr, bins=bins, range=range_)


# --------------------------------------------------------------------------
#  Convolution
# --------------------------------------------------------------------------

def gaussian_kernel_2d(sigma: float, size: int | None = None) -> np.ndarray:
    """Create a normalised 2-D Gaussian kernel.

    Parameters
    ----------
    sigma : float
        Standard deviation of the Gaussian in pixels.
    size : int, optional
        Kernel side length (must be odd). Defaults to ``int(6*sigma) | 1``.

    Returns
    -------
    ndarray
        2-D kernel array.
    """
    if size is None:
        size = int(np.ceil(sigma * 6)) | 1
    if size % 2 == 0:
        size += 1
    x = np.arange(size) - size // 2
    g1d = np.exp(-0.5 * (x / sigma) ** 2)
    kernel = np.outer(g1d, g1d)
    return kernel / kernel.sum()


def convolve_fft(
    data: np.ndarray,
    kernel: np.ndarray,
    nan_treatment: Literal["interpolate", "fill"] = "interpolate",
    fill_value: float = 0.0,
) -> np.ndarray:
    """FFT-based 2-D convolution (faster than direct for large kernels).

    Parameters
    ----------
    data : ndarray
        2-D input image.
    kernel : ndarray
        2-D convolution kernel (should be normalised).
    nan_treatment : {"interpolate", "fill"}
        How to handle NaN pixels.  ``"interpolate"`` replaces them with the
        local convolved value.  ``"fill"`` replaces them with *fill_value*.
    fill_value : float
        Value used when ``nan_treatment="fill"``.

    Returns
    -------
    ndarray
        Convolved image, same shape as *data*.
    """
    data = np.asarray(data, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)

    if data.ndim != 2 or kernel.ndim != 2:
        raise ValueError("convolve_fft requires 2-D data and kernel")

    # Handle NaN: replace with 0, track mask for weight normalisation
    nan_mask = np.isnan(data)
    data_filled = np.where(nan_mask, 0.0, data)
    weight = np.where(nan_mask, 0.0, 1.0)

    # Pad to avoid wrap-around
    pad_y = kernel.shape[0] // 2
    pad_x = kernel.shape[1] // 2
    fshape = (data.shape[0] + 2 * pad_y, data.shape[1] + 2 * pad_x)

    # FFT-based convolution
    data_fft = np.fft.rfft2(data_filled, s=fshape)
    kernel_fft = np.fft.rfft2(kernel, s=fshape)
    conv = np.fft.irfft2(data_fft * kernel_fft, s=fshape)

    # Weight convolution for NaN-aware normalisation
    weight_fft = np.fft.rfft2(weight, s=fshape)
    weight_conv = np.fft.irfft2(weight_fft * kernel_fft, s=fshape)
    weight_conv = np.clip(weight_conv, 1e-10, None)

    # Crop to original size
    result = conv[pad_y:pad_y + data.shape[0], pad_x:pad_x + data.shape[1]]
    weight_result = weight_conv[pad_y:pad_y + data.shape[0],
                                pad_x:pad_x + data.shape[1]]
    result /= weight_result

    if nan_treatment == "fill":
        result[nan_mask] = fill_value

    return result


def smooth_gaussian(
    data: np.ndarray,
    sigma: float,
    nan_treatment: Literal["interpolate", "fill"] = "interpolate",
) -> np.ndarray:
    """Smooth a 2-D image with a Gaussian kernel.

    Parameters
    ----------
    data : ndarray
        2-D image.
    sigma : float
        Gaussian standard deviation in pixels.
    nan_treatment : {"interpolate", "fill"}
        NaN handling strategy.

    Returns
    -------
    ndarray
        Smoothed image.
    """
    kernel = gaussian_kernel_2d(sigma)
    return convolve_fft(data, kernel, nan_treatment=nan_treatment)


# --------------------------------------------------------------------------
#  Resampling / Rebinning
# --------------------------------------------------------------------------

def rebin(
    data: np.ndarray,
    new_shape: tuple[int, int],
    method: Literal["sum", "mean"] = "sum",
) -> np.ndarray:
    """Rebin a 2-D array to a new shape.

    The new shape dimensions must be integer factors of the original shape.

    Parameters
    ----------
    data : ndarray
        2-D input array.
    new_shape : tuple of int
        Target ``(rows, cols)`` shape.
    method : {"sum", "mean"}
        Whether to sum or average pixels in each bin.

    Returns
    -------
    ndarray
        Rebinned array.
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("rebin requires 2-D data")
    if data.shape[0] % new_shape[0] != 0 or data.shape[1] % new_shape[1] != 0:
        raise ValueError(
            f"Cannot rebin {data.shape} -> {new_shape}: "
            f"dimensions must be integer factors"
        )
    bin_y = data.shape[0] // new_shape[0]
    bin_x = data.shape[1] // new_shape[1]
    reshaped = data.reshape(new_shape[0], bin_y, new_shape[1], bin_x)
    if method == "sum":
        return reshaped.sum(axis=(1, 3))
    return reshaped.mean(axis=(1, 3))


def resize_image(
    data: np.ndarray,
    new_shape: tuple[int, int],
    order: int = 1,
) -> np.ndarray:
    """Resize a 2-D image using interpolation (NumPy-only).

    Uses bilinear (order=1) or nearest-neighbour (order=0) interpolation
    without requiring scipy.

    Parameters
    ----------
    data : ndarray
        2-D input image.
    new_shape : tuple of int
        Target ``(rows, cols)`` shape.
    order : int
        Interpolation order: 0 = nearest, 1 = bilinear.

    Returns
    -------
    ndarray
        Resized image.
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("resize_image requires 2-D data")

    old_h, old_w = data.shape
    new_h, new_w = new_shape

    row_coords = np.linspace(0, old_h - 1, new_h)
    col_coords = np.linspace(0, old_w - 1, new_w)
    col_grid, row_grid = np.meshgrid(col_coords, row_coords)

    if order == 0:
        # Nearest-neighbour
        ri = np.round(row_grid).astype(int)
        ci = np.round(col_grid).astype(int)
        ri = np.clip(ri, 0, old_h - 1)
        ci = np.clip(ci, 0, old_w - 1)
        return data[ri, ci]

    # Bilinear interpolation
    r0 = np.floor(row_grid).astype(int)
    c0 = np.floor(col_grid).astype(int)
    r1 = np.clip(r0 + 1, 0, old_h - 1)
    c1 = np.clip(c0 + 1, 0, old_w - 1)
    r0 = np.clip(r0, 0, old_h - 1)
    c0 = np.clip(c0, 0, old_w - 1)

    dr = row_grid - r0
    dc = col_grid - c0

    return (
        data[r0, c0] * (1 - dr) * (1 - dc)
        + data[r0, c1] * (1 - dr) * dc
        + data[r1, c0] * dr * (1 - dc)
        + data[r1, c1] * dr * dc
    )


# --------------------------------------------------------------------------
#  Image Stacking
# --------------------------------------------------------------------------

def stack_images(
    images: list[np.ndarray],
    method: Literal["mean", "median", "sigma_clip"] = "median",
    sigma: float = 3.0,
    max_iters: int = 5,
) -> np.ndarray:
    """Stack multiple images into a single combined image.

    Parameters
    ----------
    images : list of ndarray
        List of 2-D images with the same shape.
    method : {"mean", "median", "sigma_clip"}
        Stacking method.
    sigma : float
        Sigma threshold for sigma-clipped stacking.
    max_iters : int
        Maximum iterations for sigma-clipped stacking.

    Returns
    -------
    ndarray
        Combined image.
    """
    if not images:
        raise ValueError("At least one image is required for stacking")

    shape0 = images[0].shape
    for i, img in enumerate(images):
        if img.shape != shape0:
            raise ValueError(
                f"Image {i} has shape {img.shape}, expected {shape0}"
            )

    cube = np.stack(images, axis=0).astype(np.float64)

    if method == "mean":
        return np.nanmean(cube, axis=0)
    elif method == "median":
        return np.nanmedian(cube, axis=0)
    elif method == "sigma_clip":
        # Pixel-wise sigma-clipped mean
        result = np.empty(shape0, dtype=np.float64)
        for _ in range(max_iters):
            with np.errstate(all="ignore"):
                med = np.nanmedian(cube, axis=0)
                std = np.nanstd(cube, axis=0, ddof=1)
            std = np.where(std > 0, std, 1.0)
            deviation = np.abs(cube - med[np.newaxis, ...])
            cube = np.where(deviation > sigma * std[np.newaxis, ...],
                            np.nan, cube)
        result = np.nanmean(cube, axis=0)
        return result
    else:
        raise ValueError(f"Unknown stacking method: {method!r}")


# --------------------------------------------------------------------------
#  Background Estimation
# --------------------------------------------------------------------------

def estimate_background(
    data: np.ndarray,
    box_size: int = 64,
    sigma_clip_threshold: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate spatially varying background and RMS maps.

    Divides the image into a grid of boxes, computes sigma-clipped statistics
    in each box, then interpolates back to the original image size.

    Parameters
    ----------
    data : ndarray
        2-D input image.
    box_size : int
        Size of each background estimation box.
    sigma_clip_threshold : float
        Sigma threshold for clipping.

    Returns
    -------
    background : ndarray
        Estimated background map (same shape as *data*).
    rms : ndarray
        Estimated RMS noise map (same shape as *data*).
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("estimate_background requires 2-D data")

    ny, nx = data.shape
    grid_ny = max(1, ny // box_size)
    grid_nx = max(1, nx // box_size)

    bg_grid = np.zeros((grid_ny, grid_nx), dtype=np.float64)
    rms_grid = np.zeros((grid_ny, grid_nx), dtype=np.float64)

    for iy in range(grid_ny):
        y0 = iy * box_size
        y1 = min(y0 + box_size, ny)
        for ix in range(grid_nx):
            x0 = ix * box_size
            x1 = min(x0 + box_size, nx)
            box = data[y0:y1, x0:x1]
            stats = sigma_clipped_stats(box, sigma=sigma_clip_threshold)
            bg_grid[iy, ix] = stats["median"]
            rms_grid[iy, ix] = stats["std"]

    # Resize grids back to original image size
    background = resize_image(bg_grid, (ny, nx))
    rms = resize_image(rms_grid, (ny, nx))

    return background, rms


# --------------------------------------------------------------------------
#  Source Detection Helpers
# --------------------------------------------------------------------------

def detect_sources(
    data: np.ndarray,
    threshold: float | None = None,
    nsigma: float = 5.0,
    min_area: int = 5,
) -> list[dict[str, float]]:
    """Simple threshold-based source detection.

    This is a basic connected-component detector suitable for quick-look
    analysis.  For production use, consider ``photutils`` or similar.

    Parameters
    ----------
    data : ndarray
        2-D background-subtracted image.
    threshold : float, optional
        Absolute detection threshold.  If ``None``, uses *nsigma* x RMS.
    nsigma : float
        Detection threshold in units of the global RMS (used if *threshold*
        is ``None``).
    min_area : int
        Minimum number of connected pixels to count as a source.

    Returns
    -------
    list of dict
        Each source has keys: ``x``, ``y`` (centroid), ``flux`` (total),
        ``peak``, ``area`` (pixels).
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("detect_sources requires 2-D data")

    if threshold is None:
        stats = sigma_clipped_stats(data)
        threshold = stats["median"] + nsigma * stats["std"]

    # Binary detection mask
    detect_mask = data > threshold

    # Simple flood-fill labelling (NumPy-only)
    labels = np.zeros_like(data, dtype=np.int32)
    current_label = 0
    ny, nx = data.shape

    for y in range(ny):
        for x in range(nx):
            if detect_mask[y, x] and labels[y, x] == 0:
                current_label += 1
                # BFS flood fill
                queue = [(y, x)]
                labels[y, x] = current_label
                while queue:
                    cy, cx = queue.pop(0)
                    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ny2, nx2 = cy + dy, cx + dx
                        if 0 <= ny2 < ny and 0 <= nx2 < nx:
                            if detect_mask[ny2, nx2] and labels[ny2, nx2] == 0:
                                labels[ny2, nx2] = current_label
                                queue.append((ny2, nx2))

    # Extract source properties
    sources: list[dict[str, float]] = []
    for label_id in range(1, current_label + 1):
        mask = labels == label_id
        area = int(np.sum(mask))
        if area < min_area:
            continue
        ys, xs = np.where(mask)
        flux_values = data[mask]
        total_flux = float(np.sum(flux_values))
        # Flux-weighted centroid
        cx = float(np.sum(xs * flux_values) / total_flux) if total_flux > 0 else float(np.mean(xs))
        cy = float(np.sum(ys * flux_values) / total_flux) if total_flux > 0 else float(np.mean(ys))
        sources.append({
            "x": cx,
            "y": cy,
            "flux": total_flux,
            "peak": float(np.max(flux_values)),
            "area": float(area),
        })

    return sources


def aperture_photometry(
    data: np.ndarray,
    x: float,
    y: float,
    radius: float,
    annulus_inner: float | None = None,
    annulus_outer: float | None = None,
) -> dict[str, float]:
    """Circular aperture photometry at a given position.

    Parameters
    ----------
    data : ndarray
        2-D image.
    x, y : float
        Source centroid position (pixel coordinates).
    radius : float
        Aperture radius in pixels.
    annulus_inner, annulus_outer : float, optional
        Inner and outer radii for background annulus.

    Returns
    -------
    dict
        Keys: ``flux``, ``area``, ``background``, ``flux_corrected``.
    """
    data = np.asarray(data, dtype=np.float64)
    ny, nx = data.shape

    yy, xx = np.ogrid[0:ny, 0:nx]
    dist = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)

    # Aperture
    aper_mask = dist <= radius
    aper_flux = float(np.nansum(data[aper_mask]))
    aper_area = float(np.sum(aper_mask))

    # Background from annulus
    bg_per_pixel = 0.0
    if annulus_inner is not None and annulus_outer is not None:
        ann_mask = (dist >= annulus_inner) & (dist <= annulus_outer)
        if np.any(ann_mask):
            ann_values = data[ann_mask]
            # Sigma-clipped median background
            clipped = sigma_clip(ann_values, sigma=3.0)
            bg_per_pixel = float(np.ma.median(clipped))

    bg_total = bg_per_pixel * aper_area
    return {
        "flux": aper_flux,
        "area": aper_area,
        "background": bg_total,
        "flux_corrected": aper_flux - bg_total,
    }


# --------------------------------------------------------------------------
#  Spectral Helpers
# --------------------------------------------------------------------------

def continuum_normalize(
    wavelength: np.ndarray,
    flux: np.ndarray,
    order: int = 5,
    sigma_clip_threshold: float = 3.0,
    max_iters: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalise a spectrum by fitting and dividing by the continuum.

    Uses iterative sigma-clipped polynomial fitting.

    Parameters
    ----------
    wavelength : ndarray
        Wavelength array.
    flux : ndarray
        Flux array.
    order : int
        Polynomial order for continuum fit.
    sigma_clip_threshold : float
        Sigma threshold for rejecting absorption/emission lines.
    max_iters : int
        Maximum fitting iterations.

    Returns
    -------
    normalized_flux : ndarray
        Continuum-normalised flux.
    continuum : ndarray
        Fitted continuum.
    """
    wavelength = np.asarray(wavelength, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    mask = np.isfinite(flux)

    for _ in range(max_iters):
        w = wavelength[mask]
        f = flux[mask]
        if len(w) < order + 1:
            break
        coeffs = np.polyfit(w, f, order)
        continuum_fit = np.polyval(coeffs, wavelength)
        residuals = flux - continuum_fit
        std = np.std(residuals[mask])
        if std == 0:
            break
        new_mask = mask & (np.abs(residuals) < sigma_clip_threshold * std)
        if np.array_equal(new_mask, mask):
            break
        mask = new_mask

    # Final fit
    coeffs = np.polyfit(wavelength[mask], flux[mask], order)
    continuum = np.polyval(coeffs, wavelength)
    normalized = np.where(continuum > 0, flux / continuum, flux)

    return normalized, continuum


def equivalent_width(
    wavelength: np.ndarray,
    normalized_flux: np.ndarray,
    line_center: float,
    width: float,
) -> float:
    """Compute the equivalent width of a spectral line.

    Parameters
    ----------
    wavelength : ndarray
        Wavelength array.
    normalized_flux : ndarray
        Continuum-normalised flux.
    line_center : float
        Central wavelength of the line.
    width : float
        Half-width of the integration window.

    Returns
    -------
    float
        Equivalent width (positive for absorption lines).
    """
    wavelength = np.asarray(wavelength, dtype=np.float64)
    normalized_flux = np.asarray(normalized_flux, dtype=np.float64)
    mask = (wavelength >= line_center - width) & (wavelength <= line_center + width)
    if np.sum(mask) < 2:
        return 0.0
    w = wavelength[mask]
    f = normalized_flux[mask]
    integrand = 1.0 - f
    return float(np.sum(0.5 * (integrand[:-1] + integrand[1:]) * np.diff(w)))


# --------------------------------------------------------------------------
#  Pixel-level Operations
# --------------------------------------------------------------------------

def cosmic_ray_clean(
    data: np.ndarray,
    sigma: float = 5.0,
    neighbor_size: int = 3,
) -> np.ndarray:
    """Simple cosmic ray cleaning using sigma-based outlier detection.

    Replaces pixels that deviate significantly from their local neighbourhood
    median with the median value.

    Parameters
    ----------
    data : ndarray
        2-D image.
    sigma : float
        Detection threshold in standard deviations.
    neighbor_size : int
        Size of the neighbourhood window.

    Returns
    -------
    ndarray
        Cleaned image.
    """
    data = np.asarray(data, dtype=np.float64).copy()
    pad = neighbor_size // 2
    padded = np.pad(data, pad, mode="reflect")

    local_median = np.zeros_like(data)
    local_std = np.zeros_like(data)

    # Compute local statistics using sliding window
    for dy in range(-pad, pad + 1):
        for dx in range(-pad, pad + 1):
            if dy == 0 and dx == 0:
                continue
            shifted = padded[pad + dy:pad + dy + data.shape[0],
                             pad + dx:pad + dx + data.shape[1]]
            local_median += shifted
    n_neighbors = neighbor_size**2 - 1
    local_median /= n_neighbors

    # Compute std from local mean
    for dy in range(-pad, pad + 1):
        for dx in range(-pad, pad + 1):
            if dy == 0 and dx == 0:
                continue
            shifted = padded[pad + dy:pad + dy + data.shape[0],
                             pad + dx:pad + dx + data.shape[1]]
            local_std += (shifted - local_median) ** 2
    local_std = np.sqrt(local_std / n_neighbors)
    local_std = np.where(local_std > 0, local_std, 1.0)

    # Detect and replace cosmic rays
    cr_mask = np.abs(data - local_median) > sigma * local_std
    data[cr_mask] = local_median[cr_mask]
    return data
