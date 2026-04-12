"""Advanced astronomical image-processing tools for NOVA.

Production-quality routines for PSF modelling, image registration and
alignment, difference imaging, CCD calibration, bad-pixel handling, and
fringe correction.  All functions operate on plain ``numpy.ndarray`` objects
and require only NumPy at import time; ``scipy.ndimage`` / ``scipy.optimize``
are imported lazily where needed.

Categories
----------
- **PSF modelling**: Moffat, Gaussian, and multi-Gaussian profiles + fitting.
- **Image registration**: WCS reprojection, feature-based alignment, sub-pixel
  shift via cross-correlation.
- **Image subtraction**: simple scaled subtraction, Alard-Lupton convolution.
- **Calibration helpers**: bias, dark, flat-field, overscan correction.
- **Bad-pixel handling**: interpolation & mask building from darks/flats.
- **Fringe correction**: fringe-map construction and removal.
"""

from __future__ import annotations

from typing import Literal, Sequence
import numpy as np

from nova.constants import MAD_TO_STD


# ---------------------------------------------------------------------------
#  Internal helpers
# ---------------------------------------------------------------------------

def _import_scipy_ndimage():  # noqa: D401
    """Lazily import *scipy.ndimage* with a clear error on failure."""
    try:
        import scipy.ndimage as ndi
        return ndi
    except ImportError:
        raise ImportError(
            "scipy is required for this function.  "
            "Install it with:  pip install scipy"
        ) from None


def _import_scipy_optimize():  # noqa: D401
    """Lazily import *scipy.optimize* with a clear error on failure."""
    try:
        import scipy.optimize as opt
        return opt
    except ImportError:
        raise ImportError(
            "scipy is required for this function.  "
            "Install it with:  pip install scipy"
        ) from None


def _cutout(image: np.ndarray, x0: float, y0: float, box_size: int) -> tuple[np.ndarray, int, int]:
    """Extract a square cutout centred near *(x0, y0)*.

    Returns the cutout array and the (row, col) origin offsets so the
    caller can convert cutout-local coordinates back to full-image coords.
    """
    half = box_size // 2
    ny, nx = image.shape[:2]
    row_c = int(round(y0))
    col_c = int(round(x0))
    r0 = max(row_c - half, 0)
    r1 = min(row_c + half + 1, ny)
    c0 = max(col_c - half, 0)
    c1 = min(col_c + half + 1, nx)
    return image[r0:r1, c0:c1].copy(), r0, c0


# ---------------------------------------------------------------------------
#  PSF Modelling
# ---------------------------------------------------------------------------

def moffat_2d(
    x: np.ndarray,
    y: np.ndarray,
    amplitude: float,
    x0: float,
    y0: float,
    alpha: float,
    beta: float,
) -> np.ndarray:
    """Evaluate a 2-D Moffat profile.

    .. math::

        I(x,y) = A \\left(1 + \\frac{(x-x_0)^2 + (y-y_0)^2}{\\alpha^2}\\right)^{-\\beta}

    Parameters
    ----------
    x, y : ndarray
        Coordinate grids (same shape).
    amplitude : float
        Peak amplitude *A*.
    x0, y0 : float
        Centre position.
    alpha : float
        Core width parameter (must be > 0).
    beta : float
        Power-law index (must be > 0; typical range 1.5 -- 5).

    Returns
    -------
    ndarray
        Moffat profile evaluated at each *(x, y)*.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    r2 = (x - x0) ** 2 + (y - y0) ** 2
    alpha2 = max(float(alpha) ** 2, 1e-30)
    return amplitude * (1.0 + r2 / alpha2) ** (-beta)


def gaussian_2d(
    x: np.ndarray,
    y: np.ndarray,
    amplitude: float,
    x0: float,
    y0: float,
    sigma_x: float,
    sigma_y: float,
    theta: float = 0.0,
) -> np.ndarray:
    """Evaluate a 2-D elliptical Gaussian with optional rotation.

    Parameters
    ----------
    x, y : ndarray
        Coordinate grids (same shape).
    amplitude : float
        Peak amplitude.
    x0, y0 : float
        Centre position.
    sigma_x, sigma_y : float
        Standard deviations along the (rotated) principal axes.
    theta : float
        Rotation angle in **radians** (counter-clockwise from +x axis).

    Returns
    -------
    ndarray
        Gaussian profile evaluated at each *(x, y)*.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    sx2 = max(float(sigma_x) ** 2, 1e-30)
    sy2 = max(float(sigma_y) ** 2, 1e-30)
    a = cos_t ** 2 / (2.0 * sx2) + sin_t ** 2 / (2.0 * sy2)
    b = np.sin(2.0 * theta) * (1.0 / (2.0 * sy2) - 1.0 / (2.0 * sx2)) / 2.0
    c = sin_t ** 2 / (2.0 * sx2) + cos_t ** 2 / (2.0 * sy2)
    dx = x - x0
    dy = y - y0
    return amplitude * np.exp(-(a * dx ** 2 + 2.0 * b * dx * dy + c * dy ** 2))


def multi_gaussian_psf(
    x: np.ndarray,
    y: np.ndarray,
    amplitudes: Sequence[float],
    x0: float,
    y0: float,
    sigmas: Sequence[float],
) -> np.ndarray:
    """Sum of concentric circular Gaussians (multi-Gaussian PSF model).

    Parameters
    ----------
    x, y : ndarray
        Coordinate grids (same shape).
    amplitudes : sequence of float
        Peak amplitude for each component.
    x0, y0 : float
        Common centre position for all components.
    sigmas : sequence of float
        Standard deviation of each circular Gaussian component.

    Returns
    -------
    ndarray
        Combined PSF evaluated at each *(x, y)*.

    Raises
    ------
    ValueError
        If *amplitudes* and *sigmas* have different lengths.
    """
    amplitudes = list(amplitudes)
    sigmas = list(sigmas)
    if len(amplitudes) != len(sigmas):
        raise ValueError(
            f"amplitudes ({len(amplitudes)}) and sigmas ({len(sigmas)}) "
            "must have the same length."
        )
    result = np.zeros_like(np.asarray(x, dtype=np.float64))
    for amp, sig in zip(amplitudes, sigmas):
        result = result + gaussian_2d(x, y, amp, x0, y0, sig, sig, theta=0.0)
    return result


def fit_moffat_psf(
    image: np.ndarray,
    x0: float,
    y0: float,
    box_size: int = 21,
) -> dict[str, float]:
    """Fit a 2-D Moffat profile to a star cutout.

    Parameters
    ----------
    image : ndarray
        2-D image containing the star.
    x0, y0 : float
        Approximate centre of the star (in full-image pixel coordinates).
    box_size : int
        Side length of the square cutout to extract.

    Returns
    -------
    dict
        Fitted parameters: ``amplitude``, ``x0``, ``y0``, ``alpha``,
        ``beta``, ``fwhm``.

    Raises
    ------
    RuntimeError
        If the fit does not converge.
    """
    opt = _import_scipy_optimize()
    stamp, r0, c0 = _cutout(image, x0, y0, box_size)
    stamp = np.asarray(stamp, dtype=np.float64)
    ny, nx = stamp.shape
    yy, xx = np.mgrid[0:ny, 0:nx]

    # Initial guesses relative to the cutout
    cx_local = x0 - c0
    cy_local = y0 - r0
    amp_guess = float(np.nanmax(stamp))
    bg_guess = float(np.nanmedian(stamp))

    def _model(coords, amp, cx, cy, alpha, beta, bg):
        xg, yg = coords
        return (moffat_2d(xg, yg, amp, cx, cy, alpha, beta) + bg).ravel()

    p0 = [amp_guess - bg_guess, cx_local, cy_local, 3.0, 2.5, bg_guess]
    bounds_lo = [0.0, -0.5, -0.5, 0.1, 1.0, -np.inf]
    bounds_hi = [np.inf, nx + 0.5, ny + 0.5, nx, 20.0, np.inf]

    valid = np.isfinite(stamp)
    xfit = xx[valid]
    yfit = yy[valid]
    zfit = stamp[valid]

    try:
        popt, _ = opt.curve_fit(
            _model, (xfit, yfit), zfit, p0=p0,
            bounds=(bounds_lo, bounds_hi), maxfev=10000,
        )
    except RuntimeError as exc:
        raise RuntimeError(f"Moffat PSF fit did not converge: {exc}") from exc

    amp, cx, cy, alpha, beta, _bg = popt
    fwhm = 2.0 * abs(alpha) * np.sqrt(2.0 ** (1.0 / max(beta, 1e-10)) - 1.0)
    return {
        "amplitude": float(amp),
        "x0": float(cx + c0),
        "y0": float(cy + r0),
        "alpha": float(alpha),
        "beta": float(beta),
        "fwhm": float(fwhm),
    }


def fit_gaussian_psf(
    image: np.ndarray,
    x0: float,
    y0: float,
    box_size: int = 21,
) -> dict[str, float]:
    """Fit a 2-D elliptical Gaussian to a star cutout.

    Parameters
    ----------
    image : ndarray
        2-D image containing the star.
    x0, y0 : float
        Approximate centre of the star (full-image pixel coordinates).
    box_size : int
        Side length of the square cutout.

    Returns
    -------
    dict
        Fitted parameters: ``amplitude``, ``x0``, ``y0``, ``sigma_x``,
        ``sigma_y``, ``theta``, ``fwhm_x``, ``fwhm_y``.

    Raises
    ------
    RuntimeError
        If the fit does not converge.
    """
    opt = _import_scipy_optimize()
    stamp, r0, c0 = _cutout(image, x0, y0, box_size)
    stamp = np.asarray(stamp, dtype=np.float64)
    ny, nx = stamp.shape
    yy, xx = np.mgrid[0:ny, 0:nx]

    cx_local = x0 - c0
    cy_local = y0 - r0
    amp_guess = float(np.nanmax(stamp))
    bg_guess = float(np.nanmedian(stamp))

    def _model(coords, amp, cx, cy, sx, sy, theta, bg):
        xg, yg = coords
        return (gaussian_2d(xg, yg, amp, cx, cy, sx, sy, theta) + bg).ravel()

    p0 = [amp_guess - bg_guess, cx_local, cy_local, 2.0, 2.0, 0.0, bg_guess]
    bounds_lo = [0.0, -0.5, -0.5, 0.1, 0.1, -np.pi, -np.inf]
    bounds_hi = [np.inf, nx + 0.5, ny + 0.5, nx, ny, np.pi, np.inf]

    valid = np.isfinite(stamp)
    xfit = xx[valid]
    yfit = yy[valid]
    zfit = stamp[valid]

    try:
        popt, _ = opt.curve_fit(
            _model, (xfit, yfit), zfit, p0=p0,
            bounds=(bounds_lo, bounds_hi), maxfev=10000,
        )
    except RuntimeError as exc:
        raise RuntimeError(f"Gaussian PSF fit did not converge: {exc}") from exc

    amp, cx, cy, sx, sy, theta, _bg = popt
    fwhm_factor = 2.0 * np.sqrt(2.0 * np.log(2.0))
    return {
        "amplitude": float(amp),
        "x0": float(cx + c0),
        "y0": float(cy + r0),
        "sigma_x": float(abs(sx)),
        "sigma_y": float(abs(sy)),
        "theta": float(theta),
        "fwhm_x": float(abs(sx) * fwhm_factor),
        "fwhm_y": float(abs(sy) * fwhm_factor),
    }


def generate_psf_image(
    shape: tuple[int, int],
    model_params: dict[str, float],
    model_type: Literal["moffat", "gaussian"] = "moffat",
) -> np.ndarray:
    """Generate a synthetic PSF image from fitted parameters.

    Parameters
    ----------
    shape : tuple of int
        Output image shape ``(ny, nx)``.
    model_params : dict
        Parameter dictionary as returned by :func:`fit_moffat_psf` or
        :func:`fit_gaussian_psf`.  The centre is used directly (in pixel
        coordinates of the output image).
    model_type : {"moffat", "gaussian"}
        Which PSF model to evaluate.

    Returns
    -------
    ndarray
        2-D image of shape *shape* containing the PSF model.

    Raises
    ------
    ValueError
        If *model_type* is not recognized.
    """
    ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    xx = xx.astype(np.float64)
    yy = yy.astype(np.float64)
    cx = model_params.get("x0", nx / 2.0)
    cy = model_params.get("y0", ny / 2.0)
    amp = model_params.get("amplitude", 1.0)

    if model_type == "moffat":
        alpha = model_params.get("alpha", 3.0)
        beta = model_params.get("beta", 2.5)
        return moffat_2d(xx, yy, amp, cx, cy, alpha, beta)

    if model_type == "gaussian":
        sx = model_params.get("sigma_x", 2.0)
        sy = model_params.get("sigma_y", 2.0)
        theta = model_params.get("theta", 0.0)
        return gaussian_2d(xx, yy, amp, cx, cy, sx, sy, theta)

    raise ValueError(
        f"Unknown model_type {model_type!r}; expected 'moffat' or 'gaussian'."
    )


# ---------------------------------------------------------------------------
#  Image Registration / Alignment
# ---------------------------------------------------------------------------

def wcs_align(
    data: np.ndarray,
    wcs_from: object,
    wcs_to: object,
    output_shape: tuple[int, int] | None = None,
    order: int = 1,
) -> np.ndarray:
    """Reproject *data* from one WCS frame to another.

    Uses an affine-approximation approach suitable for small-field TAN
    projections.  For each output pixel, the inverse WCS mapping is
    computed to find the corresponding input pixel, then interpolation
    of the requested *order* is applied via ``scipy.ndimage.map_coordinates``.

    Parameters
    ----------
    data : ndarray
        2-D input image.
    wcs_from, wcs_to : object
        WCS objects that expose ``pixel_to_world(x, y)`` and
        ``world_to_pixel(sky)`` methods (e.g. ``astropy.wcs.WCS`` or
        ``nova.wcs.NovaWCS``).
    output_shape : tuple of int or None
        Shape of the output image.  Defaults to the shape of *data*.
    order : int
        Spline interpolation order (0=nearest, 1=bilinear, 3=cubic).

    Returns
    -------
    ndarray
        Reprojected image.
    """
    ndi = _import_scipy_ndimage()
    data = np.asarray(data, dtype=np.float64)
    if output_shape is None:
        output_shape = data.shape[:2]
    ony, onx = output_shape

    # Build output pixel coordinate grids
    out_y, out_x = np.mgrid[0:ony, 0:onx]
    out_x_flat = out_x.ravel().astype(np.float64)
    out_y_flat = out_y.ravel().astype(np.float64)

    # Map output pixels -> world -> input pixels
    sky = wcs_to.pixel_to_world(out_x_flat, out_y_flat)
    in_x, in_y = wcs_from.world_to_pixel(sky)
    in_x = np.asarray(in_x, dtype=np.float64)
    in_y = np.asarray(in_y, dtype=np.float64)

    # scipy map_coordinates expects [row, col] = [y, x]
    coords = np.vstack([in_y, in_x])
    reprojected = ndi.map_coordinates(
        data, coords, order=order, mode="constant", cval=np.nan,
    )
    return reprojected.reshape(output_shape)


def _detect_peaks(image: np.ndarray, threshold: float, max_peaks: int) -> np.ndarray:
    """Detect bright peaks in *image* above *threshold*.

    Returns an (N, 2) array of ``(x, y)`` positions sorted by brightness
    (descending).
    """
    ndi = _import_scipy_ndimage()
    data = np.asarray(image, dtype=np.float64)
    mask = data > threshold
    labelled, n_labels = ndi.label(mask)
    if n_labels == 0:
        return np.empty((0, 2), dtype=np.float64)

    # Centroid of each labelled region
    indices = np.arange(1, n_labels + 1)
    centroids = ndi.center_of_mass(data, labelled, indices)
    peaks_yx = np.array(centroids, dtype=np.float64)  # (N, 2) as (row, col)
    brightness = np.array(
        [data[labelled == i].sum() for i in indices], dtype=np.float64,
    )
    order = np.argsort(-brightness)[:max_peaks]
    # Return as (x, y)
    return peaks_yx[order][:, ::-1]


def _triangle_hash(pts: np.ndarray) -> np.ndarray:
    """Compute a simple triangle invariant for a set of 2-D points.

    For each combination of 3 points, returns the sorted ratio of side
    lengths ``(s1/s3, s2/s3)`` where ``s1 <= s2 <= s3``.  This is
    scale- and rotation-invariant.

    Returns *(indices, descriptors)* arrays.
    """
    n = len(pts)
    if n < 3:
        return np.empty((0, 3), dtype=np.intp), np.empty((0, 2), dtype=np.float64)

    from itertools import combinations
    combos = list(combinations(range(n), 3))
    indices = np.array(combos, dtype=np.intp)  # (M, 3)
    descriptors = np.empty((len(combos), 2), dtype=np.float64)

    for k, (i, j, m) in enumerate(combos):
        d01 = np.linalg.norm(pts[i] - pts[j])
        d02 = np.linalg.norm(pts[i] - pts[m])
        d12 = np.linalg.norm(pts[j] - pts[m])
        sides = sorted([d01, d02, d12])
        longest = sides[2] if sides[2] > 0 else 1e-30
        descriptors[k, 0] = sides[0] / longest
        descriptors[k, 1] = sides[1] / longest

    return indices, descriptors


def feature_align(
    reference: np.ndarray,
    target: np.ndarray,
    max_features: int = 200,
    match_threshold: float = 0.7,
) -> np.ndarray:
    """Align *target* image to *reference* using source-matching triangles.

    Detects bright peaks in both images, matches constellations of three
    sources using scale-invariant triangle descriptors, then computes and
    applies an affine transformation.

    Parameters
    ----------
    reference : ndarray
        2-D reference image.
    target : ndarray
        2-D target image to align.
    max_features : int
        Maximum number of sources to detect in each image.
    match_threshold : float
        Maximum descriptor distance to accept a triangle match (0 -- 1).

    Returns
    -------
    ndarray
        Transformed *target* image aligned to *reference*.
    """
    ndi = _import_scipy_ndimage()
    ref = np.asarray(reference, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)

    # Detect peaks in both images
    ref_med = float(np.nanmedian(ref))
    ref_mad = float(np.nanmedian(np.abs(ref - ref_med))) * MAD_TO_STD
    tgt_med = float(np.nanmedian(tgt))
    tgt_mad = float(np.nanmedian(np.abs(tgt - tgt_med))) * MAD_TO_STD

    ref_thresh = ref_med + 5.0 * max(ref_mad, 1e-10)
    tgt_thresh = tgt_med + 5.0 * max(tgt_mad, 1e-10)

    ref_pts = _detect_peaks(ref, ref_thresh, max_features)
    tgt_pts = _detect_peaks(tgt, tgt_thresh, max_features)

    if len(ref_pts) < 3 or len(tgt_pts) < 3:
        # Not enough sources -- return target unchanged
        return tgt.copy()

    # Limit to brightest sources for triangle matching speed
    cap = min(30, len(ref_pts), len(tgt_pts))
    ref_pts = ref_pts[:cap]
    tgt_pts = tgt_pts[:cap]

    ref_idx, ref_desc = _triangle_hash(ref_pts)
    tgt_idx, tgt_desc = _triangle_hash(tgt_pts)

    if len(ref_desc) == 0 or len(tgt_desc) == 0:
        return tgt.copy()

    # Match triangle descriptors via nearest-neighbor in 2-D descriptor space
    matched_ref = []
    matched_tgt = []
    for ti in range(len(tgt_desc)):
        dists = np.sqrt(np.sum((ref_desc - tgt_desc[ti]) ** 2, axis=1))
        best = np.argmin(dists)
        if dists[best] < match_threshold:
            # Map point indices
            for k in range(3):
                ri = ref_idx[best, k]
                si = tgt_idx[ti, k]
                matched_ref.append(ref_pts[ri])
                matched_tgt.append(tgt_pts[si])

    if len(matched_ref) < 3:
        return tgt.copy()

    matched_ref = np.array(matched_ref, dtype=np.float64)
    matched_tgt = np.array(matched_tgt, dtype=np.float64)

    # Least-squares affine: ref = A @ tgt + t  =>  solve for A, t
    # Using homogeneous coordinates: [x, y, 1]
    n = len(matched_tgt)
    src_h = np.column_stack([matched_tgt, np.ones(n)])  # (N, 3)
    # Solve for X and Y independently
    Ax, _, _, _ = np.linalg.lstsq(src_h, matched_ref[:, 0], rcond=None)
    Ay, _, _, _ = np.linalg.lstsq(src_h, matched_ref[:, 1], rcond=None)

    # Build affine matrix for scipy (maps output -> input)
    # output (x,y) in reference frame, input (x,y) in target frame
    # We need the *inverse*: given reference pixel, find target pixel.
    # Forward: ref_x = Ax[0]*tgt_x + Ax[1]*tgt_y + Ax[2]
    #          ref_y = Ay[0]*tgt_x + Ay[1]*tgt_y + Ay[2]
    # Inverse: solve for tgt_x, tgt_y given ref_x, ref_y
    A_fwd = np.array([[Ax[0], Ax[1]], [Ay[0], Ay[1]]])
    t_fwd = np.array([Ax[2], Ay[2]])
    try:
        A_inv = np.linalg.inv(A_fwd)
    except np.linalg.LinAlgError:
        return tgt.copy()
    t_inv = -A_inv @ t_fwd

    # scipy affine_transform: output[o] = sum_j matrix[o,j]*input_index[j] + offset[o]
    # It maps output pixel -> input pixel (in row, col order).
    # Our A_inv maps (x, y) -> (x, y), but ndimage wants (row, col) = (y, x).
    matrix_rc = np.array([
        [A_inv[1, 1], A_inv[1, 0]],
        [A_inv[0, 1], A_inv[0, 0]],
    ])
    offset_rc = np.array([t_inv[1], t_inv[0]])

    aligned = ndi.affine_transform(
        tgt, matrix_rc, offset=offset_rc,
        output_shape=ref.shape, order=1, mode="constant", cval=np.nan,
    )
    return aligned


def compute_shift(
    reference: np.ndarray,
    target: np.ndarray,
    upsample_factor: int = 10,
) -> tuple[float, float]:
    """Compute sub-pixel shift between two images via phase correlation.

    Uses the Fourier-domain cross-correlation (phase correlation) method
    with upsampled matrix-multiply DFT for sub-pixel precision.

    Parameters
    ----------
    reference : ndarray
        2-D reference image.
    target : ndarray
        2-D target image (same shape as *reference*).
    upsample_factor : int
        Up-sampling factor for sub-pixel accuracy.  A factor of 10 gives
        0.1-pixel precision.

    Returns
    -------
    tuple of float
        ``(dy, dx)`` shift of *target* relative to *reference*.  Positive
        values mean *target* is shifted in the +y / +x direction.
    """
    ref = np.asarray(reference, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    if ref.shape != tgt.shape:
        raise ValueError(
            f"Image shapes must match: {ref.shape} vs {tgt.shape}"
        )

    # Replace NaNs with 0 for FFT
    ref = np.where(np.isfinite(ref), ref, 0.0)
    tgt = np.where(np.isfinite(tgt), tgt, 0.0)

    # Cross-power spectrum (tgt * conj(ref) so the IFFT peak gives the
    # shift of target relative to reference with the correct sign).
    f_ref = np.fft.fft2(ref)
    f_tgt = np.fft.fft2(tgt)
    cross_power = f_tgt * np.conj(f_ref)
    eps = np.finfo(np.float64).eps
    cross_power /= np.abs(cross_power) + eps

    # Pixel-level peak from inverse FFT
    cc = np.fft.ifft2(cross_power).real
    ny, nx = cc.shape
    peak_idx = np.unravel_index(np.argmax(cc), cc.shape)
    dy_pixel = float(peak_idx[0])
    dx_pixel = float(peak_idx[1])
    if dy_pixel > ny // 2:
        dy_pixel -= ny
    if dx_pixel > nx // 2:
        dx_pixel -= nx

    if upsample_factor <= 1:
        return (dy_pixel, dx_pixel)

    # Sub-pixel refinement via upsampled matrix-multiply DFT around the
    # integer peak (Guizar-Sicairos et al. 2008 approach).
    upsampled_size = int(np.ceil(upsample_factor * 1.5))
    dft_shift = int(np.fix(upsampled_size / 2.0))

    row_range = (np.arange(upsampled_size) - dft_shift) / upsample_factor + dy_pixel
    col_range = (np.arange(upsampled_size) - dft_shift) / upsample_factor + dx_pixel

    # Frequency indices: k = 0 .. N-1 (un-wrapped, matching FFT bin order).
    k_row = np.arange(ny, dtype=np.float64)
    k_col = np.arange(nx, dtype=np.float64)

    # kernel_row: (upsampled_size, ny) -- inverse-DFT basis
    kernel_row = np.exp(
        2j * np.pi * row_range[:, np.newaxis] * k_row[np.newaxis, :] / ny
    )
    # kernel_col: (nx, upsampled_size)
    kernel_col = np.exp(
        2j * np.pi * k_col[:, np.newaxis] * col_range[np.newaxis, :] / nx
    )

    upsampled_cc = (kernel_row @ cross_power @ kernel_col).real
    peak_up = np.unravel_index(np.argmax(upsampled_cc), upsampled_cc.shape)
    dy = (peak_up[0] - dft_shift) / upsample_factor + dy_pixel
    dx = (peak_up[1] - dft_shift) / upsample_factor + dx_pixel
    return (float(dy), float(dx))


# ---------------------------------------------------------------------------
#  Image Subtraction
# ---------------------------------------------------------------------------

def subtract_scaled(
    science: np.ndarray,
    reference: np.ndarray,
    scale: float | None = None,
) -> np.ndarray:
    """Simple scaled image subtraction.

    Computes ``science - scale * reference``.  If *scale* is ``None`` an
    optimal scale is determined by minimising the variance of the residual,
    equivalent to the ratio of the median fluxes.

    Parameters
    ----------
    science : ndarray
        2-D science image.
    reference : ndarray
        2-D reference image (same shape as *science*).
    scale : float or None
        Explicit scale factor.  If ``None`` it is computed automatically.

    Returns
    -------
    ndarray
        Difference image.
    """
    sci = np.asarray(science, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    if sci.shape != ref.shape:
        raise ValueError(
            f"Image shapes must match: {sci.shape} vs {ref.shape}"
        )
    if scale is None:
        valid = np.isfinite(sci) & np.isfinite(ref) & (ref != 0)
        if np.any(valid):
            ratio = sci[valid] / ref[valid]
            scale = float(np.median(ratio))
        else:
            scale = 1.0
    return sci - scale * ref


def _gaussian_basis_1d(size: int, sigma: float) -> np.ndarray:
    """1-D Gaussian kernel (normalised to unit sum)."""
    x = np.arange(size, dtype=np.float64) - size // 2
    g = np.exp(-0.5 * (x / max(sigma, 1e-10)) ** 2)
    g /= g.sum() + 1e-30
    return g


def alard_lupton_subtract(
    science: np.ndarray,
    reference: np.ndarray,
    kernel_size: int = 11,
    n_basis: int = 3,
    deg_spatial: int = 2,
) -> np.ndarray:
    """Alard-Lupton image subtraction.

    Convolves the *reference* image with a spatially-varying kernel so that
    it matches the *science* PSF, then subtracts.  The kernel is decomposed
    into *n_basis* basis functions (delta + Gaussian-modulated polynomials)
    with spatial-polynomial coefficients of degree *deg_spatial*.

    Parameters
    ----------
    science : ndarray
        2-D science image.
    reference : ndarray
        2-D reference image (same shape).
    kernel_size : int
        Side length of the convolution kernel (odd recommended).
    n_basis : int
        Number of basis functions for the kernel decomposition.
    deg_spatial : int
        Degree of the spatial polynomial multiplying each basis.

    Returns
    -------
    ndarray
        Difference image ``science - K (*) reference``.

    Notes
    -----
    This is a simplified implementation suitable for many practical cases.
    The basis set consists of a delta function plus ``n_basis - 1`` Gaussian
    components with increasing widths modulated by 2-D polynomial surfaces
    up to degree *deg_spatial*.
    """
    ndi = _import_scipy_ndimage()
    sci = np.asarray(science, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    if sci.shape != ref.shape:
        raise ValueError(
            f"Image shapes must match: {sci.shape} vs {ref.shape}"
        )
    ny, nx = sci.shape
    half = kernel_size // 2

    # Build basis kernels: a delta + Gaussians of increasing sigma
    basis_kernels: list[np.ndarray] = []
    # Delta basis
    delta = np.zeros((kernel_size, kernel_size), dtype=np.float64)
    delta[half, half] = 1.0
    basis_kernels.append(delta)
    for i in range(1, n_basis):
        sigma = 1.0 + (i - 1) * 1.5
        g1d = _gaussian_basis_1d(kernel_size, sigma)
        basis_kernels.append(np.outer(g1d, g1d))

    # Spatial polynomial terms: 1, x, y, x^2, xy, y^2, ... up to deg_spatial
    # Normalised coordinates in [-1, 1]
    ycoords = np.linspace(-1, 1, ny)
    xcoords = np.linspace(-1, 1, nx)
    xg, yg = np.meshgrid(xcoords, ycoords)

    spatial_terms: list[np.ndarray] = []
    for order in range(deg_spatial + 1):
        for py in range(order + 1):
            px = order - py
            spatial_terms.append(xg ** px * yg ** py)
    n_spatial = len(spatial_terms)

    # Total number of unknowns: n_basis * n_spatial + 1 (background)
    n_unknowns = n_basis * n_spatial + 1

    # Convolve reference with each basis kernel once
    ref_convolved: list[np.ndarray] = []
    for bk in basis_kernels:
        ref_convolved.append(ndi.convolve(ref, bk, mode="constant", cval=0.0))

    # Build the design matrix row-by-row (we work on the full image).
    # To keep memory manageable, we solve in a single block.
    valid = np.isfinite(sci) & np.isfinite(ref)
    for rc in ref_convolved:
        valid &= np.isfinite(rc)
    idx = np.where(valid)
    n_pix = len(idx[0])

    if n_pix < n_unknowns:
        # Not enough valid pixels -- fall back to simple subtraction
        return subtract_scaled(sci, ref)

    # Subsample if the image is very large to keep memory usage reasonable
    max_pix = 500_000
    if n_pix > max_pix:
        rng = np.random.RandomState(42)
        sel = rng.choice(n_pix, max_pix, replace=False)
    else:
        sel = np.arange(n_pix)

    rows_y = idx[0][sel]
    rows_x = idx[1][sel]
    n_rows = len(sel)

    A = np.empty((n_rows, n_unknowns), dtype=np.float64)
    col = 0
    for bi in range(n_basis):
        rc_vals = ref_convolved[bi][rows_y, rows_x]
        for si in range(n_spatial):
            sp_vals = spatial_terms[si][rows_y, rows_x]
            A[:, col] = rc_vals * sp_vals
            col += 1
    # Background column
    A[:, col] = 1.0

    b = sci[rows_y, rows_x]

    # Solve least-squares
    coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    # Reconstruct the model image
    model = np.full_like(sci, np.nan)
    model[valid] = 0.0
    col = 0
    for bi in range(n_basis):
        for si in range(n_spatial):
            model[valid] += coeffs[col] * ref_convolved[bi][valid] * spatial_terms[si][valid]
            col += 1
    model[valid] += coeffs[col]  # background

    return sci - model


# ---------------------------------------------------------------------------
#  Calibration Helpers
# ---------------------------------------------------------------------------

def subtract_bias(data: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Subtract a bias frame from *data*.

    Parameters
    ----------
    data : ndarray
        2-D science or calibration image.
    bias : ndarray
        Bias frame.  May be 2-D (full-frame master bias) or 1-D (row or
        column overscan vector that will be broadcast).

    Returns
    -------
    ndarray
        Bias-subtracted image (float64).
    """
    data = np.asarray(data, dtype=np.float64)
    bias = np.asarray(bias, dtype=np.float64)
    if bias.ndim == 1:
        # Determine broadcast axis: match the axis whose length equals
        # the bias vector length.
        if bias.shape[0] == data.shape[0]:
            bias = bias[:, np.newaxis]
        elif bias.shape[0] == data.shape[1]:
            bias = bias[np.newaxis, :]
        else:
            raise ValueError(
                f"1-D bias length {bias.shape[0]} does not match either "
                f"image dimension {data.shape}."
            )
    return data - bias


def apply_flat(
    data: np.ndarray,
    flat: np.ndarray,
    min_flat: float = 0.01,
) -> np.ndarray:
    """Divide by a normalised flat-field image.

    The flat is normalised by its median before division.  Values below
    *min_flat* (after normalisation) are clamped to *min_flat* to avoid
    division by near-zero values.

    Parameters
    ----------
    data : ndarray
        2-D image to correct.
    flat : ndarray
        Flat-field image (same shape as *data*).
    min_flat : float
        Minimum allowed value in the normalised flat.

    Returns
    -------
    ndarray
        Flat-corrected image (float64).
    """
    data = np.asarray(data, dtype=np.float64)
    flat = np.asarray(flat, dtype=np.float64)
    flat_median = np.nanmedian(flat)
    if flat_median == 0 or not np.isfinite(flat_median):
        flat_median = 1.0
    norm_flat = flat / flat_median
    norm_flat = np.where(norm_flat < min_flat, min_flat, norm_flat)
    return data / norm_flat


def subtract_dark(
    data: np.ndarray,
    dark: np.ndarray,
    exposure_data: float,
    exposure_dark: float,
) -> np.ndarray:
    """Subtract a scaled dark frame.

    The dark is multiplied by the ratio ``exposure_data / exposure_dark``
    before subtraction, so the dark current is correctly scaled to the
    science exposure time.

    Parameters
    ----------
    data : ndarray
        2-D science image.
    dark : ndarray
        Master dark frame (same shape as *data*).
    exposure_data : float
        Exposure time of the science frame (any unit, must match *exposure_dark*).
    exposure_dark : float
        Exposure time of the master dark frame.

    Returns
    -------
    ndarray
        Dark-subtracted image (float64).

    Raises
    ------
    ValueError
        If *exposure_dark* is zero.
    """
    if exposure_dark == 0:
        raise ValueError("exposure_dark must be non-zero.")
    data = np.asarray(data, dtype=np.float64)
    dark = np.asarray(dark, dtype=np.float64)
    scale = float(exposure_data) / float(exposure_dark)
    return data - scale * dark


def correct_overscan(
    data: np.ndarray,
    overscan_region: np.ndarray | slice | tuple[slice, slice],
    axis: int = 1,
    method: Literal["mean", "median", "polynomial"] = "median",
) -> np.ndarray:
    """Subtract the overscan level from *data*.

    The overscan strip is collapsed along *axis* and the resulting 1-D
    profile is subtracted from each row (or column) of *data*.

    Parameters
    ----------
    data : ndarray
        Raw 2-D CCD image **including** the overscan region.
    overscan_region : ndarray, slice, or tuple of slices
        The overscan pixels.  Can be a 2-D array already extracted, a
        single ``slice`` along *axis*, or a ``(row_slice, col_slice)``
        tuple applied to *data*.
    axis : int
        Axis along which the overscan runs (1 = columns, 0 = rows).
    method : {"mean", "median", "polynomial"}
        How to collapse the overscan strip.  ``"polynomial"`` fits a
        3rd-order polynomial along the perpendicular axis.

    Returns
    -------
    ndarray
        Overscan-corrected image (float64).
    """
    data = np.asarray(data, dtype=np.float64)

    # Extract overscan strip
    if isinstance(overscan_region, np.ndarray):
        oscan = np.asarray(overscan_region, dtype=np.float64)
    elif isinstance(overscan_region, slice):
        if axis == 1:
            oscan = data[:, overscan_region].astype(np.float64)
        else:
            oscan = data[overscan_region, :].astype(np.float64)
    elif isinstance(overscan_region, tuple):
        oscan = data[overscan_region].astype(np.float64)
    else:
        raise TypeError(
            "overscan_region must be ndarray, slice, or tuple of slices; "
            f"got {type(overscan_region).__name__}."
        )

    # Collapse along the overscan axis to get a 1-D profile
    if method == "mean":
        profile = np.nanmean(oscan, axis=axis)
    elif method == "median":
        profile = np.nanmedian(oscan, axis=axis)
    elif method == "polynomial":
        profile = np.nanmedian(oscan, axis=axis)
        idx = np.arange(len(profile), dtype=np.float64)
        valid = np.isfinite(profile)
        if np.sum(valid) > 4:
            coeffs = np.polyfit(idx[valid], profile[valid], deg=3)
            profile = np.polyval(coeffs, idx)
    else:
        raise ValueError(
            f"Unknown method {method!r}; expected 'mean', 'median', or "
            "'polynomial'."
        )

    # Subtract the 1-D profile from each row or column
    if axis == 1:
        # profile has shape (nrows,) -- subtract from each row
        return data - profile[:, np.newaxis]
    else:
        # profile has shape (ncols,) -- subtract from each column
        return data - profile[np.newaxis, :]


# ---------------------------------------------------------------------------
#  Bad Pixel Handling
# ---------------------------------------------------------------------------

def interpolate_bad_pixels(
    data: np.ndarray,
    mask: np.ndarray,
    method: Literal["linear", "nearest"] = "linear",
) -> np.ndarray:
    """Interpolate over bad pixels using their neighbours.

    Parameters
    ----------
    data : ndarray
        2-D image.
    mask : ndarray of bool
        Bad-pixel mask (``True`` = bad).
    method : {"linear", "nearest"}
        ``"linear"`` replaces each bad pixel with the median of valid
        neighbours in a 3x3 box.  ``"nearest"`` uses
        ``scipy.ndimage.distance_transform_edt`` to fill from the nearest
        valid pixel.

    Returns
    -------
    ndarray
        Image with bad pixels replaced (float64).
    """
    data = np.asarray(data, dtype=np.float64).copy()
    mask = np.asarray(mask, dtype=bool)
    if data.shape != mask.shape:
        raise ValueError(
            f"data and mask shapes must match: {data.shape} vs {mask.shape}"
        )
    bad = np.where(mask)
    if len(bad[0]) == 0:
        return data

    if method == "nearest":
        ndi = _import_scipy_ndimage()
        # Find the indices of the nearest valid pixel for each bad pixel
        _, nearest_idx = ndi.distance_transform_edt(
            mask, return_distances=True, return_indices=True,
        )
        data[mask] = data[nearest_idx[0][mask], nearest_idx[1][mask]]
        return data

    # Linear (median of valid 3x3 neighbours)
    ny, nx = data.shape
    padded = np.pad(data, 1, mode="constant", constant_values=np.nan)
    pad_mask = np.pad(mask, 1, mode="constant", constant_values=True)

    for by, bx in zip(bad[0], bad[1]):
        # Coordinates in padded array
        py, px = by + 1, bx + 1
        neighbours = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                nb_y, nb_x = py + dy, px + dx
                if not pad_mask[nb_y, nb_x] and np.isfinite(padded[nb_y, nb_x]):
                    neighbours.append(padded[nb_y, nb_x])
        if neighbours:
            data[by, bx] = float(np.median(neighbours))
        # else: leave as-is (NaN or original bad value)

    return data


def build_bad_pixel_mask(
    darks: np.ndarray,
    flats: np.ndarray,
    dark_thresh: float = 5.0,
    flat_low: float = 0.5,
    flat_high: float = 1.5,
) -> np.ndarray:
    """Build a bad-pixel mask from master dark and flat frames.

    A pixel is marked bad if:

    * its value in the master dark deviates from the median by more than
      *dark_thresh* times the robust standard deviation, **or**
    * its value in the normalised master flat is outside
      ``[flat_low, flat_high]``.

    Parameters
    ----------
    darks : ndarray
        Master dark frame (2-D) or stack of dark frames (3-D, first axis
        is the frame index -- in which case the median is taken).
    flats : ndarray
        Master flat frame (2-D) or stack of flat frames (3-D).
    dark_thresh : float
        Sigma-clipping threshold for the dark frame.
    flat_low, flat_high : float
        Acceptable range for the normalised flat.

    Returns
    -------
    ndarray of bool
        Bad-pixel mask (``True`` = bad).
    """
    darks = np.asarray(darks, dtype=np.float64)
    flats = np.asarray(flats, dtype=np.float64)

    # Combine stacks if 3-D
    if darks.ndim == 3:
        darks = np.nanmedian(darks, axis=0)
    if flats.ndim == 3:
        flats = np.nanmedian(flats, axis=0)

    # Dark mask: outlier pixels
    dark_med = np.nanmedian(darks)
    dark_mad = np.nanmedian(np.abs(darks - dark_med))
    dark_std = dark_mad * MAD_TO_STD if dark_mad > 0 else 1e-10
    dark_bad = np.abs(darks - dark_med) > dark_thresh * dark_std

    # Flat mask: outside acceptable range after normalisation
    flat_med = np.nanmedian(flats)
    if flat_med == 0 or not np.isfinite(flat_med):
        flat_med = 1.0
    norm_flat = flats / flat_med
    flat_bad = (norm_flat < flat_low) | (norm_flat > flat_high)

    # Also flag NaN / Inf
    nan_bad = ~np.isfinite(darks) | ~np.isfinite(flats)

    return dark_bad | flat_bad | nan_bad


# ---------------------------------------------------------------------------
#  Fringe Correction
# ---------------------------------------------------------------------------

def build_fringe_map(
    images: np.ndarray | Sequence[np.ndarray],
    masks: np.ndarray | Sequence[np.ndarray] | None = None,
) -> np.ndarray:
    """Build a fringe pattern from a set of dithered images.

    Each image is normalised to unit median before combining, so that
    the persistent fringe pattern is preserved while astronomical sources
    (shifted between dithers) are rejected by median combination.

    Parameters
    ----------
    images : ndarray or sequence of ndarray
        Stack of dithered images.  If a 3-D array, the first axis is the
        frame index.
    masks : ndarray, sequence of ndarray, or None
        Optional bad-pixel masks for each image (``True`` = bad).  If
        provided, masked pixels are set to NaN before combining.

    Returns
    -------
    ndarray
        2-D fringe map (normalised to zero median).
    """
    if isinstance(images, np.ndarray) and images.ndim == 3:
        stack = [images[i] for i in range(images.shape[0])]
    else:
        stack = list(images)

    if len(stack) == 0:
        raise ValueError("At least one image is required.")

    if masks is not None:
        if isinstance(masks, np.ndarray) and masks.ndim == 3:
            mask_list = [masks[i] for i in range(masks.shape[0])]
        else:
            mask_list = list(masks)
    else:
        mask_list = [None] * len(stack)

    normalised: list[np.ndarray] = []
    for img, msk in zip(stack, mask_list):
        frame = np.asarray(img, dtype=np.float64).copy()
        if msk is not None:
            frame[np.asarray(msk, dtype=bool)] = np.nan
        med = np.nanmedian(frame)
        if med == 0 or not np.isfinite(med):
            med = 1.0
        normalised.append(frame / med)

    cube = np.array(normalised)  # shape (N, ny, nx)
    fringe = np.nanmedian(cube, axis=0)
    # Normalise to zero median so the map represents only the fringe pattern
    fringe -= np.nanmedian(fringe)
    return fringe


def subtract_fringe(
    data: np.ndarray,
    fringe_map: np.ndarray,
    scale: float | None = None,
) -> np.ndarray:
    """Remove a fringe pattern from *data*.

    Computes ``data - scale * fringe_map``.  If *scale* is ``None`` it is
    determined automatically by minimising the variance of the residual
    via a simple linear regression.

    Parameters
    ----------
    data : ndarray
        2-D science image.
    fringe_map : ndarray
        Fringe pattern (same shape as *data*), e.g. from
        :func:`build_fringe_map`.
    scale : float or None
        Explicit scaling factor.  If ``None``, the optimal scale is
        computed as ``cov(data, fringe) / var(fringe)``.

    Returns
    -------
    ndarray
        Fringe-corrected image (float64).
    """
    data = np.asarray(data, dtype=np.float64)
    fringe_map = np.asarray(fringe_map, dtype=np.float64)
    if data.shape != fringe_map.shape:
        raise ValueError(
            f"data and fringe_map shapes must match: {data.shape} vs "
            f"{fringe_map.shape}"
        )

    if scale is None:
        valid = np.isfinite(data) & np.isfinite(fringe_map)
        d = data[valid]
        f = fringe_map[valid]
        f_var = np.var(f)
        if f_var > 0:
            scale = float(np.cov(d, f, ddof=0)[0, 1] / f_var)
        else:
            scale = 0.0

    return data - scale * fringe_map
