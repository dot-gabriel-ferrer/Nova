"""Advanced spectral analysis tools for NOVA.

Production-quality routines for wavelength calibration, sky subtraction,
radial-velocity measurement, spectral classification, emission-line fitting,
and multi-order echelle extraction.  All functions operate on plain
``numpy.ndarray`` objects and require only NumPy at import time;
``scipy.optimize``, ``scipy.signal``, and ``scipy.interpolate`` are imported
lazily where needed.

Categories
----------
- **Wavelength calibration**: line identification, polynomial wavelength
  solutions, and one-shot calibration pipelines.
- **Sky subtraction**: 1-D and 2-D sky-background removal for long-slit and
  multi-object spectroscopy.
- **Radial velocity**: cross-correlation RV measurement, Doppler shifting,
  and barycentric correction.
- **Spectral classification**: spectral-index measurement, equivalent-width
  calculation, and template matching.
- **Emission-line fitting**: Gaussian, Voigt (pseudo-Voigt), single-line and
  multi-line simultaneous fitting.
- **Multi-order echelle**: order tracing, extraction, and overlap merging.
"""

from __future__ import annotations

from typing import Any, Literal, Sequence

import numpy as np

from nova.constants import MAD_TO_STD


# ---------------------------------------------------------------------------
#  Module-level constants
# ---------------------------------------------------------------------------

SPEED_OF_LIGHT_KMS: float = 299792.458
"""Speed of light in km/s, used for radial-velocity calculations."""

DEFAULT_SPECTRAL_INDICES: dict[str, dict[str, tuple[float, float]]] = {
    "Ca_K": {
        "feature": (3923.0, 3943.0),
        "blue_cont": (3903.0, 3923.0),
        "red_cont": (3943.0, 3963.0),
    },
    "Ca_H": {
        "feature": (3958.0, 3978.0),
        "blue_cont": (3938.0, 3958.0),
        "red_cont": (3978.0, 3998.0),
    },
    "H_delta": {
        "feature": (4091.0, 4111.0),
        "blue_cont": (4041.0, 4081.0),
        "red_cont": (4121.0, 4161.0),
    },
    "H_gamma": {
        "feature": (4330.0, 4350.0),
        "blue_cont": (4280.0, 4320.0),
        "red_cont": (4360.0, 4400.0),
    },
    "H_beta": {
        "feature": (4851.0, 4871.0),
        "blue_cont": (4801.0, 4841.0),
        "red_cont": (4881.0, 4921.0),
    },
    "H_alpha": {
        "feature": (6553.0, 6573.0),
        "blue_cont": (6503.0, 6543.0),
        "red_cont": (6583.0, 6623.0),
    },
    "D4000": {
        "feature": (4000.0, 4100.0),
        "blue_cont": (3850.0, 3950.0),
        "red_cont": (4000.0, 4100.0),
    },
    "Mg_b": {
        "feature": (5160.0, 5192.0),
        "blue_cont": (5100.0, 5150.0),
        "red_cont": (5200.0, 5250.0),
    },
    "Na_D": {
        "feature": (5878.0, 5908.0),
        "blue_cont": (5838.0, 5868.0),
        "red_cont": (5918.0, 5948.0),
    },
}
"""Built-in spectral index definitions.

Each entry maps an index name to a dict with ``feature``, ``blue_cont``,
and ``red_cont`` wavelength ranges (Angstroms).  For the D4000 break the
index is the ratio of red-to-blue mean flux densities.
"""


# ---------------------------------------------------------------------------
#  Lazy optional imports
# ---------------------------------------------------------------------------

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


def _import_scipy_signal():  # noqa: D401
    """Lazily import *scipy.signal* with a clear error on failure."""
    try:
        import scipy.signal as sig
        return sig
    except ImportError:
        raise ImportError(
            "scipy is required for this function.  "
            "Install it with:  pip install scipy"
        ) from None


def _import_scipy_interpolate():  # noqa: D401
    """Lazily import *scipy.interpolate* with a clear error on failure."""
    try:
        import scipy.interpolate as interp
        return interp
    except ImportError:
        raise ImportError(
            "scipy is required for this function.  "
            "Install it with:  pip install scipy"
        ) from None


# ---------------------------------------------------------------------------
#  Private helpers
# ---------------------------------------------------------------------------

def _robust_std(data: np.ndarray) -> float:
    """Estimate standard deviation via MAD, robust to outliers."""
    med = np.median(data)
    mad = np.median(np.abs(data - med))
    return float(mad * MAD_TO_STD)


def _band_mean_flux(
    wavelengths: np.ndarray,
    flux: np.ndarray,
    wmin: float,
    wmax: float,
) -> float:
    """Compute mean flux within a wavelength band."""
    mask = (wavelengths >= wmin) & (wavelengths <= wmax)
    if not np.any(mask):
        return np.nan
    return float(np.mean(flux[mask]))


# =========================================================================
#  1. Wavelength Calibration
# =========================================================================

def identify_lines(
    spectrum: np.ndarray,
    threshold: float = 5.0,
    min_distance: int = 5,
) -> np.ndarray:
    """Detect emission/absorption line positions in a 1-D spectrum.

    Identifies peaks that exceed a sigma-based threshold above the local
    background estimated from the median of the spectrum.  Uses
    ``scipy.signal.find_peaks`` when available; falls back to a simple
    NumPy implementation.

    Parameters
    ----------
    spectrum : ndarray
        1-D flux array.
    threshold : float
        Detection threshold in units of sigma above the background.
    min_distance : int
        Minimum separation (pixels) between detected lines.

    Returns
    -------
    ndarray
        Sorted array of pixel positions (integer indices) of detected lines.
    """
    spec = np.asarray(spectrum, dtype=np.float64).ravel()
    if spec.size < 3:
        return np.array([], dtype=np.intp)

    background = np.median(spec)
    noise = _robust_std(spec)
    if noise == 0.0:
        noise = np.std(spec, ddof=1)
    if noise == 0.0:
        return np.array([], dtype=np.intp)

    level = background + threshold * noise

    try:
        sig = _import_scipy_signal()
        peaks, _ = sig.find_peaks(spec, height=level, distance=min_distance)
        return np.sort(peaks)
    except ImportError:
        pass

    # Fallback: simple local-maximum search
    candidates: list[int] = []
    for i in range(1, spec.size - 1):
        if spec[i] > spec[i - 1] and spec[i] > spec[i + 1] and spec[i] >= level:
            candidates.append(i)

    # Enforce minimum distance
    if not candidates:
        return np.array([], dtype=np.intp)

    filtered = [candidates[0]]
    for c in candidates[1:]:
        if c - filtered[-1] >= min_distance:
            filtered.append(c)
    return np.asarray(filtered, dtype=np.intp)


def fit_wavelength_solution(
    pixel_positions: np.ndarray,
    known_wavelengths: np.ndarray,
    order: int = 3,
) -> dict[str, Any]:
    """Fit a polynomial wavelength solution from pixel-wavelength pairs.

    Parameters
    ----------
    pixel_positions : ndarray
        Pixel coordinates of identified calibration lines.
    known_wavelengths : ndarray
        Corresponding known wavelengths (same length as *pixel_positions*).
    order : int
        Polynomial order for the fit.

    Returns
    -------
    dict
        ``coefficients`` : ndarray of polynomial coefficients (highest order
        first, compatible with ``numpy.polyval``).
        ``residual_rms`` : float RMS of fit residuals.
        ``order`` : int polynomial order used.
        ``n_lines`` : int number of calibration lines.

    Raises
    ------
    ValueError
        If inputs have mismatched lengths or too few points for the
        requested order.
    """
    pix = np.asarray(pixel_positions, dtype=np.float64).ravel()
    wav = np.asarray(known_wavelengths, dtype=np.float64).ravel()

    if pix.size != wav.size:
        raise ValueError(
            f"pixel_positions ({pix.size}) and known_wavelengths "
            f"({wav.size}) must have the same length."
        )
    if pix.size < order + 1:
        raise ValueError(
            f"Need at least {order + 1} points for a polynomial of "
            f"order {order}, got {pix.size}."
        )

    coeffs = np.polyfit(pix, wav, order)
    fitted = np.polyval(coeffs, pix)
    residual_rms = float(np.sqrt(np.mean((wav - fitted) ** 2)))

    return {
        "coefficients": coeffs,
        "residual_rms": residual_rms,
        "order": int(order),
        "n_lines": int(pix.size),
    }


def apply_wavelength_solution(
    pixels: np.ndarray,
    coefficients: np.ndarray,
) -> np.ndarray:
    """Evaluate a polynomial wavelength solution at given pixel positions.

    Parameters
    ----------
    pixels : ndarray
        Pixel positions at which to evaluate the solution.
    coefficients : ndarray
        Polynomial coefficients (highest order first), as returned by
        :func:`fit_wavelength_solution`.

    Returns
    -------
    ndarray
        Wavelength values corresponding to each pixel.
    """
    pix = np.asarray(pixels, dtype=np.float64)
    coeffs = np.asarray(coefficients, dtype=np.float64)
    return np.polyval(coeffs, pix)


def calibrate_spectrum(
    flux: np.ndarray,
    pixel_positions: np.ndarray,
    known_wavelengths: np.ndarray,
    order: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """One-shot wavelength calibration of a 1-D spectrum.

    Fits a wavelength solution from the supplied calibration lines, then
    applies it to every pixel in the spectrum.

    Parameters
    ----------
    flux : ndarray
        1-D flux array.
    pixel_positions : ndarray
        Pixel coordinates of calibration lines.
    known_wavelengths : ndarray
        Known wavelengths of those lines.
    order : int
        Polynomial order for the wavelength solution.

    Returns
    -------
    wavelengths : ndarray
        Wavelength array spanning the full spectrum.
    flux : ndarray
        Original flux array (unchanged).
    """
    flux_arr = np.asarray(flux, dtype=np.float64).ravel()
    solution = fit_wavelength_solution(pixel_positions, known_wavelengths, order)
    all_pixels = np.arange(flux_arr.size, dtype=np.float64)
    wavelengths = apply_wavelength_solution(all_pixels, solution["coefficients"])
    return wavelengths, flux_arr


# =========================================================================
#  2. Sky Subtraction
# =========================================================================

def extract_sky(
    image_2d: np.ndarray,
    sky_regions: Sequence[tuple[int, int]],
) -> np.ndarray:
    """Extract a 1-D sky spectrum from a 2-D spectral image.

    Combines rows within the specified sky regions using the median to
    produce a single representative sky spectrum.

    Parameters
    ----------
    image_2d : ndarray
        2-D spectral image (rows = spatial, columns = dispersion).
    sky_regions : sequence of (int, int)
        Row ranges ``(start_row, end_row)`` defining sky apertures.

    Returns
    -------
    ndarray
        1-D median sky spectrum (length = number of columns).

    Raises
    ------
    ValueError
        If *sky_regions* is empty or references rows outside the image.
    """
    img = np.asarray(image_2d, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError("image_2d must be 2-dimensional.")
    if not sky_regions:
        raise ValueError("sky_regions must contain at least one (start, end) pair.")

    nrows = img.shape[0]
    rows: list[np.ndarray] = []
    for start, end in sky_regions:
        if start < 0 or end > nrows or start >= end:
            raise ValueError(
                f"Invalid sky region ({start}, {end}) for image with "
                f"{nrows} rows."
            )
        rows.append(img[start:end, :])

    sky_block = np.vstack(rows)
    return np.median(sky_block, axis=0)


def subtract_sky_1d(
    spectrum: np.ndarray,
    sky: np.ndarray,
    scale: float = 1.0,
) -> np.ndarray:
    """Subtract a scaled sky spectrum from a 1-D science spectrum.

    Parameters
    ----------
    spectrum : ndarray
        1-D science spectrum.
    sky : ndarray
        1-D sky spectrum (must match length of *spectrum*).
    scale : float
        Multiplicative scale factor applied to *sky* before subtraction.

    Returns
    -------
    ndarray
        Sky-subtracted spectrum.
    """
    spec = np.asarray(spectrum, dtype=np.float64)
    sky_arr = np.asarray(sky, dtype=np.float64)
    return spec - scale * sky_arr


def subtract_sky_2d(
    image_2d: np.ndarray,
    sky_regions: Sequence[tuple[int, int]],
    method: Literal["median", "polynomial"] = "median",
) -> np.ndarray:
    """Subtract sky background from a 2-D spectral image.

    Parameters
    ----------
    image_2d : ndarray
        2-D spectral image (rows = spatial, columns = dispersion).
    sky_regions : sequence of (int, int)
        Row ranges defining sky apertures.
    method : {"median", "polynomial"}
        ``"median"`` subtracts the median sky spectrum from every row.
        ``"polynomial"`` fits a low-order polynomial along the spatial axis
        for each column, using only the sky rows, then subtracts the
        evaluated fit at every row.

    Returns
    -------
    ndarray
        Sky-subtracted 2-D image.
    """
    img = np.asarray(image_2d, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError("image_2d must be 2-dimensional.")

    if method == "median":
        sky_1d = extract_sky(img, sky_regions)
        return img - sky_1d[np.newaxis, :]

    if method == "polynomial":
        nrows, ncols = img.shape
        # Build mask of sky rows
        sky_row_indices: list[int] = []
        for start, end in sky_regions:
            sky_row_indices.extend(range(start, end))
        sky_rows = np.array(sky_row_indices, dtype=np.float64)

        result = img.copy()
        all_rows = np.arange(nrows, dtype=np.float64)

        for col in range(ncols):
            sky_vals = img[sky_row_indices, col]
            # Fit degree-2 polynomial along spatial axis
            poly_order = min(2, max(0, len(sky_row_indices) - 1))
            coeffs = np.polyfit(sky_rows, sky_vals, poly_order)
            result[:, col] = img[:, col] - np.polyval(coeffs, all_rows)

        return result

    raise ValueError(f"Unknown method {method!r}; use 'median' or 'polynomial'.")


# =========================================================================
#  3. Radial Velocity
# =========================================================================

def cross_correlate_rv(
    observed: np.ndarray,
    template: np.ndarray,
    wavelengths: np.ndarray,
    v_range: tuple[float, float] = (-500.0, 500.0),
    v_step: float = 1.0,
) -> dict[str, Any]:
    """Measure radial velocity via cross-correlation.

    Computes the cross-correlation function (CCF) between an observed
    spectrum and a template over a grid of trial velocities.  The peak of
    the CCF is refined with a parabolic fit to yield sub-step precision.

    Parameters
    ----------
    observed : ndarray
        1-D observed flux array.
    template : ndarray
        1-D template flux array (same wavelength grid as *observed*).
    wavelengths : ndarray
        Wavelength array common to both spectra.
    v_range : tuple of float
        ``(v_min, v_max)`` velocity search range in km/s.
    v_step : float
        Velocity step size in km/s.

    Returns
    -------
    dict
        ``velocity`` : best-fit radial velocity (km/s).
        ``velocity_err`` : estimated uncertainty (km/s) from parabolic fit.
        ``ccf_peak`` : peak CCF value.
        ``velocities`` : ndarray of trial velocities.
        ``ccf`` : ndarray of CCF values.
    """
    obs = np.asarray(observed, dtype=np.float64).ravel()
    tmpl = np.asarray(template, dtype=np.float64).ravel()
    wave = np.asarray(wavelengths, dtype=np.float64).ravel()

    if obs.size != tmpl.size or obs.size != wave.size:
        raise ValueError(
            "observed, template, and wavelengths must all have the "
            "same length."
        )

    # Rebin both spectra onto a uniform log-wavelength grid so that a
    # pixel shift corresponds to a constant velocity increment.
    ln_wave_orig = np.log(wave)
    n_pix = obs.size
    ln_wave_uniform = np.linspace(ln_wave_orig[0], ln_wave_orig[-1], n_pix)
    d_ln = (ln_wave_uniform[-1] - ln_wave_uniform[0]) / (n_pix - 1)
    v_per_pix = SPEED_OF_LIGHT_KMS * d_ln

    obs_rebin = np.interp(ln_wave_uniform, ln_wave_orig, obs)
    tmpl_rebin = np.interp(ln_wave_uniform, ln_wave_orig, tmpl)

    # Normalise to zero mean, unit variance
    obs_norm = obs_rebin - np.mean(obs_rebin)
    obs_std = np.std(obs_norm)
    if obs_std > 0:
        obs_norm /= obs_std

    tmpl_norm = tmpl_rebin - np.mean(tmpl_rebin)
    tmpl_std = np.std(tmpl_norm)
    if tmpl_std > 0:
        tmpl_norm /= tmpl_std

    # FFT-based cross-correlation (Tonry & Davis 1979 approach).
    # In the uniform log-lambda grid a Doppler shift is a pure pixel
    # translation.  We compute the CCF at integer-pixel lags, then
    # refine the peak location with parabolic interpolation.
    n_fft = 2 * n_pix  # zero-pad for linear (non-circular) correlation
    fft_obs = np.fft.rfft(obs_norm, n=n_fft)
    fft_tmpl = np.fft.rfft(tmpl_norm, n=n_fft)
    ccf_full = np.fft.irfft(fft_obs * np.conj(fft_tmpl), n=n_fft)

    # Map pixel lags within v_range to velocity values.
    # lag k (pixels) -> v = k * v_per_pix.
    # Negative lags wrap: ccf_full[n_fft - k] corresponds to lag -k.
    max_lag = int(np.ceil(max(abs(v_range[0]), abs(v_range[1])) / v_per_pix)) + 1
    max_lag = min(max_lag, n_pix - 1)

    lag_indices = np.arange(-max_lag, max_lag + 1)
    fft_velocities = lag_indices * v_per_pix
    ccf_at_lags = np.empty(lag_indices.size, dtype=np.float64)
    for j, lag in enumerate(lag_indices):
        ccf_at_lags[j] = ccf_full[lag % n_fft]

    # Find the coarse peak
    peak_j = int(np.argmax(ccf_at_lags))
    peak_lag = lag_indices[peak_j]

    # Sub-pixel refinement: fit a quadratic to multiple points around
    # the peak for robust centroiding even when the CCF is broad.
    half_fit = min(3, peak_j, lag_indices.size - 1 - peak_j)
    if half_fit >= 1:
        fit_slice = slice(peak_j - half_fit, peak_j + half_fit + 1)
        fit_lags = lag_indices[fit_slice].astype(np.float64)
        fit_vals = ccf_at_lags[fit_slice]
        poly = np.polyfit(fit_lags, fit_vals, 2)
        # Peak of quadratic  a*x^2 + b*x + c  is at  x = -b/(2a)
        if abs(poly[0]) > 0:
            refined_lag = -poly[1] / (2.0 * poly[0])
            # Only accept if the refined lag is within the fitting window
            if fit_lags[0] <= refined_lag <= fit_lags[-1]:
                best_v = float(refined_lag * v_per_pix)
            else:
                best_v = float(peak_lag * v_per_pix)
        else:
            best_v = float(peak_lag * v_per_pix)
    else:
        best_v = float(peak_lag * v_per_pix)

    # Now build the output CCF on the user-requested velocity grid
    # by interpolating the pixel-resolution CCF.
    velocities = np.arange(v_range[0], v_range[1] + v_step * 0.5, v_step)
    ccf = np.interp(velocities, fft_velocities, ccf_at_lags)

    # Estimate the peak CCF value and velocity uncertainty
    ccf_peak = float(np.max(ccf))
    velocity_err = v_per_pix  # conservative default

    # Refine uncertainty from curvature at the parabolic peak
    if 0 < peak_j < lag_indices.size - 1:
        y_m = ccf_at_lags[peak_j - 1]
        y_0 = ccf_at_lags[peak_j]
        y_p = ccf_at_lags[peak_j + 1]
        denom = 2.0 * (2.0 * y_0 - y_m - y_p)
        if abs(denom) > 0:
            curvature = abs(denom) / (v_per_pix ** 2)
            if curvature > 0:
                velocity_err = float(1.0 / np.sqrt(curvature))

    return {
        "velocity": best_v,
        "velocity_err": velocity_err,
        "ccf_peak": ccf_peak,
        "velocities": velocities,
        "ccf": ccf,
    }


def doppler_shift(
    wavelengths: np.ndarray,
    velocity: float,
) -> np.ndarray:
    """Apply a Doppler shift to a wavelength array.

    Uses the non-relativistic approximation:
    ``w_new = w * (1 + v / c)``

    Parameters
    ----------
    wavelengths : ndarray
        Input wavelength array (any units).
    velocity : float
        Radial velocity in km/s (positive = receding).

    Returns
    -------
    ndarray
        Shifted wavelength array.
    """
    wave = np.asarray(wavelengths, dtype=np.float64)
    return wave * (1.0 + velocity / SPEED_OF_LIGHT_KMS)


def barycentric_correction(
    wavelengths: np.ndarray,
    v_bary: float,
) -> np.ndarray:
    """Apply a barycentric velocity correction to wavelengths.

    Shifts wavelengths from the topocentric to the barycentric frame:
    ``w_bary = w * (1 + v_bary / c)``

    Parameters
    ----------
    wavelengths : ndarray
        Topocentric wavelength array.
    v_bary : float
        Barycentric velocity in km/s.

    Returns
    -------
    ndarray
        Barycentric-corrected wavelength array.
    """
    return doppler_shift(wavelengths, v_bary)


# =========================================================================
#  4. Spectral Classification
# =========================================================================

def measure_spectral_indices(
    wavelengths: np.ndarray,
    flux: np.ndarray,
    indices: dict[str, dict[str, tuple[float, float]]] | None = None,
) -> dict[str, float]:
    """Measure standard spectral indices.

    For each index the mean flux in the *feature* band is divided by the
    mean flux in the pseudo-continuum (average of *blue_cont* and
    *red_cont* bands).  For the D4000 break the index is defined as the
    ratio of the red to blue mean flux densities.

    Parameters
    ----------
    wavelengths : ndarray
        1-D wavelength array.
    flux : ndarray
        1-D flux array.
    indices : dict or None
        Index definitions.  If *None*, :data:`DEFAULT_SPECTRAL_INDICES` is
        used.  Each key maps to a dict with ``feature``, ``blue_cont``,
        and ``red_cont`` wavelength-range tuples.

    Returns
    -------
    dict
        Mapping of index name to measured value (float).  ``nan`` is
        returned for indices that cannot be measured (e.g., wavelength
        range not covered).
    """
    wave = np.asarray(wavelengths, dtype=np.float64).ravel()
    fl = np.asarray(flux, dtype=np.float64).ravel()
    if indices is None:
        indices = DEFAULT_SPECTRAL_INDICES

    results: dict[str, float] = {}
    for name, bands in indices.items():
        feat = _band_mean_flux(wave, fl, *bands["feature"])
        blue = _band_mean_flux(wave, fl, *bands["blue_cont"])
        red = _band_mean_flux(wave, fl, *bands["red_cont"])

        if name == "D4000":
            # D4000 break: ratio of red to blue flux
            if np.isnan(blue) or np.isnan(red) or blue == 0.0:
                results[name] = np.nan
            else:
                results[name] = float(red / blue)
        else:
            cont = 0.5 * (blue + red)
            if np.isnan(feat) or np.isnan(cont) or cont == 0.0:
                results[name] = np.nan
            else:
                results[name] = float(feat / cont)

    return results


def equivalent_width_spectral(
    wavelengths: np.ndarray,
    flux: np.ndarray,
    line_center: float,
    width: float = 10.0,
    continuum_width: float = 20.0,
) -> dict[str, float]:
    """Measure the equivalent width of a spectral line.

    The continuum is estimated from two bands flanking the line, and the
    equivalent width is integrated via the trapezoidal rule.

    Parameters
    ----------
    wavelengths : ndarray
        1-D wavelength array.
    flux : ndarray
        1-D flux array.
    line_center : float
        Central wavelength of the line.
    width : float
        Half-width of the line integration window (same units as
        *wavelengths*).
    continuum_width : float
        Half-width of each continuum estimation window.

    Returns
    -------
    dict
        ``ew`` : equivalent width (positive for absorption).
        ``ew_err`` : estimated uncertainty from noise in the continuum
        bands.
        ``continuum_level`` : fitted continuum flux level at line centre.
    """
    wave = np.asarray(wavelengths, dtype=np.float64).ravel()
    fl = np.asarray(flux, dtype=np.float64).ravel()

    # Continuum bands on either side of the line
    blue_lo = line_center - width - continuum_width
    blue_hi = line_center - width
    red_lo = line_center + width
    red_hi = line_center + width + continuum_width

    blue_mask = (wave >= blue_lo) & (wave <= blue_hi)
    red_mask = (wave >= red_lo) & (wave <= red_hi)
    cont_mask = blue_mask | red_mask

    if np.sum(cont_mask) < 2:
        return {"ew": np.nan, "ew_err": np.nan, "continuum_level": np.nan}

    # Linear continuum fit
    cont_wave = wave[cont_mask]
    cont_flux = fl[cont_mask]
    coeffs = np.polyfit(cont_wave, cont_flux, 1)
    continuum_at_center = float(np.polyval(coeffs, line_center))

    # Line integration window
    line_mask = (wave >= line_center - width) & (wave <= line_center + width)
    if np.sum(line_mask) < 2 or continuum_at_center == 0.0:
        return {
            "ew": np.nan,
            "ew_err": np.nan,
            "continuum_level": continuum_at_center,
        }

    line_wave = wave[line_mask]
    line_flux = fl[line_mask]
    cont_model = np.polyval(coeffs, line_wave)

    # Equivalent width: integral of (1 - F/F_cont) dw
    integrand = 1.0 - line_flux / cont_model
    _trapz = getattr(np, "trapezoid", None) or np.trapz
    ew = float(_trapz(integrand, line_wave))

    # Noise-based uncertainty estimate
    cont_residuals = cont_flux - np.polyval(coeffs, cont_wave)
    noise = float(np.std(cont_residuals, ddof=1)) if cont_residuals.size > 1 else 0.0
    n_pix = np.sum(line_mask)
    dlambda = float(np.mean(np.diff(line_wave))) if line_wave.size > 1 else 1.0
    ew_err = float(noise / continuum_at_center * np.sqrt(float(n_pix)) * dlambda)

    return {
        "ew": ew,
        "ew_err": ew_err,
        "continuum_level": continuum_at_center,
    }


def classify_spectrum(
    wavelengths: np.ndarray,
    flux: np.ndarray,
    templates: dict[str, np.ndarray],
) -> list[tuple[str, float, float]]:
    """Match an observed spectrum against a template library.

    Each template is optimally scaled to match the observation before
    computing the reduced chi-squared statistic.

    Parameters
    ----------
    wavelengths : ndarray
        1-D wavelength array (used only for length validation).
    flux : ndarray
        1-D observed flux array.
    templates : dict
        Mapping of template name to 1-D flux array (same length as
        *flux*).

    Returns
    -------
    list of (str, float, float)
        ``(name, chi2, scale)`` tuples sorted by ascending chi-squared.
    """
    obs = np.asarray(flux, dtype=np.float64).ravel()
    n = obs.size

    results: list[tuple[str, float, float]] = []
    for name, tmpl_flux in templates.items():
        tmpl = np.asarray(tmpl_flux, dtype=np.float64).ravel()
        if tmpl.size != n:
            continue
        # Optimal linear scaling: scale = sum(obs * tmpl) / sum(tmpl^2)
        denom = np.sum(tmpl ** 2)
        if denom == 0:
            continue
        scale = float(np.sum(obs * tmpl) / denom)
        residual = obs - scale * tmpl
        chi2 = float(np.sum(residual ** 2) / max(n - 1, 1))
        results.append((name, chi2, scale))

    results.sort(key=lambda x: x[1])
    return results


# =========================================================================
#  5. Emission-Line Fitting
# =========================================================================

def gaussian_line(
    wavelengths: np.ndarray,
    amplitude: float,
    center: float,
    sigma: float,
    continuum: float = 0.0,
) -> np.ndarray:
    """Gaussian emission/absorption line profile.

    .. math::
        f(\\lambda) = A \\exp\\!\\left(-\\frac{(\\lambda - \\mu)^2}
        {2\\sigma^2}\\right) + C

    Parameters
    ----------
    wavelengths : ndarray
        Wavelength array.
    amplitude : float
        Peak amplitude above continuum.
    center : float
        Line centre wavelength.
    sigma : float
        Gaussian sigma (width parameter).
    continuum : float
        Constant continuum level.

    Returns
    -------
    ndarray
        Flux values.
    """
    w = np.asarray(wavelengths, dtype=np.float64)
    return continuum + amplitude * np.exp(-0.5 * ((w - center) / sigma) ** 2)


def voigt_line(
    wavelengths: np.ndarray,
    amplitude: float,
    center: float,
    sigma: float,
    gamma: float,
    continuum: float = 0.0,
) -> np.ndarray:
    """Voigt (pseudo-Voigt) emission/absorption line profile.

    Approximation using a linear combination of Gaussian and Lorentzian
    components weighted by an *eta* mixing parameter derived from the
    Gaussian and Lorentzian FWHM contributions.

    Parameters
    ----------
    wavelengths : ndarray
        Wavelength array.
    amplitude : float
        Peak amplitude above continuum.
    center : float
        Line centre wavelength.
    sigma : float
        Gaussian sigma.
    gamma : float
        Lorentzian half-width at half-maximum.
    continuum : float
        Constant continuum level.

    Returns
    -------
    ndarray
        Flux values.
    """
    w = np.asarray(wavelengths, dtype=np.float64)

    # FWHM of each component
    fwhm_g = 2.0 * sigma * np.sqrt(2.0 * np.log(2.0))
    fwhm_l = 2.0 * gamma

    # Total FWHM (Thompson et al. 1987 approximation)
    fwhm = (
        fwhm_g ** 5
        + 2.69269 * fwhm_g ** 4 * fwhm_l
        + 2.42843 * fwhm_g ** 3 * fwhm_l ** 2
        + 4.47163 * fwhm_g ** 2 * fwhm_l ** 3
        + 0.07842 * fwhm_g * fwhm_l ** 4
        + fwhm_l ** 5
    ) ** 0.2

    # Eta mixing parameter (fraction of Lorentzian)
    if fwhm > 0:
        ratio = fwhm_l / fwhm
        eta = 1.36603 * ratio - 0.47719 * ratio ** 2 + 0.11116 * ratio ** 3
        eta = max(0.0, min(1.0, eta))
    else:
        eta = 0.0

    # Gaussian component (normalised to peak = 1)
    gauss = np.exp(-0.5 * ((w - center) / sigma) ** 2) if sigma > 0 else np.zeros_like(w)

    # Lorentzian component (normalised to peak = 1)
    lorentz = gamma ** 2 / ((w - center) ** 2 + gamma ** 2) if gamma > 0 else np.zeros_like(w)

    profile = (1.0 - eta) * gauss + eta * lorentz
    return continuum + amplitude * profile


def fit_emission_line(
    wavelengths: np.ndarray,
    flux: np.ndarray,
    center_guess: float,
    profile: Literal["gaussian", "voigt"] = "gaussian",
) -> dict[str, float]:
    """Fit a single emission line.

    Uses ``scipy.optimize.curve_fit`` for non-linear least-squares fitting
    of the selected line profile to the data.

    Parameters
    ----------
    wavelengths : ndarray
        1-D wavelength array.
    flux : ndarray
        1-D flux array.
    center_guess : float
        Initial guess for the line centre.
    profile : {"gaussian", "voigt"}
        Line profile to fit.

    Returns
    -------
    dict
        ``amplitude``, ``center``, ``sigma`` (and ``gamma`` for Voigt),
        ``fwhm``, ``flux_integrated``, ``continuum``, ``chi2``.

    Raises
    ------
    RuntimeError
        If the fit does not converge.
    """
    opt = _import_scipy_optimize()

    wave = np.asarray(wavelengths, dtype=np.float64).ravel()
    fl = np.asarray(flux, dtype=np.float64).ravel()

    cont_guess = float(np.median(fl))
    amp_guess = float(np.max(fl) - cont_guess)
    sigma_guess = 2.0  # pixels / Angstroms -- reasonable starting point

    if profile == "gaussian":
        def _model(w, amp, cen, sig, cont):
            return gaussian_line(w, amp, cen, sig, cont)

        p0 = [amp_guess, center_guess, sigma_guess, cont_guess]
        bounds_lo = [0.0, wave[0], 0.01, -np.inf]
        bounds_hi = [np.inf, wave[-1], (wave[-1] - wave[0]) / 2.0, np.inf]
        try:
            popt, _ = opt.curve_fit(
                _model, wave, fl, p0=p0,
                bounds=(bounds_lo, bounds_hi), maxfev=10000,
            )
        except RuntimeError as exc:
            raise RuntimeError(f"Gaussian fit did not converge: {exc}") from None

        amp, cen, sig, cont = popt
        fwhm = 2.0 * sig * np.sqrt(2.0 * np.log(2.0))
        flux_int = float(amp * sig * np.sqrt(2.0 * np.pi))
        fitted = _model(wave, *popt)
        residuals = fl - fitted
        chi2 = float(np.sum(residuals ** 2) / max(fl.size - 4, 1))

        return {
            "amplitude": float(amp),
            "center": float(cen),
            "sigma": float(sig),
            "fwhm": float(fwhm),
            "flux_integrated": flux_int,
            "continuum": float(cont),
            "chi2": chi2,
        }

    if profile == "voigt":
        gamma_guess = 1.0

        def _model_v(w, amp, cen, sig, gam, cont):
            return voigt_line(w, amp, cen, sig, gam, cont)

        p0 = [amp_guess, center_guess, sigma_guess, gamma_guess, cont_guess]
        bounds_lo = [0.0, wave[0], 0.01, 0.01, -np.inf]
        bounds_hi = [np.inf, wave[-1], (wave[-1] - wave[0]) / 2.0,
                     (wave[-1] - wave[0]) / 2.0, np.inf]
        try:
            popt, _ = opt.curve_fit(
                _model_v, wave, fl, p0=p0,
                bounds=(bounds_lo, bounds_hi), maxfev=10000,
            )
        except RuntimeError as exc:
            raise RuntimeError(f"Voigt fit did not converge: {exc}") from None

        amp, cen, sig, gam, cont = popt
        fwhm_g = 2.0 * sig * np.sqrt(2.0 * np.log(2.0))
        fwhm_l = 2.0 * gam
        fwhm = (
            fwhm_g ** 5
            + 2.69269 * fwhm_g ** 4 * fwhm_l
            + 2.42843 * fwhm_g ** 3 * fwhm_l ** 2
            + 4.47163 * fwhm_g ** 2 * fwhm_l ** 3
            + 0.07842 * fwhm_g * fwhm_l ** 4
            + fwhm_l ** 5
        ) ** 0.2
        flux_int = float(amp * sig * np.sqrt(2.0 * np.pi))
        fitted = _model_v(wave, *popt)
        residuals = fl - fitted
        chi2 = float(np.sum(residuals ** 2) / max(fl.size - 5, 1))

        return {
            "amplitude": float(amp),
            "center": float(cen),
            "sigma": float(sig),
            "gamma": float(gam),
            "fwhm": float(fwhm),
            "flux_integrated": flux_int,
            "continuum": float(cont),
            "chi2": chi2,
        }

    raise ValueError(f"Unknown profile {profile!r}; use 'gaussian' or 'voigt'.")


def fit_multi_lines(
    wavelengths: np.ndarray,
    flux: np.ndarray,
    center_guesses: Sequence[float],
    profile: Literal["gaussian", "voigt"] = "gaussian",
) -> list[dict[str, float]]:
    """Fit multiple emission lines simultaneously.

    Constructs a composite model of *N* lines (plus a shared continuum)
    and fits all parameters at once using ``scipy.optimize.curve_fit``.

    Parameters
    ----------
    wavelengths : ndarray
        1-D wavelength array.
    flux : ndarray
        1-D flux array.
    center_guesses : sequence of float
        Initial guesses for the centre of each line.
    profile : {"gaussian", "voigt"}
        Line profile for all lines.

    Returns
    -------
    list of dict
        One dict per line with the same keys as
        :func:`fit_emission_line`.
    """
    opt = _import_scipy_optimize()

    wave = np.asarray(wavelengths, dtype=np.float64).ravel()
    fl = np.asarray(flux, dtype=np.float64).ravel()
    n_lines = len(center_guesses)

    if n_lines == 0:
        return []

    cont_guess = float(np.median(fl))
    amp_guess = float(np.max(fl) - cont_guess)
    sigma_guess = 2.0
    gamma_guess = 1.0
    half_range = (wave[-1] - wave[0]) / 2.0

    if profile == "gaussian":
        # params: [cont, amp1, cen1, sig1, amp2, cen2, sig2, ...]
        p0 = [cont_guess]
        lo = [-np.inf]
        hi = [np.inf]
        for cg in center_guesses:
            p0.extend([amp_guess, cg, sigma_guess])
            lo.extend([0.0, wave[0], 0.01])
            hi.extend([np.inf, wave[-1], half_range])

        def _multi_gauss(w, *params):
            cont = params[0]
            result = np.full_like(w, cont)
            for k in range(n_lines):
                a = params[1 + 3 * k]
                c = params[2 + 3 * k]
                s = params[3 + 3 * k]
                result += a * np.exp(-0.5 * ((w - c) / s) ** 2)
            return result

        try:
            popt, _ = opt.curve_fit(
                _multi_gauss, wave, fl, p0=p0,
                bounds=(lo, hi), maxfev=20000,
            )
        except RuntimeError as exc:
            raise RuntimeError(
                f"Multi-line Gaussian fit did not converge: {exc}"
            ) from None

        fitted_total = _multi_gauss(wave, *popt)
        residuals = fl - fitted_total
        n_params = 1 + 3 * n_lines
        dof = max(fl.size - n_params, 1)
        total_chi2 = float(np.sum(residuals ** 2) / dof)

        results: list[dict[str, float]] = []
        cont = float(popt[0])
        for k in range(n_lines):
            amp = float(popt[1 + 3 * k])
            cen = float(popt[2 + 3 * k])
            sig = float(popt[3 + 3 * k])
            fwhm = 2.0 * sig * np.sqrt(2.0 * np.log(2.0))
            flux_int = float(amp * sig * np.sqrt(2.0 * np.pi))
            results.append({
                "amplitude": amp,
                "center": cen,
                "sigma": sig,
                "fwhm": float(fwhm),
                "flux_integrated": flux_int,
                "continuum": cont,
                "chi2": total_chi2,
            })
        return results

    if profile == "voigt":
        # params: [cont, amp1, cen1, sig1, gam1, ...]
        p0 = [cont_guess]
        lo = [-np.inf]
        hi = [np.inf]
        for cg in center_guesses:
            p0.extend([amp_guess, cg, sigma_guess, gamma_guess])
            lo.extend([0.0, wave[0], 0.01, 0.01])
            hi.extend([np.inf, wave[-1], half_range, half_range])

        def _multi_voigt(w, *params):
            cont = params[0]
            result = np.full_like(w, cont)
            for k in range(n_lines):
                a = params[1 + 4 * k]
                c = params[2 + 4 * k]
                s = params[3 + 4 * k]
                g = params[4 + 4 * k]
                result += voigt_line(w, a, c, s, g, continuum=0.0)
            return result

        try:
            popt, _ = opt.curve_fit(
                _multi_voigt, wave, fl, p0=p0,
                bounds=(lo, hi), maxfev=20000,
            )
        except RuntimeError as exc:
            raise RuntimeError(
                f"Multi-line Voigt fit did not converge: {exc}"
            ) from None

        fitted_total = _multi_voigt(wave, *popt)
        residuals = fl - fitted_total
        n_params = 1 + 4 * n_lines
        dof = max(fl.size - n_params, 1)
        total_chi2 = float(np.sum(residuals ** 2) / dof)

        results = []
        cont = float(popt[0])
        for k in range(n_lines):
            amp = float(popt[1 + 4 * k])
            cen = float(popt[2 + 4 * k])
            sig = float(popt[3 + 4 * k])
            gam = float(popt[4 + 4 * k])
            fwhm_g = 2.0 * sig * np.sqrt(2.0 * np.log(2.0))
            fwhm_l = 2.0 * gam
            fwhm = (
                fwhm_g ** 5
                + 2.69269 * fwhm_g ** 4 * fwhm_l
                + 2.42843 * fwhm_g ** 3 * fwhm_l ** 2
                + 4.47163 * fwhm_g ** 2 * fwhm_l ** 3
                + 0.07842 * fwhm_g * fwhm_l ** 4
                + fwhm_l ** 5
            ) ** 0.2
            flux_int = float(amp * sig * np.sqrt(2.0 * np.pi))
            results.append({
                "amplitude": amp,
                "center": cen,
                "sigma": sig,
                "gamma": gam,
                "fwhm": float(fwhm),
                "flux_integrated": flux_int,
                "continuum": cont,
                "chi2": total_chi2,
            })
        return results

    raise ValueError(f"Unknown profile {profile!r}; use 'gaussian' or 'voigt'.")


# =========================================================================
#  6. Multi-Order Echelle Support
# =========================================================================

def extract_orders(
    image_2d: np.ndarray,
    order_traces: Sequence[np.ndarray],
    aperture_width: int = 5,
) -> list[np.ndarray]:
    """Extract spectral orders from an echelle image.

    For each order trace (polynomial coefficients), sum the flux within an
    aperture of the given width centered on the trace.

    Parameters
    ----------
    image_2d : ndarray
        2-D echelle spectral image (rows = spatial, columns = dispersion).
    order_traces : sequence of ndarray
        List of polynomial coefficient arrays (highest order first).
        Each polynomial gives the row position of the order centre as a
        function of column index.
    aperture_width : int
        Full width (in rows) of the extraction aperture.

    Returns
    -------
    list of ndarray
        1-D extracted spectrum for each order.
    """
    img = np.asarray(image_2d, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError("image_2d must be 2-dimensional.")

    nrows, ncols = img.shape
    half_ap = aperture_width // 2
    cols = np.arange(ncols, dtype=np.float64)
    spectra: list[np.ndarray] = []

    for trace_coeffs in order_traces:
        trace_row = np.polyval(np.asarray(trace_coeffs, dtype=np.float64), cols)
        extracted = np.zeros(ncols, dtype=np.float64)

        for c in range(ncols):
            row_center = int(np.round(trace_row[c]))
            r_lo = max(0, row_center - half_ap)
            r_hi = min(nrows, row_center + half_ap + 1)
            if r_lo < r_hi:
                extracted[c] = np.sum(img[r_lo:r_hi, c])

        spectra.append(extracted)

    return spectra


def trace_order(
    image_2d: np.ndarray,
    start_col: int,
    start_row: int,
    trace_step: int = 10,
    search_width: int = 5,
) -> np.ndarray:
    """Trace a single spectral order across a 2-D image.

    Starting from ``(start_row, start_col)``, the algorithm steps along
    the dispersion axis and follows the local peak in a search window.
    A polynomial is then fitted to the measured (column, row) positions.

    Parameters
    ----------
    image_2d : ndarray
        2-D echelle spectral image.
    start_col : int
        Starting column for the trace.
    start_row : int
        Starting row (approximate position of the order).
    trace_step : int
        Column step size between trace measurements.
    search_width : int
        Half-width of the search window (rows) at each step.

    Returns
    -------
    ndarray
        Polynomial coefficients (order 3, highest first) describing the
        trace row position as a function of column.
    """
    img = np.asarray(image_2d, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError("image_2d must be 2-dimensional.")

    nrows, ncols = img.shape
    col_positions: list[float] = []
    row_positions: list[float] = []

    current_row = float(start_row)

    # Trace rightward
    for col in range(start_col, ncols, trace_step):
        r_center = int(np.round(current_row))
        r_lo = max(0, r_center - search_width)
        r_hi = min(nrows, r_center + search_width + 1)
        if r_lo >= r_hi:
            break
        strip = img[r_lo:r_hi, col]
        if strip.size == 0:
            break
        # Sub-pixel centroid
        rows_local = np.arange(r_lo, r_hi, dtype=np.float64)
        total = np.sum(strip)
        if total > 0:
            centroid = float(np.sum(rows_local * strip) / total)
        else:
            centroid = float(r_center)
        col_positions.append(float(col))
        row_positions.append(centroid)
        current_row = centroid

    # Trace leftward
    current_row = float(start_row)
    for col in range(start_col - trace_step, -1, -trace_step):
        r_center = int(np.round(current_row))
        r_lo = max(0, r_center - search_width)
        r_hi = min(nrows, r_center + search_width + 1)
        if r_lo >= r_hi:
            break
        strip = img[r_lo:r_hi, col]
        if strip.size == 0:
            break
        rows_local = np.arange(r_lo, r_hi, dtype=np.float64)
        total = np.sum(strip)
        if total > 0:
            centroid = float(np.sum(rows_local * strip) / total)
        else:
            centroid = float(r_center)
        col_positions.insert(0, float(col))
        row_positions.insert(0, centroid)
        current_row = centroid

    if len(col_positions) < 4:
        # Not enough points for a cubic; fall back to lower order
        order = max(0, len(col_positions) - 1)
    else:
        order = 3

    cols_arr = np.array(col_positions, dtype=np.float64)
    rows_arr = np.array(row_positions, dtype=np.float64)
    return np.polyfit(cols_arr, rows_arr, order)


def merge_orders(
    orders: Sequence[np.ndarray],
    wavelengths_list: Sequence[np.ndarray],
    overlap_method: Literal["weighted", "average"] = "weighted",
) -> tuple[np.ndarray, np.ndarray]:
    """Merge overlapping echelle orders into a single 1-D spectrum.

    In overlap regions the flux is combined using an SNR-weighted average
    (``"weighted"``) or a simple mean (``"average"``).

    Parameters
    ----------
    orders : sequence of ndarray
        1-D flux arrays, one per echelle order.
    wavelengths_list : sequence of ndarray
        Corresponding wavelength arrays (one per order).
    overlap_method : {"weighted", "average"}
        How to combine flux in overlap regions.

    Returns
    -------
    merged_wavelengths : ndarray
        Combined wavelength array (sorted, unique grid).
    merged_flux : ndarray
        Combined flux array.
    """
    if len(orders) != len(wavelengths_list):
        raise ValueError(
            "orders and wavelengths_list must have the same length."
        )
    if len(orders) == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    # Build a common wavelength grid from the union of all orders
    all_wave = np.concatenate(
        [np.asarray(w, dtype=np.float64).ravel() for w in wavelengths_list]
    )
    merged_wave = np.sort(np.unique(all_wave))
    n = merged_wave.size

    flux_sum = np.zeros(n, dtype=np.float64)
    weight_sum = np.zeros(n, dtype=np.float64)

    for flux_arr, wave_arr in zip(orders, wavelengths_list):
        fl = np.asarray(flux_arr, dtype=np.float64).ravel()
        wv = np.asarray(wave_arr, dtype=np.float64).ravel()

        # Interpolate this order onto the merged grid
        wmin, wmax = wv[0], wv[-1]
        mask = (merged_wave >= wmin) & (merged_wave <= wmax)
        if not np.any(mask):
            continue

        interp_flux = np.interp(merged_wave[mask], wv, fl)

        if overlap_method == "weighted":
            # SNR-based weight: use |flux| as proxy (higher flux = higher SNR)
            noise = _robust_std(fl) if fl.size > 1 else 1.0
            noise = max(noise, 1e-30)
            weight = np.abs(interp_flux) / noise
            # Taper weights near order edges to reduce edge artefacts
            n_mask = int(np.sum(mask))
            taper = np.ones(n_mask, dtype=np.float64)
            taper_len = max(1, n_mask // 10)
            ramp = np.linspace(0.0, 1.0, taper_len)
            taper[:taper_len] = ramp
            taper[-taper_len:] = ramp[::-1]
            weight *= taper
        else:
            weight = np.ones(int(np.sum(mask)), dtype=np.float64)

        flux_sum[mask] += interp_flux * weight
        weight_sum[mask] += weight

    # Avoid division by zero
    good = weight_sum > 0
    merged_flux = np.zeros(n, dtype=np.float64)
    merged_flux[good] = flux_sum[good] / weight_sum[good]

    return merged_wave, merged_flux
