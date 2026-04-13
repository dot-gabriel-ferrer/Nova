"""Spectroscopy pipeline tools for NOVA.

Provides an end-to-end spectroscopic reduction and analysis pipeline
including optimal extraction, telluric correction, redshift measurement,
spectral stacking, and continuum fitting -- all without leaving NOVA.

Key capabilities
----------------
- Optimal (Horne 1986) 1-D extraction from 2-D spectra.
- Continuum fitting with polynomial and median-filter methods.
- Telluric absorption modelling and correction.
- Spectral resampling and co-addition (stacking).
- Redshift measurement via cross-correlation.
- Spectral smoothing and rebinning.
- Signal-to-noise estimation.
- Equivalent width and line flux measurements with errors.

Example
-------
>>> from nova.spectroscopy_pipeline import optimal_extract
>>> from nova.spectroscopy_pipeline import fit_continuum
>>> spec, var = optimal_extract(image_2d, trace, profile)
>>> continuum = fit_continuum(wavelength, spec, order=5)
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from nova.constants import SPEED_OF_LIGHT_KMS


# -------------------------------------------------------------------
#  Optimal extraction (Horne 1986)
# -------------------------------------------------------------------

def optimal_extract(
    data_2d: np.ndarray,
    trace: np.ndarray,
    profile: np.ndarray | None = None,
    *,
    variance_2d: np.ndarray | None = None,
    aperture_half: int = 5,
    gain: float = 1.0,
    readnoise: float = 0.0,
) -> dict[str, np.ndarray]:
    """Optimal extraction of a 1-D spectrum from a 2-D spectral image.

    Implements the variance-weighted extraction from Horne (1986, PASP
    98, 609).  The spatial profile can be supplied or estimated from
    the data.

    Parameters
    ----------
    data_2d : np.ndarray
        2-D spectral image, shape ``(n_spatial, n_spectral)``.
    trace : np.ndarray
        1-D array of length ``n_spectral`` giving the spatial centre of
        the trace at each column.
    profile : np.ndarray or None
        Normalised spatial profile, shape ``(2*aperture_half+1, n_spectral)``.
        If None, a simple Gaussian profile is estimated from the data.
    variance_2d : np.ndarray or None
        Per-pixel variance image.  If None, estimated from gain and
        readnoise.
    aperture_half : int
        Half-width of the extraction aperture in spatial pixels.
    gain : float
        Detector gain (e-/ADU).
    readnoise : float
        Read noise (electrons).

    Returns
    -------
    dict
        ``{'flux': np.ndarray, 'variance': np.ndarray,
           'snr': np.ndarray, 'profile_used': np.ndarray}``
    """
    if not isinstance(data_2d, np.ndarray) or data_2d.ndim != 2:
        raise TypeError("data_2d must be a 2-D numpy array.")
    ny, nx = data_2d.shape
    trace = np.asarray(trace, dtype=np.float64)
    if trace.shape[0] != nx:
        raise ValueError("trace length must equal number of spectral columns.")

    ap = aperture_half
    flux_out = np.zeros(nx, dtype=np.float64)
    var_out = np.zeros(nx, dtype=np.float64)

    # If no profile supplied, build a simple Gaussian from data
    if profile is None:
        profile = np.zeros((2 * ap + 1, nx), dtype=np.float64)
        for col in range(nx):
            yc = int(round(trace[col]))
            ylo = max(0, yc - ap)
            yhi = min(ny, yc + ap + 1)
            if yhi <= ylo:
                continue
            strip = data_2d[ylo:yhi, col].astype(np.float64)
            total = strip.sum()
            if total > 0:
                p = strip / total
            else:
                p = np.ones(yhi - ylo, dtype=np.float64) / (yhi - ylo)
            # Pad to full aperture size
            p_full = np.zeros(2 * ap + 1, dtype=np.float64)
            offset = (yc - ap) - ylo
            p_full[-offset:-offset + len(p)] = p if offset <= 0 else p
            if np.sum(p_full) > 0:
                p_full /= np.sum(p_full)
            else:
                p_full = np.ones(2 * ap + 1) / (2 * ap + 1)
            profile[:, col] = p_full

    for col in range(nx):
        yc = int(round(trace[col]))
        ylo = max(0, yc - ap)
        yhi = min(ny, yc + ap + 1)
        if yhi <= ylo:
            continue

        d = data_2d[ylo:yhi, col].astype(np.float64)

        # Variance
        if variance_2d is not None:
            v = variance_2d[ylo:yhi, col].astype(np.float64)
        else:
            raw_e = np.clip(d * gain, 0, None)
            v = raw_e / (gain ** 2) + (readnoise / gain) ** 2
        v = np.where(v > 0, v, 1.0)

        # Profile for this column
        p = profile[:yhi - ylo, col]
        p_sum = np.sum(p)
        if p_sum <= 0:
            p = np.ones(yhi - ylo) / (yhi - ylo)
        else:
            p = p / p_sum

        # Optimal extraction weights: w = p / v
        w = p / v
        denom = np.sum(p * w)
        if denom <= 0:
            continue
        flux_out[col] = np.sum(w * d) / denom * np.sum(p)
        var_out[col] = np.sum(p) / denom

    snr = np.where(var_out > 0, flux_out / np.sqrt(var_out), 0.0)

    return {
        "flux": flux_out,
        "variance": var_out,
        "snr": snr,
        "profile_used": profile,
    }


# -------------------------------------------------------------------
#  Continuum fitting
# -------------------------------------------------------------------

def fit_continuum(
    wavelength: np.ndarray,
    flux: np.ndarray,
    *,
    method: str = "polynomial",
    order: int = 5,
    sigma_clip: float = 3.0,
    max_iter: int = 5,
    window: int | None = None,
) -> np.ndarray:
    """Fit the continuum of a 1-D spectrum.

    Parameters
    ----------
    wavelength : np.ndarray
        Wavelength array.
    flux : np.ndarray
        Flux array.
    method : str
        ``'polynomial'`` for Legendre polynomial fit, ``'median'`` for
        running-median filter.
    order : int
        Polynomial order (used when method='polynomial').
    sigma_clip : float
        Sigma-clipping threshold for rejecting emission/absorption lines.
    max_iter : int
        Number of sigma-clipping iterations.
    window : int or None
        Window size for median filter (used when method='median').

    Returns
    -------
    np.ndarray
        Continuum estimate, same length as *flux*.

    Raises
    ------
    ValueError
        If *method* is not recognised.
    """
    wavelength = np.asarray(wavelength, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    valid = np.isfinite(flux)

    if method == "polynomial":
        mask = valid.copy()
        for _ in range(max_iter):
            w = wavelength[mask]
            f = flux[mask]
            if len(w) < order + 1:
                break
            # Normalise wavelength to [-1, 1]
            wmin, wmax = w.min(), w.max()
            wn = 2.0 * (w - wmin) / max(wmax - wmin, 1e-30) - 1.0
            coeffs = np.polynomial.legendre.legfit(wn, f, order)
            wn_all = 2.0 * (wavelength - wmin) / max(wmax - wmin, 1e-30) - 1.0
            cont = np.polynomial.legendre.legval(wn_all, coeffs)
            resid = flux - cont
            rms = float(np.std(resid[mask]))
            if rms == 0:
                break
            mask = valid & (np.abs(resid) < sigma_clip * rms)
        return cont

    elif method == "median":
        if window is None:
            window = max(5, len(flux) // 20)
        if window % 2 == 0:
            window += 1
        half = window // 2
        cont = np.empty_like(flux)
        for i in range(len(flux)):
            lo = max(0, i - half)
            hi = min(len(flux), i + half + 1)
            seg = flux[lo:hi]
            seg = seg[np.isfinite(seg)]
            cont[i] = float(np.median(seg)) if len(seg) > 0 else flux[i]
        return cont

    else:
        raise ValueError("Unknown continuum method '{}'.".format(method))


def normalize_spectrum(
    wavelength: np.ndarray,
    flux: np.ndarray,
    **continuum_kwargs: Any,
) -> dict[str, np.ndarray]:
    """Normalize a spectrum by its continuum.

    Parameters
    ----------
    wavelength : np.ndarray
        Wavelength array.
    flux : np.ndarray
        Flux array.
    **continuum_kwargs
        Passed to ``fit_continuum()``.

    Returns
    -------
    dict
        ``{'wavelength': array, 'flux_normalized': array, 'continuum': array}``
    """
    continuum = fit_continuum(wavelength, flux, **continuum_kwargs)
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = np.where(continuum != 0, flux / continuum, 0.0)
    return {
        "wavelength": wavelength,
        "flux_normalized": normalized,
        "continuum": continuum,
    }


# -------------------------------------------------------------------
#  Telluric correction
# -------------------------------------------------------------------

def model_telluric(
    wavelength: np.ndarray,
    bands: list[tuple[float, float]],
    *,
    depth: float = 0.5,
) -> np.ndarray:
    """Generate a synthetic telluric absorption model.

    Creates a simple model with Gaussian absorption bands at the
    specified wavelength ranges.  For production use, supply an
    observed telluric standard instead.

    Parameters
    ----------
    wavelength : np.ndarray
        Wavelength array.
    bands : list of (centre, width)
        Each tuple gives the central wavelength and FWHM of a telluric
        absorption band (same units as *wavelength*).
    depth : float
        Peak absorption depth (0 = no absorption, 1 = complete).

    Returns
    -------
    np.ndarray
        Telluric transmission spectrum (1.0 = no absorption).
    """
    wavelength = np.asarray(wavelength, dtype=np.float64)
    transmission = np.ones_like(wavelength)
    for centre, width in bands:
        sigma = width / 2.3548  # FWHM -> sigma
        if sigma <= 0:
            continue
        g = np.exp(-0.5 * ((wavelength - centre) / sigma) ** 2)
        transmission -= depth * g
    return np.clip(transmission, 0.0, 1.0)


def correct_telluric(
    flux: np.ndarray,
    telluric_transmission: np.ndarray,
    *,
    min_transmission: float = 0.1,
) -> np.ndarray:
    """Divide out telluric absorption from a spectrum.

    Parameters
    ----------
    flux : np.ndarray
        Observed flux.
    telluric_transmission : np.ndarray
        Telluric transmission (0--1).
    min_transmission : float
        Floor below which transmission is clamped to avoid
        amplifying noise.

    Returns
    -------
    np.ndarray
        Telluric-corrected flux.
    """
    flux = np.asarray(flux, dtype=np.float64)
    trans = np.clip(np.asarray(telluric_transmission, dtype=np.float64),
                    min_transmission, None)
    return flux / trans


# -------------------------------------------------------------------
#  Spectral resampling and stacking
# -------------------------------------------------------------------

def resample_spectrum(
    wavelength: np.ndarray,
    flux: np.ndarray,
    new_wavelength: np.ndarray,
    *,
    variance: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Resample a spectrum onto a new wavelength grid.

    Uses linear interpolation.  Pixels outside the original range are
    set to NaN.

    Parameters
    ----------
    wavelength : np.ndarray
        Original wavelength array (sorted, ascending).
    flux : np.ndarray
        Original flux.
    new_wavelength : np.ndarray
        Target wavelength grid.
    variance : np.ndarray or None
        If provided, the variance is also resampled.

    Returns
    -------
    dict
        ``{'wavelength': array, 'flux': array, 'variance': array|None}``
    """
    wavelength = np.asarray(wavelength, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    new_wavelength = np.asarray(new_wavelength, dtype=np.float64)

    new_flux = np.interp(new_wavelength, wavelength, flux,
                         left=np.nan, right=np.nan)
    new_var = None
    if variance is not None:
        variance = np.asarray(variance, dtype=np.float64)
        new_var = np.interp(new_wavelength, wavelength, variance,
                            left=np.nan, right=np.nan)

    return {
        "wavelength": new_wavelength,
        "flux": new_flux,
        "variance": new_var,
    }


def stack_spectra(
    wavelengths: list[np.ndarray],
    fluxes: list[np.ndarray],
    *,
    common_grid: np.ndarray | None = None,
    method: str = "median",
    variances: list[np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Co-add multiple spectra.

    All spectra are resampled to a common wavelength grid, then combined.

    Parameters
    ----------
    wavelengths : list of np.ndarray
        Wavelength arrays for each spectrum.
    fluxes : list of np.ndarray
        Flux arrays.
    common_grid : np.ndarray or None
        If None, the grid of the first spectrum is used.
    method : str
        ``'median'``, ``'mean'``, or ``'weighted_mean'`` (requires
        *variances*).
    variances : list of np.ndarray or None
        Per-spectrum variance arrays (required for ``'weighted_mean'``).

    Returns
    -------
    dict
        ``{'wavelength': array, 'flux': array, 'variance': array|None,
           'n_combined': int}``
    """
    if not wavelengths or not fluxes:
        raise ValueError("Need at least one spectrum.")
    if len(wavelengths) != len(fluxes):
        raise ValueError("wavelengths and fluxes must have the same length.")

    if common_grid is None:
        common_grid = np.asarray(wavelengths[0], dtype=np.float64)
    else:
        common_grid = np.asarray(common_grid, dtype=np.float64)

    n_grid = len(common_grid)
    n_spec = len(fluxes)
    stack = np.full((n_spec, n_grid), np.nan, dtype=np.float64)
    var_stack = None
    if variances is not None:
        var_stack = np.full((n_spec, n_grid), np.nan, dtype=np.float64)

    for i in range(n_spec):
        r = resample_spectrum(wavelengths[i], fluxes[i], common_grid)
        stack[i] = r["flux"]
        if var_stack is not None and variances is not None:
            r2 = resample_spectrum(wavelengths[i], variances[i], common_grid)
            var_stack[i] = r2["flux"]

    if method == "median":
        combined = np.nanmedian(stack, axis=0)
    elif method == "mean":
        combined = np.nanmean(stack, axis=0)
    elif method == "weighted_mean":
        if var_stack is None:
            raise ValueError("weighted_mean requires variances.")
        with np.errstate(divide="ignore", invalid="ignore"):
            weights = np.where(var_stack > 0, 1.0 / var_stack, 0.0)
        w_sum = np.nansum(weights, axis=0)
        combined = np.where(
            w_sum > 0,
            np.nansum(weights * stack, axis=0) / w_sum,
            np.nan,
        )
    else:
        raise ValueError("Unknown stacking method '{}'.".format(method))

    return {
        "wavelength": common_grid,
        "flux": combined,
        "variance": None,
        "n_combined": n_spec,
    }


# -------------------------------------------------------------------
#  Redshift measurement
# -------------------------------------------------------------------

def measure_redshift(
    wavelength: np.ndarray,
    flux: np.ndarray,
    template_wavelength: np.ndarray,
    template_flux: np.ndarray,
    *,
    z_min: float = 0.0,
    z_max: float = 1.0,
    z_step: float = 0.0001,
) -> dict[str, Any]:
    """Measure redshift by cross-correlating with a template.

    Parameters
    ----------
    wavelength : np.ndarray
        Observed wavelength.
    flux : np.ndarray
        Observed flux (continuum-normalized recommended).
    template_wavelength : np.ndarray
        Rest-frame template wavelength.
    template_flux : np.ndarray
        Rest-frame template flux.
    z_min, z_max : float
        Redshift search range.
    z_step : float
        Step size in redshift space.

    Returns
    -------
    dict
        ``{'z_best': float, 'z_err': float, 'cc_peak': float,
           'z_grid': np.ndarray, 'cc_values': np.ndarray}``
    """
    wavelength = np.asarray(wavelength, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    template_wavelength = np.asarray(template_wavelength, dtype=np.float64)
    template_flux = np.asarray(template_flux, dtype=np.float64)

    z_grid = np.arange(z_min, z_max + z_step, z_step)
    cc = np.zeros_like(z_grid)

    # Normalise observed
    f_obs = flux - np.nanmean(flux)
    obs_norm = np.sqrt(np.nansum(f_obs ** 2))
    if obs_norm == 0:
        obs_norm = 1.0

    for i, z in enumerate(z_grid):
        shifted_wave = template_wavelength * (1 + z)
        t_flux = np.interp(wavelength, shifted_wave, template_flux,
                           left=np.nan, right=np.nan)
        valid = np.isfinite(t_flux) & np.isfinite(f_obs)
        if np.sum(valid) < 10:
            cc[i] = 0.0
            continue
        t = t_flux[valid] - np.mean(t_flux[valid])
        o = f_obs[valid]
        t_norm = np.sqrt(np.sum(t ** 2))
        if t_norm == 0:
            continue
        cc[i] = np.sum(o * t) / (obs_norm * t_norm)

    i_best = int(np.argmax(cc))
    z_best = float(z_grid[i_best])
    cc_peak = float(cc[i_best])

    # Rough error from CC width at half-max
    half_max = cc_peak / 2.0
    above = cc >= half_max
    if np.sum(above) > 1:
        z_err = float(z_grid[above][-1] - z_grid[above][0]) / 2.0
    else:
        z_err = z_step

    return {
        "z_best": z_best,
        "z_err": z_err,
        "cc_peak": cc_peak,
        "z_grid": z_grid,
        "cc_values": cc,
    }


# -------------------------------------------------------------------
#  Signal-to-noise estimation
# -------------------------------------------------------------------

def estimate_snr(
    flux: np.ndarray,
    variance: np.ndarray | None = None,
    *,
    method: str = "der_snr",
) -> dict[str, float]:
    """Estimate the signal-to-noise ratio of a 1-D spectrum.

    Parameters
    ----------
    flux : np.ndarray
        Flux array.
    variance : np.ndarray or None
        Variance array (used when method='variance').
    method : str
        ``'der_snr'`` -- derivative-based estimator (Stoehr+ 2008);
        ``'variance'`` -- from variance array.

    Returns
    -------
    dict
        ``{'snr_median': float, 'snr_per_pixel': np.ndarray}``
    """
    flux = np.asarray(flux, dtype=np.float64)

    if method == "der_snr":
        # DER_SNR: S/N = median(flux) / (1.4826 * median(|f_i - f_{i+2}|) / sqrt(6))
        if len(flux) < 5:
            return {"snr_median": 0.0, "snr_per_pixel": np.zeros_like(flux)}
        signal = np.abs(np.median(flux))
        noise = 1.4826 * np.median(np.abs(flux[2:] - flux[:-2])) / math.sqrt(6.0)
        if noise == 0:
            noise = 1e-30
        snr_med = signal / noise
        # Per-pixel approximation
        local_noise = np.zeros_like(flux)
        for i in range(len(flux)):
            lo = max(0, i - 2)
            hi = min(len(flux), i + 3)
            seg = flux[lo:hi]
            local_noise[i] = 1.4826 * np.median(np.abs(np.diff(seg))) if len(seg) > 1 else noise
        local_noise = np.where(local_noise > 0, local_noise, noise)
        snr_pp = np.abs(flux) / local_noise

    elif method == "variance":
        if variance is None:
            raise ValueError("variance array is required for method='variance'.")
        variance = np.asarray(variance, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            snr_pp = np.where(variance > 0, np.abs(flux) / np.sqrt(variance), 0.0)
        snr_med = float(np.median(snr_pp[snr_pp > 0])) if np.any(snr_pp > 0) else 0.0

    else:
        raise ValueError("Unknown SNR method '{}'.".format(method))

    return {
        "snr_median": float(snr_med),
        "snr_per_pixel": snr_pp,
    }


# -------------------------------------------------------------------
#  Equivalent width and line flux (with errors)
# -------------------------------------------------------------------

def equivalent_width(
    wavelength: np.ndarray,
    flux: np.ndarray,
    continuum: np.ndarray,
    *,
    line_range: tuple[float, float],
    variance: np.ndarray | None = None,
) -> dict[str, float]:
    """Measure the equivalent width of a spectral line.

    Parameters
    ----------
    wavelength : np.ndarray
        Wavelength array.
    flux : np.ndarray
        Observed flux.
    continuum : np.ndarray
        Continuum level.
    line_range : tuple
        ``(w_min, w_max)`` -- wavelength range of the line.
    variance : np.ndarray or None
        Variance array for error estimation.

    Returns
    -------
    dict
        ``{'ew': float, 'ew_err': float, 'line_flux': float,
           'line_flux_err': float}``
    """
    wavelength = np.asarray(wavelength, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    continuum = np.asarray(continuum, dtype=np.float64)

    mask = (wavelength >= line_range[0]) & (wavelength <= line_range[1])
    if np.sum(mask) < 2:
        return {"ew": 0.0, "ew_err": float("nan"),
                "line_flux": 0.0, "line_flux_err": float("nan")}

    w = wavelength[mask]
    f = flux[mask]
    c = continuum[mask]

    # EW = integral of (1 - f/c) dw
    with np.errstate(divide="ignore", invalid="ignore"):
        integrand = np.where(c > 0, 1.0 - f / c, 0.0)
    dw = np.diff(w)
    ew = float(np.sum(0.5 * (integrand[:-1] + integrand[1:]) * dw))

    # Line flux = integral of (c - f) dw
    line_integrand = c - f
    line_flux = float(np.sum(0.5 * (line_integrand[:-1] + line_integrand[1:]) * dw))

    # Error
    ew_err = float("nan")
    line_flux_err = float("nan")
    if variance is not None:
        var = np.asarray(variance, dtype=np.float64)[mask]
        pixel_width = float(np.median(dw))
        noise_sum = float(np.sum(var))
        if noise_sum > 0:
            line_flux_err = math.sqrt(noise_sum) * pixel_width
            c_med = float(np.median(c))
            if c_med > 0:
                ew_err = line_flux_err / c_med

    return {
        "ew": ew,
        "ew_err": ew_err,
        "line_flux": line_flux,
        "line_flux_err": line_flux_err,
    }


# -------------------------------------------------------------------
#  Spectral smoothing
# -------------------------------------------------------------------

def smooth_spectrum(
    flux: np.ndarray,
    kernel_size: int = 5,
    *,
    method: str = "boxcar",
) -> np.ndarray:
    """Smooth a 1-D spectrum.

    Parameters
    ----------
    flux : np.ndarray
        Input flux.
    kernel_size : int
        Smoothing kernel width (must be odd).
    method : str
        ``'boxcar'`` for running mean, ``'gaussian'`` for Gaussian
        kernel (sigma = kernel_size / 4).

    Returns
    -------
    np.ndarray
        Smoothed flux.
    """
    flux = np.asarray(flux, dtype=np.float64)
    if kernel_size < 1:
        raise ValueError("kernel_size must be >= 1.")
    if kernel_size % 2 == 0:
        kernel_size += 1

    if method == "boxcar":
        kernel = np.ones(kernel_size) / kernel_size
    elif method == "gaussian":
        sigma = kernel_size / 4.0
        x = np.arange(kernel_size) - kernel_size // 2
        kernel = np.exp(-0.5 * (x / sigma) ** 2)
        kernel /= kernel.sum()
    else:
        raise ValueError("Unknown smoothing method '{}'.".format(method))

    return np.convolve(flux, kernel, mode="same")
