
"""
ch_scalar_mixing: utilities for Chapter 3 — Scalar Mixing
Minimal dependencies: numpy (pandas/matplotlib optional for user code).
"""

from __future__ import annotations
import numpy as np

# ---------------------- Data Generation ----------------------
def generate_synthetic_channel(N: int = 3000, gradient: float = 1.0, seed: int = 7):
    """
    Build a quasi-1D channel dataset with a passive scalar c having mean gradient d<c>/dy = gradient.
    Returns dict with arrays: y, u, v, c, rho, D, Kt_true, target_flux.
    """
    rng = np.random.default_rng(seed)
    y = np.linspace(0.0, 1.0, N)

    # Mean fields
    rho = 1.2 - 0.1*y
    u_mean = 10.0 * (1.0 - (y - 0.5)**2)
    v_mean = np.zeros_like(y)

    # Scalar mean
    c_mean = 0.2 + gradient * y

    # Fluctuation amplitudes
    urms = 0.8 + 0.3*y
    vrms = 0.4*np.ones_like(y)

    # Velocity with fluctuations
    u = u_mean + urms * rng.standard_normal(N)
    v = v_mean + vrms * rng.standard_normal(N)

    # "True" eddy diffusivity profile (toy) and desired flux
    Kt_true = 0.05 + 0.15*(1.0 - (y - 0.5)**2)
    target_flux = -Kt_true * gradient

    # Construct scalar fluctuation correlated with v' to achieve target <v'c'>
    v_p = v - np.mean(v)
    alpha = target_flux / np.maximum(vrms**2, 1e-10)  # so <v' (alpha v')> ~ target_flux
    noise = 0.2 * rng.standard_normal(N)
    c = c_mean + alpha * v_p + noise

    # Molecular diffusivity
    D = 1e-3

    return dict(y=y, u=u, v=v, c=c, rho=rho, D=D, Kt_true=Kt_true, target_flux=target_flux)

# ---------------------- Binning Utilities ----------------------
def bin_mean(x, y, bins):
    """
    Bin-average x over y-bins defined by 'bins' edges.
    Returns an array of length len(bins)-1 with NaN for empty bins.
    Safe to use with values equal to the rightmost bin edge.
    """
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    bins = np.asarray(bins)
    if x.shape != y.shape:
        raise ValueError(
            f"bin_mean: x and y lengths differ: len(x)={len(x)}, len(y)={len(y)}. "
            "Use the same y that was used to generate x, or rebuild y to match x."
        )
    # Safe bin indices: map y to [0, num_bins-1]
    idx = np.searchsorted(bins, y, side='right') - 1
    idx = np.clip(idx, 0, len(bins) - 2)

    out = np.full(len(bins) - 1, np.nan)
    for k in range(len(out)):
        sel = (idx == k)
        if np.any(sel):
            out[k] = np.mean(x[sel])
    return out

def bin_grad(f_binned, centers):
    """Gradient of a binned quantity using numpy.gradient over bin centers."""
    return np.gradient(np.asarray(f_binned), np.asarray(centers))

# ---------------------- Fluxes & Closures ----------------------
def scalar_flux_bin(v, c, y, bins):
    """
    Compute bin-mean <v'c'> using fluctuations relative to bin means.
    Returns an array of length len(bins)-1.
    """
    v = np.asarray(v).reshape(-1)
    c = np.asarray(c).reshape(-1)
    y = np.asarray(y).reshape(-1)
    bins = np.asarray(bins)
    if not (len(v) == len(c) == len(y)):
        raise ValueError("scalar_flux_bin: v, c, y lengths must match.")
    idx = np.searchsorted(bins, y, side='right') - 1
    idx = np.clip(idx, 0, len(bins) - 2)
    v_b = bin_mean(v, y, bins)
    c_b = bin_mean(c, y, bins)
    v_p = v - v_b[idx]
    c_p = c - c_b[idx]
    return bin_mean(v_p * c_p, y, bins)

def estimate_Kt(vc_b, dc_dy_b):
    """Eddy diffusivity estimate Kt ≈ -<v'c'> / (d<c>/dy)."""
    vc_b = np.asarray(vc_b)
    dc_dy_b = np.asarray(dc_dy_b)
    return -vc_b / np.maximum(dc_dy_b, 1e-10)

def chi_proxy_bin(c, y, D, bins):
    """
    Scalar dissipation proxy chi = 2 D < (dc'/dy)^2 >_bin.
    Uses raw-data gradient of c' = c - <c>_global.
    """
    c = np.asarray(c).reshape(-1)
    y = np.asarray(y).reshape(-1)
    if c.shape != y.shape:
        raise ValueError("chi_proxy_bin: c and y must have the same length.")
    c_p = c - np.mean(c)
    dcdy = np.gradient(c_p, y)
    chi_local = 2.0 * D * (dcdy**2)
    return bin_mean(chi_local, y, bins)

# ---------------------- Helpers ----------------------
def coerce_y_length(y, x):
    """
    Return y' with length == len(x). If lengths differ, rebuild y linearly
    over [min(y), max(y)] (or [0,1] if y is empty).
    """
    y = np.asarray(y).reshape(-1)
    x = np.asarray(x).reshape(-1)
    if len(y) == len(x):
        return y
    y0, y1 = (float(y.min()), float(y.max())) if y.size else (0.0, 1.0)
    return np.linspace(y0, y1, len(x))

# ---------------------- Checks ----------------------
def quick_checks(c, v):
    """Basic sanity checks: finiteness & non-negative variance."""
    c = np.asarray(c).reshape(-1)
    v = np.asarray(v).reshape(-1)
    assert np.all(np.isfinite(c)), "c contains non-finite values"
    assert np.all(np.isfinite(v)), "v contains non-finite values"
    assert np.var(c - np.mean(c)) >= 0.0, "variance must be non-negative"
    return True
