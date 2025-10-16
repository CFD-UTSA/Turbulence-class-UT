
"""
Chapter 4 — Buoyancy-Generated Turbulence: Reusable Examples
------------------------------------------------------------

This module provides compact utilities to explore:
1) Boussinesq buoyancy production in TKE,
2) Mixing-length models for turbulent viscosity and thermal diffusivity,
3) 1D steady vertical heat transport with prescribed shear and stratification,
4) Simple synthetic profiles for unstable/neutral/stable cases.
"""

from dataclasses import dataclass
import numpy as np

g = 9.81  # m/s^2

@dataclass
class StratProfile:
    z: np.ndarray         # height [m]
    U: np.ndarray         # mean streamwise velocity [m/s]
    Theta: np.ndarray     # potential temperature [K]
    theta0: float         # reference potential temperature [K]

def d_dz(y, x):
    """Centered finite difference with one-sided at ends; returns dy/dx on x-grid."""
    y = np.asarray(y); x = np.asarray(x)
    out = np.empty_like(y, dtype=float)
    dx = np.diff(x)
    # interior
    out[1:-1] = (y[2:] - y[:-2]) / (x[2:] - x[:-2])
    # ends (1st order)
    out[0] = (y[1] - y[0]) / dx[0]
    out[-1] = (y[-1] - y[-2]) / dx[-1]
    return out

def mixing_length_prandtl(dU_dz, kappa=0.41, z=None, z0=1e-3, l_max=None):
    """
    Prandtl mixing length: l_m = kappa * (z - z0), optionally limited by l_max (Blackadar-type cap).
    Returns l_m array shaped like dU_dz.
    """
    if z is None:
        raise ValueError("Provide z for mixing_length_prandtl.")
    z_eff = np.maximum(z - z0, 1e-6)
    l_m = kappa * z_eff
    if l_max is not None:
        # Blackadar cap: 1/l = 1/(k z) + 1/l_max  -> l = (k z l_max)/(k z + l_max)
        l_m = (kappa * z_eff * l_max)/(kappa * z_eff + l_max)
    return l_m

def eddy_viscosity_nut(dU_dz, l_m):
    """ν_t = l_m^2 * |dU/dz|"""
    return (l_m**2) * np.abs(dU_dz)

def eddy_diffusivity_Kh(nu_t, Pr_t=0.85):
    """Thermal eddy diffusivity via turbulent Prandtl number: K_h = ν_t / Pr_t"""
    return nu_t / Pr_t

def buoyancy_production_TKE(profile: StratProfile, alpha_t=None, K_h=None):
    """
    Buoyancy term in TKE budget (Boussinesq, dry):
    G_b = (g/θ0) * (overline{w'θ'}) = -(g/θ0) * K_h * dΘ/dz
    Requires either alpha_t (same as K_h) or K_h array.
    """
    dTheta_dz = d_dz(profile.Theta, profile.z)
    if K_h is None and alpha_t is None:
        raise ValueError("Provide alpha_t (K_h) or K_h.")
    Kh = K_h if K_h is not None else alpha_t
    return -(g/profile.theta0) * Kh * dTheta_dz  # W/kg

def make_synthetic_profile(case="unstable", H=200.0, Nz=200, U0=0.0, dU=10.0, theta0=300.0, dtheta=3.0):
    """
    Build simple vertical profiles:
    - unstable: Theta decreases with z (negative gradient)
    - neutral:  Theta constant
    - stable:   Theta increases with z (positive gradient)
    Velocity increases linearly with z to create shear.
    """
    z = np.linspace(0.0, H, Nz)
    U = U0 + dU * (z/H)
    if case == "unstable":
        Theta = theta0 - dtheta * (z/H)
    elif case == "neutral":
        Theta = theta0 + 0.0*z
    elif case == "stable":
        Theta = theta0 + dtheta * (z/H)
    else:
        raise ValueError("case must be 'unstable','neutral','stable'")
    return StratProfile(z=z, U=U, Theta=Theta, theta0=theta0)

def oneD_steady_heat_flux(profile: StratProfile, Pr_t=0.85, kappa=0.41, z0=0.1, l_max=None):
    """
    Given U(z), Theta(z), estimate turbulent flux and TKE buoyancy production:
    1) compute dU/dz, mixing length l_m
    2) ν_t = l_m^2 |dU/dz|
    3) K_h = ν_t / Pr_t
    4) q_z = -K_h * dΘ/dz
    5) G_b = (g/θ0) * (overline{w'θ'}) = (g/θ0) * q_z
    Returns dict of arrays.
    """
    dU_dz = d_dz(profile.U, profile.z)
    l_m = mixing_length_prandtl(dU_dz, kappa=kappa, z=profile.z, z0=z0, l_max=l_max)
    nu_t = eddy_viscosity_nut(dU_dz, l_m)
    Kh = eddy_diffusivity_Kh(nu_t, Pr_t=Pr_t)
    dTheta_dz = d_dz(profile.Theta, profile.z)
    qz = -Kh * dTheta_dz
    Gb = (g/profile.theta0) * qz
    return dict(z=profile.z, dU_dz=dU_dz, l_m=l_m, nu_t=nu_t, Kh=Kh, dTheta_dz=dTheta_dz, qz=qz, Gb=Gb)

