
"""
scalar_mixing_utils.py
----------------------
Utility functions for the Scalar Mixing companion notebook.
Place this file in the same folder as the notebook and import as:

    import scalar_mixing_utils as smu

Dependencies: numpy (required), matplotlib (optional for plotting)
"""

from __future__ import annotations
import numpy as np

try:
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except Exception:
    _HAVE_MPL = False


# -----------------------------------------------------------------------------
# Core synthetic dataset
# -----------------------------------------------------------------------------

def generate_synthetic_dataset(
    N: int = 3000,
    y_min: float = 0.0,
    y_max: float = 1.0,
    rho0: float = 1.2,
    rho1: float = -0.1,
    T0: float = 300.0,
    dT: float = 40.0,
    c0: float = 0.2,
    dc: float = 1.0,
    v_rms: float = 0.2,
    c_rms: float = 0.05,
    corr: float = -0.6,
    D: float = 1.0e-3,
    seed: int | None = 42
) -> dict:
    """
    Generate a 1D channel-like synthetic dataset for scalar mixing.

    Returns
    -------
    data : dict
        Dictionary containing arrays:
        - y, dy
        - rho_bar, T_bar, c_bar
        - vprime, cprime
        - grad_c_bar
        - vc_flux = <v' c'>
        - c_var = <c'^2>
        - Kt (eddy diffusivity from down-gradient closure)
        - P_c (variance production proxy)
        - chi (dissipation-like proxy based on grad of fluctuations)
    """
    rng = np.random.default_rng(seed)

    # Grid
    y = np.linspace(y_min, y_max, N)
    dy = (y_max - y_min) / (N - 1)

    # Mean profiles
    rho_bar = rho0 + rho1 * y
    T_bar = T0 + dT * y
    c_bar = c0 + dc * y

    # Fluctuations with prescribed correlation
    z1 = rng.standard_normal(N)
    z2 = rng.standard_normal(N)
    vprime = v_rms * z1
    cprime = c_rms * (corr * z1 + np.sqrt(max(1.0 - corr**2, 0.0)) * z2)

    # Turbulent statistics
    grad_c_bar = np.gradient(c_bar, y, edge_order=2)
    vc_flux = np.mean(vprime * cprime) * np.ones_like(y)
    c_var = np.mean(cprime**2) * np.ones_like(y)

    # Eddy diffusivity via down-gradient closure
    eps = 1e-12
    denom = np.where(np.abs(grad_c_bar) < eps, np.sign(grad_c_bar) * eps + (grad_c_bar == 0)*eps, grad_c_bar)
    Kt = -vc_flux / denom

    # Dissipation-like proxy chi = 2D <|grad c'|^2>
    grad_cprime = np.gradient(cprime, y, edge_order=2)
    chi = 2.0 * D * np.mean(grad_cprime**2) * np.ones_like(y)

    # Variance production P_c = - <v'c'> d cbar/dy
    P_c = -vc_flux * grad_c_bar

    data = dict(
        y=y, dy=dy,
        rho_bar=rho_bar,
        T_bar=T_bar,
        c_bar=c_bar,
        vprime=vprime,
        cprime=cprime,
        grad_c_bar=grad_c_bar,
        vc_flux=vc_flux,
        c_var=c_var,
        Kt=Kt,
        P_c=P_c,
        chi=chi,
        D=D,
        meta=dict(
            N=N, rho0=rho0, rho1=rho1, T0=T0, dT=dT, c0=c0, dc=dc,
            v_rms=v_rms, c_rms=c_rms, corr=corr, seed=seed
        )
    )
    return data


# -----------------------------------------------------------------------------
# Eddy viscosity / diffusivity utilities
# -----------------------------------------------------------------------------

def van_driest_mixing_length(y_plus: np.ndarray, kappa: float = 0.41, A_plus: float = 26.0) -> np.ndarray:
    """Van Driest-damped mixing length: l_m = kappa*y^+ [1 - exp(-y^+ / A^+)]"""
    y_plus = np.asarray(y_plus, dtype=float)
    return kappa * y_plus * (1.0 - np.exp(-y_plus / A_plus))


def eddy_viscosity_from_mixing_length(l_m: np.ndarray, dUdy: np.ndarray) -> np.ndarray:
    """Eddy viscosity nu_t = l_m^2 |dU/dy|"""
    l_m = np.asarray(l_m, dtype=float)
    dUdy = np.asarray(dUdy, dtype=float)
    return (l_m**2) * np.abs(dUdy)


def eddy_diffusivity_from_nu_t(nu_t: np.ndarray, Sc_t: float | np.ndarray = 0.7) -> np.ndarray:
    """Eddy diffusivity K_t = nu_t / Sc_t"""
    nu_t = np.asarray(nu_t, dtype=float)
    return nu_t / Sc_t


def turbulent_schmidt_number(nu_t: np.ndarray, K_t: np.ndarray) -> np.ndarray:
    """Turbulent Schmidt number Sc_t = nu_t / K_t"""
    nu_t = np.asarray(nu_t, dtype=float)
    K_t = np.asarray(K_t, dtype=float)
    eps = 1e-12
    return nu_t / np.where(np.abs(K_t) < eps, np.sign(K_t)*eps + (K_t==0)*eps, K_t)


# -----------------------------------------------------------------------------
# Diffusion solver (1D, explicit)
# -----------------------------------------------------------------------------

def diffuse_1d_explicit(c: np.ndarray, D: float, dy: float, dt: float, steps: int,
                        bc: str = "neumann") -> np.ndarray:
    """
    Evolve the 1D diffusion equation c_t = D c_yy with explicit scheme.
    """
    c = np.asarray(c, dtype=float)
    c_new = c.copy()
    N = c_new.size
    r = D * dt / (dy**2)
    if r > 0.5 + 1e-12:
        raise ValueError(f"Explicit diffusion unstable: r=D*dt/dy^2={r:.3e} > 0.5")

    for _ in range(steps):
        c_old = c_new.copy()
        # interior
        c_new[1:-1] = c_old[1:-1] + r*(c_old[2:] - 2*c_old[1:-1] + c_old[:-2])

        if bc == "neumann":
            # zero-gradient: replicate boundary-adjacent values
            c_new[0] = c_new[1]
            c_new[-1] = c_new[-2]
        elif bc == "dirichlet0":
            c_new[0] = 0.0
            c_new[-1] = 0.0
        else:
            raise ValueError(f"Unknown bc='{bc}'")
    return c_new


# -----------------------------------------------------------------------------
# Time scales
# -----------------------------------------------------------------------------

def scalar_mixing_timescale(c_var: np.ndarray, chi: np.ndarray) -> np.ndarray:
    """Scalar time scale tau_c = <c'^2> / chi."""
    c_var = np.asarray(c_var, dtype=float)
    chi = np.asarray(chi, dtype=float)
    eps = 1e-20
    return c_var / np.where(np.abs(chi) < eps, np.sign(chi)*eps + (chi==0)*eps, chi)


def eddy_turnover_timescale(K_t: np.ndarray, u_rms: float | np.ndarray, l_char: float | np.ndarray) -> np.ndarray:
    """
    Heuristic eddy time scale using K_t ~ u' * l  => tau ~ l / u'.
    """
    u_rms = np.asarray(u_rms, dtype=float)
    l_char = np.asarray(l_char, dtype=float)
    eps = 1e-20
    return l_char / np.where(np.abs(u_rms) < eps, eps, u_rms)


# -----------------------------------------------------------------------------
# Optional plotting helpers
# -----------------------------------------------------------------------------

def plot_profile(y: np.ndarray, q: np.ndarray, title: str, xlabel: str, ylabel: str = "") -> None:
    """Convenience line plot for a wall-normal profile (requires matplotlib)."""
    if not _HAVE_MPL:
        return
    import matplotlib.pyplot as _plt
    _plt.figure()
    _plt.plot(y, q)
    _plt.title(title)
    _plt.xlabel(xlabel)
    _plt.ylabel(ylabel)
    _plt.tight_layout()
    _plt.show()


def demo_van_driest_profiles(
    y: np.ndarray,
    y_plus: np.ndarray,
    dUdy: np.ndarray,
    kappa: float = 0.41,
    A_plus: float = 26.0,
    Sc_t: float = 0.7,
    make_plots: bool = True
) -> dict:
    """
    Generate example nu_t and K_t profiles from Van Driest-damped mixing length
    given a user-specified mean shear dU/dy and wall coordinate y_plus.
    Returns a dict with keys: l_m, nu_t, K_t
    """
    l_m = van_driest_mixing_length(y_plus, kappa=kappa, A_plus=A_plus)
    nu_t = eddy_viscosity_from_mixing_length(l_m, dUdy)
    K_t = eddy_diffusivity_from_nu_t(nu_t, Sc_t=Sc_t)

    if make_plots and _HAVE_MPL:
        import matplotlib.pyplot as _plt
        _plt.figure()
        _plt.semilogy(y, np.maximum(nu_t, 1e-16), label="nu_t (VD)")
        _plt.semilogy(y, np.maximum(K_t, 1e-16), label="K_t (VD)")
        _plt.xlabel("y")
        _plt.ylabel("eddy coeff.")
        _plt.title("Van Driest Eddy Viscosity/Diffusivity")
        _plt.legend()
        _plt.tight_layout()
        _plt.show()

    return dict(l_m=l_m, nu_t=nu_t, K_t=K_t)
