
from __future__ import annotations
import numpy as np
import pandas as pd

def mixing_length_linear(y, kappa=0.41, y0=0.0):
    y = np.asarray(y)
    return kappa*np.maximum(0.0, y - y0)

def van_driest_damping(y, u_tau=0.05, nu=1.5e-5, A_plus=26.0, kappa=0.41):
    y = np.asarray(y)
    y_plus = y*u_tau/nu
    return kappa*y*(1.0 - np.exp(-y_plus/ A_plus))

def eddy_viscosity(lm, dUdy):
    lm = np.asarray(lm); dUdy = np.asarray(dUdy)
    return (lm**2)*np.abs(dUdy)

def eddy_diffusivity(nu_t, Pr_t=0.9):
    return nu_t/Pr_t

def shear_mixing_from_profile(y, lm, dUdy, Pr_t=0.9, dTdy=None):
    y    = np.asarray(y); lm = np.asarray(lm); dUdy = np.asarray(dUdy)
    nu_t = eddy_viscosity(lm, dUdy)
    kappa_t = eddy_diffusivity(nu_t, Pr_t)
    out = {
        "y": y,
        "lm_m": lm,
        "S_1_per_s": dUdy,
        "nu_t_m2_per_s": nu_t,
        "kappa_t_m2_per_s": kappa_t,
        "Pr_t": np.full_like(y, Pr_t, dtype=float),
    }
    if dTdy is not None:
        dTdy = np.asarray(dTdy)
        q_y = -kappa_t*dTdy
        P_theta = 2.0*kappa_t*(dTdy**2)
        out.update({"dTdy_K_per_m": dTdy, "q_y_K_m_per_s": q_y, "P_theta_K2_per_s": P_theta})
    import pandas as pd
    return pd.DataFrame(out)

def load_csv_if_exists(path):
    import pandas as pd
    try:
        return pd.read_csv(path)
    except Exception as e:
        raise FileNotFoundError(f"Could not load {path}: {e}")
