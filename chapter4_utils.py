
"""
chapter4_utils.py
Utilities for Chapter 4 companion notebooks (Turbulence Governing Equations).
"""
from __future__ import annotations
import numpy as np

try:
    import pandas as pd
    HAVE_PANDAS = True
except Exception:
    HAVE_PANDAS = False

try:
    import matplotlib.pyplot as plt  # noqa: F401
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False


def _safe_load_csv(path: str):
    """Load CSV into pandas if available, else numpy structured array."""
    if HAVE_PANDAS:
        df = pd.read_csv(path)
        return df, True
    else:
        arr = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
        return arr, False


def load_dataset_csv(path: str):
    """Uniform loader returning dict with data, type flag, and column names."""
    data, isp = _safe_load_csv(path)
    if isp:
        cols = list(data.columns)
    else:
        cols = list(data.dtype.names) if hasattr(data, 'dtype') and data.dtype.names else []
    return dict(data=data, is_pandas=isp, columns=cols)


def infer_columns(columns):
    """Infer common column roles from names; returns a mapping dict."""
    lower = [c.lower() for c in columns]
    name_map = {}

    # coordinates
    for cand in ['x','x_coord','xc']:
        if cand in lower:
            name_map['x'] = columns[lower.index(cand)]; break
    for cand in ['y','y_coord','yc']:
        if cand in lower:
            name_map['y'] = columns[lower.index(cand)]; break
    for cand in ['z','z_coord','zc']:
        if cand in lower:
            name_map['z'] = columns[lower.index(cand)]; break

    # velocities
    for cand in ['u','u_vel','ux']:
        if cand in lower:
            name_map['u'] = columns[lower.index(cand)]; break
    for cand in ['v','v_vel','uy']:
        if cand in lower:
            name_map['v'] = columns[lower.index(cand)]; break
    for cand in ['w','w_vel','uz']:
        if cand in lower:
            name_map['w'] = columns[lower.index(cand)]; break

    # thermo/scalars
    for cand in ['t','temp','temperature']:
        if cand in lower:
            name_map['T'] = columns[lower.index(cand)]; break
    for cand in ['rho','density']:
        if cand in lower:
            name_map['rho'] = columns[lower.index(cand)]; break
    for cand in ['nu','visc','kinematic_viscosity']:
        if cand in lower:
            name_map['nu'] = columns[lower.index(cand)]; break

    # gravity (optional)
    for cand in ['gx','g_x']:
        if cand in lower:
            name_map['gx'] = columns[lower.index(cand)]; break
    for cand in ['gy','g_y']:
        if cand in lower:
            name_map['gy'] = columns[lower.index(cand)]; break
    for cand in ['gz','g_z']:
        if cand in lower:
            name_map['gz'] = columns[lower.index(cand)]; break

    return name_map


def _col(dataset, name, default=None):
    """Extract a column by name from pandas DF or numpy structured array safely."""
    if name is None:
        return default
    if dataset['is_pandas']:
        df = dataset['data']
        if name in df.columns:
            return df[name].to_numpy()
        return default
    else:
        arr = dataset['data']
        if hasattr(arr, 'dtype') and arr.dtype.names and name in arr.dtype.names:
            return arr[name]
        return default


def reynolds_decomposition(u):
    """Return mean and fluctuation: u = ū + u'."""
    u = np.asarray(u, dtype=float)
    um = np.nanmean(u)
    up = u - um
    return um, up


def reynolds_stresses(u, v=None, w=None):
    """Compute Reynolds stresses; gracefully handles missing components."""
    out = {}
    if u is not None:
        um, up = reynolds_decomposition(u); out['uu'] = float(np.nanmean(up*up))
    if v is not None:
        vm, vp = reynolds_decomposition(v); out['vv'] = float(np.nanmean(vp*vp))
    if w is not None:
        wm, wp = reynolds_decomposition(w); out['ww'] = float(np.nanmean(wp*wp))
    if u is not None and v is not None:
        out['uv'] = float(np.nanmean((u-np.nanmean(u))*(v-np.nanmean(v))))
    if u is not None and w is not None:
        out['uw'] = float(np.nanmean((u-np.nanmean(u))*(w-np.nanmean(w))))
    if v is not None and w is not None:
        out['vw'] = float(np.nanmean((v-np.nanmean(v))*(w-np.nanmean(w))))
    return out


def tke_from_components(u, v=None, w=None):
    """Turbulent kinetic energy k = 0.5*(<u'^2>+<v'^2>+<w'^2>)."""
    rs = reynolds_stresses(u, v, w)
    uu = rs.get('uu', 0.0); vv = rs.get('vv', 0.0); ww = rs.get('ww', 0.0)
    return 0.5*(uu + vv + ww)


def finite_difference_grad_1d(q, coord):
    """1D gradient dq/dcoord (central differences via numpy.gradient)."""
    q = np.asarray(q, dtype=float)
    coord = np.asarray(coord, dtype=float)
    return np.gradient(q, coord, edge_order=2)


def mean_vorticity_estimate(dataset, name_map):
    """Heuristic vorticity magnitude estimate given available columns."""
    u = _col(dataset, name_map.get('u'))
    v = _col(dataset, name_map.get('v'))
    w = _col(dataset, name_map.get('w'))
    x = _col(dataset, name_map.get('x'))
    y = _col(dataset, name_map.get('y'))
    z = _col(dataset, name_map.get('z'))

    if u is None or v is None:
        return None, "Insufficient velocity components for vorticity."

    # If only a 1D profile in y is available, approximate |ω| ~ |du/dy|
    if y is not None and (x is None and z is None):
        try:
            dudy = finite_difference_grad_1d(u, y)
            omega_mag = np.abs(dudy)
            return float(np.nanmean(omega_mag)), "Approximated from du/dy along a 1D profile."
        except Exception as e:
            return None, f"Could not compute 1D gradient: {e}"

    return None, "Vorticity requires gridded 2D/3D fields; not enough coordinate information."


def gradient_richardson_number(rho, y, U=None):
    """Qualitative Ri_g proxy; returns N^2/(dU/dy)^2 if U provided, else N^2 proxy."""
    if rho is None or y is None:
        return None
    drho_dy = finite_difference_grad_1d(rho, y)
    N2 = drho_dy  # proportional proxy
    if U is None:
        return N2
    dUdy = finite_difference_grad_1d(U, y)
    eps = 1e-12
    return N2 / np.maximum(dUdy**2, eps)


def basic_dissipation_proxy(u, x=None, nu=None):
    """Isotropic surrogate: epsilon ~ 15 * nu * <(du/dx)^2> (very rough)."""
    if u is None or x is None or nu is None:
        return None
    du_dx = finite_difference_grad_1d(u, x)
    return 15.0 * float(nu) * float(np.nanmean(du_dx**2))


# +
# --- 6.3 / 6.4: Fallback for datasets that provide k (and epsilon) but not u,v,w ---

def get_col(dataset, name_candidates):
    """Return the first matching column by name (case-insensitive)."""
    if dataset is None:
        return None
    cols = [c for c in dataset["columns"]]
    lowers = {c.lower(): c for c in cols}
    for cand in name_candidates:
        if cand.lower() in lowers:
            colname = lowers[cand.lower()]
            if dataset["is_pandas"]:
                return dataset["data"][colname].to_numpy()
            else:
                return dataset["data"][colname]
    return None

def stresses_from_k(k_array):
    """HIT isotropy fallback: return mean RS and TKE from k array."""
    import numpy as np
    k_array = np.asarray(k_array, dtype=float)
    k_mean = float(np.nanmean(k_array))
    RS = {
        "uu": 2.0/3.0 * k_mean,
        "vv": 2.0/3.0 * k_mean,
        "ww": 2.0/3.0 * k_mean,
        "uv": 0.0, "uw": 0.0, "vw": 0.0,
    }
    return RS, k_mean

def print_tau_if_possible(k_array, eps_array):
    import numpy as np
    if eps_array is None:
        return
    k_array = np.asarray(k_array, dtype=float)
    eps_array = np.asarray(eps_array, dtype=float)
    tau = np.nanmean(k_array) / max(np.nanmean(eps_array), 1e-30)
    print(f"   Time scale tau = k/epsilon ≈ {tau:.3e} s")

# --- GRID ---

GRID_CSV = "chapter4-Grid_dataset.csv"
HIT_CSV  = "chapter4-HIT_dataset.csv"

grid_data = load_dataset_csv(GRID_CSV) 
hit_data  = load_dataset_csv(HIT_CSV)    

if grid_data is None:
    print("[GRID] Dataset not found.")
else:
    u = get_col(grid_data, ["u","ux","umean"])
    v = get_col(grid_data, ["v","uy","vmean"])
    w = get_col(grid_data, ["w","uz","wmean"])
    if (u is None) and (v is None) and (w is None):
        k_arr   = get_col(grid_data, ["k","tke"])
        eps_arr = get_col(grid_data, ["epsilon","eps","dissipation"])
        if k_arr is None:
            print("[GRID] No u,v,w or k found — cannot compute stresses.")
        else:
            RS, k_mean = stresses_from_k(k_arr)
            print("[GRID] (HIT fallback) Reynolds stresses from k:", RS)
            print(f"[GRID] (HIT fallback) TKE k = {k_mean:.6e}")
            print_tau_if_possible(k_arr, eps_arr)
    else:
        # original path using components (kept for completeness)
        RS = sm4.reynolds_stresses(u, v, w)
        k  = sm4.tke_from_components(u, v, w)
        print("[GRID] Reynolds stresses:", RS)
        print(f"[GRID] TKE k = {k:.6e}")

# --- HIT ---
if hit_data is None:
    print("[HIT] Dataset not found.")
else:
    u = get_col(hit_data, ["u","ux","umean"])
    v = get_col(hit_data, ["v","uy","vmean"])
    w = get_col(hit_data, ["w","uz","wmean"])
    if (u is None) and (v is None) and (w is None):
        k_arr   = get_col(hit_data, ["k","tke"])
        eps_arr = get_col(hit_data, ["epsilon","eps","dissipation"])
        if k_arr is None:
            print("[HIT] No u,v,w or k found — cannot compute stresses.")
        else:
            RS, k_mean = stresses_from_k(k_arr)
            print("[HIT] (HIT fallback) Reynolds stresses from k:", RS)
            print(f"[HIT] (HIT fallback) TKE k = {k_mean:.6e}")
            print_tau_if_possible(k_arr, eps_arr)
            # simple isotropy ratios ≈ 1:1:1 by construction
            print("[HIT] Normal-stress ratios (uu:vv:ww) ≈ 1.00 : 1.00 : 1.00")
    else:
        RS = sm4.reynolds_stresses(u, v, w)
        k  = sm4.tke_from_components(u, v, w)
        print("[HIT] Reynolds stresses:", RS)
        print(f"[HIT] TKE k = {k:.6e}")
        uu, vv, ww = RS.get('uu', np.nan), RS.get('vv', np.nan), RS.get('ww', np.nan)
        if uu > 0:
            print("[HIT] Normal-stress ratios (uu:vv:ww) ≈ "
                  f"1.00 : {vv/uu:.2f} : {ww/uu:.2f}")

# -




