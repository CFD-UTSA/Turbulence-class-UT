import numpy as np

def reynolds_avg(phi):
    return float(np.mean(phi))

def favre_avg(rho, phi):
    rho = np.asarray(rho); phi = np.asarray(phi)
    return float(np.sum(rho*phi)/np.sum(rho))

def decompose_reynolds(phi):
    m = reynolds_avg(phi)
    return m, np.asarray(phi) - m

def decompose_favre(rho, phi):
    t = favre_avg(rho, phi)
    return t, np.asarray(phi) - t

def favre_stress(rho, u, v=None):
    rho = np.asarray(rho); u = np.asarray(u)
    if v is None:
        u_t = favre_avg(rho, u)
        up = u - u_t
        return float(np.mean(rho*up*up))
    else:
        v = np.asarray(v)
        u_t = favre_avg(rho, u); v_t = favre_avg(rho, v)
        up = u - u_t; vp = v - v_t
        return float(np.mean(rho*up*vp))
