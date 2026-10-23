"""
Estimation-Theoretic Model:
Computes the Jacobian, Fisher Information, Cramer-Rao bound and a gated Gauss-Newton pose estimator
For a state defined by: x = (p_x, p_y, psi)
For a beam hitting a wall: (n, d)
"""

from __future__ import annotations
import numpy as np
from .model import SensorParams, predict

def range_jacobian(x, walls, thetas, idx):
    x = np.asarray(x, float)
    ang = x[2] + np.asarray(thetas, float)
    u = np.stack([np.cos(ang), np.sin(ang)], axis = 1)
    up = np.stack([-np.sin(ang), np.cos(ang)], axis = 1)
    H = np.zeros((len(ang), 3))
    
    for j, w in enumerate(walls):
        m = idx == j
        if not m.any():
            continue
        nu = u[m] @ w.n
        nup = up[m] @ w.n
        h = (w.d - x[:2] @ w.n) / nu
        H[m, 0:2] = -w.n[None, :] / nu[:, None]
        H[m, 2] = -h * nup / nu
    return H

"""
Fisher Info of hit component
I(x) = sum_i  w_hit,i / sigma_eff,i^2 * h_i' h_i'^T
Neglects the variance-gradient term and assumes gating removes the short/outlier components
"""
def information_matrix(x, walls, thetas, beta = 0.0, prm:SensorParams = SensorParams()):
    pr = predict(x, walls, thetas, beta, prm)
    H = range_jacobian(x, walls, thetas, pr.idx)
    ok = pr.idx >= 0
    w = np.zeros(len(pr.r))
    w[ok] = pr.w_hit[ok] / pr.sigma_eff[ok] ** 2
    return H.T @ (w[:, None] * H)

"""
Return Rank and Eigenvalues
"""
def crb(I, rtol: float = 1e-9):
    I = 0.5 * (I + I.T)
    ev = np.linalg.eigvalsh(I)
    rank = int((ev > rtol * max(ev.max(), 1e-300)).sum())
    C = np.linalg.inv(I) if rank == I.shape[0] else np.linalg.pinv(I, rcond = rtol)
    return C, rank, ev

"""
Gated and weighted Gauss-Newton localisation against the known map
"""
def gn_localize(z, thetas, walls, x0, prm: SensorParams = SensorParams(), beta = 0.0, iters: int = 25, gate_sigma: float = 4.0, gate0: float = 0.5):
    x = np.array(x0, float)
    n_used = 0
    
    for k in range(iters):
        pr = predict(x, walls, thetas, beta, prm)
        ok = np.isfinite(z) & (pr.idx >= 0)
        res = np.zeros(len(z))
        res[ok] = z[ok] - pr.r[ok]
        s = np.where(ok, pr.sigma_eff, 1.0)
        gate = np.maximum(gate_sigma * s, gate0 * 0.5 ** k)
        ok &= np.abs(res) < gate
        n_used = int(ok.sum())
        
        if n_used < 3:
            break
        H = range_jacobian(x, walls, thetas, pr.idx)
        w = np.where(ok, 1.0 / s ** 2, 0.0)
        A = H.T @ (w[:, None] * H)
        g = H.T @ (w * res)
        dx = np.linalg.lstsq(A, g, rcond = 1e-10)[0]
        x = x + dx
        
        if np.hypot(dx[0], dx[1]) < 1e-7 and abs(dx[2]) < 1e-8:
            break
    return x, n_used

