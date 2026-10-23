"""
This script creates a physics-based and probabilistic 2-D LiDAR measurement model using python.

Notes:
- We define the sensor state as x = (p_x, p_y, psi) which is the position and heading defined in the world frame
- Beam i has a body-frame angle theta_i and world_direction u_i with heading offset psi
- A wall is a segment from a to b, here we define only the front face to be visible
"""

from __future__ import annotations
from dataclasses import dataclass, replace
import numpy as np
from math import erf, sqrt

_ndtr = np.vectorize(lambda v: 0.5 * (1.0 + erf(v / sqrt(2.0))))

def ndtr(v):
    """Standard normal CDF, Phi(v)."""
    return _ndtr(v)

@dataclass
class SensorParams:
    r_min: float = 0.12     # blind zone [m]
    r_max: float = 12.0     # [m]
    
    # Signal normalization so S = 1
    rho_ref: float = 0.8
    r_ref   : float = 5.0   # [m]
    
    # Detection: P_det = Phi((S - S_min) / sigma_n)
    S_min: float = 0.08
    sigma_n: float = 0.03
    
    # Range of noise
    sigma0: float = 0.010   # Noise from electronics [m]
    sigma1: float = 0.015   # Photon limited timing noise [m]
    sigma_theta: float = float(np.deg2rad(0.1))  # [rad] beam-pointing jitter (1 sigma)
    
    # Describe outliers and fog
    z_rand: float = 0.002   # Prob. of a uniform outlier per beam
    kappa: float = 0.5  # Fog backscatter rate: lambda_f = kappa * beta [1/m]
    
    def with_(self, **kw) -> "SensorParams":
        return replace(self, **kw)
    

# Define the environment/obstacles
# Line segment defining the wall
class Wall:
    __slots__ = ("a", "b", "rho", "t", "n", "d", "L")
    
    def __init__(self, a, b, rho: float = 0.8):
        self.a = np.asarray(a, float)
        self.b = np.asarray(b, float)
        self.rho = float(rho)
        e = self.b - self.a
        self.L = float(np.hypot(e[0], e[1]))
        self.t = e / self.L
        self.n = np.array([-self.t[1], self.t[0]])
        self.d = float(self.n @ self.a)
        

# PHYSICS BASED MODEL
def raycast(p, psi, thetas, walls, prm: SensorParams):
    p = np.asarray(p, float)
    thetas = np.asarray(thetas, float)
    ang = psi + thetas
    u = np.stack([np.cos(ang), np.sin(ang)], axis = 1)
    N = len(thetas)
    best = np.full(N, np.inf)
    idx = np.full(N, -1, dtype=int)
    
    for j, w in enumerate(walls):
        nu = u @ w.n
        
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (w.d - p @ w.n) / nu
        q = p[None, :] + t[:, None] * u
        s = (q - w.a) @ w.t
        ok = (nu < -1e-9) & (t > 0.0) & (s >= 0.0) & (s <= w.L) & (t < best)
        best[ok] = t[ok]
        idx[ok] = j
        
    valid = (idx >= 0) & (best >= prm.r_min) & (best <= prm.r_max)
    r = np.where(valid, best, np.inf)
    idx = np.where(valid, idx, -1)
    nu_hit = np.einsum("ij, ij -> i", u, np.array([walls[k].n if k >= 0 else [0.0, 0.0] for k in idx]))
    alpha = np.where(valid, np.arccos(np.clip(np.abs(nu_hit), 0.0, 1.0)), np.nan)
    
    return r, idx, alpha

# Normalised lidar equation: S = (rho/rho_ref) cos(alpha) (r_ref/r)^2 exp(-2 beta r)
def received_signal(r, alpha, rho, beta, prm: SensorParams):
    return (rho / prm.rho_ref) * np.cos(alpha) * (prm.r_ref / r) ** 2 * np.exp(-2.0 * beta * r)


# PROBABILISTIC MODEL

# P_det = Phi((S - S_min) / sigma_n)
def p_detect(S, prm: SensorParams):
    return ndtr((S - prm.S_min) / prm.sigma_n)

# (sigma_r):  sigma_r^2 = sigma0^2 + sigma1^2 / S
def sigma_range(S, prm:SensorParams):
    return np.sqrt(prm.sigma0 ** 2 + prm.sigma1 ** 2 / np.maximum(S, 1e-12))

# Range noise plus pointing jitter mapped through dr/dtheta = r tan(alpha)
def sigma_eff(r, alpha, S, prm:SensorParams):
    sr = sigma_range(S, prm)
    return np.sqrt(sr ** 2 + (r * np.tan(alpha) * prm.sigma_theta) ** 2)

# Weights of the mixture density p(z)
# Determines probability of detecting the 4 possible events
def mixture_weights(r, pdet, beta, prm:SensorParams):
    r = np.asarray(r, float)
    lam = prm.kappa * beta
    r_c = np.where(np.isfinite(r), r, prm.r_max)
    e_c = np.exp(-lam * r_c)
    e_min = np.exp(-lam * prm.r_min)
    zr = prm.z_rand
    w_short = (1 - zr) * (e_min - e_c)
    w_hit = (1 - zr) * e_c * pdet
    w_max = (1 - zr) * ((1 - e_min) + e_c * (1 - pdet))
    w_rand = np.full_like(r_c, zr)
    return w_hit, w_short, w_max, w_rand


@dataclass
class Prediction:
    theta: np.ndarray
    r: np.ndarray    # true range (if no return, return inf)
    idx: np.ndarray      # wall index (if none, return -1)
    alpha: np.ndarray
    S: np.ndarray
    p_det: np.ndarray
    sigma_r: np.ndarray
    sigma_eff: np.ndarray
    w_hit: np.ndarray
    w_short: np.ndarray
    w_max: np.ndarray
    w_rand: np.ndarray


# This function returns everything the model predicts for pose, geometry, signal, noise, and mixture weights
def predict(x, walls, thetas, beta: float = 0.0, prm: SensorParams = SensorParams()) -> Prediction:
    x = np.asarray(x, float)
    thetas = np.asarray(thetas, float)
    
    r, idx, alpha = raycast(x[:2], x[2], thetas, walls, prm)
    
    hit = idx >= 0
    rho_w = np.array([w.rho for w in walls]) if walls else np.zeros(1)
    rho = np.where(hit, rho_w[np.clip(idx, 0, None)], 0.0)
    r_s = np.where(hit, r, 1.0)
    a_s = np.where(hit, alpha, 0.0)
    
    S = np.where(hit, received_signal(r_s, a_s, rho, beta, prm), 0.0)
    pdet = np.where(hit, p_detect(S, prm), 0.0)
    s_r = np.where(hit, sigma_range(S, prm), np.nan)
    s_e = np.where(hit, sigma_eff(r_s, a_s, S, prm), np.nan)
    
    w_hit, w_short, w_max, w_rand = mixture_weights(r, pdet, beta, prm)
    return Prediction(thetas, r, idx, alpha, S, pdet, s_r, s_e, w_hit, w_short, w_max, w_rand)

def sample_scan(x, walls, thetas, beta, prm: SensorParams, rng: np.random.Generator):
    x = np.asarray(x, float)
    thetas = np.asarray(thetas, float)
    N = len(thetas)
    th = thetas + rng.normal(0.0, prm.sigma_theta, N)
    r, idx, alpha = raycast(x[:2], x[2], th, walls, prm)
    hit = idx >= 0
    rho_w = np.array([w.rho for w in walls]) if walls else np.zeros(1)
    rho = np.where(hit, rho_w[np.clip(idx, 0, None)], 0.0)
    r_s = np.where(hit, r, 1.0)
    a_s = np.where(hit, alpha, 0.0)
    S = np.where(hit, received_signal(r_s, a_s, rho, beta, prm), 0.0)
    pdet = np.where(hit, p_detect(S, prm), 0.0)
    sr = sigma_range(S, prm)

    z = np.full(N, np.inf)
    det = hit & (rng.random(N) < pdet)
    z[det] = r[det] + rng.normal(0.0, sr[det])

    lam = prm.kappa * beta
    if lam > 0:
        zf = rng.exponential(1.0 / lam, N)
        r_c = np.where(hit, r, prm.r_max)
        pre = zf < r_c
        z[pre] = np.where(zf[pre] >= prm.r_min, zf[pre], np.inf)

    out = rng.random(N) < prm.z_rand
    z[out] = rng.uniform(prm.r_min, prm.r_max, out.sum())
    z[(z < prm.r_min) | (z > prm.r_max)] = np.inf
    return z


def beam_density(z, r, sigma_e, pdet, beta, prm: SensorParams):
    z = np.asarray(z, float)
    w_hit, w_short, w_max, w_rand = (float(v) for v in mixture_weights(np.array([r]), pdet, beta, prm))
    dens = np.zeros_like(z)
    
    if np.isfinite(r):
        dens += w_hit * np.exp(-0.5 * ((z - r) / sigma_e) ** 2) / (sigma_e * np.sqrt(2 * np.pi))
    lam = prm.kappa * beta
    r_c = r if np.isfinite(r) else prm.r_max
    
    if lam > 0:
        dens += (1 - prm.z_rand) * lam * np.exp(-lam * z) * ((z >= prm.r_min) & (z < r_c))
    dens += w_rand / (prm.r_max - prm.r_min) * ((z >= prm.r_min) & (z <= prm.r_max))
    return dens


# Field of view (FOV) and Line of Sight (LOS) for detection of a point target
"""
Probability that a small Lambertian target at each point produces a return
P_obs(q) = 1_FOV(q; x) * 1_LOS(q; x) * (1 - z_rand) * exp(-kappa*beta*r) * P_det(S(r, 0, rho_t, beta))
"""
def p_observe(points, x, walls, beta, prm: SensorParams, rho_t: float = 0.1, fov=(-np.pi, np.pi), return_parts: bool = False):
    q = np.atleast_2d(np.asarray(points, float))
    x = np.asarray(x, float)
    d = q - x[:2]
    rng_ = np.hypot(d[:, 0], d[:, 1])
    ang = np.arctan2(d[:, 1], d[:, 0])
    bearing = (ang - x[2] + np.pi) % (2 * np.pi) - np.pi
    in_fov = (rng_ >= prm.r_min) & (rng_ <= prm.r_max) & (bearing >= fov[0]) & (bearing <= fov[1])
    r_wall, _, _ = raycast(x[:2], 0.0, ang, walls, prm.with_(r_min=0.0, r_max=np.inf))
    los = r_wall >= rng_ - 1e-9
    r_s = np.maximum(rng_, 1e-6)
    S = received_signal(r_s, 0.0, rho_t, beta, prm)
    P = in_fov * los * (1 - prm.z_rand) * np.exp(-prm.kappa * beta * r_s) * p_detect(S, prm)
    return (P, in_fov, los) if return_parts else P