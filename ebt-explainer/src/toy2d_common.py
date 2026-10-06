"""Shared definitions for the 2D toy Energy-Based Model (toy model trained for this explainer).

Data:   context x in [0,1); target y in R^2 = mu(x) + sigma(x) * N(0, I)
        mu(x) = (A_X sin(2 pi x), A_Y sin(4 pi x))     (a figure-eight / Lissajous curve)
        sigma(x) = S_MIN + (S_MAX - S_MIN) * sin^2(pi (x - 0.25))
        -> right tip (x=0.25) almost deterministic, left tip (x=0.75) very noisy,
           both crossings (x=0 and x=0.5) medium noise.
Model:  E_theta(x, yhat) = MLP([phi(x), yhat]) + LAMBDA * ||yhat||^2
        phi(x) = [sin(2 pi k x), cos(2 pi k x) for k = 1..K]  (interleaved: sin1, cos1, sin2, cos2, ...)
        hidden layers: h = silu(W h + b), last layer linear -> scalar.
"""
import numpy as np

A_X, A_Y = 1.3, 0.9
S_MIN, S_MAX = 0.03, 0.30
K_FEAT = 3
LAMBDA = 0.02
HIDDEN = [64, 64, 64]
IN_DIM = 2 * K_FEAT + 2
EXTENT = [-2.5, 2.5, -2.5, 2.5]   # xmin, xmax, ymin, ymax of energy grids (prediction space)
GRID_N = 64
INIT_SCALE = 1.0                  # yhat_0 ~ N(0, INIT_SCALE^2 I)


def mu_np(x):
    x = np.asarray(x, dtype=np.float64)
    return np.stack([A_X * np.sin(2 * np.pi * x), A_Y * np.sin(4 * np.pi * x)], axis=-1)


def sigma_np(x):
    x = np.asarray(x, dtype=np.float64)
    return S_MIN + (S_MAX - S_MIN) * np.sin(np.pi * (x - 0.25)) ** 2


def feats_np(x):
    x = np.asarray(x, dtype=np.float64)
    out = []
    for k in range(1, K_FEAT + 1):
        out.append(np.sin(2 * np.pi * k * x))
        out.append(np.cos(2 * np.pi * k * x))
    return np.stack(out, axis=-1)


def sig4(a):
    """Round to 4 significant digits; returns nested python lists / floats."""
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 0:
        v = float(a)
        if v == 0 or not np.isfinite(v):
            return 0.0 if v == 0 else v
        return float(f"{v:.4g}")
    flat = np.array([float(f"{v:.4g}") if v != 0 else 0.0 for v in a.ravel()])
    return flat.reshape(a.shape).tolist()


def round_params(params):
    """params: list of (W [out,in], b [out]) -> rounded copies (float64), what the JSON stores."""
    out = []
    for W, b in params:
        out.append((np.array(sig4(W), dtype=np.float64), np.array(sig4(b), dtype=np.float64)))
    return out


def rnd_e(a):
    """Energies: 4 significant digits but never coarser than 3 decimals (abs error <= 5e-4)."""
    a = np.asarray(a, dtype=np.float64)
    s = np.array(sig4(a), dtype=np.float64)
    r = np.round(a, 3)
    return np.where(np.abs(a) >= 10, r, s).tolist()
