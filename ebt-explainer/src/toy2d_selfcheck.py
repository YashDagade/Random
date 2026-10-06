"""Independent plain-numpy re-implementation of the toy EBM from data/toy2d.json only.
Checks that energies on stored grids and stored GD trajectories are reproduced from the exported weights,
using only the text description in `arch` (what a browser implementation would do)."""
import json
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = json.load(open(os.path.join(ROOT, "data", "toy2d.json")))
A = D["arch"]
LAM = A["quadratic_term"]["lambda_"]
K = A["feature_map"]["K"]


def phi(x):
    return np.array([f(2 * np.pi * k * x) for k in range(1, K + 1) for f in (np.sin, np.cos)])


def silu(z):
    return z / (1 + np.exp(-z))


def dsilu(z):
    s = 1 / (1 + np.exp(-z))
    return s * (1 + z * (1 - s))


def energy_and_grad(layers, x, Y):
    """Y: [P,2] candidate predictions. Returns E [P], dE/dY [P,2]."""
    P = Y.shape[0]
    h = np.concatenate([np.repeat(phi(x)[None], P, 0), Y], 1)  # [P,8]
    zs = []
    for L in layers[:-1]:
        W, b = np.array(L["W"]), np.array(L["b"])
        z = h @ W.T + b
        zs.append((z, W))
        h = silu(z)
    W, b = np.array(layers[-1]["W"]), np.array(layers[-1]["b"])
    E = (h @ W.T + b)[:, 0] + LAM * np.sum(Y ** 2, 1)
    g = np.repeat(W, P, 0)  # [P, H]
    for z, Wl in reversed(zs):
        g = (g * dsilu(z)) @ Wl
    return E, g[:, -2:] + 2 * LAM * Y


wts = {w["step"]: w["layers"] for w in D["weights"]}
ext, n = D["extent"], D["grid_n"]
xs, ys = np.linspace(ext[0], ext[1], n), np.linspace(ext[2], ext[3], n)
GX, GY = np.meshgrid(xs, ys)
pts = np.stack([GX.ravel(), GY.ravel()], 1)

worst = 0.0
worst_rel = 0.0
for g in D["grids"]:
    E, _ = energy_and_grad(wts[g["step"]], D["contexts"][g["ctx"]]["x"], pts)
    ref = np.array(g["E"]).ravel()
    err = np.abs(E - ref)
    worst = max(worst, err.max())
    worst_rel = max(worst_rel, (err / np.maximum(np.abs(ref), 1e-9)).max())
print(f"grids: {len(D['grids'])} checked, max |E_numpy - E_json| = {worst:.2e} (max rel {worst_rel:.2e}), "
      f"|E| range {min(np.min(g['E']) for g in D['grids']):.3f}..{max(np.max(g['E']) for g in D['grids']):.3f}")
assert worst < 1e-3, worst

final = D["checkpoint_steps"][-1]
tw = 0.0
for tr in D["trajectories"]:
    x = D["contexts"][tr["ctx"]]["x"]
    for p in tr["gd"]:
        path = np.array(p["path"])
        y = path[0:1].copy()
        rec = [y[0].copy()]
        for i in range(tr["steps"]):
            _, gr = energy_and_grad(wts[final], x, y)
            y = y - tr["alpha"] * gr
            rec.append(y[0].copy())
        rec = np.array(rec)
        tw = max(tw, np.abs(rec - path).max())
        Erec, _ = energy_and_grad(wts[final], x, rec)
        tw = max(tw, np.abs(Erec - np.array(p["E"])).max())
print(f"trajectories: max abs deviation (path + energy) = {tw:.2e}")
assert tw < 2e-3, tw

# BoN chosen index is the argmin of final energies
b = D["bon"]
assert b["chosen"] == int(np.argmin([c["final_E"] for c in b["candidates"]]))
print("bon: chosen index is the lowest final energy: OK")
print("SELF-CHECK PASSED")
