"""Control analysis for the uncertainty question (toy model trained for this explainer).
Compares basin statistics at the two mirror-symmetric tips of the figure-eight for the main run
(noise low at x=0.25, high at x=0.75) and a control run with the noise pattern mirrored
(TOY2D_SIGMA_SHIFT=0.75: noise high at x=0.25, low at x=0.75). If an effect follows the noise, it is
caused by the noise; if it stays with the geometry, it is not. Prints a JSON summary."""
import os
import sys
import json

os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from toy2d_common import HIDDEN, LAMBDA, K_FEAT, mu_np, feats_np, S_MIN, S_MAX

SCR = "/tmp/claude-0/-home-user-Random/9843a8a1-7b0a-5ed7-8867-6762f4a2ea11/scratchpad"
RUNS = {"main (low noise at x=0.25)": (os.path.join(SCR, "toy2d"), 0.25),
        "control (low noise at x=0.75)": (os.path.join(SCR, "toy2d_control"), 0.75)}


def energy_single(params, f, yh):
    h = jnp.concatenate([f, yh])
    for W, b in params[:-1]:
        h = jax.nn.silu(W @ h + b)
    W, b = params[-1]
    return (W @ h + b)[0] + LAMBDA * jnp.sum(yh ** 2)


G = jax.jit(jax.grad(energy_single, argnums=2))
H = jax.jit(jax.hessian(energy_single, argnums=2))
Ej = jax.jit(energy_single)


def stats(params, x):
    f = jnp.asarray(feats_np(x))
    y = jnp.asarray(mu_np(x))
    for _ in range(400):
        y = y - 0.5 * G(params, f, y)
    ev = np.linalg.eigvalsh(np.asarray(H(params, f, y)))
    return float(Ej(params, f, y)), ev.tolist(), float(np.linalg.norm(np.asarray(y) - mu_np(x)))


out = {}
for name, (d, shift) in RUNS.items():
    Z = np.load(os.path.join(d, "train_out.npz"))
    steps = [int(s) for s in Z["ckpt_steps"]]
    for s in [5000, steps[-1]] if 5000 in steps else [steps[-1]]:
        p = [(jnp.asarray(Z[f"c{s}_W{i}"]), jnp.asarray(Z[f"c{s}_b{i}"])) for i in range(len(HIDDEN) + 1)]
        row = {}
        for x in (0.25, 0.75):
            sig = S_MIN + (S_MAX - S_MIN) * np.sin(np.pi * (x - shift)) ** 2
            e, ev, dist = stats(p, x)
            row[f"x={x} (sigma {sig:.2f})"] = dict(E_min=round(e, 4), hess_eig=[round(v, 4) for v in ev], argmin_err=round(dist, 4))
        xs = (np.arange(64) + 0.5) / 64
        sig = S_MIN + (S_MAX - S_MIN) * np.sin(np.pi * (xs - shift)) ** 2
        em, cur = [], []
        for x in xs:
            e, ev, _ = stats(p, x)
            em.append(e); cur.append(np.mean(ev))
        row["corr(sigma,E_min)"] = round(float(np.corrcoef(sig, em)[0, 1]), 3)
        row["corr(sigma,mean_curv)"] = round(float(np.corrcoef(sig, cur)[0, 1]), 3)
        out[f"{name} step {s}"] = row
print(json.dumps(out, indent=1))
