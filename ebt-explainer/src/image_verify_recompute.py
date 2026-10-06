"""Verifier: independently recompute a few headline toy numbers in data/image.json from the saved params.
python3 -I src/image_verify_recompute.py <data_npz> <run_dir> <project_root>
Re-creates the same noise draw as image_export.py (PRNGKey(2025), fold_in t, split 3) and evaluates
noisy input, FF, EBT (noise start) at 4 steps, and DDIM K=2 on the first 300 test images.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import jax, jax.numpy as jnp
import image_ebt as M

npz, run_dir, root = sys.argv[1:4]
d = json.load(open(os.path.join(root, "data/image.json")))
N = d["_meta"]["dataset"]["eval_images"]
tr, te, raw = M.load_data(npz)
Y = jnp.asarray(te[:N])
P = {t: {k: jnp.asarray(v) for k, v in np.load(os.path.join(run_dir, f"{t}_params.npz")).items()} for t in ["ebt", "ff", "diff"]}


def psnr_all(a, y):
    a = np.clip((np.asarray(a, np.float64) + 1) * 127.5, 0, 255); y = np.clip((np.asarray(y, np.float64) + 1) * 127.5, 0, 255)
    m = np.mean((a - y) ** 2)
    return 10 * np.log10(255 ** 2 / m)


# check that the saved clean PNG tile equals the dataset image
from PIL import Image
s0 = d["strips"][0]
png = np.asarray(Image.open(os.path.join(root, s0["frames"]["clean"])).convert("RGB"))[::4, ::4]
print("clean PNG == dataset test image", s0["test_index"], ":", np.array_equal(png, raw["test_x"][s0["test_index"]]))

key = jax.random.PRNGKey(2025)
gfn = jax.jit(M.energy_grad)
tr_fn = jax.jit(M.trunk)
for sig in [0.1, 0.2]:
    t = M.t_of_sigma(sig)
    kx, ky0, kb = jax.random.split(jax.random.fold_in(key, t), 3)
    X, _ = M.add_noise(kx, Y, t)
    Y0 = jax.random.normal(ky0, Y.shape)
    r = d["results"][str(sig)]
    out = {"noisy": (psnr_all(X, Y), r["noisy_input"]["psnr"]), "ff": (psnr_all(tr_fn(P["ff"], X), Y), r["ff"]["psnr"])}
    yh = Y0
    for i in range(4):
        yh = yh - M.HP["ebt_alpha"] * gfn(P["ebt"], X, yh)
    out["ebt4"] = (psnr_all(yh, Y), r["ebt"]["steps_4"]["psnr"])
    # DDIM K=2
    ts = np.unique(np.round(np.linspace(t, 0, 3)).astype(int))[::-1]
    xt = X
    for a, b in zip(ts[:-1], ts[1:]):
        ab, abn = float(M.abar_of_t(a)), float(M.abar_of_t(b))
        eps = tr_fn(P["diff"], jnp.concatenate([xt, jnp.full(xt.shape[:3] + (1,), a / 100.0)], -1))
        x0 = jnp.clip((xt - np.sqrt(1 - ab) * eps) / np.sqrt(ab), -1, 1)
        xt = np.sqrt(abn) * x0 + np.sqrt(1 - abn) * eps if b > 0 else x0
    out["ddim2"] = (psnr_all(xt, Y), [c for c in r["diff_curve"] if c["nfe"] == 2][0]["psnr"])
    for k, (mine, js) in out.items():
        print(f"sigma {sig} {k:6s} recomputed {mine:6.2f} json {js:6.2f} {'OK' if abs(mine - js) < 0.02 else 'MISMATCH'}")
