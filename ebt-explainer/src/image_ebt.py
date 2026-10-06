"""Toy image-denoising EBT vs same-size baselines (toy model trained for this explainer).

Run with:  XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
           python3 -I src/image_ebt.py <mode> <data_npz> <run_dir> [seconds]
modes: bench | ebt | ff | diff

Noise follows paper Sec 4.3 / App D.3: DDPM linear beta schedule 1e-4 -> 2e-2 over T=1000,
sigma = fraction of the schedule, t = round(sigma*T),
x_noisy = sqrt(abar_t) * y + sqrt(1 - abar_t) * eps, with abar_t = prod_{s=1..t} (1 - beta_s).
Pixels live in [-1, 1]. Train only at sigma = 0.1.

Models share one small conv trunk (32x32 RGB):
  EBT  : E(x, yhat) = 1/2 * sum_pixels f([x, yhat])^2 (scalar), think by yhat <- yhat - a * dE/dyhat
         from yhat_0 ~ N(0, I) (paper Alg. 1), unrolled steps, MSE to clean image at the last step,
         backprop through the whole optimization (second order), random step count, random step size,
         Langevin noise.
  FF   : yhat = f(x) in one forward pass (same trunk, 3 input channels).
  DIFF : eps_hat = f([x_t, t/100]) trained on t ~ U{1..100}; sampled with DDIM from t = sigma*T to 0.
"""
import json
import os
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp

T = 1000
BETAS = np.linspace(1e-4, 2e-2, T, dtype=np.float64)
ABAR = np.cumprod(1.0 - BETAS)  # ABAR[t-1] = prod_{s=1..t}(1-beta_s)


def abar_of_t(t):
    """abar after t noising steps (t=0 -> 1.0)."""
    t = np.asarray(t)
    return np.where(t <= 0, 1.0, ABAR[np.clip(t, 1, T) - 1])


def t_of_sigma(sigma):
    return int(round(sigma * T))


C1, C2 = 24, 48  # trunk widths


def conv(x, w, stride=1):
    return jax.lax.conv_general_dilated(
        x, w, (stride, stride), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC"))


def init_trunk(key, cin, cout=3, out_gain=0.1):
    ks = jax.random.split(key, 8)

    def he(k, shape, gain=1.0):
        fan_in = shape[0] * shape[1] * shape[2]
        return (jax.random.normal(k, shape) * gain * np.sqrt(2.0 / fan_in)).astype(jnp.float32)

    p = {
        "w1": he(ks[0], (3, 3, cin, C1)), "b1": jnp.zeros(C1),
        "w2": he(ks[1], (3, 3, C1, C2)), "b2": jnp.zeros(C2),
        "w3": he(ks[2], (3, 3, C2, C2)), "b3": jnp.zeros(C2),
        "w4": he(ks[3], (3, 3, C2, C2)), "b4": jnp.zeros(C2),
        "w5": he(ks[4], (3, 3, C2, C1)), "b5": jnp.zeros(C1),
        "w6": he(ks[5], (3, 3, C1, C1)), "b6": jnp.zeros(C1),
        "w7": he(ks[6], (3, 3, C1, cout), out_gain), "b7": jnp.zeros(cout),
    }
    return p


def trunk(p, inp):
    """inp: (B,32,32,cin) -> (B,32,32,3). Small U-Net-like conv net with SiLU (smooth, so
    second derivatives exist everywhere)."""
    act = jax.nn.silu
    h1 = act(conv(inp, p["w1"]) + p["b1"])                 # 32x32xC1
    B, H, W, C = h1.shape
    h = h1.reshape(B, H // 2, 2, W // 2, 2, C).mean(axis=(2, 4))   # 2x2 average pool
    h = act(conv(h, p["w2"]) + p["b2"])                    # 16x16xC2
    h = h + act(conv(h, p["w3"]) + p["b3"])
    h = h + act(conv(h, p["w4"]) + p["b4"])
    h = act(conv(h, p["w5"]) + p["b5"])                    # 16x16xC1
    h = jnp.repeat(jnp.repeat(h, 2, axis=1), 2, axis=2)    # 32x32xC1
    h = h + h1
    h = act(conv(h, p["w6"]) + p["b6"])
    return conv(h, p["w7"]) + p["b7"]


def n_params(p):
    return int(sum(np.prod(v.shape) for v in jax.tree_util.tree_leaves(p)))


# ---------------- EBT ----------------

def energy(p, x, yh):
    """Scalar energy per image: 1/2 * sum over pixels and channels of f(x, yhat)^2."""
    f = trunk(p, jnp.concatenate([x, yh], axis=-1))
    return 0.5 * jnp.sum(f * f, axis=(1, 2, 3))


def energy_grad(p, x, yh):
    e, g = jax.value_and_grad(lambda y: jnp.sum(energy(p, x, y)))(yh)
    return g


def ebt_think(p, x, y0, alpha, n_steps, noise=None):
    """Plain inference loop (python side), returns list of predictions and energies."""
    ys, es = [y0], []
    yh = y0
    for i in range(n_steps):
        es.append(energy(p, x, yh))
        yh = yh - alpha * energy_grad(p, x, yh)
        ys.append(yh)
    es.append(energy(p, x, yh))
    return ys, es


# ---------------- Noise ----------------

def add_noise(key, y, t):
    """y in [-1,1] (B,H,W,C); t int array (B,) or scalar."""
    ab = jnp.asarray(abar_of_t(t), dtype=jnp.float32).reshape(-1, 1, 1, 1)
    eps = jax.random.normal(key, y.shape)
    return jnp.sqrt(ab) * y + jnp.sqrt(1 - ab) * eps, eps


# ---------------- Adam ----------------

def adam_init(p):
    z = jax.tree_util.tree_map(jnp.zeros_like, p)
    return {"m": z, "v": jax.tree_util.tree_map(jnp.zeros_like, p), "t": 0}


def adam_update(p, g, st, lr, b1=0.9, b2=0.999, eps=1e-8, clip=1.0):
    gn = jnp.sqrt(sum(jnp.sum(x * x) for x in jax.tree_util.tree_leaves(g)))
    scale = jnp.minimum(1.0, clip / (gn + 1e-12))
    g = jax.tree_util.tree_map(lambda x: x * scale, g)
    t = st["t"] + 1
    m = jax.tree_util.tree_map(lambda m, g: b1 * m + (1 - b1) * g, st["m"], g)
    v = jax.tree_util.tree_map(lambda v, g: b2 * v + (1 - b2) * g * g, st["v"], g)
    mh = jax.tree_util.tree_map(lambda m: m / (1 - b1 ** t), m)
    vh = jax.tree_util.tree_map(lambda v: v / (1 - b2 ** t), v)
    p = jax.tree_util.tree_map(lambda p, m, v: p - lr * m / (jnp.sqrt(v) + eps), p, mh, vh)
    return p, {"m": m, "v": v, "t": t}, gn


# ---------------- data ----------------

def load_data(npz):
    d = np.load(npz)
    tr = d["train_x"].astype(np.float32) / 127.5 - 1.0
    te = d["test_x"].astype(np.float32) / 127.5 - 1.0
    return tr, te, d


def batches(rng, x, bs):
    while True:
        idx = rng.permutation(len(x))
        for i in range(0, len(idx) - bs + 1, bs):
            b = x[idx[i:i + bs]]
            flip = rng.random(bs) < 0.5
            b = np.where(flip[:, None, None, None], b[:, :, ::-1, :], b)
            yield b


# ---------------- training ----------------

HP = {
    "batch": 32,
    "lr": 1e-3,
    "warmup": 80,
    "sigma_train": 0.1,
    "ebt_alpha": 1.0,            # fixed base step size (not learned)
    "ebt_alpha_rand": 2.0,       # alpha * exp(U(-ln2, ln2)), i.e. factor in [1/2, 2]
    "ebt_steps_choices": [2, 3],
    "ebt_langevin": 0.02,        # std of eta added each training step (pixel units in [-1,1])
    "diff_t_max": 100,           # diffusion baseline trained on t ~ U{1..100}  (sigma <= 0.1)
}


def lr_at(step, total):
    w = HP["warmup"]
    if step < w:
        return HP["lr"] * (step + 1) / w
    pr = (step - w) / max(1, total - w)
    return HP["lr"] * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * min(pr, 1.0))))


def make_ebt_loss(n_steps):
    def loss_fn(p, x, y, y0, alphas, etas):
        yh = y0
        for i in range(n_steps):
            g = energy_grad(p, x, yh)       # depends on p -> second-order backprop
            yh = yh - alphas[:, None, None, None] * g + etas[i]
        return jnp.mean((yh - y) ** 2)
    return loss_fn


def train(mode, npz, run_dir, seconds, est_steps, max_steps=10**9, tag=None):
    tag = tag or mode
    os.makedirs(run_dir, exist_ok=True)
    tr, te, _ = load_data(npz)
    rng = np.random.default_rng(0)
    key = jax.random.PRNGKey({"ebt": 1, "ff": 2, "diff": 3}[mode])
    cin = {"ebt": 6, "ff": 3, "diff": 4}[mode]
    p = init_trunk(key, cin, out_gain=0.3 if mode == "ebt" else 0.1)
    st = adam_init(p)
    bs = HP["batch"]
    t_tr = t_of_sigma(HP["sigma_train"])
    gen = batches(rng, tr, bs)

    if mode == "ebt":
        fns = {}
        for k in HP["ebt_steps_choices"]:
            lf = make_ebt_loss(k)
            fns[k] = jax.jit(jax.value_and_grad(lf))
    elif mode == "ff":
        def lf(p, x, y):
            return jnp.mean((trunk(p, x) - y) ** 2)
        vg = jax.jit(jax.value_and_grad(lf))
    else:
        def lf(p, xt, tt, eps):
            tin = jnp.broadcast_to((tt / 100.0)[:, None, None, None], xt.shape[:3] + (1,))
            return jnp.mean((trunk(p, jnp.concatenate([xt, tin], -1)) - eps) ** 2)
        vg = jax.jit(jax.value_and_grad(lf))

    upd = jax.jit(adam_update, static_argnums=())
    log = []
    t0 = time.time()
    step = 0
    while time.time() - t0 < seconds and step < max_steps:
        y = jnp.asarray(next(gen))
        key, k1, k2, k3, k4 = jax.random.split(key, 5)
        if mode == "ebt":
            x, _ = add_noise(k1, y, t_tr)
            n = int(rng.choice(HP["ebt_steps_choices"]))
            y0 = jax.random.normal(k2, y.shape)
            lnr = np.log(HP["ebt_alpha_rand"])
            alphas = HP["ebt_alpha"] * jnp.exp(jax.random.uniform(k3, (bs,), minval=-lnr, maxval=lnr))
            etas = HP["ebt_langevin"] * jax.random.normal(k4, (n,) + y.shape)
            loss, g = fns[n](p, x, y, y0, alphas, etas)
        elif mode == "ff":
            x, _ = add_noise(k1, y, t_tr)
            loss, g = vg(p, x, y)
        else:
            tt = jax.random.randint(k2, (bs,), 1, HP["diff_t_max"] + 1)
            ab = jnp.asarray(ABAR, jnp.float32)[tt - 1][:, None, None, None]
            eps = jax.random.normal(k1, y.shape)
            xt = jnp.sqrt(ab) * y + jnp.sqrt(1 - ab) * eps
            loss, g = vg(p, xt, tt.astype(jnp.float32), eps)
        lr = lr_at(step, est_steps)
        p, st, gn = upd(p, g, st, lr)
        if step % 25 == 0:
            el = time.time() - t0
            log.append({"step": step, "loss": float(loss), "gnorm": float(gn), "time": round(el, 1)})
            print(mode, step, f"loss={float(loss):.5f} gn={float(gn):.3f} t={el:.0f}s", flush=True)
        step += 1
    np.savez(os.path.join(run_dir, f"{tag}_params.npz"), **{k: np.asarray(v) for k, v in p.items()})
    with open(os.path.join(run_dir, f"{tag}_log.json"), "w") as f:
        json.dump({"mode": mode, "steps": step, "seconds": time.time() - t0, "log": log,
                   "n_params": n_params(p), "hp": HP}, f)
    print("done", mode, step, "steps", n_params(p), "params")


def bench(npz):
    tr, te, _ = load_data(npz)
    key = jax.random.PRNGKey(0)
    for mode, cin in [("ff", 3), ("ebt", 6)]:
        p = init_trunk(key, cin)
        print(mode, "params", n_params(p))
        y = jnp.asarray(tr[:HP["batch"]])
        x, _ = add_noise(key, y, 100)
        if mode == "ff":
            f = jax.jit(jax.value_and_grad(lambda p: jnp.mean((trunk(p, x) - y) ** 2)))
            f(p)[0].block_until_ready()
            t0 = time.time()
            for _ in range(5):
                f(p)[0].block_until_ready()
            print("ff step", (time.time() - t0) / 5)
        else:
            for n in [2, 4]:
                lf = jax.jit(jax.value_and_grad(make_ebt_loss(n)))
                al = jnp.ones(HP["batch"])
                et = jnp.zeros((n,) + y.shape)
                lf(p, x, y, y, al, et)[0].block_until_ready()
                t0 = time.time()
                for _ in range(3):
                    lf(p, x, y, y, al, et)[0].block_until_ready()
                print("ebt step n=", n, (time.time() - t0) / 3)


if __name__ == "__main__":
    mode, npz = sys.argv[1], sys.argv[2]
    if mode == "bench":
        bench(npz)
    else:
        run_dir = sys.argv[3]
        secs = float(sys.argv[4])
        est = int(sys.argv[5])
        mx = int(sys.argv[6]) if len(sys.argv) > 6 else 10**9
        tag = sys.argv[7] if len(sys.argv) > 7 else None
        train(mode, npz, run_dir, secs, est, mx, tag)
