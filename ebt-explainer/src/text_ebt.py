"""Tiny character-level autoregressive Energy-Based Transformer (toy) + same-size
feed-forward baseline ("Transformer++ stand-in"), trained on real RedPajama-V2 text.

Toy model trained for this explainer. NOT a paper result.

Recipe follows the paper (Gladstone et al. 2025, Alg. 1, Sec. 3.3, Table D.4, Listing 1):
  * prediction yhat in R^V are logits, p = softmax(yhat) ("normalize input distribution")
  * prediction embedding e = p @ W_pe ; energy E = MLP([h, e, h*e]) -> scalar
  * thinking: yhat_{i+1} = yhat_i - alpha * grad_yhat E + eta,  yhat_0 ~ N(0, I)
  * training: 2-3 unrolled steps (randomized per sample), alpha randomized per sample
    (multiplicative factor 2), Langevin noise, replay buffer, loss = CE(softmax(yhat_N), y)
    computed only at the last step, backprop through the whole optimization (no detach,
    second-order derivatives via jax.grad of a function containing jax.grad).

Usage:
  python3 src/text_ebt.py train ebt  <work_dir> [steps] [alpha0]
  python3 src/text_ebt.py train base <work_dir> [steps]
"""
import os
import sys
import json
import time
import math

os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
import numpy as np
import jax
import jax.numpy as jnp

V = 54
CFG = dict(
    C=40,          # context length (characters)
    d_c=12,        # char embedding width
    enc_hidden=192,
    d_h=96,        # context representation width
    H=192,         # EBT energy MLP width
    Hb=256,        # baseline head MLP width (chosen to match parameter count)
    batch=256,
    lr=2e-3,
    wd=0.01,
    warmup=300,
    clip=1.0,
    # EBT System-2 style hyperparameters (paper Table D.4 analogues, retuned for V=54)
    alpha0=20.0,
    alpha_rand=2.0,      # alpha ~ alpha0 * exp(U(-ln 2, ln 2)), per sample
    n_min=2, n_max=3,    # randomized number of thinking steps per sample
    langevin=0.25,       # std of Langevin noise added on every step except the last
    replay_frac=0.25,    # fraction of each batch continued from the replay buffer
    replay_size=8192,
    wpe_std=1.0,         # init std of the prediction-embedding table (embedding-like init)
)


def silu(x):
    return x * jax.nn.sigmoid(x)


def glorot(key, fan_in, fan_out):
    lim = math.sqrt(6.0 / (fan_in + fan_out))
    return jax.random.uniform(key, (fan_in, fan_out), jnp.float32, -lim, lim)


def init_encoder(key, cfg):
    k = jax.random.split(key, 3)
    C, d_c, Hh, d_h = cfg["C"], cfg["d_c"], cfg["enc_hidden"], cfg["d_h"]
    return {
        "emb": jax.random.normal(k[0], (V, d_c)) * 0.5,
        "We1": glorot(k[1], C * d_c, Hh), "be1": jnp.zeros(Hh),
        "We2": glorot(k[2], Hh, d_h), "be2": jnp.zeros(d_h),
    }


def init_ebt(key, cfg):
    k = jax.random.split(key, 5)
    d_h, H = cfg["d_h"], cfg["H"]
    p = init_encoder(k[0], cfg)
    p.update({
        "Wpe": jax.random.normal(k[1], (V, d_h)) * cfg["wpe_std"],
        "W1": glorot(k[2], 3 * d_h, H), "b1": jnp.zeros(H),
        "W2": glorot(k[3], H, H), "b2": jnp.zeros(H),
        "W3": glorot(k[4], H, 1), "b3": jnp.zeros(1),
    })
    return p


def init_base(key, cfg):
    k = jax.random.split(key, 4)
    d_h, Hb = cfg["d_h"], cfg["Hb"]
    p = init_encoder(k[0], cfg)
    p.update({
        "W1": glorot(k[1], d_h, Hb), "b1": jnp.zeros(Hb),
        "W2": glorot(k[2], Hb, Hb), "b2": jnp.zeros(Hb),
        "W3": glorot(k[3], Hb, V), "b3": jnp.zeros(V),
    })
    return p


def n_params(p):
    return int(sum(np.prod(v.shape) for v in jax.tree_util.tree_leaves(p)))


def encode_ctx(p, ctx):
    """ctx: int (B, C) -> h (B, d_h).  MLP over concatenated char embeddings (causal by construction)."""
    x = p["emb"][ctx].reshape(ctx.shape[0], -1)
    a = silu(x @ p["We1"] + p["be1"])
    return a @ p["We2"] + p["be2"]


def energy(p, h, yhat):
    """Scalar energy per example. h: (B, d_h), yhat: (B, V) logits -> (B,)"""
    prob = jax.nn.softmax(yhat, axis=-1)
    e = prob @ p["Wpe"]
    z = jnp.concatenate([h, e, h * e], axis=-1)
    a = silu(z @ p["W1"] + p["b1"])
    a = silu(a @ p["W2"] + p["b2"])
    return (a @ p["W3"] + p["b3"])[:, 0]


def grad_energy(p, h, yhat):
    # energies of different examples are independent, so grad of the sum = per-example grads
    return jax.grad(lambda y: energy(p, h, y).sum())(yhat)


def base_logits(p, ctx):
    h = encode_ctx(p, ctx)
    a = silu(h @ p["W1"] + p["b1"])
    a = silu(a @ p["W2"] + p["b2"])
    return a @ p["W3"] + p["b3"]


def ce(logits, y):
    lp = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.take_along_axis(lp, y[:, None], axis=-1)[:, 0]


# ---------------------------------------------------------------- training (Algorithm 1 + Sec 3.3)

def ebt_loss(p, ctx, y, yhat0, alpha, nsteps, noise, cfg):
    """yhat0 (B,V); alpha (B,); nsteps (B,) int in [n_min, n_max]; noise (n_max, B, V) pre-scaled."""
    h = encode_ctx(p, ctx)
    yhat = yhat0
    E0 = energy(p, h, yhat0)
    for i in range(cfg["n_max"]):
        g = grad_energy(p, h, yhat)  # depends on p -> second-order terms in the outer gradient
        active = (i < nsteps).astype(jnp.float32)[:, None]
        last = (i == nsteps - 1).astype(jnp.float32)[:, None]
        step = -alpha[:, None] * g + (1.0 - last) * noise[i]
        yhat = yhat + active * step
    EN = energy(p, h, yhat)
    loss = ce(yhat, y).mean()
    return loss, (yhat, E0.mean(), EN.mean())


def base_loss(p, ctx, y):
    return ce(base_logits(p, ctx), y).mean(), None


def adamw_init(p):
    z = jax.tree_util.tree_map(jnp.zeros_like, p)
    return {"m": z, "v": jax.tree_util.tree_map(jnp.zeros_like, p), "t": jnp.array(0)}


def adamw_update(p, g, st, lr, cfg, b1=0.9, b2=0.999, eps=1e-8):
    gn = jnp.sqrt(sum(jnp.sum(x * x) for x in jax.tree_util.tree_leaves(g)))
    scale = jnp.minimum(1.0, cfg["clip"] / (gn + 1e-6))
    g = jax.tree_util.tree_map(lambda x: x * scale, g)
    t = st["t"] + 1
    m = jax.tree_util.tree_map(lambda m, x: b1 * m + (1 - b1) * x, st["m"], g)
    v = jax.tree_util.tree_map(lambda v, x: b2 * v + (1 - b2) * x * x, st["v"], g)
    mh = jax.tree_util.tree_map(lambda m: m / (1 - b1 ** t), m)
    vh = jax.tree_util.tree_map(lambda v: v / (1 - b2 ** t), v)
    newp = {}
    for k in p:
        decay = cfg["wd"] if p[k].ndim == 2 else 0.0
        newp[k] = p[k] - lr * (mh[k] / (jnp.sqrt(vh[k]) + eps) + decay * p[k])
    return newp, {"m": m, "v": v, "t": t}, gn


def lr_at(step, total, cfg):
    if step < cfg["warmup"]:
        return cfg["lr"] * (step + 1) / cfg["warmup"]
    prog = (step - cfg["warmup"]) / max(1, total - cfg["warmup"])
    return cfg["lr"] * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))


# ---------------------------------------------------------------- inference helpers (Algorithm 2)

def think_trace(p, h, yhat0, alpha, nsteps, noise_std=0.0, key=None):
    """Run `nsteps` gradient steps. Returns yhat trajectory (nsteps+1, B, V) and energies (nsteps+1, B)."""
    def body(carry, k):
        yhat = carry
        E = energy(p, h, yhat)
        g = grad_energy(p, h, yhat)
        nxt = yhat - alpha[:, None] * g
        if noise_std > 0:
            nxt = nxt + noise_std * jax.random.normal(k, yhat.shape)
        return nxt, (yhat, E)
    keys = jax.random.split(key if key is not None else jax.random.PRNGKey(0), nsteps)
    last, (ys, Es) = jax.lax.scan(body, yhat0, keys)
    ElastV = energy(p, h, last)
    ys = jnp.concatenate([ys, last[None]], 0)
    Es = jnp.concatenate([Es, ElastV[None]], 0)
    return ys, Es


def batches(arr, C, B, rng):
    idx = rng.integers(C, len(arr), size=B)
    ctx = arr[idx[:, None] + np.arange(-C, 0)[None, :]]
    return idx, ctx.astype(np.int32), arr[idx].astype(np.int32)


def fixed_eval_set(arr, C, n, seed):
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(np.arange(C, len(arr)), size=n, replace=False))
    ctx = arr[idx[:, None] + np.arange(-C, 0)[None, :]]
    return idx, ctx.astype(np.int32), arr[idx].astype(np.int32)


def save_params(p, path):
    np.savez(path, **{k: np.asarray(v) for k, v in p.items()})


def load_params(path):
    d = np.load(path)
    return {k: jnp.asarray(d[k]) for k in d.files}


def train(kind, work, total, alpha0=None, tag=None):
    cfg = dict(CFG)
    if alpha0 is not None:
        cfg["alpha0"] = alpha0
    tag = tag or kind
    train_arr = np.load(os.path.join(work, "train.npy"))
    val_arr = np.load(os.path.join(work, "val.npy"))
    C, B = cfg["C"], cfg["batch"]
    rng = np.random.default_rng(0)
    key = jax.random.PRNGKey(0)
    p = init_ebt(key, cfg) if kind == "ebt" else init_base(key, cfg)
    print(tag, "params", n_params(p), flush=True)
    st = adamw_init(p)
    _, vctx, vy = fixed_eval_set(val_arr, C, 4096, 123)
    _, tctx, ty = fixed_eval_set(train_arr, C, 4096, 321)
    vctx, vy, tctx, ty = map(jnp.asarray, (vctx, vy, tctx, ty))
    nmax = cfg["n_max"]

    if kind == "ebt":
        @jax.jit
        def step_fn(p, st, ctx, y, yhat0, alpha, nsteps, noise, lr):
            (loss, (yN, E0, EN)), g = jax.value_and_grad(ebt_loss, has_aux=True)(
                p, ctx, y, yhat0, alpha, nsteps, noise, cfg)
            p, st, gn = adamw_update(p, g, st, lr, cfg)
            return p, st, loss, yN, E0, EN, gn

        @jax.jit
        def eval_fn(p, ctx, y):
            # deterministic, training-like eval: fixed alpha0, n_max steps, no noise, fixed yhat0
            h = encode_ctx(p, ctx)
            y0 = jax.random.normal(jax.random.PRNGKey(7), (ctx.shape[0], V))
            al = jnp.full((ctx.shape[0],), cfg["alpha0"])
            ys, Es = think_trace(p, h, y0, al, nmax)
            return ce(ys[-1], y).mean(), Es[0].mean(), Es[-1].mean()
    else:
        @jax.jit
        def step_fn(p, st, ctx, y, lr):
            (loss, _), g = jax.value_and_grad(base_loss, has_aux=True)(p, ctx, y)
            p, st, gn = adamw_update(p, g, st, lr, cfg)
            return p, st, loss, gn

        @jax.jit
        def eval_fn(p, ctx, y):
            return ce(base_logits(p, ctx), y).mean(), 0.0, 0.0

    # replay buffer (host side): positions + last predictions
    R = cfg["replay_size"]
    buf_pos = np.zeros(R, dtype=np.int64)
    buf_y = np.zeros((R, V), dtype=np.float32)
    buf_n = 0
    n_rep = int(cfg["replay_frac"] * B)
    log = {"step": [], "train_loss": [], "val_loss": [], "train_eval_loss": [], "E0": [], "EN": [], "time": []}
    ema = None
    t0 = time.time()
    for s in range(total):
        lr = lr_at(s, total, cfg)
        if kind == "ebt":
            idx, ctx, y = batches(train_arr, C, B, rng)
            yhat0 = rng.standard_normal((B, V)).astype(np.float32)
            if buf_n >= R and n_rep > 0:
                ri = rng.integers(0, R, size=n_rep)
                idx[:n_rep] = buf_pos[ri]
                ctx[:n_rep] = train_arr[buf_pos[ri][:, None] + np.arange(-C, 0)[None, :]]
                y[:n_rep] = train_arr[buf_pos[ri]]
                yhat0[:n_rep] = buf_y[ri]
            alpha = (cfg["alpha0"] * np.exp(rng.uniform(-math.log(cfg["alpha_rand"]), math.log(cfg["alpha_rand"]), B))).astype(np.float32)
            nsteps = rng.integers(cfg["n_min"], cfg["n_max"] + 1, size=B).astype(np.int32)
            noise = (cfg["langevin"] * rng.standard_normal((nmax, B, V))).astype(np.float32)
            p, st, loss, yN, E0, EN, gn = step_fn(p, st, ctx, y, yhat0, alpha, nsteps, noise, lr)
            # write back: replayed rows continue their trajectory, fresh rows are inserted
            yN = np.asarray(yN)
            if buf_n < R:
                k = min(B, R - buf_n)
                buf_pos[buf_n:buf_n + k] = idx[:k]
                buf_y[buf_n:buf_n + k] = yN[:k]
                buf_n += k
            else:
                buf_y[ri] = yN[:n_rep]
                wi = rng.integers(0, R, size=B - n_rep)
                buf_pos[wi] = idx[n_rep:]
                buf_y[wi] = yN[n_rep:]
        else:
            idx, ctx, y = batches(train_arr, C, B, rng)
            p, st, loss, gn = step_fn(p, st, ctx, y, lr)
        loss = float(loss)
        ema = loss if ema is None else 0.98 * ema + 0.02 * loss
        if s % 100 == 0 or s == total - 1:
            vl, vE0, vEN = eval_fn(p, vctx, vy)
            tl, _, _ = eval_fn(p, tctx, ty)
            log["step"].append(s)
            log["train_loss"].append(ema)
            log["val_loss"].append(float(vl))
            log["train_eval_loss"].append(float(tl))
            log["E0"].append(float(vE0))
            log["EN"].append(float(vEN))
            log["time"].append(time.time() - t0)
            print(f"{tag} step {s} lr {lr:.2e} train(ema) {ema:.4f} train_eval {float(tl):.4f} val {float(vl):.4f} "
                  f"E0 {float(vE0):.3f} EN {float(vEN):.3f} gnorm {float(gn):.3f} t {time.time()-t0:.1f}s", flush=True)
    save_params(p, os.path.join(work, f"params_{tag}.npz"))
    json.dump({"cfg": cfg, "log": log, "n_params": n_params(p), "steps": total,
               "wall_s": time.time() - t0}, open(os.path.join(work, f"log_{tag}.json"), "w"))
    return p


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "train":
        kind, work = sys.argv[2], sys.argv[3]
        total = int(sys.argv[4]) if len(sys.argv) > 4 else 3000
        a0 = float(sys.argv[5]) if len(sys.argv) > 5 else None
        tag = sys.argv[6] if len(sys.argv) > 6 else None
        for kv in sys.argv[7:]:
            k_, v_ = kv.split("=")
            CFG[k_] = type(CFG[k_])(float(v_)) if not isinstance(CFG[k_], int) else int(float(v_))
        train(kind, work, total, a0, tag)
