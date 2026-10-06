"""Train the 2D toy EBM exactly in the style of Algorithm 1 + Sec 3.3 of Gladstone et al. 2025.

  yhat_0 ~ N(0, s^2 I)   (or, with prob P_REPLAY, a stored previous prediction from the replay buffer)
  for i < N:  yhat <- yhat - alpha * grad_yhat E(x, yhat) + eta,  eta ~ N(0, LANGEVIN^2 I)
  loss = mean ||yhat_N - y||^2      (MSE, only at the last step; no detach between steps)
  backprop through the whole unrolled optimization (second order, jax.grad of a function using jax.grad)
  alpha randomized per sample (paper App. I.1.3), N randomized per batch, Adam on theta.

Usage: python3 src/toy2d_train.py [total_steps] [out_dir]
"""
import os
import sys
import time

os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import jax
import jax.numpy as jnp
from toy2d_common import (A_X, A_Y, S_MIN, S_MAX, K_FEAT, LAMBDA, HIDDEN, IN_DIM, INIT_SCALE)

HP = dict(
    batch=256,
    total_steps=int(sys.argv[1]) if len(sys.argv) > 1 else 8000,
    lr=2e-3, lr_min=1e-4, warmup=200, adam_b1=0.9, adam_b2=0.999, adam_eps=1e-8, grad_clip=1.0,
    alpha0=1.0,            # base GD step size on yhat
    alpha_rand_factor=2.0,  # alpha ~ alpha0 * exp(U[-ln f, ln f]) independently per sample
    n_min=2, n_max=6,       # number of unrolled steps N ~ U{n_min..n_max} per batch
    langevin=0.05,          # std of eta added at every training step
    p_replay=0.25,          # prob. a batch element restarts from a replay-buffer prediction
    buffer_size=8192,
    init_scale=INIT_SCALE,
    seed=0,
)
OUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/claude-0/-home-user-Random/9843a8a1-7b0a-5ed7-8867-6762f4a2ea11/scratchpad/toy2d"
os.makedirs(OUT, exist_ok=True)


def ckpt_steps(total):
    base = [0, 10, 25, 50, 100, 200, 400, 800, 1500, 3000, 5000]
    s = [b for b in base if b < total * 0.7]
    s.append(total)
    # fill to 12 with a geometric point if needed
    while len(s) < 12:
        s.insert(-1, int((s[-2] * s[-1]) ** 0.5))
    return sorted(set(s))


def mu(x):
    return jnp.stack([A_X * jnp.sin(2 * jnp.pi * x), A_Y * jnp.sin(4 * jnp.pi * x)], -1)


def sigma(x):
    return S_MIN + (S_MAX - S_MIN) * jnp.sin(jnp.pi * (x - 0.25)) ** 2


def feats(x):
    out = []
    for k in range(1, K_FEAT + 1):
        out += [jnp.sin(2 * jnp.pi * k * x), jnp.cos(2 * jnp.pi * k * x)]
    return jnp.stack(out, -1)


def init_params(key):
    sizes = [IN_DIM] + HIDDEN + [1]
    params = []
    for i in range(len(sizes) - 1):
        key, k = jax.random.split(key)
        W = jax.random.normal(k, (sizes[i + 1], sizes[i]), jnp.float32) * jnp.sqrt(1.0 / sizes[i])
        b = jnp.zeros((sizes[i + 1],), jnp.float32)
        params.append((W, b))
    return params


def energy_single(params, f, yh):
    h = jnp.concatenate([f, yh])
    for W, b in params[:-1]:
        h = jax.nn.silu(W @ h + b)
    W, b = params[-1]
    return (W @ h + b)[0] + LAMBDA * jnp.sum(yh ** 2)


grad_y_single = jax.grad(energy_single, argnums=2)
grad_y = jax.vmap(grad_y_single, in_axes=(None, 0, 0))
energy = jax.vmap(energy_single, in_axes=(None, 0, 0))


def unroll(params, f, y0, alphas, nsteps, noise):
    """Masked unrolled GD on yhat; steps i >= nsteps are no-ops. noise: [n_max, B, 2]."""
    def body(yh, inp):
        i, eta = inp
        new = yh - alphas[:, None] * grad_y(params, f, yh) + eta
        return jnp.where(i < nsteps, new, yh), None
    yN, _ = jax.lax.scan(body, y0, (jnp.arange(noise.shape[0]), noise))
    return yN


def loss_fn(params, f, y0, y, alphas, nsteps, noise):
    yN = unroll(params, f, y0, alphas, nsteps, noise)
    return jnp.mean(jnp.sum((yN - y) ** 2, -1)), yN


def lr_at(t):
    t = t.astype(jnp.float32)
    warm = HP["lr"] * (t + 1) / HP["warmup"]
    prog = jnp.clip((t - HP["warmup"]) / max(1, HP["total_steps"] - HP["warmup"]), 0, 1)
    cos = HP["lr_min"] + 0.5 * (HP["lr"] - HP["lr_min"]) * (1 + jnp.cos(jnp.pi * prog))
    return jnp.where(t < HP["warmup"], warm, cos)


@jax.jit
def train_step(state, key):
    params, m, v, t, bx, by, byh = state
    B, nmax = HP["batch"], HP["n_max"]
    ks = jax.random.split(key, 9)
    x = jax.random.uniform(ks[0], (B,))
    y = mu(x) + sigma(x)[:, None] * jax.random.normal(ks[1], (B, 2))
    y0 = HP["init_scale"] * jax.random.normal(ks[2], (B, 2))
    # replay buffer: with prob p, continue optimizing a stored previous prediction (same x, y)
    use = jax.random.uniform(ks[3], (B,)) < HP["p_replay"]
    idx = jax.random.randint(ks[4], (B,), 0, HP["buffer_size"])
    x = jnp.where(use, bx[idx], x)
    y = jnp.where(use[:, None], by[idx], y)
    y0 = jnp.where(use[:, None], byh[idx], y0)
    lf = jnp.log(HP["alpha_rand_factor"])
    alphas = HP["alpha0"] * jnp.exp(jax.random.uniform(ks[5], (B,), minval=-lf, maxval=lf))
    nsteps = jax.random.randint(ks[6], (), HP["n_min"], HP["n_max"] + 1)
    noise = HP["langevin"] * jax.random.normal(ks[7], (nmax, B, 2))
    f = feats(x)
    (loss, yN), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, f, y0, y, alphas, nsteps, noise)
    # global-norm clip
    gn = jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(grads)))
    scale = jnp.minimum(1.0, HP["grad_clip"] / (gn + 1e-9))
    grads = jax.tree_util.tree_map(lambda g: g * scale, grads)
    # Adam
    t1 = t + 1
    b1, b2 = HP["adam_b1"], HP["adam_b2"]
    m = jax.tree_util.tree_map(lambda a, g: b1 * a + (1 - b1) * g, m, grads)
    v = jax.tree_util.tree_map(lambda a, g: b2 * a + (1 - b2) * g * g, v, grads)
    lr = lr_at(t)
    c1, c2 = 1 - b1 ** t1.astype(jnp.float32), 1 - b2 ** t1.astype(jnp.float32)
    params = jax.tree_util.tree_map(lambda p, a, s: p - lr * (a / c1) / (jnp.sqrt(s / c2) + HP["adam_eps"]), params, m, v)
    # write this batch's final predictions back into the ring buffer
    pos = (t * B + jnp.arange(B)) % HP["buffer_size"]
    bx = bx.at[pos].set(x)
    by = by.at[pos].set(y)
    byh = byh.at[pos].set(jnp.clip(jax.lax.stop_gradient(yN), -4.0, 4.0))
    return (params, m, v, t1, bx, by, byh), (loss, gn, nsteps)


def main():
    key = jax.random.PRNGKey(HP["seed"])
    key, kp, kb1, kb2, kb3 = jax.random.split(key, 5)
    params = init_params(kp)
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    S = HP["buffer_size"]
    bx = jax.random.uniform(kb1, (S,))
    by = mu(bx) + sigma(bx)[:, None] * jax.random.normal(kb2, (S, 2))
    byh = HP["init_scale"] * jax.random.normal(kb3, (S, 2))   # buffer starts as pure noise inits
    state = (params, zeros, zeros, jnp.array(0, jnp.int32), bx, by, byh)
    cks = ckpt_steps(HP["total_steps"])
    saved = {}
    losses, gnorms = [], []
    t0 = time.time()
    for step in range(HP["total_steps"] + 1):
        if step in cks:
            saved[step] = [(np.asarray(W, np.float64), np.asarray(b, np.float64)) for W, b in state[0]]
        if step == HP["total_steps"]:
            break
        key, k = jax.random.split(key)
        state, (loss, gn, n) = train_step(state, k)
        losses.append(loss)
        gnorms.append(gn)
        if step % 500 == 0 or step == 20:
            print(f"step {step:5d} loss {float(loss):.4f} gnorm {float(gn):.3f} N {int(n)} "
                  f"t {time.time() - t0:.1f}s", flush=True)
    losses = np.array([float(l) for l in losses])
    gnorms = np.array([float(g) for g in gnorms])
    wall = time.time() - t0
    print(f"done in {wall:.1f}s; final loss (mean last 200) {losses[-200:].mean():.4f}")
    np.savez(os.path.join(OUT, "train_out.npz"), losses=losses, gnorms=gnorms, wall=wall,
             ckpt_steps=np.array(sorted(saved)),
             **{f"c{s}_W{i}": W for s in saved for i, (W, b) in enumerate(saved[s])},
             **{f"c{s}_b{i}": b for s in saved for i, (W, b) in enumerate(saved[s])})
    # buffer snapshot for the replay-buffer demo
    np.savez(os.path.join(OUT, "buffer.npz"), x=np.asarray(state[4]), y=np.asarray(state[5]), yh=np.asarray(state[6]))
    import json
    with open(os.path.join(OUT, "hparams.json"), "w") as fh:
        json.dump(dict(HP, wall_seconds=wall, ckpt_steps=sorted(saved)), fh)


if __name__ == "__main__":
    main()
