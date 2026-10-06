"""Export data/toy2d.json from the checkpoints written by toy2d_train.py.

All exported quantities (grids, trajectories, BoN, basin stats, ...) are computed in float64 from the
ROUNDED weights that are stored in the JSON, so a browser recomputing E and grad_yhat E from the JSON
reproduces them. Usage: python3 src/toy2d_export.py [train_dir]
"""
import os
import sys
import json
import math

os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from toy2d_common import (A_X, A_Y, S_MIN, S_MAX, K_FEAT, LAMBDA, HIDDEN, IN_DIM, EXTENT, GRID_N,
                          INIT_SCALE, mu_np, sigma_np, feats_np, sig4, rnd_e, round_params)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = sys.argv[1] if len(sys.argv) > 1 else "/tmp/claude-0/-home-user-Random/9843a8a1-7b0a-5ed7-8867-6762f4a2ea11/scratchpad/toy2d"
OUT_JSON = os.path.join(ROOT, "data", "toy2d.json")

HP = json.load(open(os.path.join(TRAIN, "hparams.json")))
Z = np.load(os.path.join(TRAIN, "train_out.npz"))
CK = [int(s) for s in Z["ckpt_steps"]]
NL = len(HIDDEN) + 1
PARAMS = {s: round_params([(Z[f"c{s}_W{i}"], Z[f"c{s}_b{i}"]) for i in range(NL)]) for s in CK}
FINAL = CK[-1]
ALPHA0 = HP["alpha0"]
LANG = HP["langevin"]


def to_jax(p):
    return [(jnp.asarray(W), jnp.asarray(b)) for W, b in p]


def feats(x):
    out = []
    for k in range(1, K_FEAT + 1):
        out += [jnp.sin(2 * jnp.pi * k * x), jnp.cos(2 * jnp.pi * k * x)]
    return jnp.stack(out, -1)


def energy_single(params, f, yh):
    h = jnp.concatenate([f, yh])
    for W, b in params[:-1]:
        h = jax.nn.silu(W @ h + b)
    W, b = params[-1]
    return (W @ h + b)[0] + LAMBDA * jnp.sum(yh ** 2)


E_b = jax.jit(jax.vmap(energy_single, in_axes=(None, 0, 0)))
G_b = jax.jit(jax.vmap(jax.grad(energy_single, argnums=2), in_axes=(None, 0, 0)))
H_1 = jax.jit(jax.hessian(energy_single, argnums=2))


def gd_paths(params, f, y0, alphas, nsteps, noise_std=0.0, key=None):
    """Plain GD (+ optional Langevin) on yhat. f [B,2K], y0 [B,2]. Returns paths [B,n+1,2], E [B,n+1]."""
    ys = [y0]
    es = [E_b(params, f, y0)]
    y = y0
    for i in range(nsteps):
        y = y - alphas[:, None] * G_b(params, f, y)
        if noise_std > 0:
            key, k = jax.random.split(key)
            y = y + noise_std * jax.random.normal(k, y.shape)
        ys.append(y)
        es.append(E_b(params, f, y))
    return np.stack([np.asarray(a) for a in ys], 1), np.stack([np.asarray(a) for a in es], 1)


rng = np.random.default_rng(1234)

# ------------------------------------------------------------------ contexts
CTX = [
    (0.25, "right tip: near-deterministic (sigma 0.03)", "low_noise"),
    (0.75, "left tip: very noisy (sigma 0.30)", "high_noise"),
    (0.0, "crossing, moving up-right (sigma 0.165)", "mid_noise"),
    (0.5, "crossing, moving down-left (sigma 0.165)", "mid_noise"),
    (0.125, "upper-right lobe", "lobe"),
    (0.375, "lower-right lobe", "lobe"),
    (0.625, "upper-left lobe", "lobe"),
    (0.875, "lower-left lobe", "lobe"),
]
contexts = []
for x, label, role in CTX:
    contexts.append(dict(x=x, label=label, role=role, features=sig4(feats_np(x)),
                         mu=sig4(mu_np(x)), sigma=sig4(sigma_np(x))))
FOCUS = [0, 1]   # all checkpoints for these
CK_SUB = [CK[0], CK[3], CK[6], CK[-1]]  # 4 checkpoints for all contexts

# ------------------------------------------------------------------ grids
xs = np.linspace(EXTENT[0], EXTENT[1], GRID_N)
ys = np.linspace(EXTENT[2], EXTENT[3], GRID_N)
GX, GY = np.meshgrid(xs, ys)  # GX[r,c] = xs[c], GY[r,c] = ys[r]
GPTS = np.stack([GX.ravel(), GY.ravel()], -1)


def grid_E(params, x):
    f = np.repeat(feats_np(x)[None], GPTS.shape[0], 0)
    return np.asarray(E_b(params, jnp.asarray(f), jnp.asarray(GPTS))).reshape(GRID_N, GRID_N)


grids = []
for ci, c in enumerate(contexts):
    steps = CK if ci in FOCUS else CK_SUB
    for s in steps:
        E = grid_E(to_jax(PARAMS[s]), c["x"])
        grids.append(dict(ctx=ci, step=s, ckpt=CK.index(s), E=rnd_e(E)))
print("grids:", len(grids))

# ------------------------------------------------------------------ trajectories (final model)
PF = to_jax(PARAMS[FINAL])
N_START, N_STEP = 8, 40
STARTS = np.array(sig4(np.clip(INIT_SCALE * rng.standard_normal((N_START, 2)), -2.4, 2.4)))
key = jax.random.PRNGKey(7)
trajectories = []
for ci, c in enumerate(contexts):
    f = jnp.asarray(np.repeat(feats_np(c["x"])[None], N_START, 0))
    a = jnp.full((N_START,), ALPHA0)
    P, E = gd_paths(PF, f, jnp.asarray(STARTS), a, N_STEP)
    key, k = jax.random.split(key)
    PL, EL = gd_paths(PF, f, jnp.asarray(STARTS), a, N_STEP, noise_std=LANG, key=k)
    trajectories.append(dict(
        ctx=ci, alpha=ALPHA0, steps=N_STEP, langevin_sigma=LANG,
        gd=[dict(path=sig4(P[j]), E=rnd_e(E[j])) for j in range(N_START)],
        langevin=[dict(path=sig4(PL[j]), E=rnd_e(EL[j])) for j in range(N_START)],
    ))

# trajectories for the focus contexts at every checkpoint (plain GD) -> "watch the landscape learn"
traj_by_ckpt = []
for ci in FOCUS:
    f = jnp.asarray(np.repeat(feats_np(contexts[ci]["x"])[None], N_START, 0))
    for s in CK:
        P, E = gd_paths(to_jax(PARAMS[s]), f, jnp.asarray(STARTS), jnp.full((N_START,), ALPHA0), N_STEP)
        traj_by_ckpt.append(dict(ctx=ci, step=s, ckpt=CK.index(s),
                                 paths=[sig4(P[j]) for j in range(N_START)], E=[rnd_e(E[j]) for j in range(N_START)]))

# ------------------------------------------------------------------ eval set (fixed) + metrics
NV = 1024
xv = rng.random(NV)
yv = mu_np(xv) + sigma_np(xv)[:, None] * rng.standard_normal((NV, 2))
y0v = INIT_SCALE * rng.standard_normal((NV, 2))
fv = jnp.asarray(feats_np(xv))
muv = mu_np(xv)
noise_floor = float(np.mean(np.sum((yv - muv) ** 2, -1)))


def eval_ckpt(params, n_steps, alpha=ALPHA0, noise_std=0.0, key=None, y0=y0v):
    P, E = gd_paths(params, fv, jnp.asarray(y0), jnp.full((NV,), alpha), n_steps, noise_std, key)
    yN = P[:, -1]
    return P, E, float(np.mean(np.sum((yN - yv) ** 2, -1))), float(np.mean(np.sum((yN - muv) ** 2, -1)))


eval_curve = dict(step=[], mse_vs_y_N4=[], mse_vs_mu_N4=[], mse_vs_mu_N40=[], mean_final_E_N40=[])
for s in CK:
    p = to_jax(PARAMS[s])
    _, _, a4, b4 = eval_ckpt(p, 4)
    _, E40, _, b40 = eval_ckpt(p, 40)
    eval_curve["step"].append(s)
    eval_curve["mse_vs_y_N4"].append(sig4(a4))
    eval_curve["mse_vs_mu_N4"].append(sig4(b4))
    eval_curve["mse_vs_mu_N40"].append(sig4(min(b40, 1e6)))
    eval_curve["mean_final_E_N40"].append(sig4(np.mean(E40[:, -1])))

# thinking curve: final model, error vs number of GD steps (no Langevin)
P, E, _, _ = eval_ckpt(PF, 60)
th_mse_mu = np.mean(np.sum((P - muv[:, None]) ** 2, -1), 0)
th_mse_y = np.mean(np.sum((P - yv[:, None]) ** 2, -1), 0)
thinking_curve = dict(steps=list(range(61)), mse_vs_mu=sig4(th_mse_mu), mse_vs_y=sig4(th_mse_y),
                      mean_E=sig4(E.mean(0)), alpha=ALPHA0, train_N_range=[HP["n_min"], HP["n_max"]])
# thinking curve at a few checkpoints (does "thinking longer" get better as training goes on?)
thinking_by_ckpt = []
for s in CK[4:]:
    Pk, Ek, _, _ = eval_ckpt(to_jax(PARAMS[s]), 40)
    thinking_by_ckpt.append(dict(step=s, mse_vs_mu=sig4(np.minimum(np.mean(np.sum((Pk - muv[:, None]) ** 2, -1), 0), 1e6))))

# ------------------------------------------------------------------ Best-of-N (Algorithm 2) with Langevin
BON_N, BON_SIGMA = 6, LANG


def bon_eval(params, Ms, key, n_rep=1, nsteps=BON_N):
    """For each eval context, run max(M) candidates (random start, random alpha as in training, Langevin),
    pick argmin energy among the first M. Returns mean ||yhat* - mu||^2 for each M and for a random pick."""
    Mmax = max(Ms)
    finals, energies = [], []
    for j in range(Mmax):
        key, k1, k2, k3 = jax.random.split(key, 4)
        y0 = INIT_SCALE * jax.random.normal(k1, (NV, 2))
        lf = math.log(HP["alpha_rand_factor"])
        al = ALPHA0 * jnp.exp(jax.random.uniform(k2, (NV,), minval=-lf, maxval=lf))
        Pj, Ej = gd_paths(params, fv, y0, al, nsteps, BON_SIGMA, k3)
        finals.append(Pj[:, -1])
        energies.append(Ej[:, -1])
    finals = np.stack(finals, 1)
    energies = np.stack(energies, 1)
    d2 = np.sum((finals - muv[:, None]) ** 2, -1)
    out = {}
    for M in Ms:
        idx = np.argmin(energies[:, :M], 1)
        out[M] = float(np.mean(d2[np.arange(NV), idx]))
    out["oracle_best_of_max"] = float(np.mean(d2.min(1)))
    return out


Ms = [1, 2, 4, 8, 16]
bon_final = bon_eval(PF, Ms, jax.random.PRNGKey(11))
bon_vs_training = dict(step=[], M1=[], M8=[])
for s in CK[2:]:
    r = bon_eval(to_jax(PARAMS[s]), [1, 8], jax.random.PRNGKey(12))
    bon_vs_training["step"].append(s)
    bon_vs_training["M1"].append(sig4(min(r[1], 1e6)))
    bon_vs_training["M8"].append(sig4(min(r[8], 1e6)))

# single worked example for the noisy left tip
ci_ex = 1
M_EX = 8
fx = jnp.asarray(np.repeat(feats_np(contexts[ci_ex]["x"])[None], M_EX, 0))
k1, k2, k3 = jax.random.split(jax.random.PRNGKey(21), 3)
y0 = np.array(sig4(np.asarray(INIT_SCALE * jax.random.normal(k1, (M_EX, 2)))))
lf = math.log(HP["alpha_rand_factor"])
al = np.array(sig4(np.asarray(ALPHA0 * jnp.exp(jax.random.uniform(k2, (M_EX,), minval=-lf, maxval=lf)))))
Pex, Eex = gd_paths(PF, fx, jnp.asarray(y0), jnp.asarray(al), BON_N, BON_SIGMA, k3)
mu_ex = mu_np(contexts[ci_ex]["x"])
dist_ex = np.linalg.norm(Pex[:, -1] - mu_ex, axis=-1)
chosen = int(np.argmin(Eex[:, -1]))
bon = dict(
    ctx=ci_ex, M=M_EX, steps=BON_N, langevin_sigma=BON_SIGMA, alphas=sig4(al),
    candidates=[dict(path=sig4(Pex[j]), E=sig4(Eex[j]), final=sig4(Pex[j, -1]), final_E=sig4(Eex[j, -1]),
                     dist_to_mu=sig4(dist_ex[j])) for j in range(M_EX)],
    chosen=chosen, chosen_dist_to_mu=sig4(dist_ex[chosen]), mean_dist_to_mu=sig4(dist_ex.mean()),
    aggregate=dict(M=Ms, mse_vs_mu=[sig4(bon_final[M]) for M in Ms],
                   oracle_best_of_16=sig4(bon_final["oracle_best_of_max"]),
                   note=f"mean ||yhat* - mu(x)||^2 over {NV} eval contexts; each candidate: random start, "
                        f"alpha randomized as in training, {BON_N} steps, Langevin sigma {BON_SIGMA}; "
                        f"yhat* = lowest-energy candidate among the first M (Algorithm 2)."),
)

# ------------------------------------------------------------------ basin statistics vs noise (Facet 2 check)
NB = 64
xb = (np.arange(NB) + 0.5) / NB
fb = jnp.asarray(feats_np(xb))
Pm, Em = gd_paths(PF, fb, jnp.asarray(mu_np(xb)), jnp.full((NB,), 0.5), 300)   # descend from the truth to the min
ymin = Pm[:, -1]
hess = np.array([np.asarray(H_1(PF, fb[i], jnp.asarray(ymin[i]))) for i in range(NB)])
eig = np.linalg.eigvalsh(hess)
# steps to converge from 16 fixed random starts (|E_i - E_min| < 1e-3)
S16 = INIT_SCALE * rng.standard_normal((16, 2))
conv = []
for i in range(NB):
    f16 = jnp.asarray(np.repeat(feats_np(xb[i])[None], 16, 0))
    P16, E16 = gd_paths(PF, f16, jnp.asarray(S16), jnp.full((16,), ALPHA0), 60)
    within = np.abs(E16 - Em[i, -1]) < 1e-3
    first = np.where(within.any(1), within.argmax(1), 61)
    conv.append(float(np.mean(first)))
sig_b = sigma_np(xb)
basin = dict(
    x=sig4(xb), sigma=sig4(sig_b), E_min=sig4(Em[:, -1]), argmin=sig4(ymin), dist_min_to_mu=sig4(np.linalg.norm(ymin - mu_np(xb), axis=1)),
    hess_eig_small=sig4(eig[:, 0]), hess_eig_large=sig4(eig[:, 1]), steps_to_converge=sig4(np.array(conv)),
    corr_sigma_vs_Emin=sig4(np.corrcoef(sig_b, Em[:, -1])[0, 1]),
    corr_sigma_vs_mean_curvature=sig4(np.corrcoef(sig_b, eig.mean(1))[0, 1]),
    corr_sigma_vs_steps=sig4(np.corrcoef(sig_b, conv)[0, 1]),
    note="Final model. argmin found by 300 GD steps (alpha 0.5) started at mu(x). Curvature = eigenvalues of the "
         "2x2 Hessian of E w.r.t. yhat at the argmin. steps_to_converge: mean over 16 fixed random starts of the first "
         "GD step (alpha0) whose energy is within 1e-3 of E_min.",
)

# ------------------------------------------------------------------ replay buffer demo
# A prediction is stored after a short training-style run; when it is drawn again it continues from where it
# stopped, so short unrolls chain into a long trajectory (Sec 3.3). Shown at a mid checkpoint and at the end.
def buffer_demo(params, ci, seed):
    r = np.random.default_rng(seed)
    f = jnp.asarray(feats_np(contexts[ci]["x"])[None])
    y = np.array(sig4(INIT_SCALE * r.standard_normal((1, 2))))
    segs = []
    for _ in range(4):
        n = int(r.integers(HP["n_min"], HP["n_max"] + 1))
        a = float(sig4(ALPHA0 * math.exp(r.uniform(-math.log(HP["alpha_rand_factor"]), math.log(HP["alpha_rand_factor"])))))
        P, E = gd_paths(params, f, jnp.asarray(y), jnp.array([a]), n)
        segs.append(dict(N=n, alpha=a, path=sig4(P[0]), E=sig4(E[0])))
        y = np.array(sig4(P[:, -1]))
    return segs


mid_ck = CK[len(CK) // 2]
replay_buffer_demo = dict(
    note="Each segment is one training-style unroll (random N and alpha, no Langevin here for clarity). "
         "Its end point is stored in the replay buffer; the next segment starts from it. "
         "Training only ever unrolls N<=%d steps, but chained segments simulate a long trajectory." % HP["n_max"],
    examples=[dict(ctx=ci, step=s, ckpt=CK.index(s), segments=buffer_demo(to_jax(PARAMS[s]), ci, 100 + ci))
              for s in (mid_ck, FINAL) for ci in FOCUS],
)

# ------------------------------------------------------------------ parameter-space loss landscape
# Same training objective, fixed batch, fixed init noise, fixed alphas, fixed N, fixed Langevin draws, no replay.
PL_B, PL_N = 512, 4
r = np.random.default_rng(99)
xp = r.random(PL_B)
yp = mu_np(xp) + sigma_np(xp)[:, None] * r.standard_normal((PL_B, 2))
y0p = INIT_SCALE * r.standard_normal((PL_B, 2))
alp = ALPHA0 * np.exp(r.uniform(-math.log(HP["alpha_rand_factor"]), math.log(HP["alpha_rand_factor"]), PL_B))
noisep = LANG * r.standard_normal((PL_N, PL_B, 2))
fp = jnp.asarray(feats_np(xp))


def train_loss(params):
    y = jnp.asarray(y0p)
    for i in range(PL_N):
        y = y - jnp.asarray(alp)[:, None] * G_b_raw(params, fp, y) + jnp.asarray(noisep[i])
    return jnp.mean(jnp.sum((y - jnp.asarray(yp)) ** 2, -1))


G_b_raw = jax.vmap(jax.grad(energy_single, argnums=2), in_axes=(None, 0, 0))
train_loss_j = jax.jit(train_loss)


def filt_norm_dir(params, seed):
    r = np.random.default_rng(seed)
    d = []
    for W, b in params:
        D = r.standard_normal(W.shape)
        D = D * (np.linalg.norm(W, axis=1, keepdims=True) / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-12))
        d.append((D, np.zeros_like(b)))
    return d


PFn = PARAMS[FINAL]
d1, d2 = filt_norm_dir(PFn, 1), filt_norm_dir(PFn, 2)
coords = np.linspace(-1, 1, 41)
L = np.zeros((41, 41))
for i, bcoef in enumerate(coords):       # rows: direction 2
    for j, acoef in enumerate(coords):   # cols: direction 1
        p = [(jnp.asarray(W + acoef * D1 + bcoef * D2), jnp.asarray(b)) for (W, b), (D1, _), (D2, _) in zip(PFn, d1, d2)]
        L[i, j] = float(train_loss_j(p))
center_loss = float(train_loss_j(to_jax(PFn)))
# also the same slice at the initial weights, for contrast (scaled to the init weights' own filter norms)
P0n = PARAMS[CK[0]]
e1, e2 = filt_norm_dir(P0n, 1), filt_norm_dir(P0n, 2)
L0 = np.zeros((41, 41))
for i, bcoef in enumerate(coords):
    for j, acoef in enumerate(coords):
        p = [(jnp.asarray(W + acoef * D1 + bcoef * D2), jnp.asarray(b)) for (W, b), (D1, _), (D2, _) in zip(P0n, e1, e2)]
        L0[i, j] = float(train_loss_j(p))
param_landscape = dict(
    what="TRAINING LOSS over parameters theta (not the energy landscape over predictions).",
    method="Li et al. 2018 filter-normalized random directions: for every weight matrix, each row (output unit) of a "
           "Gaussian direction is rescaled to the norm of the same row of theta; bias directions are zero. "
           "loss[r][c] = L(theta + a_c d1 + b_r d2), a,b in linspace(-1,1,41).",
    objective=f"Algorithm-1 loss on a fixed batch of {PL_B} (x,y), fixed yhat_0 ~ N(0,{INIT_SCALE}^2 I), fixed per-sample "
              f"alpha, N={PL_N} unrolled steps, fixed Langevin draws (sigma {LANG}), no replay.",
    coords=sig4(coords), loss=sig4(np.minimum(L, 1e6)), center_loss=sig4(center_loss),
    init_slice=dict(step=CK[0], loss=sig4(np.minimum(L0, 1e6)), center_loss=sig4(float(train_loss_j(to_jax(P0n))))),
    noise_floor_this_batch=sig4(float(np.mean(np.sum((yp - mu_np(xp)) ** 2, -1)))),
)

# ------------------------------------------------------------------ loss curve (smoothed, ~200 pts)
losses = Z["losses"]
ema, s_, out = 0.0, [], []
for t, l in enumerate(losses):
    ema = 0.98 * ema + 0.02 * l
    s_.append(ema / (1 - 0.98 ** (t + 1)))
s_ = np.array(s_)
idx = np.unique(np.round(np.geomspace(1, len(losses), 120)).astype(int) - 1)
idx = np.unique(np.concatenate([idx, np.linspace(0, len(losses) - 1, 100).astype(int)]))
loss_curve = dict(step=[int(i + 1) for i in idx], loss=sig4(s_[idx]), raw=sig4(losses[idx]),
                  note="Training loss per Adam step (MSE of yhat_N vs noisy y, N random, Langevin on, replay on); "
                       "'loss' is a bias-corrected EMA (0.98), 'raw' the single-batch value.",
                  noise_floor=sig4(noise_floor))

# ------------------------------------------------------------------ data description + sample
xc = np.linspace(0, 1, 201)
dataset = dict(
    mu_formula=f"mu(x) = ({A_X} sin(2 pi x), {A_Y} sin(4 pi x))",
    sigma_formula=f"sigma(x) = {S_MIN} + {S_MAX - S_MIN:.2f} sin^2(pi (x - 0.25))",
    y_formula="y = mu(x) + sigma(x) * N(0, I_2), x ~ U[0,1)",
    curve=dict(x=sig4(xc), mu=sig4(mu_np(xc)), sigma=sig4(sigma_np(xc))),
    sample=sig4(np.concatenate([xv[:800, None], yv[:800]], 1)),
    sample_note="800 (x, y0, y1) rows from the fixed eval set (same distribution as training data).",
    noise_floor=sig4(noise_floor),
    noise_floor_note="E||y - mu(x)||^2: the best any predictor can reach on the noisy targets.",
)

# ------------------------------------------------------------------ weights
weights = [dict(step=s, layers=[dict(W=W.tolist(), b=b.tolist()) for W, b in PARAMS[s]]) for s in CK]

arch = dict(
    layer_sizes=[IN_DIM] + HIDDEN + [1],
    activation="silu",
    activation_formula="silu(z) = z * s(z), s(z) = 1/(1+exp(-z)); silu'(z) = s(z) * (1 + z * (1 - s(z)))",
    feature_map=dict(K=K_FEAT, formula="phi(x) = [sin(2 pi x), cos(2 pi x), sin(4 pi x), cos(4 pi x), sin(6 pi x), cos(6 pi x)]",
                     order="interleaved: sin(2 pi k x), cos(2 pi k x) for k = 1..K"),
    input="h0 = concat(phi(x), yhat) -> length 8: indices 0..5 = phi(x), index 6 = yhat[0] (horizontal), index 7 = yhat[1] (vertical)",
    forward="for hidden layers l = 0..2: h = silu(W_l h + b_l); out = W_3 h + b_3 (linear, scalar)",
    weight_layout="W[out][in] (row = output unit); W_l has shape [layer_sizes[l+1]][layer_sizes[l]]",
    energy=f"E(x, yhat) = out + lambda * (yhat[0]^2 + yhat[1]^2), lambda = {LAMBDA}",
    quadratic_term=dict(lambda_=LAMBDA, why="tiny confining term so E grows far from the data and GD cannot run off to infinity; "
                        "the MLP easily shifts the minimum, so it does not bias predictions."),
    grad=("grad_yhat E = backprop of out to h0, take entries 6..7, plus 2*lambda*yhat. Backprop: g = W_3[0]; for l = 2..0: "
          "g = g * silu'(z_l); g = W_l^T g. (z_l = W_l h_l + b_l pre-activation)."),
    update="yhat <- yhat - alpha * grad_yhat E(x, yhat) (+ eta ~ N(0, sigma^2 I) when Langevin is on)",
)

out = dict(
    _meta=dict(
        source="Toy experiment: toy model trained for this explainer (src/toy2d_train.py, src/toy2d_export.py). NOT a paper result.",
        note=("A tiny but real Energy-Based Model whose prediction yhat is a 2D point, trained with the paper's recipe "
              "(Algorithm 1: random init, N unrolled gradient steps on the energy, MSE at the last step, backprop through "
              "the optimization with second-order gradients; Sec 3.3: replay buffer, Langevin noise, randomized step size "
              "and number of steps). Energy landscape = E over yhat for a fixed x and fixed theta. "
              "param_landscape = training loss over theta. They are different objects."),
        design_choices=[
            "Context x in [0,1) is a phase; targets lie on a figure-eight with x-dependent Gaussian noise so some contexts are near-deterministic and others noisy.",
            "Smooth activation (silu) so grad_yhat E and the second-order training gradients are smooth.",
            "Small quadratic term lambda*||yhat||^2 for well-posedness far from data.",
            "alpha is randomized independently per sample (paper App. I.1.3 found per-batch alpha unstable); N is randomized per batch.",
            "Backprop through all unrolled steps (no detach), loss only at the last step, as for the paper's S2 models; we do not truncate.",
            "Replay buffer stores (x, y, yhat_N) and restarts a fraction of each batch from stored predictions.",
            "All exported numbers are computed in float64 from the rounded (4 significant digit) weights in this file.",
        ],
        grid_axes="grids[k].E[r][c] is the energy at yhat = (xs[c], ys[r]) with xs = linspace(extent[0], extent[1], grid_n), "
                  "ys = linspace(extent[2], extent[3], grid_n). Row 0 is the BOTTOM (ymin).",
        findings=None,  # filled below
    ),
    arch=arch,
    hparams=dict({k: v for k, v in HP.items() if k != "ckpt_steps"}, lambda_=LAMBDA, hidden=HIDDEN, K_features=K_FEAT,
                 loss="mean over batch of ||yhat_N - y||^2", optimizer="Adam, warmup + cosine LR, global-norm clip"),
    checkpoint_steps=CK,
    weights=weights,
    extent=EXTENT, grid_n=GRID_N,
    contexts=contexts,
    grid_focus_contexts=FOCUS, grid_subset_steps=CK_SUB,
    grids=grids,
    trajectory_starts=sig4(STARTS),
    trajectories=trajectories,
    trajectories_by_ckpt=traj_by_ckpt,
    dataset=dataset,
    loss_curve=loss_curve,
    eval=dict(
        how=f"Fixed eval set of {NV} fresh (x, y) pairs. Predict by GD on the energy from yhat_0 ~ N(0,{INIT_SCALE}^2 I) with alpha={ALPHA0}, no Langevin, no replay. "
            "mse_vs_y = mean ||yhat_N - y||^2 (what training minimizes, floor = noise_floor); mse_vs_mu = mean ||yhat_N - mu(x)||^2 (distance to the clean mean).",
        noise_floor=sig4(noise_floor), curve=eval_curve),
    thinking_curve=thinking_curve,
    thinking_by_ckpt=thinking_by_ckpt,
    bon=bon,
    bon_vs_training=dict(bon_vs_training, note="mean ||yhat* - mu||^2 on the eval set, M=1 vs M=8 candidates, same settings as bon.aggregate"),
    basin_stats=basin,
    replay_buffer_demo=replay_buffer_demo,
    param_landscape=param_landscape,
)

# ------------------------------------------------------------------ findings (computed, honest)
th = th_mse_mu
f = []
f.append(f"Final eval: mean ||yhat-mu||^2 after 40 GD steps = {th[40]:.4f}; mse vs noisy y = {th_mse_y[40]:.4f} "
         f"(noise floor {noise_floor:.4f}).")
f.append(f"Thinking longer: mse_vs_mu at steps 1/2/4/6/10/20/40/60 = " + ", ".join(f"{th[k]:.4f}" for k in [1, 2, 4, 6, 10, 20, 40, 60]) +
         f" (training used N in [{HP['n_min']},{HP['n_max']}]).")
f.append("Best-of-N (Langevin, random alpha, %d steps): mse_vs_mu for M=%s: %s; oracle best of 16 = %.4f." % (
    BON_N, Ms, ", ".join(f"{bon_final[M]:.4f}" for M in Ms), bon_final["oracle_best_of_max"]))
f.append("BoN gain over training (M=1 -> M=8): " + "; ".join(
    f"step {s}: {a} -> {b}" for s, a, b in zip(bon_vs_training["step"], bon_vs_training["M1"], bon_vs_training["M8"])))
f.append(f"Uncertainty check (64 contexts): corr(sigma, E_min) = {basin['corr_sigma_vs_Emin']}, "
         f"corr(sigma, mean Hessian eigenvalue) = {basin['corr_sigma_vs_mean_curvature']}, corr(sigma, steps to converge) = {basin['corr_sigma_vs_steps']}.")
f.append(f"Param-space training-loss slice: center {center_loss:.4f}, min {L.min():.4f}, max {L.max():.3g}.")
out["_meta"]["findings"] = f
for line in f:
    print(line)

s = json.dumps(out, separators=(",", ":"))
open(OUT_JSON, "w").write(s)
print("wrote", OUT_JSON, f"{len(s) / 1e6:.2f} MB")
