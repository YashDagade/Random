"""Independent verifier for data/text.json (toy text EBT trained for this explainer).

Re-implements the toy EBT forward pass and the gradient dE/dyhat (hand-written backprop)
in plain numpy from the base64 weights stored in data/text.json, then recomputes stored
quantities and reports max deviations. JAX is used ONLY to regenerate the random initial
predictions yhat_0 (jax.random keys) so that stored per-seed numbers can be compared; all
model math here is numpy.

  python3 src/text_verify.py <work_dir>        (work_dir holds val.npy / train.npy / params_*.npz)
"""
import os
import sys
import json
import math
import base64

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_weights(d):
    w = d["weights"]
    flat = np.frombuffer(base64.b64decode(w["data"]), dtype="<f4")
    assert flat.size == w["n_floats"], (flat.size, w["n_floats"])
    P = {}
    for k in w["order"]:
        t = w["tensors"][k]
        n = int(np.prod(t["shape"]))
        P[k] = flat[t["offset"]:t["offset"] + n].reshape(t["shape"]).astype(np.float64)
    return P


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def silu(x):
    return x * sig(x)


def dsilu(x):
    s = sig(x)
    return s * (1 + x * (1 - s))


def softmax(y):
    y = y - y.max(-1, keepdims=True)
    e = np.exp(y)
    return e / e.sum(-1, keepdims=True)


def encode(P, ids):
    """ids (B, C) -> h (B, d_h)"""
    x = P["emb"][ids].reshape(ids.shape[0], -1)
    a = silu(x @ P["We1"] + P["be1"])
    return a @ P["We2"] + P["be2"]


def energy_grad(P, h, y):
    """h (B,dh), y (B,V) -> E (B,), dE/dy (B,V). Hand-written backprop."""
    dh = h.shape[1]
    p = softmax(y)
    e = p @ P["Wpe"]
    z = np.concatenate([h, e, h * e], -1)
    u1 = z @ P["W1"] + P["b1"]
    a1 = silu(u1)
    u2 = a1 @ P["W2"] + P["b2"]
    a2 = silu(u2)
    E = (a2 @ P["W3"] + P["b3"])[:, 0]
    du2 = P["W3"][:, 0][None] * dsilu(u2)
    du1 = (du2 @ P["W2"].T) * dsilu(u1)
    dz = du1 @ P["W1"].T
    de = dz[:, dh:2 * dh] + h * dz[:, 2 * dh:]
    dp = de @ P["Wpe"].T
    g = p * (dp - (p * dp).sum(-1, keepdims=True))
    return E, g


def think(P, h, y0, alpha, n):
    ys, Es = [y0], []
    y = y0
    for i in range(n + 1):
        E, g = energy_grad(P, h, y)
        Es.append(E)
        if i == n:
            break
        y = y - np.asarray(alpha).reshape(-1, 1) * g
        ys.append(y)
    return np.stack(ys), np.stack(Es)


def ce(y, t):
    lp = y - y.max(-1, keepdims=True)
    lp = lp - np.log(np.exp(lp).sum(-1, keepdims=True))
    return -lp[np.arange(len(t)), t]


def fixed_eval_set(arr, C, n, seed):
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(np.arange(C, len(arr)), size=n, replace=False))
    ctx = arr[idx[:, None] + np.arange(-C, 0)[None, :]]
    return idx, ctx.astype(np.int64), arr[idx].astype(np.int64)


def relerr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b) / (np.abs(b) + 1e-3)))


def main(work):
    d = json.load(open(os.path.join(ROOT, "data", "text.json")))
    vocab = d["vocab"]
    stoi = {c: i for i, c in enumerate(vocab)}
    C = d["arch"]["context_len"]
    P = load_weights(d)
    report = {}

    # 0. exported weights == training checkpoint
    tag = os.environ.get("EBT_TAG", "ebt_long")
    npz = np.load(os.path.join(work, f"params_{tag}.npz"))
    report["weights_vs_checkpoint_maxabs"] = max(float(np.abs(npz[k] - P[k]).max()) for k in P)
    report["n_params_from_weights"] = int(sum(v.size for v in P.values()))

    # 1. gradient check (finite differences)
    rng = np.random.default_rng(0)
    ids = rng.integers(0, len(vocab), size=(3, C))
    h = encode(P, ids)
    y = rng.standard_normal((3, len(vocab)))
    E, g = energy_grad(P, h, y)
    fd = np.zeros_like(y)
    for v in range(len(vocab)):
        e = np.zeros_like(y); e[:, v] = 1e-5
        fd[:, v] = (energy_grad(P, h, y + e)[0] - energy_grad(P, h, y - e)[0]) / 2e-5
    report["grad_vs_finite_diff_maxabs"] = float(np.abs(fd - g).max())
    report["grad_scale"] = float(np.abs(g).max())

    # 2. reference trace
    ref = d["weights_reference_trace"]
    h = encode(P, np.array([ref["context_ids"]]))
    ys, Es = think(P, h, np.array([ref["yhat0"]]), ref["alpha"], 4)
    report["ref_trace_energy"] = {"numpy": [round(float(x), 4) for x in Es[:, 0]], "json": ref["energies"],
                                  "max_rel": relerr(Es[:, 0], ref["energies"])}
    report["ref_trace_yhat4"] = relerr(ys[4, 0, :5], ref["yhat4_first5"])

    # 3. examples: per-step energies, top-8 probabilities (yhat0 = PRNGKey(31) normal, same as ref trace)
    y0 = np.array([ref["yhat0"]])
    alpha0 = d["hparams"]["alpha0"]
    ex_report = []
    for ex in d["examples"]:
        ids = np.array([[stoi[c] for c in ex["context"]]])
        h = encode(P, ids)
        ys, Es = think(P, h, y0, alpha0, len(ex["steps"]) - 1)
        e_json = [s["energy"] for s in ex["steps"]]
        ptrue = softmax(ys[:, 0])[:, stoi[ex["true_next"]]]
        top_match = all(vocab[int(np.argmax(softmax(ys[i, 0])))] == s["top"][0][0] for i, s in enumerate(ex["steps"]))
        ex_report.append({"label": ex["label"], "energy_max_absdiff": float(np.abs(Es[:, 0] - e_json).max()),
                          "p_true_max_absdiff": float(np.abs(ptrue - [s["p_true"] for s in ex["steps"]]).max()),
                          "top1_all_steps_match": top_match})
    report["examples"] = ex_report

    # 4. energy slices (exact grid values; no randomness)
    sl_report = []
    for sl in d["energy_slices"]:
        ex = [e for e in d["examples"] if e["label"] == sl["label"]][0]
        h = encode(P, np.array([[stoi[c] for c in ex["context"]]]))
        lo, hi = sl["extent"][0], sl["extent"][1]
        gr = np.linspace(lo, hi, sl["grid_n"])
        A, B = np.meshgrid(gr, gr)
        Y = np.zeros((A.size, len(vocab)))
        Y[:, stoi[sl["token_a"]]] = A.ravel()
        Y[:, stoi[sl["token_b"]]] = B.ravel()
        E, _ = energy_grad(P, np.repeat(h, A.size, 0), Y)
        E = E.reshape(A.shape)
        # restricted GD path 0
        path = sl["paths_restricted_gd"][0]
        yv = np.zeros((1, len(vocab))); yv[0, stoi[sl["token_a"]]], yv[0, stoi[sl["token_b"]]] = path[0][0], path[0][1]
        pe = []
        for i in range(len(path)):
            e, g = energy_grad(P, h, yv)
            pe.append([yv[0, stoi[sl["token_a"]]], yv[0, stoi[sl["token_b"]]], e[0]])
            yv[0, stoi[sl["token_a"]]] -= alpha0 * g[0, stoi[sl["token_a"]]]
            yv[0, stoi[sl["token_b"]]] -= alpha0 * g[0, stoi[sl["token_b"]]]
        sl_report.append({"label": sl["label"], "grid_max_absdiff": float(np.abs(E - np.array(sl["energy"])).max()),
                          "grid_range": [float(E.min()), float(E.max())],
                          "path0_max_absdiff": float(np.abs(np.array(pe) - np.array(path)).max())})
    report["energy_slices"] = sl_report

    # 5. uncertainty sentences: energy of the uniform guess (yhat = 0) per char, no randomness
    val = np.load(os.path.join(work, "val.npy"))
    sen_report = []
    for s in d["uncertainty"]["sentences"]:
        full = s["context_before"] + s["text"]
        ids = np.array([[stoi[c] for c in full[j:j + C]] for j in range(len(s["text"]))])
        h = encode(P, ids)
        Eu, _ = energy_grad(P, h, np.zeros((len(ids), len(vocab))))
        sen_report.append({"text": s["text"][:30], "E_uniform_max_absdiff": float(np.abs(Eu - s["energy_uniform"]).max())})
    report["sentences_uniform_energy"] = sen_report

    # 6. thinking curve + baseline on held-out val (same eval positions; yhat0 from the same jax keys)
    os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
    import jax
    _, vctx, vy = fixed_eval_set(val, C, 4096, 99 + len("val"))
    hv = encode(P, vctx)
    curves = []
    for seed in range(2):
        y0 = np.asarray(jax.random.normal(jax.random.PRNGKey(1000 + seed), (len(vy), len(vocab))), np.float64)
        ys, _ = think(P, hv, y0, np.full(len(vy), alpha0), 16)
        curves.append([ce(ys[i], vy).mean() for i in range(17)])
    cf = np.mean(curves, 0)
    stored = d["thinking_curve"]["datasets"]["val"]["ebt_fixed_alpha_ce"]
    report["val_thinking_curve"] = {"numpy": [round(float(x), 4) for x in cf], "max_absdiff": float(np.abs(cf - stored).max())}
    # same curve with fresh numpy inits (does it depend on the particular seeds?)
    y0 = np.random.default_rng(12345).standard_normal((len(vy), len(vocab)))
    ys, Es = think(P, hv, y0, np.full(len(vy), alpha0), 16)
    fresh = np.array([ce(ys[i], vy).mean() for i in range(17)])
    report["val_thinking_curve_fresh_init"] = [round(float(x), 4) for x in fresh]
    # per-example paired difference N=3 vs N=16 (bootstrap CI on the mean improvement)
    diff = ce(ys[3], vy) - ce(ys[16], vy)
    bs = [diff[np.random.default_rng(k).integers(0, len(diff), len(diff))].mean() for k in range(500)]
    report["val_gain_N3_to_N16_nats"] = {"mean": float(diff.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                                         "frac_positions_improved": float((diff > 0).mean())}
    # baseline from its checkpoint
    pb = {k: np.asarray(v, np.float64) for k, v in np.load(os.path.join(work, f"params_{os.environ.get('BASE_TAG', 'base_long')}.npz")).items()}
    hb = encode(pb, vctx)
    a = silu(hb @ pb["W1"] + pb["b1"]); a = silu(a @ pb["W2"] + pb["b2"]); lb = a @ pb["W3"] + pb["b3"]
    report["baseline_val_ce"] = {"numpy": float(ce(lb, vy).mean()), "json": d["thinking_curve"]["datasets"]["val"]["baseline_ce"],
                                 "n_params": int(sum(v.size for v in pb.values()))}

    # 7. Best-of-M on val at N=3 with fresh numpy inits
    Ms, cand_ce, cand_E = [1, 2, 4, 8], [], []
    for j in range(8):
        y0 = np.random.default_rng(777 + j).standard_normal((len(vy), len(vocab)))
        ys, Es = think(P, hv, y0, np.full(len(vy), alpha0), 3)
        cand_ce.append(ce(ys[3], vy)); cand_E.append(Es[3])
    cand_ce, cand_E = np.array(cand_ce), np.array(cand_E)
    sel = [float(cand_ce[np.argmin(cand_E[:M], 0), np.arange(len(vy))].mean()) for M in Ms]
    report["bon_val_N3_fresh_init"] = {"energy_select_ce": [round(x, 4) for x in sel],
                                       "mean_ce": [round(float(cand_ce[:M].mean()), 4) for M in Ms],
                                       "json": d["bon"]["settings"][0]["datasets"]["val"]["energy_select_ce"]}
    print(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    main(sys.argv[1])
