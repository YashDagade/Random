"""Evaluate the toy text EBT + baseline and export data/text.json (+ preview PNG).

Toy model trained for this explainer, not a paper result.

  python3 src/text_eval.py <work_dir>
"""
import os
import sys
import json
import math
import base64

os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import text_ebt as T  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NMAX = 16


def r4(x):
    """Round floats (recursively) to 4 significant digits."""
    if isinstance(x, (list, tuple)):
        return [r4(v) for v in x]
    if isinstance(x, dict):
        return {k: r4(v) for k, v in x.items()}
    if isinstance(x, (np.floating, float)):
        x = float(x)
        if x == 0 or not math.isfinite(x):
            return 0.0 if not math.isfinite(x) else x
        return float(f"{x:.4g}")
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.ndarray):
        return r4(x.tolist())
    return x


def main(work):
    meta = json.load(open(os.path.join(work, "meta.json")))
    vocab = meta["vocab"]
    te, tb = os.environ.get("EBT_TAG", "ebt_main"), os.environ.get("BASE_TAG", "base_main")
    logE = json.load(open(os.path.join(work, f"log_{te}.json")))
    logB = json.load(open(os.path.join(work, f"log_{tb}.json")))
    cfg = logE["cfg"]
    for k, v in cfg.items():
        T.CFG[k] = v
    C = cfg["C"]
    alpha0 = cfg["alpha0"]
    pE = T.load_params(os.path.join(work, f"params_{te}.npz"))
    pB = T.load_params(os.path.join(work, f"params_{tb}.npz"))
    arrs = {k: np.load(os.path.join(work, k + ".npy")) for k in ["val", "ood_shakespeare", "ood_code", "train"]}

    enc = jax.jit(T.encode_ctx)
    blog = jax.jit(T.base_logits)

    @jax.jit
    def trace_fixed(h, y0, al):
        return T.think_trace(pE, h, y0, al, NMAX)

    def ce_np(logits, y):
        lp = jax.nn.log_softmax(logits, -1)
        return -np.asarray(jnp.take_along_axis(lp, jnp.asarray(y)[:, None], -1)[:, 0])

    rng = np.random.default_rng(2024)
    sets = {}
    for name in ["val", "ood_shakespeare", "ood_code", "train"]:
        n = 4096
        _, ctx, y = T.fixed_eval_set(arrs[name], C, n, 99 + len(name))
        sets[name] = (jnp.asarray(ctx), y)

    # ------------------------------------------------ thinking curves (Facet 1: more forward passes)
    thinking = {"steps": list(range(0, NMAX + 1)), "datasets": {}}
    alpha_sweep = {"alphas": [alpha0 * 0.5, alpha0, alpha0 * 2.0], "steps": list(range(0, NMAX + 1)), "val_ce": []}
    best_N = {}
    cache_h = {}
    for name, (ctx, y) in sets.items():
        h = enc(pE, ctx)
        cache_h[name] = h
        B = ctx.shape[0]
        ces_fixed, ces_rand, Es_fixed = [], [], []
        for seed in range(2):
            y0 = jax.random.normal(jax.random.PRNGKey(1000 + seed), (B, T.V))
            ys, Es = trace_fixed(h, y0, jnp.full((B,), alpha0))
            ces_fixed.append([ce_np(ys[i], y).mean() for i in range(NMAX + 1)])
            Es_fixed.append(np.asarray(Es).mean(1))
            al = alpha0 * np.exp(rng.uniform(-math.log(cfg["alpha_rand"]), math.log(cfg["alpha_rand"]), B))
            ys, Es = trace_fixed(h, y0, jnp.asarray(al, jnp.float32))
            ces_rand.append([ce_np(ys[i], y).mean() for i in range(NMAX + 1)])
        cf = np.mean(ces_fixed, 0)
        cr = np.mean(ces_rand, 0)
        base_ce = ce_np(blog(pB, ctx), y).mean()
        best_N[name] = int(np.argmin(cf[1:]) + 1)
        thinking["datasets"][name] = {
            "ebt_fixed_alpha_ce": cf, "ebt_random_alpha_ce": cr, "ebt_mean_energy": np.mean(Es_fixed, 0),
            "baseline_ce": base_ce,
            "ebt_fixed_alpha_ppl": np.exp(cf), "baseline_ppl": math.exp(base_ce),
            "best_N_fixed_alpha": best_N[name],
            "train_like_N": cfg["n_max"],
            "gain_vs_train_like_pct": 100 * (1 - math.exp(cf[best_N[name]]) / math.exp(cf[cfg["n_max"]])),
        }
        print(name, "baseline %.4f" % base_ce, "ebt fixed", np.round(cf, 3), "rand", np.round(cr, 3))
        if name == "val":
            for a in alpha_sweep["alphas"]:
                y0 = jax.random.normal(jax.random.PRNGKey(1000), (B, T.V))
                ys, _ = trace_fixed(h, y0, jnp.full((B,), a, jnp.float32))
                alpha_sweep["val_ce"].append([ce_np(ys[i], y).mean() for i in range(NMAX + 1)])

    # ------------------------------------------------ Best-of-M self-verification (Facet 3, Alg. 2)
    Mmax = 8
    bon = {"M": [1, 2, 4, 8], "note": "M candidates from different random yhat_0 ~ N(0,I), fixed alpha, N steps, no noise; "
           "keep the candidate with the lowest final energy (no external verifier). 'oracle' picks the candidate with "
           "the lowest true loss (upper bound, uses the answer); 'mean' is the average candidate.", "settings": []}
    cand = {}
    for name, (ctx, y) in sets.items():
        h = cache_h[name]
        B = ctx.shape[0]
        cc, cE = [], []
        for j in range(Mmax):
            y0 = jax.random.normal(jax.random.PRNGKey(5000 + j), (B, T.V))
            ys, Es = trace_fixed(h, y0, jnp.full((B,), alpha0))
            cc.append(np.stack([ce_np(ys[i], y) for i in range(NMAX + 1)]))
            cE.append(np.asarray(Es))
        cand[name] = (np.stack(cc), np.stack(cE))  # (M, steps, B)
    for N in sorted(set([cfg["n_max"], 8])):
        entry = {"N": N, "datasets": {}}
        for name, (ctx, y) in sets.items():
            B = ctx.shape[0]
            cand_ce, cand_E = cand[name][0][:, N], cand[name][1][:, N]
            sel, orc, mean = [], [], []
            for M in bon["M"]:
                idx = np.argmin(cand_E[:M], 0)
                sel.append(cand_ce[idx, np.arange(B)].mean())
                orc.append(cand_ce[:M].min(0).mean())
                mean.append(cand_ce[:M].mean())
            ce_c = cand_ce - cand_ce.mean(0)
            E_c = cand_E - cand_E.mean(0)
            corr = float((ce_c * E_c).sum() / (np.sqrt((ce_c ** 2).sum() * (E_c ** 2).sum()) + 1e-9))
            entry["datasets"][name] = {"energy_select_ce": sel, "oracle_ce": orc, "mean_ce": mean,
                                       "energy_loss_corr_within_example": corr}
            print("BoN N", N, name, np.round(sel, 4), "oracle", np.round(orc, 4), "corr %.3f" % corr)
        bon["settings"].append(entry)

    # ------------------------------------------------ uncertainty: easy vs hard tokens (cf. paper Fig 8)
    NU = 8

    def classify(prev_text, t):
        prev = prev_text[-1]
        if prev == "q" and t == "u":
            return "easy"
        if t.isalpha():
            # letters of the current word typed so far
            k = 0
            while k < len(prev_text) and prev_text[-1 - k].isalpha():
                k += 1
            if k == 0:
                return "hard"   # first letter of a new word
            if k >= 2:
                return "easy"   # third letter onward inside a word
        return "other"

    val_text = "".join(vocab[i] for i in arrs["val"])
    y0_shared = jax.random.normal(jax.random.PRNGKey(77), (1, T.V))

    def char_energies(positions, arr):
        ctx = jnp.asarray(arr[np.asarray(positions)[:, None] + np.arange(-C, 0)[None, :]].astype(np.int32))
        h = enc(pE, ctx)
        B = ctx.shape[0]
        y0 = jnp.repeat(y0_shared, B, 0)
        ys, Es = trace_fixed(h, y0, jnp.full((B,), alpha0))
        yt = arr[np.asarray(positions)].astype(np.int32)
        pt = np.asarray(jax.nn.softmax(ys[NU], -1))[np.arange(B), yt]
        return np.asarray(Es[:NU + 1]).T, pt, ce_np(ys[NU], yt)

    # group means over the val subset
    vctx_idx = T.fixed_eval_set(arrs["val"], C, 4096, 99 + 3)[0]
    Eall, ptall, ceall = char_energies(vctx_idx, arrs["val"])
    groups = np.array([classify(val_text[i - C:i], val_text[i]) for i in vctx_idx])
    group_curves = {}
    for gname in ["easy", "hard", "other"]:
        m = groups == gname
        group_curves[gname] = {"n": int(m.sum()), "mean_energy": Eall[m].mean(0), "std_energy": Eall[m].std(0),
                               "mean_ce_final": ceall[m].mean(), "mean_p_true_final": ptall[m].mean()}
    print("groups", {k: (v["n"], round(float(v["mean_ce_final"]), 3), np.round(v["mean_energy"], 3)) for k, v in group_curves.items()})

    # two real sentences from held-out text
    import re
    sents = []
    for mt in re.finditer(r"(?<=[\.\!\?] )([a-z][a-z ,']{55,95}[\.\!\?])", val_text):
        s0 = mt.start(1)
        if s0 < C + 5:
            continue
        sub = mt.group(1)
        if sub.count(" ") < 8 or "#" in sub:
            continue
        sents.append((s0, sub))
    picks = [sents[3], sents[len(sents) // 2]] if len(sents) > 4 else sents[:2]
    sentences = []
    for s0, sub in picks:
        pos = list(range(s0, s0 + len(sub)))
        Es, pt, cel = char_energies(pos, arrs["val"])
        sentences.append({
            "text": sub, "context_before": val_text[s0 - C:s0],
            "chars": list(sub), "energy": Es, "p_true_final": pt, "ce_final": cel,
            "group": [classify(val_text[i - C:i], val_text[i]) for i in pos],
        })
        print("sentence:", sub)

    # ------------------------------------------------ example contexts: per-step top-8
    def find(pattern, start_frac=0.0):
        st = int(len(val_text) * start_frac)
        k = val_text.find(pattern, st)
        return k

    example_specs = []
    for label, pat, off, kind in [
        ("q is almost always followed by u", "q", 1, "easy"),
        ("finishing the word 'the'", " the ", 3, "easy"),
        ("deep inside a long word", "informatio", 10, "easy"),
        ("first letter of a new word", ". ", 2, "hard"),
        ("which digit comes next in a year?", " 20", 3, "hard"),
        ("after a comma", ", ", 2, "hard"),
    ]:
        k = find(pat, 0.3)
        while k != -1 and (k < C or k + off >= len(val_text)):
            k = find(pat, (k + 1) / len(val_text))
        if k == -1:
            continue
        pos = k + off
        example_specs.append((label, pos, kind))
    NE = 12
    examples = []
    for label, pos, kind in example_specs:
        ctxa = arrs["val"][pos - C:pos].astype(np.int32)[None]
        h = enc(pE, jnp.asarray(ctxa))
        y0 = jax.random.normal(jax.random.PRNGKey(31), (1, T.V))
        ys, Es = T.think_trace(pE, h, y0, jnp.full((1,), alpha0), NE)
        ps = np.asarray(jax.nn.softmax(ys[:, 0], -1))
        true_c = vocab[int(arrs["val"][pos])]
        steps = []
        for i in range(NE + 1):
            order = np.argsort(-ps[i])[:8]
            steps.append({"step": i, "energy": float(Es[i, 0]),
                          "top": [[vocab[j], float(ps[i, j])] for j in order],
                          "p_true": float(ps[i, int(arrs["val"][pos])]),
                          "entropy": float(-(ps[i] * np.log(ps[i] + 1e-12)).sum())})
        bp = np.asarray(jax.nn.softmax(blog(pB, jnp.asarray(ctxa))[0]))
        border = np.argsort(-bp)[:8]
        examples.append({"label": label, "kind": kind, "context": val_text[pos - C:pos], "true_next": true_c,
                         "steps": steps,
                         "baseline_top": [[vocab[j], float(bp[j])] for j in border],
                         "baseline_p_true": float(bp[int(arrs["val"][pos])])})
        print("example", repr(val_text[pos - 20:pos]), "->", repr(true_c),
              [(s["top"][0][0], round(s["top"][0][1], 2), round(s["energy"], 2)) for s in steps[::3]])

    # ------------------------------------------------ 2D slice of the energy landscape over the prediction
    def slice2d(ex, grid_n=41, lo=-6.0, hi=10.0):
        pos = [p for (l, p, k) in example_specs if l == ex["label"]][0]
        ctxa = jnp.asarray(arrs["val"][pos - C:pos].astype(np.int32)[None])
        h = enc(pE, ctxa)
        a_tok = vocab.index(ex["steps"][-1]["top"][0][0])
        b_tok = vocab.index(ex["steps"][-1]["top"][1][0])
        g = np.linspace(lo, hi, grid_n)
        A, Bm = np.meshgrid(g, g)  # rows: b, cols: a
        Y = np.zeros((grid_n * grid_n, T.V), np.float32)
        Y[:, a_tok] = A.ravel()
        Y[:, b_tok] = Bm.ravel()
        hh = jnp.repeat(h, Y.shape[0], 0)
        E = np.asarray(T.energy(pE, hh, jnp.asarray(Y))).reshape(grid_n, grid_n)
        # exact gradient descent restricted to this 2D plane (other logits stay 0)
        paths = []
        for s, start in enumerate([(-4, 8), (8, -4), (-5, -5), (2, 2)]):
            y = np.zeros((1, T.V), np.float32)
            y[0, a_tok], y[0, b_tok] = start
            pts = []
            for i in range(25):
                yj = jnp.asarray(y)
                e = float(T.energy(pE, h, yj)[0])
                pts.append([float(y[0, a_tok]), float(y[0, b_tok]), e])
                gr = np.asarray(T.grad_energy(pE, h, yj))
                y[0, a_tok] -= alpha0 * gr[0, a_tok]
                y[0, b_tok] -= alpha0 * gr[0, b_tok]
            paths.append(pts)
        return {"label": ex["label"], "token_a": vocab[a_tok], "token_b": vocab[b_tok], "extent": [lo, hi, lo, hi],
                "grid_n": grid_n, "energy": E, "paths_restricted_gd": paths,
                "note": "Energy over two logits (token_a on x, token_b on y); all other 52 logits held at 0. "
                        "energy[row][col]: row = token_b logit, col = token_a logit, both from extent lo..hi. "
                        "Paths are exact gradient descent restricted to this plane (alpha = alpha0)."}

    slices = [slice2d(ex) for ex in examples[:4]]

    # ------------------------------------------------ weights export for a live in-browser demo
    order = ["emb", "We1", "be1", "We2", "be2", "Wpe", "W1", "b1", "W2", "b2", "W3", "b3"]
    flat = np.concatenate([np.asarray(pE[k], np.float32).ravel() for k in order])
    b64 = base64.b64encode(flat.astype("<f4").tobytes()).decode()
    weights = None
    if len(b64) < 1.5e6:
        offs, o = {}, 0
        for k in order:
            sz = int(np.prod(pE[k].shape))
            offs[k] = {"offset": o, "shape": list(pE[k].shape)}
            o += sz
        weights = {"encoding": "base64 little-endian float32, concatenated in 'order'; matrices row-major [in, out]",
                   "order": order, "tensors": offs, "n_floats": int(flat.size), "data": b64}
    print("weights b64 bytes", len(b64), "exported" if weights else "SKIPPED")

    # reference trace for verifying a JS port: example 0, fixed init
    ref = None
    if weights is not None:
        ex0 = example_specs[0][1]
        ctxa = arrs["val"][ex0 - C:ex0].astype(np.int32)
        y0 = np.asarray(jax.random.normal(jax.random.PRNGKey(31), (1, T.V)))[0]
        ys, Es = T.think_trace(pE, enc(pE, jnp.asarray(ctxa[None])), jnp.asarray(y0[None]), jnp.full((1,), alpha0), 4)
        ref = {"context_ids": ctxa.tolist(), "yhat0": y0.tolist(), "alpha": alpha0, "energies": np.asarray(Es[:, 0]).tolist(),
               "yhat4_first5": np.asarray(ys[4, 0, :5]).tolist()}

    # ------------------------------------------------ assemble
    def curve(log):
        return {"step": log["step"], "train_loss_ema": log["train_loss"], "val_loss": log["val_loss"],
                "train_eval_loss": log["train_eval_loss"], "wall_s": log["time"]}

    out = {
        "_meta": {
            "source": "toy experiment trained for this explainer (src/text_prep.py, src/text_ebt.py, src/text_eval.py)",
            "note": "Toy model trained for this explainer, NOT a paper result. Character-level (54 symbols), ~0.27M params, "
                    "a few minutes on 1 CPU thread. Loss in nats per character; perplexity is per character "
                    "(not comparable to the paper's per-token perplexities).",
        },
        "corpus_name": "RedPajama-Data-V2 (sample, snapshot 2023-06, shard 0000, en_head), first ~3 MB of gzip stream",
        "corpus": {
            "source_url": "https://data.together.xyz/redpajama-data-v2/v1.0.0/sample/documents/2023-06/0000/en_head.json.gz",
            "hf_dataset": "togethercomputer/RedPajama-Data-V2 (files hosted at data.together.xyz; the HF datasets-server rows API "
                          "does not serve this dataset because it uses a loading script)",
            "n_train_docs": meta["n_train_docs"], "n_val_docs": meta["n_val_docs"],
            "train_chars": meta["splits"]["train"]["chars"], "val_chars": meta["splits"]["val"]["chars"],
            "normalization": "lowercased; curly quotes/dashes mapped to ASCII; other ASCII symbols -> '#'; non-ASCII dropped; "
                             "space runs collapsed",
            "samples": {k: v["sample"] for k, v in meta["splits"].items()},
            "ood": {
                "ood_shakespeare": "tiny-shakespeare (karpathy/char-rnn), last 300k chars: different style (verse, archaic words, speaker names)",
                "ood_code": "CPython stdlib source (argparse.py, textwrap.py, json/decoder.py): far shift (code); "
                            + "%.1f%% of chars map to '#'" % (100 * meta["splits"]["ood_code"]["unk_frac"]),
            },
        },
        "vocab": vocab,
        "vocab_display": [("⏎" if c == "\n" else ("␣" if c == " " else c)) for c in vocab],
        "arch": {
            "context_len": C, "char_emb": cfg["d_c"], "encoder": f"concat {C}x{cfg['d_c']} char embeddings -> Linear({C*cfg['d_c']},{cfg['enc_hidden']}) -> SiLU -> Linear({cfg['enc_hidden']},{cfg['d_h']}) = h",
            "ebt": f"p = softmax(yhat) (yhat in R^54 are logits); e = p @ Wpe (54x{cfg['d_h']}); z = [h, e, h*e] ({3*cfg['d_h']}); "
                   f"E = Linear({cfg['H']},1)(SiLU(Linear({cfg['H']},{cfg['H']})(SiLU(Linear({3*cfg['d_h']},{cfg['H']})(z))))) -> scalar",
            "baseline": f"same encoder; logits = Linear({cfg['Hb']},54)(SiLU(Linear({cfg['Hb']},{cfg['Hb']})(SiLU(Linear({cfg['d_h']},{cfg['Hb']})(h))))) in one forward pass",
            "silu": "silu(x) = x * sigmoid(x)", "linear": "y = x @ W + b with W shaped [in, out]",
            "thinking": "yhat_{i+1} = yhat_i - alpha * dE/dyhat (yhat_i), yhat_0 ~ N(0, I)",
            "n_params_ebt": logE["n_params"], "n_params_baseline": logB["n_params"],
        },
        "hparams": {
            "batch": cfg["batch"], "train_steps": logE["steps"], "lr": cfg["lr"], "schedule": "linear warmup 300 + cosine to 10%",
            "optimizer": "AdamW (wd 0.01 on matrices), grad clip 1.0",
            "alpha0": alpha0, "alpha_random_factor": cfg["alpha_rand"],
            "alpha_sampling": "per sample: alpha0 * exp(U(-ln f, ln f)), f = alpha_random_factor",
            "train_steps_N": [cfg["n_min"], cfg["n_max"]], "N_sampling": "per sample, uniform integer",
            "langevin_sigma": cfg["langevin"], "langevin_note": "noise added on every thinking step except the last",
            "replay_frac": cfg["replay_frac"], "replay_size": cfg["replay_size"],
            "replay_note": "a quarter of each batch restarts from the stored final prediction of an earlier batch for the same "
                           "context (simulates longer thinking trajectories, Sec 3.3)",
            "loss": "cross-entropy(softmax(yhat_N), next char), last step only, backprop through all steps (second-order grads)",
            "wall_clock_s": {"ebt": logE["wall_s"], "baseline": logB["wall_s"]},
            "hardware": "1 CPU thread (JAX), shared machine",
        },
        "train_curves": {"ebt": curve(logE["log"]), "baseline": curve(logB["log"]),
                         "note": "ebt train_loss_ema includes Langevin noise, random alpha/N and replay; train_eval/val use "
                                 "deterministic thinking with alpha0 and N=3. Baseline: one forward pass."},
        "thinking_curve": thinking,
        "alpha_sweep": alpha_sweep,
        "bon": bon,
        "uncertainty": {"steps": list(range(NU + 1)), "alpha": alpha0,
                        "rule": "easy = 'u' after 'q' or 3rd+ letter inside a word; hard = first letter of a new word; "
                                "other = everything else. All positions start from the same yhat_0.",
                        "groups": group_curves, "sentences": sentences},
        "examples": examples,
        "energy_slices": slices,
        "weights": weights,
        "weights_reference_trace": ref,
    }
    out = r4(out)
    path = os.path.join(ROOT, "data", "text.json")
    s = json.dumps(out, separators=(",", ":"))
    open(path, "w").write(s)
    print("wrote", path, len(s) / 1e6, "MB")
    return out


if __name__ == "__main__":
    main(sys.argv[1])
