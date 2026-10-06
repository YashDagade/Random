"""Evaluate the toy denoising models and export data/image.json + media/toy PNGs.

python3 -I src/image_export.py <data_npz> <run_dir> <project_root>

Everything here is a TOY model trained for this explainer (CIFAR-10 32x32 subset, CPU, minutes).
The paper's real image result is Table 4 / Fig 10 / Fig 12 (COCO 128x128, DiT-L sized models).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import jax
import jax.numpy as jnp
from PIL import Image, ImageDraw, ImageFont

import image_ebt as M

N_EVAL = 500
SHOW = [13, 23, 67, 16]          # test indices for the strips (horse, truck, bird, dog)
SIGMAS = [0.1, 0.2]
EBT_MAX_STEPS = 24
STRIP_STEPS = [1, 2, 4, 8, 16]
UPS = 4
BG = (13, 19, 31)


def r4(v):
    if isinstance(v, (list, tuple)):
        return [r4(x) for x in v]
    if isinstance(v, dict):
        return {k: r4(x) for k, x in v.items()}
    if isinstance(v, (float, np.floating)):
        v = float(v)
        if v == 0 or not np.isfinite(v):
            return v
        return float(f"{v:.4g}")
    if isinstance(v, np.integer):
        return int(v)
    return v


def to255(a):
    return np.clip((np.asarray(a, dtype=np.float64) + 1.0) * 127.5, 0, 255)


def mse255(a, y):
    return np.mean((to255(a) - to255(y)) ** 2, axis=(1, 2, 3))


def summarize(per_img_mse):
    m = float(np.mean(per_img_mse))
    return {"psnr": 10 * np.log10(255.0 ** 2 / m), "mse": m,
            "psnr_mean_per_image": float(np.mean(10 * np.log10(255.0 ** 2 / np.maximum(per_img_mse, 1e-8))))}


def load_params(run_dir, tag):
    p = os.path.join(run_dir, f"{tag}_params.npz")
    if not os.path.exists(p):
        return None
    return {k: jnp.asarray(v) for k, v in np.load(p).items()}


def chunks(n, bs=100):
    for i in range(0, n, bs):
        yield slice(i, min(n, i + bs))


def main(npz, run_dir, root, n_eval=None, main_variant="ebt"):
    global N_EVAL
    if n_eval:
        N_EVAL = int(n_eval)
    tr, te, raw = M.load_data(npz)
    names = [str(s) for s in raw["label_names"]]
    Y = jnp.asarray(te[:N_EVAL])
    P = {t: load_params(run_dir, t) for t in ["ebt", "ebtx", "ff", "diff", "ff_time"]}
    logs = {}
    for t in P:
        lp = os.path.join(run_dir, f"{t}_log.json")
        if os.path.exists(lp):
            with open(lp) as f:
                logs[t] = json.load(f)
    variants = [v for v in ["ebt", "ebtx"] if P[v] is not None]
    INIT = {"ebt": "noise", "ebtx": "x_noisy"}

    def _eg(p, x, yh):
        e, g = jax.value_and_grad(lambda y: jnp.sum(M.energy(p, x, y)))(yh)
        return g
    eg_fn = jax.jit(lambda p, x, yh: (M.energy(p, x, yh), _eg(p, x, yh)))
    e_fn = jax.jit(M.energy)
    tr_fn = jax.jit(M.trunk)
    alpha = M.HP["ebt_alpha"]

    def ebt_run(p, x, y0, n):
        """returns predictions yhat_0..yhat_n and energies E(yhat_0..yhat_n)"""
        ys, es = [y0], []
        yh = y0
        for _ in range(n):
            e, g = eg_fn(p, x, yh)
            es.append(e)
            yh = yh - alpha * g
            ys.append(yh)
        es.append(e_fn(p, x, yh))
        return ys, es

    def ddim(p, x, t_start, K, return_traj=False):
        ts = np.unique(np.round(np.linspace(t_start, 0, K + 1)).astype(int))[::-1]
        xt = x
        traj = []
        for t, tn in zip(ts[:-1], ts[1:]):
            ab, abn = float(M.abar_of_t(t)), float(M.abar_of_t(tn))
            tin = jnp.full(xt.shape[:3] + (1,), t / 100.0, jnp.float32)
            eps = tr_fn(p, jnp.concatenate([xt, tin], -1))
            x0 = jnp.clip((xt - np.sqrt(1 - ab) * eps) / np.sqrt(ab), -1, 1)
            xt = np.sqrt(abn) * x0 + np.sqrt(1 - abn) * eps if tn > 0 else x0
            traj.append((int(t), int(tn), x0, xt))
        return (xt, traj) if return_traj else xt

    def spearman(a, b):
        ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
        return float(np.corrcoef(ra, rb)[0, 1])

    def p1(a, y):
        return float(10 * np.log10(255 ** 2 / mse255(a[None], y[None])[0]))

    key = jax.random.PRNGKey(2025)
    out = {"by_sigma": {}}
    strips_meta = []
    frame_dir = os.path.join(root, "media/toy/denoise")
    os.makedirs(frame_dir, exist_ok=True)
    toy_dir = os.path.join(root, "media/toy")
    W = 32 * UPS

    def save_tile(arr, rel):
        im = Image.fromarray(to255(arr).round().astype(np.uint8)).resize((W, W), Image.NEAREST)
        im.save(os.path.join(root, rel))
        return rel

    for sig in SIGMAS:
        t = M.t_of_sigma(sig)
        ab = float(M.abar_of_t(t))
        stag = f"s{int(round(sig * 100)):02d}"
        kx, ky0, kb = jax.random.split(jax.random.fold_in(key, t), 3)
        X, _ = M.add_noise(kx, Y, t)
        Y0 = jax.random.normal(ky0, Y.shape)
        res = {"t": t, "alpha_bar": ab, "sqrt_alpha_bar": np.sqrt(ab), "sqrt_one_minus_alpha_bar": np.sqrt(1 - ab),
               "ood": sig != 0.1}
        res["noisy_input"] = summarize(mse255(X, Y))
        res["noisy_rescaled"] = summarize(mse255(X / np.sqrt(ab), Y))

        # ---- EBT thinking curves (each variant)
        for v in variants:
            steps_mse = np.zeros((EBT_MAX_STEPS + 1, N_EVAL))
            steps_E = np.zeros((EBT_MAX_STEPS + 1, N_EVAL))
            for sl in chunks(N_EVAL):
                init = Y0[sl] if INIT[v] == "noise" else X[sl]
                ys, es = ebt_run(P[v], X[sl], init, EBT_MAX_STEPS)
                for i in range(EBT_MAX_STEPS + 1):
                    steps_mse[i, sl] = mse255(ys[i], Y[sl])
                    steps_E[i, sl] = np.asarray(es[i])
            curve = []
            for i in range(EBT_MAX_STEPS + 1):
                sm = summarize(steps_mse[i])
                curve.append({"steps": i, "nfe": i, "psnr": sm["psnr"], "mse": sm["mse"],
                              "energy_mean": float(np.mean(steps_E[i]))})
            rv = {"init": INIT[v], "curve": curve,
                  "best_steps": int(np.argmax([c["psnr"] for c in curve]))}
            for n in [1, 2, 3, 4, 8, 16, EBT_MAX_STEPS]:
                rv[f"steps_{n}"] = summarize(steps_mse[n])
            rv["energy_error_spearman"] = {str(n): spearman(steps_E[n], steps_mse[n]) for n in [1, 2, 4, 16]}
            rv["traces"] = [{"test_index": idx, "label": names[int(raw["test_y"][idx])],
                             "energy": [float(steps_E[i, idx]) for i in range(17)],
                             "psnr": [float(10 * np.log10(255 ** 2 / steps_mse[i, idx])) for i in range(17)]}
                            for idx in SHOW if idx < N_EVAL]
            # Best-of-M self-verification (paper Alg. 2), only meaningful with random starts
            if INIT[v] == "noise":
                Mn = 4
                bon_steps = [1, 2, 3, 4]
                mse_m = np.zeros((Mn, len(bon_steps), N_EVAL)); e_m = np.zeros_like(mse_m)
                for j in range(Mn):
                    st = jax.random.normal(jax.random.fold_in(kb, j), Y.shape)
                    for sl in chunks(N_EVAL):
                        ys, es = ebt_run(P[v], X[sl], st[sl], max(bon_steps))
                        for q, n in enumerate(bon_steps):
                            mse_m[j, q, sl] = mse255(ys[n], Y[sl]); e_m[j, q, sl] = np.asarray(es[n])
                bon = []
                for q, n in enumerate(bon_steps):
                    pick = np.argmin(e_m[:, q, :], axis=0)
                    worst = np.argmax(e_m[:, q, :], axis=0)
                    ar = np.arange(N_EVAL)
                    bon.append({"steps": n, "M": Mn, "nfe_total": n * Mn,
                                "single_psnr": summarize(mse_m[0, q])["psnr"],
                                "bon_lowest_energy_psnr": summarize(mse_m[pick, q, ar])["psnr"],
                                "highest_energy_pick_psnr": summarize(mse_m[worst, q, ar])["psnr"],
                                "oracle_best_psnr": summarize(mse_m[:, q, :].min(0))["psnr"]})
                rv["bon"] = bon
            # paper-style recursion (App. D.3): apply the model 3 times, each time to its own output
            if sig > 0.1:
                rec = []
                for n in [1, 2, 4]:
                    mm = np.zeros(N_EVAL)
                    for sl in chunks(N_EVAL):
                        ctx = X[sl]
                        for r in range(3):
                            init = jax.random.normal(jax.random.fold_in(kb, 100 + r), ctx.shape) \
                                if INIT[v] == "noise" else ctx
                            ys, _ = ebt_run(P[v], ctx, init, n)
                            ctx = ys[-1]
                        mm[sl] = mse255(ctx, Y[sl])
                    sm = summarize(mm)
                    rec.append({"steps_per_application": n, "applications": 3, "nfe": 3 * n,
                                "psnr": sm["psnr"], "mse": sm["mse"]})
                rv["recursive_x3"] = rec
            res[v] = rv
            print(f"sigma {sig} {v}: " + " ".join(f"{n}:{rv[f'steps_{n}']['psnr']:.2f}" for n in [1, 2, 4, 8, 16, EBT_MAX_STEPS]), flush=True)

        # ---- FF one-shot baselines
        for tag in ["ff", "ff_time"]:
            if P[tag] is None:
                continue
            mm = np.zeros(N_EVAL)
            for sl in chunks(N_EVAL):
                mm[sl] = mse255(tr_fn(P[tag], X[sl]), Y[sl])
            res[tag] = summarize(mm)
            res[tag]["nfe"] = 1

        # ---- diffusion (DDIM from t = sigma*T to 0, eps-prediction)
        Ks = [1, 2, 3, 5, 10, 20, 50, 100]
        dc = []
        for K in Ks:
            mm = np.zeros(N_EVAL)
            for sl in chunks(N_EVAL):
                mm[sl] = mse255(ddim(P["diff"], X[sl], t, K), Y[sl])
            sm = summarize(mm)
            dc.append({"nfe": K, "psnr": sm["psnr"], "mse": sm["mse"]})
            print(f"sigma {sig} DDIM K={K} psnr={sm['psnr']:.2f}", flush=True)
        res["diff_curve"] = dc
        bi = int(np.argmax([c["psnr"] for c in dc]))
        res["diff_best"] = dc[bi]
        res["diff_note"] = ("DDIM from t=sigma*T down to 0 in K evenly spaced steps (NFE=K). K=t (one step per "
                            "schedule step) is the natural full chain; diff_curve stops at K=100, so at sigma=0.2 the K=200 "
                            "full chain appears only in the strips (frames.diff_full). diff_best is the best K on the "
                            "eval set, an optimistic choice for the baseline.")
        if sig > 0.1:
            rec = []
            for K in [1, 2, 3, 10]:
                mm = np.zeros(N_EVAL)
                for sl in chunks(N_EVAL):
                    ctx = X[sl]
                    for r in range(3):
                        ctx = ddim(P["diff"], ctx, M.t_of_sigma(0.1), K)
                    mm[sl] = mse255(ctx, Y[sl])
                sm = summarize(mm)
                rec.append({"ddim_steps_per_application": K, "applications": 3, "nfe": 3 * K,
                            "psnr": sm["psnr"], "mse": sm["mse"]})
            res["diff_recursive_x3"] = rec
            res["recursive_note"] = ("Paper App. D.3: at OOD noise both models did best when applied 3 times "
                                     "recursively to their own output. Our reading: each application treats its "
                                     "input as noise level sigma=0.1 (t=100), the training level.")

        out["by_sigma"][str(sig)] = res

        # ---- frames + strips for showcase images
        sl = np.array(SHOW)
        Xs, Ys, Y0s = X[sl], Y[sl], Y0[sl]
        ffo = tr_fn(P["ff"], Xs)
        kbest = res["diff_best"]["nfe"]
        dbest = ddim(P["diff"], Xs, t, kbest)
        dfull = ddim(P["diff"], Xs, t, t)
        _, dtraj = ddim(P["diff"], Xs, t, 10, return_traj=True)
        runs = {}
        for v in variants:
            init = Y0s if INIT[v] == "noise" else Xs
            runs[v] = ebt_run(P[v], Xs, init, 16)
        for k, idx in enumerate(SHOW):
            nm = f"img{idx:03d}"
            files = {"clean": save_tile(Ys[k], f"media/toy/denoise/{nm}_{stag}_clean.png"),
                     "noisy": save_tile(Xs[k], f"media/toy/denoise/{nm}_{stag}_noisy.png"),
                     "ff": save_tile(ffo[k], f"media/toy/denoise/{nm}_{stag}_ff.png"),
                     "diff_best": save_tile(dbest[k], f"media/toy/denoise/{nm}_{stag}_ddim_best{kbest:03d}.png"),
                     "diff_full": save_tile(dfull[k], f"media/toy/denoise/{nm}_{stag}_ddim_full{t:03d}.png"),
                     "diff_k10_x0": [save_tile(x0[k], f"media/toy/denoise/{nm}_{stag}_ddim10_{j + 1:02d}.png")
                                     for j, (_, _, x0, _) in enumerate(dtraj)]}
            entry = {"test_index": idx, "label": names[int(raw["test_y"][idx])], "sigma": sig, "tile_px": W,
                     "gap_px": 6, "frames": files, "ddim10_timesteps": [[a, b] for (a, b, _, _) in dtraj],
                     "ddim_best_K": kbest,
                     "psnr": {"noisy_input": p1(Xs[k], Ys[k]), "ff": p1(ffo[k], Ys[k]),
                              "diff_best": p1(dbest[k], Ys[k]), "diff_full": p1(dfull[k], Ys[k])},
                     "variants": {}}
            for v in variants:
                ys, es = runs[v]
                suffix = "" if v == "ebt" else "x"
                fr = [save_tile(ys[i][k], f"media/toy/denoise/{nm}_{stag}_{'step' if v == 'ebt' else 'xstep'}{i:02d}.png")
                      for i in range(17)]
                cols = [Ys[k], Xs[k], ys[0][k]] + [ys[q][k] for q in STRIP_STEPS] + [ffo[k], dbest[k]]
                colnames = ["clean", "noisy input", "EBT start y0" + (" (noise)" if INIT[v] == "noise" else " (= x)")] + \
                           [f"EBT step {q}" for q in STRIP_STEPS] + ["FF (1 pass)", f"DDIM ({kbest} passes)"]
                gap = 6
                strip = Image.new("RGB", (len(cols) * (W + gap) - gap, W), BG)
                for c, a in enumerate(cols):
                    im = Image.fromarray(to255(a).round().astype(np.uint8)).resize((W, W), Image.NEAREST)
                    strip.paste(im, (c * (W + gap), 0))
                sp = f"media/toy/denoise_{nm}_{stag}{'' if v == 'ebt' else '_xinit'}.png"
                strip.save(os.path.join(root, sp))
                entry["variants"][v] = {"init": INIT[v], "strip": sp, "strip_columns": colnames, "step_frames": fr,
                                        "energy_steps": [float(es[i][k]) for i in range(17)],
                                        "psnr_steps": [p1(ys[i][k], Ys[k]) for i in range(17)]}
            strips_meta.append(entry)

        # ---- 2D slice of the energy landscape over predictions for one image (fixed x, fixed theta)
        v = main_variant if main_variant in variants else variants[0]
        ys, _ = runs[v]
        k = 0
        yst = ys[16][k]
        d1 = ys[0][k] - yst
        n1 = float(jnp.linalg.norm(d1)); d1 = d1 / n1
        r = Ys[k] - yst  # plane contains start, final prediction AND the true clean image
        r = r - jnp.sum(r * d1) * d1; d2 = r / jnp.linalg.norm(r)
        G = 41
        a = np.linspace(-0.3 * n1, 1.15 * n1, G); b = np.linspace(-0.725 * n1, 0.725 * n1, G)
        grid = np.zeros((G, G))
        xk = jnp.broadcast_to(Xs[k], (G,) + Xs[k].shape)
        for i, bb in enumerate(b):
            cand = yst[None] + a[:, None, None, None] * d1[None] + bb * d2[None]
            grid[i] = np.asarray(e_fn(P[v], xk, cand))
        path = [[float(jnp.sum((ys[i][k] - yst) * d1)), float(jnp.sum((ys[i][k] - yst) * d2))] for i in range(17)]
        yclean = [float(jnp.sum((Ys[k] - yst) * d1)), float(jnp.sum((Ys[k] - yst) * d2))]
        res["energy_slice"] = {
            "what": "ENERGY landscape (not the training loss): E(x, yhat) for ONE fixed noisy context x and the "
                    "final trained weights, on a 2D plane through yhat-space (3072 dims). Origin = EBT step-16 "
                    "prediction; axis u (columns) = direction to the start yhat0; axis v (rows) = direction to the "
                    "true clean image (orthogonalized), so start, final prediction and truth all lie in the plane. "
                    "Path = projection of the 16 gradient steps onto this plane. "
                    "energy[i][j] is at (u[j], v[i]).",
            "variant": v, "test_index": SHOW[k], "u": a.tolist(), "v": b.tolist(), "energy": grid.tolist(),
            "path": path, "clean_projection": yclean, "start_distance": n1,
            "clean_off_plane_distance": float(jnp.sqrt(jnp.maximum(jnp.sum((Ys[k] - yst) ** 2) - yclean[0] ** 2 - yclean[1] ** 2, 0)))}

    # ---- labelled overview sheets (for humans)
    try:
        font = ImageFont.truetype(os.path.join(root, "assets/fonts/JetBrainsMono.ttf"), 14)
    except Exception:
        font = ImageFont.load_default()
    sheets = []
    for v in variants:
        for sig in SIGMAS:
            rows = [s for s in strips_meta if s["sigma"] == sig]
            gap = 6; top = 30; lab = 22
            colnames = rows[0]["variants"][v]["strip_columns"]
            ncol = len(colnames)
            sheet = Image.new("RGB", (ncol * (W + gap) - gap + 20, top + len(rows) * (W + lab + 10) + 10), BG)
            dr = ImageDraw.Draw(sheet)
            dr.text((10, 7), f"Toy EBT denoising (start: {INIT[v]}), sigma={sig} "
                             f"({'in-distribution' if sig == 0.1 else 'OOD'}), CIFAR-10 32px, "
                             f"toy model trained for this explainer", fill=(233, 238, 246), font=font)
            for rI, s in enumerate(rows):
                vv = s["variants"][v]
                im = Image.open(os.path.join(root, vv["strip"]))
                y0 = top + rI * (W + lab + 10)
                sheet.paste(im, (10, y0))
                ps = [None, s["psnr"]["noisy_input"], vv["psnr_steps"][0]] + \
                     [vv["psnr_steps"][q] for q in STRIP_STEPS] + [s["psnr"]["ff"], s["psnr"]["diff_best"]]
                short = ["clean", "noisy", "y0"] + [f"step {q}" for q in STRIP_STEPS] + ["FF", f"DDIM{s['ddim_best_K']}"]
                for c in range(ncol):
                    txt = short[c] if ps[c] is None else f"{short[c]} {ps[c]:.1f}dB"
                    dr.text((10 + c * (W + gap), y0 + W + 3), txt, fill=(147, 161, 184), font=font)
            sp = f"media/toy/denoise_sheet_s{int(round(sig * 100)):02d}{'' if v == 'ebt' else '_xinit'}.png"
            sheet.save(os.path.join(root, sp))
            sheets.append(sp)

    # ---- assemble json
    n_par = {}
    for tag in P:
        if P[tag] is not None:
            n_par[tag] = int(sum(np.prod(v.shape) for v in P[tag].values()))
    train_curves = {}
    for tag, lg in logs.items():
        train_curves[tag] = {"step": [e["step"] for e in lg["log"]], "loss": [e["loss"] for e in lg["log"]],
                             "seconds": [e["time"] for e in lg["log"]], "total_steps": lg["steps"],
                             "wall_seconds": lg["seconds"]}
    T = M.T
    js = {
        "_meta": {
            "source": "toy experiment (toy model trained for this explainer, src/image_ebt.py + src/image_export.py)",
            "note": "Tiny conv EBT vs same-size one-shot denoiser and same-size diffusion (DDIM) denoiser on a "
                    "5000-image CIFAR-10 subset at 32x32 RGB, trained for a few minutes on one CPU core each. "
                    "NOT the paper's result: the paper's image numbers are Table 4 / Fig 10 / Fig 12 (COCO 2014 "
                    "128x128, DiT-L sized models, 100k steps). Images come from the Hugging Face datasets-server "
                    "JPEG previews of uoft-cs/cifar10 (quality ~75), so 'clean' means the decoded JPEG.",
            "dataset": {"name": "uoft-cs/cifar10 (plain_text)", "train_images": int(len(tr)), "eval_images": N_EVAL,
                        "eval_split": f"test rows 0..{N_EVAL - 1} (first {N_EVAL} of the {len(te)} downloaded test rows)",
                        "train_split": f"train rows 0..{len(tr) - 1}", "resolution": "32x32 RGB", "pixel_range_model": "[-1, 1]",
                        "label_names": names},
            "psnr_definition": "PSNR = 10*log10(255^2 / MSE), MSE on 0..255 pixels averaged over all eval pixels "
                               "(this matches how paper Table 4 PSNR and MSE relate). psnr_mean_per_image also given.",
        },
        "noise": {
            "schedule": "DDPM linear beta from 1e-4 to 2e-2 (paper Sec 4.3, p.13) over T=1000 steps (T assumed: DiT default, "
                        "the paper does not state the number of steps)",
            "sigma_definition": "sigma = fraction of the schedule; t = round(sigma*T); "
                                "x_noisy = sqrt(abar_t)*y + sqrt(1-abar_t)*eps, abar_t = prod_{s=1..t}(1-beta_s)",
            "alpha_bar": {str(s): {"t": M.t_of_sigma(s), "alpha_bar": float(M.abar_of_t(M.t_of_sigma(s))),
                                   "alpha_bar_if_0_indexed": float(M.ABAR[M.t_of_sigma(s)]),
                                   "sqrt_alpha_bar": float(np.sqrt(M.abar_of_t(M.t_of_sigma(s)))),
                                   "noise_std_in_[-1,1]_units": float(np.sqrt(1 - M.abar_of_t(M.t_of_sigma(s))))}
                          for s in SIGMAS},
            "train_sigma": 0.1, "test_sigmas": SIGMAS,
            "index_note": "The paper does not state the index convention; DiT/guided-diffusion code indexes "
                          "alphas_cumprod from 0, which would use abar after t+1 steps. The difference is tiny "
                          "(see alpha_bar_if_0_indexed).",
        },
        "hparams": {
            "trunk": f"conv net, SiLU: conv3x3 {M.C1} @32px -> avgpool -> conv {M.C2} @16px -> 2 residual conv {M.C2} "
                     f"-> conv {M.C1} -> nearest upsample + skip -> conv {M.C1} -> conv3x3 to 3 channels",
            "ebt": {"input": "concat(x_noisy, yhat) = 6 channels", "energy": "E = 1/2 * sum(f(x, yhat)^2) (scalar per image)",
                    "think": "yhat_{i+1} = yhat_i - alpha * dE/dyhat_i, yhat_0 ~ N(0, I) (paper Alg. 1)",
                    "init_choice_reason": "We start from pure noise, exactly like paper Algorithm 1, so every bit of "
                                          "the clean image has to come from descending the learned energy. Starting "
                                          "from x_noisy would also work for denoising but would hide that the "
                                          "energy landscape alone carries the prediction.",
                    "train_steps_per_example": M.HP["ebt_steps_choices"], "alpha": M.HP["ebt_alpha"],
                    "alpha_random_factor": f"x exp(U(-ln {M.HP['ebt_alpha_rand']}, ln {M.HP['ebt_alpha_rand']}))",
                    "langevin_std": M.HP["ebt_langevin"], "loss": "MSE(yhat_N, y) at the last step, backprop through all "
                    "steps (second order)", "learnable_alpha": False, "replay_buffer": False,
                    "inference": f"deterministic gradient descent, same alpha, no noise, up to {EBT_MAX_STEPS} steps"},
            "ff": {"input": "x_noisy (3 channels)", "output": "clean image in one forward pass", "loss": "MSE"},
            "ff_time": {"note": "same FF model given the same WALL-CLOCK budget as the EBT (more optimizer steps)"},
            "diff": {"input": "concat(x_t, t/100) = 4 channels", "output": "eps", "train_t": "U{1..100} (sigma <= 0.1)",
                     "sampler": "DDIM (eta=0) from t=sigma*T to 0 with K steps, x0 clipped to [-1,1]",
                     "ood_note": "At sigma=0.2 (t=200) the diffusion model also sees an unseen timestep input."},
            "optim": {"optimizer": "Adam", "lr": M.HP["lr"], "warmup": M.HP["warmup"], "schedule": "cosine to 0.1x",
                      "batch": M.HP["batch"], "grad_clip": 1.0, "augmentation": "horizontal flip"},
            "n_params": n_par,
            "nfe_note": "NFE counts network evaluations. One EBT step = one forward + one backward through the "
                        "energy net (about 3x the FLOPs of a forward pass), the paper also counts it as one pass.",
            "training_compute_note": "Each EBT training step backpropagates through 2-3 gradient steps (second "
                                     "order); on the shared CPU it took about 2.6 s vs 0.8-1.0 s for an FF step "
                                     "(roughly 3x), see train_curves wall_seconds.",
        },
        "train_curves": train_curves,
        "results": out["by_sigma"],
        "strips": strips_meta,
        "sheets": sheets,
        "main_variant": main_variant,
        "paper_reference": {"table4": {"DiT": {"0.1": {"psnr": 26.58, "mse": 142.98}, "0.2": {"psnr": 19.56, "mse": 718.7}},
                                       "EBT": {"0.1": {"psnr": 27.25, "mse": 122.55}, "0.2": {"psnr": 23.29, "mse": 305.2}}},
                            "source": "paper Table 4, p.13 (COCO 2014 128x128)"},
    }
    paper_ref = js.pop("paper_reference")
    js = r4(js)
    js["paper_reference"] = paper_ref  # paper Table 4 values kept exact (not rounded to 4 significant digits)
    with open(os.path.join(root, "data/image.json"), "w") as f:
        json.dump(js, f, separators=(",", ":"))
    print("wrote image.json", os.path.getsize(os.path.join(root, "data/image.json")), "bytes")
    # console summary
    for sig in SIGMAS:
        r = out["by_sigma"][str(sig)]
        line = f"== sigma {sig}: noisy {r['noisy_rescaled']['psnr']:.2f} (rescaled) {r['noisy_input']['psnr']:.2f} (raw) | FF {r['ff']['psnr']:.2f}"
        if "ff_time" in r:
            line += f" | FFtime {r['ff_time']['psnr']:.2f}"
        line += f" | DDIM best {r['diff_best']['psnr']:.2f}@{r['diff_best']['nfe']} K1 {r['diff_curve'][0]['psnr']:.2f} K{r['t']} n/a"
        print(line)
        for v in variants:
            rv = r[v]
            print(f"   {v}: " + " ".join(f"{n}:{rv[f'steps_{n}']['psnr']:.2f}" for n in [1, 2, 3, 4, 8, 16, EBT_MAX_STEPS]),
                  "spearman", {k: round(x, 2) for k, x in rv["energy_error_spearman"].items()})
            if "bon" in rv:
                print("     BoN", [(b["steps"], round(b["single_psnr"], 2), round(b["bon_lowest_energy_psnr"], 2),
                                    round(b["highest_energy_pick_psnr"], 2)) for b in rv["bon"]])
            if "recursive_x3" in rv:
                print("     rec", [(q["nfe"], round(q["psnr"], 2)) for q in rv["recursive_x3"]])
        if "diff_recursive_x3" in r:
            print("   diff rec", [(q["nfe"], round(q["psnr"], 2)) for q in r["diff_recursive_x3"]])


if __name__ == "__main__":
    main(*sys.argv[1:6])
