"""Matplotlib previews of data/toy2d.json (toy model trained for this explainer).
Writes media/toy/toy2d_landscape_{early,mid,final}.png and a diagnostics sheet to argv[1] (optional)."""
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = json.load(open(os.path.join(ROOT, "data", "toy2d.json")))
OUT = os.path.join(ROOT, "media", "toy")
os.makedirs(OUT, exist_ok=True)
BG, PANEL, RULE, INK, MUTED, FAINT = "#0d131f", "#141c2b", "#283449", "#e9eef6", "#93a1b8", "#5d6b82"
EBT, TRUTH, BAD = "#ffc95c", "#74e0c0", "#ff6b81"
CMAP = LinearSegmentedColormap.from_list("energy", ["#0b1a33", "#15466b", "#2b7a8a", "#6fae84", "#e3b75a", "#f7e6bd"])
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": PANEL, "axes.edgecolor": RULE, "text.color": INK,
                     "axes.labelcolor": MUTED, "xtick.color": FAINT, "ytick.color": FAINT, "font.size": 11})

ext = D["extent"]
n = D["grid_n"]
xs, ys = np.linspace(ext[0], ext[1], n), np.linspace(ext[2], ext[3], n)
CK = D["checkpoint_steps"]
curve = np.array(D["dataset"]["curve"]["mu"])
sample = np.array(D["dataset"]["sample"])


def grid(ctx, step):
    for g in D["grids"]:
        if g["ctx"] == ctx and g["step"] == step:
            return np.array(g["E"])


def traj(ctx, step):
    for t in D["trajectories_by_ckpt"]:
        if t["ctx"] == ctx and t["step"] == step:
            return [np.array(p) for p in t["paths"]]


def landscape_panel(ax, ctx, step):
    E = grid(ctx, step)
    c = D["contexts"][ctx]
    # per-panel robust color range so the shape is visible at every checkpoint
    lo, hi = np.percentile(E, 1), np.percentile(E, 97)
    ax.imshow(E, origin="lower", extent=ext, cmap=CMAP, vmin=lo, vmax=hi, interpolation="bilinear")
    ax.contour(xs, ys, E, levels=np.linspace(lo, hi, 14), colors=[INK], linewidths=0.4, alpha=0.35)
    ax.plot(curve[:, 0], curve[:, 1], color=TRUTH, lw=0.8, alpha=0.35, ls="--")
    near = np.abs(((sample[:, 0] - c["x"] + 0.5) % 1) - 0.5) < 0.02
    ax.scatter(sample[near, 1], sample[near, 2], s=7, color=TRUTH, alpha=0.7, lw=0)
    m = c["mu"]
    ax.add_patch(plt.Circle(m, 2 * c["sigma"], fill=False, ec=TRUTH, lw=1.2, alpha=0.8))
    ax.plot(*m, marker="x", color=TRUTH, ms=9, mew=2)
    for p in traj(ctx, step):
        p = np.clip(p, ext[0], ext[1])
        ax.plot(p[:, 0], p[:, 1], color=EBT, lw=1.3, alpha=0.95)
        ax.plot(p[0, 0], p[0, 1], "o", color=EBT, ms=4, mfc=BG)
        ax.plot(p[-1, 0], p[-1, 1], "o", color=EBT, ms=4)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_aspect("equal")
    ax.set_title(f"x = {c['x']}  ({c['label']})", color=INK, fontsize=10.5)
    ax.set_xlabel("ŷ[0]"); ax.set_ylabel("ŷ[1]")


def landscape_fig(step, name, tag):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6.1))
    for ax, ctx in zip(axs, D["grid_focus_contexts"]):
        landscape_panel(ax, ctx, step)
    fig.suptitle(f"Energy landscape E(x, ŷ) over the prediction ŷ, {tag}: training step {step}   "
                 f"(toy model trained for this explainer)", color=INK, fontsize=12.5)
    fig.text(0.5, 0.015, "amber: 40 GD steps on the energy from 8 random starts (α = %.3g).  mint: true mean μ(x) (×), 2σ circle, "
             "nearby training targets.  colors: low → high energy" % D["trajectories"][0]["alpha"],
             ha="center", color=MUTED, fontsize=9.5)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(os.path.join(OUT, name), dpi=110)
    plt.close(fig)


landscape_fig(CK[2], "toy2d_landscape_early.png", "early")
landscape_fig(CK[5], "toy2d_landscape_mid.png", "mid-training")
landscape_fig(CK[-1], "toy2d_landscape_final.png", "final")

# ---------------------------------------------------------------- diagnostics sheet (not part of the site)
if len(sys.argv) > 1:
    fig, axs = plt.subplots(3, 4, figsize=(20, 14))
    a = axs[0, 0]
    lc = D["loss_curve"]
    a.plot(lc["step"], lc["raw"], color=FAINT, lw=0.6); a.plot(lc["step"], lc["loss"], color=EBT)
    a.axhline(lc["noise_floor"], color=TRUTH, ls="--"); a.set_xscale("log"); a.set_yscale("log"); a.set_title("train loss")
    a = axs[0, 1]
    pl = D["param_landscape"]
    L = np.array(pl["loss"]); co = pl["coords"]
    im = a.imshow(np.log10(L), origin="lower", extent=[co[0], co[-1], co[0], co[-1]], cmap=CMAP)
    a.contour(co, co, np.log10(L), levels=15, colors=[INK], linewidths=0.4); plt.colorbar(im, ax=a)
    a.set_title(f"param landscape log10 loss, center {pl['center_loss']}")
    a = axs[0, 2]
    L0 = np.array(pl["init_slice"]["loss"])
    im = a.imshow(np.log10(L0), origin="lower", extent=[co[0], co[-1], co[0], co[-1]], cmap=CMAP); plt.colorbar(im, ax=a)
    a.set_title("param landscape at init (log10)")
    a = axs[0, 3]
    mid = len(co) // 2
    a.plot(co, L[mid], color=EBT, label="d1 slice final"); a.plot(co, L[:, mid], color="#8ea7ff", label="d2 slice final")
    a.plot(co, L0[mid], color=FAINT, label="d1 init"); a.set_yscale("log"); a.legend(); a.set_title("1D cuts")
    a = axs[1, 0]
    th = D["thinking_curve"]
    a.plot(th["steps"], th["mse_vs_mu"], color=EBT, label="vs mu"); a.plot(th["steps"], th["mse_vs_y"], color=TRUTH, label="vs y")
    for t in D["thinking_by_ckpt"]:
        a.plot(range(len(t["mse_vs_mu"])), t["mse_vs_mu"], color=FAINT, lw=0.7)
    a.set_yscale("log"); a.legend(); a.set_title("thinking curve (final + ckpts gray)")
    a = axs[1, 1]
    bs = D["basin_stats"]
    a.plot(bs["x"], np.array(bs["sigma"]) * 10, color=TRUTH, label="sigma*10")
    a.plot(bs["x"], bs["E_min"], color=EBT, label="E_min")
    a.plot(bs["x"], bs["hess_eig_small"], color="#8ea7ff", label="hess small")
    a.plot(bs["x"], bs["hess_eig_large"], color="#c99bff", label="hess large")
    a.plot(bs["x"], np.array(bs["steps_to_converge"]) / 10, color=BAD, label="steps/10")
    a.legend(fontsize=8); a.set_title("basin stats vs x")
    a = axs[1, 2]
    bv = D["bon_vs_training"]
    a.plot(bv["step"], bv["M1"], "o-", color=FAINT, label="M=1"); a.plot(bv["step"], bv["M8"], "o-", color=EBT, label="M=8")
    a.set_xscale("log"); a.set_yscale("log"); a.legend(); a.set_title("BoN vs training step")
    a = axs[1, 3]
    ev = D["eval"]["curve"]
    a.plot(ev["step"], ev["mse_vs_y_N4"], "o-", color=TRUTH, label="vs y N4"); a.plot(ev["step"], ev["mse_vs_mu_N4"], "o-", color=EBT, label="vs mu N4")
    a.plot(ev["step"], ev["mse_vs_mu_N40"], "o-", color=BAD, label="vs mu N40")
    a.set_xscale("symlog", linthresh=10); a.set_yscale("log"); a.legend(); a.set_title("eval vs ckpt")
    # all contexts final
    for k, ci in enumerate([2, 3, 4, 5]):
        a = axs[2, k]
        E = grid(ci, CK[-1]); c = D["contexts"][ci]
        a.imshow(E, origin="lower", extent=ext, cmap=CMAP, vmin=np.percentile(E, 1), vmax=np.percentile(E, 97))
        a.contour(xs, ys, E, levels=14, colors=[INK], linewidths=0.4, alpha=0.4)
        tr = D["trajectories"][ci]
        for p in tr["langevin"]:
            p = np.array(p["path"]); a.plot(p[:, 0], p[:, 1], color=BAD, lw=0.8)
        for p in tr["gd"]:
            p = np.array(p["path"]); a.plot(p[:, 0], p[:, 1], color=EBT, lw=1)
        a.plot(*c["mu"], "x", color=TRUTH, ms=9, mew=2); a.set_title(c["label"])
    fig.tight_layout()
    fig.savefig(sys.argv[1], dpi=70)
    # BoN example + replay
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    b = D["bon"]; ci = b["ctx"]
    E = grid(ci, CK[-1])
    for a in axs:
        a.imshow(E, origin="lower", extent=ext, cmap=CMAP, vmin=np.percentile(E, 1), vmax=np.percentile(E, 97))
    for j, c in enumerate(b["candidates"]):
        p = np.array(c["path"])
        axs[0].plot(p[:, 0], p[:, 1], color=EBT if j == b["chosen"] else MUTED, lw=1.5 if j == b["chosen"] else 0.8)
    axs[0].plot(*D["contexts"][ci]["mu"], "x", color=TRUTH, ms=9, mew=2)
    axs[0].set_title(f"BoN example chosen {b['chosen']} dist {b['chosen_dist_to_mu']} mean {b['mean_dist_to_mu']}")
    for k, ex in enumerate(D["replay_buffer_demo"]["examples"][:2]):
        a = axs[1 + k]
        Eg = grid(ex["ctx"], ex["step"])
        a.imshow(Eg, origin="lower", extent=ext, cmap=CMAP, vmin=np.percentile(Eg, 1), vmax=np.percentile(Eg, 97))
        cols = [EBT, BAD, TRUTH, "#8ea7ff"]
        for s, col in zip(ex["segments"], cols):
            p = np.array(s["path"]); a.plot(p[:, 0], p[:, 1], "o-", color=col, ms=3)
        a.set_title(f"replay demo ctx {ex['ctx']} step {ex['step']}")
    fig.tight_layout(); fig.savefig(sys.argv[1].replace(".png", "_bon.png"), dpi=70)
print("previews written")
