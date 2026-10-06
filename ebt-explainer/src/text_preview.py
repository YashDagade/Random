"""Preview figure for data/text.json -> media/toy/text_preview.png (toy model, not paper results)."""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for f in ["Atkinson-Regular.ttf", "Atkinson-Bold.ttf", "JetBrainsMono.ttf"]:
    font_manager.fontManager.addfont(os.path.join(ROOT, "assets/fonts", f))
BG, PANEL, RULE, INK, MUTED, FAINT = "#0d131f", "#141c2b", "#283449", "#e9eef6", "#93a1b8", "#5d6b82"
AR, EBT, TRUTH, BAD, DIFF, RNN = "#8ea7ff", "#ffc95c", "#74e0c0", "#ff6b81", "#ff8f7a", "#c99bff"
ECMAP = LinearSegmentedColormap.from_list("energy", ["#0b1a33", "#15466b", "#2b7a8a", "#6fae84", "#e3b75a", "#f7e6bd"])
plt.rcParams.update({"font.family": "Atkinson Hyperlegible", "text.color": INK, "axes.labelcolor": MUTED,
                     "xtick.color": FAINT, "ytick.color": FAINT, "axes.edgecolor": RULE, "font.size": 11})


def style(ax, title):
    ax.set_facecolor(PANEL)
    ax.set_title(title, color=INK, fontsize=12.5, loc="left", pad=8, fontweight="bold")
    ax.grid(color=RULE, lw=0.6)
    for s in ax.spines.values():
        s.set_visible(False)


def main():
    d = json.load(open(os.path.join(ROOT, "data/text.json")))
    fig = plt.figure(figsize=(18, 10.5), facecolor=BG)
    gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.22, left=0.045, right=0.965, top=0.885, bottom=0.07)
    fig.text(0.045, 0.955, "Tiny character-level EBT vs same-size feed-forward baseline (toy model trained for this explainer)",
             fontsize=17, color=INK, fontweight="bold")
    fig.text(0.045, 0.927, f"{d['corpus_name']}  |  EBT {d['arch']['n_params_ebt']:,} params, baseline "
             f"{d['arch']['n_params_baseline']:,} params  |  loss in nats per character", fontsize=11.5, color=MUTED)

    # 1 training curves
    ax = fig.add_subplot(gs[0, 0]); style(ax, "Training: held-out loss vs optimizer step")
    tc = d["train_curves"]
    ax.plot(tc["baseline"]["step"], tc["baseline"]["val_loss"], color=AR, lw=2, label="baseline (1 forward pass)")
    ax.plot(tc["ebt"]["step"], tc["ebt"]["val_loss"], color=EBT, lw=2, label="EBT (think N=3, alpha0)")
    ax.set_xlabel("optimizer step"); ax.set_ylabel("val loss (nats/char)"); ax.legend(facecolor=PANEL, edgecolor=RULE, labelcolor=INK)

    # 2 thinking curve
    ax = fig.add_subplot(gs[0, 1]); style(ax, "Thinking longer: loss vs gradient steps on the energy")
    th = d["thinking_curve"]
    cols = {"val": TRUTH, "ood_shakespeare": RNN, "ood_code": DIFF}
    names = {"val": "RedPajama held-out", "ood_shakespeare": "OOD Shakespeare", "ood_code": "OOD Python code"}
    for k, c in cols.items():
        ds = th["datasets"][k]
        ax.plot(th["steps"][1:], ds["ebt_fixed_alpha_ce"][1:], color=c, lw=2, marker="o", ms=3, label=f"EBT, {names[k]}")
        ax.axhline(ds["baseline_ce"], color=c, lw=1.2, ls="--", alpha=0.8)
    ax.axvspan(2, 3, color=EBT, alpha=0.12)
    ax.text(2.1, ax.get_ylim()[1] * 0.98, "trained\nwith 2-3", color=EBT, fontsize=9, va="top")
    ax.set_xlabel("thinking steps N (forward passes)"); ax.set_ylabel("loss (nats/char); dashed = baseline")
    ax.legend(facecolor=PANEL, edgecolor=RULE, labelcolor=INK, fontsize=9)

    # 3 BoN
    ax = fig.add_subplot(gs[0, 2]); style(ax, "Self-verification: best of M by lowest energy")
    for st, ls in zip(d["bon"]["settings"], ["-", ":"]):
        for k, c in cols.items():
            ds = st["datasets"][k]
            ax.plot(d["bon"]["M"], ds["energy_select_ce"], color=c, lw=2, ls=ls, marker="o", ms=4,
                    label=f"{names[k]}, N={st['N']}" if True else None)
    ax.set_xscale("log", base=2); ax.set_xticks(d["bon"]["M"]); ax.set_xticklabels(d["bon"]["M"])
    ax.set_xlabel("candidates M"); ax.set_ylabel("loss of chosen candidate")
    ax.legend(facecolor=PANEL, edgecolor=RULE, labelcolor=INK, fontsize=8, ncol=2)

    # 4 easy vs hard
    ax = fig.add_subplot(gs[1, 0]); style(ax, "Uncertainty: energy relative to a uniform guess")
    u = d["uncertainty"]
    for g, c in [("easy", TRUTH), ("hard", BAD), ("other", MUTED)]:
        gg = u["groups"][g]
        m, s = np.array(gg["mean_rel_energy"])[1:], np.array(gg["std_rel_energy"])[1:]
        ax.plot(u["steps"][1:], m, color=c, lw=2.2, marker="o", ms=3,
                label=f"{g} (n={gg['n']}, final loss {gg['mean_ce_final']:.2f})")
        ax.fill_between(u["steps"][1:], m - 0.25 * s, m + 0.25 * s, color=c, alpha=0.12)
    ev = u["energy_vs_loss"]
    ax.text(0.02, 0.04, f"Spearman(final energy, char loss) = {ev['spearman_final_energy_vs_char_loss']:.2f}; "
            f"relative: {ev['spearman_final_rel_energy_vs_char_loss']:.2f}", transform=ax.transAxes, color=MUTED, fontsize=9.5)
    ax.text(0.3, 0.6, f"Raw final energy (not relative):\neasy {u['groups']['easy']['mean_energy'][-1]:.2f}, "
            f"hard {u['groups']['hard']['mean_energy'][-1]:.2f}.\nRaw energy does not separate them.",
            transform=ax.transAxes, color=BAD, fontsize=9.5, va="top")
    ax.set_xlabel("thinking step"); ax.set_ylabel("E(x, yhat_i) - E(x, uniform)"); ax.legend(facecolor=PANEL, edgecolor=RULE, labelcolor=INK, fontsize=9)

    # 5 example distribution over steps
    ax = fig.add_subplot(gs[1, 1]); ax.set_facecolor(PANEL)
    ex = d["examples"][0]
    vocab = d["vocab_display"]
    toks = [t for t, _ in ex["steps"][-1]["top"]]
    M = np.zeros((len(toks), len(ex["steps"])))
    for j, s in enumerate(ex["steps"]):
        dd = dict(s["top"])
        for i, t in enumerate(toks):
            M[i, j] = dd.get(t, 0.0)
    ax.imshow(M, aspect="auto", cmap=ECMAP, vmin=0, vmax=1)
    ax.set_yticks(range(len(toks))); ax.set_yticklabels([("space" if t == " " else ("newline" if t == "\n" else t)) for t in toks], fontfamily="JetBrains Mono")
    ax.set_xticks(range(len(ex["steps"]))); ax.set_xlabel("thinking step")
    for j, s in enumerate(ex["steps"]):
        ax.text(j, -0.75, f"{s['energy']:.1f}", ha="center", color=EBT, fontsize=8)
    ctx = ex["context"][-24:].replace("\n", "⏎")
    ax.set_title(f"“...{ctx}” -> true '{ex['true_next']}' (probability; energy on top)", color=INK, fontsize=11, loc="left", pad=16)

    # 6 sentence energy heatmap
    ax = fig.add_subplot(gs[1, 2]); ax.set_facecolor(PANEL)
    sent = u["sentences"][0]
    E = np.array(sent["rel_energy"]).T[1:]  # steps 1.. x chars
    n = min(48, E.shape[1])
    im = ax.imshow(E[:, :n], aspect="auto", cmap=ECMAP)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.015)
    cb.outline.set_visible(False); cb.ax.tick_params(colors=FAINT, labelsize=8)
    cb.set_label("E - E(uniform)", color=MUTED, fontsize=9)
    ax.set_xticks(range(n)); ax.set_xticklabels([("·" if c == " " else c) for c in sent["chars"][:n]], fontfamily="JetBrains Mono", fontsize=8)
    for i, g in enumerate(sent["group"][:n]):
        ax.get_xticklabels()[i].set_color(TRUTH if g == "easy" else (BAD if g == "hard" else FAINT))
    ax.set_ylabel("thinking step"); ax.set_yticks(range(E.shape[0])); ax.set_yticklabels(range(1, E.shape[0] + 1))
    ax.set_title("Relative energy per character, steps 1-8 (mint = easy, red = hard)", color=INK, fontsize=11, loc="left", pad=8)

    out = os.path.join(ROOT, "media/toy/text_preview.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=100, facecolor=BG)
    print("wrote", out)


if __name__ == "__main__":
    main()
