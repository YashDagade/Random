"""Add a flat comparison table + plain-language findings to data/image.json (python3 -I src/image_table.py <root>)."""
import json
import os
import sys


def main(root):
    p = os.path.join(root, "data/image.json")
    with open(p) as f:
        d = json.load(f)
    tc = d.get("train_curves", {})
    steps = {k: v.get("total_steps") for k, v in tc.items()}
    rows = []
    for sig, r in d["results"].items():
        s = float(sig)
        dist = "in-distribution" if s == 0.1 else "OOD"

        def add(model, family, nfe, psnr, mse, note=""):
            rows.append({"sigma": s, "dist": dist, "model": model, "family": family, "nfe": nfe,
                         "psnr": psnr, "mse": mse, "note": note})

        add("noisy input (no denoising)", "data", 0, r["noisy_input"]["psnr"], r["noisy_input"]["mse"])
        add("noisy input rescaled by 1/sqrt(abar)", "data", 0, r["noisy_rescaled"]["psnr"], r["noisy_rescaled"]["mse"])
        add(f"FF one-shot ({steps.get('ff')} train steps, step-matched)", "ff", 1, r["ff"]["psnr"], r["ff"]["mse"])
        if "ff_time" in r:
            add(f"FF one-shot ({steps.get('ff_time')} train steps, wall-clock-matched)", "ff", 1,
                r["ff_time"]["psnr"], r["ff_time"]["mse"])
        longest = max(r["diff_curve"], key=lambda c: c["nfe"])
        add(f"Diffusion DDIM {longest['nfe']} steps from t={r['t']} to 0", "diff", longest["nfe"], longest["psnr"],
            longest["mse"], "one network call per schedule step" if longest["nfe"] == r["t"] else "strided schedule")
        add(f"Diffusion DDIM best K={r['diff_best']['nfe']}", "diff", r["diff_best"]["nfe"], r["diff_best"]["psnr"],
            r["diff_best"]["mse"], "best K chosen on the eval set (optimistic for the baseline)")
        for v, lab in [("ebt", "EBT, start from noise"), ("ebtx", "EBT, start from x_noisy")]:
            if v not in r:
                continue
            for n in [1, 2, 4, 8, 16]:
                c = r[v]["curve"][n]
                add(f"{lab}, {n} steps ({steps.get(v)} train steps)", "ebt", n, c["psnr"], c["mse"])
            for q in r[v].get("recursive_x3", []):
                add(f"{lab}, recursive x3 ({q['steps_per_application']} steps each)", "ebt", q["nfe"], q["psnr"], q["mse"],
                    "paper App. D.3 style recursion")
        for q in r.get("diff_recursive_x3", []):
            add(f"Diffusion DDIM recursive x3 (K={q['ddim_steps_per_application']} each, from t=100)", "diff",
                q["nfe"], q["psnr"], q["mse"], "paper App. D.3 style recursion")
    d["table"] = rows

    # plain findings computed from the numbers
    f = []
    for sig, r in d["results"].items():
        e = r["ebt"]
        best_e = e["curve"][e["best_steps"]]  # best_steps = argmax of the unrounded PSNR (image_export.py)
        flat_from = next(c["steps"] for c in e["curve"] if best_e["psnr"] - c["psnr"] < 0.05)
        bits = [f"sigma={sig}: EBT (noise start) goes from {e['curve'][1]['psnr']:.1f} dB after 1 step to "
                f"{e['curve'][4]['psnr']:.1f} dB after 4 and {best_e['psnr']:.1f} dB at its best ({best_e['steps']} steps; "
                f"within 0.05 dB of that from step {flat_from} on); "
                f"one-shot FF gets {r['ff']['psnr']:.1f} dB, DDIM best {r['diff_best']['psnr']:.1f} dB at {r['diff_best']['nfe']} passes."]
        longest = max(r["diff_curve"], key=lambda c: c["nfe"])
        bits.append(f"The longest DDIM chain we ran ({longest['nfe']} passes from t={r['t']}) gets {longest['psnr']:.1f} dB.")
        if "ff_time" in r:
            bits.append(f"FF trained for the same wall-clock as the EBT gets {r['ff_time']['psnr']:.1f} dB.")
        for q in r["ebt"].get("recursive_x3", []):
            if q["steps_per_application"] == 4:
                bits.append(f"EBT applied 3x recursively (4 steps each, 12 passes) gets {q['psnr']:.1f} dB.")
        if r.get("diff_recursive_x3"):
            bd = max(r["diff_recursive_x3"], key=lambda c: c["psnr"])
            bits.append(f"DDIM applied 3x recursively gets up to {bd['psnr']:.1f} dB ({bd['nfe']} passes).")
        f.append(" ".join(bits))
    d["findings"] = f
    r1, r2 = d["results"]["0.1"], d["results"]["0.2"]
    xpk = sorted({r1["ebtx"]["best_steps"], r2["ebtx"]["best_steps"]})
    xpk_txt = f"{xpk[0]} to {xpk[-1]}" if len(xpk) > 1 else f"{xpk[0]}"
    d["_meta"]["verdict"] = (
        "In this toy, the baselines win on PSNR: one-shot FF and few-step DDIM beat the EBT at sigma=0.1 and "
        "sigma=0.2, and the gap grows if FF gets the same wall-clock instead of the same number of steps. "
        "What the toy does show: (1) the noise-start EBT's quality rises with thinking steps and then plateaus "
        f"without collapsing (the x-start EBT peaks at {xpk_txt} steps and then drifts down slowly), while the toy "
        "diffusion model degrades with long chains and collapses at OOD noise (it was trained only on t <= 100, so "
        "a long chain from t=200 feeds it timesteps it never saw); (2) OOD inputs get "
        f"higher energy (mean energy after 16 steps {r1['ebt']['curve'][16]['energy_mean']:.0f} at sigma=0.1 vs "
        f"{r2['ebt']['curve'][16]['energy_mean']:.0f} at sigma=0.2); (3) paper-style recursion helps at OOD noise "
        "for both families; the best EBT number at sigma=0.2 (x-start, recursive x3, "
        f"{max(q['psnr'] for q in r2['ebtx']['recursive_x3']):.1f} dB) beats step-matched FF "
        f"({r2['ff']['psnr']:.1f} dB) and direct DDIM ({r2['diff_best']['psnr']:.1f} dB) but not wall-clock-matched FF "
        f"({r2['ff_time']['psnr']:.1f} dB) or recursive DDIM "
        f"({max(q['psnr'] for q in r2['diff_recursive_x3']):.1f} dB); (4) Best-of-4 by lowest "
        f"energy adds only {r1['ebt']['bon'][1]['bon_lowest_energy_psnr'] - r1['ebt']['bon'][1]['single_psnr']:.2f} dB "
        "at 2 steps (all starts fall into the same bowl), but it always beats picking the highest-energy sample. "
        "The paper's real result (EBT beats DiT, Table 4 / Fig 12) is at far larger scale.")
    d["_meta"]["caveats"] = [
        f"Training steps (batch 32): EBT noise-start {steps.get('ebt')}, EBT x-start {steps.get('ebtx')} (stopped by "
        f"the ~8 minute cap), FF {steps.get('ff')} and diffusion {steps.get('diff')} (step-matched to the EBT), "
        f"FF wall-clock-matched {steps.get('ff_time')}.",
        "The CPU was shared with other jobs (load 8-20 on 4 cores), so wall-clock numbers are noisy.",
        "Models are tiny (about 69k parameters) and far from converged; all numbers are toy demonstrations.",
        "diff_best picks the best K on the eval set, which flatters the diffusion baseline.",
        "Recursive x3 follows paper App. D.3 in spirit; the exact recursion details are our interpretation.",
        f"EBT x-start starts from the noisy observation; it peaks after {r1['ebtx']['best_steps']} steps (sigma=0.1) "
        f"or {r2['ebtx']['best_steps']} steps (sigma=0.2) and then slowly gets worse, so more steps are not always "
        "better for that variant.",
        "The toy diffusion collapse at sigma=0.2 with long DDIM chains is partly an artifact of our setup: the "
        "diffusion baseline was trained only on t in 1..100 (sigma <= 0.1), so steps with t > 100 use timestep "
        "inputs it never saw. The paper's DiT used recursive DDIM and did not collapse (Table 4: 19.56 dB at sigma=0.2).",
        "T=1000 diffusion steps is assumed (DiT default); the paper gives the beta range but not T.",
        f"Eval set: first {d['_meta']['dataset']['eval_images']} CIFAR-10 test images (one fixed noise draw per sigma).",
    ]
    with open(p, "w") as fh:
        json.dump(d, fh, separators=(",", ":"))
    for row in rows:
        print(f"{row['sigma']:.1f} {row['nfe']:>4} {row['psnr']:6.2f} {row['mse']:8.1f}  {row['model']}")
    print("\n".join(f))


if __name__ == "__main__":
    main(sys.argv[1])
