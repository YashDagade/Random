"""Verifier: recompute PSNR of saved toy denoising PNGs against the saved clean PNG and compare to image.json.
python3 -I src/image_verify_png_psnr.py <project_root>
"""
import json, os, sys
import numpy as np
from PIL import Image

root = sys.argv[1]
d = json.load(open(os.path.join(root, "data/image.json")))


def tile32(path):
    a = np.asarray(Image.open(os.path.join(root, path)).convert("RGB"), dtype=np.float64)
    assert a.shape == (128, 128, 3), (path, a.shape)
    # NEAREST upsample x4: every 4x4 block constant
    blk = a.reshape(32, 4, 32, 4, 3)
    assert np.all(blk == blk[:, :1, :, :1, :]), path + " not a clean x4 nearest upsample"
    return blk[:, 0, :, 0, :]


def psnr(a, b):
    m = np.mean((a - b) ** 2)
    return 10 * np.log10(255 ** 2 / m)


worst = 0.0
n = 0
for s in d["strips"]:
    fr = s["frames"]
    clean = tile32(fr["clean"])
    checks = [("noisy_input", fr["noisy"]), ("ff", fr["ff"]), ("diff_best", fr["diff_best"]), ("diff_full", fr["diff_full"])]
    for k, pth in checks:
        p = psnr(tile32(pth), clean); j = s["psnr"][k]
        worst = max(worst, abs(p - j)); n += 1
        flag = "  <-- MISMATCH" if abs(p - j) > 0.1 else ""
        print(f"img{s['test_index']:03d} s{s['sigma']} {k:12s} png {p:6.2f} json {j:6.2f}{flag}")
    for v, vv in s["variants"].items():
        for i in [0, 1, 2, 4, 8, 16]:
            p = psnr(tile32(vv["step_frames"][i]), clean); j = vv["psnr_steps"][i]
            worst = max(worst, abs(p - j)); n += 1
            flag = "  <-- MISMATCH" if abs(p - j) > 0.1 else ""
            print(f"img{s['test_index']:03d} s{s['sigma']} {v:4s} step{i:02d}   png {p:6.2f} json {j:6.2f}{flag}")
        # strip consistency: column c of the strip equals the per-step frames
        st = np.asarray(Image.open(os.path.join(root, vv["strip"])).convert("RGB"), dtype=np.float64)
        W, gap = s["tile_px"], s["gap_px"]
        ncol = len(vv["strip_columns"])
        assert st.shape == (W, ncol * (W + gap) - gap, 3), (vv["strip"], st.shape)
        colsrc = [fr["clean"], fr["noisy"], vv["step_frames"][0]] + [vv["step_frames"][q] for q in [1, 2, 4, 8, 16]] + [fr["ff"], fr["diff_best"]]
        for c, src in enumerate(colsrc):
            crop = st[:, c * (W + gap): c * (W + gap) + W]
            ref = np.asarray(Image.open(os.path.join(root, src)).convert("RGB"), dtype=np.float64)
            if not np.array_equal(crop, ref):
                print("STRIP COLUMN MISMATCH", vv["strip"], c, vv["strip_columns"][c], src)
print(f"checked {n} tiles, max |png PSNR - json PSNR| = {worst:.3f} dB")
