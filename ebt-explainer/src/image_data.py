"""CIFAR-10 subset for the toy image-denoising EBT (run with python3 -I).

Usage:
  python3 -I src/image_data.py urls  <download_dir>   -> writes <download_dir>/urls.txt ("<split>_<idx> <url>")
  python3 -I src/image_data.py pack  <download_dir> <out_npz>
      -> decodes <download_dir>/img/<split>_<idx>.jpg into uint8 arrays

Source: Hugging Face datasets-server rows API for "uoft-cs/cifar10" (plain_text).
The server serves each 32x32 image as a JPEG preview (quality ~75), so the "clean"
targets below are those decoded JPEGs. Downloaded files are treated as data only:
we parse JSON with the json module and decode images with Pillow, nothing else.
"""
import json
import os
import sys

import numpy as np
from PIL import Image


def cmd_urls(d):
    rows_dir = os.path.join(d, "rows")
    out = []
    labels = {}
    names = None
    for fn in sorted(os.listdir(rows_dir)):
        if not fn.endswith(".json"):
            continue
        split = fn.split("_")[0]
        with open(os.path.join(rows_dir, fn)) as f:
            js = json.load(f)
        if names is None:
            for ft in js.get("features", []):
                if ft.get("name") == "label":
                    names = ft["type"]["names"]
        for r in js.get("rows", []):
            idx = int(r["row_idx"])
            src = r["row"]["img"]["src"]
            if not isinstance(src, str) or not src.startswith("https://datasets-server.huggingface.co/"):
                continue
            key = f"{split}_{idx}"
            out.append(f"{key} {src}")
            labels[key] = int(r["row"]["label"])
    with open(os.path.join(d, "urls.txt"), "w") as f:
        f.write("\n".join(out) + "\n")
    with open(os.path.join(d, "labels.json"), "w") as f:
        json.dump({"names": names, "labels": labels}, f)
    print(len(out), "urls")


def cmd_pack(d, out_npz):
    with open(os.path.join(d, "labels.json")) as f:
        lab = json.load(f)
    imgdir = os.path.join(d, "img")
    data = {"train": [], "test": []}
    for key, y in lab["labels"].items():
        split, idx = key.split("_")
        p = os.path.join(imgdir, key + ".jpg")
        if not os.path.exists(p):
            continue
        try:
            im = Image.open(p)
            im.load()
            im = im.convert("RGB")
        except Exception:
            continue
        if im.size != (32, 32):
            continue
        data[split].append((int(idx), np.asarray(im, dtype=np.uint8), y))
    arrs = {}
    for split, items in data.items():
        items.sort(key=lambda t: t[0])
        arrs[f"{split}_x"] = np.stack([t[1] for t in items])
        arrs[f"{split}_y"] = np.array([t[2] for t in items], dtype=np.int16)
        arrs[f"{split}_idx"] = np.array([t[0] for t in items], dtype=np.int32)
        print(split, arrs[f"{split}_x"].shape)
    arrs["label_names"] = np.array(lab["names"])
    np.savez_compressed(out_npz, **arrs)


if __name__ == "__main__":
    if sys.argv[1] == "urls":
        cmd_urls(sys.argv[2])
    elif sys.argv[1] == "pack":
        cmd_pack(sys.argv[2], sys.argv[3])
