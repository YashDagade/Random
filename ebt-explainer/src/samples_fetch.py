"""Fetch raw rows / files from the Hugging Face datasets-server into a downloads dir.

Everything fetched is untrusted data and is only saved to disk here.

Usage:
  python3 -I samples_fetch.py rows <out_dir> <name> <dataset> <config> <split> <offset> <length>
  python3 -I samples_fetch.py coco-candidates <out_dir>
  python3 -I samples_fetch.py in1k-candidates <out_dir>
  python3 -I samples_fetch.py ssv2-candidates <out_dir>
  python3 -I samples_fetch.py images <out_dir> <rows_json> <prefix> <idx> [<idx> ...]
      downloads row[idx].row.image.src (or .image) to <out_dir>/<prefix>_<idx>.jpg
"""
import json
import sys
import time
from pathlib import Path

import ssl
import urllib.error
import urllib.parse
import urllib.request

ROWS = "https://datasets-server.huggingface.co/rows"
UA = {"User-Agent": "ebt-explainer-sample-fetch/1.0"}


CTX = ssl.create_default_context(cafile="/root/.ccr/ca-bundle.crt")


class Resp:
    def __init__(self, content):
        self.content = content

    def json(self):
        return json.loads(self.content.decode("utf-8"))


def get(url, params=None, tries=4):
    if params:
        url = url + "?" + urllib.parse.urlencode(params)
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60, context=CTX) as r:
                return Resp(r.read())
        except urllib.error.HTTPError as e:
            print(f"  HTTP {e.code} for {url[:160]}: {e.read()[:200]!r}", file=sys.stderr)
        except Exception as e:  # noqa: BLE001
            print(f"  error {e} for {url[:160]}", file=sys.stderr)
        time.sleep(2 * (k + 1))
    return None


def fetch_rows(out_dir, name, dataset, config, split, offset, length):
    p = {"dataset": dataset, "config": config, "split": split, "offset": offset, "length": length}
    r = get(ROWS, p)
    if r is None:
        return None
    d = r.json()
    d["_request"] = p
    out = Path(out_dir) / f"{name}.json"
    out.write_text(json.dumps(d))
    print(f"saved {out} rows={len(d.get('rows', []))} total={d.get('num_rows_total')}")
    return d


def coco_candidates(out_dir):
    # AbdoTW/COCO_2014 is the HF mirror the paper cites (ref [120]).
    for split, offs in (("train", [0, 9000, 21000, 37000, 52000, 68000, 80000]),
                        ("validation", [100, 8000, 17000, 26000, 35000])):
        for o in offs:
            fetch_rows(out_dir, f"coco_{split}_{o}", "AbdoTW/COCO_2014", "default", split, o, 12)


def in1k_candidates(out_dir):
    for o in (0, 6100, 12300, 18700, 24400, 30900, 37200, 43500, 49000):
        fetch_rows(out_dir, f"in1k_val_{o}", "benjamin-paine/imagenet-1k-128x128", "default",
                   "validation", o, 10)


def ssv2_candidates(out_dir):
    for o in (0, 40, 120):
        fetch_rows(out_dir, f"ssv2grid_{o}", "moondream/ssv2-3x3", "default", "test", o, 12)


def images(out_dir, rows_json, prefix, idxs):
    d = json.loads(Path(rows_json).read_text())
    rows = d["rows"]
    for i in idxs:
        row = rows[int(i)]["row"]
        img = row.get("image")
        src = img.get("src") if isinstance(img, dict) else None
        if not src:
            print(f"no image src for row {i}", file=sys.stderr)
            continue
        r = get(src)
        if r is None:
            continue
        out = Path(out_dir) / f"{prefix}_{i}.jpg"
        out.write_bytes(r.content)
        print(f"saved {out} ({len(r.content)} bytes)")


def main():
    cmd = sys.argv[1]
    if cmd == "rows":
        fetch_rows(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6],
                   int(sys.argv[7]), int(sys.argv[8]))
    elif cmd == "coco-candidates":
        coco_candidates(sys.argv[2])
    elif cmd == "in1k-candidates":
        in1k_candidates(sys.argv[2])
    elif cmd == "ssv2-candidates":
        ssv2_candidates(sys.argv[2])
    elif cmd == "images":
        images(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:])
    else:
        raise SystemExit(f"unknown command {cmd}")


if __name__ == "__main__":
    main()
