"""Independent verifier for data/samples.json (run with python3 -I).

Usage: python3 -I src/samples_verify.py <downloads_dir> <project_root> [<fresh_dir>]
  <fresh_dir>: independent datasets-server responses named fresh_gsm8k*.json, fresh_squad*.json, fresh_dyck*.json,
  fresh_emqa*.json (fetched with curl into scratchpad/ebt/downloads/samples_verify/).

Checks (prints PASS/FAIL lines, exits non-zero on any FAIL):
  text   : re-tokenizes every snippet with the GPT-NeoX tokenizer.json, checks ids, text spans, BPE pieces,
           next-token pairs, and that each snippet is the prefix of the raw RPv2 document with that doc id.
  eval   : every eval example equals a row in the saved datasets-server responses (same row_idx).
           With <fresh_dir>, rows re-fetched independently from datasets-server are compared too.
  images : recomputes alpha_bar for sigma 0.1 / 0.2 (authors' scaled schedule and plain schedule), checks the
           noisy JPEGs statistically (residual mean ~0, std ~ sqrt(1 - alpha_bar), channels uncorrelated),
           recomputes PSNR / MSE from the saved files, rebuilds the clean crop from the original download.
  json   : _meta present in each section, all media paths exist and open.
All downloaded files are treated as data only.
"""
import gzip
import io
import json
import math
import sys
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

FAILS = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def read_partial_gz(path):
    raw = Path(path).read_bytes()
    d = zlib.decompressobj(16 + zlib.MAX_WBITS)
    out = d.decompress(raw)
    lines = out.decode("utf-8", errors="replace").split("\n")
    return lines[:-1]  # last line may be truncated


def verify_text(S, DL):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(DL / "gptneox_tokenizer.json"))
    vs = tok.get_vocab_size(with_added_tokens=True)
    check(vs == 50277, f"tokenizer vocab size with added tokens = {vs} (paper Table D.3: 50277)")
    # known GPT-NeoX ids (sanity: confirms this is the NeoX vocab, not GPT-2)
    known = {" the": 253, ",": 13, ".": 15, "\n": 187, " a": 247}
    for s, i in known.items():
        ids = tok.encode(s).ids
        check(ids == [i], f"NeoX id for {s!r} = {ids} (expected [{i}])")
    gpt2_the = tok.encode(" the").ids
    check(gpt2_the != [262], "not the GPT-2 vocab (' the' would be 262 there)")

    docs = {}
    for b in ("en_head", "en_middle"):
        for k, line in enumerate(read_partial_gz(DL / f"rpv2_{b}.part.gz")):
            try:
                docs[f"2023-06/0003/{b}.json.gz/{k}"] = json.loads(line)
            except json.JSONDecodeError:
                pass
    text = S["text"]
    for doc in text["docs"]:
        sn = doc["snippet"]
        enc = tok.encode(sn)
        t = doc["tokens"]
        n = len(t["ids"])
        check(enc.ids[:n] == t["ids"], f"{doc['id']}: stored ids == fresh NeoX encoding of snippet ({n} tokens)")
        check(len(enc.ids) == doc["snippet_tokens_total"], f"{doc['id']}: snippet_tokens_total {doc['snippet_tokens_total']} == {len(enc.ids)}")
        check("".join(t["text"]) == sn[:len("".join(t["text"]))], f"{doc['id']}: token text spans join to snippet prefix")
        dec = tok.decode(t["ids"])
        check(dec == "".join(t["text"]), f"{doc['id']}: decode(ids) == joined text spans")
        pieces = [tok.id_to_token(i) for i in t["ids"]]
        check(pieces == t["bpe"], f"{doc['id']}: bpe pieces == id_to_token(ids)")
        p = doc["pairs"]
        check(p["input_ids"] == t["ids"][:-1] and p["target_ids"] == t["ids"][1:], f"{doc['id']}: pairs are a 1-token shift")
        raw = docs.get(doc["rpv2_doc_id"])
        check(raw is not None, f"{doc['id']}: rpv2 doc id {doc['rpv2_doc_id']} found in downloaded shard")
        if raw is not None:
            rc = raw["raw_content"]
            check(rc.startswith(sn), f"{doc['id']}: snippet is the prefix of raw_content")
            check(len(rc) == doc["doc_chars"], f"{doc['id']}: doc_chars {doc['doc_chars']} == {len(rc)}")
            meta = raw.get("meta") or {}
            if isinstance(meta, str):
                meta = json.loads(meta)
            if "url" in meta:
                check(meta["url"] == doc["url"], f"{doc['id']}: url matches shard meta")
    pd = text["pairs_demo"]
    src = next(d for d in text["docs"] if d["id"] == pd["doc"])
    ok = True
    for st in pd["pairs"]:
        i = st["position"]
        ok &= st["context_ids"] == src["tokens"]["ids"][: i + 1]
        ok &= st["target_id"] == src["tokens"]["ids"][i + 1]
        ok &= st["context_text"] == "".join(src["tokens"]["text"][: i + 1])
        ok &= st["target_text"] == src["tokens"]["text"][i + 1]
        ok &= tok.decode(st["context_ids"]) == st["context_text"]
    check(ok, f"pairs_demo ({len(pd['pairs'])} steps) consistent with {pd['doc']} tokens")


def load_rows(DL, prefix):
    rows = {}
    for f in sorted(DL.glob(prefix + "*.json")):
        d = json.loads(f.read_text())
        for r in d.get("rows", []):
            rows[r["row_idx"]] = r["row"]
    return rows


def verify_eval(S, DL, FRESH=None):
    E = S["eval"]
    g = load_rows(DL, "eval_gsm8k")
    for ex in E["gsm8k"]:
        r = g[ex["row_idx"]]
        check(r["question"] == ex["question"] and r["answer"] == ex["solution"],
              f"gsm8k row {ex['row_idx']} matches saved row")
        check(r["answer"].split("####")[-1].strip() == ex["final_answer"], f"gsm8k row {ex['row_idx']} final answer")
    q = load_rows(DL, "eval_squad")
    for ex in E["squad"]:
        r = q[ex["row_idx"]]
        check(r["id"] == ex["id"] and r["context"] == ex["context"] and r["question"] == ex["question"]
              and r["answers"]["text"][: len(ex["answers"])] == ex["answers"][: len(r["answers"]["text"])],
              f"squad row {ex['row_idx']} matches saved row")
    dy = load_rows(DL, "eval_dyck")
    for ex in E["bb_dyck"]:
        r = dy[ex["row_idx"]]
        check(r["inputs"] == ex["prompt"] and ex["target"] in r["targets"], f"bb_dyck row {ex['row_idx']} matches saved row")
        check(ex["input_sequence"] in r["inputs"], f"bb_dyck row {ex['row_idx']} input_sequence inside prompt")
    em = load_rows(DL, "eval_emqa")
    for ex in E["bb_elementary_math_qa"]:
        r = em[ex["row_idx"]]
        check(r["inputs"] == ex["prompt"] and ex["target"] in r["targets"],
              f"bb_emqa row {ex['row_idx']} matches saved row")
        check(sorted(ex["choices"]) == sorted(r["multiple_choice_targets"]), f"bb_emqa row {ex['row_idx']} choices")
    if FRESH:
        for name, key in (("gsm8k", "gsm8k"), ("squad", "squad"), ("dyck", "bb_dyck"), ("emqa", "bb_elementary_math_qa")):
            fr = load_rows(FRESH, f"fresh_{name}")
            for ex in E[key]:
                if ex["row_idx"] in fr:
                    r = fr[ex["row_idx"]]
                    fields = {"gsm8k": r.get("question") == ex.get("question"),
                              "squad": r.get("question") == ex.get("question"),
                              "dyck": r.get("inputs") == ex.get("prompt"),
                              "emqa": r.get("inputs") == ex.get("prompt")}[name]
                    check(fields, f"FRESH {key} row {ex['row_idx']} equals an independent datasets-server fetch")


def alpha_bar_scaled(sigma, T=1000):
    n = int(round(T / sigma))
    s = 1000.0 / n
    betas = np.linspace(s * 1e-4, s * 2e-2, n)
    return float(np.cumprod(1 - betas)[T - 1])


def alpha_bar_plain(sigma, T=1000):
    betas = np.linspace(1e-4, 2e-2, T)
    return float(np.prod(1 - betas[: int(round(sigma * T))]))


def verify_images(S, ROOT, DL):
    I = S["images"]
    N = I["noise"]
    for sg in (0.1, 0.2):
        a1, a2 = alpha_bar_scaled(sg), alpha_bar_plain(sg)
        js = N["alpha_bar"][str(sg)]
        check(abs(a1 - js) < 5e-4, f"alpha_bar({sg}) scaled-schedule {a1:.5f} vs json {js}")
        check(abs(a1 - a2) < 2e-3, f"alpha_bar({sg}) plain-schedule {a2:.5f} agrees with scaled {a1:.5f}")
        check(abs(math.sqrt(1 - a1) - N["noise_std_in_[-1,1]_units"][str(sg)]) < 5e-4, f"noise std({sg}) = {math.sqrt(1 - a1):.4f}")
        check(abs(math.sqrt(a1) - N["signal_scale"][str(sg)]) < 5e-4, f"signal scale({sg}) = {math.sqrt(a1):.4f}")
    curve_ok = all(abs(c["alpha_bar"] - (alpha_bar_scaled(c["sigma"]) if c["sigma"] > 0 else 1.0)) < 1e-3
                   for c in N["curve"])
    check(curve_ok, f"noise curve ({len(N['curve'])} points) matches recomputed alpha_bar")

    # Exact check: rebuild each clean crop from the original download (ADM/DiT center crop), re-add the noise
    # with the producer's documented seed rule (seed = 1000*(k+1) + int(sigma*100)), and compare to the files.
    def center_crop_arr(pil_image, image_size):  # ADM / DiT crop, re-typed from the authors' model_utils.py
        while min(*pil_image.size) >= 2 * image_size:
            pil_image = pil_image.resize(tuple(v // 2 for v in pil_image.size), resample=Image.BOX)
        scale = image_size / min(*pil_image.size)
        pil_image = pil_image.resize(tuple(round(v * scale) for v in pil_image.size), resample=Image.BICUBIC)
        arr = np.array(pil_image)
        cy, cx = (arr.shape[0] - image_size) // 2, (arr.shape[1] - image_size) // 2
        return arr[cy: cy + image_size, cx: cx + image_size]

    import io
    srcs = {f.name: f for f in (DL / "img").glob("coco_*.jpg")}
    for k, c in enumerate(I["coco"]):
        clean_file = np.asarray(Image.open(ROOT / c["file"]).convert("RGB"), dtype=np.float64)
        check(clean_file.shape == (128, 128, 3), f"coco {c['id']} clean is 128x128x3")
        # find the original download whose crop, JPEG q95 re-encoded, equals the clean file
        hit = None
        for name, f in srcs.items():
            a = center_crop_arr(Image.open(f).convert("RGB"), 128)
            buf = io.BytesIO()
            Image.fromarray(a).save(buf, "JPEG", quality=95)
            if np.abs(np.asarray(Image.open(buf).convert("RGB"), np.float64) - clean_file).mean() < 0.05:
                hit = a
                break
        check(hit is not None, f"coco {c['id']}: clean file = center_crop_arr(original download), bit-exact after JPEG")
        if hit is None:
            continue
        x = hit.astype(np.float64) / 127.5 - 1
        for tag, info in c["noisy"].items():
            sg = info["sigma"]
            a = alpha_bar_scaled(sg)
            out = []
            for seed in (1000 * (k + 1) + int(sg * 100), 1000 * (k + 1) + int(sg * 100) + 1):
                eps = np.random.default_rng(seed).standard_normal(x.shape)
                u8 = np.clip(np.round((np.clip(math.sqrt(a) * x + math.sqrt(1 - a) * eps, -1, 1) + 1) * 127.5), 0, 255)
                out.append(u8)
            nf = np.asarray(Image.open(ROOT / info["file"]).convert("RGB"), dtype=np.float64)
            d_ok, d_bad = np.abs(out[0] - nf).mean(), np.abs(out[1] - nf).mean()
            mse = np.mean((out[0] - hit) ** 2)
            ps = 10 * math.log10(255 ** 2 / mse)
            check(d_ok < 7 and d_bad > 25,
                  f"coco {c['id']} {tag}: file = sqrt(ab)x + sqrt(1-ab)eps (mean abs diff {d_ok:.2f} = JPEG error; "
                  f"wrong seed gives {d_bad:.1f})")
            check(abs(ps - info["psnr_db_vs_clean"]) < 0.01 and abs(mse - info["mse_pixel_vs_clean"]) / mse < 1e-3,
                  f"coco {c['id']} {tag}: PSNR {ps:.2f} / MSE {mse:.0f} match json")
            rc = (nf / 127.5 - 1) - math.sqrt(a) * x
            cc = np.corrcoef(rc.reshape(-1, 3).T)
            check(np.max(np.abs(cc - np.eye(3))) < 0.1, f"coco {c['id']} {tag}: noise independent across RGB")


def verify_json(S, ROOT):
    for k, v in S.items():
        if isinstance(v, dict) and k != "_meta":
            check("_meta" in v, f"section {k} has _meta")
    check("source" in S["_meta"] and "note" in S["_meta"], "top-level _meta has source and note")
    paths = set()

    def walk(o):
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
        elif isinstance(o, str) and o.startswith("media/"):
            paths.add(o)
    walk(S)
    bad = []
    for p in sorted(paths):
        try:
            with Image.open(ROOT / p) as im:
                im.load()
        except Exception as e:  # noqa: BLE001
            bad.append((p, str(e)))
    check(not bad, f"{len(paths)} media paths exist and decode ({bad[:3]})")


def main():
    DL, ROOT = Path(sys.argv[1]), Path(sys.argv[2])
    FRESH = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    S = json.loads((ROOT / "data" / "samples.json").read_text())
    verify_json(S, ROOT)
    verify_text(S, DL)
    verify_eval(S, DL, FRESH)
    verify_images(S, ROOT, DL)
    print(f"\n{len(FAILS)} FAIL(s)")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
