"""Build data/samples.json and media/samples/* from raw downloads (untrusted data).

Usage: python3 -I build_samples.py <downloads_dir> <project_root>

Inputs (all fetched by samples_fetch.py / curl into <downloads_dir>):
  rpv2_en_head.part.gz, rpv2_en_middle.part.gz          RedPajama-V2 "sample" docs (range download)
  rpv2_en_head_signals.part.gz, ..._middle_signals...  matching quality-signal lines
  gptneox_tokenizer.json                               EleutherAI/gpt-neox-20b tokenizer
  eval_*.json                                          datasets-server rows (GSM8K, SQuAD, BigBench)
  img/*.jpg                                            COCO 2014, ImageNet-1k (128px mirror), SSV2 3x3 grids
  mmnist_test_0.json                                   Detection Moving MNIST rows (illustration only)
Nothing in the inputs is executed; they are parsed as JSON / decoded as images.
"""
import json
import math
import sys
import zlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

DL = Path(sys.argv[1])
ROOT = Path(sys.argv[2])
MEDIA = ROOT / "media" / "samples"
FONT_DIR = ROOT / "assets" / "fonts"

# visual tokens from notes/BRIEF.md
BG = (13, 19, 31)
PANEL = (20, 28, 43)
RULE = (40, 52, 73)
INK = (233, 238, 246)
MUTED = (147, 161, 184)
EBT = (255, 201, 92)
TRUTH = (116, 224, 192)


def font(size, mono=False):
    cands = (["JetBrainsMono.ttf", "JetBrainsMono-Regular.ttf"] if mono else
             ["Atkinson-Regular.ttf", "AtkinsonHyperlegible-Regular.ttf"])
    for c in cands:
        p = FONT_DIR / c
        if p.exists():
            try:
                return ImageFont.truetype(str(p), size)
            except Exception:  # noqa: BLE001
                pass
    for p in sorted(FONT_DIR.glob("*Mono*.ttf" if mono else "*Atkinson*Regular*.ttf")):
        try:
            return ImageFont.truetype(str(p), size)
        except Exception:  # noqa: BLE001
            pass
    return ImageFont.load_default()


def rnd(x, sig=4):
    if x is None or x == 0 or not math.isfinite(x):
        return x
    return float(f"{x:.{sig}g}")


def read_partial_gz(path):
    raw = Path(path).read_bytes()
    d = zlib.decompressobj(16 + zlib.MAX_WBITS)
    return d.decompress(raw).decode("utf-8", errors="ignore").split("\n")[:-1]


def rows(name):
    return json.loads((DL / f"{name}.json").read_text())["rows"]


# ----------------------------------------------------------------------------------------------
# 1) Pretraining text: RedPajama-V2
# ----------------------------------------------------------------------------------------------
RPV2_PICKS = [  # (bucket, line index in the 2023-06/0003 shard, topic label for the UI)
    ("en_head", 688, "science news"),
    ("en_head", 473, "programming"),
    ("en_head", 575, "education"),
    ("en_head", 519, "vehicles news"),
    ("en_middle", 186, "nature"),
    ("en_middle", 76, "software community"),
    ("en_middle", 16, "museum science"),
    ("en_middle", 319, "astronomy abstract"),
    ("en_middle", 337, "travel story"),
    ("en_middle", 177, "car history"),
    ("en_head", 229, "music history"),
    ("en_middle", 121, "personal blog"),
]
N_TOK = 80
SNIP_CHARS = 600


def cut_snippet(txt, n=SNIP_CHARS):
    if len(txt) <= n:
        return txt
    cut = txt[:n]
    sp = max(cut.rfind(" "), cut.rfind("\n"))
    return cut[:sp] if sp > n * 0.8 else cut


def build_text():
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(DL / "gptneox_tokenizer.json"))
    docs_cache, sig_cache = {}, {}
    out_docs = []
    for k, (bucket, idx, topic) in enumerate(RPV2_PICKS):
        if bucket not in docs_cache:
            docs_cache[bucket] = read_partial_gz(DL / f"rpv2_{bucket}.part.gz")
            sig_cache[bucket] = read_partial_gz(DL / f"rpv2_{bucket}_signals.part.gz")
        doc = json.loads(docs_cache[bucket][idx])
        sig = json.loads(sig_cache[bucket][idx])
        expect_id = f"2023-06/0003/{bucket}.json.gz/{idx}"
        assert sig["id"] == expect_id, (sig["id"], expect_id)
        assert sig["metadata"]["url"] == doc["url"], "signals/doc misaligned"
        qs = sig["quality_signals"]

        def q(name):
            v = qs.get(name)
            return rnd(v[0][2]) if v and v[0][2] is not None else None

        text = doc["raw_content"]
        snip = cut_snippet(text)
        enc = tok.encode(snip)
        ids = enc.ids[:N_TOK]
        offs = enc.offsets[:N_TOK]
        # contiguous spans: whitespace that byte-level BPE folds into a token ('Ġ') is given to that token,
        # so "".join(strs) == snip[:end of last token]. A token inside a split multi-byte char gets "".
        strs, cur = [], 0
        for a, b in offs:
            strs.append(snip[cur:b] if b > cur else "")
            cur = max(cur, b)
        pieces = enc.tokens[:N_TOK]
        out_docs.append({
            "id": f"rpv2_{k + 1:02d}",
            "topic": topic,
            "rpv2_doc_id": expect_id,
            "bucket": bucket.split("_")[1],
            "language": "en",
            "snapshot": "2023-06",
            "title": doc.get("title"),
            "source_domain": doc.get("source_domain"),
            "url": doc.get("url"),
            "date_download": doc.get("date_download"),
            "doc_chars": len(text),
            "signals": {
                "ccnet_perplexity": q("ccnet_perplexity"),
                "ccnet_language_score": q("ccnet_language_score"),
                "rps_doc_word_count": q("rps_doc_word_count"),
                "rps_doc_stop_word_fraction": q("rps_doc_stop_word_fraction"),
            },
            "snippet": snip,
            "snippet_tokens_total": len(enc.ids),
            "tokens": {"ids": ids, "text": strs, "bpe": pieces, "char_offsets": [list(o) for o in offs]},
            "pairs": {"input_ids": ids[:-1], "target_ids": ids[1:]},
        })

    d0 = out_docs[11]  # "On Thursday afternoon I received a cheery phone call ..." reads naturally token by token
    demo = []
    for i in range(16):
        demo.append({
            "position": i,
            "context_text": d0["snippet"][: d0["tokens"]["char_offsets"][i][1]],
            "context_ids": d0["tokens"]["ids"][: i + 1],
            "target_id": d0["tokens"]["ids"][i + 1],
            "target_text": d0["tokens"]["text"][i + 1],
        })
    n_chars = sum(len(d["snippet"]) for d in out_docs)
    n_toks = sum(d["snippet_tokens_total"] for d in out_docs)
    return {
        "_meta": {
            "source": "togethercomputer/RedPajama-Data-V2 (Hugging Face), 'sample' config files "
                      "sample/documents/2023-06/0003/en_head.json.gz and en_middle.json.gz, plus matching "
                      "sample/quality_signals/... lines. Only the first ~3 MB of each gzip shard was range-downloaded.",
            "dataset_id": "togethercomputer/RedPajama-Data-V2",
            "config": "sample (the paper used the 100B-token sample; these documents come from the small 'sample' "
                      "subset of the same corpus, CC snapshot 2023-06)",
            "license": "Data: Common Crawl Foundation Terms of Use. RedPajama-V2 code: Apache-2.0.",
            "selection": "12 English documents hand-picked for topic diversity after an automatic filter "
                         "(length, prose-like lines, no emails/phone numbers, keyword blocklist). Raw web text, "
                         "unedited, including boilerplate where present.",
            "bucket_note": "head / middle = CCNet perplexity bucket (head = most Wikipedia-like).",
            "snippet": f"first <= {SNIP_CHARS} characters of raw_content, cut at a word boundary",
            "tokenizer": "EleutherAI/gpt-neox-20b tokenizer.json via the `tokenizers` library "
                         f"(vocab size {tok.get_vocab_size()} incl. added tokens; paper Table D.3 lists 50277). "
                         "No BOS token is added.",
            "tokens_fields": "ids = token ids; text = exact substring of the snippet each token covers, "
                             "including its leading space/newline (concatenating text reproduces the snippet prefix); bpe = raw byte-level BPE piece ('Ġ' = leading space, 'Ċ' = newline). "
                             f"Only the first {N_TOK} tokens are stored.",
            "pairs": "Next-token training pairs: input_ids = ids[:-1], target_ids = ids[1:]. "
                     "Position i is trained to predict target_ids[i] from input_ids[:i+1] (causal).",
            "paper_context_length": 256,
            "paper_context_length_ref": "Table D.2 (p.33) and Table D.3 (p.35): NLP context length 256",
            "paper_split": "Paper: manual split of 66M train / 33K validation samples (p.8).",
            "chars_per_token": rnd(n_chars / n_toks),
        },
        "docs": out_docs,
        "pairs_demo": {"doc": d0["id"], "pairs": demo},
    }


# ----------------------------------------------------------------------------------------------
# 2) Downstream eval examples
# ----------------------------------------------------------------------------------------------
def build_eval():
    g = rows("eval_gsm8k")
    gsm = []
    for i in (1, 0, 6):
        r = g[i]["row"]
        final = r["answer"].split("####")[-1].strip()
        gsm.append({"row_idx": g[i]["row_idx"], "question": r["question"], "solution": r["answer"],
                    "final_answer": final})
    sq = []
    for name, i in (("eval_squad", 0), ("eval_squad_8800", 3), ("eval_squad_5300", 5)):
        rr = rows(name)[i]
        r = rr["row"]
        sq.append({"row_idx": rr["row_idx"], "id": r["id"], "title": r["title"], "context": r["context"],
                   "question": r["question"], "answers": sorted(set(r["answers"]["text"]))})
    dy = []
    dyr = rows("eval_dyck")
    for i in (0, 4, 17):
        r = dyr[i]["row"]
        seq = r["inputs"].split("Input:")[-1].split("Output:")[0].strip()
        dy.append({"row_idx": dyr[i]["row_idx"], "prompt": r["inputs"], "input_sequence": seq,
                   "target": r["targets"][0]})
    em = []
    for name, i in (("eval_emqa", 5), ("eval_emqa_2500", 5), ("eval_emqa_5000", 3)):
        rr = rows(name)[i]
        r = rr["row"]
        em.append({"row_idx": rr["row_idx"], "prompt": r["inputs"], "target": r["targets"][0],
                   "choices": r["multiple_choice_targets"]})
    return {
        "_meta": {
            "note": "The paper reports PERPLEXITY on these four downstream datasets (not accuracy), because its "
                    "small from-scratch models do not reach meaningful accuracy (p.9). Lower is better. The paper "
                    "does not specify the exact prompt/answer formatting used to compute downstream perplexity, so "
                    "the raw dataset fields are shown here.",
            "sources": {
                "gsm8k": {"dataset_id": "openai/gsm8k", "config": "main", "split": "test", "license": "MIT"},
                "squad": {"dataset_id": "rajpurkar/squad", "config": "plain_text", "split": "validation",
                          "license": "CC BY-SA 4.0"},
                "bb_dyck": {"dataset_id": "tasksource/bigbench", "config": "dyck_languages", "split": "validation",
                            "license": "Apache-2.0 (BIG-bench)",
                            "note": "tasksource re-split of BIG-bench; the paper cites BIG-bench [99] without naming a split"},
                "bb_elementary_math_qa": {"dataset_id": "tasksource/bigbench", "config": "elementary_math_qa",
                                          "split": "validation", "license": "Apache-2.0 (BIG-bench)",
                                          "note": "the task mixes formats: plain arithmetic chains, nested function "
                                                  "form, and word problems with a hint"},
            },
            "access": "Hugging Face datasets-server /rows API",
        },
        "paper_table3": {
            "ref": "Table 3, p.12 (perplexity, lower is better)",
            "columns": ["Pretrain", "GSM8K", "SQuAD", "BB Math QA", "BB Dyck"],
            "Transformer++": [31.36, 49.6, 52.3, 79.8, 131.5],
            "EBT": [33.43, 43.3, 53.1, 72.6, 125.3],
        },
        "gsm8k": gsm,
        "squad": sq,
        "bb_dyck": dy,
        "bb_elementary_math_qa": em,
    }


# ----------------------------------------------------------------------------------------------
# 3) Images
# ----------------------------------------------------------------------------------------------
COCO_LICENSES = {
    1: "CC BY-NC-SA 2.0", 2: "CC BY-NC 2.0", 3: "CC BY-NC-ND 2.0", 4: "CC BY 2.0",
    5: "CC BY-SA 2.0", 6: "CC BY-ND 2.0", 7: "No known copyright restrictions", 8: "United States Government Work",
}
COCO_PICKS = [  # (rows file, row idx in file)  -> train 8 then validation 4
    ("coco_train_0", 1), ("coco_train_0", 8), ("coco_train_21000", 4), ("coco_train_52000", 1),
    ("coco_train_68000", 4), ("coco_train_68000", 9), ("coco_train_80000", 9), ("coco_train_9000", 9),
    ("coco_validation_26000", 1), ("coco_validation_26000", 3), ("coco_validation_35000", 4),
    ("coco_validation_35000", 7),
]
IN1K_PICKS = [("in1k_val_0", 2), ("in1k_val_18700", 7), ("in1k_val_18700", 4), ("in1k_val_24400", 5),
              ("in1k_val_43500", 4), ("in1k_val_43500", 8), ("in1k_val_49000", 8), ("in1k_val_24400", 4)]


def center_crop_arr(pil_image, image_size):
    """ADM / DiT center crop, identical to model/model_utils.py in the authors' repo."""
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)
    arr = np.array(pil_image)
    cy = (arr.shape[0] - image_size) // 2
    cx = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[cy: cy + image_size, cx: cx + image_size])


def alpha_bar_repo(sigma, T=1000, t_index=None):
    """Authors' code: create_diffusion(diffusion_steps=int(T/sigma)) with the linear schedule scaled by
    1000/N (beta 1e-4*s .. 2e-2*s over N steps), then q_sample at t = T-1 (0-indexed)."""
    n = int(T / sigma)
    s = 1000.0 / n
    betas = np.linspace(s * 1e-4, s * 2e-2, n, dtype=np.float64)
    ab = np.cumprod(1.0 - betas)
    return float(ab[(T - 1) if t_index is None else t_index])


def alpha_bar_simple(sigma, T=1000):
    """Brief's reading: fraction sigma of the plain 1000-step schedule, t = round(sigma*T) steps applied."""
    betas = np.linspace(1e-4, 2e-2, T, dtype=np.float64)
    t = int(round(sigma * T))
    return float(np.prod(1.0 - betas[:t])) if t > 0 else 1.0


def add_noise(img_u8, sigma, seed):
    x = img_u8.astype(np.float64) / 127.5 - 1.0
    ab = alpha_bar_repo(sigma)
    eps = np.random.default_rng(seed).standard_normal(x.shape)
    xt = math.sqrt(ab) * x + math.sqrt(1 - ab) * eps
    u8 = np.clip(np.round((np.clip(xt, -1, 1) + 1) * 127.5), 0, 255).astype(np.uint8)
    return u8, ab


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 10 * math.log10(255.0 ** 2 / mse), mse


def build_images():
    cdir = MEDIA / "coco"
    idir = MEDIA / "imagenet"
    cdir.mkdir(parents=True, exist_ok=True)
    idir.mkdir(parents=True, exist_ok=True)
    coco = []
    clean_arrays = []
    for k, (fname, i) in enumerate(COCO_PICKS):
        rr = rows(fname)[i]
        r = rr["row"]
        split = "train" if "train" in fname else "validation"
        src = DL / "img" / f"{fname}_{i}.jpg"
        im = center_crop_arr(Image.open(src).convert("RGB"), 128)
        nn = f"{k + 1:02d}"
        im.save(cdir / f"{nn}.jpg", quality=95)
        arr = np.array(im)
        clean_arrays.append(arr)
        entry = {"id": nn, "file": f"media/samples/coco/{nn}.jpg", "split": split, "coco_id": r["id"],
                 "row_idx": rr["row_idx"], "orig_size": [r["width"], r["height"]],
                 "caption": r["caption"][0], "captions": r["caption"],
                 "license": COCO_LICENSES.get(r["license"], str(r["license"])),
                 "flickr_url": r.get("flickr_url"), "noisy": {}}
        for sig, tag in ((0.1, "s10"), (0.2, "s20")):
            noisy, ab = add_noise(arr, sig, seed=1000 * (k + 1) + int(sig * 100))
            Image.fromarray(noisy).save(cdir / f"{nn}_{tag}.jpg", quality=92, subsampling=0)
            p, m = psnr(noisy, arr)
            entry["noisy"][tag] = {"sigma": sig, "file": f"media/samples/coco/{nn}_{tag}.jpg",
                                   "psnr_db_vs_clean": rnd(p), "mse_pixel_vs_clean": rnd(m)}
        coco.append(entry)

    # noise ladder strip for image 01 (illustrates how sigma maps to corruption)
    ladder_sig = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]
    tiles = []
    ladder_meta = []
    for j, s in enumerate(ladder_sig):
        if s == 0:
            tiles.append(clean_arrays[0])
            ladder_meta.append({"sigma": 0.0, "alpha_bar": 1.0, "signal_scale": 1.0, "noise_std": 0.0})
            continue
        n, ab = add_noise(clean_arrays[0], s, seed=77 + j)
        tiles.append(n)
        ladder_meta.append({"sigma": s, "alpha_bar": rnd(ab), "signal_scale": rnd(math.sqrt(ab)),
                            "noise_std": rnd(math.sqrt(1 - ab))})
    pad, lab = 8, 34
    W = len(tiles) * (128 + pad) + pad
    strip = Image.new("RGB", (W, 128 + 2 * pad + lab), BG)
    dr = ImageDraw.Draw(strip)
    f = font(18, mono=True)
    for j, t in enumerate(tiles):
        x = pad + j * (128 + pad)
        strip.paste(Image.fromarray(t), (x, pad))
        s = ladder_sig[j]
        col = EBT if s in (0.1, 0.2) else MUTED
        txt = f"σ={s:g}"
        tw = dr.textlength(txt, font=f)
        dr.text((x + (128 - tw) / 2, 128 + pad + 8), txt, fill=col, font=f)
    strip.save(cdir / "noise_ladder.png")

    # patch grid illustration (128 px, patch 16 -> 8x8 = 64 tokens of 16*16*3 = 768 values)
    src = Image.fromarray(clean_arrays[0])
    S = 4
    big = src.resize((128 * S, 128 * S), Image.NEAREST)
    m = 40
    can = Image.new("RGB", (128 * S + 2 * m, 128 * S + 2 * m), BG)
    can.paste(big, (m, m))
    dr = ImageDraw.Draw(can)
    for g in range(9):
        p = m + g * 16 * S
        dr.line([(p, m), (p, m + 128 * S)], fill=INK, width=2)
        dr.line([(m, p), (m + 128 * S, p)], fill=INK, width=2)
    fs = font(15, mono=True)
    for r_ in range(8):
        for c_ in range(8):
            idx = r_ * 8 + c_
            x, y = m + c_ * 16 * S + 4, m + r_ * 16 * S + 3
            dr.rectangle([x - 2, y - 1, x + 20, y + 17], fill=BG)
            dr.text((x, y), f"{idx}", fill=EBT, font=fs)
    fl = font(20)
    dr.text((m, 8), "128 × 128 image = 8 × 8 patches of 16 px = 64 tokens", fill=INK, font=fl)
    dr.text((m, m + 128 * S + 8), "each token = 16·16·3 = 768 pixel values", fill=MUTED, font=fl)
    can.save(cdir / "patch_grid.png")

    # patches as a token sequence (raster order), 16 per row
    ps, gap = 48, 6
    cols = 16
    rows_n = 4
    seq = Image.new("RGB", (cols * (ps + gap) + gap, rows_n * (ps + gap + 18) + gap), BG)
    dr = ImageDraw.Draw(seq)
    fs2 = font(12, mono=True)
    for idx in range(64):
        r_, c_ = divmod(idx, 8)
        patch = src.crop((c_ * 16, r_ * 16, c_ * 16 + 16, r_ * 16 + 16)).resize((ps, ps), Image.NEAREST)
        rr_, cc_ = divmod(idx, cols)
        x, y = gap + cc_ * (ps + gap), gap + rr_ * (ps + gap + 18)
        seq.paste(patch, (x, y))
        dr.rectangle([x - 1, y - 1, x + ps, y + ps], outline=RULE)
        dr.text((x + 2, y + ps + 2), str(idx), fill=MUTED, font=fs2)
    seq.save(cdir / "patch_tokens.png")

    # ImageNet-1k validation images for the linear-probe explanation
    imnet = []
    for k, (fname, i) in enumerate(IN1K_PICKS):
        d = json.loads((DL / f"{fname}.json").read_text())
        names = d["features"][1]["type"]["names"]
        rr = d["rows"][i]
        lab = rr["row"]["label"]
        im = center_crop_arr(Image.open(DL / "img" / f"{fname}_{i}.jpg").convert("RGB"), 128)
        nn = f"{k + 1:02d}"
        im.save(idir / f"{nn}.jpg", quality=95)
        imnet.append({"id": nn, "file": f"media/samples/imagenet/{nn}.jpg", "label_index": lab,
                      "label": names[lab].split(",")[0].strip(), "label_full": names[lab],
                      "row_idx": rr["row_idx"]})

    ab10, ab20 = alpha_bar_repo(0.1), alpha_bar_repo(0.2)
    curve = []
    for s in np.linspace(0.0, 1.0, 21):
        if s == 0:
            curve.append({"sigma": 0.0, "alpha_bar": 1.0, "signal_scale": 1.0, "noise_std": 0.0})
            continue
        a = alpha_bar_repo(float(s))
        curve.append({"sigma": rnd(float(s)), "alpha_bar": rnd(a), "signal_scale": rnd(math.sqrt(a)),
                      "noise_std": rnd(math.sqrt(1 - a))})
    mean_psnr = {tag: rnd(float(np.mean([c["noisy"][tag]["psnr_db_vs_clean"] for c in coco])))
                 for tag in ("s10", "s20")}
    mean_mse = {tag: rnd(float(np.mean([c["noisy"][tag]["mse_pixel_vs_clean"] for c in coco])))
                for tag in ("s10", "s20")}
    return {
        "_meta": {
            "coco_source": "AbdoTW/COCO_2014 (Hugging Face; the exact mirror the paper cites as ref [120] and its "
                           "code loads), splits train and validation, via datasets-server /rows image URLs.",
            "coco_license": "COCO annotations CC BY 4.0; each image keeps its Flickr license (per-image `license`).",
            "preprocess": "center_crop_arr(image, 128) exactly as in the authors' code (ADM/DiT crop: halve with BOX "
                          "filter while min side >= 256, BICUBIC resize min side to 128, center crop), then pixels "
                          "scaled to [-1, 1] (Normalize mean 0.5, std 0.5).",
            "splits_note": "COCO 01-08 are from train (what the model learns on); 09-12 from validation "
                           "(what evaluation uses). Paper: 'its train/validation split' (p.34).",
            "pixel_space": "Image denoising runs in raw pixel space, 3 x 128 x 128, patch 16 -> 64 tokens (from the "
                           "authors' public code, model/model_utils.py setup_bidirectional_ebt; the paper text says "
                           "128x128, patch size 16).",
            "imagenet_source": "benjamin-paine/imagenet-1k-128x128 (Hugging Face, validation split; ImageNet-1k images "
                               "pre-resized to 128 px). License: ImageNet terms of access (non-commercial research and "
                               "education). Used only to illustrate the linear-probe evaluation.",
            "linear_probe": "Paper (method p.14 and p.35; Table 4 on p.13): average all final patch tokens, train a linear classifier on "
                            "ImageNet-1k; for DiT feed T = 0. Top-1 0.31% (DiT) vs 5.32% (EBT); top-5 1.36% vs 13.2%.",
        },
        "noise": {
            "schedule": "linear beta 1e-4 -> 2e-2 over T = 1000 (DiT), sigma = fraction of the schedule",
            "paper_ref": "Sec 4.3, p.13: sigma = 0.1 for training / in-distribution, 0.2 for OOD testing",
            "formula": "x_t = sqrt(alpha_bar) * x + sqrt(1 - alpha_bar) * eps, eps ~ N(0, I), x in [-1, 1]",
            "implementation": "As in the authors' code (model/img/ebt_denoise.py): create_diffusion(diffusion_steps = "
                              "int(1000 / sigma)) rescales the linear schedule by 1000/N, then q_sample at t = 999. "
                              "This equals applying the first sigma*1000 steps of the 1000-step schedule up to "
                              "discretization.",
            "alpha_bar": {"0.1": rnd(ab10), "0.2": rnd(ab20)},
            "alpha_bar_plain_schedule_check": {"0.1": rnd(alpha_bar_simple(0.1)), "0.2": rnd(alpha_bar_simple(0.2))},
            "signal_scale": {"0.1": rnd(math.sqrt(ab10)), "0.2": rnd(math.sqrt(ab20))},
            "noise_std_in_[-1,1]_units": {"0.1": rnd(math.sqrt(1 - ab10)), "0.2": rnd(math.sqrt(1 - ab20))},
            "noisy_input_mean_psnr_db": mean_psnr,
            "noisy_input_mean_mse_pixel": mean_mse,
            "noisy_psnr_note": "PSNR/MSE of the NOISY INPUT vs the clean image over our 12 COCO crops, 0-255 pixel "
                               "units after clipping to [-1, 1] (computed before JPEG). For scale: paper Table 4 "
                               "denoised outputs reach PSNR 27.25 (EBT) / 26.58 (DiT) at sigma 0.1 and 23.29 / 19.56 "
                               "at sigma 0.2.",
            "curve": curve,
            "seeds": "noise seeds fixed per image (numpy default_rng); JPEG q92 4:4:4 adds slight compression",
        },
        "coco": coco,
        "noise_ladder": {"file": "media/samples/coco/noise_ladder.png", "image": "01", "levels": ladder_meta},
        "patch_grid": {"file": "media/samples/coco/patch_grid.png", "tokens_file": "media/samples/coco/patch_tokens.png",
                       "image": "01", "image_px": 128, "patch_px": 16, "grid": [8, 8], "n_tokens": 64,
                       "token_dim": 768, "order": "raster (row-major), index = row*8 + col"},
        "imagenet": imnet,
    }


# ----------------------------------------------------------------------------------------------
# 4) Video
# ----------------------------------------------------------------------------------------------
MMNIST_ROW = 1  # two digits that cross (occlusion mid-clip)
SSV2_PICKS = [("ssv2grid_120", 9), ("ssv2grid_120", 11), ("ssv2grid_40", 7), ("ssv2grid_0", 1)]


def split_grid(img):
    a = np.asarray(img.convert("L"), dtype=np.float64)

    def runs(profile, n=3):
        bright = profile > 20
        segs, start = [], None
        for i, b in enumerate(bright):
            if b and start is None:
                start = i
            if not b and start is not None:
                segs.append((start, i))
                start = None
        if start is not None:
            segs.append((start, len(bright)))
        segs = sorted(segs, key=lambda s: s[1] - s[0], reverse=True)[:n]
        return sorted(segs)

    cols = runs(a.mean(axis=0))
    rws = runs(a.mean(axis=1))
    if len(cols) != 3 or len(rws) != 3:
        h, w = a.shape
        cols = [(i * w // 3, (i + 1) * w // 3) for i in range(3)]
        rws = [(i * h // 3, (i + 1) * h // 3) for i in range(3)]
    tiles = []
    for (y0, y1) in rws:
        for (x0, x1) in cols:
            tiles.append(img.crop((x0, y0, x1, y1)))
    return tiles


def build_video():
    vdir = MEDIA / "ssv2"
    vdir.mkdir(parents=True, exist_ok=True)
    clips = []
    for k, (fname, i) in enumerate(SSV2_PICKS):
        rr = rows(fname)[i]
        r = rr["row"]
        g = Image.open(DL / "img" / f"{fname}_{i}.jpg").convert("RGB")
        tiles = split_grid(g)
        files = []
        for j, t in enumerate(tiles):
            p = vdir / f"c{k + 1:02d}_f{j + 1}.jpg"
            t.save(p, quality=90)
            files.append(f"media/samples/ssv2/c{k + 1:02d}_f{j + 1}.jpg")
        clips.append({"id": f"c{k + 1:02d}", "video_id": r["video_id"], "label": r["label_text"],
                      "template": r["annotation_text"], "frames": files, "frame_size": list(tiles[0].size),
                      "row_idx": rr["row_idx"]})

    mdir = MEDIA / "moving_mnist"
    mdir.mkdir(parents=True, exist_ok=True)
    for old in mdir.glob("*.png"):
        old.unlink()
    mm = rows("mmnist_classic_0")
    best = MMNIST_ROW
    row = mm[best]["row"]
    vid = np.array(row["video"], dtype=np.uint8)
    files = []
    for j in range(vid.shape[0]):
        p = mdir / f"f{j + 1:02d}.png"
        Image.fromarray(vid[j]).save(p, optimize=True)
        files.append(f"media/samples/moving_mnist/f{j + 1:02d}.png")
    T = vid.shape[0]
    fp = vid.shape[1]
    strip = Image.new("L", (T * fp + (T - 1) * 4, fp), 13)
    for j in range(T):
        strip.paste(Image.fromarray(vid[j]), (j * (fp + 4), 0))
    strip.save(mdir / "strip.png", optimize=True)
    return {
        "_meta": {
            "paper_dataset": "Something-Something V2 (SSV2), standard train/val split; frames 224x224 (Resize, "
                             "aspect not preserved, in the authors' code), 0.25 s between frames, encoded by a frozen "
                             "SD-XL VAE into 3136-dim features (p.12; our arithmetic: 4 latent channels x 28 x 28 for "
                             "a 224 px frame with 8x downsampling); Smooth L1 loss with beta 1.0 (p.12); context "
                             "length 16 frames (Tables D.2, D.3).",
            "ssv2_access": "Official SSV2 requires accepting Qualcomm's license (academic research use). These 4 clips "
                           "come from an unofficial public Hugging Face mirror, moondream/ssv2-3x3 (test split), "
                           "which stores 9 frames per video as a 3x3 collage; we split the collage into frames. "
                           "Frame sampling interval is not documented by the mirror (row-major order matches time, "
                           "checked visually). Lead: decide whether to ship these frames publicly; for a fully clean "
                           "option use the paper's own Fig 11.",
            "ssv2_license": "SSV2 terms (Qualcomm / 20BN), research use; mirror card states no license.",
            "moving_mnist": "Classic Moving MNIST (Srivastava et al. 2015, mnist_test_seq.npy, 20 frames of 64x64, two "
                            "bouncing MNIST digits) via the Hugging Face mirror ryushinn/MMNIST (license not stated on "
                            "the mirror; original distributed freely at cs.toronto.edu/~nitish/unsupervised_video). "
                            "NOT the paper's dataset. Use only to illustrate next-frame prediction mechanics.",
            "fig11": "Paper Fig 11 (p.14): EBT energy per frame on an SSV2 clip. Energy is high at the start "
                     "(frame mostly empty, unpredictable), drops as a blue garment is placed into the frame, and "
                     "rises again when it is removed. Use the paper's figure for the actual energies.",
        },
        "ssv2": {"available": True, "source": "moondream/ssv2-3x3", "split": "test",
                 "is_official_release": False, "clips": clips},
        "moving_mnist": {"source": "ryushinn/MMNIST", "split": "train (the mirror's only split; it holds the "
                         "original mnist_test_seq.npy)", "row_idx": mm[best]["row_idx"], "not_paper_dataset": True,
                         "frame_px": int(fp), "n_frames": int(T), "frames": files,
                         "strip": "media/samples/moving_mnist/strip.png"},
        "paper_fig11": {"page": 14, "description": "energy per frame tracks uncertainty (see _meta.fig11)"},
    }


def build_setups():
    """Splits & setups table. Every entry is from notes/paper_text.txt (page numbers given)."""
    return {
        "_meta": {"source": "arXiv 2507.02092v1 text; page numbers refer to the PDF pages",
                  "note": "Facts about the paper's setups, for a 'Splits & setups' table. Not model outputs."},
        "rows": [
            {"modality": "Text (autoregressive LM)",
             "data": "RedPajamaV2, 100B-token sample from Hugging Face (p.8)",
             "split": "manual split: 66M training / 33K validation samples (p.8)",
             "input": "GPT-NeoX tokenizer, vocab 50277, context length 256 (p.8; Tables D.2, D.3)",
             "loss": "categorical cross-entropy (p.7)",
             "baseline": "Transformer++ (Llama 2 implementation; EBTs built on the same backbone, p.33)",
             "metrics": "pretraining (validation) perplexity; downstream perplexity on GSM8K, SQuAD, BigBench "
                        "Elementary Math QA, BigBench Dyck Languages (p.9)",
             "regime": "each sample seen once due to dataset size (p.12)"},
            {"modality": "Video (autoregressive next frame)",
             "data": "Something-Something V2 (p.12)",
             "split": "standard SSV2 train and validation split (p.34)",
             "input": "224 x 224 frames, 0.25 s apart, encoded by a frozen SD-XL VAE into 3136-dim features; "
                      "context length 16 (p.12, p.34; Tables D.2, D.3)",
             "loss": "Smooth L1, beta = 1.0 (p.12)",
             "baseline": "Transformer++ (p.12)",
             "metrics": "minimum validation loss achieved (p.12)",
             "regime": "small dataset, many epochs: 'how well can models fit a fixed dataset?' (p.12)"},
            {"modality": "Images (bidirectional denoising)",
             "data": "COCO 2014 (Hugging Face AbdoTW/COCO_2014) (p.13, ref [120])",
             "split": "COCO train / validation split (p.34)",
             "input": "128 x 128 pixels, patch size 16; noise from a linear variance schedule 1e-4 to 2e-2 with "
                      "sigma = fraction of the schedule: 0.1 for training and in-distribution test, 0.2 for OOD "
                      "test (p.13)",
             "loss": "mean squared error (paper text, p.7). Note: the authors' public code (model/img/ebt_denoise.py, "
                     "checked at commit 19420cb) trains the EBT denoiser with Smooth L1 (beta 1.0); metrics are "
                     "PSNR / pixel MSE.",
             "baseline": "DiT (Diffusion Transformer implementation from [26]); best inference found was DDIM "
                         "applied recursively on its own output (p.35)",
             "metrics": "PSNR and pixel MSE at sigma 0.1 and 0.2; ImageNet-1k linear probe top-1 / top-5 using "
                        "the average of final patch tokens (DiT fed T = 0) (Table 4 p.13; p.14; p.35)",
             "regime": "large model size, lr 1e-4, 100,000 steps (p.34). For OOD noise both models denoise their "
                       "own output twice more (3 applications): 300 forward passes for DiT vs 3 for EBT (Fig 12, "
                       "p.35)"},
        ],
    }


def main():
    out = {
        "_meta": {
            "source": "Real dataset samples fetched from Hugging Face for the EBT explainer (see per-section _meta).",
            "note": "Everything here is real data from public datasets, not model outputs. Paper numbers quoted "
                    "(Table 3, Table 4) are from arXiv 2507.02092v1. Images live in media/samples/.",
            "datasets": {
                "pretraining_text": "togethercomputer/RedPajama-Data-V2 (config sample, 2023-06 en_head/en_middle)",
                "tokenizer": "EleutherAI/gpt-neox-20b (tokenizer.json)",
                "gsm8k": "openai/gsm8k main/test",
                "squad": "rajpurkar/squad plain_text/validation",
                "bigbench_dyck": "tasksource/bigbench dyck_languages/validation",
                "bigbench_elementary_math_qa": "tasksource/bigbench elementary_math_qa/validation",
                "coco": "AbdoTW/COCO_2014 default/train + validation",
                "imagenet": "benjamin-paine/imagenet-1k-128x128 default/validation",
                "ssv2": "moondream/ssv2-3x3 default/test (unofficial mirror of Something-Something V2)",
                "moving_mnist": "ryushinn/MMNIST default/train = classic Moving MNIST test sequences (illustration only)",
            },
            "built_by": "src/samples_fetch.py + src/build_samples.py",
        },
        "text": build_text(),
        "eval": build_eval(),
        "images": build_images(),
        "video": build_video(),
        "setups": build_setups(),
    }
    p = ROOT / "data" / "samples.json"
    p.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    print(f"wrote {p} ({p.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
