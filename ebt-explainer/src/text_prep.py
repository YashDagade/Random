"""Prepare the character-level corpora for the tiny text EBT toy.

Reads (untrusted, downloaded) raw files from a directory given on the command
line and writes integer-encoded arrays + metadata to an output directory.

  python3 -I src/text_prep.py <raw_dir> <out_dir>

In-distribution corpus: RedPajama-Data-V2, sample, snapshot 2023-06, shard 0000,
`en_head.json.gz` (the first ~3 MB of the gzip stream, fetched with an HTTP range
request from data.together.xyz, the official RedPajama-V2 file host).
OOD corpora: tiny-shakespeare (karpathy/char-rnn) and a few CPython stdlib files.
"""
import json
import sys
import zlib
import os
import numpy as np

VOCAB = [" ", "\n"] + list("abcdefghijklmnopqrstuvwxyz") + list("0123456789") + list(".,;:!?'\"-()/&%$") + ["#"]
UNK = "#"
STOI = {c: i for i, c in enumerate(VOCAB)}

TRANS = {
    "’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-",
    "…": "...", " ": " ", "\t": " ", "\r": "",
}


def normalize(s):
    out = []
    for ch in s:
        ch = TRANS.get(ch, ch)
        for c in ch:
            c = c.lower()
            if c in STOI:
                out.append(c)
            elif ord(c) < 128:
                out.append(UNK)
            # non-ascii (accents, emoji, cjk) dropped
    s = "".join(out)
    # collapse runs of spaces and long newline runs
    import re
    s = re.sub(r" {2,}", " ", s)
    s = re.sub(r" *\n *", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s


def encode(s):
    return np.array([STOI[c] for c in s], dtype=np.uint8)


def main(raw_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    d = zlib.decompressobj(16 + zlib.MAX_WBITS)
    raw = d.decompress(open(os.path.join(raw_dir, "rpv2_part.gz"), "rb").read())
    lines = raw.split(b"\n")
    docs = []
    for ln in lines[:-1]:  # last line may be truncated
        try:
            o = json.loads(ln)
        except Exception:
            continue
        t = normalize(o["raw_content"])
        if len(t) > 400:
            docs.append(t)
    print("docs", len(docs))
    # document-level split: last 6% of documents -> validation
    n_val = max(1, int(0.06 * len(docs)))
    train_docs, val_docs = docs[:-n_val], docs[-n_val:]
    train = "\n\n".join(train_docs)
    val = "\n\n".join(val_docs)
    shk = normalize(open(os.path.join(raw_dir, "tinyshakespeare.txt"), encoding="utf-8").read())
    shk = shk[-300000:]  # last 300k chars as OOD eval text
    code = "\n\n".join(normalize(open(os.path.join(raw_dir, f), encoding="utf-8").read())
                       for f in ["code_argparse.py", "code_textwrap.py", "code_json_decoder.py"])
    out = {}
    for name, s in [("train", train), ("val", val), ("ood_shakespeare", shk), ("ood_code", code)]:
        arr = encode(s)
        np.save(os.path.join(out_dir, name + ".npy"), arr)
        counts = np.bincount(arr, minlength=len(VOCAB))
        out[name] = {"chars": int(len(arr)), "unk_frac": float(counts[STOI[UNK]] / len(arr)),
                     "sample": s[1000:1400]}
        print(name, len(arr), "unk frac %.4f" % (counts[STOI[UNK]] / len(arr)))
    meta = {"vocab": VOCAB, "n_train_docs": len(train_docs), "n_val_docs": len(val_docs), "splits": out}
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
