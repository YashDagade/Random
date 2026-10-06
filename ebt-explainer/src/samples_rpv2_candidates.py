"""List candidate RedPajama-V2 documents from partial (range-downloaded) gzip shards.

Usage: python3 -I samples_rpv2_candidates.py <downloads_dir> [max_per_file]
Prints index, bucket, domain, title, length and an excerpt so a human can pick
diverse, benign documents. Reads untrusted data; does not execute anything.
"""
import gzip
import json
import re
import sys
import zlib
from pathlib import Path

BLOCK = re.compile(
    r"\b(sex|porn|xxx|nude|escort|casino|viagra|cialis|dating|hookup|onlyfans|bet|betting|"
    r"loan|payday|crypto|forex|gun|kill|murder|rape|drug|weed|cbd|vape)\b",
    re.I,
)
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE = re.compile(r"\(?\d{3}\)?[-. ]\d{3}[-. ]\d{4}")


def read_partial_gz(path):
    """Decompress as much of a truncated gzip file as possible."""
    raw = Path(path).read_bytes()
    d = zlib.decompressobj(16 + zlib.MAX_WBITS)
    out = d.decompress(raw)
    text = out.decode("utf-8", errors="ignore")
    lines = text.split("\n")
    return lines[:-1]  # last line is probably cut


def main():
    ddir = Path(sys.argv[1])
    maxn = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    for bucket in ("en_head", "en_middle"):
        lines = read_partial_gz(ddir / f"rpv2_{bucket}.part.gz")
        shown = 0
        for i, ln in enumerate(lines):
            try:
                doc = json.loads(ln)
            except Exception:
                continue
            txt = doc.get("raw_content", "")
            head = txt[:700]
            if len(txt) < 900:
                continue
            if BLOCK.search(txt[:3000]) or EMAIL.search(head) or PHONE.search(head):
                continue
            words = head.split()
            if len(words) < 80:
                continue
            # prefer prose: share of lines ending with punctuation in the head
            hl = [h for h in head.split("\n") if h.strip()]
            prose = sum(1 for h in hl if h.strip()[-1:] in ".!?\"'") / max(1, len(hl))
            if prose < 0.4:
                continue
            title = (doc.get("title") or "")[:70]
            ex = head[:160].replace("\n", " | ")
            print(f"{bucket}#{i}\t{doc.get('source_domain')}\t{len(txt)}\t{title}\n    {ex}")
            shown += 1
            if shown >= maxn:
                break


if __name__ == "__main__":
    main()
