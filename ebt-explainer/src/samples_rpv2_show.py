"""Print the first N characters of selected RedPajama-V2 docs (by bucket#index).

Usage: python3 -I samples_rpv2_show.py <downloads_dir> <nchars> en_head#12 en_middle#3 ...
"""
import json
import sys
import zlib
from pathlib import Path


def read_partial_gz(path):
    raw = Path(path).read_bytes()
    d = zlib.decompressobj(16 + zlib.MAX_WBITS)
    return d.decompress(raw).decode("utf-8", errors="ignore").split("\n")[:-1]


def main():
    ddir = Path(sys.argv[1])
    n = int(sys.argv[2])
    cache = {}
    for key in sys.argv[3:]:
        bucket, idx = key.split("#")
        if bucket not in cache:
            cache[bucket] = read_partial_gz(ddir / f"rpv2_{bucket}.part.gz")
        doc = json.loads(cache[bucket][int(idx)])
        print(f"===== {key} | {doc.get('source_domain')} | {doc.get('title')}")
        print(repr(doc["raw_content"][:n]))


if __name__ == "__main__":
    main()
