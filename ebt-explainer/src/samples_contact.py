"""Make a labelled contact sheet of images (for visual review only).

Usage: python3 -I samples_contact.py <out.png> <thumb_px> <img> [<img> ...]
"""
import sys
from pathlib import Path

from PIL import Image, ImageDraw


def main():
    out, px = sys.argv[1], int(sys.argv[2])
    paths = sys.argv[3:]
    cols = 6
    rows = (len(paths) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (px + 8), rows * (px + 24)), (13, 19, 31))
    dr = ImageDraw.Draw(sheet)
    for k, p in enumerate(paths):
        im = Image.open(p).convert("RGB")
        im.thumbnail((px, px))
        x, y = (k % cols) * (px + 8), (k // cols) * (px + 24)
        sheet.paste(im, (x, y))
        dr.text((x + 2, y + px + 4), Path(p).stem[-28:], fill=(233, 238, 246))
    sheet.save(out)
    print(out, sheet.size)


if __name__ == "__main__":
    main()
