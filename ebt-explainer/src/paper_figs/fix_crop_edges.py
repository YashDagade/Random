"""Widen plot crops in media/paper/ so no axes frame (spine) is cut off (checker pass).

Run:  python3 -I src/paper_figs/fix_crop_edges.py   (after crop_figs.py)

Why: crop_figs.py takes the union of element rectangles, but pymupdf's Rect union ignores
empty rectangles, and a vertical spine is a zero-width rectangle. Where nothing else reaches
past the right spine, the crop ended up to 4 pt inside the plot frame (Fig 6a, 9, 12, B.1, C.1),
or exactly on it with no margin (Fig 4c, 5a-c). This script grows each plot crop so every axes
frame inside it, plus its line width and a small white margin, is included, re-renders the PNG
at the same 200 dpi, and updates clip_pt/size_px in crops.json and width/height in
data/paper_figs.json. It never moves a crop edge inwards.
"""
import os, sys, json, site
sys.path.append(site.getusersitepackages())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pymupdf as fitz
import vecplot as V

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
CROPS = os.environ.get('EBT_CROPS_JSON', '/tmp/claude-0/-home-user-Random/9843a8a1-7b0a-5ed7-8867-6762f4a2ea11/scratchpad/ebt/digitize_check/crops.json')
FIGS = os.path.join(ROOT, 'data', 'paper_figs.json')
DPI = 200
MARGIN = 2.0      # pt of white beyond the outer edge of a spine (crop_figs.py pads by 2.0 too)
SPINE = 0.21      # half of the 0.41 pt spine width

crops = json.load(open(CROPS))
byid = {c['id']: c for c in crops}
figs = json.load(open(FIGS))
fbyid = {f['id']: f for f in figs['figures']}
changed = []
for c in crops:
    if fbyid.get(c['id'], {}).get('kind') != 'plot': continue
    page = V.doc()[c['page'] - 1]
    clip = fitz.Rect(c['clip_pt'])
    frames = [fr for fr in V.find_axes(c['page']) if clip.contains(fitz.Point((fr.x0 + fr.x1) / 2, (fr.y0 + fr.y1) / 2))]
    if not frames: continue
    # content that belongs to other panels / body text: anything whose centre is outside this crop
    # and outside this crop's own axes frames (a cut-off spine has its centre outside the old crop)
    inflated = [fitz.Rect(fr.x0 - 1, fr.y0 - 1, fr.x1 + 1, fr.y1 + 1) for fr in frames]
    def foreign(r):
        ctr = fitz.Point((r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2)
        return not clip.contains(ctr) and not any(f.contains(ctr) for f in inflated)
    others = []
    for d in page.get_drawings():
        if d['type'] == 'f' and d.get('fill') and tuple(round(v, 3) for v in d['fill']) == (1, 1, 1): continue
        r = d['rect']
        if foreign(r): others.append(fitz.Rect(r))
    for b in page.get_text('dict')['blocks']:
        for l in b.get('lines', []):
            for s in l['spans']:
                r = fitz.Rect(s['bbox'])
                if s['text'].strip() and foreign(r): others.append(r)
    new = fitz.Rect(clip)
    for fr in frames:
        need = fitz.Rect(fr.x0 - SPINE - MARGIN, fr.y0 - SPINE - MARGIN, fr.x1 + SPINE + MARGIN, fr.y1 + SPINE + MARGIN)
        # right edge: stop before any foreign content
        if need.x1 > new.x1:
            x1 = need.x1
            for o in others:
                if o.x0 >= new.x1 - 0.01 and o.x0 < x1 and o.y1 > new.y0 and o.y0 < new.y1: x1 = min(x1, o.x0 - 0.5)
            new.x1 = max(new.x1, x1)
        if need.x0 < new.x0:
            x0 = need.x0
            for o in others:
                if o.x1 <= new.x0 + 0.01 and o.x1 > x0 and o.y1 > new.y0 and o.y0 < new.y1: x0 = max(x0, o.x1 + 0.5)
            new.x0 = min(new.x0, x0)
        if need.y0 < new.y0: new.y0 = need.y0
        if need.y1 > new.y1: new.y1 = need.y1
    if max(abs(a - b) for a, b in zip(new, clip)) < 0.02: continue
    pix = page.get_pixmap(dpi=DPI, clip=new, alpha=False)
    pix.save(os.path.join(ROOT, c['file']))
    old_px = c['size_px']
    c['clip_pt'] = [round(v, 2) for v in new]
    c['size_px'] = [pix.width, pix.height]
    if c['id'] in fbyid:
        fbyid[c['id']]['width'], fbyid[c['id']]['height'] = pix.width, pix.height
    changed.append(c['id'])
    print(f"{c['id']:7s} clip {[round(v, 1) for v in clip]} -> {[round(v, 1) for v in new]}  px {old_px} -> {[pix.width, pix.height]}")

json.dump(crops, open(CROPS, 'w'), indent=1)
json.dump(figs, open(FIGS, 'w'), ensure_ascii=False, indent=1)
print('re-cropped', len(changed), ':', ' '.join(changed))
