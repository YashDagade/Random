"""Vector-plot extraction helpers (matplotlib PDFs embedded in the paper)."""
import os
import pymupdf as fitz
import numpy as np

PDF = os.environ.get('EBT_PDF', '/tmp/claude-0/-home-user-Random/9843a8a1-7b0a-5ed7-8867-6762f4a2ea11/scratchpad/ebt/paper/ebt.pdf')
_doc = None
def doc():
    global _doc
    if _doc is None: _doc = fitz.open(PDF)
    return _doc

def col(c): return None if c is None else tuple(round(x,3) for x in c)

def find_axes(pno):
    """Axes = white filled rect nested inside another (larger) white filled rect (figure patch)."""
    p = doc()[pno-1]
    wr = [d['rect'] for d in p.get_drawings() if d['type']=='f' and col(d.get('fill'))==(1,1,1) and d['items'][0][0]=='re']
    out=[]
    for r in wr:
        if any((o.contains(r) and o!=r) for o in wr): out.append(r)
    return sorted(out, key=lambda r:(round(r.y0/50),r.x0))

def collect(pno, frame):
    p = doc()[pno-1]
    res = dict(frame=frame, xticks=[], yticks=[], markers=[], lines=[], legend=None, glyphs=[])
    tol = 0.15
    for d in p.get_drawings():
        it = d['items']
        c = col(d.get('color')); f = col(d.get('fill')); w = d.get('width') or 0
        r = d['rect']
        if d['type']=='s' and len(it)==1 and it[0][0]=='l' and c==(0,0,0):
            a,b = it[0][1], it[0][2]
            L = abs(a.x-b.x)+abs(a.y-b.y)
            if L < 5 and L > 0.3:
                if abs(a.x-b.x)<0.01 and frame.x0-tol<=a.x<=frame.x1+tol and (abs(min(a.y,b.y)-frame.y1)<tol):
                    res['xticks'].append((a.x, round(w,2), round(L,2)))
                if abs(a.y-b.y)<0.01 and frame.y0-tol<=a.y<=frame.y1+tol and (abs(max(a.x,b.x)-frame.x0)<tol):
                    res['yticks'].append((a.y, round(w,2), round(L,2)))
        if d['type']=='fs' and c==(0.8,0.8,0.8) and f==(1,1,1) and frame.contains(r):
            res['legend']=r
    for d in p.get_drawings():
        it = d['items']; c = col(d.get('color')); f = col(d.get('fill')); r = d['rect']
        if d['type']=='f' and f==(0,0,0):
            res['glyphs'].append(r); continue
        if c in [(0,0,0),(0.502,0.502,0.502),(0.8,0.8,0.8)] : continue
        center = fitz.Point((r.x0+r.x1)/2,(r.y0+r.y1)/2)
        if not fitz.Rect(frame.x0-3,frame.y0-3,frame.x1+3,frame.y1+3).contains(center): continue
        if res['legend'] is not None and res['legend'].contains(center): continue
        if d['type'] in ('fs','f') and r.width<6 and r.height<6:
            res['markers'].append(dict(color=f or c, x=center.x, y=center.y, n=len(it), w=r.width, h=r.height))
        elif d['type']=='s' and len(it)>=1:
            pts=[]
            for item in it:
                if item[0]=='l':
                    if not pts: pts.append((item[1].x,item[1].y))
                    pts.append((item[2].x,item[2].y))
                elif item[0]=='c':
                    if not pts: pts.append((item[1].x,item[1].y))
                    pts.append((item[4].x,item[4].y))
            res['lines'].append(dict(color=c, pts=pts, dashes=d.get('dashes'), width=d.get('width')))
    res['xticks'].sort(); res['yticks'].sort()
    return res
