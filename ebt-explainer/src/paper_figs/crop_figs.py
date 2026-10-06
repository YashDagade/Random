import os
import sys, json, re, os
import pymupdf as fitz
PDF=os.environ.get('EBT_PDF', '/tmp/claude-0/-home-user-Random/9843a8a1-7b0a-5ed7-8867-6762f4a2ea11/scratchpad/ebt/paper/ebt.pdf')
ROOT='/home/user/Random/ebt-explainer'
OUTDIR=ROOT+'/media/paper'
DPI=200
doc=fitz.open(PDF)

# id, page, region (pt), caption prefix, splits (x positions) for panels, panel labels
FIGS = [
 dict(id='fig01', page=2, region=(100,62,512,206), cap='Figure 1:', splits=[203,302,406], labels=['(a) AR Transformer','(b) RNN','(c) Diffusion Transformer','(d) EBT']),
 dict(id='fig02', page=4, region=(100,62,512,207), cap='Figure 2:'),
 dict(id='fig03', page=5, region=(100,62,512,231), cap='Figure 3:'),
 dict(id='fig04', page=9, region=(100,62,512,182), cap='Figure 4:', splits=[245.5,378], labels=['(a) Scaling for data.','(b) Scaling for batch size.','(c) Scaling for depth.']),
 dict(id='fig05', page=10, region=(100,62,512,183), cap='Figure 5:', splits=[245.5,378], labels=['(a) Scaling for number of Parameters.','(b) Scaling for number of FLOPs.','(c) Scaling for the embed. dimension.']),
 dict(id='fig06', page=11, region=(100,62,512,221), cap='Figure 6:', splits=[305], labels=['(a) OOD Thinking Performance Comparison.','(b) Verification Capabilities as Scale Increases.']),
 dict(id='fig07', page=11, region=(302,384,512,541), cap='Figure 7:'),
 dict(id='fig08', page=12, region=(100,180,512,347), cap='Figure 8:', splits=[306], labels=['(a) "The quick brown fox jumps over the lazy dog."','(b) "System 2 Thinking is a challenging but interesting research problem."']),
 dict(id='fig09', page=13, region=(100,62,512,220), cap='Figure 9:', splits=[305], labels=['(a) Scaling for the embedding dimension.','(b) Scaling for the Non-Embedding Parameters.']),
 dict(id='fig10', page=14, region=(100,62,512,189), cap='Figure 10:'),
 dict(id='fig11', page=14, region=(302,340,512,543), cap='Figure 11:'),
 dict(id='fig12', page=15, region=(302,192,512,347), cap='Figure 12:'),
 dict(id='figB1', page=28, region=(100,62,512,220), cap='Figure B.1:', splits=[305], labels=['(a) Self-verification with BoN-10 versus BoN-2.','(b) Results in Fig. 6b projected to Llama3 scale [84].']),
 dict(id='figB2', page=29, region=(100,62,512,262), cap='Figure B.2:'),
 dict(id='figB3', page=29, region=(100,330,512,502), cap='Figure B.3:', splits=[305], labels=['(a) Data Scaling Higher Scale.','(b) Data Scaling Zoomed In.']),
 dict(id='figC1', page=31, region=(100,62,512,206), cap='Figure C.1:'),
 dict(id='figE1', page=37, region=(100,62,512,228), cap='Figure E.1:', splits=[306], labels=['(a) Diffusion Model','(b) Energy-Based Model (EBM)']),
 dict(id='figH1', page=40, region=(100,495,512,636), cap='Figure H.1:', splits=[289], labels=['(a) Feed Forward Model','(b) Energy-Based Model']),
]

def elements(page, region):
    R=fitz.Rect(region); els=[]
    for d in page.get_drawings():
        if d['type']=='f' and d.get('fill') and tuple(round(v,3) for v in d['fill'])==(1,1,1): continue
        r=d['rect']
        c=fitz.Point((r.x0+r.x1)/2,(r.y0+r.y1)/2)
        if R.contains(c): els.append(('d',fitz.Rect(r)))
    for im in page.get_images(full=True):
        for r in page.get_image_rects(im[0]):
            c=fitz.Point((r.x0+r.x1)/2,(r.y0+r.y1)/2)
            if R.contains(c): els.append(('i',fitz.Rect(r)))
    td=page.get_text('dict', clip=R)
    for b in td['blocks']:
        for l in b.get('lines',[]):
            for s in l['spans']:
                if not s['text'].strip(): continue
                r=fitz.Rect(s['bbox']); c=fitz.Point((r.x0+r.x1)/2,(r.y0+r.y1)/2)
                if R.contains(c): els.append(('t',r,s['text']))
    return els

def subcap_rows(els):
    rows=[]
    for e in els:
        if e[0]=='t' and re.match(r'^\([a-h]\)', e[2].strip()): rows.append(e[1].y0)
    return min(rows) if rows else None

def union(els):
    u=None
    for e in els:
        r=e[1]
        u = fitz.Rect(r) if u is None else u | r
    return u

def caption_text(page, prefix):
    blocks=page.get_text('blocks')
    for b in blocks:
        t=b[4].strip()
        if t.startswith(prefix):
            rect=fitz.Rect(b[:4]); texts=[t]
            # follow continuation blocks (PyMuPDF sometimes splits a caption paragraph)
            while True:
                nxt=[c for c in blocks if 0 <= c[1]-rect.y1 < 3.0 and abs(c[0]-rect.x0) < 4 and c[4].strip()]
                if not nxt: break
                c=min(nxt, key=lambda c:c[1]); texts.append(c[4].strip()); rect=rect|fitz.Rect(c[:4])
            lines=[ln.strip() for t in texts for ln in t.split('\n') if ln.strip()]
            out=''
            for ln in lines:
                if out.endswith('-') and re.match(r'^[a-z]', ln) and re.search(r'[A-Za-z]-$', out):
                    out=out[:-1]+ln   # line-break hyphenation inside a word ("Re- sults")
                elif out: out+=' '+ln
                else: out=ln
            out=out.replace('\u02c6y','\u0177').replace('ˆy','ŷ')
            return out, fitz.Rect(b[:4])
    return None, None

def patches(page, region):
    R=fitz.Rect(region)
    wr=[d['rect'] for d in page.get_drawings() if d['type']=='f' and d.get('fill') and tuple(round(v,3) for v in d['fill'])==(1,1,1) and d['items'][0][0]=='re']
    outer=[r for r in wr if not any(o.contains(r) and o!=r for o in wr) and R.contains(fitz.Point((r.x0+r.x1)/2,(r.y0+r.y1)/2))]
    return sorted(outer, key=lambda r:r.x0)

def save(page, rect, path, pad=2.0, limit=None):
    r=fitz.Rect(rect.x0-pad, rect.y0-pad, rect.x1+pad, rect.y1+pad)
    if limit is not None: r = r & limit
    pix=page.get_pixmap(dpi=DPI, clip=r, alpha=False)
    pix.save(path)
    return [round(v,2) for v in r], pix.width, pix.height

records=[]
for F in FIGS:
    page=doc[F['page']-1]
    cap, caprect = caption_text(page, F['cap'])
    assert cap, F['id']
    els=elements(page, F['region'])
    # guard: nothing may overlap the caption
    u=union(els)
    assert u.y1 <= caprect.y0+0.5, (F['id'], u, caprect)
    fpath=f"{OUTDIR}/{F['id']}.png"
    clip,w,h=save(page,u,fpath)
    rec=dict(id=F['id'], file=f"media/paper/{F['id']}.png", caption=cap, page=F['page'], panel=None, clip_pt=clip, size_px=[w,h])
    sub_y=subcap_rows(els)
    if F.get('splits'):
        rec['panels']=[]
        pts=patches(page, F['region'])
        if len(pts)==len(F['labels']):
            prs=[(p.x0,p.y0,p.x1,p.y1) for p in pts]; lims=[fitz.Rect(p.x0-0.6,p.y0-0.6,p.x1+0.6,p.y1+0.6) for p in pts]
        else:
            xs=[F['region'][0]]+F['splits']+[F['region'][2]]
            prs=[(xs[k], F['region'][1], xs[k+1], (sub_y-0.5) if sub_y else F['region'][3]) for k in range(len(xs)-1)]
            lims=[fitz.Rect(pr) for pr in prs]
        for k,pr in enumerate(prs):
            pe=[e for e in elements(page, pr) if not (e[0]=='t' and re.match(r'^\([a-h]\)', e[2].strip()))]
            pu=union(pe)
            pid=F['id']+'abcdefgh'[k]
            ppath=f"{OUTDIR}/{pid}.png"
            pclip,pw,ph=save(page,pu,ppath,limit=lims[k])
            print('   panel', pid, 'mode', 'patch' if len(pts)==len(F['labels']) else 'split', [round(v,1) for v in pclip])
            records.append(dict(id=pid, file=f"media/paper/{pid}.png", caption=cap, page=F['page'], panel=F['labels'][k], parent=F['id'], clip_pt=pclip, size_px=[pw,ph]))
            rec['panels'].append(pid)
    records.insert(len(records)-len(rec.get('panels',[])), rec)
    print(F['id'], 'fig bbox', [round(v,1) for v in u], 'subcap_y', sub_y, '->', fpath, w, h)

json.dump(records, open(sys.argv[1],'w'), indent=1)
