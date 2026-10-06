import os
import json, io, sys
import numpy as np, pymupdf as fitz
from PIL import Image
PDF=os.environ.get('EBT_PDF', '/tmp/claude-0/-home-user-Random/9843a8a1-7b0a-5ed7-8867-6762f4a2ea11/scratchpad/ebt/paper/ebt.pdf')
OUT='/home/user/Random/ebt-explainer/data/paper_heatmaps.json'
doc=fitz.open(PDF)
def img(xref):
    d=doc.extract_image(xref)
    return np.asarray(Image.open(io.BytesIO(d['image'])).convert('RGB')).astype(float)[::-1]  # flip: PDF places these images y-flipped
SPECS=[
 dict(id='fig8a', figure='Fig 8a', page=12, heat=2274, cbar=1805, image='media/paper/fig08a.png', title='Token Energies Across Thinking Steps',
      rows=['The','quick','brown','fox','jumps','over','the','lazy','dog','.'], row_axis='Token'),
 dict(id='fig8b', figure='Fig 8b', page=12, heat=2275, cbar=1808, image='media/paper/fig08b.png', title='Token Energies Across Thinking Steps',
      rows=['System','2','Thinking','is','a','challenging','but','interesting','research','problem','.'], row_axis='Token'),
 dict(id='fig11', figure='Fig 11', page=14, heat=2276, cbar=1856, image='media/paper/fig11.png', title='Frame Energies Across Thinking Steps',
      rows=None, row_axis='Frame'),
 dict(id='figB2_seq1', figure='Fig B.2 (Sequence 1)', page=29, heat=2277, cbar=1911, image='media/paper/figB2.png', title='Energy Comparison for Different Sequences: Sequence 1 (likely seen in training)',
      rows=['The','quick','brown','fox','jumps','over','the','lazy','dog','.'], row_axis='Token'),
 dict(id='figB2_seq2', figure='Fig B.2 (Sequence 2)', page=29, heat=2278, cbar=1911, image='media/paper/figB2.png', title='Energy Comparison for Different Sequences: Sequence 2 (random, OOD)',
      rows=['NO','$','#',')','h','v','-','3','h','-'], row_axis='Token'),
]
def boundaries(a, axis):
    # mean abs difference between consecutive lines along axis
    d = np.abs(np.diff(a, axis=axis)).sum(axis=2).mean(axis=1-axis)
    thr = d.max()*0.15
    idx = np.where(d > max(thr, 1.0))[0]
    # merge adjacent indices
    out=[]
    for i in idx:
        if not out or i-out[-1][-1] > 3: out.append([i])
        else: out[-1].append(i)
    return [int(np.mean(g))+1 for g in out]
res=[]
for S in SPECS:
    H=img(S['heat']); C=img(S['cbar'])
    cb = C[:, C.shape[1]//2, :]           # top row = value 1.0, bottom = 0.0 (after flip)
    vals = np.linspace(1.0, 0.0, len(cb))
    nc = 12; nr = len(S['rows']) if S['rows'] else 16   # regular grid: iterations 0..11; Fig 11 has 16 frame thumbnails
    xs=np.round(np.linspace(0,H.shape[1],nc+1)).astype(int); ys=np.round(np.linspace(0,H.shape[0],nr+1)).astype(int)
    grid=np.zeros((nr,nc)); spread=np.zeros((nr,nc))
    for i in range(nr):
        for j in range(nc):
            y0,y1=ys[i],ys[i+1]; x0,x1=xs[j],xs[j+1]
            patch=H[y0+(y1-y0)//4:y1-(y1-y0)//4, x0+(x1-x0)//4:x1-(x1-x0)//4].reshape(-1,3)
            c=np.median(patch,axis=0); spread_in=np.abs(patch-c).max()
            dist=np.sqrt(((cb-c)**2).sum(1)); k=int(np.argmin(dist))
            grid[i,j]=vals[k]; spread[i,j]=max(dist[k], spread_in)
    print(S['id'], 'grid', nr, 'x', nc, 'max colour mismatch / in-cell spread', round(spread.max(),2), 'row px', np.diff(ys).min(), np.diff(ys).max(), 'col px', np.diff(xs).min(), np.diff(xs).max())
    if S['rows'] is not None: assert nr==len(S['rows']), (S['id'], nr)
    rec=dict(id=S['id'], figure=S['figure'], page=S['page'], title=S['title'], image=S['image'], row_axis=S['row_axis'],
             rows=S['rows'] if S['rows'] else [f'frame {i+1}' for i in range(nr)],
             iterations=list(range(nc)), energy=[[round(float(v),3) for v in r] for r in grid],
             approx=True, label=f"approx., read from {S['figure']}")
    res.append(rec)
out={'_meta':{'source':'digitized from paper heatmaps (Gladstone et al., arXiv 2507.02092v1): Fig 8, Fig 11, Fig B.2',
              'note':'approx. Each cell is the colour of the embedded raster heatmap mapped back through the figure\'s own colourbar (0 = yellow, 1 = dark purple). Values are the paper\'s "Normalized Energy" (normalisation scheme not stated in the paper). energy[row][iteration]; iteration 0 = initial random prediction. Colour quantisation gives about ±0.005 error.'},
     'heatmaps':res}
json.dump(out, open(OUT,'w'), indent=1)
print('wrote', OUT)
