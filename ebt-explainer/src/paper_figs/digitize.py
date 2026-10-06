import os
import sys, json, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vecplot as V
import numpy as np

ORANGE=(0.961,0.663,0.239); BLUE=(0.365,0.553,0.996)
# axis index = order from V.find_axes(page)
PLOTS = [
 dict(id='fig4a', page=9, ax=0, xlog=False, ylog=False, xt={0:2,1:4,2:6}, yt={0:70,1:60,2:50,3:40},
      colors={ORANGE:'Transformer++', BLUE:'EBT'}),
 dict(id='fig4b', page=9, ax=1, xlog=False, ylog=False, xt={0:10,1:20,2:30,3:40,4:50,5:60}, yt={0:55,1:50,2:45,3:40,4:35},
      colors={ORANGE:'Transformer++', BLUE:'EBT'}),
 dict(id='fig4c', page=9, ax=2, xlog=True, ylog=True, xt={0:4,1:5,2:6,3:7,4:8,5:9,6:16}, yt={0:60,1:50},
      colors={ORANGE:'Transformer++', BLUE:'EBT'}),
 dict(id='fig5a', page=10, ax=0, xlog=True, ylog=True, xt={4:10,13:100,0:6}, yt={0:60,1:50,2:40,3:30,4:20},
      colors={ORANGE:'Transformer++', BLUE:'EBT'}),
 dict(id='fig5b', page=10, ax=1, xlog=True, ylog=True, xt={7:1e17,16:1e18,25:1e19,34:1e20}, yt={0:60,1:50,2:40,3:30,4:20},
      colors={ORANGE:'Transformer++', BLUE:'EBT'}),
 dict(id='fig5c', page=10, ax=2, xlog=True, ylog=True, xt={0:200,1:300}, yt={0:60,1:50},
      colors={ORANGE:'Transformer++', BLUE:'EBT'}),
 dict(id='fig6a', page=11, ax=0, xlog=False, ylog=False, xt={0:2,1:3,2:6,3:15,4:30}, yt={0:45.0,1:42.5,2:40.0,3:37.5,4:35.0,5:32.5},
      colors={ORANGE:'Transformer++', BLUE:'EBT'}),
 dict(id='fig6b', page=11, ax=1, xlog=False, ylog=False, xt={0:0,1:5,2:10,3:15,4:20,5:25,6:30}, yt={0:14,1:12,2:10,3:8,4:6,5:4},
      colors={BLUE:'EBT'}),
 dict(id='fig7', page=11, ax=2, xlog=False, ylog=False, xt={0:1.0,1:1.5,2:2.0,3:2.5,4:3.0,5:3.5,6:4.0,7:4.5}, yt={0:22,1:20,2:18,3:16,4:14,5:12},
      colors={BLUE:'EBT'}),
 dict(id='fig9a', page=13, ax=0, xlog=True, ylog=True, xt={0:400,1:500,2:600,3:700,4:800,5:900,6:1536,7:2000}, yt={0:1.9,1:1.8,2:1.7,3:1.6,4:1.5},
      colors={ORANGE:'Transformer++', BLUE:'EBT'}),
 dict(id='fig9b', page=13, ax=1, xlog=True, ylog=True, xt={5:10,14:100}, yt={0:1.9,1:1.8,2:1.7,3:1.6,4:1.5},
      colors={ORANGE:'Transformer++', BLUE:'EBT'}),
 dict(id='fig12', page=15, ax=0, xlog=True, ylog=True, xt={2:1,11:10,20:100}, yt={0:22,1:20,2:18,3:16,4:14},
      colors={ORANGE:'DiT', BLUE:'EBT'}),
 dict(id='figB1a', page=28, ax=0, xlog=False, ylog=False, xt={0:5,1:10,2:15,3:20,4:25,5:30}, yt={0:2.5,1:2.0,2:1.5,3:1.0,4:0.5,5:0.0},
      colors={BLUE:'EBT'}),
 dict(id='figB1b', page=28, ax=1, xlog=True, ylog=True, xt={3:1e9,12:1e10,21:1e11,30:1e12,39:1e13}, yt={0:1000,9:100,18:10},
      colors={BLUE:'EBT'}),
 dict(id='figB3a', page=29, ax=3, xlog=False, ylog=False, xt={0:0,1:20,2:40,3:60,4:80,5:100,6:120}, yt={0:45,1:40,2:35,3:30,4:25},
      colors={ORANGE:'Transformer++', BLUE:'EBT'}),
 dict(id='figB3b', page=29, ax=4, xlog=False, ylog=False, xt={0:60,1:80,2:100,3:120}, yt={0:27,1:26,2:25},
      colors={ORANGE:'Transformer++', BLUE:'EBT'}),
 dict(id='figC1', page=31, ax=0, xlog=True, ylog=True, xt={2:1.0,11:10.0,0:0.8}, yt={0:80,1:70,2:60,3:50},
      colors={ORANGE:'EBT-S1', BLUE:'EBT-S2'}),
]

def axis_map(ticks, valmap, log):
    pos = np.array([ticks[i][0] for i in valmap]); val = np.array(list(valmap.values()), float)
    f = np.log10(val) if log else val
    A = np.vstack([f, np.ones_like(f)]).T
    (b, a), *_ = np.linalg.lstsq(A, pos, rcond=None)
    resid = np.max(np.abs(A@[b,a]-pos)) if len(pos)>2 else 0.0
    def inv(p):
        u = (np.asarray(p)-a)/b
        return 10**u if log else u
    return inv, resid, b

def run(cfg):
    fr = V.find_axes(cfg['page'])[cfg['ax']]
    r = V.collect(cfg['page'], fr)
    fx, rx, bx = axis_map(r['xticks'], cfg['xt'], cfg['xlog'])
    fy, ry, by = axis_map(r['yticks'], cfg['yt'], cfg['ylog'])
    out = dict(id=cfg['id'], page=cfg['page'], frame=[round(v,2) for v in fr], tick_fit_resid_pt=[float(rx), float(ry)],
               xlim=[float(fx(fr.x0)), float(fx(fr.x1))], ylim=[float(fy(fr.y1)), float(fy(fr.y0))], markers=[], lines=[])
    for m in r['markers']:
        c = tuple(round(v,3) for v in m['color'])
        out['markers'].append(dict(model=cfg['colors'].get(c, str(c)), x=float(fx(m['x'])), y=float(fy(m['y'])), shape={8:'circle',3:'triangle',10:'star',9:'star'}.get(m['n'], m['n']), px=[m['x'], m['y']]))
    for l in r['lines']:
        pts = np.array(l['pts'])
        out['lines'].append(dict(model=cfg['colors'].get(tuple(round(v,3) for v in l['color']), str(l['color'])), pts=[[float(fx(x)), float(fy(y))] for x,y in pts]))
    return out

if __name__=='__main__':
    res = [run(c) for c in PLOTS]
    json.dump(dict(plots=res, cfg=[{k:(v if k!='colors' else {str(kk):vv for kk,vv in v.items()}) for k,v in c.items() if k not in ('xt','yt')} for c in PLOTS]), open(sys.argv[1],'w'))
    for o in res:
        from collections import Counter
        print(o['id'], 'resid', [round(v,3) for v in o['tick_fit_resid_pt']], 'xlim', [float('%.4g'%v) for v in o['xlim']], 'ylim', [float('%.4g'%v) for v in o['ylim']],
              dict(Counter((m['model'],m['shape']) for m in o['markers'])), [(l['model'],len(l['pts'])) for l in o['lines']])
