import os
import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
ROOT='/home/user/Random/ebt-explainer'
OUT=os.environ.get('EBT_CHECK_DIR', '/tmp/claude-0/-home-user-Random/9843a8a1-7b0a-5ed7-8867-6762f4a2ea11/scratchpad/ebt/digitize_check')
d=json.load(open(ROOT+'/data/scaling.json'))
mk={'EBT No Thinking':'o','EBT Thinking Longer':'*','EBT Self-Verification':'^'}
for p in d['plots']:
    fig,(a0,a1)=plt.subplots(1,2,figsize=(13,4.6),gridspec_kw=dict(width_ratios=[1,1]))
    a0.imshow(Image.open(ROOT+'/'+p['image'])); a0.axis('off'); a0.set_title('original crop: '+p['image'].split('/')[-1], fontsize=9)
    for s in p['series']:
        pts=np.array(s['points']); c=s['paper_color']
        if s.get('point_labels'):
            for (x,y),lab in zip(pts,s['point_labels']): a1.plot(x,y,mk[lab],color=c,ms=8)
        else:
            a1.plot(pts[:,0],pts[:,1],'o',color=c,ms=5,label=s['legend'])
        pc=s.get('paper_curve')
        if pc: 
            q=np.array(pc['points']); a1.plot(q[:,0],q[:,1],'-',color=c,lw=1.2)
        f=s.get('fit')
        if f:
            xs=np.geomspace(pts[:,0].min(),pts[:,0].max(),50) if p['x']['log'] or f['space']=='log10-log10' else np.linspace(pts[:,0].min(),pts[:,0].max(),50)
            if f['space']=='log10-log10': ys=10**(f['intercept']+f['slope']*np.log10(xs))
            else: ys=f['slope']*xs+f['intercept']
            a1.plot(xs,ys,'--',color='k',lw=0.8,alpha=0.6)
        if s.get('extrapolated_point'):
            a1.plot(*s['extrapolated_point'],'*',color=c,ms=10)
    if p['x']['log']: a1.set_xscale('log')
    if p['y']['log']: a1.set_yscale('log')
    a1.set_xlim(p['x']['lim']); a1.set_ylim(p['y']['lim'])
    a1.grid(alpha=0.3, which='both' if p['x']['log'] else 'major')
    a1.set_xlabel(p['x']['label']); a1.set_ylabel(p['y']['label'])
    t=f"digitized {p['id']}: {p['printed_title']}"
    if p['rate_gain'] is not None: t+=f"\nrate_gain computed {100*p['rate_gain']:.2f}%" + (f" | printed {100*p['rate_gain_printed']:.2f}%" if p['rate_gain_printed'] is not None else '')
    a1.set_title(t,fontsize=9); a1.legend(fontsize=7)
    # show linear-axis tick values like the paper
    fig.tight_layout(); fig.savefig(f"{OUT}/{p['id']}_check.png",dpi=90); plt.close(fig)
    print('wrote', p['id'])
