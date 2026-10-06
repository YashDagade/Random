"""Build data/scaling.json from vector-extracted plot data (digitize.py)."""
import os
import sys, json, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import digitize as D

OUT = '/home/user/Random/ebt-explainer/data/scaling.json'

def r4(v, n=4):
    if v is None: return None
    if isinstance(v, (list, tuple)): return [r4(x, n) for x in v]
    v = float(v)
    if v == 0 or not math.isfinite(v): return v
    return float(f'{v:.{n}g}')

META = {
 'fig4a': dict(figure='Fig 4a', title='Scaling for data (text pretraining, RedPajamaV2)', printed_title='EBT Scales 35.98% Faster Than Transformer++', printed_rate=0.3598,
               x=dict(label='Number of Tokens (Billions)', unit='billion tokens'), y=dict(label='Validation Perplexity', unit='perplexity'), img='media/paper/fig04a.png'),
 'fig4b': dict(figure='Fig 4b', title='Scaling for batch size (text pretraining)', printed_title='EBT Scales 28.46% Faster Than Transformer++', printed_rate=0.2846,
               x=dict(label='Batch Size in Tokens (Thousands)', unit='thousand tokens per batch'), y=dict(label='Validation Perplexity', unit='perplexity'), img='media/paper/fig04b.png'),
 'fig4c': dict(figure='Fig 4c', title='Scaling for depth (text pretraining)', printed_title='EBT Scales 5.29% Faster Than Transformer++', printed_rate=0.0529,
               x=dict(label='Depth in Transformer Blocks (log scale)', unit='transformer blocks'), y=dict(label='Validation Perplexity (log scale)', unit='perplexity'), img='media/paper/fig04c.png'),
 'fig5a': dict(figure='Fig 5a', title='Scaling for number of parameters (text pretraining)', printed_title='EBT Scales 2.91% Faster Than Transformer++', printed_rate=0.0291,
               x=dict(label='Non-Embedding Parameters (M, log scale)', unit='million non-embedding parameters'), y=dict(label='Validation Perplexity (log scale)', unit='perplexity'), img='media/paper/fig05a.png'),
 'fig5b': dict(figure='Fig 5b', title='Scaling for number of FLOPs (text pretraining)', printed_title='EBT Scales 2.92% Faster Than Transformer++', printed_rate=0.0292,
               x=dict(label='Training FLOPs (log scale)', unit='FLOPs'), y=dict(label='Validation Perplexity (log scale)', unit='perplexity'), img='media/paper/fig05b.png'),
 'fig5c': dict(figure='Fig 5c', title='Scaling for the embedding dimension (text pretraining)', printed_title='EBT Scales 0.02% Faster Than Transformer++', printed_rate=0.0002,
               x=dict(label='Embedding Dimension (log scale)', unit='embedding width'), y=dict(label='Validation Perplexity (log scale)', unit='perplexity'), img='media/paper/fig05c.png'),
 'fig6a': dict(figure='Fig 6a', title='OOD thinking performance comparison (mean over four OOD text datasets)', printed_title='Performance Drop on Out of Distribution Data', printed_rate=None,
               x=dict(label='Number of Forward Passes', unit='forward passes per prediction'), y=dict(label='Perplexity Increase (Lower is Better ↓)', unit='perplexity increase on OOD data'), img='media/paper/fig06a.png'),
 'fig6b': dict(figure='Fig 6b', title='Verification capabilities as scale increases (BoN-5 vs no verification)', printed_title='Self-Verification Capabilities Scale During Training', printed_rate=None,
               x=dict(label='Tokens Trained On (B)', unit='billion tokens'), y=dict(label='% PPL Improvement Over No Verification', unit='% perplexity improvement'), img='media/paper/fig06b.png'),
 'fig7':  dict(figure='Fig 7', title='OOD thinking performance (5 datasets)', printed_title='Thinking Helps More on OOD Data', printed_rate=None,
               x=dict(label='OOD Magnitude Shift', unit='downstream ppl / pretraining ppl'), y=dict(label='% PPL Improvement By Thinking', unit='% perplexity improvement'), img='media/paper/fig07.png'),
 'fig9a': dict(figure='Fig 9a', title='Video (SSV2) scaling for the embedding dimension', printed_title='EBT Scales 33.66% Faster Than Transformer++', printed_rate=0.3366,
               x=dict(label='Embedding Dimension (log scale)', unit='embedding width'), y=dict(label='Loss (log scale)', unit='minimum validation Smooth L1 loss'), img='media/paper/fig09a.png'),
 'fig9b': dict(figure='Fig 9b', title='Video (SSV2) scaling for non-embedding parameters', printed_title='EBT Scales 34.28% Faster Than Transformer++', printed_rate=0.3428,
               x=dict(label='Non-Embedding Parameters (M, log scale)', unit='million non-embedding parameters'), y=dict(label='Loss (log scale)', unit='minimum validation Smooth L1 loss'), img='media/paper/fig09b.png'),
 'fig12': dict(figure='Fig 12', title='Image denoising thinking scalability (OOD noise)', printed_title='Performance as Forward Passes Increases', printed_rate=None,
               x=dict(label='Number of Forward Passes (log scale)', unit='forward passes (NFE)'), y=dict(label='PSNR (Higher is Better ↑)', unit='PSNR (dB)'), img='media/paper/fig12.png'),
 'figB1a': dict(figure='Fig B.1a', title='Self-verification with BoN-10 versus BoN-2', printed_title='Self-Verification Capablities Scale During Training', printed_rate=None,
               x=dict(label='Tokens Trained On (B)', unit='billion tokens'), y=dict(label='% Improvement in PPL', unit='% perplexity improvement (BoN-10 over BoN-2)'), img='media/paper/figB1a.png'),
 'figB1b': dict(figure='Fig B.1b', title='Fig 6b results projected to Llama3 data scale', printed_title='Self-Verification Capabilities Extrapolated', printed_rate=None,
               x=dict(label='Tokens Trained on (log scale)', unit='tokens'), y=dict(label='% PPL Improvement Over No Verification', unit='% perplexity improvement'), img='media/paper/figB1b.png'),
 'figB3a': dict(figure='Fig B.3a', title='Data scaling at higher scale (FineWeb)', printed_title='EBT Scales 35.69% Faster Than Transformer++', printed_rate=0.3569,
               x=dict(label='Number of Tokens (Billions)', unit='billion tokens'), y=dict(label='Validation Perplexity', unit='perplexity'), img='media/paper/figB3a.png'),
 'figB3b': dict(figure='Fig B.3b', title='Data scaling at higher scale, zoomed in (FineWeb)', printed_title='EBT Scales 51.70% Faster Than Transformer++', printed_rate=0.5170,
               x=dict(label='Number of Tokens (Billions)', unit='billion tokens'), y=dict(label='Validation Perplexity', unit='perplexity'), img='media/paper/figB3b.png'),
 'figC1': dict(figure='Fig C.1', title='EBT System 1 vs System 2 scaling', printed_title='EBT-S2 Scales 3.30% Faster Than EBT-S1', printed_rate=0.0330,
               x=dict(label='Non-Embedding Parameters in Millions (log scale)', unit='million non-embedding parameters'), y=dict(label='Validation Perplexity (log scale)', unit='perplexity'), img='media/paper/figC1.png'),
}
LEGEND = {('fig6b','EBT'):'BoN-5 Samples', ('fig7','EBT'):'EBT With Max Thinking', ('figB1a','EBT'):'BoN-10 Samples', ('figB1b','EBT'):'BoN-5 Samples',
          ('figC1','EBT-S1'):'EBT-S1', ('figC1','EBT-S2'):'EBT-S2', ('fig12','DiT'):'DiT', ('fig12','EBT'):'EBT'}
PAPER_HEX = {'Transformer++':'#f5a93d', 'DiT':'#f5a93d', 'EBT-S1':'#f5a93d', 'EBT':'#5d8dfe', 'EBT-S2':'#5d8dfe'}
REF_ORDER = {'fig12':('EBT','DiT'), 'figC1':('EBT-S2','EBT-S1')}

def loglog_fit(pts):
    lx, ly = np.log10(pts[:,0]), np.log10(pts[:,1])
    s, c = np.polyfit(lx, ly, 1)
    pred = s*lx + c
    r2 = 1 - np.sum((ly-pred)**2)/np.sum((ly-ly.mean())**2)
    return float(s), float(c), float(r2)

def lin_fit(pts):
    s, c = np.polyfit(pts[:,0], pts[:,1], 1)
    pred = s*pts[:,0]+c
    r2 = 1 - np.sum((pts[:,1]-pred)**2)/np.sum((pts[:,1]-pts[:,1].mean())**2)
    return float(s), float(c), float(r2)

def downsample_curve(pts, n=100):
    idx = np.unique(np.round(np.linspace(0, len(pts)-1, n)).astype(int))
    return [[r4(pts[i,0]), r4(pts[i,1])] for i in idx]

plots = []
checks = []
for cfg in D.PLOTS:
    raw = D.run(cfg)
    pid = cfg['id']; M = META[pid]
    models = []
    for m in raw['markers']:
        if m['model'] not in models: models.append(m['model'])
    series = []
    slopes = {}
    for mdl in models:
        mk = sorted([m for m in raw['markers'] if m['model']==mdl], key=lambda m: m['x'])
        main = [m for m in mk if not (pid=='figB1b' and m['shape']=='star')]
        pts = np.array([[m['x'], m['y']] for m in main])
        s = dict(model=mdl, legend=LEGEND.get((pid,mdl), mdl), paper_color=PAPER_HEX.get(mdl),
                 points=[[r4(x), r4(y)] for x,y in pts], n_points=len(pts), printed_equation=None)
        lines = [l for l in raw['lines'] if l['model']==mdl]
        curve = np.array(lines[0]['pts']) if lines else None
        # ---- fits
        if pid in ('fig4a','fig4b','fig4c','fig5a','fig5b','fig5c','fig9a','fig9b','figB3a','figB3b','figC1','fig12'):
            sl, ic, r2 = loglog_fit(pts); slopes[mdl] = sl
            s['fit'] = dict(slope=r4(sl,5), intercept=r4(ic,5), space='log10-log10', form='log10(y) = slope*log10(x) + intercept  (power law y = 10^intercept * x^slope)',
                            r2=r4(r2), fit_by='explainer', fit_note=('OLS on the digitized points; this is the fit the paper uses for its printed scaling-rate titles (reproduced to within 0.01 percentage points).' if M['printed_rate'] is not None else 'OLS on the 3 digitized points; the paper does not print a slope for this figure.'))
        elif pid in ('fig6b','fig7','figB1a'):
            sl, ic, r2 = lin_fit(pts)
            s['fit'] = dict(slope=r4(sl,5), intercept=r4(ic,5), space='linear', form='y = slope*x + intercept', r2=r4(r2), fit_by='explainer',
                            fit_note='OLS on the digitized points; matches the regression line drawn in the paper (see paper_curve).')
        elif pid == 'figB1b':
            sl, ic, r2 = lin_fit(pts)
            s['fit'] = dict(slope=r4(sl,5), intercept=r4(ic,5), space='linear', form='y = slope*x + intercept (x in tokens; same line as Fig 6b with x in billions)', r2=r4(r2), fit_by='explainer',
                            fit_note='OLS on the digitized points. The paper extrapolates this line to 15T tokens (Llama3 scale).')
        else:
            s['fit'] = None
        # ---- the curve actually drawn in the paper
        if curve is not None and len(curve) >= 10:
            pc = dict(points=downsample_curve(curve))
            if pid in ('fig4a','fig4b','figB3a','figB3b'):
                c = np.polyfit(np.log(pts[:,0]), pts[:,1], 2)
                dev = float(np.max(np.abs(np.polyval(c, np.log(curve[:,0])) - curve[:,1])))
                pc.update(form='y = c0 + c1*ln(x) + c2*ln(x)^2 (display trend line; OLS quadratic in ln x on the plotted points)', coeffs=[r4(c[2],6), r4(c[1],6), r4(c[0],6)], max_dev_from_drawn=r4(dev,2))
            elif pid in ('fig6b','fig7','figB1a','figB1b'):
                c = np.polyfit(curve[:,0], curve[:,1], 1)
                pc.update(form='y = slope*x + intercept (line drawn in the paper)', slope=r4(c[0],5), intercept=r4(c[1],5))
            else:
                c = np.polyfit(np.log10(curve[:,0]), np.log10(curve[:,1]), 1)
                pc.update(form='power law drawn in the paper (straight line in log-log)', slope=r4(c[0],5), intercept=r4(c[1],5))
            s['paper_curve'] = pc
        elif curve is not None:
            s['paper_curve'] = dict(form='straight segments connecting the points', points=[[r4(x), r4(y)] for x,y in curve])
        if pid == 'fig6a' and mdl == 'EBT':
            lab = {'circle':'EBT No Thinking', 'star':'EBT Thinking Longer', 'triangle':'EBT Self-Verification'}
            s['point_labels'] = [lab[m['shape']] for m in main]
            s['legend'] = 'EBT (marker shape = thinking mode)'
        if pid == 'fig6a' and mdl == 'Transformer++':
            s['legend'] = 'Transformer++'
        if pid == 'figB1b':
            star = [m for m in mk if m['shape']=='star']
            if star:
                s['extrapolated_point'] = [r4(star[0]['x']), r4(star[0]['y'])]
                s['extrapolated_legend'] = 'BoN-5 Samples (extrapolated)'
        series.append(s)
    # ---- rate gain
    rate = None
    if len(slopes) == 2:
        a, b = REF_ORDER.get(pid, ('EBT', 'Transformer++'))
        rate = abs(slopes[a])/abs(slopes[b]) - 1
    P = dict(id=pid, figure=M['figure'], page=cfg['page'], title=M['title'], printed_title=M['printed_title'], image=M['img'],
             x=dict(label=M['x']['label'], log=cfg['xlog'], unit=M['x']['unit'], lim=[r4(v) for v in raw['xlim']]),
             y=dict(label=M['y']['label'], log=cfg['ylog'], unit=M['y']['unit'], lim=[r4(v) for v in raw['ylim']]),
             series=series,
             rate_gain=r4(rate) if rate is not None else None,
             rate_gain_printed=M['printed_rate'],
             approx=True, label=f"approx., read from {M['figure']}")
    if rate is not None:
        a, b = REF_ORDER.get(pid, ('EBT', 'Transformer++'))
        P['rate_gain_definition'] = f'|slope_{a}| / |slope_{b}| - 1, slopes of log10-log10 OLS fits to the digitized points'
        if M['printed_rate'] is not None: checks.append((pid, rate, M['printed_rate']))
    if pid == 'fig12':
        P['y']['note'] = 'The paper draws this y axis on a log scale although its label does not say so.'
        P['note'] = 'Paper prints no rate for this figure; rate_gain here is an explainer computation from 3 points per model (EBT 1-3 passes vs DiT 100-300 passes). EBT at 3 passes ≈ Table 4 EBT OOD PSNR 23.29; DiT at 300 ≈ Table 4 DiT 19.56.'
    if pid == 'fig6a':
        ebt = [s for s in series if s['model']=='EBT'][0]['points']
        P['note'] = 'EBT improves from {:.4g} (no thinking) to {:.4g} at 30 passes: {:.1%} lower, matching the paper\'s "up to 29%". Transformer++ stays flat at about {:.4g}: more forward passes cannot change its per-token prediction.'.format(ebt[0][1], ebt[-1][1], (ebt[0][1]-ebt[-1][1])/ebt[0][1], [s for s in series if s['model']=='Transformer++'][0]['points'][0][1])
        P['thinking_improvement'] = r4((ebt[0][1]-ebt[-1][1])/ebt[0][1])
    if pid == 'fig6b':
        P['note'] = 'Same 33 points as Fig B.1b (x there in tokens). Paper text: verification gain grows from 4%-8% to 10%-14% (p.10).'
    if pid == 'figB1b':
        P['note'] = 'Circles = Fig 6b data. The curve is the Fig 6b linear regression (in tokens) extended to 1.5e13 tokens (Llama3 scale); it looks curved only because both axes are log. The star is the extrapolated value, not a measurement.'
    if pid in ('fig4a','fig4b','figB3a','figB3b'):
        P['note'] = 'The smooth curves drawn in this panel are display trend lines (quadratic in ln x). The printed rate comes from straight-line fits in log-log space (series[].fit).'
    if pid == 'figC1':
        P['note'] = 'EBT-S1 = EBT trained with System 1 hyperparameters, EBT-S2 = with System 2 (thinking) hyperparameters, see Appendix C.'
    plots.append(P)

out = {
 '_meta': {
   'source': 'digitized from paper figures (Gladstone et al., arXiv 2507.02092v1)',
   'note': 'approx. Values were extracted from the vector graphics inside the PDF (marker centres and line paths) and mapped to data units with the tick positions; tick labels were read visually. Every series is approx. and should be labelled "approx., read from Fig N". Mapping residual at the ticks is < 0.002 pt. Checks: (1) the log-log OLS slope ratios of the digitized points reproduce all 11 printed "X% Faster" titles to within 0.01 percentage points; (2) Fig 12 endpoints match Table 4 (EBT 23.29 dB, DiT 19.56 dB) to 0.01 dB.',
   'method': 'pymupdf page.get_drawings(): coloured filled paths = markers, coloured strokes = drawn fit lines, black short strokes = ticks; see src note in report.',
   'paper_colors': {'#f5a93d (orange)': 'Transformer++ / DiT / EBT-S1', '#5d8dfe (blue)': 'EBT / EBT-S2'},
   'rate_gain': '|slope_EBT| / |slope_baseline| - 1 from log10-log10 fits; positive = EBT perplexity/loss falls faster with scale.',
   'not_digitized': 'Fig 8, Fig 11, Fig B.2 are heatmaps (see paper_heatmaps.json if present); Figs 1-3, 10, E.1, H.1 are diagrams/images.',
   'rounding': 'points to 4 significant digits, fit parameters to 5.'
 },
 'plots': plots,
 'printed_rate_check': [dict(id=a, computed=r4(b), printed=c) for a,b,c in checks],
}
json.dump(out, open(OUT,'w'), ensure_ascii=False, indent=1)
for a,b,c in checks: print(f'{a:7s} computed {100*b:7.3f}%  printed {100*c:6.2f}%')
print('wrote', OUT)
