"""Post-process data/scaling.json after build_scaling.py (checker pass).

Run:  python3 -I src/paper_figs/refine_scaling.py   (after build_scaling.py)

What it fixes, and why:
1. Precision. build_scaling.py stored points at 4 significant digits. That is too coarse for
   the small slope differences behind the printed "X% Faster" titles: refitting the stored
   points gave 33.50% for Fig 9a (printed 33.66), 34.14% for 9b (34.28), 51.59% for B.3b
   (51.70), 5.37% for 4c (5.29) and 0.08% for 5c (0.02). The site's scaling lab refits the
   stored points live, so it showed numbers that disagree with the paper. Here y is stored
   to 6 significant digits (x stays at 4), which reproduces all 11 printed rates.
2. Star markers. A matplotlib '*' marker's bounding-box centre sits 0.0955*R above the data
   point (the top tip reaches R, the bottom tips only 0.809*R). build_scaling.py used the
   bbox centre for every marker. For stars this script uses the centroid of the 10 vertices,
   which is the exact centre. Affects Fig 6a (EBT Thinking Longer, 35.92 -> 35.89) and the
   Fig B.1b extrapolated star (1345% -> 1330%, which now matches the Fig 6b line at 15T
   tokens exactly). Circles and '^' triangles are centred on their bounding box already.
"""
import os, sys, json, math, site
sys.path.append(site.getusersitepackages())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pymupdf as fitz
import vecplot as V
import digitize as D

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'scaling.json')
OUT = os.path.normpath(OUT)

def sig(v, n):
    v = float(v)
    if v == 0 or not math.isfinite(v): return v
    return float(f'{v:.{n}g}')

def markers(cfg):
    """Markers of one plot with exact centres (vertex centroid for stars)."""
    page = V.doc()[cfg['page'] - 1]
    fr = V.find_axes(cfg['page'])[cfg['ax']]
    r = V.collect(cfg['page'], fr)
    fx, _, _ = D.axis_map(r['xticks'], cfg['xt'], cfg['xlog'])
    fy, _, _ = D.axis_map(r['yticks'], cfg['yt'], cfg['ylog'])
    out = []
    for d in page.get_drawings():
        it = d['items']; c = V.col(d.get('color')); f = V.col(d.get('fill')); rr = d['rect']
        if d['type'] not in ('fs', 'f') or not (rr.width < 6 and rr.height < 6): continue
        if f == (0, 0, 0) or c in [(0, 0, 0), (0.502, 0.502, 0.502), (0.8, 0.8, 0.8)]: continue
        bc = fitz.Point((rr.x0 + rr.x1) / 2, (rr.y0 + rr.y1) / 2)
        if not fitz.Rect(fr.x0 - 3, fr.y0 - 3, fr.x1 + 3, fr.y1 + 3).contains(bc): continue
        if r['legend'] is not None and r['legend'].contains(bc): continue
        n = len(it)
        shape = {8: 'circle', 3: 'triangle', 10: 'star', 9: 'star'}.get(n, n)
        if shape == 'star':
            vx = np.array([(i[1].x, i[1].y) for i in it if i[0] == 'l'])
            cx, cy = vx.mean(0)
        else:
            cx, cy = bc.x, bc.y
        col = tuple(round(v, 3) for v in (f or c))
        out.append(dict(model=cfg['colors'].get(col, str(col)), shape=shape, x=float(fx(cx)), y=float(fy(cy))))
    return out

def loglog_slope(p):
    p = np.asarray(p, float)
    return float(np.polyfit(np.log10(p[:, 0]), np.log10(p[:, 1]), 1)[0])

S = json.load(open(OUT))
P = {p['id']: p for p in S['plots']}
report = []
for cfg in D.PLOTS:
    pid = cfg['id']; plot = P[pid]
    mk = markers(cfg)
    for s in plot['series']:
        mm = sorted([m for m in mk if m['model'] == s['model']], key=lambda m: m['x'])
        main = [m for m in mm if not (pid == 'figB1b' and m['shape'] == 'star')]
        assert len(main) == len(s['points']) == s['n_points'], (pid, s['model'], len(main), len(s['points']))
        new = [[sig(m['x'], 4), sig(m['y'], 6)] for m in main]
        for (ox, oy), (nx, ny), m in zip(s['points'], new, main):
            assert abs(ox - nx) <= 1e-3 * abs(ox) + 1e-9, (pid, ox, nx)
            tol = 2e-3 * abs(oy) if m['shape'] == 'star' else 6e-4 * abs(oy) + 1e-9
            assert abs(oy - ny) <= tol, (pid, s['model'], oy, ny)
            if abs(sig(ny, 4) - sig(oy, 4)) > 1e-9:
                report.append(f'{pid} {s["model"]} x={nx}: y {sig(oy, 4)} -> {sig(ny, 4)} (centre fix, {m["shape"]})')
        s['points'] = new
        if pid == 'figB1b':
            star = [m for m in mm if m['shape'] == 'star']
            if star:
                old = s.get('extrapolated_point')
                s['extrapolated_point'] = [sig(star[0]['x'], 4), sig(star[0]['y'], 6)]
                report.append(f'figB1b extrapolated star {old} -> {s["extrapolated_point"]}')
    # rate recomputed from the stored points
    if plot.get('rate_gain') is not None:
        sl = {s['model']: loglog_slope(s['points']) for s in plot['series']}
        a, b = {'fig12': ('EBT', 'DiT'), 'figC1': ('EBT-S2', 'EBT-S1')}.get(pid, ('EBT', 'Transformer++'))
        rg = abs(sl[a]) / abs(sl[b]) - 1
        plot['rate_gain_from_stored_points'] = sig(rg, 4)
        plot['rate_gain_source'] = ('explainer computation; the paper prints no rate for this figure'
                                    if plot.get('rate_gain_printed') is None else
                                    'reproduces the rate printed in the figure title')
        if plot.get('rate_gain_printed') is not None:
            assert abs(rg - plot['rate_gain_printed']) < 1e-4, (pid, rg, plot['rate_gain_printed'])
        report.append(f'{pid}: rate from stored points {100*rg:.3f}% (printed {plot.get("rate_gain_printed")})')

# notes that depend on point values
p6 = P['fig6a']
ebt = [s for s in p6['series'] if s['model'] == 'EBT'][0]['points']
p6['thinking_improvement'] = sig((ebt[0][1] - ebt[-1][1]) / ebt[0][1], 4)
pb = P['figB1b']
pb['note'] = ('Circles = Fig 6b data. The curve is the Fig 6b linear regression (in tokens) extended to 1.5e13 tokens '
              '(Llama3 scale); it looks curved only because both axes are log. The star is the extrapolated value '
              '(about 1330%), not a measurement; it lies on that line.')
# which data each thinking plot was measured on (Appendix D.1.2, p.34) and cross-check caveats
DATA_NOTES = {
 'fig6a': ('Mean over the four downstream datasets of Table 3 (GSM8K, SQuAD, BigBench Elementary Math QA, BigBench Dyck '
           'Languages), p.34. The paper does not say exactly how "perplexity increase" is computed, and the plotted values do '
           'not follow from Table 3 by a simple mean, so show them as plotted.'),
 'fig6b': 'BigBench Dyck Languages only (p.34: "Figure 6b was solely from the Dyck Languages benchmark").',
 'fig7': ('Five points: the RedPajamaV2 pretraining data (shift 1.0) and the four downstream datasets of Table 3 (p.34). '
          'The x values are not the Table 3 perplexity ratios of either model, so use them as plotted.'),
 'figB1a': 'RedPajamaV2 validation set (p.34).',
 'figB1b': 'Same Dyck Languages points as Fig 6b, x in tokens.',
 'figB3a': 'FineWeb, small models, batch 256, context 1024, 500,000 steps (p.28).',
 'figB3b': 'FineWeb, same runs as Fig B.3a, only the points from about 51B tokens on (p.28-29).',
}
for pid, txt in DATA_NOTES.items():
    P[pid]['data_note'] = txt

S['_meta']['rounding'] = ('points: x to 4 significant digits, y to 6 (y needs the extra digits so that refitting the stored '
                          'points reproduces the printed rates; with 4 digits Fig 9a refits to 33.50% instead of 33.66%); '
                          'fit parameters to 5; paper_curve points to 4.')
S['_meta']['method'] = ('pymupdf page.get_drawings(): coloured filled paths = markers, coloured strokes = drawn fit lines, '
                        'black short strokes = ticks (src/paper_figs/vecplot.py, digitize.py, build_scaling.py), then '
                        'src/paper_figs/refine_scaling.py. Independent x-axis checks: Fig 5b Transformer++ FLOPs equal '
                        '6*N*steps*batch*256 from App. D.1.1 (3.189e16 for xxs), EBT FLOPs are 6.67x that (App. D.5), and '
                        'the token axes of Figs 4a, 6b, B.3a are whole multiples of steps*batch*context.')
S['_meta']['marker_centres'] = ('Circles and triangles: bounding-box centre of the marker path. Stars: centroid of the 10 '
                                'star vertices (a star\'s bounding-box centre sits 0.0955 x radius above the data point).')
S['_meta']['checked_by'] = ('Independent checker pass: every plot overlaid on its crop (points, ticks and tick labels line '
                            'up), log/linear axis flags tested against tick spacing, legend colours checked, rates refitted '
                            'from the stored points (see plots[].rate_gain_from_stored_points). Post-processed by '
                            'src/paper_figs/refine_scaling.py.')
for c in S['printed_rate_check']:
    c['from_stored_points'] = P[c['id']]['rate_gain_from_stored_points']
json.dump(S, open(OUT, 'w'), ensure_ascii=False, indent=1)
print('\n'.join(report))
print('wrote', OUT)
