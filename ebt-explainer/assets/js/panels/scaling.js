/* Panel 19: scaling laws (learning scalability).
   Data: lib.data('scaling') = data/scaling.json, points digitized from the paper's vector figures (approx.).
   Everything in the figure is computed live from those points: OLS fits in log10-log10 space, rate gains,
   drop-one ranges, crossing points, extrapolations and the irreducible-floor sensitivity analysis. */
(function () {
  'use strict';
  const L10 = Math.log10, P10 = (v) => Math.pow(10, v);
  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
  const MINUS = '−';
  const SUP = { '-': '⁻', '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹' };
  const sup = (s) => String(s).split('').map(c => SUP[c] || c).join('');
  const neg = (s) => String(s).replace(/-/g, MINUS);
  function sci(v, d = 1) {
    if (v == null || !isFinite(v)) return '–';
    if (v === 0) return '0';
    let e = Math.floor(L10(Math.abs(v)));
    let ms = (v / P10(e)).toFixed(d);
    if (Math.abs(parseFloat(ms)) >= 10) { e += 1; ms = (v / P10(e)).toFixed(d); }
    ms = ms.replace(/\.0+$/, '');
    return (ms === '1' ? '' : neg(ms) + '×') + '10' + sup(e);
  }
  function num(v, sig = 3) {
    if (v == null || !isFinite(v)) return '–';
    const a = Math.abs(v);
    if (a !== 0 && (a >= 1e6 || a < 1e-3)) return sci(v, 1);
    if (a >= 1000) return neg(Math.round(v).toLocaleString('en-US'));
    return neg(String(+v.toPrecision(sig)));
  }
  const tickFmt = (v) => (Math.abs(v) >= 1e4) ? sci(v, 0) : num(v, 3);
  const pct = (g, d = 2) => (g >= 0 ? '+' : MINUS) + Math.abs(g * 100).toFixed(d) + '%';
  const KIND = {
    tokB: v => v >= 1000 ? num(v / 1000) + 'T tokens' : num(v) + 'B tokens',
    tokK: v => v >= 1000 ? num(v / 1000) + 'M tokens/batch' : num(v) + 'K tokens/batch',
    depth: v => num(v) + ' blocks',
    width: v => 'width ' + num(v),
    paramM: v => v >= 1e6 ? num(v / 1e6) + 'T params' : v >= 1000 ? num(v / 1000) + 'B params' : num(v) + 'M params',
    flops: v => sci(v, 1) + ' FLOPs',
  };

  // OLS of log10(y - E) on log10(x)
  function ols(pts, E = 0) {
    const q = pts.filter(p => p[0] > 0 && p[1] - E > 0); const n = q.length; if (n < 2) return null;
    let sx = 0, sy = 0, sxx = 0, sxy = 0, syy = 0;
    q.forEach(([x, y]) => { const a = L10(x), c = L10(y - E); sx += a; sy += c; sxx += a * a; sxy += a * c; syy += c * c; });
    const mx = sx / n, my = sy / n, vxx = sxx / n - mx * mx, vxy = sxy / n - mx * my, vyy = syy / n - my * my;
    if (!(vxx > 0)) return null;
    const m = vxy / vxx;
    return { m, b: my - m * mx, r2: vyy > 0 ? (vxy * vxy) / (vxx * vyy) : 1, n, E };
  }
  const fitY = (f, x) => f.E + P10(f.b + f.m * L10(x));
  const gainOf = (fA, fB) => (fA && fB) ? Math.abs(fB.m) / Math.abs(fA.m) - 1 : NaN;

  function linTicks(a, b, n) {
    const span = b - a; if (!(span > 0)) return [a];
    const s0 = span / n, mag = P10(Math.floor(L10(s0))), r = s0 / mag;
    const step = (r < 1.5 ? 1 : r < 3 ? 2 : r < 7 ? 5 : 10) * mag; const out = [];
    for (let v = Math.ceil(a / step) * step; v <= b + 1e-9 * span; v += step) out.push(+v.toPrecision(12));
    return out;
  }
  function logTicks(la, lb, maxN) {
    const d0 = Math.ceil(la - 1e-9), d1 = Math.floor(lb + 1e-9), dec = [];
    for (let d = d0; d <= d1; d++) dec.push(d);
    if (dec.length >= 3) { const st = Math.ceil(dec.length / (maxN || 6)); return dec.filter(d => (d - d0) % st === 0).map(P10); }
    for (const ms of [[1, 2, 5], [1, 2, 3, 4, 5, 6, 7, 8, 9]]) {
      const t = [];
      for (let d = Math.floor(la); d <= Math.ceil(lb); d++) ms.forEach(m => { const v = m * P10(d), lv = L10(v); if (lv >= la - 1e-9 && lv <= lb + 1e-9) t.push(v); });
      if (t.length >= 3 && t.length <= (maxN || 7)) return t;
    }
    return linTicks(P10(la), P10(lb), 4).filter(v => v > 0);
  }

  // responsive canvas whose logical size equals its CSS size (labels stay legible on phones)
  function rcanvas(lib, parent, o) {
    const box = lib.h('div', { class: 'canvas-box' }); parent.appendChild(box);
    const c = lib.h('canvas', { role: 'img', 'aria-label': o.label || '' }); box.appendChild(c);
    const g = c.getContext('2d'); const R = { canvas: c, ctx: g, w: 0, h: 0, dpr: 1, box };
    R.fit = function () {
      const w = Math.max(260, Math.round(box.clientWidth || 600)), hh = Math.round(o.aspect(w));
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      if (w === R.w && hh === R.h && dpr === R.dpr) return false;
      R.w = w; R.h = hh; R.dpr = dpr; c.width = Math.round(w * dpr); c.height = Math.round(hh * dpr); c.style.height = hh + 'px';
      g.setTransform(dpr, 0, 0, dpr, 0, 0); return true;
    };
    R.fit();
    if (window.ResizeObserver) new ResizeObserver(() => { if (R.fit() && o.draw) o.draw(); }).observe(box);
    R.local = (ev) => { const r = c.getBoundingClientRect(); const p = ev.touches && ev.touches.length ? ev.touches[0] : ev; return [(p.clientX - r.left) / r.width * R.w, (p.clientY - r.top) / r.height * R.h]; };
    R.clear = () => { g.save(); g.setTransform(R.dpr, 0, 0, R.dpr, 0, 0); g.fillStyle = '#ffffff'; g.fillRect(0, 0, R.w, R.h); g.restore(); };
    return R;
  }

  EBT.panel({
    id: 'scaling',
    nav: 'Scaling laws',
    title: 'Scaling laws: who improves faster, and by how much',
    lede: 'The abstract reports an “up to 35% higher scaling rate” than Transformer++. That number is a ratio of two slopes, and it comes from one axis. Refit every axis from the paper’s own plots, see what a steeper slope does and does not buy, and test how robust each rate is.',
    text: `
      <p>A <b>scaling law</b> describes how a model's held-out loss $L$ falls as it gets more of a resource $R$: training tokens, batch size, depth, parameters, FLOPs. Empirically the curve is close to a power law,</p>
      <div class="eq">$$\\begin{gathered}L(R)\\approx A\\,R^{-\\beta}\\\\ \\log_{10}L=\\log_{10}A-\\beta\\,\\log_{10}R\\end{gathered}$$<span class="why">A sets the height, β the speed. On log–log axes this is a straight line with slope −β.</span></div>
      <p>Every 10× more $R$ multiplies $L$ by $10^{-\\beta}$, every doubling by $2^{-\\beta}$. Here $L$ is validation perplexity (text) or the minimum validation Smooth-L1 loss (video).</p>
      <p>The paper compares EBT with the Transformer++ recipe (Llama 2 style, p.8) on six text axes (Figs 4 and 5), two video axes (Fig 9) and a larger FineWeb run (Fig B.3). It never defines its "X% faster" titles. Fitting straight lines to the digitized points in log–log space reproduces all eleven printed values to 0.01 points as</p>
      <div class="eq">$$\\text{rate gain}=\\frac{|\\beta_{\\text{EBT}}|}{|\\beta_{\\text{T++}}|}-1 .$$<span class="why">derived: our refit, matches every printed title</span></div>`,
    steps: [
      { label: 'Read the plot as printed', html: '<p>Fig 4a (p.9): xxs models (6.18M non-embedding parameters), batch 128, only the number of training tokens changes. EBT (blue) starts worse, ≈74 vs ≈64 perplexity at 0.5B tokens, crosses near 3B and ends slightly better, ≈38.8 vs ≈40.0 at 6.9B. The thin lines are the paper\'s drawn trend curves. Hover a point to read it.</p>' },
      { label: 'Take logs: power laws become lines', html: '<p>On log–log axes both series are close to straight, so an ordinary least-squares fit of $\\log L$ on $\\log R$ summarizes each by a slope: $\\beta_{\\text{T++}}\\approx0.166$, $\\beta_{\\text{EBT}}\\approx0.226$ for data. Click any point to drop that run from both fits; the readout refits live and the drop-one range shows how much a single run moves the rate.</p>' },
      { label: 'The rate is a ratio of slopes', html: '<p>$0.226/0.166-1\\approx36\\%$, the 35.98% in the title. Concretely, each doubling of data multiplies EBT perplexity by $2^{-0.226}\\approx0.855$ and Transformer++ perplexity by $2^{-0.166}\\approx0.891$. That is all "scales 36% faster" means: a steeper line, not a lower one.</p>' },
      { label: 'Slope is not height', html: '<p>Parameters (Fig 5a, xxs to large): the gain is 2.91%, and EBT has the higher perplexity at every size from 6M to 396M. With slopes this close, the fitted lines meet only far beyond the largest model. Depth, width and both video axes look the same way: EBT is worse or level at every measured point. Only data and batch size (and the FineWeb run) actually cross over. The paper\'s own word for parameters and FLOPs is "slightly out-scale" (p.10).</p>' },
      { label: 'Equal compute: the FLOPs axis', html: '<p>Fig 5b plots the same runs against training FLOPs. Each EBT run sits ≈6.67× to the right of its Transformer++ twin (arrows), because one two-step EBT training step costs ≈6.66× a Transformer++ step: forward, backward for $\\nabla_{\\hat y}E$, a Hessian-vector product, on a sequence of doubled length (p.35–36). At every FLOP budget that was measured, Transformer++ has the lower perplexity. The 2.92% is a statement about slope only: extended, these two lines would not meet until about $10^{52}$ FLOPs [derived].</p>' },
      { label: 'Extrapolate, with care', html: '<p>Drag <b>extend the fits</b>. Beyond the hatched line there is no data. At the measured slopes, the parameter-scaling lines would cross at ≈260B non-embedding parameters, ≈650× the largest model trained. The paper\'s "at the scale of modern foundation models ... we expect the pretraining performance of EBTs to be significantly better" (p.9) is this move. It assumes the slope difference, estimated from 4 to 5 runs with one seed each and no error bars, holds over three orders of magnitude.</p>' },
      { label: 'How robust is the rate?', html: '<p>Real loss curves flatten toward an irreducible floor, $L=E_0+A R^{-\\beta}$ (the form used by Hoffmann et al. 2022; not used in the paper). Set an assumed floor and every fit uses $\\log(L-E_0)$ instead. The data and batch gains barely move. The ≈3% gains on parameters and FLOPs fall to zero once $E_0$ reaches about half the lowest measured perplexity. The list on the right refits all eleven axes under the same assumption.</p>' },
    ],
    after: `
      <h3>Why a model that starts behind can look steeper</h3>
      <p>With a floor, the local log–log slope is not $-\\beta$ but</p>
      <div class="eq">$$\\frac{d\\log L}{d\\log R}=-\\beta\\,\\frac{A R^{-\\beta}}{E_0+A R^{-\\beta}} ,$$<span class="why">our analysis, not in the paper</span></div>
      <p>The fraction is the reducible share of the loss. Two models with the same exponent $\\beta$ and the same floor, one of them worse (larger $A$), have different fitted slopes: the worse one is further above the floor, so its line is steeper. EBT starts behind on most axes, so part of a small slope advantage can come from this effect rather than a better exponent. The big data and batch gains survive any reasonable floor; the 3% ones do not. Nobody knows the true floor, and the paper fits pure power laws (Hoffmann et al. 2022 is our reference for the floor form, not the paper's).</p>
      <h3>The scorecard</h3>
      <p>Data 35.98% and batch 28.46%: steeper, and EBT ends below Transformer++ inside the measured range. Depth 5.29%, parameters 2.91%, FLOPs 2.92%: steeper by a little, worse at every measured point. Width 0.02%: the same line. Video width 33.66% and parameters 34.28% (Fig 9): much steeper, higher loss at every size up to 708M. FineWeb (Fig B.3): 35.69% over 4–130B tokens, 51.70% if you fit only from 51B, crossing near 90B. "Up to 35%" is the data axis.</p>
      <p class="note">Points are digitized from the vector graphics of each figure (approx., read from Figs 4, 5, 9, B.3, C.1); the printed percentages in the figure titles are exact. Refits, drop-one ranges, crossings, extrapolations and the floor analysis are our computations on those points. Text batch scaling is 28.46%, under the "over 30%" of the discussion (p.14).</p>`,
    source: [{ kind: 'paper', note: 'Figs 4, 5, 9, B.3, C.1, digitized (approx.); printed rates exact' }, { kind: 'ext', note: 'refits, crossings, extrapolation and floor analysis are ours' }],
    figure(stage, ctx) {
      const { lib } = ctx, h = lib.h, C = lib.C;
      const D = lib.data('scaling');
      if (!D || !D.plots) { stage.appendChild(h('p', { class: 'callout warn' }, 'Scaling data missing (data/scaling.json).')); return {}; }
      const P = {}; D.plots.forEach(p => { P[p.id] = p; });
      const AX = [
        { key: 'data', id: 'fig4a', group: 'Text · RedPajamaV2', name: 'Data', kind: 'tokB', res: 'data',
          note: 'Fig 4a. xxs models (6.18M non-embedding), batch 128, context 256; only the number of training tokens changes (p.34).' },
        { key: 'batch', id: 'fig4b', group: 'Text · RedPajamaV2', name: 'Batch', kind: 'tokK', res: 'batch size',
          note: 'Fig 4b. xxs models, 105k steps each; only the batch size changes (p.34). With a fixed step count a bigger batch also means more tokens [derived].' },
        { key: 'depth', id: 'fig4c', group: 'Text · RedPajamaV2', name: 'Depth', kind: 'depth', res: 'depth',
          note: 'Fig 4c. Only the number of Transformer blocks changes (4, 8, 12, 16). EBT is slightly above Transformer++ at every depth.' },
        { key: 'params', id: 'fig5a', group: 'Text · RedPajamaV2', name: 'Params', kind: 'paramM', res: 'parameters',
          note: 'Fig 5a. xxs to large (6.18M to 396M non-embedding), 105k steps, batch 32 to 256 (p.34).' },
        { key: 'flops', id: 'fig5b', group: 'Text · RedPajamaV2', name: 'FLOPs', kind: 'flops', res: 'training FLOPs',
          note: 'Fig 5b. The Fig 5a runs plotted against training FLOPs: 6N per token for Transformer++, ≈6.66× that for two-step EBTs (p.35–36).' },
        { key: 'width', id: 'fig5c', group: 'Text · RedPajamaV2', name: 'Width', kind: 'width', res: 'width',
          note: 'Fig 5c. Only the embedding dimension changes (≈180 to 360; these widths are not in Table D.1). The two fits are practically the same line.' },
        { key: 'vwidth', id: 'fig9a', group: 'Video · Something-Something V2', name: 'Width', kind: 'width', res: 'width',
          note: 'Fig 9a. Next-frame prediction on SD-XL VAE latents; y is the minimum validation Smooth-L1 loss (p.12). Batch 256.' },
        { key: 'vparams', id: 'fig9b', group: 'Video · Something-Something V2', name: 'Params', kind: 'paramM', res: 'parameters',
          note: 'Fig 9b. xxs to xl (6.18M to 708M non-embedding), batch 256 (p.34). EBT loss is higher at every size.' },
        { key: 'fw', id: 'figB3a', group: 'Larger run · FineWeb (App. B)', name: 'Data', kind: 'tokB', res: 'data',
          note: 'Fig B.3a. Small models, batch 256, context 1024, 500k steps (p.28). EBT crosses below Transformer++ near 90B tokens.' },
        { key: 'fwz', id: 'figB3b', group: 'Larger run · FineWeb (App. B)', name: '≥51B', kind: 'tokB', res: 'data',
          note: 'Fig B.3b. The same FineWeb runs, fitted only from ≈51B tokens on. Same models, narrower window, a different rate (51.70% vs 35.69%).' },
        { key: 's2', id: 'figC1', group: 'Recipe · EBT-S2 vs EBT-S1 (App. C)', name: 'S2 vs S1', kind: 'paramM', res: 'parameters',
          note: 'Fig C.1. Both are EBTs: S1 tuned for stable pretraining, S2 for thinking (p.30). Tiny models, 0.9M to 12.4M.' },
      ].filter(a => P[a.id] && P[a.id].series && P[a.id].series.length === 2);
      AX.forEach(a => { a.p = P[a.id]; });
      const byKey = {}; AX.forEach(a => { byKey[a.key] = a; });
      const names = (ax) => ax.key === 's2' ? ['EBT-S1', 'EBT-S2'] : ['Transformer++', 'EBT'];
      const shortN = (ax) => ax.key === 's2' ? ['S1', 'S2'] : ['T++', 'EBT'];
      const isLoss = (ax) => /loss/i.test(ax.p.y.label);
      const cleanLab = (s) => s.replace(/,\s*log scale/i, '').replace(/\s*\(log scale\)/i, '');
      const yMin = (ax) => Math.min(...ax.p.series[0].points.map(q => q[1]), ...ax.p.series[1].points.map(q => q[1]));

      const S = { axis: 'data', tx: 0, ty: 0, ttx: 0, tty: 0, lines: 'curves', slopes: false, pairs: false, ext: 0, floor: 0, drop: {}, hover: null };
      const dropOf = (k) => S.drop[k] || (S.drop[k] = new Set());
      const floorOf = (ax) => S.floor * yMin(ax);

      function analyze(ax) {
        const A = ax.p.series[0], B = ax.p.series[1], drop = dropOf(ax.key), E = floorOf(ax);
        const n = Math.min(A.points.length, B.points.length);
        const act = (i) => !drop.has(i);
        const aP = A.points.filter((_, i) => act(i)), bP = B.points.filter((_, i) => act(i));
        const fA = ols(aP, E), fB = ols(bP, E), gain = gainOf(fA, fB);
        const idx = []; for (let i = 0; i < n; i++) if (act(i)) idx.push(i);
        let jk = null;
        if (idx.length >= 4) {
          let lo = Infinity, hi = -Infinity;
          idx.forEach(j => { const g = gainOf(ols(A.points.filter((_, i) => act(i) && i !== j), E), ols(B.points.filter((_, i) => act(i) && i !== j), E)); lo = Math.min(lo, g); hi = Math.max(hi, g); });
          jk = [lo, hi];
        }
        const uc = (fA && fB && Math.abs(fA.m - fB.m) > 1e-12) ? (fB.b - fA.b) / (fA.m - fB.m) : NaN;
        // values at the largest run both models reached
        const lastA = A.points[A.points.length - 1], lastB = B.points[B.points.length - 1];
        let lead;
        if (Math.abs(lastA[0] - lastB[0]) / lastA[0] < 0.02) lead = { x: lastA[0], a: lastA[1], b: lastB[1], measured: true };
        else { const xc = Math.min(lastA[0], lastB[0]); const a = lastA[0] === xc ? lastA[1] : fitY(fA, xc), b = lastB[0] === xc ? lastB[1] : fitY(fB, xc); lead = { x: xc, a, b, measured: false }; }
        const xsAll = A.points.concat(B.points).map(q => q[0]);
        return { ax, A, B, n, aP, bP, fA, fB, gain, jk, uc, lead, E, nAct: idx.length, xmin: Math.min(...xsAll), xmax: Math.max(...xsAll) };
      }

      // ---------- layout ----------
      const top = lib.frame(stage, { label: 'Scaling explorer' });
      const subEl = top.wrap.querySelector('.fig-label');
      let R = analyze(byKey[S.axis]);
      const cv = rcanvas(lib, top.frame, { aspect: (w) => clamp(Math.round(w * 0.55), 290, 380), label: 'Validation loss against a scaling resource for Transformer++ (hollow ink circles) and EBT (blue dots), with fitted power laws', draw: () => draw() });
      const ctl = h('div', { class: 'controls' }); stage.appendChild(ctl);
      const modeSeg = lib.segmented({ label: 'Axes', options: [['paper', 'as printed'], ['log', 'log–log']], value: 'paper', onchange: (v) => { setMode(v); } });
      ctl.appendChild(h('span', { class: 'fig-label' }, 'axes')); ctl.appendChild(modeSeg.el);
      const lineSeg = lib.segmented({ label: 'Lines', options: [['curves', 'paper'], ['fits', 'fits'], ['both', 'both']], value: 'curves', onchange: (v) => { S.lines = v; draw(); } });
      ctl.appendChild(h('span', { class: 'fig-label' }, 'lines')); ctl.appendChild(lineSeg.el);
      const bSlopes = lib.button('slopes', () => { S.slopes = !S.slopes; bSlopes.setAttribute('aria-pressed', String(S.slopes)); if (S.slopes) { if (S.lines === 'curves') { S.lines = 'fits'; lineSeg.set('fits'); } setMode('log'); } draw(); }, { aria: 'show slope triangles' });
      bSlopes.setAttribute('aria-pressed', 'false'); ctl.appendChild(bSlopes);
      const ctl2 = h('div', { class: 'controls' }); stage.appendChild(ctl2);
      const extS = lib.slider({ id: 'scaling-ext', label: 'extend the fits', min: 0, max: 3, step: 0.01, value: 0, fmt: v => v < 0.01 ? 'off' : '×' + num(P10(v), 2) + ' largest', oninput: v => { S.ext = v; if (v > 0.01) { if (S.lines === 'curves') { S.lines = 'fits'; lineSeg.set('fits'); } setMode('log', true); } refresh(); } });
      const floorS = lib.slider({ id: 'scaling-floor', label: 'assumed floor E₀', min: 0, max: 0.9, step: 0.01, value: 0, fmt: v => v < 0.005 ? 'none (paper)' : Math.round(v * 100) + '% of lowest', oninput: v => { S.floor = v; if (v > 0.005 && S.lines === 'curves') { S.lines = 'fits'; lineSeg.set('fits'); } refresh(); } });
      ctl2.appendChild(extS.el); ctl2.appendChild(floorS.el);

      const row = h('div', { class: 'fig-row sc-row2' }); stage.appendChild(row);
      const RO = lib.frame(row, { label: 'Fit readout' }); RO.wrap.style.flex = '1 1 220px';
      const roBox = h('div', { class: 'sc-ro', 'aria-live': 'polite' }); RO.frame.appendChild(roBox);
      roBox.addEventListener('click', (ev) => { if (ev.target && ev.target.dataset && ev.target.dataset.restore) { dropOf(S.axis).clear(); refresh(); } });
      const AXF = lib.frame(row, { label: 'All axes · click to select' }); AXF.wrap.style.flex = '1.3 1 300px';
      const list = h('div', { class: 'sc-axes', role: 'group', 'aria-label': 'Scaling axis' }); AXF.frame.appendChild(list);
      list.appendChild(h('div', { class: 'sc-head' }, h('span', {}, 'axis · fig'), h('span', {}, 'printed  | refit'), h('span', { class: 'r' }, 'gain'), h('span', { class: 'r', title: 'which model has the lower loss at the largest run both reached' }, 'lower')));
      const gmax = 0.55, gmin = -0.1;
      const rowsEl = {};
      let grp = null;
      AX.forEach(a => {
        if (a.group !== grp) { grp = a.group; list.appendChild(h('div', { class: 'sc-grp' }, a.group)); }
        const bar = h('i'), tick = h('b', { class: 'tk' }), val = h('span', { class: 'v' }), lead = h('span', { class: 'ld' });
        const b = h('button', { type: 'button', class: 'sc-row', 'aria-pressed': 'false', title: a.p.printed_title || '' },
          h('span', { class: 'nm' }, a.name, h('span', { class: 'fg' }, a.p.figure.replace('Fig ', ''))),
          h('span', { class: 'br' }, h('span', { class: 'z', style: { left: (-gmin / (gmax - gmin) * 100) + '%' } }), bar, tick), val, lead);
        b.addEventListener('click', () => { selectAxis(a.key); });
        list.appendChild(b); rowsEl[a.key] = { b, bar, tick, val, lead };
      });
      const pos = (g) => clamp((g - gmin) / (gmax - gmin), 0, 1) * 100;
      function updateList() {
        AX.forEach(a => {
          const r = rowsEl[a.key], pr = a.p.rate_gain_printed != null ? a.p.rate_gain_printed : a.p.rate_gain;
          const A2 = a === byKey[S.axis] ? R : analyze(a);
          const z = pos(0), x = pos(pr);
          r.bar.style.left = Math.min(z, x) + '%'; r.bar.style.width = Math.max(0.6, Math.abs(x - z)) + '%';
          r.tick.style.left = pos(A2.gain) + '%';
          const changed = Math.abs(A2.gain - pr) > 0.0006;
          r.tick.style.display = changed ? 'block' : 'none';
          r.val.innerHTML = (pr * 100).toFixed(2) + '%' + (changed ? ` <em>→ ${(A2.gain * 100).toFixed(1)}</em>` : '');
          const L = A2.lead, nm = shortN(a);
          r.lead.textContent = L.b < L.a ? nm[1] : nm[0];
          r.lead.classList.toggle('ebt', L.b < L.a);
          r.b.setAttribute('aria-pressed', String(a.key === S.axis));
        });
      }

      // ---------- drawing ----------
      let hatchPat = null;
      function hatch(g) { const c = document.createElement('canvas'); c.width = c.height = 8; const q = c.getContext('2d'); q.strokeStyle = 'rgba(47,60,255,0.16)'; q.lineWidth = 1; q.beginPath(); q.moveTo(-1, 9); q.lineTo(9, -1); q.stroke(); return g.createPattern(c, 'repeat'); }
      let MAP = null;
      function geometry() {
        const ax = R.ax, W = cv.w, H = cv.h, g = cv.ctx;
        const title = `${ax.p.figure} · ${ax.p.printed_title || ax.p.title}`;
        const tm = lib.measure(g, title, { size: 11.5, kind: 'mono', maxWidth: W - 20 });
        const rect = { x: 58, y: 18 + tm.h + 8, w: W - 58 - 14, h: 0 }; rect.h = H - rect.y - 46;
        const xmaxV = R.xmax * P10(S.ext);
        const xs = [R.xmin, xmaxV], spx = R.xmax - R.xmin;
        const xl = [R.xmin - spx * 0.05, xmaxV + spx * 0.05];
        const lx = [L10(R.xmin), L10(xmaxV)], slx = (lx[1] - lx[0]) || 0.3;
        const xg = [lx[0] - slx * 0.05, lx[1] + slx * 0.05];
        const ys = R.A.points.concat(R.B.points).map(q => q[1]);
        if (S.ext > 0.01) { [R.fA, R.fB].forEach(f => { if (f) { const v = fitY(f, xmaxV); if (isFinite(v) && v > 0) ys.push(v); } }); }
        const ylo = Math.min(...ys), yhi = Math.max(...ys), spy = (yhi - ylo) || yhi * 0.1;
        const yl = [ylo - spy * 0.07, yhi + spy * 0.07];
        const ly = [L10(ylo), L10(yhi)], sly = (ly[1] - ly[0]) || 0.05;
        const yg = [ly[0] - sly * 0.07, ly[1] + sly * 0.07];
        const tx = S.tx, ty = S.ty;
        const nx = (v) => (1 - tx) * (v - xl[0]) / (xl[1] - xl[0]) + (tx > 0 ? tx * (L10(Math.max(v, 1e-300)) - xg[0]) / (xg[1] - xg[0]) : 0);
        const ny = (v) => (1 - ty) * (v - yl[0]) / (yl[1] - yl[0]) + (ty > 0 ? ty * (L10(Math.max(v, 1e-300)) - yg[0]) / (yg[1] - yg[0]) : 0);
        MAP = { rect, title, xl, xg, yl, yg, xmaxV, X: (v) => rect.x + nx(v) * rect.w, Y: (v) => rect.y + rect.h - ny(v) * rect.h, xs };
        return MAP;
      }
      function polyline(g, pts, col, o = {}) {
        if (pts.length < 2) return; g.save(); g.strokeStyle = col; g.lineWidth = o.w || 1.5; g.lineJoin = 'round'; g.lineCap = 'round'; g.globalAlpha *= (o.alpha == null ? 1 : o.alpha);
        if (o.dash) g.setLineDash(o.dash); g.beginPath(); g.moveTo(pts[0][0], pts[0][1]); for (let i = 1; i < pts.length; i++) g.lineTo(pts[i][0], pts[i][1]); g.stroke(); g.restore();
      }
      function draw() {
        if (!cv.w) return;
        const g = cv.ctx, ax = R.ax, M = geometry(), r = M.rect, W = cv.w;
        cv.clear();
        if (!hatchPat) hatchPat = hatch(g);
        const [nA, nB] = names(ax);
        // title
        lib.text(g, M.title, 10, 12, { size: 11.5, kind: 'mono', color: C.ink, maxWidth: W - 20 });
        // extrapolation region
        const xEdge = M.X(R.xmax);
        if (S.ext > 0.01 && xEdge < r.x + r.w - 2) {
          g.save(); g.fillStyle = hatchPat; g.fillRect(xEdge, r.y, r.x + r.w - xEdge, r.h); g.restore();
          polyline(g, [[xEdge, r.y], [xEdge, r.y + r.h]], C.bad, { w: 1, dash: [3, 3] });
          if (r.x + r.w - xEdge > 120) lib.text(g, 'EXTRAPOLATION · NO DATA', r.x + r.w - 6, r.y + 6, { size: 10.5, kind: 'mono', color: C.bad, align: 'right', spacing: 0.5 });
        }
        // ticks + grid
        const logX = S.tx > 0.5, logY = S.ty > 0.5;
        const xt = logX ? logTicks(M.xg[0], M.xg[1], Math.max(3, Math.floor(r.w / 80))) : linTicks(M.xl[0], M.xl[1], Math.max(3, Math.floor(r.w / 90)));
        const yt = logY ? logTicks(M.yg[0], M.yg[1], 6) : linTicks(M.yl[0], M.yl[1], 5);
        g.save(); g.strokeStyle = C.rule; g.lineWidth = 1;
        xt.forEach(v => { const X = M.X(v); if (X < r.x - 0.5 || X > r.x + r.w + 0.5) return; g.beginPath(); g.moveTo(X, r.y); g.lineTo(X, r.y + r.h); g.stroke(); });
        yt.forEach(v => { const Y = M.Y(v); if (Y < r.y - 0.5 || Y > r.y + r.h + 0.5) return; g.beginPath(); g.moveTo(r.x, Y); g.lineTo(r.x + r.w, Y); g.stroke(); });
        g.restore();
        let lastXr = -1e9;
        xt.forEach(v => { const X = M.X(v); if (X < r.x - 0.5 || X > r.x + r.w + 0.5) return; const s = tickFmt(v); const m = lib.measure(g, s, { size: 11, kind: 'mono' }); if (X - m.w / 2 < lastXr + 6) return; lastXr = X + m.w / 2; lib.text(g, s, X, r.y + r.h + 6, { size: 11, kind: 'mono', color: C.muted, align: 'center' }); });
        yt.forEach(v => { const Y = M.Y(v); if (Y < r.y - 0.5 || Y > r.y + r.h + 0.5) return; lib.text(g, tickFmt(v), r.x - 6, Y, { size: 11, kind: 'mono', color: C.muted, align: 'right', baseline: 'middle' }); });
        g.save(); g.strokeStyle = C.ink; g.lineWidth = 1; g.beginPath(); g.moveTo(r.x, r.y); g.lineTo(r.x, r.y + r.h); g.lineTo(r.x + r.w, r.y + r.h); g.stroke(); g.restore();
        lib.text(g, cleanLab(ax.p.x.label) + (logX ? ' · log' : ''), r.x + r.w / 2, r.y + r.h + 24, { size: 11.5, kind: 'mono', color: C.ink, align: 'center', maxWidth: r.w });
        g.save(); g.translate(13, r.y + r.h / 2); g.rotate(-Math.PI / 2); lib.text(g, cleanLab(ax.p.y.label) + (logY ? ' · log' : ''), 0, 0, { size: 11.5, kind: 'mono', color: C.ink, align: 'center', baseline: 'middle' }); g.restore();

        g.save(); g.beginPath(); g.rect(r.x + 1, r.y - 4, r.w + 4, r.h + 4); g.clip();
        // floor
        if (R.E > 0) {
          const Yf = M.Y(R.E);
          if (Yf >= r.y && Yf <= r.y + r.h) { polyline(g, [[r.x, Yf], [r.x + r.w, Yf]], C.muted, { w: 1, dash: [2, 4] }); lib.text(g, 'assumed floor E₀ = ' + num(R.E, 3), r.x + 6, Yf - 4, { size: 11, kind: 'mono', color: C.muted, baseline: 'alphabetic' }); }
        }
        // paper curves
        if (S.lines !== 'fits') {
          [[R.A, C.ink, 0.45], [R.B, C.blue, 0.55]].forEach(([s, col, al]) => { if (s.paper_curve && s.paper_curve.points) polyline(g, s.paper_curve.points.map(q => [M.X(q[0]), M.Y(q[1])]), col, { w: 1.2, alpha: al }); });
        }
        // fits
        if (S.lines !== 'curves') {
          [[R.fA, R.A, C.ink, 1.4], [R.fB, R.B, C.blue, 2]].forEach(([f, s, col, w]) => {
            if (!f) return; const xs = s.points.map(q => q[0]); const a = L10(Math.min(...xs)), b = L10(Math.max(...xs)), e = L10(M.xmaxV);
            const seg = (u0, u1, n) => { const out = []; for (let i = 0; i <= n; i++) { const u = u0 + (u1 - u0) * i / n; out.push([M.X(P10(u)), M.Y(fitY(f, P10(u)))]); } return out; };
            polyline(g, seg(a, b, 80), col, { w });
            if (e > b + 1e-6) polyline(g, seg(b, e, 80), col, { w, dash: [6, 5] });
          });
        }
        // FLOPs pairing arrows
        if (S.pairs && ax.key === 'flops') {
          R.A.points.forEach((q, i) => { const e2 = R.B.points[i]; if (!e2) return; const x1 = M.X(q[0]) + 5, y1 = M.Y(q[1]), x2 = M.X(e2[0]) - 6, y2 = M.Y(e2[1]); lib.arrow(g, x1, y1, x2, y2, { color: C.muted, width: 1, head: 6, dash: [3, 3] }); });
          const q = R.A.points[1], e2 = R.B.points[1]; const ratio = R.B.points.reduce((s, p, i) => s + p[0] / R.A.points[i][0], 0) / R.B.points.length;
          if (q && e2) lib.text(g, 'same model, ×' + ratio.toFixed(2) + ' FLOPs', (M.X(q[0]) + M.X(e2[0])) / 2, Math.min(M.Y(q[1]), M.Y(e2[1])) - 16, { size: 11, kind: 'mono', color: C.ink, align: 'center' });
        }
        // gap bracket at the largest run both reached
        if (S.lines !== 'curves' && R.fA && R.fB) {
          const L = R.lead, X = M.X(L.x), ya = M.Y(L.a), yb = M.Y(L.b);
          if (X > r.x && X < r.x + r.w) {
            const bx = X + 9;
            polyline(g, [[bx - 3, ya], [bx, ya], [bx, yb], [bx - 3, yb]], C.ink, { w: 1 });
            const gp = L.b / L.a - 1, s = `${shortN(ax)[1]} ${gp >= 0 ? '+' : MINUS}${Math.abs(gp * 100).toFixed(1)}%`;
            lib.text(g, s, X - 6, Math.max(ya, yb) + 10, { size: 11, kind: 'mono', color: C.ink, align: 'right', weight: 600 });
          }
        }
        // crossing
        if (S.lines !== 'curves' && isFinite(R.uc) && logX) {
          const xc = P10(R.uc), X = M.X(xc);
          if (X > r.x + 4 && X < r.x + r.w - 4 && R.fA) {
            const Y = M.Y(fitY(R.fA, xc));
            if (Y > r.y && Y < r.y + r.h) {
              g.save(); g.strokeStyle = C.ink; g.lineWidth = 1; g.beginPath(); g.arc(X, Y, 7, 0, 7); g.stroke(); g.beginPath(); g.moveTo(X - 4, Y); g.lineTo(X + 4, Y); g.moveTo(X, Y - 4); g.lineTo(X, Y + 4); g.stroke(); g.restore();
              const s = 'fits cross ≈ ' + KIND[ax.kind](xc), tw = lib.measure(g, s, { size: 11, kind: 'mono' }).w;
              const lx = Math.max(r.x + 4 + tw, X - 10);
              lib.text(g, s, lx, Y + 12, { size: 11, kind: 'mono', color: C.ink, align: 'right' });
            }
          }
        }
        // slope triangles
        if (S.slopes && S.lines !== 'curves' && logX && logY && S.tx > 0.95 && S.ty > 0.95) {
          [[R.fA, R.A, C.ink, 0.1, 0.42, 'below'], [R.fB, R.B, C.blue, 0.55, 0.88, 'above']].forEach(([f, s, col, p0, p1]) => {
            if (!f) return; const xs = s.points.map(q => q[0]); const a = L10(Math.min(...xs)), b = L10(Math.max(...xs));
            const x0 = P10(a + (b - a) * p0), x1 = P10(a + (b - a) * p1);
            const X0 = M.X(x0), X1 = M.X(x1), Y0 = M.Y(fitY(f, x0)), Y1 = M.Y(fitY(f, x1));
            polyline(g, [[X0, Y0], [X1, Y0], [X1, Y1]], col, { w: 1.2, dash: [4, 3] });
            const sl = 'slope ' + neg(f.m.toFixed(3));
            const tw = lib.measure(g, sl, { size: 11, kind: 'mono', weight: 600 }).w;
            const tx = X1 + 6 + tw < r.x + r.w ? X1 + 6 : X1 - 6 - tw;
            lib.text(g, sl, tx, (Y0 + Y1) / 2, { size: 11, kind: 'mono', color: col, weight: 600, baseline: 'middle' });
          });
        }
        // points
        const drop = dropOf(ax.key);
        [[R.A, 'A'], [R.B, 'B']].forEach(([s, w]) => {
          s.points.forEach((q, i) => {
            const X = M.X(q[0]), Y = M.Y(q[1]), off = drop.has(i);
            if (off) { g.save(); g.strokeStyle = C.faint; g.lineWidth = 1; g.beginPath(); g.arc(X, Y, 4, 0, 7); g.stroke(); g.beginPath(); g.moveTo(X - 3, Y - 3); g.lineTo(X + 3, Y + 3); g.moveTo(X + 3, Y - 3); g.lineTo(X - 3, Y + 3); g.stroke(); g.restore(); return; }
            if (w === 'A') lib.dot(g, X, Y, 3.6, '#ffffff', { stroke: C.ink, lw: 1.3 });
            else lib.dot(g, X, Y, 3.8, C.blue, { stroke: '#ffffff', lw: 1 });
          });
        });
        g.restore();
        // legend (top right inside plot)
        const lg1 = '○ ' + nA, lg2 = '● ' + nB;
        const w2 = lib.measure(g, lg2, { size: 11, kind: 'mono' }).w;
        lib.text(g, lg1, r.x + r.w - 8, r.y + (S.ext > 0.01 ? 24 : 6), { size: 11, kind: 'mono', color: C.ink, align: 'right' });
        lib.text(g, lg2, r.x + r.w - 8, r.y + (S.ext > 0.01 ? 40 : 22), { size: 11, kind: 'mono', color: C.blue, align: 'right' });
        void w2;
        // hover tooltip
        if (S.hover) {
          const { s, i } = S.hover, q = (s === 'A' ? R.A : R.B).points[i]; const X = M.X(q[0]), Y = M.Y(q[1]);
          const l1 = `${s === 'A' ? nA : nB} · ${KIND[ax.kind](q[0])}`, l2 = `${isLoss(ax) ? 'loss' : 'ppl'} ${num(q[1], 4)} · approx.`, l3 = drop.has(i) ? 'dropped · click to restore' : 'click to drop this run';
          const tw = Math.max(...[l1, l2, l3].map(t => lib.measure(g, t, { size: 11, kind: 'mono' }).w)) + 14;
          let bx = X + 12, by = Y - 50; if (bx + tw > W - 4) bx = X - 12 - tw; if (by < 2) by = Y + 12;
          g.save(); g.fillStyle = 'rgba(255,255,255,0.96)'; g.strokeStyle = C.ink; g.lineWidth = 1; g.fillRect(bx, by, tw, 50); g.strokeRect(bx + 0.5, by + 0.5, tw - 1, 49); g.restore();
          lib.text(g, l1, bx + 7, by + 6, { size: 11, kind: 'mono', color: s === 'A' ? C.ink : C.blue });
          lib.text(g, l2, bx + 7, by + 20, { size: 11, kind: 'mono', color: C.ink });
          lib.text(g, l3, bx + 7, by + 34, { size: 11, kind: 'mono', color: C.muted });
          g.save(); g.strokeStyle = C.ink; g.lineWidth = 1; g.beginPath(); g.arc(X, Y, 7, 0, 7); g.stroke(); g.restore();
        }
      }

      // ---------- readout ----------
      function cell(k, v, s) { return `<div class="k">${k}</div><div class="v">${v}${s ? `<span class="s">${s}</span>` : ''}</div>`; }
      function updateReadout() {
        const ax = R.ax, [nA, nB] = shortN(ax), pr = ax.p.rate_gain_printed != null ? ax.p.rate_gain_printed : ax.p.rate_gain;
        const out = [];
        if (!R.fA || !R.fB) { roBox.innerHTML = cell('fit', 'needs at least 2 runs per model'); return; }
        const nd = dropOf(ax.key).size;
        out.push(cell('slope m', `${neg(R.fA.m.toFixed(3))} · <b>${neg(R.fB.m.toFixed(3))}</b>`, `${nA} · ${nB}` + (R.E > 0 ? `, slope of L − E₀ (E₀ = ${num(R.E, 3)})` : '')));
        out.push(cell('rate gain', `<b>${pct(R.gain)}</b> <span class="s0">printed ${(pr * 100).toFixed(2)}%</span>`, `${R.nAct} runs fitted` + (nd ? ` · ${nd} dropped · <a href="javascript:void 0" data-restore="1">restore</a>` : '')));
        out.push(cell('per 10×', `×${P10(R.fA.m).toFixed(3)} · <b>×${P10(R.fB.m).toFixed(3)}</b>`, (R.E > 0 ? 'L − E₀' : (isLoss(ax) ? 'loss' : 'perplexity')) + ' per 10× ' + ax.res));
        if (R.jk) out.push(cell('drop one', `${pct(R.jk[0], 1)} … ${pct(R.jk[1], 1)}`, 'gain with any one run left out'));
        const L = R.lead, gp = L.b / L.a - 1;
        out.push(cell('largest run', `${nB} <b>${gp >= 0 ? '+' : MINUS}${Math.abs(gp * 100).toFixed(1)}%</b> vs ${nA}`, `${KIND[ax.kind](L.x)}: ${num(L.b, 4)} vs ${num(L.a, 4)}${L.measured ? '' : ' (fit)'}`));
        if (isFinite(R.uc)) {
          const xc = P10(R.uc), lo = L10(R.xmin), hi = L10(R.xmax);
          let s;
          if (Math.abs(R.gain) < 0.005) s = 'slopes nearly equal: means little';
          else if (R.uc < lo) s = 'below the measured range';
          else if (R.uc <= hi) s = 'inside the measured range';
          else s = `<span class="warn">×${num(xc / R.xmax, 3)} past the largest run</span>`;
          out.push(cell('fits cross', `≈${KIND[ax.kind](xc)}`, s));
        }
        if (S.ext > 0.01) {
          const xe = R.xmax * P10(S.ext), ya = fitY(R.fA, xe), yb = fitY(R.fB, xe), ge = yb / ya - 1;
          out.push(cell('extrapolated', `<span class="warn">${nB} ${ge >= 0 ? '+' : MINUS}${Math.abs(ge * 100).toFixed(1)}% vs ${nA}</span>`, `at ${KIND[ax.kind](xe)}: no run exists there`));
        }
        roBox.innerHTML = out.join('');
      }

      function refresh() { R = analyze(byKey[S.axis]); updateReadout(); updateList(); draw(); }
      function selectAxis(k, keep) {
        if (!byKey[k]) return; S.axis = k; S.hover = null;
        const ax = byKey[k];
        subEl.textContent = 'Scaling explorer · ' + ax.group;
        ctx.setCaption(ax.note + ' Hollow ink circles: ' + names(ax)[0] + '. Blue dots: ' + names(ax)[1] + '. Values approx., read from ' + ax.p.figure + '.');
        if (!keep) setMode(modeSeg.value(), true);
        refresh();
      }
      // axis-mode morph
      const anim = lib.loop((dt) => {
        const k = Math.min(1, dt * 7);
        S.tx += (S.ttx - S.tx) * k; S.ty += (S.tty - S.ty) * k;
        const done = Math.abs(S.ttx - S.tx) < 0.004 && Math.abs(S.tty - S.ty) < 0.004;
        if (done) { S.tx = S.ttx; S.ty = S.tty; }
        draw(); return !done;
      });
      function setMode(m, instant) {
        if (S.ext > 0.01 || S.slopes) m = 'log';
        modeSeg.set(m);
        const p = byKey[S.axis].p;
        S.ttx = m === 'log' ? 1 : (p.x.log ? 1 : 0); S.tty = m === 'log' ? 1 : (p.y.log ? 1 : 0);
        if (instant || lib.reducedMotion || !ctx.visible()) { S.tx = S.ttx; S.ty = S.tty; draw(); } else anim.start();
      }
      // hover / click
      function nearest(ev) {
        if (!MAP) return null; const [px, py] = cv.local(ev); let best = null, bd = 14;
        [['A', R.A], ['B', R.B]].forEach(([s, ser]) => ser.points.forEach((q, i) => { const d = Math.hypot(MAP.X(q[0]) - px, MAP.Y(q[1]) - py); if (d < bd) { bd = d; best = { s, i }; } }));
        return best;
      }
      cv.canvas.addEventListener('mousemove', (ev) => { const n = nearest(ev); const same = (n && S.hover && n.s === S.hover.s && n.i === S.hover.i) || (!n && !S.hover); S.hover = n; cv.canvas.style.cursor = n ? 'pointer' : 'default'; if (!same) draw(); });
      cv.canvas.addEventListener('mouseleave', () => { if (S.hover) { S.hover = null; draw(); } });
      cv.canvas.addEventListener('click', (ev) => {
        const n = nearest(ev); if (!n) return; const d = dropOf(S.axis);
        if (d.has(n.i)) d.delete(n.i); else if (R.n - d.size > 2) d.add(n.i);
        if (S.lines === 'curves') { S.lines = 'fits'; lineSeg.set('fits'); }
        S.hover = n; refresh();
      });

      const CFG = [
        { axis: 'data', mode: 'paper', lines: 'curves', slopes: false, pairs: false, ext: 0, floor: 0 },
        { axis: 'data', mode: 'log', lines: 'fits', slopes: false, pairs: false, ext: 0, floor: 0 },
        { axis: 'data', mode: 'log', lines: 'fits', slopes: true, pairs: false, ext: 0, floor: 0 },
        { axis: 'params', mode: 'log', lines: 'fits', slopes: false, pairs: false, ext: 0, floor: 0 },
        { axis: 'flops', mode: 'log', lines: 'fits', slopes: false, pairs: true, ext: 0, floor: 0 },
        { axis: 'params', mode: 'log', lines: 'fits', slopes: false, pairs: false, ext: 3, floor: 0 },
        { axis: 'params', mode: 'log', lines: 'fits', slopes: false, pairs: false, ext: 0, floor: 0.5 },
      ];
      function apply(c) {
        Object.keys(S.drop).forEach(k => S.drop[k].clear());
        S.lines = c.lines; lineSeg.set(c.lines);
        S.slopes = c.slopes; bSlopes.setAttribute('aria-pressed', String(c.slopes));
        S.pairs = c.pairs; S.ext = c.ext; extS.set(c.ext); S.floor = c.floor; floorS.set(c.floor);
        const prevAxis = S.axis;
        S.axis = c.axis;
        if (prevAxis !== c.axis) { const p = byKey[c.axis].p; S.tx = p.x.log ? 1 : 0; S.ty = p.y.log ? 1 : 0; }
        selectAxis(c.axis, true);
        setMode(c.mode);
      }
      apply(CFG[0]);
      return {
        step(i) { apply(CFG[Math.max(0, Math.min(CFG.length - 1, i))]); },
        show() { draw(); },
        hide() { anim.stop(); S.tx = S.ttx; S.ty = S.tty; },
      };
    },
  });
})();
