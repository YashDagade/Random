/* Panel: results (learning scalability, thinking, images, cost).
   Data: lib.data('scaling') = data/scaling.json, points digitized from the paper's vector figures (approx.).
   Learning view: OLS fits in log10-log10 space computed live (rate gain = |slope EBT| / |slope T++| - 1),
   drop-a-point refits, crossings and extrapolation. Thinking views: Fig 6a (text) and Fig 12 (images) with a pass budget. */
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
  // OLS of log10(y) on log10(x)
  function ols(pts) {
    const q = pts.filter(p => p[0] > 0 && p[1] > 0); const n = q.length; if (n < 2) return null;
    let sx = 0, sy = 0, sxx = 0, sxy = 0;
    q.forEach(([x, y]) => { const a = L10(x), c = L10(y); sx += a; sy += c; sxx += a * a; sxy += a * c; });
    const mx = sx / n, my = sy / n, vxx = sxx / n - mx * mx, vxy = sxy / n - mx * my;
    if (!(vxx > 0)) return null;
    const m = vxy / vxx;
    return { m, b: my - m * mx, n };
  }
  const fitY = (f, x) => P10(f.b + f.m * L10(x));
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
    id: 'results',
    nav: 'Results',
    title: 'Results: steeper slopes, thinking that pays, at a price',
    lede: 'EBTs improve faster with scale but start behind, extra thinking helps most far from the training data, and every training step costs several times more compute.',
    text: `
      <p>A <b>scaling law</b> says loss falls as a power of a resource $R$ (tokens, parameters, FLOPs): $L\\approx A\\,R^{-\\beta}$. On log-log axes that is a straight line with slope $-\\beta$. The paper's "X% faster" compares slopes, $|\\beta_{\\text{EBT}}|/|\\beta_{\\text{T++}}|-1$; our refit reproduces every printed rate this way. A steeper line is not yet a lower one.</p>`,
    steps: [
      { label: 'Data: the 35.98%', html: '<p>Fig 4a: small models, more training tokens. Each doubling of data multiplies EBT perplexity by about 0.855 and Transformer++ perplexity by 0.891, slopes 0.226 vs 0.166: a 36% steeper line. EBT starts worse and crosses below after about 3B tokens. Click a point to drop it and refit.</p>' },
      { label: 'Other axes: from 28% down to 0.02%', html: '<p>Batch size gives 28.46%, depth 5.29%, parameters 2.91%, FLOPs 2.92%, width 0.02% (pick any row). On parameters EBT is worse at every measured size. Extended, the two fits would meet near 255B parameters, hundreds of times past the largest model: an extrapolation, not a result.</p>' },
      { label: 'The price of a training step', html: '<p>An EBT step needs a forward pass, a backward pass for $\\nabla_{\\hat y}E$, and a Hessian-vector product to train through it: about 1.66× a Transformer++ step. Context plus guesses doubles the sequence (3.33×), and two steps double it again: 6.66× (p.36). The arrows shift each run right by that factor. At every measured budget Transformer++ is lower.</p>' },
      { label: 'Thinking: 29% against itself', html: '<p>Fig 6a, averaged over four out-of-distribution text sets. Without extra thinking EBT (≈44.8) is worse than Transformer++ (≈38.4), which cannot use more passes. One extra step already beats it; best-of-N verification reaches ≈31.8, 29% below EBT\'s own baseline. Fig 7 adds that the further the data is from pretraining, the larger the gain (≈12% up to ≈23%).</p>' },
      { label: 'Images: 3 passes versus 300', html: '<p>Table 4, denoising at a noise level neither model trained on: EBT reaches 23.29 dB after 3 passes, a Diffusion Transformer 19.56 dB after 300. With 1 or 2 passes EBT only ties DiT at 100 or 200, and in distribution the gap is small (27.25 vs 26.58). Each EBT pass also runs backward, so the compute saving is under 100×.</p>' },
    ],
    after: `<p class="note">One seed, no error bars, models up to 708M parameters and below $10^{21}$ training FLOPs. Plotted points are digitized (approx.); printed rates and table values are exact.</p>`,
    source: [{ kind: 'paper', note: 'Figs 4, 5, 6a, 9, 12, B.3, Table 4 (digitized, approx.)' }, { kind: 'ext', note: 'refits, crossings and extrapolation are ours' }],
    figure(stage, ctx) {
      const { lib } = ctx, h = lib.h, C = lib.C;
      const D = lib.data('scaling');
      if (!D || !D.plots) { stage.appendChild(h('p', { class: 'callout warn' }, 'Scaling data missing (data/scaling.json).')); return {}; }
      const P = {}; D.plots.forEach(p => { P[p.id] = p; });
      const AX = [
        { key: 'data', id: 'fig4a', group: 'Text', name: 'Data', kind: 'tokB', res: 'data' },
        { key: 'batch', id: 'fig4b', group: 'Text', name: 'Batch', kind: 'tokK', res: 'batch size' },
        { key: 'depth', id: 'fig4c', group: 'Text', name: 'Depth', kind: 'depth', res: 'depth' },
        { key: 'params', id: 'fig5a', group: 'Text', name: 'Params', kind: 'paramM', res: 'parameters' },
        { key: 'flops', id: 'fig5b', group: 'Text', name: 'FLOPs', kind: 'flops', res: 'training FLOPs' },
        { key: 'width', id: 'fig5c', group: 'Text', name: 'Width', kind: 'width', res: 'width' },
        { key: 'vwidth', id: 'fig9a', group: 'Video', name: 'Width', kind: 'width', res: 'width' },
        { key: 'vparams', id: 'fig9b', group: 'Video', name: 'Params', kind: 'paramM', res: 'parameters' },
        { key: 'fw', id: 'figB3a', group: 'Text · FineWeb', name: 'Data', kind: 'tokB', res: 'data' },
      ].filter(a => P[a.id] && P[a.id].series && P[a.id].series.length === 2);
      AX.forEach(a => { a.p = P[a.id]; });
      const byKey = {}; AX.forEach(a => { byKey[a.key] = a; });
      const isLoss = (ax) => /loss/i.test(ax.p.y.label);
      const cleanLab = (s) => s.replace(/,\s*log scale/i, '').replace(/\s*\(log scale\)/i, '');
      const ser = (id, i) => (P[id] && P[id].series[i]) ? P[id].series[i].points : [];

      const S = { view: 'learn', axis: 'data', mode: 'log', tx: 1, ty: 1, ttx: 1, tty: 1, slopes: false, pairs: false, ext: 0, drop: {}, hover: null, bThink: 30, bImg: 300 };
      const dropOf = (k) => S.drop[k] || (S.drop[k] = new Set());

      function analyze(ax) {
        const A = ax.p.series[0], B = ax.p.series[1], drop = dropOf(ax.key);
        const n = Math.min(A.points.length, B.points.length);
        const aP = A.points.filter((_, i) => !drop.has(i)), bP = B.points.filter((_, i) => !drop.has(i));
        const fA = ols(aP), fB = ols(bP), gain = gainOf(fA, fB);
        const uc = (fA && fB && Math.abs(fA.m - fB.m) > 1e-12) ? (fB.b - fA.b) / (fA.m - fB.m) : NaN;
        const lastA = A.points[A.points.length - 1], lastB = B.points[B.points.length - 1];
        let lead;
        if (Math.abs(lastA[0] - lastB[0]) / lastA[0] < 0.02) lead = { x: lastA[0], a: lastA[1], b: lastB[1], measured: true };
        else { const xc = Math.min(lastA[0], lastB[0]); const a = lastA[0] === xc ? lastA[1] : fitY(fA, xc), b = lastB[0] === xc ? lastB[1] : fitY(fB, xc); lead = { x: xc, a, b, measured: false }; }
        const xsAll = A.points.concat(B.points).map(q => q[0]);
        return { ax, A, B, n, fA, fB, gain, uc, lead, nAct: n - drop.size, xmin: Math.min(...xsAll), xmax: Math.max(...xsAll) };
      }

      // ---------- layout ----------
      const vrow = h('div', { class: 'controls' }); stage.appendChild(vrow);
      const viewSeg = lib.segmented({ label: 'Result', options: [['learn', 'learning · scaling'], ['think', 'thinking · text'], ['img', 'thinking · images']], value: 'learn', onchange: (v) => { setView(v); } });
      vrow.append(h('span', { class: 'fig-label' }, 'show'), viewSeg.el);
      const top = lib.frame(stage, { label: 'Fig 4a' });
      const labEl = top.wrap.querySelector('.fig-label');
      let R = analyze(byKey[S.axis]);
      const cv = rcanvas(lib, top.frame, { aspect: (w) => w < 520 ? Math.round(w * 0.82) : clamp(Math.round(w * 0.52), 290, 360), label: 'Results chart: scaling fits for Transformer++ and EBT, or performance against forward passes', draw: () => draw() });
      const ro = h('div', { class: 'readout res-ro', 'aria-live': 'polite' }); stage.appendChild(ro);
      // learning controls
      const cL = h('div', { class: 'controls' }); stage.appendChild(cL);
      const modeSeg = lib.segmented({ label: 'Axes', options: [['paper', 'as printed'], ['log', 'log-log + fits']], value: 'log', onchange: (v) => { setMode(v); } });
      const bSlopes = lib.button('slopes', () => { S.slopes = !S.slopes; bSlopes.setAttribute('aria-pressed', String(S.slopes)); if (S.slopes) setMode('log'); draw(); }, { aria: 'show slope triangles' });
      bSlopes.setAttribute('aria-pressed', 'false');
      const extS = lib.slider({ id: 'results-ext', label: 'extend the fits', min: 0, max: 3, step: 0.01, value: 0, fmt: v => v < 0.01 ? 'off' : '×' + num(P10(v), 2) + ' largest', oninput: v => { S.ext = v; if (v > 0.01) setMode('log', true); refresh(); } });
      cL.append(modeSeg.el, bSlopes, extS.el);
      // thinking controls
      const cT = h('div', { class: 'controls' }); stage.appendChild(cT);
      const thS = lib.slider({ id: 'results-tbudget', label: 'budget, passes per token', min: 2, max: 30, step: 1, value: S.bThink, fmt: v => v + ' passes', oninput: v => { sweep.stop(); S.bThink = v; draw(); } });
      const imS = lib.slider({ id: 'results-ibudget', label: 'budget, forward passes', min: 0, max: L10(300), step: 0.005, value: L10(S.bImg), fmt: v => String(Math.round(P10(v))), oninput: v => { sweep.stop(); S.bImg = Math.round(P10(v)); draw(); } });
      cT.append(thS.el, imS.el);
      // axis list
      const AXF = lib.frame(stage, { label: 'Rate gain per axis · click to plot' });
      const list = h('div', { class: 'res-axes', role: 'group', 'aria-label': 'Scaling axis' }); AXF.frame.appendChild(list);
      list.appendChild(h('div', { class: 'res-head' }, h('span', {}, 'axis · fig'), h('span', {}, 'steeper by'), h('span', { class: 'r' }, 'printed'), h('span', { class: 'r', title: 'which model has the lower loss at the largest run both reached' }, 'lower')));
      const gmax = 0.4, gmin = -0.02, rowsEl = {};
      let grp = null;
      AX.forEach(a => {
        if (a.group !== grp) { grp = a.group; list.appendChild(h('div', { class: 'res-grp' }, a.group)); }
        const bar = h('i'), tick = h('b', { class: 'tk' }), val = h('span', { class: 'v' }), lead = h('span', { class: 'ld' });
        const b = h('button', { type: 'button', class: 'res-row', 'aria-pressed': 'false', title: a.p.printed_title || '' },
          h('span', { class: 'nm' }, a.name, h('span', { class: 'fg' }, a.p.figure.replace('Fig ', ''))),
          h('span', { class: 'br' }, h('span', { class: 'z', style: { left: (-gmin / (gmax - gmin) * 100) + '%' } }), bar, tick), val, lead);
        b.addEventListener('click', () => { if (S.view !== 'learn') setView('learn'); selectAxis(a.key); });
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
          const rv = A2.gain * 100;
          r.val.innerHTML = (pr * 100).toFixed(2) + '%' + (changed ? ` <em>→ ${Math.abs(rv) < 0.05 ? '0.0' : neg(rv.toFixed(1))}</em>` : '');
          const L = A2.lead;
          r.lead.textContent = L.b < L.a ? 'EBT' : 'T++';
          r.lead.classList.toggle('ebt', L.b < L.a);
          r.b.setAttribute('aria-pressed', String(a.key === S.axis && S.view === 'learn'));
        });
      }

      // ---------- shared drawing helpers ----------
      let HITS = [];
      const T = (g, s, x, y, o = {}) => lib.text(g, s, x, y, Object.assign({ size: 11, kind: 'mono', color: C.ink }, o));
      function poly(g, pts, col, o = {}) {
        if (pts.length < 2) return; g.save(); g.strokeStyle = col; g.lineWidth = o.w || 1.5; g.lineJoin = 'round'; g.lineCap = 'round'; g.globalAlpha *= (o.alpha == null ? 1 : o.alpha);
        if (o.dash) g.setLineDash(o.dash); g.beginPath(); g.moveTo(pts[0][0], pts[0][1]); for (let i = 1; i < pts.length; i++) g.lineTo(pts[i][0], pts[i][1]); g.stroke(); g.restore();
      }
      function marker(g, kind, x, y, r, fill, stroke) {
        g.save(); g.beginPath();
        if (kind === 'star') { for (let i = 0; i < 10; i++) { const a = -Math.PI / 2 + i * Math.PI / 5, rr = i % 2 ? r * 0.45 : r * 1.25; g.lineTo(x + rr * Math.cos(a), y + rr * Math.sin(a)); } g.closePath(); }
        else if (kind === 'tri') { g.moveTo(x, y - r * 1.2); g.lineTo(x + r * 1.1, y + r * 0.8); g.lineTo(x - r * 1.1, y + r * 0.8); g.closePath(); }
        else g.arc(x, y, r, 0, Math.PI * 2);
        if (fill) { g.fillStyle = fill; g.fill(); } if (stroke) { g.strokeStyle = stroke; g.lineWidth = 1.2; g.stroke(); } g.restore();
      }
      function axesBox(g, o) {
        const W = cv.w, H = cv.h, rect = { x: 56, y: 16, w: W - 56 - 14, h: H - 16 - 44 };
        const tx = (v) => o.xlog ? L10(v) : v, ty = (v) => o.ylog ? L10(v) : v;
        const [xa, xb] = o.xlim.map(tx), [ya, yb] = o.ylim.map(ty);
        const X = (v) => rect.x + (tx(v) - xa) / (xb - xa) * rect.w, Y = (v) => rect.y + rect.h - (ty(v) - ya) / (yb - ya) * rect.h;
        g.save(); g.strokeStyle = C.rule; g.lineWidth = 1;
        (o.xticks || []).forEach(v => { g.beginPath(); g.moveTo(X(v), rect.y); g.lineTo(X(v), rect.y + rect.h); g.stroke(); });
        (o.yticks || []).forEach(v => { g.beginPath(); g.moveTo(rect.x, Y(v)); g.lineTo(rect.x + rect.w, Y(v)); g.stroke(); });
        g.strokeStyle = C.ink; g.beginPath(); g.moveTo(rect.x, rect.y); g.lineTo(rect.x, rect.y + rect.h); g.lineTo(rect.x + rect.w, rect.y + rect.h); g.stroke(); g.restore();
        let lastR = -1e9;
        (o.xticks || []).forEach(v => { const s = num(v), w = lib.measure(g, s, { size: 11, kind: 'mono' }).w, x = X(v); if (x - w / 2 < lastR + 4) return; lastR = x + w / 2; T(g, s, x, rect.y + rect.h + 6, { color: C.muted, align: 'center' }); });
        (o.yticks || []).forEach(v => T(g, num(v), rect.x - 6, Y(v), { color: C.muted, align: 'right', baseline: 'middle' }));
        T(g, o.xlabel, rect.x + rect.w / 2, rect.y + rect.h + 24, { size: 11.5, align: 'center', maxWidth: rect.w });
        g.save(); g.translate(13, rect.y + rect.h / 2); g.rotate(-Math.PI / 2); T(g, o.ylabel, 0, 0, { size: 11.5, align: 'center', baseline: 'middle', maxWidth: rect.h }); g.restore();
        return { rect, X, Y };
      }
      function budgetLine(g, M, x, label) {
        const r = M.rect, xb = M.X(x);
        g.save(); g.fillStyle = 'rgba(17,17,17,0.035)'; g.fillRect(xb, r.y, r.x + r.w - xb, r.h); g.restore();
        poly(g, [[xb, r.y], [xb, r.y + r.h]], C.ink, { w: 1, dash: [3, 4], alpha: 0.6 });
        const bw = lib.measure(g, label, { size: 11, kind: 'mono' }).w, rt = xb + 5 + bw > r.x + r.w;
        T(g, label, rt ? xb - 5 : xb + 5, r.y + 2, { align: rt ? 'right' : 'left', color: C.muted });
      }
      const ring = (g, x, y, col) => { g.save(); g.strokeStyle = col; g.lineWidth = 1.2; g.beginPath(); g.arc(x, y, 9, 0, 7); g.stroke(); g.restore(); };
      function tooltip(g) {
        if (!S.hover) return; const { x, y, lines } = S.hover;
        const tw = Math.max(...lines.map(t => lib.measure(g, t, { size: 11, kind: 'mono' }).w)) + 14, th = lines.length * 14 + 10;
        let bx = x + 12, by = y - th - 6; if (bx + tw > cv.w - 4) bx = x - 12 - tw; if (by < 2) by = y + 12;
        g.save(); g.fillStyle = 'rgba(255,255,255,0.96)'; g.fillRect(bx, by, tw, th); g.strokeStyle = C.ink; g.lineWidth = 1; g.strokeRect(bx + 0.5, by + 0.5, tw - 1, th - 1); g.beginPath(); g.arc(x, y, 7, 0, 7); g.stroke(); g.restore();
        lines.forEach((t, i) => T(g, t, bx + 7, by + 6 + i * 14, { color: i === 0 ? C.ink : C.muted }));
      }
      let hatchPat = null;
      function hatch(g) { const c = document.createElement('canvas'); c.width = c.height = 8; const q = c.getContext('2d'); q.strokeStyle = 'rgba(47,60,255,0.16)'; q.lineWidth = 1; q.beginPath(); q.moveTo(-1, 9); q.lineTo(9, -1); q.stroke(); return g.createPattern(c, 'repeat'); }

      // ---------- learning view: the scaling explorer ----------
      function geometry() {
        const W = cv.w, H = cv.h;
        const rect = { x: 58, y: 14, w: W - 58 - 14, h: 0 }; rect.h = H - rect.y - 46;
        const xmaxV = R.xmax * P10(S.ext), spx = R.xmax - R.xmin;
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
        return { rect, xl, xg, yl, yg, xmaxV, X: (v) => rect.x + nx(v) * rect.w, Y: (v) => rect.y + rect.h - ny(v) * rect.h };
      }
      function drawLearn(g) {
        const ax = R.ax, M = geometry(), r = M.rect;
        const showFits = S.mode === 'log' || dropOf(ax.key).size > 0, showCurves = S.mode !== 'log';
        if (!hatchPat) hatchPat = hatch(g);
        const xEdge = M.X(R.xmax);
        if (S.ext > 0.01 && xEdge < r.x + r.w - 2) {
          g.save(); g.fillStyle = hatchPat; g.fillRect(xEdge, r.y, r.x + r.w - xEdge, r.h); g.restore();
          poly(g, [[xEdge, r.y], [xEdge, r.y + r.h]], C.bad, { w: 1, dash: [3, 3] });
          if (r.x + r.w - xEdge > 120) T(g, 'EXTRAPOLATION · NO DATA', r.x + r.w - 8, r.y + 40, { size: 10.5, color: C.bad, align: 'right', spacing: 0.5 });
        }
        const logX = S.tx > 0.5, logY = S.ty > 0.5;
        const xt = logX ? logTicks(M.xg[0], M.xg[1], Math.max(3, Math.floor(r.w / 80))) : linTicks(M.xl[0], M.xl[1], Math.max(3, Math.floor(r.w / 90)));
        const yt = logY ? logTicks(M.yg[0], M.yg[1], 6) : linTicks(M.yl[0], M.yl[1], 5);
        g.save(); g.strokeStyle = C.rule; g.lineWidth = 1;
        xt.forEach(v => { const X = M.X(v); if (X < r.x - 0.5 || X > r.x + r.w + 0.5) return; g.beginPath(); g.moveTo(X, r.y); g.lineTo(X, r.y + r.h); g.stroke(); });
        yt.forEach(v => { const Y = M.Y(v); if (Y < r.y - 0.5 || Y > r.y + r.h + 0.5) return; g.beginPath(); g.moveTo(r.x, Y); g.lineTo(r.x + r.w, Y); g.stroke(); });
        g.restore();
        let lastXr = -1e9;
        xt.forEach(v => { const X = M.X(v); if (X < r.x - 0.5 || X > r.x + r.w + 0.5) return; const s = tickFmt(v); const m = lib.measure(g, s, { size: 11, kind: 'mono' }); if (X - m.w / 2 < lastXr + 6) return; lastXr = X + m.w / 2; T(g, s, X, r.y + r.h + 6, { color: C.muted, align: 'center' }); });
        yt.forEach(v => { const Y = M.Y(v); if (Y < r.y - 0.5 || Y > r.y + r.h + 0.5) return; T(g, tickFmt(v), r.x - 6, Y, { color: C.muted, align: 'right', baseline: 'middle' }); });
        poly(g, [[r.x, r.y], [r.x, r.y + r.h], [r.x + r.w, r.y + r.h]], C.ink, { w: 1 });
        T(g, cleanLab(ax.p.x.label) + (logX ? ' · log' : ''), r.x + r.w / 2, r.y + r.h + 24, { size: 11.5, align: 'center', maxWidth: r.w });
        g.save(); g.translate(13, r.y + r.h / 2); g.rotate(-Math.PI / 2); T(g, cleanLab(ax.p.y.label) + (logY ? ' · log' : ''), 0, 0, { size: 11.5, align: 'center', baseline: 'middle' }); g.restore();

        g.save(); g.beginPath(); g.rect(r.x + 1, r.y - 4, r.w + 4, r.h + 4); g.clip();
        if (showCurves) [[R.A, C.ink, 0.45], [R.B, C.blue, 0.55]].forEach(([s, col, al]) => { if (s.paper_curve && s.paper_curve.points) poly(g, s.paper_curve.points.map(q => [M.X(q[0]), M.Y(q[1])]), col, { w: 1.2, alpha: al }); });
        if (showFits) {
          [[R.fA, R.A, C.ink, 1.4], [R.fB, R.B, C.blue, 2]].forEach(([f, s, col, w]) => {
            if (!f) return; const xs = s.points.map(q => q[0]); const a = L10(Math.min(...xs)), b = L10(Math.max(...xs)), e = L10(M.xmaxV);
            const seg = (u0, u1, n) => { const out = []; for (let i = 0; i <= n; i++) { const u = u0 + (u1 - u0) * i / n; out.push([M.X(P10(u)), M.Y(fitY(f, P10(u)))]); } return out; };
            poly(g, seg(a, b, 80), col, { w });
            if (e > b + 1e-6) poly(g, seg(b, e, 80), col, { w, dash: [6, 5] });
          });
        }
        // FLOPs: same model, shifted right by the EBT cost factor
        if (S.pairs && ax.key === 'flops') {
          R.A.points.forEach((q, i) => { const e2 = R.B.points[i]; if (!e2) return; lib.arrow(g, M.X(q[0]) + 5, M.Y(q[1]), M.X(e2[0]) - 6, M.Y(e2[1]), { color: C.muted, width: 1, head: 6, dash: [3, 3] }); });
          const q = R.A.points[1], e2 = R.B.points[1], ratio = R.B.points.reduce((s, p, i) => s + p[0] / R.A.points[i][0], 0) / R.B.points.length;
          if (q && e2) T(g, 'same model, ×' + ratio.toFixed(2) + ' FLOPs', (M.X(q[0]) + M.X(e2[0])) / 2, Math.min(M.Y(q[1]), M.Y(e2[1])) - 16, { align: 'center' });
        }
        // crossing of the fits
        if (showFits && isFinite(R.uc) && logX && Math.abs(R.gain) >= 0.005 && R.fA) {
          const xc = P10(R.uc), X = M.X(xc);
          if (X > r.x + 4 && X < r.x + r.w - 4) {
            const Y = M.Y(fitY(R.fA, xc));
            if (Y > r.y && Y < r.y + r.h) {
              ring(g, X, Y, C.ink);
              const s = 'fits cross ≈ ' + KIND[ax.kind](xc), tw = lib.measure(g, s, { size: 11, kind: 'mono' }).w;
              T(g, s, Math.max(r.x + 4 + tw, X - 12), Y + 12, { align: 'right' });
            }
          }
        }
        // slope triangles
        if (S.slopes && showFits && S.tx > 0.95 && S.ty > 0.95) {
          [[R.fA, R.A, C.ink, 0.1, 0.42], [R.fB, R.B, C.blue, 0.55, 0.88]].forEach(([f, s, col, p0, p1]) => {
            if (!f) return; const xs = s.points.map(q => q[0]); const a = L10(Math.min(...xs)), b = L10(Math.max(...xs));
            const x0 = P10(a + (b - a) * p0), x1 = P10(a + (b - a) * p1);
            const X0 = M.X(x0), X1 = M.X(x1), Y0 = M.Y(fitY(f, x0)), Y1 = M.Y(fitY(f, x1));
            poly(g, [[X0, Y0], [X1, Y0], [X1, Y1]], col, { w: 1.2, dash: [4, 3] });
            const sl = 'slope ' + neg(f.m.toFixed(3)), tw = lib.measure(g, sl, { size: 11, kind: 'mono', weight: 600 }).w;
            T(g, sl, X1 + 6 + tw < r.x + r.w ? X1 + 6 : X1 - 6 - tw, (Y0 + Y1) / 2, { color: col, weight: 600, baseline: 'middle' });
          });
        }
        // points
        const drop = dropOf(ax.key), unit = isLoss(ax) ? 'loss' : 'ppl';
        [[R.A, 'Transformer++'], [R.B, 'EBT']].forEach(([s, nm], si) => {
          s.points.forEach((q, i) => {
            const X = M.X(q[0]), Y = M.Y(q[1]), off = drop.has(i);
            HITS.push({ x: X, y: Y, i, lines: [`${nm} · ${KIND[ax.kind](q[0])}`, `${unit} ${num(q[1], 4)} · approx.`, off ? 'dropped · click to restore' : 'click to drop this run'] });
            if (off) { g.save(); g.strokeStyle = C.faint; g.lineWidth = 1; g.beginPath(); g.arc(X, Y, 4, 0, 7); g.stroke(); g.beginPath(); g.moveTo(X - 3, Y - 3); g.lineTo(X + 3, Y + 3); g.moveTo(X + 3, Y - 3); g.lineTo(X - 3, Y + 3); g.stroke(); g.restore(); return; }
            if (si === 0) lib.dot(g, X, Y, 3.6, '#ffffff', { stroke: C.ink, lw: 1.3 });
            else lib.dot(g, X, Y, 3.8, C.blue, { stroke: '#ffffff', lw: 1 });
          });
        });
        g.restore();
        T(g, '○ Transformer++', r.x + r.w - 8, r.y + 4, { align: 'right' });
        T(g, '● EBT', r.x + r.w - 8, r.y + 20, { align: 'right', color: C.blue });
      }
      function readLearn() {
        const ax = R.ax, pr = ax.p.rate_gain_printed != null ? ax.p.rate_gain_printed : ax.p.rate_gain;
        if (!R.fA || !R.fB) return ['fit needs at least 2 runs per model'];
        const nd = dropOf(ax.key).size, L = R.lead, gp = L.b / L.a - 1, out = [];
        out.push(`slopes T++ <b>${neg(R.fA.m.toFixed(3))}</b> · EBT <b>${neg(R.fB.m.toFixed(3))}</b>`);
        out.push(`EBT steeper by <b>${pct(R.gain)}</b> (printed ${(pr * 100).toFixed(2)}%${nd ? `, ${nd} dropped` : ''})`);
        out.push(`largest run: EBT <b>${gp >= 0 ? '+' : MINUS}${Math.abs(gp * 100).toFixed(1)}%</b> ${isLoss(ax) ? 'loss' : 'perplexity'} vs T++`);
        if (isFinite(R.uc) && Math.abs(R.gain) >= 0.005) {
          const lo = L10(R.xmin), hi = L10(R.xmax), xc = P10(R.uc);
          out.push(R.uc < lo ? 'fits cross below the measured range' : R.uc <= hi ? `fits cross inside the data, ≈${KIND[ax.kind](xc)}` : `fits cross ≈${KIND[ax.kind](xc)}, <b>×${num(xc / R.xmax, 3)}</b> past the largest run`);
        }
        if (nd) out.push('<a href="javascript:void 0" data-restore="1">restore all runs</a>');
        return out;
      }
      ro.addEventListener('click', (ev) => { if (ev.target && ev.target.dataset && ev.target.dataset.restore) { dropOf(S.axis).clear(); refresh(); } });

      // ---------- thinking view: Fig 6a ----------
      function bestThink() { const E = ser('fig6a', 1); let k = 0; E.forEach((q, i) => { if (q[0] <= S.bThink + 1e-9) k = i; }); return k; }
      function drawThink(g) {
        const Tp = ser('fig6a', 0), E = ser('fig6a', 1), labs = P.fig6a.series[1].point_labels || [];
        const M = axesBox(g, { xlim: [0, 32], ylim: [29.5, 46.5], xticks: [2, 3, 6, 15, 30], yticks: [30, 34, 38, 42, 46], xlabel: 'forward passes per token', ylabel: 'perplexity increase, OOD text (↓)' });
        const r = M.rect, k = bestThink(), yT = Tp[0][1];
        g.save(); g.fillStyle = 'rgba(47,60,255,0.05)'; g.fillRect(r.x, M.Y(yT), r.w, r.y + r.h - M.Y(yT)); g.restore();
        T(g, r.w < 400 ? '↓ better than T++' : 'below this line: better than Transformer++', r.x + r.w - 6, M.Y(yT) + 6, { align: 'right', color: C.muted, size: 10.5 });
        budgetLine(g, M, S.bThink, 'budget ' + S.bThink);
        poly(g, [[M.X(Tp[0][0]), M.Y(yT)], [M.X(Tp[Tp.length - 1][0]), M.Y(yT)]], C.ink, { w: 1.4 });
        Tp.forEach(q => { marker(g, 'o', M.X(q[0]), M.Y(q[1]), 3.6, '#fff', C.ink); HITS.push({ x: M.X(q[0]), y: M.Y(q[1]), lines: ['Transformer++ · ' + q[0] + ' passes', num(q[1], 4) + ' (approx.)', 'same prediction however many passes'] }); });
        T(g, 'Transformer++ ≈' + yT.toFixed(1), M.X(15), M.Y(yT) - 16, { align: 'center' });
        const pts = E.map(q => [M.X(q[0]), M.Y(q[1])]);
        poly(g, pts.slice(0, k + 1), C.blue, { w: 2 });
        if (k < E.length - 1) poly(g, pts.slice(k), C.blue, { w: 1.2, dash: [4, 4], alpha: 0.45 });
        E.forEach((q, i) => {
          const on = i <= k;
          marker(g, i === 0 ? 'o' : i === 1 ? 'star' : 'tri', pts[i][0], pts[i][1], 4.2, on ? C.blue : '#fff', on ? '#fff' : C.blue2);
          HITS.push({ x: pts[i][0], y: pts[i][1], lines: [(labs[i] || 'EBT') + ' · ' + q[0] + ' passes', num(q[1], 4) + ' (approx.)'] });
        });
        const q = E[k], X = pts[k][0], Y = pts[k][1];
        ring(g, X, Y, C.ink);
        const short = ['no thinking', 'thinking longer', 'self-verification'][Math.min(2, k)];
        const right = X < r.x + r.w * 0.6, lx = X + (right ? 12 : -12), al = right ? 'left' : 'right', ly0 = k >= 2 ? Y - 40 : Y + 8;
        T(g, `EBT ${short} · ${q[1].toFixed(1)}`, lx, ly0, { align: al, color: C.blue, weight: 600 });
        if (k) T(g, `${MINUS}${((1 - q[1] / E[0][1]) * 100).toFixed(1)}% vs no thinking`, lx, ly0 + 14, { align: al, color: C.blue });
        if (k > 0) { const y0 = pts[0][1], bx = Math.max(pts[0][0] - 12, r.x + 4); poly(g, [[bx + 4, y0], [bx, y0], [bx, Y], [bx + 4, Y]], C.blue, { w: 1 }); poly(g, [[bx, Y], [X - 9, Y]], C.blue, { w: 0.8, dash: [2, 3], alpha: 0.6 }); }
        if (r.w >= 400) { let ly = r.y + 22; [['o', 'no thinking'], ['star', 'thinking longer'], ['tri', 'self-verification (best of N)']].forEach(([mk, t]) => { const tw = lib.measure(g, t, { size: 10.5, kind: 'mono' }).w; g.save(); g.fillStyle = 'rgba(255,255,255,0.92)'; g.fillRect(r.x + r.w - 8 - tw - 18, ly - 1, tw + 22, 15); g.restore(); marker(g, mk, r.x + r.w - 8 - tw - 10, ly + 6, 3.4, C.blue, null); T(g, t, r.x + r.w - 8, ly, { align: 'right', size: 10.5, color: C.muted }); ly += 15; }); }
      }
      function readThink() {
        const E = ser('fig6a', 1), Tp = ser('fig6a', 0), k = bestThink(), q = E[k];
        const vsNo = (1 - q[1] / E[0][1]) * 100, vsT = (1 - q[1] / Tp[0][1]) * 100;
        return [`budget <b>${S.bThink}</b> passes`, `best EBT <b>${q[1].toFixed(1)}</b>`, k ? `vs EBT no thinking <b>${vsNo.toFixed(1)}% lower</b>` : 'EBT with no extra thinking', `vs Transformer++ <b>${Math.abs(vsT).toFixed(1)}% ${vsT > 0 ? 'better' : 'worse'}</b>`];
      }

      // ---------- images view: Fig 12 + Table 4 ----------
      const f12 = P.fig12;
      const imgPts = () => ({ Dt: f12 ? f12.series[0].points : [[100, 14.31], [200, 18.86], [300, 19.57]], Eb: f12 ? f12.series[1].points : [[1, 13.92], [2, 18.82], [3, 23.29]] });
      const bestIn = (arr, b) => { let o = null; arr.forEach(q => { if (q[0] <= b + 1e-9 && (!o || q[1] > o[1])) o = q; }); return o; };
      function drawImg(g) {
        const { Dt, Eb } = imgPts();
        const M = axesBox(g, { xlog: true, xlim: [0.7, 450], ylim: [12, 25], xticks: [1, 2, 3, 10, 30, 100, 200, 300], yticks: [13, 15, 17, 19, 21, 23, 25], xlabel: 'forward passes, log', ylabel: 'PSNR on noisier images, dB (↑)' });
        const r = M.rect;
        budgetLine(g, M, clamp(S.bImg, 0.7, 450), 'budget ' + S.bImg);
        Eb.forEach((q, i) => { const d = Dt[i]; if (d) poly(g, [[M.X(q[0]), M.Y(q[1])], [M.X(d[0]), M.Y(d[1])]], C.faint, { w: 1, dash: [2, 4] }); });
        if (r.w >= 380) T(g, 'dotted: same number of re-applications (1, 2, 3)', r.x + 6, r.y + r.h - 16, { color: C.muted, size: 10.5 });
        poly(g, Dt.map(q => [M.X(q[0]), M.Y(q[1])]), C.ink, { w: 1.4 });
        poly(g, Eb.map(q => [M.X(q[0]), M.Y(q[1])]), C.blue, { w: 2 });
        Dt.forEach(q => { const on = q[0] <= S.bImg; marker(g, 'o', M.X(q[0]), M.Y(q[1]), 4.5, '#fff', on ? C.ink : C.faint); HITS.push({ x: M.X(q[0]), y: M.Y(q[1]), lines: [`DiT · ${q[0]} passes`, `${q[1].toFixed(2)} dB (approx.)`] }); });
        Eb.forEach(q => { const on = q[0] <= S.bImg; marker(g, 'o', M.X(q[0]), M.Y(q[1]), 5, on ? C.blue : '#fff', on ? '#fff' : C.blue2); HITS.push({ x: M.X(q[0]), y: M.Y(q[1]), lines: [`EBT · ${q[0]} pass${q[0] > 1 ? 'es' : ''}`, `${q[1].toFixed(2)} dB (approx.)`] }); });
        const bE = bestIn(Eb, S.bImg), bD = bestIn(Dt, S.bImg);
        if (bE) { ring(g, M.X(bE[0]), M.Y(bE[1]), C.blue); T(g, `EBT ${bE[1].toFixed(1)} dB`, M.X(bE[0]) + 12, M.Y(bE[1]) - 16, { color: C.blue, weight: 600 }); }
        if (bD) { ring(g, M.X(bD[0]), M.Y(bD[1]), C.ink); T(g, `DiT ${bD[1].toFixed(1)} dB`, M.X(bD[0]) - 12, M.Y(bD[1]) - 22, { align: 'right', weight: 600 }); }
        if (r.w >= 380) { const lgx = Math.max(r.x + 8, M.X(12)); T(g, '● EBT', lgx, r.y + 4, { color: C.blue }); T(g, '○ Diffusion Transformer (DiT)', lgx, r.y + 20); }
      }
      function readImg() {
        const { Dt, Eb } = imgPts(), bE = bestIn(Eb, S.bImg), bD = bestIn(Dt, S.bImg);
        return [`budget <b>${S.bImg}</b> passes`, `EBT ${bE ? `<b>${bE[1].toFixed(1)} dB</b>` : 'none yet'}`, `DiT ${bD ? `<b>${bD[1].toFixed(1)} dB</b>` : 'needs 100 passes'}`, 'Table 4: noisier <b>23.29</b> vs <b>19.56</b> dB · trained noise <b>27.25</b> vs <b>26.58</b>'];
      }

      // ---------- draw / readout ----------
      const LAB = { think: 'Fig 6a · thinking on out-of-distribution text', img: 'Fig 12 · denoising images noisier than in training' };
      function draw() {
        if (!cv.w) return;
        const g = cv.ctx; cv.clear(); HITS = [];
        if (S.view === 'learn') drawLearn(g); else if (S.view === 'think') drawThink(g); else drawImg(g);
        tooltip(g);
        const lines = S.view === 'learn' ? readLearn() : S.view === 'think' ? readThink() : readImg();
        ro.innerHTML = lines.map(s => `<span>${s}</span>`).join('');
        labEl.textContent = S.view === 'learn' ? `${R.ax.p.figure} · ${R.ax.group.toLowerCase()} · ${R.ax.p.printed_title || ''}` : LAB[S.view];
      }
      function refresh() { R = analyze(byKey[S.axis]); updateList(); draw(); }
      function caption() {
        ctx.setCaption(S.view === 'learn' ? 'Hollow ink circles: Transformer++. Blue dots: EBT. Values approx., read from ' + R.ax.p.figure + '. Hover a point to read it, click to drop it from the fits.'
          : S.view === 'think' ? 'Mean over four out-of-distribution text sets (approx., read from Fig 6a). The y axis formula is not given in the paper; lower is better.'
            : 'Noise level σ = 0.2, outside training (approx., read from Fig 12). Both models re-denoise their own output up to three times.');
      }
      function setView(v) {
        S.view = v; S.hover = null; viewSeg.set(v);
        cL.hidden = v !== 'learn'; AXF.wrap.hidden = v !== 'learn'; cT.hidden = v === 'learn';
        thS.el.hidden = v !== 'think'; imS.el.hidden = v !== 'img';
        caption(); refresh();
      }
      function selectAxis(k) {
        if (!byKey[k]) return; S.axis = k; S.hover = null;
        const p = byKey[k].p; if (S.mode === 'paper') { S.tx = S.ttx = p.x.log ? 1 : 0; S.ty = S.tty = p.y.log ? 1 : 0; }
        caption(); refresh();
      }
      const anim = lib.loop((dt) => {
        const k = Math.min(1, dt * 7);
        S.tx += (S.ttx - S.tx) * k; S.ty += (S.tty - S.ty) * k;
        const done = Math.abs(S.ttx - S.tx) < 0.004 && Math.abs(S.tty - S.ty) < 0.004;
        if (done) { S.tx = S.ttx; S.ty = S.tty; }
        draw(); return !done;
      });
      function setMode(m, instant) {
        if (S.ext > 0.01 || S.slopes) m = 'log';
        S.mode = m; modeSeg.set(m);
        const p = byKey[S.axis].p;
        S.ttx = m === 'log' ? 1 : (p.x.log ? 1 : 0); S.tty = m === 'log' ? 1 : (p.y.log ? 1 : 0);
        if (instant || lib.reducedMotion || !ctx.visible()) { S.tx = S.ttx; S.ty = S.tty; draw(); } else anim.start();
      }
      // budget sweep when a thinking step is entered
      let sw = null;
      const sweep = lib.loop((dt) => {
        if (!sw) return false; sw.t = Math.min(1, sw.t + dt / 1.8);
        const v = sw.from + (sw.to - sw.from) * sw.t;
        if (sw.kind === 'think') { S.bThink = Math.round(v); thS.set(S.bThink); } else { S.bImg = Math.max(1, Math.round(P10(v))); imS.set(L10(S.bImg)); }
        draw(); if (sw.t >= 1) { sw = null; return false; } return true;
      });
      function startSweep(kind) {
        const from = kind === 'think' ? 2 : 0, to = kind === 'think' ? 30 : L10(300);
        if (lib.reducedMotion || !ctx.visible()) { if (kind === 'think') { S.bThink = 30; thS.set(30); } else { S.bImg = 300; imS.set(L10(300)); } draw(); return; }
        sw = { kind, from, to, t: 0 }; sweep.start();
      }
      // hover / click
      function nearest(ev) { const [px, py] = cv.local(ev); let best = null, bd = 14; HITS.forEach(t => { const d = Math.hypot(t.x - px, t.y - py); if (d < bd) { bd = d; best = t; } }); return best; }
      cv.canvas.addEventListener('mousemove', (ev) => { const n = nearest(ev); const same = (n && S.hover && n.x === S.hover.x && n.y === S.hover.y) || (!n && !S.hover); S.hover = n; cv.canvas.style.cursor = n && n.i != null ? 'pointer' : 'default'; if (!same) draw(); });
      cv.canvas.addEventListener('mouseleave', () => { if (S.hover) { S.hover = null; draw(); } });
      cv.canvas.addEventListener('click', (ev) => {
        const n = nearest(ev); if (!n || n.i == null || S.view !== 'learn') return; const d = dropOf(S.axis);
        if (d.has(n.i)) d.delete(n.i); else if (R.n - d.size > 2) d.add(n.i);
        S.hover = null; refresh();
      });

      const CFG = [
        { view: 'learn', axis: 'data', slopes: true, pairs: false, ext: 0 },
        { view: 'learn', axis: 'params', slopes: false, pairs: false, ext: 3 },
        { view: 'learn', axis: 'flops', slopes: false, pairs: true, ext: 0 },
        { view: 'think' },
        { view: 'img' },
      ];
      function apply(c) {
        sweep.stop(); sw = null;
        Object.keys(S.drop).forEach(k => S.drop[k].clear());
        if (c.view === 'learn') {
          S.slopes = c.slopes; bSlopes.setAttribute('aria-pressed', String(c.slopes));
          S.pairs = c.pairs; S.ext = c.ext; extS.set(c.ext);
          S.axis = c.axis; setView('learn'); selectAxis(c.axis);
          const p = byKey[c.axis].p; if (S.tx > 0.99 && S.ty > 0.99) { S.tx = p.x.log ? 1 : 0; S.ty = p.y.log ? 1 : 0; }
          setMode('log');
        } else { setView(c.view); startSweep(c.view); }
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
