/* Panel 20: does thinking actually help? (thinking scalability results)
   Data: lib.data('scaling') (Figs 6a, 6b, 7, B.1a, B.1b, digitized, approx.), lib.data('paper_heatmaps') (Fig 8, B.2),
   lib.data('text') (our char-level toy EBT: thinking curves and best-of-M). Bootstrap and leave-one-out refits are computed live. */
(function () {
  'use strict';
  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
  const L10 = Math.log10, P10 = (v) => Math.pow(10, v);
  const MINUS = '−';
  const neg = (s) => String(s).replace(/-/g, MINUS);
  const num = (v, sig = 3) => (v == null || !isFinite(v)) ? '–' : neg(String(+v.toPrecision(sig)));
  const sgn = (v, d = 1) => (v >= 0 ? '+' : MINUS) + Math.abs(v).toFixed(d);
  function linfit(pts) {
    const n = pts.length; if (n < 2) return null; let sx = 0, sy = 0, sxx = 0, sxy = 0, syy = 0;
    pts.forEach(([x, y]) => { sx += x; sy += y; sxx += x * x; sxy += x * y; syy += y * y; });
    const mx = sx / n, my = sy / n, vxx = sxx / n - mx * mx, vxy = sxy / n - mx * my, vyy = syy / n - my * my;
    if (!(vxx > 0)) return null; const m = vxy / vxx;
    return { m, b: my - m * mx, r2: vyy > 0 ? vxy * vxy / (vxx * vyy) : 1, n };
  }
  function linTicks(a, b, n) {
    const span = b - a; if (!(span > 0)) return [a];
    const s0 = span / n, mag = P10(Math.floor(L10(s0))), r = s0 / mag;
    const step = (r < 1.5 ? 1 : r < 3 ? 2 : r < 7 ? 5 : 10) * mag; const out = [];
    for (let v = Math.ceil(a / step) * step; v <= b + 1e-9 * span; v += step) out.push(+v.toPrecision(12));
    return out;
  }
  const pctile = (arr, q) => { const v = arr.slice().sort((a, b) => a - b); const i = clamp((v.length - 1) * q, 0, v.length - 1), lo = Math.floor(i), hi = Math.ceil(i); return v[lo] + (v[hi] - v[lo]) * (i - lo); };
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
    id: 'thinking-results',
    nav: 'Does thinking help?',
    title: 'Does thinking actually help?',
    lede: 'At inference an EBT can spend more forward passes on a single token: more descent steps, or several candidates judged by its own energy. The paper reports gains of up to 29%. Here is exactly what that number measures, how the gains change with training and with distribution shift, and how much of each trend is noise.',
    text: `
      <p>Two knobs, both applied to every single prediction (Sec 3.3, Alg. 2, p.7–8):</p>
      <ul><li><b>Thinking longer</b>: more steps of $\\hat y_{i+1}=\\hat y_i-\\alpha\\nabla_{\\hat y}E_\\theta(x,\\hat y_i)$.</li>
      <li><b>Self-verification</b>: run $M$ chains from different random $\\hat y_{0,j}$ and keep the one the model itself scores best, $\\hat y^*=\\arg\\min_j E_\\theta(x,\\hat y_{N,j})$. There is no reward model; the verifier is the energy.</li></ul>
      <p>Cost is counted in forward passes (NFEs), one per optimization step (p.8), so $M$ chains of $N$ steps cost $M\\!\\cdot\\!N$. A Transformer++ makes exactly one forward pass per token and has no such knob.</p>
      <p>All text results are teacher-forced next-token perplexities, $\\mathrm{PPL}=\\exp\\big(-\\tfrac1T\\sum_t\\log p(x_t\\mid x_{\\lt t})\\big)$, lower is better. Gains are reported as "% perplexity improvement" over a no-thinking or no-verification reference; the exact formula is not printed (the general definition is Def. C.1, p.30). The thinking experiments use one small model pair (xxs, 6.18M non-embedding parameters, batch 128, 1M steps ≈ 33B tokens, p.34).</p>`,
    steps: [
      { label: 'Transformer++ has no knob', html: '<p>Fig 6a (p.11) averages four downstream datasets that are out of distribution relative to RedPajamaV2 pretraining (GSM8K, SQuAD, BigBench Math QA, BigBench Dyck). Its y axis, "perplexity increase", is lower-is-better; the exact formula is not given. Transformer++ sits at ≈38.4 whatever the budget. EBT with no extra thinking (2 passes, the step count it was trained with) is at ≈44.8: <b>worse</b> than Transformer++.</p>' },
      { label: 'One extra step', html: '<p>Raise the budget to 3 passes: one more descent step drops EBT to ≈35.9, already below Transformer++. Training used 2–3 randomized steps (Table D.4), so this is still inside what the model practised.</p>' },
      { label: 'Several candidates, and the 29%', html: '<p>Points at 6, 15 and 30 passes add self-verification; $M=2,5,10$ candidates of 3 steps each fits the budgets [derived]. EBT reaches ≈31.8, which is $(44.8-31.8)/44.8\\approx29\\%$ below its <em>own</em> no-thinking point: that is the paper\'s 29%. Against Transformer++ the improvement is ≈17% [derived]. Returns fall off fast: going from 6 to 30 passes buys only ≈0.7.</p>' },
      { label: 'Does verification improve with training?', html: '<p>Fig 6b follows one model through training, on BigBench Dyck only; "We did not observe this trend in other benchmarks" (p.34). The fitted line rises from ≈7.9% to ≈10.6%. The text\'s "4%−8%" to "10%−14%" describes the spread of the dots. The faint lines are refits on bootstrap resamples of the 33 points, computed in your browser: the slope is positive in ≈99% of them, but its 95% range spans roughly 0.01 to 0.16 points per billion tokens.</p>' },
      { label: 'More candidates can hurt early', html: '<p>Fig B.1a (RedPajamaV2 validation) compares best-of-10 with best-of-2. At one early checkpoint ten candidates did worse than two, "likely because the EBT found an adversarial sample (a sample with low energy that is in fact not a good prediction)" (p.28). Searching harder exposes the verifier\'s mistakes. The gain grows to ≈2% by 33B tokens and the dips fade, which the authors read as verification becoming robust with scale.</p>' },
      { label: 'The 15T-token projection', html: '<p>Fig B.1b extends the Fig 6b line to 15T tokens, the Llama 3 data scale. The star (≈1330%) is that straight line evaluated ≈460× past the last checkpoint: $7.78+0.0882\\times15{,}000\\approx1330$ [derived]. With the bootstrap slopes, the same projection ranges from about 200% to 2300%. A gain above 100% only makes sense as a ratio, which the paper does not define. Treat it as an extrapolation, not a result.</p>' },
      { label: 'Further out of distribution, bigger gains', html: '<p>Fig 7 (p.11): five datasets, the pretraining data at shift 1.0 and four downstream sets, with shift = downstream perplexity ÷ pretraining perplexity. The gain from max thinking (longer + verification) rises ≈3.3 points per unit of shift. Click a dot to drop it and refit: the slope stays between ≈3.1 and ≈3.7. Which dot is which dataset is not stated, and the x values do not match Table 3\'s ratios [derived], so we leave them unlabeled.</p>' },
      { label: 'Our toy: same mechanism, smaller effect', html: '<p>Our char-level text EBT (toy, not the paper) gains from thinking longer in distribution: loss 2.331 → 2.250 nats/char from 3 to 16 steps (paired 95% CI of the gain 0.063 to 0.093 nats). Best-of-8 by energy reaches 2.250 vs 2.329 for one candidate, while the average candidate stays at 2.330. But on Python code thinking past 5 steps <em>hurts</em>, the opposite of Fig 7, and a same-size feed-forward model beats the toy EBT outright (1.599).</p>' },
    ],
    after: `
      <h3>Which regularizers make thinking work</h3>
      <p>Table 2 (p.10, tab <b>Table 2</b>) ablates the S2 recipe on BigBench Dyck. Without a randomized step size, thinking longer gives −1.47% and verification 0.19%: the gains nearly vanish. Without a randomized number of steps, thinking longer gives exactly 0.00%. Without Langevin noise, thinking longer is best (17.2%) but the combined score drops (17.0 vs 18.7): less exploration helps a single path and hurts the diversity that verification needs. The full configuration wins only in the combined column.</p>
      <h3>Which tokens stay uncertain</h3>
      <p>Fig 8 (p.12, tab <b>8</b>) shows per-token energies over 12 iterations. Almost the whole drop happens at the first update; what differs is where tokens settle. Easy tokens (".", "is", "but") end low, hard ones ("quick", "research", "problem") stay high. See the <a href="#tokens">tokens</a> and <a href="#uncertainty">uncertainty</a> panels for what this does and does not show.</p>
      <h3>How much weight to put on these numbers</h3>
      <p>One xxs model pair, one seed (33), no error bars, perplexity only, no generated text. Fig 6a, Fig 7 and Table 3 cannot be reconciled from their printed values (paper_facts), so their numbers should not be mixed. The paper also notes that these small models did not benefit from chain-of-thought (footnote 9, p.10): "thinking" here is iterative refinement of one continuous prediction, not a written reasoning trace.</p>
      <p class="note">Paper curves are digitized from the PDF (approx., read from Figs 6a, 6b, 7, B.1a, B.1b, 8, B.2). Bootstrap, leave-one-out and cursor values are computed live from those points. The toy tab uses data/text.json, a 0.27M-parameter character-level EBT trained for this explainer.</p>`,
    source: [{ kind: 'paper', note: 'Figs 6a, 6b, 7, B.1, 8, B.2, Table 2 (digitized, approx.)' }, { kind: 'ext', note: 'bootstrap and leave-one-out refits are ours' }, { kind: 'toy', note: 'char-level text EBT (toy tab)' }],
    figure(stage, ctx) {
      const { lib } = ctx, h = lib.h, C = lib.C;
      const SD = lib.data('scaling'), HM = lib.data('paper_heatmaps'), TX = lib.data('text');
      if (!SD) { stage.appendChild(h('p', { class: 'callout warn' }, 'Data missing (data/scaling.json).')); return {}; }
      const P = {}; SD.plots.forEach(p => { P[p.id] = p; });
      const ser = (id, i = 0) => (P[id] && P[id].series[i]) ? P[id].series[i].points : [];
      const S = { tab: '6a', budget: 2, cur6b: 32.65, curB1a: 32.65, ext: L10(1.5e13), boot: true, seed: 3, drop7: new Set(), t2: 4, hm: 'fig8a', hmIt: 1, toyDs: 'val', toyN: 16, toyBonN: 0, hover: null };
      // bootstrap cache
      const bootCache = {};
      function boot(id, seed) {
        const key = id + ':' + seed; if (bootCache[key]) return bootCache[key];
        const pts = ser(id), n = pts.length, r = lib.rng(seed * 7919 + 13), fits = [];
        for (let k = 0; k < 400; k++) { const q = []; for (let i = 0; i < n; i++) q.push(pts[Math.floor(r() * n)]); const f = linfit(q); if (f) fits.push(f); }
        return (bootCache[key] = fits);
      }
      const F = { '6b': linfit(ser('fig6b')), b1a: linfit(ser('figB1a')) };

      // ---------- layout ----------
      const tabs = [['6a', '6a · passes'], ['6b', '6b · BoN-5 vs training'], ['b1a', 'B.1a · BoN-10 vs 2'], ['b1b', 'B.1b · to 15T'], ['7', '7 · OOD shift'], ['t2', 'Table 2'], ['8', '8 · token energies'], ['toy', 'toy · text EBT']];
      const tabRow = h('div', { class: 'controls tr-tabs' }); stage.appendChild(tabRow);
      tabRow.appendChild(h('span', { class: 'fig-label' }, 'exhibit'));
      const tabSeg = lib.segmented({ label: 'Exhibit', options: tabs, value: S.tab, onchange: (v) => setTab(v) }); tabRow.appendChild(tabSeg.el);
      const fr = lib.frame(stage, { label: 'Fig 6a', sub: '&nbsp;' });
      const labEl = fr.wrap.querySelector('.fig-label'), subEl = fr.wrap.querySelector('.fig-sub');
      const cv = rcanvas(lib, fr.frame, { aspect: (w) => clamp(Math.round(w * 0.58), 300, 380), label: 'Thinking results chart', draw: () => draw() });
      const tctl = h('div', { class: 'controls' }); stage.appendChild(tctl);
      const ro = h('div', { class: 'readout tr-ro', 'aria-live': 'polite' }); stage.appendChild(ro);

      // ---------- plotting helpers ----------
      let HITS = [];
      function axesBox(g, o) {
        const W = cv.w, H = cv.h, rect = { x: o.left || 56, y: o.top || 16, w: W - (o.left || 56) - (o.right || 14), h: H - (o.top || 16) - (o.bottom || 44) };
        const tx = (v) => o.xlog ? L10(v) : v, ty = (v) => o.ylog ? L10(v) : v;
        const [xa, xb] = o.xlim.map(tx), [ya, yb] = o.ylim.map(ty);
        const X = (v) => rect.x + (tx(v) - xa) / (xb - xa) * rect.w, Y = (v) => rect.y + rect.h - (ty(v) - ya) / (yb - ya) * rect.h;
        g.save(); g.strokeStyle = C.rule; g.lineWidth = 1;
        (o.xticks || []).forEach(v => { g.beginPath(); g.moveTo(X(v), rect.y); g.lineTo(X(v), rect.y + rect.h); g.stroke(); });
        (o.yticks || []).forEach(v => { g.beginPath(); g.moveTo(rect.x, Y(v)); g.lineTo(rect.x + rect.w, Y(v)); g.stroke(); });
        g.strokeStyle = C.ink; g.beginPath(); g.moveTo(rect.x, rect.y); g.lineTo(rect.x, rect.y + rect.h); g.lineTo(rect.x + rect.w, rect.y + rect.h); g.stroke(); g.restore();
        const xf = o.xfmt || ((v) => num(v)), yf = o.yfmt || ((v) => num(v));
        let lastR = -1e9;
        (o.xticks || []).forEach(v => { const s = xf(v), w = lib.measure(g, s, { size: 11, kind: 'mono' }).w, x = X(v); if (x - w / 2 < lastR + 4) return; lastR = x + w / 2; lib.text(g, s, x, rect.y + rect.h + 6, { size: 11, kind: 'mono', color: C.muted, align: 'center' }); });
        (o.yticks || []).forEach(v => lib.text(g, yf(v), rect.x - 6, Y(v), { size: 11, kind: 'mono', color: C.muted, align: 'right', baseline: 'middle' }));
        if (o.xlabel) lib.text(g, o.xlabel, rect.x + rect.w / 2, rect.y + rect.h + 24, { size: 11.5, kind: 'mono', color: C.ink, align: 'center', maxWidth: rect.w });
        if (o.ylabel) { g.save(); g.translate(13, rect.y + rect.h / 2); g.rotate(-Math.PI / 2); lib.text(g, o.ylabel, 0, 0, { size: 11.5, kind: 'mono', color: C.ink, align: 'center', baseline: 'middle', maxWidth: rect.h }); g.restore(); }
        return { rect, X, Y };
      }
      function poly(g, pts, col, o = {}) { if (pts.length < 2) return; g.save(); g.strokeStyle = col; g.lineWidth = o.w || 1.5; g.globalAlpha *= (o.alpha == null ? 1 : o.alpha); g.lineJoin = 'round'; g.lineCap = 'round'; if (o.dash) g.setLineDash(o.dash); g.beginPath(); g.moveTo(pts[0][0], pts[0][1]); for (let i = 1; i < pts.length; i++) g.lineTo(pts[i][0], pts[i][1]); g.stroke(); g.restore(); }
      function marker(g, kind, x, y, r, fill, stroke) {
        g.save(); g.beginPath();
        if (kind === 'star') { for (let i = 0; i < 10; i++) { const a = -Math.PI / 2 + i * Math.PI / 5, rr = i % 2 ? r * 0.45 : r * 1.25; g.lineTo(x + rr * Math.cos(a), y + rr * Math.sin(a)); } g.closePath(); }
        else if (kind === 'tri') { g.moveTo(x, y - r * 1.2); g.lineTo(x + r * 1.1, y + r * 0.8); g.lineTo(x - r * 1.1, y + r * 0.8); g.closePath(); }
        else g.arc(x, y, r, 0, Math.PI * 2);
        if (fill) { g.fillStyle = fill; g.fill(); } if (stroke) { g.strokeStyle = stroke; g.lineWidth = 1.2; g.stroke(); } g.restore();
      }
      const T = (g, s, x, y, o = {}) => lib.text(g, s, x, y, Object.assign({ size: 11, kind: 'mono', color: C.ink }, o));
      const hit = (x, y, lines) => HITS.push({ x, y, lines });
      function tooltip(g) {
        if (!S.hover) return; const { x, y, lines } = S.hover;
        const tw = Math.max(...lines.map(t => lib.measure(g, t, { size: 11, kind: 'mono' }).w)) + 14, th = lines.length * 14 + 10;
        let bx = x + 12, by = y - th - 6; if (bx + tw > cv.w - 4) bx = x - 12 - tw; if (by < 2) by = y + 12;
        g.save(); g.fillStyle = 'rgba(255,255,255,0.96)'; g.fillRect(bx, by, tw, th); g.strokeStyle = C.ink; g.lineWidth = 1; g.strokeRect(bx + 0.5, by + 0.5, tw - 1, th - 1); g.beginPath(); g.arc(x, y, 7, 0, 7); g.stroke(); g.restore();
        lines.forEach((t, i) => T(g, t, bx + 7, by + 6 + i * 14, { color: i === 0 ? C.ink : C.muted }));
      }
      function hatch(g) { const c = document.createElement('canvas'); c.width = c.height = 8; const q = c.getContext('2d'); q.strokeStyle = 'rgba(47,60,255,0.16)'; q.lineWidth = 1; q.beginPath(); q.moveTo(-1, 9); q.lineTo(9, -1); q.stroke(); return g.createPattern(c, 'repeat'); }
      let hatchPat = null;

      // ---------- tabs ----------
      const TABS = {};
      // Fig 6a
      TABS['6a'] = {
        label: 'Fig 6a · OOD thinking performance', sub: 'mean over four OOD text datasets · approx., read from Fig 6a', src: 'paper',
        controls(el) {
          const sl = lib.slider({ id: 'tr-budget', label: 'forward-pass budget per token', min: 2, max: 30, step: 1, value: S.budget, fmt: v => v + ' passes', oninput: v => { stopSweep(); S.budget = v; draw(); readout(); } });
          S._budgetSl = sl; el.appendChild(sl.el);
          el.appendChild(lib.button('sweep 2 → 30', () => { sweep(2, 30); }));
        },
        best() { const E = ser('fig6a', 1); let k = 0; E.forEach((q, i) => { if (q[0] <= S.budget + 1e-9) k = i; }); return k; },
        draw(g) {
          const Tp = ser('fig6a', 0), E = ser('fig6a', 1), labs = P.fig6a.series[1].point_labels || [];
          const M = axesBox(g, { xlim: [0, 32], ylim: [29.5, 46.5], xticks: [2, 3, 6, 15, 30], yticks: [30, 34, 38, 42, 46], xlabel: 'forward passes per token', ylabel: 'perplexity increase on OOD data (↓)' });
          const r = M.rect, k = this.best(), yT = Tp[0][1];
          // EBT-better region
          g.save(); g.fillStyle = 'rgba(47,60,255,0.05)'; g.fillRect(r.x, M.Y(yT), r.w, r.y + r.h - M.Y(yT)); g.restore();
          T(g, 'below this line: better than Transformer++', r.x + r.w - 6, M.Y(yT) + 6, { align: 'right', color: C.muted, size: 10.5 });
          // budget line
          const xb = M.X(S.budget); poly(g, [[xb, r.y], [xb, r.y + r.h]], C.ink, { w: 1, dash: [3, 4], alpha: 0.6 });
          const bl = 'budget ' + S.budget, bw = lib.measure(g, bl, { size: 11, kind: 'mono' }).w;
          T(g, bl, xb + 5 + bw > r.x + r.w ? xb - 5 : xb + 5, r.y + 2, { align: xb + 5 + bw > r.x + r.w ? 'right' : 'left' });
          // Transformer++
          poly(g, [[M.X(Tp[0][0]), M.Y(yT)], [M.X(Tp[Tp.length - 1][0]), M.Y(yT)]], C.ink, { w: 1.4 });
          Tp.forEach(q => { marker(g, 'o', M.X(q[0]), M.Y(q[1]), 3.6, '#fff', C.ink); hit(M.X(q[0]), M.Y(q[1]), ['Transformer++ · ' + q[0] + ' passes', num(q[1], 4) + ' (approx.)', 'same prediction however many passes']); });
          T(g, 'Transformer++ ≈' + yT.toFixed(1), M.X(15), M.Y(yT) - 16, { align: 'center' });
          // EBT
          const pts = E.map(q => [M.X(q[0]), M.Y(q[1])]);
          poly(g, pts.slice(0, k + 1), C.blue, { w: 2 });
          if (k < E.length - 1) poly(g, pts.slice(k), C.blue, { w: 1.2, dash: [4, 4], alpha: 0.45 });
          E.forEach((q, i) => {
            const kind = i === 0 ? 'o' : i === 1 ? 'star' : 'tri', on = i <= k;
            marker(g, kind, pts[i][0], pts[i][1], 4.2, on ? C.blue : '#fff', on ? '#fff' : C.blue2);
            hit(pts[i][0], pts[i][1], [(labs[i] || 'EBT') + ' · ' + q[0] + ' passes', num(q[1], 4) + ' (approx.)', i ? `${sgn(-(1 - q[1] / E[0][1]) * 100)}% vs no thinking` : 'trained with 2 steps']);
          });
          // bracket from no-thinking to current best
          const q = E[k], X = pts[k][0], Y = pts[k][1];
          g.save(); g.strokeStyle = C.ink; g.lineWidth = 1.2; g.beginPath(); g.arc(X, Y, 8, 0, 7); g.stroke(); g.restore();
          const short = ['no thinking', 'thinking longer', 'self-verification'][Math.min(2, k)];
          const right = X < r.x + r.w * 0.6;
          T(g, `EBT ${short} · ${q[1].toFixed(1)}`, X + (right ? 12 : -12), Y + (k ? 10 : -20), { align: right ? 'left' : 'right', color: C.blue, weight: 600 });
          if (k > 0) {
            const y0 = pts[0][1], bx = Math.max(pts[0][0] - 14, r.x + 4);
            poly(g, [[bx + 4, y0], [bx, y0], [bx, Y], [bx + 4, Y]], C.blue, { w: 1 });
            poly(g, [[bx, Y], [X - 9, Y]], C.blue, { w: 0.8, dash: [2, 3], alpha: 0.6 });
            T(g, sgn(-(1 - q[1] / E[0][1]) * 100) + '%', bx + 6, (y0 + Y) / 2, { color: C.blue, baseline: 'middle', weight: 600 });
          }
          // legend
          const lx = r.x + r.w - 8; let ly = r.y + 18;
          [['o', 'no thinking'], ['star', 'thinking longer'], ['tri', 'self-verification (BoN)']].forEach(([mk, t]) => { const tw = lib.measure(g, t, { size: 10.5, kind: 'mono' }).w; marker(g, mk, lx - tw - 10, ly + 6, 3.4, C.blue, null); T(g, t, lx, ly, { align: 'right', size: 10.5, color: C.muted }); ly += 15; });
        },
        readout() {
          const E = ser('fig6a', 1), Tp = ser('fig6a', 0), k = this.best(), q = E[k], labs = P.fig6a.series[1].point_labels || [];
          const vsNo = (1 - q[1] / E[0][1]) * 100, vsT = (1 - q[1] / Tp[0][1]) * 100;
          return [`budget <b>${S.budget}</b> passes`, `best EBT <b>${q[1].toFixed(2)}</b> (${(labs[k] || '').replace('EBT ', '')}, ${q[0]} passes)`, `vs EBT no thinking <b>${sgn(-vsNo)}%</b>`, `vs Transformer++ <b>${sgn(-vsT)}%</b> (${vsT > 0 ? 'better' : 'worse'})`, k ? `last step: ${sgn(q[1] - E[k - 1][1], 2)} for ${q[0] - E[k - 1][0]} more passes` : 'Transformer++ ≈' + Tp[0][1].toFixed(2) + ' at any budget'];
        },
      };
      // Fig 6b and B.1a share a scatter + bootstrap view
      function scatterTab(id, o) {
        return {
          label: o.label, sub: o.sub, src: 'paper',
          controls(el) {
            const sl = lib.slider({ id: 'tr-cur-' + id, label: 'tokens trained on', min: o.dom[0], max: o.dom[1], step: 0.01, value: S[o.curKey], fmt: v => v.toFixed(1) + 'B', oninput: v => { S[o.curKey] = v; draw(); readout(); } });
            el.appendChild(sl.el);
            const bb = lib.button('bootstrap fits', () => { S.boot = !S.boot; bb.setAttribute('aria-pressed', String(S.boot)); draw(); readout(); }); bb.setAttribute('aria-pressed', String(S.boot)); el.appendChild(bb);
            el.appendChild(lib.button('resample', () => { S.seed++; S.boot = true; bb.setAttribute('aria-pressed', 'true'); draw(); readout(); }));
          },
          draw(g) {
            const pts = ser(o.pid), f = F[o.fk], bs = boot(o.pid, S.seed);
            const M = axesBox(g, { xlim: o.xlim, ylim: o.ylim, xticks: linTicks(0, o.xlim[1], 6).filter(v => v >= o.xlim[0]), yticks: o.yticks, xfmt: v => num(v) + 'B', yfmt: v => num(v) + '%', xlabel: 'tokens trained on (billions)', ylabel: o.ylabel });
            const r = M.rect, x0 = o.dom[0], x1 = o.dom[1];
            if (o.zero) { poly(g, [[r.x, M.Y(0)], [r.x + r.w, M.Y(0)]], C.ink, { w: 1, alpha: 0.5 }); T(g, 'more candidates = worse below this line', r.x + 6, M.Y(0) + 4, { size: 10.5, color: C.muted }); }
            g.save(); g.beginPath(); g.rect(r.x, r.y, r.w, r.h); g.clip();
            if (S.boot) bs.slice(0, 80).forEach(b => poly(g, [[M.X(x0), M.Y(b.b + b.m * x0)], [M.X(x1), M.Y(b.b + b.m * x1)]], C.blue, { w: 1, alpha: 0.07 }));
            poly(g, [[M.X(x0), M.Y(f.b + f.m * x0)], [M.X(x1), M.Y(f.b + f.m * x1)]], C.blue, { w: 2 });
            g.restore();
            pts.forEach((q, i) => {
              const X = M.X(q[0]), Y = M.Y(q[1]), past = q[0] <= S[o.curKey] + 1e-9;
              const bad = o.zero && q[1] < 0;
              marker(g, 'o', X, Y, 3.6, past ? (bad ? C.bad : C.blue) : '#fff', past ? '#fff' : C.blue2);
              hit(X, Y, [`checkpoint at ${q[0].toFixed(1)}B tokens`, `${q[1].toFixed(2)}% (approx.)`, bad ? 'BoN-10 worse than BoN-2' : `${sgn(q[1] - (f.b + f.m * q[0]), 2)} from the fitted line`]);
            });
            if (o.zero) { const q = pts.find(p => p[1] < 0); if (q) T(g, 'adversarial sample? (p.28)', M.X(q[0]) + 9, M.Y(q[1]) - 4, { color: C.bad, size: 10.5 }); }
            // cursor + bootstrap interval
            const c = S[o.curKey], Xc = M.X(c), yc = f.b + f.m * c, ys = bs.map(b => b.b + b.m * c), lo = pctile(ys, 0.025), hi = pctile(ys, 0.975);
            poly(g, [[Xc, r.y], [Xc, r.y + r.h]], C.ink, { w: 1, dash: [3, 4], alpha: 0.6 });
            g.save(); g.strokeStyle = C.ink; g.lineWidth = 1.4; g.beginPath(); g.moveTo(Xc - 5, M.Y(lo)); g.lineTo(Xc + 5, M.Y(lo)); g.moveTo(Xc, M.Y(lo)); g.lineTo(Xc, M.Y(hi)); g.moveTo(Xc - 5, M.Y(hi)); g.lineTo(Xc + 5, M.Y(hi)); g.stroke(); g.restore();
            marker(g, 'o', Xc, M.Y(yc), 4.5, C.ink, '#fff');
            const lab = `fit ${yc.toFixed(1)}% [${lo.toFixed(1)}, ${hi.toFixed(1)}]`, tw = lib.measure(g, lab, { size: 11, kind: 'mono' }).w;
            const left = Xc + 10 + tw > r.x + r.w;
            T(g, lab, left ? Xc - 10 : Xc + 10, M.Y(hi) - 14, { align: left ? 'right' : 'left', weight: 600 });
            T(g, o.legend, r.x + 8, r.y + 4, { color: C.blue, size: 10.5 });
          },
          readout() {
            const f = F[o.fk], bs = boot(o.pid, S.seed), ms = bs.map(b => b.m), c = S[o.curKey];
            const pos = ms.filter(m => m > 0).length / ms.length;
            return [`slope <b>${sgn(f.m, 3)}</b> pts per 1B tokens`, `bootstrap 95%: [${pctile(ms, 0.025).toFixed(3)}, ${pctile(ms, 0.975).toFixed(3)}]`, `slope &gt; 0 in <b>${(pos * 100).toFixed(1)}%</b> of ${bs.length} resamples`, `r² ${f.r2.toFixed(2)} · ${ser(o.pid).length} checkpoints`, `line at ${c.toFixed(1)}B: <b>${(f.b + f.m * c).toFixed(2)}%</b>`];
          },
        };
      }
      TABS['6b'] = scatterTab('6b', { pid: 'fig6b', fk: '6b', curKey: 'cur6b', dom: [0.98, 32.65], xlim: [-0.6, 34.3], ylim: [3, 15.2], yticks: [4, 6, 8, 10, 12, 14], ylabel: '% PPL gain, BoN-5 vs none', label: 'Fig 6b · verification vs training', sub: 'BoN-5 vs no verification · BigBench Dyck only (p.34) · approx.', legend: '● BoN-5 checkpoints (Dyck)' });
      TABS.b1a = scatterTab('b1a', { pid: 'figB1a', fk: 'b1a', curKey: 'curB1a', dom: [2.95, 32.65], xlim: [1.4, 34.3], ylim: [-0.5, 2.9], yticks: [0, 0.5, 1, 1.5, 2, 2.5], ylabel: '% PPL gain, BoN-10 vs BoN-2', label: 'Fig B.1a · BoN-10 vs BoN-2', sub: 'RedPajamaV2 validation (p.34) · approx.', zero: true, legend: '● BoN-10 over BoN-2 checkpoints' });
      // Fig B.1b
      TABS.b1b = {
        label: 'Fig B.1b · projected to Llama 3 scale', sub: 'the Fig 6b line extended to 15T tokens · approx.', src: 'paper',
        controls(el) {
          const sl = lib.slider({ id: 'tr-ext', label: 'evaluate the line at', min: 10.5, max: L10(1.5e13), step: 0.01, value: S.ext, fmt: v => { const x = P10(v); return x >= 1e12 ? num(x / 1e12) + 'T tokens' : num(x / 1e9) + 'B tokens'; }, oninput: v => { S.ext = v; draw(); readout(); } });
          el.appendChild(sl.el);
          el.appendChild(lib.button('resample', () => { S.seed++; draw(); readout(); }));
        },
        draw(g) {
          if (!hatchPat) hatchPat = hatch(g);
          const pts = ser('figB1b'), f = F['6b'], bs = boot('fig6b', S.seed), last = pts[pts.length - 1][0];
          const M = axesBox(g, { xlim: [6e8, 2.4e13], ylim: [3, 3000], xlog: true, ylog: true, xticks: [1e9, 1e10, 1e11, 1e12, 1e13], yticks: [3, 10, 30, 100, 300, 1000, 3000], xfmt: v => v >= 1e12 ? num(v / 1e12) + 'T' : num(v / 1e9) + 'B', yfmt: v => num(v) + '%', xlabel: 'tokens trained on · log', ylabel: '% PPL gain over no verification · log' });
          const r = M.rect, xe = P10(S.ext);
          const xL = M.X(last); g.save(); g.fillStyle = hatchPat; g.fillRect(xL, r.y, r.x + r.w - xL, r.h); g.restore();
          poly(g, [[xL, r.y], [xL, r.y + r.h]], C.bad, { w: 1, dash: [3, 3] });
          T(g, 'EXTRAPOLATION · NO DATA', r.x + r.w - 6, r.y + 6, { align: 'right', color: C.bad, size: 10.5 });
          // bootstrap band
          const band = []; const N = 60;
          for (let i = 0; i <= N; i++) { const x = P10(9 + (L10(xe) - 9) * i / N), ys = bs.map(b => b.b + b.m * x / 1e9); band.push([x, Math.max(3, pctile(ys, 0.025)), pctile(ys, 0.975)]); }
          g.save(); g.beginPath(); g.rect(r.x, r.y, r.w, r.h); g.clip();
          g.fillStyle = 'rgba(47,60,255,0.10)'; g.beginPath(); band.forEach((b, i) => { const X = M.X(b[0]), Y = M.Y(b[2]); i ? g.lineTo(X, Y) : g.moveTo(X, Y); }); band.slice().reverse().forEach(b => g.lineTo(M.X(b[0]), M.Y(b[1]))); g.closePath(); g.fill();
          const lineAt = (x) => f.b + f.m * x / 1e9;
          const s1 = [], s2 = []; for (let i = 0; i <= 80; i++) { const x = P10(9 + (L10(last) - 9) * i / 80); s1.push([M.X(x), M.Y(lineAt(x))]); } for (let i = 0; i <= 80; i++) { const x = P10(L10(last) + (L10(xe) - L10(last)) * i / 80); s2.push([M.X(x), M.Y(lineAt(x))]); }
          poly(g, s1, C.blue, { w: 2 }); if (xe > last) poly(g, s2, C.blue, { w: 2, dash: [6, 5] });
          g.restore();
          pts.forEach(q => { marker(g, 'o', M.X(q[0]), M.Y(q[1]), 3, C.blue, '#fff'); });
          const ex = P.figB1b.series[0].extrapolated_point;
          if (ex) { marker(g, 'star', M.X(ex[0]), M.Y(ex[1]), 6, '#fff', C.ink); hit(M.X(ex[0]), M.Y(ex[1]), ['paper\'s projected point (star)', '≈' + num(ex[1], 4) + '% at 15T tokens', 'not a measurement']); T(g, 'paper\'s star ≈' + Math.round(ex[1]) + '%', M.X(ex[0]) - 12, M.Y(ex[1]) - 4, { align: 'right', baseline: 'middle' }); }
          const yv = lineAt(xe); marker(g, 'o', M.X(xe), M.Y(yv), 4.5, C.ink, '#fff');
          T(g, '33 checkpoints (Fig 6b)', M.X(2e9), M.Y(4.2), { color: C.blue, size: 10.5 });
          T(g, 'shaded: 95% of bootstrap lines', r.x + 8, r.y + 4, { color: C.muted, size: 10.5 });
        },
        readout() {
          const f = F['6b'], bs = boot('fig6b', S.seed), xe = P10(S.ext), last = ser('figB1b').slice(-1)[0][0];
          const ys = bs.map(b => b.b + b.m * xe / 1e9), y = f.b + f.m * xe / 1e9;
          return [`at ${xe >= 1e12 ? num(xe / 1e12) + 'T' : num(xe / 1e9) + 'B'} tokens: <b>${num(y, 4)}%</b>`, `bootstrap 95%: ${Math.round(pctile(ys, 0.025))}% to ${Math.round(pctile(ys, 0.975))}%`, xe > last ? `<span class="warn">×${num(xe / last, 3)} past the last checkpoint</span>` : 'inside the measured range', `= ${f.b.toFixed(2)} + ${f.m.toFixed(4)} × (tokens in B)`];
        },
      };
      // Fig 7
      TABS['7'] = {
        label: 'Fig 7 · thinking helps more on OOD data', sub: 'EBT with max thinking (longer + verification) · approx. · click a dot to drop it', src: 'paper',
        controls(el) { el.appendChild(lib.button('restore all', () => { S.drop7.clear(); draw(); readout(); })); el.appendChild(h('span', { class: 'mono', style: { color: 'var(--muted)' } }, 'faint lines: the five leave-one-out refits')); },
        fit() { return linfit(ser('fig7').filter((_, i) => !S.drop7.has(i))); },
        draw(g) {
          const pts = ser('fig7'), f = this.fit();
          const M = axesBox(g, { xlim: [0.7, 4.7], ylim: [9, 25], xticks: [1, 2, 3, 4], yticks: [10, 14, 18, 22], yfmt: v => v + '%', xlabel: 'OOD shift = downstream ppl ÷ pretraining ppl', ylabel: '% PPL gain from thinking' });
          const r = M.rect;
          pts.forEach((_, j) => { const fj = linfit(pts.filter((__, i) => i !== j)); poly(g, [[M.X(0.8), M.Y(fj.b + fj.m * 0.8)], [M.X(4.6), M.Y(fj.b + fj.m * 4.6)]], C.ink, { w: 1, alpha: 0.18 }); });
          if (f) poly(g, [[M.X(0.8), M.Y(f.b + f.m * 0.8)], [M.X(4.6), M.Y(f.b + f.m * 4.6)]], C.blue, { w: 2 });
          pts.forEach((q, i) => {
            const X = M.X(q[0]), Y = M.Y(q[1]), off = S.drop7.has(i);
            if (off) { marker(g, 'o', X, Y, 4.5, '#fff', C.faint); poly(g, [[X - 3, Y - 3], [X + 3, Y + 3]], C.faint, { w: 1 }); poly(g, [[X + 3, Y - 3], [X - 3, Y + 3]], C.faint, { w: 1 }); }
            else marker(g, 'o', X, Y, 4.5, C.blue, '#fff');
            hit(X, Y, [i === 0 ? 'pretraining data (shift 1.0)' : 'downstream dataset (not identified)', `shift ${q[0].toFixed(2)} · gain ${q[1].toFixed(1)}% (approx.)`, off ? 'dropped · click to restore' : 'click to drop and refit']);
          });
          T(g, 'pretraining data', M.X(1) + 8, M.Y(pts[0][1]) + 6, { color: C.muted, size: 10.5 });
          if (f) T(g, `slope ${f.m.toFixed(2)} pts per unit shift`, r.x + 8, r.y + 4, { color: C.blue, weight: 600 });
        },
        readout() {
          const pts = ser('fig7'), f = this.fit(); if (!f) return ['need at least 2 points'];
          const loo = pts.map((_, j) => linfit(pts.filter((__, i) => i !== j)).m);
          return [`slope <b>${f.m.toFixed(2)}</b> points per unit shift`, `r² ${f.r2.toFixed(3)} from ${f.n} datasets`, `at shift 1: ${(f.b + f.m).toFixed(1)}% · at 4: ${(f.b + 4 * f.m).toFixed(1)}%`, `leave-one-out slopes ${Math.min(...loo).toFixed(2)} to ${Math.max(...loo).toFixed(2)}`];
        },
        click(px, py) { const n = nearestHit(px, py); if (!n) return false; const pts = ser('fig7'); const i = pts.findIndex(q => Math.abs(this._M.X(q[0]) - n.x) < 0.5); return i; },
      };
      // Table 2
      const T2 = [
        { name: 'No Random Step Size', a: -1.47, b: 0.19, why: 'Removing the randomized step size "nearly eliminates Thinking gains" (p.10). Our reading: the landscape is shaped only along the one step size used in training.' },
        { name: 'No Random Num. Steps', a: 0.00, b: 9.65, why: 'Exactly 0.00 from thinking longer: a model that always took the same number of steps in training gets nothing from more [derived]. Verification still helps (9.65).' },
        { name: 'No Langevin Dynamics', a: 17.2, b: 17.0, why: 'Best single-path result (bold in the paper). Less noise in training means "less energy landscape exploration, which improves single path performance ... at the expense of self-verification" (Table 2 caption).' },
        { name: 'No Replay Buffer', a: 14.8, b: 17.8, why: 'Thinking longer still works well; combined is slightly below the full recipe. The buffer is meant to make the landscape well defined near its minimum (p.7).' },
        { name: 'Full System 2 Configuration', a: 7.19, b: 18.7, why: 'The default S2 recipe: best only in the combined column (18.7). Its single-path gain (7.19) is lower than two ablations. Noise trades single-path quality for diversity that verification can use.' },
      ];
      TABS.t2 = {
        label: 'Table 2 · System 2 thinking ablations', sub: '% perplexity improvement on BigBench Dyck (OOD), xxs S2 models, p.10 · exact', src: 'paper',
        controls(el) { el.appendChild(h('span', { class: 'mono', style: { color: 'var(--muted)', fontSize: '12px' } }, 'hover or tap a row for what it shows')); },
        draw(g) {
          const W = cv.w, H = cv.h, narrow = W < 520;
          const lx = narrow ? 10 : 16, labW = narrow ? 0 : 200, x0 = lx + labW + 8, x1 = W - 16;
          const vmin = -3, vmax = 20, X = (v) => x0 + (v - vmin) / (vmax - vmin) * (x1 - x0);
          const top = 34, rowH = (H - top - 40) / T2.length;
          T(g, '□ thinking longer   ■ thinking longer + self-verification', lx, 10, { size: 10.5, color: C.muted });
          [0, 5, 10, 15, 20].forEach(v => { poly(g, [[X(v), top - 4], [X(v), H - 34]], v === 0 ? C.ink : C.rule, { w: 1 }); T(g, v + '%', X(v), H - 28, { align: 'center', color: C.muted }); });
          T2.forEach((rw, i) => {
            const y = top + i * rowH, sel = S.t2 === i, bh = Math.min(13, rowH * 0.22);
            if (sel) { g.save(); g.fillStyle = 'rgba(47,60,255,0.06)'; g.fillRect(lx - 6, y, W - lx - 4, rowH - 4); g.restore(); }
            const ny = narrow ? y + 4 : y + rowH / 2 - 8;
            T(g, rw.name, lx, ny, { size: 11.5, weight: sel ? 700 : 400, color: sel ? C.ink : C.ink2 || C.ink });
            const by1 = narrow ? y + 22 : y + rowH / 2 - bh - 2, by2 = by1 + bh + 3;
            g.save(); g.strokeStyle = C.ink; g.lineWidth = 1; g.strokeRect(Math.min(X(0), X(rw.a)) + 0.5, by1 + 0.5, Math.max(1, Math.abs(X(rw.a) - X(0))), bh); g.fillStyle = C.blue; g.fillRect(Math.min(X(0), X(rw.b)), by2, Math.max(1.5, Math.abs(X(rw.b) - X(0))), bh); g.restore();
            T(g, rw.a.toFixed(2).replace('-', MINUS), Math.max(X(rw.a), X(0)) + 5, by1 + bh / 2, { baseline: 'middle', size: 10.5, weight: rw.a === 17.2 ? 700 : 400 });
            T(g, rw.b.toFixed(2), X(rw.b) + 5, by2 + bh / 2, { baseline: 'middle', size: 10.5, color: C.blue, weight: rw.b === 18.7 ? 700 : 400 });
            HITS.push({ x: (x0 + x1) / 2, y: y + rowH / 2, row: i, rect: [lx - 6, y, W - lx, rowH], lines: [rw.name, `longer ${rw.a}% · + verification ${rw.b}%`] });
          });
          T(g, '% perplexity improvement (higher is better)', (x0 + x1) / 2, H - 14, { align: 'center', size: 10.5, color: C.muted });
        },
        readout() { const rw = T2[S.t2]; return [`<b>${rw.name}</b>: ${rw.a}% / ${rw.b}%`, rw.why]; },
      };
      // Fig 8 / B.2 heatmaps
      TABS['8'] = {
        label: 'Fig 8 · token energies across thinking steps', sub: 'normalized energy (0–1, normalization not stated) · approx., read from the figure', src: 'paper',
        controls(el) {
          const opts = [['fig8a', '8a · quick fox'], ['fig8b', '8b · System 2'], ['figB2_seq1', 'B.2 · seen text'], ['figB2_seq2', 'B.2 · random tokens']].filter(o => HM && HM.heatmaps.find(x => x.id === o[0]));
          el.appendChild(lib.segmented({ label: 'Sequence', options: opts, value: S.hm, onchange: v => { S.hm = v; draw(); readout(); } }).el);
          const sl = lib.slider({ id: 'tr-it', label: 'iteration', min: 0, max: 11, step: 1, value: S.hmIt, fmt: v => String(v), oninput: v => { S.hmIt = v; draw(); readout(); } }); el.appendChild(sl.el);
        },
        hmap() { return HM ? HM.heatmaps.find(x => x.id === S.hm) : null; },
        draw(g) {
          const hm = this.hmap(); if (!hm) { T(g, 'heatmap data missing', 20, 20); return; }
          const W = cv.w, H = cv.h, rows = hm.rows.length, cols = hm.iterations.length;
          const labW = 92, barW = Math.min(130, W * 0.22), x0 = labW, x1 = W - barW - 18, top = 24, bot = 34;
          const cw = (x1 - x0) / cols, ch = (H - top - bot) / rows;
          T(g, 'iteration →', x0, 6, { color: C.muted, size: 10.5 });
          T(g, 'mean, iterations 1–11', x1 + 12, 6, { color: C.muted, size: 10.5 });
          hm.energy.forEach((rowv, i) => {
            const y = top + i * ch;
            T(g, hm.rows[i], labW - 8, y + ch / 2, { align: 'right', baseline: 'middle', size: 11.5 });
            rowv.forEach((v, j) => {
              const c = lib.cmap(1 - (1 - v) * 0.95); g.fillStyle = `rgb(${c[0]},${c[1]},${c[2]})`; g.fillRect(x0 + j * cw, y, cw - 1, ch - 1);
              if (cw > 30 && ch > 15) T(g, v.toFixed(2), x0 + j * cw + cw / 2, y + ch / 2, { align: 'center', baseline: 'middle', size: 9.5, color: v > 0.45 ? C.ink : '#fff' });
            });
            const plateau = rowv.slice(1).reduce((a, b) => a + b, 0) / (cols - 1);
            g.fillStyle = C.ink; g.fillRect(x1 + 12, y + ch * 0.25, plateau * barW, ch * 0.5);
            T(g, plateau.toFixed(2), x1 + 16 + plateau * barW, y + ch / 2, { baseline: 'middle', size: 10.5 });
            hit(x1 + 12 + plateau * barW, y + ch / 2, [`“${hm.rows[i]}”`, `iteration 0: ${rowv[0].toFixed(2)} · 1: ${rowv[1].toFixed(2)}`, `mean of 1–11: ${plateau.toFixed(2)}`]);
          });
          const xi = x0 + S.hmIt * cw; g.save(); g.strokeStyle = C.ink; g.lineWidth = 1.5; g.strokeRect(xi + 0.5, top - 2, cw - 1, rows * ch + 2); g.restore();
          [0, 3, 6, 9, 11].forEach(j => T(g, String(j), x0 + j * cw + cw / 2, top + rows * ch + 6, { align: 'center', color: C.muted, size: 10.5 }));
          T(g, 'deep blue = low energy (confident) · pale = high', x0, H - 14, { color: C.muted, size: 10.5 });
        },
        readout() {
          const hm = this.hmap(); if (!hm) return ['no data'];
          const E = hm.energy, it = S.hmIt, n = E.length;
          const d1 = E.reduce((a, r) => a + (r[0] - r[1]), 0) / n, d111 = E.reduce((a, r) => a + (r[1] - r[11]), 0) / n;
          const pl = E.map((r, i) => [hm.rows[i], r.slice(1).reduce((a, b) => a + b, 0) / 11]).sort((a, b) => b[1] - a[1]);
          const col = E.map((r, i) => [hm.rows[i], r[it]]).sort((a, b) => b[1] - a[1]);
          return [`mean drop at iteration 1: <b>${d1.toFixed(2)}</b>`, `mean change 1 → 11: ${sgn(-d111, 2)}`, `highest after thinking: <b>“${pl[0][0]}”</b> ${pl[0][1].toFixed(2)}`, `lowest: “${pl[pl.length - 1][0]}” ${pl[pl.length - 1][1].toFixed(2)}`, `at iteration ${it}: highest “${col[0][0]}”`];
        },
      };
      // toy
      const DS = [['val', 'held-out'], ['ood_shakespeare', 'Shakespeare'], ['ood_code', 'Python code'], ['train', 'train']];
      TABS.toy = {
        label: 'Toy · char-level text EBT', sub: 'trained for this explainer · nats per character · not comparable with the paper', src: 'toy',
        controls(el) {
          if (!TX) return;
          el.appendChild(lib.segmented({ label: 'Dataset', options: DS, value: S.toyDs, onchange: v => { S.toyDs = v; draw(); readout(); } }).el);
          const sl = lib.slider({ id: 'tr-toyn', label: 'thinking steps N', min: 0, max: 16, step: 1, value: S.toyN, fmt: v => String(v), oninput: v => { S.toyN = v; draw(); readout(); } }); el.appendChild(sl.el);
        },
        draw(g) {
          if (!TX) { T(g, 'toy text data missing (data/text.json)', 20, 20); return; }
          const tc = TX.thinking_curve, W = cv.w, H = cv.h, wide = W >= 520;
          const curves = DS.map(([k]) => tc.datasets[k]).filter(Boolean);
          const all = curves.flatMap(d => d.ebt_fixed_alpha_ce.slice(1).concat([d.baseline_ce]));
          const lo = Math.floor(Math.min(...all) * 2) / 2 - 0.1, hi = 3.6;
          const save = cv.w; void save;
          // left: CE vs steps
          const split = wide ? Math.round(W * 0.62) : W;
          const M = (() => { const ow = cv.w; cv.w = split; const m = axesBox(g, { xlim: [0, 16.5], ylim: [lo, hi], xticks: [0, 2, 4, 6, 8, 10, 12, 14, 16], yticks: linTicks(lo, hi, 5), yfmt: v => v.toFixed(1), xlabel: 'thinking steps N', ylabel: 'loss, nats/char (↓)', bottom: wide ? 44 : Math.round(H * 0.42) + 44 }); cv.w = ow; return m; })();
          const r = M.rect;
          DS.forEach(([k, nm]) => {
            const d = tc.datasets[k]; if (!d) return; const sel = k === S.toyDs;
            const ys = d.ebt_fixed_alpha_ce; const pts = ys.map((v, i) => [M.X(i), M.Y(Math.min(v, hi + 1))]);
            g.save(); g.beginPath(); g.rect(r.x, r.y, r.w, r.h); g.clip();
            poly(g, pts, sel ? C.blue : C.faint, { w: sel ? 2 : 1.2 });
            if (sel) poly(g, [[r.x, M.Y(d.baseline_ce)], [r.x + r.w, M.Y(d.baseline_ce)]], C.ink, { w: 1.2, dash: [5, 4] });
            g.restore();
            const yl = M.Y(ys[16]); T(g, nm, r.x + r.w - 4, yl - (sel ? 13 : 12), { align: 'right', size: 10.5, color: sel ? C.blue : C.muted });
            if (sel) { T(g, 'same-size feed-forward ' + d.baseline_ce.toFixed(3), r.x + r.w - 4, M.Y(d.baseline_ce) + 4, { align: 'right', size: 10.5, color: C.ink }); const X = M.X(S.toyN), Y = M.Y(ys[S.toyN]); marker(g, 'o', X, Y, 4.5, C.ink, '#fff'); hit(X, Y, [`${nm}, N = ${S.toyN}`, `${ys[S.toyN].toFixed(3)} nats/char`, `ppl/char ${Math.exp(ys[S.toyN]).toFixed(2)}`]); }
          });
          poly(g, [[M.X(3), r.y], [M.X(3), r.y + r.h]], C.ink, { w: 1, dash: [2, 4], alpha: 0.5 });
          T(g, 'trained with 2–3', M.X(3) + 4, r.y + 2, { size: 10.5, color: C.muted });
          // right/bottom: best-of-M
          const bon = TX.bon && TX.bon.settings && TX.bon.settings[0] && TX.bon.settings[0].datasets[S.toyDs];
          if (!bon) return;
          const Ms = TX.bon.M; const bx = wide ? split + 46 : 56, by = wide ? 16 : r.y + r.h + 58, bw = (wide ? W - split - 60 : W - 70), bh = wide ? H - 60 : H - by - 40;
          const vals = bon.energy_select_ce.concat(bon.oracle_ce, bon.mean_ce), blo = Math.min(...vals) - 0.05, bhi = Math.max(...vals) + 0.05;
          const BX = (i) => bx + i / (Ms.length - 1) * bw, BY = (v) => by + bh - (v - blo) / (bhi - blo) * bh;
          g.save(); g.strokeStyle = C.ink; g.lineWidth = 1; g.beginPath(); g.moveTo(bx, by); g.lineTo(bx, by + bh); g.lineTo(bx + bw, by + bh); g.stroke(); g.restore();
          Ms.forEach((m, i) => T(g, 'M=' + m, BX(i), by + bh + 5, { align: 'center', size: 10.5, color: C.muted }));
          linTicks(blo, bhi, 3).forEach(v => T(g, v.toFixed(1), bx - 5, BY(v), { align: 'right', baseline: 'middle', size: 10.5, color: C.muted }));
          T(g, `best-of-M, N = ${TX.bon.settings[0].N}`, bx, by - 14, { size: 10.5, color: C.ink, weight: 600 });
          [[bon.mean_ce, C.faint, 'average'], [bon.energy_select_ce, C.blue, 'lowest energy'], [bon.oracle_ce, C.ink, 'oracle']].forEach(([arr, col, nm], j) => {
            const pts = arr.map((v, i) => [BX(i), BY(v)]); poly(g, pts, col, { w: col === C.blue ? 2 : 1.2, dash: nm === 'oracle' ? [4, 3] : null });
            pts.forEach((p, i) => { marker(g, 'o', p[0], p[1], 3, col === C.faint ? '#fff' : col, col === C.faint ? C.muted : '#fff'); hit(p[0], p[1], [`${nm}, M = ${Ms[i]}`, `${arr[i].toFixed(3)} nats/char`]); });
            T(g, nm, bx + bw, pts[pts.length - 1][1] + (nm === 'average' ? -14 : 3), { align: 'right', size: 10.5, color: col === C.faint ? C.muted : col });
          });
        },
        readout() {
          if (!TX) return ['toy data missing'];
          const d = TX.thinking_curve.datasets[S.toyDs], ys = d.ebt_fixed_alpha_ce, n = S.toyN;
          const g3 = (1 - Math.exp(ys[n] - ys[3])) * 100;
          const bon = TX.bon.settings[0].datasets[S.toyDs];
          return [`N = <b>${n}</b>: ${ys[n].toFixed(3)} nats/char`, `vs N = 3 (training length): <b>${sgn(g3)}%</b> ppl`, `feed-forward baseline ${d.baseline_ce.toFixed(3)}`, `best-of-8 by energy ${bon.energy_select_ce[3].toFixed(3)} vs single ${bon.energy_select_ce[0].toFixed(3)} (oracle ${bon.oracle_ce[3].toFixed(3)})`];
        },
      };

      // ---------- generic draw / readout / interaction ----------
      function nearestHit(px, py) { let best = null, bd = 14; HITS.forEach(hh => { if (hh.rect) return; const d = Math.hypot(hh.x - px, hh.y - py); if (d < bd) { bd = d; best = hh; } }); return best; }
      function draw() {
        if (!cv.w) return; const g = cv.ctx; cv.clear(); HITS = [];
        const tb = TABS[S.tab]; try { tb.draw(g); } catch (e) { console.error('thinking-results', e); }
        if (S.hover && !S.hover.rect) tooltip(g);
      }
      function readout() { const tb = TABS[S.tab]; ro.innerHTML = (tb.readout ? tb.readout() : []).map(s => `<span>${s}</span>`).join(''); }
      function setTab(v, keepCtl) {
        if (!TABS[v]) return; S.tab = v; S.hover = null; tabSeg.set(v);
        const tb = TABS[v];
        labEl.textContent = tb.label; subEl.innerHTML = tb.sub;
        if (!keepCtl) { tctl.innerHTML = ''; tb.controls && tb.controls(tctl); }
        const tag = { paper: 'from the paper', toy: 'toy model trained for this explainer', ext: 'beyond the paper' }[tb.src] || '';
        ctx.setCaption(`<span class="src ${tb.src}">${tag}</span> ${CAP[v] || ''}`);
        draw(); readout();
      }
      const CAP = {
        '6a': 'Circle: no thinking (2 passes). Star: thinking longer (3). Triangles: self-verification (6, 15, 30). Dashed line: your budget; the ringed point is the best EBT setting that fits in it.',
        '6b': 'Each dot is a checkpoint of one xxs EBT. Blue line: least-squares fit (matches the paper\'s). Faint lines: refits on bootstrap resamples. Bar at the cursor: 95% range of the bootstrap lines there.',
        b1a: 'Same training run, best-of-10 compared with best-of-2. The red dot is the checkpoint where more candidates hurt.',
        b1b: 'Both axes log, so the straight Fig 6b line looks bent. Shaded: where 95% of the bootstrap lines fall.',
        '7': 'Five datasets, one dot each; the leftmost is the pretraining data. Faint lines: refits leaving one dataset out.',
        t2: 'Outlined bars: thinking longer only. Blue bars: thinking longer plus self-verification. Bold values are bold in the paper.',
        '8': 'Rows are tokens, columns thinking iterations. Our colormap (deep blue = low energy); the paper uses reversed viridis. Right: each token\'s mean energy after the first update.',
        toy: 'Left: held-out loss against thinking steps for four datasets (selected in blue, feed-forward baseline dashed). Right: best-of-M by lowest energy vs the average candidate and an oracle that knows the answer.',
      };
      cv.canvas.addEventListener('mousemove', (ev) => {
        const [px, py] = cv.local(ev);
        if (S.tab === 't2') { const row = HITS.find(hh => hh.rect && px >= hh.rect[0] && px <= hh.rect[0] + hh.rect[2] && py >= hh.rect[1] && py <= hh.rect[1] + hh.rect[3]); if (row && row.row !== S.t2) { S.t2 = row.row; draw(); readout(); } return; }
        const n = nearestHit(px, py); const same = n === S.hover || (n && S.hover && n.x === S.hover.x && n.y === S.hover.y);
        S.hover = n; cv.canvas.style.cursor = n ? 'pointer' : 'default'; if (!same) draw();
      });
      cv.canvas.addEventListener('mouseleave', () => { if (S.hover) { S.hover = null; draw(); } });
      cv.canvas.addEventListener('click', (ev) => {
        const [px, py] = cv.local(ev);
        if (S.tab === 't2') { const row = HITS.find(hh => hh.rect && px >= hh.rect[0] && px <= hh.rect[0] + hh.rect[2] && py >= hh.rect[1] && py <= hh.rect[1] + hh.rect[3]); if (row) { S.t2 = row.row; draw(); readout(); } return; }
        if (S.tab === '7') {
          const n = nearestHit(px, py); if (!n) return; const pts = ser('fig7');
          let bi = -1, bd = 1e9; pts.forEach((q, i) => { const hh = HITS[i]; if (!hh) return; const d = Math.hypot(hh.x - n.x, hh.y - n.y); if (d < bd) { bd = d; bi = i; } });
          if (bi < 0) return; if (S.drop7.has(bi)) S.drop7.delete(bi); else if (pts.length - S.drop7.size > 2) S.drop7.add(bi);
          draw(); readout(); return;
        }
        const n = nearestHit(px, py); if (n) { S.hover = n; draw(); }
      });
      // sweep animation for the 6a budget
      let sweepT = null;
      function stopSweep() { if (sweepT) { clearInterval(sweepT); sweepT = null; } }
      function sweep(a, b) {
        stopSweep(); S.budget = a; if (S._budgetSl) S._budgetSl.set(a); draw(); readout();
        if (lib.reducedMotion) { S.budget = b; if (S._budgetSl) S._budgetSl.set(b); draw(); readout(); return; }
        sweepT = setInterval(() => { S.budget = Math.min(b, S.budget + 1); if (S._budgetSl) S._budgetSl.set(S.budget); draw(); readout(); if (S.budget >= b) stopSweep(); }, 70);
      }
      const CFG = [
        () => { stopSweep(); S.budget = 2; setTab('6a'); },
        () => { stopSweep(); S.budget = 3; setTab('6a'); },
        () => { setTab('6a'); if (ctx.visible()) sweep(3, 30); else { S.budget = 30; S._budgetSl && S._budgetSl.set(30); draw(); readout(); } },
        () => { stopSweep(); S.boot = true; S.cur6b = 32.65; setTab('6b'); },
        () => { stopSweep(); S.boot = true; S.curB1a = 8.85; setTab('b1a'); },
        () => { stopSweep(); S.ext = L10(1.5e13); setTab('b1b'); },
        () => { stopSweep(); S.drop7.clear(); setTab('7'); },
        () => { stopSweep(); S.toyDs = 'val'; S.toyN = 16; setTab('toy'); },
      ];
      setTab('6a');
      return {
        step(i) { CFG[Math.max(0, Math.min(CFG.length - 1, i))](); },
        show() { draw(); },
        hide() { stopSweep(); },
      };
    },
  });
})();
