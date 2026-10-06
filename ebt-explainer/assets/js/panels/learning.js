/* Panel: watching a landscape learn. Checkpoint scrubber over the real toy 2-D EBT training run (data/toy2d.json):
   stored 64x64 energy grids for two contexts at all 12 checkpoints (blended while scrubbing), stored 40-step descents at each
   checkpoint, training loss vs eval errors, and the Best-of-N gain by checkpoint. Basin minimum is computed live from the weights. */
EBT.panel({
  id: 'learning',
  nav: 'Watching a landscape learn',
  title: 'Watching a landscape learn',
  lede: 'Scrub through 12 checkpoints of a real training run. The basin forms early and the loss soon goes flat, yet the energy keeps getting better at judging answers.',
  text: `
    <p>Two descents are nested. Learning updates the weights, $\\theta_{t+1}=\\theta_t-\\eta\\nabla_\\theta\\mathcal L$, which reshapes the landscape; the slider moves $t$. Thinking then rolls down that landscape: blue paths, 40 steps from 8 fixed starts. Left: almost noise-free targets. Right: noisy ones (dashed ring: target noise; crosshair: clean mean $\\mu$).</p>`,
  steps: [
    { label: 'Random weights', html: '<p>The untrained network gives a shallow bowl with an arbitrary minimum, far from $\\mu$.</p>' },
    { label: 'A basin is dragged to the data', html: '<p>Each update backpropagates through short descents, so the minimum slides toward $\\mu(x)$. By step 50 it is still shallow and has not yet reached $\\mu$.</p>' },
    { label: 'The loss hits its floor', html: '<p>By step 100 the basin sits on $\\mu$ and the loss reaches the noise floor (dashed): what remains is noise in the targets, which no predictor can remove.</p>' },
    { label: 'Refinement the loss cannot see', html: '<p>The loss stays flat, yet the error to the clean mean (hollow dots) falls another 17&times; after step 200. That gain hides inside the loss\'s batch-to-batch jitter.</p>' },
    { label: 'Verification keeps improving', html: '<p>Run 8 noisy descents and keep the lowest-energy one. This Best-of-8 gain over one descent grows from 1.5&times; to 7.8&times;, mostly after the loss went flat. The paper reports the same direction (Fig 6b): Best-of-5 gains rose from "4%&minus;8%" to "10%&minus;14%" with more training (p.10).</p>' },
  ],
  after: `
    <p class="note">Caveats: Fig 6b is one benchmark (BigBench Dyck; "we did not observe this trend in other benchmarks", p.34) and noisy. The toy agrees in direction, not size. Toy: 12,000 Adam steps of Algorithm 1 with the paper's regularizers; heatmaps are stored per checkpoint.</p>`,
  source: [{ kind: 'toy', note: 'toy 2-D EBT, 12 real checkpoints' }, { kind: 'paper', note: 'Fig 6b direction only, p.10, p.34' }],
  figure(stage, ctx) {
    const { lib } = ctx, h = lib.h, C = lib.C, M = window.EBT.toy2d;
    if (!M || !M.ready) { stage.appendChild(h('p', { class: 'callout warn' }, 'Toy model data missing (data/toy2d.json).')); return {}; }
    const D = M.data, STEPS = M.steps, nC = M.nCkpt, EXT = M.extent;
    const S = { pos: 0, anim: null, drawT: 1 };
    const fmtStep = (s) => s.toLocaleString('en-US');

    // ---------------------------------------------------------------- layout
    const row = h('div', { class: 'fig-row' }); stage.appendChild(row);
    const panes = [0, 1].map(ci => {
      const c = M.contexts[ci];
      const fr = lib.frame(row, { label: ci === 0 ? 'x = 0.25 · near-deterministic' : 'x = 0.75 · noisy', sub: 'target noise ' + c.sigma.toFixed(2) + ' · energy over ŷ' });
      fr.wrap.style.flex = '1 1 250px';
      const cv = lib.canvas(fr.frame, 300, 300, { label: 'Energy landscape of the toy model for context x = ' + c.x + ' at the chosen training checkpoint, with stored descents' });
      const ro = h('div', { class: 'readout lrn-ro' }); fr.wrap.appendChild(ro);
      return { cv, ro, fr, ci };
    });
    const ctl = h('div', { class: 'controls' }); stage.appendChild(ctl);
    const sl = lib.slider({ id: 'lrn-ck', label: 'training step', min: 0, max: nC - 1, step: 0.01, value: 0, fmt: (v) => fmtStep(STEPS[Math.round(v)]), oninput: (v) => { stopAnim(); S.pos = v; S.drawT = 1; drawAll(); } });
    sl.input.addEventListener('change', () => { S.pos = Math.round(S.pos); sl.set(S.pos); S.drawT = 0; animatePaths(); });
    sl.el.style.maxWidth = '340px'; sl.el.style.flex = '2 1 220px';
    const bPrev = lib.button('◀', () => go(Math.max(0, Math.ceil(S.pos - 1e-6) - 1)), { aria: 'previous checkpoint' });
    const bPlay = lib.button('play', () => { if (S.anim) { stopAnim(); return; } animateTo(S.pos >= nC - 1 ? 0 : S.pos, nC - 1, 1.0, S.pos >= nC - 1); }, { primary: true });
    const bNext = lib.button('▶', () => go(Math.min(nC - 1, Math.floor(S.pos + 1e-6) + 1)), { aria: 'next checkpoint' });
    ctl.append(bPrev, bPlay, bNext, sl.el);
    const row3 = h('div', { class: 'fig-row' }); stage.appendChild(row3);
    const FL = lib.frame(row3, { label: 'Loss vs where thinking lands', sub: 'log–log · click or drag to scrub' }); FL.wrap.style.flex = '1 1 320px';
    const LW = 360, LH = 230;
    const lc = lib.canvas(FL.frame, LW, LH, { label: 'Training loss curve with the noise floor and eval errors at each checkpoint, log-log, with a cursor at the chosen checkpoint' });
    const legL = h('div', { class: 'readout lrn-leg', html: '<span><i class="ln"></i>training loss</span><span><i class="dt"></i>error vs noisy y</span><span><i class="dt hol"></i>error to clean μ</span><span><i class="ln dash"></i>noise floor</span>' }); FL.wrap.appendChild(legL);
    const FV = lib.frame(row3, { label: 'Verification, by checkpoint', sub: 'Best-of-8 gain over one descent' }); FV.wrap.style.flex = '1 1 260px';
    const vc = lib.canvas(FV.frame, 300, 230, { label: 'Best-of-8 gain over a single descent versus training step' });
    const roV = h('div', { class: 'readout' }); FV.wrap.appendChild(roV);

    // ---------------------------------------------------------------- per-checkpoint data
    const box = { x: 0, y: 0, w: 300, h: 300 };
    const P = (q) => M.toPx(q, box);
    const gridCache = {};
    const gridOf = (ci, k) => gridCache[ci + ':' + k] || (gridCache[ci + ':' + k] = M.storedGrid(ci, STEPS[k]));
    const statCache = {};
    function stats(ci, k) { // minimum polished by descent and E_min (live, from weights)
      const key = ci + ':' + k; if (statCache[key]) return statCache[key];
      const g = gridOf(ci, k), n = g.length, x = M.contexts[ci].x; let best = Infinity, at = [0, 0];
      g.forEach((r, i) => r.forEach((v, j) => { if (v < best) { best = v; at = [EXT[0] + j * (EXT[1] - EXT[0]) / (n - 1), EXT[3] - i * (EXT[3] - EXT[2]) / (n - 1)]; } }));
      const mn = M.descend(k, x, at, { alpha: 0.3, steps: 80 }).path.pop();
      return (statCache[key] = { min: mn, E: M.energy(k, x, mn) });
    }
    const trajOf = (ci, k) => D.trajectories_by_ckpt.find(t => t.ctx === ci && t.ckpt === k);

    // ---------------------------------------------------------------- drawing
    function cross(c, p, r = 7) { c.save(); c.lineWidth = 3; c.strokeStyle = 'rgba(17,17,17,0.35)'; c.beginPath(); c.moveTo(p[0] - r, p[1]); c.lineTo(p[0] + r, p[1]); c.moveTo(p[0], p[1] - r); c.lineTo(p[0], p[1] + r); c.stroke(); c.lineWidth = 1.4; c.strokeStyle = '#fff'; c.stroke(); c.restore(); }
    function plate(c, s, x, y, o = {}) { c.save(); c.font = '11px "JetBrains Mono", monospace'; const w = c.measureText(s).width + 10; const xx = o.align === 'right' ? x - w : x; c.fillStyle = 'rgba(255,255,255,0.88)'; c.fillRect(xx, y, w, 16); c.restore(); lib.text(c, s, xx + 5, y + 2, { size: 11, kind: 'mono', color: o.color || '#111' }); }
    function drawPane(pn) {
      const ci = pn.ci, c = pn.cv.ctx; pn.cv.clear();
      const k0 = Math.floor(S.pos + 1e-9), k1 = Math.min(nC - 1, k0 + 1), f = S.pos - k0, kn = Math.round(S.pos);
      const a = gridOf(ci, k0), b = gridOf(ci, k1);
      const E = f < 1e-3 ? a : a.map((r, i) => r.map((v, j) => v + (b[i][j] - v) * f));
      const flat = E.flat().sort((x, y) => x - y), lo = flat[0], hi = flat[Math.floor(0.6 * (flat.length - 1))];
      lib.heatmap(c, E, 0, 0, 300, 300, { range: [lo, hi], gamma: 0.75 });
      const lv = Array.from({ length: 10 }, (_, i) => lo + (hi - lo) * Math.pow((i + 1) / 10, 1.3));
      lib.contours(c, E, 0, 0, 300, 300, lv, { color: 'rgba(17,17,17,0.2)', width: 1 });
      const mu = P(M.contexts[ci].mu), rs = M.contexts[ci].sigma / (EXT[1] - EXT[0]) * 300;
      c.save(); c.strokeStyle = '#fff'; c.lineWidth = 1.2; c.setLineDash([3, 3]); c.beginPath(); c.arc(mu[0], mu[1], Math.max(3, rs), 0, 7); c.stroke(); c.restore();
      const near = Math.abs(S.pos - kn) < 0.02, al = near ? 1 : Math.max(0, 1 - Math.abs(S.pos - kn) * 3);
      if (al > 0) {
        const tj = trajOf(ci, kn);
        if (tj) { c.save(); c.globalAlpha = al; tj.paths.forEach(p => { const px = p.map(P); lib.line(c, px, { color: C.blue, width: 1.5, progress: S.drawT }); lib.dot(c, px[0][0], px[0][1], 3, '#fff', { stroke: C.blue, lw: 1.3 }); if (S.drawT >= 1) { const e = px[px.length - 1]; lib.dot(c, e[0], e[1], 3.4, C.blue, { stroke: '#fff', lw: 1 }); } }); c.restore(); }
      }
      cross(c, mu);
      if (near) { const st = stats(ci, kn), m = P(st.min); c.save(); c.strokeStyle = '#111'; c.lineWidth = 1; c.beginPath(); c.arc(m[0], m[1], 5, 0, 7); c.stroke(); c.restore(); }
      plate(c, 'step ' + fmtStep(STEPS[kn]), 6, 278);
      plate(c, 'E ∈ [' + lo.toFixed(2) + ', ' + hi.toFixed(2) + ']', 294, 278, { align: 'right', color: '#6b6b70' });
      {
        const st = stats(ci, kn), m0 = M.contexts[ci].mu;
        const f2 = (v) => (Math.abs(v) < 0.005 ? 0 : v).toFixed(2);
        pn.ro.innerHTML = `<span>min <b>(${f2(st.min[0])}, ${f2(st.min[1])})</b> vs μ (${f2(m0[0])}, ${f2(m0[1])})</span><span>E<sub>min</sub> <b>${st.E.toFixed(3)}</b></span>`;
      }
    }
    // loss chart: x = training step (log; step 0 drawn at 1), y = log
    const LA = { x: 52, y: 10, w: LW - 64, h: LH - 58 };
    let lax = null;
    // centered mean of 9 neighbouring single-batch losses (samples are log-spaced, so the window is about ±10% of the step).
    // Unlike the stored EMA it does not lag, so the curve drops where the eval dots drop.
    const LRAW = D.loss_curve.raw, LSM = LRAW.map((_, i) => { let s = 0, n = 0; for (let j = Math.max(0, i - 4); j <= Math.min(LRAW.length - 1, i + 4); j++) { s += LRAW[j]; n++; } return s / n; });
    function drawLoss() {
      const c = lc.ctx; lc.clear(); const lcv = D.loss_curve, ev = D.eval.curve;
      lax = lib.axes(c, { ...LA, xlim: [1, 12000], ylim: [5e-5, 5], xlog: true, ylog: true, xticks: [1, 10, 100, 1000, 10000], yticks: [1e-4, 1e-3, 1e-2, 0.1, 1], xfmt: (v) => (v >= 1000 ? v / 1000 + 'k' : String(v)), yfmt: (v) => (v >= 0.1 ? String(v) : v.toExponential(0).replace('e-', 'e−')), size: 11 });
      lib.plot(c, lax, lcv.step.map((s, i) => [s, lcv.raw[i]]), { color: C.blue, width: 1, alpha: 0.25 });
      lib.plot(c, lax, lcv.step.map((s, i) => [s, LSM[i]]), { color: C.blue, width: 2 });
      const nf = lcv.noise_floor; c.save(); c.setLineDash([5, 4]); c.strokeStyle = '#111'; c.lineWidth = 1.1; c.beginPath(); c.moveTo(lax.X(1), lax.Y(nf)); c.lineTo(lax.X(12000), lax.Y(nf)); c.stroke(); c.restore();
      lib.text(c, 'noise floor ' + nf.toFixed(4), lax.X(1) + 4, lax.Y(nf) + 3, { size: 10.5, kind: 'mono', color: '#33333a' });
      const sx = (s) => Math.max(1, s);
      lib.plot(c, lax, ev.step.map((s, i) => [sx(s), ev.mse_vs_mu_N4[i]]), { color: '#111', width: 1, alpha: 0.5, dash: [2, 3] });
      ev.step.forEach((s, i) => { lib.dot(c, lax.X(sx(s)), lax.Y(ev.mse_vs_y_N4[i]), 3, '#111'); lib.dot(c, lax.X(sx(s)), lax.Y(ev.mse_vs_mu_N4[i]), 3.2, '#fff', { stroke: '#111', lw: 1.2 }); });
      // cursor in log-step space between checkpoints
      const k0 = Math.floor(S.pos + 1e-9), k1 = Math.min(nC - 1, k0 + 1), f = S.pos - k0;
      const l0 = Math.log10(sx(STEPS[k0])), l1 = Math.log10(sx(STEPS[k1])), X = lax.X(Math.pow(10, l0 + (l1 - l0) * f));
      c.save(); c.strokeStyle = C.blue; c.lineWidth = 1.2; c.beginPath(); c.moveTo(X, LA.y); c.lineTo(X, LA.y + LA.h); c.stroke(); c.restore();
      lib.text(c, 'training step (0 drawn at 1)', LA.x + LA.w / 2, LH - 18, { size: 11, kind: 'mono', color: '#6b6b70', align: 'center' });
    }
    function posFromX(px) { const v = Math.pow(10, (px - LA.x) / LA.w * Math.log10(12000)); const ls = STEPS.map(s => Math.log10(Math.max(1, s))), lv = Math.log10(Math.max(1, v)); if (lv <= ls[0]) return 0; for (let k = 0; k < nC - 1; k++) if (lv <= ls[k + 1]) return k + (ls[k + 1] - ls[k] > 0 ? (lv - ls[k]) / (ls[k + 1] - ls[k]) : 1); return nC - 1; }
    let dragging = false;
    const onPtr = (ev) => { const [px] = lc.toLocal(ev); stopAnim(); S.pos = posFromX(px); sl.set(S.pos); S.drawT = 1; drawAll(); };
    lc.canvas.style.cursor = 'ew-resize';
    lc.canvas.addEventListener('pointerdown', (ev) => { dragging = true; lc.canvas.setPointerCapture(ev.pointerId); onPtr(ev); });
    lc.canvas.addEventListener('pointermove', (ev) => { if (dragging) onPtr(ev); });
    lc.canvas.addEventListener('pointerup', () => { dragging = false; S.pos = Math.round(S.pos); sl.set(S.pos); S.drawT = 0; animatePaths(); });

    const fmtE = (v) => (v >= 0.01 ? v.toFixed(3) : v.toExponential(1).replace('e-', 'e−'));
    // verification: gain = error(M=1) / error(M=8) by training step
    const BV = D.bon_vs_training, GAIN = BV.step.map((s, i) => BV.M1[i] / BV.M8[i]);
    function drawVerif() {
      const c = vc.ctx; vc.clear(); const kn = Math.round(S.pos);
      const ax = lib.axes(c, { x: 36, y: 8, w: 254, h: 180, xlim: [10, 15000], ylim: [0, 8.5], xlog: true, xticks: [10, 100, 1000, 10000], yticks: [0, 2, 4, 6, 8], xfmt: (v) => (v >= 1000 ? v / 1000 + 'k' : String(v)), yfmt: (v) => v + '×', size: 10.5 });
      c.save(); c.strokeStyle = 'rgba(17,17,17,0.4)'; c.setLineDash([2, 3]); c.beginPath(); c.moveTo(ax.X(10), ax.Y(1)); c.lineTo(ax.X(15000), ax.Y(1)); c.stroke(); c.restore();
      lib.text(c, 'no gain', ax.X(15000) - 3, ax.Y(1) + 3, { size: 10, kind: 'mono', color: '#6b6b70', align: 'right' });
      lib.plot(c, ax, BV.step.map((s, i) => [s, GAIN[i]]), { color: C.blue, width: 1.8, markers: 3 });
      const k0 = Math.floor(S.pos + 1e-9), k1 = Math.min(nC - 1, k0 + 1), f = S.pos - k0;
      const l0 = Math.log10(Math.max(10, STEPS[k0])), l1 = Math.log10(Math.max(10, STEPS[k1])), X = ax.X(Math.pow(10, l0 + (l1 - l0) * f));
      c.save(); c.strokeStyle = C.blue; c.lineWidth = 1.2; c.beginPath(); c.moveTo(X, 8); c.lineTo(X, 188); c.stroke(); c.restore();
      const j = BV.step.indexOf(STEPS[kn]);
      if (j >= 0) { lib.dot(c, ax.X(BV.step[j]), ax.Y(GAIN[j]), 5, C.blue, { stroke: '#fff', lw: 1.5 }); roV.innerHTML = `<span>M = 1 error <b>${fmtE(BV.M1[j])}</b></span><span>M = 8 <b>${fmtE(BV.M8[j])}</b></span><span>gain <b>${GAIN[j].toFixed(1)}×</b></span>`; }
      else roV.innerHTML = '<span>no Best-of-N measurement before step 25</span>';
      lib.text(c, 'training step', 163, 214, { size: 10.5, kind: 'mono', color: '#6b6b70', align: 'center' });
    }
    function drawAll() { panes.forEach(drawPane); drawLoss(); drawVerif(); }

    // ---------------------------------------------------------------- animation
    const loop = lib.loop((dt) => {
      let busy = false;
      if (S.anim) { const A = S.anim; A.t = Math.min(1, A.t + dt / A.dur); S.pos = A.from + (A.to - A.from) * A.t; sl.set(S.pos); S.drawT = 1; if (A.t >= 1) { S.anim = null; bPlay.textContent = 'play'; S.pos = Math.round(S.pos); sl.set(S.pos); S.drawT = 0; } busy = true; }
      else if (S.drawT < 1) { S.drawT = Math.min(1, S.drawT + dt / 0.9); busy = true; }
      drawAll();
      if (!busy) return false;
    });
    function animateTo(from, to, perCk, jump) { if (jump) S.pos = from; const dist = Math.abs(to - S.pos); if (lib.reducedMotion || !ctx.visible() || dist < 1e-6) { S.pos = to; sl.set(to); S.drawT = 1; drawAll(); return; } S.anim = { from: S.pos, to, t: 0, dur: Math.max(0.3, dist * perCk) }; bPlay.textContent = 'pause'; loop.start(); }
    function stopAnim() { if (S.anim) { S.anim = null; } bPlay.textContent = 'play'; }
    function animatePaths() { if (lib.reducedMotion || !ctx.visible()) { S.drawT = 1; drawAll(); return; } S.drawT = 0; loop.start(); }
    function go(k) { stopAnim(); animateTo(S.pos, k, 0.5); }
    const _stop = loop.stop; loop.stop = () => { _stop(); if (!S.anim) bPlay.textContent = 'play'; };
    const hl = (which) => [FL, FV].forEach(F => F.frame.classList.toggle('lrn-hl', F === which));
    const caps = [
      'Energy over ŷ for two contexts at the selected training step (darker = lower; color range per frame). Crosshair: clean mean μ(x); dashed ring: target noise; small black ring: the landscape\'s minimum. Blue: 40-step descents from 8 fixed starts.',
      'Scrubbing from step 0 to step 50. Between checkpoints the heatmap blends the two stored grids; descents are drawn at checkpoints only.',
      'Loss plot: blue = training loss (local mean; faint = single batches), black dots = eval error against noisy targets after 4 thinking steps, dashed = noise floor.',
      'Hollow dots: eval error to the clean mean μ after 4 steps. It keeps falling after the loss and the black dots have gone flat.',
      'Best-of-8 gain = error of one noisy descent ÷ error of the lowest-energy of 8, on 1024 eval contexts, at each checkpoint.',
    ];
    drawAll();
    return {
      step(i) {
        ctx.setCaption(caps[i]); stopAnim();
        hl([null, null, FL, FL, FV][i]);
        if (i === 0) { S.pos = 0; sl.set(0); animatePaths(); }
        if (i === 1) animateTo(0, 3, 0.6, true);
        if (i === 2) animateTo(3, 5, 0.7, true);
        if (i === 3) animateTo(5, 11, 0.3, true);
        if (i === 4) animateTo(2, 11, 0.22, true);
        drawAll();
      },
      show() { if (S.anim || S.drawT < 1) loop.start(); },
      hide() { loop.stop(); },
    };
  },
});
