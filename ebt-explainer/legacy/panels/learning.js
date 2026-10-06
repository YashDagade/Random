/* Panel: watching a landscape learn. Checkpoint scrubber over the real toy 2-D EBT training run (data/toy2d.json):
   stored 64×64 energy grids for two contexts at all 12 checkpoints (blended while scrubbing), stored 40-step descents at each
   checkpoint, training loss vs eval errors, thinking curves by checkpoint, Best-of-N gain by checkpoint. Basin minimum and Hessian
   are computed live from the checkpoint's weights. */
EBT.panel({
  id: 'learning',
  nav: 'Watching a landscape learn',
  title: 'Watching a landscape learn',
  lede: 'Scrub through 12 checkpoints of a real training run. The basin forms and the training loss hits its floor early. Then two things the loss curve cannot show keep improving: how close thinking lands to the truth, and how well the energy ranks its own candidates.',
  text: `
    <p>This is the toy 2-D EBT used throughout the site, trained for 12,000 Adam steps with Algorithm 1 and all four regularizers of Sec 3.3 (about 2.6 CPU-minutes). Each heatmap is the energy over the prediction $\\hat y$ for one fixed context, using the weights saved at that training step. Two descents are nested here, and the slider moves only the outer one:</p>
    <div class="eq">$$\\theta_{t+1} = \\theta_t - \\eta\\,\\nabla_\\theta\\,\\mathcal L(\\theta_t)$$<span class="why">learning: reshapes the landscape. The slider moves t.</span></div>
    <div class="eq">$$\\hat y_{i+1} = \\hat y_i - \\alpha\\,\\nabla_{\\hat y}E_{\\theta_t}(x,\\hat y_i)$$<span class="why">thinking: moves on it. Blue paths: 40 steps, α = 0.5, from the same 8 starts.</span></div>
    <p>The left context ($x=0.25$) has nearly deterministic targets ($\\sigma=0.03$); the right one ($x=0.75$) is noisy ($\\sigma=0.30$). The crosshair is the clean mean $\\mu(x)$, the dashed ring has radius $\\sigma$. Training targets are $y=\\mu(x)+\\sigma(x)\\,\\varepsilon$, and the loss is $\\lVert\\hat y_N-y\\rVert^2$.</p>`,
  steps: [
    { label: 'Step 0: random weights', html: '<p>The untrained network adds a gentle random tilt to the small bowl $0.02\\lVert\\hat y\\rVert^2$. Each context gets a shallow minimum in an arbitrary place (readout), nowhere near $\\mu$, and descents creep toward it. After 4 steps the error to $\\mu$ averaged over 1024 contexts is 2.45, barely better than the random start (3.05).</p>' },
    { label: 'Steps 10–50: a basin is dragged toward the data', html: '<p>Each update backpropagates through short descents like these, and a basin slides toward $\\mu(x)$. At step 50 it is still off target and shallow, so 4 steps stop short: error 0.76, and the training loss is far above its floor (loss plot).</p>' },
    { label: 'Steps 100–200: the loss hits the noise floor', html: '<p>By step 100 the basin sits on $\\mu$. The error against the noisy targets after 4 steps is 0.079, essentially the noise floor $\\mathbb E\\lVert y-\\mu(x)\\rVert^2=0.0758$ (dashed). What is left is noise in the targets, which no predictor can remove. The training loss (blue) sits a little above it, because it is measured with Langevin noise and random strides on, and from here on it stays flat.</p>' },
    { label: 'Steps 400–12,000: refinement the loss cannot see', html: '<p>The loss no longer moves, yet the squared error between thinking\'s answer and the clean mean (hollow dots) keeps falling, with small bumps, from 0.0018 at step 200 to 0.0001 at the end, about 17× better. The loss cannot show this: in expectation it equals this error plus the noise floor, $\\mathbb E\\lVert\\hat y-y\\rVert^2=\\mathbb E\\lVert\\hat y-\\mu\\rVert^2+0.0758$, so a gain of 0.0017 hides inside its batch-to-batch jitter (about 0.01). The curvature at the minimum settles near 0.85 to 0.9 (readout; 0.78 to 0.92 across 64 contexts at the end), close to the 0.8 that the previous panel\'s stride analysis predicts for training with random step sizes. The depth $E_{\\min}$ drifts: no loss term pins absolute energy levels, only the shape is trained.</p>' },
    { label: 'Thinking improves with training', html: '<p>Bottom left: error versus thinking steps at each checkpoint (gray: all stored checkpoints; blue: the current one). Later checkpoints settle lower: after 40 steps, 0.0043 at step 100 versus 0.00012 at step 12,000. Training unrolls were only 2 to 6 steps long (shaded), yet extra steps do not wander off: the minimum is a fixed point, as the random number of steps intends.</p>' },
    { label: 'Verification keeps improving after the loss plateaus', html: '<p>Bottom right: Best-of-8. Run 8 noisy descents per context and keep the lowest-energy one (Algorithm 2, plus Langevin noise that Algorithm 2 itself does not include). Its gain over a single descent grows from 1.5× at step 25 to 7.8× at step 12,000, and most of that growth comes after the loss went flat: the energy keeps learning to rank near-misses it was never told about. The paper reports the same direction on one benchmark (Fig 6b, BigBench Dyck only): Best-of-5 gains rose from "4%−8%" to "10%−14%" with more training data (p.10). See below for how noisy that figure is.</p>' },
  ],
  after: `
    <h3>Three curves, three stories</h3>
    <p>The training loss compares predictions with noisy targets, so it bottoms out once the basin is centered. The error to the clean mean measures where thinking actually lands; it keeps improving for about 60× longer. The Best-of-N gain measures the energy as a verifier of its own candidates. A loss curve alone would have called training finished long before the landscape was.</p>
    <h3>Reading the paper's Fig 6b honestly</h3>
    <p>Fig 6b is from one benchmark (BigBench Dyck), and the authors "did not observe this trend in other benchmarks" (p.34). The scatter is noisy: individual points range from about 4.4% to 13.9%, and a linear fit rises only from about 7.9% to 10.6% (approx., read from Fig 6b). The toy agrees in direction, not in size. In the toy, plain descent without noise already reaches the noise floor, so its Best-of-N gain measures how well the energy picks the least-disturbed noisy candidate, not a gain over plain thinking.</p>
    <p class="note">Toy recipe: SiLU MLP 8→64→64→64→1 on [φ(x), ŷ], batch 256, Adam lr 0.002 with warmup and cosine decay, N ∈ 2..6 per batch, α = 1 with random factor 2 per example, Langevin σ = 0.05, replay p = 0.25. Heatmaps: stored 64×64 energies at each checkpoint, blended linearly only while the slider is between checkpoints, color range per frame. Eval: 1024 fresh contexts, ŷ₀ ~ N(0, I), α = 1, no noise. Best-of-8: random α as in training, 6 steps, Langevin 0.05. Minimum, E<sub>min</sub> and Hessian eigenvalues are computed live from the checkpoint weights.</p>`,
  source: [{ kind: 'toy', note: 'toy 2-D EBT, 12 real checkpoints' }, { kind: 'paper', note: 'Fig 6b direction only, p.10–11, p.34' }],
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
      const fr = lib.frame(row, { label: ci === 0 ? 'x = 0.25 · near-deterministic' : 'x = 0.75 · noisy', sub: 'σ = ' + c.sigma.toFixed(2) + ' · energy over ŷ at this step' });
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
    const FL = lib.frame(stage, { label: 'Training loss vs what thinking achieves', sub: 'log–log · click or drag to scrub' });
    const narrow = (stage.getBoundingClientRect().width || 700) < 520, LW = narrow ? 340 : 620, LH = narrow ? 210 : 182;
    const lc = lib.canvas(FL.frame, LW, LH, { label: 'Training loss curve with the noise floor and eval errors at each checkpoint, log-log, with a cursor at the chosen checkpoint' });
    const legL = h('div', { class: 'readout lrn-leg', html: '<span><i class="ln"></i>training loss (local mean; faint = single batches)</span><span><i class="dt"></i>eval error vs noisy y</span><span><i class="dt hol"></i>error to clean mean μ</span><span><i class="ln dash"></i>noise floor</span>' }); FL.wrap.appendChild(legL);
    const row3 = h('div', { class: 'fig-row' }); stage.appendChild(row3);
    const FT = lib.frame(row3, { label: 'Thinking, by checkpoint', sub: 'error to μ vs descent steps (log)' }); FT.wrap.style.flex = '1 1 250px';
    const tc = lib.canvas(FT.frame, 300, 200, { label: 'Error to the clean mean versus number of thinking steps, for every stored checkpoint' });
    const FV = lib.frame(row3, { label: 'Verification, by checkpoint', sub: 'Best-of-8 gain over one descent' }); FV.wrap.style.flex = '1 1 250px';
    const vc = lib.canvas(FV.frame, 300, 200, { label: 'Best-of-8 gain over a single descent versus training step' });
    const roT = h('div', { class: 'readout' }); FT.wrap.appendChild(roT);
    const roV = h('div', { class: 'readout' }); FV.wrap.appendChild(roV);

    // ---------------------------------------------------------------- per-checkpoint data
    const box = { x: 0, y: 0, w: 300, h: 300 };
    const P = (q) => M.toPx(q, box);
    const gridCache = {};
    const gridOf = (ci, k) => gridCache[ci + ':' + k] || (gridCache[ci + ':' + k] = M.storedGrid(ci, STEPS[k]));
    const statCache = {};
    function stats(ci, k) { // minimum polished by descent, E_min, Hessian eigenvalues by central differences (live, from weights)
      const key = ci + ':' + k; if (statCache[key]) return statCache[key];
      const g = gridOf(ci, k), n = g.length, x = M.contexts[ci].x; let best = Infinity, at = [0, 0];
      g.forEach((r, i) => r.forEach((v, j) => { if (v < best) { best = v; at = [EXT[0] + j * (EXT[1] - EXT[0]) / (n - 1), EXT[3] - i * (EXT[3] - EXT[2]) / (n - 1)]; } }));
      const mn = M.descend(k, x, at, { alpha: 0.3, steps: 80 }).path.pop();
      const e = 0.01, f = (a, b) => M.energy(k, x, [a, b]);
      const fxx = (f(mn[0] + e, mn[1]) - 2 * f(mn[0], mn[1]) + f(mn[0] - e, mn[1])) / (e * e);
      const fyy = (f(mn[0], mn[1] + e) - 2 * f(mn[0], mn[1]) + f(mn[0], mn[1] - e)) / (e * e);
      const fxy = (f(mn[0] + e, mn[1] + e) - f(mn[0] + e, mn[1] - e) - f(mn[0] - e, mn[1] + e) + f(mn[0] - e, mn[1] - e)) / (4 * e * e);
      const tr = (fxx + fyy) / 2, dt = Math.sqrt(Math.max(0, ((fxx - fyy) / 2) ** 2 + fxy * fxy));
      return (statCache[key] = { min: mn, E: f(mn[0], mn[1]), eig: [tr - dt, tr + dt] });
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
        pn.ro.innerHTML = `<span>min <b>(${st.min[0].toFixed(2)}, ${st.min[1].toFixed(2)})</b> vs μ (${m0[0].toFixed(2)}, ${m0[1].toFixed(2)})</span><span>E<sub>min</sub> <b>${st.E.toFixed(3)}</b></span><span>curvature <b>${st.eig[0].toFixed(2)}, ${st.eig[1].toFixed(2)}</b></span>`;
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
      lib.text(c, narrow ? 'training step (0 drawn at 1)' : 'training step (step 0 drawn at 1)', LA.x + LA.w / 2, LH - 18, { size: 11, kind: 'mono', color: '#6b6b70', align: 'center' });
    }
    function posFromX(px) { const v = Math.pow(10, (px - LA.x) / LA.w * Math.log10(12000)); const ls = STEPS.map(s => Math.log10(Math.max(1, s))), lv = Math.log10(Math.max(1, v)); if (lv <= ls[0]) return 0; for (let k = 0; k < nC - 1; k++) if (lv <= ls[k + 1]) return k + (ls[k + 1] - ls[k] > 0 ? (lv - ls[k]) / (ls[k + 1] - ls[k]) : 1); return nC - 1; }
    let dragging = false;
    const onPtr = (ev) => { const [px] = lc.toLocal(ev); stopAnim(); S.pos = posFromX(px); sl.set(S.pos); S.drawT = 1; drawAll(); };
    lc.canvas.style.cursor = 'ew-resize';
    lc.canvas.addEventListener('pointerdown', (ev) => { dragging = true; lc.canvas.setPointerCapture(ev.pointerId); onPtr(ev); });
    lc.canvas.addEventListener('pointermove', (ev) => { if (dragging) onPtr(ev); });
    lc.canvas.addEventListener('pointerup', () => { dragging = false; S.pos = Math.round(S.pos); sl.set(S.pos); S.drawT = 0; animatePaths(); });

    // thinking curves: x = steps i (drawn at i+1, log), y = error to mu (log)
    const TBC = D.thinking_by_ckpt, INIT = D.thinking_curve.mse_vs_mu[0];
    function drawThink() {
      const c = tc.ctx; tc.clear(); const kn = Math.round(S.pos);
      const ax = lib.axes(c, { x: 44, y: 8, w: 246, h: 150, xlim: [1, 41], ylim: [5e-5, 6], xlog: true, ylog: true, xticks: [1, 2, 3, 5, 11, 21, 41], yticks: [1e-4, 1e-3, 1e-2, 0.1, 1], xfmt: (v) => String(v - 1), yfmt: (v) => (v >= 0.1 ? String(v) : v.toExponential(0).replace('e-', 'e−')), size: 10.5 });
      c.fillStyle = C.blue4; c.fillRect(ax.X(3), 8, ax.X(7) - ax.X(3), 150); lib.text(c, 'training N', ax.X(3) + 3, 144, { size: 10, kind: 'mono', color: C.blue });
      TBC.forEach(t => lib.plot(c, ax, t.mse_vs_mu.map((v, i) => [i + 1, v]), { color: '#9a9aa0', width: 1, alpha: 0.6 }));
      const t = TBC.find(q => q.step === STEPS[kn]); const ev = D.eval.curve;
      if (t) { lib.plot(c, ax, t.mse_vs_mu.map((v, i) => [i + 1, v]), { color: C.blue, width: 2.2 }); roT.innerHTML = `<span>step <b>${fmtStep(STEPS[kn])}</b></span><span>after 4 steps <b>${fmtE(t.mse_vs_mu[4])}</b></span><span>after 40 <b>${fmtE(t.mse_vs_mu[40])}</b></span>`; }
      else { const pts = [[1, INIT], [5, ev.mse_vs_mu_N4[kn]], [41, ev.mse_vs_mu_N40[kn]]]; lib.plot(c, ax, pts, { color: C.blue, width: 1.4, dash: [4, 3], markers: 3.4 }); roT.innerHTML = `<span>step <b>${fmtStep(STEPS[kn])}</b></span><span>after 4 steps <b>${fmtE(ev.mse_vs_mu_N4[kn])}</b></span><span>after 40 <b>${fmtE(ev.mse_vs_mu_N40[kn])}</b></span><span>(full curves stored from step 100)</span>`; }
      lib.text(c, 'thinking steps i', 167, 184, { size: 10.5, kind: 'mono', color: '#6b6b70', align: 'center' });
    }
    const fmtE = (v) => (v >= 0.01 ? v.toFixed(3) : v.toExponential(1).replace('e-', 'e−'));
    // verification: gain = error(M=1) / error(M=8) by training step
    const BV = D.bon_vs_training, GAIN = BV.step.map((s, i) => BV.M1[i] / BV.M8[i]);
    function drawVerif() {
      const c = vc.ctx; vc.clear(); const kn = Math.round(S.pos);
      const ax = lib.axes(c, { x: 36, y: 8, w: 254, h: 150, xlim: [10, 15000], ylim: [0, 8.5], xlog: true, xticks: [10, 100, 1000, 10000], yticks: [0, 2, 4, 6, 8], xfmt: (v) => (v >= 1000 ? v / 1000 + 'k' : String(v)), yfmt: (v) => v + '×', size: 10.5 });
      c.save(); c.strokeStyle = 'rgba(17,17,17,0.4)'; c.setLineDash([2, 3]); c.beginPath(); c.moveTo(ax.X(10), ax.Y(1)); c.lineTo(ax.X(15000), ax.Y(1)); c.stroke(); c.restore();
      lib.text(c, 'no gain', ax.X(15000) - 3, ax.Y(1) + 3, { size: 10, kind: 'mono', color: '#6b6b70', align: 'right' });
      lib.plot(c, ax, BV.step.map((s, i) => [s, GAIN[i]]), { color: C.blue, width: 1.8, markers: 3 });
      const k0 = Math.floor(S.pos + 1e-9), k1 = Math.min(nC - 1, k0 + 1), f = S.pos - k0;
      const l0 = Math.log10(Math.max(10, STEPS[k0])), l1 = Math.log10(Math.max(10, STEPS[k1])), X = ax.X(Math.pow(10, l0 + (l1 - l0) * f));
      c.save(); c.strokeStyle = C.blue; c.lineWidth = 1.2; c.beginPath(); c.moveTo(X, 8); c.lineTo(X, 158); c.stroke(); c.restore();
      const j = BV.step.indexOf(STEPS[kn]);
      if (j >= 0) { lib.dot(c, ax.X(BV.step[j]), ax.Y(GAIN[j]), 5, C.blue, { stroke: '#fff', lw: 1.5 }); roV.innerHTML = `<span>M = 1 error <b>${fmtE(BV.M1[j])}</b></span><span>M = 8 <b>${fmtE(BV.M8[j])}</b></span><span>gain <b>${GAIN[j].toFixed(1)}×</b></span>`; }
      else roV.innerHTML = '<span>no Best-of-N measurement before step 25</span>';
      lib.text(c, 'training step', 163, 184, { size: 10.5, kind: 'mono', color: '#6b6b70', align: 'center' });
    }
    function drawAll() { panes.forEach(drawPane); drawLoss(); drawThink(); drawVerif(); }

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
    const hl = (which) => [FL, FT, FV].forEach(F => F.frame.classList.toggle('lrn-hl', F === which));
    const caps = [
      'Energy over ŷ for two contexts at the selected training step (darker = lower; color range per frame, printed bottom right). Crosshair: clean mean μ(x); dashed ring: noise σ; small black ring: the landscape\'s minimum. Blue: 40-step descents from 8 fixed starts at that checkpoint.',
      'Scrubbing from step 0 to step 50. Between checkpoints the heatmap is a blend of the two stored grids; descents are shown at checkpoints only.',
      'Loss plot: blue = training loss, black dots = eval error against noisy targets after 4 steps, dashed = noise floor.',
      'Hollow dots: eval error to the clean mean μ after 4 steps. It keeps falling after the loss and the black dots have gone flat.',
      'Thinking by checkpoint. Gray: error to μ versus descent steps for each stored checkpoint (from step 100). Blue: the current checkpoint. Shaded: unroll lengths used in training.',
      'Best-of-8 gain = error of one noisy descent ÷ error of the lowest-energy of 8, on 1024 eval contexts, at each checkpoint.',
    ];
    drawAll();
    return {
      step(i) {
        ctx.setCaption(caps[i]); stopAnim();
        hl([null, null, FL, FL, FT, FV][i]);
        if (i === 0) { S.pos = 0; sl.set(0); animatePaths(); }
        if (i === 1) animateTo(0, 3, 0.6, true);
        if (i === 2) animateTo(3, 5, 0.7, true);
        if (i === 3) animateTo(5, 11, 0.3, true);
        if (i === 4) animateTo(4, 11, 0.28, true);
        if (i === 5) animateTo(2, 11, 0.22, true);
        drawAll();
      },
      show() { if (S.anim || S.drawT < 1) loop.start(); },
      hide() { loop.stop(); },
    };
  },
});
