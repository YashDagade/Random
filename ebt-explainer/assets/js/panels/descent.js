/* Panel: thinking = gradient descent on the prediction (reference panel written by the lead). */
EBT.panel({
  id: 'descent',
  nav: 'Thinking is gradient descent',
  title: 'Thinking is gradient descent on the prediction',
  lede: 'An EBT never outputs an answer directly. It starts from noise and improves its guess by following the slope of its own energy, one forward and backward pass at a time.',
  text: `
    <p>Fix the context $x$ and the weights $\\theta$. The energy $E_\\theta(x,\\hat y)$ is then just a function of the candidate prediction $\\hat y$, a surface over the space of possible answers. Low regions are answers the model judges compatible with the context.</p>
    <p>To predict, the EBT does what any optimizer would do on such a surface. It computes the gradient of the energy with respect to the <em>prediction</em> (not the weights) and steps downhill:</p>
    <div class="eq">$$\\hat y_{i+1} \\;=\\; \\hat y_i \\;-\\; \\alpha\\, \\nabla_{\\hat y}\\, E_\\theta(x, \\hat y_i), \\qquad \\hat y_0 \\sim \\mathcal N(0, I)$$<span class="why">Eq. 1 of the paper (p.7). α is the step size.</span></div>
    <p>Each step costs one forward pass (to get $E$) plus one backward pass to the input (to get $\\nabla_{\\hat y}E$). The paper counts thinking in <b>function evaluations</b> (NFEs): one per optimization step.</p>`,
  steps: [
    { label: 'Start from noise', html: '<p>The first guess is random: $\\hat y_0 \\sim \\mathcal N(0, I)$. Nothing about the context is used yet. Press <b>[ new start ]</b> to draw another.</p>' },
    { label: 'Ask the verifier', html: '<p>One forward pass returns a single number, the energy of this guess. Backpropagating that number to the input gives $\\nabla_{\\hat y}E$, the direction in which the guess gets <em>worse</em> fastest. The arrow shows $-\\alpha\\nabla_{\\hat y}E$.</p>' },
    { label: 'Take one step', html: '<p>Move the guess along the arrow. Energy drops. This single update is one unit of "thinking".</p>' },
    { label: 'Repeat until it settles', html: '<p>Keep stepping. When the energy stops falling, $|E_{i+1}-E_i| < \\varepsilon$, the model has converged and can stop. A harder context would need more steps: this is <b>dynamic compute</b>, Facet 1 of the paper.</p>' },
    { label: 'Step size matters', html: '<p>Too small an $\\alpha$ crawls. Too large an $\\alpha$ overshoots and bounces across the valley. Drag the α slider and run again. The paper randomizes α during training so the learned landscape works for a range of step sizes (Sec 3.3).</p>' },
  ],
  after: `
    <h3>Why this is different</h3>
    <p>A Transformer++ spends exactly one forward pass per token, whatever the difficulty. Here the number of steps is a dial you can turn at inference time, and every intermediate guess comes with a score. Those two properties are what the rest of the paper builds on.</p>
    <p class="note">The landscape is a real model: a 2-D toy EBT trained for this explainer with the paper's recipe (unrolled steps, Langevin noise, replay buffer, random α and step count). Energies and gradients are computed live in your browser from its weights.</p>`,
  source: [{ kind: 'toy', note: 'toy 2-D EBT, final checkpoint' }, { kind: 'paper', note: 'Eq. 1, Alg. 2' }],
  figure(stage, ctx) {
    const { lib } = ctx, h = lib.h, M = window.EBT.toy2d, C = lib.C;
    if (!M || !M.ready) { stage.appendChild(h('p', { class: 'callout warn' }, 'Toy model data missing (data/toy2d.json).')); return {}; }
    const K = M.nCkpt - 1, ext = M.extent;
    const ctxs = M.contexts.slice(0, 3);
    let ci = 0, alpha = 1.0, y = null, path = [], energies = [], showArrow = false, timer = null;
    const row = h('div', { class: 'fig-row' }); stage.appendChild(row);
    const L = lib.frame(row, { label: 'Energy landscape', sub: 'E<sub>θ</sub>(x, ŷ) over ŷ ∈ ℝ² · click to place ŷ<sub>0</sub>' });
    L.wrap.style.flex = '1 1 360px';
    const cv = lib.canvas(L.frame, 560, 560, { label: 'Energy landscape heatmap with the current prediction and its gradient-descent path' });
    const R = lib.frame(row, { label: 'Energy per step', sub: 'E(x, ŷ<sub>i</sub>) as the model thinks' });
    R.wrap.style.flex = '1 1 240px';
    const pv = lib.canvas(R.frame, 320, 260, { label: 'Energy at each thinking step' });
    const ro = h('div', { class: 'readout' }); R.wrap.appendChild(ro);
    const box = { x: 0, y: 0, w: 560, h: 560 };
    const grids = {};
    const gridFor = (k) => grids[k] || (grids[k] = M.grid(K, ctxs[k].x, 96));
    function draw() {
      const g = gridFor(ci), c = cv.ctx;
      cv.clear();
      if (g.q == null) g.q = lib.quantile(g.E, 0.55);
      lib.heatmap(c, g.E, 0, 0, 560, 560, { key: 'descent-' + ci, range: [g.lo, g.q], gamma: 0.7 });
      const levels = Array.from({ length: 10 }, (_, i) => g.lo + (g.q - g.lo) * Math.pow((i + 1) / 10, 1.4));
      lib.contours(c, g.E, 0, 0, 560, 560, levels, { color: 'rgba(17,17,17,0.22)', width: 1 });
      // target mean (crosshair) for this context
      const mu = M.toPx(ctxs[ci].mu, box);
      c.save(); c.strokeStyle = '#111'; c.lineWidth = 1.2; c.setLineDash([3, 3]); c.beginPath(); c.arc(mu[0], mu[1], 14, 0, 7); c.stroke(); c.setLineDash([]);
      c.beginPath(); c.moveTo(mu[0] - 7, mu[1]); c.lineTo(mu[0] + 7, mu[1]); c.moveTo(mu[0], mu[1] - 7); c.lineTo(mu[0], mu[1] + 7); c.stroke(); c.restore();
      lib.text(c, 'data mean', mu[0] + 18, mu[1] + 12, { size: 12, kind: 'mono', color: '#fff' });
      if (path.length) {
        const px = path.map(p => M.toPx(p, box));
        lib.line(c, px, { color: C.blue, width: 2 });
        px.forEach((p, i) => lib.dot(c, p[0], p[1], i === px.length - 1 ? 6 : 3, i === px.length - 1 ? C.blue : '#fff', { stroke: C.blue, lw: 1.5 }));
        lib.text(c, 'ŷ' + sub(path.length - 1), px[px.length - 1][0] + 10, px[px.length - 1][1] + 6, { size: 14, kind: 'mono', color: C.blue });
        if (showArrow) {
          const { g: gr } = M.energyGrad(K, ctxs[ci].x, y);
          const tgt = [y[0] - alpha * gr[0], y[1] - alpha * gr[1]], a = px[px.length - 1], b = M.toPx(tgt, box);
          lib.arrow(c, a[0], a[1], b[0], b[1], { color: '#111', width: 1.6, head: 9, dash: [5, 4] });
          lib.text(c, '−α∇E', (a[0] + b[0]) / 2 + 8, (a[1] + b[1]) / 2 - 18, { size: 13, kind: 'mono', color: '#111' });
        }
      }
      // axes ticks
      lib.text(c, 'ŷ₁ →', 548, 540, { size: 12, kind: 'mono', color: '#fff', align: 'right' });
      drawPlot(); readout();
    }
    const sub = (n) => String(n).split('').map(d => '₀₁₂₃₄₅₆₇₈₉'[+d]).join('');
    function drawPlot() {
      const c = pv.ctx; pv.clear();
      const n = Math.max(8, energies.length);
      const lo = Math.min(...energies, gridFor(ci).lo), hi = Math.max(...energies, gridFor(ci).lo + 0.5);
      const ax = lib.axes(c, { x: 46, y: 14, w: 260, h: 200, xlim: [0, n - 1], ylim: [lo, hi], xticks: [0, Math.round((n - 1) / 2), n - 1], yticks: [lo, hi], yfmt: (v) => v.toFixed(2), xlabel: 'step i', size: 11 });
      if (energies.length) { lib.plot(c, ax, energies.map((e, i) => [i, e]), { color: C.blue, width: 2, markers: 3 }); }
    }
    function readout() {
      if (!y) { ro.innerHTML = ''; return; }
      const { E, g } = M.energyGrad(K, ctxs[ci].x, y); const mu = ctxs[ci].mu;
      ro.innerHTML = `<span>step <b>${path.length - 1}</b></span><span>E <b>${E.toFixed(3)}</b></span><span>‖∇E‖ <b>${Math.hypot(g[0], g[1]).toFixed(3)}</b></span><span>dist to mean <b>${Math.hypot(y[0] - mu[0], y[1] - mu[1]).toFixed(3)}</b></span><span>NFEs <b>${path.length - 1}</b></span>`;
    }
    let seed = 7;
    const rnd = () => { const r = lib.rng(seed++ * 9973); return [r.normal() * 1.2, r.normal() * 1.2]; };
    function start(p) { stop(); y = p || rnd(); path = [y.slice()]; energies = [M.energy(K, ctxs[ci].x, y)]; draw(); }
    function stepOnce() { if (!y) start(); const { g } = M.energyGrad(K, ctxs[ci].x, y); y = [y[0] - alpha * g[0], y[1] - alpha * g[1]]; y = y.map(v => Math.max(ext[0] * 1.6, Math.min(ext[1] * 1.6, v))); path.push(y.slice()); energies.push(M.energy(K, ctxs[ci].x, y)); draw(); }
    function run(n = 25) { stop(); let k = 0; timer = setInterval(() => { stepOnce(); k++; const m = energies.length; if (k >= n || (m > 3 && Math.abs(energies[m - 1] - energies[m - 2]) < 1e-5)) stop(); }, lib.reducedMotion ? 0 : 160); }
    function stop() { if (timer) clearInterval(timer); timer = null; }
    cv.canvas.addEventListener('click', (ev) => { const [px, py] = cv.toLocal(ev); start(M.fromPx(px, py, box)); showArrow = true; draw(); ctx.goStep(1); });
    const controls = h('div', { class: 'controls' }); stage.appendChild(controls);
    const seg = lib.segmented({ label: 'Context', options: ctxs.map((c, i) => [i, 'x=' + c.x + ' · σ ' + c.sigma]), value: 0, onchange: (v) => { ci = v; start(); } });
    controls.appendChild(h('span', { class: 'fig-label' }, 'context'));
    controls.appendChild(seg.el);
    const sl = lib.slider({ id: 'descent-alpha', label: 'step size α', min: 0.05, max: 2.6, step: 0.05, value: alpha, fmt: (v) => v.toFixed(2), oninput: (v) => { alpha = v; draw(); } });
    const controls2 = h('div', { class: 'controls' }); stage.appendChild(controls2);
    controls2.appendChild(lib.button('new start', () => { start(); ctx.goStep(0); }));
    controls2.appendChild(lib.button('step', () => { showArrow = true; stepOnce(); }, { primary: true }));
    controls2.appendChild(lib.button('run', () => { showArrow = true; run(); }));
    controls2.appendChild(sl.el);
    ctx.setCaption('Heatmap: energy of every candidate ŷ for one fixed context (darker blue = lower energy). Dashed circle: the mean of the training targets for this context. Blue path: the prediction as it thinks.');
    start([-1.7, 1.6]);
    return {
      step(i) {
        stop();
        if (i === 0) { showArrow = false; start([-1.7, 1.6]); }
        if (i === 1) { showArrow = true; if (path.length > 1) start([-1.7, 1.6]); draw(); }
        if (i === 2) { showArrow = true; if (path.length > 1) start([-1.7, 1.6]); alpha = 1.0; sl.set(1.0); stepOnce(); }
        if (i === 3) { showArrow = true; alpha = 1.0; sl.set(1.0); start([-1.7, 1.6]); run(); }
        if (i === 4) { showArrow = true; alpha = 2.4; sl.set(2.4); start([-1.7, 1.6]); run(18); }
      },
      hide() { stop(); },
    };
  },
});
