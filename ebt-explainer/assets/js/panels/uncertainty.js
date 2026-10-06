/* Panel: where uncertainty lives in an EBT, and where it does not.
   Real computation: the final toy 2-D EBT (data/toy2d.json) is evaluated live (energies, gradients, finite-difference Hessians,
   thinking runs); toy2d.basin_stats gives the 64-context sweep; text.json gives the char-level text toy's easy/hard energies;
   paper_heatmaps.json gives digitized Figs 8, 11, B.2 (approx.). */
(function () {
  EBT.panel({
    id: 'uncertainty',
    nav: 'Where uncertainty lives',
    title: 'Where uncertainty lives (and where it does not)',
    lede: 'The paper reads the energy as an uncertainty signal (Facet 2). Our toy shows that Algorithm 1 does not force this: the noise level in the data sets neither the width nor the height of the basin. Here is why, what sets the basin instead, and what the paper actually shows.',
    text: `
      <p>Facet 2 is "Modeling Uncertainty in Continuous State Spaces" (p.3): EBMs model "the relative unnormalized likelihoods of predictions". Read $e^{-E_\\theta(x,\\hat y)}$ as an unnormalized density over predictions. Near its minimum $\\hat y^*$ the energy is locally quadratic, so that density is a Gaussian:</p>
      <div class="eq">$$\\begin{aligned}E_\\theta(x,\\hat y)&\\approx E^*+\\tfrac12(\\hat y-\\hat y^*)^\\top H\\,(\\hat y-\\hat y^*)\\\\ \\Rightarrow\\ e^{-E_\\theta}&\\propto\\mathcal N\\big(\\hat y^*,\\,H^{-1}\\big)\\end{aligned}$$<span class="why">H: Hessian of E in ŷ at the minimum, eigenvalues λ. Width along each eigenvector ∝ λ^(−1/2). Relative only: e^(−E) is unnormalized.</span></div>
      <p>Our toy has a built-in test. Its targets are $y=\\mu(x)+\\sigma(x)\\,\\varepsilon$, $\\varepsilon\\sim\\mathcal N(0,I)$, with $\\sigma$ from 0.03 at $x=0.25$ to 0.30 at $x=0.75$. If the energy were the negative log-density of the targets, the curvature would be $\\lambda=1/\\sigma^2$: 1,111 at one tip and 11 at the other.</p>`,
    steps: [
      { label: 'What a density would look like', html: '<p>Both panels show the energy a true density model would have, $\\|\\hat y-\\mu\\|^2/2\\sigma^2$, on one color scale (0 at the minimum). Circles are target samples from the toy\'s data distribution. Left: a pinpoint pit. Right: a basin 10 times wider. Below: the curvature $1/\\sigma^2$ this needs at every context (log scale).</p>' },
      { label: 'What the toy actually learned', html: '<p>Now the learned energy, computed live from the weights. Both basins are the same bowl: the blue contour $\\Delta E=\\tfrac12$ has radius about 1.1 in both, the density contour (dashed) has radius $\\sigma$. Live Hessians give $\\lambda\\approx 0.85$ to $0.91$ at both tips, and over 64 contexts $\\lambda$ stays in 0.78 to 0.92 while $1/\\sigma^2$ moves 100×. Blue dots: 24 thinking runs from random starts, all on one point. A control run with the noise pattern mirrored looks the same.</p>' },
      { label: 'Squared error only sees the mean', html: '<p>The toy is trained with squared error on the last step. Averaged over the noisy target of one context, $$\\mathbb E_y\\|\\hat y_N-y\\|^2=\\|\\hat y_N-\\mu(x)\\|^2+2\\sigma(x)^2.$$ The noise adds a constant, so the gradient $2(\\hat y_N-\\mu)$ is the same for every $\\sigma$ (parallel tangents below). The best $\\hat y_N$ is the mean, and nothing in the expected signal asks the basin to widen. A larger $\\sigma$ only makes per-sample gradients noisier.</p>' },
      { label: 'Only the slope of E is trained', html: '<p>Algorithm 1 touches the energy only through $\\nabla_{\\hat y}E$. Replace $E$ by $E+c(x)$ for any function of the context: $\\nabla_{\\hat y}c(x)=0$, so every thinking path, every $\\hat y_N$ and the loss are identical. The level of one context against another is never trained. Drag $k$: the correlation of noise with minimum energy changes sign at identical loss (our runs gave 0.45 at the end, −0.03 at step 5,000, −0.19 in the control). Best-of-N is safe: it compares energies within one context, where $c(x)$ cancels.</p>' },
      { label: 'The step size sets the curvature', html: '<p>Near a minimum, one step multiplies the error along an eigenvector by $1-\\alpha\\lambda$. The toy trains with $\\alpha\\in[0.5,2]$ (log-uniform) and 2 to 6 steps, so the expected leftover is $r(\\lambda)=\\mathbb E_{\\alpha,N}(1-\\alpha\\lambda)^{2N}$. Numerically it is smallest at $\\lambda^*=0.80$, which is $2/(\\alpha_{\\min}+\\alpha_{\\max})$, the curvature at which the smallest and the largest step overshoot equally (our derivation). The toy sits right there. A density-shaped bowl for $\\sigma=0.3$ ($\\lambda=11$) would multiply the error by up to 21 per step. The optimizer chooses the width. Drag $\\alpha_0$ and $\\lambda^*$ follows as $\\approx 0.8/\\alpha_0$. The paper saw the same coupling from the other side: "a smaller step size results in larger generated gradients, whereas a larger step size results in smaller gradients" (p.42).</p>' },
      { label: 'Where the paper sees uncertainty', html: '<p>Figs 8, 11 and B.2, redrawn from digitized values (the paper draws high energy dark; here darker means lower, as elsewhere on this site). Hard tokens such as "quick", "research", "problem" stay higher; easy ones such as ".", "is", "but" go lower (p.10–12). Step [4] applies to the paper too: no loss ever touches energy values (Fig E.1, p.37), so these cross-token levels are emergent. That makes the pattern interesting, and not guaranteed. Fig 11 is the video EBT, trained with Smooth-L1, a regression loss like our toy\'s, yet its frame levels differ: levels the loss leaves free can still end up tracking difficulty (our reading). Caveats: the normalization is not stated, iteration-0 values already differ, and almost all change happens at iteration 1. For text, cross-entropy also puts uncertainty in <em>where</em> the minimum is: the optimal $\\mathrm{softmax}(\\hat y)$ is the predicted distribution itself.</p>' },
      { label: 'Checking it on our text toy', html: '<p>Our character-level text EBT (cross-entropy, Algorithm 1, $\\alpha=10$, 8 thinking steps) gives a mixed answer. Raw final energy is slightly <em>lower</em> for hard characters (first letter of a word, −4.07) than for easy ones (−3.97): the opposite of Fig 8. Subtracting each context\'s own reference, $E(x,\\hat y)-E(x,\\text{uniform})$ (our normalization, not the paper\'s), restores the paper\'s order (hard −2.91, easy −3.19). The entropy of the prediction tracks per-character loss better (Spearman 0.58) than energy does (0.29): in this toy, uncertainty lives mostly in where the minimum is, not in how high it sits.</p>' },
    ],
    after: `
      <h3>What to take away</h3>
      <p>Within one context the energy is a grounded verifier: differences between candidates are exactly what training shapes, which is why Best-of-N works. Across contexts the level is a free constant of the objective. It may come to track difficulty, as the paper reports, but Algorithm 1 does not guarantee it. Pinning levels down would need a term that constrains them, such as a contrastive or likelihood objective (our suggestion, not the paper's), which the paper avoids for scalability (p.6–7).</p>
      <p>The paper's Fig 3 caption offers a second reading: uncertainty "can be represented by landscapes that are harder to optimize or by landscapes with many local minima" (p.5). The toy shows neither. Every basin is one smooth bowl, and the noisiest context converges in fewer steps than the cleanest (2.2 vs 2.6 gradient steps on average to come within $10^{-3}$ of $E_{\\min}$; across 64 contexts the correlation of steps with $\\sigma$ is −0.27).</p>
      <p>This is the continuous-space problem of Sec 6.1: for feed-forward models "the normalization process for continuous state spaces is not as well-defined as it is for discrete spaces using softmax" (p.15). An MSE-trained regressor outputs $\\mu(x)$ and nothing else. An MSE-trained EBT, as here, inherits the same blind spot.</p>
      <p class="note">Toy: 2-D EBT, Algorithm 1 with MSE, N ∈ 2..6, α ∈ [0.5, 2], Langevin 0.05, replay buffer. basin_stats: 64 contexts, argmin by 300 GD steps, 2×2 Hessian. Control: 6,000-step run with the noise mirrored; the lower E_min stays at x = 0.25 in both runs, so it follows geometry, not noise. Text toy: 54-symbol character model; easy = 3rd+ letter of a word or 'u' after 'q', hard = first letter of a word.</p>`,
    source: [{ kind: 'toy', note: '2-D EBT final checkpoint (live), basin_stats, text toy uncertainty' }, { kind: 'paper', note: 'Figs 8, 11, B.2 digitized (approx.); Facet 2 (p.3)' }],

    figure(stage, ctx) {
      const { lib } = ctx, h = lib.h, C = lib.C, M = window.EBT.toy2d, D = M && M.data;
      if (!M || !M.ready) { stage.appendChild(h('p', { class: 'callout warn' }, 'Toy model data missing (data/toy2d.json).')); return {}; }
      const BS = D.basin_stats, TX = (lib.data('text') || {}).uncertainty, PH = ((lib.data('paper_heatmaps') || {}).heatmaps) || [];
      const K = M.nCkpt - 1, LAM = D.arch.quadratic_term.lambda_, KF = D.arch.feature_map.K, HID = 64;
      const muOf = (x) => [1.3 * Math.sin(2 * Math.PI * x), 0.9 * Math.sin(4 * Math.PI * x)];      // toy2d.dataset.mu_formula
      const sigOf = (x) => 0.03 + 0.27 * Math.pow(Math.sin(Math.PI * (x - 0.25)), 2);              // toy2d.dataset.sigma_formula
      const HP = D.hparams, AF = HP.alpha_rand_factor || 2, NMIN = HP.n_min || 2, NMAX = HP.n_max || 6;

      // ---------------- fast evaluator for the final checkpoint ----------------
      const NET = D.weights[K].layers.map(l => ({ no: l.W.length, ni: l.W[0].length, W: Float64Array.from([].concat(...l.W)), b: Float64Array.from(l.b) }));
      const z0 = new Float64Array(HID), h1 = new Float64Array(HID), z1 = new Float64Array(HID), h2 = new Float64Array(HID), z2 = new Float64Array(HID), h3 = new Float64Array(HID);
      const gA = new Float64Array(HID), gB = new Float64Array(HID), gC = new Float64Array(HID);
      const sg = (z) => 1 / (1 + Math.exp(-z));
      function ctxConst(x) { const L = NET[0], c = new Float64Array(HID), f = []; for (let k = 1; k <= KF; k++) f.push(Math.sin(2 * Math.PI * k * x), Math.cos(2 * Math.PI * k * x));
        for (let o = 0; o < HID; o++) { let s = L.b[o]; for (let i = 0; i < f.length; i++) s += L.W[o * L.ni + i] * f[i]; c[o] = s; } return c; }
      function energy(c0, y0, y1, grad) {
        const [L0, L1, L2, L3] = NET, n0 = L0.ni, W0 = L0.W, W1 = L1.W, W2 = L2.W, W3 = L3.W;
        for (let o = 0; o < HID; o++) { const z = c0[o] + W0[o * n0 + n0 - 2] * y0 + W0[o * n0 + n0 - 1] * y1; z0[o] = z; h1[o] = z * sg(z); }
        for (let o = 0; o < HID; o++) { let s = L1.b[o]; const off = o * HID; for (let i = 0; i < HID; i++) s += W1[off + i] * h1[i]; z1[o] = s; h2[o] = s * sg(s); }
        for (let o = 0; o < HID; o++) { let s = L2.b[o]; const off = o * HID; for (let i = 0; i < HID; i++) s += W2[off + i] * h2[i]; z2[o] = s; h3[o] = s * sg(s); }
        let E = L3.b[0]; for (let i = 0; i < HID; i++) E += W3[i] * h3[i]; E += LAM * (y0 * y0 + y1 * y1);
        if (!grad) return E;
        for (let i = 0; i < HID; i++) { const s = sg(z2[i]); gA[i] = W3[i] * s * (1 + z2[i] * (1 - s)); }
        gB.fill(0); for (let o = 0; o < HID; o++) { const g = gA[o], off = o * HID; for (let j = 0; j < HID; j++) gB[j] += W2[off + j] * g; }
        for (let j = 0; j < HID; j++) { const s = sg(z1[j]); gB[j] *= s * (1 + z1[j] * (1 - s)); }
        gC.fill(0); for (let o = 0; o < HID; o++) { const g = gB[o], off = o * HID; for (let j = 0; j < HID; j++) gC[j] += W1[off + j] * g; }
        for (let j = 0; j < HID; j++) { const s = sg(z0[j]); gC[j] *= s * (1 + z0[j] * (1 - s)); }
        let g0 = 0, g1 = 0; for (let o = 0; o < HID; o++) { g0 += W0[o * n0 + n0 - 2] * gC[o]; g1 += W0[o * n0 + n0 - 1] * gC[o]; }
        return [E, g0 + 2 * LAM * y0, g1 + 2 * LAM * y1];
      }
      // basin of context x: argmin by GD (α 0.5) from μ(x) as in basin_stats (90 steps: the error shrinks ×0.58 per step, so this is converged); Hessian by central differences of ∇E
      function basin(x, steps) {
        const c0 = ctxConst(x); let y = muOf(x);
        for (let i = 0; i < (steps || 90); i++) { const r = energy(c0, y[0], y[1], true); y = [y[0] - 0.5 * r[1], y[1] - 0.5 * r[2]]; }
        const e = 1e-3, g = (p) => { const r = energy(c0, p[0], p[1], true); return [r[1], r[2]]; };
        const ax = g([y[0] + e, y[1]]), bx = g([y[0] - e, y[1]]), ay = g([y[0], y[1] + e]), by = g([y[0], y[1] - e]);
        const a = (ax[0] - bx[0]) / (2 * e), d = (ay[1] - by[1]) / (2 * e), b = ((ax[1] - bx[1]) + (ay[0] - by[0])) / (4 * e);
        const tr = (a + d) / 2, disc = Math.sqrt(Math.max(0, tr * tr - (a * d - b * b)));
        return { x, c0, y, E: energy(c0, y[0], y[1], false), lam: [tr - disc, tr + disc], mu: muOf(x), sigma: sigOf(x) };
      }
      const r0 = lib.rng(31), STARTS = Array.from({ length: 24 }, () => [r0.normal(), r0.normal()]);
      const EPS = [lib.rng(11), lib.rng(12)].map(r => Array.from({ length: 60 }, () => [r.normal(), r.normal()]));
      function endpoints(B) { return STARTS.map(s => { let y = s.slice(); for (let i = 0; i < 4; i++) { const r = energy(B.c0, y[0], y[1], true); y = [y[0] - r[1], y[1] - r[2]]; } return y; }); }

      // ---------------- layout ----------------
      const S = { mode: 'ideal', view: 'curv', xB: 0.75, k: 0, a0: 1, fig: 'fig8a', ends: false, path: false, band: false };
      const row = h('div', { class: 'fig-row', style: { columnGap: '18px' } }); stage.appendChild(row);
      const mk = (label, sub) => { const F = lib.frame(row, { label, sub }); F.wrap.style.flex = '1 1 225px'; const cv = lib.canvas(F.frame, 300, 300, { label: 'Energy around the minimum for one context, with target samples' }); const ro = h('div', { class: 'readout' }); F.wrap.appendChild(ro); return { F, cv, ro }; };
      const FA = mk('Low-noise context · A', 'x = 0.25 · σ = 0.03 · window μ ± 1.5');
      const FB = mk('Second context · B', '&nbsp;');
      const subB = FB.F.wrap.querySelector('.fig-sub');
      const ctl1 = h('div', { class: 'controls' }); stage.appendChild(ctl1);
      const segM = lib.segmented({ label: 'Energy shown', options: [['ideal', 'density model'], ['learned', 'learned E_θ (toy)']], value: S.mode, onchange: (v) => { S.mode = v; drawFrames(); } });
      ctl1.appendChild(segM.el);
      const slB = lib.slider({ id: 'unc-xB', label: 'context x of B', min: 0, max: 0.995, step: 0.005, value: S.xB, fmt: (v) => v.toFixed(3) + ' · σ ' + sigOf(v).toFixed(3), oninput: (v) => setB(v, true) });
      slB.input.addEventListener('change', () => setB(slB.value(), false));
      ctl1.appendChild(slB.el);

      const W = Math.round(Math.max(320, Math.min(640, (stage.clientWidth || 600) - 2))), H = 244;
      const FC = lib.frame(stage, { label: 'The evidence, one chart per step', sub: 'click the curvature or level chart to move B' });
      const tabs = lib.segmented({ label: 'Chart', options: [['curv', 'curvature'], ['mse', 'MSE'], ['level', 'energy level'], ['alpha', 'step size'], ['paper', 'paper figs'], ['text', 'text toy']], value: S.view, onchange: (v) => { S.view = v; S.band = S.band || v !== 'curv'; drawBottom(); } });
      FC.frame.appendChild(h('div', { style: { padding: '8px 10px 0' } }, tabs.el));
      const cvC = lib.canvas(FC.frame, W, H, { label: 'Chart for the current step' });
      const ctl2 = h('div', { class: 'controls', style: { padding: '0 10px 8px' } }); FC.frame.appendChild(ctl2);
      const slK = lib.slider({ id: 'unc-k', label: 'add c(x) = k·sin 2πx, k', min: -0.4, max: 0.4, step: 0.01, value: 0, fmt: (v) => v.toFixed(2), oninput: (v) => { S.k = v; drawFrames(); drawBottom(); } });
      const slA = lib.slider({ id: 'unc-a0', label: 'training step size α₀', min: -2, max: 2, step: 0.05, value: 0, fmt: (v) => Math.pow(2, v).toFixed(2), oninput: (v) => { S.a0 = Math.pow(2, v); drawBottom(); } });
      const figOpts = [['fig8a', 'Fig 8a'], ['fig8b', 'Fig 8b'], ['fig11', 'Fig 11'], ['figB2', 'Fig B.2']].filter(o => o[0] === 'figB2' ? PH.some(p => p.id === 'figB2_seq1') : PH.some(p => p.id === o[0]));
      const segF = lib.segmented({ label: 'Paper figure', options: figOpts, value: S.fig, onchange: (v) => { S.fig = v; drawBottom(); } });
      ctl2.append(slK.el, slA.el, segF.el);
      const roC = h('div', { class: 'readout', style: { padding: '0 10px 10px' } }); FC.frame.appendChild(roC);
      const CAP = {
        ideal: 'Top: the energy a density model of the targets would have, ΔE = ‖ŷ − μ‖²/2σ², for two contexts on one scale from 0 (dark) to 1.2 (white). Circles: target samples y. Crosshair: target mean μ(x). Black circle: ΔE = ½ (radius σ). Scale bar: 0.5 units of ŷ.',
        learned: 'Top: the learned energy around its minimum for two contexts, on a common scale ΔE = E − E<sub>min</sub> from 0 (dark) to 1.2 (white). Circles: target samples y. Crosshair: target mean μ(x). Blue contour: learned ΔE = ½. Dashed: where ΔE = ½ would be for a density model (radius σ). ◇: the learned minimum.',
      };
      let capMode = null;

      let A = basin(0.25), B = basin(S.xB), gridA = null, gridB = null, endsA = null, endsB = null;
      const HW = 1.5, BOX = { x: 10, y: 10, w: 280, h: 280 }, NG = 50;
      function gridFor(Bs, n) { n = n || NG; const E = []; for (let r = 0; r < n; r++) { const yv = Bs.mu[1] + HW - 2 * HW * r / (n - 1), row = new Array(n); for (let c = 0; c < n; c++) { const xv = Bs.mu[0] - HW + 2 * HW * c / (n - 1); row[c] = energy(Bs.c0, xv, yv, false) - Bs.E; } E.push(row); } return E; }
      function layerFor(F, grid) { // heatmap + contours of the learned energy, cached until the grid changes
        const o = document.createElement('canvas'), dpr = F.cv.dpr; o.width = Math.round(BOX.w * dpr); o.height = Math.round(BOX.h * dpr);
        const g = o.getContext('2d'); g.scale(dpr, dpr);
        lib.heatmap(g, grid, 0, 0, BOX.w, BOX.h, { range: [0, 1.2], gamma: 0.85 });
        lib.contours(g, grid, 0, 0, BOX.w, BOX.h, [0.125, 0.25, 1.0], { color: 'rgba(17,17,17,0.25)', width: 0.8 });
        lib.contours(g, grid, 0, 0, BOX.w, BOX.h, [0.5], { color: C.blue, width: 2 });
        return o;
      }
      gridA = gridFor(A); endsA = endpoints(A); FA.layer = layerFor(FA, gridA);
      let rafB = 0, pendB = null;
      function setB(x, fast) {
        if (fast) { pendB = x; if (rafB) return; rafB = requestAnimationFrame(() => { rafB = 0; computeB(pendB, true); }); return; }
        computeB(x, false);
      }
      function computeB(x, fast) {
        S.xB = x; B = basin(x, fast ? 50 : 90); gridB = gridFor(B, fast ? 30 : NG); endsB = endpoints(B); FB.layer = layerFor(FB, gridB);
        drawFrames(); if (S.view === 'curv' || S.view === 'level' || S.view === 'mse' || S.view === 'alpha') drawBottom();
      }
      gridB = gridFor(B); endsB = endpoints(B); FB.layer = layerFor(FB, gridB);

      // ---------------- frames ----------------
      const toPx = (Bs, y) => [BOX.x + (y[0] - (Bs.mu[0] - HW)) / (2 * HW) * BOX.w, BOX.y + (Bs.mu[1] + HW - y[1]) / (2 * HW) * BOX.h];
      function idealDisc(c, Bs) { // exact radial picture of ΔE = r²/2σ² on the same color scale
        const p = toPx(Bs, Bs.mu), pxPerU = BOX.w / (2 * HW), R = Bs.sigma * Math.sqrt(2 * 1.2) * pxPerU;
        c.fillStyle = (() => { const q = lib.cmap(1); return `rgb(${q[0]},${q[1]},${q[2]})`; })(); c.fillRect(BOX.x, BOX.y, BOX.w, BOX.h);
        const gr = c.createRadialGradient(p[0], p[1], 0, p[0], p[1], Math.max(1, R));
        for (let i = 0; i <= 12; i++) { const t = i / 12, dE = 1.2 * t * t, q = lib.cmap(dE / 1.2); gr.addColorStop(t, `rgb(${q[0]},${q[1]},${q[2]})`); }
        c.fillStyle = gr; c.beginPath(); c.arc(p[0], p[1], Math.max(1, R), 0, 7); c.fill();
        [0.125, 0.25, 1.0].forEach(dE => { const rr = Bs.sigma * Math.sqrt(2 * dE) * pxPerU; if (rr > 3) { c.strokeStyle = 'rgba(17,17,17,0.25)'; c.lineWidth = 0.8; c.beginPath(); c.arc(p[0], p[1], rr, 0, 7); c.stroke(); } });
      }
      function drawFrame(F, Bs, grid, ends, label) {
        const c = F.cv.ctx; F.cv.clear();
        c.save(); c.beginPath(); c.rect(BOX.x, BOX.y, BOX.w, BOX.h); c.clip();
        if (S.mode === 'ideal') idealDisc(c, Bs);
        else {
          c.drawImage(F.layer, BOX.x, BOX.y, BOX.w, BOX.h);
        }
        // density-model ΔE = 1/2 circle (radius σ)
        const p = toPx(Bs, Bs.mu), rs = Bs.sigma * BOX.w / (2 * HW);
        c.save(); c.strokeStyle = C.ink; c.lineWidth = 1.3; c.setLineDash(rs > 8 ? [4, 3] : []); c.beginPath(); c.arc(p[0], p[1], Math.max(2.5, rs), 0, 7); c.stroke(); c.restore();
        // targets
        const eps = EPS[F === FA ? 0 : 1];
        c.strokeStyle = 'rgba(17,17,17,0.75)'; c.lineWidth = 0.9;
        eps.forEach(e => { const q = toPx(Bs, [Bs.mu[0] + Bs.sigma * e[0], Bs.mu[1] + Bs.sigma * e[1]]); c.beginPath(); c.arc(q[0], q[1], 2.4, 0, 7); c.stroke(); });
        if (S.mode === 'learned') {
          const pm = toPx(Bs, Bs.y); c.save(); c.fillStyle = '#fff'; c.strokeStyle = C.ink; c.lineWidth = 1.2; c.beginPath(); c.moveTo(pm[0], pm[1] - 5); c.lineTo(pm[0] + 5, pm[1]); c.lineTo(pm[0], pm[1] + 5); c.lineTo(pm[0] - 5, pm[1]); c.closePath(); c.fill(); c.stroke(); c.restore();
          if (S.ends) ends.forEach(y => { const q = toPx(Bs, y); lib.dot(c, q[0], q[1], 3.2, C.blue, { stroke: '#fff', lw: 1 }); });
          if (S.path) {
            const st = [Bs.mu[0] - 0.75, Bs.mu[1] + 0.65]; let y = st.slice(); const pts = [y.slice()], ds = [Math.hypot(y[0] - Bs.y[0], y[1] - Bs.y[1])];
            for (let i = 0; i < 4; i++) { const r = energy(Bs.c0, y[0], y[1], true); y = [y[0] - r[1], y[1] - r[2]]; pts.push(y.slice()); ds.push(Math.hypot(y[0] - Bs.y[0], y[1] - Bs.y[1])); }
            const px = pts.map(q => toPx(Bs, q)); lib.line(c, px, { color: C.blue, width: 1.8 });
            px.forEach((q, i) => { lib.dot(c, q[0], q[1], i ? 3 : 4, i ? C.blue : '#fff', { stroke: C.blue, lw: 1.4 }); });
            { const m = [(px[0][0] + px[1][0]) / 2, (px[0][1] + px[1][1]) / 2]; lib.text(c, '×' + (ds[1] / ds[0]).toFixed(2), m[0] + 7, m[1] - 4, { size: 11, kind: 'mono', color: C.ink }); }
            F.ratios = ds.slice(1).map((d, i) => d / ds[i]);
          }
        }
        // μ crosshair
        c.strokeStyle = C.ink; c.lineWidth = 1.2; c.beginPath(); c.moveTo(p[0] - 7, p[1]); c.lineTo(p[0] + 7, p[1]); c.moveTo(p[0], p[1] - 7); c.lineTo(p[0], p[1] + 7); c.stroke();
        c.restore();
        c.strokeStyle = C.ink; c.lineWidth = 1; c.strokeRect(BOX.x + 0.5, BOX.y + 0.5, BOX.w - 1, BOX.h - 1);
        lib.text(c, label, BOX.x + 8, BOX.y + 7, { size: 11.5, kind: 'mono', color: C.ink });
        lib.text(c, S.mode === 'ideal' ? 'density model: ΔE = ‖ŷ−μ‖²/2σ²' : 'learned: ΔE = E − E_min', BOX.x + 8, BOX.y + 23, { size: 10.5, kind: 'mono', color: C.muted });
        const u = BOX.w / (2 * HW) * 0.5; c.fillStyle = C.ink; c.fillRect(BOX.x + BOX.w - 12 - u, BOX.y + BOX.h - 14, u, 1.5);
        lib.text(c, '0.5', BOX.x + BOX.w - 12 - u / 2, BOX.y + BOX.h - 28, { size: 10.5, kind: 'mono', color: C.ink, align: 'center' });
        // readout
        const lm = (Bs.lam[0] + Bs.lam[1]) / 2, cE = Bs.E + S.k * Math.sin(2 * Math.PI * Bs.x);
        let spread = 0; if (ends) { const m = ends.reduce((a, y) => [a[0] + y[0] / ends.length, a[1] + y[1] / ends.length], [0, 0]); spread = Math.sqrt(ends.reduce((a, y) => a + (y[0] - m[0]) ** 2 + (y[1] - m[1]) ** 2, 0) / ends.length); }
        if (S.mode === 'ideal') { F.ro.innerHTML = `<span>σ <b>${Bs.sigma.toFixed(3)}</b></span><span>density model: λ = 1/σ² <b>${(1 / Bs.sigma ** 2).toFixed(0)}</b></span><span>width <b>${Bs.sigma.toFixed(2)}</b></span>`; return; }
        F.ro.innerHTML = `<span>σ <b>${Bs.sigma.toFixed(3)}</b></span><span>λ live <b>${Bs.lam[0].toFixed(2)}, ${Bs.lam[1].toFixed(2)}</b></span><span>width 1/√λ <b>${(1 / Math.sqrt(lm)).toFixed(2)}</b> vs σ <b>${Bs.sigma.toFixed(2)}</b></span>`
          + `<span>E<sub>min</sub>${S.k ? ' + c(x)' : ''} <b>${cE.toFixed(3)}</b></span>`
          + (S.path && F.ratios ? `<span>error ratio per step <b>${F.ratios.map(r => r.toFixed(2)).join(', ')}</b> → |1−λ| <b>${Math.abs(1 - Bs.lam[1]).toFixed(2)}–${Math.abs(1 - Bs.lam[0]).toFixed(2)}</b></span>` : `<span>spread of 24 runs <b>${spread.toFixed(4)}</b></span>`);
      }
      function drawFrames() {
        if (capMode !== S.mode) { capMode = S.mode; ctx.setCaption(CAP[S.mode]); }
        subB.innerHTML = 'x = ' + S.xB.toFixed(3) + ' · σ = ' + sigOf(S.xB).toFixed(3) + ' · window μ ± 1.5';
        drawFrame(FA, A, gridA, endsA, 'A · x = 0.25');
        drawFrame(FB, B, gridB, endsB, 'B · x = ' + S.xB.toFixed(3));
      }

      // ---------------- bottom charts ----------------
      const T = (c, s, x, y, o = {}) => lib.text(c, s, x, y, Object.assign({ size: 10.5, kind: 'mono', color: C.muted }, o));
      const title = (c, s, short) => { const m = lib.measure(c, s, { size: 10.5, kind: 'mono', spacing: 0.5 }); T(c, (short && m.w > W - 24) ? short : s, 12, 8, { color: C.ink, size: 10.5, spacing: 0.5 }); };
      const xs = Array.from({ length: 201 }, (_, i) => i / 200);
      const pearson = (a, b) => { const n = a.length, ma = a.reduce((s, v) => s + v, 0) / n, mb = b.reduce((s, v) => s + v, 0) / n; let sab = 0, saa = 0, sbb = 0; for (let i = 0; i < n; i++) { sab += (a[i] - ma) * (b[i] - mb); saa += (a[i] - ma) ** 2; sbb += (b[i] - mb) ** 2; } return sab / Math.sqrt(saa * sbb); };
      let hit = null; // for clicks: {x0, w}
      function cursor(c, ax, x, lab, yTop, yBot) { c.save(); c.strokeStyle = C.ink; c.lineWidth = 0.8; c.setLineDash([2, 3]); c.beginPath(); c.moveTo(ax.X(x), yTop); c.lineTo(ax.X(x), yBot); c.stroke(); c.restore(); T(c, lab, ax.X(x) + 3, yTop, { color: C.ink }); }
      function viewCurv(c) {
        title(c, 'BASIN CURVATURE vs NOISE · 64 CONTEXTS · LOG SCALE', 'CURVATURE vs NOISE · 64 CONTEXTS · LOG');
        const ax = lib.axes(c, { x: 52, y: 30, w: W - 70, h: H - 76, xlim: [0, 1], ylim: [0.01, 3000], ylog: true, xticks: [0, 0.25, 0.5, 0.75, 1], yticks: [0.01, 0.1, 1, 10, 100, 1000], yfmt: (v) => v >= 1 ? String(v) : String(v), size: 10.5 });
        T(c, 'context x', ax.X(1), H - 16, { align: 'right' });
        lib.plot(c, ax, xs.map(x => [x, sigOf(x)]), { color: C.ink, width: 1.2 });
        c.save(); c.setLineDash([5, 3]); lib.plot(c, ax, xs.map(x => [x, 1 / sigOf(x) ** 2]), { color: C.ink, width: 1.4 }); c.restore();
        T(c, '1/σ²: what a density model needs (dashed)', ax.X(0.47), ax.Y(300), { color: C.ink });
        T(c, 'σ(x): noise in the targets', ax.X(0.52), ax.Y(0.2) + 2, { color: C.ink });
        if (S.band) {
          c.save(); c.fillStyle = C.blue3; c.beginPath();
          BS.x.forEach((x, i) => { const p = [ax.X(x), ax.Y(BS.hess_eig_large[i])]; i ? c.lineTo(p[0], p[1]) : c.moveTo(p[0], p[1]); });
          for (let i = BS.x.length - 1; i >= 0; i--) c.lineTo(ax.X(BS.x[i]), ax.Y(BS.hess_eig_small[i]));
          c.closePath(); c.fill(); c.restore();
          lib.plot(c, ax, BS.x.map((x, i) => [x, (BS.hess_eig_small[i] + BS.hess_eig_large[i]) / 2]), { color: C.blue, width: 2 });
          T(c, 'λ learned by the toy (both eigenvalues)', ax.X(0.03), ax.Y(0.85) - 18, { color: C.blue });
        }
        cursor(c, ax, 0.25, 'A', ax.Y(3000), ax.Y(0.01)); cursor(c, ax, S.xB, 'B', ax.Y(3000), ax.Y(0.01));
        hit = { x0: ax.X(0), w: ax.X(1) - ax.X(0) };
        const i = Math.round(S.xB * 64 - 0.5), j = Math.max(0, Math.min(63, i));
        roC.innerHTML = `<span>σ range <b>0.03–0.30</b> (10×)</span><span>1/σ² range <b>11–1111</b> (100×)</span><span>learned λ range <b>${Math.min(...BS.hess_eig_small).toFixed(2)}–${Math.max(...BS.hess_eig_large).toFixed(2)}</b></span><span>corr(σ, mean λ) <b>${BS.corr_sigma_vs_mean_curvature.toFixed(2)}</b></span>`;
      }
      function viewMse(c) {
        title(c, 'EXPECTED LOSS ALONG ŷ₁ − μ₁ · A vs B');
        const sA = A.sigma, sB = B.sigma;
        const ax = lib.axes(c, { x: 52, y: 30, w: W - 70, h: H - 76, xlim: [-1, 1], ylim: [0, 1.3], xticks: [-1, -0.5, 0, 0.5, 1], yticks: [0, 0.5, 1], yfmt: (v) => v.toFixed(1), xfmt: (v) => v < 0 ? '−' + Math.abs(v) : String(v), size: 10.5 });
        T(c, 'prediction offset u = ŷ₁ − μ₁', ax.X(1), H - 16, { align: 'right' });
        const us = Array.from({ length: 101 }, (_, i) => -1 + i / 50);
        lib.plot(c, ax, us.map(u => [u, u * u + 2 * sA * sA]), { color: C.ink, width: 1.5 });
        c.save(); c.setLineDash([5, 3]); lib.plot(c, ax, us.map(u => [u, u * u + 2 * sB * sB]), { color: C.ink, width: 1.5 }); c.restore();
        const eA = us.map(u => [u, energy(A.c0, A.mu[0] + u, A.mu[1], false) - energy(A.c0, A.mu[0], A.mu[1], false)]), eB = us.map(u => [u, energy(B.c0, B.mu[0] + u, B.mu[1], false) - energy(B.c0, B.mu[0], B.mu[1], false)]);
        lib.plot(c, ax, eA, { color: C.blue, width: 1.4 }); c.save(); c.setLineDash([5, 3]); lib.plot(c, ax, eB, { color: C.blue, width: 1.4 }); c.restore();
        const u0 = 0.6; [sA, sB].forEach(s => { const y0 = u0 * u0 + 2 * s * s, d = 0.2; lib.line(c, [[ax.X(u0 - d), ax.Y(y0 - 2 * u0 * d)], [ax.X(u0 + d), ax.Y(y0 + 2 * u0 * d)]], { color: C.muted, width: 2.2 }); lib.dot(c, ax.X(u0), ax.Y(y0), 2.6, C.ink); });
        const yA1 = 1 + 2 * sA * sA, yB1 = 1 + 2 * sB * sB;
        T(c, 'loss A', ax.X(0.86), ax.Y(yA1) + 6, { color: C.ink, align: 'right' });
        T(c, 'loss B', ax.X(0.80), ax.Y(Math.min(1.25, yB1)) - 16, { color: C.ink, align: 'right' });
        T(c, 'E_θ A, B', ax.X(-0.97), ax.Y(eA[0][1]) - 16, { color: C.blue });
        T(c, 'parallel tangents', ax.X(0.42), ax.Y(0.62), { color: C.muted, align: 'right' });
        roC.innerHTML = `<span>black: expected loss u² + 2σ² (solid A, dashed B)</span><span>offset 2σ²: A <b>${(2 * sA * sA).toFixed(4)}</b>, B <b>${(2 * sB * sB).toFixed(3)}</b></span><span>gray tangents at u = ${u0}: slope <b>${(2 * u0).toFixed(1)}</b> for both</span><span>blue: learned energy along the same line</span><span>minimizer: <b>μ(x)</b> for both</span>`;
      }
      function viewLevel(c) {
        title(c, 'MINIMUM ENERGY PER CONTEXT, PLUS c(x) = k·sin 2πx', 'E_min + k·sin 2πx PER CONTEXT');
        const ax = lib.axes(c, { x: 52, y: 30, w: W - 92, h: H - 76, xlim: [0, 1], ylim: [-1.05, 0.15], xticks: [0, 0.25, 0.5, 0.75, 1], yticks: [-1, -0.5, 0], yfmt: (v) => v < 0 ? '−' + Math.abs(v) : String(v), size: 10.5 });
        T(c, 'context x', ax.X(1), H - 16, { align: 'right' });
        // σ on a right axis
        const Ys = (s) => ax.Y(-1.05) - (s - 0) / 0.35 * (ax.Y(-1.05) - ax.Y(0.15));
        lib.plot(c, { X: ax.X, Y: Ys }, xs.map(x => [x, sigOf(x)]), { color: C.ink, width: 1.1 });
        [0.1, 0.2, 0.3].forEach(v => T(c, v.toFixed(1), ax.X(1) + 6, Ys(v), { baseline: 'middle' }));
        T(c, 'σ(x) →', ax.X(1) + 4, Ys(0.34) - 4, { color: C.ink, align: 'right' });
        const ys = BS.E_min.map((e, i) => e + S.k * Math.sin(2 * Math.PI * BS.x[i]));
        BS.x.forEach((x, i) => lib.dot(c, ax.X(x), ax.Y(ys[i]), 2.8, C.blue));
        T(c, 'blue dots: E_min + c(x)', ax.X(0), H - 16, { color: C.blue });
        cursor(c, ax, 0.25, 'A', ax.Y(0.15), ax.Y(-1.05)); cursor(c, ax, S.xB, 'B', ax.Y(0.15), ax.Y(-1.05));
        hit = { x0: ax.X(0), w: ax.X(1) - ax.X(0) };
        const r = pearson(BS.sigma, ys);
        if (W >= 450) T(c, 'corr(σ, E_min + c) = ' + (r < 0 ? '−' : '') + Math.abs(r).toFixed(2), ax.X(1) - 4, ax.Y(-1.05) - 3, { color: C.ink, align: 'right', baseline: 'bottom', size: 12 });
        roC.innerHTML = `<span>k <b>${S.k.toFixed(2)}</b></span><span>corr(σ, E<sub>min</sub>+c) <b>${r.toFixed(2)}</b></span><span>thinking paths, ŷ<sub>N</sub>, training loss: <b>unchanged</b> (∇<sub>ŷ</sub>c = 0)</span>`;
      }
      function resid(lam, a0, na) { // E over α ~ a0·exp(U(−ln f, ln f)), N ~ U{nmin..nmax} of (1 − αλ)^(2N)
        let tot = 0; na = na || 48;
        for (let n = NMIN; n <= NMAX; n++) for (let j = 0; j < na; j++) { const a = a0 * Math.exp(((j + 0.5) / na * 2 - 1) * Math.log(AF)); tot += Math.pow(1 - a * lam, 2 * n); }
        return tot / ((NMAX - NMIN + 1) * na);
      }
      function viewAlpha(c) {
        title(c, 'LEFTOVER ERROR AFTER THINKING vs BASIN CURVATURE λ', 'LEFTOVER ERROR vs CURVATURE λ');
        const ax = lib.axes(c, { x: 52, y: 30, w: W - 70, h: H - 76, xlim: [0.1, 30], ylim: [1e-4, 1e4], xlog: true, ylog: true, xticks: [0.1, 0.3, 1, 3, 10, 30], yticks: [1e-4, 1e-2, 1, 1e2, 1e4], yfmt: (v) => v === 1 ? '1' : '10' + ({ '-4': '⁻⁴', '-2': '⁻²', '2': '²', '4': '⁴' })[String(Math.round(Math.log10(v)))], size: 10.5 });
        T(c, 'curvature λ (log)', ax.X(30), H - 16, { align: 'right' });
        const lo = Math.min(...BS.hess_eig_small), hi = Math.max(...BS.hess_eig_large);
        c.fillStyle = C.blue4; c.fillRect(ax.X(lo), ax.Y(1e4), ax.X(hi) - ax.X(lo), ax.Y(1e-4) - ax.Y(1e4));
        T(c, 'toy λ', ax.X(Math.sqrt(lo * hi)), ax.Y(1e4) + 2, { color: C.blue, align: 'center' });
        c.save(); c.setLineDash([3, 3]); lib.line(c, [[ax.X(0.1), ax.Y(1)], [ax.X(30), ax.Y(1)]], { color: C.faint, width: 1 }); c.restore();
        T(c, 'r = 1: no progress', ax.X(0.11), ax.Y(1) - 14);
        const ls = Array.from({ length: 241 }, (_, i) => Math.pow(10, -1 + i / 240 * Math.log10(300)));
        const rs = ls.map(l => [l, Math.max(1e-4, Math.min(1e4, resid(l, S.a0)))]);
        lib.plot(c, ax, rs, { color: C.ink, width: 1.6 });
        let best = ls[0], bv = Infinity; ls.forEach(l => { const v = resid(l, S.a0); if (v < bv) { bv = v; best = l; } });
        { let lo = best / 1.03, hi = best * 1.03; for (let it = 0; it < 40; it++) { const m1 = lo + (hi - lo) / 3, m2 = hi - (hi - lo) / 3; if (resid(m1, S.a0, 400) < resid(m2, S.a0, 400)) hi = m2; else lo = m1; } best = (lo + hi) / 2; bv = resid(best, S.a0, 400); }
        lib.dot(c, ax.X(best), ax.Y(Math.max(1e-4, bv)), 4, C.blue, { stroke: '#fff', lw: 1 });
        T(c, 'λ* = ' + best.toFixed(2), ax.X(best) + 6, ax.Y(Math.max(1e-4, bv)) - 2, { color: C.blue });
        const lB = 1 / B.sigma ** 2;
        if (lB < 30) { c.save(); c.setLineDash([4, 3]); lib.line(c, [[ax.X(lB), ax.Y(1e4)], [ax.X(lB), ax.Y(1e-4)]], { color: C.ink, width: 1.1 }); c.restore(); T(c, 'density bowl, B', ax.X(lB) - 4, ax.Y(1e-3), { color: C.ink, align: 'right' }); }
        { const s = 'A: λ = 1111 →', m = lib.measure(c, s, { size: 10.5, kind: 'mono' }); c.fillStyle = '#fff'; c.fillRect(ax.X(30) - 4 - m.w, ax.Y(1e-3) + 12, m.w + 4, 15); T(c, s, ax.X(30) - 2, ax.Y(1e-3) + 14, { color: C.ink, align: 'right' }); }
        const amax = S.a0 * AF, amin = S.a0 / AF;
        roC.innerHTML = `<span>α ∈ <b>[${amin.toFixed(2)}, ${amax.toFixed(2)}]</b>, N ∈ <b>${NMIN}..${NMAX}</b></span><span>λ* <b>${best.toFixed(2)}</b> = 2/(α<sub>min</sub>+α<sub>max</sub>) = <b>${(2 / (amin + amax)).toFixed(2)}</b></span><span>at λ = 1/σ<sub>B</sub>² = ${lB.toFixed(1)}: |1 − α<sub>max</sub>λ| = <b>${Math.abs(1 - amax * lB).toFixed(1)}</b> per step</span>`;
      }
      function heat(c, Hm, x, y, w, hh, o = {}) {
        const R = Hm.rows.length, nI = Hm.energy[0].length, lw = o.labelW, cw = (w - lw) / nI, ch = Math.min(o.maxRow || 16, (hh - 18) / R);
        Hm.energy.forEach((row, r) => row.forEach((v, i) => { const q = lib.cmap(v); c.fillStyle = `rgb(${q[0]},${q[1]},${q[2]})`; c.fillRect(x + lw + i * cw, y + r * ch, Math.ceil(cw), Math.ceil(ch)); }));
        c.strokeStyle = C.ink; c.lineWidth = 0.8; c.strokeRect(x + lw + 0.5, y + 0.5, nI * cw, R * ch);
        Hm.rows.forEach((t, r) => T(c, t, x + lw - 5, y + r * ch + ch / 2, { align: 'right', baseline: 'middle', color: C.ink, size: Math.min(10.5, ch - 1) }));
        [0, 5, 11].forEach(i => T(c, String(i), x + lw + (i + 0.5) * cw, y + R * ch + 3, { align: 'center' }));
        return { ch, cw, R, bottom: y + R * ch };
      }
      function viewPaper(c) {
        const id = S.fig;
        if (id === 'figB2') {
          const h1 = PH.find(p => p.id === 'figB2_seq1'), h2 = PH.find(p => p.id === 'figB2_seq2');
          title(c, 'FIG B.2 · FAMILIAR SENTENCE vs RANDOM TOKENS (approx.)', 'FIG B.2 · SENTENCE vs RANDOM (approx.)');
          const half = (W - 24) / 2, lw = Math.min(70, half * 0.3);
          const g1 = heat(c, h1, 12, 40, half - 6, H - 84, { labelW: lw }), g2 = heat(c, h2, 12 + half + 6, 40, half - 6, H - 84, { labelW: lw });
          T(c, 'familiar sentence', 12 + lw, 25, { color: C.ink }); T(c, 'random tokens', 12 + half + 6 + lw, 25, { color: C.ink });
          const m = (Hm) => Hm.energy.reduce((s, r) => s + r.slice(1).reduce((a, b) => a + b, 0) / (r.length - 1), 0) / Hm.energy.length;
          T(c, 'thinking iteration →', 12 + half - 6, g1.bottom + 16, { align: 'right' });
          roC.innerHTML = `<span>mean normalized energy, iterations 1–11: sentence <b>${m(h1).toFixed(2)}</b>, random tokens <b>${m(h2).toFixed(2)}</b></span><span>paper: "EBTs learn to know what they don't know" (p.28)</span>`;
        } else {
          const Hm = PH.find(p => p.id === id); if (!Hm) return;
          title(c, Hm.figure.toUpperCase() + ' · ' + Hm.title.toUpperCase() + ' (approx.)', Hm.figure.toUpperCase() + ' (approx.)');
          const barW = Math.min(120, W * 0.22), lw = Math.min(90, W * 0.2);
          const g = heat(c, Hm, 12, 30, W - 24 - barW - 10, H - 74, { labelW: lw });
          T(c, 'thinking iteration →', 12 + (W - 24 - barW - 10), g.bottom + 16, { align: 'right' });
          const bx = W - 12 - barW, means = Hm.energy.map(r => r.slice(1).reduce((a, b) => a + b, 0) / (r.length - 1)), mx = Math.max(...means, 0.5);
          T(c, 'mean, it. 1–11', W - 12, 18, { align: 'right' });
          means.forEach((v, r) => { const y = 30 + r * g.ch; c.fillStyle = C.blue; c.fillRect(bx, y + g.ch * 0.2, v / mx * (barW - 34), g.ch * 0.6); T(c, v.toFixed(2), bx + v / mx * (barW - 34) + 3, y + g.ch / 2, { baseline: 'middle', size: Math.min(10, g.ch) }); });
          const order = means.map((v, i) => [v, Hm.rows[i]]).sort((a, b) => b[0] - a[0]);
          roC.innerHTML = `<span>highest after thinking: <b>${order.slice(0, 3).map(o => o[1]).join(', ')}</b></span><span>lowest: <b>${order.slice(-3).reverse().map(o => o[1]).join(', ')}</b></span><span>${Hm.label}; color: darker = lower energy</span>`;
        }
        // colorbar
        const cbx = 12, cby = H - 14, cbw = 120;
        for (let i = 0; i < cbw; i++) { const q = lib.cmap(i / (cbw - 1)); c.fillStyle = `rgb(${q[0]},${q[1]},${q[2]})`; c.fillRect(cbx + i, cby, 1.2, 6); }
        T(c, '0', cbx, cby - 13); T(c, '1  "Normalized Energy"', cbx + cbw - 4, cby - 13);
      }
      function viewText(c) {
        if (!TX) { title(c, 'text toy data missing'); return; }
        title(c, 'TEXT TOY · MEAN ENERGY PER THINKING STEP · EASY vs HARD CHARACTERS', 'TEXT TOY · EASY vs HARD CHARACTERS');
        const G = TX.groups, steps = TX.steps.slice(1), half = (W - 12) / 2;
        const one = (x0, key, lab) => {
          const e = G.easy[key].slice(1), hd = G.hard[key].slice(1), all = e.concat(hd), lo = Math.min(...all), hi = Math.max(...all), pad = (hi - lo) * 0.1;
          const ax = lib.axes(c, { x: x0 + 44, y: 48, w: half - 62, h: H - 96, xlim: [1, steps[steps.length - 1]], ylim: [lo - pad, hi + pad], xticks: [1, 4, 8], yticks: [lo, hi], yfmt: (v) => (v < 0 ? '−' : '') + Math.abs(v).toFixed(2), size: 10.5 });
          T(c, lab, x0 + 44, 30, { color: C.ink });
          lib.plot(c, ax, steps.map((s, i) => [s, e[i]]), { color: C.ink, width: 1.6, markers: 2 });
          lib.plot(c, ax, steps.map((s, i) => [s, hd[i]]), { color: C.blue, width: 1.8, markers: 2.2 });
          const L = steps.length - 1;
          T(c, 'easy', ax.X(steps[L]) - 2, ax.Y(e[L]) + (e[L] > hd[L] ? -15 : 4), { color: C.ink, align: 'right' });
          T(c, 'hard', ax.X(steps[L]) - 2, ax.Y(hd[L]) + (hd[L] > e[L] ? -15 : 4), { color: C.blue, align: 'right' });
          T(c, 'thinking step', ax.X(steps[L]), H - 18, { align: 'right' });
        };
        one(0, 'mean_energy', 'raw E(x, ŷᵢ)');
        one(half + 12, 'mean_rel_energy', 'E(x, ŷᵢ) − E(x, uniform)');
        const EL = TX.energy_vs_loss;
        roC.innerHTML = `<span>n <b>${G.easy.n}</b> easy, <b>${G.hard.n}</b> hard</span><span>final raw E: easy <b>${G.easy.mean_energy.slice(-1)[0].toFixed(3)}</b>, hard <b>${G.hard.mean_energy.slice(-1)[0].toFixed(3)}</b></span><span>Spearman with char loss: energy <b>${EL.spearman_final_energy_vs_char_loss.toFixed(2)}</b>, entropy <b>${EL.spearman_final_entropy_vs_char_loss.toFixed(2)}</b></span><span>step 0 (random start) not shown</span>`;
      }
      function drawBottom() {
        const c = cvC.ctx; cvC.clear(); hit = null;
        slK.el.style.display = S.view === 'level' ? '' : 'none';
        slA.el.style.display = S.view === 'alpha' ? '' : 'none';
        segF.el.style.display = S.view === 'paper' ? '' : 'none';
        ctl2.style.display = (S.view === 'level' || S.view === 'alpha' || S.view === 'paper') ? '' : 'none';
        ({ curv: viewCurv, mse: viewMse, level: viewLevel, alpha: viewAlpha, paper: viewPaper, text: viewText })[S.view](c);
        cvC.canvas.style.cursor = hit ? 'pointer' : 'default';
      }
      cvC.canvas.addEventListener('click', (ev) => { if (!hit) return; const [px] = cvC.toLocal(ev); const v = Math.round(Math.max(0, Math.min(0.995, (px - hit.x0) / hit.w)) * 200) / 200; slB.set(v); setB(v); });

      drawFrames(); drawBottom();
      const P = [
        { mode: 'ideal', view: 'curv', ends: false, path: false, band: false },
        { mode: 'learned', view: 'curv', ends: true, path: false, band: true },
        { mode: 'learned', view: 'mse', ends: true, path: false, band: true },
        { mode: 'learned', view: 'level', ends: false, path: false, band: true },
        { mode: 'learned', view: 'alpha', ends: false, path: true, band: true },
        { mode: 'learned', view: 'paper', ends: false, path: false, band: true },
        { mode: 'learned', view: 'text', ends: false, path: false, band: true },
      ];
      // The stage is sticky (base.css). When it is taller than the window, stick it higher so that everything down to
      // the step bar (the frames and the step's chart) stays on screen; a stacked (very tall) stage just scrolls.
      function fitSticky() {
        const bar = stage.querySelector('.stepbar'); if (!bar) return;
        const need = bar.getBoundingClientRect().bottom - stage.getBoundingClientRect().top + 14, room = window.innerHeight;
        if (need + 24 <= room) { stage.style.position = ''; stage.style.top = ''; }
        else if (need > room * 1.45) { stage.style.position = 'static'; stage.style.top = ''; }
        else { stage.style.position = ''; stage.style.top = Math.round(room - need) + 'px'; }
      }
      window.addEventListener('resize', () => requestAnimationFrame(fitSticky));
      setTimeout(fitSticky, 0);
      return {
        show() { fitSticky(); },
        step(i) {
          const p = P[i] || P[0]; Object.assign(S, p);
          if (i === 3 && S.k === 0) { S.k = 0.3; slK.set(0.3); }
          if (i !== 3) { S.k = 0; slK.set(0); }
          if (i === 4) { S.a0 = 1; slA.set(0); }
          if (Math.abs(S.xB - 0.75) > 1e-9) { slB.set(0.75); setB(0.75); }
          segM.set(S.mode); tabs.set(S.view); drawFrames(); drawBottom();
        },
      };
    },
  });
})();
