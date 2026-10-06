/* Panel: the loss landscape (training loss over the weights θ) is not the energy landscape (energy over the prediction ŷ).
   Real computation: the toy 2-D EBT (data/toy2d.json). The left map is the stored slice toy2d.param_landscape,
   L(θ* + a·d1 + b·d2). The two filter-normalized directions d1, d2 used to make that slice are embedded at the end of
   this file (numpy default_rng(1) and (2) standard normals, int16-quantized at 1/8192; rows rescaled here to the row
   norms of θ, exactly as in src/toy2d_export.py), so every point you pick on the left is rebuilt as a full set of
   weights and its energy landscape is recomputed live on the right. */
(function () {
  EBT.panel({
    id: 'two-landscapes',
    nav: 'Loss vs energy landscape',
    title: 'The loss landscape is not the energy landscape',
    lede: 'Two bowls, two gradient descents. Learning walks down the training loss over the weights. Thinking walks down the energy over the prediction. Every point of the first bowl is a full set of weights, and so defines a whole new version of the second.',
    text: `
      <p>Both objects get drawn as bowls, so they are easy to confuse. They are functions on different spaces:</p>
      <div class="eq">$$\\begin{aligned}\\text{energy:}&\\quad E_\\theta(x,\\hat y)\\ \\text{ as a function of } \\hat y\\\\ \\text{loss:}&\\quad \\mathcal L(\\theta)=\\mathbb E_{(x,y)}\\big[J(\\hat y_N(\\theta),\\,y)\\big]\\end{aligned}$$<span class="why">the energy landscape lives over predictions ŷ, the loss landscape over weights θ.</span></div>
      <p>The <b>energy</b> scores a candidate prediction $\\hat y$ for a context $x$, with the weights $\\theta$ frozen. Its domain is the prediction space: 2 numbers in our toy, a vector of 50,277 logits per token for text (Table D.4), a 3,136-dim latent per video frame (p.12).</p>
      <p>The <b>loss</b> scores a whole set of weights. $\\hat y_N(\\theta)$ is the prediction after $N$ thinking steps on $E_\\theta$, $J$ is the task loss (squared error in this toy, cross-entropy for text, p.7), and the expectation runs over training pairs. Its domain is weight space: 8,961 parameters here, 6.18M non-embedding parameters in the smallest paper model (Table D.1).</p>
      <div class="eq">$$\\begin{aligned}\\text{think:}&\\quad \\hat y_{i+1}=\\hat y_i-\\alpha\\,\\nabla_{\\hat y}E_\\theta(x,\\hat y_i)\\\\ \\text{learn:}&\\quad \\theta\\leftarrow\\theta-\\eta\\,\\nabla_{\\theta}\\mathcal L(\\theta)\\end{aligned}$$<span class="why">α: thinking step size (Eq. 1, p.7). η: learning rate of the weight optimizer (AdamW in the paper).</span></div>`,
    steps: [
      { label: 'Two surfaces, two spaces', html: '<p>Left: the toy\'s training loss on a 2-D slice through weight space, centred on the trained weights $\\theta^*$ (blue dot). Right: the energy over predictions for one context, computed from exactly those weights. The blue path is 4 thinking steps from $\\hat y_0$ and ends on the target mean $\\mu(x)$ (crosshair). At $\\theta^*$ that happens for almost every context, which is why $\\mathcal L(\\theta^*)=0.076$ sits close to this batch\'s noise floor of 0.068.</p>' },
      { label: 'A point on the left is a surface on the right', html: '<p>We moved $\\theta$ to $\\theta^*-0.5\\,d_1-0.5\\,d_2$. That changes all 8,768 matrix weights at once, so the right panel is a different function. Its minimum (◇) slid from $\\mu(x)$ to about $(0.99,-0.30)$; the dashed contour shows where the basin was at $\\theta^*$. Thinking still converges, but to the wrong place, and the loss on the left rises from 0.076 to 0.99. Drag the dot anywhere: the right panel is recomputed from the new weights.</p>' },
      { label: 'Thinking moves on the energy landscape', html: '<p>At $\\theta^*-0.8\\,d_1+0.6\\,d_2$ the bowl is misplaced and much steeper: Hessian eigenvalues $\\lambda\\approx 2.5$ and $3.6$ at its minimum (readout) instead of about $0.85$. Near a minimum one step with $\\alpha=1$ multiplies the error along each eigenvector by $1-\\alpha\\lambda$, here $-1.5$ and $-2.6$. Every step overshoots by more than it corrects, so the path bounces across the valley and the energy <em>rises</em>. The loss here is 6.98. $\\mathcal L$ grades the whole procedure: the landscape together with the step size and step count it is used with: with $\\alpha<2/\\lambda_{\\max}\\approx 0.56$ this same bowl would converge, though still to the wrong place.</p>' },
      { label: 'Learning moves on the loss landscape', html: '<p>Now descend the left surface: $(a,b)\\leftarrow(a,b)-\\eta\\,\\nabla_{(a,b)}\\mathcal L$. The two components are $d_1\\!\\cdot\\!\\nabla_\\theta\\mathcal L$ and $d_2\\!\\cdot\\!\\nabla_\\theta\\mathcal L$, the weight gradient projected onto the plane (estimated here from the stored slice). As $\\theta$ slides home, the energy landscape is rebuilt at every step, its curvature relaxes and thinking lands near $\\mu(x)$ again. The loss settles just above the noise floor of this batch (dashed): no weights can predict the noise in $y$.</p>' },
      { label: 'Where training starts', html: '<p>The same kind of slice around the initial weights $\\theta_0$ (training step 0, same color scale). The whole neighbourhood is high: 2.83 at the centre, 2.65 at best. The energy at $\\theta_0$ is almost flat and knows nothing about the context: its minimum ◇ sits far from $\\mu(x)$, and the 4 thinking steps creep toward it. Real training travels from $\\theta_0$ to $\\theta^*$ along directions that lie in neither plane; see <a href="#learning">Watching a landscape learn</a>.</p>' },
      { label: 'Same picture, different object', html: '<p>In 3-D both are bowls (drag to rotate). The paper only ever talks about energy landscapes. Its Fig 3 is a schematic energy landscape for "The dog caught the ___", marked "Adapted from [57]", and [57] is Li et al., <i>Visualizing the loss landscape of neural nets</i> (p.5, p.20). Li et al. drew loss over weights; the paper borrowed the look for energy over predictions.</p>' },
    ],
    after: `
      <h3>Why the distinction matters</h3>
      <ul>
        <li>"Convex surrounding the ground truth solution" (p.7) is a claim about $E_\\theta$ over $\\hat y$, a shape that training creates. It says nothing about $\\mathcal L$ over $\\theta$, which for a deep network is non-convex.</li>
        <li>$\\mathcal L$ is defined through the energy. Its gradient must pass through every thinking step, $\\partial\\hat y_N/\\partial\\theta$, which is why training needs Hessian-vector products (p.7; see <a href="#second-order">Gradients of gradients</a>).</li>
        <li>The two loops nest. Every learning step first runs the $N$ thinking steps on each training example, scores where they ended, and only then moves $\\theta$ (Alg. 1). At test time only the inner loop runs.</li>
        <li>Thinking never sees the target $y$. At test time the energy is the only signal (Alg. 2). Learning is the only place $y$ enters.</li>
      </ul>
      <p class="note">How the left map is made: Li et al. (2018) filter-normalized random directions. Each row of a Gaussian direction is rescaled to the norm of the same row of θ, so a = 1 means a perturbation as large as each unit's own weights; biases are not moved. Height: the Algorithm 1 loss on a fixed batch of 512 pairs, N = 4 steps, per-sample α ∈ [0.5, 2], Langevin σ = 0.05, no replay (toy2d.param_landscape). A 2-D slice of a function of 8,961 weights can hide every other direction. "live" re-runs the same objective on a new random batch of 512 in your browser; away from θ* a few contexts fail badly, so it can differ from the slice by batch noise (for example 1.26 to 1.80 at one point).</p>`,
    source: [{ kind: 'toy', note: 'toy 2-D EBT: weights, loss slice param_landscape, energies recomputed live' }, { kind: 'paper', note: 'Eq. 1, Alg. 1 (p.7), Fig 3 (p.5)' }],

    figure(stage, ctx) {
      const { lib } = ctx, h = lib.h, C = lib.C, M = window.EBT.toy2d;
      const D = M && M.data;
      if (!M || !M.ready || !D.param_landscape) { stage.appendChild(h('p', { class: 'callout warn' }, 'Toy model data missing (data/toy2d.json).')); return {}; }
      const PL = D.param_landscape, EXT = M.extent, LAM = D.arch.quadratic_term.lambda_, KF = D.arch.feature_map.K;
      const FINAL = M.nCkpt - 1, NSTEP = 4, ALPHA = 1.0, HID = 64;
      const muOf = (x) => [1.3 * Math.sin(2 * Math.PI * x), 0.9 * Math.sin(4 * Math.PI * x)];          // toy2d.dataset.mu_formula
      const sigOf = (x) => 0.03 + 0.27 * Math.pow(Math.sin(Math.PI * (x - 0.25)), 2);                  // toy2d.dataset.sigma_formula
      const sub = (n) => String(n).split('').map(d => '₀₁₂₃₄₅₆₇₈₉'[+d]).join('');

      // ---------------- weights θ(a, b) = θ_base + a d1 + b d2 ----------------
      const RAW = decodeDirs();
      const shapes = D.weights[FINAL].layers.map(l => [l.W.length, l.W[0].length]);
      function makeBase(ck) {
        const out = []; let off = 0;
        D.weights[ck].layers.forEach((l, li) => {
          const [no, ni] = shapes[li], W = new Float64Array(no * ni), d1 = new Float64Array(no * ni), d2 = new Float64Array(no * ni);
          for (let o = 0; o < no; o++) {
            let wn = 0, n1 = 0, n2 = 0;
            for (let i = 0; i < ni; i++) { const w = l.W[o][i], k = off + o * ni + i; W[o * ni + i] = w; wn += w * w; n1 += RAW[0][k] * RAW[0][k]; n2 += RAW[1][k] * RAW[1][k]; }
            const s1 = Math.sqrt(wn) / (Math.sqrt(n1) + 1e-12), s2 = Math.sqrt(wn) / (Math.sqrt(n2) + 1e-12);
            for (let i = 0; i < ni; i++) { const k = off + o * ni + i; d1[o * ni + i] = RAW[0][k] * s1; d2[o * ni + i] = RAW[1][k] * s2; }
          }
          off += no * ni; out.push({ no, ni, W, b: Float64Array.from(l.b), d1, d2 });
        });
        return out;
      }
      const BASES = { final: makeBase(FINAL), init: makeBase(0) };
      const NPARAM = BASES.final.reduce((s, L) => s + L.W.length + L.b.length, 0);
      function makeNet(which, a, b) {
        return BASES[which].map(L => { const W = new Float64Array(L.W.length); for (let k = 0; k < W.length; k++) W[k] = L.W[k] + a * L.d1[k] + b * L.d2[k]; return { no: L.no, ni: L.ni, W, b: L.b }; });
      }
      const feats = (x) => { const f = []; for (let k = 1; k <= KF; k++) f.push(Math.sin(2 * Math.PI * k * x), Math.cos(2 * Math.PI * k * x)); return f; };
      function ctxConst(N, x) { // φ(x) part of the first pre-activation (constant over ŷ)
        const L = N[0], f = feats(x), c = new Float64Array(L.no);
        for (let o = 0; o < L.no; o++) { let s = L.b[o]; for (let i = 0; i < f.length; i++) s += L.W[o * L.ni + i] * f[i]; c[o] = s; }
        return c;
      }
      const z0 = new Float64Array(HID), h1 = new Float64Array(HID), z1 = new Float64Array(HID), h2 = new Float64Array(HID), z2 = new Float64Array(HID), h3 = new Float64Array(HID);
      const gA = new Float64Array(HID), gB = new Float64Array(HID), gC = new Float64Array(HID);
      const sg = (z) => 1 / (1 + Math.exp(-z));
      // E(x, ŷ) and (optionally) ∇_ŷ E for the net N; c0 = ctxConst(N, x)
      function energy(N, c0, y0, y1, grad) {
        const L0 = N[0], L1 = N[1], L2 = N[2], L3 = N[3], n0 = L0.ni, W0 = L0.W, W1 = L1.W, W2 = L2.W, W3 = L3.W;
        for (let o = 0; o < HID; o++) { const z = c0[o] + W0[o * n0 + n0 - 2] * y0 + W0[o * n0 + n0 - 1] * y1; z0[o] = z; h1[o] = z * sg(z); }
        for (let o = 0; o < HID; o++) { let s = L1.b[o]; const off = o * HID; for (let i = 0; i < HID; i++) s += W1[off + i] * h1[i]; z1[o] = s; h2[o] = s * sg(s); }
        for (let o = 0; o < HID; o++) { let s = L2.b[o]; const off = o * HID; for (let i = 0; i < HID; i++) s += W2[off + i] * h2[i]; z2[o] = s; h3[o] = s * sg(s); }
        let E = L3.b[0]; for (let i = 0; i < HID; i++) E += W3[i] * h3[i];
        E += LAM * (y0 * y0 + y1 * y1);
        if (!grad) return E;
        for (let i = 0; i < HID; i++) { const s = sg(z2[i]); gA[i] = W3[i] * s * (1 + z2[i] * (1 - s)); }
        gB.fill(0); for (let o = 0; o < HID; o++) { const g = gA[o], off = o * HID; for (let j = 0; j < HID; j++) gB[j] += W2[off + j] * g; }
        for (let j = 0; j < HID; j++) { const s = sg(z1[j]); gB[j] *= s * (1 + z1[j] * (1 - s)); }
        gC.fill(0); for (let o = 0; o < HID; o++) { const g = gB[o], off = o * HID; for (let j = 0; j < HID; j++) gC[j] += W1[off + j] * g; }
        for (let j = 0; j < HID; j++) { const s = sg(z0[j]); gC[j] *= s * (1 + z0[j] * (1 - s)); }
        let g0 = 0, g1 = 0; for (let o = 0; o < HID; o++) { g0 += W0[o * n0 + n0 - 2] * gC[o]; g1 += W0[o * n0 + n0 - 1] * gC[o]; }
        return [E, g0 + 2 * LAM * y0, g1 + 2 * LAM * y1];
      }
      const clampY = (v) => Math.max(-8, Math.min(8, v)), clampL = (v) => Math.max(-1e3, Math.min(1e3, v));
      function think(N, c0, y0, steps, alpha) {
        let y = y0.slice(); const path = [y.slice()], Es = [];
        for (let i = 0; i < steps; i++) { const r = energy(N, c0, y[0], y[1], true); Es.push(r[0]); y = [clampY(y[0] - alpha * r[1]), clampY(y[1] - alpha * r[2])]; path.push(y.slice()); }
        Es.push(energy(N, c0, y[0], y[1], false));
        return { path, Es };
      }
      function basinMin(N, c0, Gd) { // start at the grid argmin, refine by small-step descent, then a finite-difference Hessian
        let best = Infinity, y = [0, 0];
        for (let r = 0; r < Gd.n; r++) for (let c = 0; c < Gd.n; c++) if (Gd.E[r][c] < best) { best = Gd.E[r][c]; y = [EXT[0] + (EXT[1] - EXT[0]) * c / (Gd.n - 1), EXT[3] - (EXT[3] - EXT[2]) * r / (Gd.n - 1)]; }
        const hess = (p) => { const e = 1e-3, g = (q) => { const r = energy(N, c0, q[0], q[1], true); return [r[1], r[2]]; };
          const ax = g([p[0] + e, p[1]]), bx = g([p[0] - e, p[1]]), ay = g([p[0], p[1] + e]), by = g([p[0], p[1] - e]);
          const a = (ax[0] - bx[0]) / (2 * e), d = (ay[1] - by[1]) / (2 * e), b = ((ax[1] - bx[1]) + (ay[0] - by[0])) / (4 * e);
          const tr = (a + d) / 2, disc = Math.sqrt(Math.max(0, tr * tr - (a * d - b * b))); return [tr - disc, tr + disc]; };
        let lam = hess(y); const st = 0.5 / Math.max(0.5, Math.abs(lam[1]));
        for (let i = 0; i < 160; i++) { const r = energy(N, c0, y[0], y[1], true); y = [y[0] - st * r[1], y[1] - st * r[2]]; }
        return { y, lam: hess(y) };
      }
      function grid(N, c0, n) {
        const E = []; let lo = Infinity, hi = -Infinity;
        for (let r = 0; r < n; r++) { const yv = EXT[3] - (EXT[3] - EXT[2]) * r / (n - 1), row = new Array(n);
          for (let c = 0; c < n; c++) { const xv = EXT[0] + (EXT[1] - EXT[0]) * c / (n - 1), e = energy(N, c0, xv, yv, false); row[c] = e; if (e < lo) lo = e; if (e > hi) hi = e; }
          E.push(row); }
        return { E, lo, hi, n };
      }
      // the Algorithm-1 objective of the stored slice, re-run on a fresh batch (B = 512, N = 4, α per sample in [0.5, 2], Langevin 0.05)
      const BATCH = (() => { const r = lib.rng(2024), B = 512, out = [];
        for (let k = 0; k < B; k++) { const x = r(), mu = muOf(x), s = sigOf(x);
          out.push({ x, f: feats(x), y: [mu[0] + s * r.normal(), mu[1] + s * r.normal()], y0: [r.normal(), r.normal()], a: Math.exp((2 * r() - 1) * Math.log(2)), eta: Array.from({ length: NSTEP }, () => [0.05 * r.normal(), 0.05 * r.normal()]) }); }
        return out; })();
      function liveLoss(N, batch) {
        batch = batch || BATCH; let tot = 0; const L0 = N[0], c0 = new Float64Array(HID);
        for (const s of batch) {
          for (let o = 0; o < HID; o++) { let v = L0.b[o]; for (let i = 0; i < 6; i++) v += L0.W[o * L0.ni + i] * s.f[i]; c0[o] = v; }
          let y0 = s.y0[0], y1 = s.y0[1];
          for (let i = 0; i < NSTEP; i++) { const g = energy(N, c0, y0, y1, true); y0 = clampL(y0 - s.a * g[1] + s.eta[i][0]); y1 = clampL(y1 - s.a * g[2] + s.eta[i][1]); }
          tot += (y0 - s.y[0]) ** 2 + (y1 - s.y[1]) ** 2;
        }
        return tot / batch.length;
      }

      // ---------------- the stored loss slice (log10), Catmull-Rom interpolation ----------------
      const SL = { final: PL.loss.map(r => r.map(v => Math.log10(v))), init: PL.init_slice.loss.map(r => r.map(v => Math.log10(v))) };
      const NL = PL.loss.length; // 41, rows = b from −1 (row 0) to +1, cols = a
      const LLO = Math.min(lib.gridRange(SL.final)[0], lib.gridRange(SL.init)[0]), LHI = Math.max(lib.gridRange(SL.final)[1], lib.gridRange(SL.init)[1]);
      const cr = (p0, p1, p2, p3, t) => 0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t + (-p0 + 3 * p1 - 3 * p2 + p3) * t * t * t);
      function sliceLog(which, a, b) {
        const G = SL[which], u = (Math.max(-1, Math.min(1, a)) + 1) / 2 * (NL - 1), v = (Math.max(-1, Math.min(1, b)) + 1) / 2 * (NL - 1);
        const i = Math.min(NL - 2, Math.floor(v)), j = Math.min(NL - 2, Math.floor(u)), tv = v - i, tu = u - j;
        const R = (k) => G[Math.max(0, Math.min(NL - 1, k))], Cc = (row, k) => row[Math.max(0, Math.min(NL - 1, k))];
        const rows = [-1, 0, 1, 2].map(di => { const row = R(i + di); return cr(Cc(row, j - 1), Cc(row, j), Cc(row, j + 1), Cc(row, j + 2), tu); });
        return cr(rows[0], rows[1], rows[2], rows[3], tv);
      }
      const sliceL = (which, a, b) => Math.pow(10, sliceLog(which, a, b));

      // ---------------- layout ----------------
      const CTXS = [0, 1, 2, 4].map(i => M.contexts[i]);
      const S = { slice: 'final', a: 0, b: 0, ci: 0, y0: [-1.6, 1.4], view: 'map', k: NSTEP, learn: [], live: null, liveFor: '' };
      const view3 = { yaw: -0.62, pitch: 0.95 };
      const row = h('div', { class: 'fig-row', style: { columnGap: '18px' } }); stage.appendChild(row);
      const FL = lib.frame(row, { label: 'Loss landscape · learning', sub: 'over weights: ℒ(θ*+a·d₁+b·d₂) · drag' });
      FL.wrap.style.flex = '1 1 225px';
      const cvL = lib.canvas(FL.frame, 300, 332, { label: 'Training loss over a two-dimensional slice of weight space. Drag the blue point to choose the weights.' });
      const roL = h('div', { class: 'readout' }); FL.wrap.appendChild(roL);
      const FE = lib.frame(row, { label: 'Energy landscape · thinking', sub: 'over predictions: E<sub>θ</sub>(x, ŷ) · click' });
      FE.wrap.style.flex = '1 1 225px';
      const cvE = lib.canvas(FE.frame, 300, 332, { label: 'Energy over the two-dimensional prediction at the chosen weights, with a thinking path.' });
      const roE = h('div', { class: 'readout' }); FE.wrap.appendChild(roE);
      const row2 = h('div', { class: 'fig-row' }); stage.appendChild(row2);
      const FP = lib.frame(row2, { label: 'Two descents', sub: 'outer loop: ℒ per learning step · inner loop: E per thinking step', dashed: true });
      FP.wrap.style.flex = '1 1 100%';
      const pr = h('div', { class: 'fig-row', style: { gap: '0', flexWrap: 'wrap' } }); FP.frame.appendChild(pr);
      const pA = h('div', { style: { flex: '1 1 260px', minWidth: '0' } }), pB = h('div', { style: { flex: '1 1 260px', minWidth: '0' } }); pr.append(pA, pB);
      const cvP1 = lib.canvas(pA, 300, 150, { label: 'Training loss at each learning step' });
      const cvP2 = lib.canvas(pB, 300, 150, { label: 'Energy at each thinking step' });

      const controls = h('div', { class: 'controls' }); stage.appendChild(controls);
      controls.appendChild(h('span', { class: 'fig-label' }, 'context'));
      const segC = lib.segmented({ label: 'Context', options: CTXS.map((c, i) => [i, 'x=' + c.x]), value: 0, onchange: (v) => { S.ci = v; S.live = null; recompute(false); } });
      controls.appendChild(segC.el);
      controls.appendChild(h('span', { class: 'fig-label' }, 'slice around'));
      const segS = lib.segmented({ label: 'Slice around', options: [['final', 'trained θ*'], ['init', 'initial θ₀']], value: 'final', onchange: (v) => { stopAll(); S.slice = v; S.learn = []; S.live = null; recompute(false); } });
      controls.appendChild(segS.el);
      const controls2 = h('div', { class: 'controls' }); stage.appendChild(controls2);
      controls2.appendChild(lib.button('think', () => animateThink(), { primary: true }));
      controls2.appendChild(lib.button('learn', () => animateLearn()));
      controls2.appendChild(lib.button('reset θ', () => { stopAll(); S.a = 0; S.b = 0; S.learn = []; S.live = null; recompute(false); }));
      controls2.appendChild(h('span', { class: 'fig-label' }, 'view'));
      const segV = lib.segmented({ label: 'View', options: [['map', 'map'], ['3d', '3-D']], value: 'map', onchange: (v) => { S.view = v; draw(); } });
      controls2.appendChild(segV.el);
      ctx.setCaption('Left: each point is a full set of ' + NPARAM.toLocaleString('en-US') + ' weights; color = training loss (log scale, lighter = higher). Right: for the weights at the blue dot, the energy of every candidate ŷ for one context x (darker = lower). Crosshair: the target mean μ(x). Dashed contour: where the basin sits at θ*.');

      // ---------------- computation state ----------------
      let net = null, c0 = null, G = null, P = null, B = null, refBasin = {}, busyT = null;
      function recompute(fast) {
        net = makeNet(S.slice, S.a, S.b); const x = CTXS[S.ci].x; c0 = ctxConst(net, x);
        G = grid(net, c0, fast ? 30 : 60); G.q = lib.quantile(G.E, 0.55); energyLayer = null;
        P = think(net, c0, S.y0, NSTEP, ALPHA); B = basinMin(net, c0, G);
        draw();
        if (!fast) scheduleLive();
      }
      let livePending = false;
      function scheduleLive() {
        clearTimeout(busyT);
        if (!ctx.visible()) { livePending = true; return; }
        livePending = false;
        const key = S.slice + S.a.toFixed(4) + S.b.toFixed(4);
        if (S.liveFor === key && S.live != null) return;
        busyT = setTimeout(() => { S.live = liveLoss(net); S.liveFor = key; readouts(); }, 120);
      }
      function basinAtStar(ci) { // the trained landscape's basin outline for this context (dashed reference)
        if (refBasin[ci]) return refBasin[ci];
        const N = makeNet('final', 0, 0), cc = ctxConst(N, CTXS[ci].x), g = grid(N, cc, 60);
        return (refBasin[ci] = { E: g.E, level: g.lo + 0.5 });
      }

      // ---------------- drawing ----------------
      const BOX = { x: 30, y: 6, w: 262, h: 262 };
      function offscreen(dpr, paint) { // cached layer at the canvas' pixel density, drawn in box-local coordinates
        const o = document.createElement('canvas'); o.width = Math.round(BOX.w * dpr); o.height = Math.round(BOX.h * dpr);
        const g = o.getContext('2d'); g.scale(dpr, dpr); paint(g); return o;
      }
      const lossLayers = {};
      let energyLayer = null;
      function buildEnergyLayer() {
        energyLayer = offscreen(cvE.dpr, (g) => {
          lib.heatmap(g, G.E, 0, 0, BOX.w, BOX.h, { range: [G.lo, G.q], gamma: 0.7 });
          const levels = Array.from({ length: 10 }, (_, i) => G.lo + (G.q - G.lo) * Math.pow((i + 1) / 10, 1.4));
          lib.contours(g, G.E, 0, 0, BOX.w, BOX.h, levels, { color: 'rgba(17,17,17,0.22)', width: 0.8 });
          const moved = S.slice !== 'final' || Math.hypot(S.a, S.b) > 1e-6;
          if (moved) { const rb = basinAtStar(S.ci); g.save(); g.setLineDash([2, 2.6]); lib.contours(g, rb.E, 0, 0, BOX.w, BOX.h, [rb.level], { color: 'rgba(17,17,17,0.9)', width: 1.3 }); g.restore(); }
        });
      }
      function colorbar(c, x, y, w, hh, lo, hi, ticks, fmt, label) {
        for (let i = 0; i < w; i++) { const col = lib.cmap(i / (w - 1)); c.fillStyle = `rgb(${col[0]},${col[1]},${col[2]})`; c.fillRect(x + i, y, 1.2, hh); }
        c.strokeStyle = C.faint; c.lineWidth = 0.6; c.strokeRect(x + 0.5, y + 0.5, w - 1, hh - 1);
        ticks.forEach(v => { const px = x + (v - lo) / (hi - lo) * w; if (px < x - 1 || px > x + w + 1) return; c.fillStyle = C.ink; c.fillRect(px - 0.5, y + hh, 1, 3); lib.text(c, fmt(v), px, y + hh + 4, { size: 10.5, kind: 'mono', color: C.muted, align: 'center' }); });
        if (label) lib.text(c, label, x + w + 6, y - 2, { size: 10.5, kind: 'mono', color: C.muted });
      }
      function frameTicks(c, vals, toPx, fmt) {
        vals.forEach(v => { const [px, py] = toPx(v);
          lib.text(c, fmt(v), Math.max(BOX.x + 7, Math.min(BOX.x + BOX.w - 7, px)), BOX.y + BOX.h + 4, { size: 10.5, kind: 'mono', color: C.muted, align: 'center' });
          lib.text(c, fmt(v), BOX.x - 4, Math.max(BOX.y + 6, Math.min(BOX.y + BOX.h - 8, py)), { size: 10.5, kind: 'mono', color: C.muted, align: 'right', baseline: 'middle' }); });
      }
      const abToPx = (a, b) => [BOX.x + (a + 1) / 2 * BOX.w, BOX.y + (1 - (b + 1) / 2) * BOX.h];
      const pxToAb = (px, py) => [Math.max(-1, Math.min(1, (px - BOX.x) / BOX.w * 2 - 1)), Math.max(-1, Math.min(1, 1 - (py - BOX.y) / BOX.h * 2))];
      const yToPx = (y) => M.toPx(y, BOX);
      function crosshair(c, p, col) {
        c.save(); c.strokeStyle = col || C.ink; c.lineWidth = 1.2; c.setLineDash([3, 3]); c.beginPath(); c.arc(p[0], p[1], 11, 0, 7); c.stroke(); c.setLineDash([]);
        c.beginPath(); c.moveTo(p[0] - 6, p[1]); c.lineTo(p[0] + 6, p[1]); c.moveTo(p[0], p[1] - 6); c.lineTo(p[0], p[1] + 6); c.stroke(); c.restore();
      }
      function drawLossMap() {
        const c = cvL.ctx, Gs = SL[S.slice];
        if (!lossLayers[S.slice]) lossLayers[S.slice] = offscreen(cvL.dpr, (g) => {
          lib.heatmap(g, Gs, 0, 0, BOX.w, BOX.h, { range: [LLO, LHI], flipY: true, gamma: 0.9 });
          const lev = []; for (let v = -1; v <= 1.75; v += 0.25) lev.push(v);
          lib.contours(g, Gs, 0, 0, BOX.w, BOX.h, lev, { flipY: true, color: 'rgba(17,17,17,0.22)', width: 0.8 });
        });
        c.drawImage(lossLayers[S.slice], BOX.x, BOX.y, BOX.w, BOX.h);
        c.strokeStyle = C.ink; c.lineWidth = 1; c.strokeRect(BOX.x + 0.5, BOX.y + 0.5, BOX.w - 1, BOX.h - 1);
        frameTicks(c, [-1, 0, 1], (v) => abToPx(v, v), (v) => v === 0 ? '0' : (v < 0 ? '−1' : '1'));
        lib.text(c, 'a (along d₁) →', BOX.x + BOX.w, BOX.y + BOX.h + 18, { size: 10.5, kind: 'mono', color: C.ink, align: 'right' });
        lib.text(c, 'b', BOX.x - 4, BOX.y + 22, { size: 10.5, kind: 'mono', color: C.ink, align: 'right' });
        lib.text(c, '↑', BOX.x - 4, BOX.y + 36, { size: 10.5, kind: 'mono', color: C.ink, align: 'right' });
        const ctr = abToPx(0, 0);
        c.save(); c.strokeStyle = C.ink; c.lineWidth = 1; c.beginPath(); c.moveTo(ctr[0] - 5, ctr[1]); c.lineTo(ctr[0] + 5, ctr[1]); c.moveTo(ctr[0], ctr[1] - 5); c.lineTo(ctr[0], ctr[1] + 5); c.stroke(); c.restore();
        lib.text(c, S.slice === 'final' ? 'θ*' : 'θ₀', ctr[0] + 9, ctr[1] + 7, { size: 11, kind: 'mono', color: C.ink });
        if (S.learn.length > 1) lib.line(c, S.learn.map(p => abToPx(p[0], p[1])), { color: C.blue, width: 1.6 });
        const p = abToPx(S.a, S.b); lib.dot(c, p[0], p[1], 6, C.blue, { stroke: '#fff', lw: 2 });
        colorbar(c, BOX.x, 306, 170, 6, LLO, LHI, [-1, 0, 1], (v) => ({ '-1': '0.1', '0': '1', '1': '10' })[String(v)], 'ℒ (log)');
      }
      function drawEnergyMap() {
        const c = cvE.ctx;
        c.save(); c.beginPath(); c.rect(BOX.x, BOX.y, BOX.w, BOX.h); c.clip();
        if (!energyLayer) buildEnergyLayer();
        c.drawImage(energyLayer, BOX.x, BOX.y, BOX.w, BOX.h);
        const mu = muOf(CTXS[S.ci].x); crosshair(c, yToPx(mu));
        const pm = yToPx(B.y); c.save(); c.strokeStyle = C.ink; c.fillStyle = '#fff'; c.lineWidth = 1.3; c.beginPath(); c.moveTo(pm[0], pm[1] - 6); c.lineTo(pm[0] + 6, pm[1]); c.lineTo(pm[0], pm[1] + 6); c.lineTo(pm[0] - 6, pm[1]); c.closePath(); c.fill(); c.stroke(); c.restore();
        if (Math.hypot(B.y[0] - mu[0], B.y[1] - mu[1]) > 0.25) lib.text(c, 'min', pm[0] + 8, pm[1] - 6, { size: 10.5, kind: 'mono', color: C.ink });
        const k = Math.min(S.k, P.path.length - 1), pts = P.path.slice(0, k + 1).map(yToPx);
        lib.line(c, pts, { color: C.blue, width: 1.8 });
        pts.forEach((q, i) => lib.dot(c, q[0], q[1], i === pts.length - 1 ? 5.5 : 3, i === pts.length - 1 ? C.blue : '#fff', { stroke: C.blue, lw: 1.5 }));
        c.restore();
        // off-plot marker for an escaped prediction
        const last = P.path[k];
        if (Math.abs(last[0]) > EXT[1] || Math.abs(last[1]) > EXT[3]) lib.text(c, 'ŷ' + sub(k) + ' off plot (' + last[0].toFixed(1) + ', ' + last[1].toFixed(1) + ')', BOX.x + 6, BOX.y + BOX.h - 18, { size: 10.5, kind: 'mono', color: C.bad });
        const q0 = yToPx(P.path[0]); if (q0[0] > BOX.x && q0[0] < BOX.x + BOX.w) lib.text(c, 'ŷ₀', q0[0] - 8, q0[1] - 16, { size: 11, kind: 'mono', color: C.ink, align: 'right' });
        c.strokeStyle = C.ink; c.lineWidth = 1; c.strokeRect(BOX.x + 0.5, BOX.y + 0.5, BOX.w - 1, BOX.h - 1);
        frameTicks(c, [-2, 0, 2], (v) => yToPx([v, v]), (v) => v < 0 ? '−' + Math.abs(v) : String(v));
        lib.text(c, 'ŷ₁ (prediction, 1st coord.) →', BOX.x + BOX.w, BOX.y + BOX.h + 18, { size: 10.5, kind: 'mono', color: C.ink, align: 'right' });
        lib.text(c, 'ŷ₂', BOX.x - 4, BOX.y + 66, { size: 10.5, kind: 'mono', color: C.ink, align: 'right' });
        lib.text(c, '↑', BOX.x - 4, BOX.y + 80, { size: 10.5, kind: 'mono', color: C.ink, align: 'right' });
        colorbar(c, BOX.x, 306, 170, 6, G.lo, G.q, [G.lo, G.q], (v) => v.toFixed(2), 'E (clipped)');
      }
      function drawSurface(cv, grid3, range, overlay) {
        const c = cv.ctx;
        const proj = lib.surface(c, grid3, { cx: 150, cy: 168, size: 190, yaw: view3.yaw, pitch: view3.pitch, zscale: 120, range, step: 1, wire: 'rgba(17,17,17,0.18)' });
        overlay && overlay(c, proj);
        lib.text(c, 'drag to rotate', 296, 314, { size: 10.5, kind: 'mono', color: C.faint, align: 'right' });
      }
      function draw3D() {
        const GsTD = SL[S.slice].slice().reverse(); // top-down: row 0 = b = +1
        cvL.clear();
        drawSurface(cvL, GsTD, [LLO, LHI], (c, proj) => {
          const u = (S.a + 1) / 2, v = (1 - S.b) / 2, q = proj(u, v, sliceLog(S.slice, S.a, S.b));
          if (S.learn.length > 1) lib.line(c, S.learn.map(p => { const r = proj((p[0] + 1) / 2, (1 - p[1]) / 2, sliceLog(S.slice, p[0], p[1])); return [r[0], r[1]]; }), { color: C.blue, width: 1.6 });
          lib.dot(c, q[0], q[1], 5.5, C.blue, { stroke: '#fff', lw: 2 }); lib.text(c, 'θ', q[0] + 8, q[1] - 16, { size: 12, kind: 'mono', color: C.ink });
          lib.text(c, 'height = log₁₀ ℒ', 6, 314, { size: 10.5, kind: 'mono', color: C.muted });
        });
        cvE.clear();
        const hiE = G.lo + (G.q - G.lo) * 1.6, Ec = G.E.map(r => r.map(v => Math.min(v, hiE)));
        drawSurface(cvE, Ec, [G.lo, hiE], (c, proj) => {
          const k = Math.min(S.k, P.path.length - 1);
          const U = (y) => [(Math.max(EXT[0], Math.min(EXT[1], y[0])) - EXT[0]) / (EXT[1] - EXT[0]), (EXT[3] - Math.max(EXT[2], Math.min(EXT[3], y[1]))) / (EXT[3] - EXT[2])];
          const pts = P.path.slice(0, k + 1).map((y, i) => { const u = U(y), r = proj(u[0], u[1], Math.min(P.Es[i], hiE)); return [r[0], r[1] - 2]; });
          lib.line(c, pts, { color: C.blue, width: 2 }); pts.forEach((q, i) => lib.dot(c, q[0], q[1], i === pts.length - 1 ? 5 : 2.6, i === pts.length - 1 ? C.blue : '#fff', { stroke: C.blue, lw: 1.4 }));
          const mu = muOf(CTXS[S.ci].x), um = U(mu), qm = proj(um[0], um[1], Math.min(energy(net, c0, mu[0], mu[1], false), hiE));
          crosshair(c, qm);
          lib.text(c, 'height = E (clipped)', 6, 314, { size: 10.5, kind: 'mono', color: C.muted });
        });
      }
      function drawPlots() {
        // learning curve
        let c = cvP1.ctx; cvP1.clear();
        const pts = S.learn.length ? S.learn.map((p, t) => [t, sliceL(S.slice, p[0], p[1])]) : [[0, sliceL(S.slice, S.a, S.b)]];
        const nT = Math.max(10, pts.length - 1), floor = PL.noise_floor_this_batch;
        const lo = Math.min(floor * 0.4, ...pts.map(p => p[1])), hi = Math.max(10, ...pts.map(p => p[1]));
        const yt = [0.1, 1, 10, 100].filter(v => v >= lo * 0.999 && v <= hi * 1.001);
        const ax = lib.axes(c, { x: 44, y: 22, w: 240, h: 92, xlim: [0, nT], ylim: [lo, hi], ylog: true, xticks: [0, nT], yticks: yt, yfmt: (v) => String(v), size: 10.5 });
        lib.text(c, 'LEARNING · ℒ(θₜ)', 44, 4, { size: 10.5, kind: 'mono', color: C.ink, spacing: 1 });
        lib.text(c, 'learning step t', 284, 136, { size: 10.5, kind: 'mono', color: C.muted, align: 'right' });
        c.save(); c.setLineDash([4, 3]); lib.line(c, [[ax.X(0), ax.Y(floor)], [ax.X(nT), ax.Y(floor)]], { color: C.ink, width: 1 }); c.restore();
        lib.text(c, 'noise floor ' + floor.toFixed(3), ax.X(0) + 4, ax.Y(floor) + 2, { size: 10, kind: 'mono', color: C.muted });
        lib.plot(c, ax, pts, { color: C.blue, width: 1.8, markers: pts.length < 40 ? 2.2 : 0 });
        const e = pts[pts.length - 1]; lib.dot(c, ax.X(e[0]), ax.Y(e[1]), 4, C.blue);
        // thinking curve
        c = cvP2.ctx; cvP2.clear();
        const k = Math.min(S.k, P.path.length - 1), Es = P.Es.slice(0, k + 1);
        const elo = Math.min(...P.Es), ehi = Math.max(...P.Es), pad = (ehi - elo) * 0.08 + 1e-3;
        const ax2 = lib.axes(c, { x: 44, y: 22, w: 240, h: 92, xlim: [0, NSTEP], ylim: [elo - pad, ehi + pad], xticks: [0, 1, 2, 3, 4], yticks: [elo, ehi], yfmt: (v) => v.toFixed(2), size: 10.5 });
        lib.text(c, 'THINKING · E(x, ŷᵢ)', 44, 4, { size: 10.5, kind: 'mono', color: C.ink, spacing: 1 });
        lib.text(c, 'thinking step i', 284, 136, { size: 10.5, kind: 'mono', color: C.muted, align: 'right' });
        lib.plot(c, ax2, Es.map((v, i) => [i, v]), { color: C.blue, width: 1.8, markers: 2.6 });
      }
      function readouts() {
        const Ls = sliceL(S.slice, S.a, S.b), key = S.slice + S.a.toFixed(4) + S.b.toFixed(4);
        const live = (S.live != null && S.liveFor === key) ? S.live.toFixed(3) : '…';
        roL.innerHTML = `<span>a <b>${S.a.toFixed(2)}</b> b <b>${S.b.toFixed(2)}</b></span><span>ℒ slice <b>${Ls < 1 ? Ls.toFixed(3) : Ls.toFixed(2)}</b></span><span>ℒ live, new batch <b>${live}</b></span>`;
        const k = Math.min(S.k, P.path.length - 1), y = P.path[k], mu = muOf(CTXS[S.ci].x), d = Math.hypot(y[0] - mu[0], y[1] - mu[1]);
        roE.innerHTML = `<span>ŷ${sub(k)} <b>(${y[0].toFixed(2)}, ${y[1].toFixed(2)})</b></span><span>‖ŷ${sub(k)}−μ‖ <b>${d.toFixed(3)}</b></span><span>E <b>${P.Es[k].toFixed(3)}</b></span><span>min ◇ <b>(${B.y[0].toFixed(2)}, ${B.y[1].toFixed(2)})</b></span><span>Hessian λ at ◇ <b>${B.lam[0].toFixed(2)}, ${B.lam[1].toFixed(2)}</b></span>`;
      }
      function draw() {
        if (S.view === '3d') draw3D();
        else { cvL.clear(); drawLossMap(); cvE.clear(); drawEnergyMap(); }
        drawPlots(); readouts();
      }

      // ---------------- interaction ----------------
      let drag = null, rot = null;
      cvL.canvas.style.cursor = 'crosshair'; cvE.canvas.style.cursor = 'crosshair';
      const onDownL = (ev) => {
        if (S.view === '3d') { rot = { x: ev.clientX, y: ev.clientY, yaw: view3.yaw, pitch: view3.pitch }; return; }
        stopAll(); drag = true; S.learn = []; moveTheta(ev, true);
      };
      function moveTheta(ev, fast) { const [px, py] = cvL.toLocal(ev); const [a, b] = pxToAb(px, py); S.a = a; S.b = b; S.k = NSTEP; recompute(fast); }
      let rafPending = false, lastEv = null;
      const capture = (el, ev) => { try { el.setPointerCapture(ev.pointerId); } catch (e) { /* synthetic or stale pointer */ } };
      cvL.canvas.addEventListener('pointerdown', (ev) => { capture(cvL.canvas, ev); onDownL(ev); });
      cvL.canvas.addEventListener('pointermove', (ev) => {
        if (rot) { rotate(ev); return; }
        if (!drag) return; lastEv = ev; if (rafPending) return; rafPending = true;
        requestAnimationFrame(() => { rafPending = false; if (drag && lastEv) moveTheta(lastEv, true); });
      });
      const up = () => {
        if (drag) { drag = null; if (lastEv) { const [px, py] = cvL.toLocal(lastEv); [S.a, S.b] = pxToAb(px, py); } lastEv = null; recompute(false); }
        rot = null;
      };
      cvL.canvas.addEventListener('pointerup', up); cvL.canvas.addEventListener('pointercancel', up);
      function rotate(ev) { view3.yaw = rot.yaw + (ev.clientX - rot.x) * 0.01; view3.pitch = Math.max(0.3, Math.min(1.35, rot.pitch + (ev.clientY - rot.y) * 0.006)); draw(); }
      cvE.canvas.addEventListener('pointerdown', (ev) => {
        if (S.view === '3d') { capture(cvE.canvas, ev); rot = { x: ev.clientX, y: ev.clientY, yaw: view3.yaw, pitch: view3.pitch }; return; }
        const [px, py] = cvE.toLocal(ev); if (px < BOX.x || px > BOX.x + BOX.w || py < BOX.y || py > BOX.y + BOX.h) return;
        S.y0 = M.fromPx(px, py, BOX); P = think(net, c0, S.y0, NSTEP, ALPHA); animateThink();
      });
      cvE.canvas.addEventListener('pointermove', (ev) => { if (rot) rotate(ev); });
      cvE.canvas.addEventListener('pointerup', up); cvE.canvas.addEventListener('pointercancel', up);

      let tThink = null, tLearn = null;
      function stopAll() { clearInterval(tThink); clearInterval(tLearn); tThink = tLearn = null; }
      function animateThink() {
        clearInterval(tThink); P = think(net, c0, S.y0, NSTEP, ALPHA);
        if (lib.reducedMotion) { S.k = NSTEP; draw(); return; }
        S.k = 0; draw();
        tThink = setInterval(() => { S.k++; draw(); if (S.k >= NSTEP) { clearInterval(tThink); tThink = null; } }, 420);
      }
      function animateLearn() {
        stopAll(); S.k = NSTEP;
        if (!S.learn.length || Math.hypot(S.learn[S.learn.length - 1][0] - S.a, S.learn[S.learn.length - 1][1] - S.b) > 1e-9) S.learn = [[S.a, S.b]];
        const eta = 0.12, hstep = 0.01; let it = 0; // plain GD on the plane; each step capped at 0.05 in (a, b)
        const tick = () => {
          const L = (a, b) => sliceL(S.slice, a, b);
          const ga = (L(S.a + hstep, S.b) - L(S.a - hstep, S.b)) / (2 * hstep), gb = (L(S.a, S.b + hstep) - L(S.a, S.b - hstep)) / (2 * hstep);
          let da = -eta * ga, db = -eta * gb; const n = Math.hypot(da, db), cap = 0.05;
          if (n > cap) { da *= cap / n; db *= cap / n; }
          S.a = Math.max(-1, Math.min(1, S.a + da)); S.b = Math.max(-1, Math.min(1, S.b + db)); S.learn.push([S.a, S.b]); it++;
          const done = Math.hypot(da, db) < 0.001 || it >= 160;
          recompute(!done);
          return done;
        };
        if (lib.reducedMotion) { while (!tick()); return; }
        tLearn = setInterval(() => { if (tick()) { clearInterval(tLearn); tLearn = null; } }, 60);
      }

      // The stage is sticky (base.css). When it is taller than the window, stick it higher so that everything down to
      // the step bar stays on screen while the reader works through the steps; a stacked (very tall) stage just scrolls.
      function fitSticky() {
        const bar = stage.querySelector('.stepbar'); if (!bar) return;
        const need = bar.getBoundingClientRect().bottom - stage.getBoundingClientRect().top + 14, room = window.innerHeight;
        if (need + 24 <= room) { stage.style.position = ''; stage.style.top = ''; }
        else if (need > room * 1.45) { stage.style.position = 'static'; stage.style.top = ''; }
        else { stage.style.position = ''; stage.style.top = Math.round(room - need) + 'px'; }
      }
      window.addEventListener('resize', () => requestAnimationFrame(fitSticky));
      setTimeout(fitSticky, 0);

      recompute(false);
      const presets = {
        0: () => { S.slice = 'final'; S.a = 0; S.b = 0; },
        1: () => { S.slice = 'final'; S.a = -0.5; S.b = -0.5; },
        2: () => { S.slice = 'final'; S.a = -0.8; S.b = 0.6; },
        3: () => { S.slice = 'final'; S.a = -0.8; S.b = 0.6; },
        4: () => { S.slice = 'init'; S.a = 0; S.b = 0; },
        5: () => { S.slice = 'final'; S.a = 0; S.b = 0; },
      };
      return {
        step(i) {
          stopAll(); S.learn = []; S.live = null; S.y0 = [-1.6, 1.4]; S.ci = 0; segC.set(0);
          presets[i] && presets[i]();
          segS.set(S.slice); S.view = i === 5 ? '3d' : 'map'; segV.set(S.view);
          S.k = NSTEP; recompute(false);
          if (i === 2) animateThink();
          if (i === 3) animateLearn();
        },
        show() { fitSticky(); if (livePending) scheduleLive(); },
        hide() { stopAll(); },
      };
    },
  });

  function decodeDirs() {
    const s = atob(dirsB64()), n = s.length / 2, out = new Float64Array(n);
    for (let i = 0; i < n; i++) { let v = s.charCodeAt(2 * i) | (s.charCodeAt(2 * i + 1) << 8); if (v >= 32768) v -= 65536; out[i] = v / 8192; }
    return [out.subarray(0, n / 2), out.subarray(n / 2)];
  }
  // Raw Gaussian directions d1 (numpy default_rng(1)) and d2 (default_rng(2)) for W0..W3 (row-major W[out][in]), int16 / 8192, little-endian.
  function dirsB64() { return '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'; }
})();
