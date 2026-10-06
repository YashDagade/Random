/* Panel 04: the energy landscape of the toy 2-D EBT, live from its weights (EBT.toy2d).
   Top view (heatmap + contours, click to drop ŷ0 and watch gradient descent), a lit 3-D surface (drag to rotate) that
   shares the same overlays, a context strip (x in [0,1), grids recomputed live and cached), and a slice through the
   minimum with the local quadratic from a finite-difference Hessian. Checkpoints show what training changes. */
EBT.panel({
  id: 'landscape',
  nav: 'The energy landscape',
  title: 'The energy landscape',
  lede: 'Fix the context and the weights, and the energy becomes a surface over the space of possible predictions. Its shape decides what the model predicts and how easily thinking gets there.',
  text: `
    <p>For a fixed context $x$ and fixed weights $\\theta$, plotting $E_\\theta(x,\\hat y)$ over every candidate $\\hat y$ gives the <b>energy landscape</b>: in the paper's words, a surface "resulting from mapping all possible predictions to scalar values" (p.40). Thinking is a walk downhill on it. This is not the loss landscape, whose axes are weights (next section).</p>
    <p>The landscape here is measured, not drawn. It belongs to a toy EBT trained for this explainer with the paper's recipe (Algorithm 1 with the Sec 3.3 regularizers). Its prediction $\\hat y=(\\hat y_1,\\hat y_2)$ is a point in the plane. Its context $x\\in[0,1)$ is a phase: the training targets for context $x$ scatter around</p>
    <div class="eq">$$\\mu(x)=\\big(1.3\\sin 2\\pi x,\\; 0.9\\sin 4\\pi x\\big)$$<span class="why">a figure-eight, with Gaussian noise σ(x) = 0.03 + 0.27 sin²(π(x − 0.25)): nearly exact at x = 0.25, very noisy at x = 0.75.</span></div>
    <p>Every energy, gradient and path in the figure is computed in your browser from the toy's weights.</p>`,
  steps: [
    { label: 'The floor is the prediction space', html: '<p>Each point of the floor is one candidate $\\hat y$; three are marked with their energies. Here the space is $\\mathbb R^2$, so you can see all of it. In the paper it is far larger: a 50,277-dimensional logit vector per text token (Table D.3), a 3,136-dimensional VAE latent per video frame (p.12), a whole 128×128 image for denoising. Any picture of those landscapes, Fig 3 included, is a 2-D slice.</p>' },
    { label: 'Height is the energy', html: '<p>The 3-D view lifts every candidate to its energy (drag to rotate). Hover the top view to read $E$ anywhere, with the downhill step $-\\alpha\\nabla_{\\hat y}E$ as an arrow and a pin in 3-D. Low means compatible. Only differences matter, so the absolute height, about $-0.6$ at the bottom, carries no meaning by itself.</p>' },
    { label: 'The context moves the basin', html: '<p>Drag the context strip or watch the sweep. The weights stay fixed, yet each $x$ gives a new landscape whose basin slides along the dashed figure-eight of training targets. One network encodes a whole family of landscapes, one per context. Each frame is a fresh grid of 1,600 to 4,096 energy evaluations.</p>' },
    { label: 'Convex around the truth', html: '<p>Starts from all over the plane flow into one basin centred on $\\mu(x)$, the mean of this context\'s targets (crosshair; dashed ring $=2\\sigma$). The slice cuts the bowl through its minimum $\\hat y^*$: it is convex and close to the quadratic $E^*+\\tfrac12\\lambda s^2$, with $\\lambda$ a Hessian eigenvalue. Training "pushes the energy landscape to be convex surrounding the ground truth solution" (p.7). Convexity is what makes the answer independent of the random start and lets plain gradient descent find it.</p>' },
    { label: 'Before training', html: '<p>The same starts on the weights from training step 50: the bowl is already convex, but it sits in the wrong place, so every path converges, confidently, to a wrong answer. Try <b>step 0</b>: the untrained surface is nearly flat, with non-convex patches (hatched). Learning moves and shapes the basin; thinking only rolls down it.</p>' },
    { label: 'Shaped for its optimizer', html: '<p>Near its minimum the bowl has curvature $\\lambda\\approx0.85$ in every direction, and one step multiplies the distance to the bottom by $|1-\\alpha\\lambda|$. Training used step sizes around $\\alpha_0=1$, so at $\\alpha=1$ a start lands almost on the bottom in one or two jumps; at $\\alpha=0.5$ it takes about ten steps; at $\\alpha=2$ it overshoots and zig-zags. The landscape was learned together with the optimizer that descends it.</p>' },
  ],
  after: `
    <h3>Reading the paper's Fig 3 against the toy</h3>
    <figure class="paper-fig"><img src="media/paper/fig03.png" alt="Paper Figure 3: a red-to-blue 3-D energy landscape for the context 'The dog caught the', with a white path from step 0 down into a narrow basin marked Converged."></figure>
    <p class="note" style="border-top:0;padding-top:0">Paper Fig 3 (p.5), "Thinking Process Visualization". A schematic adapted from a loss-landscape paper [57]; not measured from a trained model.</p>
    <p>Fig 3 shows a predicted token distribution sharpening onto "frisbee" while a path runs down a wide slope into a narrow well, and says to "Continue Thinking Process until Energy Convergence". Its caption adds that uncertainty "can be represented by landscapes that are harder to optimize or by landscapes with many local minima" (p.5).</p>
    <p>The toy shows the measurable core of that picture: one smooth basin per context, placed on the conditional mean, with a curvature tuned to the step size. It does <em>not</em> show uncertainty in the basin shape: its low-noise and high-noise contexts have nearly the same curvature (panel 06 explains why). The convexity that makes thinking reliable also has a cost the paper names: when one context has many good answers, a convex bowl "merges" them into one averaged prediction (p.29), which is why EBTs struggle with many-mode distributions (p.17).</p>
    <p class="note">Toy: E(x, ŷ) = MLP([φ(x), ŷ]) + 0.02‖ŷ‖², φ(x) = Fourier features of x, SiLU MLP 8→64→64→64→1, 12,000 training steps with 2 to 6 unrolled descent steps, α randomized in [0.5, 2]. Top-view colors and contours are scaled per landscape; 3-D heights share one scale per context (the trained landscape's), clipped at its 72nd percentile, so an untrained model looks flat. Paths use α = 0.5 unless you change it.</p>`,
  source: [
    { kind: 'toy', note: 'toy 2-D EBT, energies computed live from its weights' },
    { kind: 'paper', note: 'Fig 3 (p.5), Sec 3.2 (p.7), App. H (p.40)' },
  ],
  figure(stage, ctx) {
    const { lib } = ctx, h = lib.h, C = lib.C, M = window.EBT && window.EBT.toy2d;
    if (!M || !M.ready) { stage.appendChild(h('p', { class: 'callout warn' }, 'Toy model data missing (data/toy2d.json).')); return {}; }
    const D = M.data, EXT = M.extent, LAST = M.nCkpt - 1, STEPS = M.steps, TAU = Math.PI * 2;
    const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
    const fx = (v, d = 2) => !isFinite(v) ? '–' : (v < 0 && Math.abs(v) >= 0.5 * Math.pow(10, -d) ? '−' : '') + Math.abs(v).toFixed(d);
    const vec = (p, d = 2) => '(' + fx(p[0], d) + ', ' + fx(p[1], d) + ')';
    const muOf = (x) => [1.3 * Math.sin(TAU * x), 0.9 * Math.sin(2 * TAU * x)];
    const sigOf = (x) => 0.03 + 0.27 * Math.pow(Math.sin(Math.PI * (x - 0.25)), 2);
    const ck50 = Math.max(0, STEPS.indexOf(50));
    const CAP = 0.72, GAM = 0.65, XQ = 240;
    const S = { ck: LAST, x: 0.25, alpha: 0.5, paths: [], probe: null, cands: true, pinT: 0, hatch: false, sweep: null, view: { yaw: -0.62, pitch: 0.9 }, rot: false, coarse: false };

    // ---------------------------------------------------------------- fields (energy grids), cached
    const fields = new Map();
    function field(ck, x, n) {
      const xq = ((Math.round(x * XQ) / XQ) % 1 + 1) % 1, key = ck + '|' + xq.toFixed(5) + '|' + n;
      if (fields.has(key)) return fields.get(key);
      const g = M.grid(ck, xq, n, EXT), E = g.E, q = lib.quantile(E, CAP);
      let best = Infinity, br = 0, bc = 0; E.forEach((row, r) => row.forEach((v, c) => { if (v < best) { best = v; br = r; bc = c; } }));
      const F = { E, n, lo: g.lo, hi: g.hi, q, key, x: xq, ck, gmin: [EXT[0] + (EXT[1] - EXT[0]) * bc / (n - 1), EXT[3] - (EXT[3] - EXT[2]) * br / (n - 1)] };
      F.tf = (v) => Math.pow(clamp((v - F.lo) / (F.q - F.lo || 1), 0, 1), GAM);
      F.levels = Array.from({ length: 11 }, (_, k) => F.lo + (F.q - F.lo) * Math.pow((k + 1) / 12, 1 / GAM));
      // convexity from second differences of the grid: non-convex cells (Hessian not positive definite)
      const hh = (EXT[1] - EXT[0]) / (n - 1); F.nonconvex = []; let tot = 0;
      for (let r = 1; r < n - 1; r++) for (let c = 1; c < n - 1; c++) {
        const exx = (E[r][c + 1] - 2 * E[r][c] + E[r][c - 1]) / (hh * hh), eyy = (E[r + 1][c] - 2 * E[r][c] + E[r - 1][c]) / (hh * hh), exy = (E[r + 1][c + 1] - E[r + 1][c - 1] - E[r - 1][c + 1] + E[r - 1][c - 1]) / (4 * hh * hh);
        tot++; if (!(exx > 0 && exx * eyy - exy * exy > 0)) F.nonconvex.push([r, c]);
      }
      F.convexFrac = 1 - F.nonconvex.length / tot;
      if (fields.size > 260) fields.delete(fields.keys().next().value);
      fields.set(key, F); return F;
    }
    const cur = () => field(S.ck, S.x, S.coarse ? 40 : 64);
    // shared 3-D height scale per context: the trained landscape's range
    const zRange = (x) => { const F = field(LAST, x, 40); return [F.lo, F.q]; };
    // refined minimum + Hessian (finite differences of the exact gradient)
    const basins = new Map();
    function hess(ck, x, y) {
      const e = 1e-3, g = (p) => M.energyGrad(ck, x, p).g;
      const gx1 = g([y[0] + e, y[1]]), gx0 = g([y[0] - e, y[1]]), gy1 = g([y[0], y[1] + e]), gy0 = g([y[0], y[1] - e]);
      const a = (gx1[0] - gx0[0]) / (2 * e), d = (gy1[1] - gy0[1]) / (2 * e), b = ((gx1[1] - gx0[1]) + (gy1[0] - gy0[0])) / (4 * e);
      const tr = (a + d) / 2, disc = Math.sqrt(Math.max(0, tr * tr - (a * d - b * b)));
      return { H: [[a, b], [b, d]], eig: [tr - disc, tr + disc] };
    }
    function basin(F) {
      if (basins.has(F.key)) return basins.get(F.key);
      let y = F.gmin.slice(); const h0 = hess(F.ck, F.x, y), st = 0.8 / Math.max(0.05, h0.eig[1]);
      for (let i = 0; i < 60; i++) { const g = M.energyGrad(F.ck, F.x, y).g; y = [clamp(y[0] - st * g[0], EXT[0], EXT[1]), clamp(y[1] - st * g[1], EXT[2], EXT[3])]; }
      const B = Object.assign({ y, E: M.energy(F.ck, F.x, y) }, hess(F.ck, F.x, y)); basins.set(F.key, B); return B;
    }

    // ---------------------------------------------------------------- DOM
    const row = h('div', { class: 'fig-row' }); stage.appendChild(row);
    const FT = lib.frame(row, { label: 'Top view', sub: 'E<sub>θ</sub>(x, ŷ) over ŷ ∈ ℝ² · click to drop ŷ<sub>0</sub>' }); FT.wrap.style.flex = '1 1 280px';
    const F3 = lib.frame(row, { label: '3-D view', sub: 'height = energy · drag to rotate' }); F3.wrap.style.flex = '1 1 280px';
    const sideW = (fr) => clamp(Math.floor(fr.clientWidth || 300), 260, 520);
    const TW = sideW(FT.frame), cvT = lib.canvas(FT.frame, TW, TW, { label: 'Energy landscape seen from above. Click to drop a starting guess and watch gradient descent.' });
    const W3 = sideW(F3.frame), cv3 = lib.canvas(F3.frame, W3, W3, { label: 'The same energy landscape as a 3-D surface. Drag or use the arrow keys to rotate.' });
    cv3.canvas.tabIndex = 0; cvT.canvas.style.cursor = 'crosshair'; cv3.canvas.style.cursor = 'grab';
    const FS = lib.frame(stage, { label: 'Context x', sub: 'drag to change the context · the weights stay fixed' });
    const SW = clamp(Math.floor(FS.frame.clientWidth || 600), 300, 700), cvS = lib.canvas(FS.frame, SW, 64, { label: 'Context strip: the noise level of the training targets as a function of x, and the current x' });
    cvS.canvas.style.cursor = 'ew-resize';
    const FL = lib.frame(stage, { label: 'Slice through the minimum', sub: 'E(ŷ* + s·u) along the dashed line · dashed: quadratic from the Hessian' });
    const LW = SW, LH = SW < 480 ? 140 : 132, cvL = lib.canvas(FL.frame, LW, LH, { label: 'One-dimensional cut through the bottom of the basin, with the local quadratic approximation' });
    const c1 = h('div', { class: 'controls' }); stage.appendChild(c1);
    c1.appendChild(h('span', { class: 'fig-label' }, 'weights'));
    const ckSeg = lib.segmented({ label: 'Training checkpoint', options: [[0, 'step 0'], [ck50, 'step 50'], [LAST, 'step ' + STEPS[LAST].toLocaleString('en-US')]], value: LAST, onchange: (v) => { S.ck = v; S.hatch = S.hatch || v === 0; refresh(true); } });
    c1.appendChild(ckSeg.el);
    c1.appendChild(h('span', { class: 'fig-label' }, 'step size α'));
    const aSeg = lib.segmented({ label: 'Step size', options: [[0.25, '0.25'], [0.5, '0.5'], [1, '1'], [2, '2']], value: 0.5, onchange: (v) => { S.alpha = v; refresh(true); } });
    c1.appendChild(aSeg.el);
    const c2 = h('div', { class: 'controls' }); stage.appendChild(c2);
    c2.appendChild(lib.button('drop 8 starts', () => { dropStarts(); }, { primary: true }));
    c2.appendChild(lib.button('sweep x', () => { startSweep(S.x, S.x + 1, 7); }));
    const hatchBtn = lib.button('non-convex regions', () => { S.hatch = !S.hatch; hatchBtn.setAttribute('aria-pressed', String(S.hatch)); render(); });
    hatchBtn.setAttribute('aria-pressed', 'false'); c2.appendChild(hatchBtn);
    c2.appendChild(lib.button('clear', () => { S.paths = []; S.cands = false; render(); }));
    const ro = h('div', { class: 'readout', 'aria-live': 'polite' }); stage.appendChild(ro);

    // ---------------------------------------------------------------- geometry
    const tb = { x: 30, y: 8, w: TW - 38, h: TW - 38 }; // top-view plot box
    const P = (y) => M.toPx(y, tb), Pinv = (px, py) => M.fromPx(px, py, tb);
    const uv = (y) => [(clamp(y[0], EXT[0], EXT[1]) - EXT[0]) / (EXT[1] - EXT[0]), (EXT[3] - clamp(y[1], EXT[2], EXT[3])) / (EXT[3] - EXT[2])];
    const L3 = () => ({ cx: W3 / 2, cy: W3 * 0.6, size: W3 * 0.6, zs: W3 * 0.34 });

    // ---------------------------------------------------------------- bitmaps
    function heatBitmap(F) {
      const k = 'top|' + TW + '|' + cvT.dpr + '|' + (S.hatch ? 1 : 0); if (F.bm && F.bm.k === k) return F.bm.cv;
      const n = F.n, img = document.createElement('canvas'); img.width = n; img.height = n; const g0 = img.getContext('2d'), id = g0.createImageData(n, n);
      for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) { const col = lib.cmap(F.tf(F.E[r][c])), i = 4 * (r * n + c); id.data[i] = col[0]; id.data[i + 1] = col[1]; id.data[i + 2] = col[2]; id.data[i + 3] = 255; }
      g0.putImageData(id, 0, 0);
      const cv = document.createElement('canvas'); cv.width = Math.round(tb.w * cvT.dpr); cv.height = Math.round(tb.h * cvT.dpr);
      const g = cv.getContext('2d'); g.setTransform(cvT.dpr, 0, 0, cvT.dpr, 0, 0); g.imageSmoothingEnabled = true; g.imageSmoothingQuality = 'high';
      const cw = tb.w / (n - 1), ch = tb.h / (n - 1); g.drawImage(img, -cw / 2, -ch / 2, tb.w + cw, tb.h + ch);
      lib.contours(g, F.E, 0, 0, tb.w, tb.h, F.levels, { color: 'rgba(17,17,17,0.2)', width: 1 });
      if (S.hatch && F.nonconvex.length) {
        g.save(); g.beginPath(); F.nonconvex.forEach(([r, c]) => g.rect((c - 0.5) * cw, (r - 0.5) * ch, cw, ch)); g.clip();
        g.strokeStyle = 'rgba(212,66,28,0.55)'; g.lineWidth = 1; g.beginPath(); for (let d = -tb.h; d < tb.w; d += 6) { g.moveTo(d, tb.h); g.lineTo(d + tb.h, 0); } g.stroke(); g.restore();
      }
      F.bm = { k, cv }; return cv;
    }
    const LV = (() => { const l = [-0.45, -0.55, 0.7], m = Math.hypot(...l); return l.map(v => v / m); })();
    function surfaceBitmap(F, zr, step) {
      const v = S.view, k = ['s3', W3, cv3.dpr, v.yaw.toFixed(3), v.pitch.toFixed(3), step, zr[0].toFixed(4), zr[1].toFixed(4)].join('|');
      if (F.sb && F.sb.k === k) return F.sb;
      const cv = (F.sb && F.sb.cv) || document.createElement('canvas'); cv.width = Math.round(W3 * cv3.dpr); cv.height = Math.round(W3 * cv3.dpr);
      const g = cv.getContext('2d'); g.setTransform(cv3.dpr, 0, 0, cv3.dpr, 0, 0); g.clearRect(0, 0, W3, W3);
      const o = L3(), E = F.E, n = F.n, cyw = Math.cos(v.yaw), syw = Math.sin(v.yaw), cp = Math.cos(v.pitch), sp = Math.sin(v.pitch);
      const zn = (z) => clamp((z - zr[0]) / (zr[1] - zr[0] || 1), 0, 1) * o.zs;
      const project = (u, w, z) => { const X0 = (u - 0.5) * o.size, Y0 = (w - 0.5) * o.size, Z = zn(z); const xr = X0 * cyw - Y0 * syw, yr = X0 * syw + Y0 * cyw; return [o.cx + xr, o.cy + yr * cp - Z * sp, yr * sp + Z * cp]; };
      // floor
      const fl = [[0, 0], [1, 0], [1, 1], [0, 1]].map(([a, b]) => project(a, b, zr[0]));
      g.strokeStyle = C.rule; g.lineWidth = 1; g.beginPath(); fl.forEach((p, i) => i ? g.lineTo(p[0], p[1]) : g.moveTo(p[0], p[1])); g.closePath(); g.stroke();
      const quads = [], cell = step / (n - 1) * o.size;
      for (let r = 0; r + step < n; r += step) for (let c = 0; c + step < n; c += step) {
        const z00 = E[r][c], z01 = E[r][c + step], z11 = E[r + step][c + step], z10 = E[r + step][c];
        const u0 = c / (n - 1), u1 = (c + step) / (n - 1), w0 = r / (n - 1), w1 = (r + step) / (n - 1);
        const p = [project(u0, w0, z00), project(u1, w0, z01), project(u1, w1, z11), project(u0, w1, z10)];
        const du = ((zn(z01) - zn(z00)) + (zn(z11) - zn(z10))) / (2 * cell), dv = ((zn(z10) - zn(z00)) + (zn(z11) - zn(z01))) / (2 * cell);
        const lam = (-du * LV[0] - dv * LV[1] + LV[2]) / Math.hypot(du, dv, 1);
        quads.push({ p, d: p[0][2] + p[1][2] + p[2][2] + p[3][2], z: (z00 + z01 + z11 + z10) / 4, s: 0.62 + 0.38 * clamp(lam, 0, 1) });
      }
      quads.sort((a, b) => a.d - b.d);
      g.lineJoin = 'round'; g.lineWidth = 0.7;
      for (const q of quads) { const col = lib.cmap(F.tf(q.z)), fill = 'rgb(' + Math.round(col[0] * q.s) + ',' + Math.round(col[1] * q.s) + ',' + Math.round(col[2] * q.s) + ')'; g.fillStyle = fill; g.strokeStyle = fill; g.beginPath(); g.moveTo(q.p[0][0], q.p[0][1]); for (let k2 = 1; k2 < 4; k2++) g.lineTo(q.p[k2][0], q.p[k2][1]); g.closePath(); g.fill(); g.stroke(); }
      F.sb = { k, cv, project, fl }; return F.sb;
    }

    // ---------------------------------------------------------------- paths (real gradient descent on the toy)
    function descend(y0) {
      let y = y0.slice(); const pts = [y.slice()], Es = [M.energy(S.ck, S.x, y)];
      for (let i = 0; i < 40; i++) {
        const { g } = M.energyGrad(S.ck, S.x, y); const yn = [clamp(y[0] - S.alpha * g[0], -4, 4), clamp(y[1] - S.alpha * g[1], -4, 4)];
        const moved = Math.hypot(yn[0] - y[0], yn[1] - y[1]); y = yn; pts.push(y.slice()); Es.push(M.energy(S.ck, S.x, y));
        if (moved < 2e-4 && i > 2) break;
      }
      return { y0: y0.slice(), pts, Es, t: 0 };
    }
    const STARTS = (D.trajectory_starts || []).slice(0, 8);
    function dropStarts(animate = true) { S.cands = false; S.paths = STARTS.map(s => descend(s)); if (!animate || lib.reducedMotion) S.paths.forEach(p => { p.t = p.pts.length - 1; }); kick(); render(); }
    function recomputePaths() { S.paths = S.paths.map(p => { const q = descend(p.y0); q.t = q.pts.length - 1; return q; }); }
    const posAt = (p, t) => { const i = Math.min(p.pts.length - 1, Math.floor(t)), f = t - i; if (i >= p.pts.length - 1) return p.pts[p.pts.length - 1]; const a = p.pts[i], b = p.pts[i + 1], e = lib.ease(f); return [a[0] + (b[0] - a[0]) * e, a[1] + (b[1] - a[1]) * e]; };

    // ---------------------------------------------------------------- rendering
    const CANDS = [[-1.0, 1.4, 'A'], [0.3, -1.9, 'B'], [1.3, 0.0, 'C']];
    const halo = (c, fn) => { c.save(); c.strokeStyle = 'rgba(255,255,255,0.9)'; c.lineWidth = 3.2; fn(true); c.restore(); c.save(); fn(false); c.restore(); };
    function crosshair(c, q, r, col) { halo(c, (hl) => { c.strokeStyle = hl ? 'rgba(255,255,255,0.9)' : col; c.lineWidth = hl ? 3 : 1.3; c.beginPath(); c.moveTo(q[0] - r, q[1]); c.lineTo(q[0] + r, q[1]); c.moveTo(q[0], q[1] - r); c.lineTo(q[0], q[1] + r); c.stroke(); }); }
    function xmark(c, q, r, col) { halo(c, (hl) => { c.strokeStyle = hl ? 'rgba(255,255,255,0.9)' : col; c.lineWidth = hl ? 3.2 : 1.6; c.beginPath(); c.moveTo(q[0] - r, q[1] - r); c.lineTo(q[0] + r, q[1] + r); c.moveTo(q[0] + r, q[1] - r); c.lineTo(q[0] - r, q[1] + r); c.stroke(); }); }
    function label(c, s, x, y, o = {}) { c.save(); c.font = (o.weight || 500) + ' ' + (o.size || 12) + 'px ' + lib.F.mono; c.textAlign = o.align || 'left'; c.textBaseline = o.base || 'middle'; c.lineWidth = 3; c.strokeStyle = 'rgba(255,255,255,0.92)'; c.lineJoin = 'round'; c.strokeText(s, x, y); c.fillStyle = o.color || C.ink; c.fillText(s, x, y); c.restore(); }
    function sliceDir(B) {
      const p = S.paths.length ? S.paths[S.paths.length - 1].y0 : null;
      if (p) { const d = [p[0] - B.y[0], p[1] - B.y[1]], m = Math.hypot(d[0], d[1]); if (m > 0.2) return [d[0] / m, d[1] / m]; }
      return [1, 0];
    }
    function drawTop(F, B) {
      const c = cvT.ctx; cvT.clear();
      c.drawImage(heatBitmap(F), tb.x, tb.y, tb.w, tb.h);
      [-2, -1, 0, 1, 2].forEach(v => { const px = P([v, 0])[0], py = P([0, v])[1]; lib.text(c, fx(v, 0), px, tb.y + tb.h + 5, { size: 11, kind: 'mono', color: C.muted, align: 'center' }); lib.text(c, fx(v, 0), tb.x - 6, py, { size: 11, kind: 'mono', color: C.muted, align: 'right', baseline: 'middle' }); });
      c.save(); c.beginPath(); c.rect(tb.x, tb.y, tb.w, tb.h); c.clip();
      // data manifold: where the targets of all contexts live
      const man = D.dataset.curve.mu.map(P);
      c.save(); c.strokeStyle = 'rgba(255,255,255,0.75)'; c.lineWidth = 2.6; c.beginPath(); man.forEach((q, i) => i ? c.lineTo(q[0], q[1]) : c.moveTo(q[0], q[1])); c.stroke();
      c.strokeStyle = 'rgba(17,17,17,0.55)'; c.lineWidth = 1; c.setLineDash([3, 4]); c.stroke(); c.restore();
      // targets of this context (eval samples with nearby x), mean and 2σ ring
      const mu = muOf(S.x), sg = sigOf(S.x), pm = P(mu);
      D.dataset.sample.forEach(s => { let dx = Math.abs(s[0] - S.x); dx = Math.min(dx, 1 - dx); if (dx < 0.03) { const q = P([s[1], s[2]]); lib.dot(c, q[0], q[1], 1.9, C.ink, { stroke: '#fff', lw: 0.8 }); } });
      halo(c, (hl) => { c.strokeStyle = hl ? 'rgba(255,255,255,0.85)' : C.ink; c.lineWidth = hl ? 2.6 : 1; c.setLineDash([3, 3]); c.beginPath(); c.arc(pm[0], pm[1], Math.max(5, 2 * sg / (EXT[1] - EXT[0]) * tb.w), 0, TAU); c.stroke(); });
      crosshair(c, pm, 8, C.ink);
      // slice line
      const u = sliceDir(B), a0 = P([B.y[0] - 6 * u[0], B.y[1] - 6 * u[1]]), a1 = P([B.y[0] + 6 * u[0], B.y[1] + 6 * u[1]]);
      c.save(); c.strokeStyle = 'rgba(17,17,17,0.5)'; c.lineWidth = 1; c.setLineDash([6, 5]); c.beginPath(); c.moveTo(a0[0], a0[1]); c.lineTo(a1[0], a1[1]); c.stroke(); c.restore();
      xmark(c, P(B.y), 5, C.blue);
      // candidates of step 1
      if (S.cands) CANDS.forEach(([a, b, nm]) => { const q = P([a, b]), e = M.energy(S.ck, S.x, [a, b]); lib.dot(c, q[0], q[1], 5, '#fff', { stroke: C.ink, lw: 1.5 }); label(c, nm + '  E = ' + fx(e), q[0] + 9, q[1], { size: 12, weight: 600 }); });
      // paths
      S.paths.forEach((p, j) => {
        const n = Math.floor(p.t), pts = p.pts.slice(0, n + 1).map(P); if (p.t > n && n < p.pts.length - 1) pts.push(P(posAt(p, p.t)));
        lib.line(c, pts, { color: C.blue, width: 1.8 });
        p.pts.slice(1, n + 1).forEach(y => { const q = P(y); lib.dot(c, q[0], q[1], 2.4, '#fff', { stroke: C.blue, lw: 1.2 }); });
        const s0 = P(p.y0); lib.dot(c, s0[0], s0[1], 4, C.ink, { stroke: '#fff', lw: 1.3 });
        const hd = P(posAt(p, p.t)); lib.dot(c, hd[0], hd[1], j === S.paths.length - 1 ? 5 : 4, C.blue, { stroke: '#fff', lw: 1.4 });
      });
      // hover probe
      if (S.probe) {
        const eg = M.energyGrad(S.ck, S.x, S.probe), q0 = P(S.probe), q1 = P([S.probe[0] - S.alpha * eg.g[0], S.probe[1] - S.alpha * eg.g[1]]);
        lib.arrow(c, q0[0], q0[1], q1[0], q1[1], { color: C.ink, width: 1.4, head: 7 });
        lib.dot(c, q0[0], q0[1], 3, C.ink);
        const right = q0[0] < tb.x + tb.w * 0.55; label(c, 'E ' + fx(eg.E) + '  |∇E| ' + Math.hypot(eg.g[0], eg.g[1]).toFixed(2), q0[0] + (right ? 9 : -9), q0[1] + (q0[1] < tb.y + 20 ? 14 : -12), { align: right ? 'left' : 'right', size: 11.5 });
      }
      label(c, 'ŷ₁ →', tb.x + tb.w - 5, tb.y + tb.h - 10, { align: 'right', size: 12 });
      label(c, '↑ ŷ₂', tb.x + 6, tb.y + 12, { size: 12 });
      c.restore();
      c.strokeStyle = C.rule; c.lineWidth = 1; c.strokeRect(tb.x + 0.5, tb.y + 0.5, tb.w - 1, tb.h - 1);
    }
    function draw3D(F, B) {
      const c = cv3.ctx; cv3.clear(); const zr = zRange(F.x);
      const sb = surfaceBitmap(F, zr, S.rot ? 2 : 1); c.drawImage(sb.cv, 0, 0, W3, W3);
      const pr = (y, E) => { const t = uv(y); return sb.project(t[0], t[1], E); };
      const o = L3(), fl = sb.fl;
      // axes: ŷ₁ along the v = 1 edge, ŷ₂ along the u = 0 edge, E as a vertical at the leftmost floor corner
      const mid = (a, b) => [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2], ctr = mid(fl[0], fl[2]);
      const lab = (p, s) => { const d = [p[0] - ctr[0], p[1] - ctr[1]], m = Math.hypot(d[0], d[1]) || 1; label(c, s, p[0] + d[0] / m * 16, p[1] + d[1] / m * 12, { align: 'center', size: 12, color: C.muted }); };
      lab(mid(fl[3], fl[2]), 'ŷ₁'); lab(mid(fl[0], fl[3]), 'ŷ₂');
      let li = 0; fl.forEach((p, i) => { if (p[0] < fl[li][0]) li = i; });
      const base = fl[li]; c.save(); c.strokeStyle = C.faint; c.lineWidth = 1; c.beginPath(); c.moveTo(base[0], base[1]); c.lineTo(base[0], base[1] - o.zs * Math.sin(S.view.pitch)); c.stroke(); c.restore();
      label(c, 'E', base[0] - 8, base[1] - o.zs * Math.sin(S.view.pitch), { align: 'right', size: 12, color: C.muted });
      // truth and minimum
      const mu = muOf(S.x); crosshair(c, pr(mu, M.energy(S.ck, S.x, mu)), 6, C.ink);
      xmark(c, pr(B.y, B.E), 4.5, C.blue);
      if (S.cands) CANDS.forEach(([a, b, nm]) => {
        const e = M.energy(S.ck, S.x, [a, b]), f = lib.easeOut(S.pinT), top = pr([a, b], zr[0] + (Math.min(e, zr[1]) - zr[0]) * f), bot = pr([a, b], zr[0]);
        c.save(); c.strokeStyle = C.ink; c.lineWidth = 1; c.setLineDash([2, 3]); c.beginPath(); c.moveTo(bot[0], bot[1]); c.lineTo(top[0], top[1]); c.stroke(); c.restore();
        lib.dot(c, top[0], top[1], 4.5, '#fff', { stroke: C.ink, lw: 1.5 }); label(c, nm, top[0] + 8, top[1] - 6, { size: 12, weight: 600 });
      });
      S.paths.forEach((p, j) => {
        const n = Math.floor(p.t), pts = []; for (let i = 0; i <= n && i < p.pts.length; i++) pts.push(pr(p.pts[i], p.Es[i]));
        if (p.t > n && n < p.pts.length - 1) { const y = posAt(p, p.t); pts.push(pr(y, M.energy(S.ck, S.x, y))); }
        lib.line(c, pts, { color: 'rgba(255,255,255,0.9)', width: 3.6 }); lib.line(c, pts, { color: C.blue, width: 1.8 });
        const hd = pts[pts.length - 1]; lib.dot(c, hd[0], hd[1], j === S.paths.length - 1 ? 5 : 4, C.blue, { stroke: '#fff', lw: 1.4 });
      });
      if (S.probe) {
        const e = M.energy(S.ck, S.x, S.probe), top = pr(S.probe, e), bot = pr(S.probe, zr[0]);
        c.save(); c.strokeStyle = C.ink; c.lineWidth = 1; c.setLineDash([3, 3]); c.beginPath(); c.moveTo(bot[0], bot[1]); c.lineTo(top[0], top[1]); c.stroke(); c.restore();
        lib.dot(c, top[0], top[1], 3.5, C.ink, { stroke: '#fff', lw: 1 }); label(c, 'E ' + fx(e), top[0] + 7, top[1] - 8, { size: 11.5 });
      }
      label(c, 'drag to rotate', W3 - 8, 12, { align: 'right', size: 11, color: C.faint });
      label(c, 'height clipped at E = ' + fx(zr[1], 1), 8, W3 - 12, { size: 11, color: C.faint });
    }
    const named = (D.contexts || []).map(cx => cx.x);
    function drawStrip() {
      const c = cvS.ctx; cvS.clear(); const x0 = 34, x1 = SW - 12, yb = 46, yt = 8, X = (v) => x0 + v * (x1 - x0), Ys = (s) => yb - (s / 0.3) * (yb - yt);
      c.strokeStyle = C.faint; c.lineWidth = 1; c.beginPath(); c.moveTo(x0, yb + 0.5); c.lineTo(x1, yb + 0.5); c.stroke();
      [0, 0.25, 0.5, 0.75, 1].forEach(v => lib.text(c, String(v), X(v), yb + 4, { size: 10.5, kind: 'mono', color: C.muted, align: 'center' }));
      named.forEach(v => { c.strokeStyle = C.faint; c.beginPath(); c.moveTo(X(v), yb); c.lineTo(X(v), yb + 3); c.stroke(); });
      c.beginPath(); for (let i = 0; i <= 200; i++) { const v = i / 200; i ? c.lineTo(X(v), Ys(sigOf(v))) : c.moveTo(X(v), Ys(sigOf(v))); } c.strokeStyle = C.ink; c.lineWidth = 1.2; c.stroke();
      lib.text(c, 'σ(x)', 4, yt, { size: 11, kind: 'mono', color: C.muted });
      lib.text(c, 'exact', X(0.25), yb - 13, { size: 10, kind: 'mono', color: C.muted, align: 'center' });
      const px = X(S.x); c.strokeStyle = C.blue; c.lineWidth = 1.6; c.beginPath(); c.moveTo(px, yt - 2); c.lineTo(px, yb); c.stroke(); lib.dot(c, px, Ys(sigOf(S.x)), 4.5, C.blue, { stroke: '#fff', lw: 1.3 });
      const lt = 'x = ' + S.x.toFixed(3), rightSide = S.x < 0.82;
      lib.text(c, lt, px + (rightSide ? 6 : -6), yt + 2, { size: 11.5, kind: 'mono', color: C.blue, align: rightSide ? 'left' : 'right', weight: 600 });
    }
    function drawSlice(F, B) {
      const c = cvL.ctx; cvL.clear(); const u = sliceDir(B), lu = u[0] * (B.H[0][0] * u[0] + B.H[0][1] * u[1]) + u[1] * (B.H[1][0] * u[0] + B.H[1][1] * u[1]);
      const sR = 2.6, N = 105, key = F.key + '|' + u[0].toFixed(3) + '|' + u[1].toFixed(3);
      if (!F.slice || F.slice.key !== key) { const pts = []; for (let i = 0; i < N; i++) { const s = -sR + 2 * sR * i / (N - 1); pts.push([s, M.energy(S.ck, S.x, [B.y[0] + s * u[0], B.y[1] + s * u[1]])]); } F.slice = { key, pts }; }
      const pts = F.slice.pts; let lo = Math.min(...pts.map(p => p[1])), hi = Math.max(...pts.map(p => p[1])); hi = lo + Math.max(0.5, (hi - lo) * 1.08);
      const ax = lib.axes(c, { x: 40, y: 8, w: LW - 52, h: LH - 34, xlim: [-sR, sR], ylim: [lo, hi], xticks: [-2, -1, 0, 1, 2], yticks: [lo, (lo + hi) / 2, hi], xfmt: (v) => fx(v, 0), yfmt: (v) => fx(v, 1), size: 10.5 });
      c.save(); c.beginPath(); c.rect(40, 8, LW - 52, LH - 34); c.clip();
      lib.plot(c, ax, pts.map(p => [p[0], B.E + 0.5 * lu * p[0] * p[0]]), { color: C.ink, width: 1, dash: [4, 4] });
      lib.plot(c, ax, pts, { color: C.blue, width: 2 });
      const p = S.paths.length ? S.paths[S.paths.length - 1] : null;
      if (p) { const n = Math.min(p.pts.length - 1, Math.floor(p.t)); for (let i = 0; i <= n; i++) { const y = p.pts[i], s = (y[0] - B.y[0]) * u[0] + (y[1] - B.y[1]) * u[1]; lib.dot(c, ax.X(s), ax.Y(p.Es[i]), i === n ? 4 : 2.6, i === n ? C.blue : '#fff', { stroke: C.blue, lw: 1.2 }); } }
      c.restore();
      lib.text(c, 's', LW - 12, LH - 24, { size: 11, kind: 'mono', color: C.ink, align: 'right' });
      lib.text(c, 'λ along this line ' + lu.toFixed(2) + (lu > 0 ? ' > 0: convex' : ' ≤ 0: not convex'), LW - 14, 12, { size: 11, kind: 'mono', color: C.muted, align: 'right' });
    }
    function readout(F, B) {
      const mu = muOf(S.x), e = B.eig, last = S.paths.length ? S.paths[S.paths.length - 1] : null;
      const k1 = Math.abs(1 - S.alpha * e[0]), k2 = Math.abs(1 - S.alpha * e[1]);
      let s = `<span>x <b>${S.x.toFixed(3)}</b></span><span>σ(x) <b>${sigOf(S.x).toFixed(3)}</b></span><span>weights <b>step ${STEPS[S.ck].toLocaleString('en-US')}</b></span>` +
        `<span>argmin ŷ* <b>${vec(B.y)}</b></span><span>μ(x) <b>${vec(mu)}</b></span><span>E(ŷ*) <b>${fx(B.E, 3)}</b></span>` +
        `<span>Hessian eigenvalues <b>${fx(e[0])}, ${fx(e[1])}</b>${e[0] > 0 ? ' (convex at ŷ*)' : ''}</span><span>convex on <b>${Math.round(100 * F.convexFrac)}%</b> of the view</span>` +
        `<span>one step at α = ${S.alpha} keeps <b>${Math.min(k1, k2).toFixed(2)}–${Math.max(k1, k2).toFixed(2)}</b> of the distance</span>`;
      if (last) { const n = Math.min(last.pts.length - 1, Math.floor(last.t)), y = last.pts[n]; s += `<span>last path: step <b>${n}</b>, E ${fx(last.Es[0])} → <b>${fx(last.Es[n])}</b>, distance to μ(x) <b>${Math.hypot(y[0] - mu[0], y[1] - mu[1]).toFixed(3)}</b></span>`; }
      ro.innerHTML = s;
    }
    function render() {
      const F = cur(), B = S.coarse ? { y: F.gmin, E: M.energy(S.ck, S.x, F.gmin), H: [[1, 0], [0, 1]], eig: [NaN, NaN] } : basin(F);
      drawTop(F, B); draw3D(F, B); drawStrip(); if (!S.coarse) { drawSlice(F, B); readout(F, B); }
    }

    // ---------------------------------------------------------------- animation
    const loop = lib.loop((dt) => {
      let busy = false;
      if (S.sweep) {
        const sw = S.sweep; sw.t += dt; const f = Math.min(1, sw.t / sw.dur); S.x = ((sw.from + (sw.to - sw.from) * f) % 1 + 1) % 1; S.coarse = f < 1;
        if (f >= 1) { S.sweep = null; S.x = Math.round(S.x * XQ) / XQ % 1; recomputePaths(); } busy = true;
      }
      S.paths.forEach(p => { if (p.t < p.pts.length - 1) { p.t = Math.min(p.pts.length - 1, p.t + dt * 4.5); busy = true; } });
      if (S.pinT < 1 && S.pinAnim) { S.pinT = Math.min(1, S.pinT + dt / 1.1); busy = true; }
      if (S.rotTo) { const r = S.rotTo; r.t += dt; const f = lib.ease(Math.min(1, r.t / r.dur)); S.view.yaw = r.y0 + (r.y1 - r.y0) * f; if (r.t >= r.dur) S.rotTo = null; busy = true; }
      render(); return busy;
    });
    const kick = () => loop.start();
    function startSweep(from, to, dur) { S.paths.forEach(p => { p.t = p.pts.length - 1; }); if (lib.reducedMotion) { S.x = ((to % 1) + 1) % 1; recomputePaths(); render(); return; } S.sweep = { from, to, dur, t: 0 }; kick(); }
    function refresh(paths) { if (paths) recomputePaths(); render(); }

    // ---------------------------------------------------------------- interaction
    cvT.canvas.addEventListener('pointermove', (ev) => {
      const [px, py] = cvT.toLocal(ev); if (ev.pointerType !== 'mouse') return;
      if (px >= tb.x && px <= tb.x + tb.w && py >= tb.y && py <= tb.y + tb.h) { S.probe = Pinv(px, py); if (!loop.running) render(); } else if (S.probe) { S.probe = null; render(); }
    });
    cvT.canvas.addEventListener('pointerleave', () => { if (S.probe) { S.probe = null; if (!loop.running) render(); } });
    cvT.canvas.addEventListener('pointerdown', (ev) => {
      const [px, py] = cvT.toLocal(ev); if (px < tb.x || px > tb.x + tb.w || py < tb.y || py > tb.y + tb.h) return;
      S.cands = false; S.sweep = null; S.coarse = false; const p = descend(Pinv(px, py)); if (lib.reducedMotion) p.t = p.pts.length - 1;
      S.paths.push(p); if (S.paths.length > 10) S.paths.shift(); kick(); render();
    });
    let rdrag = null;
    cv3.canvas.addEventListener('pointerdown', (ev) => { rdrag = { x: ev.clientX, y: ev.clientY, yaw: S.view.yaw, pitch: S.view.pitch }; S.rot = true; S.rotTo = null; cv3.canvas.style.cursor = 'grabbing'; try { cv3.canvas.setPointerCapture(ev.pointerId); } catch (_) { } });
    cv3.canvas.addEventListener('pointermove', (ev) => { if (!rdrag) return; S.view.yaw = rdrag.yaw + (ev.clientX - rdrag.x) * 0.012; S.view.pitch = clamp(rdrag.pitch - (ev.clientY - rdrag.y) * 0.008, 0.2, 1.45); if (!loop.running) render(); });
    const rup = () => { if (!rdrag) return; rdrag = null; S.rot = false; cv3.canvas.style.cursor = 'grab'; render(); };
    cv3.canvas.addEventListener('pointerup', rup); cv3.canvas.addEventListener('pointercancel', rup);
    cv3.canvas.addEventListener('keydown', (ev) => { const d = { ArrowLeft: [-0.12, 0], ArrowRight: [0.12, 0], ArrowUp: [0, -0.08], ArrowDown: [0, 0.08] }[ev.key]; if (!d) return; ev.preventDefault(); S.view.yaw += d[0]; S.view.pitch = clamp(S.view.pitch + d[1], 0.2, 1.45); render(); });
    let sdrag = false;
    const setXFrom = (ev) => { const [px] = cvS.toLocal(ev); S.x = clamp((px - 34) / (SW - 46), 0, 0.999); S.x = Math.round(S.x * XQ) / XQ % 1; };
    cvS.canvas.addEventListener('pointerdown', (ev) => { sdrag = true; S.sweep = null; S.coarse = true; try { cvS.canvas.setPointerCapture(ev.pointerId); } catch (_) { } setXFrom(ev); render(); });
    cvS.canvas.addEventListener('pointermove', (ev) => { if (!sdrag) return; setXFrom(ev); render(); });
    const sup = () => { if (!sdrag) return; sdrag = false; S.coarse = false; recomputePaths(); render(); };
    cvS.canvas.addEventListener('pointerup', sup); cvS.canvas.addEventListener('pointercancel', sup);

    ctx.setCaption('Top view: energy for the current context (darker blue = lower; contours every equal step in color). Dashed figure-eight: the targets of all contexts. Black dots, crosshair and dashed ring: this context\'s targets, their mean μ(x) and 2σ. Blue ×: the model\'s minimum ŷ*. Blue paths: gradient descent from each start.');
    S.pinT = 0; render();

    function setState(o) {
      S.sweep = null; S.coarse = false; S.rotTo = null;
      if (o.ck != null) { S.ck = o.ck; ckSeg.set(o.ck); }
      if (o.alpha != null) { S.alpha = o.alpha; aSeg.set(o.alpha); }
      if (o.hatch != null) { S.hatch = o.hatch; hatchBtn.setAttribute('aria-pressed', String(o.hatch)); }
      if (o.x != null) S.x = o.x;
    }
    return {
      step(i) {
        if (i === 0) { setState({ ck: LAST, alpha: 0.5, hatch: false, x: 0.25 }); S.paths = []; S.cands = true; S.pinT = 0; S.pinAnim = false; S.view = { yaw: -0.62, pitch: 0.9 }; render(); }
        if (i === 1) { setState({ ck: LAST, hatch: false }); S.paths = []; S.cands = true; S.pinT = lib.reducedMotion ? 1 : 0; S.pinAnim = true; S.rotTo = lib.reducedMotion ? null : { y0: S.view.yaw, y1: S.view.yaw + 0.7, t: 0, dur: 1.8 }; kick(); render(); }
        if (i === 2) { setState({ ck: LAST, hatch: false }); S.cands = false; S.paths = []; startSweep(S.x, S.x + 1.5, 8); }
        if (i === 3) { setState({ ck: LAST, alpha: 0.5, hatch: true }); if (Math.abs(S.x - 0.25) > 0.2 && Math.abs(S.x - 0.75) > 0.2) S.x = 0.75; dropStarts(); }
        if (i === 4) { setState({ ck: ck50, alpha: 0.5, hatch: true }); dropStarts(); }
        if (i === 5) { setState({ ck: LAST, alpha: 1, hatch: false }); dropStarts(); }
      },
      show() { render(); },
      hide() { loop.stop(); if (S.sweep) { S.x = ((S.sweep.to % 1) + 1) % 1; S.sweep = null; S.coarse = false; recomputePaths(); } S.paths.forEach(p => { p.t = p.pts.length - 1; }); S.pinT = 1; S.rotTo = null; },
    };
  },
});
