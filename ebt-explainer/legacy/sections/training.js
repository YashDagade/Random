/* Chapter: Training lab.
   How an EBT is trained (Algorithm 1, p.7): start from noise, unroll N gradient steps on the energy, score only the
   final prediction with an ordinary loss, and backpropagate through every step (second-order, Hessian-vector products).
   Then the landscape regularizers of Sec 3.3 (p.7) and the Table 2 ablation (p.10).
   Parts: (1) Algorithm 1 stepper on the real toy 2D EBT with exact weight gradients through the unroll,
          (2) a 3-parameter 1D energy you train by hand (exact gradients),
          (3) checkpoint scrubber over the real toy training run,
          (4) regularizer sampler + Table 2. */
(function () {
  'use strict';
  const SUBS = '₀₁₂₃₄₅₆₇₈₉';
  const sub = (i) => String(i).split('').map((d) => SUBS[+d] || d).join('');
  const sigm = (z) => 1 / (1 + Math.exp(-z));
  const LAM = 0.02;

  // ---------------------------------------------------------------------------------------------
  // Small MLP engine with weight gradients. Same math as EBT.toy2d (checked against it at mount):
  // E(x, ŷ) = MLP([phi(x), ŷ]) + λ‖ŷ‖², SiLU, 8→64→64→64→1. Weights flat (row-major W[out][in]).
  // ---------------------------------------------------------------------------------------------
  function makeNet(layers) { return layers.map((l) => ({ rows: l.W.length, cols: l.W[0].length, W: Float64Array.from(l.W.flat()), b: Float64Array.from(l.b) })); }
  const cloneNet = (net) => net.map((l) => ({ rows: l.rows, cols: l.cols, W: l.W.slice(), b: l.b.slice() }));
  const zerosLike = (net) => net.map((l) => ({ rows: l.rows, cols: l.cols, W: new Float64Array(l.W.length), b: new Float64Array(l.b.length) }));
  const nParams = (net) => net.reduce((s, l) => s + l.W.length + l.b.length, 0);
  function axpy(dst, src, s) {
    for (let l = 0; l < dst.length; l++) {
      const a = dst[l].W, b = src[l].W; for (let i = 0; i < a.length; i++) a[i] += s * b[i];
      const c = dst[l].b, d = src[l].b; for (let i = 0; i < c.length; i++) c[i] += s * d[i];
    }
    return dst;
  }
  function dot(A, B) {
    let s = 0;
    for (let l = 0; l < A.length; l++) {
      const a = A[l].W, b = B[l].W; for (let i = 0; i < a.length; i++) s += a[i] * b[i];
      const c = A[l].b, d = B[l].b; for (let i = 0; i < c.length; i++) s += c[i] * d[i];
    }
    return s;
  }
  const norm = (A) => Math.sqrt(dot(A, A));

  // forward + backward. Returns E, gy = ∇_ŷ E, and (optionally) TG = ∇_θ E with the same layout as net.
  function fb(net, phi, y, wantTheta) {
    const nl = net.length; const h0 = new Float64Array(phi.length + 2);
    for (let j = 0; j < phi.length; j++) h0[j] = phi[j];
    h0[phi.length] = y[0]; h0[phi.length + 1] = y[1];
    const hs = [h0], zs = []; let h = h0, out = 0;
    for (let l = 0; l < nl; l++) {
      const L = net[l], o = new Float64Array(L.rows);
      for (let i = 0; i < L.rows; i++) { let s = L.b[i]; const off = i * L.cols; for (let j = 0; j < L.cols; j++) s += L.W[off + j] * h[j]; o[i] = s; }
      if (l < nl - 1) { zs.push(o); const hn = new Float64Array(L.rows); for (let i = 0; i < L.rows; i++) hn[i] = o[i] * sigm(o[i]); hs.push(hn); h = hn; } else out = o[0];
    }
    let g = net[nl - 1].W.slice(); // d out / d h_last (last layer has a single row)
    const TG = wantTheta ? new Array(nl) : null;
    if (wantTheta) TG[nl - 1] = { W: hs[nl - 1].slice(), b: Float64Array.of(1) };
    for (let l = nl - 2; l >= 0; l--) {
      const z = zs[l], L = net[l], gz = new Float64Array(L.rows);
      for (let i = 0; i < L.rows; i++) { const s = sigm(z[i]); gz[i] = g[i] * s * (1 + z[i] * (1 - s)); }
      const hin = hs[l], gin = new Float64Array(L.cols);
      if (wantTheta) { const W = new Float64Array(L.W.length); for (let i = 0; i < L.rows; i++) { const gi = gz[i], off = i * L.cols; for (let j = 0; j < L.cols; j++) W[off + j] = gi * hin[j]; } TG[l] = { W, b: gz }; }
      for (let i = 0; i < L.rows; i++) { const gi = gz[i]; if (gi === 0) continue; const off = i * L.cols; for (let j = 0; j < L.cols; j++) gin[j] += L.W[off + j] * gi; }
      g = gin;
    }
    const n = h0.length;
    return { E: out + LAM * (y[0] * y[0] + y[1] * y[1]), gy: [g[n - 2] + 2 * LAM * y[0], g[n - 1] + 2 * LAM * y[1]], TG };
  }
  // energy on an n×n grid, rows TOP-DOWN (row 0 = ymax), fast path (feature part of layer 0 precomputed)
  function netGrid(net, phi, n, ext) {
    const L0 = net[0], nf = phi.length, base = new Float64Array(L0.rows), w6 = new Float64Array(L0.rows), w7 = new Float64Array(L0.rows);
    for (let i = 0; i < L0.rows; i++) { let s = L0.b[i]; for (let j = 0; j < nf; j++) s += L0.W[i * L0.cols + j] * phi[j]; base[i] = s; w6[i] = L0.W[i * L0.cols + nf]; w7[i] = L0.W[i * L0.cols + nf + 1]; }
    const bufs = net.map((l) => new Float64Array(l.rows));
    const E = []; let lo = Infinity, hi = -Infinity;
    for (let r = 0; r < n; r++) {
      const yv = ext[3] - (ext[3] - ext[2]) * r / (n - 1); const row = new Array(n);
      for (let c = 0; c < n; c++) {
        const xv = ext[0] + (ext[1] - ext[0]) * c / (n - 1);
        let h = bufs[0]; for (let i = 0; i < L0.rows; i++) { const z = base[i] + w6[i] * xv + w7[i] * yv; h[i] = z * sigm(z); }
        let out = 0;
        for (let l = 1; l < net.length; l++) {
          const L = net[l], o = bufs[l], last = l === net.length - 1;
          for (let i = 0; i < L.rows; i++) { let s = L.b[i]; const off = i * L.cols; for (let j = 0; j < L.cols; j++) s += L.W[off + j] * h[j]; o[i] = last ? s : s * sigm(s); }
          if (last) out = o[0]; h = o;
        }
        const e = out + LAM * (xv * xv + yv * yv); row[c] = e; if (e < lo) lo = e; if (e > hi) hi = e;
      }
      E.push(row);
    }
    return { E, lo, hi };
  }
  function hess(net, phi, y) {
    const e = 1e-4;
    const col = (k) => { const p = y.slice(), m = y.slice(); p[k] += e; m[k] -= e; const gp = fb(net, phi, p).gy, gm = fb(net, phi, m).gy; return [(gp[0] - gm[0]) / (2 * e), (gp[1] - gm[1]) / (2 * e)]; };
    const c0 = col(0), c1 = col(1), off = (c0[1] + c1[0]) / 2;
    return [[c0[0], off], [off, c1[1]]];
  }
  const eig2 = (H) => { const a = H[0][0], b = H[0][1], d = H[1][1], t = (a + d) / 2, s = Math.sqrt(((a - d) / 2) ** 2 + b * b); return [t - s, t + s]; };
  function unrollFwd(net, phi, y0, alpha, N) {
    const path = [y0.slice()], E = []; let y = y0.slice();
    for (let i = 0; i < N; i++) { const r = fb(net, phi, y); E.push(r.E); y = [y[0] - alpha * r.gy[0], y[1] - alpha * r.gy[1]]; path.push(y.slice()); }
    E.push(fb(net, phi, y).E);
    return { path, E };
  }
  const sqd = (p, t) => (p[0] - t[0]) ** 2 + (p[1] - t[1]) ** 2;
  function lossOf(path, tgt, detach) { if (!detach) return sqd(path[path.length - 1], tgt); let s = 0; for (let i = 1; i < path.length; i++) s += sqd(path[i], tgt); return s; }
  // Algorithm 1 forward + exact backward through the unroll.
  // full (S2-style): L = ‖ŷN − y‖², adjoint a_i = ∂L/∂ŷ_i flows back: a_{i-1} = (I − αH(ŷ_{i-1})) a_i
  // detached (S1-style): L = Σ_i ‖ŷ_i − y‖², ŷ_{i-1} treated as a constant input of step i (no Hessian term)
  // step i's piece of ∂L/∂θ: c_i = −α ∇_θ[a_i · ∇_ŷE(θ, ŷ_{i-1})], a mixed second derivative, computed as the
  // central difference of two ordinary weight-gradients at ŷ_{i-1} ± ε·a_i/|a_i| (same two backprops also give H·a_i).
  function trainUnroll(net, phi, y0, tgt, alpha, N, detach) {
    const { path, E } = unrollFwd(net, phi, y0, alpha, N);
    const err = (p) => [2 * (p[0] - tgt[0]), 2 * (p[1] - tgt[1])];
    const total = zerosLike(net), cs = new Array(N + 1).fill(null), adj = new Array(N + 1).fill(null), H = new Array(N + 1).fill(null);
    let a = err(path[N]); adj[N] = a;
    for (let i = N; i >= 1; i--) {
      const ai = detach ? err(path[i]) : a; if (detach) adj[i] = ai;
      const yp = path[i - 1], na = Math.hypot(ai[0], ai[1]), c = zerosLike(net); let Hv = [0, 0];
      if (na > 1e-15) {
        const e = 1e-4, u = [ai[0] / na, ai[1] / na];
        const P = fb(net, phi, [yp[0] + e * u[0], yp[1] + e * u[1]], true), M = fb(net, phi, [yp[0] - e * u[0], yp[1] - e * u[1]], true);
        const s = alpha * na / (2 * e); axpy(c, P.TG, -s); axpy(c, M.TG, s);
        Hv = [(P.gy[0] - M.gy[0]) * na / (2 * e), (P.gy[1] - M.gy[1]) * na / (2 * e)];
      }
      cs[i] = c; axpy(total, c, 1); H[i - 1] = hess(net, phi, yp);
      if (!detach) { a = [ai[0] - alpha * Hv[0], ai[1] - alpha * Hv[1]]; adj[i - 1] = a; }
    }
    const tn = norm(total);
    return { path, E, L: lossOf(path, tgt, detach), total, tn, cn: cs.map((c) => (c ? norm(c) : 0)), share: cs.map((c) => (c && tn > 0 ? dot(c, total) / (tn * tn) : 0)), adj, H, N, alpha, detach, tgt, y0 };
  }

  // ---------------------------------------------------------------------------------------------
  // 1D trainable energy for part 2: E_θ(ŷ) = a(ŷ − m)² + w·sin(3ŷ), θ = (a, m, w). Exact gradients.
  // ---------------------------------------------------------------------------------------------
  const K1 = 3;
  const e1 = (th, y) => th.a * (y - th.m) ** 2 + th.w * Math.sin(K1 * y);
  const g1 = (th, y) => 2 * th.a * (y - th.m) + th.w * K1 * Math.cos(K1 * y);
  const h1 = (th, y) => 2 * th.a - th.w * K1 * K1 * Math.sin(K1 * y);
  function toyRun(th, batch, N, tgt) {
    let L = 0; const G = { a: 0, m: 0, w: 0 }, paths = [];
    batch.starts.forEach((y0, j) => {
      const al = batch.alphas[j], p = [y0]; let y = y0;
      for (let i = 0; i < N; i++) { y = Math.max(-40, Math.min(40, y - al * g1(th, y))); p.push(y); }
      paths.push(p); const yN = p[N]; L += (yN - tgt) ** 2;
      let a = 2 * (yN - tgt);
      for (let i = N - 1; i >= 0; i--) {
        const yi = p[i];
        G.a += -al * 2 * (yi - th.m) * a; G.m += -al * (-2 * th.a) * a; G.w += -al * K1 * Math.cos(K1 * yi) * a;
        a *= 1 - al * h1(th, yi);
      }
    });
    const n = batch.starts.length; L /= n; G.a /= n; G.m /= n; G.w /= n;
    return { paths, L, G };
  }

  EBT.section({
    id: 'training',
    nav: 'Training',
    kicker: 'Lab · Algorithm 1',
    title: 'Training lab: backpropagate through thinking',
    lede: 'An EBT is never told what its energy should be. Training runs the model\'s own gradient descent for a few steps from noise, scores only where it lands with an ordinary loss, and backpropagates through every step. Below you can step through that loop on the real toy model, train a 1D landscape by hand, watch the toy landscape form over training, and see what each landscape regularizer is for.',
    mount(el, lib) {
      const h = lib.h, C = lib.C, M = EBT.toy2d, D = M && M.ready ? M.data : null;
      const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
      const f2 = (v) => lib.fmt(v, 2), f3 = (v) => lib.fmt(v, 3);
      const sci = (v) => { if (!isFinite(v)) return '–'; const a = Math.abs(v); return a !== 0 && (a < 1e-3 || a >= 1e4) ? v.toExponential(1) : a < 0.01 ? v.toFixed(4) : a < 1 ? v.toFixed(3) : v.toFixed(2); };
      const vec = (p) => '(' + f2(p[0]) + ', ' + f2(p[1]) + ')';
      const reduce = lib.reducedMotion;
      if (!D) { el.appendChild(lib.callout('warn', 'Data missing', 'data/toy2d.json was not bundled. Run <code>python3 src/bundle_data.py</code>.')); return; }
      const STEPS = M.steps, EXT = M.extent;

      // ---------- shared helpers ----------
      function rcanvas(parent, o) {
        const box = h('div', { class: 'canvas-box training-cv' }); const cv = h('canvas', { role: 'img', 'aria-label': o.label });
        box.appendChild(cv); parent.appendChild(box);
        const ctx = cv.getContext('2d'); const R = { box, cv, ctx, w: 0, h: 0, dpr: 1 };
        R.fit = () => {
          const w = Math.max(220, Math.floor(box.clientWidth || (parent.clientWidth || 600)));
          const hh = Math.max(40, Math.round(o.height(w))); const dpr = Math.min(2, window.devicePixelRatio || 1);
          if (w !== R.w || hh !== R.h || dpr !== R.dpr) { cv.width = Math.round(w * dpr); cv.height = Math.round(hh * dpr); cv.style.height = hh + 'px'; R.w = w; R.h = hh; R.dpr = dpr; }
          ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        };
        R.render = () => { R.fit(); ctx.clearRect(0, 0, R.w, R.h); o.draw(ctx, R.w, R.h); };
        R.toLocal = (ev) => { const r = cv.getBoundingClientRect(); return [(ev.clientX - r.left) / r.width * R.w, (ev.clientY - r.top) / r.height * R.h]; };
        if (window.ResizeObserver) { let pw = 0; new ResizeObserver(() => { const w = Math.floor(box.clientWidth); if (w && w !== pw) { pw = w; R.render(); } }).observe(box); }
        return R;
      }
      const T = (g, s, x, y, o = {}) => lib.text(g, s, x, y, Object.assign({ size: 13, kind: 'body', color: C.muted }, o));
      const btn = (label, onclick, cls) => { const b = h('button', { type: 'button', class: 'btn' + (cls ? ' ' + cls : '') }, label); b.addEventListener('click', onclick); return b; };
      function playBtn(onToggle) { const b = h('button', { type: 'button', class: 'btn primary', 'aria-pressed': 'false' }, 'Play'); b.addEventListener('click', () => onToggle()); b.setPlaying = (p) => { b.textContent = p ? 'Pause' : 'Play'; b.setAttribute('aria-pressed', String(p)); }; return b; }
      function quantile(grid, q) { const a = []; grid.forEach((r) => r.forEach((v) => a.push(v))); a.sort((x, y) => x - y); return a[Math.min(a.length - 1, Math.floor(q * (a.length - 1)))]; }
      function gridLoHi(grid) { let lo = Infinity; grid.forEach((r) => r.forEach((v) => { if (v < lo) lo = v; })); return [lo, quantile(grid, 0.9)]; }
      function heatImage(grid, lo, hi, gamma) {
        const n = grid.length, m = grid[0].length, c = document.createElement('canvas'); c.width = m; c.height = n;
        const g = c.getContext('2d'), img = g.createImageData(m, n);
        for (let r = 0; r < n; r++) for (let k = 0; k < m; k++) { let v = clamp((grid[r][k] - lo) / (hi - lo || 1), 0, 1); if (gamma) v = Math.pow(v, gamma); const col = lib.cmap(v), i = 4 * (r * m + k); img.data[i] = col[0]; img.data[i + 1] = col[1]; img.data[i + 2] = col[2]; img.data[i + 3] = 255; }
        g.putImageData(img, 0, 0); return c;
      }
      // grid points sit exactly on box edges (matches EBT.toy2d.toPx and lib.contours)
      function drawHeat(g, im, box) {
        const cw = box.w / (im.width - 1), ch = box.h / (im.height - 1);
        g.save(); g.beginPath(); g.rect(box.x, box.y, box.w, box.h); g.clip(); g.imageSmoothingEnabled = true; g.imageSmoothingQuality = 'high';
        g.drawImage(im, box.x - cw / 2, box.y - ch / 2, box.w + cw, box.h + ch); g.restore();
      }
      const imgCache = new WeakMap();
      function drawLandscape(g, grid, box, o = {}) {
        let ent = imgCache.get(grid);
        if (!ent) { const [lo0, hi0] = gridLoHi(grid); ent = { lo: lo0, hi: hi0, img: o.img || heatImage(grid, lo0, hi0, 0.8) }; imgCache.set(grid, ent); }
        const lo = ent.lo, hi = ent.hi, cl = o.clip || box;
        g.save(); g.beginPath(); g.rect(cl.x, cl.y, cl.w, cl.h); g.clip();
        drawHeat(g, ent.img, box);
        const lv = []; for (let k = 1; k <= (o.levels || 9); k++) lv.push(lo + (hi - lo) * Math.pow(k / ((o.levels || 9) + 1), 1.25));
        lib.contours(g, grid, box.x, box.y, box.w, box.h, lv, { color: 'rgba(233,238,246,0.22)', width: 1 });
        g.restore();
        g.save(); g.strokeStyle = C.rule; g.lineWidth = 1; g.strokeRect(cl.x + 0.5, cl.y + 0.5, cl.w - 1, cl.h - 1); g.restore();
      }
      // a label on a dark plate (readable over the light parts of the colormap)
      function plate(g, s, x, y, o = {}) {
        const size = o.size || 12.5, kind = o.kind || 'body'; const m = lib.measure(g, s, { size, kind });
        const pw = m.w + 12, ph = size + 9, px = o.align === 'right' ? x - pw : x, py = y - ph;
        g.save(); g.fillStyle = lib.rgba(C.bg, 0.78); lib.rr(g, px, py, pw, ph, 5); g.fill(); g.restore();
        T(g, s, px + 6, py + ph / 2 + 0.5, { size, kind, color: o.color || C.ink, baseline: 'middle' });
      }
      // view window (in ŷ) -> virtual box for the full-extent grid so that the window fills [0, w]²
      function zoomBox(view, w) { const s = w / (view[1] - view[0]); return { x: -(view[0] - EXT[0]) * s, y: -(EXT[3] - view[3]) * s, w: (EXT[1] - EXT[0]) * s, h: (EXT[3] - EXT[2]) * s }; }
      function viewAround(pts, margin, minSize) {
        let xa = Infinity, xb = -Infinity, ya = Infinity, yb = -Infinity; pts.forEach((q) => { xa = Math.min(xa, q[0]); xb = Math.max(xb, q[0]); ya = Math.min(ya, q[1]); yb = Math.max(yb, q[1]); });
        const size = Math.min(EXT[1] - EXT[0], Math.max(minSize, xb - xa + 2 * margin, yb - ya + 2 * margin));
        const cx = clamp((xa + xb) / 2, EXT[0] + size / 2, EXT[1] - size / 2), cy = clamp((ya + yb) / 2, EXT[2] + size / 2, EXT[3] - size / 2);
        return [cx - size / 2, cx + size / 2, cy - size / 2, cy + size / 2];
      }
      function axisTicks(g, view, w) { // integer ŷ ticks along the bottom and left edges of a zoomed view
        const s = w / (view[1] - view[0]);
        for (let v = Math.ceil(view[0] - 1e-9); v <= view[1]; v += view[1] - view[0] > 2.6 ? 1 : 0.5) { const x = (v - view[0]) * s; if (x < 14 || x > w - 14) continue; g.fillStyle = lib.rgba(C.ink, 0.6); g.fillRect(x - 0.5, w - 5, 1, 5); }
        for (let v = Math.ceil(view[2] - 1e-9); v <= view[3]; v += view[3] - view[2] > 2.6 ? 1 : 0.5) { const y = (view[3] - v) * s; if (y < 14 || y > w - 14) continue; g.fillStyle = lib.rgba(C.ink, 0.6); g.fillRect(0, y - 0.5, 5, 1); }
      }
      const P = (y, box) => M.toPx(y, box);
      function marker(g, x, y, kind, col) {
        g.save(); g.strokeStyle = col; g.fillStyle = col; g.lineWidth = 2.2;
        if (kind === 'cross') { g.beginPath(); g.moveTo(x - 6, y - 6); g.lineTo(x + 6, y + 6); g.moveTo(x + 6, y - 6); g.lineTo(x - 6, y + 6); g.stroke(); }
        else if (kind === 'ring') { g.beginPath(); g.arc(x, y, 6, 0, 7); g.stroke(); }
        else if (kind === 'diamond') { g.beginPath(); g.moveTo(x, y - 7); g.lineTo(x + 7, y); g.lineTo(x, y + 7); g.lineTo(x - 7, y); g.closePath(); g.stroke(); }
        g.restore();
      }
      function partHeader(n, title, badges, intro) {
        return h('header', { class: 'training-head' },
          h('div', { class: 'training-num' }, 'Part ' + n),
          h('h3', {}, title),
          intro ? h('p', { class: 'training-intro', html: intro }) : null);
      }
      const ctxLabel = (i) => { const c = M.contexts[i]; return 'x = ' + c.x + ' · σ ' + c.sigma.toFixed(2); };
      const phiOf = (i) => M.features(M.contexts[i].x);

      // runtime cross-check of our weight-gradient engine against EBT.toy2d (energy and ∇ŷE)
      (function selfCheck() {
        try {
          const net = makeNet(D.weights[3].layers), phi = phiOf(0);
          let worst = 0; [[0.3, -0.7], [-1.4, 1.1]].forEach((y) => { const a = fb(net, phi, y), b = M.energyGrad(3, M.contexts[0].x, y); worst = Math.max(worst, Math.abs(a.E - b.E), Math.abs(a.gy[0] - b.g[0]), Math.abs(a.gy[1] - b.g[1])); });
          if (worst > 1e-9) console.warn('training: gradient engine differs from EBT.toy2d by', worst);
        } catch (e) { console.warn('training: self-check failed', e); }
      })();

      // ---------- intro: two nested loops ----------
      const loops = h('div', { class: 'training-loops' },
        h('div', { class: 'training-loop inner' },
          h('div', { class: 'training-loop-tag' }, 'Inner loop · thinking · changes ŷ'),
          h('div', { class: 'training-eq', html: 'ŷ<sub>i+1</sub> = ŷ<sub>i</sub> − α ∇<sub>ŷ</sub>E<sub>θ</sub>(x, ŷ<sub>i</sub>)' }),
          h('p', {}, 'Gradient descent on the prediction, starting from noise ŷ₀ ~ N(0, I). It descends the energy landscape. The paper\'s thinking models train with only 2 to 3 such steps (Table D.4, p.36).')),
        h('div', { class: 'training-loop outer' },
          h('div', { class: 'training-loop-tag' }, 'Outer loop · learning · changes θ'),
          h('div', { class: 'training-eq', html: 'θ ← θ − η ∇<sub>θ</sub> J(ŷ<sub>N</sub>, y)' }),
          h('p', {}, 'An ordinary loss on the final prediction (cross-entropy for text, MSE for images), backpropagated through the whole inner loop. That needs gradients of gradients (p.7). It descends the loss landscape over θ.')));
      el.appendChild(h('div', { class: 'stack' }, h('div', { class: 'training-badges' }, lib.badge('paper', 'Algorithm 1, Eq. 1, p.7')), loops));

      // =============================================================================================
      // PART 1: Algorithm 1 stepper on the real toy model
      // =============================================================================================
      (function part1() {
        const sec = h('section', { class: 'training-part', 'aria-labelledby': 'training-p1' });
        el.appendChild(sec);
        const hd = partHeader(1, 'Unroll, score, backpropagate', null, 'The model below is the real 2D toy EBT (its prediction ŷ is a point in the plane), loaded from a checkpoint of its training run. Press <b>Step</b> to run Algorithm 1 one operation at a time: N forward steps of gradient descent on the energy, one loss, then the backward pass through every step. Every number is computed live from the checkpoint\'s weights.');
        hd.querySelector('h3').id = 'training-p1'; sec.appendChild(hd);

        const ckOpts = [[0, '0'], [2, '25'], [3, '50'], [4, '100'], [6, '400'], [11, '12k']];
        const S = { ck: 3, ctx: 0, N: 4, alpha: 1, detach: false, seed: 5, eta: 0.03, edits: 0, stage: 0, prog: 1, playing: false, lastL: null, msg: '' };
        let net = null, phi = null, grid = null, img = null, R = null, prev = null, fdErr = null;
        const gridCache = new Map();

        function sample() { const r = lib.rng(1000 + S.seed * 7919); const y0 = [r.normal(), r.normal()]; const c = M.contexts[S.ctx]; const n = [r.normal(), r.normal()]; return { y0, tgt: [c.mu[0] + c.sigma * n[0], c.mu[1] + c.sigma * n[1]] }; }
        let smp = sample();
        function loadNet() { net = makeNet(D.weights[S.ck].layers); phi = phiOf(S.ctx); S.edits = 0; }
        function loadGrid() {
          const key = S.ck + ':' + S.ctx;
          if (S.edits === 0 && gridCache.has(key)) { ({ grid, img } = gridCache.get(key)); return; }
          let G = S.edits === 0 ? M.storedGrid(S.ctx, STEPS[S.ck]) : null;
          if (!G) G = netGrid(net, phi, 64, EXT).E;
          const [lo, hi] = gridLoHi(G); grid = G; img = heatImage(G, lo, hi, 0.8);
          if (S.edits === 0) gridCache.set(key, { grid, img });
        }
        function compute() {
          R = trainUnroll(net, phi, smp.y0, smp.tgt, S.alpha, S.N, S.detach);
          // what one SGD step on θ would do for this same example
          const n2 = axpy(cloneNet(net), R.total, -S.eta); const f = unrollFwd(n2, phi, smp.y0, S.alpha, S.N);
          prev = { net: n2, path: f.path, L: lossOf(f.path, smp.tgt, S.detach) };
          // finite-difference check of the whole ∂L/∂θ along a random direction
          const r = lib.rng(99), dir = zerosLike(net); dir.forEach((l) => { for (let i = 0; i < l.W.length; i++) l.W[i] = r.normal(); for (let i = 0; i < l.b.length; i++) l.b[i] = r.normal(); });
          const dn = norm(dir); dir.forEach((l) => { l.W.forEach((v, i) => { l.W[i] = v / dn; }); l.b.forEach((v, i) => { l.b[i] = v / dn; }); });
          const eps = 1e-5, Lp = lossOf(unrollFwd(axpy(cloneNet(net), dir, eps), phi, smp.y0, S.alpha, S.N).path, smp.tgt, S.detach), Lm = lossOf(unrollFwd(axpy(cloneNet(net), dir, -eps), phi, smp.y0, S.alpha, S.N).path, smp.tgt, S.detach);
          const fd = (Lp - Lm) / (2 * eps), an = dot(R.total, dir); fdErr = { fd, an, rel: Math.abs(fd - an) / Math.max(1e-12, Math.abs(fd) + Math.abs(an)) };
        }
        loadNet(); loadGrid(); compute();
        const nP = nParams(net);
        const SS = () => 2 * S.N + 2; // last stage index

        // ----- controls -----
        const ckSeg = lib.segmented({ label: 'Weights from training step', options: ckOpts, value: S.ck, onchange: (v) => { S.ck = v; loadNet(); loadGrid(); recompute(true); } });
        const ctxSeg = lib.segmented({ label: 'Context', options: [[0, 'x = 0.25 · low noise'], [1, 'x = 0.75 · high noise']], value: S.ctx, onchange: (v) => { S.ctx = v; smp = sample(); loadNet(); loadGrid(); recompute(true); } });
        const nSl = lib.slider({ id: 'training-p1-n', label: 'Unrolled steps N', min: 1, max: 6, step: 1, value: S.N, oninput: (v) => { S.N = v; recompute(true); } });
        const aSl = lib.slider({ id: 'training-p1-a', label: 'Step size α', min: 0.25, max: 2, step: 0.05, value: S.alpha, fmt: f2, oninput: (v) => { S.alpha = v; recompute(true); } });
        const modeSeg = lib.segmented({ label: 'Backprop mode', options: [[false, 'Full unroll'], [true, 'Detached steps']], value: S.detach, onchange: (v) => { S.detach = v; recompute(true); } });
        const etaSl = lib.slider({ id: 'training-p1-eta', label: 'Learning rate η for one update', min: -2.5, max: -0.5, step: 0.05, value: Math.log10(S.eta), fmt: (v) => Math.pow(10, v).toFixed(3), oninput: (v) => { S.eta = Math.pow(10, v); recompute(false); } });
        const stepB = btn('Step', () => { stop(); if (S.stage < SS()) { S.stage++; S.prog = 0; animOnce(); } });
        const play = playBtn(() => { if (S.playing) stop(); else { if (S.stage >= SS()) { S.stage = 0; S.prog = 1; } S.playing = true; play.setPlaying(true); loop.start(); } });
        const resetB = btn('Reset', () => { stop(); S.stage = 0; S.prog = 1; S.msg = ''; drawAll(); });
        const newB = btn('New ŷ₀ and y', () => { S.seed++; smp = sample(); recompute(true); });
        const applyB = btn('Apply update to θ', () => {
          const before = R.L; net = prev.net; S.edits++; loadGrid(); compute();
          S.msg = 'Update ' + S.edits + ' applied (θ ← θ − η ∂L/∂θ). Same ŷ₀ and y, new weights: L went from ' + f3(before) + ' to ' + f3(R.L) + '. ';
          S.stage = SS(); S.prog = 1; drawAll();
        }, 'primary');
        const restoreB = btn('Restore checkpoint', () => { loadNet(); loadGrid(); S.msg = ''; recompute(true); });
        function recompute(full) { compute(); if (full) { S.stage = SS(); S.prog = 1; } drawAll(); }

        const left = h('div', { class: 'card training-ctls' },
          h('div', { class: 'ctl' }, h('label', {}, h('span', {}, 'Weights from training step')), ckSeg.el),
          h('div', { class: 'ctl' }, h('label', {}, h('span', {}, 'Context x (one input of the toy)')), ctxSeg.el),
          nSl.el, aSl.el,
          h('div', { class: 'ctl' }, h('label', {}, h('span', {}, 'Backprop mode')), modeSeg.el),
          h('div', { class: 'training-btns' }, stepB, play, resetB),
          h('div', { class: 'training-btns' }, newB),
          h('hr', { class: 'training-hr' }),
          etaSl.el,
          h('div', { class: 'training-btns' }, applyB, restoreB),
          h('p', { class: 'training-note' }, 'One example only. Real training averages ∂L/∂θ over a batch (256 in our toy) and uses Adam.'));

        // ----- instrument -----
        const right = h('div', { class: 'card stack training-inst' });
        right.appendChild(h('div', { class: 'training-badges' }, lib.badge('toy', 'live, from checkpoint weights'), h('span', { class: 'training-small' }, 'Algorithm 1 · p.7')));
        const graphC = rcanvas(right, { label: 'Unrolled computation graph of Algorithm 1: noise ŷ0, N gradient steps that all use the shared weights θ, the loss, and the backward pass through every step.', height: (w) => graphLayout(w, S.N).h, draw: drawGraph });
        const duo = h('div', { class: 'training-duo' }); right.appendChild(duo);
        const landBox = h('div', { class: 'stack' }); duo.appendChild(landBox);
        const landC = rcanvas(landBox, { label: 'Energy landscape of the toy model for the chosen context, with the unrolled path, the target y, and backward-pass arrows.', height: (w) => w, draw: drawLand });
        landBox.appendChild(h('div', { class: 'legend' }, h('span', {}, h('i', { style: 'background:var(--ebt)' }), 'path ŷ₀ … ŷN'), h('span', {}, h('i', { style: 'background:var(--truth)' }), 'target y'), h('span', {}, h('i', { style: 'background:var(--bad)' }), '−∂L/∂ŷᵢ'), h('span', {}, h('i', { style: 'background:var(--ink)' }), 'after update')));
        const contribC = rcanvas(duo, { label: 'Bar charts: how much loss gradient reaches each ŷ_i, and how much each step contributes to the gradient for θ.', height: (w) => 2 * 118 + 30, draw: drawContrib });
        const status = h('div', { class: 'training-status', 'aria-live': 'polite' }); right.appendChild(status);
        const readout = h('div', { class: 'readout' }); right.appendChild(readout);
        sec.appendChild(h('div', { class: 'bench' }, left, right));

        // asides: HVP + what to notice
        sec.appendChild(h('div', { class: 'grid2' },
          lib.callout('', 'Hessian-vector products, simply', 'Backprop through one step multiplies the incoming error vector <b>a</b> by <b>I − αH</b>, where H = ∂²E/∂ŷ² is the Hessian. You never build H. <b>Ha</b> is just "how much the gradient changes if you nudge ŷ along a": Ha ≈ [∇E(ŷ + εa) − ∇E(ŷ − εa)] / 2ε. Autodiff gets it exactly for about the price of one more backward pass, so the cost grows linearly with model size (p.7, p.41). Here ŷ has 2 numbers, so H is 2×2. In the paper\'s text model ŷ is a 50,277-dim logit vector per token, so H would have about 2.5 billion entries per position [derived]. This lab computes Ha and the θ-term with the two-sided difference above (ε = 10⁻⁴), and checks the full ∂L/∂θ against finite differences of the loss.'),
          lib.callout('insight', 'What to notice', 'Early weights (step <b>25</b> or <b>50</b>): the landscape around the path is nearly flat, so I − αH ≈ I passes the error back almost unchanged and every step gets a similar share of ∂L/∂θ. Trained weights (<b>400</b> or <b>12k</b>): the bowl\'s curvature is close to 1/α, so I − αH nearly erases the error one step back, and the last step carries most of the learning signal. <b>Detached steps</b> is the paper\'s S1 recipe (loss at every step, no gradient between steps). Even then each step needs the mixed derivative ∂(∇ŷE)/∂θ, a gradient of a gradient (p.30, p.43). Cost: each step is about F + B + B, roughly 1.66× a feed-forward step (p.35, p.41).')));

        // ----- layout of the unrolled graph -----
        function graphLayout(w, N) {
          const vertical = w < 560;
          if (!vertical) {
            const cy = 128, x0 = 60, xN = w - 178, dx = (xN - x0) / N;
            const nodes = []; for (let i = 0; i <= N; i++) nodes.push({ x: x0 + i * dx, y: cy });
            const ops = [null]; for (let i = 1; i <= N; i++) ops.push({ x: (nodes[i - 1].x + nodes[i].x) / 2, y: cy });
            const opW = Math.max(40, Math.min(62, dx - 46));
            return { vertical, h: 236, nodes, ops, opW, opH: 28, J: { x: w - 104, y: cy }, Y: { x: w - 104, y: cy + 78 }, Lv: { x: w - 44, y: cy }, theta: { x: 12, y: 12, w: w - 24, h: 46 },
              opPath: (i) => [[ops[i].x, 58], [ops[i].x, cy - 14]], eLab: (i) => [nodes[i].x, cy - 44, 'center'], aLab: (i) => [nodes[i].x, cy + 52, 'center'] };
          }
          const top = 70, step = 66, cx = 34;
          const nodes = []; for (let i = 0; i <= N; i++) nodes.push({ x: cx, y: top + i * step });
          const ops = [null]; for (let i = 1; i <= N; i++) ops.push({ x: cx, y: (nodes[i - 1].y + nodes[i].y) / 2 });
          const jy = nodes[N].y + 64;
          return { vertical, h: jy + 98, nodes, ops, opW: 56, opH: 24, J: { x: cx + 6, y: jy }, Y: { x: cx + 110, y: jy }, Lv: { x: cx + 6, y: jy + 44 }, theta: { x: 12, y: 10, w: w - 24, h: 40 },
            opPath: (i) => [[w - 26, 50], [w - 26, ops[i].y], [cx + 28, ops[i].y]], eLab: (i) => [cx + 26, nodes[i].y - 9, 'left'], aLab: (i) => [cx + 26, nodes[i].y + 9, 'left'] };
        }
        function along(pts, t) { let tot = 0; const seg = []; for (let q = 1; q < pts.length; q++) { const d = Math.hypot(pts[q][0] - pts[q - 1][0], pts[q][1] - pts[q - 1][1]); seg.push(d); tot += d; } let r = clamp(t, 0, 1) * tot; for (let q = 1; q < pts.length; q++) { if (r <= seg[q - 1] || q === pts.length - 1) { const f = seg[q - 1] ? Math.min(1, r / seg[q - 1]) : 1; return [lib.lerp(pts[q - 1][0], pts[q][0], f), lib.lerp(pts[q - 1][1], pts[q][1], f)]; } r -= seg[q - 1]; } return pts[pts.length - 1]; }
        const stepCol = (i, N) => lib.mix(C.faint, C.ebt, N <= 1 ? 1 : 0.25 + 0.75 * (i - 1) / (N - 1));

        function drawGraph(g, w) {
          const N = S.N, L = graphLayout(w, N), st = S.stage, p = lib.ease(S.prog), k = st - (N + 1);
          const th = L.theta;
          // θ bar
          lib.panel(g, th.x, th.y, th.w, th.h, { fill: C.panel2, stroke: C.rule, r: 10, lw: 1 });
          const tx = th.x + 14, tyy = th.y + th.h / 2;
          T(g, 'θ', tx, tyy, { size: 20, kind: 'display', weight: 600, color: C.ink, baseline: 'middle' });
          T(g, L.vertical || w < 780 ? 'shared weights, used by every step' : 'shared weights · ' + nP.toLocaleString() + ' numbers · used by every step', tx + 22, tyy, { baseline: 'middle', size: 13 });
          // accumulator (stacked by deposit order)
          const dep = []; for (let i = N; i >= 1; i--) if (k >= N - i + 1) dep.push(i);
          if (!L.vertical && dep.length) {
            const ax = th.x + th.w - 250, aw = 236, sumN = R.cn.reduce((a, b) => a + b, 0) || 1; let cx = ax;
            T(g, '∂L/∂θ so far', ax, th.y + 7, { size: 12, color: C.bad });
            dep.forEach((i) => { const ww = aw * R.cn[i] / sumN; g.fillStyle = stepCol(i, N); g.fillRect(cx, th.y + 26, Math.max(1, ww - 1), 10); cx += ww; });
          }
          // edges θ → ops
          for (let i = 1; i <= N; i++) {
            const used = st >= i, deposited = k >= N - i + 1, active = k === N - i + 1 && S.prog < 1;
            const pp = L.opPath(i);
            g.save(); g.setLineDash([3, 4]); g.lineWidth = deposited ? 1.8 : 1.2;
            g.strokeStyle = deposited ? lib.rgba(C.bad, 0.7) : used ? lib.rgba(C.ebt, 0.45) : lib.rgba(C.faint, 0.5);
            g.beginPath(); g.moveTo(pp[0][0], pp[0][1]); for (let q = 1; q < pp.length; q++) g.lineTo(pp[q][0], pp[q][1]); g.stroke(); g.restore();
            if (st === i && S.prog < 1) { const d = along(pp, p); lib.dot(g, d[0], d[1], 4, C.ebt); }
            if (active) { const d = along(pp, 1 - p); lib.dot(g, d[0], d[1], 5, C.bad); }
          }
          // forward chain
          const r0 = 17;
          for (let i = 1; i <= N; i++) {
            const a = L.nodes[i - 1], b = L.nodes[i], o = L.ops[i];
            const vis = st >= i, prog = st === i ? p : 1;
            const ow = L.opW, oh = L.opH;
            if (vis) {
              if (!L.vertical) { lib.arrow(g, a.x + r0 + 2, a.y, o.x - ow / 2 - 3, o.y, { color: C.ebt, width: 2, head: 7, progress: clamp(prog * 2, 0, 1) }); if (prog > 0.5) lib.arrow(g, o.x + ow / 2 + 3, o.y, b.x - r0 - 3, b.y, { color: C.ebt, width: 2, head: 7, progress: (prog - 0.5) * 2 }); }
              else { lib.arrow(g, a.x, a.y + r0 + 2, o.x, o.y - oh / 2 - 3, { color: C.ebt, width: 2, head: 7, progress: clamp(prog * 2, 0, 1) }); if (prog > 0.5) lib.arrow(g, o.x, o.y + oh / 2 + 3, b.x, b.y - r0 - 3, { color: C.ebt, width: 2, head: 7, progress: (prog - 0.5) * 2 }); }
            }
            const active = (st === i && S.prog < 1) || (k === N - i + 1);
            lib.panel(g, o.x - ow / 2, o.y - oh / 2, ow, oh, { fill: active ? lib.rgba(k === N - i + 1 ? C.bad : C.ebt, 0.22) : C.panel2, stroke: vis ? (k >= N - i + 1 ? C.bad : C.ebt) : C.rule, r: 7, lw: 1.4 });
            T(g, '−α∇E', o.x, o.y, { size: 12.5, kind: 'mono', color: vis ? C.ink : C.faint, align: 'center', baseline: 'middle' });
            if (S.detach && i < N && vis) { // stop-gradient marks on the edge leaving ŷ_i
              const m = L.vertical ? [b.x, b.y + r0 + 10] : [b.x + r0 + 10, b.y];
              g.save(); g.strokeStyle = C.muted; g.lineWidth = 1.6; g.beginPath();
              if (L.vertical) { g.moveTo(m[0] - 7, m[1] - 3); g.lineTo(m[0] + 7, m[1] - 3); g.moveTo(m[0] - 7, m[1] + 2); g.lineTo(m[0] + 7, m[1] + 2); }
              else { g.moveTo(m[0] - 3, m[1] - 7); g.lineTo(m[0] - 3, m[1] + 7); g.moveTo(m[0] + 2, m[1] - 7); g.lineTo(m[0] + 2, m[1] + 7); }
              g.stroke(); g.restore();
            }
          }
          // backward arcs (full mode)
          if (!S.detach && k >= 1) {
            for (let i = N; i >= 1; i--) {
              const kk = N - i + 1; if (k < kk) continue; const pr = k === kk ? p : 1;
              const a = L.nodes[i], b = L.nodes[i - 1];
              g.save(); g.strokeStyle = lib.rgba(C.bad, 0.85); g.fillStyle = C.bad; g.lineWidth = 2; g.beginPath();
              let ex, ey;
              if (!L.vertical) { const yb = a.y + 30; const mx = lib.lerp(a.x, b.x, pr); g.moveTo(a.x, a.y + r0 + 3); g.quadraticCurveTo(a.x, yb, lib.lerp(a.x, (a.x + b.x) / 2, Math.min(1, pr * 2)), yb); if (pr > 0.5) g.quadraticCurveTo(b.x, yb, b.x, lib.lerp(yb, b.y + r0 + 3, (pr - 0.5) * 2)); ex = pr > 0.5 ? b.x : mx; ey = pr > 0.5 ? lib.lerp(yb, b.y + r0 + 3, (pr - 0.5) * 2) : yb; }
              else { const xb = a.x - 26; g.moveTo(a.x - r0 - 3, a.y); g.quadraticCurveTo(xb, a.y, xb, lib.lerp(a.y, (a.y + b.y) / 2, Math.min(1, pr * 2))); if (pr > 0.5) g.quadraticCurveTo(xb, b.y, lib.lerp(xb, b.x - r0 - 3, (pr - 0.5) * 2), b.y); ex = 0; ey = 0; }
              g.stroke(); g.restore();
              if (pr >= 1) { if (!L.vertical) lib.arrow(g, b.x + 0.01, b.y + r0 + 12, b.x, b.y + r0 + 4, { color: C.bad, width: 2, head: 7 }); else lib.arrow(g, b.x - r0 - 12, b.y + 0.01, b.x - r0 - 4, b.y, { color: C.bad, width: 2, head: 7 }); }
            }
          }
          // nodes
          for (let i = 0; i <= N; i++) {
            const n = L.nodes[i], vis = i <= st, isNew = st === i && S.prog < 1;
            g.save(); g.globalAlpha = vis ? (isNew ? 0.35 + 0.65 * p : 1) : 0.35;
            lib.dot(g, n.x, n.y, r0, i === 0 ? C.panel2 : lib.rgba(C.ebt, 0.18), { stroke: vis ? (i === 0 ? C.ink : C.ebt) : C.faint, lw: 2 });
            T(g, 'ŷ' + sub(i), n.x, n.y + 1, { size: 15, kind: 'mono', color: vis ? C.ink : C.faint, align: 'center', baseline: 'middle' });
            g.restore();
            if (vis && (!isNew || p > 0.6)) { const [lx, ly, al] = L.eLab(i); T(g, 'E ' + f2(R.E[i]), lx, ly, { size: 12.5, kind: 'mono', color: C.muted, align: al, baseline: 'middle' }); }
            const known = S.detach ? (i >= 1 && k >= N - i + 1) || (i === N && k >= 0) : k >= N - i;
            if (known && R.adj[i]) { const [lx, ly, al] = L.aLab(i); T(g, (L.vertical ? '|∂L/∂ŷ| ' : '|∂L/∂ŷ| ') + sci(Math.hypot(R.adj[i][0], R.adj[i][1])), lx, ly, { size: 12.5, kind: 'mono', color: C.bad, align: al, baseline: 'middle' }); }
          }
          // loss
          const lossVis = st >= N + 1, lp = st === N + 1 ? p : 1, J = L.J, Y = L.Y, last = L.nodes[N];
          g.save(); g.globalAlpha = lossVis ? 1 : 0.35;
          if (lossVis) {
            if (!L.vertical) { lib.arrow(g, last.x + r0 + 2, last.y, J.x - 30, J.y, { color: C.ink, width: 2, head: 7, progress: lp }); lib.arrow(g, Y.x, Y.y - 13, Y.x, J.y + 17, { color: C.truth, width: 2, head: 7, progress: lp }); }
            else { lib.arrow(g, last.x, last.y + r0 + 2, J.x - 6, J.y - 17, { color: C.ink, width: 2, head: 7, progress: lp }); lib.arrow(g, Y.x - 16, Y.y, J.x + 36, J.y, { color: C.truth, width: 2, head: 7, progress: lp }); }
          }
          lib.panel(g, J.x - 28, J.y - 15, 56, 30, { fill: lossVis ? lib.rgba(C.bad, 0.15) : C.panel2, stroke: lossVis ? C.bad : C.rule, r: 7, lw: 1.4 });
          T(g, S.detach ? 'ΣJ' : 'J', J.x, J.y + 1, { size: 15, kind: 'mono', color: C.ink, align: 'center', baseline: 'middle' });
          lib.dot(g, Y.x, Y.y, 12, lib.rgba(C.truth, 0.18), { stroke: C.truth, lw: 2 });
          T(g, 'y', Y.x, Y.y, { size: 14, kind: 'mono', color: C.truth, align: 'center', baseline: 'middle' });
          if (!L.vertical) T(g, 'target', Y.x + 18, Y.y, { size: 12.5, color: C.truth, baseline: 'middle' });
          if (lossVis) {
            if (!L.vertical) { T(g, 'L', L.Lv.x, J.y - 12, { size: 13, kind: 'mono', color: C.muted, align: 'center', baseline: 'middle' }); T(g, f3(R.L), L.Lv.x, J.y + 8, { size: 14, kind: 'mono', color: C.ink, align: 'center', baseline: 'middle' }); lib.arrow(g, J.x + 30, J.y, L.Lv.x - 24, J.y, { color: C.muted, width: 1.5, head: 6, progress: lp }); }
            else T(g, 'L = ' + f3(R.L), J.x + 40, L.Lv.y, { size: 14, kind: 'mono', color: C.ink, baseline: 'middle' });
          }
          g.restore();
          if (L.vertical) { // θ accumulator text line for the narrow layout
            if (dep.length) T(g, '∂L/∂θ so far: ' + dep.length + ' of ' + N + ' pieces', w - 14, L.h - 14, { size: 12.5, color: C.bad, align: 'right', baseline: 'middle' });
          }
        }

        // colour range taken from the grid cells inside the current view, so a zoomed basin keeps its contrast
        let vimg = { key: '', img: null };
        function viewImage(view) {
          const key = view.map((v) => v.toFixed(3)).join(',') + '|' + S.ck + '|' + S.ctx + '|' + S.edits; if (vimg.key === key && vimg.grid === grid) return vimg.img;
          const n = grid.length, vals = []; let lo = Infinity;
          for (let r = 0; r < n; r++) { const yv = EXT[3] - (EXT[3] - EXT[2]) * r / (n - 1); if (yv < view[2] - 0.1 || yv > view[3] + 0.1) continue; for (let c = 0; c < n; c++) { const xv = EXT[0] + (EXT[1] - EXT[0]) * c / (n - 1); if (xv < view[0] - 0.1 || xv > view[1] + 0.1) continue; vals.push(grid[r][c]); if (grid[r][c] < lo) lo = grid[r][c]; } }
          vals.sort((a, b) => a - b); const hi = vals.length ? vals[Math.floor(0.95 * (vals.length - 1))] : lo + 1;
          vimg = { key, grid, img: heatImage(grid, lo, hi, 0.8) }; return vimg.img;
        }
        function drawLand(g, w) {
          const N = S.N, st = S.stage, p = lib.ease(S.prog), k = st - (N + 1);
          const view = viewAround(R.path.concat(prev ? prev.path : [], [smp.tgt, M.contexts[S.ctx].mu]), 0.45, 2.2), box = zoomBox(view, w);
          drawLandscape(g, grid, box, { img: viewImage(view), clip: { x: 0, y: 0, w, h: w } }); axisTicks(g, view, w);
          g.save(); g.beginPath(); g.rect(0, 0, w, w); g.clip();
          const pts = R.path.map((q) => P(q, box));
          // μ and target
          const c = M.contexts[S.ctx], mu = P(c.mu, box), tg = P(smp.tgt, box);
          g.save(); g.strokeStyle = lib.rgba(C.truth, 0.5); g.setLineDash([3, 3]); g.lineWidth = 1.2; g.beginPath(); g.arc(mu[0], mu[1], Math.max(4, c.sigma / (EXT[1] - EXT[0]) * box.w), 0, 7); g.stroke(); g.restore();
          marker(g, tg[0], tg[1], 'cross', C.truth);
          T(g, 'y', tg[0] + 9, tg[1] - 16, { size: 13, kind: 'mono', color: C.truth });
          // preview path (after update)
          if (st >= 2 * N + 2 && prev) { const pp = prev.path.map((q) => P(q, box)); lib.line(g, pp, { color: lib.rgba(C.ink, 0.8), width: 1.6, dash: [5, 4] }); const e = pp[pp.length - 1]; marker(g, e[0], e[1], 'ring', C.ink); }
          // forward path
          const upto = Math.min(st, N);
          for (let i = 1; i <= upto; i++) { const pr = st === i ? p : 1; lib.line(g, [pts[i - 1], [lib.lerp(pts[i - 1][0], pts[i][0], pr), lib.lerp(pts[i - 1][1], pts[i][1], pr)]], { color: C.ebt, width: 2.4 }); }
          for (let i = 0; i <= upto; i++) { if (st === i && S.prog < 1 && i > 0) continue; lib.dot(g, pts[i][0], pts[i][1], i === 0 ? 5.5 : 4.5, i === 0 ? C.ink : C.ebt, { stroke: C.bg, lw: 1.5 }); }
          T(g, 'ŷ₀', pts[0][0] + 8, pts[0][1] + 4, { size: 13, kind: 'mono', color: C.ink });
          if (upto === N && st > N) T(g, 'ŷ' + sub(N), pts[N][0] + 8, pts[N][1] + 4, { size: 13, kind: 'mono', color: C.ebt });
          // backward arrows: −∂L/∂ŷ_i (where the loss wants ŷ_i to have been)
          const known = []; for (let i = 0; i <= N; i++) { const kn = S.detach ? (i >= 1 && (k >= N - i + 1 || (i === N && k >= 0))) : k >= N - i; if (kn && R.adj[i]) known.push(i); }
          const mx = Math.max(1e-12, ...R.adj.filter(Boolean).map((a) => Math.hypot(a[0], a[1])));
          known.forEach((i) => { const a = R.adj[i], na = Math.hypot(a[0], a[1]); if (na < 1e-12) return; const len = Math.max(8, 0.16 * w * na / mx); const d = [-a[0] / na, a[1] / na]; lib.arrow(g, pts[i][0], pts[i][1], pts[i][0] + d[0] * len, pts[i][1] + d[1] * len, { color: C.bad, width: 2.2, head: 7 }); });
          g.restore();
          plate(g, 'E(x, ŷ) · step ' + STEPS[S.ck].toLocaleString() + (S.edits ? ' + ' + S.edits + ' update' + (S.edits > 1 ? 's' : '') : ''), 8, w - 8);
          plate(g, 'view ' + f2(view[1] - view[0]) + ' wide', w - 8, w - 8, { align: 'right', color: C.muted, size: 12 });
        }

        function drawContrib(g, w) {
          const N = S.N, k = S.stage - (N + 1), pad = 10, lw = 46;
          const bw = (w - lw - pad) / (N + 1);
          // top: |∂L/∂ŷ_i|
          const top = 22, hh = 72;
          T(g, S.detach ? 'Local loss gradient at each ŷᵢ  |∂Jᵢ/∂ŷᵢ|' : 'Error reaching each ŷᵢ  |∂L/∂ŷᵢ|', 0, 2, { size: 13, color: C.ink });
          const an = R.adj.map((a) => (a ? Math.hypot(a[0], a[1]) : 0)), amax = Math.max(1e-12, ...an);
          for (let i = 0; i <= N; i++) {
            const x = lw + i * bw, known = S.detach ? (i >= 1 && (k >= N - i + 1 || (i === N && k >= 0))) : k >= N - i;
            const v = known ? an[i] / amax : 0, bh = v * hh;
            g.fillStyle = lib.rgba(C.bad, known ? 0.85 : 0.15); g.fillRect(x + bw * 0.2, top + hh - Math.max(known ? 1.5 : 0, bh), bw * 0.6, Math.max(known ? 1.5 : 0, bh));
            if (!known) { g.strokeStyle = C.rule; g.strokeRect(x + bw * 0.2 + 0.5, top + 0.5, bw * 0.6 - 1, hh - 1); }
            T(g, 'ŷ' + sub(i), x + bw / 2, top + hh + 6, { size: 12.5, kind: 'mono', align: 'center' });
          }
          // bottom: ‖c_i‖ and share
          const top2 = top + hh + 44, hh2 = 72;
          T(g, 'Step i\'s piece of ∂L/∂θ  ‖cᵢ‖ (share)', 0, top2 - 22, { size: 13, color: C.ink });
          const cmax = Math.max(1e-12, ...R.cn);
          for (let i = 1; i <= N; i++) {
            const x = lw + i * bw - bw / 2, known = k >= N - i + 1, v = known ? R.cn[i] / cmax : 0, bh = v * hh2;
            g.fillStyle = known ? stepCol(i, N) : lib.rgba(C.faint, 0.15); g.fillRect(x + bw * 0.2, top2 + hh2 - Math.max(known ? 1.5 : 0, bh), bw * 0.6, Math.max(known ? 1.5 : 0, bh));
            if (!known) { g.strokeStyle = C.rule; g.strokeRect(x + bw * 0.2 + 0.5, top2 + 0.5, bw * 0.6 - 1, hh2 - 1); }
            T(g, 'c' + sub(i), x + bw / 2, top2 + hh2 + 6, { size: 12.5, kind: 'mono', align: 'center' });
            if (known) T(g, Math.round(R.share[i] * 100) + '%', x + bw / 2, top2 + hh2 - bh - 16, { size: 12, kind: 'mono', align: 'center', color: C.ink });
          }
          // axis stubs
          g.strokeStyle = C.faint; g.lineWidth = 1; g.beginPath(); g.moveTo(lw, top + hh + 0.5); g.lineTo(w - pad, top + hh + 0.5); g.moveTo(lw, top2 + hh2 + 0.5); g.lineTo(w - pad, top2 + hh2 + 0.5); g.stroke();
          T(g, sci(amax), lw - 4, top, { size: 12, kind: 'mono', align: 'right', color: C.faint });
          T(g, sci(cmax), lw - 4, top2, { size: 12, kind: 'mono', align: 'right', color: C.faint });
        }

        function statusHTML() {
          const N = S.N, st = S.stage, k = st - (N + 1), pth = R.path, al = f2(S.alpha);
          if (st === 0) return '<b>Sample ŷ₀ ~ N(0, I).</b> The guess starts as noise at ' + vec(pth[0]) + ', energy ' + f3(R.E[0]) + '. The target y ' + vec(smp.tgt) + ' is one noisy sample around μ(x); training never sees μ.';
          if (st <= N) { const gi = [(pth[st - 1][0] - pth[st][0]) / S.alpha, (pth[st - 1][1] - pth[st][1]) / S.alpha]; return '<b>Forward step ' + st + ' of ' + N + '.</b> ŷ' + sub(st) + ' = ŷ' + sub(st - 1) + ' − α∇E with α = ' + al + ' and ∇E = ' + vec(gi) + ', so ŷ moves to ' + vec(pth[st]) + '. Energy ' + f3(R.E[st - 1]) + ' → ' + f3(R.E[st]) + '. Getting ∇E costs a forward and a backward pass, and the graph is kept for later (<code>create_graph=True</code>, p.44).'; }
          if (k === 0) return S.detach ? '<b>Score every step.</b> Detached mode puts a loss on each ŷᵢ: L = Σ‖ŷᵢ − y‖² = ' + f3(R.L) + '. Nothing ever tells the model what the energy value should be (Fig E.1, p.37).' : '<b>Score only the end.</b> J = ‖ŷ' + sub(N) + ' − y‖² = ' + f3(R.L) + '. This is the only supervision. Nothing ever tells the model what the energy value should be (Fig E.1, p.37). Its gradient a' + sub(N) + ' = 2(ŷ' + sub(N) + ' − y) has length ' + sci(Math.hypot(...R.adj[N])) + '.';
          if (k >= 1 && k <= N) {
            const i = N - k + 1, Hm = R.H[i - 1], ev = eig2(Hm), f = ev.map((l) => Math.abs(1 - S.alpha * l));
            let s = '<b>Backprop through step ' + i + '.</b> (1) θ gets its piece c' + sub(i) + ' = −α ∇θ[a' + sub(i) + ' · ∇ŷE(ŷ' + sub(i - 1) + ')], ‖c' + sub(i) + '‖ = ' + sci(R.cn[i]) + ': a gradient of a gradient. ';
            if (S.detach) s += '(2) Nothing flows back to ŷ' + sub(i - 1) + ': the step is detached, so no Hessian term.';
            else s += '(2) The error moves back: a' + sub(i - 1) + ' = (I − αH) a' + sub(i) + '. H at ŷ' + sub(i - 1) + ' has eigenvalues ' + f2(ev[0]) + ' and ' + f2(ev[1]) + ', so I − αH scales the error by ' + f2(f[0]) + ' and ' + f2(f[1]) + ' (' + (Math.max(f[0], f[1]) >= 1 ? 'can grow: not a bowl here' : Math.min(f[0], f[1]) > 0.75 ? 'almost unchanged: the landscape is nearly flat here' : Math.max(f[0], f[1]) < 0.35 ? 'nearly erased: a bowl with curvature close to 1/α' : 'shrinks: a bowl') + '). |a| ' + sci(Math.hypot(...R.adj[i])) + ' → ' + sci(Math.hypot(...R.adj[i - 1])) + '.';
            return s;
          }
          let big = 1; for (let i = 1; i <= N; i++) if (R.share[i] > R.share[big]) big = i;
          return (S.msg ? '<span class="training-msg">' + S.msg + '</span>' : '') + '<b>Sum the pieces.</b> ∂L/∂θ = ' + Array.from({ length: N }, (_, j) => 'c' + sub(j + 1)).join(' + ') + ', ‖·‖ = ' + sci(R.tn) + '. Step ' + big + ' carries the largest share (' + Math.round(R.share[big] * 100) + '%). One update with η = ' + S.eta.toFixed(3) + ' would move the same start to the white ring, and L from ' + f3(R.L) + ' to ' + f3(prev.L) + (prev.L > R.L ? ' (worse: η too large)' : '') + '. Press <b>Apply update to θ</b> to make it real.';
        }
        function drawAll() {
          graphC.render(); landC.render(); contribC.render();
          status.innerHTML = statusHTML();
          readout.innerHTML = '<span>stage <b>' + S.stage + '/' + SS() + '</b></span><span>L <b>' + f3(R.L) + '</b></span><span>‖∂L/∂θ‖ <b>' + sci(R.tn) + '</b></span><span>check vs finite differences: rel. error <b>' + (fdErr ? fdErr.rel.toExponential(0) : '–') + '</b></span>';
          restoreB.disabled = S.edits === 0;
          stepB.disabled = S.stage >= SS();
        }
        const loop = lib.loop((dt) => {
          S.prog = Math.min(1, S.prog + dt / (S.playing ? 0.85 : 0.45));
          if (S.prog >= 1) {
            if (S.playing && S.stage < SS()) { S.stage++; S.prog = 0; }
            else if (!S.playing || S.stage >= SS()) { drawAll(); if (S.playing) { S.playing = false; play.setPlaying(false); } return false; }
          }
          drawAll();
        });
        function animOnce() { if (reduce) { S.prog = 1; drawAll(); return; } loop.start(); }
        function stop() { S.playing = false; play.setPlaying(false); loop.stop(); S.prog = 1; }
        // start informative: full picture drawn; auto-play once when first visible
        S.stage = SS(); drawAll();
        lib.whenVisible(sec, () => { if (reduce) return; setTimeout(() => { if (S.stage === SS() && !S.playing && S.edits === 0) { S.stage = 0; S.prog = 1; S.playing = true; play.setPlaying(true); loop.start(); } }, 400); });
      })();

      // =============================================================================================
      // PART 2: a 1D energy you train by hand
      // =============================================================================================
      (function part2() {
        const sec = h('section', { class: 'training-part', 'aria-labelledby': 'training-p2' }); el.appendChild(sec);
        const hd = partHeader(2, 'Train a one-dimensional energy by hand', null, 'A landscape with three knobs: <b>E<sub>θ</sub>(ŷ) = a·(ŷ − m)² + w·sin(3ŷ)</b>. The curvature a sets how steep the bowl is, m where it sits, w how tall the bumps are. Each <b>Train step</b> runs Algorithm 1 on a mini-batch of 12 random starts and moves θ against the exact gradient. Watch the bowl slide to the target, the bumps flatten, and the curvature tune itself to the step size.');
        hd.querySelector('h3').id = 'training-p2'; sec.appendChild(hd);
        const TH0 = { a: 0.15, m: -0.8, w: 0.3 }, TGT = 1.2, B = 12;
        const S = { N: 2, alpha: 1, randA: false, lr: 0.2, th: Object.assign({}, TH0), prev: null, t: 0, hist: [], seed: 11, playing: false };
        let batch = null, run = null;
        function newBatch() { const r = lib.rng(5000 + S.seed * 31 + S.t * 977); const starts = [], alphas = []; for (let j = 0; j < B; j++) { starts.push(r.normal()); alphas.push(S.randA ? S.alpha * Math.exp((2 * r() - 1) * Math.LN2) : S.alpha); } batch = { starts, alphas }; }
        function evalRun() { run = toyRun(S.th, batch, S.N, TGT); }
        function trainStep() {
          const { G, L } = run; if (!isFinite(L) || ![G.a, G.m, G.w].every(isFinite)) { S.t++; newBatch(); evalRun(); return; }
          S.hist.push(L);
          // a is updated in log space so it stays positive; global-norm clip 1 (as in our toy's training)
          const gr = S.th.a * G.a, gn = Math.hypot(gr, G.m, G.w), s = Math.min(1, 1 / (gn + 1e-12));
          S.prev = Object.assign({}, S.th);
          S.th = { a: Math.exp(Math.log(S.th.a) - S.lr * s * gr), m: S.th.m - S.lr * s * G.m, w: S.th.w - S.lr * s * G.w };
          S.t++; newBatch(); evalRun();
        }
        newBatch(); evalRun();

        const nSl = lib.slider({ id: 'training-p2-n', label: 'Unrolled steps N', min: 1, max: 6, step: 1, value: S.N, oninput: (v) => { S.N = v; newBatch(); evalRun(); draw(); } });
        const aSl = lib.slider({ id: 'training-p2-a', label: 'Step size α', min: 0.25, max: 2, step: 0.05, value: S.alpha, fmt: f2, oninput: (v) => { S.alpha = v; newBatch(); evalRun(); draw(); } });
        const lrSl = lib.slider({ id: 'training-p2-lr', label: 'Learning rate η (outer loop)', min: 0.02, max: 0.5, step: 0.01, value: S.lr, fmt: f2, oninput: (v) => { S.lr = v; draw(); } });
        const randB = h('button', { type: 'button', class: 'btn', 'aria-pressed': 'false' }, 'Random α per start (×½ to ×2)');
        randB.addEventListener('click', () => { S.randA = !S.randA; randB.setAttribute('aria-pressed', String(S.randA)); newBatch(); evalRun(); draw(); });
        const stepB = btn('Train step', () => { stop(); trainStep(); draw(); }, 'primary');
        const play = playBtn(() => { if (S.playing) stop(); else { S.playing = true; play.setPlaying(true); acc = 0; loop.start(); } });
        const resetB = btn('Reset', () => { stop(); S.th = Object.assign({}, TH0); S.prev = null; S.t = 0; S.hist = []; newBatch(); evalRun(); draw(); });
        const left = h('div', { class: 'card training-ctls' }, nSl.el, aSl.el, randB, lrSl.el, h('div', { class: 'training-btns' }, stepB, play, resetB),
          h('p', { class: 'training-note' }, 'Train step = one outer-loop update of θ. Play runs them continuously.'));
        const mathCard = h('div', { class: 'card stack' }, h('h4', {}, 'The exact gradient, written out'), h('div', { class: 'training-math', html: 'forward&nbsp; ŷ<sub>i+1</sub> = ŷ<sub>i</sub> − α E′(ŷ<sub>i</sub>)<br>loss&nbsp;&nbsp;&nbsp;&nbsp; L = mean (ŷ<sub>N</sub> − y)²<br>backward a<sub>N</sub> = 2(ŷ<sub>N</sub> − y)<br>&nbsp;&nbsp;∂L/∂θ += −α · ∂E′(ŷ<sub>i</sub>)/∂θ · a<sub>i+1</sub><br>&nbsp;&nbsp;a<sub>i</sub> = (1 − α E″(ŷ<sub>i</sub>)) · a<sub>i+1</sub><br><span class="training-dim">E′ = 2a(ŷ−m) + 3w cos 3ŷ<br>E″ = 2a − 9w sin 3ŷ &nbsp;(the 1D "Hessian")</span>' }),
          h('p', { class: 'training-note' }, 'In 1D the Hessian-vector product is just multiplication by E″. ∂L/∂a is applied in log space so a stays positive, and the update is clipped to norm 1, as in our toy\'s training.'));
        const right = h('div', { class: 'card stack' });
        right.appendChild(h('div', { class: 'training-badges' }, lib.badge('ext', 'illustration: 3-parameter energy, exact gradients'), h('span', { class: 'training-small' }, 'same algorithm as the paper, toy-sized')));
        const mainC = rcanvas(right, { label: 'Top: the 1D energy curve now (solid) and before the last update (dashed), with the target. Bottom: each start of the mini-batch descending for N steps.', height: (w) => 200 + 26 + (S.N + 1) * 30 + 16, draw: drawMain });
        const tbl = h('div', { class: 'tbl' }); right.appendChild(tbl);
        const duo = h('div', { class: 'training-duo eq' }); right.appendChild(duo);
        const lossC = rcanvas(duo, { label: 'Training loss of each mini-batch against the number of train steps.', height: () => 150, draw: drawLoss });
        const note = h('div', { class: 'training-status', 'aria-live': 'polite' }); duo.appendChild(note);
        sec.appendChild(h('div', { class: 'bench' }, left, right));
        sec.appendChild(h('div', { class: 'grid2' }, mathCard, lib.callout('insight', 'The landscape is shaped for its own optimizer', 'With N = 1, training drives the curvature to exactly a = 1/(2α): the bowl whose single gradient step drops any start onto the bottom (E″ = 1/α). Change α and retrain: the bowl follows. The real 2D toy EBT learned the same thing. Its Hessian eigenvalues at the minima are 0.78 to 0.92 across 64 contexts (toy data, <code>basin_stats</code>), close to 1/α₀ = 1 for its base step size α₀ = 1. One step with α = 1 then removes about 85% of the error.')));

        const X = (y, w) => 44 + (y + 3) / 6 * (w - 58);
        function drawMain(g, w) {
          const th = S.th, ys = []; for (let i = 0; i <= 240; i++) ys.push(-3 + 6 * i / 240);
          const cur = ys.map((y) => e1(th, y)), gh = S.prev ? ys.map((y) => e1(S.prev, y)) : null;
          let lo = Math.min(...cur, ...(gh || [])), hi = Math.max(...cur, ...(gh || [])); hi = Math.min(hi, lo + 12); const pad = (hi - lo) * 0.08; lo -= pad; hi += pad;
          const ax = { x: 44, y: 22, w: w - 58, h: 170 };
          const Yv = (e) => ax.y + ax.h - (e - lo) / (hi - lo) * ax.h;
          // frame
          g.strokeStyle = C.rule; g.lineWidth = 1; g.strokeRect(ax.x + 0.5, ax.y + 0.5, ax.w, ax.h);
          T(g, 'energy E(ŷ)', ax.x, 2, { size: 13, color: C.ink });
          T(g, 'step ' + S.t, w - 14, 2, { size: 13, kind: 'mono', align: 'right', color: C.muted });
          [-3, -2, -1, 0, 1, 2, 3].forEach((v) => { g.strokeStyle = lib.rgba(C.rule, 0.6); g.beginPath(); g.moveTo(X(v, w), ax.y); g.lineTo(X(v, w), ax.y + ax.h); g.stroke(); });
          // target line spanning both panels
          const fanTop = ax.y + ax.h + 30, rowH = 30, fanH = S.N * rowH;
          g.save(); g.setLineDash([5, 4]); g.strokeStyle = C.truth; g.lineWidth = 1.6; g.beginPath(); g.moveTo(X(TGT, w), ax.y); g.lineTo(X(TGT, w), fanTop + fanH + 6); g.stroke(); g.restore();
          { const tx = X(TGT, w), right = tx + 100 > w; T(g, 'target y = ' + TGT, right ? tx - 6 : tx + 6, ax.y + 4, { size: 13, color: C.truth, align: right ? 'right' : 'left' }); }
          g.save(); g.beginPath(); g.rect(ax.x, ax.y, ax.w, ax.h); g.clip();
          if (gh) lib.line(g, ys.map((y, i) => [X(y, w), Yv(gh[i])]), { color: lib.rgba(C.muted, 0.8), width: 1.6, dash: [5, 4] });
          lib.line(g, ys.map((y, i) => [X(y, w), Yv(cur[i])]), { color: C.ebt, width: 2.6 });
          // where the batch ends: dots on the curve
          run.paths.forEach((p) => { const yN = p[S.N]; if (yN < -3 || yN > 3) return; lib.dot(g, X(yN, w), Yv(e1(th, yN)), 4, C.ebt, { stroke: C.bg, lw: 1.2 }); });
          g.restore();
          // fan: rows i = 0..N
          for (let i = 0; i <= S.N; i++) { const y = fanTop + i * rowH; g.strokeStyle = lib.rgba(C.rule, 0.8); g.beginPath(); g.moveTo(ax.x, y); g.lineTo(ax.x + ax.w, y); g.stroke(); T(g, 'ŷ' + sub(i), 6, y, { size: 13, kind: 'mono', baseline: 'middle', color: i === S.N ? C.ebt : C.muted }); }
          run.paths.forEach((p) => {
            const pts = p.map((v, i) => [clamp(X(v, w), ax.x, ax.x + ax.w), fanTop + i * rowH]);
            lib.line(g, pts, { color: lib.rgba(C.ebt, 0.55), width: 1.4 });
            lib.dot(g, pts[0][0], pts[0][1], 3, C.ink); const e = pts[pts.length - 1]; lib.dot(g, e[0], e[1], 4, C.ebt);
          });
          [-3, -2, -1, 0, 1, 2].forEach((v) => T(g, String(v), X(v, w), fanTop + fanH + 10, { size: 12, kind: 'mono', align: 'center', color: C.faint }));
          T(g, 'ŷ →', ax.x + ax.w, fanTop + fanH + 10, { size: 13, kind: 'mono', align: 'right', color: C.muted });
        }
        function drawLoss(g, w, hh) {
          const hs = S.hist.concat([run.L]), n = hs.length, ax = { x: 54, y: 22, w: w - 66, h: hh - 50 };
          let lo = Math.max(1e-6, Math.min(...hs.filter((v) => v > 0)) * 0.7), hi = Math.max(...hs) * 1.3;
          const yt = []; for (let e = Math.ceil(Math.log10(lo)); e <= Math.floor(Math.log10(hi)); e++) yt.push(Math.pow(10, e));
          if (yt.length < 2) { lo = Math.min(lo, Math.pow(10, Math.floor(Math.log10(lo)))); hi = Math.max(hi, Math.pow(10, Math.ceil(Math.log10(hi)))); yt.length = 0; for (let e = Math.round(Math.log10(lo)); e <= Math.round(Math.log10(hi)); e++) yt.push(Math.pow(10, e)); }
          const A = lib.axes(g, { x: ax.x, y: ax.y, w: ax.w, h: ax.h, xlim: [0, Math.max(10, n - 1)], ylim: [lo, hi], ylog: true, xticks: [0, Math.max(10, n - 1)], yticks: yt.length > 5 ? yt.filter((_, i) => i % 2 === 0) : yt, yfmt: (v) => (v >= 1 ? String(v) : v >= 1e-3 ? String(Number(v.toPrecision(1))) : '1e' + Math.round(Math.log10(v))), size: 12 });
          T(g, 'batch loss L (log)', ax.x, 2, { size: 13, color: C.ink });
          lib.plot(g, A, hs.map((v, i) => [i, Math.max(lo, v)]), { color: C.ebt, width: 2 });
          lib.dot(g, A.X(n - 1), A.Y(Math.max(lo, run.L)), 4, C.ebt);
          T(g, 'train steps →', ax.x + ax.w / 2, ax.y + ax.h + 10, { size: 12, align: 'center', color: C.faint });
        }
        function draw() {
          mainC.render(); lossC.render();
          const G = run.G, th = S.th, aStar = 1 / (2 * S.alpha);
          const row = (name, role, v, gr, extra) => '<tr><td><span class="mono">' + name + '</span> <span class="training-dim">' + role + '</span></td><td class="num">' + f3(v) + '</td><td class="num">' + sci(gr) + '</td><td>' + (extra || '') + '</td></tr>';
          tbl.innerHTML = '<table><thead><tr><th>θ</th><th>value</th><th>∂L/∂θ</th><th>reference</th></tr></thead><tbody>' +
            row('a', 'curvature', th.a, G.a, '1/(2α) = ' + f2(aStar)) + row('m', 'center', th.m, G.m, 'target ' + TGT) + row('w', 'bumps', th.w, G.w, '0 = no traps') + '</tbody></table>';
          const misses = run.paths.filter((p) => Math.abs(p[S.N] - TGT) > 0.3).length;
          note.innerHTML = '<b>Batch loss ' + sci(run.L) + '</b> after ' + S.t + ' train steps. ' + misses + ' of ' + B + ' starts end more than 0.3 from the target. ' +
            (S.t === 0 ? 'The bowl sits at m = −0.8 and is too shallow for ' + S.N + ' step' + (S.N > 1 ? 's' : '') + ' to reach y. Press Train step or Play.' : Math.abs(th.a - aStar) / aStar < 0.08 ? 'Curvature a = ' + f3(th.a) + ' ≈ 1/(2α): one step lands on the bottom.' : 'Curvature a = ' + f3(th.a) + ' vs 1/(2α) = ' + f2(aStar) + '.' + (S.N > 1 ? ' With N > 1 a range of curvatures works, so a stops moving once the loss is small.' : ''));
        }
        let acc = 0;
        const loop = lib.loop((dt) => { acc += dt; let k = 0; while (acc > 0.09 && k < 3) { acc -= 0.09; trainStep(); k++; } draw(); if (S.t >= 400) { stop(); return false; } });
        function stop() { S.playing = false; play.setPlaying(false); loop.stop(); }
        draw();
      })();

      // =============================================================================================
      // PART 3: checkpoint scrubber over the real toy training run
      // =============================================================================================
      (function part3() {
        const sec = h('section', { class: 'training-part', 'aria-labelledby': 'training-p3' }); el.appendChild(sec);
        const hd = partHeader(3, 'Watch the toy landscape form over training', null, 'Twelve real checkpoints of the 2D toy EBT, trained for 12,000 Adam steps with the full recipe (random α and N, Langevin noise, replay buffer). Left: a near-deterministic context; right: a noisy one. White paths are 4-step descents with α = 1, the training setting, from the same 8 starts. Scrub or press Play.');
        hd.querySelector('h3').id = 'training-p3'; sec.appendChild(hd);
        const S = { pos: 3, playing: false };
        const nC = M.nCkpt, starts = D.trajectory_starts;
        const cache = new Map();
        function ckData(k, ci) {
          const key = k + ':' + ci; if (cache.has(key)) return cache.get(key);
          const grid = M.storedGrid(ci, STEPS[k]); const x = M.contexts[ci].x;
          const paths = starts.map((s) => M.descend(k, x, s, { alpha: 1, steps: 4 }).path);
          // minimum and curvature from the grid (3-point stencil, spacing 5/63)
          let best = Infinity, br = 0, bc = 0; grid.forEach((row, r) => row.forEach((v, c) => { if (v < best) { best = v; br = r; bc = c; } }));
          const n = grid.length, hstep = (EXT[1] - EXT[0]) / (n - 1), cc = clamp(bc, 1, n - 2), rr = clamp(br, 1, n - 2);
          const cx = (grid[br][cc + 1] - 2 * grid[br][cc] + grid[br][cc - 1]) / hstep / hstep, cy = (grid[rr + 1][bc] - 2 * grid[rr][bc] + grid[rr - 1][bc]) / hstep / hstep;
          const at = [EXT[0] + bc * hstep, EXT[3] - br * hstep];
          const ends = paths.map((p) => p[p.length - 1]); const mu = M.contexts[ci].mu;
          const err = ends.reduce((s, e) => s + sqd(e, mu), 0) / ends.length;
          const d = { grid, paths, Emin: best, at, curv: [cx, cy], err }; cache.set(key, d); return d;
        }
        const sl = lib.slider({ id: 'training-p3-k', label: 'Checkpoint', min: 0, max: nC - 1, step: 0.01, value: S.pos, fmt: (v) => 'step ' + STEPS[Math.round(v)].toLocaleString(), oninput: (v) => { stop(); S.pos = v; draw(); } });
        sl.input.addEventListener('change', () => { S.pos = Math.round(S.pos); sl.set(S.pos); draw(); });
        const play = playBtn(() => { if (S.playing) stop(); else { if (S.pos >= nC - 1) S.pos = 0; S.playing = true; play.setPlaying(true); loop.start(); } });
        const prevB = btn('◀ Prev', () => { stop(); S.pos = Math.max(0, Math.ceil(S.pos) - 1); sl.set(S.pos); draw(); });
        const nextB = btn('Next ▶', () => { stop(); S.pos = Math.min(nC - 1, Math.floor(S.pos) + 1); sl.set(S.pos); draw(); });
        const story = h('div', { class: 'training-status', 'aria-live': 'polite' });
        const left = h('div', { class: 'card training-ctls' }, sl.el, h('div', { class: 'training-btns' }, prevB, play, nextB), story,
          h('p', { class: 'training-note' }, 'Between checkpoints the picture is a blend of the two neighbouring stored grids, shown only while it moves.'));
        const right = h('div', { class: 'card stack' });
        right.appendChild(h('div', { class: 'training-badges' }, lib.badge('toy', 'stored checkpoints'), h('span', { class: 'training-small' }, 'energy over ŷ, fixed x, fixed θ')));
        const pairC = rcanvas(right, { label: 'Energy landscapes of the toy model at the chosen training checkpoint, for a low-noise and a high-noise context, with 4-step descents.', height: (w) => (w >= 520 ? (w - 16) / 2 + 28 : 2 * w + 56), draw: drawPair });
        right.appendChild(h('div', { class: 'legend' }, h('span', {}, h('i', { style: 'background:var(--ink)' }), 'start ŷ₀ and 4-step path'), h('span', {}, h('i', { style: 'background:var(--ebt)' }), 'where ŷ₄ lands'), h('span', {}, h('i', { style: 'background:var(--truth)' }), 'μ(x), dashed ring = noise σ')));
        const tbl = h('div', { class: 'tbl' }); right.appendChild(tbl);
        const lossC = rcanvas(right, { label: 'Training loss curve of the toy model on log-log axes with a cursor at the chosen checkpoint.', height: (w) => (w < 500 ? 190 : 210), draw: drawLoss });
        right.appendChild(h('div', { class: 'legend' }, h('span', {}, h('i', { style: 'background:var(--ebt)' }), 'training loss per Adam step (EMA; faint = raw)'), h('span', {}, h('i', { style: 'background:var(--ink)' }), 'eval after 4 steps, at checkpoints'), h('span', {}, h('i', { style: 'background:var(--truth)' }), 'noise floor: error of a perfect predictor')));
        sec.appendChild(h('div', { class: 'bench' }, left, right));

        function blendGrid(k0, k1, f, ci) { const a = ckData(k0, ci).grid, b = ckData(k1, ci).grid; if (f <= 0) return a; if (f >= 1) return b; return a.map((row, r) => row.map((v, c) => v + (b[r][c] - v) * f)); }
        function drawPair(g, w) {
          const side = w >= 520, s = side ? (w - 16) / 2 : w;
          const k0 = Math.floor(S.pos), k1 = Math.min(nC - 1, k0 + 1), f = S.pos - k0, kn = Math.round(S.pos), crisp = Math.abs(S.pos - kn) < 0.02;
          [0, 1].forEach((ci) => {
            const box = side ? { x: ci * (s + 16), y: 26, w: s, h: s } : { x: 0, y: 26 + ci * (s + 28), w: s, h: s };
            const c = M.contexts[ci];
            T(g, (ci === 0 ? 'x = 0.25 · σ = 0.03 · near-deterministic' : 'x = 0.75 · σ = 0.30 · noisy'), box.x, box.y - 22, { size: 13, color: C.ink });
            const grid = blendGrid(k0, k1, f, ci); drawLandscape(g, grid, box);
            const mu = P(c.mu, box);
            g.save(); g.strokeStyle = lib.rgba(C.truth, 0.85); g.setLineDash([3, 3]); g.lineWidth = 1.4; g.beginPath(); g.arc(mu[0], mu[1], Math.max(4, c.sigma / (EXT[1] - EXT[0]) * s), 0, 7); g.stroke(); g.restore();
            marker(g, mu[0], mu[1], 'cross', C.truth);
            const d = ckData(kn, ci), al = crisp ? 1 : clamp(1 - Math.abs(S.pos - kn) * 3, 0, 1);
            if (al > 0) {
              g.save(); g.globalAlpha = al;
              d.paths.forEach((p) => { const pts = p.map((q) => P(q, box)); lib.line(g, pts, { color: lib.rgba(C.ink, 0.8), width: 1.4 }); lib.dot(g, pts[0][0], pts[0][1], 2.8, C.ink); const e = pts[pts.length - 1]; lib.dot(g, e[0], e[1], 4, C.ebt, { stroke: C.bg, lw: 1.2 }); });
              g.restore();
            }
            plate(g, 'step ' + STEPS[kn].toLocaleString(), box.x + box.w - 8, box.y + box.h - 8, { align: 'right', kind: 'mono' });
          });
        }
        function drawLoss(g, w, hh) {
          const lc = D.loss_curve, ev = D.eval.curve, ax = { x: 46, y: 26, w: w - 58, h: hh - 52 };
          const A = lib.axes(g, { x: ax.x, y: ax.y, w: ax.w, h: ax.h, xlim: [1, 12000], ylim: [0.05, 4], xlog: true, ylog: true, xticks: [1, 10, 100, 1000, 10000], yticks: [0.1, 1], xfmt: (v) => (v >= 1000 ? v / 1000 + 'k' : String(v)), yfmt: String, size: 12 });
          T(g, 'training loss (MSE of ŷN vs noisy y)', ax.x, 4, { size: 13, color: C.ink });
          lib.plot(g, A, lc.step.map((s, i) => [s, lc.raw[i]]), { color: lib.rgba(C.ebt, 0.25), width: 1 });
          lib.plot(g, A, lc.step.map((s, i) => [s, lc.loss[i]]), { color: C.ebt, width: 2.2 });
          const nf = lc.noise_floor; g.save(); g.setLineDash([5, 4]); g.strokeStyle = C.truth; g.lineWidth = 1.4; g.beginPath(); g.moveTo(A.X(1), A.Y(nf)); g.lineTo(A.X(12000), A.Y(nf)); g.stroke(); g.restore();
          T(g, 'noise floor ' + nf.toFixed(3), A.X(1) + 6, A.Y(nf) + 4, { size: 12, color: C.truth });
          ev.step.forEach((s, i) => { if (s < 1) return; lib.dot(g, A.X(s), A.Y(ev.mse_vs_y_N4[i]), 3.2, C.ink); });
          // cursor (interpolated in log step)
          const k0 = Math.floor(S.pos), k1 = Math.min(nC - 1, k0 + 1), f = S.pos - k0, l0 = Math.log10(Math.max(1, STEPS[k0])), l1 = Math.log10(Math.max(1, STEPS[k1]));
          const cx = A.X(Math.pow(10, l0 + (l1 - l0) * f));
          g.strokeStyle = C.ink; g.lineWidth = 1.5; g.beginPath(); g.moveTo(cx, ax.y); g.lineTo(cx, ax.y + ax.h); g.stroke();
        }
        function storyText(k) {
          const s = STEPS[k];
          if (s <= 10) return '<b>Random weights.</b> The energy is a shallow bowl near the origin (the small λ‖ŷ‖² term plus a random tilt). It knows nothing about the data, so 4 descent steps end far from μ. Eval error after 4 steps: ' + f3(D.eval.curve.mse_vs_mu_N4[k]) + '.';
          if (s <= 50) return '<b>A basin is being dragged toward μ.</b> It is still shallow, so 4 steps with α = 1 stop short and the training loss is far above the noise floor. Every bit of this reshaping came from backpropagating through descents like these.';
          if (s <= 200) return '<b>The basin sits on μ.</b> It is now steep enough that 4 steps land in it, and the training loss reaches the noise floor: the remaining error is the noise in y, which no predictor can remove.';
          return '<b>Refinement.</b> The bowl\'s curvature settles near 1/α (Hessian eigenvalues 0.78 to 0.92 at the end), so one step already lands near the bottom. The energy value at the minimum drifts (' + f2(ckData(6, 0).Emin) + ' at step 400, ' + f2(ckData(11, 0).Emin) + ' at the end): no loss pins absolute energies, only the shape is trained.';
        }
        function draw() {
          pairC.render(); lossC.render();
          const kn = Math.round(S.pos); story.innerHTML = storyText(kn);
          const d0 = ckData(kn, 0), d1 = ckData(kn, 1), mu0 = M.contexts[0].mu, mu1 = M.contexts[1].mu;
          const tr = (lab, a, b) => '<tr><td>' + lab + '</td><td class="num">' + a + '</td><td class="num">' + b + '</td></tr>';
          tbl.innerHTML = '<table><thead><tr><th>at step ' + STEPS[kn].toLocaleString() + '</th><th>x = 0.25</th><th>x = 0.75</th></tr></thead><tbody>' +
            tr('lowest grid point', vec(d0.at), vec(d1.at)) + tr('true mean μ(x)', vec(mu0), vec(mu1)) + tr('energy there', f2(d0.Emin), f2(d1.Emin)) +
            tr('curvature ≈ (∂²E/∂ŷ₀², ∂²E/∂ŷ₁²)', f2(d0.curv[0]) + ', ' + f2(d0.curv[1]), f2(d1.curv[0]) + ', ' + f2(d1.curv[1])) + tr('‖ŷ₄ − μ‖², mean of 8 starts', sci(d0.err), sci(d1.err)) + '</tbody></table>';
        }
        const loop = lib.loop((dt) => { S.pos = Math.min(nC - 1, S.pos + dt / 1.1); sl.set(S.pos); draw(); if (S.pos >= nC - 1) { stop(); draw(); return false; } });
        function stop() { if (!S.playing) return; S.playing = false; play.setPlaying(false); loop.stop(); S.pos = Math.round(S.pos); sl.set(S.pos); }
        draw();
      })();

      // =============================================================================================
      // PART 4: landscape regularizers + Table 2
      // =============================================================================================
      (function part4() {
        const sec = h('section', { class: 'training-part', 'aria-labelledby': 'training-p4' }); el.appendChild(sec);
        const hd = partHeader(4, 'Landscape regularizers: where does the training signal land?', null, 'The weight gradient only flows through ∇ŷE at the points the unrolled descents actually visit (part 1: one piece c<sub>i</sub> per visited ŷ). So the landscape is only shaped where training paths go. The four tricks of Sec 3.3 (p.7) change where they go. Below, 5 training batches × 8 examples are sampled on a toy checkpoint with your settings. Every white dot is a point where the energy gradient receives training signal.');
        hd.querySelector('h3').id = 'training-p4'; sec.appendChild(hd);
        const HP = D.hparams, RF = 0.6; // zoom half-width around the minimum
        const S = { rA: true, rN: true, lang: true, replay: true, sigma: HP.langevin, ck: 6, ctx: 0, seed: 3 };
        let sampled = null; const zCache = new Map();
        function zoomData() {
          const key = S.ck + ':' + S.ctx; if (zCache.has(key)) return zCache.get(key);
          const grid = M.storedGrid(S.ctx, STEPS[S.ck]), n = grid.length, x = M.contexts[S.ctx].x; let best = Infinity, at = [0, 0];
          grid.forEach((row, rr) => row.forEach((v, c) => { if (v < best) { best = v; at = [EXT[0] + c * (EXT[1] - EXT[0]) / (n - 1), EXT[3] - rr * (EXT[3] - EXT[2]) / (n - 1)]; } }));
          const mn = M.descend(S.ck, x, at, { alpha: 0.5, steps: 60 }).path.pop(); // polish the grid minimum
          const zext = [mn[0] - RF, mn[0] + RF, mn[1] - RF, mn[1] + RF];
          const z = { grid, min: mn, zext, zgrid: netGrid(makeNet(D.weights[S.ck].layers), phiOf(S.ctx), 56, zext).E };
          zCache.set(key, z); return z;
        }
        function sampleAll() {
          const r = lib.rng(777 + S.seed * 104729), x = M.contexts[S.ctx].x, buf = [], unrolls = [];
          const alphaOf = () => (S.rA ? HP.alpha0 * Math.exp((2 * r() - 1) * Math.log(HP.alpha_rand_factor)) : HP.alpha0);
          const doOne = (y0, N, rep) => { const a = alphaOf(); const res = M.descend(S.ck, x, y0, { alpha: a, steps: N, sigma: S.lang ? S.sigma : 0, seed: Math.floor(r() * 1e9) + 1 }); buf.push(res.path[N].map((v) => clamp(v, -4, 4))); return { path: res.path, alpha: a, N, replay: rep }; };
          for (let j = 0; j < 12; j++) doOne([r.normal(), r.normal()], 4, false); // warm the buffer (not drawn)
          const Ns = [];
          for (let b = 0; b < 5; b++) {
            const N = S.rN ? HP.n_min + Math.floor(r() * (HP.n_max - HP.n_min + 1)) : 4; Ns.push(N);
            for (let j = 0; j < 8; j++) { const rep = S.replay && r() < HP.p_replay; const y0 = rep ? buf[Math.floor(r() * buf.length)].slice() : [r.normal(), r.normal()]; unrolls.push(doOne(y0, N, rep)); }
          }
          const z = zoomData(); let pts = 0, near = 0; const cells = new Set(), fine = new Set(); // fine = 0.05×0.05 cells on the floor
          unrolls.forEach((u) => { for (let i = 0; i < u.N; i++) { const q = u.path[i]; pts++; const d = Math.hypot(q[0] - z.min[0], q[1] - z.min[1]); if (d < 0.25) { near++; fine.add(Math.floor((q[0] - z.min[0]) / 0.05) + ':' + Math.floor((q[1] - z.min[1]) / 0.05)); } cells.add(Math.floor((q[0] - EXT[0]) / 0.25) + ':' + Math.floor((q[1] - EXT[2]) / 0.25)); } });
          const alphas = unrolls.map((u) => u.alpha);
          sampled = { unrolls, Ns, z, pts, near, cells: cells.size, fine: fine.size, amin: Math.min(...alphas), amax: Math.max(...alphas), nrep: unrolls.filter((u) => u.replay).length };
        }
        const toggles = [
          ['rA', 'Random step size α', 'Each example gets its own α (toy: α₀·e<sup>u</sup> with u uniform in [−ln 2, ln 2]; paper: random factor 2). Strides that undershoot and overshoot trace different paths through the basin, so the landscape cannot be fitted to one step size. In the toy it roughly doubles the area around the minimum that gets signal.', 'Removing it nearly kills thinking: −1.47 and 0.19.'],
          ['rN', 'Random number of steps N', 'N varies per batch (toy: 2 to 6; the paper\'s thinking models: 2 to 3). Some unrolls stop early, some keep going after they reach the floor, so the model is trained on what to do after many steps too. It barely changes this picture: its job is about how many steps, not where they go.', 'Without it, thinking longer gave exactly 0.00.'],
          ['lang', 'Langevin noise', 'Adds η ~ N(0, σ²) to every step (Eq. 2, p.7). Paths wander off the straight line to the minimum, so the neighbourhood of each path gets shaped too. In the toy, σ = 0.05 spreads the points on the basin floor over about 1.5× as many cells.', 'Without it: thinking longer 17.2 (better), combined 17.0 (worse than 18.7).'],
          ['replay', 'Replay buffer', 'Some examples restart from an earlier final prediction instead of fresh noise (toy: 25%). Chained short unrolls act like one long trajectory, so more signal lands on the floor of the basin (p.7). Clearest at step 50, where it more than doubles the points on the floor. The paper gives no buffer details; this is our toy\'s version.', 'Without it: 14.8 and 17.8.'],
        ];
        const tEls = {}, why = {};
        const tWrap = h('div', { class: 'training-toggles', role: 'group', 'aria-label': 'Landscape regularizers' });
        toggles.forEach(([k, label]) => {
          const b = h('button', { type: 'button', class: 'btn training-tg', 'aria-pressed': String(S[k]) }, h('span', { class: 'training-tg-box', 'aria-hidden': 'true' }), label);
          b.addEventListener('click', () => setT(k, !S[k]));
          tEls[k] = b; tWrap.appendChild(b);
        });
        function setT(k, v) { S[k] = v; tEls[k].setAttribute('aria-pressed', String(v)); why[k].classList.toggle('off', !v); if (k === 'lang') sigSl.el.hidden = !v; update(); }
        const sigSl = lib.slider({ id: 'training-p4-sig', label: 'Langevin σ (toy used 0.05)', min: 0, max: 0.3, step: 0.01, value: S.sigma, fmt: f2, oninput: (v) => { S.sigma = v; update(); } });
        const ckSeg = lib.segmented({ label: 'Checkpoint', options: [[3, 'step 50'], [6, 'step 400'], [11, 'step 12k']], value: S.ck, onchange: (v) => { S.ck = v; update(); } });
        const ctxSeg = lib.segmented({ label: 'Context', options: [[0, 'low noise'], [1, 'high noise']], value: S.ctx, onchange: (v) => { S.ctx = v; update(); } });
        const resB = btn('Resample', () => { S.seed++; update(); });
        const allOff = btn('All off', () => { ['rA', 'rN', 'lang', 'replay'].forEach((k) => { S[k] = false; tEls[k].setAttribute('aria-pressed', 'false'); why[k].classList.add('off'); }); sigSl.el.hidden = true; update(); });
        const allOn = btn('All on', () => { ['rA', 'rN', 'lang', 'replay'].forEach((k) => { S[k] = true; tEls[k].setAttribute('aria-pressed', 'true'); why[k].classList.remove('off'); }); sigSl.el.hidden = false; update(); });
        const left = h('div', { class: 'card training-ctls' }, h('h4', {}, 'Training recipe'), tWrap, sigSl.el, h('div', { class: 'training-btns' }, allOn, allOff, resB),
          h('div', { class: 'ctl' }, h('label', {}, h('span', {}, 'Landscape from training step')), ckSeg.el), h('div', { class: 'ctl' }, h('label', {}, h('span', {}, 'Context')), ctxSeg.el),
          h('p', { class: 'training-note' }, 'Try All off, then switch one on at a time, and watch the zoom on the right.'));
        const right = h('div', { class: 'card stack' });
        right.appendChild(h('div', { class: 'training-badges' }, lib.badge('toy', 'sampled live on a checkpoint'), h('span', { class: 'training-small' }, 'paths = training unrolls, not inference')));
        const landC = rcanvas(right, { label: 'Toy energy landscape with sampled training unrolls (left: whole plane; right: zoom on the basin floor). Dots mark the points where the energy gradient receives training signal.', height: (w) => (w >= 560 ? (w - 16) / 2 + 26 : 2 * w + 52), draw: drawLand });
        right.appendChild(h('div', { class: 'legend' }, h('span', {}, h('i', { style: 'background:var(--ink)' }), 'visited ŷᵢ (gets signal)'), h('span', {}, h('i', { style: 'background:var(--ebt)' }), 'unroll from noise'), h('span', {}, h('i', { style: 'background:var(--rnn)' }), 'unroll from replay buffer'), h('span', {}, h('i', { style: 'background:var(--truth)' }), 'μ(x)')));
        const readout = h('div', { class: 'readout' }); right.appendChild(readout);
        sec.appendChild(h('div', { class: 'bench' }, left, right));
        const whyGrid = h('div', { class: 'training-why' });
        toggles.forEach(([k, label, txt, ev]) => { why[k] = h('div', { class: 'card training-whycard' }, h('h4', {}, label), h('p', { html: txt }), h('p', { class: 'training-ev' }, h('span', { class: 'badge paper' }, 'Table 2'), ' ', ev)); whyGrid.appendChild(why[k]); });
        sec.appendChild(whyGrid);

        // Table 2 + recipe comparison
        const t2 = h('div', { class: 'card stack' });
        t2.appendChild(h('div', { class: 'training-badges' }, lib.badge('paper', 'Table 2, p.10'), h('span', { class: 'training-small' }, '% perplexity improvement, OOD BigBench Dyck')));
        t2.appendChild(h('h4', {}, 'What removing each regularizer did in the paper'));
        const t2C = rcanvas(t2, { label: 'Table 2 of the paper as bars: percent perplexity improvement from thinking longer, and from thinking longer plus self-verification, for each ablation.', height: (w) => 5 * (w < 480 ? 58 : 46) + 54, draw: drawT2 });
        const t2note = h('p', { class: 'training-note', 'aria-live': 'polite' }); t2.appendChild(t2note);
        t2.appendChild(h('p', { class: 'training-note' }, 'Read the two columns together. Full regularization gives the best combined score (18.7) but only 7.19 for thinking longer alone: per the paper, less exploration helps a single path and hurts self-verification. Our toy was trained once with all four on; we did not re-run these ablations on it.'));
        const recipe = h('div', { class: 'card stack' });
        recipe.appendChild(h('div', { class: 'training-badges' }, lib.badge('paper', 'Table D.4, p.36'), lib.badge('toy')));
        recipe.appendChild(h('h4', {}, 'Recipe: paper thinking models vs our toy'));
        recipe.appendChild(h('div', { class: 'tbl', html: '<table><thead><tr><th></th><th>Paper S2 (text)</th><th>Our toy</th></tr></thead><tbody>' +
          '<tr><td>Steps N</td><td>2 to 3, random</td><td>2 to 6, random per batch</td></tr>' +
          '<tr><td>Step size α</td><td>5, random factor 2</td><td>1.0, random factor 2 per example</td></tr>' +
          '<tr><td>Langevin noise</td><td>3 (on logits)</td><td>0.05</td></tr>' +
          '<tr><td>Replay buffer</td><td>yes</td><td>yes, 25% of each batch</td></tr>' +
          '<tr><td>Detach between steps</td><td>no</td><td>no</td></tr>' +
          '<tr><td>Loss</td><td>last step, truncated backprop</td><td>last step, full backprop</td></tr>' +
          '<tr><td>Optimizer</td><td>AdamW, LR 0.0012</td><td>Adam, LR 0.002, batch 256</td></tr></tbody></table>' }));
        sec.appendChild(h('div', { class: 'grid2' }, t2, recipe));

        const T2 = [['No random step size', -1.47, 0.19, 'rA'], ['No random num. steps', 0.0, 9.65, 'rN'], ['No Langevin dynamics', 17.2, 17.0, 'lang'], ['No replay buffer', 14.8, 17.8, 'replay'], ['Full System 2 config', 7.19, 18.7, null]];
        function activeRow() { const off = ['rA', 'rN', 'lang', 'replay'].filter((k) => !S[k]); if (off.length === 0) return 4; if (off.length === 1) return T2.findIndex((r) => r[3] === off[0]); return -1; }
        function drawT2(g, w) {
          const narrow = w < 480, rowH = narrow ? 58 : 46, lab = narrow ? 0 : 168, x0 = lab + 34, x1 = w - 46, top = 30;
          const X = (v) => x0 + (v + 2) / 22 * (x1 - x0);
          const act = activeRow();
          g.fillStyle = lib.rgba(C.ebt, 0.08); g.strokeStyle = C.ebt; g.lineWidth = 1.2; g.fillRect(0, 4, 14, 12); g.strokeRect(0.5, 4.5, 13, 11);
          T(g, 'thinking longer', 20, 3, { size: 12.5 });
          g.fillStyle = C.ebt; g.fillRect(140, 4, 14, 12); T(g, '+ self-verification', 160, 3, { size: 12.5 });
          [0, 5, 10, 15, 20].forEach((v) => { g.strokeStyle = v === 0 ? C.faint : lib.rgba(C.rule, 0.8); g.lineWidth = 1; g.beginPath(); g.moveTo(X(v), top); g.lineTo(X(v), top + 5 * rowH); g.stroke(); T(g, String(v), X(v), top + 5 * rowH + 6, { size: 12, kind: 'mono', align: 'center', color: C.faint }); });
          T2.forEach((r, i) => {
            const y = top + i * rowH, on = i === act;
            if (on) lib.panel(g, 0, y + 1, w, rowH - 2, { fill: lib.rgba(C.ebt, 0.08), stroke: lib.rgba(C.ebt, 0.5), r: 8, lw: 1 });
            const ly = narrow ? y + 6 : y + rowH / 2;
            T(g, r[0], narrow ? 6 : 8, ly, { size: 13, color: on ? C.ink : C.muted, baseline: narrow ? 'top' : 'middle', weight: i === 4 ? 700 : 400 });
            const by = narrow ? y + 26 : y + 9, bh = narrow ? 11 : 12;
            [[r[1], false], [r[2], true]].forEach(([v, solid], j) => {
              const yy = by + j * (bh + 3), xa = X(Math.min(0, v)), xb = X(Math.max(0, v)), bw = Math.max(1.5, xb - xa);
              if (solid) { g.fillStyle = on ? C.ebt : lib.rgba(C.ebt, 0.6); g.fillRect(xa, yy, bw, bh); }
              else { g.fillStyle = lib.rgba(C.ebt, 0.08); g.fillRect(xa, yy, bw, bh); g.strokeStyle = lib.rgba(C.ebt, on ? 1 : 0.65); g.lineWidth = 1.2; g.strokeRect(xa + 0.5, yy + 0.5, Math.max(1, bw - 1), bh - 1); }
              T(g, v.toFixed(v === 0 ? 2 : Math.abs(v) < 10 ? 2 : 1), (v >= 0 ? xb + 5 : xa - 5), yy + bh / 2, { size: 12, kind: 'mono', color: on ? C.ink : C.muted, baseline: 'middle', align: v >= 0 ? 'left' : 'right' });
            });
          });
        }
        function drawLand(g, w) {
          const side = w >= 560, s = side ? (w - 16) / 2 : w, z = sampled.z;
          const b1 = { x: 0, y: 24, w: s, h: s }, b2 = side ? { x: s + 16, y: 24, w: s, h: s } : { x: 0, y: s + 52, w: s, h: s };
          T(g, 'Whole plane', b1.x, b1.y - 20, { size: 13, color: C.ink });
          T(g, 'Zoom: basin floor (±' + RF + ' around the minimum)', b2.x, b2.y - 20, { size: 13, color: C.ink });
          // whole plane
          drawLandscape(g, z.grid, b1);
          const mn1 = P(z.min, b1), zr = [P([z.zext[0], z.zext[3]], b1), P([z.zext[1], z.zext[2]], b1)];
          g.save(); g.beginPath(); g.rect(b1.x, b1.y, b1.w, b1.h); g.clip();
          sampled.unrolls.forEach((u) => lib.line(g, u.path.map((q) => P(q, b1)), { color: lib.rgba(u.replay ? C.rnn : C.ebt, 0.55), width: 1.2 }));
          sampled.unrolls.forEach((u) => { for (let i = 0; i < u.N; i++) { const q = P(u.path[i], b1); lib.dot(g, q[0], q[1], 2.2, lib.rgba(C.ink, 0.9)); } });
          g.restore();
          g.save(); g.strokeStyle = C.ink; g.lineWidth = 1.3; g.setLineDash([4, 3]); g.strokeRect(zr[0][0], zr[0][1], zr[1][0] - zr[0][0], zr[1][1] - zr[0][1]); g.restore();
          const mu1 = P(M.contexts[S.ctx].mu, b1); marker(g, mu1[0], mu1[1], 'cross', C.truth);
          plate(g, 'step ' + STEPS[S.ck].toLocaleString() + ' · x = ' + M.contexts[S.ctx].x, b1.x + 8, b1.y + b1.h - 8);
          // zoom
          const Pz = (q) => M.toPx(q, b2, z.zext);
          drawLandscape(g, z.zgrid, b2, { levels: 12 });
          g.save(); g.beginPath(); g.rect(b2.x, b2.y, b2.w, b2.h); g.clip();
          const mz = Pz(z.min), rr = 0.25 / (2 * RF) * b2.w;
          g.save(); g.strokeStyle = lib.rgba(C.ink, 0.7); g.setLineDash([4, 4]); g.lineWidth = 1.2; g.beginPath(); g.arc(mz[0], mz[1], rr, 0, 7); g.stroke(); g.restore();
          const inZ = (q) => Math.abs(q[0] - z.min[0]) < 1.6 * RF && Math.abs(q[1] - z.min[1]) < 1.6 * RF;
          sampled.unrolls.forEach((u) => { for (let i = 1; i <= u.N; i++) { const a = u.path[i - 1], b = u.path[i]; lib.line(g, [Pz(a), Pz(b)], { color: lib.rgba(u.replay ? C.rnn : C.ebt, inZ(a) ? 0.85 : 0.18), width: 1.5 }); } });
          sampled.unrolls.forEach((u) => { for (let i = 0; i < u.N; i++) { const q = Pz(u.path[i]); lib.dot(g, q[0], q[1], 3, C.ink, { stroke: C.bg, lw: 0.8 }); } const e = Pz(u.path[u.N]); lib.dot(g, e[0], e[1], 3.6, u.replay ? C.rnn : C.ebt, { stroke: C.bg, lw: 1 }); });
          const muz = Pz(M.contexts[S.ctx].mu); marker(g, muz[0], muz[1], 'cross', C.truth);
          g.restore();
          plate(g, 'dashed ring: r = 0.25 · box ' + (2 * RF).toFixed(1) + ' wide', b2.x + 8, b2.y + b2.h - 8, { size: 12 });
        }
        function update() {
          sampleAll(); landC.render(); t2C.render();
          const z = sampled, act = activeRow();
          readout.innerHTML = '<span>N per batch <b>' + z.Ns.join(', ') + '</b></span><span>α range <b>' + f2(z.amin) + '–' + f2(z.amax) + '</b></span><span>replay starts <b>' + z.nrep + '/40</b></span><span>signal points <b>' + z.pts + '</b></span>' +
            '<span>on the floor (inside ring) <b>' + z.near + '</b> points</span><span>floor area touched <b>' + z.fine + '</b> cells of 0.05²</span><span>plane area touched <b>' + z.cells + '</b> cells of 0.25²</span>';
          t2note.innerHTML = act === 4 ? 'Your toggles match the <b>full System 2 configuration</b> (highlighted).' : act >= 0 ? 'Your toggles match the ablation <b>' + T2[act][0].toLowerCase() + '</b> (highlighted).' : 'The paper removed one regularizer at a time; your combination is not in Table 2.';
        }
        update(); // cheap: ~400 gradient evaluations + one 56×56 zoom grid per checkpoint/context
      })();
    },
  });
})();
