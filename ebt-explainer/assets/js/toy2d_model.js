/* EBT.toy2d: the toy 2D Energy-Based Model trained for this explainer (data/toy2d.json), runnable in the browser.
   E(x, ŷ) = MLP([phi(x), ŷ]) + λ‖ŷ‖², SiLU MLP 8→64→64→64→1, weights at 12 training checkpoints.
   API (all pure; ckpt = checkpoint index 0..n-1, x = context in [0,1), y = [y0, y1]):
     EBT.toy2d.ready                      -> bool (data present)
     EBT.toy2d.data                       -> the raw JSON
     EBT.toy2d.nCkpt, .steps              -> number of checkpoints, training step of each
     EBT.toy2d.contexts                   -> [{x, label, role, mu:[..], sigma}]
     EBT.toy2d.extent                     -> [xmin, xmax, ymin, ymax] of the prediction plane
     EBT.toy2d.energy(ckpt, x, y)         -> scalar energy
     EBT.toy2d.energyGrad(ckpt, x, y)     -> {E, g:[dE/dy0, dE/dy1]}
     EBT.toy2d.descend(ckpt, x, y0, {alpha, steps, sigma, seed}) -> {path:[[y0,y1],...], energies:[...]}  (Langevin noise if sigma > 0)
     EBT.toy2d.grid(ckpt, x, n, extent)   -> {E: rows TOP-DOWN (row 0 = ymax), lo, hi}  ready for lib.heatmap / lib.contours / lib.surface
     EBT.toy2d.storedGrid(ctxIdx, step)   -> stored 64×64 grid TOP-DOWN, or null (stored grids in JSON are bottom-up; this flips them)
     EBT.toy2d.toPx(y, box) / fromPx(px, py, box) -> map prediction coords <-> pixel coords for box {x, y, w, h} (y up)
   Video scenes can use the same object: include assets/js/toy2d_model.js is not needed there; scenes may copy these functions. */
(function () {
  const EBT = window.EBT = window.EBT || {};
  const D = (window.EBT_DATA || {}).toy2d;
  const M = { ready: !!D, data: D };
  EBT.toy2d = M;
  if (!D) return;
  const K = D.arch.feature_map.K || 3, LAM = (D.arch.quadratic_term && D.arch.quadratic_term.lambda_) || 0.02;
  M.nCkpt = D.weights.length; M.steps = D.weights.map(w => w.step); M.contexts = D.contexts; M.extent = D.extent;
  const sig = (z) => 1 / (1 + Math.exp(-z));
  const feats = (x) => { const f = []; for (let k = 1; k <= K; k++) { f.push(Math.sin(2 * Math.PI * k * x), Math.cos(2 * Math.PI * k * x)); } return f; };
  M.features = feats;
  // forward with cache for backprop
  function forward(layers, h0) {
    const hs = [h0], zs = []; let h = h0;
    for (let l = 0; l < layers.length; l++) {
      const W = layers[l].W, b = layers[l].b, out = new Array(W.length);
      for (let i = 0; i < W.length; i++) { let s = b[i]; const Wi = W[i]; for (let j = 0; j < Wi.length; j++) s += Wi[j] * h[j]; out[i] = s; }
      if (l < layers.length - 1) { zs.push(out); h = out.map(z => z * sig(z)); hs.push(h); } else return { out: out[0], hs, zs };
    }
  }
  M.energyGrad = function (ckpt, x, y) {
    const layers = D.weights[ckpt].layers; const h0 = feats(x).concat([y[0], y[1]]);
    const { out, hs, zs } = forward(layers, h0);
    // backprop d out / d h0
    let g = layers[layers.length - 1].W[0].slice();
    for (let l = layers.length - 2; l >= 0; l--) {
      const z = zs[l]; g = g.map((gi, i) => { const s = sig(z[i]); return gi * s * (1 + z[i] * (1 - s)); });
      const W = layers[l].W, nIn = W[0].length, gIn = new Array(nIn).fill(0);
      for (let i = 0; i < W.length; i++) { const gi = g[i]; if (gi === 0) continue; const Wi = W[i]; for (let j = 0; j < nIn; j++) gIn[j] += Wi[j] * gi; }
      g = gIn;
    }
    const n = h0.length;
    return { E: out + LAM * (y[0] * y[0] + y[1] * y[1]), g: [g[n - 2] + 2 * LAM * y[0], g[n - 1] + 2 * LAM * y[1]] };
  };
  M.energy = function (ckpt, x, y) {
    const layers = D.weights[ckpt].layers; const { out } = forward(layers, feats(x).concat([y[0], y[1]]));
    return out + LAM * (y[0] * y[0] + y[1] * y[1]);
  };
  function rng(seed) { let a = seed >>> 0; const f = () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; f.normal = () => { const u = Math.max(1e-9, f()), v = f(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }; return f; }
  M.descend = function (ckpt, x, y0, o = {}) {
    const alpha = o.alpha == null ? 1.0 : o.alpha, steps = o.steps == null ? 20 : o.steps, sigma = o.sigma || 0, r = rng(o.seed || 1);
    let y = [y0[0], y0[1]]; const path = [y.slice()], energies = [];
    for (let i = 0; i < steps; i++) {
      const { E, g } = M.energyGrad(ckpt, x, y); energies.push(E);
      y = [y[0] - alpha * g[0] + sigma * r.normal(), y[1] - alpha * g[1] + sigma * r.normal()];
      path.push(y.slice());
    }
    energies.push(M.energy(ckpt, x, y));
    return { path, energies };
  };
  M.grid = function (ckpt, x, n = 64, ext = D.extent) {
    const E = []; let lo = Infinity, hi = -Infinity;
    for (let r = 0; r < n; r++) { const yv = ext[3] - (ext[3] - ext[2]) * r / (n - 1); const row = [];
      for (let c = 0; c < n; c++) { const xv = ext[0] + (ext[1] - ext[0]) * c / (n - 1); const e = M.energy(ckpt, x, [xv, yv]); row.push(e); if (e < lo) lo = e; if (e > hi) hi = e; }
      E.push(row); }
    return { E, lo, hi };
  };
  M.storedGrid = function (ctxIdx, step) {
    const g = D.grids.find(k => k.ctx === ctxIdx && k.step === step); if (!g) return null;
    return g.E.slice().reverse();
  };
  M.toPx = (y, b, ext = D.extent) => [b.x + (y[0] - ext[0]) / (ext[1] - ext[0]) * b.w, b.y + b.h - (y[1] - ext[2]) / (ext[3] - ext[2]) * b.h];
  M.fromPx = (px, py, b, ext = D.extent) => [ext[0] + (px - b.x) / b.w * (ext[1] - ext[0]), ext[2] + (b.y + b.h - py) / b.h * (ext[3] - ext[2])];
})();
