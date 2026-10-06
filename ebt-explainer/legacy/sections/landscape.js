/* Chapter: Energy landscape lab (core lab).
   Everything numeric here is computed live from the toy EBT in data/toy2d.json through EBT.toy2d
   (energy, gradient, finite-difference Hessian, gradient descent with optional Langevin noise, Best-of-N),
   or read from the toy's exported results (Best-of-N aggregates, basin statistics, training-loss slice). */
EBT.section({
  id: 'landscape',
  nav: 'Energy landscape',
  kicker: 'Lab',
  title: 'Energy landscape lab: watch a prediction think',
  lede: 'An EBT gives one number, the <b>energy</b>, for a context x and a candidate prediction ŷ. Low energy means they fit. To predict, it starts from noise and walks downhill on that energy. Here ŷ is a 2D point, so you can see the whole landscape, drop guesses on it, and run the same update rule the paper uses.',
  mount(el, lib) {
    const h = lib.h, C = lib.C, T = window.EBT && window.EBT.toy2d;
    if (!T || !T.ready) { el.appendChild(lib.callout('warn', 'Data missing', 'data/toy2d.json is not in data/bundle.js. Run <code>python3 src/bundle_data.py</code>.')); return; }

    // ------------------------------------------------------------------ constants & small helpers
    const D = T.data, EXT = T.extent, NCK = T.nCkpt, STEPS = T.steps, LAST = NCK - 1, HP = D.hparams || {};
    const TAU = Math.PI * 2, MAXM = 16, GAMMA = 0.5;
    const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
    const muOf = (x) => [1.3 * Math.sin(TAU * x), 0.9 * Math.sin(2 * TAU * x)];            // dataset.mu_formula
    const sigOf = (x) => 0.03 + 0.27 * Math.pow(Math.sin(Math.PI * (x - 0.25)), 2);        // dataset.sigma_formula
    const fx = (v, d = 3) => !isFinite(v) ? '–' : (v < -0.5 * Math.pow(10, -d) ? '−' : '') + Math.abs(v).toFixed(d);
    const vec = (p, d = 3) => '(' + fx(p[0], d) + ', ' + fx(p[1], d) + ')';
    const sub = (n) => String(n).split('').map(c => '₀₁₂₃₄₅₆₇₈₉'[+c] || c).join('');
    const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
    const ckLabel = (i) => 'step ' + STEPS[i].toLocaleString('en-US') + (i === LAST ? ' (final)' : i === 0 ? ' (untrained)' : '');
    const nParams = D.weights[0].layers.reduce((s, L) => s + L.W.length * L.W[0].length + L.b.length, 0);
    // sanity: our closed-form mu/sigma must match the exported contexts
    D.contexts.forEach(c => { const m = muOf(c.x); if (dist(m, c.mu) > 1e-3 || Math.abs(sigOf(c.x) - c.sigma) > 1e-3) console.warn('landscape: mu/sigma formula mismatch at x =', c.x); });
    const CTX_NAMES = ['Right tip', 'Left tip', 'Crossing ↗', 'Crossing ↙', 'Upper right', 'Lower right', 'Upper left', 'Lower left'];

    function niceTicks(lo, hi, n = 4) {
      const span = hi - lo; if (!(span > 0)) return [lo];
      const s0 = span / n, mag = Math.pow(10, Math.floor(Math.log10(s0))), e = s0 / mag;
      const st = (e >= 7.5 ? 10 : e >= 3.5 ? 5 : e >= 1.5 ? 2 : 1) * mag, out = [];
      for (let v = Math.ceil(lo / st - 1e-9) * st; v <= hi + 1e-9; v += st) out.push(+v.toFixed(10));
      return out;
    }
    function rgbScale(col, s) { return 'rgb(' + Math.round(col[0] * s) + ',' + Math.round(col[1] * s) + ',' + Math.round(col[2] * s) + ')'; }

    // responsive canvas: logical size = CSS size, so canvas text keeps its pixel size on phones
    const canvases = [];
    function rcanvas(parent, { height, label, cls, onfit }) {
      const box = h('div', { class: 'landscape-cv' + (cls ? ' ' + cls : '') });
      const c = h('canvas', { role: 'img', 'aria-label': label || '' });
      box.appendChild(c); parent.appendChild(box);
      const o = { canvas: c, box, ctx: c.getContext('2d'), w: 0, h: 0, dpr: 1 };
      o.fit = function () {
        const w = Math.max(220, Math.floor(box.clientWidth || 0)); if (!box.clientWidth) return false;
        const hh = Math.round(height(w)), dpr = Math.min(2, window.devicePixelRatio || 1);
        if (w === o.w && hh === o.h && dpr === o.dpr) return false;
        o.w = w; o.h = hh; o.dpr = dpr; c.width = Math.round(w * dpr); c.height = Math.round(hh * dpr); c.style.height = hh + 'px';
        onfit && onfit(); return true;
      };
      o.begin = function () { o.ctx.setTransform(o.dpr, 0, 0, o.dpr, 0, 0); o.ctx.clearRect(0, 0, o.w, o.h); return o.ctx; };
      o.toLocal = (ev) => { const r = c.getBoundingClientRect(); return [(ev.clientX - r.left) / r.width * o.w, (ev.clientY - r.top) / r.height * o.h]; };
      canvases.push(o); return o;
    }
    function txt(ctx, s, x, y, o = {}) {
      ctx.save(); ctx.font = (o.weight || 400) + ' ' + (o.size || 12) + 'px ' + (o.mono === false ? lib.F.body : lib.F.mono);
      ctx.fillStyle = o.color || C.muted; ctx.textAlign = o.align || 'left'; ctx.textBaseline = o.base || 'alphabetic';
      if (o.halo) { ctx.lineWidth = 3; ctx.strokeStyle = o.halo; ctx.lineJoin = 'round'; ctx.strokeText(s, x, y); }
      ctx.fillText(s, x, y); const w = ctx.measureText(s).width; ctx.restore(); return w;
    }
    // simple axes with 1px rules
    function axes(ctx, o) {
      const tx = (v) => o.xlog ? Math.log10(v) : v, ty = (v) => o.ylog ? Math.log10(v) : v;
      const [xa, xb] = o.xlim.map(tx), [ya, yb] = o.ylim.map(ty);
      const X = (v) => o.x + (tx(v) - xa) / (xb - xa) * o.w, Y = (v) => o.y + o.h - (ty(v) - ya) / (yb - ya) * o.h;
      ctx.save(); ctx.lineWidth = 1; ctx.strokeStyle = C.rule;
      (o.yticks || []).forEach(v => { const yy = Math.round(Y(v)) + 0.5; ctx.beginPath(); ctx.moveTo(o.x, yy); ctx.lineTo(o.x + o.w, yy); ctx.stroke(); txt(ctx, (o.yfmt || String)(v), o.x - 6, yy, { align: 'right', base: 'middle', size: 11.5 }); });
      (o.xticks || []).forEach(v => { const xx = Math.round(X(v)) + 0.5; ctx.beginPath(); ctx.moveTo(xx, o.y); ctx.lineTo(xx, o.y + o.h); ctx.stroke(); txt(ctx, (o.xfmt || String)(v), xx, o.y + o.h + 15, { align: 'center', size: 11.5 }); });
      ctx.strokeStyle = C.faint; ctx.beginPath(); ctx.moveTo(o.x + 0.5, o.y); ctx.lineTo(o.x + 0.5, o.y + o.h + 0.5); ctx.lineTo(o.x + o.w, o.y + o.h + 0.5); ctx.stroke();
      ctx.restore(); return { X, Y };
    }
    function poly(ctx, pts, color, width, o = {}) {
      if (pts.length < 2) return; ctx.save(); ctx.strokeStyle = color; ctx.lineWidth = width; ctx.lineJoin = 'round'; ctx.lineCap = 'round'; ctx.globalAlpha = o.alpha == null ? 1 : o.alpha; if (o.dash) ctx.setLineDash(o.dash);
      ctx.beginPath(); ctx.moveTo(pts[0][0], pts[0][1]); for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]); ctx.stroke(); ctx.restore();
    }
    function circle(ctx, x, y, r, fill, stroke, lw = 1.5) { ctx.beginPath(); ctx.arc(x, y, r, 0, TAU); if (fill) { ctx.fillStyle = fill; ctx.fill(); } if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = lw; ctx.stroke(); } }
    function arrowPx(ctx, x1, y1, x2, y2, color, w = 1.6) {
      const L = Math.hypot(x2 - x1, y2 - y1); if (L < 2) return; const a = Math.atan2(y2 - y1, x2 - x1), hs = Math.min(8, 3 + L * 0.25);
      ctx.save(); ctx.strokeStyle = ctx.fillStyle = color; ctx.lineWidth = w; ctx.lineCap = 'round';
      ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2 - hs * 0.6 * Math.cos(a), y2 - hs * 0.6 * Math.sin(a)); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(x2, y2); ctx.lineTo(x2 - hs * Math.cos(a - 0.42), y2 - hs * Math.sin(a - 0.42)); ctx.lineTo(x2 - hs * Math.cos(a + 0.42), y2 - hs * Math.sin(a + 0.42)); ctx.closePath(); ctx.fill(); ctx.restore();
    }
    function xMark(ctx, x, y, r, color, lw = 2.2) { ctx.save(); ctx.strokeStyle = 'rgba(13,19,31,0.9)'; ctx.lineWidth = lw + 2.5; ctx.lineCap = 'round'; ctx.beginPath(); ctx.moveTo(x - r, y - r); ctx.lineTo(x + r, y + r); ctx.moveTo(x + r, y - r); ctx.lineTo(x - r, y + r); ctx.stroke(); ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.stroke(); ctx.restore(); }

    // ------------------------------------------------------------------ energy grids (cached; never per frame)
    const gridCache = new Map();
    function colorImage(E, lo, hi, tf) { // tf maps value -> [0,1]
      const n = E.length, m = E[0].length, cv = document.createElement('canvas'); cv.width = m; cv.height = n;
      const g = cv.getContext('2d'), im = g.createImageData(m, n);
      for (let r = 0; r < n; r++) for (let c = 0; c < m; c++) { const col = lib.cmap(tf(E[r][c])); const i = 4 * (r * m + c); im.data[i] = col[0]; im.data[i + 1] = col[1]; im.data[i + 2] = col[2]; im.data[i + 3] = 255; }
      g.putImageData(im, 0, 0); return cv;
    }
    function makeField(E, lo, hi, key, tf, levels) { return { E, lo, hi, n: E.length, key, tf, levels, img: colorImage(E, lo, hi, tf), cache: null }; }
    function energyField(ck, x, coarse) {
      const key = ck + '|' + x.toFixed(4) + (coarse ? '|c' : '');
      if (gridCache.has(key)) return gridCache.get(key);
      let E = null, src = 'live';
      const ci = D.contexts.findIndex(c => Math.abs(c.x - x) < 1e-6);
      if (ci >= 0 && !coarse) { E = T.storedGrid(ci, STEPS[ck]); if (E) src = 'stored'; }
      let lo = Infinity, hi = -Infinity;
      if (E) { for (const r of E) for (const v of r) { if (v < lo) lo = v; if (v > hi) hi = v; } }
      else { const g = T.grid(ck, x, coarse ? 32 : 64, EXT); E = g.E; lo = g.lo; hi = g.hi; }
      const tf = (v) => Math.pow(clamp((v - lo) / (hi - lo || 1), 0, 1), GAMMA);
      const levels = []; for (let k = 1; k < 14; k++) levels.push(lo + (hi - lo) * Math.pow(k / 14, 1 / GAMMA));
      const F = makeField(E, lo, hi, key, tf, levels); F.src = src;
      if (gridCache.size > 48) gridCache.delete(gridCache.keys().next().value);
      gridCache.set(key, F); return F;
    }
    // heat + contours of a field into a w×h offscreen canvas (cached per size)
    function fieldBitmap(F, w, hh, dpr) {
      const k = w + 'x' + hh + '@' + dpr; if (F.cache && F.cache.k === k) return F.cache.cv;
      const cv = document.createElement('canvas'); cv.width = Math.round(w * dpr); cv.height = Math.round(hh * dpr);
      const g = cv.getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0);
      const n = F.n, cw = w / (n - 1), ch = hh / (n - 1);
      g.imageSmoothingEnabled = true; g.imageSmoothingQuality = 'high';
      g.drawImage(F.img, -cw / 2, -ch / 2, w + cw, hh + ch); // pixel centres on grid nodes
      lib.contours(g, F.E, 0, 0, w, hh, F.levels, { color: 'rgba(233,238,246,0.2)', width: 1 });
      F.cache = { k, cv }; return cv;
    }
    // shared 3D height range over every stored grid, so training growth is visible
    let ZLO = Infinity, ZHI = -Infinity;
    D.grids.forEach(g => g.E.forEach(r => r.forEach(v => { if (v < ZLO) ZLO = v; if (v > ZHI) ZHI = v; })));

    // 3D surface (same projection as lib.surface, but coloured by the heatmap's colour scale and lit, so the basin reads)
    const LV = (() => { const l = [-0.45, -0.55, 0.7], n = Math.hypot(...l); return l.map(v => v / n); })();
    function surface(ctx, F, o) {
      const E = F.E, n = E.length, st = o.step || 1;
      const cyw = Math.cos(o.yaw), syw = Math.sin(o.yaw), cp = Math.cos(o.pitch), sp = Math.sin(o.pitch);
      const zn = (z) => (clamp(z, o.zlo, o.zhi) - o.zlo) / (o.zhi - o.zlo || 1) * o.zs;
      const project = (u, v, z) => { const X0 = (u - 0.5) * o.size, Y0 = (v - 0.5) * o.size, Z = zn(z); const xr = X0 * cyw - Y0 * syw, yr = X0 * syw + Y0 * cyw; return [o.cx + xr, o.cy + yr * cp - Z * sp, yr * sp + Z * cp]; };
      // floor outline + corner post
      const fl = [[0, 0], [1, 0], [1, 1], [0, 1]].map(([u, v]) => project(u, v, o.zlo));
      ctx.save(); ctx.strokeStyle = C.rule; ctx.lineWidth = 1; ctx.beginPath(); fl.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])); ctx.closePath(); ctx.stroke(); ctx.restore();
      const quads = [], cell = st / (n - 1) * o.size;
      for (let r = 0; r + st < n; r += st) for (let c = 0; c + st < n; c += st) {
        const z00 = E[r][c], z01 = E[r][c + st], z11 = E[r + st][c + st], z10 = E[r + st][c];
        const u0 = c / (n - 1), u1 = (c + st) / (n - 1), v0 = r / (n - 1), v1 = (r + st) / (n - 1);
        const p = [project(u0, v0, z00), project(u1, v0, z01), project(u1, v1, z11), project(u0, v1, z10)];
        const du = ((zn(z01) - zn(z00)) + (zn(z11) - zn(z10))) / (2 * cell), dv = ((zn(z10) - zn(z00)) + (zn(z11) - zn(z01))) / (2 * cell);
        const nl = Math.hypot(du, dv, 1), lam = (-du * LV[0] - dv * LV[1] + LV[2]) / nl;
        quads.push({ p, d: p[0][2] + p[1][2] + p[2][2] + p[3][2], z: (z00 + z01 + z11 + z10) / 4, s: 0.55 + 0.45 * clamp(lam, 0, 1) });
      }
      quads.sort((a, b) => a.d - b.d);
      ctx.save(); ctx.lineJoin = 'round'; ctx.lineWidth = 0.7;
      for (const q of quads) {
        const col = rgbScale(lib.cmap(F.tf(q.z)), q.s); ctx.fillStyle = col; ctx.strokeStyle = col;
        ctx.beginPath(); ctx.moveTo(q.p[0][0], q.p[0][1]); for (let k = 1; k < 4; k++) ctx.lineTo(q.p[k][0], q.p[k][1]); ctx.closePath(); ctx.fill(); ctx.stroke();
      }
      ctx.restore();
      return project;
    }
    function surfaceLayout(w, hh) { const size = Math.min(w * 0.6, hh * 0.68); return { cx: w / 2, cy: hh * 0.6, size, zs: hh * 0.36 }; }
    function surfaceBitmap(cache, F, w, hh, dpr, view, zlo, zhi, step) {
      const k = [F.key, w, hh, dpr, view.yaw.toFixed(3), view.pitch.toFixed(3), step].join('|');
      if (cache.k === k) return cache;
      const cv = cache.cv || document.createElement('canvas'); cv.width = Math.round(w * dpr); cv.height = Math.round(hh * dpr);
      const g = cv.getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, w, hh);
      const L = surfaceLayout(w, hh);
      const project = surface(g, F, Object.assign({}, L, { yaw: view.yaw, pitch: view.pitch, zlo, zhi, step }));
      return Object.assign(cache, { k, cv, project });
    }
    function attachRotate(cvo, view, onchange) {
      let drag = null;
      const cv = cvo.canvas; cv.tabIndex = 0;
      cv.addEventListener('pointerdown', (e) => { drag = { x: e.clientX, y: e.clientY, yaw: view.yaw, pitch: view.pitch }; view.dragging = true; try { cv.setPointerCapture(e.pointerId); } catch (_) {} });
      cv.addEventListener('pointermove', (e) => { if (!drag) return; view.yaw = drag.yaw + (e.clientX - drag.x) * 0.012; view.pitch = clamp(drag.pitch - (e.clientY - drag.y) * 0.008, 0.15, 1.45); onchange(); });
      const end = () => { if (!drag) return; drag = null; view.dragging = false; onchange(); };
      cv.addEventListener('pointerup', end); cv.addEventListener('pointercancel', end);
      cv.addEventListener('keydown', (e) => {
        const d = { ArrowLeft: [-0.12, 0], ArrowRight: [0.12, 0], ArrowUp: [0, -0.08], ArrowDown: [0, 0.08] }[e.key]; if (!d) return;
        e.preventDefault(); view.yaw += d[0]; view.pitch = clamp(view.pitch + d[1], 0.15, 1.45); onchange();
      });
    }

    // ------------------------------------------------------------------ state
    const S = { x: 0.75, ck: LAST, alpha: 1.0, N: 4, sigma: 0, M: 6, randAlpha: true, truth: true, seed: 5, user: null };
    let runs = [], k = S.N, frac = 0, playing = false, probe = null, F = null, basin = null, coarse = false;
    const view3 = { yaw: -0.62, pitch: 0.92, dragging: false };

    function hessEig(ck, x, y) {
      const e = 1e-3, g = (p) => T.energyGrad(ck, x, p).g;
      const gx1 = g([y[0] + e, y[1]]), gx0 = g([y[0] - e, y[1]]), gy1 = g([y[0], y[1] + e]), gy0 = g([y[0], y[1] - e]);
      const a = (gx1[0] - gx0[0]) / (2 * e), d = (gy1[1] - gy0[1]) / (2 * e), b = ((gx1[1] - gx0[1]) + (gy1[0] - gy0[0])) / (4 * e);
      const tr = (a + d) / 2, disc = Math.sqrt(Math.max(0, tr * tr - (a * d - b * b)));
      return [tr - disc, tr + disc];
    }
    // the basin bottom for this verifier and context: start at the grid argmin, refine with a curvature-scaled GD
    const basinCache = new Map();
    function basinOf(ck, x, Fd) {
      const key = ck + '|' + x.toFixed(4); if (basinCache.has(key)) return basinCache.get(key);
      let best = Infinity, y = [0, 0]; const n = Fd.n;
      for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) if (Fd.E[r][c] < best) { best = Fd.E[r][c]; y = [EXT[0] + (EXT[1] - EXT[0]) * c / (n - 1), EXT[3] - (EXT[3] - EXT[2]) * r / (n - 1)]; }
      let lam = hessEig(ck, x, y); const a = 0.8 / Math.max(0.03, lam[1]);
      for (let i = 0; i < 40; i++) { const g = T.energyGrad(ck, x, y).g; y = [clamp(y[0] - a * g[0], EXT[0], EXT[1]), clamp(y[1] - a * g[1], EXT[2], EXT[3])]; }
      lam = hessEig(ck, x, y);
      const B = { y, E: T.energy(ck, x, y), lam, d: dist(y, muOf(x)) }; basinCache.set(key, B); return B;
    }
    // the starts and per-candidate randomness are tied to the seed, so changing M or N keeps earlier candidates identical
    function draws(seed) {
      const r = lib.rng(seed * 7919 + 17), starts = [], af = [];
      for (let j = 0; j < MAXM; j++) { starts.push([r.normal(), r.normal()]); af.push(Math.exp((2 * r() - 1) * Math.log(HP.alpha_rand_factor || 2))); }
      return { starts, af };
    }
    function runOne(y0, a, seed) {
      const rn = lib.rng(seed); let y = y0.slice(); const ys = [y.slice()], Es = [], gs = [], etas = [];
      for (let i = 0; i <= S.N; i++) {
        const eg = T.energyGrad(S.ck, S.x, y); Es.push(eg.E); gs.push(eg.g);
        if (i === S.N) break;
        const e0 = rn.normal() * S.sigma, e1 = rn.normal() * S.sigma; etas.push([e0, e1]);
        y = [clamp(y[0] - a * eg.g[0] + e0, -30, 30), clamp(y[1] - a * eg.g[1] + e1, -30, 30)]; ys.push(y.slice());
      }
      return { y0: y0.slice(), alpha: a, ys, Es, gs, etas };
    }
    function simulate(onlyUser) {
      const dr = draws(S.seed);
      const mk = (j) => runOne(j === 0 && S.user ? S.user : dr.starts[j], S.randAlpha ? S.alpha * dr.af[j] : S.alpha, S.seed * 1000 + j + 1);
      if (onlyUser && runs.length === S.M) runs[0] = mk(0);
      else { runs = []; for (let j = 0; j < S.M; j++) runs.push(mk(j)); }
    }
    function refreshField() {
      F = energyField(S.ck, S.x, coarse);
      basin = coarse ? null : basinOf(S.ck, S.x, F);
    }
    const chosenAt = (kk) => { let b = 0; runs.forEach((r, j) => { if (r.Es[kk] < runs[b].Es[kk]) b = j; }); return b; };
    const closestAt = (kk) => { const m = muOf(S.x); let b = 0; runs.forEach((r, j) => { if (dist(r.ys[kk], m) < dist(runs[b].ys[kk], m)) b = j; }); return b; };
    const posAt = (r, kk, f) => { kk = clamp(kk, 0, S.N); if (kk >= S.N || !(f > 0)) return r.ys[Math.min(kk, S.N)]; const e = lib.ease(f), a = r.ys[kk], b = r.ys[kk + 1]; return [a[0] + (b[0] - a[0]) * e, a[1] + (b[1] - a[1]) * e]; };

    // ------------------------------------------------------------------ DOM: intro
    el.appendChild(h('div', { class: 'prose' },
      h('p', { html: 'The energy E<sub>θ</sub>(x, ŷ) is a learned <b>verifier</b>. Thinking is gradient descent on the prediction, with the weights θ frozen: <span class="landscape-eq">ŷ<sub>i+1</sub> = ŷ<sub>i</sub> − α ∇<sub>ŷ</sub>E<sub>θ</sub>(x, ŷ<sub>i</sub>)</span> (Eq. 1, p.7). Self-verification runs several guesses and keeps the one with the lowest energy (Algorithm 2, p.7).' }),
      h('p', { html: 'The landscape below is real. It comes from a tiny EBT (an 8-64-64-64-1 MLP, ' + nParams.toLocaleString('en-US') + ' weights) that we trained for this page with the paper\'s recipe: random start, 2 to 6 unrolled descent steps, squared error at the last step, backpropagation through the whole unroll, plus Langevin noise, random step sizes and a replay buffer (Algorithm 1 and Sec 3.3). Its context x picks a spot on a figure-eight. The targets scatter around that spot with a noise level σ(x) that changes along the curve.' })));

    const H3 = (n, t, badge) => h('h3', { class: 'landscape-h3' }, h('span', { class: 'n' }, n), t, badge || null);

    // ------------------------------------------------------------------ DOM: lab controls
    el.appendChild(H3('1', 'Drop a guess and watch it think'));
    const bench = h('div', { class: 'bench landscape-bench' }); el.appendChild(bench);
    const ctl = h('div', { class: 'card landscape-ctl' }); bench.appendChild(ctl);
    const inst = h('div', { class: 'card landscape-inst' }); bench.appendChild(inst);

    ctl.appendChild(h('h4', {}, 'Context x'));
    const chips = h('div', { class: 'landscape-chips', role: 'group', 'aria-label': 'Example contexts' }); ctl.appendChild(chips);
    const chipBtns = D.contexts.map((c, i) => {
      const b = h('button', { type: 'button', class: 'landscape-chip', 'aria-pressed': 'false' }, h('b', {}, CTX_NAMES[i] || c.label), h('span', {}, 'x ' + c.x.toFixed(3).replace(/0+$/, '').replace(/\.$/, '') + ' · σ ' + c.sigma.toFixed(2)));
      b.addEventListener('click', () => { setX(c.x); }); chips.appendChild(b); return b;
    });
    const xS = lib.slider({ id: 'ls-x', label: 'Fine-tune x (any value works)', min: 0, max: 0.995, step: 0.005, value: S.x, fmt: (v) => v.toFixed(3), oninput: (v) => setX(v, true) });
    ctl.appendChild(xS.el);
    const ctxInfo = h('p', { class: 'landscape-hint' }); ctl.appendChild(ctxInfo);

    ctl.appendChild(h('h4', {}, 'Verifier (training checkpoint)'));
    const ckS = lib.slider({ id: 'ls-ck', label: 'Weights θ from', min: 0, max: LAST, step: 1, value: S.ck, fmt: (i) => ckLabel(i), oninput: (v) => { S.ck = v; changed(true); } });
    ctl.appendChild(ckS.el);

    ctl.appendChild(h('h4', {}, 'Thinking'));
    const sliders = h('div', { class: 'landscape-sliders' }); ctl.appendChild(sliders);
    const aS = lib.slider({ id: 'ls-a', label: 'Step size α', min: 0.05, max: 3, step: 0.05, value: S.alpha, fmt: (v) => v.toFixed(2), oninput: (v) => { S.alpha = v; changed(); } });
    const nS = lib.slider({ id: 'ls-n', label: 'Steps N', min: 1, max: 40, step: 1, value: S.N, oninput: (v) => { S.N = v; changed(); } });
    const sS = lib.slider({ id: 'ls-s', label: 'Langevin noise σ', min: 0, max: 0.3, step: 0.01, value: S.sigma, fmt: (v) => v.toFixed(2), oninput: (v) => { S.sigma = v; changed(); } });
    const mS = lib.slider({ id: 'ls-m', label: 'Candidates M', min: 1, max: MAXM, step: 1, value: S.M, oninput: (v) => { S.M = v; changed(); } });
    [aS, nS, sS, mS].forEach(s => sliders.appendChild(s.el));
    const raBox = h('input', { type: 'checkbox', id: 'ls-ra' }); raBox.checked = S.randAlpha;
    raBox.addEventListener('change', () => { S.randAlpha = raBox.checked; changed(); });
    const trBox = h('input', { type: 'checkbox', id: 'ls-tr' }); trBox.checked = S.truth;
    trBox.addEventListener('change', () => { S.truth = trBox.checked; render(); });
    ctl.appendChild(h('label', { class: 'landscape-check', for: 'ls-ra' }, raBox, h('span', { html: 'Random α per candidate (×½ to ×2, as in training)' })));
    ctl.appendChild(h('label', { class: 'landscape-check', for: 'ls-tr' }, trBox, h('span', {}, 'Show the truth μ(x) and nearby targets')));
    ctl.appendChild(h('p', { class: 'landscape-hint', html: 'α: how far one step moves. N: how long it thinks. σ: random kick added each step (Eq. 2). M: how many guesses the verifier compares.' }));

    ctl.appendChild(h('h4', {}, 'Try this'));
    const presets = h('div', { class: 'landscape-presets' }); ctl.appendChild(presets);
    const PRESETS = [
      ['One step, 8 guesses', 'With one step, the guesses land at different distances. The energy alone picks the closest.', { x: 0.25, ck: LAST, alpha: 1, N: 1, sigma: 0, M: 8, randAlpha: false }],
      ['Step too large', 'α = 2.6 overshoots: each step jumps past the bottom and the energy bounces instead of settling.', { x: 0.25, ck: LAST, alpha: 2.6, N: 14, sigma: 0, M: 3, randAlpha: false }],
      ['Tiny steps', 'α = 0.15 is stable but slow. Each extra step still helps: this is thinking longer.', { x: 0.125, ck: LAST, alpha: 0.15, N: 30, sigma: 0, M: 4, randAlpha: false }],
      ['Untrained verifier', 'At training step 25 the bowl sits in the wrong place. Descent still converges, confidently wrong.', { x: 0.25, ck: 2, alpha: 1, N: 12, sigma: 0, M: 6, randAlpha: false }],
      ['Noisy guesses, best of 16', 'Langevin noise spreads 16 guesses around the bottom. The lowest energy one is usually among the closest.', { x: 0.75, ck: LAST, alpha: 1, N: 6, sigma: 0.15, M: 16, randAlpha: true }],
    ];
    PRESETS.forEach(([name, note, cfg]) => {
      const b = h('button', { type: 'button', class: 'btn landscape-preset', title: note }, name);
      b.addEventListener('click', () => applyPreset(cfg, name)); presets.appendChild(b);
    });

    // ------------------------------------------------------------------ DOM: instrument
    const tbar = h('div', { class: 'landscape-tbar' }); inst.appendChild(tbar);
    const bStep = h('button', { type: 'button', class: 'btn' }, 'Step');
    const bPlay = h('button', { type: 'button', class: 'btn primary' }, 'Play');
    const bReset = h('button', { type: 'button', class: 'btn', title: 'Draw new random starts ŷ₀ ~ N(0, I)' }, 'New guesses');
    const status = h('span', { class: 'landscape-status', 'aria-live': 'polite' });
    tbar.append(h('div', { class: 'row', style: 'gap:6px' }, bStep, bPlay, bReset), status);
    inst.appendChild(h('div', { class: 'row landscape-badges' }, lib.badge('toy', 'live model'), h('span', { class: 'landscape-src' })));
    const srcSpan = inst.querySelector('.landscape-src');

    const pair = h('div', { class: 'landscape-pair' }); inst.appendChild(pair);
    const figMap = h('figure', { class: 'landscape-fig' }); pair.appendChild(figMap);
    const figSurf = h('figure', { class: 'landscape-fig' }); pair.appendChild(figSurf);
    const mapCv = rcanvas(figMap, { height: (w) => w - 2, cls: 'drag', label: 'Energy landscape heatmap over the 2D prediction. Click or drag to place your own starting guess; arrow keys move it.' });
    mapCv.canvas.tabIndex = 0;
    const cbarLo = h('span'), cbarHi = h('span');
    figMap.appendChild(h('figcaption', {},
      h('div', { class: 'landscape-cbar' }, cbarLo, h('div', { class: 'cbar' }), cbarHi),
      h('div', { class: 'legend' },
        h('span', {}, h('i', { style: 'background:var(--ebt)' }), 'lowest energy now'),
        h('span', {}, h('i', { style: 'background:rgba(233,238,246,.6)' }), 'other guesses'),
        h('span', {}, h('i', { style: 'background:var(--truth)' }), 'truth μ(x), 2σ ring'),
        h('span', {}, h('i', { class: 'ring' }), 'your start')),
      h('div', { class: 'landscape-cap' }, 'Top view. Click to drop your own start ŷ₀, drag to move it. Hover to probe the energy and the step one update would take.')));
    const surfCv = rcanvas(figSurf, { height: (w) => w - 2, cls: 'rot', label: 'The same energy landscape as a 3D surface. Drag or use arrow keys to rotate.' });
    figSurf.appendChild(h('figcaption', {}, h('div', { class: 'landscape-cap', html: 'Same landscape in 3D, height = energy. <b>Drag to rotate</b> (arrow keys work too). Height uses one scale for every checkpoint, so an untrained model looks flat.' })));

    const plots = h('div', { class: 'landscape-plots' }); inst.appendChild(plots);
    const ePlot = rcanvas(plots, { height: (w) => clamp(w * 0.46, 150, 190), label: 'Energy of every candidate at each thinking step' });
    const dPlot = rcanvas(plots, { height: (w) => clamp(w * 0.46, 150, 190), label: 'Distance of every candidate to the truth at each step, log scale' });
    const readout = h('div', { class: 'readout landscape-readout', 'aria-live': 'polite' }); inst.appendChild(readout);
    const lower = h('div', { class: 'landscape-lower' }); inst.appendChild(lower);
    const eqCard = h('div', { class: 'landscape-eqcard' }); lower.appendChild(eqCard);
    const notes = h('div', { class: 'callout insight landscape-notes' }); lower.appendChild(notes);

    // ------------------------------------------------------------------ interactions
    function setX(x, fromSlider) {
      S.x = clamp(x, 0, 0.995); if (!fromSlider) xS.set(S.x);
      chipBtns.forEach((b, i) => b.setAttribute('aria-pressed', String(Math.abs(D.contexts[i].x - S.x) < 1e-6)));
      if (fromSlider) { coarse = !D.contexts.some(c => Math.abs(c.x - S.x) < 1e-6); clearTimeout(setX.t); setX.t = setTimeout(() => { coarse = false; changed(true); }, 180); }
      else coarse = false;
      changed(true);
    }
    function changed(fieldToo) {
      stop(); activePreset = null;
      if (fieldToo) refreshField();
      simulate(); k = S.N; frac = 0; render(); measureStale();
    }
    function applyPreset(cfg, name) {
      Object.assign(S, cfg); S.user = null;
      xS.set(S.x); ckS.set(S.ck); aS.set(S.alpha); nS.set(S.N); sS.set(S.sigma); mS.set(S.M); raBox.checked = S.randAlpha;
      chipBtns.forEach((b, i) => b.setAttribute('aria-pressed', String(Math.abs(D.contexts[i].x - S.x) < 1e-6)));
      coarse = false; refreshField(); simulate(); k = 0; frac = 0; measureStale(); activePreset = name;
      if (lib.reducedMotion) { k = S.N; render(); } else play();
    }
    let activePreset = null;
    const loop = lib.loop((dt) => {
      if (!playing) return false;
      const dur = clamp(3.6 / S.N, 0.14, 0.75);
      frac += Math.max(0, dt) / dur;
      while (frac >= 1) { frac -= 1; k++; if (k >= S.N) { k = S.N; frac = 0; playing = false; bPlay.textContent = 'Play'; } }
      render(); return playing;
    });
    function play() { if (k >= S.N) { k = 0; frac = 0; } playing = true; loop.start(); bPlay.textContent = 'Pause'; render(); }
    function stop() { playing = false; loop.stop(); frac = 0; bPlay.textContent = 'Play'; }
    bPlay.addEventListener('click', () => { if (playing) { stop(); render(); } else play(); });
    bStep.addEventListener('click', () => { stop(); if (k >= S.N) k = 0; else k++; render(); });
    bReset.addEventListener('click', () => { stop(); activePreset = null; S.seed++; S.user = null; simulate(); k = 0; render(); measureStale(); });

    // map pointer: click = drop a start and play; drag = move it with the full path shown live; hover = probe
    const mapBox = () => { const m = { l: 34, r: 8, t: 8, b: 26 }; return { x: m.l, y: m.t, w: mapCv.w - m.l - m.r, h: mapCv.h - m.t - m.b }; };
    let mdrag = null;
    const inBox = (p, b) => p[0] >= b.x && p[0] <= b.x + b.w && p[1] >= b.y && p[1] <= b.y + b.h;
    function placeUser(p) { activePreset = null; const b = mapBox(); const y = T.fromPx(clamp(p[0], b.x, b.x + b.w), clamp(p[1], b.y, b.y + b.h), b); S.user = y; simulate(true); }
    mapCv.canvas.addEventListener('pointerdown', (e) => {
      const p = mapCv.toLocal(e); if (!inBox(p, mapBox())) return;
      e.preventDefault(); try { mapCv.canvas.setPointerCapture(e.pointerId); } catch (_) {}
      mdrag = { x: e.clientX, y: e.clientY, moved: false }; stop(); placeUser(p); probe = null;
      k = 0; frac = 0; render();
    });
    mapCv.canvas.addEventListener('pointermove', (e) => {
      const p = mapCv.toLocal(e), b = mapBox();
      if (mdrag) {
        if (Math.hypot(e.clientX - mdrag.x, e.clientY - mdrag.y) > 3) mdrag.moved = true;
        if (mdrag.moved) { placeUser(p); k = S.N; frac = 0; render(); }
        return;
      }
      if (e.pointerType === 'mouse' && inBox(p, b)) { probe = T.fromPx(p[0], p[1], b); if (!playing) render(); } else if (probe) { probe = null; render(); }
    });
    const mapUp = () => { if (!mdrag) return; const moved = mdrag.moved; mdrag = null; measureStale(); if (!moved && !lib.reducedMotion) play(); else { k = S.N; render(); } };
    mapCv.canvas.addEventListener('pointerup', mapUp); mapCv.canvas.addEventListener('pointercancel', mapUp);
    mapCv.canvas.addEventListener('pointerleave', () => { if (probe) { probe = null; render(); } });
    mapCv.canvas.addEventListener('keydown', (e) => {
      const d = { ArrowLeft: [-0.1, 0], ArrowRight: [0.1, 0], ArrowUp: [0, 0.1], ArrowDown: [0, -0.1] }[e.key];
      if (d) { e.preventDefault(); stop(); const u = S.user || runs[0].y0; S.user = [clamp(u[0] + d[0], EXT[0], EXT[1]), clamp(u[1] + d[1], EXT[2], EXT[3])]; simulate(true); k = S.N; render(); }
      else if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); playing ? stop() : play(); }
    });
    const surfCache = {};
    attachRotate(surfCv, view3, () => render());

    // ------------------------------------------------------------------ rendering: map
    const manifold = (D.dataset && D.dataset.curve && D.dataset.curve.mu) || [];
    const samples = (D.dataset && D.dataset.sample) || [];
    function renderMap() {
      if (!mapCv.w) return; const ctx = mapCv.begin(), b = mapBox();
      ctx.fillStyle = C.bg; ctx.fillRect(0, 0, mapCv.w, mapCv.h);
      ctx.drawImage(fieldBitmap(F, b.w, b.h, mapCv.dpr), b.x, b.y, b.w, b.h);
      // ticks
      [-2, -1, 0, 1, 2].forEach(v => {
        const px = T.toPx([v, 0], b)[0], py = T.toPx([0, v], b)[1];
        txt(ctx, fx(v, 0), px, b.y + b.h + 16, { align: 'center', size: 11.5 });
        txt(ctx, fx(v, 0), b.x - 6, py, { align: 'right', base: 'middle', size: 11.5 });
      });
      ctx.save(); ctx.beginPath(); ctx.rect(b.x, b.y, b.w, b.h); ctx.clip();
      const P = (y) => T.toPx(y, b);
      // where targets live across all contexts (the figure-eight)
      poly(ctx, manifold.map(P), 'rgba(233,238,246,0.28)', 1, { dash: [3, 4] });
      const m = muOf(S.x), sg = sigOf(S.x);
      if (S.truth) {
        samples.forEach(s => { let dx = Math.abs(s[0] - S.x); dx = Math.min(dx, 1 - dx); if (dx < 0.02) { const q = P([s[1], s[2]]); circle(ctx, q[0], q[1], 2.1, 'rgba(116,224,192,0.75)'); } });
        const pm = P(m), rr = 2 * sg / (EXT[1] - EXT[0]) * b.w;
        ctx.save(); ctx.setLineDash([4, 3]); circle(ctx, pm[0], pm[1], rr, null, 'rgba(116,224,192,0.85)', 1.3); ctx.restore();
      }
      // candidate paths
      const ch = chosenAt(k), you = S.user ? 0 : -1;
      const order = runs.map((_, j) => j).filter(j => j !== ch).concat([ch]);
      order.forEach(j => {
        const r = runs[j], pts = r.ys.slice(0, k + 1).map(P), cur = posAt(r, k, frac);
        if (frac > 0 && k < S.N) pts.push(P(cur));
        const isC = j === ch;
        poly(ctx, pts, isC ? C.ebt : 'rgba(233,238,246,0.55)', isC ? 2.4 : 1.3);
        r.ys.slice(1, k + 1).forEach(y => { const q = P(y); circle(ctx, q[0], q[1], isC ? 2.4 : 1.6, isC ? C.ebt : 'rgba(233,238,246,0.6)'); });
        const s0 = P(r.y0); circle(ctx, s0[0], s0[1], 3.2, 'rgba(13,19,31,0.85)', isC ? C.ebt : 'rgba(233,238,246,0.7)', 1.4);
      });
      // next-step arrows (deterministic part −α∇E) when paused
      if (!playing && k < S.N) runs.forEach((r, j) => {
        const y = r.ys[k], g = r.gs[k], q0 = P(y), q1 = P([y[0] - r.alpha * g[0], y[1] - r.alpha * g[1]]);
        arrowPx(ctx, q0[0], q0[1], q1[0], q1[1], j === ch ? C.ebt : 'rgba(233,238,246,0.7)', j === ch ? 2 : 1.3);
      });
      // current positions
      order.forEach(j => { const q = P(posAt(runs[j], k, frac)); const isC = j === ch; circle(ctx, q[0], q[1], isC ? 6 : 4.2, isC ? C.ebt : C.ink, 'rgba(13,19,31,0.9)', 1.6); });
      if (you === 0) { const q = P(runs[0].y0); circle(ctx, q[0], q[1], 8, null, C.ink, 2); txt(ctx, 'your ŷ₀', q[0] + 11, q[1] - 9, { size: 12, color: C.ink, halo: 'rgba(13,19,31,0.85)' }); }
      if (S.truth) { const pm = P(m); xMark(ctx, pm[0], pm[1], 6, C.truth); }
      // basin bottom of this verifier
      if (basin) { const q = P(basin.y); ctx.save(); ctx.strokeStyle = 'rgba(247,230,189,0.9)'; ctx.lineWidth = 1.2; ctx.beginPath(); ctx.moveTo(q[0] - 5, q[1]); ctx.lineTo(q[0] + 5, q[1]); ctx.moveTo(q[0], q[1] - 5); ctx.lineTo(q[0], q[1] + 5); ctx.stroke(); ctx.restore(); }
      // hover probe
      if (probe) {
        const eg = T.energyGrad(S.ck, S.x, probe), q0 = P(probe), q1 = P([probe[0] - S.alpha * eg.g[0], probe[1] - S.alpha * eg.g[1]]);
        arrowPx(ctx, q0[0], q0[1], q1[0], q1[1], C.e5 || '#f7e6bd', 1.8);
        circle(ctx, q0[0], q0[1], 3, C.ink);
        const lab = 'E = ' + fx(eg.E, 3) + '   |∇E| = ' + Math.hypot(eg.g[0], eg.g[1]).toFixed(2);
        const right = q0[0] < b.x + b.w * 0.55;
        txt(ctx, lab, q0[0] + (right ? 10 : -10), q0[1] + (q0[1] < b.y + 24 ? 18 : -10), { align: right ? 'left' : 'right', size: 12, color: C.ink, halo: 'rgba(13,19,31,0.9)' });
      }
      txt(ctx, 'ŷ[0] →', b.x + b.w - 6, b.y + b.h - 7, { align: 'right', size: 12, color: C.ink, halo: 'rgba(13,19,31,0.8)' });
      txt(ctx, '↑ ŷ[1]', b.x + 6, b.y + 15, { size: 12, color: C.ink, halo: 'rgba(13,19,31,0.8)' });
      ctx.restore();
      ctx.strokeStyle = C.rule; ctx.lineWidth = 1; ctx.strokeRect(b.x + 0.5, b.y + 0.5, b.w - 1, b.h - 1);
    }

    // ------------------------------------------------------------------ rendering: 3D surface
    function renderSurf() {
      if (!surfCv.w) return; const ctx = surfCv.begin(), w = surfCv.w, hh = surfCv.h;
      ctx.fillStyle = C.bg; ctx.fillRect(0, 0, w, hh);
      const sb = surfaceBitmap(surfCache, F, w, hh, surfCv.dpr, view3, ZLO, ZHI, view3.dragging ? 2 : 1);
      ctx.drawImage(sb.cv, 0, 0, w, hh);
      const pr = sb.project, U = (y) => [(y[0] - EXT[0]) / (EXT[1] - EXT[0]), (EXT[3] - y[1]) / (EXT[3] - EXT[2])];
      const inside = (y) => y[0] >= EXT[0] && y[0] <= EXT[1] && y[1] >= EXT[2] && y[1] <= EXT[3];
      const P3 = (y, E) => { const u = U([clamp(y[0], EXT[0], EXT[1]), clamp(y[1], EXT[2], EXT[3])]); return pr(u[0], u[1], E); };
      const ch = chosenAt(k);
      const order = runs.map((_, j) => j).filter(j => j !== ch).concat([ch]);
      order.forEach(j => {
        const r = runs[j], isC = j === ch, pts = [];
        for (let i = 0; i <= k; i++) pts.push(P3(r.ys[i], r.Es[i]));
        const cur = posAt(r, k, frac), Ec = (frac > 0 && k < S.N) ? T.energy(S.ck, S.x, cur) : r.Es[Math.min(k, S.N)];
        if (frac > 0 && k < S.N) pts.push(P3(cur, Ec));
        poly(ctx, pts, 'rgba(13,19,31,0.6)', isC ? 4.5 : 3);
        poly(ctx, pts, isC ? C.ebt : 'rgba(233,238,246,0.8)', isC ? 2.2 : 1.2);
        const q = P3(cur, Ec); circle(ctx, q[0], q[1] - (isC ? 4 : 3), isC ? 5.5 : 3.8, isC ? C.ebt : C.ink, 'rgba(13,19,31,0.9)', 1.4);
        if (!inside(cur)) txt(ctx, 'off map', q[0] + 6, q[1] - 8, { size: 11, color: C.bad });
      });
      if (S.truth) { const m = muOf(S.x), q = P3(m, T.energy(S.ck, S.x, m)); xMark(ctx, q[0], q[1], 5, C.truth, 2); }
      txt(ctx, 'height = E (shared scale ' + fx(ZLO, 1) + ' to ' + fx(ZHI, 1) + ')', 10, hh - 10, { size: 11.5, color: C.faint });
      txt(ctx, 'drag to rotate', w - 10, 18, { size: 11.5, color: C.faint, align: 'right' });
    }

    // ------------------------------------------------------------------ rendering: per-step plots
    function renderPlots() {
      const ch = chosenAt(k), kk = k + (playing ? clamp(frac, 0, 0.999) : 0);
      { // energy per step
        const cv = ePlot; if (!cv.w) return; const ctx = cv.begin(); ctx.fillStyle = C.panel2; ctx.fillRect(0, 0, cv.w, cv.h);
        let lo = Infinity, hi = -Infinity; runs.forEach(r => r.Es.forEach(v => { if (v < lo) lo = v; if (v > hi) hi = v; }));
        if (basin && basin.E < lo) lo = basin.E; const pad = (hi - lo) * 0.08 || 0.1; lo -= pad; hi += pad;
        const yt = niceTicks(lo, hi, 3), xt = niceTicks(0, S.N, Math.min(S.N, 5)).filter(v => Number.isInteger(v));
        const ax = axes(ctx, { x: 46, y: 26, w: cv.w - 58, h: cv.h - 50, xlim: [0, Math.max(1, S.N)], ylim: [lo, hi], xticks: xt, yticks: yt, yfmt: (v) => fx(v, Math.abs(hi - lo) < 0.5 ? 2 : 1) });
        txt(ctx, 'Energy per step', 10, 16, { size: 12.5, color: C.ink, mono: false, weight: 700 });
        txt(ctx, 'step i', cv.w - 12, 16, { size: 11.5, align: 'right', color: C.faint });
        if (basin) { const yy = ax.Y(basin.E); poly(ctx, [[ax.X(0), yy], [ax.X(Math.max(1, S.N)), yy]], 'rgba(247,230,189,0.55)', 1, { dash: [2, 3] }); txt(ctx, 'basin bottom', ax.X(Math.max(1, S.N)) - 2, yy - 4, { size: 11, align: 'right', color: C.faint }); }
        const order = runs.map((_, j) => j).filter(j => j !== ch).concat([ch]);
        order.forEach(j => {
          const r = runs[j], isC = j === ch, n = Math.floor(kk), pts = [];
          for (let i = 0; i <= Math.min(n, S.N); i++) pts.push([ax.X(i), ax.Y(r.Es[i])]);
          if (kk > n && n < S.N) { const f = kk - n; pts.push([ax.X(n + f), ax.Y(r.Es[n] + (r.Es[n + 1] - r.Es[n]) * f)]); }
          poly(ctx, pts, isC ? C.ebt : 'rgba(233,238,246,0.45)', isC ? 2.4 : 1.2);
          if (pts.length) { const q = pts[pts.length - 1]; circle(ctx, q[0], q[1], isC ? 3.5 : 2.2, isC ? C.ebt : 'rgba(233,238,246,0.7)'); }
        });
      }
      { // distance to truth (log)
        const cv = dPlot; if (!cv.w) return; const ctx = cv.begin(); ctx.fillStyle = C.panel2; ctx.fillRect(0, 0, cv.w, cv.h);
        const m = muOf(S.x); let hi = 0.1; runs.forEach(r => r.ys.forEach(y => { hi = Math.max(hi, dist(y, m)); }));
        const top = Math.pow(10, Math.ceil(Math.log10(hi * 1.05))), bot = 1e-3;
        const yt = []; for (let e = -3; Math.pow(10, e) <= top + 1e-12; e++) yt.push(Math.pow(10, e));
        const xt = niceTicks(0, S.N, Math.min(S.N, 5)).filter(v => Number.isInteger(v));
        const ax = axes(ctx, { x: 46, y: 26, w: cv.w - 58, h: cv.h - 50, xlim: [0, Math.max(1, S.N)], ylim: [bot, top], ylog: true, xticks: xt, yticks: yt, yfmt: (v) => v >= 1 ? String(v) : v.toString() });
        txt(ctx, 'Distance to truth ‖ŷᵢ − μ(x)‖', 10, 16, { size: 12.5, color: C.ink, mono: false, weight: 700 });
        const sy = ax.Y(sigOf(S.x)); poly(ctx, [[ax.X(0), sy], [ax.X(Math.max(1, S.N)), sy]], 'rgba(116,224,192,0.7)', 1, { dash: [4, 3] });
        txt(ctx, 'target noise σ(x)', ax.X(Math.max(1, S.N)) - 2, sy - 4, { size: 11, align: 'right', color: C.truth });
        const order = runs.map((_, j) => j).filter(j => j !== ch).concat([ch]);
        order.forEach(j => {
          const r = runs[j], isC = j === ch, n = Math.floor(kk), pts = [];
          for (let i = 0; i <= Math.min(n, S.N); i++) pts.push([ax.X(i), ax.Y(Math.max(bot, dist(r.ys[i], m)))]);
          if (kk > n && n < S.N) { const f = kk - n, p = posAt(r, n, f); pts.push([ax.X(n + f), ax.Y(Math.max(bot, dist(p, m)))]); }
          poly(ctx, pts, isC ? C.ebt : 'rgba(233,238,246,0.45)', isC ? 2.4 : 1.2);
          if (pts.length) { const q = pts[pts.length - 1]; circle(ctx, q[0], q[1], isC ? 3.5 : 2.2, isC ? C.ebt : 'rgba(233,238,246,0.7)'); }
        });
      }
    }

    // ------------------------------------------------------------------ readouts, equation card, notes
    function rd(label, value, title) { return '<span' + (title ? ' title="' + title + '"' : '') + '>' + label + ' <b>' + value + '</b></span>'; }
    function renderText() {
      const ch = chosenAt(k), cl = closestAt(k), m = muOf(S.x), r = runs[ch];
      status.textContent = 'step ' + k + ' of ' + S.N + ' · ' + (S.M * k) + ' gradient evaluations (M × steps)';
      const sg = sigOf(S.x), ci = D.contexts.findIndex(c => Math.abs(c.x - S.x) < 1e-6);
      ctxInfo.innerHTML = 'x = ' + S.x.toFixed(3) + (ci >= 0 ? ' · ' + (CTX_NAMES[ci] || '') : '') + '<br>targets y ~ N(μ(x), σ²I), μ(x) = ' + vec(m, 2) + ', σ = ' + sg.toFixed(3);
      cbarLo.textContent = fx(F.lo, 2); cbarHi.textContent = fx(F.hi, 2);
      srcSpan.textContent = (F.src === 'stored' ? 'grid: exported 64×64' : coarse ? 'grid: live 32×32 (refining)' : 'grid: computed live, 64×64') + ' · θ at ' + ckLabel(S.ck) + ' · color ∝ √(E − E_min)';
      const dC = dist(r.ys[k], m), dCl = dist(runs[cl].ys[k], m), meanD = runs.reduce((s, q) => s + dist(q.ys[k], m), 0) / runs.length;
      const lam = basin ? basin.lam : [NaN, NaN];
      const aMax = S.randAlpha ? S.alpha * (HP.alpha_rand_factor || 2) : S.alpha;
      readout.innerHTML = [
        rd('lowest energy:', 'guess ' + (ch + 1) + ' of ' + S.M + ', E = ' + fx(r.Es[k], 3)),
        rd('its distance to truth:', fx(dC, 3)),
        rd('closest guess:', (cl === ch ? 'the same one' : 'guess ' + (cl + 1) + ' (' + fx(dCl, 3) + ')')),
        rd('mean distance:', fx(meanD, 3)),
        rd('basin curvature λ:', basin ? fx(lam[0], 2) + ', ' + fx(lam[1], 2) : '…', 'Eigenvalues of the 2×2 Hessian of E at the basin bottom, by finite differences of the live gradient'),
        rd('largest α·λ:', basin ? fx(aMax * lam[1], 2) + (aMax * lam[1] < 2 ? ' < 2, stable' : ' > 2, unstable') : '…', 'Near a quadratic bottom, gradient descent converges only if α·λmax < 2'),
      ].join('');
      // live math for the focus candidate
      const fj = S.user ? 0 : ch, fr = runs[fj], i = Math.min(k, S.N - 1), y = fr.ys[i], g = fr.gs[i], et = fr.etas[i] || [0, 0], y1 = fr.ys[i + 1];
      const tag = S.user ? 'your guess' : 'lowest-energy guess';
      eqCard.innerHTML = '<div class="landscape-eqh">One update, live numbers <span>(' + tag + ', step ' + i + ' → ' + (i + 1) + ')</span></div>' +
        '<table><tbody>' +
        '<tr><td>ŷ' + sub(i) + '</td><td>' + vec(y) + '</td><td>E = ' + fx(fr.Es[i], 3) + '</td></tr>' +
        '<tr><td>∇<sub>ŷ</sub>E</td><td>' + vec(g) + '</td><td>|∇E| = ' + Math.hypot(g[0], g[1]).toFixed(3) + '</td></tr>' +
        '<tr><td>− α∇E</td><td>' + vec([-fr.alpha * g[0], -fr.alpha * g[1]]) + '</td><td>α = ' + fr.alpha.toFixed(3) + '</td></tr>' +
        '<tr><td>+ η</td><td>' + vec(et) + '</td><td>η ~ N(0, ' + S.sigma.toFixed(2) + '²)</td></tr>' +
        '<tr class="sum"><td>ŷ' + sub(i + 1) + '</td><td>' + vec(y1) + '</td><td>E = ' + fx(fr.Es[i + 1], 3) + '</td></tr>' +
        '</tbody></table>';
      // what is going on
      const out = [];
      const fin = runs.map(q => q.ys[S.N]), spread = Math.max(...fin.map(p => dist(p, fin[0])));
      const Ech = runs[chosenAt(S.N)].Es;
      const bouncing = Ech.length > 3 && Ech[S.N] > Math.min(...Ech) + 0.02;
      if (basin && aMax * lam[1] > 2) out.push(['Overshooting', 'The step α·λ is ' + fx(aMax * lam[1], 2) + ', above 2. Each update jumps past the bottom of the bowl, so the energy bounces' + (bouncing ? ' (see the plot)' : '') + '. Near a quadratic bottom, descent is stable only when α·λmax < 2. Training used α from 0.5 to 2, and the learned curvature λ ≈ ' + fx(lam[1], 2) + ' keeps even α = 2 stable.']);
      if (STEPS[S.ck] <= 50) out.push(['Untrained verifier', 'At ' + ckLabel(S.ck) + ' the lowest point of the landscape is ' + (basin ? fx(basin.d, 2) : '?') + ' away from the truth. Descent still settles, so a low energy here does not mean a good answer. Move the checkpoint slider to watch the bowl travel to the truth.']);
      if (S.M > 1 && cl !== ch && dist(runs[ch].ys[k], m) - dCl > 0.03) out.push(['Verifier fooled', 'Guess ' + (ch + 1) + ' has the lowest energy, but guess ' + (cl + 1) + ' is closer to the truth. The energy is a learned score, not an oracle.']);
      if (k === S.N && S.M > 1 && spread < 0.01 && !out.length) out.push(['Nothing to choose between', 'All ' + S.M + ' guesses end at the same point (spread ' + fx(spread, 4) + '): this landscape has one smooth basin. Best-of-N only helps when guesses end in different places: fewer steps, noise σ, or random α.']);
      if (!out.length && k === S.N && S.N <= 2 && S.M > 1) out.push(['Short thinking, wide search', 'After ' + S.N + ' step' + (S.N > 1 ? 's' : '') + ' the guesses sit at different distances, and the lowest energy picks one ' + (cl === ch ? 'that is also the closest' : 'near the closest') + '. Spending compute on more guesses can replace thinking longer.']);
      if (!out.length && S.sigma > 0) out.push(['Noise on', 'Each step adds a random kick of size σ (Eq. 2). It keeps guesses apart, so the verifier has real choices, but no single guess settles exactly. In the paper, noise is a training regularizer; the ablation (Table 2, p.10) trains with and without it.']);
      if (!out.length) {
        const conv = (() => { for (let i2 = 1; i2 <= S.N; i2++) if (Math.abs(Ech[i2] - Ech[i2 - 1]) < 1e-3) return i2; return null; })();
        out.push(['Converging', conv ? 'The lowest-energy guess stops changing its energy (|ΔE| < 0.001) at step ' + conv + '. The paper\'s stopping rule is the same idea: think until the energy converges (Fig 2 caption, p.4).' : 'The energy is still falling at step ' + S.N + '. More steps would keep improving the guess: this is thinking longer.']);
      }
      if (activePreset) { const p = PRESETS.find(q => q[0] === activePreset); if (p) out.unshift(['Preset: ' + p[0], p[1]]); }
      notes.innerHTML = out.slice(0, 2).map(([t, s]) => '<p><b>' + t + '.</b> ' + s + '</p>').join('');
    }

    let raf = 0;
    function render() {
      if (raf) return; // coalesce
      raf = requestAnimationFrame(() => {
        raf = 0; const dbg = window.EBT_DEBUG_LANDSCAPE, t = [performance.now()];
        renderMap(); dbg && t.push(performance.now()); renderSurf(); dbg && t.push(performance.now()); renderPlots(); dbg && t.push(performance.now());
        renderText(); dbg && t.push(performance.now()); syncLinked(); dbg && t.push(performance.now());
        if (dbg) dbg.push(t.slice(1).map((v, i) => +(v - t[i]).toFixed(2)));
      });
    }

    // ================================================================== Part 2: self-verification, measured
    el.appendChild(H3('2', 'Self-verification: is the lowest energy really the best guess?'));
    el.appendChild(h('p', { class: 'landscape-p', html: 'Algorithm 2 (p.7) runs M guesses and returns the one with the lowest energy. No external judge is used: the model verifies itself. Measure it on the toy with your current lab settings. Each run predicts for 96 contexts spread along the curve and scores each prediction by its squared distance to the clean truth μ(x).' }));
    const vgrid = h('div', { class: 'landscape-vgrid' }); el.appendChild(vgrid);
    const mCard = h('div', { class: 'card' }); vgrid.appendChild(mCard);
    mCard.appendChild(h('div', { class: 'row landscape-cardhead' }, h('h4', { style: 'margin:0' }, 'Live measurement'), lib.badge('toy', 'computed in your browser')));
    const mBtn = h('button', { type: 'button', class: 'btn primary' }, 'Measure with current settings');
    const mCfg = h('div', { class: 'landscape-hint' });
    const mOut = h('div', { class: 'landscape-bars' });
    const mNote = h('p', { class: 'landscape-hint' });
    mCard.append(h('div', { class: 'row' }, mBtn), mCfg, mOut, mNote);
    let mJob = 0, mDoneCfg = null;
    function cfgStr(c) { return 'θ at ' + ckLabel(c.ck) + ' · α ' + c.alpha.toFixed(2) + (c.randAlpha ? ' (random ×½ to ×2)' : '') + ' · N ' + c.N + ' · σ ' + c.sigma.toFixed(2) + ' · M ' + c.M; }
    function measureStale() { if (!mDoneCfg) return; const now = cfgStr(S); mBtn.textContent = now === mDoneCfg ? 'Measure again' : 'Settings changed: measure again'; mBtn.classList.toggle('primary', now !== mDoneCfg); }
    function measure() {
      const job = ++mJob, c = { ck: S.ck, alpha: S.alpha, N: S.N, sigma: S.sigma, M: S.M, randAlpha: S.randAlpha }, NC = 96, rr = lib.rng(4242);
      let i = 0, e1 = 0, eB = 0, eO = 0, wins = 0; const lf = Math.log(HP.alpha_rand_factor || 2);
      mBtn.disabled = true; mCfg.textContent = cfgStr(c);
      const tick = () => {
        if (job !== mJob) return; const t0 = performance.now();
        while (i < NC && performance.now() - t0 < 14) {
          const x = (i + 0.5) / NC, m = muOf(x); let bestE = Infinity, bestErr = 0, orc = Infinity, first = 0;
          for (let j = 0; j < c.M; j++) {
            let y = [rr.normal(), rr.normal()]; const a = c.randAlpha ? c.alpha * Math.exp((2 * rr() - 1) * lf) : c.alpha;
            for (let s = 0; s < c.N; s++) { const g = T.energyGrad(c.ck, x, y).g; y = [clamp(y[0] - a * g[0] + c.sigma * rr.normal(), -30, 30), clamp(y[1] - a * g[1] + c.sigma * rr.normal(), -30, 30)]; }
            const E = T.energy(c.ck, x, y), err = (y[0] - m[0]) ** 2 + (y[1] - m[1]) ** 2;
            if (j === 0) first = err; if (E < bestE) { bestE = E; bestErr = err; } if (err < orc) orc = err;
          }
          e1 += first; eB += bestErr; eO += orc; if (bestErr <= orc + 1e-12) wins++; i++;
        }
        mOut.innerHTML = '<div class="landscape-prog"><i style="width:' + (100 * i / NC).toFixed(0) + '%"></i></div>';
        if (i < NC) { setTimeout(tick, 0); return; }
        mBtn.disabled = false; mDoneCfg = cfgStr(c); measureStale();
        const v = [['One guess (M = 1)', e1 / NC, 'var(--muted)'], ['Lowest energy of ' + c.M, eB / NC, 'var(--ebt)'], ['Oracle: closest of ' + c.M, eO / NC, 'var(--truth)']];
        const mx = Math.max(...v.map(q => q[1])) || 1;
        mOut.innerHTML = v.map(([l, val, col]) => '<div class="landscape-bar"><span class="l">' + l + '</span><span class="b"><i style="width:' + Math.max(0.6, 100 * val / mx).toFixed(1) + '%;background:' + col + '"></i></span><span class="v">' + val.toExponential(2).replace('-', '−') + '</span></div>').join('');
        const gain = (e1 / NC) / (eB / NC);
        mNote.innerHTML = 'Mean squared distance to μ(x), lower is better. Compute: ' + c.M + ' × ' + c.N + ' = <b>' + (c.M * c.N) + '</b> gradient steps per prediction. ' +
          (c.M === 1 ? 'With one guess there is nothing to verify.' : gain > 1.15 ? 'Choosing by energy cuts the error <b>' + gain.toFixed(1) + '×</b>; it picked the truly closest guess in ' + wins + ' of ' + NC + ' contexts.' : gain < 0.95 ? 'Choosing by energy made things <b>worse</b> (' + gain.toFixed(2) + '×): this verifier rewards the wrong guesses.' : 'Choosing by energy barely matters here (' + gain.toFixed(2) + '×): the guesses end too close together to tell apart, or the verifier cannot tell them apart.');
      };
      tick();
    }
    mBtn.addEventListener('click', measure);

    // stored toy results (1024 eval contexts) as charts
    const cCard = h('div', { class: 'card' }); vgrid.appendChild(cCard);
    cCard.appendChild(h('div', { class: 'row landscape-cardhead' }, h('h4', { style: 'margin:0' }, 'Exported toy results'), lib.badge('toy', '1024 eval contexts')));
    const bonA = rcanvas(cCard, { height: (w) => clamp(w * 0.5, 170, 210), label: 'Toy: error versus number of candidates M, with oracle line' });
    const bonT = rcanvas(cCard, { height: (w) => clamp(w * 0.5, 170, 210), label: 'Toy: error with 1 versus 8 candidates across training' });
    cCard.appendChild(h('p', { class: 'landscape-hint', html: 'Each guess: random start, random α, 6 steps, Langevin σ 0.05. Error = mean ‖ŷ* − μ(x)‖². The amber dot on the lower chart follows the lab\'s checkpoint.' }));
    function renderBon() {
      const B = D.bon && D.bon.aggregate, BT = D.bon_vs_training; if (!B || !BT) return;
      { const cv = bonA; if (!cv.w) return; const ctx = cv.begin(); ctx.fillStyle = C.panel2; ctx.fillRect(0, 0, cv.w, cv.h);
        const ys = B.mse_vs_mu.concat([B.oracle_best_of_16]); const lo = Math.pow(10, Math.floor(Math.log10(Math.min(...ys)))), hi = Math.pow(10, Math.ceil(Math.log10(Math.max(...ys))));
        const yt = []; for (let v = lo; v <= hi * 1.001; v *= 10) yt.push(v);
        const ax = axes(ctx, { x: 52, y: 30, w: cv.w - 66, h: cv.h - 56, xlim: [1, 16], ylim: [lo, hi], xlog: true, ylog: true, xticks: B.M, yticks: yt, yfmt: (v) => v.toExponential(0).replace('-', '−') });
        txt(ctx, 'More guesses, lower error', 10, 18, { size: 12.5, color: C.ink, mono: false, weight: 700 });
        if (cv.w > 380) txt(ctx, 'candidates M →', cv.w - 12, 18, { size: 11.5, align: 'right', color: C.faint });
        const yo = ax.Y(B.oracle_best_of_16); poly(ctx, [[ax.X(1), yo], [ax.X(16), yo]], C.truth, 1.2, { dash: [4, 3] }); txt(ctx, 'oracle, best of 16', ax.X(1) + 4, yo + 14, { size: 11, color: C.truth });
        const pts = B.M.map((m, i) => [ax.X(m), ax.Y(B.mse_vs_mu[i])]); poly(ctx, pts, C.ebt, 2.2); pts.forEach(p => circle(ctx, p[0], p[1], 3.5, C.ebt));
        txt(ctx, 'M = 16: ' + (B.mse_vs_mu[0] / B.mse_vs_mu[B.M.length - 1]).toFixed(0) + '× lower than M = 1', ax.X(16) - 4, ax.Y(hi) + 16, { size: 11.5, align: 'right', color: C.ebt, halo: C.panel2 });
      }
      { const cv = bonT; if (!cv.w) return; const ctx = cv.begin(); ctx.fillStyle = C.panel2; ctx.fillRect(0, 0, cv.w, cv.h);
        const all = BT.M1.concat(BT.M8); const lo = Math.pow(10, Math.floor(Math.log10(Math.min(...all)))), hi = Math.pow(10, Math.ceil(Math.log10(Math.max(...all))));
        const yt = []; for (let v = lo; v <= hi * 1.001; v *= 10) yt.push(v);
        const ax = axes(ctx, { x: 52, y: 30, w: cv.w - 66, h: cv.h - 56, xlim: [BT.step[0], BT.step[BT.step.length - 1]], ylim: [lo, hi], xlog: true, ylog: true, xticks: [25, 100, 1000, 10000].filter(v => v >= BT.step[0]), yticks: yt, yfmt: (v) => v.toExponential(0).replace('-', '−'), xfmt: (v) => v >= 1000 ? (v / 1000) + 'k' : String(v) });
        txt(ctx, 'Verification improves with training', 10, 18, { size: 12.5, color: C.ink, mono: false, weight: 700 });
        if (cv.w > 420) txt(ctx, 'training step →', cv.w - 12, 18, { size: 11.5, align: 'right', color: C.faint });
        const p1 = BT.step.map((s, i) => [ax.X(s), ax.Y(BT.M1[i])]), p8 = BT.step.map((s, i) => [ax.X(s), ax.Y(BT.M8[i])]);
        poly(ctx, p1, 'rgba(233,238,246,0.7)', 1.6); poly(ctx, p8, C.ebt, 2.2);
        p1.forEach(p => circle(ctx, p[0], p[1], 2.5, 'rgba(233,238,246,0.8)')); p8.forEach(p => circle(ctx, p[0], p[1], 2.5, C.ebt));
        const L = BT.step.length - 1; txt(ctx, 'M = 1', p1[L][0] - 4, p1[L][1] - 8, { size: 11.5, align: 'right', color: C.ink });
        txt(ctx, 'M = 8', p8[L][0] - 4, p8[L][1] + 16, { size: 11.5, align: 'right', color: C.ebt });
        const r0 = BT.M1[0] / BT.M8[0], r1 = BT.M1[L] / BT.M8[L];
        txt(ctx, 'gap ' + r0.toFixed(1) + '× → ' + r1.toFixed(1) + '×', ax.X(BT.step[0]) + 6, ax.Y(lo) - 8, { size: 11.5, color: C.muted });
        const si = BT.step.indexOf(STEPS[S.ck]);
        if (si >= 0) { const xx = ax.X(BT.step[si]); poly(ctx, [[xx, ax.Y(hi)], [xx, ax.Y(lo)]], 'rgba(255,201,92,0.45)', 1, { dash: [3, 3] }); circle(ctx, p8[si][0], p8[si][1], 5, null, C.ebt, 2); }
      }
    }
    el.appendChild(h('div', { class: 'landscape-paperrow' },
      lib.badge('paper', 'Fig 6b, Table 2, p.10, p.28'),
      h('p', { html: 'The paper sees the same two effects at scale. Self-verification gains grow as the EBT trains: BoN-5 improves perplexity by 4 to 8% early and 10 to 14% later (Fig 6b, Dyck only). Early in training, verifying 10 samples was sometimes worse than verifying 2, "likely because the EBT found an adversarial sample" (p.28). And noise during training trades single-path thinking for verification: without Langevin noise, thinking longer improves 17.2% but adding self-verification gives 17.0%, while the full setup gives 7.19% and 18.7% (Table 2, OOD Dyck).' })));

    // ================================================================== Part 3: uncertainty
    el.appendChild(H3('3', 'Does the shape of the basin show uncertainty?'));
    const ugrid = h('div', { class: 'landscape-ugrid' }); el.appendChild(ugrid);
    const uText = h('div', { class: 'stack landscape-utext' }); ugrid.appendChild(uText);
    const uCard = h('div', { class: 'card' }); ugrid.appendChild(uCard);
    const BS = D.basin_stats;
    uText.append(
      h('p', { html: 'Facet 2 in the paper is modelling uncertainty: a harder prediction should end at a higher energy or in a wider basin. Our toy has a built-in test, since its target noise σ(x) ranges from 0.03 to 0.30 along the curve. The chart measures the final model\'s basin at 64 contexts.' }),
      h('p', { html: 'The trend points the right way but is weak. Noisier contexts end slightly higher (r = ' + fx(BS.corr_sigma_vs_Emin, 2) + ') in a slightly flatter bowl (r = ' + fx(BS.corr_sigma_vs_mean_curvature, 2) + '), yet the curvature changes by under 20% while σ changes 10×.' }),
      lib.callout('', 'Why so weak', 'With a squared-error loss on noisy targets, the best answer is the mean μ(x) whatever the noise, so training has no reason to widen the bowl. Also, training only ever uses ∇<sub>ŷ</sub>E: adding any c(x) to the energy leaves it unchanged. Energies of guesses for the <i>same</i> context are comparable (Best-of-N uses exactly that); energy levels across contexts are not trained directly. The paper\'s uncertainty evidence comes from text and video EBTs (Figs 8 and 11).'));
    uCard.appendChild(h('div', { class: 'row landscape-cardhead' }, h('h4', { style: 'margin:0' }, 'Basin of the final model at 64 contexts'), lib.badge('toy')));
    const uCv = rcanvas(uCard, { height: (w) => clamp(w * 0.62, 250, 320), label: 'Noise level, minimum energy and basin curvature across 64 contexts. Click to set the lab context.' });
    uCv.canvas.style.cursor = 'pointer';
    const uRead = h('div', { class: 'readout' }); uCard.appendChild(uRead);
    uCard.appendChild(h('p', { class: 'landscape-hint' }, 'Click the chart to send that context to the lab above. The amber line is the lab\'s current x.'));
    let uAx = null;
    function renderU() {
      const cv = uCv; if (!cv.w || !BS) return; const ctx = cv.begin(); ctx.fillStyle = C.panel2; ctx.fillRect(0, 0, cv.w, cv.h);
      const rows = [
        ['target noise σ(x)', BS.sigma, C.truth, [0, 0.32]],
        ['lowest energy E_min', BS.E_min, C.ebt, null],
        ['curvature λ (both eigenvalues)', null, C.ar, null],
      ];
      const x0 = 52, w = cv.w - 64, top = 10, gap = 14, rh = (cv.h - top - 26 - gap * 2) / 3, labels = [];
      const X = (v) => x0 + v * w;
      rows.forEach(([lab, arr, col, lim], ri) => {
        const y0 = top + ri * (rh + gap);
        let lo, hi;
        if (ri === 2) { lo = Math.min(...BS.hess_eig_small); hi = Math.max(...BS.hess_eig_large); } else { lo = lim ? lim[0] : Math.min(...arr); hi = lim ? lim[1] : Math.max(...arr); }
        const pad = (hi - lo) * 0.12; lo -= pad; hi += pad;
        const Y = (v) => y0 + rh - (v - lo) / (hi - lo) * rh;
        ctx.strokeStyle = C.rule; ctx.lineWidth = 1; ctx.strokeRect(x0 + 0.5, y0 + 0.5, w - 1, rh - 1);
        niceTicks(lo, hi, 2).forEach(v => { if (Y(v) > y0 + 4 && Y(v) < y0 + rh - 4) txt(ctx, fx(v, 2), x0 - 5, Y(v), { align: 'right', base: 'middle', size: 11 }); });
        ctx.save(); ctx.font = '400 11.5px ' + lib.F.mono; const lw = ctx.measureText(lab).width; ctx.restore();
        labels.push([lab, x0 + 6, y0 + 14, col, lw]);
        if (ri === 2) {
          ctx.save(); ctx.fillStyle = 'rgba(142,167,255,0.25)'; ctx.beginPath();
          BS.x.forEach((xx, i) => { const p = [X(xx), Y(BS.hess_eig_large[i])]; i ? ctx.lineTo(...p) : ctx.moveTo(...p); });
          for (let i = BS.x.length - 1; i >= 0; i--) ctx.lineTo(X(BS.x[i]), Y(BS.hess_eig_small[i]));
          ctx.closePath(); ctx.fill(); ctx.restore();
          poly(ctx, BS.x.map((xx, i) => [X(xx), Y((BS.hess_eig_small[i] + BS.hess_eig_large[i]) / 2)]), col, 1.8);
        } else if (ri === 0) {
          const xs = []; for (let i = 0; i <= 200; i++) xs.push(i / 200); poly(ctx, xs.map(v => [X(v), Y(sigOf(v))]), col, 2);
        } else {
          BS.x.forEach((xx, i) => circle(ctx, X(xx), Y(arr[i]), 2.4, col));
        }
      });
      [0, 0.25, 0.5, 0.75].forEach(v => txt(ctx, v.toFixed(2), X(v), cv.h - 8, { align: 'center', size: 11 }));
      txt(ctx, 'context x →', x0 + w, cv.h - 8, { align: 'right', size: 11, color: C.faint });
      const cx = X(S.x); poly(ctx, [[cx, top], [cx, cv.h - 24]], 'rgba(255,201,92,0.8)', 1.4);
      labels.forEach(([lab, lx, ly, col, lw]) => { ctx.fillStyle = 'rgba(26,36,55,0.88)'; ctx.fillRect(lx - 4, ly - 12, lw + 8, 17); txt(ctx, lab, lx, ly, { size: 11.5, color: col }); });
      uAx = { x0, w };
      let bi = 0; BS.x.forEach((xx, i) => { if (Math.abs(xx - S.x) < Math.abs(BS.x[bi] - S.x)) bi = i; });
      uRead.innerHTML = rd('nearest x', BS.x[bi].toFixed(3)) + rd('σ', BS.sigma[bi].toFixed(3)) + rd('E_min', fx(BS.E_min[bi], 3)) + rd('λ', BS.hess_eig_small[bi].toFixed(2) + ', ' + BS.hess_eig_large[bi].toFixed(2)) + rd('steps to settle', BS.steps_to_converge[bi].toFixed(1));
    }
    uCv.canvas.addEventListener('click', (e) => { if (!uAx) return; const p = uCv.toLocal(e); const v = clamp((p[0] - uAx.x0) / uAx.w, 0, 0.995); setX(Math.round(v * 200) / 200); });

    // ================================================================== Part 4: loss landscape vs energy landscape
    el.appendChild(H3('4', 'Loss landscape vs energy landscape'));
    el.appendChild(h('p', { class: 'landscape-p', html: 'Both pictures are bowls you descend, and both get called "landscapes". They are different objects. The <b>energy landscape</b> lives over predictions ŷ, for one context and frozen weights; walking down it is <b>thinking</b>. The <b>loss landscape</b> lives over the weights θ; walking down it is <b>learning</b>, and every point on it defines a whole energy landscape.' }));
    const PL = D.param_landscape;
    const lCard = h('div', { class: 'card landscape-lcard' }); el.appendChild(lCard);
    const thetaSeg = lib.segmented({ label: 'Weights', options: [['init', 'θ at step 0'], ['final', 'θ after training']], value: 'final', onchange: () => renderL() });
    const viewSeg = lib.segmented({ label: 'View', options: [['3d', '3D'], ['map', 'Map']], value: '3d', onchange: () => renderL() });
    lCard.appendChild(h('div', { class: 'row landscape-lhead' }, h('span', { class: 'landscape-hint' }, 'Weights:'), thetaSeg.el, h('span', { class: 'landscape-hint' }, 'View:'), viewSeg.el, lib.badge('toy')));
    const lPair = h('div', { class: 'landscape-pair' }); lCard.appendChild(lPair);
    const lFigA = h('figure', { class: 'landscape-fig' }), lFigB = h('figure', { class: 'landscape-fig' }); lPair.append(lFigA, lFigB);
    lFigA.appendChild(h('div', { class: 'landscape-ltitle' }, h('b', {}, 'Loss landscape'), ' L(θ): training loss over weights'));
    const lossCv = rcanvas(lFigA, { height: (w) => clamp(w * 0.78, 240, 380), cls: 'rot', label: 'Training loss over a 2D slice of weight space. Drag to rotate in 3D view.' });
    const lossCap = h('figcaption', { class: 'landscape-cap' }); lFigA.appendChild(lossCap);
    lFigB.appendChild(h('div', { class: 'landscape-ltitle' }, h('b', {}, 'Energy landscape'), ' E(x, ŷ): energy over predictions'));
    const enCv = rcanvas(lFigB, { height: (w) => clamp(w * 0.78, 240, 380), cls: 'rot', label: 'Energy over the 2D prediction for the lab context, at the matching weights. Drag to rotate in 3D view.' });
    const enCap = h('figcaption', { class: 'landscape-cap' }); lFigB.appendChild(enCap);
    const viewL = { yaw: -0.62, pitch: 0.95, dragging: false }, lossCache = {}, enCache = {};
    attachRotate(lossCv, viewL, () => renderL()); attachRotate(enCv, viewL, () => renderL());
    let lossHover = null;
    // loss fields: rows of PL.loss run from b = −1 (bottom) to +1, flip to top-down; log colour on a scale shared by both slices
    let LLO = Infinity, LHI = -Infinity;
    const lossField = (() => {
      if (!PL) return {};
      const mk = (L, key) => {
        const E = L.slice().reverse().map(r => r.map(v => Math.log10(Math.max(1e-6, v)))); return E;
      };
      const Ef = mk(PL.loss), Ei = PL.init_slice ? mk(PL.init_slice.loss) : null;
      [Ef, Ei].forEach(E => E && E.forEach(r => r.forEach(v => { if (v < LLO) LLO = v; if (v > LHI) LHI = v; })));
      const tf = (v) => clamp((v - LLO) / (LHI - LLO), 0, 1);
      const lev = []; for (let q = Math.ceil(LLO * 4) / 4; q < LHI; q += 0.25) lev.push(q);
      return { final: makeField(Ef, LLO, LHI, 'loss-final', tf, lev), init: Ei ? makeField(Ei, LLO, LHI, 'loss-init', tf, lev) : null };
    })();
    function drawLike(cvo, Fd, cache, mode, opts) {
      const ctx = cvo.begin(), w = cvo.w, hh = cvo.h; ctx.fillStyle = C.bg; ctx.fillRect(0, 0, w, hh);
      if (mode === '3d') {
        const sb = surfaceBitmap(cache, Fd, w, hh, cvo.dpr, viewL, opts.zlo, opts.zhi, viewL.dragging ? 2 : 1);
        ctx.drawImage(sb.cv, 0, 0, w, hh); return { mode, project: sb.project };
      }
      const s = Math.min(w - 40, hh - 32), b = { x: Math.round((w - s) / 2) + 10, y: 6, w: s, h: s };
      ctx.drawImage(fieldBitmap(Fd, b.w, b.h, cvo.dpr), b.x, b.y, b.w, b.h);
      ctx.strokeStyle = C.rule; ctx.strokeRect(b.x + 0.5, b.y + 0.5, b.w - 1, b.h - 1);
      opts.ticks.forEach(([v, lab]) => { txt(ctx, lab, b.x + v * b.w, b.y + b.h + 15, { align: 'center', size: 11 }); txt(ctx, lab, b.x - 5, b.y + (1 - v) * b.h, { align: 'right', base: 'middle', size: 11 }); });
      return { mode, b };
    }
    function renderL() {
      if (!lossCv.w || !enCv.w || !PL) return;
      const th = thetaSeg.value(), mode = viewSeg.value(), Lf = th === 'init' && lossField.init ? lossField.init : lossField.final;
      const ck = th === 'init' ? 0 : LAST;
      // loss side
      const A = drawLike(lossCv, Lf, lossCache, mode, { zlo: LLO, zhi: LHI, ticks: [[0, '−1'], [0.5, '0'], [1, '1']] });
      const cLoss = th === 'init' ? PL.init_slice.center_loss : PL.center_loss;
      const ctxA = lossCv.ctx;
      if (A.mode === '3d') { const q = A.project(0.5, 0.5, Math.log10(cLoss)); circle(ctxA, q[0], q[1] - 3, 5.5, C.ebt, 'rgba(13,19,31,0.9)', 1.5); txt(ctxA, 'θ (' + (th === 'init' ? 'step 0' : 'trained') + ')', q[0] + 9, q[1] - 8, { size: 12, color: C.ink, halo: 'rgba(13,19,31,0.85)' }); txt(ctxA, 'height = log₁₀ loss', 10, lossCv.h - 10, { size: 11.5, color: C.faint }); }
      else {
        const b = A.b, q = [b.x + b.w / 2, b.y + b.h / 2]; circle(ctxA, q[0], q[1], 5.5, C.ebt, 'rgba(13,19,31,0.9)', 1.5); txt(ctxA, 'θ', q[0] + 9, q[1] - 7, { size: 12, color: C.ink, halo: 'rgba(13,19,31,0.85)' });
        if (lossHover) { const n = Lf.n, c = Math.round(lossHover[0] * (n - 1)), r = Math.round(lossHover[1] * (n - 1)), v = Math.pow(10, Lf.E[r][c]); const px = b.x + c / (n - 1) * b.w, py = b.y + r / (n - 1) * b.h; circle(ctxA, px, py, 3, C.ink); txt(ctxA, 'L = ' + (v < 1 ? v.toFixed(3) : v.toFixed(1)), px + (px < b.x + b.w / 2 ? 8 : -8), py - 8, { size: 12, color: C.ink, align: px < b.x + b.w / 2 ? 'left' : 'right', halo: 'rgba(13,19,31,0.9)' }); }
        lossCv.mapBox = b;
      }
      lossCap.innerHTML = 'Each point is a full set of ' + nParams.toLocaleString('en-US') + ' weights θ + a·d₁ + b·d₂, with d₁, d₂ random filter-normalized directions (a, b from −1 to 1). Height: the Algorithm 1 training loss on a fixed batch. Center: L = <b>' + cLoss.toFixed(3) + '</b>' + (th === 'final' ? ' (noise floor of this batch ' + PL.noise_floor_this_batch.toFixed(3) + ').' : ', before any training.') + ' Color on a log scale shared by both slices.';
      // energy side: same lab context, matching weights
      const Fe = energyField(ck, S.x, false);
      const B2 = drawLike(enCv, Fe, enCache, mode, { zlo: ZLO, zhi: ZHI, ticks: [[0, '−2.5'], [0.5, '0'], [1, '2.5']] });
      const ctxB = enCv.ctx, m = muOf(S.x), start = [-1.6, 1.6], path = T.descend(ck, S.x, start, { alpha: 1, steps: 12 });
      if (B2.mode === '3d') {
        const U = (y) => [(clamp(y[0], EXT[0], EXT[1]) - EXT[0]) / (EXT[1] - EXT[0]), (EXT[3] - clamp(y[1], EXT[2], EXT[3])) / (EXT[3] - EXT[2])];
        const pts = path.path.map((y, i) => { const u = U(y); return B2.project(u[0], u[1], path.energies[i]); });
        poly(ctxB, pts, 'rgba(13,19,31,0.6)', 4); poly(ctxB, pts, C.ebt, 2); const q = pts[pts.length - 1]; circle(ctxB, q[0], q[1] - 3, 5, C.ebt, 'rgba(13,19,31,0.9)', 1.4);
        const um = U(m), qm = B2.project(um[0], um[1], T.energy(ck, S.x, m)); xMark(ctxB, qm[0], qm[1], 5, C.truth);
        txt(ctxB, 'height = E', 10, enCv.h - 10, { size: 11.5, color: C.faint });
      } else {
        const b = B2.b, P = (y) => T.toPx(y, b);
        ctxB.save(); ctxB.beginPath(); ctxB.rect(b.x, b.y, b.w, b.h); ctxB.clip();
        poly(ctxB, path.path.map(P), C.ebt, 2); path.path.forEach(y => { const q = P(y); circle(ctxB, q[0], q[1], 2.4, C.ebt); });
        const pm = P(m); xMark(ctxB, pm[0], pm[1], 6, C.truth); ctxB.restore();
      }
      const dEnd = dist(path.path[path.path.length - 1], m);
      enCap.innerHTML = 'Each point is one prediction ŷ for the lab\'s context x = ' + S.x.toFixed(3) + ', with θ fixed at ' + ckLabel(ck) + '. Height: the energy. Amber: 12 thinking steps from ŷ₀ = (−1.6, 1.6), ending ' + dEnd.toFixed(2) + ' from the truth (mint ×).' + (th === 'init' ? ' Untrained, the energy is almost flat, so thinking barely moves.' : '');
    }
    lossCv.canvas.addEventListener('pointermove', (e) => {
      if (viewSeg.value() !== 'map' || !lossCv.mapBox) return; const p = lossCv.toLocal(e), b = lossCv.mapBox;
      const u = (p[0] - b.x) / b.w, v = (p[1] - b.y) / b.h; lossHover = (u >= 0 && u <= 1 && v >= 0 && v <= 1) ? [u, v] : null; renderL();
    });
    lossCv.canvas.addEventListener('pointerleave', () => { if (lossHover) { lossHover = null; renderL(); } });

    const lgrid = h('div', { class: 'landscape-lgrid' }); el.appendChild(lgrid);
    lgrid.appendChild(h('div', { class: 'card' }, h('h4', {}, 'Two nested loops (Algorithm 1)'),
      h('pre', { class: 'landscape-code', html:
        '<span class="c"># learning: walks the LOSS landscape over θ</span>\n' +
        'for each batch (x, y):\n' +
        '    ŷ ← noise ~ N(0, I)\n' +
        '    <span class="c"># thinking: walks the ENERGY landscape over ŷ, θ frozen</span>\n' +
        '    for i in 1..N:\n' +
        '        ŷ ← ŷ − α ∇<sub>ŷ</sub>E<sub>θ</sub>(x, ŷ)\n' +
        '    L ← ‖ŷ − y‖²\n' +
        '    θ ← θ − lr ∇<sub>θ</sub>L   <span class="c"># backprop through all N steps</span>\n\n' +
        '<span class="c"># inference runs only the inner loop</span>' }),
      h('p', { class: 'landscape-hint', html: 'The gradient ∇<sub>θ</sub>L flows back through every inner step, so it needs gradients of gradients. The paper computes them with Hessian-vector products (p.7). The energy itself is never supervised; only the final prediction is.' })));
    const rows = [
      ['A point is', 'one candidate prediction ŷ', 'one setting of all weights θ'],
      ['Height', 'energy E<sub>θ</sub>(x, ŷ): how badly ŷ fits x', 'training loss: how far the unrolled predictions land from the targets'],
      ['Held fixed', 'the context x and the weights θ', 'the data batch and the random draws'],
      ['Who descends it', 'the model, at inference: <b>thinking</b>', 'the optimizer, in training: <b>learning</b>'],
      ['Dimensions here', '2 (ŷ is a 2D point)', nParams.toLocaleString('en-US') + ' (shown as a random 2D slice)'],
      ['How many', 'one per context, per θ', 'one for the whole model'],
      ['In the paper', 'Fig 3 schematic and every thinking result', 'not shown; Fig 3 only borrows the drawing style of a loss-landscape paper [57]'],
    ];
    lgrid.appendChild(h('div', { class: 'card flush' }, h('div', { class: 'tbl', style: 'border:0' }, h('table', {}, h('thead', {}, h('tr', {}, h('th', {}, ''), h('th', {}, 'Energy landscape'), h('th', {}, 'Loss landscape'))),
      h('tbody', {}, rows.map(([a, b, c]) => h('tr', {}, h('th', { scope: 'row' }, a), h('td', { html: b }), h('td', { html: c }))))))));
    const fig3 = h('figure', { class: 'landscape-fig3' },
      h('div', { class: 'paper-fig' }, h('img', { src: 'media/paper/fig03.png', alt: 'Paper Figure 3: a schematic 3D energy landscape for the context "The dog caught the", with a trajectory descending from step 0 to a converged minimum.', decoding: 'async' })),
      h('figcaption', {}, lib.badge('paper', 'Fig 3, p.5'), ' ', h('span', { html: 'The paper\'s picture of thinking: a prediction for "The dog caught the ___" descends an energy landscape until the energy converges. It is a schematic, "Adapted from [57]" (a loss-landscape visualization paper). What it depicts is an energy landscape over predictions, the left-hand object in the table, like the toy\'s real one above.' })));
    el.appendChild(fig3);

    // ------------------------------------------------------------------ links between parts, resize, start
    let lastLinked = '';
    function syncLinked() {
      const key = S.x.toFixed(4) + '|' + S.ck;
      if (key === lastLinked) return; lastLinked = key;
      renderBon(); renderU(); renderL();
    }
    function fitAll() { let any = false; canvases.forEach(c => { if (c.fit()) any = true; }); return any; }
    const ro = new ResizeObserver(() => { if (fitAll()) { lastLinked = ''; render(); } });
    canvases.forEach(c => ro.observe(c.box));

    refreshField(); simulate(); k = S.N;
    chipBtns.forEach((b, i) => b.setAttribute('aria-pressed', String(Math.abs(D.contexts[i].x - S.x) < 1e-6)));
    fitAll(); render();
    lib.whenVisible(inst, () => { if (!lib.reducedMotion) setTimeout(() => { if (!playing && k === S.N && !S.user) play(); }, 400); });
    lib.whenVisible(mCard, () => setTimeout(measure, 300));
    // test hook (used by src/shot_section.js ACTIONS); harmless otherwise
    window.EBT.landscapeLab = { S, probe: () => probe, preset: (i) => applyPreset(PRESETS[i][2], PRESETS[i][0]), step: () => bStep.click(), set: (o) => { Object.assign(S, o); xS.set(S.x); ckS.set(S.ck); aS.set(S.alpha); nS.set(S.N); sS.set(S.sigma); mS.set(S.M); raBox.checked = S.randAlpha; changed(true); }, measure, view: (th, mode) => { thetaSeg.set(th); viewSeg.set(mode); renderL(); } };
  },
});
