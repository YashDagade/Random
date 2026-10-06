/* Panel: an EBT-style energy head for JEPA world models. BEYOND THE PAPER (discussion).
   Figure: (1) a JEPA diagram that grows an energy head and a planner; (2) the real toy 2-D EBT (data/toy2d.json,
   final checkpoint) read as a latent dynamics model: a weak "JEPA predictor" guess, the learned energy over candidate
   next latents, thinking from the guess, and joint descent on (action, predicted latent) toward a goal (App. A.3). */
(function () {
  'use strict';
  const TAU = 2 * Math.PI;
  // the toy's data-generating process (data/toy2d.json dataset): next latent = mu(x) + sigma(x) * N(0, I)
  const mu = (x) => [1.3 * Math.sin(TAU * x), 0.9 * Math.sin(2 * TAU * x)];
  const sig = (x) => 0.03 + 0.27 * Math.sin(Math.PI * (x - 0.25)) ** 2;
  // a deliberately weak predictor: least squares on first-harmonic features of x. It recovers the loop's width (1.3 sin 2πx)
  // but projects the figure-eight twist (0.9 sin 4πx) to zero.
  const pred = (x) => [1.3 * Math.sin(TAU * x), 0];

  EBT.panel({
    id: 'world-models',
    nav: 'JEPA world models',
    title: 'An energy head for JEPA world models',
    lede: 'Beyond the paper: what an EBT-style verifier would add to a world model that predicts in latent space. A toy EBT trained for this explainer stands in for the latent dynamics.',
    text: `
      <p>The paper never mentions JEPA (joint-embedding predictive architecture). Its predictor regresses the next latent, $\\hat z=g_\\phi(z_t,a_t)$. Read as an energy, $E_{\\text{JEPA}}=\\|g_\\phi(z_t,a_t)-z'\\|^2$ always has its minimum at the guess, with value 0. An energy head $E_\\psi(z_t,a_t,\\hat z)$ trained with Alg. 1 learns the shape instead.</p>`,
    steps: [
      { label: 'JEPA as it is', html: '<p>A 2-D latent space for one context. Dashed loop: the mean next latent of every context; crosshair: one outcome; square: a weak predictor\'s guess. The bowl sits on the guess, so its error is invisible.</p>' },
      { label: 'Add an energy head', html: '<p>Shading is now the toy EBT\'s energy. Its basin sits on the data, so the guess scores above the basin floor: the model can tell it is off before the outcome arrives.</p>' },
      { label: 'Think from the guess', html: '<p>Descend $E$ from the guess (blue) and from noise (gray). Both reach the floor within a few steps: Alg. 1 trained this landscape to be descended from noise, so here the warm start buys little. App. A.4 still proposes the split, EBTs as "the verifier of predictions initialized by standard feed-forward models"; it should pay off where descent is slow or basins are many.</p>' },
      { label: 'Plan through the energy', html: '<p>Free the action: descend $E_\\psi+\\lambda\\|\\hat z-z_{\\text{goal}}\\|^2$ jointly in action and predicted latent, as A.3 suggests, keeping the best of M = 6 starts. Drag the goal.</p>' },
    ],
    after: `<p class="note">Caveats: energy levels across contexts are not trained (Alg. 1 only uses $\\nabla_{\\hat y}E$), and training jointly with a JEPA encoder would still need its anti-collapse machinery.</p>`,
    source: [{ kind: 'ext', note: 'discussion' }, { kind: 'toy', note: 'latent stand-in' }, { kind: 'paper', note: 'App. A.3–A.4' }],
    figure(stage, ctx) {
      const { lib } = ctx, h = lib.h, C = lib.C, T2 = window.EBT && EBT.toy2d;
      const INK = C.ink || '#111111', BLUE = C.blue || '#2f3cff', MUTED = C.muted || '#6b6b70', FAINT = C.faint || '#a3a3a8', RULE = C.rule || '#e4e4e7', GRAY = '#8d8d93';
      const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
      const MONO = (lib.F && lib.F.mono) || 'monospace';
      if (!T2 || !T2.ready) { stage.appendChild(h('p', { class: 'callout warn' }, 'Toy model data missing (data/toy2d.json).')); return {}; }

      // ---------- fast exact evaluator of the toy EBT, with ∂E/∂ŷ and ∂E/∂x (x = context phase) ----------
      const toy = makeToy(T2.data);
      try { [[0.2, [0.3, -0.4]], [0.71, [-1.2, 0.5]]].forEach(([x, y]) => { const r = T2.energyGrad(T2.nCkpt - 1, x, y), E = toy.fwdGrad(toy.ctxBias(x), y[0], y[1]); if (Math.abs(r.E - E) > 1e-6 || Math.abs(r.g[0] - toy.G.a) > 1e-6) console.warn('world-models: evaluator mismatch', r, E); }); } catch (e) { console.warn(e); }

      // ---------- state ----------
      const S = { mode: 'jepa', x: 0.125, alpha: 1.0, seed: 3, eps: [0.6, -0.9], y0: [-1.4, -0.9], think: null, plan: null, goal: [-0.95, -0.75] };
      let visible = false, animT = 0;

      // ---------- layout ----------
      const D = lib.frame(stage, { label: 'A JEPA world model, plus an energy', sub: 'black: JEPA as trained · blue: the part this step adds' });
      const diag = buildDiagram(D.frame);
      const Fl = lib.frame(stage, { label: 'Latent plane · toy EBT', sub: 'shading: E_JEPA, a bowl at the guess' });
      const cpl = autoCanvas(Fl.frame, { aspect: 0.6, minH: 290, maxH: 380, label: 'Latent plane: energy over candidate next latents for one context, with the predictor guess, thinking paths and the plan toward a goal', draw: () => draw() });
      const ro = h('div', { class: 'readout' }); stage.appendChild(ro);
      // controls (one group per mode)
      const cThink = h('div', { class: 'controls' }), cPlan = h('div', { class: 'controls' });
      stage.append(cThink, cPlan);
      const slX = lib.slider({ id: 'wm-x', label: 'context phase x', min: 0, max: 0.995, step: 0.005, value: S.x, fmt: (v) => v.toFixed(3), oninput: (v) => { S.x = v; S.think = null; if (S.mode === 'think') runThink(); draw(); } });
      const slA = lib.slider({ id: 'wm-alpha', label: 'step size α', min: 0.1, max: 1.5, step: 0.05, value: S.alpha, fmt: (v) => v.toFixed(2), oninput: (v) => { S.alpha = v; if (S.mode === 'think') runThink(); draw(); } });
      const bOut = lib.button('new outcome', () => { const r = lib.rng(++S.seed * 7919); S.eps = [r.normal(), r.normal()]; S.y0 = [r.normal() * 1.1, r.normal() * 1.1]; if (S.mode === 'think') runThink(); draw(); });
      const bThink = lib.button('think', () => { runThink(); }, { primary: true });
      cThink.append(bThink, bOut, slX.el, slA.el);
      const bPlan = lib.button('plan', () => runPlan(), { primary: true });
      const bGoal = lib.button('new goal', () => { const r = lib.rng(++S.seed * 31); const xg = r(); const m = mu(xg); S.goal = [clamp(m[0] + 0.35 * r.normal(), -1.9, 1.9), clamp(m[1] + 0.35 * r.normal(), -1.9, 1.9)]; runPlan(); });
      cPlan.append(bPlan, bGoal, h('span', { class: 'wm-hint' }, 'or drag the goal on the plane'));

      // ---------- heatmap caches ----------
      const EXT = [-2.2, 2.2, -2.2, 2.2], HN = 46;
      const gridCache = new Map();
      function ebtGrid(x) {
        const key = Math.round(x * 400); if (gridCache.has(key)) return gridCache.get(key);
        const zb = toy.ctxBias(key / 400), E = []; let lo = Infinity;
        for (let r = 0; r < HN; r++) { const yv = EXT[3] - (EXT[3] - EXT[2]) * r / (HN - 1), row2 = []; for (let c = 0; c < HN; c++) { const xv = EXT[0] + (EXT[1] - EXT[0]) * c / (HN - 1), e = toy.fwd(zb, xv, yv); row2.push(e); if (e < lo) lo = e; } E.push(row2); }
        const g = { E, lo, q: lib.quantile(E, 0.6), key: 'wm-ebt-' + key };
        if (gridCache.size > 160) gridCache.delete(gridCache.keys().next().value);
        gridCache.set(key, g); return g;
      }
      function jepaGrid(p) {
        const E = []; for (let r = 0; r < HN; r++) { const yv = EXT[3] - (EXT[3] - EXT[2]) * r / (HN - 1), row2 = []; for (let c = 0; c < HN; c++) { const xv = EXT[0] + (EXT[1] - EXT[0]) * c / (HN - 1); row2.push((xv - p[0]) ** 2 + (yv - p[1]) ** 2); } E.push(row2); }
        return { E, lo: 0, q: lib.quantile(E, 0.6) };
      }
      const obsOf = (x) => { const m = mu(x), s = sig(x); return [m[0] + s * S.eps[0], m[1] + s * S.eps[1]]; };

      // ---------- thinking ----------
      function descend(x, p0, alpha, n) { const zb = toy.ctxBias(x); let p = p0.slice(); const path = [p.slice()], Es = []; for (let i = 0; i < n; i++) { Es.push(toy.fwdGrad(zb, p[0], p[1])); p = [p[0] - alpha * toy.G.a, p[1] - alpha * toy.G.c]; path.push(p.slice()); } Es.push(toy.fwd(zb, p[0], p[1])); return { path, Es }; }
      function runThink() { const n = 10; S.think = { a: descend(S.x, pred(S.x), S.alpha, n), b: descend(S.x, S.y0, S.alpha, n), n, t0: performance.now() }; animT = lib.reducedMotion ? 1e9 : 0; kick(); draw(); }
      // ---------- planning: joint descent on (action phase a, predicted latent ẑ) ----------
      // Best-of-M (Alg. 2 applied to plans): M starts spread around the action dial, keep the lowest objective J = E + λ‖ẑ − goal‖².
      const LAMG = 3.0, AA = 0.005, AZ = 0.25, PIT = 60, MP = 6;
      function planFrom(a0) {
        let a = a0, p = pred(a0); const tr = [];
        for (let i = 0; i <= PIT; i++) {
          const zb = toy.ctxBias(a), E = toy.fwdGrad(zb, p[0], p[1], a), dg = [p[0] - S.goal[0], p[1] - S.goal[1]], d2 = dg[0] ** 2 + dg[1] ** 2;
          tr.push({ a, p: p.slice(), E, J: E + LAMG * d2, d: Math.sqrt(d2) });
          const gx = toy.G.x, ga = toy.G.a + 2 * LAMG * dg[0], gc = toy.G.c + 2 * LAMG * dg[1];
          a = ((a - AA * gx) % 1 + 1) % 1; p = [p[0] - AZ * ga, p[1] - AZ * gc];
        }
        return tr;
      }
      // verification: think again at a fixed action, from the predictor's guess (12 steps at α = 1)
      function thinkAt(a) { const zb = toy.ctxBias(a); let p = pred(a); for (let k = 0; k < 12; k++) { toy.fwdGrad(zb, p[0], p[1]); p = [p[0] - toy.G.a, p[1] - toy.G.c]; } return p; }
      function runPlan() {
        const runs = []; for (let j = 0; j < MP; j++) runs.push(planFrom((S.x + j / MP) % 1));
        let b = 0; runs.forEach((r, j) => { if (r[PIT].J < runs[b][PIT].J) b = j; });
        const zs = thinkAt(runs[b][PIT].a);
        S.plan = { tr: runs[b], runs, best: b, zs, re: Math.hypot(zs[0] - S.goal[0], zs[1] - S.goal[1]) };
        animT = lib.reducedMotion ? 1e9 : 0; kick(); draw();
      }

      // ---------- drawing ----------
      const nfmt = (v) => Math.round(v).toLocaleString('en-US');
      function draw() { drawPlane(); readouts(); }
      function plane() { const { w, h: hh } = cpl, sz = Math.min(w, hh), bx = (w - sz) / 2, by = (hh - sz) / 2; return { sz, bx, by, P: (p) => [bx + (p[0] - EXT[0]) / (EXT[1] - EXT[0]) * sz, by + sz - (p[1] - EXT[2]) / (EXT[3] - EXT[2]) * sz], inv: (px, py) => [EXT[0] + (px - bx) / sz * (EXT[1] - EXT[0]), EXT[2] + (by + sz - py) / sz * (EXT[3] - EXT[2])] }; }
      function drawPlane() {
        const g = cpl.ctx, { w, h: hh } = cpl, pl = plane(), P = pl.P;
        g.save(); g.setTransform(cpl.dpr, 0, 0, cpl.dpr, 0, 0); g.fillStyle = '#fff'; g.fillRect(0, 0, w, hh);
        const md = S.mode;
        let xNow = S.x;
        if (md === 'plan' && S.plan) { const k = Math.min(S.plan.tr.length - 1, Math.floor(animT * 14)); xNow = S.plan.tr[k].a; }
        // heatmap
        if (md === 'jepa') { const gr = jepaGrid(pred(S.x)); lib.heatmap(g, gr.E, pl.bx, pl.by, pl.sz, pl.sz, { range: [0, gr.q], gamma: 0.8 }); lib.contours(g, gr.E, pl.bx, pl.by, pl.sz, pl.sz, [0.1, 0.3, 0.6, 1, 1.5, 2.1].map(v => v * gr.q / 2.1), { color: 'rgba(17,17,17,0.18)', width: 1 }); }
        else { const gr = ebtGrid(xNow); lib.heatmap(g, gr.E, pl.bx, pl.by, pl.sz, pl.sz, { key: gr.key + '-' + Math.round(pl.sz), range: [gr.lo, gr.q], gamma: 0.7 }); const lv = []; for (let i = 1; i <= 9; i++) lv.push(gr.lo + (gr.q - gr.lo) * Math.pow(i / 9, 1.4)); lib.contours(g, gr.E, pl.bx, pl.by, pl.sz, pl.sz, lv, { color: 'rgba(17,17,17,0.18)', width: 1 }); }
        // axes ticks
        g.strokeStyle = 'rgba(17,17,17,0.12)'; g.lineWidth = 1; g.beginPath(); const o = P([0, 0]); g.moveTo(pl.bx, o[1] + 0.5); g.lineTo(pl.bx + pl.sz, o[1] + 0.5); g.moveTo(o[0] + 0.5, pl.by); g.lineTo(o[0] + 0.5, pl.by + pl.sz); g.stroke();
        // loop of mean next latents (and the changed loop after t = 80)
        const loop = (fn) => { const pts = []; for (let k = 0; k <= 200; k++) pts.push(P(fn(k / 200))); return pts; };
        lib.line(g, loop(mu), { color: INK, width: 1.1, dash: [4, 4], alpha: 0.55 });
        if (md === 'jepa' || md === 'ebt' || md === 'think') {
          const m = mu(S.x), s2 = 2 * sig(S.x), pm = P(m), r = s2 / (EXT[1] - EXT[0]) * pl.sz;
          g.setLineDash([3, 3]); g.strokeStyle = INK; g.lineWidth = 1; g.beginPath(); g.arc(pm[0], pm[1], Math.max(3, r), 0, 7); g.stroke(); g.setLineDash([]);
          const ob = P(obsOf(S.x)); cross(g, ob, INK, 7); tag(g, 'outcome z', ob[0] + 9, ob[1] - 10, INK);
          const pg = P(pred(S.x)); square(g, pg, INK); tag(g, 'guess', pg[0] + 9, pg[1] + 12, INK);
          if (md === 'think' && S.think) drawThink(g, P);
        }
        if (md === 'plan') drawPlan(g, P, pl);
        g.restore();
      }
      function drawThink(g, P) {
        const k = Math.min(S.think.n, Math.floor(animT * 5));
        const seg = (path, col, dash) => { const pts = path.slice(0, k + 1).map(P); halo(pts, col, 2, dash); pts.forEach((p, i) => lib.dot(g, p[0], p[1], i === pts.length - 1 ? 4.5 : 2.4, i === pts.length - 1 ? col : '#fff', { stroke: i === pts.length - 1 ? '#fff' : col, lw: 1.4 })); return pts[pts.length - 1]; };
        const p0 = P(S.y0); g.strokeStyle = GRAY; g.lineWidth = 1.4; g.beginPath(); g.arc(p0[0], p0[1], 4, 0, 7); g.stroke(); tag(g, 'noise ŷ₀', p0[0] + 8, p0[1] + 12, GRAY);
        seg(S.think.b.path, GRAY, [4, 3]);
        const e = seg(S.think.a.path, BLUE); tag(g, 'ẑ*', e[0] - 22, e[1] - 8, BLUE);
      }
      function drawPlan(g, P, pl) {
        // goal
        const pg = P(S.goal); g.setLineDash([3, 3]); g.strokeStyle = INK; g.lineWidth = 1.2; g.beginPath(); g.arc(pg[0], pg[1], 11, 0, 7); g.stroke(); g.setLineDash([]); cross(g, pg, INK, 6); tag(g, 'goal', pg[0] - 44, pg[1] - 14, INK);
        if (!S.plan) return;
        const k = Math.min(PIT, Math.floor(animT * 14)), tr = S.plan.tr, done = k >= PIT;
        // the other candidates, faint
        S.plan.runs.forEach((run, j) => { if (j === S.plan.best) return; const pts = run.slice(0, k + 1).map(s => P(s.p)); poly(g, pts, BLUE, 1.1, null, 0.35); const e = pts[pts.length - 1]; lib.dot(g, e[0], e[1], 2.4, '#fff', { stroke: lib.rgba(BLUE, 0.6), lw: 1.2 }); });
        const pts = tr.slice(0, k + 1).map(s => P(s.p)); halo(pts, BLUE, 2);
        pts.forEach((p, i) => { if (i % 3 === 0 || i === pts.length - 1) lib.dot(g, p[0], p[1], i === pts.length - 1 ? 5 : 2.2, i === pts.length - 1 ? BLUE : '#fff', { stroke: i === pts.length - 1 ? '#fff' : BLUE, lw: 1.4 }); });
        const st = P(tr[0].p); square(g, st, INK); tag(g, 'start', st[0] + 9, st[1] - 8, INK);
        if (pts.length > 1) { const e = pts[pts.length - 1]; tag(g, 'ẑ', e[0] + 8, e[1] + 4, BLUE); }
        if (done) { // verification: the model's own prediction at the chosen action
          const zp = P(S.plan.zs), e = pts[pts.length - 1];
          g.setLineDash([2, 3]); g.strokeStyle = BLUE; g.lineWidth = 1; g.beginPath(); g.moveTo(e[0], e[1]); g.lineTo(zp[0], zp[1]); g.stroke(); g.setLineDash([]);
          g.fillStyle = '#fff'; g.strokeStyle = BLUE; g.lineWidth = 1.6; g.beginPath(); g.arc(zp[0], zp[1], 5, 0, 7); g.fill(); g.stroke();
          tag(g, 'ẑ*(a)', zp[0] - 44, zp[1] + 16, BLUE);
        }
        // action dial: faint ticks = the M starts, gray = winning start, blue = current action
        const a = tr[k].a, cx = pl.bx + pl.sz - 34, cy = pl.by + 34, r = 22, ang = (v) => [cx + r * Math.sin(TAU * v), cy - r * Math.cos(TAU * v)];
        g.fillStyle = 'rgba(255,255,255,0.9)'; g.fillRect(cx - 30, cy - 30, 60, 74);
        g.strokeStyle = INK; g.lineWidth = 1; g.beginPath(); g.arc(cx, cy, r, 0, 7); g.stroke();
        S.plan.runs.forEach((run) => { const q = ang(run[0].a), q2 = [cx + (r - 5) * Math.sin(TAU * run[0].a), cy - (r - 5) * Math.cos(TAU * run[0].a)]; g.strokeStyle = FAINT; g.beginPath(); g.moveTo(q2[0], q2[1]); g.lineTo(q[0], q[1]); g.stroke(); });
        const a0 = ang(tr[0].a); g.strokeStyle = GRAY; g.beginPath(); g.moveTo(cx, cy); g.lineTo(a0[0], a0[1]); g.stroke();
        const a1 = ang(a); g.strokeStyle = BLUE; g.lineWidth = 2; g.beginPath(); g.moveTo(cx, cy); g.lineTo(a1[0], a1[1]); g.stroke();
        lib.text(g, 'a = ' + a.toFixed(3), cx, cy + r + 6, { size: 10.5, kind: 'mono', color: BLUE, align: 'center' });
      }
      function halo(pts, col, wdt, dash) { if (pts.length < 2) return; lib.line(g0(), pts, { color: 'rgba(255,255,255,0.9)', width: wdt + 3 }); lib.line(g0(), pts, { color: col, width: wdt, dash }); }
      function g0() { return cpl.ctx; }
      function cross(g, p, col, r) { g.strokeStyle = col; g.lineWidth = 1.5; g.beginPath(); g.moveTo(p[0] - r, p[1]); g.lineTo(p[0] + r, p[1]); g.moveTo(p[0], p[1] - r); g.lineTo(p[0], p[1] + r); g.stroke(); }
      function square(g, p, col) { g.fillStyle = '#fff'; g.fillRect(p[0] - 5, p[1] - 5, 10, 10); g.strokeStyle = col; g.lineWidth = 1.6; g.strokeRect(p[0] - 5, p[1] - 5, 10, 10); }
      function tag(g, t, x, y, col) {
        g.save(); g.font = `400 11px ${MONO}`; const tw = g.measureText(t).width; x = clamp(x, 2, cpl.w - tw - 6); y = clamp(y, 12, cpl.h - 4);
        g.fillStyle = 'rgba(255,255,255,0.82)'; g.fillRect(x - 3, y - 11, tw + 6, 15); g.fillStyle = col; g.textBaseline = 'alphabetic'; g.fillText(t, x, y); g.restore();
      }
      const poly = (g, pts, col, wdt, dash, alpha) => { if (pts.length < 2) return; g.save(); g.globalAlpha = alpha == null ? 1 : alpha; g.strokeStyle = col; g.lineWidth = wdt; g.setLineDash(dash || []); g.beginPath(); pts.forEach((p, i) => i ? g.lineTo(p[0], p[1]) : g.moveTo(p[0], p[1])); g.stroke(); g.restore(); };
      function readouts() {
        const md = S.mode, f3 = (v) => (v < 0 ? '−' : '') + Math.abs(v).toFixed(3);
        if (md === 'jepa' || md === 'ebt' || md === 'think') {
          const zb = toy.ctxBias(S.x), pg = pred(S.x), m = mu(S.x), eg = toy.fwd(zb, pg[0], pg[1]), th = descend(S.x, pg, 1, 12), emin = th.Es[th.Es.length - 1];
          const err = Math.hypot(pg[0] - m[0], pg[1] - m[1]);
          if (md === 'jepa') ro.innerHTML = `<span>x <b>${S.x.toFixed(3)}</b></span><span>σ(x) <b>${sig(S.x).toFixed(2)}</b></span><span>E<sub>JEPA</sub>(guess) <b>0</b> always</span><span>guess error <b>${err.toFixed(2)}</b>, invisible to it</span>`;
          else if (md === 'ebt') ro.innerHTML = `<span>x <b>${S.x.toFixed(3)}</b></span><span>E<sub>ψ</sub>(guess) <b>${f3(eg)}</b></span><span>basin floor <b>${f3(emin)}</b></span><span>gap <b>${(eg - emin).toFixed(3)}</b></span><span>guess error <b>${err.toFixed(2)}</b></span>`;
          else { const T = S.think; if (!T) { ro.innerHTML = ''; return; } const k = Math.min(T.n, Math.floor(animT * 5)), pa = T.a.path[k], pb = T.b.path[k]; ro.innerHTML = `<span>step <b>${k}</b></span><span>from guess: E <b>${f3(T.a.Es[k])}</b>, ‖ẑ − μ‖ <b>${Math.hypot(pa[0] - m[0], pa[1] - m[1]).toFixed(3)}</b></span><span>from noise: E <b>${f3(T.b.Es[k])}</b>, ‖ẑ − μ‖ <b>${Math.hypot(pb[0] - m[0], pb[1] - m[1]).toFixed(3)}</b></span>` + (k === T.n ? (() => { const hit = (P) => { const j = P.findIndex(q => Math.hypot(q[0] - m[0], q[1] - m[1]) < 0.05); return j < 0 ? '> ' + T.n : String(j); }; return `<span>within 0.05 of μ: guess at step <b>${hit(T.a.path)}</b>, noise at step <b>${hit(T.b.path)}</b></span>`; })() : ''); }
        } else if (md === 'plan') {
          if (!S.plan) { ro.innerHTML = ''; return; } const tr = S.plan.tr, k = Math.min(PIT, Math.floor(animT * 14)), s = tr[k], done = k >= PIT;
          ro.innerHTML = `<span>iteration <b>${k}</b></span><span>action a <b>${s.a.toFixed(3)}</b> (start ${tr[0].a.toFixed(3)}, best of ${MP})</span><span>E <b>${f3(s.E)}</b></span><span>‖ẑ − goal‖ <b>${s.d.toFixed(2)}</b></span>` +
            (done ? `<span>verify: ‖ẑ*(a) − goal‖ <b>${S.plan.re.toFixed(2)}</b></span>` : '') + `<span>≈ <b>${nfmt(3 * MP * k + (done ? 36 : 0))}</b> forward-pass equivalents</span>`;
        }
      }

      // ---------- animation ----------
      const loop = lib.loop((dt) => {
        if (!visible) return false;
        let more = false;
        if ((S.mode === 'think' && S.think && animT * 5 < S.think.n + 1) || (S.mode === 'plan' && S.plan && animT * 14 < PIT + 1)) { animT += dt; more = true; }
        draw(); return more;
      });
      function kick() { if (visible) loop.start(); }

      // ---------- dragging the goal ----------
      let drag = false;
      const loc = (ev) => { const r = cpl.c.getBoundingClientRect(); return plane().inv((ev.clientX - r.left) / r.width * cpl.w, (ev.clientY - r.top) / r.height * cpl.h); };
      cpl.c.addEventListener('pointerdown', (ev) => { if (S.mode !== 'plan') return; const p = loc(ev); if (Math.hypot(p[0] - S.goal[0], p[1] - S.goal[1]) < 0.45) { drag = true; try { cpl.c.setPointerCapture(ev.pointerId); } catch (e) { /* ignore */ } ev.preventDefault(); } });
      cpl.c.addEventListener('pointermove', (ev) => { if (S.mode !== 'plan') { cpl.c.style.cursor = 'default'; return; } const p = loc(ev); if (!drag) { cpl.c.style.cursor = Math.hypot(p[0] - S.goal[0], p[1] - S.goal[1]) < 0.45 ? 'grab' : 'default'; return; } S.goal = [clamp(p[0], -2.1, 2.1), clamp(p[1], -2.1, 2.1)]; runPlan(); animT = 1e9; draw(); });
      const end = () => { if (drag) { drag = false; runPlan(); } };
      cpl.c.addEventListener('pointerup', end); cpl.c.addEventListener('pointercancel', end);

      // ---------- modes / steps ----------
      function setMode(m, layer) {
        S.mode = m; diag.setLayer(layer);
        cThink.hidden = !(m === 'think' || m === 'jepa' || m === 'ebt'); bThink.hidden = slA.el.hidden = m !== 'think';
        cPlan.hidden = m !== 'plan';
        Fl.wrap.querySelector('.fig-sub').textContent = m === 'jepa' ? 'shading: E_JEPA, a bowl at the guess' : m === 'plan' ? 'shading: E_ψ at the current action a' : 'shading: learned E_ψ over ẑ';
        ctx.setCaption(CAP[m]);
      }
      const CAP = {
        jepa: 'Darker = lower energy. Dashed loop: mean next latent of every context; dashed circle: ±2σ of outcomes here; crosshair: one outcome; square: the weak predictor\'s guess.',
        ebt: 'Shaded by the toy EBT\'s energy for this context. Floor: energy after 12 thinking steps from the guess; gap: E(guess) minus the floor.',
        think: 'Blue: thinking from the predictor\'s guess; gray, dashed: from noise. Dots are steps; ẑ* is where the blue run ends.',
        plan: 'Bold: best of 6 joint descents on (a, ẑ); faint: other starts. Hollow circle: the model\'s own prediction at the chosen a (verification). Dial ticks: start actions.',
      };
      const STEPS = [
        () => { setMode('jepa', 0); draw(); },
        () => { setMode('ebt', 1); draw(); },
        () => { setMode('think', 1); runThink(); },
        () => { setMode('plan', 2); runPlan(); },
      ];
      setMode('jepa', 0);
      (document.fonts && document.fonts.ready ? document.fonts.ready : Promise.resolve()).then(() => draw());
      draw();
      return {
        step(i) { (STEPS[i] || STEPS[0])(); },
        show() { visible = true; cpl.fit(); draw(); kick(); },
        hide() { visible = false; loop.stop(); },
      };

      // ================= helpers (hoisted) =================
      function autoCanvas(parent, o) {
        const box = h('div', { class: 'canvas-box' }); parent.appendChild(box);
        const c = h('canvas', { role: 'img', 'aria-label': o.label || '' }); box.appendChild(c);
        const r = { c, ctx: c.getContext('2d'), w: 0, h: 0, box, dpr: 1 };
        r.fit = () => {
          const w = Math.max(200, Math.round(box.clientWidth || 300)), hh = Math.round(clamp(w * o.aspect, o.minH || 0, o.maxH || 1e9)), dpr = Math.min(2, window.devicePixelRatio || 1);
          if (w === r.w && hh === r.h && dpr === r.dpr) return false;
          r.w = w; r.h = hh; r.dpr = dpr; c.width = Math.round(w * dpr); c.height = Math.round(hh * dpr); c.style.height = hh + 'px'; return true;
        };
        r.fit();
        if (window.ResizeObserver) new ResizeObserver(() => { if (r.fit() && o.draw) o.draw(); }).observe(box);
        return r;
      }
    },
  });

  // ===================================================================================================
  // toy EBT evaluator (same weights and formula as EBT.toy2d), with the gradient w.r.t. the context phase x as well
  // ===================================================================================================
  function makeToy(D) {
    const ck = D.weights.length - 1, LAY = D.weights[ck].layers;
    const LAM = (D.arch.quadratic_term && D.arch.quadratic_term.lambda_) || 0.02, KF = (D.arch.feature_map && D.arch.feature_map.K) || 3, NF = 2 * KF;
    const L = LAY.map(l => ({ W: Float64Array.from([].concat(...l.W)), b: Float64Array.from(l.b), nOut: l.W.length, nIn: l.W[0].length }));
    const nL = L.length, ZS = L.map(l => new Float64Array(l.nOut)), HS = L.map(l => new Float64Array(l.nOut)), GH = L.map(l => new Float64Array(l.nOut));
    const sg = (z) => 1 / (1 + Math.exp(-z));
    function ctxBias(x) { const l = L[0], zb = new Float64Array(l.nOut), f = []; for (let k = 1; k <= KF; k++) f.push(Math.sin(TAU * k * x), Math.cos(TAU * k * x)); for (let i = 0; i < l.nOut; i++) { let s = l.b[i]; for (let j = 0; j < NF; j++) s += l.W[i * l.nIn + j] * f[j]; zb[i] = s; } return zb; }
    function fwd(zb, a, c) {
      const l0 = L[0];
      for (let i = 0; i < l0.nOut; i++) { const z = zb[i] + l0.W[i * l0.nIn + NF] * a + l0.W[i * l0.nIn + NF + 1] * c; ZS[0][i] = z; HS[0][i] = z * sg(z); }
      let out = 0;
      for (let k = 1; k < nL; k++) { const l = L[k], hin = HS[k - 1]; for (let i = 0; i < l.nOut; i++) { let s = l.b[i]; const base = i * l.nIn; for (let j = 0; j < l.nIn; j++) s += l.W[base + j] * hin[j]; if (k < nL - 1) { ZS[k][i] = s; HS[k][i] = s * sg(s); } else out = s; } }
      return out + LAM * (a * a + c * c);
    }
    const G = { a: 0, c: 0, x: 0 };
    function fwdGrad(zb, a, c, x) {
      const E = fwd(zb, a, c), last = L[nL - 1], g0 = GH[nL - 2]; for (let j = 0; j < last.nIn; j++) g0[j] = last.W[j];
      for (let k = nL - 2; k >= 0; k--) {
        const z = ZS[k], g = GH[k]; for (let i = 0; i < g.length; i++) { const s = sg(z[i]); g[i] = g[i] * s * (1 + z[i] * (1 - s)); }
        const l = L[k];
        if (k > 0) { const gp = GH[k - 1]; gp.fill(0); for (let i = 0; i < l.nOut; i++) { const gi = g[i], base = i * l.nIn; for (let j = 0; j < l.nIn; j++) gp[j] += l.W[base + j] * gi; } }
        else {
          let s0 = 0, s1 = 0; for (let i = 0; i < l.nOut; i++) { s0 += l.W[i * l.nIn + NF] * g[i]; s1 += l.W[i * l.nIn + NF + 1] * g[i]; } G.a = s0 + 2 * LAM * a; G.c = s1 + 2 * LAM * c;
          if (x != null) { let gx = 0; for (let kk = 1; kk <= KF; kk++) { const wv = TAU * kk; let gs = 0, gc = 0; for (let i = 0; i < l.nOut; i++) { gs += l.W[i * l.nIn + 2 * (kk - 1)] * g[i]; gc += l.W[i * l.nIn + 2 * (kk - 1) + 1] * g[i]; } gx += gs * wv * Math.cos(wv * x) - gc * wv * Math.sin(wv * x); } G.x = gx; }
        }
      }
      return E;
    }
    return { ctxBias, fwd, fwdGrad, G };
  }

  // ===================================================================================================
  // diagram: one SVG per layout (wide / narrow), layers 0 JEPA, 1 energy head, 2 planner
  // ===================================================================================================
  function buildDiagram(parent) {
    const NS = 'http://www.w3.org/2000/svg';
    const el = (tag, attrs, kids) => { const e = document.createElementNS(NS, tag); Object.entries(attrs || {}).forEach(([k, v]) => e.setAttribute(k, v)); (kids || []).forEach(k => e.appendChild(k)); return e; };
    // text with _{..} subscripts
    const txt = (s, attrs) => { const t = el('text', attrs); String(s).split(/(_\{[^}]*\})/).forEach(part => { if (!part) return; const m = /^_\{(.*)\}$/.exec(part); if (m) { t.appendChild(el('tspan', { dy: 3, 'font-size': '0.78em' }, [document.createTextNode(m[1])])); t.appendChild(el('tspan', { dy: -3 }, [document.createTextNode('​')])); } else t.appendChild(document.createTextNode(part)); }); return t; };
    const N = { // [layer, kind, label, sub, wide [x,y,w,h], narrow [x,y,w,h]]
      goal: [2, 'pill', 'goal', '', [168, 7, 52, 22], [6, 17, 46, 22]],
      planner: [2, 'box', 'PLANNER', 'CEM, or ∇ through E', [246, 0, 150, 36], [66, 8, 176, 40]],
      a: [0, 'pill', 'a_{t}', '', [301, 43, 40, 16], [126, 58, 52, 22]],
      ot: [0, 'img', 'o_{t}', '', [8, 70, 34, 34], [6, 98, 34, 34]],
      enc: [0, 'box', 'ENCODER', 'f_{θ} · ViT', [62, 66, 108, 42], [66, 94, 176, 42]],
      zt: [0, 'pill', 'z_{t}', 'sparse', [188, 76, 42, 22], [130, 146, 46, 22]],
      pred: [0, 'box', 'PREDICTOR', 'g_{φ} · AdaLN-zero', [246, 66, 150, 42], [66, 178, 176, 42]],
      zh: [0, 'pill', 'ẑ_{t+1}', '', [416, 76, 56, 22], [124, 232, 58, 22]],
      head: [1, 'box', 'ENERGY HEAD', 'E_{ψ}(z_{t}, a_{t}, ẑ) → scalar', [486, 118, 148, 42], [66, 290, 176, 42]],
      ot1: [0, 'img', 'o_{t+1}', '', [8, 184, 34, 34], [6, 376, 34, 34]],
      tenc: [0, 'box', 'TARGET ENC.', 'EMA · stop-grad', [62, 180, 108, 42], [66, 372, 176, 42]],
      z1: [0, 'pill', 'z_{t+1}', '', [188, 190, 52, 22], [252, 382, 58, 22]],
      loss: [0, 'box', 'JEPA LOSS', '‖ẑ − sg(z)‖²', [276, 180, 156, 42], [196, 232, 138, 42]],
    };
    // edges: [layer, wide points, narrow points, label (wide), label (narrow), loop?]
    const E = [
      [0, [[42, 87], [62, 87]], [[40, 115], [66, 115]]],
      [0, [[170, 87], [188, 87]], [[154, 136], [154, 146]]],
      [0, [[230, 87], [246, 87]], [[154, 168], [154, 178]]],
      [0, [[396, 87], [416, 87]], [[154, 220], [154, 232]]],
      [0, [[321, 59], [321, 66]], [[178, 69], [262, 69], [262, 199], [242, 199]]],
      [0, [[42, 201], [62, 201]], [[40, 393], [66, 393]]],
      [0, [[170, 201], [188, 201]], [[242, 393], [252, 393]]],
      [0, [[240, 201], [276, 201]], [[296, 382], [296, 274]]],
      [0, [[444, 98], [444, 168], [354, 168], [354, 180]], [[182, 243], [196, 243]], 'compare', ''],
      [1, [[472, 87], [560, 87], [560, 118]], [[154, 254], [154, 290]]],
      [1, [[610, 118], [610, 58], [444, 58], [444, 76]], [[66, 311], [36, 311], [36, 243], [124, 243]], '−α∇_{ẑ}E (think)', '−α∇E', true],
      [2, [[634, 139], [640, 139], [640, 18], [396, 18]], [[242, 311], [324, 311], [324, 28], [242, 28]], '∇_{a}E · scores', '', false, [520, 13, 'middle']],
      [2, [[220, 18], [246, 18]], [[52, 28], [66, 28]]],
      [2, [[321, 36], [321, 43]], [[154, 48], [154, 58]]],
    ];
    const mk = (lay) => {
      const vb = lay === 'wide' ? [0, -4, 646, 248] : [0, 0, 340, 424];
      const svg = el('svg', { viewBox: vb.join(' '), class: 'wm-diag wm-' + lay, role: 'img', 'aria-label': 'Diagram: a JEPA world model (encoder, action-conditioned predictor, EMA target encoder and loss) with an energy head that scores predicted latents and a planner that uses its gradient' });
      svg.appendChild(el('defs', {}, [el('marker', { id: 'wm-ar-' + lay, viewBox: '0 0 8 8', refX: 7, refY: 4, markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse' }, [el('path', { d: 'M0,0 L8,4 L0,8 z', fill: 'context-stroke' })])]));
      const gE = el('g'), gN = el('g'); svg.append(gE, gN);
      E.forEach(([L, pw, pn, lw, ln, isLoop, posW, posN]) => {
        const pts = lay === 'wide' ? pw : pn, lab = lay === 'wide' ? lw : ln;
        const g = el('g', { class: 'wm-l wm-l' + L });
        g.appendChild(el('polyline', { points: pts.map(p => p.join(',')).join(' '), fill: 'none', 'marker-end': `url(#wm-ar-${lay})`, class: 'wm-edge' + (isLoop ? ' wm-loop' : '') }));
        const pos = lay === 'wide' ? posW : posN;
        if (lab && pos) g.appendChild(txt(lab, { x: pos[0], y: pos[1], class: 'wm-elab', 'text-anchor': pos[2] }));
        else if (lab) { const a = pts[Math.floor((pts.length - 1) / 2)], b = pts[Math.floor((pts.length - 1) / 2) + 1]; g.appendChild(txt(lab, { x: (a[0] + b[0]) / 2 + (a[1] === b[1] ? 0 : 6), y: (a[1] + b[1]) / 2 - (a[1] === b[1] ? 5 : -3), class: 'wm-elab', 'text-anchor': a[1] === b[1] ? 'middle' : 'start' })); }
        gE.appendChild(g);
      });
      Object.entries(N).forEach(([id, [L, kind, label, sub, rw, rn]]) => {
        const [x, y, w, hh] = lay === 'wide' ? rw : rn, g = el('g', { class: 'wm-l wm-l' + L + ' wm-n-' + kind });
        if (kind === 'box') {
          g.appendChild(el('rect', { x, y, width: w, height: hh, class: 'wm-box' }));
          [[x - 4, y - 4], [x + w + 4, y - 4], [x - 4, y + hh + 4], [x + w + 4, y + hh + 4]].forEach(([cx, cy]) => g.appendChild(el('text', { x: cx, y: cy + 3, class: 'wm-plus', 'text-anchor': 'middle' }, [document.createTextNode('+')])));
          g.appendChild(txt(label, { x: x + 8, y: y + 16, class: 'wm-lab' }));
          if (sub) g.appendChild(txt(sub, { x: x + 8, y: y + 32, class: 'wm-sub' }));
        } else if (kind === 'pill') {
          g.appendChild(el('rect', { x, y, width: w, height: hh, rx: hh / 2, class: 'wm-pill' }));
          g.appendChild(txt(label, { x: x + w / 2, y: y + hh / 2 + 4, class: 'wm-var', 'text-anchor': 'middle' }));
        } else { // observation frame: hatched square
          g.appendChild(el('rect', { x, y, width: w, height: hh, class: 'wm-img' }));
          for (let d = 6; d < w + hh; d += 6) { const x1 = x + Math.max(0, d - hh), y1 = y + Math.min(hh, d), x2 = x + Math.min(w, d), y2 = y + Math.max(0, d - w); g.appendChild(el('line', { x1, y1, x2, y2, class: 'wm-hatch' })); }
          g.appendChild(txt(label, { x: x + w / 2, y: y + hh + 12, class: 'wm-var', 'text-anchor': 'middle' }));
        }
        gN.appendChild(g);
      });
      return svg;
    };
    const wrap = document.createElement('div'); wrap.className = 'wm-diag-wrap'; parent.appendChild(wrap);
    const wide = mk('wide'), narrow = mk('narrow'); wrap.append(wide, narrow);
    return { setLayer(k) { [wide, narrow].forEach(s => { s.querySelectorAll('.wm-l').forEach(g => { const L = +g.getAttribute('class').match(/wm-l(\d)/)[1]; g.classList.toggle('off', L > k); g.classList.toggle('new', L === k && k > 0); }); }); } };
  }

})();
