/* Chapter: Planning and MPC with energy. BEYOND THE PAPER, grounded only in App. A.3 (p.26).
   A 2-D point robot plans by gradient descent on a hand-written energy over (actions, imagined states), with exact analytic
   gradients, Best-of-M self-verification, Langevin noise, warm starts and receding-horizon MPC.
   The numerical core (Core) is pure and also loads in Node for tests: require('assets/js/sections/planning.js'). */
(function (root) {
  'use strict';

  // =====================================================================================================
  // Core: world, energy + exact gradient, gradient-descent "thinking", MPC simulator. No DOM.
  // =====================================================================================================
  const Core = (function () {
    const WX = 10, WY = 6, REACH = 0.4;
    // energy weights that the user does not change (w_dyn is a slider)
    const W = { wg: 6, wr: 0.05, wo: 25, m: 0.35, we: 0.15, ws: 1, vmax: 0.8, wv: 20 };
    const CLIP = 0.6; // safety: no 2-D variable moves more than this per iteration

    function rng(seed) { // mulberry32 + Box-Muller (same generator as the video engine)
      let a = seed >>> 0;
      const f = () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
      f.normal = () => { const u = Math.max(1e-9, f()), v = f(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
      return f;
    }
    // obstacle + wall penalty for one point: sum of max(0, r + m - dist)^2. Adds d/dp to g, returns the cost.
    function obsCost(world, x, y, m, g) {
      let c = 0;
      for (const o of world.obs) {
        const dx = x - o.x, dy = y - o.y, d = Math.sqrt(dx * dx + dy * dy) + 1e-9, pen = o.r + m - d;
        if (pen > 0) { c += pen * pen; const k = -2 * pen / d; g[0] += k * dx; g[1] += k * dy; }
      }
      let p;
      if ((p = m - x) > 0) { c += p * p; g[0] -= 2 * p; }
      if ((p = x - (WX - m)) > 0) { c += p * p; g[0] += 2 * p; }
      if ((p = m - y) > 0) { c += p * p; g[1] -= 2 * p; }
      if ((p = y - (WY - m)) > 0) { c += p * p; g[1] += 2 * p; }
      return c;
    }
    const hub = (dx, dy) => Math.sqrt(dx * dx + dy * dy + 1) - 1; // pseudo-Huber distance
    // positions P_0..P_H. joint: imagined states S; shoot: model rollout of A
    function positions(s0, plan, P) {
      const H = P.H, X = new Float64Array(2 * (H + 1)); X[0] = s0[0]; X[1] = s0[1];
      for (let t = 1; t <= H; t++) {
        if (P.mode === 'joint') { X[2 * t] = plan.S[2 * t - 2]; X[2 * t + 1] = plan.S[2 * t - 1]; }
        else { X[2 * t] = X[2 * t - 2] + plan.A[2 * t - 2]; X[2 * t + 1] = X[2 * t - 1] + plan.A[2 * t - 1]; }
      }
      return X;
    }
    function rollout(s0, A, H) { // what the actions do under the planner's model s' = s + a
      const X = new Float64Array(2 * (H + 1)); X[0] = s0[0]; X[1] = s0[1];
      for (let t = 1; t <= H; t++) { X[2 * t] = X[2 * t - 2] + A[2 * t - 2]; X[2 * t + 1] = X[2 * t - 1] + A[2 * t - 1]; }
      return X;
    }
    // E(x, ŷ) and its exact gradient, written into preallocated buffers (hot path). x = (s0, goal, map), ŷ = plan {A, S}
    function energyCore(world, s0, plan, P, B) {
      const H = P.H, X = B.X, gX = B.gX, gA = B.gA, gS = B.gS, T = B.T, A = plan.A, joint = P.mode === 'joint';
      X[0] = s0[0]; X[1] = s0[1];
      for (let t = 1; t <= H; t++) {
        if (joint) { X[2 * t] = plan.S[2 * t - 2]; X[2 * t + 1] = plan.S[2 * t - 1]; }
        else { X[2 * t] = X[2 * t - 2] + A[2 * t - 2]; X[2 * t + 1] = X[2 * t - 1] + A[2 * t - 1]; }
      }
      gX.fill(0); gA.fill(0); gS.fill(0); T.goal = T.obs = T.eff = T.smooth = T.dyn = 0;
      const gx = world.goal[0], gy = world.goal[1];
      for (let t = 1; t <= H; t++) { // goal: pseudo-Huber pull on every state (w_r) plus the last state (w_g)
        const w = t === H ? P.wr + P.wg : P.wr, dx = X[2 * t] - gx, dy = X[2 * t + 1] - gy, q = Math.sqrt(dx * dx + dy * dy + 1);
        T.goal += w * (q - 1); gX[2 * t] += w * dx / q; gX[2 * t + 1] += w * dy / q;
      }
      const g = B.g;
      for (let t = 1; t <= H; t++) { // obstacles at each state and each segment midpoint
        g[0] = g[1] = 0; T.obs += P.wo * obsCost(world, X[2 * t], X[2 * t + 1], P.m, g);
        gX[2 * t] += P.wo * g[0]; gX[2 * t + 1] += P.wo * g[1];
        const mx = 0.5 * (X[2 * t - 2] + X[2 * t]), my = 0.5 * (X[2 * t - 1] + X[2 * t + 1]);
        g[0] = g[1] = 0; T.obs += P.wo * obsCost(world, mx, my, P.m, g);
        gX[2 * t] += 0.5 * P.wo * g[0]; gX[2 * t + 1] += 0.5 * P.wo * g[1];
        gX[2 * t - 2] += 0.5 * P.wo * g[0]; gX[2 * t - 1] += 0.5 * P.wo * g[1];
      }
      for (let i = 0; i < 2 * H; i++) { T.eff += P.we * A[i] * A[i]; gA[i] += 2 * P.we * A[i]; }
      for (let t = 0; t < H; t++) { // speed limit |a| <= vmax, counted as effort
        const ax = A[2 * t], ay = A[2 * t + 1], n = Math.sqrt(ax * ax + ay * ay) + 1e-9, p = n - P.vmax;
        if (p > 0) { T.eff += P.wv * p * p; const k = 2 * P.wv * p / n; gA[2 * t] += k * ax; gA[2 * t + 1] += k * ay; }
      }
      for (let t = 1; t < H; t++) for (let k = 0; k < 2; k++) {
        const d = A[2 * t + k] - A[2 * t - 2 + k]; T.smooth += P.ws * d * d; gA[2 * t + k] += 2 * P.ws * d; gA[2 * t - 2 + k] -= 2 * P.ws * d;
      }
      if (joint) {
        for (let t = 0; t < H; t++) for (let k = 0; k < 2; k++) {
          const r = X[2 * (t + 1) + k] - X[2 * t + k] - A[2 * t + k];
          T.dyn += P.wd * r * r; gX[2 * (t + 1) + k] += 2 * P.wd * r; gX[2 * t + k] -= 2 * P.wd * r; gA[2 * t + k] -= 2 * P.wd * r;
        }
        for (let i = 0; i < 2 * H; i++) gS[i] = gX[i + 2];
      } else { // backprop through the rollout: dP_t/da_u = I for every t > u
        let cx = 0, cy = 0;
        for (let u = H - 1; u >= 0; u--) { cx += gX[2 * (u + 1)]; cy += gX[2 * (u + 1) + 1]; gA[2 * u] += cx; gA[2 * u + 1] += cy; }
      }
      return T.goal + T.obs + T.eff + T.smooth + T.dyn;
    }
    const newBuf = (H) => ({ X: new Float64Array(2 * (H + 1)), gX: new Float64Array(2 * (H + 1)), gA: new Float64Array(2 * H), gS: new Float64Array(2 * H), g: [0, 0], T: { goal: 0, obs: 0, eff: 0, smooth: 0, dyn: 0 } });
    const bufs = new Map(); const sharedBuf = (H) => { let b = bufs.get(H); if (!b) { b = newBuf(H); bufs.set(H, b); } return b; };
    // public version: fresh buffers, safe to keep. Returns {E, T (per term), gA, gS, X}
    function energy(world, s0, plan, P) { const B = newBuf(P.H); const E = energyCore(world, s0, plan, P, B); return { E, T: B.T, gA: B.gA, gS: B.gS, X: B.X }; }
    // energy only (no allocation), for scoring candidates
    function energyOnly(world, s0, plan, P) { return energyCore(world, s0, plan, P, sharedBuf(P.H)); }
    // ŷ0 ~ N(0, sd² I) for the actions; imagined states start as the model rollout of those actions
    function initPlan(s0, H, r, sd) {
      const A = new Float64Array(2 * H); for (let i = 0; i < 2 * H; i++) A[i] = sd * r.normal();
      return { A, S: rollout(s0, A, H).slice(2) };
    }
    // one thinking step (Eq. 1, plus Langevin noise as in Eq. 2): ŷ ← ŷ − α∇E + η
    function gdStep(world, s0, plan, P, alpha, sigma, r) {
      const B = sharedBuf(P.H), E = energyCore(world, s0, plan, P, B);
      const upd = (V, G) => {
        for (let i = 0; i < V.length; i += 2) {
          let dx = -alpha * G[i], dy = -alpha * G[i + 1]; const n = Math.hypot(dx, dy);
          if (n > CLIP) { dx *= CLIP / n; dy *= CLIP / n; }
          V[i] += dx + (sigma > 0 ? sigma * r.normal() : 0); V[i + 1] += dy + (sigma > 0 ? sigma * r.normal() : 0);
        }
      };
      upd(plan.A, B.gA); if (P.mode === 'joint') upd(plan.S, B.gS);
      return E;
    }
    function shift(plan, H) { // receding horizon: drop the executed step, coast at the end
      for (let i = 0; i < 2 * (H - 1); i++) { plan.A[i] = plan.A[i + 2]; plan.S[i] = plan.S[i + 2]; }
      plan.A[2 * H - 2] = 0; plan.A[2 * H - 1] = 0;
    }
    function segHits(world, ax, ay, bx, by) { // true collision test (no margin)
      if (bx < 0 || bx > WX || by < 0 || by > WY) return true;
      for (const o of world.obs) {
        const vx = bx - ax, vy = by - ay, L = vx * vx + vy * vy;
        let t = L > 0 ? ((o.x - ax) * vx + (o.y - ay) * vy) / L : 0; t = Math.max(0, Math.min(1, t));
        const px = ax + t * vx - o.x, py = ay + t * vy - o.y; if (px * px + py * py < o.r * o.r) return true;
      }
      return false;
    }
    // largest Hessian eigenvalue of E at a plan (power iteration on Hessian-vector products of the exact gradient)
    function lambdaMax(world, s0, plan, P, iters) {
      const H = P.H, n = P.mode === 'joint' ? 4 * H : 2 * H, h = 1e-5;
      const x0 = new Float64Array(n); x0.set(plan.A); if (P.mode === 'joint') x0.set(plan.S, 2 * H);
      const tmp = { A: new Float64Array(2 * H), S: new Float64Array(2 * H) };
      const grad = (x) => { tmp.A.set(x.subarray(0, 2 * H)); if (P.mode === 'joint') tmp.S.set(x.subarray(2 * H)); else tmp.S.set(plan.S); const e = energy(world, s0, tmp, P, true); const g = new Float64Array(n); g.set(e.gA); if (P.mode === 'joint') g.set(e.gS, 2 * H); return g; };
      const r = rng(99); let v = new Float64Array(n).map(() => r() - 0.5), lam = 0;
      const xp = new Float64Array(n), xm = new Float64Array(n);
      for (let k = 0; k < (iters || 30); k++) {
        let nv = 0; for (let i = 0; i < n; i++) nv += v[i] * v[i]; nv = Math.sqrt(nv) || 1;
        for (let i = 0; i < n; i++) { v[i] /= nv; xp[i] = x0[i] + h * v[i]; xm[i] = x0[i] - h * v[i]; }
        const gp = grad(xp), gm = grad(xm); const Hv = new Float64Array(n); lam = 0;
        for (let i = 0; i < n; i++) { Hv[i] = (gp[i] - gm[i]) / (2 * h); lam += Hv[i] * v[i]; }
        v = Hv;
      }
      return lam;
    }
    const params = (cfg) => Object.assign({ mode: cfg.vars, H: cfg.H, wd: cfg.wd }, W);

    // ---------- presets ----------
    const cup = []; for (let a = -90; a <= 90; a += 30) { const t = a * Math.PI / 180; cup.push({ x: +(5 + 1.7 * Math.cos(t)).toFixed(2), y: +(3 + 1.7 * Math.sin(t)).toFixed(2), r: 0.7 }); }
    const PRESETS = {
      pillar: { label: 'Pillar: two equal routes', world: { start: [1.2, 3], goal: [8.8, 3], obs: [{ x: 5, y: 3, r: 1.3 }] }, cfg: { M: 6, seed: 3 },
        hint: 'One pillar sits on the straight line. Going over and going under are equally good, so candidates split between two valleys of the energy.' },
      cup: { label: 'Cup: a local-minimum trap', world: { start: [1.2, 3], goal: [8.8, 3], obs: cup }, cfg: { M: 1, seed: 3 },
        hint: 'A cup opens toward the robot. Gradient descent from most starts slides into it and stops: a local minimum. Raise the number of candidates M.' },
      gap: { label: 'Gusty gap: wind the model ignores', world: { start: [1.2, 3], goal: [8.8, 3], obs: [{ x: 5, y: 4.45, r: 1.05 }, { x: 5, y: 1.55, r: 1.05 }] }, cfg: { wind: 0.1, M: 4, seed: 2 },
        hint: 'A steady wind pushes the robot down by 0.1 per step. The planner’s model does not know about it. Compare open loop with MPC.' },
      slalom: { label: 'Slalom: think a little, often', world: { start: [1.2, 3], goal: [8.8, 3], obs: [{ x: 3.3, y: 2.4, r: 0.8 }, { x: 5, y: 3.8, r: 0.8 }, { x: 6.7, y: 2.3, r: 0.8 }] }, cfg: { K: 3, M: 4, seed: 1 },
        hint: 'Only 3 thinking steps per move. With warm start each decision continues the previous plan, so that is enough. Turn warm start off to see why it matters.' },
    };
    const DEFAULTS = { mode: 'mpc', vars: 'joint', H: 12, alpha: 0.009, N0: 2000, K: 30, sigma: 0, M: 6, wd: 10, wind: 0, warm: true, adaptive: false, sd: 1, maxSteps: 40, seed: 1 };

    // ---------- the simulator: think (one GD iteration on all candidates) and act (one control step) ----------
    class Sim {
      constructor(world, cfg) { this.world = world; this.cfg = cfg; this.reset(); }
      P() { return params(this.cfg); }
      fresh() { return initPlan(this.s, this.cfg.H, this.r, this.cfg.sd); }
      reset() {
        const c = this.cfg; this.r = rng(c.seed * 7919 + 13);
        this.s = this.world.start.slice(); this.trail = [this.s.slice()]; this.k = 0; this.it = 0; this.itStep = 0;
        this.status = 'planning'; this.frozen = null; this.crash = null; this.steps = []; this.cur = []; this.prev = null;
        this.cands = []; for (let j = 0; j < c.M; j++) this.cands.push(this.fresh());
        this.evalAll();
      }
      get terminal() { return ['reached', 'crashed', 'missed', 'timeout', 'stuck'].includes(this.status); }
      setM(M) { while (this.cands.length < M) this.cands.push(this.fresh()); if (this.cands.length > M) { const order = this.E.map((e, j) => [e, j]).sort((a, b) => a[0] - b[0]).slice(0, M).map(p => p[1]).sort((a, b) => a - b); this.cands = order.map(j => this.cands[j]); } this.cfg.M = M; this.evalAll(); }
      evalAll() {
        const P = this.P(); this.E = this.cands.map(c => energyOnly(this.world, this.s, c, P));
        let b = 0; this.E.forEach((e, j) => { if (e < this.E[b]) b = j; }); this.best = b;
      }
      budget() { return this.k === 0 ? this.cfg.N0 : (this.cfg.mode === 'open' ? 0 : this.cfg.K); }
      wantsThink() { // still thinking for this decision?
        if (this.terminal || this.frozen) return false;
        if (this.itStep >= this.budget()) return false;
        if (this.cfg.adaptive && this.cur.length > 10) { const a = this.cur[this.cur.length - 11].min, b = this.cur[this.cur.length - 1].min; if ((a - b) / Math.max(1, b) < 0.002) return false; }
        return true;
      }
      think() {
        if (this.terminal || this.frozen) return;
        const P = this.P(), c = this.cfg;
        const E = this.E = this.cands.map(cd => gdStep(this.world, this.s, cd, P, c.alpha, c.sigma, this.r)); // energy at the start of this step
        let b = 0; for (let j = 1; j < E.length; j++) if (E[j] < E[b]) b = j; this.best = b; this.it++; this.itStep++;
        this.cur.push({ E: Float32Array.from(this.E), min: this.E[this.best] });
        if (this.cur.length > 6000) this.cur.splice(0, this.cur.length - 6000);
      }
      act() {
        if (this.terminal) return null;
        this.evalAll(); const c = this.cfg, P = this.P(), H = c.H; let a;
        if (c.mode === 'open') {
          if (!this.frozen) { const b = this.cands[this.best]; this.frozen = { k0: this.k, A: b.A.slice(), X: positions(this.s, b, P), R: rollout(this.s, b.A, H), H }; }
          const i = this.k - this.frozen.k0; a = [this.frozen.A[2 * i], this.frozen.A[2 * i + 1]];
        } else { const b = this.cands[this.best]; a = [b.A[0], b.A[1]]; }
        const n = Math.hypot(a[0], a[1]); if (n > W.vmax) { a = [a[0] * W.vmax / n, a[1] * W.vmax / n]; }
        const from = this.s.slice(), to = [from[0] + a[0], from[1] + a[1] - c.wind];
        this.steps.push({ iters: this.itStep, E: this.E[this.best] });
        this.prev = this.cur.map(p => p.min); this.cur = []; this.itStep = 0; this.k++;
        if (segHits(this.world, from[0], from[1], to[0], to[1])) { this.status = 'crashed'; this.crash = to; this.trail.push(to.slice()); return { from, to, hit: true }; }
        this.s = to; this.trail.push(to.slice());
        const dg = Math.hypot(to[0] - this.world.goal[0], to[1] - this.world.goal[1]);
        if (dg < REACH) this.status = 'reached';
        else if (this.frozen && this.k - this.frozen.k0 >= this.frozen.H) this.status = 'missed';
        else if (this.k >= c.maxSteps) this.status = 'timeout';
        else if (!this.frozen && this.k >= 10 && dg > 1 && Math.hypot(to[0] - this.trail[this.k - 8][0], to[1] - this.trail[this.k - 8][1]) < 0.3) this.status = 'stuck';
        else this.status = 'moving';
        if (!this.terminal && c.mode === 'mpc') {
          if (c.warm) {
            this.cands.forEach(cd => shift(cd, H)); this.evalAll();
            if (this.cands.length >= 3) { let w = 0; this.E.forEach((e, j) => { if (e > this.E[w]) w = j; }); this.cands[w] = this.fresh(); } // keep exploring
          } else this.cands = this.cands.map(() => this.fresh());
        }
        this.evalAll();
        return { from, to, hit: false };
      }
      distToGoal() { return Math.hypot(this.s[0] - this.world.goal[0], this.s[1] - this.world.goal[1]); }
    }
    return { WX, WY, REACH, W, CLIP, rng, obsCost, hub, positions, rollout, energy, energyOnly, initPlan, gdStep, shift, segHits, lambdaMax, params, PRESETS, DEFAULTS, Sim };
  })();
  if (typeof module !== 'undefined' && module.exports) module.exports = Core;
  if (!root.EBT || !root.EBT.section) return;

  // =====================================================================================================
  // UI
  // =====================================================================================================
  root.EBT.section({
    id: 'planning',
    nav: 'Planning & MPC',
    kicker: 'Lab · beyond the paper',
    title: 'Planning is thinking about actions',
    lede: 'Swap the prediction ŷ for a plan, and the same gradient-descent “thinking” becomes planning. The paper only sketches this idea for world models (App. A.3, p.26) and runs no such experiment. Here it runs live, on a hand-written energy with exact gradients.',
    mount(el, lib) {
      const h = lib.h, C = Core;
      const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim() || null;
      const col = { bg: css('--bg') || '#0d131f', panel: css('--panel') || '#141c2b', panel2: css('--panel2') || '#1a2437', rule: css('--rule') || '#283449', ink: css('--ink') || '#e9eef6', muted: css('--muted') || '#93a1b8', faint: css('--faint') || '#5d6b82', ebt: css('--ebt') || '#ffc95c', truth: css('--truth') || '#74e0c0', bad: css('--bad') || '#ff6b81', ar: css('--ar') || '#8ea7ff', rnn: css('--rnn') || '#c99bff' };
      const rgba = lib.rgba || ((hx, a) => hx);
      const fmtE = (v) => !isFinite(v) ? '–' : v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v.toFixed(2);
      const nf = (v) => Math.round(v).toLocaleString('en-US');

      // ---------------- state ----------------
      const cfg = Object.assign({}, C.DEFAULTS);
      let presetId = 'pillar';
      let world = null, sim = null;
      const view = { field: true, avg: true, speed: 'normal' };
      let lam = null, lamDirty = true, framesSinceLam = 0;
      let anim = null; // robot move animation {from, to, t}
      let playing = false;

      function cloneWorld(w) { return { start: w.start.slice(), goal: w.goal.slice(), obs: w.obs.map(o => Object.assign({}, o)) }; }
      function loadPreset(id, over) {
        presetId = id; const p = C.PRESETS[id];
        Object.assign(cfg, C.DEFAULTS, p.cfg, over || {});
        world = cloneWorld(p.world); sim = new C.Sim(world, cfg);
        fieldDirty = true; lamDirty = true; anim = null;
        syncControls(); hintEl.textContent = p.hint;
      }

      // ---------------- layout ----------------
      const intro = h('div', { class: 'planning-intro' },
        h('div', { class: 'prose' },
          h('p', { html: 'An EBT predicts by starting from noise and walking downhill on an energy E<sub>θ</sub>(x, ŷ). Nothing in that recipe says ŷ must be a token or an image. Here ŷ is a <b>plan</b>: the next H velocity commands a<sub>0</sub>…a<sub>H−1</sub> and the states ŝ<sub>1</sub>…ŝ<sub>H</sub> they should lead to. The context x (where the robot is, the goal, the map) stays fixed while the plan descends the energy.' }),
          h('p', { html: 'The paper only speculates about this. It says world models “could implicitly be used as policies”, by “holding the current context (past states) constant, and minimizing the energy by propagating the gradient back to the action inputs and future state predictions” (App. A.3, p.26). That sentence is exactly the default setting below.' }),
        ),
        h('div', { class: 'planning-eq card' },
          h('div', { class: 'row', style: 'justify-content:space-between' }, h('h4', {}, 'The energy being minimized'), lib.badge('ext', 'hand-written, not learned')),
          h('div', { class: 'planning-terms' },
            term('goal', 'w<sub>g</sub>·ρ(ŝ<sub>H</sub> − g) + w<sub>r</sub>·Σ ρ(ŝ<sub>t</sub> − g)', 'pull toward the goal, ρ = pseudo-Huber distance'),
            term('obstacles', 'w<sub>o</sub>·Σ max(0, r + m − ‖ŝ − c‖)²', 'checked at every imagined state and segment midpoint'),
            term('effort', 'w<sub>e</sub>·Σ ‖a<sub>t</sub>‖² + speed limit', 'small, slow commands are cheap'),
            term('smoothness', 'w<sub>s</sub>·Σ ‖a<sub>t</sub> − a<sub>t−1</sub>‖²', 'no jerky steering'),
            term('dynamics', 'w<sub>dyn</sub>·Σ ‖ŝ<sub>t+1</sub> − ŝ<sub>t</sub> − a<sub>t</sub>‖²', 'imagined states must follow from the actions'),
          ),
          h('p', { class: 'planning-note', html: 'Each term is a separate energy and they simply add. That is the composability the paper lists as Facet 4 (p.38): adding an obstacle adds a term, with no retraining. Every gradient here is analytic and was checked against finite differences.' }),
        ),
      );
      el.appendChild(intro);

      // ---- controls card
      const ctlCard = h('div', { class: 'card planning-ctl' });
      const presetSel = h('select', { id: 'planning-preset', 'aria-label': 'Scenario' }, Object.entries(C.PRESETS).map(([id, p]) => h('option', { value: id }, p.label)));
      presetSel.addEventListener('change', () => { stop(); loadPreset(presetSel.value); render(); });
      const hintEl = h('p', { class: 'planning-hint' });
      const segMode = lib.segmented({ label: 'Control loop', options: [['mpc', 'Re-plan every step (MPC)'], ['open', 'Plan once (open loop)']], value: cfg.mode, onchange: (v) => { cfg.mode = v; restart(); } });
      const segVars = lib.segmented({ label: 'Plan variables', options: [['joint', 'Actions + imagined states'], ['shoot', 'Actions only']], value: cfg.vars, onchange: (v) => { cfg.vars = v; restart(); } });
      const sl = {};
      const mk = (key, o) => { sl[key] = lib.slider(Object.assign({ id: 'planning-' + key, value: cfg[key] }, o)); return sl[key].el; };
      const lamOut = h('p', { class: 'planning-lam', 'aria-live': 'polite' });
      const chk = (id, label, get, set) => { const i = h('input', { type: 'checkbox', id: 'planning-' + id }); i.checked = get(); i.addEventListener('change', () => { set(i.checked); render(); }); return { el: h('label', { class: 'planning-chk', for: 'planning-' + id }, i, h('span', {}, label)), input: i }; };
      const cWarm = chk('warm', 'Warm start: continue from the previous plan', () => cfg.warm, (v) => { cfg.warm = v; });
      const cAdapt = chk('adapt', 'Stop thinking early once energy stops falling', () => cfg.adaptive, (v) => { cfg.adaptive = v; });
      const cAvg = chk('avg', 'Show the average of two routes', () => view.avg, (v) => { view.avg = v; });
      const cField = chk('field', 'Show the energy field', () => view.field, (v) => { view.field = v; });
      const segSpeed = lib.segmented({ label: 'Animation speed', options: [['slow', 'Slow'], ['normal', 'Normal'], ['fast', 'Fast']], value: view.speed, onchange: (v) => { view.speed = v; } });

      ctlCard.append(
        group('Scenario', h('div', { class: 'ctl' }, h('label', { for: 'planning-preset' }, h('span', {}, 'World')), presetSel), hintEl),
        group('Control loop', segMode.el, mk('wind', { label: 'Wind the model ignores (push down per step)', min: 0, max: 0.2, step: 0.01, fmt: v => v.toFixed(2), oninput: v => { cfg.wind = v; render(); } })),
        group('Thinking: one decision', segVars.el,
          mk('alpha', { label: 'Step size α', min: 0.001, max: 0.03, step: 0.001, fmt: v => v.toFixed(3), oninput: v => { cfg.alpha = v; render(); } }), lamOut,
          mk('N0', { label: 'Thinking steps before the first move N₀', min: 0, max: 4000, step: 100, fmt: v => nf(v), oninput: v => { cfg.N0 = v; render(); } }),
          mk('K', { label: 'Thinking steps per MPC step K', min: 0, max: 200, step: 1, fmt: v => String(v), oninput: v => { cfg.K = v; render(); } }),
          mk('M', { label: 'Candidates M (Best-of-M)', min: 1, max: 12, step: 1, fmt: v => String(v), oninput: v => { sim.setM(v); lamDirty = true; render(); } }),
          mk('sigma', { label: 'Langevin noise σ', min: 0, max: 0.05, step: 0.002, fmt: v => v.toFixed(3), oninput: v => { cfg.sigma = v; render(); } }),
          mk('H', { label: 'Horizon H (steps planned ahead)', min: 4, max: 24, step: 1, fmt: v => String(v), oninput: v => { cfg.H = v; restart(); } }),
          mk('wd', { label: 'Dynamics weight w_dyn', min: 0, max: 30, step: 0.5, fmt: v => v.toFixed(1), oninput: v => { cfg.wd = v; lamDirty = true; render(); } }),
          cWarm.el, cAdapt.el),
        group('Display', cAvg.el, cField.el, h('div', { class: 'ctl' }, h('label', {}, h('span', {}, 'Animation speed')), segSpeed.el)),
      );

      // ---- instrument card
      const inst = h('div', { class: 'card planning-inst' });
      const bPlay = h('button', { type: 'button', class: 'btn primary', 'aria-label': 'Play' }, 'Play');
      const bThink = h('button', { type: 'button', class: 'btn', title: 'One gradient step on every candidate plan' }, 'Think 1');
      const bThink50 = h('button', { type: 'button', class: 'btn', title: 'Fifty gradient steps' }, 'Think 50');
      const bAct = h('button', { type: 'button', class: 'btn', title: 'Execute the first action of the chosen plan' }, 'Act 1');
      const bReset = h('button', { type: 'button', class: 'btn', title: 'Same random draw, start over' }, 'Reset');
      const bNew = h('button', { type: 'button', class: 'btn', title: 'New random initial plans and noise' }, 'New draw');
      const status = h('span', { class: 'planning-status', role: 'status', 'aria-live': 'polite' });
      const bar = h('div', { class: 'planning-transport' }, h('div', { class: 'row' }, bPlay, bThink, bThink50, bAct, bReset, bNew), status);
      const head = h('div', { class: 'row', style: 'justify-content:space-between;gap:8px' }, h('h4', { style: 'margin:0' }, 'World: imagined plans and what really happens'), lib.badge('ext', 'App. A.3, p.26'));
      const worldBox = h('div', { class: 'planning-world' });
      const legend = h('div', { class: 'legend planning-legend' },
        h('span', {}, h('i', { style: `background:${col.ebt}` }), 'chosen plan (lowest energy)'),
        h('span', {}, h('i', { style: `background:${rgba(col.ebt, 0.35)}` }), 'other candidates'),
        h('span', {}, h('i', { class: 'dash', style: `border-color:${col.ink}` }), 'what its actions really do'),
        h('span', {}, h('i', { class: 'dash', style: `border-color:${col.bad}` }), 'average of two routes'),
        h('span', {}, h('i', { style: `background:${col.truth}` }), 'goal'),
      );
      const tip = h('p', { class: 'planning-tip' }, 'Drag obstacles, the goal or the start. Drag an obstacle’s edge to resize it. Double-click empty space to add an obstacle, double-click one to remove it.');
      const ro = h('div', { class: 'planning-readouts' });
      const RO = {};
      [['step', 'MPC step'], ['iters', 'Thinking steps'], ['E', 'Chosen plan energy'], ['dist', 'Distance to goal'], ['gap', 'Imagined vs real end']].forEach(([k, l]) => { RO[k] = h('b'); ro.appendChild(h('div', {}, h('span', {}, l), RO[k])); });
      // plots
      const pThink = h('div', { class: 'planning-plot' }, h('div', { class: 'row', style: 'justify-content:space-between' }, h('h4', {}, 'Energy while thinking about this decision'), lib.badge('ext')));
      const pTime = h('div', { class: 'planning-plot' }, h('h4', {}, 'Think, act, think, act: steps spent per decision'));
      const timeLegend = h('div', { class: 'legend' }, h('span', {}, h('i', { style: `background:${col.ebt}` }), 'thinking steps (log scale)'), h('span', {}, h('i', { style: `background:${col.truth};border-radius:50%` }), 'chosen plan energy after thinking'));
      const pBreak = h('div', { class: 'planning-plot' }, h('h4', {}, 'Where the chosen plan’s energy comes from'));
      const breakRows = h('div', { class: 'planning-break' });
      pBreak.appendChild(breakRows);
      const pSlice = h('div', { class: 'planning-plot' }, h('h4', {}, 'A slice through plan space'));
      const sliceNote = h('p', { class: 'planning-note' });
      const plots = h('div', { class: 'planning-plots' }, h('div', { class: 'stack' }, pThink, pTime, timeLegend), h('div', { class: 'stack' }, pBreak, pSlice, sliceNote));
      inst.append(head, bar, worldBox, legend, tip, ro, plots);
      el.appendChild(h('div', { class: 'bench planning-bench' }, ctlCard, inst));

      // ---------------- responsive canvases ----------------
      function rcanvas(parent, aspect, label, minH) {
        const box = h('div', { class: 'planning-cbox' }); parent.appendChild(box);
        const c = h('canvas', { role: 'img', 'aria-label': label }); box.appendChild(c);
        const o = { c, ctx: c.getContext('2d'), w: 0, h: 0, box };
        const fit = () => {
          const w = Math.max(200, box.clientWidth || 600), hh = Math.max(minH || 0, Math.round(w * aspect)), dpr = Math.min(2, window.devicePixelRatio || 1);
          if (w === o.w && hh === o.h) return false;
          o.w = w; o.h = hh; c.width = Math.round(w * dpr); c.height = Math.round(hh * dpr); c.style.height = hh + 'px';
          o.ctx.setTransform(dpr, 0, 0, dpr, 0, 0); return true;
        };
        fit(); o.fit = fit; return o;
      }
      const cw = rcanvas(worldBox, 0.6, 'Planning world: start, goal, obstacles, candidate plans as paths, the chosen plan in amber, and the robot trail.');
      const ct = rcanvas(pThink, 0.5, 'Energy of every candidate plan versus thinking step for the current decision, log scale.', 170);
      const cl = rcanvas(pTime, 0.32, 'Bars: thinking steps used for each MPC decision. Dots: chosen plan energy after thinking.', 110);
      const cs = rcanvas(pSlice, 0.5, 'Energy along the straight line in plan space from the chosen plan to a rival plan.', 170);
      const ro2 = new ResizeObserver(() => { let ch = false; [cw, ct, cl, cs].forEach(o => { ch = o.fit() || ch; }); if (ch) render(); });
      [cw, ct, cl, cs].forEach(o => ro2.observe(o.box));

      // ---------------- energy field (energy felt by the last imagined state ŝ_H) ----------------
      let fieldDirty = true, fieldImg = null, fieldGrid = null, fieldLayer = null;
      function buildField() {
        const nx = 120, ny = 72, P = sim.P(), g = [0, 0];
        const vals = new Float32Array(nx * ny); let hi = 0;
        for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
          const x = (i + 0.5) / nx * C.WX, y = C.WY - (j + 0.5) / ny * C.WY;
          const e = P.wg * C.hub(x - world.goal[0], y - world.goal[1]) + P.wo * C.obsCost(world, x, y, P.m, g);
          const v = Math.log1p(e); vals[j * nx + i] = v; if (v > hi) hi = v;
        }
        const cv = document.createElement('canvas'); cv.width = nx; cv.height = ny; const g2 = cv.getContext('2d'); const img = g2.createImageData(nx, ny);
        const grid = [];
        for (let j = 0; j < ny; j++) { const row = []; for (let i = 0; i < nx; i++) { const v = vals[j * nx + i] / hi; row.push(v); const c3 = lib.cmap(Math.pow(v, 1.5)); const k = 4 * (j * nx + i); img.data[k] = c3[0]; img.data[k + 1] = c3[1]; img.data[k + 2] = c3[2]; img.data[k + 3] = 255; } grid.push(row); }
        g2.putImageData(img, 0, 0); fieldImg = cv; fieldGrid = grid; fieldDirty = false;
      }

      // ---------------- derived quantities for display ----------------
      function rival() { // the candidate most different from the chosen one, among reasonably converged ones
        if (sim.cands.length < 2) return -1;
        const b = sim.best, P = sim.P(), Xb = C.positions(sim.s, sim.cands[b], P); let best = -1, bd = -1;
        const lim = sim.E[b] * 3 + 5;
        sim.cands.forEach((c, j) => { if (j === b) return; const X = C.positions(sim.s, c, P); let d = 0; for (let i = 2; i < X.length; i++) d += (X[i] - Xb[i]) ** 2; d = Math.sqrt(d / (X.length - 2)); const ok = sim.E[j] < lim; const score = d + (ok ? 1000 : 0); if (score > bd) { bd = score; best = j; } });
        return best;
      }
      function mixPlan(a, b, t) { const A = a.A.map((v, i) => (1 - t) * v + t * b.A[i]); const S = a.S.map((v, i) => (1 - t) * v + t * b.S[i]); return { A, S }; }

      // ---------------- drawing: world ----------------
      function drawWorld() {
        const { ctx, w, h: H } = cw, sc = w / C.WX, X = (x) => x * sc, Y = (y) => (C.WY - y) * sc;
        ctx.save(); ctx.clearRect(0, 0, w, H);
        ctx.fillStyle = col.panel2; ctx.fillRect(0, 0, w, H);
        if (view.field) {
          if (fieldDirty || !fieldImg) buildField();
          if (!fieldLayer || fieldLayer.w !== w || fieldLayer.h !== H || fieldLayer.src !== fieldImg) { // cache heatmap + contours at display size
            const dpr = Math.min(2, window.devicePixelRatio || 1), cv = document.createElement('canvas'); cv.width = Math.round(w * dpr); cv.height = Math.round(H * dpr);
            const g2 = cv.getContext('2d'); g2.scale(dpr, dpr); g2.globalAlpha = 0.5; g2.imageSmoothingEnabled = true; g2.drawImage(fieldImg, 0, 0, w, H); g2.globalAlpha = 1;
            lib.contours(g2, fieldGrid, 0, 0, w, H, [0.35, 0.45, 0.55, 0.65, 0.75], { color: 'rgba(233,238,246,0.10)', width: 1 });
            fieldLayer = { cv, w, h: H, src: fieldImg };
          }
          ctx.drawImage(fieldLayer.cv, 0, 0, w, H);
        }
        // grid
        ctx.strokeStyle = 'rgba(147,161,184,0.10)'; ctx.lineWidth = 1; ctx.beginPath();
        for (let x = 1; x < C.WX; x++) { ctx.moveTo(X(x), 0); ctx.lineTo(X(x), H); } for (let y = 1; y < C.WY; y++) { ctx.moveTo(0, Y(y)); ctx.lineTo(w, Y(y)); } ctx.stroke();
        // obstacles
        const m = C.W.m;
        world.obs.forEach((o, i) => {
          ctx.setLineDash([4, 4]); ctx.strokeStyle = 'rgba(147,161,184,0.45)'; ctx.lineWidth = 1; ctx.beginPath(); ctx.arc(X(o.x), Y(o.y), (o.r + m) * sc, 0, Math.PI * 2); ctx.stroke(); ctx.setLineDash([]);
          ctx.fillStyle = 'rgba(13,19,31,0.92)'; ctx.beginPath(); ctx.arc(X(o.x), Y(o.y), o.r * sc, 0, Math.PI * 2); ctx.fill();
          ctx.strokeStyle = drag && drag.kind !== 'goal' && drag.kind !== 'start' && drag.i === i ? col.ebt : col.muted; ctx.lineWidth = 1.5; ctx.stroke();
        });
        // goal
        ctx.fillStyle = rgba(col.truth, 0.18); ctx.strokeStyle = col.truth; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(X(world.goal[0]), Y(world.goal[1]), C.REACH * sc, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(X(world.goal[0]) - 5, Y(world.goal[1])); ctx.lineTo(X(world.goal[0]) + 5, Y(world.goal[1])); ctx.moveTo(X(world.goal[0]), Y(world.goal[1]) - 5); ctx.lineTo(X(world.goal[0]), Y(world.goal[1]) + 5); ctx.stroke();
        label(ctx, 'goal g', X(world.goal[0]), Y(world.goal[1]) - C.REACH * sc - 6, col.truth, 'center');
        // start
        ctx.strokeStyle = col.muted; ctx.lineWidth = 1.5; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.arc(X(world.start[0]), Y(world.start[1]), 9, 0, Math.PI * 2); ctx.stroke(); ctx.setLineDash([]);
        if (sim.k === 0) label(ctx, 'start', X(world.start[0]), Y(world.start[1]) + 14, col.muted, 'center', 'top');
        const P = sim.P(), path = (Xs, o) => { ctx.beginPath(); for (let t = 0; t < Xs.length / 2; t++) { const px = X(Xs[2 * t]), py = Y(Xs[2 * t + 1]); t ? ctx.lineTo(px, py) : ctx.moveTo(px, py); } ctx.strokeStyle = o.color; ctx.lineWidth = o.width || 1.5; ctx.setLineDash(o.dash || []); ctx.globalAlpha = o.alpha == null ? 1 : o.alpha; ctx.stroke(); ctx.setLineDash([]); if (o.dots) { ctx.fillStyle = o.color; for (let t = 1; t < Xs.length / 2; t++) { ctx.beginPath(); ctx.arc(X(Xs[2 * t]), Y(Xs[2 * t + 1]), o.dots, 0, Math.PI * 2); ctx.fill(); } } ctx.globalAlpha = 1; };
        const s0 = anim ? anim.to : sim.s;
        if (sim.frozen) {
          path(sim.frozen.X, { color: col.ebt, width: 2.5, dots: 3, alpha: 0.85, dash: [7, 5] });
          label(ctx, 'committed plan', X(sim.frozen.X[2 * sim.frozen.H]), Y(sim.frozen.X[2 * sim.frozen.H + 1]) + 16, col.ebt, 'center', 'top');
        } else if (!sim.terminal) {
          const eb = sim.E[sim.best]; sim.cands.forEach((c, j) => { if (j !== sim.best) path(C.positions(s0, c, P), { color: col.ebt, width: 1.4, alpha: sim.E[j] < 3 * eb + 2 ? 0.34 : 0.1, dots: 1.6 }); });
          const r = rival();
          if (view.avg && r >= 0) { const avg = mixPlan(sim.cands[sim.best], sim.cands[r], 0.5); path(C.positions(s0, avg, P), { color: col.bad, width: 1.8, dash: [6, 5], alpha: 0.9 }); }
          const b = sim.cands[sim.best], Xb = C.positions(s0, b, P);
          if (P.mode === 'joint') path(C.rollout(s0, b.A, P.H), { color: col.ink, width: 1.5, dash: [3, 4], alpha: 0.85 });
          path(Xb, { color: col.ebt, width: 3, dots: 3.2 });
        }
        // trail
        const tr = sim.trail.slice(); if (anim) { tr[tr.length - 1] = [lib.lerp(anim.from[0], anim.to[0], anim.e), lib.lerp(anim.from[1], anim.to[1], anim.e)]; }
        if (tr.length > 1) { ctx.beginPath(); tr.forEach((p, i) => i ? ctx.lineTo(X(p[0]), Y(p[1])) : ctx.moveTo(X(p[0]), Y(p[1]))); ctx.strokeStyle = col.ink; ctx.lineWidth = 2.2; ctx.stroke(); ctx.fillStyle = col.ink; tr.slice(0, -1).forEach(p => { ctx.beginPath(); ctx.arc(X(p[0]), Y(p[1]), 2.2, 0, Math.PI * 2); ctx.fill(); }); }
        const rp = tr[tr.length - 1];
        if (sim.status === 'crashed' && !anim) {
          ctx.strokeStyle = col.bad; ctx.lineWidth = 3; const cx = X(rp[0]), cy = Y(rp[1]); ctx.beginPath(); ctx.moveTo(cx - 8, cy - 8); ctx.lineTo(cx + 8, cy + 8); ctx.moveTo(cx + 8, cy - 8); ctx.lineTo(cx - 8, cy + 8); ctx.stroke();
        } else {
          ctx.fillStyle = col.ink; ctx.beginPath(); ctx.arc(X(rp[0]), Y(rp[1]), 6.5, 0, Math.PI * 2); ctx.fill(); ctx.strokeStyle = col.ebt; ctx.lineWidth = 2; ctx.stroke();
        }
        // wind gauge
        if (cfg.wind > 0) {
          const x0 = w - 70, y0 = 14; ctx.fillStyle = 'rgba(13,19,31,0.75)'; ctx.fillRect(x0 - 8, y0 - 6, 70, 54);
          label(ctx, 'wind', x0 + 26, y0 + 2, col.muted, 'center', 'top');
          for (let i = 0; i < 3; i++) lib.arrow(ctx, x0 + 8 + i * 18, y0 + 20, x0 + 8 + i * 18, y0 + 22 + Math.min(22, cfg.wind * 140), { color: col.ar, width: 1.6, head: 6 });
        }
        // banner
        const msg = banner(); if (msg) { ctx.font = `600 14px ${lib.F ? lib.F.body : 'sans-serif'}`; const tw = ctx.measureText(msg.t).width + 24; ctx.fillStyle = 'rgba(13,19,31,0.88)'; ctx.strokeStyle = msg.c; ctx.lineWidth = 1.5; lib.rr(ctx, 10, 10, tw, 30, 8); ctx.fill(); ctx.stroke(); ctx.fillStyle = msg.c; ctx.textBaseline = 'middle'; ctx.textAlign = 'left'; ctx.fillText(msg.t, 22, 25); }
        ctx.restore();
      }
      function label(ctx, t, x, y, color, align, base) { ctx.font = `13px ${lib.F ? lib.F.body : 'sans-serif'}`; ctx.textAlign = align || 'left'; ctx.textBaseline = base || 'bottom'; ctx.fillStyle = 'rgba(13,19,31,0.7)'; const tw = ctx.measureText(t).width; const bx = align === 'center' ? x - tw / 2 : x; ctx.fillRect(bx - 3, base === 'top' ? y - 1 : y - 16, tw + 6, 17); ctx.fillStyle = color; ctx.fillText(t, x, y); }
      function banner() {
        const s = sim.status;
        if (s === 'reached') return { t: `Reached the goal in ${sim.k} steps`, c: col.truth };
        if (s === 'crashed') return { t: `Crashed at step ${sim.k}`, c: col.bad };
        if (s === 'missed') return { t: `Plan used up. Missed the goal by ${sim.distToGoal().toFixed(2)}`, c: col.bad };
        if (s === 'timeout') return { t: `Out of time after ${sim.k} steps`, c: col.bad };
        if (s === 'stuck') return { t: `Stuck after ${sim.k} steps: a local minimum of the energy`, c: col.bad };
        return null;
      }

      // ---------------- drawing: plots ----------------
      function axesBox(o, pad) { return { x0: pad.l, y0: pad.t, x1: o.w - pad.r, y1: o.h - pad.b }; }
      function tickText(ctx, t, x, y, align, base) { ctx.font = `12px ${lib.F ? lib.F.mono : 'monospace'}`; ctx.fillStyle = col.muted; ctx.textAlign = align; ctx.textBaseline = base; ctx.fillText(t, x, y); }
      function logTicks(lo, hi) { const out = []; for (let e = Math.floor(lo); e <= Math.ceil(hi); e++) if (e >= lo - 1e-9 && e <= hi + 1e-9) out.push(e); return out; }
      const fmtPow = (e) => e >= 0 && e <= 4 ? String(Math.pow(10, e)) : '1e' + e;
      function drawThink() {
        const { ctx, w, h: H } = ct; ctx.clearRect(0, 0, w, H);
        const b = axesBox(ct, { l: 44, r: 12, t: 10, b: 40 }); const cur = sim.cur, budget = Math.max(1, sim.budget(), sim.itStep);
        const series = cur.length ? cur : null; let lo = Infinity, hi = -Infinity;
        const consider = (v) => { if (v > 0 && isFinite(v)) { const l = Math.log10(v); if (l < lo) lo = l; if (l > hi) hi = l; } };
        if (series) series.forEach(p => p.E.forEach(consider)); if (sim.prev) sim.prev.forEach(consider); sim.E.forEach(consider);
        if (!isFinite(lo)) { lo = 0; hi = 2; } if (hi - lo < 0.5) { lo -= 0.25; hi += 0.25; }
        lo = Math.floor(lo * 2) / 2; hi = Math.ceil(hi * 2) / 2;
        const xs = (i) => b.x0 + (i / budget) * (b.x1 - b.x0), ys = (v) => b.y1 - (Math.log10(Math.max(1e-9, v)) - lo) / (hi - lo) * (b.y1 - b.y0);
        ctx.strokeStyle = col.rule; ctx.lineWidth = 1; logTicks(lo, hi).forEach(e => { const y = b.y1 - (e - lo) / (hi - lo) * (b.y1 - b.y0); ctx.beginPath(); ctx.moveTo(b.x0, y); ctx.lineTo(b.x1, y); ctx.stroke(); tickText(ctx, fmtPow(e), b.x0 - 6, y, 'right', 'middle'); });
        ctx.strokeStyle = col.faint; ctx.beginPath(); ctx.moveTo(b.x0, b.y0); ctx.lineTo(b.x0, b.y1); ctx.lineTo(b.x1, b.y1); ctx.stroke();
        tickText(ctx, '0', b.x0, b.y1 + 6, 'center', 'top'); tickText(ctx, nf(budget), b.x1, b.y1 + 6, 'right', 'top');
        tickText(ctx, sim.k === 0 ? 'thinking step, before the first move' : `thinking step, decision ${sim.k + 1}`, (b.x0 + b.x1) / 2, b.y1 + 22, 'center', 'top');
        if (sim.prev && sim.prev.length > 1 && sim.k > 0) { const pb = Math.max(1, sim.prev.length); ctx.strokeStyle = rgba(col.muted, 0.5); ctx.setLineDash([3, 3]); ctx.lineWidth = 1.2; ctx.beginPath(); sim.prev.forEach((v, i) => { const x = b.x0 + (i / Math.max(budget, pb)) * (b.x1 - b.x0), y = ys(v); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }); ctx.stroke(); ctx.setLineDash([]); }
        if (series) {
          const n = series.length, stride = Math.max(1, Math.floor(n / 500)), M = series[n - 1].E.length;
          for (let j = 0; j < M; j++) { ctx.strokeStyle = rgba(col.ebt, 0.28); ctx.lineWidth = 1; ctx.beginPath(); let st = false; for (let i = 0; i < n; i += stride) { if (j >= series[i].E.length) continue; const x = xs(i + 1), y = ys(series[i].E[j]); st ? ctx.lineTo(x, y) : ctx.moveTo(x, y); st = true; } ctx.stroke(); }
          ctx.strokeStyle = col.ebt; ctx.lineWidth = 2.4; ctx.beginPath(); for (let i = 0; i < n; i += stride) { const x = xs(i + 1), y = ys(series[i].min); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); } ctx.lineTo(xs(n), ys(series[n - 1].min)); ctx.stroke();
          ctx.fillStyle = col.ebt; ctx.beginPath(); ctx.arc(xs(n), ys(series[n - 1].min), 3.5, 0, Math.PI * 2); ctx.fill();
        } else {
          sim.E.forEach(v => { ctx.fillStyle = rgba(col.ebt, 0.6); ctx.beginPath(); ctx.arc(b.x0 + 3, ys(v), 3, 0, Math.PI * 2); ctx.fill(); });
          ctx.font = `13px ${lib.F ? lib.F.body : 'sans-serif'}`; ctx.fillStyle = col.muted; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
          const msg = sim.frozen ? 'Open loop: no more thinking, just executing.' : sim.terminal ? 'Episode over.' : playing ? '' : sim.k > 0 && cfg.K === 0 ? 'K = 0: acting without thinking.' : 'Press Play or Think to start descending.'; if (msg) ctx.fillText(msg, (b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2);
        }
      }
      function drawTimeline() {
        const { ctx, w, h: H } = cl; ctx.clearRect(0, 0, w, H);
        const b = axesBox(cl, { l: 44, r: 10, t: 8, b: 22 }); const steps = sim.steps.slice(); const live = !sim.terminal && !sim.frozen ? { iters: sim.itStep, E: sim.E[sim.best], live: true } : null; if (live) steps.push(live);
        const N = Math.max(cfg.mode === 'open' ? cfg.H + 1 : 20, steps.length), bw = (b.x1 - b.x0) / N;
        const maxI = Math.max(10, cfg.N0, ...steps.map(s => s.iters)); const yl = (v) => b.y1 - Math.log10(1 + v) / Math.log10(1 + maxI) * (b.y1 - b.y0);
        ctx.strokeStyle = col.rule; ctx.lineWidth = 1; [1, 10, 100, 1000].filter(v => v <= maxI).forEach(v => { const y = yl(v); ctx.beginPath(); ctx.moveTo(b.x0, y); ctx.lineTo(b.x1, y); ctx.stroke(); tickText(ctx, String(v), b.x0 - 6, y, 'right', 'middle'); });
        steps.forEach((s, i) => { const x = b.x0 + i * bw; ctx.fillStyle = s.live ? rgba(col.ebt, 0.45) : col.ebt; const y = yl(s.iters); ctx.fillRect(x + bw * 0.15, y, Math.max(1, bw * 0.7), b.y1 - y); });
        // energy after thinking (dots, own log scale)
        const Es = steps.map(s => s.E).filter(v => v > 0 && isFinite(v));
        if (Es.length) { const lo = Math.log10(Math.min(...Es)), hi = Math.max(lo + 0.5, Math.log10(Math.max(...Es))); const ye = (v) => b.y0 + 4 + (1 - (Math.log10(v) - lo) / (hi - lo)) * (b.y1 - b.y0 - 8); ctx.strokeStyle = rgba(col.truth, 0.7); ctx.lineWidth = 1.2; ctx.beginPath(); steps.forEach((s, i) => { const x = b.x0 + (i + 0.5) * bw, y = ye(s.E); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }); ctx.stroke(); ctx.fillStyle = col.truth; steps.forEach((s, i) => { ctx.beginPath(); ctx.arc(b.x0 + (i + 0.5) * bw, ye(s.E), 2.4, 0, Math.PI * 2); ctx.fill(); }); }
        ctx.strokeStyle = col.faint; ctx.beginPath(); ctx.moveTo(b.x0, b.y0); ctx.lineTo(b.x0, b.y1); ctx.lineTo(b.x1, b.y1); ctx.stroke();
        tickText(ctx, 'decision 1', b.x0, b.y1 + 5, 'left', 'top'); tickText(ctx, String(N), b.x1, b.y1 + 5, 'right', 'top');
      }
      function drawSlice() {
        const { ctx, w, h: H } = cs; ctx.clearRect(0, 0, w, H);
        const b = axesBox(cs, { l: 44, r: 12, t: 12, b: 30 }); const r = sim.terminal || sim.frozen ? -1 : rival();
        if (r < 0) { ctx.font = `13px ${lib.F ? lib.F.body : 'sans-serif'}`; ctx.fillStyle = col.muted; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; const few = sim.cands.length < 2; ctx.fillText(few ? 'Needs at least 2 candidates (M ≥ 2).' : 'No live plans to compare.', w / 2, H / 2); sliceNote.innerHTML = few ? 'The slice compares the chosen plan with the most different candidate. With one candidate there is nothing to compare.' : sim.frozen ? 'Open loop committed to one plan, so there are no candidates left to compare.' : 'The episode is over. Reset or Replay to compare candidate plans again.'; return; }
        const P = sim.P(), A = sim.cands[sim.best], B = sim.cands[r], ls = [], es = [];
        for (let i = 0; i <= 60; i++) { const l = -0.25 + 1.5 * i / 60; ls.push(l); es.push(C.energyOnly(world, sim.s, mixPlan(A, B, l), P)); }
        const lo = Math.floor(Math.log10(Math.min(...es)) * 2) / 2, hi = Math.max(lo + 0.5, Math.ceil(Math.log10(Math.max(...es)) * 2) / 2);
        const xs = (l) => b.x0 + (l + 0.25) / 1.5 * (b.x1 - b.x0), ys = (v) => b.y1 - (Math.log10(v) - lo) / (hi - lo) * (b.y1 - b.y0);
        ctx.strokeStyle = col.rule; ctx.lineWidth = 1; logTicks(lo, hi).forEach(e => { const y = b.y1 - (e - lo) / (hi - lo) * (b.y1 - b.y0); ctx.beginPath(); ctx.moveTo(b.x0, y); ctx.lineTo(b.x1, y); ctx.stroke(); tickText(ctx, fmtPow(e), b.x0 - 6, y, 'right', 'middle'); });
        ctx.strokeStyle = col.faint; ctx.beginPath(); ctx.moveTo(b.x0, b.y0); ctx.lineTo(b.x0, b.y1); ctx.lineTo(b.x1, b.y1); ctx.stroke();
        ctx.strokeStyle = col.ink; ctx.lineWidth = 2; ctx.beginPath(); ls.forEach((l, i) => { const x = xs(l), y = ys(es[i]); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }); ctx.stroke();
        const eA = es[10], eB = es[50], eM = es[30];
        const mark = (l, v, c, t, al) => { ctx.fillStyle = c; ctx.beginPath(); ctx.arc(xs(l), ys(v), 5, 0, Math.PI * 2); ctx.fill(); tickText(ctx, t, xs(l), b.y1 + 6, al || 'center', 'top'); };
        ctx.setLineDash([3, 3]); ctx.strokeStyle = rgba(col.bad, 0.6); ctx.beginPath(); ctx.moveTo(xs(0.5), b.y0); ctx.lineTo(xs(0.5), b.y1); ctx.stroke(); ctx.setLineDash([]);
        mark(0, eA, col.ebt, 'chosen'); mark(1, eB, rgba(col.ebt, 0.55), 'rival'); mark(0.5, eM, col.bad, 'average');
        const ratio = eM / Math.max(1e-9, eA), barrier = es.slice(10, 51).reduce((m, v) => Math.max(m, v), 0);
        const twoBasins = barrier > 1.5 * Math.max(eA, eB) + 0.5;
        sliceNote.innerHTML = `Energy along the straight line in plan space from the chosen plan (λ = 0) to the most different candidate (λ = 1). ` + (twoBasins
          ? `There is a ridge between them: two separate valleys. The averaged plan costs <b>${fmtE(eM)}</b>, ${ratio >= 10 ? Math.round(ratio) + '×' : ratio.toFixed(1) + '×'} the chosen plan.`
          : `No ridge between them right now: both sit in one valley (or have not converged yet).`);
      }
      function drawBreak() {
        const P = sim.P(), b = sim.cands[sim.best]; if (!b) return;
        const e = C.energy(world, sim.s, b, P); const keys = [['goal', 'goal'], ['obs', 'obstacles'], ['eff', 'effort'], ['smooth', 'smoothness'], ['dyn', 'dynamics']];
        const tot = Math.max(1e-9, e.E); breakRows.innerHTML = '';
        keys.forEach(([k, l]) => { const v = e.T[k], pct = 100 * v / tot; const off = k === 'dyn' && P.mode !== 'joint'; breakRows.appendChild(h('div', { class: 'planning-brow' + (off ? ' off' : '') }, h('span', { class: 'l' }, l), h('span', { class: 'track' }, h('i', { style: `width:${Math.min(100, pct).toFixed(1)}%;background:${k === 'obs' && v > 0.5 ? col.bad : col.ebt}` })), h('span', { class: 'v' }, off ? 'exact' : fmtE(v)))); });
        breakRows.appendChild(h('div', { class: 'planning-brow total' }, h('span', { class: 'l' }, 'total E'), h('span', { class: 'track' }), h('span', { class: 'v' }, fmtE(e.E))));
      }
      function updateReadouts() {
        RO.step.textContent = `${sim.k} / ${cfg.mode === 'open' ? cfg.H : cfg.maxSteps}`;
        RO.iters.textContent = `${nf(sim.itStep)} now · ${nf(sim.it)} total`;
        RO.E.textContent = sim.frozen ? 'committed' : fmtE(sim.E[sim.best]);
        RO.dist.textContent = sim.distToGoal().toFixed(2);
        if (cfg.vars === 'joint' && !sim.frozen && sim.cands[sim.best]) { const b = sim.cands[sim.best], H = cfg.H, R = C.rollout(sim.s, b.A, H); RO.gap.textContent = Math.hypot(R[2 * H] - b.S[2 * H - 2], R[2 * H + 1] - b.S[2 * H - 1]).toFixed(2); }
        else RO.gap.textContent = cfg.vars === 'joint' ? '–' : '0 (exact)';
        const st = sim.status, txt = { planning: playing ? 'Thinking before the first move' : sim.itStep >= cfg.N0 ? 'Ready: press Play to act' : 'Ready', moving: playing ? (sim.wantsThink() ? 'Thinking' : 'Moving') : 'Paused', reached: 'Reached the goal', crashed: 'Crashed', missed: 'Missed the goal', timeout: 'Out of time', stuck: 'Stuck' }[st] || st;
        status.textContent = txt; status.dataset.state = st;
        bPlay.textContent = playing && !thinkOnly ? 'Pause' : sim.terminal ? 'Replay' : 'Play'; bPlay.setAttribute('aria-label', bPlay.textContent);
        bThink.disabled = bThink50.disabled = !!(sim.terminal || sim.frozen); bAct.disabled = sim.terminal;
        sl.K.input.disabled = cfg.mode === 'open';
      }
      function updateLam() {
        if (!lamDirty && framesSinceLam < 20) return; framesSinceLam = 0; lamDirty = false;
        const b = sim.cands[sim.best]; if (!b) return;
        lam = C.lambdaMax(world, sim.s, b, sim.P(), 30);
        const lim = 2 / Math.max(1e-9, lam), ok = cfg.alpha < lim;
        lamOut.innerHTML = `Curvature of E at the chosen plan: λ<sub>max</sub> ≈ ${lam.toFixed(0)}. Plain gradient descent is stable only for α &lt; 2/λ<sub>max</sub> ≈ <b>${lim.toFixed(4)}</b>. ` + (ok ? '<span class="ok">Current α is safe.</span>' : '<span class="no">Current α is too large: expect oscillation.</span>');
      }
      function render() { drawWorld(); drawThink(); drawTimeline(); drawSlice(); drawBreak(); updateLam(); updateReadouts(); }

      // ---------------- control flow ----------------
      const RATE = { slow: 1, normal: 20, fast: 150 }, MOVE = { slow: 0.55, normal: 0.22, fast: 0.08 };
      let thinkOnly = false; // auto preview: think through the first decision, then wait for the user
      const thinkBurst = (n) => { const t0 = performance.now(); for (let i = 0; i < n && performance.now() - t0 < 9; i++) sim.think(); framesSinceLam++; };
      const loop = lib.loop((dt) => {
        if (anim) { anim.t += dt / MOVE[view.speed]; anim.e = lib.ease ? lib.ease(Math.min(1, anim.t)) : Math.min(1, anim.t); if (anim.t >= 1) anim = null; render(); return true; }
        if (drag) { if (!sim.terminal && !sim.frozen && !playing) thinkBurst(8); render(); return true; } // plans bend live while you drag
        if (!playing) { render(); return false; }
        if (sim.terminal) { playing = false; render(); return false; }
        if (sim.wantsThink()) { let n = RATE[view.speed]; while (n > 0 && sim.wantsThink()) { const before = sim.it; thinkBurst(1); n--; if (sim.it === before) break; } }
        else if (thinkOnly) { thinkOnly = false; playing = false; render(); return false; }
        else doAct();
        render(); return true;
      });
      function doAct() { const r = sim.act(); if (r && !lib.reducedMotion) anim = { from: r.from, to: r.to, t: 0, e: 0 }; lamDirty = true; }
      function stop() { playing = false; thinkOnly = false; }
      function restart() { stop(); sim.cfg = cfg; sim.reset(); fieldDirty = true; lamDirty = true; anim = null; render(); }
      bPlay.addEventListener('click', () => { if (playing && thinkOnly) { thinkOnly = false; return; } if (playing) { playing = false; render(); return; } if (sim.terminal) { sim.reset(); lamDirty = true; } playing = true; thinkOnly = false; loop.start(); });
      bThink.addEventListener('click', () => { stop(); sim.think(); lamDirty = true; render(); });
      bThink50.addEventListener('click', () => { stop(); for (let i = 0; i < 50; i++) sim.think(); lamDirty = true; render(); });
      bAct.addEventListener('click', () => { stop(); doAct(); loop.start(); render(); });
      bReset.addEventListener('click', () => { stop(); sim.reset(); lamDirty = true; anim = null; render(); });
      bNew.addEventListener('click', () => { stop(); cfg.seed = (cfg.seed % 997) + 1 + Math.floor(Math.random() * 50); sim.reset(); lamDirty = true; anim = null; render(); });

      // ---------------- direct manipulation on the world canvas ----------------
      let drag = null;
      const toWorld = (ev) => { const r = cw.c.getBoundingClientRect(); return [(ev.clientX - r.left) / r.width * C.WX, C.WY - (ev.clientY - r.top) / r.height * C.WY]; };
      function hit(p) {
        const sc = cw.w / C.WX, tol = 10 / sc;
        if (Math.hypot(p[0] - world.goal[0], p[1] - world.goal[1]) < C.REACH + tol * 0.5) return { kind: 'goal' };
        if (Math.hypot(p[0] - world.start[0], p[1] - world.start[1]) < 0.3 + tol * 0.5) return { kind: 'start' };
        for (let i = world.obs.length - 1; i >= 0; i--) { const o = world.obs[i], d = Math.hypot(p[0] - o.x, p[1] - o.y); if (Math.abs(d - o.r) < tol * 0.8) return { kind: 'rim', i }; if (d < o.r) return { kind: 'obs', i, dx: p[0] - o.x, dy: p[1] - o.y }; }
        return null;
      }
      const clampW = (v, a, b) => Math.max(a, Math.min(b, v));
      cw.c.addEventListener('pointerdown', (ev) => { const p = toWorld(ev), t = hit(p); if (!t) return; drag = t; cw.c.setPointerCapture(ev.pointerId); ev.preventDefault(); loop.start(); });
      cw.c.addEventListener('pointermove', (ev) => {
        const p = toWorld(ev);
        if (!drag) { const t = hit(p); cw.c.style.cursor = !t ? 'crosshair' : t.kind === 'rim' ? 'nwse-resize' : 'grab'; return; }
        if (drag.kind === 'goal') { world.goal = [clampW(p[0], 0.3, C.WX - 0.3), clampW(p[1], 0.3, C.WY - 0.3)]; }
        else if (drag.kind === 'start') { world.start = [clampW(p[0], 0.3, C.WX - 0.3), clampW(p[1], 0.3, C.WY - 0.3)]; sim.reset(); }
        else if (drag.kind === 'obs') { const o = world.obs[drag.i]; o.x = clampW(p[0] - drag.dx, 0, C.WX); o.y = clampW(p[1] - drag.dy, 0, C.WY); }
        else if (drag.kind === 'rim') { const o = world.obs[drag.i]; o.r = clampW(Math.hypot(p[0] - o.x, p[1] - o.y), 0.25, 2.2); }
        fieldDirty = true; lamDirty = true; sim.evalAll();
      });
      const endDrag = () => { if (drag) { drag = null; render(); } };
      cw.c.addEventListener('pointerup', endDrag); cw.c.addEventListener('pointercancel', endDrag);
      cw.c.addEventListener('dblclick', (ev) => { const p = toWorld(ev), t = hit(p); if (t && (t.kind === 'obs' || t.kind === 'rim')) world.obs.splice(t.i, 1); else if (!t) world.obs.push({ x: p[0], y: p[1], r: 0.6 }); fieldDirty = true; lamDirty = true; sim.evalAll(); render(); });

      // ---------------- sync controls with cfg ----------------
      function syncControls() {
        presetSel.value = presetId; segMode.set(cfg.mode); segVars.set(cfg.vars);
        ['wind', 'alpha', 'N0', 'K', 'M', 'sigma', 'H', 'wd'].forEach(k => sl[k].set(cfg[k]));
        cWarm.input.checked = cfg.warm; cAdapt.input.checked = cfg.adaptive;
      }

      // ---------------- guided experiments ----------------
      const exps = h('div', { class: 'planning-exps' });
      const run = (preset, over, opts) => () => { stop(); loadPreset(preset, over); if (opts && opts.view) Object.assign(view, opts.view); cAvg.input.checked = view.avg; cField.input.checked = view.field; render(); worldBox.scrollIntoView({ behavior: lib.reducedMotion ? 'auto' : 'smooth', block: 'center' }); playing = true; loop.start(); };
      const EXPS = [
        { t: 'Two routes, one choice', d: 'Watch the candidates split above and below the pillar. Algorithm 2 keeps the lowest-energy one. The red dashed line is the average of two routes: straight through the pillar.', why: 'An energy can hold several good answers as separate valleys and rank them (Facets 2 and 3). A regressor trained on both routes would output their average.', b: [['Run', 'pillar', { M: 8 }, { view: { avg: true } }]] },
        { t: 'A trap for gradient descent', d: 'With one candidate, descent slides into the cup and stops: a local minimum, so the robot gets stuck. With eight candidates, some start outside the cup, and their lower energy wins.', why: 'Best-of-M is verification: the same energy that guides descent also picks the winner, with no external reward model (p.8).', b: [['1 candidate', 'cup', { M: 1 }], ['8 candidates', 'cup', { M: 8 }]] },
        { t: 'Wind the model does not know', d: 'Open loop plans once and executes blindly, so the push adds up until it misses or hits a rock. MPC executes one action, looks where it really is, and thinks again.', why: 'Re-planning turns a wrong model into a small, repeatedly corrected error. A world model only needs to be right a few steps ahead.', b: [['Open loop', 'gap', { mode: 'open' }], ['MPC', 'gap', { mode: 'mpc' }]] },
        { t: 'Think a little, often', d: 'Three thinking steps per move. With warm start, each decision continues from the previous plan, shifted by one step, and the robot glides through. Cold start throws the plan away each time.', why: 'Like the paper’s replay buffer, which keeps optimizing earlier predictions to simulate longer trajectories (Sec 3.3), warm starts spread thinking across time.', b: [['Warm start', 'slalom', { warm: true }], ['Cold start', 'slalom', { warm: false }]] },
        { t: 'Imagination without physics', d: 'Lower the dynamics weight and the imagined states (amber) sail around the pillar, while the actions’ real effect (white dashed) falls far short. Run it open loop to see the robot act on a fantasy.', why: 'Energy over imagined futures is only useful if the energy also checks that those futures are reachable. In a learned system that check is the world model.', b: [['w_dyn = 0.3, open loop', 'pillar', { wd: 0.3, mode: 'open', M: 4 }], ['w_dyn = 10', 'pillar', { wd: 10, mode: 'open', M: 4 }]] },
        { t: 'Step size and curvature', d: 'Push α above the stability limit shown under the slider and the plans start to shake instead of settle. Switch to “Actions only” and the limit drops sharply.', why: 'Backpropagating through a long rollout makes the energy much stiffer than optimizing imagined states jointly. The paper also reports step size as a sensitive hyperparameter (p.17) and randomizes it in training (Sec 3.3).', b: [['α = 0.024', 'pillar', { alpha: 0.024, M: 4 }], ['Actions only', 'pillar', { vars: 'shoot', alpha: 0.004, M: 4 }]] },
      ];
      EXPS.forEach((x, i) => {
        exps.appendChild(h('div', { class: 'card planning-exp' },
          h('div', { class: 'n' }, String(i + 1)),
          h('h3', {}, x.t), h('p', {}, x.d), h('p', { class: 'why' }, x.why),
          h('div', { class: 'row' }, x.b.map(([lbl, pre, over, opts]) => h('button', { type: 'button', class: 'btn', onclick: run(pre, over, opts) }, lbl)))));
      });
      el.appendChild(h('h3', { class: 'planning-sub' }, 'Six experiments'));
      el.appendChild(h('p', { class: 'planning-subp' }, 'Each button loads a world and settings, then plays. Every number on screen comes from the live simulation.'));
      el.appendChild(exps);

      // ---------------- mapping + takeaways ----------------
      const rows = [
        ['Context x', 'robot state s₀, goal g, obstacle map', 'encoded observation history and a goal embedding'],
        ['Prediction ŷ', 'actions a₀…a<sub>H−1</sub> and imagined states ŝ₁…ŝ<sub>H</sub>', 'future latent states and the actions between them'],
        ['Energy E<sub>θ</sub>(x, ŷ)', 'hand-written sum of five terms', 'a learned scalar head scoring (past, future, actions) together'],
        ['Thinking: ŷ ← ŷ − α∇E (Eq. 1)', 'one press of Think', 'gradient steps on imagined latents and actions'],
        ['Langevin noise η (Eq. 2)', 'σ slider', 'exploration in plan space'],
        ['Best-of-M by lowest energy (Alg. 2)', 'M candidate plans', 'self-verification of imagined futures'],
        ['Replay buffer (Sec 3.3)', 'warm start from the shifted previous plan', 'amortized planning across time'],
        ['More steps for hard predictions (Facet 1)', '“stop early” toggle and the per-decision bars', 'adaptive compute per decision'],
      ];
      const tbl = h('div', { class: 'tbl' }, h('table', {}, h('thead', {}, h('tr', {}, h('th', {}, 'In the paper (EBT)'), h('th', {}, 'In this lab'), h('th', {}, 'In a JEPA-style world model (our extrapolation)'))), h('tbody', {}, rows.map(r => h('tr', {}, r.map(c => h('td', { html: c })))))));
      el.appendChild(h('div', { class: 'row', style: 'justify-content:space-between' }, h('h3', { class: 'planning-sub' }, 'How the pieces map'), lib.badge('ext')));
      el.appendChild(tbl);
      el.appendChild(h('div', { class: 'grid2 planning-callouts' },
        lib.callout('insight', 'For JEPA world models', 'A JEPA predictor maps (context, action) to the next latent and is trained by regression, so it outputs one future per input. To plan with it you roll it forward and score the rollout. That is the <i>Actions only</i> setting here, where gradients flow back through the whole rollout. An EBT-style world model would score (context, future, actions) jointly with one scalar. That gives a verifier to rank imagined futures, an uncertainty signal (energy that stays high when no plan fits), and joint optimization of states and actions. In this lab the joint form tolerates a step size roughly 4× larger; the λ<sub>max</sub> readout shows it live.'),
        lib.callout('', 'Why hundreds of steps here, and two or three in the paper', 'This energy is written by hand, so nothing shapes it for gradient descent. The stiffest terms (obstacles, dynamics) cap α, and the weakest (effort, goal pull) set the pace. An EBT learns its energy by backpropagating through the descent itself (Alg. 1), so the landscape is trained to be easy to descend: the paper’s models train with two or three steps (p.26). A learned planner energy would be shaped the same way.'),
        lib.callout('warn', 'Two valleys is a choice we made', 'Our energy has one valley per route, so averaging is visibly wrong. The paper reports that its training pushes the energy to be convex around training examples, so when one context has many valid answers the valleys can merge into one bowl around their average. For images that means blur (App. B.2, p.29; Limitations, p.17). For a planner, the average of “over” and “under” is the red dashed path into the pillar.'),
        lib.callout('', 'What is real and what is not', 'Real: every energy value, gradient, curvature estimate and trajectory on this page is computed live from the formula above. Not from the paper: the whole planning setup. The paper has no experiments with actions, rewards or control. The only grounding is App. A.3 (p.26).'),
      ));

      // ---------------- helpers used above (hoisted) ----------------
      function group(title, ...kids) { return h('fieldset', { class: 'planning-group' }, h('legend', {}, title), ...kids); }
      function term(name, f, d) { return h('div', { class: 'planning-term' }, h('span', { class: 'n' }, name), h('span', { class: 'f', html: f }), h('span', { class: 'd' }, d)); }

      // ---------------- start in an informative state: the first plan already thought through ----------------
      // start: random initial plans ŷ0 ~ N(0, I); when the lab scrolls into view it thinks through the first decision once
      loadPreset('pillar');
      render();
      lib.whenVisible(worldBox, () => {
        [cw, ct, cl, cs].forEach(o => o.fit());
        if (lib.reducedMotion) { while (sim.wantsThink()) sim.think(); render(); return; }
        if (!playing && sim.it === 0) { thinkOnly = true; playing = true; loop.start(); } else render();
      });
    },
  });
})(typeof window !== 'undefined' ? window : globalThis);
