/* Panel: planning and MPC with an energy. BEYOND THE PAPER: grounded only in App. A.3 (p.26).
   A 2-D point robot plans by gradient descent on a hand-written energy over (actions, imagined states) with exact analytic
   gradients (Eq. 1 / Eq. 2 with the plan as ŷ), Best-of-M selection (Alg. 2), warm-started receding-horizon MPC, and a
   live compute race against the cross-entropy method (CEM) on the same energy.
   The numerical core is pure and loads in Node for tests: require('assets/js/panels/planning.js') -> {Core, Sim, SCEN}. */
(function (root) {
  'use strict';

  // ===================================================================================================
  // Core: world, energy + exact gradient, one gradient step, CEM, curvature. No DOM.
  // ===================================================================================================
  const Core = (function () {
    const WX = 10, WY = 5, REACH = 0.4;
    const W = { wg: 4, wr: 0.04, wo: 12, m: 0.3, we: 0.1, ws: 0.6, vmax: 0.8, wv: 10 };
    const CLIP = 0.5; // safety: no 2-D block of the plan moves more than this in one gradient step
    function rng(seed) {
      let a = seed >>> 0;
      const f = () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
      f.normal = () => { const u = Math.max(1e-9, f()), v = f(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
      return f;
    }
    // Σ max(0, r + m − ‖p − c‖)² over obstacles, plus the same margin penalty for the 4 walls. Adds ∂/∂p into g.
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
    const rho = (dx, dy) => Math.sqrt(dx * dx + dy * dy + 1) - 1; // pseudo-Huber distance
    function rollout(s0, A, H) {
      const X = new Float64Array(2 * (H + 1)); X[0] = s0[0]; X[1] = s0[1];
      for (let t = 1; t <= H; t++) { X[2 * t] = X[2 * t - 2] + A[2 * t - 2]; X[2 * t + 1] = X[2 * t - 1] + A[2 * t - 1]; }
      return X;
    }
    function positions(s0, plan, P) {
      if (P.mode !== 'joint') return rollout(s0, plan.A, P.H);
      const H = P.H, X = new Float64Array(2 * (H + 1)); X[0] = s0[0]; X[1] = s0[1];
      for (let t = 1; t <= H; t++) { X[2 * t] = plan.S[2 * t - 2]; X[2 * t + 1] = plan.S[2 * t - 1]; }
      return X;
    }
    // E(x, ŷ) and its exact gradient. x = (s0, goal, map); ŷ = {A: 2H actions, S: 2H imagined states}.
    // P.mode 'joint': A and S are free (App. A.3: actions and future states); 'shoot': S is the rollout of A.
    function energyCore(world, s0, plan, P, B) {
      const H = P.H, X = B.X, gX = B.gX, gA = B.gA, gS = B.gS, T = B.T, A = plan.A, joint = P.mode === 'joint';
      X[0] = s0[0]; X[1] = s0[1];
      for (let t = 1; t <= H; t++) {
        if (joint) { X[2 * t] = plan.S[2 * t - 2]; X[2 * t + 1] = plan.S[2 * t - 1]; }
        else { X[2 * t] = X[2 * t - 2] + A[2 * t - 2]; X[2 * t + 1] = X[2 * t - 1] + A[2 * t - 1]; }
      }
      gX.fill(0); gA.fill(0); gS.fill(0); T.goal = T.obs = T.eff = T.smooth = T.dyn = 0;
      const gx = world.goal[0], gy = world.goal[1];
      for (let t = 1; t <= H; t++) { // goal: strong pull on the last state, weak pull on every state
        const w = t === H ? P.wr + P.wg : P.wr, dx = X[2 * t] - gx, dy = X[2 * t + 1] - gy, q = Math.sqrt(dx * dx + dy * dy + 1);
        T.goal += w * (q - 1); gX[2 * t] += w * dx / q; gX[2 * t + 1] += w * dy / q;
      }
      const g = B.g;
      for (let t = 1; t <= H; t++) { // obstacles at every state and every segment midpoint
        g[0] = g[1] = 0; T.obs += P.wo * obsCost(world, X[2 * t], X[2 * t + 1], P.m, g);
        gX[2 * t] += P.wo * g[0]; gX[2 * t + 1] += P.wo * g[1];
        const mx = 0.5 * (X[2 * t - 2] + X[2 * t]), my = 0.5 * (X[2 * t - 1] + X[2 * t + 1]);
        g[0] = g[1] = 0; T.obs += P.wo * obsCost(world, mx, my, P.m, g);
        gX[2 * t] += 0.5 * P.wo * g[0]; gX[2 * t + 1] += 0.5 * P.wo * g[1];
        gX[2 * t - 2] += 0.5 * P.wo * g[0]; gX[2 * t - 1] += 0.5 * P.wo * g[1];
      }
      for (let i = 0; i < 2 * H; i++) { T.eff += P.we * A[i] * A[i]; gA[i] += 2 * P.we * A[i]; }
      for (let t = 0; t < H; t++) { // speed limit ‖a‖ ≤ vmax, counted as effort
        const ax = A[2 * t], ay = A[2 * t + 1], n = Math.sqrt(ax * ax + ay * ay) + 1e-9, p = n - P.vmax;
        if (p > 0) { T.eff += P.wv * p * p; const k = 2 * P.wv * p / n; gA[2 * t] += k * ax; gA[2 * t + 1] += k * ay; }
      }
      for (let t = 1; t < H; t++) for (let k = 0; k < 2; k++) {
        const d = A[2 * t + k] - A[2 * t - 2 + k]; T.smooth += P.ws * d * d; gA[2 * t + k] += 2 * P.ws * d; gA[2 * t - 2 + k] -= 2 * P.ws * d;
      }
      if (joint) { // dynamics consistency: imagined states must follow from the actions
        for (let t = 0; t < H; t++) for (let k = 0; k < 2; k++) {
          const r = X[2 * (t + 1) + k] - X[2 * t + k] - A[2 * t + k];
          T.dyn += P.wd * r * r; gX[2 * (t + 1) + k] += 2 * P.wd * r; gX[2 * t + k] -= 2 * P.wd * r; gA[2 * t + k] -= 2 * P.wd * r;
        }
        for (let i = 0; i < 2 * H; i++) gS[i] = gX[i + 2];
      } else { // backprop through the rollout: ∂s_t/∂a_u = I for every t > u
        let cx = 0, cy = 0;
        for (let u = H - 1; u >= 0; u--) { cx += gX[2 * (u + 1)]; cy += gX[2 * (u + 1) + 1]; gA[2 * u] += cx; gA[2 * u + 1] += cy; }
      }
      return T.goal + T.obs + T.eff + T.smooth + T.dyn;
    }
    const newBuf = (H) => ({ X: new Float64Array(2 * (H + 1)), gX: new Float64Array(2 * (H + 1)), gA: new Float64Array(2 * H), gS: new Float64Array(2 * H), g: [0, 0], T: { goal: 0, obs: 0, eff: 0, smooth: 0, dyn: 0 } });
    const bufs = new Map(); const sharedBuf = (H) => { let b = bufs.get(H); if (!b) { b = newBuf(H); bufs.set(H, b); } return b; };
    function energy(world, s0, plan, P) { const B = newBuf(P.H); const E = energyCore(world, s0, plan, P, B); return { E, T: Object.assign({}, B.T), gA: B.gA, gS: B.gS, X: B.X }; }
    function energyOnly(world, s0, plan, P) { return energyCore(world, s0, plan, P, sharedBuf(P.H)); }
    function initPlan(s0, H, r, sd) { // ŷ0 ~ N(0, sd² I) on the actions; imagined states start as their rollout
      const A = new Float64Array(2 * H); for (let i = 0; i < 2 * H; i++) A[i] = sd * r.normal();
      return { A, S: rollout(s0, A, H).slice(2) };
    }
    // one thinking step (Eq. 1; Eq. 2 when sigma > 0): ŷ ← ŷ − α∇E + η. Returns the energy before the step.
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
    function shiftPlan(plan, H) { // receding horizon: drop the executed step; coast (zero action) at the end
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
    // Cross-entropy method over action sequences (states = rollout of the actions), same energy.
    function cemInit(H, sd) { return { mu: new Float64Array(2 * H), sd: new Float64Array(2 * H).fill(sd), best: null, bestE: Infinity, last: null }; }
    function cemIter(world, s0, st, P, cfg, r) {
      const H = P.H, n = cfg.pop, ke = cfg.elite, Ps = Object.assign({}, P, { mode: 'shoot' }), S = [];
      for (let j = 0; j < n; j++) {
        const A = new Float64Array(2 * H);
        if (j === 0) A.set(st.mu); else for (let i = 0; i < 2 * H; i++) A[i] = st.mu[i] + st.sd[i] * r.normal();
        S.push({ A, E: energyOnly(world, s0, { A, S: null }, Ps) });
      }
      S.sort((a, b) => a.E - b.E);
      for (let i = 0; i < 2 * H; i++) {
        let m = 0; for (let e = 0; e < ke; e++) m += S[e].A[i]; m /= ke;
        let v = 0; for (let e = 0; e < ke; e++) v += (S[e].A[i] - m) ** 2; v /= ke;
        st.mu[i] = (1 - cfg.mom) * m + cfg.mom * st.mu[i];
        st.sd[i] = Math.max(cfg.minSd, (1 - cfg.mom) * Math.sqrt(v) + cfg.mom * st.sd[i]);
      }
      if (S[0].E < st.bestE) { st.bestE = S[0].E; st.best = S[0].A.slice(); }
      st.last = S;
      return n; // energy evaluations spent (forward passes)
    }
    function cemShift(st, H, sd0) {
      for (let i = 0; i < 2 * (H - 1); i++) { st.mu[i] = st.mu[i + 2]; st.sd[i] = st.sd[i + 2]; }
      for (let i = 2 * (H - 1); i < 2 * H; i++) { st.mu[i] = 0; st.sd[i] = sd0; }
      for (let i = 0; i < 2 * H; i++) st.sd[i] = Math.max(st.sd[i], 0.5 * sd0);
      if (st.best) { const b = st.best; for (let i = 0; i < 2 * (H - 1); i++) b[i] = b[i + 2]; b[2 * H - 2] = b[2 * H - 1] = 0; }
      st.bestE = Infinity; st.last = null;
    }
    // largest Hessian eigenvalue at a plan: power iteration on finite-difference Hessian-vector products of the exact gradient
    function lambdaMax(world, s0, plan, P, iters) {
      const H = P.H, joint = P.mode === 'joint', n = joint ? 4 * H : 2 * H, h = 1e-5;
      const x0 = new Float64Array(n); x0.set(plan.A); if (joint) x0.set(plan.S, 2 * H);
      const tmp = { A: new Float64Array(2 * H), S: new Float64Array(2 * H) };
      const grad = (x) => { tmp.A.set(x.subarray(0, 2 * H)); if (joint) tmp.S.set(x.subarray(2 * H)); const e = energy(world, s0, tmp, P); const g = new Float64Array(n); g.set(e.gA); if (joint) g.set(e.gS, 2 * H); return g; };
      const r = rng(99); let v = new Float64Array(n).map(() => r() - 0.5), lam = 0;
      const xp = new Float64Array(n), xm = new Float64Array(n);
      for (let k = 0; k < (iters || 25); k++) {
        let nv = 0; for (let i = 0; i < n; i++) nv += v[i] * v[i]; nv = Math.sqrt(nv) || 1;
        for (let i = 0; i < n; i++) { v[i] /= nv; xp[i] = x0[i] + h * v[i]; xm[i] = x0[i] - h * v[i]; }
        const gp = grad(xp), gm = grad(xm); const Hv = new Float64Array(n); lam = 0;
        for (let i = 0; i < n; i++) { Hv[i] = (gp[i] - gm[i]) / (2 * h); lam += Hv[i] * v[i]; }
        v = Hv;
      }
      return lam;
    }
    return { WX, WY, REACH, W, CLIP, rng, obsCost, rho, rollout, positions, energy, energyOnly, initPlan, gdStep, shiftPlan, segHits, cemInit, cemIter, cemShift, lambdaMax };
  })();

  // ---------- scenarios ----------
  const cup = []; for (let a = -90; a <= 90; a += 30) { const t = a * Math.PI / 180; cup.push({ x: +(5.3 + 1.2 * Math.cos(t)).toFixed(2), y: +(2.5 + 1.2 * Math.sin(t)).toFixed(2), r: 0.5 }); }
  const SCEN = {
    pillar: { label: 'pillar', world: { start: [1, 2.5], goal: [9, 2.5], obs: [{ x: 5, y: 2.5, r: 1.0 }] } },
    slalom: { label: 'slalom', world: { start: [1, 2.5], goal: [9, 2.5], obs: [{ x: 3.2, y: 1.9, r: 0.7 }, { x: 5, y: 3.1, r: 0.7 }, { x: 6.8, y: 1.9, r: 0.7 }] } },
    gap: { label: 'gap', world: { start: [1, 2.5], goal: [9, 2.5], obs: [{ x: 5, y: 4.05, r: 0.95 }, { x: 5, y: 0.95, r: 0.95 }] } },
    cup: { label: 'cup', world: { start: [1, 2.5], goal: [9, 2.5], obs: cup } },
  };
  const DEFAULTS = { planner: 'grad', vars: 'joint', loop: 'mpc', R: 1, H: 12, alphaJ: 0.035, alphaS: 0.007, sigma: 0, M: 6, K: 25, N0: 600, cem0: 30, cemK: 6, pop: 64, elite: 8, mom: 0.1, minSd: 0.02, wind: 0, warm: true, sd: 1, maxSteps: 40, seed: 1, wd: 4 };

  // ---------- simulator: think (one iteration of the planner) and act (one control step) ----------
  class Sim {
    constructor(world, cfg) { this.world = world; this.cfg = cfg; this.reset(); }
    P() { const c = this.cfg; return Object.assign({ mode: c.planner === 'cem' ? 'shoot' : c.vars, H: c.H }, Core.W, { wd: c.wd }); }
    alpha() { return this.cfg.vars === 'joint' ? this.cfg.alphaJ : this.cfg.alphaS; }
    reset() {
      const c = this.cfg; this.r = Core.rng(c.seed * 7919 + 13);
      this.s = this.world.start.slice(); this.trail = [this.s.slice()]; this.k = 0; this.it = 0; this.itDec = 0; this.fe = 0; this.feDec = 0;
      this.status = 'ready'; this.frozen = null; this.log = []; this.cur = []; this.since = 0; this.crash = null;
      this.cands = []; for (let j = 0; j < c.M; j++) this.cands.push(Core.initPlan(this.s, c.H, this.r, c.sd));
      this.cem = Core.cemInit(c.H, c.sd);
      this.evalAll();
    }
    get terminal() { return ['reached', 'crashed', 'missed', 'timeout', 'stuck'].includes(this.status); }
    setM(M) {
      while (this.cands.length < M) this.cands.push(Core.initPlan(this.s, this.cfg.H, this.r, this.cfg.sd));
      if (this.cands.length > M) { const keep = this.E.map((e, j) => [e, j]).sort((a, b) => a[0] - b[0]).slice(0, M).map(p => p[1]).sort((a, b) => a - b); this.cands = keep.map(j => this.cands[j]); }
      this.cfg.M = M; this.evalAll();
    }
    evalAll() {
      const P = this.P();
      if (this.cfg.planner === 'cem') { this.E = [Core.energyOnly(this.world, this.s, this.plan(), P)]; this.best = 0; return; }
      this.E = this.cands.map(cd => Core.energyOnly(this.world, this.s, cd, P)); let b = 0; this.E.forEach((e, j) => { if (e < this.E[b]) b = j; }); this.best = b;
    }
    plan() {
      if (this.cfg.planner === 'cem') { const A = (this.cem.best || this.cem.mu).slice(); return { A, S: Core.rollout(this.s, A, this.cfg.H).slice(2) }; }
      return this.cands[this.best];
    }
    budget() { const c = this.cfg; if (this.frozen) return 0; if (this.k === 0) return c.planner === 'cem' ? c.cem0 : c.N0; if (this.since > 0) return 0; return c.planner === 'cem' ? c.cemK : c.K; }
    wantsThink() { return !this.terminal && this.itDec < this.budget(); }
    think() {
      if (this.terminal || this.frozen) return;
      const P = this.P(), c = this.cfg;
      if (c.planner === 'cem') {
        const n = Core.cemIter(this.world, this.s, this.cem, P, c, this.r); this.fe += n; this.feDec += n;
        this.E = [this.cem.bestE]; this.best = 0;
      } else {
        const a = this.alpha(); this.E = this.cands.map(cd => Core.gdStep(this.world, this.s, cd, P, a, c.sigma, this.r));
        let b = 0; for (let j = 1; j < this.E.length; j++) if (this.E[j] < this.E[b]) b = j; this.best = b;
        this.fe += 3 * this.cands.length; this.feDec += 3 * this.cands.length;
      }
      this.it++; this.itDec++;
      this.cur.push({ E: Float32Array.from(this.E), min: this.E[this.best], fe: this.feDec });
      if (this.cur.length > 4000) this.cur.splice(0, this.cur.length - 4000);
      if (this.status === 'ready') this.status = 'thinking';
    }
    act() {
      if (this.terminal) return null;
      if (this.cfg.planner !== 'cem') this.evalAll();
      const c = this.cfg, H = c.H; let a;
      if (c.loop === 'open') {
        if (!this.frozen) { const b = this.plan(); this.frozen = { k0: this.k, A: b.A.slice(), X: Core.positions(this.s, b, this.P()), R: Core.rollout(this.s, b.A, H) }; }
        const i = this.k - this.frozen.k0; a = [this.frozen.A[2 * i], this.frozen.A[2 * i + 1]];
      } else { const b = this.plan(); a = [b.A[0], b.A[1]]; }
      const n = Math.hypot(a[0], a[1]); if (n > Core.W.vmax) a = [a[0] * Core.W.vmax / n, a[1] * Core.W.vmax / n];
      const from = this.s.slice(), to = [from[0] + a[0], from[1] + a[1] - c.wind];
      if (this.since === 0) this.log.push({ iters: this.itDec, fe: this.feDec, E: this.E[this.best] });
      this.k++;
      if (Core.segHits(this.world, from[0], from[1], to[0], to[1])) { this.status = 'crashed'; this.crash = to; this.trail.push(to.slice()); return { from, to, hit: true }; }
      this.s = to; this.trail.push(to.slice());
      const dg = Math.hypot(to[0] - this.world.goal[0], to[1] - this.world.goal[1]);
      if (dg < Core.REACH) this.status = 'reached';
      else if (this.frozen && this.k - this.frozen.k0 >= H) this.status = 'missed';
      else if (this.k >= c.maxSteps) this.status = 'timeout';
      else if (!this.frozen && this.k >= 10 && dg > 1 && Math.hypot(to[0] - this.trail[this.k - 8][0], to[1] - this.trail[this.k - 8][1]) < 0.35) this.status = 'stuck';
      else this.status = 'moving';
      if (!this.terminal && c.loop === 'mpc') {
        this.since = (this.since + 1) % c.R;
        if (this.since === 0) { this.cur = []; this.itDec = 0; this.feDec = 0; }
        if (c.planner === 'cem') Core.cemShift(this.cem, H, c.sd * 0.5);
        else if (c.warm || this.since > 0) {
          this.cands.forEach(cd => Core.shiftPlan(cd, H));
          if (this.since === 0 && this.cands.length >= 3) { this.evalAll(); let w = 0; this.E.forEach((e, j) => { if (e > this.E[w]) w = j; }); this.cands[w] = Core.initPlan(this.s, H, this.r, c.sd); } // keep exploring
        } else this.cands = this.cands.map(() => Core.initPlan(this.s, H, this.r, c.sd));
      }
      this.evalAll();
      return { from, to, hit: false };
    }
    dist() { return Math.hypot(this.s[0] - this.world.goal[0], this.s[1] - this.world.goal[1]); }
  }

  // compute race: gradient (shooting, M plans) vs CEM from the same state, same energy, advanced by equal compute
  class Race {
    constructor(world, s0, cfg, seed) {
      this.world = world; this.s0 = s0.slice(); this.cfg = cfg; this.H = cfg.H;
      this.P = Object.assign({ mode: 'shoot', H: cfg.H }, Core.W, { wd: cfg.wd });
      const r1 = Core.rng(seed * 31 + 7); this.r1 = r1; this.M = cfg.M;
      this.cands = []; for (let j = 0; j < this.M; j++) this.cands.push(Core.initPlan(s0, cfg.H, r1, cfg.sd));
      this.r2 = Core.rng(seed * 57 + 3); this.cem = Core.cemInit(cfg.H, cfg.sd);
      this.g = []; this.c = []; this.gFE = 0; this.cFE = 0; this.gIt = 0; this.cIt = 0; this.gBest = Infinity; this.gE = this.cands.map(cd => Core.energyOnly(world, s0, cd, this.P));
      this.budget = 6000; this.done = false;
    }
    advance(fe) { // advance a shared compute clock by `fe` forward-pass equivalents; each planner spends up to the clock
      this.clock = Math.min(this.budget, (this.clock || 0) + fe); const tg = this.clock, tc = this.clock;
      while (this.gFE + 3 * this.M <= tg) {
        this.gE = this.cands.map(cd => Core.gdStep(this.world, this.s0, cd, this.P, this.cfg.alphaS, 0, this.r1));
        this.gFE += 3 * this.M; this.gIt++; this.gBest = Math.min(this.gBest, ...this.gE); this.g.push([this.gFE, this.gBest, this.gIt]);
      }
      while (this.cFE + this.cfg.pop <= tc) {
        this.cFE += Core.cemIter(this.world, this.s0, this.cem, this.P, this.cfg, this.r2); this.cIt++; this.c.push([this.cFE, this.cem.bestE, this.cIt]);
      }
      if (this.clock >= this.budget) this.done = true;
    }
    gPlan() { let b = 0; const E = this.cands.map(cd => Core.energyOnly(this.world, this.s0, cd, this.P)); E.forEach((e, j) => { if (e < E[b]) b = j; }); return { plan: this.cands[b], E: E[b], all: this.cands, Es: E }; }
    cPlan() { const A = (this.cem.best || this.cem.mu); return { A, S: null }; }
    reach(thr) { const f = (h) => { const k = h.find(p => p[1] <= thr); return k ? { fe: k[0], it: k[2] } : null; }; return { g: f(this.g), c: f(this.c) }; }
  }

  if (typeof module !== 'undefined' && module.exports) module.exports = { Core, Sim, Race, SCEN, DEFAULTS };
  if (!root.EBT || !root.EBT.panel) return;

  // ===================================================================================================
  // Panel
  // ===================================================================================================
  root.EBT.panel({
    id: 'planning',
    nav: 'Planning and MPC',
    title: 'Planning and MPC with an energy',
    lede: 'Make the prediction ŷ a plan and the same machinery plans: thinking bends a trajectory downhill on an energy, Best-of-M chooses between routes, and model predictive control thinks again after every move. The paper only sketches this (App. A.3); everything here is our extension, computed live.',
    text: `
      <p>The paper runs no planning experiments. Its one passage on control (App. A.3, p.26) imagines a world model of "the current context, future, as well as future actions" that acts as a policy by "holding the current context (past states) constant, and minimizing the energy by propagating the gradient back to the action inputs and future state predictions". This panel does exactly that, with a hand-written energy standing in for a learned one.</p>
      <p>The context is $x=(s_0,g,\\text{map})$: robot position, goal, obstacles. The prediction is a plan over $H=12$ steps, $\\hat y=(a_{0:H-1},\\,\\hat s_{1:H})$: velocity commands $a_t$ and the imagined positions $\\hat s_t$ they should reach, 48 numbers in all. Its energy sums simple terms:</p>
      <div class="eq">$$\\begin{aligned}E(x,\\hat y)=\\;&w_g\\,\\rho(\\hat s_H-g)+w_o{\\textstyle\\sum_t}\\,\\phi(\\hat s_t)\\\\&+{\\textstyle\\sum_t}\\big(w_e\\|a_t\\|^2+w_s\\|a_t-a_{t-1}\\|^2\\big)\\\\&+w_d{\\textstyle\\sum_t}\\|\\hat s_{t+1}-\\hat s_t-a_t\\|^2\\end{aligned}$$<span class="why">ρ(d) = √(‖d‖²+1) − 1 is a smooth distance to the goal g. φ(s) = Σ max(0, r + m − ‖s − c‖)² penalizes entering a margin m = 0.3 around each obstacle (centre c, radius r). The middle line prices effort and jerky steering; the last asks imagined states to follow from the actions, with ŝ₀ = s₀ fixed. Also in E but not written out: a weak pull of every ŝ_t toward g (weight 0.04), φ at segment midpoints and at the walls, and a speed limit ‖a_t‖ ≤ 0.8 counted as effort. Weights: w_g = 4, w_o = 12, w_e = 0.1, w_s = 0.6, w_d = 4.</span></div>
      <p>Planning is the paper's inference procedure: $M$ random plans (actions $a_t\\sim\\mathcal N(0,I)$, imagined states initialized where those actions lead), steps $\\hat y_{i+1}=\\hat y_i-\\alpha\\nabla_{\\hat y}E+\\eta_i$ (Eq. 1; Eq. 2 adds Langevin noise $\\eta_i$), and the plan with the lowest final energy wins (Alg. 2, p.7). One safety change: no point of the plan moves more than 0.5 per step. The paper mentions clamping prediction gradients as a stabilizer but never uses it in its experiments (p.42). <b>Model predictive control</b> (MPC) wraps this in a loop: execute only $a_0$, observe where the robot really is, shift the plan one step forward as a warm start, think a little, repeat.</p>`,
    steps: [
      { label: 'A plan is a prediction', html: '<p>Six random plans $\\hat y_0$, drawn faintly, bold for the lowest energy. Each is 12 velocity commands; the dots are the imagined positions $\\hat s_t$. The map has not been consulted yet, so they wander and cut through the pillar. The bars split the bold plan\'s energy into its terms.</p>' },
      { label: 'Thinking bends the plans', html: '<p>Each iteration costs one forward pass for $E$ and one backward pass for $\\nabla_{\\hat y}E$ over all 48 numbers. The plans slide off the pillar and straighten toward the goal, and the plot tracks every candidate\'s energy. This is Eq. 1 with a trajectory as $\\hat y$. The line under the bars estimates $\\lambda_{\\max}$, the sharpest curvature of $E$ at the bold plan: plain gradient descent is stable only for $\\alpha<2/\\lambda_{\\max}$, which is why $\\alpha$ is small here and 600 iterations are spent before the first move.</p>' },
      { label: 'Verification picks a route', html: '<p>Eight candidates split above and below the pillar: two valleys of $E$. Alg. 2 keeps the lowest. The dashed line averages the best plan of each valley. It runs through the pillar at about 25 times the best plan\'s energy, and it is what a regressor trained on both routes would predict, since squared error is minimized by the mean. The paper reports the same averaging for its convex-basin training on multi-modal data (blurry images, p.29; Limitations, p.17).</p>' },
      { label: 'Imagined states must be reachable', html: '<p>A.3 optimizes "future state predictions" alongside actions. Here the dynamics weight is lowered to $w_d=0.3$: the imagined positions (blue) still bend around the pillar, but the real effect of the actions (dotted) does not. The loop is open, so <b>[ play ]</b> executes all 12 actions without re-planning, and the robot follows them into the pillar (10 of 10 draws). Switch the loop to MPC and it arrives anyway, in about 16 moves instead of 12: re-planning from the true position corrects the bad imagination one move at a time. In a learned world model, this consistency term is the learned compatibility of states and actions.</p>' },
      { label: 'Act, shift, think again', html: '<p>MPC executes $a_0$, shifts every candidate one step forward in time and spends $K=25$ iterations from there. The bars count thinking per move: 600 iterations before the first move, 25 after each. The shift (a warm start) is what makes 25 enough. Turn it off under <b>[ more ]</b> and each move starts from fresh random plans: on this slalom the robot then reached the goal in only 4 of 10 draws, against 10 of 10 with the shift, which still succeeds with $K=5$. The paper\'s replay buffer is a training-time cousin: restarting from earlier predictions simulates longer optimization trajectories than the 2 or 3 steps actually run (Sec 3.3).</p>' },
      { label: 'Re-planning absorbs model error', html: '<p>A wind pushes the robot down 0.06 per step, and the planner\'s model does not include it. The gray robot plans once and executes blindly (open loop): it drifts by about 12 × 0.06 = 0.72 and misses (or crashes, in 2 of 10 draws). The black robot re-plans from where it really is after every move and reaches the goal in 10 of 10 draws; re-planning every 5 moves (under <b>[ more ]</b>) does too. Feedback lets an imperfect model work: it only has to be right until the next re-plan, not over the whole horizon.</p>' },
      { label: 'Gradient vs CEM: compute', html: '<p>Same energy, same start, equal compute. CEM (black, dashed) samples 64 action sequences per iteration, keeps the 8 best and refits its Gaussian. Gradient descent (blue) moves $M=2$ plans, each step a forward and a backward pass, charged as 3 forward-pass equivalents. Both search over actions only here (shooting), so the race is fair. The counters show who first gets within 10% of the best plan either method found.</p>' },
      { label: 'Traps: imagined states escape', html: '<p>A cup opens toward the robot. CEM over actions (gray) heads straight for the goal and stalls in front of the cup (10 of 10 draws). Gradient descent over actions <em>and</em> imagined states, with eight candidates, can move positions around the rim first and let the dynamics term pull the actions along: it escapes in 10 of 10. Trajectory optimization calls this collocation, as opposed to shooting. The win is not gradients alone: with the same 8 candidates over actions only, gradient descent crashed into the cup in 9 of 10 draws, and a single collocation plan ($M=1$) got stuck in 9 of 10. Both toggles are below.</p>' },
    ],
    after: `
      <h3>What the race shows</h3>
      <p>Over ten draws on the slalom with $M=2$, CEM needed 2.8 to 8.9 times more forward-pass equivalents than gradient descent to get within 10% of the best plan in nine of them; in the tenth it settled on a worse route and never got there. Gradient descent needed 1.2 to 3.8 times more sequential iterations. With $M=6$ the compute gap shrinks to 1.5 to 3 times, because every candidate pays for its own gradient. Two caveats pull in opposite directions. Charging 3 forward passes per gradient step (F ≈ 2N, B ≈ 4N FLOPs per token, p.35) is conservative for gradients: a backward pass to the input alone skips the weight gradients and costs about one forward pass. But CEM needs no gradient and its 64 samples run in parallel, so on a GPU its latency is set by its fewer iterations.</p>
      <h3>Why shooting needs short horizons</h3>
      <p>Over actions alone, the first action moves every later state, so the curvature of $E$ grows with the horizon. At converged slalom plans, $\\lambda_{\\max}\\approx$ 190, 530 and 830 for $H=$ 6, 12 and 24; over actions and imagined states it stays near 40 to 60. A step size that is stable at $H=12$ is not at $H=24$, and in our runs gradient shooting with $M=2$ then stalled at high energy while CEM still found a good plan. Optimizing the imagined states too (collocation) is what keeps gradient planning well conditioned here. Try <b>horizon H</b> under <b>[ more ]</b> and watch the $\\lambda_{\\max}$ line. The familiar argument that gradients beat sampling in high dimensions, the curse of dimensionality the paper raises against contrastive training (p.6–7), concerns how many samples a search needs; this toy does not test it.</p>
      <h3>Replacing the hand-written energy</h3>
      <p>A learned planner would train $E_\\theta(x,\\hat y)$ with Alg. 1 on logged trajectories: $x$ the encoded past, $\\hat y$ the future states and actions, $J$ the distance from $\\hat y_N$ to the logged future. Trained this way it scores what the logged behavior would do; the goal is what turns imitation into planning. It can enter as an extra additive energy term, as here (EBM energies compose, Facet 4, p.38), or as context, where the state a logged trajectory actually reached can serve as its goal (hindsight relabeling). Three lessons carry over. EBTs train with only 2 or 3 descent steps (p.26), so a learned landscape is shaped for a few large steps, not the hundreds needed here. The convex-basin bias merges alternative routes (step 3, p.29). And a planner is an adversary of its verifier: at small data scale, Best-of-10 sometimes picked low-energy wrong answers (p.28), and an optimizer searches harder than random samples do.</p>
      <p class="note">Beyond the paper: this whole panel extends App. A.3. The energy is hand-written; its gradients are analytic and checked against finite differences; every number is computed live in your browser. The draw counts quoted above come from running this same code over 10 seeds.</p>`,
    source: [{ kind: 'ext', note: 'hand-written energy' }, { kind: 'paper', note: 'A.3, Eq. 1–2, Alg. 2' }],
    figure(stage, ctx) {
      const { lib } = ctx, h = lib.h, C = lib.C, X = Core;
      const INK = C.ink || '#111111', BLUE = C.blue || '#2f3cff', MUTED = C.muted || '#6b6b70', FAINT = C.faint || '#a3a3a8', RULE = C.rule || '#e4e4e7', GRAY = '#8d8d93';
      const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
      const nf = (v) => Math.round(v).toLocaleString('en-US');
      const fmtE = (v) => !isFinite(v) ? '–' : v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v.toFixed(2);
      const fmtK = (v) => v >= 1e4 ? (v / 1000).toFixed(1) + 'k' : nf(v);

      // ---------- state ----------
      const cfg = Object.assign({}, DEFAULTS);
      let scen = 'pillar', world = null, sim = null, ghost = null, ghostKind = null, race = null;
      const view = { avg: false, plotX: 'iter' };
      let playing = false, autoThink = false, anim = null, ganim = null, drag = null, visible = false, fieldDirty = true;
      const cloneW = (w) => ({ start: w.start.slice(), goal: w.goal.slice(), obs: w.obs.map(o => Object.assign({}, o)) });

      // ---------- layout ----------
      const wrapW = lib.frame(stage, { label: 'World · plan space', sub: 'blue: candidate plans ŷ (bold = lowest energy) · ink: where the robot really went' });
      const cw = autoCanvas(wrapW.frame, { aspect: 0.5, minH: 170, label: 'Planning world: start, goal, obstacles, candidate plans and the executed trail', draw: () => draw() });
      const ro = h('div', { class: 'readout', 'aria-live': 'off' }); wrapW.wrap.appendChild(ro);
      const row = h('div', { class: 'fig-row' }); stage.appendChild(row);
      const Fp = lib.frame(row, { label: 'Energy while thinking', sub: 'E of every candidate, log scale' }); const fpSub = Fp.wrap.querySelector('.fig-sub');
      Fp.wrap.style.flex = '1 1 300px';
      const cp = autoCanvas(Fp.frame, { aspect: 0.5, minH: 172, maxH: 180, label: 'Energy of each candidate plan per thinking iteration, and thinking spent per move', draw: () => draw() });
      const Ft = lib.frame(row, { label: 'Energy terms', sub: 'of the chosen plan' });
      Ft.wrap.style.flex = '1 1 190px';
      const terms = h('div', { class: 'planning-terms' }); Ft.frame.appendChild(terms);
      const TERM = [['goal', 'goal'], ['obs', 'obstacles'], ['eff', 'effort'], ['smooth', 'smoothness'], ['dyn', 'dynamics']];
      const termEls = {};
      TERM.forEach(([k, l]) => { const bar = h('i'), v = h('span', { class: 'v' }); termEls[k] = { bar, v }; terms.appendChild(h('div', { class: 'planning-trow' }, h('span', { class: 'l' }, l), h('span', { class: 'tr' }, bar), v)); });
      const totEl = h('span', { class: 'v' }); terms.appendChild(h('div', { class: 'planning-trow tot' }, h('span', { class: 'l' }, 'total E'), h('span', { class: 'tr' }), totEl));
      const lamEl = h('div', { class: 'planning-lam' }); terms.appendChild(lamEl);

      // controls
      const c1 = h('div', { class: 'controls' }); stage.appendChild(c1);
      const bThink = lib.button('think ×1', () => { stopPlay(); clearRace(); sim.think(); if (ghost && ghostKind) ghost.think(); draw(); });
      const bThink20 = lib.button('×20', () => { stopPlay(); clearRace(); for (let i = 0; i < 20; i++) { sim.think(); if (ghost) ghost.think(); } draw(); });
      const bAct = lib.button('act', () => { stopPlay(); clearRace(); doAct(); kick(); });
      const bPlay = lib.button('play', () => { if (race) { if (race.done) startRace(); kick(); return; } if (playing && autoThink) { autoThink = false; kick(); draw(); return; } if (playing) { stopPlay(); draw(); return; } if (sim.terminal) { resetSims(); } playing = true; autoThink = false; kick(); }, { primary: true });
      const bReset = lib.button('reset', () => { stopPlay(); resetSims(); draw(); });
      const bNew = lib.button('new draw', () => { stopPlay(); cfg.seed = (cfg.seed % 97) + 1 + Math.floor(Math.random() * 40); resetSims(); if (race) startRace(); draw(); });
      c1.append(bThink, bThink20, bAct, bPlay, bReset, bNew);
      const c2 = h('div', { class: 'controls' }); stage.appendChild(c2);
      const segPlanner = lib.segmented({ label: 'Planner', options: [['grad', 'gradient'], ['cem', 'CEM']], value: cfg.planner, onchange: (v) => { cfg.planner = v; segVars.el.classList.toggle('planning-dim', v === 'cem'); clearRace(); restart(); } });
      const segVars = lib.segmented({ label: 'Plan variables', options: [['joint', 'a + ŝ'], ['shoot', 'a only']], value: cfg.vars, onchange: (v) => { cfg.vars = v; clearRace(); restart(); } });
      const segLoop = lib.segmented({ label: 'Control loop', options: [['mpc', 'MPC'], ['open', 'open']], value: cfg.loop, onchange: (v) => { cfg.loop = v; clearRace(); restart(); } });
      c2.append(h('span', { class: 'fig-label' }, 'planner'), segPlanner.el, h('span', { class: 'fig-label' }, 'ŷ'), segVars.el, h('span', { class: 'fig-label' }, 'loop'), segLoop.el);
      const c3 = h('div', { class: 'controls' }); stage.appendChild(c3);
      const slM = lib.slider({ id: 'planning-M', label: 'candidates M', min: 1, max: 12, step: 1, value: cfg.M, fmt: (v) => String(v), oninput: (v) => { cfg.M = v; if (race) { startRace(); return; } sim.setM(v); draw(); } });
      const slWind = lib.slider({ id: 'planning-wind', label: 'wind (unknown to planner)', min: 0, max: 0.15, step: 0.01, value: cfg.wind, fmt: (v) => v.toFixed(2), oninput: (v) => { cfg.wind = v; if (ghost) ghost.cfg.wind = v; draw(); } });
      c3.append(slM.el, slWind.el);
      // more settings (hidden until [ more ] is pressed, to keep the figure within one screen)
      const det = h('div', { class: 'planning-more', hidden: true }); stage.appendChild(det);
      const bMore = lib.button('more', () => { det.hidden = !det.hidden; bMore.setAttribute('aria-pressed', String(!det.hidden)); bMore.setAttribute('aria-expanded', String(!det.hidden)); });
      bMore.setAttribute('aria-expanded', 'false'); c1.appendChild(bMore);
      const c4 = h('div', { class: 'controls' }); det.appendChild(c4);
      const segScen = lib.segmented({ label: 'World', options: Object.keys(SCEN).map(k => [k, SCEN[k].label]), value: scen, onchange: (v) => { stopPlay(); clearRace(); loadScen(v); draw(); } });
      const segR = lib.segmented({ label: 'Replan every', options: [[1, '1 move'], [5, '5 moves']], value: cfg.R, onchange: (v) => { cfg.R = v; restart(); } });
      const segWarm = lib.segmented({ label: 'Warm start', options: [[true, 'shift plan'], [false, 'fresh random']], value: cfg.warm, onchange: (v) => { cfg.warm = v; restart(); } });
      const segH = lib.segmented({ label: 'Horizon H', options: [[6, '6'], [12, '12'], [24, '24']], value: cfg.H, onchange: (v) => { cfg.H = v; if (race) { startRace(); draw(); return; } restart(); } });
      c4.append(h('span', { class: 'fig-label' }, 'world'), segScen.el, h('span', { class: 'fig-label' }, 'replan every'), segR.el);
      const c4b = h('div', { class: 'controls' }); det.appendChild(c4b);
      c4b.append(h('span', { class: 'fig-label' }, 'warm start'), segWarm.el, h('span', { class: 'fig-label' }, 'horizon H'), segH.el);
      const c5 = h('div', { class: 'controls' }); det.appendChild(c5);
      const slSig = lib.slider({ id: 'planning-sigma', label: 'Langevin noise σ', min: 0, max: 0.05, step: 0.005, value: cfg.sigma, fmt: (v) => v.toFixed(3), oninput: (v) => { cfg.sigma = v; } });
      const slWd = lib.slider({ id: 'planning-wd', label: 'dynamics weight w_d', min: 0, max: 8, step: 0.1, value: cfg.wd, fmt: (v) => v.toFixed(1), oninput: (v) => { cfg.wd = v; sim.evalAll(); draw(); } });
      const slK = lib.slider({ id: 'planning-K', label: 'iterations per move K', min: 0, max: 100, step: 1, value: cfg.K, fmt: (v) => String(v), oninput: (v) => { cfg.K = v; } });
      c5.append(slSig.el, slWd.el, slK.el);
      det.appendChild(h('p', { class: 'planning-tip' }, 'Drag the obstacles, the goal or the start. Double-click empty space to add an obstacle, or an obstacle to remove it.'));

      // ---------- world / sims ----------
      function loadScen(id) { scen = id; segScen.set(id); world = cloneW(SCEN[id].world); fieldDirty = true; resetSims(); }
      function resetSims() { sim = new Sim(world, cfg); if (ghostKind) makeGhost(ghostKind); else ghost = null; anim = ganim = null; }
      function restart() { stopPlay(); resetSims(); draw(); }
      function makeGhost(kind) {
        ghostKind = kind;
        if (!kind) { ghost = null; return; }
        const gc = Object.assign({}, cfg, kind === 'open' ? { loop: 'open' } : { planner: 'cem', loop: 'mpc' });
        ghost = new Sim(world, gc);
      }
      function syncControls() {
        segVars.el.classList.toggle('planning-dim', cfg.planner === 'cem');
        segPlanner.set(cfg.planner); segVars.set(cfg.vars); segLoop.set(cfg.loop); segR.set(cfg.R); segScen.set(scen); segWarm.set(cfg.warm); segH.set(cfg.H);
        slM.set(cfg.M); slWind.set(cfg.wind); slSig.set(cfg.sigma); slWd.set(cfg.wd); slK.set(cfg.K);
      }
      function startRace() { race = new Race(world, sim.s, cfg, cfg.seed); view.plotX = 'fe'; }
      function clearRace() { if (race) { race = null; view.plotX = 'iter'; } }

      // ---------- energy field (cost felt by a single state: goal pull + obstacles) ----------
      let field = null;
      function buildField(w, hh) {
        const nx = 100, ny = 50, g = [0, 0], P = Object.assign({}, X.W), grid = [];
        let hi = 0, lo = Infinity;
        for (let j = 0; j < ny; j++) { const rowv = []; for (let i = 0; i < nx; i++) { const x = (i + 0.5) / nx * X.WX, y = X.WY - (j + 0.5) / ny * X.WY; const e = Math.log1p(P.wg * X.rho(x - world.goal[0], y - world.goal[1]) + P.wo * X.obsCost(world, x, y, P.m, g)); rowv.push(e); if (e > hi) hi = e; if (e < lo) lo = e; } grid.push(rowv); }
        const dpr = Math.min(2, window.devicePixelRatio || 1), cv = document.createElement('canvas'); cv.width = Math.round(w * dpr); cv.height = Math.round(hh * dpr);
        const g2 = cv.getContext('2d'); g2.scale(dpr, dpr);
        const q = lib.quantile(grid, 0.8);
        g2.globalAlpha = 0.34; lib.heatmap(g2, grid, 0, 0, w, hh, { range: [lo, q], gamma: 0.9 }); g2.globalAlpha = 1;
        const levels = []; for (let k = 1; k <= 7; k++) levels.push(lo + (q - lo) * k / 8);
        lib.contours(g2, grid, 0, 0, w, hh, levels, { color: 'rgba(17,17,17,0.08)', width: 1 });
        field = { cv, w, h: hh }; fieldDirty = false;
      }

      // ---------- drawing: world ----------
      function draw() { drawWorld(); drawPlot(); drawTerms(); readouts(); }
      function drawWorld() {
        const { ctx: g, w, h: hh } = cw, sc = w / X.WX, PX = (x) => x * sc, PY = (y) => (X.WY - y) * sc;
        g.save(); g.setTransform(cw.dpr, 0, 0, cw.dpr, 0, 0); g.fillStyle = '#fff'; g.fillRect(0, 0, w, hh);
        if (fieldDirty || !field || field.w !== w || field.h !== hh) buildField(w, hh);
        g.drawImage(field.cv, 0, 0, w, hh);
        // grid
        g.strokeStyle = 'rgba(17,17,17,0.06)'; g.lineWidth = 1; g.beginPath();
        for (let x = 1; x < X.WX; x++) { g.moveTo(PX(x) + 0.5, 0); g.lineTo(PX(x) + 0.5, hh); } for (let y = 1; y < X.WY; y++) { g.moveTo(0, PY(y) + 0.5); g.lineTo(w, PY(y) + 0.5); } g.stroke();
        // obstacles: white disc, hairline outline, hatching, dashed margin
        world.obs.forEach((o, i) => {
          const cx = PX(o.x), cy = PY(o.y), r = o.r * sc;
          g.setLineDash([3, 4]); g.strokeStyle = 'rgba(17,17,17,0.28)'; g.lineWidth = 1; g.beginPath(); g.arc(cx, cy, (o.r + X.W.m) * sc, 0, 7); g.stroke(); g.setLineDash([]);
          g.fillStyle = '#fff'; g.beginPath(); g.arc(cx, cy, r, 0, 7); g.fill();
          g.save(); g.beginPath(); g.arc(cx, cy, r, 0, 7); g.clip(); g.strokeStyle = 'rgba(17,17,17,0.22)'; g.lineWidth = 1; g.beginPath();
          for (let d = -2 * r; d <= 2 * r; d += 6) { g.moveTo(cx + d - r, cy + r); g.lineTo(cx + d + r, cy - r); } g.stroke(); g.restore();
          g.strokeStyle = drag && drag.i === i ? BLUE : INK; g.lineWidth = 1.2; g.beginPath(); g.arc(cx, cy, r, 0, 7); g.stroke();
        });
        // goal: dashed reach circle + crosshair
        const gx = PX(world.goal[0]), gy = PY(world.goal[1]);
        g.strokeStyle = INK; g.lineWidth = 1.2; g.setLineDash([3, 3]); g.beginPath(); g.arc(gx, gy, X.REACH * sc, 0, 7); g.stroke(); g.setLineDash([]);
        g.beginPath(); g.moveTo(gx - 7, gy); g.lineTo(gx + 7, gy); g.moveTo(gx, gy - 7); g.lineTo(gx, gy + 7); g.stroke();
        label(g, 'goal g', gx, gy - X.REACH * sc - 4, INK, 'center', 'bottom');
        // start
        const sx = PX(world.start[0]), sy = PY(world.start[1]);
        g.strokeStyle = MUTED; g.lineWidth = 1.2; g.beginPath(); g.arc(sx, sy, 8, 0, 7); g.stroke();
        if (sim.k === 0 && !anim) label(g, 's₀', sx, sy + 11, MUTED, 'center', 'top');
        const path = (Xs, o) => {
          g.save(); g.globalAlpha = o.alpha == null ? 1 : o.alpha; g.strokeStyle = o.color; g.lineWidth = o.width || 1.4; g.setLineDash(o.dash || []); g.lineJoin = 'round'; g.beginPath();
          for (let t = 0; t < Xs.length / 2; t++) { const px = PX(Xs[2 * t]), py = PY(Xs[2 * t + 1]); t ? g.lineTo(px, py) : g.moveTo(px, py); }
          g.stroke(); g.setLineDash([]);
          if (o.dots) { g.fillStyle = o.dotFill || o.color; for (let t = 1; t < Xs.length / 2; t++) { g.beginPath(); g.arc(PX(Xs[2 * t]), PY(Xs[2 * t + 1]), o.dots, 0, 7); g.fill(); if (o.dotStroke) { g.strokeStyle = o.dotStroke; g.lineWidth = 1; g.stroke(); } } }
          g.restore();
        };
        const P = sim.P();
        // ---- race view: CEM samples (gray) and gradient candidates (blue) from the same start
        if (race) {
          const st = race.cem;
          if (st.last) st.last.forEach((smp, j) => { if (j % 2 === 0 || j < race.cfg.elite) path(X.rollout(race.s0, smp.A, race.H), { color: j < race.cfg.elite ? INK : GRAY, width: 1, alpha: j < race.cfg.elite ? 0.35 : 0.13 }); });
          const cA = race.cPlan().A; path(X.rollout(race.s0, cA, race.H), { color: INK, width: 2.2, dash: [6, 4], dots: 2.6 });
          const gp = race.gPlan();
          gp.all.forEach((cd, j) => { if (cd !== gp.plan) path(X.rollout(race.s0, cd.A, race.H), { color: BLUE, width: 1.2, alpha: 0.3 }); });
          path(X.rollout(race.s0, gp.plan.A, race.H), { color: BLUE, width: 2.6, dots: 3 });
        } else {
          // ---- ghost (open loop or CEM): gray
          if (ghost) {
            if (ghost.frozen) path(ghost.frozen.X, { color: GRAY, width: 1.6, dash: [5, 4], alpha: 0.8 });
            else if (!ghost.terminal) {
              const gs = ganim ? ganim.to : ghost.s;
              if (ghost.cfg.planner === 'cem' && ghost.cem.last && ghost.k === 0) ghost.cem.last.forEach((smp, j) => { if (j % 3 === 0) path(X.rollout(gs, smp.A, ghost.cfg.H), { color: GRAY, width: 1, alpha: 0.12 }); });
              path(X.positions(gs, ghost.plan(), ghost.P()), { color: GRAY, width: 1.8, dash: [5, 4], alpha: 0.9 });
            }
            drawTrail(g, ghost, ganim, PX, PY, true);
          }
          // ---- main sim
          const s0 = anim ? anim.to : sim.s;
          if (sim.frozen) {
            path(sim.frozen.X, { color: BLUE, width: 2.2, dots: 2.6, alpha: 0.85, dash: [7, 5] });
          } else if (!sim.terminal) {
            if (cfg.planner === 'cem') {
              if (sim.cem.last) sim.cem.last.forEach((smp, j) => { if (j % 2 === 0) path(X.rollout(s0, smp.A, cfg.H), { color: j < cfg.elite ? BLUE : GRAY, width: 1, alpha: j < cfg.elite ? 0.3 : 0.12 }); });
              path(X.positions(s0, sim.plan(), P), { color: BLUE, width: 2.6, dots: 3 });
            } else {
              const eb = sim.E[sim.best];
              const fresh = sim.it === 0; sim.cands.forEach((cd, j) => { if (j !== sim.best) path(X.positions(s0, cd, P), { color: BLUE, width: 1.2, alpha: fresh ? 0.42 : sim.E[j] < 3 * eb + 2 ? 0.3 : 0.07, dots: 1.6 }); });
              const b = sim.cands[sim.best];
              if (view.avg) { const r = rival(); if (r >= 0) { const avg = mixPlan(b, sim.cands[r], 0.5), Xa = X.positions(s0, avg, P); path(Xa, { color: INK, width: 1.6, dash: [6, 5] }); const ea = X.energyOnly(world, s0, avg, P); const mid = Math.floor(cfg.H / 2); label(g, 'average of the two routes · E ' + fmtE(ea) + ' (' + Math.round(ea / Math.max(1e-9, sim.E[sim.best])) + '× the best)', PX(Xa[2 * mid]), PY(Xa[2 * mid + 1]) - 10, INK, 'center', 'bottom'); } }
              if (P.mode === 'joint') { const Rr = X.rollout(s0, b.A, cfg.H); let gap = 0; for (let i = 2; i < Rr.length; i++) gap = Math.max(gap, Math.abs(Rr[i] - (i < 2 ? 0 : b.S[i - 2]))); if (gap > 0.35) { path(Rr, { color: INK, width: 1.4, dash: [2, 3], alpha: 0.85 }); const t = cfg.H; label(g, 'real effect of the actions', PX(Rr[2 * t]), PY(Rr[2 * t + 1]) + 8, INK, 'center', 'top'); } }
              path(X.positions(s0, b, P), { color: BLUE, width: 2.6, dots: 3, dotFill: '#fff', dotStroke: BLUE });
            }
          }
          drawTrail(g, sim, anim, PX, PY, false);
        }
        // wind gauge
        if (cfg.wind > 0) {
          const x0 = w - 64, y0 = 10;
          g.fillStyle = 'rgba(255,255,255,0.85)'; g.fillRect(x0 - 6, y0 - 4, 62, 50);
          label(g, 'wind', x0 + 25, y0, MUTED, 'center', 'top');
          for (let i = 0; i < 3; i++) lib.arrow(g, x0 + 9 + i * 16, y0 + 18, x0 + 9 + i * 16, y0 + 20 + Math.min(22, cfg.wind * 160), { color: INK, width: 1.2, head: 5 });
        }
        // status banner
        const msg = banner(); if (msg) { g.font = `500 12px ${lib.F ? lib.F.mono : 'monospace'}`; const tw = g.measureText(msg).width + 18; g.fillStyle = 'rgba(255,255,255,0.94)'; g.fillRect(8, 8, tw, 24); g.strokeStyle = INK; g.lineWidth = 1; g.strokeRect(8.5, 8.5, tw - 1, 23); g.fillStyle = INK; g.textBaseline = 'middle'; g.textAlign = 'left'; g.fillText(msg, 17, 20.5); }
        g.restore();
      }
      function drawTrail(g, S, an, PX, PY, isGhost) {
        const tr = S.trail.slice(); if (an) tr[tr.length - 1] = [lib.lerp(an.from[0], an.to[0], an.e), lib.lerp(an.from[1], an.to[1], an.e)];
        const col = isGhost ? GRAY : INK;
        if (tr.length > 1) { g.save(); g.strokeStyle = col; g.lineWidth = isGhost ? 1.6 : 2; if (isGhost) g.setLineDash([5, 4]); g.beginPath(); tr.forEach((p, i) => i ? g.lineTo(PX(p[0]), PY(p[1])) : g.moveTo(PX(p[0]), PY(p[1]))); g.stroke(); g.restore(); g.fillStyle = col; tr.slice(0, -1).forEach(p => { g.beginPath(); g.arc(PX(p[0]), PY(p[1]), 2, 0, 7); g.fill(); }); }
        const rp = tr[tr.length - 1], cx = PX(rp[0]), cy = PY(rp[1]);
        if (S.status === 'crashed' && !an) { g.strokeStyle = isGhost ? GRAY : INK; g.lineWidth = 2.4; g.beginPath(); g.moveTo(cx - 7, cy - 7); g.lineTo(cx + 7, cy + 7); g.moveTo(cx + 7, cy - 7); g.lineTo(cx - 7, cy + 7); g.stroke(); }
        else if (isGhost) { g.fillStyle = '#fff'; g.strokeStyle = GRAY; g.lineWidth = 2; g.beginPath(); g.arc(cx, cy, 6, 0, 7); g.fill(); g.stroke(); }
        else { g.fillStyle = INK; g.beginPath(); g.arc(cx, cy, 6.5, 0, 7); g.fill(); g.strokeStyle = BLUE; g.lineWidth = 2; g.beginPath(); g.arc(cx, cy, 9, 0, 7); g.stroke(); }
        if (isGhost && S.k > 0) label(g, ghostKind === 'open' ? 'open loop' : 'CEM', cx, cy + 10, MUTED, 'center', 'top');
      }
      function label(g, t, x, y, color, align, base) {
        g.save(); g.font = `400 11.5px ${lib.F ? lib.F.mono : 'monospace'}`; g.textAlign = align || 'left'; g.textBaseline = base || 'bottom';
        const tw = g.measureText(t).width; if (align === 'center') x = clamp(x, tw / 2 + 4, cw.w - tw / 2 - 4); y = clamp(y, base === 'top' ? 2 : 16, cw.h - (base === 'top' ? 16 : 2));
        const bx = align === 'center' ? x - tw / 2 : align === 'right' ? x - tw : x, by = base === 'top' ? y : base === 'middle' ? y - 7 : y - 14;
        g.fillStyle = 'rgba(255,255,255,0.82)'; g.fillRect(bx - 3, by - 1, tw + 6, 16); g.fillStyle = color; g.fillText(t, x, y); g.restore();
      }
      function banner() {
        const s = sim.status, ex = (S, who) => {
          if (S.status === 'reached') return `${who}reached the goal in ${S.k} moves`;
          if (S.status === 'crashed') return `${who}crashed at move ${S.k}`;
          if (S.status === 'missed') return `${who}plan used up, missed by ${S.dist().toFixed(2)}`;
          if (S.status === 'timeout') return `${who}out of time after ${S.k} moves`;
          if (S.status === 'stuck') return `${who}stuck: a local minimum of E`;
          return null;
        };
        if (race) return race.done ? 'equal budget spent: ' + nf(race.budget) + ' forward-pass equivalents each' : 'race: ' + nf(race.clock || 0) + ' / ' + nf(race.budget) + ' FE each';
        const a = ex(sim, ghost ? (cfg.planner === 'cem' ? 'CEM: ' : 'MPC: ') : ''), b = ghost ? ex(ghost, ghostKind === 'open' ? 'open loop: ' : 'CEM: ') : null;
        if (ghostKind === 'cem' && a) return a.replace('MPC: ', 'gradient: ') + (b ? ' · ' + b : '');
        return [a, b].filter(Boolean).join(' · ') || (s === 'ready' ? null : null);
      }
      function rival() { // the converged candidate whose route differs most from the chosen one
        if (sim.cands.length < 2) return -1;
        const P = sim.P(), b = sim.best, Xb = X.positions(sim.s, sim.cands[b], P), lim = sim.E[b] * 2 + 3; let best = -1, bd = 0.6;
        sim.cands.forEach((cd, j) => { if (j === b || !(sim.E[j] < lim)) return; const Xj = X.positions(sim.s, cd, P); let d = 0; for (let i = 2; i < Xj.length; i++) d += (Xj[i] - Xb[i]) ** 2; d = Math.sqrt(d / (Xj.length - 2)); if (d > bd) { bd = d; best = j; } });
        return best;
      }
      function mixPlan(a, b, t) { return { A: a.A.map((v, i) => (1 - t) * v + t * b.A[i]), S: a.S.map((v, i) => (1 - t) * v + t * b.S[i]) }; }

      // ---------- drawing: energy plot ----------
      function drawPlot() {
        const { ctx: g, w, h: hh } = cp;
        g.save(); g.setTransform(cp.dpr, 0, 0, cp.dpr, 0, 0); g.fillStyle = '#fff'; g.fillRect(0, 0, w, hh);
        const strip = race ? 0 : 46, b = { x: 46, y: 10, w: w - 58, h: hh - 44 - strip };
        let series = [], ghostSeries = null, xmax = 1, xlog = true, xlabel = 'thinking iteration, this move (log)';
        const subTxt = race ? 'lowest E found so far, by compute spent' : ghostKind === 'cem' ? 'lowest E, by compute this move' : cfg.planner === 'cem' ? 'lowest E sampled so far, log scale' : 'E of every candidate, log scale';
        if (fpSub.textContent !== subTxt) fpSub.textContent = subTxt;
        if (race) {
          xmax = race.budget; xlabel = 'forward-pass equivalents (log)';
        } else if (ghostKind === 'cem') {
          xlabel = 'forward-pass equivalents, this move (log)';
          series = sim.cur; ghostSeries = ghost.cur; xmax = Math.max(1, ...series.map(p => p.fe), ...ghostSeries.map(p => p.fe), sim.k === 0 ? 3 * cfg.M * cfg.N0 : 1);
        } else {
          series = sim.cur; xmax = Math.max(10, sim.budget(), series.length);
          if (cfg.planner === 'cem') xlabel = 'CEM iteration, this move (log)';
        }
        // y range (log)
        let lo = Infinity, hi = -Infinity; const see = (v) => { if (v > 0 && isFinite(v)) { const l = Math.log10(v); if (l < lo) lo = l; if (l > hi) hi = l; } };
        if (race) { race.g.forEach(p => see(p[1])); race.c.forEach(p => see(p[1])); race.gE.forEach(see); }
        else { series.forEach(p => { p.E.forEach(see); }); if (ghostSeries) ghostSeries.forEach(p => see(p.min)); sim.E.forEach(see); }
        if (!isFinite(lo)) { lo = 0; hi = 2; } lo = Math.floor(lo * 2) / 2; hi = Math.max(lo + 0.5, Math.ceil(hi * 2) / 2);
        const x0 = race || ghostKind === 'cem' ? 10 : 1, XS = (v) => b.x + (Math.log10(Math.max(x0, v)) - Math.log10(x0)) / (Math.log10(Math.max(xmax, x0 * 10)) - Math.log10(x0)) * b.w;
        const YS = (v) => b.y + b.h - (Math.log10(Math.max(1e-9, v)) - lo) / (hi - lo) * b.h;
        // grid + ticks
        g.font = `400 11px ${lib.F ? lib.F.mono : 'monospace'}`; g.fillStyle = MUTED; g.strokeStyle = RULE; g.lineWidth = 1;
        for (let e = Math.ceil(lo); e <= hi + 1e-9; e++) { const y = Math.round(YS(Math.pow(10, e))) + 0.5; g.beginPath(); g.moveTo(b.x, y); g.lineTo(b.x + b.w, y); g.stroke(); g.textAlign = 'right'; g.textBaseline = 'middle'; g.fillText(e >= 0 && e <= 3 ? String(Math.pow(10, e)) : '1e' + e, b.x - 6, y); }
        g.strokeStyle = FAINT; g.beginPath(); g.moveTo(b.x + 0.5, b.y); g.lineTo(b.x + 0.5, b.y + b.h + 0.5); g.lineTo(b.x + b.w, b.y + b.h + 0.5); g.stroke();
        g.textAlign = 'center'; g.textBaseline = 'top';
        [1, 10, 100, 1000, 10000].forEach(v => { if (v >= x0 && v < xmax * 0.7) { g.fillText(fmtK(v), XS(v), b.y + b.h + 4); g.strokeStyle = RULE; g.beginPath(); g.moveTo(Math.round(XS(v)) + 0.5, b.y); g.lineTo(Math.round(XS(v)) + 0.5, b.y + b.h); g.stroke(); } });
        g.textAlign = 'right'; g.fillText(fmtK(xmax), b.x + b.w, b.y + b.h + 4);
        g.textAlign = 'center'; g.fillStyle = MUTED; g.fillText(xlabel, b.x + b.w / 2, b.y + b.h + 18);
        const poly = (pts, col, wdt, dash, alpha) => { if (pts.length < 1) return; g.save(); g.globalAlpha = alpha == null ? 1 : alpha; g.strokeStyle = col; g.lineWidth = wdt; g.setLineDash(dash || []); g.beginPath(); pts.forEach((p, i) => i ? g.lineTo(p[0], p[1]) : g.moveTo(p[0], p[1])); if (pts.length === 1) g.lineTo(pts[0][0] + 1, pts[0][1]); g.stroke(); g.restore(); };
        g.save(); g.beginPath(); g.rect(b.x, b.y - 4, b.w + 4, b.h + 8); g.clip();
        if (race) {
          poly(race.c.map(p => [XS(p[0]), YS(p[1])]), INK, 2, [6, 4]);
          poly(race.g.map(p => [XS(p[0]), YS(p[1])]), BLUE, 2.4);
          const lg = race.g[race.g.length - 1], lc = race.c[race.c.length - 1];
          if (lg) lib.dot(g, XS(lg[0]), YS(lg[1]), 3.5, BLUE); if (lc) lib.dot(g, XS(lc[0]), YS(lc[1]), 3.5, INK);
        } else if (series.length || (ghostSeries && ghostSeries.length)) {
          const xOf = ghostKind === 'cem' ? (p) => XS(p.fe) : (p, i) => XS(i + 1);
          if (ghostKind !== 'cem' && series.length) { const M = series[series.length - 1].E.length, stride = Math.max(1, Math.floor(series.length / 400)); for (let j = 0; j < M && M > 1; j++) { const pts = []; for (let i = 0; i < series.length; i += stride) if (j < series[i].E.length) pts.push([xOf(series[i], i), YS(series[i].E[j])]); poly(pts, BLUE, 1, null, 0.28); } }
          if (ghostSeries && ghostSeries.length) poly(ghostSeries.map((p) => [XS(p.fe), YS(p.min)]), GRAY, 2, [6, 4]);
          if (series.length) { const stride = Math.max(1, Math.floor(series.length / 600)), pts = []; for (let i = 0; i < series.length; i += stride) pts.push([xOf(series[i], i), YS(series[i].min)]); pts.push([xOf(series[series.length - 1], series.length - 1), YS(series[series.length - 1].min)]); poly(pts, BLUE, 2.4); const lp = pts[pts.length - 1]; lib.dot(g, lp[0], lp[1], 3.5, BLUE); }
        } else {
          sim.E.forEach(v => lib.dot(g, b.x + 4, YS(v), 3, BLUE, { alpha: 0.6 }));
        }
        g.restore();
        // legend for comparisons
        if (race || ghostKind === 'cem') {
          g.font = `400 11px ${lib.F ? lib.F.mono : 'monospace'}`; g.textAlign = 'right'; g.textBaseline = 'top';
          g.fillStyle = BLUE; g.fillText(race ? 'gradient, M = ' + race.M : 'gradient, a + ŝ, M = ' + cfg.M, b.x + b.w - 4, b.y + 2); g.fillStyle = race ? INK : GRAY; g.fillText(race ? 'CEM, 64 samples' : 'CEM over a, 64 samples', b.x + b.w - 4, b.y + 16);
        } else if (!series.length) {
          g.font = `400 11.5px ${lib.F ? lib.F.mono : 'monospace'}`; g.fillStyle = MUTED; g.textAlign = 'center'; g.textBaseline = 'middle';
          g.fillText(sim.frozen ? 'open loop: executing, no more thinking' : sim.terminal ? 'episode over' : playing ? (cfg.R > 1 && sim.since > 0 ? 'executing the plan, no re-planning' : 'moving') : 'iteration 0: press think or play', b.x + b.w / 2, b.y + b.h / 2);
        }
        // strip: thinking per move
        if (strip) {
          const sy = hh - strip + 8, sh = strip - 22, steps = sim.log.slice(); const live = !sim.terminal && !sim.frozen && sim.since === 0 ? { iters: sim.itDec, live: true } : null; if (live) steps.push(live);
          const N = Math.max(cfg.loop === 'open' ? cfg.H : 16, steps.length), bw = b.w / N, maxI = Math.max(10, cfg.planner === 'cem' ? cfg.cem0 : cfg.N0, ...steps.map(s => s.iters));
          const yl = (v) => sy + sh - Math.log10(1 + v) / Math.log10(1 + maxI) * sh;
          g.strokeStyle = RULE; g.beginPath(); g.moveTo(b.x, sy + sh + 0.5); g.lineTo(b.x + b.w, sy + sh + 0.5); g.stroke();
          steps.forEach((s, i) => { const y = yl(s.iters); g.fillStyle = s.live ? lib.rgba(BLUE, 0.4) : BLUE; g.fillRect(b.x + i * bw + bw * 0.18, y, Math.max(1, bw * 0.64), sy + sh - y); });
          g.font = `400 10.5px ${lib.F ? lib.F.mono : 'monospace'}`; g.fillStyle = MUTED; g.textAlign = 'left'; g.textBaseline = 'top';
          g.fillText('thinking per move (log)', b.x, sy + sh + 3); g.textAlign = 'right'; g.fillText(sim.log.length ? 'max ' + nf(maxI) : '', b.x + b.w, sy + sh + 3);
        }
        g.restore();
      }
      function drawTerms() {
        const b = race ? race.gPlan().plan : sim.plan(); if (!b) return;
        const P = race ? race.P : sim.P(), e = X.energy(world, race ? race.s0 : anim ? anim.to : sim.s, b, P), tot = Math.max(1e-9, e.E);
        TERM.forEach(([k]) => { const v = e.T[k], off = k === 'dyn' && P.mode !== 'joint'; termEls[k].bar.style.width = (off ? 0 : Math.min(100, 100 * v / tot)).toFixed(1) + '%'; termEls[k].v.textContent = off ? 'exact' : fmtE(v); });
        totEl.textContent = fmtE(e.E);
      }
      let lamCache = { key: '', txt: '' };
      function updateLam() {
        if (race) { lamEl.innerHTML = `gradient plan shown · CEM best E = ${fmtE(race.cem.bestE)} · α = ${cfg.alphaS}`; return; }
        if (cfg.planner === 'cem' || sim.terminal) { lamEl.innerHTML = ''; return; }
        const key = sim.it + ':' + sim.k + ':' + cfg.vars + ':' + cfg.wd;
        if (lamCache.key !== key && (sim.it % 25 === 0 || !lamCache.txt || sim.itDec <= 1)) {
          const lam = X.lambdaMax(world, sim.s, sim.cands[sim.best], sim.P(), 20), a = sim.alpha();
          lamCache = { key, txt: lam > 0 ? `λ<sub>max</sub> ≈ ${lam.toFixed(0)}, so descent is stable for α &lt; ${(2 / lam).toFixed(3)}; α = ${a}` : `curvature ≈ ${lam.toFixed(0)} &lt; 0: not convex here; α = ${a}` };
        }
        lamEl.innerHTML = lamCache.txt;
      }
      function readouts() {
        if (race) {
          const best = Math.min(race.gBest, race.cem.bestE), thr = best * 1.1 + 0.05, rr = race.reach(thr);
          const f = (x) => x ? `<b>${fmtK(x.fe)}</b> FE · <b>${x.it}</b> it` : '<b>not yet</b>';
          ro.innerHTML = `<span>to reach E ≤ ${fmtE(thr)}:</span><span>gradient ${f(rr.g)}</span><span>CEM ${f(rr.c)}</span>`;
          updateLam(); return;
        }
        const gapTxt = () => { if (cfg.planner === 'cem' || cfg.vars !== 'joint' || sim.frozen) return '0'; const b = sim.plan(), R = X.rollout(sim.s, b.A, cfg.H); return Math.hypot(R[2 * cfg.H] - b.S[2 * cfg.H - 2], R[2 * cfg.H + 1] - b.S[2 * cfg.H - 1]).toFixed(2); };
        ro.innerHTML = `<span>move <b>${sim.k}</b></span><span>iter <b>${nf(sim.itDec)}</b> now · <b>${nf(sim.it)}</b> all</span><span><b>${fmtK(sim.fe)}</b> FE</span><span>E <b>${sim.frozen ? 'fixed' : fmtE(sim.E[sim.best])}</b></span><span>to goal <b>${sim.dist().toFixed(2)}</b></span><span>ŝ gap <b>${gapTxt()}</b></span>`;
        updateLam();
        bPlay.textContent = playing ? 'pause' : sim.terminal ? 'replay' : 'play';
      }

      // ---------- control flow ----------
      function ipf(S) { return S.cfg.planner === 'cem' ? (S.itDec < 6 ? 1 : 2) : Math.min(40, 1 + Math.floor(S.itDec / 12)); } // slow first, then faster
      function doAct() {
        const r = sim.act(); if (r && !lib.reducedMotion) anim = { from: r.from, to: r.to, t: 0, e: 0 };
        if (ghost) { const r2 = ghost.act(); if (r2 && !lib.reducedMotion) ganim = { from: r2.from, to: r2.to, t: 0, e: 0 }; }
      }
      const loop = lib.loop((dt) => {
        if (!visible && !drag) return false;
        if (anim || ganim) {
          [anim, ganim].forEach(a => { if (a) { a.t += dt / 0.16; a.e = lib.ease ? lib.ease(Math.min(1, a.t)) : Math.min(1, a.t); } });
          if (anim && anim.t >= 1) anim = null; if (ganim && ganim.t >= 1) ganim = null; draw(); return true;
        }
        if (drag) { if (!playing && !race && !sim.terminal && !sim.frozen) { for (let i = 0; i < 6; i++) sim.think(); } draw(); return true; }
        if (race && !race.done) { race.advance(lib.reducedMotion ? race.budget : 8 + (race.clock || 0) * 0.012); draw(); return true; }
        if (!playing) { draw(); return false; }
        const t0 = performance.now();
        const busy = (S) => S && !S.terminal && S.wantsThink();
        if (busy(sim) || busy(ghost)) {
          const n1 = ipf(sim), n2 = ghost ? ipf(ghost) : 0;
          for (let i = 0; i < Math.max(n1, n2) && performance.now() - t0 < 8; i++) { if (i < n1 && busy(sim)) sim.think(); if (ghost && i < n2 && busy(ghost)) ghost.think(); }
        } else if (autoThink) { playing = false; autoThink = false; draw(); return false; }
        else if (sim.terminal && (!ghost || ghost.terminal)) { playing = false; draw(); return false; }
        else { if (!sim.terminal) { const r = sim.act(); if (r && !lib.reducedMotion) anim = { from: r.from, to: r.to, t: 0, e: 0 }; } if (ghost && !ghost.terminal) { const r2 = ghost.act(); if (r2 && !lib.reducedMotion) ganim = { from: r2.from, to: r2.to, t: 0, e: 0 }; } }
        draw(); return true;
      });
      function kick() { if (visible || drag) loop.start(); else draw(); }
      function stopPlay() { playing = false; autoThink = false; }
      function finishInstant() { // reduced motion: jump to the end state of the current activity
        let guard = 0;
        if (race) { while (!race.done && guard++ < 1000) race.advance(race.budget); return; }
        while (playing && guard++ < 200000) {
          const busy = (S) => S && !S.terminal && S.wantsThink();
          if (busy(sim) || busy(ghost)) { if (busy(sim)) sim.think(); if (busy(ghost)) ghost.think(); }
          else if (autoThink || (sim.terminal && (!ghost || ghost.terminal))) { playing = false; autoThink = false; }
          else { if (!sim.terminal) sim.act(); if (ghost && !ghost.terminal) ghost.act(); }
        }
      }

      // ---------- direct manipulation ----------
      const toWorld = (ev) => { const r = cw.c.getBoundingClientRect(); return [(ev.clientX - r.left) / r.width * X.WX, X.WY - (ev.clientY - r.top) / r.height * X.WY]; };
      function hit(p) {
        const tol = 10 / (cw.w / X.WX);
        if (Math.hypot(p[0] - world.goal[0], p[1] - world.goal[1]) < X.REACH + tol * 0.4) return { kind: 'goal' };
        if (Math.hypot(p[0] - world.start[0], p[1] - world.start[1]) < 0.25 + tol * 0.4) return { kind: 'start' };
        for (let i = world.obs.length - 1; i >= 0; i--) { const o = world.obs[i]; if (Math.hypot(p[0] - o.x, p[1] - o.y) < o.r) return { kind: 'obs', i, dx: p[0] - o.x, dy: p[1] - o.y }; }
        return null;
      }
      cw.c.addEventListener('pointerdown', (ev) => { const p = toWorld(ev), t = hit(p); if (!t) return; drag = t; clearRace(); try { cw.c.setPointerCapture(ev.pointerId); } catch (e) { /* ignore */ } ev.preventDefault(); loop.start(); });
      cw.c.addEventListener('pointermove', (ev) => {
        const p = toWorld(ev);
        if (!drag) { const t = hit(p); cw.c.style.cursor = t ? 'grab' : 'default'; return; }
        if (drag.kind === 'goal') world.goal = [clamp(p[0], 0.4, X.WX - 0.4), clamp(p[1], 0.4, X.WY - 0.4)];
        else if (drag.kind === 'start') { world.start = [clamp(p[0], 0.4, X.WX - 0.4), clamp(p[1], 0.4, X.WY - 0.4)]; resetSims(); }
        else { const o = world.obs[drag.i]; o.x = clamp(p[0] - drag.dx, 0, X.WX); o.y = clamp(p[1] - drag.dy, 0, X.WY); }
        fieldDirty = true; sim.evalAll(); if (ghost) ghost.evalAll();
      });
      const endDrag = () => { if (drag) { drag = null; draw(); } };
      cw.c.addEventListener('pointerup', endDrag); cw.c.addEventListener('pointercancel', endDrag);
      cw.c.addEventListener('dblclick', (ev) => { const p = toWorld(ev), t = hit(p); if (t && t.kind === 'obs') world.obs.splice(t.i, 1); else if (!t) world.obs.push({ x: p[0], y: p[1], r: 0.6 }); fieldDirty = true; clearRace(); sim.evalAll(); if (ghost) ghost.evalAll(); draw(); });

      // ---------- steps ----------
      function setup(o) {
        stopPlay(); race = null; view.plotX = 'iter'; view.avg = !!o.avg;
        Object.assign(cfg, DEFAULTS, o.cfg || {});
        ghostKind = o.ghost || null;
        loadScen(o.scen || 'pillar'); syncControls();
      }
      function play(auto) { playing = true; autoThink = !!auto; if (lib.reducedMotion) { finishInstant(); draw(); } else kick(); }
      const STEPS = [
        () => { setup({ scen: 'pillar', cfg: { M: 6, seed: 2 } }); draw(); },
        () => { setup({ scen: 'pillar', cfg: { M: 6, seed: 2 } }); play(true); },
        () => { setup({ scen: 'pillar', cfg: { M: 8, seed: 5 }, avg: true }); play(true); },
        () => { setup({ scen: 'pillar', cfg: { M: 4, seed: 2, wd: 0.3, loop: 'open' } }); play(true); },
        () => { setup({ scen: 'slalom', cfg: { M: 6, seed: 1 } }); play(false); },
        () => { setup({ scen: 'gap', cfg: { M: 6, seed: 2, wind: 0.06 }, ghost: 'open' }); play(false); },
        () => { setup({ scen: 'slalom', cfg: { M: 2, seed: 3, vars: 'shoot' } }); startRace(); if (lib.reducedMotion) finishInstant(); kick(); draw(); },
        () => { setup({ scen: 'cup', cfg: { M: 8, seed: 3 }, ghost: 'cem' }); play(false); },
      ];
      loadScen('pillar');
      (document.fonts && document.fonts.ready ? document.fonts.ready : Promise.resolve()).then(() => draw());
      ctx.setCaption('Shading: energy one position would get from the goal and obstacle terms (darker = lower); dashed rings: obstacle margins. FE: forward-pass equivalents, 3 per gradient step per plan, 1 per CEM sample. ŝ gap: imagined end vs where the actions really lead.');
      return {
        step(i) { (STEPS[i] || STEPS[0])(); },
        show() { visible = true; [cw, cp].forEach(o => o.fit()); draw(); if (playing || (race && !race.done)) loop.start(); },
        hide() { visible = false; loop.stop(); },
      };

      // ---------- canvas helper: logical size = CSS size, so text is drawn at its true size on every screen ----------
      function autoCanvas(parent, o) {
        const box = h('div', { class: 'canvas-box' }); parent.appendChild(box);
        const c = h('canvas', { role: 'img', 'aria-label': o.label || '' }); box.appendChild(c);
        const r = { c, ctx: c.getContext('2d'), w: 0, h: 0, box, dpr: 1 };
        r.fit = () => {
          const w = Math.max(200, Math.round(box.clientWidth || 600)), hh = Math.round(clamp(w * o.aspect, o.minH || 0, o.maxH || 1e9)), dpr = Math.min(2, window.devicePixelRatio || 1);
          if (w === r.w && hh === r.h && dpr === r.dpr) return false;
          r.w = w; r.h = hh; r.dpr = dpr; c.width = Math.round(w * dpr); c.height = Math.round(hh * dpr); c.style.height = hh + 'px'; return true;
        };
        r.fit();
        if (window.ResizeObserver) new ResizeObserver(() => { if (r.fit() && o.draw) o.draw(); }).observe(box);
        return r;
      }
    },
  });
})(typeof window !== 'undefined' ? window : globalThis);
