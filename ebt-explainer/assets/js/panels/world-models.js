/* Panel: what EBTs could mean for JEPA world models, continual learning and memory. BEYOND THE PAPER (discussion).
   Figure: (1) a JEPA diagram that grows an energy head, a planner and a surprise monitor; (2) the real toy 2-D EBT
   (data/toy2d.json, final checkpoint) read as a latent dynamics model: a weak "JEPA predictor" guess, the learned energy
   over candidate next latents, thinking from the guess, joint descent on (action, predicted latent) toward a goal
   (App. A.3), and energy surprise on a stream whose dynamics change. Below the panel: a searchable glossary and a map
   of every figure and table in the paper. */
(function () {
  'use strict';
  const TAU = 2 * Math.PI;
  // the toy's data-generating process (data/toy2d.json dataset): next latent = mu(x) + sigma(x) * N(0, I)
  const mu = (x) => [1.3 * Math.sin(TAU * x), 0.9 * Math.sin(2 * TAU * x)];
  const sig = (x) => 0.03 + 0.27 * Math.sin(Math.PI * (x - 0.25)) ** 2;
  // a deliberately weak predictor: least squares on first-harmonic features of x. It recovers the loop's width (1.3 sin 2πx)
  // but projects the figure-eight twist (0.9 sin 4πx) to zero.
  const pred = (x) => [1.3 * Math.sin(TAU * x), 0];
  const grp = (x) => { const s = sig(x); return s < 0.1 ? 0 : s > 0.2 ? 2 : 1; };

  EBT.panel({
    id: 'world-models',
    nav: 'JEPA world models',
    title: 'For JEPA world models, continual learning and memory',
    lede: 'A discussion panel, beyond the paper: what an energy-based verifier would add to a world model that predicts in latent space, plans, and keeps learning. A real toy EBT, trained for this explainer, stands in for the latent dynamics.',
    text: `
      <p>None of this is in the paper. It never mentions JEPA, and its only world-model passage is the App. A.3 paragraph on the previous panel. What follows is our reading, for a typical JEPA world model: a ViT encoder, an action-conditioned Transformer predictor (for example with AdaLN-zero conditioning on the action), an EMA target encoder, and a sampling planner such as CEM that re-plans every few steps.</p>
      <details class="wm-quote"><summary>App. A.3 "World Models" (p.26), verbatim</summary><blockquote>"In this work, we focus on autoregressive and bidirectional models over just state information (no actions). EBTs offer high promise in modeling states and actions due to the nature of EBMs learning a distribution over all possible inputs. Particularly, given a model trained to estimate the unnormalized joint distribution of the current context, future, as well as future actions, such world models could implicitly be used as policies to generate actions to achieve a specific state, similar to [136–138]. This would involve holding the current context (past states) constant, and minimizing the energy by propagating the gradient back to the action inputs and future state predictions. Thus, world models trained in this manner become capable of more than just predicting the future, but also in decision making to achieve a specific goal state."</blockquote><p class="wm-small">[136–138] include "Planning with diffusion for flexible behavior synthesis" and DINO-WM. No experiment in the paper uses actions.</p></details>
      <p>A JEPA trains a deterministic predictor to regress the next latent, $\\hat z_{t+1}=g_\\phi(z_t,a_t)$, against an EMA target encoder $\\bar f$ with a stop-gradient, for example $\\mathcal L=\\|\\hat z_{t+1}-\\mathrm{sg}(\\bar f(o_{t+1}))\\|^2$ (L1 is also common). Read as an energy-based model, as in Dawid and LeCun [42], which the paper cites for the unnormalized view (p.6), its energy is</p>
      <div class="eq">$$E_{\\text{JEPA}}(z_t,a_t,z')=\\|g_\\phi(z_t,a_t)-z'\\|^2 .$$<span class="why">z' is any candidate next latent. The minimum is always at the predictor's output, with value 0, in every context.</span></div>
      <p>That energy has a fixed shape. Inference is amortized into one forward pass, so there is nothing to descend, nothing to rank, and no energy level to read as confidence until the future arrives. An EBT-style head $E_\\psi(z_t,a_t,\\hat z)$ learns the shape instead. Trained with Alg. 1 on latent targets, its basin follows the data, and the paper's tools apply: think by descending $\\hat z$, verify by comparing candidates (Alg. 2), and read the energy as a signal.</p>`,
    steps: [
      { label: 'JEPA as it is', html: '<p>The plane is the toy\'s 2-D latent space for one context (read the context phase $x$ as $(z_t,a_t)$). The dashed loop holds the mean next latent of every context; the dashed circle spans ±2σ of outcomes here; the crosshair is one outcome that occurred. The square is a weak predictor\'s guess: it has the loop\'s width but not its twist. Shading is $E_{\\text{JEPA}}$, a bowl centered on the guess.</p>' },
      { label: 'Add an energy head', html: '<p>Shading is now a learned energy: the toy EBT from the landscape panels (an 8→64→64→64→1 MLP), with the context as $x$. Its basin sits on the data, so the guess has measurably higher energy than the basin floor, and a verifier that thinks can tell the guess is off <em>before</em> the outcome arrives. The slice plot compares both energies on the line from guess to outcome. Across 80 contexts the gap $E(\\text{guess})-E(\\hat z^*)$ tracks the squared error of the guess almost exactly (correlation 0.996, about half the squared error), because every basin in this toy has curvature near 0.87. The raw level $E(\\text{guess})$ tracks it less well (0.915): the floor itself moves between −0.61 and −0.28 from context to context. Step 5 runs into this.</p>' },
      { label: 'Think from the guess', html: '<p>Descend $E$ from the predictor\'s guess (blue) and from noise $\\hat y_0\\sim\\mathcal N(0,I)$ (gray). Over 200 contexts, getting within 0.05 of the data mean takes 1.8 steps from the guess and 2.4 from noise at $\\alpha=1$, the step size Alg. 1 shaped this landscape for. At $\\alpha=0.3$ it takes 7.5 against 10.6, and the warm start pays more. A.4 proposes this division of labor: EBTs as "the verifier of predictions initialized by standard feed-forward models" (p.26).</p>' },
      { label: 'Plan through the energy', html: '<p>Now the action is free. Hold $z_t$ and descend $E_\\psi(z_t,a,\\hat z)+\\lambda\\|\\hat z-z_{\\text{goal}}\\|^2$ jointly in the action $a$ and the predicted latent $\\hat z$: A.3\'s "propagating the gradient back to the action inputs and future state predictions". The toy has no separate state, so its context phase plays the action (the dial); the basin moves as $a$ turns, and $\\hat z$ rides it toward the goal. The objective is not convex in $a$, so one start often stalls; we run Alg. 2 on plans instead: $M=6$ starts spread around the dial, keep the lowest objective. Then verify: think again at the chosen $a$, starting from the guess. The model\'s own prediction (hollow circle) lands farther from the goal than $\\hat z$, because the goal term pulls $\\hat z$ off the basin floor, the imagined-state gap of the planning panel. Drag the goal.</p>' },
      { label: 'Surprise as a learning signal', html: '<p>A stream of transitions; at $t=80$ the dynamics rotate by 15°. Before each outcome the model thinks to $\\hat z^*$ (6 steps from noise); when the outcome arrives, surprise is $\\Delta E=E(x,z_{\\text{obs}})-E(x,\\hat z^*)$. The alarm line is the 95th percentile of scores on unchanged data, so 5% of unchanged transitions raise false alarms. With raw ΔE they all come from noisy contexts (11% there, none in precise ones), and precise contexts catch none of the changed transitions. Dividing ΔE by a memory of its typical value in each context (the median over 24 earlier transitions in each of 16 phase bins) spreads the false alarms (7% and 3%) and lets precise contexts catch 92%. Noisy contexts still catch few: a 15° turn is small next to their noise. Switch the score and the kind of change.</p>' },
    ],
    after: `
      <h3>Why raw energy over-alarms</h3>
      <p>Alg. 1 only ever uses $\\nabla_{\\hat y}E$, so adding any function $c(x)$ to the energy changes nothing in training: energy <em>levels</em> across contexts are not trained, only shapes within a context. With MSE targets the expected gradient does not see the noise level either, so this toy's basin is about as sharp for precise as for noisy contexts, and surprise behaves like $c\\,\\|z_{\\text{obs}}-\\hat z^*\\|^2$. The paper does find uncertainty in energies (Figs 8, 11, B.2), for text trained with cross-entropy and for video latents. Whether a JEPA latent behaves the same is open. Ranking candidates within one context (Best-of-N) is grounded; thresholds across contexts need calibration.</p>
      <h3>Continual learning, memory, cost</h3>
      <p>Thinking gains grow with distribution shift (Fig 7, p.11), which suits a learner that keeps meeting new situations. But thinking only repairs the generator. When the dynamics change, the old energy still has its minimum at the old answer, and descent converges to it confidently; surprise is how you notice. The paper's replay buffer is not experience replay: it exists to simulate longer optimization trajectories so the landscape is well defined near its minima (p.7), not to fight forgetting. Training costs ≈3.33× a Transformer++ step per optimization step (p.36), paid on every update by an online learner, and each thinking step at inference is a forward plus a backward pass.</p>
      <p>For memory, two links are grounded in the paper. Descending an energy is how Hopfield networks retrieve stored patterns, and the paper lists Hopfield networks among implicit EBMs (p.41); it also stresses that EBTs, unlike the Energy Transformer, use no associative memory (p.38). EBT thinking is the same operation applied to a prediction instead of a stored pattern. And normalized surprise is a natural write gate: keep or replay the transitions the model found surprising for their context. The per-context memory of typical ΔE in step 5 is the calibration such a gate needs.</p>
      <h3>Collapse</h3>
      <p>The video EBT predicts in a frozen VAE latent (p.12), so representation collapse never arises in the paper. Training $E_\\psi$ jointly with a JEPA encoder would still need JEPA's anti-collapse machinery (EMA target, stop-gradient, variance terms): Alg. 1's loss alone is satisfied by a constant embedding.</p>
      <p class="note">Beyond the paper: our extrapolation, not the authors'. The latent plane runs the toy 2-D EBT trained for this explainer (final checkpoint), with exact gradients in your browser; the weak predictor and the changing stream are ours. A glossary and a map of every figure and table follow below.</p>`,
    source: [{ kind: 'ext', note: 'discussion' }, { kind: 'toy', note: 'latent stand-in' }, { kind: 'paper', note: 'A.3–A.4' }],
    figure(stage, ctx) {
      const { lib } = ctx, h = lib.h, C = lib.C, T2 = window.EBT && EBT.toy2d;
      const INK = C.ink || '#111111', BLUE = C.blue || '#2f3cff', MUTED = C.muted || '#6b6b70', FAINT = C.faint || '#a3a3a8', RULE = C.rule || '#e4e4e7', GRAY = '#8d8d93';
      const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
      const MONO = (lib.F && lib.F.mono) || 'monospace';
      buildReference(ctx, lib);
      if (!T2 || !T2.ready) { stage.appendChild(h('p', { class: 'callout warn' }, 'Toy model data missing (data/toy2d.json).')); return {}; }

      // ---------- fast exact evaluator of the toy EBT, with ∂E/∂ŷ and ∂E/∂x (x = context phase) ----------
      const toy = makeToy(T2.data);
      try { [[0.2, [0.3, -0.4]], [0.71, [-1.2, 0.5]]].forEach(([x, y]) => { const r = T2.energyGrad(T2.nCkpt - 1, x, y), E = toy.fwdGrad(toy.ctxBias(x), y[0], y[1]); if (Math.abs(r.E - E) > 1e-6 || Math.abs(r.g[0] - toy.G.a) > 1e-6) console.warn('world-models: evaluator mismatch', r, E); }); } catch (e) { console.warn(e); }

      // ---------- state ----------
      const S = { mode: 'jepa', x: 0.125, alpha: 1.0, seed: 3, eps: [0.6, -0.9], y0: [-1.4, -0.9], think: null, plan: null, goal: [-0.95, -0.75], stream: null, score: 'raw', kind: 'rotate', t: 0 };
      let visible = false, playing = false, animT = 0;

      // ---------- layout ----------
      const D = lib.frame(stage, { label: 'A JEPA world model, plus an energy', sub: 'black: JEPA as trained · blue: the part this step adds' });
      const diag = buildDiagram(D.frame);
      const row = h('div', { class: 'fig-row' }); stage.appendChild(row);
      const Fl = lib.frame(row, { label: 'Latent plane · toy EBT', sub: 'shading: E_JEPA, a bowl at the guess' }); Fl.wrap.style.flex = '1 1 260px';
      const Fr = lib.frame(row, { label: 'Energy on a slice', sub: 'the line from guess to outcome' }); Fr.wrap.style.flex = '1 1 260px';
      const frTitle = Fr.wrap.querySelector('.fig-label'), frSub = Fr.wrap.querySelector('.fig-sub');
      const cpl = autoCanvas(Fl.frame, { aspect: 1, minH: 240, maxH: 272, label: 'Latent plane: energy over candidate next latents for one context, with the predictor guess, thinking paths and observed outcomes', draw: () => draw() });
      const crt = autoCanvas(Fr.frame, { aspect: 1, minH: 220, maxH: 272, label: 'Side plot for the current step', draw: () => draw() });
      const ro = h('div', { class: 'readout' }); stage.appendChild(ro);
      // controls (one group per mode)
      const cThink = h('div', { class: 'controls' }), cPlan = h('div', { class: 'controls' }), cSurp = h('div', { class: 'controls' });
      stage.append(cThink, cPlan, cSurp);
      const slX = lib.slider({ id: 'wm-x', label: 'context phase x', min: 0, max: 0.995, step: 0.005, value: S.x, fmt: (v) => v.toFixed(3), oninput: (v) => { S.x = v; S.think = null; if (S.mode === 'think') runThink(); draw(); } });
      const slA = lib.slider({ id: 'wm-alpha', label: 'step size α', min: 0.1, max: 1.5, step: 0.05, value: S.alpha, fmt: (v) => v.toFixed(2), oninput: (v) => { S.alpha = v; if (S.mode === 'think') runThink(); draw(); } });
      const bOut = lib.button('new outcome', () => { const r = lib.rng(++S.seed * 7919); S.eps = [r.normal(), r.normal()]; S.y0 = [r.normal() * 1.1, r.normal() * 1.1]; if (S.mode === 'think') runThink(); draw(); });
      const bThink = lib.button('think', () => { runThink(); }, { primary: true });
      cThink.append(bThink, bOut, slX.el, slA.el);
      const bPlan = lib.button('plan', () => runPlan(), { primary: true });
      const bGoal = lib.button('new goal', () => { const r = lib.rng(++S.seed * 31); const xg = r(); const m = mu(xg); S.goal = [clamp(m[0] + 0.35 * r.normal(), -1.9, 1.9), clamp(m[1] + 0.35 * r.normal(), -1.9, 1.9)]; runPlan(); });
      cPlan.append(bPlan, bGoal, h('span', { class: 'wm-hint' }, 'or drag the goal on the plane'));
      const segScore = lib.segmented({ label: 'Surprise score', options: [['raw', 'raw ΔE'], ['norm', 'ΔE ÷ memory']], value: S.score, onchange: (v) => { S.score = v; recompute(); draw(); } });
      const segKind = lib.segmented({ label: 'Change at t = 80', options: [['none', 'no change'], ['rotate', 'rotate 15°'], ['dynamics', 'shift']], value: S.kind, onchange: (v) => { S.kind = v; recompute(); draw(); } });
      const bPlay = lib.button('play', () => { if (!S.stream) return; if (S.t >= TT) S.t = 0; playing = !playing; kick(); draw(); }, { primary: true });
      const bEnd = lib.button('end', () => { playing = false; S.t = TT; draw(); });
      cSurp.append(bPlay, bEnd, segScore.el, segKind.el);

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

      // ---------- surprise stream ----------
      const TT = 160, TC = 80, NB = 16, PER = 24, NTH = 6;
      function observe(kind, x, e0, e1) {
        let m = mu(x), s = sig(x);
        if (kind === 'rotate') { const an = 15 * Math.PI / 180, c = Math.cos(an), sn = Math.sin(an); m = [m[0] * c - m[1] * sn, m[0] * sn + m[1] * c]; }
        else if (kind === 'dynamics') { const x2 = (x + 0.05) % 1; m = mu(x2); s = sig(x2); }
        return [m[0] + s * e0, m[1] + s * e1];
      }
      function thinkFrom(zb, a, c, keep) { const P = keep ? [[a, c]] : null; for (let k = 0; k < NTH; k++) { toy.fwdGrad(zb, a, c); a -= toy.G.a; c -= toy.G.c; if (P) P.push([a, c]); } return { E: toy.fwd(zb, a, c), y: [a, c], P }; }
      function draws(n, seed, strat, keep) { const r = lib.rng(seed), out = []; for (let i = 0; i < n; i++) { const x = strat ? (Math.floor(i / PER) + r()) / NB : r(); const d = { x, g: grp(x), e0: r.normal(), e1: r.normal(), a0: r.normal(), c0: r.normal() }; d.zb = toy.ctxBias(x); d.th = thinkFrom(d.zb, d.a0, d.c0, keep); out.push(d); } return out; }
      let cal = null, test = null, R = null;
      function initStream() {
        if (S.stream) return;
        cal = draws(NB * PER, 101, true, false); test = draws(NB * PER, 202, true, false);
        cal.forEach(d => { const y = observe('none', d.x, d.e0, d.e1); d.dE = toy.fwd(d.zb, y[0], y[1]) - d.th.E; });
        const by = Array.from({ length: NB }, () => []); cal.forEach(d => by[Math.min(NB - 1, Math.floor(d.x * NB))].push(d.dE));
        S.mem = by.map(a => { a.sort((p, q) => p - q); return Math.max(1e-6, a[a.length >> 1]); });
        S.stream = draws(TT, 11, false, true);
        recompute();
      }
      const binOf = (x) => Math.min(NB - 1, Math.floor(x * NB));
      function recompute() {
        if (!S.stream) return;
        const sc = (dE, x) => S.score === 'raw' ? dE : dE / S.mem[binOf(x)];
        const cs = cal.map(d => sc(d.dE, d.x)).sort((a, b) => a - b), thr = cs[Math.ceil(0.95 * cs.length) - 1];
        const fa = [0, 0, 0, 0], hit = [0, 0, 0, 0], n = [0, 0, 0, 0];
        test.forEach(d => { const y0 = observe('none', d.x, d.e0, d.e1), y1 = observe(S.kind, d.x, d.e0, d.e1); const s0 = sc(toy.fwd(d.zb, y0[0], y0[1]) - d.th.E, d.x), s1 = sc(toy.fwd(d.zb, y1[0], y1[1]) - d.th.E, d.x); [d.g, 3].forEach(g => { n[g]++; if (s0 > thr) fa[g]++; if (s1 > thr) hit[g]++; }); });
        const recs = S.stream.map((d, t) => { const ch = t >= TC && S.kind !== 'none', y = observe(ch ? S.kind : 'none', d.x, d.e0, d.e1), dE = toy.fwd(d.zb, y[0], y[1]) - d.th.E, s = sc(dE, d.x); return { t, x: d.x, g: d.g, y, dE, s, alarm: s > thr, ch, P: d.th.P }; });
        R = { thr, recs, fa: fa.map((v, i) => v / n[i]), hit: hit.map((v, i) => v / n[i]) };
      }

      // ---------- drawing ----------
      const pct = (v) => Math.round(100 * v) + '%', nfmt = (v) => Math.round(v).toLocaleString('en-US');
      function draw() { drawPlane(); drawSide(); readouts(); }
      function plane() { const { w, h: hh } = cpl, sz = Math.min(w, hh), bx = (w - sz) / 2, by = (hh - sz) / 2; return { sz, bx, by, P: (p) => [bx + (p[0] - EXT[0]) / (EXT[1] - EXT[0]) * sz, by + sz - (p[1] - EXT[2]) / (EXT[3] - EXT[2]) * sz], inv: (px, py) => [EXT[0] + (px - bx) / sz * (EXT[1] - EXT[0]), EXT[2] + (by + sz - py) / sz * (EXT[3] - EXT[2])] }; }
      function drawPlane() {
        const g = cpl.ctx, { w, h: hh } = cpl, pl = plane(), P = pl.P;
        g.save(); g.setTransform(cpl.dpr, 0, 0, cpl.dpr, 0, 0); g.fillStyle = '#fff'; g.fillRect(0, 0, w, hh);
        const md = S.mode;
        let xNow = S.x;
        if (md === 'plan' && S.plan) { const k = Math.min(S.plan.tr.length - 1, Math.floor(animT * 14)); xNow = S.plan.tr[k].a; }
        if (md === 'surprise' && R && S.t > 0) xNow = R.recs[S.t - 1].x;
        // heatmap
        if (md === 'jepa') { const gr = jepaGrid(pred(S.x)); lib.heatmap(g, gr.E, pl.bx, pl.by, pl.sz, pl.sz, { range: [0, gr.q], gamma: 0.8 }); lib.contours(g, gr.E, pl.bx, pl.by, pl.sz, pl.sz, [0.1, 0.3, 0.6, 1, 1.5, 2.1].map(v => v * gr.q / 2.1), { color: 'rgba(17,17,17,0.18)', width: 1 }); }
        else { const gr = ebtGrid(xNow); lib.heatmap(g, gr.E, pl.bx, pl.by, pl.sz, pl.sz, { key: gr.key + '-' + Math.round(pl.sz), range: [gr.lo, gr.q], gamma: 0.7 }); const lv = []; for (let i = 1; i <= 9; i++) lv.push(gr.lo + (gr.q - gr.lo) * Math.pow(i / 9, 1.4)); lib.contours(g, gr.E, pl.bx, pl.by, pl.sz, pl.sz, lv, { color: 'rgba(17,17,17,0.18)', width: 1 }); }
        // axes ticks
        g.strokeStyle = 'rgba(17,17,17,0.12)'; g.lineWidth = 1; g.beginPath(); const o = P([0, 0]); g.moveTo(pl.bx, o[1] + 0.5); g.lineTo(pl.bx + pl.sz, o[1] + 0.5); g.moveTo(o[0] + 0.5, pl.by); g.lineTo(o[0] + 0.5, pl.by + pl.sz); g.stroke();
        // loop of mean next latents (and the changed loop after t = 80)
        const loop = (fn) => { const pts = []; for (let k = 0; k <= 200; k++) pts.push(P(fn(k / 200))); return pts; };
        lib.line(g, loop(mu), { color: INK, width: 1.1, dash: [4, 4], alpha: 0.55 });
        if (md === 'surprise' && S.kind !== 'none' && S.t > TC) lib.line(g, loop((x) => observe(S.kind, x, 0, 0)), { color: GRAY, width: 1.3, dash: [2, 3] });
        if (md === 'jepa' || md === 'ebt' || md === 'think') {
          const m = mu(S.x), s2 = 2 * sig(S.x), pm = P(m), r = s2 / (EXT[1] - EXT[0]) * pl.sz;
          g.setLineDash([3, 3]); g.strokeStyle = INK; g.lineWidth = 1; g.beginPath(); g.arc(pm[0], pm[1], Math.max(3, r), 0, 7); g.stroke(); g.setLineDash([]);
          const ob = P(obsOf(S.x)); cross(g, ob, INK, 7); tag(g, 'outcome z', ob[0] + 9, ob[1] - 10, INK);
          const pg = P(pred(S.x)); square(g, pg, INK); tag(g, 'guess', pg[0] + 9, pg[1] + 12, INK);
          if (md === 'think' && S.think) drawThink(g, P);
        }
        if (md === 'plan') drawPlan(g, P, pl);
        if (md === 'surprise') drawSurpPlane(g, P);
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
      function drawSurpPlane(g, P) {
        if (!R) { lib.text(g, 'computing…', 12, 12, { size: 12, kind: 'mono', color: MUTED }); return; }
        for (let t = Math.max(0, S.t - 40); t < S.t - 1; t++) { const r = R.recs[t], p = P(r.y), al = 0.3 + 0.6 * (t - (S.t - 40)) / 40; mark(g, p, r, al, 3); }
        if (S.t > 0) {
          const r = R.recs[S.t - 1], pts = r.P.map(P);
          halo(pts, BLUE, 1.6); lib.dot(g, pts[pts.length - 1][0], pts[pts.length - 1][1], 4.5, BLUE, { stroke: '#fff', lw: 1.4 });
          const py = P(r.y); cross(g, py, INK, 7); if (r.alarm) { g.strokeStyle = BLUE; g.lineWidth = 2; g.beginPath(); g.arc(py[0], py[1], 11, 0, 7); g.stroke(); }
          g.setLineDash([3, 3]); g.strokeStyle = 'rgba(17,17,17,0.6)'; g.lineWidth = 1; g.beginPath(); g.moveTo(pts[pts.length - 1][0], pts[pts.length - 1][1]); g.lineTo(py[0], py[1]); g.stroke(); g.setLineDash([]);
        }
      }
      function mark(g, p, r, al, rad) { // precise: filled ink; mid: filled gray; noisy: hollow gray. alarm: blue ring
        g.save(); g.globalAlpha = al;
        if (r.g === 2) { g.strokeStyle = GRAY; g.lineWidth = 1.3; g.beginPath(); g.arc(p[0], p[1], rad, 0, 7); g.stroke(); }
        else { g.fillStyle = r.g === 0 ? INK : GRAY; g.beginPath(); g.arc(p[0], p[1], rad, 0, 7); g.fill(); }
        if (r.alarm) { g.globalAlpha = Math.min(1, al + 0.3); g.strokeStyle = BLUE; g.lineWidth = 1.8; g.beginPath(); g.arc(p[0], p[1], rad + 3, 0, 7); g.stroke(); }
        g.restore();
      }
      function halo(pts, col, wdt, dash) { if (pts.length < 2) return; lib.line(g0(), pts, { color: 'rgba(255,255,255,0.9)', width: wdt + 3 }); lib.line(g0(), pts, { color: col, width: wdt, dash }); }
      function g0() { return cpl.ctx; }
      function cross(g, p, col, r) { g.strokeStyle = col; g.lineWidth = 1.5; g.beginPath(); g.moveTo(p[0] - r, p[1]); g.lineTo(p[0] + r, p[1]); g.moveTo(p[0], p[1] - r); g.lineTo(p[0], p[1] + r); g.stroke(); }
      function square(g, p, col) { g.fillStyle = '#fff'; g.fillRect(p[0] - 5, p[1] - 5, 10, 10); g.strokeStyle = col; g.lineWidth = 1.6; g.strokeRect(p[0] - 5, p[1] - 5, 10, 10); }
      function tag(g, t, x, y, col) {
        g.save(); g.font = `400 11px ${MONO}`; const tw = g.measureText(t).width; x = clamp(x, 2, cpl.w - tw - 6); y = clamp(y, 12, cpl.h - 4);
        g.fillStyle = 'rgba(255,255,255,0.82)'; g.fillRect(x - 3, y - 11, tw + 6, 15); g.fillStyle = col; g.textBaseline = 'alphabetic'; g.fillText(t, x, y); g.restore();
      }
      // ---------- side plot ----------
      function axesBox() { const { w, h: hh } = crt; return { x: 42, y: 14, w: w - 54, h: hh - 52 }; }
      function frameAxes(g, b, xr, yr, xt, yt, xlab, yfmt) {
        const X = (v) => b.x + (v - xr[0]) / (xr[1] - xr[0]) * b.w, Y = (v) => b.y + b.h - (v - yr[0]) / (yr[1] - yr[0]) * b.h;
        g.font = `400 10.5px ${MONO}`; g.fillStyle = MUTED; g.strokeStyle = RULE; g.lineWidth = 1;
        yt.forEach(v => { const y = Math.round(Y(v)) + 0.5; g.beginPath(); g.moveTo(b.x, y); g.lineTo(b.x + b.w, y); g.stroke(); g.textAlign = 'right'; g.textBaseline = 'middle'; g.fillText(yfmt ? yfmt(v) : String(v), b.x - 5, y); });
        xt.forEach(v => { g.textAlign = 'center'; g.textBaseline = 'top'; g.fillText(String(v), X(v), b.y + b.h + 4); });
        g.strokeStyle = FAINT; g.beginPath(); g.moveTo(b.x + 0.5, b.y); g.lineTo(b.x + 0.5, b.y + b.h + 0.5); g.lineTo(b.x + b.w, b.y + b.h + 0.5); g.stroke();
        g.textAlign = 'center'; g.textBaseline = 'top'; g.fillText(xlab, b.x + b.w / 2, b.y + b.h + 19);
        return { X, Y };
      }
      const poly = (g, pts, col, wdt, dash, alpha) => { if (pts.length < 2) return; g.save(); g.globalAlpha = alpha == null ? 1 : alpha; g.strokeStyle = col; g.lineWidth = wdt; g.setLineDash(dash || []); g.beginPath(); pts.forEach((p, i) => i ? g.lineTo(p[0], p[1]) : g.moveTo(p[0], p[1])); g.stroke(); g.restore(); };
      const legend = (g, b, items) => { g.font = `400 10.5px ${MONO}`; g.textAlign = 'right'; g.textBaseline = 'top'; items.forEach(([t, col], i) => { const tw = g.measureText(t).width; g.fillStyle = 'rgba(255,255,255,0.85)'; g.fillRect(b.x + b.w - tw - 8, b.y + 2 + 14 * i, tw + 6, 14); g.fillStyle = col; g.fillText(t, b.x + b.w - 4, b.y + 3 + 14 * i); }); };
      function drawSide() {
        const g = crt.ctx, { w, h: hh } = crt, b = axesBox(), md = S.mode;
        g.save(); g.setTransform(crt.dpr, 0, 0, crt.dpr, 0, 0); g.fillStyle = '#fff'; g.fillRect(0, 0, w, hh);
        if (md === 'jepa' || md === 'ebt') {
          frTitle.textContent = 'Energy on a slice'; frSub.textContent = 'the line from guess to outcome';
          const pg = pred(S.x), ob = obsOf(S.x), zb = toy.ctxBias(S.x), d2 = (ob[0] - pg[0]) ** 2 + (ob[1] - pg[1]) ** 2;
          const ss = [], ej = [], ee = []; for (let i = 0; i <= 80; i++) { const s = -0.6 + 2.2 * i / 80, p = [pg[0] + s * (ob[0] - pg[0]), pg[1] + s * (ob[1] - pg[1])]; ss.push(s); ej.push(s * s * d2); ee.push(toy.fwd(zb, p[0], p[1])); }
          const all = md === 'ebt' ? ej.concat(ee) : ej; let lo = Math.min(...all), hi = Math.max(...all); if (hi - lo < 0.5) hi = lo + 0.5; const pad = (hi - lo) * 0.08; lo -= pad; hi += pad;
          const nice = (v) => Math.round(v * 2) / 2, yt = []; for (let v = Math.ceil(lo * 2) / 2; v <= hi; v += Math.max(0.5, nice((hi - lo) / 4))) yt.push(+v.toFixed(2));
          const ax = frameAxes(g, b, [-0.6, 1.6], [lo, hi], [], yt, 'position on the slice', (v) => v.toFixed(1));
          [0, 1].forEach(v => { g.strokeStyle = 'rgba(17,17,17,0.25)'; g.setLineDash([2, 3]); g.beginPath(); g.moveTo(ax.X(v), b.y); g.lineTo(ax.X(v), b.y + b.h); g.stroke(); g.setLineDash([]); });
          g.font = `400 10.5px ${MONO}`; g.fillStyle = MUTED; g.textAlign = 'center'; g.textBaseline = 'top'; g.fillText('guess', ax.X(0), b.y + b.h + 4); g.fillText('outcome', ax.X(1), b.y + b.h + 4);
          poly(g, ss.map((s, i) => [ax.X(s), ax.Y(ej[i])]), INK, 1.8, [6, 4]);
          if (md === 'ebt') { poly(g, ss.map((s, i) => [ax.X(s), ax.Y(ee[i])]), BLUE, 2.2); { const zb0 = toy.ctxBias(S.x); lib.dot(g, ax.X(0), ax.Y(toy.fwd(zb0, pg[0], pg[1])), 3.5, BLUE); } }
          lib.dot(g, ax.X(0), ax.Y(0), 3.5, INK);
          legend(g, b, md === 'ebt' ? [['E_JEPA = ‖guess − z′‖²', INK], ['E_ψ (toy EBT)', BLUE]] : [['E_JEPA = ‖guess − z′‖²', INK]]);
        } else if (md === 'think') {
          frTitle.textContent = 'Energy per thinking step'; frSub.textContent = 'from the guess and from noise';
          if (!S.think) runThink();
          const A = S.think.a.Es, B2 = S.think.b.Es, n = S.think.n, k = Math.min(n, Math.floor(animT * 5));
          const all = A.concat(B2); let lo = Math.min(...all), hi = Math.max(...all); const pad = (hi - lo) * 0.08 + 0.05; lo -= pad; hi += pad;
          const st = (hi - lo) > 3 ? 1 : 0.5, yt = []; for (let v = Math.ceil(lo / st) * st; v <= hi; v += st) yt.push(+v.toFixed(2));
          const ax = frameAxes(g, b, [0, n], [lo, hi], [0, 2, 4, 6, 8, 10], yt, 'thinking step i', (v) => v.toFixed(1));
          poly(g, B2.slice(0, k + 1).map((e, i) => [ax.X(i), ax.Y(e)]), GRAY, 1.8, [4, 3]); B2.slice(0, k + 1).forEach((e, i) => lib.dot(g, ax.X(i), ax.Y(e), 2.4, GRAY));
          poly(g, A.slice(0, k + 1).map((e, i) => [ax.X(i), ax.Y(e)]), BLUE, 2.2); A.slice(0, k + 1).forEach((e, i) => lib.dot(g, ax.X(i), ax.Y(e), 2.8, BLUE));
          legend(g, b, [['from the guess', BLUE], ['from noise ŷ₀', GRAY]]);
        } else if (md === 'plan') {
          frTitle.textContent = 'Planning iterations'; frSub.textContent = 'objective per start · distance to goal';
          if (!S.plan) runPlan();
          const tr = S.plan.tr, k = Math.min(PIT, Math.floor(animT * 14));
          const Js = tr.map(s => s.J), ds = tr.map(s => s.d), allJ = [].concat(...S.plan.runs.map(r => r.map(s => s.J)));
          let lo = Math.min(0, ...allJ), hi = Math.max(...Js, ...ds); hi += (hi - lo) * 0.08;
          const st = hi - lo > 4 ? 1 : 0.5, yt = []; for (let v = Math.ceil(lo / st) * st; v <= hi; v += st) yt.push(+v.toFixed(2));
          const ax = frameAxes(g, b, [0, PIT], [lo, hi], [0, 20, 40, 60], yt, 'iteration (one ∇ step on a and ẑ)', (v) => v.toFixed(1));
          g.save(); g.beginPath(); g.rect(b.x, b.y, b.w, b.h); g.clip();
          S.plan.runs.forEach((run, j) => { if (j !== S.plan.best) poly(g, run.slice(0, k + 1).map((s, i) => [ax.X(i), ax.Y(s.J)]), BLUE, 1, null, 0.3); });
          g.restore();
          poly(g, ds.slice(0, k + 1).map((v, i) => [ax.X(i), ax.Y(v)]), INK, 1.8, [6, 4]);
          poly(g, Js.slice(0, k + 1).map((v, i) => [ax.X(i), ax.Y(v)]), BLUE, 2.2);
          lib.dot(g, ax.X(k), ax.Y(Js[k]), 3.5, BLUE); lib.dot(g, ax.X(k), ax.Y(ds[k]), 3, INK);
          legend(g, b, [['E + λ‖ẑ − goal‖²', BLUE], ['other starts', lib.rgba(BLUE, 0.55)], ['‖ẑ − goal‖', INK]]);
        } else {
          frTitle.textContent = 'Surprise over time'; frSub.textContent = 'score ÷ alarm line, log scale';
          if (!R) { lib.text(g, 'computing…', b.x + 8, b.y + 8, { size: 12, kind: 'mono', color: MUTED }); g.restore(); return; }
          const lo = -2, hi = 1.6, X = (t) => b.x + (t + 0.5) / TT * b.w, Y = (r) => b.y + b.h - (clamp(Math.log10(Math.max(r, 1e-9)), lo, hi) - lo) / (hi - lo) * b.h;
          if (S.kind !== 'none') { g.fillStyle = lib.rgba(BLUE, 0.06); g.fillRect(X(TC - 0.5), b.y, b.x + b.w - X(TC - 0.5), b.h); }
          g.font = `400 10.5px ${MONO}`; g.fillStyle = MUTED; g.strokeStyle = RULE; g.lineWidth = 1;
          [0.01, 0.1, 1, 10].forEach(v => { const y = Math.round(Y(v)) + 0.5; g.beginPath(); g.moveTo(b.x, y); g.lineTo(b.x + b.w, y); g.stroke(); g.textAlign = 'right'; g.textBaseline = 'middle'; g.fillText(v >= 1 ? v + '×' : v + '×', b.x - 5, y); });
          [0, 40, 80, 120, 160].forEach(v => { g.textAlign = 'center'; g.textBaseline = 'top'; g.fillText(String(v), X(v - 0.5), b.y + b.h + 4); });
          g.fillText('observation t', b.x + b.w / 2, b.y + b.h + 19);
          g.strokeStyle = FAINT; g.beginPath(); g.moveTo(b.x + 0.5, b.y); g.lineTo(b.x + 0.5, b.y + b.h + 0.5); g.lineTo(b.x + b.w, b.y + b.h + 0.5); g.stroke();
          g.strokeStyle = BLUE; g.setLineDash([5, 4]); g.lineWidth = 1.2; g.beginPath(); g.moveTo(b.x, Y(1)); g.lineTo(b.x + b.w, Y(1)); g.stroke(); g.setLineDash([]);
          { const tw = g.measureText('alarm line').width; g.fillStyle = 'rgba(255,255,255,0.85)'; g.fillRect(b.x + 3, Y(1) - 15, tw + 4, 13); g.fillStyle = BLUE; g.textAlign = 'left'; g.textBaseline = 'bottom'; g.fillText('alarm line', b.x + 5, Y(1) - 2); }
          if (S.kind !== 'none') { g.fillStyle = MUTED; g.textAlign = 'left'; g.textBaseline = 'top'; g.fillText('world changes', X(TC - 0.5) + 4, b.y + 2); }
          for (let t = 0; t < S.t; t++) { const r = R.recs[t], ratio = R.thr > 0 ? r.s / R.thr : 0; mark(g, [X(t), Y(ratio)], r, 0.95, 2.4); }
        }
        g.restore();
      }
      function readouts() {
        const md = S.mode, f3 = (v) => (v < 0 ? '−' : '') + Math.abs(v).toFixed(3);
        if (md === 'jepa' || md === 'ebt' || md === 'think') {
          const zb = toy.ctxBias(S.x), pg = pred(S.x), m = mu(S.x), eg = toy.fwd(zb, pg[0], pg[1]), th = descend(S.x, pg, 1, 12), emin = th.Es[th.Es.length - 1];
          const err = Math.hypot(pg[0] - m[0], pg[1] - m[1]);
          if (md === 'jepa') ro.innerHTML = `<span>x <b>${S.x.toFixed(3)}</b></span><span>σ(x) <b>${sig(S.x).toFixed(2)}</b></span><span>E<sub>JEPA</sub>(guess) <b>0</b> always</span><span>guess error <b>${err.toFixed(2)}</b>, invisible to it</span>`;
          else if (md === 'ebt') ro.innerHTML = `<span>x <b>${S.x.toFixed(3)}</b></span><span>E<sub>ψ</sub>(guess) <b>${f3(eg)}</b></span><span>basin floor <b>${f3(emin)}</b></span><span>gap <b>${(eg - emin).toFixed(3)}</b></span><span>guess error <b>${err.toFixed(2)}</b></span>`;
          else { const T = S.think; if (!T) { ro.innerHTML = ''; return; } const k = Math.min(T.n, Math.floor(animT * 5)), pa = T.a.path[k], pb = T.b.path[k]; ro.innerHTML = `<span>step <b>${k}</b></span><span>from guess: E <b>${f3(T.a.Es[k])}</b>, ‖ẑ − μ‖ <b>${Math.hypot(pa[0] - m[0], pa[1] - m[1]).toFixed(3)}</b></span><span>from noise: E <b>${f3(T.b.Es[k])}</b>, ‖ẑ − μ‖ <b>${Math.hypot(pb[0] - m[0], pb[1] - m[1]).toFixed(3)}</b></span>`; }
        } else if (md === 'plan') {
          if (!S.plan) { ro.innerHTML = ''; return; } const tr = S.plan.tr, k = Math.min(PIT, Math.floor(animT * 14)), s = tr[k], done = k >= PIT;
          ro.innerHTML = `<span>iteration <b>${k}</b></span><span>action a <b>${s.a.toFixed(3)}</b> (start ${tr[0].a.toFixed(3)}, best of ${MP})</span><span>E <b>${f3(s.E)}</b></span><span>‖ẑ − goal‖ <b>${s.d.toFixed(2)}</b></span>` +
            (done ? `<span>verify: ‖ẑ*(a) − goal‖ <b>${S.plan.re.toFixed(2)}</b></span>` : '') + `<span>≈ <b>${nfmt(3 * MP * k + (done ? 36 : 0))}</b> forward-pass equivalents</span>`;
        } else {
          if (!R) { ro.innerHTML = 'computing the stream…'; return; }
          const r = S.t > 0 ? R.recs[S.t - 1] : null;
          ro.innerHTML = (r ? `<span>t <b>${S.t}</b>${r.ch ? ' (changed)' : ''}</span><span>ΔE <b>${f3(r.dE)}</b></span><span>score <b>${(r.s / R.thr).toFixed(2)}×</b> line ${r.alarm ? '<b>alarm</b>' : ''}</span>` : '<span>t <b>0</b>: press play</span>') +
            `<span>false alarms: precise <b>${pct(R.fa[0])}</b> · noisy <b>${pct(R.fa[2])}</b></span>` + (S.kind === 'none' ? '' : `<span>caught after change: precise <b>${pct(R.hit[0])}</b> · noisy <b>${pct(R.hit[2])}</b> · all <b>${pct(R.hit[3])}</b></span>`);
        }
        bPlay.textContent = playing ? 'pause' : S.t >= TT ? 'replay' : 'play';
      }

      // ---------- animation ----------
      const loop = lib.loop((dt) => {
        if (!visible) return false;
        let more = false;
        if ((S.mode === 'think' && S.think && animT * 5 < S.think.n + 1) || (S.mode === 'plan' && S.plan && animT * 14 < PIT + 1)) { animT += dt; more = true; }
        if (S.mode === 'surprise' && playing) { S.tAcc = (S.tAcc || 0) + dt * 22; while (S.tAcc >= 1 && S.t < TT) { S.t++; S.tAcc -= 1; } if (S.t >= TT) playing = false; more = more || playing; }
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
        S.mode = m; playing = false; diag.setLayer(layer);
        cThink.hidden = !(m === 'think' || m === 'jepa' || m === 'ebt'); bThink.hidden = slA.el.hidden = m !== 'think';
        cPlan.hidden = m !== 'plan'; cSurp.hidden = m !== 'surprise';
        Fl.wrap.querySelector('.fig-sub').textContent = m === 'jepa' ? 'shading: E_JEPA, a bowl at the guess' : m === 'surprise' ? 'recent outcomes · ring = alarm' : m === 'plan' ? 'shading: E_ψ at the current action a' : 'shading: learned E_ψ over ẑ';
        ctx.setCaption(CAP[m]);
      }
      const CAP = {
        jepa: 'Darker = lower energy. Dashed loop: mean next latent of every context; dashed circle: ±2σ of outcomes here; crosshair: one outcome; square: the weak predictor\'s guess.',
        ebt: 'Shaded by the toy EBT\'s energy for this context. Floor: energy after 12 thinking steps from the guess; gap: E(guess) minus the floor.',
        think: 'Blue: thinking from the predictor\'s guess; gray, dashed: from noise. Dots are steps; ẑ* is where the blue run ends.',
        plan: 'Bold: best of 6 joint descents on (a, ẑ); faint: other starts. Hollow circle: the model\'s own prediction at the chosen a (verification). Dial ticks: start actions.',
        surprise: '● ink: precise context (σ &lt; 0.1) · ● gray: medium · ○: noisy (σ &gt; 0.2) · blue ring: alarm. Dotted gray loop: the dynamics after the change.',
      };
      const STEPS = [
        () => { setMode('jepa', 0); draw(); },
        () => { setMode('ebt', 1); draw(); },
        () => { setMode('think', 1); runThink(); },
        () => { setMode('plan', 2); runPlan(); },
        () => { setMode('surprise', 3); initStream(); S.t = 0; playing = !lib.reducedMotion; if (lib.reducedMotion) S.t = TT; kick(); draw(); },
      ];
      setMode('jepa', 0);
      (document.fonts && document.fonts.ready ? document.fonts.ready : Promise.resolve()).then(() => draw());
      draw();
      return {
        step(i) { (STEPS[i] || STEPS[0])(); },
        show() { visible = true; [cpl, crt].forEach(o => o.fit()); draw(); kick(); },
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
  // diagram: one SVG per layout (wide / narrow), layers 0 JEPA, 1 energy head, 2 planner, 3 surprise
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
      ot1: [0, 'img', 'o_{t+1}', '', [8, 184, 34, 34], [6, 446, 34, 34]],
      tenc: [0, 'box', 'TARGET ENC.', 'EMA · stop-grad', [62, 180, 108, 42], [66, 442, 176, 42]],
      z1: [0, 'pill', 'z_{t+1}', '', [188, 190, 52, 22], [252, 452, 58, 22]],
      loss: [0, 'box', 'JEPA LOSS', '‖ẑ − sg(z)‖²', [276, 180, 156, 42], [196, 232, 138, 42]],
      surp: [3, 'box', 'SURPRISE', 'ΔE → memory, update gate', [470, 180, 164, 42], [66, 372, 176, 42]],
    };
    // edges: [layer, wide points, narrow points, label (wide), label (narrow), loop?]
    const E = [
      [0, [[42, 87], [62, 87]], [[40, 115], [66, 115]]],
      [0, [[170, 87], [188, 87]], [[154, 136], [154, 146]]],
      [0, [[230, 87], [246, 87]], [[154, 168], [154, 178]]],
      [0, [[396, 87], [416, 87]], [[154, 220], [154, 232]]],
      [0, [[321, 59], [321, 66]], [[178, 69], [262, 69], [262, 199], [242, 199]]],
      [0, [[42, 201], [62, 201]], [[40, 463], [66, 463]]],
      [0, [[170, 201], [188, 201]], [[242, 463], [252, 463]]],
      [0, [[240, 201], [276, 201]], [[296, 452], [296, 274]]],
      [0, [[444, 98], [444, 168], [354, 168], [354, 180]], [[182, 243], [196, 243]], 'compare', ''],
      [1, [[472, 87], [560, 87], [560, 118]], [[154, 254], [154, 290]]],
      [1, [[610, 118], [610, 58], [444, 58], [444, 76]], [[66, 311], [36, 311], [36, 243], [124, 243]], '−α∇_{ẑ}E (think)', '−α∇E', true],
      [2, [[634, 139], [640, 139], [640, 18], [396, 18]], [[242, 311], [324, 311], [324, 28], [242, 28]], '∇_{a}E · scores', '', false, [520, 13, 'middle']],
      [2, [[220, 18], [246, 18]], [[52, 28], [66, 28]]],
      [2, [[321, 36], [321, 43]], [[154, 48], [154, 58]]],
      [3, [[560, 160], [560, 180]], [[154, 332], [154, 372]], 'E(ẑ*)', 'E(ẑ*)', false, [566, 173, 'start'], [160, 356, 'start']],
      [3, [[214, 212], [214, 236], [552, 236], [552, 222]], [[281, 452], [281, 393], [242, 393]], 'observed z_{t+1}', '', false, [383, 232, 'middle']],
    ];
    const mk = (lay) => {
      const vb = lay === 'wide' ? [0, -4, 646, 248] : [0, 0, 340, 492];
      const svg = el('svg', { viewBox: vb.join(' '), class: 'wm-diag wm-' + lay, role: 'img', 'aria-label': 'Diagram: a JEPA world model (encoder, action-conditioned predictor, EMA target encoder and loss) with an energy head that scores predicted latents, a planner that uses its gradient, and a surprise monitor' });
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

  // ===================================================================================================
  // reference block below the panel: searchable glossary + paper map (every figure, table, algorithm, listing)
  // ===================================================================================================
  function buildReference(ctx, lib) {
    const h = lib.h;
    const panelName = (id) => { const d = window.EBT && EBT.panels && EBT.panels[id]; return d ? (d.nav || id) : id; };
    const go = (id) => h('a', { href: '#' + id, class: 'wm-go' }, panelName(id) + ' →');
    const sec = h('section', { class: 'wm-ref', 'aria-label': 'Glossary and paper map' }); ctx.panel.appendChild(sec);
    sec.appendChild(h('div', { class: 'wm-ref-head' }, h('div', { class: 'p-num' }, 'Reference'), h('h3', {}, 'Glossary and paper map')));
    const cols = h('div', { class: 'wm-ref-cols' }); sec.appendChild(cols);

    // ---------- glossary ----------
    const G = h('div', { class: 'wm-gloss' }); cols.appendChild(G);
    G.appendChild(h('div', { class: 'fig-label' }, 'Glossary'));
    const data = lib.data('glossary'), fromData = ((data && data.terms) || []).map(t => ({ term: t.term, def: t.short, long: t.long, ext: /model predictive|MPC/i.test(t.term) }));
    const OURS = [
      ['JEPA (joint-embedding predictive architecture)', 'Predicts the representation of a target from a context in a learned embedding space, trained by regression against an EMA target encoder. Not mentioned in the paper.'],
      ['CEM (cross-entropy method)', 'Sampling-based planner: draw action sequences from a Gaussian, keep the lowest-cost elites, refit the Gaussian, repeat. Needs no gradient.'],
      ['Shooting vs collocation', 'Shooting optimizes actions and rolls the model forward to get states. Collocation optimizes states and actions jointly with a consistency penalty, as in App. A.3.'],
      ['Surprise ΔE', 'In this explainer: E(x, z_obs) − E(x, ẑ*), the energy of what happened minus the energy of what the model expected after thinking.'],
      ['Memory-normalized surprise', 'ΔE divided by the typical ΔE remembered for similar contexts, so noisy contexts do not raise false alarms.'],
      ['Representation collapse', 'A jointly trained encoder maps everything to the same embedding, making prediction trivial. JEPAs prevent it with EMA targets, stop-gradients or variance terms.'],
    ].map(([term, def]) => ({ term, def, ext: true }));
    const ALL = fromData.concat(OURS).sort((a, b) => a.term.localeCompare(b.term));
    const inp = h('input', { type: 'text', class: 'wm-search', placeholder: 'search terms', 'aria-label': 'Search the glossary' });
    const cnt = h('span', { class: 'wm-count', 'aria-live': 'polite' });
    G.appendChild(h('div', { class: 'wm-bar' }, inp, cnt));
    const dl = h('dl', { class: 'wm-dl' }); G.appendChild(dl);
    const esc = (t) => String(t).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);
    const mark = (t, q) => { const e = esc(t); if (!q) return e; const i = e.toLowerCase().indexOf(esc(q).toLowerCase()); return i < 0 ? e : e.slice(0, i) + '<mark>' + e.slice(i, i + q.length) + '</mark>' + e.slice(i + q.length); };
    function renderG() {
      const q = inp.value.trim(), ql = q.toLowerCase(); dl.replaceChildren();
      const hits = ALL.filter(e => !ql || (e.term + ' ' + e.def + ' ' + (e.long || '')).toLowerCase().includes(ql));
      hits.forEach(e => {
        const dt = h('dt', { html: mark(e.term, q) }); if (e.ext) dt.appendChild(h('span', { class: 'src ext' }, 'beyond the paper'));
        const dd = h('dd', {}, h('span', { html: mark(e.def, q) }));
        if (e.long) { const det = h('details', {}, h('summary', {}, 'more'), h('p', { html: mark(e.long, q) })); if (ql && e.long.toLowerCase().includes(ql) && !(e.term + e.def).toLowerCase().includes(ql)) det.open = true; dd.appendChild(det); }
        dl.append(dt, dd);
      });
      cnt.textContent = hits.length + ' of ' + ALL.length;
      if (!hits.length) dl.appendChild(h('p', { class: 'wm-small' }, 'No match. Try a shorter word.'));
    }
    inp.addEventListener('input', renderG); renderG();

    // ---------- paper map ----------
    const Mp = h('div', { class: 'wm-map' }); cols.appendChild(Mp);
    Mp.appendChild(h('div', { class: 'fig-label' }, 'Paper map · arXiv 2507.02092v1'));
    const M = [
      ['Fig 1', 'fig', 2, 'Four ways to predict the next element: AR Transformer, RNN, Diffusion Transformer, EBT. Only the EBT outputs an energy for a candidate.', ['families'], 'fig01'],
      ['Table 1', 'tab', 3, 'Three facets of System 2 thinking by architecture: EBTs have all three; diffusion has dynamic compute only.', ['families']],
      ['Fig 2', 'fig', 4, 'An EBT predicting the next token and the next video frame: a random guess sharpens step by step as its energy falls.', ['tokens', 'descent'], 'fig02'],
      ['Fig 3', 'fig', 5, 'Thinking as descent on an energy landscape until the energy converges. A schematic adapted from a loss-landscape paper, not a measured landscape.', ['landscape'], 'fig03'],
      ['Eq. 1', 'alg', 7, 'The thinking update: ŷ moves against the gradient of E(x, ŷ) with respect to ŷ, scaled by the step size α.', ['descent']],
      ['Eq. 2', 'alg', 7, 'The same update plus Langevin noise η ~ N(0, σ).', ['langevin-bon']],
      ['Algorithm 1', 'alg', 7, 'Training: start from noise, take N gradient steps on the energy, put the loss on the final prediction, backpropagate through the optimization.', ['alg1', 'second-order']],
      ['Algorithm 2', 'alg', 7, 'Inference with verification: optimize M candidates and keep the one with the lowest energy.', ['langevin-bon']],
      ['Fig 4', 'fig', 9, 'Text scaling with data (35.98% faster), batch size (28.46%) and depth (5.29%). EBT crosses below Transformer++ only for data and batch.', ['scaling'], 'fig04'],
      ['Fig 5', 'fig', 10, 'Text scaling with parameters (2.91%), FLOPs (2.92%) and width (0.02%). EBT perplexity is higher at every measured size and FLOP budget; width is a wash.', ['scaling', 'costs'], 'fig05'],
      ['Table 2', 'tab', 10, 'Landscape-regularizer ablations on OOD Dyck. Without random step size thinking nearly vanishes (−1.47 / 0.19); the full recipe is best with self-verification (18.7).', ['regularizers']],
      ['Fig 6', 'fig', 11, '(a) OOD perplexity increase falls by up to 29% with more forward passes, relative to EBT\'s own no-thinking point; Transformer++ is flat. (b) The Best-of-5 gain grows with training tokens (Dyck only, noisy).', ['thinking-results'], 'fig06'],
      ['Fig 7', 'fig', 11, 'The gain from thinking grows roughly linearly with OOD shift (downstream ÷ pretraining perplexity), from about 12% to 23% (approx.).', ['thinking-results'], 'fig07'],
      ['Table 3', 'tab', 12, 'Worse pretraining perplexity (33.43 vs 31.36) but better on GSM8K, BB Math QA and BB Dyck; slightly worse on SQuAD.', ['text-data', 'thinking-results']],
      ['Fig 8', 'fig', 12, 'Token energies over iterations 0 to 11: most of the drop happens at iteration 1, and hard tokens stay higher.', ['tokens', 'uncertainty'], 'fig08'],
      ['Fig 9', 'fig', 13, 'Video next-frame scaling on SSV2: more than 33% faster with width (33.66%) and parameters (34.28%), but higher loss at every measured size.', ['video-data', 'scaling'], 'fig09'],
      ['Table 4', 'tab', 13, 'Denoising PSNR and MSE at σ = 0.1 and OOD σ = 0.2 (EBT 23.29 vs DiT 19.56 PSNR) and ImageNet-1k linear probe (5.32% vs 0.31% top-1).', ['image-results', 'image-data']],
      ['Fig 10', 'fig', 14, 'OOD denoising examples: EBT after 1, 2, 3 steps vs DiT after 100, 200, 300 steps.', ['image-results'], 'fig10'],
      ['Fig 11', 'fig', 14, 'Frame energies across a video: high while the scene is unpredictable, low once the garment is in view.', ['video-data', 'uncertainty'], 'fig11'],
      ['Fig 12', 'fig', 15, 'PSNR vs forward passes on OOD noise: EBT with 3 passes (≈23) beats DiT with 300 (≈19.6); about equal at 1 vs 100 and 2 vs 200 (approx.).', ['image-results'], 'fig12'],
      ['App. A.3', 'text', 26, 'The only world-model passage: hold the past fixed and descend the energy with respect to future actions and states, so the world model acts as a policy. No experiments.', ['planning', 'world-models']],
      ['Fig B.1', 'fig', 28, '(a) Best-of-10 over Best-of-2 gains grow with data (RedPajama validation). (b) Fig 6b extrapolated to 15T tokens: speculative.', ['thinking-results'], 'figB1'],
      ['Fig B.2', 'fig', 29, 'Familiar text keeps lower energy than a random token sequence: epistemic uncertainty.', ['uncertainty'], 'figB2'],
      ['Fig B.3', 'fig', 29, 'FineWeb data scaling to about 130B tokens: 35.69% faster overall, 51.70% in the zoomed tail where EBT crosses below.', ['scaling'], 'figB3'],
      ['Fig C.1', 'fig', 31, 'S2 (thinking-tuned) vs S1 EBTs on tiny models: S2 starts higher but scales 3.30% faster.', ['regularizers'], 'figC1'],
      ['Table D.1', 'tab', 33, 'Model sizes xxs to xl: 6.18M to 708M non-embedding parameters, 6 to 24 layers.', ['scaling', 'costs']],
      ['Table D.2', 'tab', 33, 'Transformer++ hyperparameters for video and text: AdamW, context 16 frames or 256 tokens.', ['text-data', 'video-data']],
      ['Table D.3', 'tab', 35, 'EBT scaling hyperparameters: 2 optimization steps, step size 500 (text) and 30,000 (video), learnable step size.', ['alg1', 'costs']],
      ['Table D.4', 'tab', 36, 'S2 thinking hyperparameters: 2 to 3 random steps, step size 5, Langevin noise 3, replay buffer, no detaching.', ['regularizers']],
      ['Fig E.1', 'fig', 37, 'Diffusion is supervised at every denoising step; an EBM only at the end of its optimization.', ['families', 'contrastive'], 'figE1'],
      ['Fig H.1', 'fig', 40, 'A feed-forward model maps x to ŷ; an EBM scores the pair (x, ŷ) with one energy.', ['energy'], 'figH1'],
      ['Listing 1', 'code', 44, 'PyTorch pseudocode for language EBT training: logits as predictions, softmax then projection, create_graph=True for second-order gradients.', ['architecture', 'alg1']],
      ['Listing 2', 'code', 45, 'The same for video: next-frame embeddings with a Smooth L1 loss.', ['video-data']],
    ];
    const KIND = { fig: 'figures', tab: 'tables', alg: 'algorithms', code: 'code', text: 'text' };
    let filt = 'all', sel = 'Fig 3';
    const seg = lib.segmented({ label: 'Filter the paper map', options: [['all', 'all'], ['fig', 'figures'], ['tab', 'tables'], ['alg', 'algorithms + code']], value: filt, onchange: (v) => { filt = v; renderM(); } });
    const inp2 = h('input', { type: 'text', class: 'wm-search', placeholder: 'search (e.g. OOD, Dyck, PSNR)', 'aria-label': 'Search the paper map' });
    const cnt2 = h('span', { class: 'wm-count' });
    inp2.addEventListener('input', () => renderM());
    Mp.appendChild(h('div', { class: 'wm-bar' }, seg.el, inp2, cnt2));
    const prev = h('figure', { class: 'wm-prev' });
    const tb = h('tbody'); const table = h('table', { class: 'wm-table' }, h('thead', {}, h('tr', {}, h('th', {}, 'Item'), h('th', { class: 'num' }, 'Page'), h('th', {}, 'What it shows'), h('th', {}, 'Explore'))), tb);
    Mp.appendChild(prev); Mp.appendChild(h('div', { class: 'tbl wm-tbl' }, table));
    function preview(r) {
      prev.replaceChildren();
      if (!r || !r[5]) { prev.appendChild(h('p', { class: 'wm-small' }, (r ? r[0] + ' (p.' + r[2] + '): ' : '') + 'no image crop; the key numbers are in the row and the linked panel.')); return; }
      prev.append(h('a', { href: 'media/paper/' + r[5] + '.png', target: '_blank', rel: 'noopener', class: 'paper-fig' }, h('img', { src: 'media/paper/' + r[5] + '.png', alt: 'Paper ' + r[0] + ': ' + r[3], loading: 'lazy' })),
        h('figcaption', { class: 'wm-small' }, h('span', { class: 'src paper' }, 'from the paper · ' + r[0] + ', p.' + r[2])));
    }
    function renderM() {
      const q = inp2.value.trim().toLowerCase(); tb.replaceChildren(); let n = 0;
      M.forEach(r => {
        if (filt === 'fig' && r[1] !== 'fig') return; if (filt === 'tab' && r[1] !== 'tab') return; if (filt === 'alg' && !(r[1] === 'alg' || r[1] === 'code')) return;
        if (q && !(r[0] + ' ' + r[3] + ' ' + KIND[r[1]]).toLowerCase().includes(q)) return; n++;
        const btn = h('button', { type: 'button', class: 'wm-item' + (r[0] === sel ? ' on' : ''), 'aria-label': 'Preview ' + r[0] }, r[0]);
        btn.addEventListener('click', () => { sel = r[0]; preview(r); renderM(); });
        tb.appendChild(h('tr', { class: r[0] === sel ? 'hl' : '' }, h('td', {}, btn), h('td', { class: 'num' }, 'p.' + r[2]), h('td', {}, r[3]), h('td', { class: 'wm-links' }, ...r[4].map(go))));
      });
      cnt2.textContent = n + ' of ' + M.length;
    }
    renderM(); preview(M.find(r => r[0] === sel));
  }
})();
