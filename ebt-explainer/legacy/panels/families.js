/* Panel: four ways to predict the next token (paper Fig 1, Table 1, Sec 6.1–6.4).
   Stage: four lanes (AR Transformer, RNN, Diffusion Transformer, EBT) answer the same next-token question under a shared
   budget of forward passes. Three candidate tokens, so every prediction is a point in the probability simplex (a triangle).
   - Diffusion: a real DDIM sampler over the 3 logits with the exact denoiser for a Gaussian target around its belief.
   - EBT: real gradient descent on E(ŷ) = CE(b, softmax(ŷ)) (gradient softmax(ŷ) − b), stopping when E ≤ τ or when one
     step lowers E by less than 0.002.
   - AR and RNN: one pass gives softmax(belief); internals are schematic.
   Every lane gets the same size of knowledge error (logit noise), so this is about mechanism, not accuracy.
   Prose slots: "numbers vs arrows" lab (step 4; energy vs a learned vector field with controllable curl) and Table 1
   (step 6; hover a cell for the paper's reason and page). Ported and restyled from legacy/sections/families.js. */
(function () {
  'use strict';
  const PKEYS = ['easy', 'medium', 'hard'];
  const PROMPTS = {
    easy: { ctx: 'Once upon a', toks: ['time', 'day', 'hill'], p: [0.90, 0.07, 0.03] },
    medium: { ctx: 'The cat sat on the', toks: ['mat', 'floor', 'sofa'], p: [0.55, 0.30, 0.15] },
    hard: { ctx: 'My favorite color is', toks: ['blue', 'red', 'green'], p: [0.38, 0.34, 0.28] },
  };
  const LANES = [
    { id: 'ar', name: 'AR Transformer', eq: 'p = softmax(f(x≤t))', facets: [0, 0, 0] },
    { id: 'rnn', name: 'RNN', eq: 'h_t = g(h_t−1, x_t)', facets: [0, 0, 0] },
    { id: 'diff', name: 'Diffusion Transformer', eq: 'outputs a noise estimate ε', facets: [1, 0, 0] },
    { id: 'ebt', name: 'EBT', eq: 'outputs energy E(x, ŷ)', facets: [1, 1, 1] },
  ];
  const TD = 8, ERR = 0.12, S_DIFF = 0.15, EPS = 0.002, ALPHA = 1.0, BMAX = 16;

  // ---------- math ----------
  const softmax = (z) => { const m = Math.max(...z); const e = z.map(v => Math.exp(v - m)); const s = e.reduce((a, b) => a + b, 0); return e.map(v => v / s); };
  const ce = (p, q) => -p.reduce((a, pk, k) => a + pk * Math.log(Math.max(1e-12, q[k])), 0);
  const kl = (p, q) => ce(p, q) - ce(p, p);
  const center = (v) => { const m = v.reduce((a, b) => a + b, 0) / v.length; return v.map(x => x - m); };
  const abar = (u) => { const c = Math.cos((u + 0.008) / 1.008 * Math.PI / 2); return Math.max(0, c * c); }; // cosine schedule, u in [0,1]
  function belief(lib, lane, pk) { const r = lib.rng(1000 + lane * 97 + PKEYS.indexOf(pk) * 13); return PROMPTS[pk].p.map(v => Math.log(v) + ERR * r.normal()); }
  function runEBT(lib, pk, o) {
    const b = softmax(belief(lib, 3, pk)), E = (y) => ce(b, softmax(y));
    const r = lib.rng(o.seed * 104729 + 5); let y = [r.normal(), r.normal(), r.normal()];
    const ys = [y], Es = [E(y)]; let reason = 'budget';
    for (let i = 1; i <= o.budget; i++) {
      const q = softmax(y); y = y.map((v, k) => v - ALPHA * (q[k] - b[k])); ys.push(y); Es.push(E(y));
      if (o.auto) { if (Es[i] <= o.tau) { reason = 'good enough'; break; } if (Es[i - 1] - Es[i] < EPS) { reason = 'settled'; break; } }
    }
    return { b, floor: ce(b, b), ys, Es, n: ys.length - 1, reason, E };
  }
  // DDIM (eta = 0) over the logit vector with the exact posterior-mean denoiser for N(mu, s² I)
  function denoiseStep(mu, z, a, a2) {
    const s2 = S_DIFF * S_DIFF, sa = Math.sqrt(a), c = sa * s2 / (a * s2 + 1 - a);
    const x0 = mu.map((m, k) => m + c * (z[k] - sa * m));
    const e = z.map((zk, k) => (zk - sa * x0[k]) / Math.sqrt(Math.max(1e-12, 1 - a)));
    return { x0, e, z: x0.map((x, k) => Math.sqrt(a2) * x + Math.sqrt(1 - a2) * e[k]) };
  }
  function runDiff(lib, pk, o) {
    const mu = center(belief(lib, 2, pk)), T = Math.min(o.budget, TD), r = lib.rng(o.seed * 7919 + 3);
    let z = [r.normal(), r.normal(), r.normal()]; const zs = [z], eps = [null];
    for (let j = T; j >= 1; j--) { const st = denoiseStep(mu, z, abar(j / T), j - 1 === 0 ? 1 : abar((j - 1) / T)); z = st.z; zs.push(z); eps.push(st.e); }
    return { T, zs, eps, mu };
  }

  // ---------- numbers vs arrows (prose lab) ----------
  const WELLS = [{ m: [-1.15, -0.45], s: 0.85, w: 0.55 }, { m: [1.35, 0.75], s: 0.62, w: 0.45 }];
  const SWIRL = [0.1, 0.15], NPATH = 64, DOM = { x: [-3, 3], y: [-2.25, 2.25] };
  const lw2 = (y) => WELLS.map(W => Math.log(W.w) - ((y[0] - W.m[0]) ** 2 + (y[1] - W.m[1]) ** 2) / (2 * W.s * W.s));
  function E2(y) { const ls = lw2(y), m = Math.max(...ls); return -(m + Math.log(ls.reduce((a, l) => a + Math.exp(l - m), 0))); }
  function gradE2(y) { const ls = lw2(y), m = Math.max(...ls), ex = ls.map(l => Math.exp(l - m)), Z = ex.reduce((a, b) => a + b, 0); let gx = 0, gy = 0; WELLS.forEach((W, k) => { const rk = ex[k] / Z; gx += rk * (y[0] - W.m[0]) / (W.s * W.s); gy += rk * (y[1] - W.m[1]) / (W.s * W.s); }); return [gx, gy]; }
  function field(y, c) { const g = gradE2(y), dx = y[0] - SWIRL[0], dy = y[1] - SWIRL[1], env = Math.exp(-(dx * dx + dy * dy) / (2 * 1.6 * 1.6)); return [-g[0] - c * env * dy, -g[1] + c * env * dx]; }
  function bez(A, B, side) { const mx = (A[0] + B[0]) / 2, my = (A[1] + B[1]) / 2, dx = B[0] - A[0], dy = B[1] - A[1], L = Math.hypot(dx, dy) || 1, d = 0.55 * L + 0.5; const Cc = [mx - dy / L * d * side, my + dx / L * d * side]; return (t) => [(1 - t) * (1 - t) * A[0] + 2 * (1 - t) * t * Cc[0] + t * t * B[0], (1 - t) * (1 - t) * A[1] + 2 * (1 - t) * t * Cc[1] + t * t * B[1]]; }
  function pathIntegral(P, c) { let s = 0, prev = P(0); for (let i = 1; i <= NPATH; i++) { const cur = P(i / NPATH), mid = P((i - 0.5) / NPATH), f = field(mid, c); s -= f[0] * (cur[0] - prev[0]) + f[1] * (cur[1] - prev[1]); prev = cur; } return s; }

  // ---------- Table 1 ----------
  const FACETS = [
    { n: 'Facet 1', t: 'Dynamic compute allocation', d: 'Spend more computation on harder predictions, at the granularity of each prediction (p.3, footnote 3). Lunch versus a career change.' },
    { n: 'Facet 2', t: 'Uncertainty in continuous spaces', d: 'Know how unsure a prediction is, also for continuous outputs such as video frames, where there is no softmax (p.3). A pedestrian who might step out from behind a parked car.' },
    { n: 'Facet 3', t: 'Verification of predictions', d: 'Score a candidate, so the model can stop early, think longer, or keep the best of many. "verifying solutions is exponentially easier than generating solutions" (p.3).' },
  ];
  const ROWS = [
    { n: 'FF Transformers', why: [
      'Fixed depth and width and a single forward pass per prediction, so they "are unable to dynamically allocate more computation to each prediction" (Sec 6.1, p.15). Chain-of-thought adds tokens, but each token still gets a fixed budget (Sec G.1.2, p.39).',
      'Softmax gives token probabilities for text (p.3), but for continuous outputs "the normalization process ... is not as well-defined", so uncertainty needs tricks such as Vector Quantization or ELBO objectives (p.15).',
      'Not trained to verify samples, so improving a single prediction at inference "often requires external models" (p.15).'] },
    { n: 'RNNs', why: [
      '"most modern RNNs only update their internal state with new information, meaning they cannot be used for thinking longer" (p.2). Recurrent-depth variants can loop but are not widely adopted (footnote 2, p.3).',
      'Like Transformers, standard RNNs "generally do not provide strong or reliable uncertainty estimates" for continuous outputs without discretization or pseudo-objectives (p.3).',
      'Recurrent-depth RNNs "still lack mechanisms for explicit verification" (p.2): they "amortize gradient prediction of the energy function" (p.16).'] },
    { n: 'Diffusion Transformers', why: [
      'More denoising steps means more computation per prediction (p.3). But "a fixed denoising schedule ... restricts their ability to adaptively halt or extend computation" (p.16), and they "typically fail to benefit from denoising steps beyond what they were trained on" (p.2).',
      'Each step predicts noise, not an energy, so "diffusion models cannot give unnormalized likelihood estimates at each step" (Fig 1 caption, p.2). Likelihoods need the full reverse process with ELBOs or solvers (p.37). Score-based diffusion can express uncertainty but is less widely used (footnote 4, p.3).',
      '"in practice an external verifier is necessary to improve performance at inference time beyond increasing denoising steps" (p.16).'] },
    { n: 'EBTs', why: [
      'Thinking is gradient descent on the energy, so an EBT can iterate "for any number of steps" (Table 1 caption, p.3) and stop when the energy converges (Fig 2 caption, p.4). Extra passes cut the OOD perplexity increase by up to 29% from EBT\'s own no-thinking point (Fig 6a).',
      'The energy is an unnormalized likelihood, defined for continuous predictions too (p.3, p.4). Hard tokens keep higher energy than easy ones (Fig 8, p.12); nearly empty early video frames have higher energy (Fig 11, p.14). The evidence is qualitative (heatmaps with an unspecified normalization), and energy values are never supervised directly (p.37).',
      'The forward pass is the verifier: one scalar per candidate. Algorithm 2 optimizes M candidates and keeps the lowest-energy one (p.7), with no external reward or verifier model (p.8).'] },
  ];

  EBT.panel({
    id: 'families',
    nav: 'Four model families',
    title: 'Four ways to predict the next token',
    lede: 'Every family in the paper\'s Figure 1 predicts token t+1 from tokens 1..t. They differ in what one forward pass returns, and in whether a second pass on the same token can help.',
    text: `
      <figure class="fam-fig1"><div class="paper-fig"><img src="media/paper/fig01.png" alt="Paper Figure 1: (a) AR Transformer maps x1..xt to the next token; (b) an RNN chains states over x1..xt; (c) a Diffusion Transformer takes x1..xt and a candidate and outputs Noise(candidate); (d) an EBT takes x1..xt and a candidate and outputs Energy(candidate)."></div><figcaption><span class="src paper">from the paper · Fig 1, p.2</span></figcaption></figure>
      <p>Look at what comes out of each box: a token, a token, Noise($\\hat x_{t+1}$), Energy($\\hat x_{t+1}$). The last two also take a candidate prediction <em>as input</em>, so they can be run again on an improved candidate. That single difference decides who can think longer, and who can tell how good its guess is.</p>
      <p>The figure runs all four on one next-token question with a shared budget of forward passes. We keep three candidate tokens so that every prediction, a distribution $q$ over the candidates, is a point in a triangle; a real model has 50,277 (Table D.2). The dashed crosshair marks the true next-token distribution $p$.</p>`,
    steps: [
      { label: 'Same question, four machines', html: '<p>Press <b>[ play ]</b>. Each tick, every lane that can still use a pass runs one. The KL chart under the lanes tracks the gap to the truth, $\\mathrm{KL}(p\\,\\|\\,q)$, which we can measure and the models cannot. All lanes share the same size of knowledge error, so they end near the same point; what differs is how they get there and what they know about it.</p>' },
      { label: 'AR and RNN: compute is fixed per token', html: '<p>An AR Transformer computes $$p_\\theta(x_{t+1}\\mid x_{\\le t}) = \\mathrm{softmax}\\big(f_\\theta(x_{\\le t})\\big)$$ with a fixed stack of layers, so every token costs the same (Sec 6.1, p.15), and a rerun on the same input returns the same $p$. An RNN updates $h_t = g(h_{t-1}, x_t)$ only when a new token $x_t$ arrives (p.2, p.16); with no new token there is nothing to update. Their extra passes are crossed out.</p>' },
      { label: 'Diffusion: more passes, fixed schedule', html: '<p>A diffusion Transformer starts from noise and denoises over a schedule of $T$ steps, so it does spend more compute per prediction (Table 1, Facet 1). But $T$ is set before it starts. You can pick a different $T$ in advance, but the model cannot decide half-way that this token needs more or fewer steps: the schedule "restricts their ability to adaptively halt or extend computation" (p.16), and models "typically fail to benefit from denoising steps beyond what they were trained on" (p.2). Here $T = 8$: a larger budget goes unused, a smaller one squeezes the same schedule into fewer, coarser steps (drag the budget below 8).</p>' },
      { label: 'Diffusion outputs arrows, not scores', html: '<p>Each denoising step outputs a noise estimate, which is a scaled score: $$\\epsilon_\\theta(x_t, t) \\approx -\\sigma_t\\,\\nabla_{x_t}\\log p_t(x_t)$$ where $\\sigma_t = \\sqrt{1-\\bar\\alpha_t}$ is the noise scale at step $t$. Writing $E_t = -\\log p_t + \\text{const}$ gives $\\epsilon_\\theta \\approx \\sigma_t\\nabla E_t$: the gradient of an energy (one per noise level), never the energy itself. That is why the paper calls diffusion an implicit EBM (p.36, p.41). The diffusion triangle now shows its arrow field; the EBT triangle shows a number at every point. To rank two candidates with arrows alone you must integrate along a path, and a learned field need not be an exact gradient:</p><div class="fam-slot" data-slot="arrows"></div>' },
      { label: 'EBT: a number per candidate, descended until it settles', html: '<p>An EBT returns one scalar $E_\\theta(x,\\hat y)$ and predicts by descending it (Eq. 1, p.7): $$\\hat y_{i+1} = \\hat y_i - \\alpha\\,\\nabla_{\\hat y}E_\\theta(x,\\hat y_i).$$ The paper counts one function evaluation per step (p.8); each also needs a backward pass to $\\hat y$. The lane stops when $E \\le \\tau = 0.45$ (good enough) or when a step lowers $E$ by less than 0.002 (converged, as in the Fig 2 caption). From the default start, the easy prompt stops after 6 passes, because its energy falls below τ. The medium and hard prompts never get that low: the energy levels off near the entropy of the lane\'s belief, and the lane stops only when it stops falling (9 and 8 passes). The pass count also depends on where the random start lands; press <b>[ new noise ]</b>. The level of the plateau does not: thinking longer cannot make a favorite color predictable, and the high plateau is the model reporting that (Facets 1 and 2), much as hard tokens keep higher energy in the paper\'s Fig 8.</p>' },
      { label: 'Table 1, cell by cell', html: '<p>The paper condenses the comparison into Table 1 (p.3). Hover, tap or tab to a cell for the paper\'s reason; hovering a row highlights that lane. The ✗ marks are "generally" true: recurrent-depth RNNs and score-based diffusion or mixture density networks are partial exceptions (footnotes 2 and 4, p.3).</p><div class="fam-slot" data-slot="table1"></div>' },
    ],
    after: `
      <h3>What EBT adds, and what it costs</h3>
      <p>The paper calls diffusion the closest relative: diffusion models "can be seen as predicting the gradient of the data density/energy function", so EBMs are "a generalization of diffusion models that learn to explicitly verify predictions" (p.16). Recurrent-depth RNNs likewise "amortize gradient prediction of the energy function" (p.16). Learning the scalar itself is what buys a stopping rule, an uncertainty readout and Best-of-N selection without a second model.</p>
      <p>The price: every EBT step needs a backward pass to $\\hat y$, and training differentiates through those steps (gradients of gradients). Each optimization step of an autoregressive EBT costs about 3.33× the FLOPs of a Transformer++ training step, so the two-step pretraining runs cost 6.66× (p.36). The advantages over diffusion in Sec E.2 hold "under the assumption that the energy landscape is well formed and that optimization is well behaved" (p.37).</p>
      <p class="note">What is real in the lanes: the diffusion lane runs DDIM with the exact denoiser for a Gaussian target around its belief; the EBT lane runs gradient descent on $E(\\hat y) = \\mathrm{CE}(b, \\mathrm{softmax}(\\hat y))$, whose gradient is $\\mathrm{softmax}(\\hat y) - b$ and whose minimum is the entropy of its belief $b$. AR and RNN internals are schematic. Beliefs are hand-set, with the same logit noise for every lane. The stopping threshold τ assumes energies are comparable across prompts; the paper never trains energy levels directly (p.37, p.41).</p>`,
    source: [{ kind: 'concept', note: 'hand-set beliefs, real DDIM and descent' }, { kind: 'paper', note: 'Fig 1, Table 1, Sec 6.1–6.4' }],

    figure(stage, ctx) {
      const { lib } = ctx, h = lib.h, C = lib.C, T_ = lib.text;
      const S = { prompt: 'medium', budget: 12, auto: true, tau: 0.45, seed: 1, k: 0, phase: 1, playing: false, stepping: false, focus: null, quiver: false, facets: false, hl: -1 };
      const canvases = [];
      function autoCanvas(parent, o) {
        const box = h('div', { class: 'canvas-box' }); parent.appendChild(box);
        const c = h('canvas', { role: 'img', 'aria-label': o.label || '' }); box.appendChild(c);
        const g = c.getContext('2d'); const st = { w: 0, h: 0, ctx: g, canvas: c, box, dpr: 1 };
        const resize = () => { if (!box.clientWidth) return false; const w = Math.max(220, Math.round(box.clientWidth)); const hh = Math.round(o.height(w)), dpr = Math.min(2, window.devicePixelRatio || 1); if (w === st.w && hh === st.h) return false; st.w = w; st.h = hh; st.dpr = dpr; c.width = Math.round(w * dpr); c.height = Math.round(hh * dpr); c.style.height = hh + 'px'; g.setTransform(dpr, 0, 0, dpr, 0, 0); return true; };
        st.draw = () => { resize(); if (!st.w) return; g.clearRect(0, 0, st.w, st.h); o.draw(g, st.w, st.h); };
        st.toLocal = (ev) => { const r = c.getBoundingClientRect(); return [(ev.clientX - r.left) / r.width * st.w, (ev.clientY - r.top) / r.height * st.h]; };
        if (window.ResizeObserver) new ResizeObserver(() => { if (resize()) st.draw(); }).observe(box);
        canvases.push(st); return st;
      }
      const tick = (g, x1, y1, x2, y2, col, lw, dash) => { g.save(); g.strokeStyle = col; g.lineWidth = lw || 1; if (dash) g.setLineDash(dash); g.beginPath(); g.moveTo(x1, y1); g.lineTo(x2, y2); g.stroke(); g.restore(); };
      const box = (g, x, y, w, hh, o = {}) => { g.save(); if (o.fill) { g.fillStyle = o.fill; g.fillRect(x, y, w, hh); } g.strokeStyle = o.stroke || C.ink; g.lineWidth = o.lw || 1; if (o.dash) g.setLineDash(o.dash); g.strokeRect(x + 0.5, y + 0.5, w - 1, hh - 1); g.restore(); };

      // ---------- computation ----------
      let R = null;
      function compute() {
        const pk = S.prompt;
        const o = { budget: S.budget, auto: S.auto, tau: S.tau, seed: S.seed };
        const ebt = runEBT(lib, pk, o), diff = runDiff(lib, pk, o);
        R = { P: PROMPTS[pk], ar: softmax(belief(lib, 0, pk)), rnn: softmax(belief(lib, 1, pk)), ebt, diff, ticks: Math.max(1, diff.T, ebt.n) };
        R.all = PKEYS.map(k => ({ pk: k, r: runEBT(lib, k, o) }));
      }
      const laneTicks = (id) => id === 'ar' || id === 'rnn' ? 1 : id === 'diff' ? R.diff.T : R.ebt.n;
      function laneTime(id) { const lt = laneTicks(id); if (S.k > lt) return { kl: lt, act: false, f: 1 }; const act = S.phase < 1 && S.k >= 1; return { kl: S.k, act, f: act ? lib.ease(S.phase) : 1 }; }
      function qAt(id, k) {
        if (id === 'ar') return k >= 1 ? R.ar : null;
        if (id === 'rnn') return k >= 1 ? R.rnn : null;
        if (id === 'diff') return softmax(R.diff.zs[Math.min(k, R.diff.T)]);
        return softmax(R.ebt.ys[Math.min(k, R.ebt.n)]);
      }
      function curQ(id) { const t = laneTime(id), q1 = qAt(id, t.kl); if (!t.act) return q1; const q0 = qAt(id, t.kl - 1); if (!q0) return t.f > 0.8 ? q1 : null; return q0.map((v, i) => lib.lerp(v, q1[i], t.f)); }

      // ---------- stage layout ----------
      const LF = lib.frame(stage, { label: 'Four lanes, one token', sub: '·' });
      const subEl = LF.wrap.querySelector('.fig-sub');
      const lanesCv = autoCanvas(LF.frame, { height: (w) => (w >= 520 ? 92 : 190) * 4, label: 'Four model families predicting the same next token: mechanism, forward passes used, and the current prediction as a point in the probability triangle', draw: drawLanes });
      const controls = h('div', { class: 'controls' }); stage.appendChild(controls);
      const promptSeg = lib.segmented({ label: 'Prompt difficulty', value: S.prompt, options: PKEYS.map(k => [k, k]), onchange: (v) => { S.prompt = v; restart(true, true); } });
      const budget = lib.slider({ id: 'fam-budget', label: 'budget: passes for this token', min: 1, max: BMAX, step: 1, value: S.budget, oninput: (v) => { S.budget = v; restart(false); } });
      controls.append(h('span', { class: 'fig-label' }, 'prompt'), promptSeg.el, budget.el);
      const controls2 = h('div', { class: 'controls' }); stage.appendChild(controls2);
      const btnPlay = lib.button('play', () => S.playing ? pause() : play(), { primary: true });
      const btnStep = lib.button('step', () => stepOnce());
      const autoBtn = lib.button('EBT stops on its own: on', () => { S.auto = !S.auto; autoBtn.textContent = 'EBT stops on its own: ' + (S.auto ? 'on' : 'off'); autoBtn.setAttribute('aria-pressed', String(S.auto)); restart(false); });
      autoBtn.setAttribute('aria-pressed', 'true');
      controls2.append(btnPlay, btnStep, lib.button('reset', () => { S.playing = false; S.k = 0; S.phase = 1; render(); }), lib.button('new noise', () => { S.seed++; restart(true, true); }), autoBtn);
      const chartRow = h('div', { class: 'fig-row fam-charts' }); stage.appendChild(chartRow);
      const QF = lib.frame(chartRow, { label: 'Gap to the truth, KL(p‖q)' }); QF.wrap.style.flex = '1 1 260px';
      const qCv = autoCanvas(QF.frame, { height: (w) => Math.round(Math.max(126, Math.min(144, w * 0.48))), label: 'KL divergence from the true next-token distribution to each model\'s current guess, against forward passes used', draw: drawQuality });
      const EF = lib.frame(chartRow, { label: 'What the EBT sees: its energy' }); EF.wrap.style.flex = '1 1 260px';
      const eCv = autoCanvas(EF.frame, { height: (w) => Math.round(Math.max(126, Math.min(144, w * 0.48))), label: 'EBT energy at each optimization step for the easy, medium and hard prompt', draw: drawEnergy });
      const legend = h('div', { class: 'readout fam-legend', html: '<span><i class="fam-k ar"></i>AR</span><span><i class="fam-k rnn"></i>RNN</span><span><i class="fam-k diff"></i>diffusion</span><span><i class="fam-k ebt"></i>EBT</span><span><i class="fam-k truth"></i>truth p</span>' });
      stage.appendChild(legend);

      // ---------- lanes ----------
      function lay(w, i) {
        const wide = w >= 520, RH = wide ? 92 : 190, y0 = i * RH;
        if (wide) { const s = 86, tri = { x: w - s - 8, y: y0 + 4, s }; return { wide, RH, y0, tri, mech: { x: 0, y: y0 + 21, w: w - s - 36, h: 44 }, slotY: y0 + 74 }; }
        const s = 92; return { wide, RH, y0, tri: { x: w - s - 8, y: y0 + 92, s }, mech: { x: 0, y: y0 + 21, w, h: 44 }, slotY: y0 + 75, info: { x: 0, y: y0 + 100, w: w - s - 20 } };
      }
      const V = (t) => [[t.x + t.s / 2, t.y], [t.x, t.y + t.s * 0.866], [t.x + t.s, t.y + t.s * 0.866]];
      const qPt = (t, q) => { const v = V(t); return [q[0] * v[0][0] + q[1] * v[1][0] + q[2] * v[2][0], q[0] * v[0][1] + q[1] * v[1][1] + q[2] * v[2][1]]; };
      const heatCache = {};
      function heatImage(pk, s, dpr) {
        const key = pk + ':' + s + ':' + dpr + ':' + S.seed; if (heatCache[key]) return heatCache[key];
        const b = R.ebt.b, n = Math.round(s * dpr), hh = Math.round(s * 0.866 * dpr), cv = document.createElement('canvas'); cv.width = n; cv.height = hh;
        const g = cv.getContext('2d'), img = g.createImageData(n, hh), lo = ce(b, b), hi = lo + 1.6;
        for (let py = 0; py < hh; py++) for (let px = 0; px < n; px++) {
          // barycentric coordinates of the pixel in the triangle (apex = token 0, bottom-left = token 1, bottom-right = token 2)
          // point = q0·apex + q1·bottom-left + q2·bottom-right  ⇒  y = 1 − q0, x = q0/2 + q2
          const x = (px + 0.5) / n, y = (py + 0.5) / hh, q0 = 1 - y, q2 = x - q0 * 0.5, q1 = 1 - q0 - q2;
          if (q1 < -0.01 || q2 < -0.01) continue;
          const E = ce(b, [Math.max(q0, 1e-4), Math.max(q1, 1e-4), Math.max(q2, 1e-4)]);
          const col = lib.cmap(Math.pow(lib.clamp((E - lo) / (hi - lo)), 0.75)), i4 = 4 * (py * n + px);
          img.data[i4] = col[0]; img.data[i4 + 1] = col[1]; img.data[i4 + 2] = col[2]; img.data[i4 + 3] = 170;
        }
        g.putImageData(img, 0, 0); return (heatCache[key] = cv);
      }
      const isoCache = {};
      function isoGrid(pk) { // energy on a grid over the triangle's bounding box (outside points clamped, so they read as high energy)
        const key = pk; if (isoCache[key]) return isoCache[key];
        const b = R.ebt.b, n = 44, E = [];
        for (let r = 0; r < n; r++) { const row = [], y = r / (n - 1), q0 = 1 - y; for (let c = 0; c < n; c++) { const x = c / (n - 1), q2 = x - q0 * 0.5, q1 = 1 - q0 - q2; row.push(ce(b, [Math.max(q0, 1e-4), Math.max(q1, 1e-4), Math.max(q2, 1e-4)])); } E.push(row); }
        return (isoCache[key] = { E, lo: ce(b, b) });
      }
      function drawSimplex(g, t, L, trail, q, dim) {
        const v = V(t), toks = R.P.toks;
        if (L.id === 'ebt') {
          g.drawImage(heatImage(S.prompt, t.s, lanesCv.dpr), t.x, t.y, t.s, t.s * 0.866);
          const G = isoGrid(S.prompt); g.save(); g.beginPath(); g.moveTo(...v[0]); g.lineTo(...v[1]); g.lineTo(...v[2]); g.closePath(); g.clip();
          lib.contours(g, G.E, t.x, t.y, t.s, t.s * 0.866, [0.03, 0.1, 0.25, 0.5, 0.9].map(d => G.lo + d), { color: 'rgba(255,255,255,0.75)', width: 0.8 });
          g.restore();
        }
        if (L.id === 'diff' && S.quiver) drawQuiver(g, t);
        g.save(); g.strokeStyle = C.ink; g.lineWidth = 1; g.beginPath(); g.moveTo(...v[0]); g.lineTo(...v[1]); g.lineTo(...v[2]); g.closePath(); g.stroke(); g.restore();
        T_(g, toks[0], v[0][0] - 6, v[0][1] - 1, { size: 10.5, kind: 'mono', color: C.ink, align: 'right' });
        T_(g, toks[1], v[1][0], v[1][1] + 3, { size: 10.5, kind: 'mono', color: C.ink });
        T_(g, toks[2], v[2][0], v[2][1] + 3, { size: 10.5, kind: 'mono', color: C.ink, align: 'right' });
        const pp = qPt(t, R.P.p);
        g.save(); g.strokeStyle = C.ink; g.lineWidth = 1; g.setLineDash([2, 2]); g.beginPath(); g.arc(pp[0], pp[1], 6, 0, 7); g.stroke(); g.setLineDash([]); g.beginPath(); g.moveTo(pp[0] - 4, pp[1]); g.lineTo(pp[0] + 4, pp[1]); g.moveTo(pp[0], pp[1] - 4); g.lineTo(pp[0], pp[1] + 4); g.stroke(); g.restore();
        const col = L.id === 'ebt' ? C.blue : C.ink;
        if (trail && trail.length > 1) { const pts = trail.map(qq => qPt(t, qq)); if (L.id === 'ebt') lib.line(g, pts, { color: '#fff', width: 3 }); lib.line(g, pts, { color: col, width: 1.3, dash: L.id === 'diff' ? [3, 2] : null }); pts.slice(0, -1).forEach(p => lib.dot(g, p[0], p[1], 1.6, col)); }
        if (q) { const p = qPt(t, q); lib.dot(g, p[0], p[1], 4, col, { stroke: '#fff', lw: 1.5 }); }
        void dim;
      }
      function drawQuiver(g, t) { // diffusion's learned arrows at a mid noise level, mapped onto the triangle
        const mu = R.diff.mu, a = abar(0.5), a2 = abar(0.375), n = 6;
        for (let i = 1; i < n; i++) for (let j = 1; j < n - i; j++) {
          const q = [i / n, j / n, (n - i - j) / n], z = center(q.map(Math.log)), st = denoiseStep(mu, z, a, a2);
          const q2 = softmax(z.map((zk, k) => zk + 0.35 * (st.z[k] - zk))), p1 = qPt(t, q), p2 = qPt(t, q2);
          const d = Math.hypot(p2[0] - p1[0], p2[1] - p1[1]); if (d < 0.3) continue; const sc = 10 / d;
          lib.arrow(g, p1[0] - (p2[0] - p1[0]) * sc / 2, p1[1] - (p2[1] - p1[1]) * sc / 2, p1[0] + (p2[0] - p1[0]) * sc / 2, p1[1] + (p2[1] - p1[1]) * sc / 2, { color: 'rgba(17,17,17,0.7)', width: 1.1, head: 4 });
        }
      }
      function slots(g, x, y, used, active, blockedFrom, col, maxW) {
        const B = S.budget, gap = 3, sw = Math.max(5, Math.min(11, (maxW - (B - 1) * gap) / B));
        for (let i = 0; i < B; i++) {
          const sx = x + i * (sw + gap);
          if (i < used) { g.fillStyle = col; g.fillRect(sx, y, sw, 9); }
          else if (i === used && active) { g.fillStyle = lib.rgba(col === C.blue ? '#2f3cff' : '#111111', 0.3); g.fillRect(sx, y, sw, 9); }
          else if (i >= blockedFrom) { box(g, sx, y, sw, 9, { stroke: C.faint }); tick(g, sx + 1.5, y + 1.5, sx + sw - 1.5, y + 7.5, C.faint, 1); tick(g, sx + sw - 1.5, y + 1.5, sx + 1.5, y + 7.5, C.faint, 1); }
          else box(g, sx, y, sw, 9, { stroke: C.rule });
        }
        return x + B * (sw + gap) + 6;
      }
      function drawLanes(g, w) {
        if (!R) return;
        LANES.forEach((L, i) => {
          const Ly = lay(w, i), m = Ly.mech, t = laneTime(L.id), col = L.id === 'ebt' ? C.blue : C.ink;
          const dim = S.focus && !S.focus.includes(L.id);
          if (S.hl === i) { g.fillStyle = C.blue4; g.fillRect(0, Ly.y0, w, Ly.RH); }
          if (i > 0) tick(g, 0, Ly.y0 + 0.5, w, Ly.y0 + 0.5, C.rule, 1);
          g.save(); if (dim) g.globalAlpha = 0.28;
          // header
          T_(g, (i + 1) + '  ' + L.name.toUpperCase(), 0, Ly.y0 + 5, { size: 11, kind: 'mono', weight: 700, color: col });
          g.font = lib.font(11, 'mono', 700); const nw = g.measureText((i + 1) + '  ' + L.name.toUpperCase()).width;
          if (S.facets) {
            let fx = nw + 12; L.facets.forEach((f, j) => { const s = 'F' + (j + 1) + (f ? ' ✓' : ' ✗'); T_(g, s, fx, Ly.y0 + 5, { size: 10.5, kind: 'mono', color: f ? (L.id === 'ebt' ? C.blue : C.ink) : C.faint, weight: f ? 700 : 400 }); g.font = lib.font(10.5, 'mono'); fx += g.measureText(s).width + 8; });
          } else if (Ly.wide || nw < w - 150) T_(g, L.eq, nw + 12, Ly.y0 + 5, { size: 10.5, kind: 'mono', color: C.muted });
          // mechanism
          const capY = Ly.slotY - 1;
          let cap = '';
          if (L.id === 'ar') {
            const cw = Math.min(124, m.w * 0.3), bh = 28, by = m.y;
            box(g, m.x, by, cw, bh); T_(g, R.P.ctx + ' ___', m.x + 5, by + 2, { size: 10.5, kind: 'mono', maxWidth: cw - 8, lh: 1.2 });
            const lx = m.x + cw + 16; lib.arrow(g, m.x + cw + 2, by + bh / 2, lx - 3, by + bh / 2, { color: C.faint, width: 1, head: 5 });
            const nL = 6, lwd = 8, lit = t.kl >= 1 ? (t.act ? Math.floor(t.f * (nL + 0.99)) : nL) : 0;
            for (let k = 0; k < nL; k++) box(g, lx + k * (lwd + 4), by, lwd, bh, { fill: k < lit ? C.ink : '#fff', stroke: C.ink });
            T_(g, '6 fixed layers', lx, by + bh + 2, { size: 9.5, kind: 'mono', color: C.muted });
            const ox = lx + nL * (lwd + 4) + 14; lib.arrow(g, ox - 12, by + bh / 2, ox - 2, by + bh / 2, { color: C.faint, width: 1, head: 5 });
            box(g, ox, by, 64, bh, { stroke: t.kl >= 1 && !t.act ? C.ink : C.rule }); T_(g, 'softmax', ox + 32, by + 2, { size: 9.5, kind: 'mono', color: C.muted, align: 'center' }); T_(g, 'p(·)', ox + 32, by + 14, { size: 11, kind: 'mono', color: t.kl >= 1 ? C.ink : C.faint, align: 'center' });
            const x2 = slots(g, m.x, Ly.slotY, t.kl >= 1 ? 1 : 0, t.act, 1, C.ink, Ly.wide ? 190 : m.w * 0.55);
            cap = S.budget > 1 ? `${t.kl >= 1 ? 1 : 0} of ${S.budget} · a rerun gives the same p` : '1 of 1 · exactly what it needs';
            T_(g, cap, Ly.wide ? x2 : m.x, Ly.wide ? capY : Ly.slotY + 12, { size: 10, kind: 'mono', color: C.muted });
          } else if (L.id === 'rnn') {
            const words = R.P.ctx.split(' '), n = words.length, by = m.y + 13, bh = 19, bw = 22;
            const sp = Math.min(58, (m.w - 70) / n);
            words.forEach((wd, k) => {
              const x = m.x + k * sp, last = k === n - 1, lit = last && t.kl >= 1;
              T_(g, wd, x + bw / 2, m.y, { size: 10, kind: 'mono', color: last ? C.ink : C.muted, align: 'center' });
              const r = lib.rng(31 + k * 7 + PKEYS.indexOf(S.prompt) * 5);
              for (let c = 0; c < 4; c++) { const v = Math.abs(Math.tanh(r.normal())); g.fillStyle = lib.rgba('#111111', (last && t.kl < 1 ? 0.05 : 0.12 + 0.6 * v)); g.fillRect(x + 3, by + 2 + c * 4.2, bw - 6, 3); }
              box(g, x, by, bw, bh, { stroke: lit ? C.ink : C.faint, lw: lit ? 1.4 : 1 });
              if (k < n - 1) lib.arrow(g, x + bw + 1, by + bh / 2, x + sp - 2, by + bh / 2, { color: C.faint, width: 1, head: 4 });
            });
            const ex = m.x + (n - 1) * sp + bw + 4; lib.arrow(g, ex, by + bh / 2, ex + 18, by + bh / 2, { color: t.kl >= 1 ? C.ink : C.faint, width: 1, head: 5 });
            T_(g, 'p(·)', ex + 22, by + bh / 2 - 6, { size: 11, kind: 'mono', color: t.kl >= 1 ? C.ink : C.faint });
            const x2 = slots(g, m.x, Ly.slotY, t.kl >= 1 ? 1 : 0, t.act, 1, C.ink, Ly.wide ? 190 : m.w * 0.55);
            cap = S.budget > 1 ? `${t.kl >= 1 ? 1 : 0} of ${S.budget} · no new token, no update` : '1 of 1 · one state update';
            T_(g, cap, Ly.wide ? x2 : m.x, Ly.wide ? capY : Ly.slotY + 12, { size: 10, kind: 'mono', color: C.muted });
          } else if (L.id === 'diff') {
            const D = R.diff, rw = 9, gap = 4, hmax = 22, base = m.y + 36;
            T_(g, 'noise level', m.x, m.y - 1, { size: 9.5, kind: 'mono', color: C.muted });
            for (let r = 1; r <= D.T; r++) { // left = noisiest level
              const xx = m.x + (r - 1) * (rw + gap), j = D.T - r + 1, sig = Math.sqrt(1 - abar(j / D.T)), hg = Math.max(2, sig * hmax);
              const done = t.kl >= r && !(t.act && t.kl === r);
              g.fillStyle = done ? C.ink : (t.act && t.kl === r ? lib.rgba('#111111', 0.35) : '#fff'); g.fillRect(xx, base - hg, rw, hg); box(g, xx, base - hg, rw, hg, { stroke: C.ink });
            }
            T_(g, 'schedule: T = ' + D.T + (D.T < TD ? ' (squeezed)' : ''), m.x, base + 3, { size: 9.5, kind: 'mono', color: C.muted });
            const ex = m.x + TD * (rw + gap) + 18, eps = t.kl >= 1 ? D.eps[Math.min(t.kl, D.T)] : null;
            T_(g, 'step output ε', ex, m.y - 1, { size: 9.5, kind: 'mono', color: C.muted });
            const mid = m.y + 26;
            tick(g, ex, mid, ex + 54, mid, C.rule, 1);
            [0, 1, 2].forEach(k => { const v = eps ? Math.max(-1, Math.min(1, eps[k] / 1.8)) : 0; g.fillStyle = C.ink; g.fillRect(ex + 4 + k * 17, v >= 0 ? mid - v * 16 : mid, 11, Math.max(1, Math.abs(v) * 16)); });
            if (m.w - (ex - m.x) > 190) T_(g, 'a direction for each logit,\nnot a score for the guess', ex + 64, m.y + 8, { size: 9.5, kind: 'mono', color: C.muted, lh: 1.3 });
            const x2 = slots(g, m.x, Ly.slotY, Math.min(t.kl, D.T), t.act, D.T, C.ink, Ly.wide ? 190 : m.w * 0.55);
            cap = S.budget > TD ? `${Math.min(t.kl, D.T)} of ${S.budget} · schedule ends at ${TD}` : S.budget < TD ? `${Math.min(t.kl, D.T)} of ${S.budget} · schedule squeezed to ${D.T}` : `${Math.min(t.kl, D.T)} of ${S.budget} · one per noise level`;
            T_(g, cap, Ly.wide ? x2 : m.x, Ly.wide ? capY : Ly.slotY + 12, { size: 10, kind: 'mono', color: C.muted });
          } else {
            const Eb = R.ebt, kk = Math.min(t.kl, Eb.n), by = m.y, bh = 28;
            const q = curQ('ebt') || softmax(Eb.ys[kk]);
            box(g, m.x, by, 58, bh, { stroke: C.blue }); T_(g, 'ŷ', m.x + 4, by + 2, { size: 10, kind: 'mono', color: C.blue });
            q.forEach((v, k) => { const hb = v * (bh - 8); g.fillStyle = C.blue; g.fillRect(m.x + 18 + k * 12, by + bh - 4 - hb, 9, hb); });
            const ex = m.x + 76; lib.arrow(g, m.x + 60, by + bh / 2, ex - 2, by + bh / 2, { color: C.faint, width: 1, head: 5 });
            box(g, ex, by, 66, bh, { stroke: C.blue }); T_(g, 'E(x, ŷ)', ex + 33, by + 8, { size: 10.5, kind: 'mono', color: C.blue, align: 'center' });
            const eX = ex + 84; lib.arrow(g, ex + 68, by + bh / 2, eX - 2, by + bh / 2, { color: C.faint, width: 1, head: 5 });
            const Eshow = t.act && kk >= 1 ? lib.lerp(Eb.Es[kk - 1], Eb.Es[kk], t.f) : Eb.Es[kk];
            T_(g, 'E = ' + Eshow.toFixed(3), eX, by - 1, { size: 14, kind: 'mono', weight: 700, color: C.blue });
            T_(g, kk >= 1 ? 'ΔE ' + (Eb.Es[kk] - Eb.Es[kk - 1]).toFixed(3) + '  floor ' + Eb.floor.toFixed(3) : 'random start ŷ₀ ~ N(0, I)', eX, by + 17, { size: 9.5, kind: 'mono', color: C.muted });
            // return loop
            const ay = by + bh + 1, x1 = eX + 10, x0 = m.x + 29;
            g.save(); g.strokeStyle = lib.rgba('#2f3cff', 0.6); g.lineWidth = 1; g.setLineDash([3, 3]); g.beginPath(); g.moveTo(x1, by + 28); g.bezierCurveTo(x1, ay + 16, x0, ay + 16, x0, ay + 2); g.stroke(); g.restore();
            g.fillStyle = C.blue; g.beginPath(); g.moveTo(x0, ay); g.lineTo(x0 - 3.5, ay + 6); g.lineTo(x0 + 3.5, ay + 6); g.fill();
            { const lb = 'ŷ ← ŷ − α∇ŷE'; g.font = lib.font(9.5, 'mono'); const tw = g.measureText(lb).width, lx = (x0 + x1) / 2; g.fillStyle = S.hl === i ? C.blue4 : '#fff'; g.fillRect(lx - tw / 2 - 3, ay + 5, tw + 6, 12); T_(g, lb, lx, ay + 6, { size: 9.5, kind: 'mono', color: C.blue, align: 'center' }); }
            const done = t.kl >= Eb.n && !t.act;
            const x2 = slots(g, m.x, Ly.slotY, t.act ? Math.max(0, kk - 1) : kk, t.act, BMAX + 1, C.blue, Ly.wide ? 190 : m.w * 0.55);
            cap = t.kl === 0 ? 'random start · each pass: E, ∇E, one step' : !done ? `thinking · ${kk} pass${kk > 1 ? 'es' : ''}` : Eb.reason === 'budget' ? `${kk} of ${S.budget} · budget used` : `${kk} of ${S.budget} · stopped: ${Eb.reason}`;
            T_(g, cap, Ly.wide ? x2 : m.x, Ly.wide ? capY : Ly.slotY + 12, { size: 10, kind: 'mono', color: done ? C.blue : C.muted });
          }
          // simplex
          const trailK = laneTime(L.id).kl;
          let trail = null;
          if (L.id === 'diff') trail = R.diff.zs.slice(0, Math.min(trailK, R.diff.T) + 1).map(softmax).concat(curQ('diff') ? [curQ('diff')] : []);
          if (L.id === 'ebt') trail = R.ebt.ys.slice(0, Math.min(trailK, R.ebt.n) + 1).map(softmax).concat(curQ('ebt') ? [curQ('ebt')] : []);
          drawSimplex(g, Ly.tri, L, trail, curQ(L.id), dim);
          if (!Ly.wide) { // phone: numeric readout beside the triangle
            const q = curQ(L.id), inf = Ly.info;
            T_(g, 'q = ' + (q ? '(' + q.map(v => v.toFixed(2)).join(', ') + ')' : 'none yet'), inf.x, inf.y + 6, { size: 10.5, kind: 'mono', color: C.ink });
            T_(g, 'KL to truth ' + (q ? kl(R.P.p, q).toFixed(3) : '–'), inf.x, inf.y + 22, { size: 10.5, kind: 'mono', color: C.muted });
            if (L.id === 'ebt') T_(g, 'energy ' + R.ebt.Es[Math.min(t.kl, R.ebt.n)].toFixed(3), inf.x, inf.y + 38, { size: 10.5, kind: 'mono', color: C.blue });
            else T_(g, L.id === 'diff' ? 'no score for its guess' : 'no score for its guess', inf.x, inf.y + 38, { size: 10.5, kind: 'mono', color: C.faint });
          }
          g.restore();
        });
      }

      // ---------- charts ----------
      function chartAxes(g, w, hh, ylim, yt, xlab, ylab) {
        const L = 36, Rr = w - 8, top = 30, bot = hh - 26;
        const X = (v) => L + v / BMAX * (Rr - L), Y = (v) => bot - (v - ylim[0]) / (ylim[1] - ylim[0]) * (bot - top);
        const dec = yt.some(v => Math.abs(v * 10 - Math.round(v * 10)) > 1e-9) ? 2 : 1;
        yt.forEach(v => { tick(g, L, Y(v), Rr, Y(v), C.rule, 1); T_(g, v.toFixed(dec), L - 5, Y(v), { size: 9.5, kind: 'mono', color: C.muted, align: 'right', baseline: 'middle' }); });
        tick(g, L, bot + 0.5, Rr, bot + 0.5, C.ink, 1);
        [0, 4, 8, 12, 16].forEach(v => T_(g, String(v), X(v), bot + 4, { size: 9.5, kind: 'mono', color: C.muted, align: 'center' }));
        T_(g, xlab, Rr, bot + 14, { size: 9.5, kind: 'mono', color: C.muted, align: 'right' });
        T_(g, ylab, L - 32, 1, { size: 9.5, kind: 'mono', color: C.muted });
        g.save(); g.strokeStyle = C.faint; g.setLineDash([2, 3]); g.beginPath(); g.moveTo(X(S.budget), top); g.lineTo(X(S.budget), bot); g.stroke(); g.restore();
        T_(g, 'budget', X(S.budget) + (S.budget > 12 ? -3 : 3), top + 1, { size: 9.5, kind: 'mono', color: C.faint, align: S.budget > 12 ? 'right' : 'left' });
        return { X, Y, L, Rr, top, bot };
      }
      function drawQuality(g, w, hh) {
        if (!R) return;
        const p = R.P.p, Eb = R.ebt, D = R.diff;
        const ebtPts = Eb.ys.map((y, i) => [i, kl(p, softmax(y))]), difPts = D.zs.map((z, j) => [j, kl(p, softmax(z))]);
        const hi = Math.max(0.2, ...ebtPts.map(q => q[1]), ...difPts.map(q => q[1])) * 1.08;
        const yt = hi > 1 ? [0, 0.5, 1, 1.5, 2].filter(v => v <= hi) : [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8].filter(v => v <= hi);
        const A = chartAxes(g, w, hh, [0, hi], yt.length > 4 ? yt.filter((_, i) => i % 2 === 0) : yt, 'forward passes used', 'KL(p‖q), nats');
        const kE = laneTime('ebt').kl, kD = laneTime('diff').kl, k1 = laneTime('ar').kl;
        if (k1 >= 1) [['ar', R.ar, null, C.ink], ['rnn', R.rnn, [1, 2], '#6b6b70']].forEach(([id, q, dash, col]) => { const y = kl(p, q); tick(g, A.X(1), A.Y(y), A.X(BMAX), A.Y(y), col, 1.2, dash || [6, 0]); lib.dot(g, A.X(1), A.Y(y), 3, col); });
        lib.plot(g, A, difPts.slice(0, Math.min(kD, D.T) + 1), { color: C.ink, width: 1.3, dash: [5, 3], markers: 2.2 });
        lib.plot(g, A, ebtPts.slice(0, Math.min(kE, Eb.n) + 1), { color: C.blue, width: 1.8, markers: 2.6 });
      }
      function drawEnergy(g, w, hh) {
        if (!R) return;
        const all = R.all, hi = Math.max(...all.map(a => Math.max(...a.r.Es))) * 1.04, lo = Math.max(0, Math.min(S.tau, ...all.map(a => a.r.floor)) - 0.15);
        const yt = [0.5, 1, 1.5, 2, 2.5, 3, 3.5].filter(v => v >= lo && v <= hi);
        const A = chartAxes(g, w, hh, [lo, hi], yt, 'optimization step = pass', 'energy E');
        if (S.auto) { tick(g, A.L, A.Y(S.tau), A.Rr, A.Y(S.tau), C.ink, 1, [4, 3]); T_(g, 'τ: good enough', A.Rr, A.Y(S.tau) - 12, { size: 9.5, kind: 'mono', color: C.ink, align: 'right' }); }
        const kE = laneTime('ebt').kl;
        let lx = A.Rr; [...all].reverse().forEach(({ pk, r }) => { const curP = pk === S.prompt, sL = pk + ' ' + r.n; g.font = lib.font(10, 'mono', curP ? 700 : 400); const tw = g.measureText(sL).width; T_(g, sL, lx, A.top - 14, { size: 10, kind: 'mono', weight: curP ? 700 : 400, color: curP ? C.blue : C.blue2, align: 'right' }); lx -= tw + 10; });
        T_(g, 'stops at:', lx, A.top - 14, { size: 10, kind: 'mono', color: C.muted, align: 'right' });
        all.forEach(({ pk, r }) => {
          const curP = pk === S.prompt, n = curP ? Math.min(kE, r.n) : r.n;
          const pts = r.Es.slice(0, n + 1).map((e, i) => [i, e]);
          lib.plot(g, A, pts, { color: curP ? C.blue : C.blue2, width: curP ? 2 : 1, markers: curP ? 2.6 : 1.6 });
          tick(g, A.X(0), A.Y(r.floor), A.X(BMAX), A.Y(r.floor), curP ? C.blue2 : C.blue3, 1, [2, 3]);
          if (n === r.n) { const ex = A.X(r.n), ey = A.Y(r.Es[r.n]); tick(g, ex, ey - 5, ex, ey + 5, curP ? C.blue : C.blue2, 1.5); }
        });
      }

      // ---------- prose labs ----------
      const slot = (name) => ctx.panel.querySelector(`.fam-slot[data-slot="${name}"]`);
      const guard = (el, stepIdx) => { ['click', 'keydown', 'pointerdown'].forEach(ev => el.addEventListener(ev, (e) => { if (ctx.step === stepIdx) e.stopPropagation(); })); };
      // numbers vs arrows
      const arEl = slot('arrows');
      const bS = { A: [-0.55, -1.15], B: [0.7, 1.25], c: 0.3, drag: null };
      let bCv = null, bOut = null, egrid = null;
      if (arEl) {
        guard(arEl, 3); arEl.className = 'fam-slot fam-lab';
        const sw = lib.slider({ id: 'fam-swirl', label: 'curl in the learned arrows', min: 0, max: 2, step: 0.05, value: bS.c, fmt: (v) => v.toFixed(2), oninput: (v) => { bS.c = v; bCv.draw(); bRead(); } });
        const row = h('div', { class: 'controls' }, sw.el, lib.button('swap A, B', () => { const t0 = bS.A; bS.A = bS.B; bS.B = t0; bCv.draw(); bRead(); }), lib.button('random pair', () => { const r = lib.rng(++pairSeed * 7717); bS.A = [-2.4 + 4.8 * r(), -1.8 + 3.6 * r()]; bS.B = [-2.4 + 4.8 * r(), -1.8 + 3.6 * r()]; bCv.draw(); bRead(); }));
        bCv = autoCanvas(arEl, { height: (w) => (w >= 360 ? Math.round((w - 12) / 2 * 0.75) + 18 : Math.round(w * 0.75) * 2 + 40), label: 'Left: an energy, one number per point, with candidates A and B. Right: a learned arrow field and two routes from A to B. Drag A or B.', draw: drawLab });
        arEl.appendChild(row);
        bOut = h('div', { class: 'fam-verdicts' }); arEl.appendChild(bOut);
        arEl.appendChild(h('p', { class: 'fam-small' }, 'Real math: E is a two-well energy; the arrows are −∇E plus curl × a rotational field; each route estimate is −∫ arrows · dl with ' + NPATH + ' evaluations. Real diffusion likelihoods need an ODE solve or an ELBO over many noise levels (Sec E.2, p.37); this keeps only the core point. Trained score networks are close to, not exactly, conservative; how close varies, so treat the curl slider as a what-if.'));
        bCv.canvas.addEventListener('pointerdown', (ev) => { const [px, py] = bCv.toLocal(ev); for (const i of [0, 1]) for (const nm of ['A', 'B']) { const r = panelRect(i, bCv.w), p = toPxL(r, bS[nm]); if (Math.hypot(px - p[0], py - p[1]) < 16) { bS.drag = { nm, i }; bCv.canvas.setPointerCapture(ev.pointerId); ev.preventDefault(); return; } } });
        bCv.canvas.addEventListener('pointermove', (ev) => { if (!bS.drag) return; const [px, py] = bCv.toLocal(ev), r = panelRect(bS.drag.i, bCv.w), q = toDom(r, px, py); bS[bS.drag.nm] = [lib.clamp(q[0], DOM.x[0] + 0.1, DOM.x[1] - 0.1), lib.clamp(q[1], DOM.y[0] + 0.1, DOM.y[1] - 0.1)]; bCv.draw(); bRead(); });
        const end = () => { bS.drag = null; }; bCv.canvas.addEventListener('pointerup', end); bCv.canvas.addEventListener('pointercancel', end);
      }
      let pairSeed = 3;
      function panelRect(i, w) { const two = w >= 360; if (two) { const pw = (w - 12) / 2; return { x: i * (pw + 12), y: 18, w: pw, h: pw * 0.75 }; } return { x: 0, y: 18 + i * (w * 0.75 + 22), w, h: w * 0.75 }; }
      const toPxL = (r, p) => [r.x + (p[0] - DOM.x[0]) / (DOM.x[1] - DOM.x[0]) * r.w, r.y + (DOM.y[1] - p[1]) / (DOM.y[1] - DOM.y[0]) * r.h];
      const toDom = (r, px, py) => [DOM.x[0] + (px - r.x) / r.w * (DOM.x[1] - DOM.x[0]), DOM.y[1] - (py - r.y) / r.h * (DOM.y[1] - DOM.y[0])];
      function drawLab(g, w) {
        if (!egrid) { const n = 64; egrid = []; for (let r = 0; r < n; r++) { const rowv = []; for (let c = 0; c < n; c++) rowv.push(E2([DOM.x[0] + c / (n - 1) * 6, DOM.y[1] - r / (n - 1) * 4.5])); egrid.push(rowv); } }
        const lo = lib.gridRange(egrid)[0], hi = lo + 7, lv = Array.from({ length: 8 }, (_, i) => lo + (hi - lo) * (i + 1) / 9);
        const r0 = panelRect(0, w), r1 = panelRect(1, w);
        T_(g, 'EBT: a number per point', r0.x, r0.y - 16, { size: 10.5, kind: 'mono', weight: 700, color: C.blue });
        T_(g, 'diffusion: an arrow per point', r1.x, r1.y - 16, { size: 10.5, kind: 'mono', weight: 700, color: C.ink });
        lib.heatmap(g, egrid, r0.x, r0.y, r0.w, r0.h, { range: [lo, hi], gamma: 0.8, alpha: 0.6 });
        lib.contours(g, egrid, r0.x, r0.y, r0.w, r0.h, lv, { color: 'rgba(17,17,17,0.18)', width: 1 });
        box(g, r0.x, r0.y, r0.w, r0.h, { stroke: C.ink }); box(g, r1.x, r1.y, r1.w, r1.h, { stroke: C.ink });
        const nx = 13, ny = 10, cell = r1.w / nx;
        for (let i = 0; i < nx; i++) for (let j = 0; j < ny; j++) {
          const p = [DOM.x[0] + (i + 0.5) / nx * 6, DOM.y[1] - (j + 0.5) / ny * 4.5], f = field(p, bS.c), mag = Math.hypot(f[0], f[1]); if (mag < 1e-6) continue;
          const len = Math.min(cell * 0.8, 4 + mag * cell * 0.22), [px, py] = toPxL(r1, p);
          lib.arrow(g, px - f[0] / mag * len / 2, py + f[1] / mag * len / 2, px + f[0] / mag * len / 2, py - f[1] / mag * len / 2, { color: 'rgba(17,17,17,0.35)', width: 1, head: 3.5 });
        }
        [[-1, null], [1, [5, 4]]].forEach(([side, dash]) => { const P = bez(bS.A, bS.B, side), pts = []; for (let i = 0; i <= 48; i++) pts.push(toPxL(r1, P(i / 48))); g.save(); g.beginPath(); g.rect(r1.x, r1.y, r1.w, r1.h); g.clip(); lib.line(g, pts, { color: C.ink, width: 1.6, dash }); g.restore(); });
        const pa = toPxL(r1, bS.A), pb = toPxL(r1, bS.B), cm = [(pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2];
        [[-1, 'route 1'], [1, 'route 2']].forEach(([side, nm]) => { const q = toPxL(r1, bez(bS.A, bS.B, side)(0.5)); let ux = q[0] - cm[0], uy = q[1] - cm[1]; const L = Math.hypot(ux, uy) || 1; ux /= L; uy /= L; const tx = lib.clamp(q[0] + ux * 14, r1.x + 30, r1.x + r1.w - 30), ty = lib.clamp(q[1] + uy * 10, r1.y + 8, r1.y + r1.h - 8); T_(g, nm, tx, ty, { size: 10, kind: 'mono', color: C.ink, align: 'center', baseline: 'middle' }); });
        [r0, r1].forEach((r, i) => [['A', bS.A], ['B', bS.B]].forEach(([nm, p]) => {
          const [px, py] = toPxL(r, p); lib.dot(g, px, py, 7.5, i === 0 ? C.blue : C.ink, { stroke: '#fff', lw: 1.5 });
          T_(g, nm, px, py + 0.5, { size: 10, kind: 'mono', weight: 700, color: '#fff', align: 'center', baseline: 'middle' });
          if (i === 0) { const lab = 'E ' + E2(p).toFixed(2); g.font = lib.font(10, 'mono'); const tw = g.measureText(lab).width; const lx = px + 10 + tw > r.x + r.w ? px - 10 - tw : px + 10; g.fillStyle = 'rgba(255,255,255,0.85)'; g.fillRect(lx - 2, py - 16, tw + 4, 13); T_(g, lab, lx, py - 15, { size: 10, kind: 'mono', color: C.ink }); }
        }));
      }
      function bRead() {
        if (!bOut) return;
        const eA = E2(bS.A), eB = E2(bS.B), d = eB - eA, i1 = pathIntegral(bez(bS.A, bS.B, -1), bS.c), i2 = pathIntegral(bez(bS.A, bS.B, 1), bS.c);
        const verdict = (x) => Math.abs(x) < 0.02 ? 'a tie' : x > 0 ? 'A is more likely' : 'B is more likely', sgn = (x) => (x >= 0 ? '+' : '') + x.toFixed(2);
        const agree = Math.sign(i1) === Math.sign(i2);
        bOut.innerHTML = `<div class="ebt"><b>energy</b><span>E(B) − E(A) = ${sgn(d)} → ${verdict(d)}</span><i>2 passes</i></div>`
          + `<div><b>route 1</b><span>−∫ f·dl = ${sgn(i1)} → ${verdict(i1)}</span><i>${NPATH} passes</i></div>`
          + `<div><b>route 2</b><span>−∫ f·dl = ${sgn(i2)} → ${verdict(i2)}</span><i>${NPATH} passes</i></div>`
          + `<p class="${agree ? '' : 'warn'}">${bS.c < 0.02 ? 'No curl: the arrows are an exact gradient, so both routes match the energy, at 64 times the cost.' : agree ? `The routes differ by ${Math.abs(i1 - i2).toFixed(2)}: the answer depends on the route. Raise the curl or move B to make them disagree on the ranking.` : 'The routes disagree on which candidate is better: with arrows that are not an exact gradient, "how much better is B than A" has no single answer.'}</p>`;
      }
      // Table 1
      const tEl = slot('table1');
      if (tEl) {
        guard(tEl, 5); tEl.className = 'fam-slot fam-t1';
        const why = h('div', { class: 'fam-why', 'aria-live': 'polite' }); const cells = [];
        const showCell = (ri, fi) => { cells.forEach(c => c.el.classList.toggle('on', c.ri === ri && c.fi === fi)); const v = LANES[ri].facets[fi]; why.innerHTML = `<div class="fam-why-h"><b>${ROWS[ri].n}</b> × <b>${FACETS[fi].n}: ${FACETS[fi].t}</b> <span class="${v ? 'yes' : 'no'}">${v ? '✓ yes' : '✗ no'}</span></div><p>${ROWS[ri].why[fi]}</p><p class="fam-small">${FACETS[fi].d}</p>`; };
        const table = h('table', { class: 'fam-table' },
          h('thead', {}, h('tr', {}, h('th', {}, 'Architecture'), FACETS.map(F => h('th', { class: 'c' }, F.n, h('span', {}, F.t))))),
          h('tbody', {}, ROWS.map((R0, ri) => { const tr = h('tr', { class: ri === 3 ? 'ebt' : '' }, h('td', {}, R0.n), LANES[ri].facets.map((v, fi) => { const b = h('button', { type: 'button', class: 'fam-cell ' + (v ? 'yes' : 'no'), 'aria-label': `${R0.n}, ${FACETS[fi].t}: ${v ? 'yes' : 'no'}. Show the reason.` }, v ? '✓' : '✗'); ['mouseenter', 'focus', 'click'].forEach(e => b.addEventListener(e, () => showCell(ri, fi))); cells.push({ el: b, ri, fi }); return h('td', { class: 'c' }, b); })); tr.addEventListener('mouseenter', () => { S.hl = ri; lanesCv.draw(); }); tr.addEventListener('mouseleave', () => { S.hl = -1; lanesCv.draw(); }); return tr; })));
        tEl.append(h('div', { class: 'tbl' }, table), why, h('p', { class: 'fam-small' }, 'Table 1 (p.3), verbatim marks. Reasons quote or paraphrase the pages cited.'));
        showCell(2, 0);
      }

      // ---------- animation ----------
      const PASS = () => lib.reducedMotion ? 0.01 : 0.42;
      function updateStatus() {
        btnPlay.textContent = S.playing ? 'pause' : (S.k >= R.ticks && S.phase >= 1 ? 'replay' : 'play');
        subEl.innerHTML = `“${R.P.ctx} ___” · candidates: ${R.P.toks.join(', ')} · pass ${Math.min(S.k, R.ticks)} of ${R.ticks}`;
      }
      function render() { if (!R) return; lanesCv.draw(); qCv.draw(); eCv.draw(); updateStatus(); }
      const loop = lib.loop((dt) => {
        if (!S.playing && !S.stepping) return false;
        S.phase += dt / PASS();
        if (S.phase >= 1) { S.phase = 1; if (S.stepping) { S.stepping = false; render(); return false; } if (S.k < R.ticks) { S.k++; S.phase = 0; } else { S.playing = false; render(); return false; } }
        render(); return true;
      });
      function play() { if (S.k >= R.ticks && S.phase >= 1) { S.k = 0; S.phase = 1; } S.playing = true; S.stepping = false; if (S.phase >= 1 && S.k < R.ticks) { S.k++; S.phase = 0; } loop.start(); render(); }
      function pause() { S.playing = false; S.stepping = false; S.phase = 1; render(); }
      function stepOnce() { S.playing = false; if (S.k >= R.ticks) { S.phase = 1; render(); return; } S.k++; S.phase = 0; S.stepping = true; loop.start(); }
      function restart(resetK, autoplay) { compute(); if (resetK) { S.k = 0; S.phase = 1; } else { S.k = R.ticks; S.phase = 1; } if (autoplay && !lib.reducedMotion) play(); else { if (resetK) S.k = R.ticks; render(); } }

      ctx.setCaption('Slots: filled = used, crossed = unusable, empty = not needed. Triangle: the guess q over 3 tokens; crosshair = truth p; EBT shading = its energy over every possible guess (darker = lower, white lines = equal energy); its minimum sits at the lane\'s belief, a little off the truth.');
      compute(); S.k = R.ticks; S.phase = 1;
      const kick = () => { canvases.forEach(c => c.draw()); bRead(); updateStatus(); };
      kick(); if (window.EBTV && EBTV.fontsReady) EBTV.fontsReady.then(kick);
      let shown = false;
      const CFG = [
        { prompt: 'medium', budget: 12, focus: null, quiver: false, facets: false, play: true },
        { prompt: 'medium', budget: 12, focus: ['ar', 'rnn'], quiver: false, facets: false },
        { prompt: 'medium', budget: 12, focus: ['diff'], quiver: false, facets: false },
        { prompt: 'medium', budget: 12, focus: ['diff', 'ebt'], quiver: true, facets: false },
        { prompt: 'hard', budget: 12, focus: ['ebt'], quiver: false, facets: false, play: true },
        { prompt: 'medium', budget: 12, focus: null, quiver: false, facets: true },
      ];
      return {
        step(i) {
          const c = CFG[i]; S.playing = false; S.stepping = false; loop.stop();
          Object.assign(S, { prompt: c.prompt, budget: c.budget, focus: c.focus, quiver: c.quiver, facets: c.facets, auto: true });
          promptSeg.set(S.prompt); budget.set(S.budget); autoBtn.textContent = 'EBT stops on its own: on'; autoBtn.setAttribute('aria-pressed', 'true');
          restart(true, !!c.play && shown);
        },
        show() { kick(); if (!shown) { shown = true; if (!lib.reducedMotion && ctx.step === 0) { S.k = 0; S.phase = 1; setTimeout(play, 300); } } },
        hide() { if (S.playing || S.stepping) { S.playing = false; S.stepping = false; loop.stop(); S.k = R.ticks; S.phase = 1; render(); } },
      };
    },
  });
})();
