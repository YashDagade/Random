/* Chapter "families": Four ways to predict the next token.
   Lab A: four lanes (AR Transformer, RNN, Diffusion Transformer, EBT) predict the next word under a shared pass budget.
          Diffusion runs a real DDIM sampler (exact denoiser for a Gaussian target). EBT runs real gradient descent on
          E(y) = cross-entropy(belief, softmax(y)), gradient softmax(y) - belief. AR / RNN internals are schematic.
   Lab B: numbers vs arrows. An energy ranks two candidates in 2 passes; a learned vector field (what diffusion learns)
          needs a path integral, and if it has any rotational part the answer depends on the path.
   Table 1 (interactive) and a closing note on diffusion as a learned energy gradient (Sec 6.4, E.1). */
(function () {
  'use strict';
  const PKEYS = ['easy', 'medium', 'hard'];
  const PROMPTS = {
    easy: { label: 'Easy', ctx: ['Once', 'upon', 'a'], toks: ['time', 'day', 'hill', 'dream', 'star', 'night', 'year', 'king'], p: [0.86, 0.03, 0.02, 0.02, 0.02, 0.02, 0.015, 0.015] },
    medium: { label: 'Medium', ctx: ['The', 'cat', 'sat', 'on', 'the'], toks: ['mat', 'floor', 'sofa', 'bed', 'rug', 'chair', 'roof', 'table'], p: [0.38, 0.2, 0.12, 0.1, 0.08, 0.06, 0.03, 0.03] },
    hard: { label: 'Hard', ctx: ['My', 'favorite', 'color', 'is'], toks: ['blue', 'red', 'green', 'purple', 'black', 'pink', 'yellow', 'orange'], p: [0.2, 0.17, 0.15, 0.13, 0.11, 0.09, 0.08, 0.07] },
  };
  const LANES = [
    { id: 'ar', name: 'AR Transformer', color: 'ar', table: 0 },
    { id: 'rnn', name: 'RNN', color: 'rnn', table: 1 },
    { id: 'diff', name: 'Diffusion Transformer', color: 'diff', table: 2 },
    { id: 'ebt', name: 'Energy-Based Transformer', color: 'ebt', table: 3 },
  ];
  const TD = 8;            // diffusion: trained schedule length (illustration)
  const ERR = 0.2;         // same-size knowledge error for every lane (std of logit error)
  const S_DIFF = 0.15;     // spread of the diffusion target around its mean logits
  const EPS_SETTLE = 0.003;// EBT: energy counts as settled when one step lowers it by less than this
  const BMAX = 16;

  // ---------- math ----------
  const softmax = (z) => { const m = Math.max(...z); const e = z.map(v => Math.exp(v - m)); const s = e.reduce((a, b) => a + b, 0); return e.map(v => v / s); };
  const ce = (p, q) => -p.reduce((a, pk, k) => a + pk * Math.log(Math.max(1e-12, q[k])), 0);
  const entropy = (p) => ce(p, p);
  const add = (a, b) => a.map((v, i) => v + b[i]);
  const center = (v) => { const m = v.reduce((a, b) => a + b, 0) / v.length; return v.map(x => x - m); };
  const abar = (u) => { const c = Math.cos((u + 0.008) / 1.008 * Math.PI / 2); return Math.max(0, c * c); }; // cosine schedule, u in [0,1]
  const hashStr = (s) => { let x = 2166136261; for (let i = 0; i < s.length; i++) { x ^= s.charCodeAt(i); x = Math.imul(x, 16777619); } return x >>> 0; };
  const argmin = (a) => a.reduce((bi, v, i) => v < a[bi] ? i : bi, 0);

  function laneBelief(lib, laneIdx, pi, logp) { const r = lib.rng(1000 + laneIdx * 97 + pi * 13); return add(logp, logp.map(() => ERR * r.normal())); }

  // EBT: M candidates of gradient descent on E(y) = CE(belief, softmax y). Returns per-candidate trajectories.
  function runEBT(lib, pk, o) {
    const P = PROMPTS[pk], pi = PKEYS.indexOf(pk), logp = P.p.map(Math.log);
    const belief = softmax(laneBelief(lib, 4, pi, logp));
    const Meff = Math.max(1, Math.min(o.M, o.budget)), cap = Math.max(1, Math.floor(o.budget / Meff));
    const E = (y) => ce(belief, softmax(y));
    const cands = [];
    for (let c = 0; c < Meff; c++) {
      const r = lib.rng(o.seed * 104729 + c * 31 + 5);
      let y = P.p.map(() => r.normal());
      const ys = [y], Es = [E(y)]; let stop = cap, reason = 'budget';
      for (let i = 1; i <= cap; i++) {
        const q = softmax(y); y = y.map((v, k) => v - o.alpha * (q[k] - belief[k]));
        const e = E(y); ys.push(y); Es.push(e);
        if (o.auto) {
          if (e <= o.tau) { stop = i; reason = 'confident'; break; }
          if (Es[i - 1] - e < EPS_SETTLE) { stop = i; reason = 'settled'; break; }
        }
      }
      cands.push({ ys, Es, stop, reason });
    }
    const ticks = Math.max(...cands.map(c => c.stop));
    const at = (k) => { // state after tick k
      const idx = cands.map(c => Math.min(k, c.stop));
      const Ek = cands.map((c, j) => c.Es[idx[j]]);
      const best = argmin(Ek);
      const passes = idx.reduce((a, b) => a + b, 0);
      return { idx, Ek, best, passes, y: cands[best].ys[idx[best]], E: Ek[best] };
    };
    return { belief, floor: entropy(belief), cands, cap, Meff, ticks, at };
  }

  // Diffusion: DDIM (eta = 0) over the logit vector with the exact posterior-mean denoiser for N(mu, s^2 I).
  function runDiff(lib, pk, o) {
    const P = PROMPTS[pk], pi = PKEYS.indexOf(pk), logp = P.p.map(Math.log);
    const mu = center(laneBelief(lib, 3, pi, logp));
    const T = Math.min(o.budget, TD), s2 = S_DIFF * S_DIFF;
    const r = lib.rng(o.seed * 7919 + 3);
    let z = mu.map(() => r.normal());
    const zs = [z], eps = [null], rungs = [null];
    for (let j = T; j >= 1; j--) {
      const a = abar(j / T), a2 = j - 1 === 0 ? 1 : abar((j - 1) / T), sa = Math.sqrt(a);
      const c = sa * s2 / (a * s2 + 1 - a);
      const x0 = mu.map((m, k) => m + c * (z[k] - sa * m));
      const e = z.map((zk, k) => (zk - sa * x0[k]) / Math.sqrt(Math.max(1e-12, 1 - a)));
      z = x0.map((x, k) => Math.sqrt(a2) * x + Math.sqrt(1 - a2) * e[k]);
      zs.push(z); eps.push(e); rungs.push(Math.max(1, Math.round(j / T * TD)));
    }
    return { T, zs, eps, rungs, ticks: T };
  }

  // RNN hidden states over the prompt words (random-weight tanh RNN, schematic only).
  function runRNN(lib, pk) {
    const P = PROMPTS[pk], H = 8, r = lib.rng(4242);
    const W = [...Array(H)].map(() => [...Array(H)].map(() => r.normal() * 0.9 / Math.sqrt(H)));
    const Um = [...Array(H)].map(() => [...Array(H)].map(() => r.normal() * 0.8));
    let hcur = Array(H).fill(0); const hs = [];
    P.ctx.forEach(w => { const re = lib.rng(hashStr(w.toLowerCase())); const e = [...Array(H)].map(() => re.normal()); hcur = hcur.map((_, i) => Math.tanh(W[i].reduce((a, wij, j) => a + wij * hcur[j], 0) + Um[i].reduce((a, uij, j) => a + uij * e[j], 0))); hs.push(hcur); });
    return hs;
  }

  // ---------- Lab B: energy vs learned vector field ----------
  const WELLS = [{ m: [-1.15, -0.45], s: 0.85, w: 0.55 }, { m: [1.35, 0.75], s: 0.62, w: 0.45 }];
  const SWIRL = [0.1, 0.15];
  function E2(y) { const ls = WELLS.map(W => Math.log(W.w) - ((y[0] - W.m[0]) ** 2 + (y[1] - W.m[1]) ** 2) / (2 * W.s * W.s)); const m = Math.max(...ls); return -(m + Math.log(ls.reduce((a, l) => a + Math.exp(l - m), 0))); }
  function gradE2(y) {
    const ls = WELLS.map(W => Math.log(W.w) - ((y[0] - W.m[0]) ** 2 + (y[1] - W.m[1]) ** 2) / (2 * W.s * W.s)); const m = Math.max(...ls);
    const ex = ls.map(l => Math.exp(l - m)), Z = ex.reduce((a, b) => a + b, 0); let gx = 0, gy = 0;
    WELLS.forEach((W, k) => { const rk = ex[k] / Z; gx += rk * (y[0] - W.m[0]) / (W.s * W.s); gy += rk * (y[1] - W.m[1]) / (W.s * W.s); });
    return [gx, gy];
  }
  function field(y, c) { // learned "score": -grad E plus a rotational error of size c
    const g = gradE2(y), dx = y[0] - SWIRL[0], dy = y[1] - SWIRL[1], env = Math.exp(-(dx * dx + dy * dy) / (2 * 1.6 * 1.6));
    return [-g[0] - c * env * dy, -g[1] + c * env * dx];
  }
  function bez(A, B, side) { const mx = (A[0] + B[0]) / 2, my = (A[1] + B[1]) / 2, dx = B[0] - A[0], dy = B[1] - A[1], L = Math.hypot(dx, dy) || 1, d = 0.55 * L + 0.5; const Cc = [mx - dy / L * d * side, my + dx / L * d * side]; return (t) => [(1 - t) * (1 - t) * A[0] + 2 * (1 - t) * t * Cc[0] + t * t * B[0], (1 - t) * (1 - t) * A[1] + 2 * (1 - t) * t * Cc[1] + t * t * B[1]]; }
  const NPATH = 64;
  function pathIntegral(P, c) { // estimate of E(B) - E(A) from the field alone: -∫ f · dl (midpoint rule, NPATH field evaluations)
    let s = 0, prev = P(0); for (let i = 1; i <= NPATH; i++) { const cur = P(i / NPATH), mid = P((i - 0.5) / NPATH), f = field(mid, c); s -= f[0] * (cur[0] - prev[0]) + f[1] * (cur[1] - prev[1]); prev = cur; } return s;
  }

  EBT.section({
    id: 'families',
    nav: 'Model families',
    kicker: 'Lab',
    title: 'Four ways to predict the next token',
    lede: 'Every model below must name the next word. Watch how each one spends its compute, and what it can tell you about its own guess. Only one of them can score its own guess with a single number, for text and for continuous outputs alike.',
    mount(el, lib) {
      const h = lib.h, C = lib.C, T = lib.text;
      const reduced = !!lib.reducedMotion;
      const col = (name) => C[name];
      const canvases = [];

      // auto-sized canvas: logical size = CSS size, so canvas text renders at true pixel size
      function autoCanvas(parent, o) {
        const box = h('div', { class: 'canvas-box families-cv' }); parent.appendChild(box);
        const c = h('canvas', { role: 'img', 'aria-label': o.label || '' }); box.appendChild(c);
        const ctx = c.getContext('2d'); const st = { w: 0, h: 0, ctx, canvas: c, box };
        const resize = () => { const w = Math.max(240, Math.round(box.clientWidth || 600)), hh = Math.round(o.height(w)), dpr = Math.min(2, window.devicePixelRatio || 1); if (w === st.w && hh === st.h) return false; st.w = w; st.h = hh; c.width = Math.round(w * dpr); c.height = Math.round(hh * dpr); c.style.height = hh + 'px'; ctx.setTransform(dpr, 0, 0, dpr, 0, 0); return true; };
        st.draw = () => { resize(); ctx.clearRect(0, 0, st.w, st.h); o.draw(ctx, st.w, st.h); };
        st.toLocal = (ev) => { const r = c.getBoundingClientRect(); return [(ev.clientX - r.left) / r.width * st.w, (ev.clientY - r.top) / r.height * st.h]; };
        if (window.ResizeObserver) new ResizeObserver(() => { if (resize()) st.draw(); }).observe(box);
        canvases.push(st); return st;
      }
      const redrawAll = () => canvases.forEach(cv => cv.draw());

      // ============ intro ============
      el.appendChild(h('div', { class: 'families-intro' },
        h('div', { class: 'prose' },
          h('p', { html: 'All four families can be autoregressive: they predict word <i>t</i>+1 from words 1..<i>t</i> (paper Fig 1). They differ in <b>what one forward pass computes</b>, and in <b>whether more passes help</b>.' }),
          h('ul', { class: 'families-list' },
            h('li', { html: '<b class="c-ar">AR Transformer</b>: context in, distribution out, one pass through a fixed stack of layers.' }),
            h('li', { html: '<b class="c-rnn">RNN</b>: a state that is updated once per incoming word.' }),
            h('li', { html: '<b class="c-diff">Diffusion Transformer</b>: start from noise and remove predicted noise over a fixed schedule. Each step outputs a noise vector.' }),
            h('li', { html: '<b class="c-ebt">EBT</b>: score a candidate with one number, the energy. Improve the candidate by gradient descent on that number.' })),
        ),
        h('figure', { class: 'families-fig1' },
          h('div', { class: 'paper-fig' }, h('img', { src: 'media/paper/fig01.png', alt: 'Paper Figure 1: AR Transformer outputs the next token, RNN chains states, Diffusion Transformer outputs Noise(x_t+1), EBT outputs Energy(x_t+1).' })),
          h('figcaption', {}, lib.badge('paper', 'Fig 1, p.2'), ' The paper’s sketch. Note the outputs: a token, a token, Noise(x̂), Energy(x̂).'))));

      // ============ Lab A: lanes ============
      const S = { prompt: 'medium', budget: 12, alpha: 2.5, M: 1, auto: true, tau: 0.7, seed: 1, k: 0, phase: 1, playing: false, stepping: false };
      let R = null;
      const PASS_DUR = reduced ? 0.01 : 0.8;

      function compute() {
        const P = PROMPTS[S.prompt], pi = PKEYS.indexOf(S.prompt), logp = P.p.map(Math.log);
        const ar = softmax(laneBelief(lib, 1, pi, logp)), rnn = softmax(laneBelief(lib, 2, pi, logp));
        const diff = runDiff(lib, S.prompt, S), ebt = runEBT(lib, S.prompt, S), hs = runRNN(lib, S.prompt);
        const maxQ = Math.max(...P.p);
        R = { P, ar, rnn, hs, diff, ebt, scale: Math.min(1, Math.max(0.32, maxQ * 1.35)), ticks: Math.max(1, diff.ticks, ebt.ticks) };
        // compute allocation for every prompt (Facet 1 summary)
        R.alloc = PKEYS.map(pk => { const e = runEBT(lib, pk, Object.assign({}, S, { auto: true })); const last = e.at(e.ticks); return { pk, ebt: last.passes, E: last.E, reason: e.cands[last.best].reason, diff: Math.min(S.budget, TD) }; });
      }

      // ----- controls -----
      const ctl = h('div', { class: 'card families-ctl' });
      const inst = h('div', { class: 'families-inst' });
      el.appendChild(h('div', { class: 'bench families-bench' }, ctl, inst));

      const promptSeg = lib.segmented({ label: 'Prompt difficulty', value: S.prompt, options: PKEYS.map(k => [k, PROMPTS[k].label]), onchange: v => { S.prompt = v; restart(); } });
      const promptLine = h('div', { class: 'families-prompt', 'aria-live': 'polite' });
      const budget = lib.slider({ id: 'fam-budget', label: 'Compute budget (forward passes for this word)', min: 1, max: BMAX, step: 1, value: S.budget, oninput: v => { S.budget = v; restart(false); } });
      const alpha = lib.slider({ id: 'fam-alpha', label: 'EBT step size α', min: 0.5, max: 6, step: 0.25, value: S.alpha, fmt: v => v.toFixed(2), oninput: v => { S.alpha = v; restart(false); } });
      const mSeg = lib.segmented({ label: 'EBT candidates M', value: S.M, options: [[1, 'M = 1'], [2, '2'], [4, '4']], onchange: v => { S.M = v; restart(false); } });
      const autoBtn = h('button', { type: 'button', class: 'btn', 'aria-pressed': 'true' }, 'Stop on its own: on');
      autoBtn.addEventListener('click', () => { S.auto = !S.auto; autoBtn.setAttribute('aria-pressed', String(S.auto)); autoBtn.textContent = 'Stop on its own: ' + (S.auto ? 'on' : 'off'); tauCtl.el.classList.toggle('families-dim', !S.auto); restart(false); });
      const tauCtl = lib.slider({ id: 'fam-tau', label: 'Good-enough energy τ', min: 0.3, max: 2.4, step: 0.05, value: S.tau, fmt: v => v.toFixed(2), oninput: v => { S.tau = v; restart(false); } });
      const btnStep = h('button', { type: 'button', class: 'btn' }, 'Step');
      const btnPlay = h('button', { type: 'button', class: 'btn primary' }, 'Play');
      const btnReset = h('button', { type: 'button', class: 'btn' }, 'Reset');
      const btnSeed = h('button', { type: 'button', class: 'btn', title: 'Draw a new random starting point for diffusion and EBT' }, 'New noise');
      const status = h('div', { class: 'families-status', 'aria-live': 'polite' });
      const hint = h('div', { class: 'families-hint', 'aria-live': 'polite' });

      const TRIES = [
        { t: 'Budget 16: who uses it?', s: { prompt: 'medium', budget: 16, M: 1, auto: false }, n: 'AR and RNN finish after 1 pass and cannot use the rest. Diffusion stops at the end of its 8-step schedule. Only the EBT keeps refining, and its energy flattens out as it converges.' },
        { t: 'Easy word: EBT stops early', s: { prompt: 'easy', budget: 12, M: 1, auto: true }, n: 'After a couple of steps the energy is already below τ, so the EBT judges its guess good and stops. Compare with the Hard prompt.' },
        { t: 'Hard word: EBT keeps thinking', s: { prompt: 'hard', budget: 12, M: 1, auto: true }, n: 'Many colors fit, so even the best guess has high energy. The EBT never gets below τ. It keeps stepping until the energy stops falling. High final energy = high uncertainty (Facet 2).' },
        { t: 'Self-verify with 4 candidates', s: { prompt: 'medium', budget: 16, M: 4, auto: false }, n: 'Four random starts share the budget (4 steps each). At every pass the lane shows the candidate with the lowest energy: this is Best-of-N selection by the model itself (Algorithm 2, p.7).' },
        { t: 'Too few passes', s: { prompt: 'medium', budget: 2, M: 1, auto: false }, n: 'With 2 passes the EBT is still far from its minimum. The AR model is cheaper here: one pass is enough for it. EBTs trade extra compute per prediction for the abilities on the right.' },
      ];
      const tryRow = h('div', { class: 'families-tries' }, TRIES.map(tr => { const b = h('button', { type: 'button', class: 'families-try' }, tr.t); b.addEventListener('click', () => applyTry(tr)); return b; }));
      function applyTry(tr) {
        Object.assign(S, tr.s); promptSeg.set(S.prompt); budget.set(S.budget); mSeg.set(S.M);
        autoBtn.setAttribute('aria-pressed', String(S.auto)); autoBtn.textContent = 'Stop on its own: ' + (S.auto ? 'on' : 'off'); tauCtl.el.classList.toggle('families-dim', !S.auto);
        hint.innerHTML = '<b>What to notice.</b> ' + tr.n; restart(true, true);
      }

      ctl.append(
        h('h4', {}, 'Prompt'), promptSeg.el, promptLine,
        h('div', { class: 'families-gap' }), budget.el,
        h('h4', { class: 'families-h4' }, 'EBT thinking'), alpha.el,
        h('div', { class: 'ctl' }, h('span', { class: 'families-lbl' }, 'Candidates (self-verification, Best-of-M)'), mSeg.el),
        h('div', { class: 'row families-auto' }, autoBtn), tauCtl.el,
        h('p', { class: 'families-small' }, 'Stop rule: stop when energy ≤ τ (good enough) or when a step lowers it by less than ' + EPS_SETTLE + ' (settled).'),
        h('div', { class: 'row families-transport' }, btnStep, btnPlay, btnReset, btnSeed), status,
        h('h4', { class: 'families-h4' }, 'Try this'), tryRow, hint);

      // ----- instrument: badges + lanes -----
      inst.appendChild(h('div', { class: 'row families-badges' }, lib.badge('paper', 'Fig 1, Table 1'), lib.badge('ext', 'illustration')));
      const lanesBox = h('div', { class: 'families-lanes' }); inst.appendChild(lanesBox);
      const laneEls = {};
      LANES.forEach((L, li) => {
        const card = h('div', { class: 'families-lane', style: `--lc: var(--${L.color})`, 'data-lane': L.id });
        const chips = { ar: ['1 pass per word', 'output: logits'], rnn: ['1 state update per word', 'output: logits'], diff: ['fixed 8-step schedule', 'output per step: noise vector'], ebt: ['any number of steps', 'output per step: energy E'] }[L.id];
        card.appendChild(h('div', { class: 'families-lane-head' }, h('b', {}, L.name), chips.map(c => h('span', { class: 'families-chip' }, c))));
        const cv = autoCanvas(card, { height: w => w >= 620 ? 184 : 350, label: L.name + ' lane: mechanism, passes used, and current next-word distribution', draw: (ctx, w, hh) => drawLane(L, ctx, w, hh) });
        const ro = h('div', { class: 'readout families-ro' }); card.appendChild(ro);
        card.addEventListener('mouseenter', () => highlightRow(L.table, true)); card.addEventListener('mouseleave', () => highlightRow(L.table, false));
        lanesBox.appendChild(card); laneEls[L.id] = { card, cv, ro };
      });

      // charts
      const chartRow = h('div', { class: 'families-charts' }); inst.appendChild(chartRow);
      const qCard = h('div', { class: 'card families-chartcard' }, h('h4', {}, 'How good is each guess? (we can see this, the models cannot)'));
      const eCard = h('div', { class: 'card families-chartcard' }, h('h4', {}, 'What the EBT sees: its own energy per step'));
      chartRow.append(qCard, eCard);
      const qCv = autoCanvas(qCard, { height: w => w >= 420 ? 230 : 210, label: 'Cross-entropy to the true next-word distribution versus forward passes used, for each model', draw: drawQuality });
      qCard.appendChild(h('div', { class: 'legend' }, [['AR (1 pass, then idle)', 'ar'], ['RNN', 'rnn'], ['Diffusion', 'diff'], ['EBT', 'ebt'], ['floor', 'truth']].map(([t, c]) => h('span', {}, h('i', { style: `background:var(--${c})` }), t))));
      qCard.appendChild(h('p', { class: 'families-small' }, 'Cross-entropy between the true next-word distribution and each model’s current guess, against forward passes spent. Lower is better. The dashed mint line is the floor: the entropy of the truth.'));
      const eCv = autoCanvas(eCard, { height: w => w >= 420 ? 230 : 210, label: 'EBT energy per optimization step for each candidate', draw: drawEnergy });
      eCard.appendChild(h('p', { class: 'families-small' }, 'One line per candidate; bold = current lowest energy. AR, RNN and diffusion produce no such number, so there is nothing to plot for them.'));

      // compute allocation (Facet 1) + honesty note
      const allocCard = h('div', { class: 'card families-alloc' });
      inst.appendChild(allocCard);
      inst.appendChild(lib.callout('', 'What is real in this lab', 'The <b>diffusion</b> lane runs a DDIM sampler with the exact denoiser for a Gaussian target around its belief. The <b>EBT</b> lane runs gradient descent on E(ŷ) = cross-entropy(belief, softmax(ŷ)), whose true gradient is softmax(ŷ) − belief; the lowest reachable energy is the entropy of its belief. <b>AR and RNN</b> internals are drawn schematically (the RNN state is a random-weight tanh RNN). Every model gets the same size of knowledge error, so this lab is about mechanism, not accuracy. The true next-word distributions are hand-set. One pass = one network evaluation (the paper counts one function evaluation per EBT optimization step, p.8).'));

      // ---------- lane drawing ----------
      function laneLayout(w, hh) {
        if (w >= 620) { const mw = Math.round(w * 0.55); return { wide: true, mech: { x: 0, y: 0, w: mw - 18, h: hh }, bars: { x: mw, y: 0, w: w - mw, h: hh } }; }
        return { wide: false, mech: { x: 0, y: 0, w, h: 176 }, bars: { x: 0, y: 186, w, h: hh - 186 } };
      }
      const laneTicks = (id) => id === 'ar' || id === 'rnn' ? 1 : id === 'diff' ? R.diff.ticks : R.ebt.ticks;
      // returns {kl, act, f}: lane-local completed tick, whether a pass is animating, eased progress of that pass
      function laneTime(id) {
        const lt = laneTicks(id);
        if (S.k > lt) return { kl: lt, act: false, f: 1 };
        const act = S.phase < 1 && S.k >= 1;
        return { kl: S.k, act, f: act ? lib.ease(S.phase) : 1 };
      }
      const qOf = (id, k) => {
        if (id === 'ar') return k >= 1 ? R.ar : null;
        if (id === 'rnn') return k >= 1 ? R.rnn : null;
        if (id === 'diff') return softmax(R.diff.zs[Math.min(k, R.diff.ticks)]);
        return softmax(R.ebt.at(k).y);
      };
      function curQ(id) {
        const t = laneTime(id), q1 = qOf(id, t.kl);
        if (!t.act) return q1;
        const q0 = qOf(id, t.kl - 1);
        if (!q0) return t.f > 0.85 ? q1 : null;
        return q0.map((v, i) => lib.lerp(v, q1[i], t.f));
      }
      function slots(ctx, x, y, w, used, active, unusedMark, color) {
        const B = S.budget, gap = 3, sw = Math.min(16, (w - (B - 1) * gap) / B);
        for (let i = 0; i < B; i++) {
          const sx = x + i * (sw + gap); ctx.beginPath(); ctx.roundRect(sx, y, sw, 14, 3);
          if (i < used) { ctx.fillStyle = color; ctx.fill(); }
          else if (i === used && active) { ctx.fillStyle = lib.rgba(color, 0.35 + 0.3 * Math.sin(performance.now() / 120)); ctx.fill(); }
          else { ctx.strokeStyle = C.rule; ctx.lineWidth = 1.2; ctx.stroke(); if (i >= unusedMark) { ctx.strokeStyle = C.faint; ctx.beginPath(); ctx.moveTo(sx + 4, y + 4); ctx.lineTo(sx + sw - 4, y + 10); ctx.moveTo(sx + sw - 4, y + 4); ctx.lineTo(sx + 4, y + 10); ctx.stroke(); } }
        }
      }
      function box(ctx, x, y, w, hh, o = {}) { ctx.beginPath(); ctx.roundRect(x, y, w, hh, 7); ctx.fillStyle = o.fill || C.panel2; ctx.fill(); ctx.strokeStyle = o.stroke || C.rule; ctx.lineWidth = o.lw || 1.2; ctx.stroke(); }
      function strip(ctx, x, y, vals, color, cw = 9, ch = 12) { // horizontal strip of cells; vals in [0,1]
        vals.forEach((v, i) => { ctx.fillStyle = lib.rgba(color, 0.12 + 0.88 * lib.clamp(v)); ctx.fillRect(x + i * (cw + 2), y, cw, ch); });
      }
      function drawBars(ctx, z, q, color, note) {
        const P = R.P, scale = R.scale;
        T(ctx, 'p(next word)', z.x, z.y + 2, { size: 12.5, kind: 'mono', color: C.muted });
        T(ctx, '| = truth', z.x + z.w - 2, z.y + 2, { size: 12.5, kind: 'mono', color: C.truth, align: 'right' });
        const top = z.y + 22, rowH = Math.min(18.5, (z.h - 26) / P.toks.length), labW = 66, valW = 42, bw = Math.max(40, z.w - labW - valW);
        P.toks.forEach((tk, i) => {
          const yy = top + i * rowH;
          T(ctx, tk, z.x + labW - 8, yy + rowH / 2, { size: 13, kind: 'mono', align: 'right', baseline: 'middle', color: C.ink });
          ctx.fillStyle = 'rgba(233,238,246,0.05)'; ctx.fillRect(z.x + labW, yy + 3, bw, rowH - 6);
          if (q) {
            const v = Math.min(1, q[i] / scale); ctx.fillStyle = color; ctx.fillRect(z.x + labW, yy + 3, v * bw, rowH - 6);
            if (q[i] > scale) { ctx.fillStyle = C.ink; ctx.beginPath(); ctx.moveTo(z.x + labW + bw - 6, yy + 4); ctx.lineTo(z.x + labW + bw, yy + rowH / 2); ctx.lineTo(z.x + labW + bw - 6, yy + rowH - 4); ctx.fill(); }
            T(ctx, q[i].toFixed(2), z.x + labW + bw + 6, yy + rowH / 2, { size: 12.5, kind: 'mono', baseline: 'middle', color: C.muted });
          }
          const tx = z.x + labW + Math.min(1, P.p[i] / scale) * bw; ctx.strokeStyle = C.truth; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(tx, yy + 1); ctx.lineTo(tx, yy + rowH - 1); ctx.stroke();
        });
        if (!q) T(ctx, note || 'no guess yet: press Step or Play', z.x + labW + 10, top + rowH * 3.6, { size: 13, color: C.muted, maxWidth: bw });
      }
      function caption(ctx, z, s, color) { T(ctx, s, z.x, z.y + z.h - 36, { size: 13, color: color || C.muted, maxWidth: z.w, lh: 1.28 }); }

      function drawLane(L, ctx, w, hh) {
        if (!R) return;
        const lay = laneLayout(w, hh), m = lay.mech, cc = col(L.color), t = laneTime(L.id);
        const dy = m.y + 4, dh = m.h - 64; // diagram area
        const slotY = m.y + m.h - 56;
        if (!lay.wide) { ctx.strokeStyle = C.rule; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(0, lay.bars.y - 6); ctx.lineTo(w, lay.bars.y - 6); ctx.stroke(); }
        if (L.id === 'ar') {
          const cw = Math.min(110, m.w * 0.3), lw = Math.min(100, m.w * 0.27), ow = Math.min(92, m.w * 0.24);
          const by = dy + 10, bh = Math.min(66, dh - 26);
          box(ctx, m.x, by, cw, bh); T(ctx, R.P.ctx.join(' ') + ' ___', m.x + 8, by + 8, { size: 13, kind: 'mono', maxWidth: cw - 14, lh: 1.25 });
          const lx = m.x + cw + 22; lib.arrow(ctx, m.x + cw + 4, by + bh / 2, lx - 4, by + bh / 2, { color: C.faint, width: 1.6, head: 7 });
          const nL = 6, lwid = (lw - (nL - 1) * 5) / nL, lit = t.kl >= 1 ? (t.act ? Math.floor(t.f * (nL + 0.99)) : nL) : 0;
          for (let i = 0; i < nL; i++) { ctx.beginPath(); ctx.roundRect(lx + i * (lwid + 5), by, lwid, bh, 3); ctx.fillStyle = i < lit ? lib.rgba(cc, 0.85) : C.panel2; ctx.fill(); ctx.strokeStyle = C.rule; ctx.lineWidth = 1; ctx.stroke(); }
          T(ctx, '6 fixed layers', lx + lw / 2, by + bh + 5, { size: 12.5, color: C.muted, align: 'center' });
          const ox = lx + lw + 22; lib.arrow(ctx, lx + lw + 4, by + bh / 2, ox - 4, by + bh / 2, { color: C.faint, width: 1.6, head: 7 });
          box(ctx, ox, by + bh / 2 - 22, ow, 44, { stroke: t.kl >= 1 && !t.act ? cc : C.rule });
          T(ctx, 'softmax', ox + ow / 2, by + bh / 2 - 15, { size: 12.5, kind: 'mono', color: C.muted, align: 'center' });
          T(ctx, 'p(·)', ox + ow / 2, by + bh / 2 + 1, { size: 14, kind: 'mono', color: t.kl >= 1 ? cc : C.faint, align: 'center' });
          slots(ctx, m.x, slotY, m.w, t.kl >= 1 ? 1 : 0, t.act, 1, cc);
          caption(ctx, m, S.budget > 1 ? `${t.kl >= 1 ? 1 : 0} of ${S.budget} passes used. Rerunning on the same input gives the same output.` : 'Budget of 1 pass: exactly what it needs.');
        } else if (L.id === 'rnn') {
          const n = R.P.ctx.length, by = dy + 22, bw = 30, bh = Math.min(58, dh - 30);
          ctx.save(); ctx.font = lib.font(12.5, 'mono'); const longest = Math.max(...R.P.ctx.map(s => ctx.measureText(s).width)); ctx.restore();
          const sp = Math.min(Math.max(48, longest + 12), (m.w - 96) / n);
          R.P.ctx.forEach((word, i) => {
            const x = m.x + i * sp + (sp - bw) / 2, last = i === n - 1;
            const hv = last && t.kl < 1 ? (i > 0 ? R.hs[i - 1] : R.hs[0].map(() => 0)) : R.hs[i];
            const lit = last && t.kl >= 1;
            const glow = last && t.act ? 0.5 + 0.5 * Math.sin(performance.now() / 110) : 0;
            box(ctx, x, by, bw, bh, { stroke: lit || glow ? cc : C.rule, lw: lit ? 1.8 : 1.2 });
            hv.forEach((v, k) => { ctx.fillStyle = v >= 0 ? lib.rgba(cc, 0.15 + 0.85 * Math.abs(v)) : lib.rgba(C.ar, 0.12 + 0.6 * Math.abs(v)); ctx.fillRect(x + 6, by + 5 + k * ((bh - 10) / 8), bw - 12, (bh - 10) / 8 - 1.5); });
            T(ctx, word, x + bw / 2, by - 18, { size: 12.5, kind: 'mono', align: 'center', color: last ? C.ink : C.muted });
            if (i < n - 1) lib.arrow(ctx, x + bw + 2, by + bh / 2, x + sp - 2, by + bh / 2, { color: C.faint, width: 1.5, head: 6 });
          });
          const ex = m.x + n * sp - (sp - bw) / 2 + 4; lib.arrow(ctx, ex, by + bh / 2, ex + 26, by + bh / 2, { color: t.kl >= 1 ? cc : C.faint, width: 1.6, head: 7 });
          T(ctx, 'p(·)', ex + 32, by + bh / 2, { size: 14, kind: 'mono', baseline: 'middle', color: t.kl >= 1 ? cc : C.faint });
          T(ctx, 'state h, updated once per word', m.x, by + bh + 6, { size: 12.5, color: C.muted });
          slots(ctx, m.x, slotY, m.w, t.kl >= 1 ? 1 : 0, t.act, 1, cc);
          caption(ctx, m, S.budget > 1 ? `${t.kl >= 1 ? 1 : 0} of ${S.budget} passes used. No new word arrives, so the state cannot change.` : 'Budget of 1 pass: one state update.');
        } else if (L.id === 'diff') {
          const D = R.diff, rw = Math.min(14, (m.w * 0.48 - 7 * 6) / 8), gap = 6, hmax = Math.min(64, dh - 34), base = dy + 22 + hmax;
          T(ctx, 'noise level, fixed schedule', m.x, dy, { size: 12.5, color: C.muted });
          const used = new Set(D.rungs.slice(1)), curR = t.kl >= 1 ? D.rungs[t.kl] : null;
          for (let r = TD; r >= 1; r--) {
            const i = TD - r, x = m.x + i * (rw + gap), sig = Math.sqrt(1 - abar(r / TD)), hgt = Math.max(3, sig * hmax);
            const done = t.kl >= 1 && D.rungs.slice(1, t.kl + 1).includes(r) && !(t.act && r === curR);
            ctx.fillStyle = done ? lib.rgba(cc, 0.9) : used.has(r) ? lib.rgba(cc, 0.22) : 'rgba(147,161,184,0.12)';
            ctx.fillRect(x, base - hgt, rw, hgt);
            if (r === curR && t.act) { ctx.strokeStyle = C.ink; ctx.lineWidth = 1.5; ctx.strokeRect(x - 1.5, base - hgt - 1.5, rw + 3, hgt + 3); }
          }
          T(ctx, 'noisy', m.x, base + 4, { size: 12, color: C.faint }); T(ctx, 'clean', m.x + 8 * (rw + gap) - gap, base + 4, { size: 12, color: C.faint, align: 'right' });
          const ex = m.x + 8 * (rw + gap) + 18, eps = t.kl >= 1 ? D.eps[t.kl] : null;
          T(ctx, 'step output ε̂:', ex, dy + 14, { size: 12.5, color: C.muted });
          if (eps) { eps.forEach((v, i) => { ctx.fillStyle = v >= 0 ? lib.rgba(cc, 0.15 + 0.85 * Math.min(1, Math.abs(v) / 1.8)) : lib.rgba(C.ar, 0.15 + 0.85 * Math.min(1, Math.abs(v) / 1.8)); ctx.fillRect(ex + i * 13, dy + 36, 11, 16); }); }
          else { ctx.strokeStyle = C.rule; ctx.strokeRect(ex, dy + 36, 8 * 13 - 2, 16); }
          T(ctx, '8 numbers (a direction).', ex, dy + 58, { size: 12.5, color: C.muted, maxWidth: m.w - (ex - m.x) });
          T(ctx, 'No score for the guess.', ex, dy + 76, { size: 12.5, color: C.diff, maxWidth: m.w - (ex - m.x) });
          slots(ctx, m.x, slotY, m.w, t.kl, t.act, D.T, cc);
          const capS = S.budget > TD ? `${t.kl} of ${S.budget} passes used. The schedule ends at step ${TD}; the rest go unused.` : S.budget < TD ? `${t.kl} of ${S.budget} passes used. Schedule squeezed from ${TD} to ${D.T} steps.` : `${t.kl} of ${S.budget} passes used: one per noise level.`;
          caption(ctx, m, capS);
        } else {
          const Eb = R.ebt, st = Eb.at(t.kl), prev = t.kl >= 1 ? Eb.at(t.kl - 1) : st;
          const by = dy + 12, bh = 46, yw = Math.min(110, m.w * 0.28), ew = Math.min(78, m.w * 0.2);
          // candidate stack
          for (let j = Math.min(Eb.Meff, 4) - 1; j >= 1; j--) box(ctx, m.x + j * 4, by - j * 4, yw, bh, { fill: C.panel });
          box(ctx, m.x, by, yw, bh, { stroke: cc });
          ctx.save(); ctx.font = lib.font(12, 'mono'); let yl = Eb.Meff > 1 ? `ŷ best of ${Eb.Meff}` : 'ŷ (guess)'; if (ctx.measureText(yl).width > yw - 12) yl = Eb.Meff > 1 ? `ŷ ×${Eb.Meff}` : 'ŷ'; ctx.restore();
          T(ctx, yl, m.x + 7, by + 5, { size: 12, kind: 'mono', color: C.muted });
          const q = curQ('ebt') || softmax(st.y), mq = Math.max(...q);
          strip(ctx, m.x + 8, by + 24, q.map(v => v / mq), cc, Math.max(6, (yw - 16 - 14) / 8), 14);
          const exb = m.x + yw + 20; lib.arrow(ctx, m.x + yw + 3, by + bh / 2, exb - 3, by + bh / 2, { color: C.faint, width: 1.6, head: 7 });
          box(ctx, exb, by, ew, bh, { stroke: lib.rgba(cc, 0.7) }); T(ctx, 'EBT', exb + ew / 2, by + 7, { size: 14, weight: 700, align: 'center', color: cc }); T(ctx, '(x, ŷ)', exb + ew / 2, by + 26, { size: 12, kind: 'mono', align: 'center', color: C.muted });
          const exE = exb + ew + 20; lib.arrow(ctx, exb + ew + 3, by + bh / 2, exE - 3, by + bh / 2, { color: C.faint, width: 1.6, head: 7 });
          const Eshow = t.act ? lib.lerp(prev.E, st.E, t.f) : st.E;
          T(ctx, 'E = ' + Eshow.toFixed(3), exE, by + 6, { size: 18, kind: 'mono', weight: 600, color: cc });
          if (t.kl >= 1) T(ctx, 'ΔE ' + (st.E - prev.E >= 0 ? '+' : '') + (st.E - prev.E).toFixed(3), exE, by + 30, { size: 12.5, kind: 'mono', color: C.muted });
          else T(ctx, 'random start', exE, by + 30, { size: 12.5, color: C.muted });
          // return arc: E -> gradient -> y
          const ay = by + bh + 6, x1 = exE + 30, x0 = m.x + yw / 2;
          ctx.strokeStyle = lib.rgba(cc, 0.55); ctx.lineWidth = 1.6; ctx.setLineDash([5, 4]); ctx.beginPath(); ctx.moveTo(x1, ay); ctx.bezierCurveTo(x1, ay + 22, x0, ay + 22, x0, ay + 2); ctx.stroke(); ctx.setLineDash([]);
          ctx.fillStyle = lib.rgba(cc, 0.8); ctx.beginPath(); ctx.moveTo(x0, ay); ctx.lineTo(x0 - 5, ay + 8); ctx.lineTo(x0 + 5, ay + 8); ctx.fill();
          T(ctx, 'ŷ ← ŷ − α∇ŷE', (x0 + x1) / 2, ay + 20, { size: 12.5, kind: 'mono', color: C.ink, align: 'center' });
          if (t.act) { // travelling dot round the loop
            const f = t.f; let px, py;
            if (f < 0.5) { const u = f / 0.5; px = lib.lerp(m.x + yw, exE, u); py = by + bh / 2; }
            else { const u = (f - 0.5) / 0.5, a = 1 - u; px = a * a * a * x1 + 3 * a * a * u * x1 + 3 * a * u * u * x0 + u * u * u * x0; py = a * a * a * ay + 3 * a * a * u * (ay + 22) + 3 * a * u * u * (ay + 22) + u * u * u * (ay + 2); }
            lib.dot(ctx, px, py, 4.5, C.ink);
          }
          const passes = t.act ? Eb.at(t.kl - 1).passes : st.passes, done = t.kl >= Eb.ticks && !t.act;
          slots(ctx, m.x, slotY, m.w, passes, t.act, done ? passes : BMAX + 1, cc);
          let capS;
          const bestC = Eb.cands[st.best];
          if (t.kl === 0) capS = 'Starts from random noise. Each pass: energy + gradient, then one step downhill.';
          else if (!done) capS = `Thinking: ${st.passes} pass${st.passes > 1 ? 'es' : ''} so far` + (Eb.Meff > 1 ? ` (${Eb.Meff} candidates in parallel).` : '.');
          else if (bestC.reason === 'confident' && S.auto) capS = `Stopped after ${st.passes} of ${S.budget}: energy ≤ τ, judged good enough.`;
          else if (bestC.reason === 'settled' && S.auto) capS = `Stopped after ${st.passes} of ${S.budget}: energy stopped falling.`;
          else capS = `Used ${st.passes} of ${S.budget} passes` + (Eb.Meff > 1 ? ` (${Eb.Meff} × ${Eb.cap} steps), kept the lowest energy.` : '.');
          caption(ctx, m, capS, done ? C.ink : C.muted);
        }
        drawBars(ctx, lay.bars, curQ(L.id), lib.rgba(cc, 0.85), L.id === 'ar' || L.id === 'rnn' ? 'no guess until its single pass runs' : null);
      }

      // ---------- charts ----------
      function chartFrame(ctx, w, hh, ylim, yt, ylabel, xlabel) {
        const pad = { l: 60, r: 12, t: 12, b: 42 };
        const ax = lib.axes(ctx, { x: pad.l, y: pad.t, w: w - pad.l - pad.r, h: hh - pad.t - pad.b, xlim: [0, BMAX], ylim, xticks: [0, 4, 8, 12, 16], yticks: yt, size: 12, yfmt: v => v.toFixed(1) });
        T(ctx, xlabel, pad.l + (w - pad.l - pad.r) / 2, hh - 17, { size: 12.5, color: C.muted, align: 'center' });
        ctx.save(); ctx.translate(11, pad.t + (hh - pad.t - pad.b) / 2); ctx.rotate(-Math.PI / 2); T(ctx, ylabel, 0, 0, { size: 12.5, color: C.muted, align: 'center', baseline: 'middle' }); ctx.restore();
        return ax;
      }
      const niceTicks = (lo, hi) => { const span = hi - lo, st = span > 3 ? 1 : span > 1.4 ? 0.5 : 0.25; const out = []; for (let v = Math.ceil(lo / st) * st; v <= hi + 1e-9; v += st) out.push(+v.toFixed(2)); return out; };
      const hline = (ctx, ax, y, color, dash) => { ctx.save(); ctx.strokeStyle = color; ctx.lineWidth = 1.5; ctx.setLineDash(dash); ctx.beginPath(); ctx.moveTo(ax.X(0), ax.Y(y)); ctx.lineTo(ax.X(BMAX), ax.Y(y)); ctx.stroke(); ctx.restore(); };
      function drawQuality(ctx, w, hh) {
        if (!R) return;
        const p = R.P.p, Hs = entropy(p), Eb = R.ebt, D = R.diff;
        const ebtPts = []; for (let k = 0; k <= Eb.ticks; k++) { const a = Eb.at(k); ebtPts.push([a.passes, ce(p, softmax(a.y))]); }
        const difPts = D.zs.map((z, j) => [j, ce(p, softmax(z))]);
        const all = [...ebtPts, ...difPts].map(q => q[1]).concat([ce(p, R.ar), ce(p, R.rnn)]);
        const lo = Math.max(0, Hs - 0.15), hi = Math.max(...all) * 1.05;
        const ax = chartFrame(ctx, w, hh, [lo, hi], niceTicks(lo, hi), 'cross-entropy (nats)', 'forward passes used');
        hline(ctx, ax, Hs, C.truth, [5, 5]);
        ctx.save(); ctx.strokeStyle = C.faint; ctx.setLineDash([2, 4]); ctx.beginPath(); ctx.moveTo(ax.X(S.budget), ax.Y(hi)); ctx.lineTo(ax.X(S.budget), ax.Y(lo)); ctx.stroke(); ctx.restore();
        T(ctx, 'budget', ax.X(S.budget) + (S.budget > 12 ? -4 : 4), ax.Y(hi) + 2, { size: 12, color: C.faint, align: S.budget > 12 ? 'right' : 'left' });
        const kE = laneTime('ebt').kl, kD = laneTime('diff').kl, k1 = laneTime('ar').kl;
        if (k1 >= 1) { [['ar', R.ar, [2, 4]], ['rnn', R.rnn, [7, 4]]].forEach(([id, q, dash]) => { const y = ce(p, q); ctx.save(); ctx.strokeStyle = lib.rgba(C[id], 0.7); ctx.lineWidth = 1.8; ctx.setLineDash(dash); ctx.beginPath(); ctx.moveTo(ax.X(1), ax.Y(y)); ctx.lineTo(ax.X(S.budget), ax.Y(y)); ctx.stroke(); ctx.restore(); lib.dot(ctx, ax.X(1), ax.Y(y), 4.5, C[id]); }); }
        lib.plot(ctx, ax, difPts.slice(0, kD + 1), { color: C.diff, width: 2.2, markers: 3 });
        lib.plot(ctx, ax, ebtPts.slice(0, kE + 1), { color: C.ebt, width: 2.6, markers: 3.2 });
      }
      function drawEnergy(ctx, w, hh) {
        if (!R) return;
        const Eb = R.ebt, kE = laneTime('ebt').kl, st = Eb.at(kE);
        const allE = Eb.cands.flatMap(c => c.Es.slice(0, c.stop + 1));
        const minL = Math.min(Eb.floor, S.auto ? S.tau : Eb.floor), hi = Math.max(...allE, S.auto ? S.tau : 0) * 1.05, lo = Math.max(0, minL - Math.max(0.15, 0.16 * (hi - minL)));
        const ax = chartFrame(ctx, w, hh, [lo, hi], niceTicks(lo, hi), 'energy E', Eb.Meff > 1 ? 'optimization step (each candidate)' : 'optimization step = forward pass');
        hline(ctx, ax, Eb.floor, lib.rgba(C.truth, 0.85), [5, 5]);
        const close = S.auto && Math.abs(ax.Y(S.tau) - ax.Y(Eb.floor)) < 20; // when the lines are close: lower label below its line, upper label above
        T(ctx, 'floor: entropy of belief', ax.X(BMAX) - 2, ax.Y(Eb.floor) + (close && Eb.floor < S.tau ? 4 : -17), { size: 12, color: C.truth, align: 'right' });
        if (S.auto) { hline(ctx, ax, S.tau, lib.rgba(C.ebt, 0.75), [5, 5]); T(ctx, 'τ (good enough)', ax.X(0) + 4, ax.Y(S.tau) + (close && S.tau < Eb.floor ? 4 : -17), { size: 12, color: C.ebt }); }
        Eb.cands.forEach((c, j) => { const n = Math.min(kE, c.stop); const pts = c.Es.slice(0, n + 1).map((e, i) => [i, e]); lib.plot(ctx, ax, pts, { color: j === st.best ? C.ebt : lib.rgba(C.ebt, 0.35), width: j === st.best ? 2.6 : 1.4, markers: j === st.best ? 3 : 0 }); });
        if (kE === 0) T(ctx, 'press Play: each step adds a point', ax.X(0.6), ax.Y(hi) + 4, { size: 12.5, color: C.muted });
      }

      // ---------- readouts, allocation table ----------
      function updateReadouts() {
        const p = R.P.p, fmt = (v) => v.toFixed(2);
        LANES.forEach(L => {
          const t = laneTime(L.id), ro = laneEls[L.id].ro;
          if (L.id === 'ar' || L.id === 'rnn') {
            const q = t.kl >= 1 ? R[L.id] : null;
            ro.innerHTML = `<span>passes <b>${q ? 1 : 0} / ${S.budget}</b></span><span>self-check <b>${q ? 'softmax entropy ' + fmt(entropy(q)) + ' nats' : '–'}</b> <i class="families-note">(text only; no softmax for continuous outputs)</i></span><span>CE vs truth <b>${q ? fmt(ce(p, q)) : '–'}</b></span>`;
          } else if (L.id === 'diff') {
            const z = R.diff.zs[t.kl];
            ro.innerHTML = `<span>passes <b>${t.kl} / ${S.budget}</b></span><span>noise level <b>${t.kl >= R.diff.T ? 'clean' : 'σ = ' + fmt(Math.sqrt(1 - abar((R.diff.T - t.kl) / R.diff.T)))}</b></span><span>self-check <b>none</b></span><span>CE vs truth <b>${fmt(ce(p, softmax(z)))}</b></span>`;
          } else {
            const st = R.ebt.at(t.kl);
            ro.innerHTML = `<span>passes <b>${st.passes} / ${S.budget}</b></span><span>self-check <b>energy ${st.E.toFixed(3)}</b></span><span>CE vs truth <b>${fmt(ce(p, softmax(st.y)))}</b></span>`;
          }
        });
        const k = S.k, tk = R.ticks;
        status.textContent = k === 0 ? 'Ready: no passes run yet.' : k >= tk && S.phase >= 1 ? `Done after ${tk} tick${tk > 1 ? 's' : ''}. Press Reset, then Play to watch again.` : `Tick ${k} of ${tk}: every lane that can still use a pass runs one.`;
        btnPlay.textContent = S.playing ? 'Pause' : (k >= tk && S.phase >= 1 ? 'Replay' : 'Play');
        btnStep.disabled = k >= tk && S.phase >= 1 && !S.playing;
        
        promptLine.innerHTML = `<span class="mono">${R.P.ctx.join(' ')} <u>&nbsp;?&nbsp;</u></span>`;
      }
      function renderAlloc() {
        const rows = [['AR Transformer', 'ar', PKEYS.map(() => '1')], ['RNN', 'rnn', PKEYS.map(() => '1')], ['Diffusion', 'diff', PKEYS.map(() => String(Math.min(S.budget, TD)))], ['EBT (stop rule on)', 'ebt', R.alloc.map(a => `${a.ebt}<span class="families-sub">E ${a.E.toFixed(2)}</span>`)]];
        allocCard.innerHTML = '';
        allocCard.append(h('h4', {}, 'Facet 1 in one table: passes each model spends per word'),
          h('div', { class: 'tbl' }, h('table', {}, h('thead', {}, h('tr', {}, h('th', {}, 'Model'), PKEYS.map(pk => h('th', { class: pk === S.prompt ? 'families-cur' : '' }, PROMPTS[pk].label, h('span', { class: 'families-sub families-thsub' }, '…' + PROMPTS[pk].ctx.slice(-2).join(' ')))))),
            h('tbody', {}, rows.map(([n, c, vals]) => h('tr', {}, h('td', { html: `<span class="families-dot" style="background:var(--${c})"></span>${n}` }), vals.map((v, i) => h('td', { class: 'num' + (PKEYS[i] === S.prompt ? ' families-cur' : ''), html: v }))))))),
          h('p', { class: 'families-small' }, `Same settings as above (budget ${S.budget}, α ${S.alpha}, M ${S.M}, τ ${S.tau}). Only the EBT row changes with difficulty: it reads its own energy and decides. Its final energy (E) is also an uncertainty readout: low for “time”, high for a favorite color.`));
      }

      // ---------- animation ----------
      function render() { if (!R) return; LANES.forEach(L => laneEls[L.id].cv.draw()); qCv.draw(); eCv.draw(); updateReadouts(); }
      const loop = lib.loop((dt) => {
        if (!S.playing && !S.stepping) return false;
        S.phase += dt / PASS_DUR;
        if (S.phase >= 1) {
          S.phase = 1;
          if (S.stepping) { S.stepping = false; render(); return false; }
          if (S.k < R.ticks) { S.k++; S.phase = 0; }
          else { S.playing = false; render(); return false; }
        }
        render(); return true;
      });
      function play() { if (S.k >= R.ticks && S.phase >= 1) { S.k = 0; S.phase = 1; } S.playing = true; S.stepping = false; if (S.phase >= 1 && S.k < R.ticks) { S.k++; S.phase = 0; } loop.start(); render(); }
      function pause() { S.playing = false; S.stepping = false; S.phase = 1; render(); }
      function step() { S.playing = false; if (S.k >= R.ticks) { S.phase = 1; S.stepping = false; render(); return; } if (S.phase < 1) S.phase = 1; S.k++; S.phase = 0; S.stepping = true; loop.start(); }
      function restart(resetK = true, autoplay = false) {
        compute(); renderAlloc();
        if (resetK) { S.k = 0; S.phase = 1; } else { S.k = Math.min(S.k, R.ticks); S.phase = 1; }
        if (!resetK && !S.playing) { /* keep position */ }
        if (autoplay) play(); else render();
      }
      btnPlay.addEventListener('click', () => S.playing ? pause() : play());
      btnStep.addEventListener('click', step);
      btnReset.addEventListener('click', () => { S.playing = false; S.stepping = false; S.k = 0; S.phase = 1; render(); });
      btnSeed.addEventListener('click', () => { S.seed++; restart(true); });

      // start in an informative state: everything computed and run to the end; replay once when first visible
      compute(); renderAlloc(); S.k = R.ticks; S.phase = 1;
      hint.innerHTML = '<b>Start here.</b> Press Play to watch one word being predicted, or pick a “Try this” button.';

      // ============ Lab B: numbers vs arrows ============
      const bCard = h('div', { class: 'card families-lab2' }); el.appendChild(bCard);
      bCard.append(h('div', { class: 'row families-badges' }, lib.badge('ext', 'illustration of Sec 6.4 and E.2'), lib.badge('paper', 'claim: p.16, p.36–37')),
        h('h3', {}, 'Why a number beats an arrow: ranking two guesses'),
        h('div', { class: 'prose' },
          h('p', { html: 'The paper says diffusion models (and recurrent-depth RNNs) learn the <b>gradient</b> of an energy, while EBTs learn the <b>energy itself</b> (Sec 6.4, E.1). Why does that matter? Try to answer one question with each: <i>which of two candidates, A or B, is more likely?</i>' }),
          h('p', { html: 'With an energy you evaluate E(A) and E(B): two passes. With only arrows you must add up arrows along a route from A to B, one evaluation per segment. And a network that outputs a vector is not forced to be the gradient of anything. Give it a small <b>swirl</b> (rotational error) and two routes from A to B give two different answers.' })));
      const bS = { A: [-0.55, -1.15], B: [0.7, 1.25], c: 0.35, drag: null };
      const bCtl = h('div', { class: 'families-lab2-ctl' });
      const swirl = lib.slider({ id: 'fam-swirl', label: 'Swirl in the learned arrows (non-gradient part)', min: 0, max: 2, step: 0.05, value: bS.c, fmt: v => v.toFixed(2), oninput: v => { bS.c = v; bCv.draw(); bRead(); } });
      const bSwap = h('button', { type: 'button', class: 'btn' }, 'Swap A and B');
      const bRand = h('button', { type: 'button', class: 'btn' }, 'Random pair');
      let pairSeed = 3;
      bSwap.addEventListener('click', () => { const t0 = bS.A; bS.A = bS.B; bS.B = t0; bCv.draw(); bRead(); });
      bRand.addEventListener('click', () => { const r = lib.rng(++pairSeed * 7717); bS.A = [-2.4 + 4.8 * r(), -1.8 + 3.6 * r()]; bS.B = [-2.4 + 4.8 * r(), -1.8 + 3.6 * r()]; bCv.draw(); bRead(); });
      bCtl.append(swirl.el, h('div', { class: 'row' }, bSwap, bRand, h('span', { class: 'families-small' }, 'Drag A or B in either panel, or focus one and use the arrow keys.')));
      bCard.appendChild(bCtl);
      const DOM = { x: [-3, 3], y: [-2.25, 2.25] };
      let egrid = null;
      const bLayout = (w) => w >= 640 ? { two: true, pw: (w - 20) / 2, ph: (w - 20) / 2 * 0.75 + 30 } : { two: false, pw: w, ph: w * 0.75 + 30 };
      const bCv = autoCanvas(bCard, {
        height: w => { const L = bLayout(w); return L.two ? L.ph : 2 * L.ph + 16; },
        label: 'Left: energy heatmap with candidates A and B and their energies. Right: the learned arrow field and two integration paths from A to B.',
        draw: drawLab2,
      });
      const bOut = h('div', { class: 'families-lab2-out' }); bCard.appendChild(bOut);
      bCard.appendChild(h('p', { class: 'families-small' }, 'Real math: E is a two-well energy; the arrows are −∇E plus swirl × a rotational field; each route estimate is −∫ arrows · dl with ' + NPATH + ' midpoint evaluations. Real diffusion likelihoods use an ODE solve over noise levels, with hundreds to thousands of steps, and give bounds or approximations (Sec E.2, p.37). This toy keeps only the core point: arrows alone do not give you a score in one pass.'));
      function panelRect(i, w) { const L = bLayout(w); return L.two ? { x: i * (L.pw + 20), y: 30, w: L.pw, h: L.ph - 30 } : { x: 0, y: i * (L.ph + 16) + 30, w: L.pw, h: L.ph - 30 }; }
      const toPx = (r, p) => [r.x + (p[0] - DOM.x[0]) / (DOM.x[1] - DOM.x[0]) * r.w, r.y + (DOM.y[1] - p[1]) / (DOM.y[1] - DOM.y[0]) * r.h];
      const toDom = (r, px, py) => [DOM.x[0] + (px - r.x) / r.w * (DOM.x[1] - DOM.x[0]), DOM.y[1] - (py - r.y) / r.h * (DOM.y[1] - DOM.y[0])];
      function drawLab2(ctx, w) {
        if (!egrid) { const n = 72; egrid = []; for (let r = 0; r < n; r++) { const row = []; for (let c = 0; c < n; c++) row.push(E2([DOM.x[0] + c / (n - 1) * 6, DOM.y[1] - r / (n - 1) * 4.5])); egrid.push(row); } }
        const L = lib.gridRange(egrid), lv = []; for (let i = 1; i <= 9; i++) lv.push(L[0] + (Math.min(L[1], L[0] + 9) - L[0]) * i / 10);
        const r0 = panelRect(0, w), r1 = panelRect(1, w);
        T(ctx, 'EBT: a number per point, E(ŷ)', r0.x, r0.y - 24, { size: 14, weight: 700, color: C.ebt });
        T(ctx, 'Diffusion: an arrow per point, ε̂ ∝ −∇E', r1.x, r1.y - 24, { size: 14, weight: 700, color: C.diff });
        ctx.save(); ctx.beginPath(); ctx.roundRect(r0.x, r0.y, r0.w, r0.h, 8); ctx.clip();
        lib.heatmap(ctx, egrid, r0.x, r0.y, r0.w, r0.h, { range: [L[0], L[0] + 9], gamma: 0.8 });
        lib.contours(ctx, egrid, r0.x, r0.y, r0.w, r0.h, lv, { color: 'rgba(233,238,246,0.22)', width: 1 });
        ctx.restore();
        ctx.beginPath(); ctx.roundRect(r1.x, r1.y, r1.w, r1.h, 8); ctx.fillStyle = '#0f1724'; ctx.fill(); ctx.strokeStyle = C.rule; ctx.lineWidth = 1; ctx.stroke();
        // quiver
        const nx = 17, ny = 13, cell = r1.w / nx;
        for (let i = 0; i < nx; i++) for (let j = 0; j < ny; j++) {
          const p = [DOM.x[0] + (i + 0.5) / nx * 6, DOM.y[1] - (j + 0.5) / ny * 4.5], f = field(p, bS.c), mag = Math.hypot(f[0], f[1]);
          if (mag < 1e-6) continue; const len = Math.min(cell * 0.85, 6 + mag * cell * 0.22), [px, py] = toPx(r1, p);
          lib.arrow(ctx, px - f[0] / mag * len / 2, py + f[1] / mag * len / 2, px + f[0] / mag * len / 2, py - f[1] / mag * len / 2, { color: 'rgba(147,161,184,0.55)', width: 1.2, head: 4.5 });
        }
        // paths
        [[-1, C.diff, []], [1, C.diff, [6, 5]]].forEach(([side, cc, dash]) => { const P = bez(bS.A, bS.B, side), pts = []; for (let i = 0; i <= 48; i++) pts.push(toPx(r1, P(i / 48))); lib.line(ctx, pts, { color: cc, width: 2.4, dash }); });
        const pa = toPx(r1, bS.A), pb = toPx(r1, bS.B), cm = [(pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2];
        [[-1, 'route 1'], [1, 'route 2']].forEach(([side, nm]) => { const q = toPx(r1, bez(bS.A, bS.B, side)(0.5)); let nx = q[0] - cm[0], ny = q[1] - cm[1]; const L = Math.hypot(nx, ny) || 1; nx /= L; ny /= L; T(ctx, nm, q[0] + nx * 18, q[1] + ny * 14, { size: 12.5, kind: 'mono', color: C.diff, align: nx > 0.3 ? 'left' : nx < -0.3 ? 'right' : 'center', baseline: 'middle' }); });
        // A, B markers in both panels
        [r0, r1].forEach((r, i) => [['A', bS.A], ['B', bS.B]].forEach(([nm, p]) => {
          const [px, py] = toPx(r, p); lib.dot(ctx, px, py, 8, i === 0 ? C.ebt : C.ink, { stroke: C.bg, lw: 2 });
          T(ctx, nm, px, py, { size: 11, weight: 700, color: C.bg, align: 'center', baseline: 'middle' });
          if (i === 0) T(ctx, 'E = ' + E2(p).toFixed(2), px + 12, py - 8, { size: 13, kind: 'mono', color: C.ink });
        }));
        placeHandles();
      }
      function bRead() {
        const eA = E2(bS.A), eB = E2(bS.B), d = eB - eA;
        const i1 = pathIntegral(bez(bS.A, bS.B, -1), bS.c), i2 = pathIntegral(bez(bS.A, bS.B, 1), bS.c);
        const verdict = (x) => Math.abs(x) < 0.02 ? 'a tie' : x > 0 ? 'A is more likely' : 'B is more likely';
        const agree = Math.sign(i1) === Math.sign(i2);
        bOut.innerHTML = `<div class="families-verdict ebt"><b>EBT</b><span>E(B) − E(A) = <b>${(d >= 0 ? '+' : '') + d.toFixed(2)}</b> → ${verdict(d)}</span><span class="families-cost">cost: 2 passes</span></div>`
          + `<div class="families-verdict diff"><b>Arrows, route 1</b><span>estimate <b>${(i1 >= 0 ? '+' : '') + i1.toFixed(2)}</b> → ${verdict(i1)}</span><span class="families-cost">cost: ${NPATH} passes</span></div>`
          + `<div class="families-verdict diff"><b>Arrows, route 2</b><span>estimate <b>${(i2 >= 0 ? '+' : '') + i2.toFixed(2)}</b> → ${verdict(i2)}</span><span class="families-cost">cost: ${NPATH} passes</span></div>`
          + `<p class="families-small ${agree ? '' : 'families-warn'}">${bS.c < 0.02 ? 'No swirl: the arrows are an exact gradient, so both routes agree with the EBT, but they cost 64× more passes.' : agree ? `The routes differ by ${Math.abs(i1 - i2).toFixed(2)}: the answer depends on the route. Raise the swirl to make them disagree on the ranking.` : 'The two routes disagree on which candidate is better. With arrows that are not an exact gradient, “how much better is B than A” has no single answer.'}</p>`;
      }
      // draggable, keyboard-movable handles over A and B (the canvas itself stays scrollable on touch screens)
      const handles = [];
      [0, 1].forEach(pi => ['A', 'B'].forEach(nm => {
        const hb = h('button', { type: 'button', class: 'families-handle', 'aria-label': `Move candidate ${nm} (${pi === 0 ? 'energy' : 'arrow'} panel). Drag, or use arrow keys.` });
        bCv.box.appendChild(hb); handles.push({ hb, pi, nm });
        hb.addEventListener('pointerdown', (ev) => { hb.setPointerCapture(ev.pointerId); bS.drag = { nm, i: pi }; ev.preventDefault(); });
        hb.addEventListener('pointermove', (ev) => { if (!bS.drag || bS.drag.nm !== nm || bS.drag.i !== pi) return; const [px, py] = bCv.toLocal(ev), r = panelRect(pi, bCv.w), q = toDom(r, px, py); bS[nm] = [lib.clamp(q[0], DOM.x[0] + 0.1, DOM.x[1] - 0.1), lib.clamp(q[1], DOM.y[0] + 0.1, DOM.y[1] - 0.1)]; bCv.draw(); bRead(); });
        const end = () => { bS.drag = null; }; hb.addEventListener('pointerup', end); hb.addEventListener('pointercancel', end);
        hb.addEventListener('keydown', (ev) => { const d = ev.shiftKey ? 0.3 : 0.1, mv = { ArrowLeft: [-d, 0], ArrowRight: [d, 0], ArrowUp: [0, d], ArrowDown: [0, -d] }[ev.key]; if (!mv) return; ev.preventDefault(); bS[nm] = [lib.clamp(bS[nm][0] + mv[0], DOM.x[0] + 0.1, DOM.x[1] - 0.1), lib.clamp(bS[nm][1] + mv[1], DOM.y[0] + 0.1, DOM.y[1] - 0.1)]; bCv.draw(); bRead(); });
      }));
      function placeHandles() { handles.forEach(({ hb, pi, nm }) => { const [px, py] = toPx(panelRect(pi, bCv.w), bS[nm]); hb.style.left = px + 'px'; hb.style.top = py + 'px'; }); }

      // ============ Table 1 ============
      const FACETS = [
        { n: 'Facet 1', s: 'compute', t: 'Dynamic compute allocation', d: 'Spend more computation on harder predictions, per prediction (p.3). Like taking longer to decide a career change than lunch.' },
        { n: 'Facet 2', s: 'uncertainty', t: 'Uncertainty in continuous spaces', d: 'Know how unsure you are before committing, also when outputs are continuous (video frames, latents), where there is no softmax (p.3).' },
        { n: 'Facet 3', s: 'verify', t: 'Verifying predictions', d: 'Score a candidate, so you can stop early when it is good, think longer when it is not, or pick the best of many (p.3).' },
      ];
      const ROWS = [
        { n: 'FF Transformers', c: 'ar', v: [0, 0, 0], why: [
          'Fixed depth and width, one forward pass per token. Rerunning on the same input gives the same output (p.3 footnote 3; Sec 6.1, p.15). Chain-of-thought adds tokens, but each token still gets a fixed budget (Sec G.1.2, p.39).',
          'For text, softmax gives token probabilities (p.3). For continuous outputs there is no well-defined softmax, so uncertainty needs tricks such as vector quantization or ELBO objectives (p.3, p.15).',
          'Not trained to check a given answer. Improving a single prediction at inference usually needs an external model (p.15).'] },
        { n: 'RNNs', c: 'rnn', v: [0, 0, 0], why: [
          'Most modern RNNs update their state only with new input, so they cannot think longer about the current token (p.2; Sec 6.2, p.16). Recurrent-depth variants can loop [25], but are not widely adopted (p.3 footnote 2).',
          'Same readout story as FF Transformers: softmax for tokens, no native uncertainty for continuous outputs (p.3).',
          'Even recurrent-depth RNNs lack explicit verification (p.2): they amortize the gradient of an energy instead of outputting one (p.16, p.36).'] },
        { n: 'Diffusion Transformers', c: 'diff', v: [1, 0, 0], why: [
          'More denoising steps means more computation, so yes (p.3). But the fixed denoising schedule restricts adaptive halting or extending (p.16), and models typically fail to benefit from steps beyond those they were trained on (p.2).',
          'Each step predicts noise, not an energy, so there is no unnormalized likelihood at each step (Fig 1 caption, p.2). Likelihoods need ELBO bounds or ODE solvers with approximation error (p.37).',
          'No built-in score for its own sample; beyond adding steps, improving at inference needs an external verifier (p.2, p.16).'] },
        { n: 'EBTs', c: 'ebt', v: [1, 1, 1], why: [
          'Thinking is gradient descent on the energy, so it can run for any number of steps and stop when the energy converges (Fig 2, p.4). Up to 29% more improvement from extra passes than Transformer++ on language (p.4, Fig 6a).',
          'The energy is an unnormalized likelihood, defined for continuous predictions too (p.4, Sec 3.1). Easy tokens reach low energy, hard tokens stay high (Fig 8, p.12).',
          'The forward pass is a verifier: one scalar per candidate. Algorithm 2 (p.7) optimizes M candidates and keeps the lowest-energy one, with no extra model.'] },
      ];
      const tCard = h('div', { class: 'card families-t1' }); el.appendChild(tCard);
      tCard.append(h('div', { class: 'row families-badges' }, lib.badge('paper', 'Table 1, p.3')), h('h3', {}, 'Table 1, explained cell by cell'), h('p', { class: 'families-small' }, 'Hover, tap, or tab to a cell to see why the paper marks it that way. Hovering a lane above highlights its row.'));
      const why = h('div', { class: 'families-why', 'aria-live': 'polite' });
      const cells = [];
      const showCell = (ri, fi) => {
        cells.forEach(c => c.el.classList.toggle('on', c.ri === ri && c.fi === fi));
        const R0 = ROWS[ri], F = FACETS[fi];
        why.innerHTML = `<div class="families-why-h"><span class="families-dot" style="background:var(--${R0.c})"></span><b>${R0.n}</b> × <b>${F.n}: ${F.t}</b> <span class="families-mark ${R0.v[fi] ? 'yes' : 'no'}">${R0.v[fi] ? '✓ yes' : '✗ no'}</span></div><p>${R0.why[fi]}</p><p class="families-small">${F.n}: ${F.d}</p>`;
      };
      const rowEls = [];
      const table = h('table', { class: 'families-table' },
        h('thead', {}, h('tr', {}, h('th', {}, 'Architecture'), FACETS.map(F => h('th', {}, h('span', { class: 'families-fn' }, F.n), h('br', {}), h('span', { class: 'families-fl' }, F.t), h('span', { class: 'families-fs' }, F.s))))),
        h('tbody', {}, ROWS.map((R0, ri) => { const tr = h('tr', {}, h('td', { html: `<span class="families-dot" style="background:var(--${R0.c})"></span>${R0.n}` }), R0.v.map((v, fi) => { const b = h('button', { type: 'button', class: 'families-cell ' + (v ? 'yes' : 'no'), 'aria-label': `${R0.n}, ${FACETS[fi].t}: ${v ? 'yes' : 'no'}` }, v ? '✓' : '✗'); ['mouseenter', 'focus', 'click'].forEach(e => b.addEventListener(e, () => showCell(ri, fi))); cells.push({ el: b, ri, fi }); return h('td', { class: 'families-cellwrap' }, b); })); rowEls.push(tr); tr.addEventListener('mouseenter', () => { const L = LANES.find(l => l.table === ri); laneEls[L.id].card.classList.add('hl'); }); tr.addEventListener('mouseleave', () => LANES.forEach(l => laneEls[l.id].card.classList.remove('hl'))); return tr; })));
      tCard.append(h('div', { class: 'families-t1grid' }, h('div', { class: 'tbl' }, table), why));
      function highlightRow(ri, on) { rowEls.forEach((r, i) => r.classList.toggle('hl', on && i === ri)); }
      showCell(2, 0);

      // ============ closing: diffusion = learned gradient of an energy ============
      el.appendChild(h('div', { class: 'families-close' },
        h('figure', {}, h('div', { class: 'paper-fig' }, h('img', { src: 'media/paper/figE1.png', alt: 'Paper Figure E.1: diffusion models are supervised at every step of a denoising path; EBMs learn an energy landscape and are supervised only at the end.' })),
          h('figcaption', {}, lib.badge('paper', 'Fig E.1, p.37'), ' Diffusion: supervision at every step of a set denoising path. EBM: a landscape, supervision only at the end, update = gradient descent on the energy.')),
        lib.callout('insight', 'Diffusion is the closest relative', '<p>Diffusion models can be seen as predicting the gradient of the energy (the score), so EBMs are a generalization of diffusion that also outputs the energy itself (Sec 6.4, p.16; Sec E.1, p.36). Same iterative spirit, different interface:</p><ul class="families-list"><li><b>Diffusion</b> follows a fixed schedule and outputs a direction per step. It cannot say how good its current guess is.</li><li><b>EBT</b> outputs a number per step. That number gives a stopping rule, an uncertainty readout, and Best-of-N selection for free.</li><li>Because the EBT learns a whole landscape, small errors at one step need not accumulate: the minimum can still be reached (Sec E.2, p.37, assuming a well-formed landscape).</li></ul><p class="families-small">Fair caveats from the paper: EBTs add hyperparameters (step size, steps), cost more per training step, and currently struggle with distributions that have many modes, likely due to the convex-landscape assumption (Limitations, p.17).</p>')));

      // ---------- start ----------
      const kick = () => { redrawAll(); bRead(); render(); };
      kick();
      if (window.EBTV && EBTV.fontsReady) EBTV.fontsReady.then(kick);
      lib.whenVisible(lanesBox, () => { if (!reduced) { S.k = 0; S.phase = 1; setTimeout(play, 350); } });
    },
  });
})();
