/* Panel: why per-prediction thinking matters, and four ways to predict the next token (paper Fig 1, Table 1, Sec 6.1-6.4).
   Stage: four lanes (AR Transformer, RNN, Diffusion Transformer, EBT) answer the same next-token question under a shared
   budget of forward passes. Three candidate tokens, so every prediction is a point in the probability simplex (a triangle).
   - Diffusion: a real DDIM sampler over the 3 logits with the exact denoiser for a Gaussian target around its belief.
   - EBT: real gradient descent on E(ŷ) = CE(b, softmax(ŷ)) (gradient softmax(ŷ) − b), stopping when E ≤ τ or when one
     step lowers E by less than 0.002.
   - AR and RNN: one pass gives softmax(belief); internals are schematic.
   Every lane gets the same size of knowledge error (logit noise), so this is about mechanism, not accuracy.
   Step 5 holds a compact Table 1 in the prose; hovering a row highlights that lane. Consolidated from legacy system2 + families. */
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

  const T1 = `<table class="fam-table"><thead><tr><th></th><th class="c">think longer<span>Facet 1</span></th><th class="c">uncertainty<span>Facet 2</span></th><th class="c">verify<span>Facet 3</span></th></tr></thead><tbody>`
    + [['FF Transformer', 0, 0, 0], ['RNN', 0, 0, 0], ['Diffusion', 1, 0, 0], ['EBT', 1, 1, 1]].map(([n, ...f], i) => `<tr data-lane="${i}"${i === 3 ? ' class="ebt"' : ''}><td>${n}</td>${f.map(v => `<td class="c ${v ? 'yes' : 'no'}">${v ? '✓' : '✗'}</td>`).join('')}</tr>`).join('')
    + '</tbody></table>';

  EBT.panel({
    id: 'families',
    nav: 'Four model families',
    title: 'Four ways to predict the next token',
    lede: 'Some predictions deserve more thought than others. Whether a model can give it depends on what one forward pass returns, and whether a second pass can help.',
    text: `
      <p>Psychology calls fast, automatic thinking <b>System 1</b> and slow, deliberate thinking <b>System 2</b>: lunch takes a moment, a career change far longer (p.3). The paper wants System 2 for every single prediction, learned from unlabeled data alone, without rewards, answer checkers or a second model (p.2). That takes two things: a way to turn extra compute into a better answer, and a signal for when to stop.</p>
      <p>In the figure, four families share one prompt and a budget of passes. With three candidate tokens, each prediction is a point in a triangle; the dashed crosshair is the true distribution $p$.</p>`,
    steps: [
      { label: 'Same question, four machines', html: '<p>Press <b>play</b>. Each tick, every lane that can still use a pass runs one. All lanes share the same knowledge error, so they land close together (KL chart); what differs is how they get there.</p>' },
      { label: 'AR and RNN: fixed compute per token', html: '<p>An AR Transformer maps the context through a fixed stack of layers to $\\mathrm{softmax}(f_\\theta(x_{\\le t}))$: "the" and the last digit of a long sum get the same pass, and a rerun returns the same answer. An RNN updates its state only when a new token arrives. Extra passes are crossed out.</p>' },
      { label: 'Diffusion: a fixed schedule of arrows', html: '<p>A diffusion Transformer denoises over $T$ steps, so it does spend more compute, but $T$ is fixed in advance: it cannot halt or extend half-way (p.16). Each step predicts noise, $\\epsilon_\\theta \\approx \\sigma_t\\nabla E_t$: an amortized gradient of an energy it never outputs (arrows in its triangle). Arrows say which way is better, not how good a guess is: it cannot score its own candidates.</p>' },
      { label: 'EBT: a score, descended until it settles', html: '<p>An EBT takes the candidate as input and returns one number, the energy $E_\\theta(x,\\hat y)$, shaded over the triangle. Thinking is gradient descent on it, $\\hat y \\leftarrow \\hat y-\\alpha\\nabla_{\\hat y}E_\\theta$, stopping when $E$ is low enough or stops falling. On this hard prompt the energy levels off high: more thought cannot make a favorite color predictable, and the plateau reports that.</p>' },
      { label: 'Table 1: three facets', html: '<p>Table 1 (p.3); hover a row to find its lane. ✗ holds "generally": recurrent-depth RNNs and score-based diffusion are partial exceptions.</p><div class="fam-t1">' + T1 + '</div>' },
    ],
    after: `
      <p class="note">The price: each EBT step needs a backward pass to $\\hat y$, and training differentiates through the steps, about 3.33× a Transformer++ training step per optimization step (p.36). The diffusion lane runs real DDIM, the EBT lane real descent on a cross-entropy energy; AR and RNN internals are schematic.</p>`,
    source: [{ kind: 'concept', note: 'hand-set beliefs, real DDIM and descent' }, { kind: 'paper', note: 'p.2-3, Fig 1, Table 1, Sec 6.1-6.4' }],

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

      // ---------- Table 1 rows (step 5 prose) highlight their lane ----------
      ctx.panel.querySelectorAll('.fam-t1 tr[data-lane]').forEach(tr => {
        const i = +tr.dataset.lane;
        tr.addEventListener('mouseenter', () => { S.hl = i; lanesCv.draw(); });
        tr.addEventListener('mouseleave', () => { S.hl = -1; lanesCv.draw(); });
      });

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
      const kick = () => { canvases.forEach(c => c.draw()); updateStatus(); };
      kick(); if (window.EBTV && EBTV.fontsReady) EBTV.fontsReady.then(kick);
      let shown = false;
      const CFG = [
        { prompt: 'medium', budget: 12, focus: null, quiver: false, facets: false, play: true },
        { prompt: 'medium', budget: 12, focus: ['ar', 'rnn'], quiver: false, facets: false },
        { prompt: 'medium', budget: 12, focus: ['diff'], quiver: true, facets: false },
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
