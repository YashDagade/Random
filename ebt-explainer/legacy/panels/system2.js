/* Panel: System 1 vs System 2, today's recipes, the paper's core question, and its answer.
   Figure: a "difficulty dial". Eight tasks, each the minimum of a stretched quadratic valley E_k(ŷ) = ½(u²/κ_k + v²).
   Gray (System 1) takes a fixed K gradient steps on every task. Blue (System 2) keeps stepping until a stop signal fires:
   an answer key, an external checker that exists only for math and code, no check at all (a budget T fixed in advance),
   or its own energy settling. Everything is computed live (8 tasks × 24 random starts).
   The recipes table (step 3) and the three headline-result mini charts (digitized paper data, data/scaling.json) are
   placed in the prose column next to the text that explains them, so the sticky figure stays within one screen. */
(function () {
  'use strict';
  // ---------------- toy math (illustration, not a trained model) ----------------
  const NT = 8, KMAX = 12, R_TOL = 0.15, EPS_E = 1e-4, CAP = 60, NSTART = 24, EXT = 3;
  const YSTAR = [0.8, -0.4], Y0 = [-2.1, 1.6];
  const DOMAIN = ['math', 'essay', 'code', 'image', 'math', 'video', 'code', 'essay'];
  const CHECKABLE = (d) => d === 'math' || d === 'code';
  const TASKS = Array.from({ length: NT }, (_, k) => {
    const kap = Math.pow(KMAX, k / (NT - 1)), th = (25 + 17 * k) * Math.PI / 180;
    return { k, kap, c: Math.cos(th), s: Math.sin(th), dom: DOMAIN[k] };
  });
  const uv = (T, y) => { const dx = y[0] - YSTAR[0], dy = y[1] - YSTAR[1]; return [T.c * dx + T.s * dy, -T.s * dx + T.c * dy]; };
  const energy = (T, y) => { const [u, v] = uv(T, y); return 0.5 * (u * u / T.kap + v * v); };
  const grad = (T, y) => { const [u, v] = uv(T, y); const gu = u / T.kap, gv = v; return [T.c * gu - T.s * gv, T.s * gu + T.c * gv]; };
  const dist = (y) => Math.hypot(y[0] - YSTAR[0], y[1] - YSTAR[1]);
  // pol: {kind:'fixed', n} | {kind:'key'} | {kind:'energy'}; step size α = 1 throughout
  function run(T, y0, pol) {
    let y = y0.slice(); const path = [y.slice()], Es = [energy(T, y)]; let reason = null;
    for (let i = 0; ; i++) {
      if (pol.kind === 'fixed' && i >= pol.n) { reason = 'budget'; break; }
      if (pol.kind === 'key' && dist(y) < R_TOL) { reason = 'checked'; break; }
      if (i >= CAP) { reason = 'cap'; break; }
      const g = grad(T, y); y = [y[0] - g[0], y[1] - g[1]]; path.push(y.slice()); Es.push(energy(T, y));
      if (pol.kind === 'energy' && Es[Es.length - 2] - Es[Es.length - 1] < EPS_E) { reason = 'settled'; break; }
    }
    return { path, Es, n: path.length - 1, hit: dist(y) < R_TOL, reason };
  }
  const MODES = {
    key: { label: 'answer key', pol: () => ({ kind: 'key' }) },
    checker: { label: 'checker: math, code', pol: (T, S) => CHECKABLE(T.dom) ? { kind: 'key' } : { kind: 'fixed', n: S.T } },
    none: { label: 'no check', pol: (T, S) => ({ kind: 'fixed', n: S.T }) },
    energy: { label: 'own energy', pol: () => ({ kind: 'energy' }) },
  };
  const REASON = { checked: 'answer checked', settled: 'energy settled', budget: 'budget used up', cap: CAP + '-step cap' };

  EBT.panel({
    id: 'system2',
    nav: 'System 1, System 2',
    title: 'Thinking fast, thinking slow, and the question this paper asks',
    lede: 'Some predictions deserve more thought than others. Today\'s models either spend the same compute on every token, or need a reward, an answer checker or a second model to think harder. The paper asks whether a model can learn to think harder from unlabeled data alone.',
    text: `
      <p>Psychology splits thinking into two modes (p.1). <b>System 1</b> is fast, intuitive and automatic. <b>System 2</b> is slow, deliberate and analytical: the mode for "mathematics, programming, multistep reasoning, or novel out-of-distribution situations" (p.2). The paper's example: deciding what to eat for lunch takes a moment, deciding whether to change careers takes much longer (p.3).</p>
      <p>The paper means this per <em>prediction</em>, for example per token (footnote 3, p.3). A feed-forward Transformer has fixed depth and width, so the word "the" and the last digit of a long sum get exactly the same forward pass. To think harder where it matters, a model needs two things: a way to turn extra compute into a better answer, and a signal that tells it when to stop.</p>
      <p>The figure makes "difficulty" precise with a toy. Each task asks for the minimum $y^*$ of a stretched valley</p>
      <div class="eq">$$E_\\kappa(\\hat y) = \\tfrac12\\Big(\\frac{u^2}{\\kappa} + v^2\\Big),$$<span class="why">u, v: coordinates of ŷ − y* along and across the valley. κ ≥ 1: how stretched it is (its condition number).</span></div>
      <p>Gradient descent with step size 1 removes the error across the valley in one step and shrinks the error along it by a factor $1-1/\\kappa$ per step. From an initial error $u_0$, reaching tolerance $r$ takes</p>
      <div class="eq">$$n(\\kappa) = \\Big\\lceil \\frac{\\ln(r/|u_0|)}{\\ln(1-1/\\kappa)} \\Big\\rceil \;\\approx\; \\kappa \\ln\\frac{|u_0|}{r}$$<span class="why">≈ for large κ. Steps grow linearly with difficulty, so any fixed budget K fails once n(κ) &gt; K.</span></div>`,
    steps: [
      { label: 'System 1: the same budget for every task', html: '<p>Gray is a System 1 model. It takes $K = 3$ steps on every task, the way a Transformer spends one forward pass on every token. Black ticks mark the steps each task actually needs (mean over 24 random starts). Gray solves the first two tasks from every start, then a shrinking share as κ grows: 41% of all starts. Drag the K slider: a bigger K rescues harder tasks but charges every easy task the same price.</p>' },
      { label: 'System 2: spend where it is needed', html: '<p>Blue keeps stepping until its guess is within tolerance. Its bars follow the black ticks, from 1 step at $\\kappa = 1$ to about 24 at $\\kappa = 12$, and it solves every task with 69 steps in all, against gray\'s 24. But it cheats: it stops by comparing against the <b>answer key</b>, which no model has at test time. Real thinking needs its own way to know when to stop.</p>' },
      { label: 'Today\'s recipes bring the check from outside', html: '<p>Current reasoning methods get that check from somewhere else. The paper sorts their limits into three kinds: they are "modality-specific (e.g., working only in text), problem-specific (e.g., verifiable domains like math and coding), or require additional supervision/training on top of unsupervised pretraining (e.g., verifiers or verifiable rewards)" (p.1). RL-trained reasoning models (o1, R1, Grok3, Claude 3.7 Sonnet) are the prominent case: their training "only works in domains where rule-based rewards can easily verify answers, such as math and coding" (p.2).</p><div class="s2-slot" data-slot="recipes"></div><p>In the figure the checker now exists only for math and code. Essays, images and video fall back to a budget fixed in advance ($T = 12$): hatched bars are steps spent beyond what was needed. The video task now fails from about a third of starts, and the hardest essay almost always fails.</p>' },
      { label: 'The question', html: '<p>Remove the outside checker entirely. Blue must now fix $T$ before it starts: 12 steps where 1 would do, too few from $\\kappa \\approx 6$ on. It spends 96 steps and solves 74% of starts. No single $T$ fixes this: a larger one wastes more on easy tasks, a smaller one fails more hard ones (drag T). The paper\'s core question (p.2) is whether the stop-and-choose signal itself can be learned:</p><p class="s2-quote">"Can we rely entirely on unsupervised learning to develop System 2 Thinking?"</p><p>That would extend System 2 Thinking "to any problem, any modality, and avoid the reliance on external human, reward, or model supervision" (p.2).</p>' },
      { label: 'The answer: learn the verifier, then optimize against it', html: '<p>The paper\'s answer is yes, "by learning to explicitly verify the compatibility between inputs and candidate-predictions, and then re-framing prediction problems as optimization with respect to this verifier" (p.1). Blue now stops when its <b>own energy</b> stops falling, $E_i - E_{i+1} < 10^{-4}$. Its allocation tracks the answer key\'s, plus about 1 to 5 steps per task to see that the energy has stopped falling. It solves every task with 100 steps in all, about what the fixed $T = 12$ spent for 74%, now placed where it is needed. No label is needed at test time, so the rule works for any modality, and the same scalar can rank several attempts (Facet 3).</p>' },
    ],
    after: `
      <h3>How the paper scores thinking</h3>
      <p>Definition C.1 (p.30) measures System 2 Thinking as the expected relative change in a metric $P$ when a model gets $F$ function evaluations instead of the minimum $F_0$ it needs for a valid prediction:</p>
      <div class="eq">$$\\mathrm{STT}(x,\\theta,F) = \\mathbb E_x\\Big[\\frac{P(x,\\theta,F)}{P(x,\\theta,F_0)} - 1\\Big]$$<span class="why">P can be accuracy, perplexity, FID, ... One function evaluation is one forward pass; for an EBT, one optimization step (p.8).</span></div>
      <p>For a feed-forward Transformer, extra passes on the same input return the same prediction, so its score is zero. The question is whether a model can make this number positive without outside help.</p>
      <h3>The three headline results, read carefully</h3>
      <div class="s2-slot" data-slot="res0"></div>
      <p><b>35%</b> is the best of six text axes: data, 35.98% (Fig 4a). Batch size gives 28.46%, depth 5.29%, parameters 2.91%, FLOPs 2.92%, width 0.02% (Figs 4, 5). The paper never defines "scaling rate"; the printed numbers equal the ratio of fitted log-log slopes minus 1, which the chart reproduces. At the measured scales EBT perplexity is higher on most axes; it ends below Transformer++ only for data and batch size, after a crossover (width is a wash). Counted in training FLOPs, which include EBT's 6.66× training cost (p.36), EBT is worse at every measured budget (Fig 5b).</p>
      <div class="s2-slot" data-slot="res1"></div>
      <p><b>29%</b> ("29% more than the Transformer++", p.1) is EBT's relative gain from its own no-thinking point (Fig 6a, approx. 44.8 → 31.8, lower is better), against a Transformer++ gain of zero: more passes cannot change a feed-forward prediction. EBT's no-thinking point is worse than Transformer++ (≈38.4); one extra step already beats it, and choosing among several candidates (Best-of-N) adds the rest.</p>
      <div class="s2-slot" data-slot="res2"></div>
      <p><b>99% fewer forward passes</b> means 3 EBT passes against 300 DiT passes when denoising images at an out-of-distribution noise level (p.35, Fig 12). Both models re-denoise their own output three times; the DiT runs DDIM each time. EBT is clearly ahead only at 3 vs 300; at 1 vs 100 and 2 vs 200 the two are about equal. Each EBT pass also needs a backward pass, so the compute saving is smaller than 100×.</p>
      <p class="note">The difficulty dial is an illustration with a hand-written energy, not a trained model: its "own energy" is exact, and only the stopping rule can mislead it. A learned energy can be wrong, and its scale is arbitrary, so ε must be tuned. A model that only outputs update directions could also stop when its steps shrink; it could not compare two candidates or tell a high plateau from a low one (next panel). "Thinking" in this paper means refining one continuous prediction step by step, not chain-of-thought; at this model size chain-of-thought did not help (footnote 9, p.10). Whether inference-time compute captures what psychologists call System 2 is "still actively debated" (p.39).</p>`,
    source: [{ kind: 'concept', note: 'difficulty dial: hand-written energies, real gradient descent' }, { kind: 'paper', note: 'p.1–3, p.16; Figs 4a, 6a, 12 digitized (approx.)' }],

    figure(stage, ctx) {
      const { lib } = ctx, h = lib.h, C = lib.C, T_ = lib.text;
      const S = { mode: 'key', showBlue: false, showDom: false, K: 3, T: 12, task: 6, y0: Y0.slice(), tick: 999 };
      const rs = lib.rng(3); const STARTS = Array.from({ length: NSTART }, () => [rs.normal() * 1.3, rs.normal() * 1.3]);
      let startIdx = -1;
      const canvases = [];
      // auto-sized canvas: logical width = CSS width, so text renders at its true size on every screen
      function autoCanvas(parent, o) {
        const box = h('div', { class: 'canvas-box' }); parent.appendChild(box);
        const c = h('canvas', { role: 'img', 'aria-label': o.label || '' }); box.appendChild(c);
        const g = c.getContext('2d'); const st = { w: 0, h: 0, ctx: g, canvas: c, box };
        const resize = () => { if (!box.clientWidth) return false; const w = Math.max(200, Math.round(box.clientWidth)); const hh = Math.round(o.height(w)), dpr = Math.min(2, window.devicePixelRatio || 1); if (w === st.w && hh === st.h) return false; st.w = w; st.h = hh; c.width = Math.round(w * dpr); c.height = Math.round(hh * dpr); c.style.height = hh + 'px'; g.setTransform(dpr, 0, 0, dpr, 0, 0); return true; };
        st.draw = () => { resize(); if (!st.w) return; g.clearRect(0, 0, st.w, st.h); o.draw(g, st.w, st.h); };
        st.toLocal = (ev) => { const r = c.getBoundingClientRect(); return [(ev.clientX - r.left) / r.width * st.w, (ev.clientY - r.top) / r.height * st.h]; };
        if (window.ResizeObserver) new ResizeObserver(() => { if (resize()) st.draw(); }).observe(box);
        canvases.push(st); return st;
      }

      // ---------- statistics over the 24 starts ----------
      let stats = null, cur = null;
      const bluePol = (T) => MODES[S.mode].pol(T, S);
      function computeStats() {
        const mean = (a, f) => a.reduce((s, r) => s + f(r), 0) / a.length;
        stats = TASKS.map(T => {
          const key = STARTS.map(y => run(T, y, { kind: 'key' })), gr = STARTS.map(y => run(T, y, { kind: 'fixed', n: S.K })), bl = STARTS.map(y => run(T, y, bluePol(T)));
          return { needed: mean(key, r => r.n), gray: { steps: mean(gr, r => r.n), rate: mean(gr, r => r.hit ? 1 : 0) }, blue: { steps: mean(bl, r => r.n), rate: mean(bl, r => r.hit ? 1 : 0), fixed: bluePol(T).kind === 'fixed' } };
        });
      }
      function computeCurrent() { const T = TASKS[S.task]; cur = { gray: run(T, S.y0, { kind: 'fixed', n: S.K }), blue: run(T, S.y0, bluePol(T)) }; }

      // ---------- layout ----------
      const row = h('div', { class: 'fig-row' }); stage.appendChild(row);
      const A = lib.frame(row, { label: 'Difficulty dial', sub: 'mean of 24 starts · click a task' });
      A.wrap.style.flex = '1 1 330px';
      const dial = autoCanvas(A.frame, { height: (w) => Math.round(Math.max(300, Math.min(330, w * 0.92))), label: 'Steps spent and share of starts solved for eight tasks of rising difficulty, for a fixed-budget model and a thinking model. Click a task to inspect it.', draw: drawDial });
      const roA = h('div', { class: 'readout s2-ro' }); A.wrap.appendChild(roA);
      const B = lib.frame(row, { label: 'One task, up close', sub: 'E(ŷ) · click to set ŷ<sub>0</sub>' });
      B.wrap.style.flex = '1 1 230px';
      const land = autoCanvas(B.frame, { height: (w) => w, label: 'Energy landscape of the selected task with both models\' gradient-descent paths. Click to choose the starting guess.', draw: drawLand });
      const roB = h('div', { class: 'readout s2-ro s2-roB' }); B.wrap.appendChild(roB);

      const controls = h('div', { class: 'controls' }); stage.appendChild(controls);
      controls.appendChild(h('span', { class: 'fig-label' }, 'blue stops when'));
      const seg = lib.segmented({ label: 'Stop signal of the thinking model', options: Object.entries(MODES).map(([k, m]) => [k, m.label]), value: S.mode, onchange: (v) => { S.mode = v; S.showBlue = true; S.showDom = S.showDom || v === 'checker' || v === 'none'; refresh(true, true); } });
      controls.appendChild(seg.el);
      const controls2 = h('div', { class: 'controls' }); stage.appendChild(controls2);
      const slK = lib.slider({ id: 's2-K', label: 'gray budget K', min: 1, max: 16, step: 1, value: S.K, oninput: (v) => { S.K = v; refresh(true, false); } });
      const slT = lib.slider({ id: 's2-T', label: 'blue budget T (no check)', min: 2, max: 40, step: 1, value: S.T, oninput: (v) => { S.T = v; refresh(true, false); } });
      controls2.append(slK.el, slT.el, lib.button('new start', () => { startIdx = (startIdx + 1) % NSTART; S.y0 = STARTS[startIdx].slice(); refresh(false, true); }), lib.button('replay', () => animate()));

      // ---------- recipes table (into the step-3 prose) ----------
      const slot = (name) => ctx.panel.querySelector(`.s2-slot[data-slot="${name}"]`);
      const RECIPES = [
        { n: 'RL with verifiable rewards', pipe: ['pretrain', '+ RL, rule-checked reward'], lim: 'math and code only; "often deteriorates performance in other tasks such as writing" (p.2)' },
        { n: 'Best-of-N with a verifier', pipe: ['pretrain', '+ a separate verifier'], lim: 'a second model and extra supervision on top of pretraining (p.1, p.8)' },
        { n: 'Chain-of-thought', pipe: ['pretrain', '+ reasoning tokens'], lim: 'discrete text; they "don\'t seamlessly transfer to continuous modalities" (p.16)' },
        { n: 'Diffusion, more steps', pipe: ['pretrain', '+ more denoising steps'], lim: 'typically no gain beyond the trained number of steps; needs an external verifier (p.2)' },
        { n: 'EBT (this paper)', pipe: ['pretrain only, unlabeled data'], lim: 'is that enough? (step 4)', ebt: true },
      ];
      const recSlot = slot('recipes');
      if (recSlot) {
        recSlot.className = 's2-slot s2-recipes';
        recSlot.appendChild(h('div', { class: 's2-rh' }, 'where the check comes from · our summary of p.1, p.2, p.8, p.16'));
        RECIPES.forEach(r => recSlot.appendChild(h('div', { class: 's2-r' + (r.ebt ? ' ebt' : '') },
          h('div', { class: 's2-rn' }, h('b', {}, r.n), h('span', { class: 's2-pipe' }, r.pipe.map((p, i) => h('span', { class: 's2-box' + (i > 0 ? ' extra' : '') + (r.ebt ? ' ebt' : '') }, p)))),
          h('div', { class: 's2-lim' }, r.lim))));
      }

      // ---------- headline result mini charts (into the "after" prose) ----------
      const sc = lib.data('scaling'); const plotById = (id) => sc && sc.plots ? sc.plots.find(p => p.id === id) : null;
      const RES = [
        { id: 'fig4a', big: '35.98%', t: 'faster data scaling', fig: 'Fig 4a, p.9' },
        { id: 'fig6a', big: '29%', t: 'thinking gain on OOD text', fig: 'Fig 6a, p.11' },
        { id: 'fig12', big: '3 vs 300', t: 'forward passes, OOD image denoising', fig: 'Fig 12, p.15' },
      ];
      RES.forEach((R, i) => {
        const el = slot('res' + i); if (!el) return;
        el.className = 's2-slot s2-res';
        el.appendChild(h('div', { class: 's2-resh' }, h('b', {}, R.big), h('span', {}, R.t), h('span', { class: 'src paper' }, R.fig + ' · approx.')));
        autoCanvas(el, { height: (w) => Math.round(Math.max(170, Math.min(200, w * 0.48))), label: R.fig + ' digitized: ' + R.t, draw: (g, w, hh) => drawRes(R, g, w, hh) });
      });

      // ---------- drawing helpers ----------
      const tick = (g, x1, y1, x2, y2, col, lw, dash) => { g.save(); g.strokeStyle = col; g.lineWidth = lw || 1; if (dash) g.setLineDash(dash); g.beginPath(); g.moveTo(x1, y1); g.lineTo(x2, y2); g.stroke(); g.restore(); };
      function hatch(g, x, y, w, hh, col) { if (hh <= 0) return; g.save(); g.beginPath(); g.rect(x, y, w, hh); g.clip(); g.strokeStyle = col; g.lineWidth = 1; for (let k = -hh; k < w + hh; k += 4) { g.beginPath(); g.moveTo(x + k, y + hh); g.lineTo(x + k + hh, y); g.stroke(); } g.restore(); g.strokeStyle = col; g.lineWidth = 1; g.strokeRect(x + 0.5, y + 0.5, Math.max(0, w - 1), Math.max(0, hh - 1)); }

      function dialGeom(w, hh) {
        const L = 34, Rr = w - 6, top0 = 22, top1 = Math.round(hh * 0.47), lab0 = top1 + 6;
        const bot0 = top1 + (S.showDom ? 64 : 46), bot1 = hh - 22;
        const sw = (Rr - L) / NT;
        return { L, Rr, top0, top1, lab0, bot0, bot1, sw, cx: (k) => L + sw * (k + 0.5) };
      }
      function drawDial(g, w, hh) {
        if (!stats) return;
        const G = dialGeom(w, hh), blueOn = S.showBlue;
        let ymax = Math.max(...stats.map(s => s.needed), S.K, blueOn ? Math.max(...stats.map(s => s.blue.steps)) : 0);
        ymax = Math.max(10, Math.ceil(ymax / 10) * 10);
        const Y = (v) => G.top1 - (v / ymax) * (G.top1 - G.top0);
        g.fillStyle = C.blue4; g.fillRect(G.L + G.sw * S.task + 1, G.top0 - 8, G.sw - 2, G.bot1 - G.top0 + 8);
        T_(g, 'steps spent', G.L - 30, 2, { size: 10.5, kind: 'mono', color: C.muted });
        [0, ymax / 2, ymax].forEach(v => { tick(g, G.L, Y(v), G.Rr, Y(v), C.rule, 1); T_(g, String(v), G.L - 6, Y(v), { size: 10, kind: 'mono', color: C.muted, align: 'right', baseline: 'middle' }); });
        tick(g, G.L, G.top1 + 0.5, G.Rr, G.top1 + 0.5, C.ink, 1);
        const bw = Math.max(5, Math.min(13, G.sw * 0.3));
        stats.forEach((s, k) => {
          const cx = G.cx(k);
          const gx = blueOn ? cx - bw - 1 : cx - bw / 2;
          const gs = s.gray.steps, gu = Math.min(gs, s.needed); // fixed budget: solid up to what was needed, hatched beyond (waste)
          hatch(g, gx, Y(gs), bw, G.top1 - Y(gs), '#6b6b70');
          g.fillStyle = '#d4d4d8'; g.fillRect(gx, Y(gu), bw, G.top1 - Y(gu)); g.strokeStyle = C.ink; g.lineWidth = 0.8; g.strokeRect(gx + 0.4, Y(gs) + 0.4, bw - 0.8, G.top1 - Y(gs) - 0.8);
          if (blueOn) {
            const bx = cx + 1, bs = s.blue.steps;
            if (s.blue.fixed) { // budget fixed in advance: solid up to what was needed, hatched beyond (waste)
              const used = Math.min(bs, s.needed);
              hatch(g, bx, Y(bs), bw, G.top1 - Y(bs), C.blue);
              g.fillStyle = C.blue; g.fillRect(bx, Y(used), bw, G.top1 - Y(used));
            } else { g.fillStyle = C.blue; g.fillRect(bx, Y(bs), bw, G.top1 - Y(bs)); }
          }
          const ny = Y(s.needed); tick(g, cx - bw - 5, ny, cx + bw + 5, ny, C.ink, 1.6);
        });
        stats.forEach((s, k) => {
          const cx = G.cx(k), T = TASKS[k];
          T_(g, 'κ ' + (T.kap < 10 ? T.kap.toFixed(1) : T.kap.toFixed(0)), cx, G.lab0, { size: 10, kind: 'mono', color: k === S.task ? C.ink : C.muted, align: 'center', weight: k === S.task ? 700 : 400 });
          if (S.showDom) {
            const chk = CHECKABLE(T.dom);
            T_(g, T.dom, cx, G.lab0 + 14, { size: 10, kind: 'mono', color: chk ? C.ink : C.muted, align: 'center' });
            if (S.mode === 'checker') T_(g, chk ? 'check' : 'none', cx, G.lab0 + 27, { size: 9.5, kind: 'mono', color: chk ? C.blue : C.faint, align: 'center' });
          }
        });
        const Yb = (v) => G.bot1 - v * (G.bot1 - G.bot0);
        T_(g, 'solved (share of 24 starts)', G.L - 30, G.bot0 - 16, { size: 10.5, kind: 'mono', color: C.muted });
        [0, 0.5, 1].forEach(v => { tick(g, G.L, Yb(v), G.Rr, Yb(v), C.rule, 1); T_(g, Math.round(v * 100) + '%', G.L - 6, Yb(v), { size: 10, kind: 'mono', color: C.muted, align: 'right', baseline: 'middle' }); });
        const line = (key, col, dash, lw) => { const pts = stats.map((s, k) => [G.cx(k), Yb(s[key].rate)]); lib.line(g, pts, { color: col, width: lw, dash }); pts.forEach(p => lib.dot(g, p[0], p[1], key === 'blue' ? 3.2 : 2.8, key === 'blue' ? col : '#fff', { stroke: col, lw: 1.3 })); };
        line('gray', C.ink, [4, 3], 1.3);
        if (blueOn) line('blue', C.blue, null, 2);
        T_(g, 'easy · lunch', G.L, hh - 13, { size: 10, kind: 'mono', color: C.muted });
        T_(g, 'career change · hard', G.Rr, hh - 13, { size: 10, kind: 'mono', color: C.muted, align: 'right' });
      }

      const NG = 64, grids = {};
      function gridFor(k) {
        if (grids[k]) return grids[k];
        const T = TASKS[k], E = [];
        for (let r = 0; r < NG; r++) { const yv = EXT - 2 * EXT * r / (NG - 1), rowv = []; for (let c = 0; c < NG; c++) rowv.push(energy(T, [-EXT + 2 * EXT * c / (NG - 1), yv])); E.push(rowv); }
        return (grids[k] = { E, q: lib.quantile(E, 0.6) });
      }
      const toPx = (y, w) => [(y[0] + EXT) / (2 * EXT) * w, (EXT - y[1]) / (2 * EXT) * w];
      const fromPx = (px, py, w) => [px / w * 2 * EXT - EXT, EXT - py / w * 2 * EXT];
      function drawLand(g, w) {
        if (!cur) return;
        const Gd = gridFor(S.task);
        lib.heatmap(g, Gd.E, 0, 0, w, w, { key: 's2-' + S.task + '-' + w, range: [0, Gd.q], gamma: 0.8, alpha: 0.55 });
        const lv = Array.from({ length: 9 }, (_, i) => Gd.q * Math.pow((i + 1) / 10, 1.6));
        lib.contours(g, Gd.E, 0, 0, w, w, lv, { color: 'rgba(17,17,17,0.2)', width: 1 });
        const [sx, sy] = toPx(YSTAR, w), rp = R_TOL / (2 * EXT) * w;
        g.save(); g.strokeStyle = C.ink; g.lineWidth = 1; g.setLineDash([3, 3]); g.beginPath(); g.arc(sx, sy, Math.max(rp, 5), 0, 7); g.stroke(); g.setLineDash([]);
        g.beginPath(); g.moveTo(sx - 9, sy); g.lineTo(sx + 9, sy); g.moveTo(sx, sy - 9); g.lineTo(sx, sy + 9); g.stroke(); g.restore();
        T_(g, 'answer y*', sx + 10, sy + 6, { size: 10.5, kind: 'mono', color: C.ink });
        const drawPath = (r, col, lw, dash, filled) => {
          const n = Math.min(S.tick, r.n), px = r.path.slice(0, n + 1).map(p => toPx(p, w));
          if (px.length > 1) lib.line(g, px, { color: col, width: lw, dash });
          px.forEach((p, i) => { if (i === 0) return; lib.dot(g, p[0], p[1], i === px.length - 1 ? 4 : 1.8, filled ? col : '#fff', { stroke: col, lw: 1.1 }); });
        };
        drawPath(cur.gray, C.ink, 1.3, [4, 3], false);
        if (S.showBlue) drawPath(cur.blue, C.blue, 2, null, true);
        const p0 = toPx(S.y0, w); g.save(); g.strokeStyle = C.ink; g.lineWidth = 1.2; g.fillStyle = '#fff'; g.beginPath(); g.arc(p0[0], p0[1], 5, 0, 7); g.fill(); g.stroke(); g.restore();
        T_(g, 'ŷ₀', p0[0] - 6, p0[1] + 8, { size: 12, kind: 'mono', color: C.ink, align: 'right' });
        const T = TASKS[S.task], tag = `task ${S.task + 1} · κ ${T.kap.toFixed(1)}` + (S.showDom ? ' · ' + T.dom : '');
        g.font = lib.font(10.5, 'mono'); const tw = g.measureText(tag).width; g.fillStyle = 'rgba(255,255,255,0.9)'; g.fillRect(4, 4, tw + 10, 17);
        T_(g, tag, 9, 7, { size: 10.5, kind: 'mono', color: C.ink });
      }

      function drawRes(R, g, w, hh) {
        const P = plotById(R.id); if (!P) { T_(g, 'data/scaling.json missing', 8, 8, { size: 11, kind: 'mono', color: C.bad }); return; }
        const L = 34, Rr = w - 10, top = 22, bot = hh - 28;
        const logx = R.id !== 'fig4a', logy = R.id === 'fig12';
        const xl = R.id === 'fig4a' ? [0.2, 7.2] : R.id === 'fig6a' ? [1.7, 36] : [0.75, 420];
        const yl = R.id === 'fig4a' ? [36, 76] : R.id === 'fig6a' ? [30.5, 46] : [13, 25];
        const fx = (v) => logx ? Math.log10(v) : v, fy = (v) => logy ? Math.log10(v) : v;
        const X = (v) => L + (fx(v) - fx(xl[0])) / (fx(xl[1]) - fx(xl[0])) * (Rr - L);
        const Y = (v) => bot - (fy(v) - fy(yl[0])) / (fy(yl[1]) - fy(yl[0])) * (bot - top);
        const xt = R.id === 'fig4a' ? [1, 2, 3, 4, 5, 6, 7] : R.id === 'fig6a' ? [2, 3, 6, 15, 30] : [1, 2, 3, 10, 100, 200, 300];
        const yt = R.id === 'fig4a' ? [40, 50, 60, 70] : R.id === 'fig6a' ? [32, 36, 40, 44] : [14, 17, 20, 23];
        yt.forEach(v => { tick(g, L, Y(v), Rr, Y(v), C.rule, 1); T_(g, String(v), L - 5, Y(v), { size: 10, kind: 'mono', color: C.muted, align: 'right', baseline: 'middle' }); });
        tick(g, L, bot + 0.5, Rr, bot + 0.5, C.ink, 1);
        xt.forEach(v => T_(g, String(v), X(v), bot + 5, { size: 10, kind: 'mono', color: C.muted, align: 'center' }));
        const xlab = R.id === 'fig4a' ? 'training tokens (billions)' : R.id === 'fig6a' ? 'forward passes per token (log)' : 'forward passes (log)';
        T_(g, xlab, Rr, bot + 17, { size: 10, kind: 'mono', color: C.muted, align: 'right' });
        const ylab = R.id === 'fig4a' ? 'validation perplexity, lower is better' : R.id === 'fig6a' ? 'perplexity increase on OOD data, lower is better' : 'PSNR in dB (log axis, as in the paper), higher is better';
        T_(g, ylab, L - 30, 3, { size: 10, kind: 'mono', color: C.muted });
        P.series.forEach(s => {
          const isE = /EBT/.test(s.model), col = isE ? C.blue : C.ink;
          const pts = s.points.map(p => [X(p[0]), Y(p[1])]);
          if (R.id === 'fig4a' && s.fit) { // the straight log-log fit behind the printed 35.98% (slope ratio − 1)
            const fpts = []; for (let i = 0; i <= 50; i++) { const xv = 0.45 + (7.1 - 0.45) * i / 50; fpts.push([X(xv), Y(Math.pow(10, s.fit.intercept) * Math.pow(xv, s.fit.slope))]); }
            lib.line(g, fpts, { color: col, width: isE ? 1.6 : 1.2, dash: isE ? null : [4, 3] });
          } else lib.line(g, pts, { color: col, width: isE ? 1.6 : 1.2, dash: isE ? null : [4, 3] });
          pts.forEach(p => lib.dot(g, p[0], p[1], 2.6, isE ? col : '#fff', { stroke: col, lw: 1 }));
        });
        const lab = (s, x, y, col, al) => T_(g, s, x, y, { size: 10.5, kind: 'mono', color: col, align: al || 'left' });
        if (R.id === 'fig4a') {
          const fit = (s, x) => Math.pow(10, s.fit.intercept) * Math.pow(x, s.fit.slope);
          const Tp = P.series.find(s => !/EBT/.test(s.model)), Eb = P.series.find(s => /EBT/.test(s.model));
          void fit;
          lab('log-log slope of the fit', Rr, Y(74), C.muted, 'right');
          lab('EBT  ' + Eb.fit.slope.toFixed(4), Rr, Y(74) + 14, C.blue, 'right');
          lab('Transformer++  ' + Tp.fit.slope.toFixed(4), Rr, Y(74) + 28, C.ink, 'right');
          lab(Math.abs(Eb.fit.slope).toFixed(4) + ' / ' + Math.abs(Tp.fit.slope).toFixed(4) + ' − 1 = ' + (100 * (Eb.fit.slope / Tp.fit.slope - 1)).toFixed(2) + '%', Rr, Y(74) + 42, C.ink, 'right');
        }
        if (R.id === 'fig6a') { lab('Transformer++, flat', X(30), Y(38.41) - 15, C.ink, 'right'); lab('EBT, no thinking', X(2) + 7, Y(44.77) - 4, C.blue); lab('+1 step', X(3) + 7, Y(35.89) - 15, C.blue); lab('Best-of-N', X(30), Y(31.76) - 15, C.blue, 'right'); }
        if (R.id === 'fig12') { lab('EBT: 1, 2, 3 passes', X(3) + 7, Y(23.29) - 4, C.blue); lab('DiT: 100, 200, 300', X(100) - 6, Y(14.31) - 6, C.ink, 'right'); }
      }

      // ---------- readouts ----------
      function readouts() {
        const tg = stats.reduce((a, s) => a + s.gray.steps, 0), tb = stats.reduce((a, s) => a + s.blue.steps, 0);
        const rg = stats.reduce((a, s) => a + s.gray.rate, 0) / NT, rb = stats.reduce((a, s) => a + s.blue.rate, 0) / NT;
        roA.innerHTML = `<span><i class="s2-k gray"></i>System 1, K = ${S.K}</span>` + (S.showBlue ? `<span><i class="s2-k blue"></i>System 2, ${MODES[S.mode].label}</span>` : '') + `<span><i class="s2-k tickk"></i>needed</span>` +
          `<span>all 8 tasks: gray <b>${tg.toFixed(0)}</b> steps, <b>${Math.round(rg * 100)}%</b> solved` + (S.showBlue ? ` · blue <b>${tb.toFixed(0)}</b> steps, <b>${Math.round(rb * 100)}%</b>` : '') + `</span>`;
        const f = (r, nm) => `<span><i class="s2-k ${nm}"></i>${r.n} step${r.n === 1 ? '' : 's'} · ${r.hit ? '<b>solved</b>' : dist(r.path[r.path.length - 1]).toFixed(2) + ' from y*'} · ${REASON[r.reason]}</span>`;
        roB.innerHTML = f(cur.gray, 'gray') + (S.showBlue ? f(cur.blue, 'blue') : '');
      }

      // ---------- animation ----------
      let timer = null;
      const stopAnim = () => { if (timer) clearTimeout(timer); timer = null; };
      function animate() {
        stopAnim(); const N = Math.max(cur.gray.n, S.showBlue ? cur.blue.n : 0);
        if (lib.reducedMotion) { S.tick = 999; land.draw(); return; }
        S.tick = 0; land.draw();
        const go = () => { S.tick++; land.draw(); if (S.tick < N) timer = setTimeout(go, S.tick < 6 ? 120 : 50); else timer = null; };
        timer = setTimeout(go, 160);
      }
      function refresh(withStats, anim) {
        if (withStats) computeStats(); computeCurrent();
        if (!anim) S.tick = 999;
        dial.draw(); land.draw(); readouts();
        const tUsed = S.showBlue && (S.mode === 'checker' || S.mode === 'none'); slT.el.style.opacity = tUsed ? '' : '0.4';
        if (anim) animate();
      }
      dial.canvas.addEventListener('click', (ev) => { const [px] = dial.toLocal(ev), G = dialGeom(dial.w, dial.h); S.task = Math.max(0, Math.min(NT - 1, Math.floor((px - G.L) / G.sw))); refresh(false, true); });
      dial.canvas.tabIndex = 0; dial.canvas.setAttribute('aria-label', dial.canvas.getAttribute('aria-label') + ' Use the left and right arrow keys to choose a task.');
      dial.canvas.addEventListener('keydown', (ev) => { const d = ev.key === 'ArrowLeft' ? -1 : ev.key === 'ArrowRight' ? 1 : 0; if (!d) return; ev.preventDefault(); ev.stopPropagation(); S.task = Math.max(0, Math.min(NT - 1, S.task + d)); refresh(false, true); });
      land.canvas.addEventListener('click', (ev) => { const [px, py] = land.toLocal(ev); S.y0 = fromPx(px, py, land.w).map(v => Math.max(-EXT + 0.05, Math.min(EXT - 0.05, v))); refresh(false, true); });
      ctx.setCaption('Bars: steps each model spends on eight tasks of rising difficulty κ; black ticks: steps the task needs; lines: share of 24 random starts that end within tolerance. Hatched: steps from a budget fixed in advance, spent beyond what was needed. Right: the selected task, its answer (crosshair; dashed circle = tolerance) and both paths from one start.');

      const setCtl = (o) => { Object.assign(S, o); seg.set(S.mode); slK.set(S.K); slT.set(S.T); };
      computeStats(); computeCurrent(); readouts();
      const kick = () => canvases.forEach(c => c.draw());
      kick(); if (window.EBTV && EBTV.fontsReady) EBTV.fontsReady.then(kick);
      const CFG = [
        { mode: 'key', task: 6, blue: false, dom: false },
        { mode: 'key', task: 6, blue: true, dom: false },
        { mode: 'checker', task: 7, blue: true, dom: true },
        { mode: 'none', task: 7, blue: true, dom: true },
        { mode: 'energy', task: 7, blue: true, dom: true },
      ];
      return {
        step(i) {
          stopAnim(); const c = CFG[i]; S.y0 = Y0.slice(); startIdx = -1;
          setCtl({ mode: c.mode, K: 3, T: 12, task: c.task }); S.showBlue = c.blue; S.showDom = c.dom;
          refresh(true, ctx.visible());
        },
        show() { kick(); },
        hide() { stopAnim(); if (S.tick < 999) { S.tick = 999; land.draw(); } },
      };
    },
  });
})();
