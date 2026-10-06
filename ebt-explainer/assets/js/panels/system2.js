/* Panel: System 1 vs System 2, today's recipes, the paper's core question, and its answer.
   Figure: a "difficulty dial". Eight tasks, each the minimum of a stretched quadratic valley E_k(ŷ) = ½(u²/κ_k + v²).
   Gray (System 1) takes a fixed K gradient steps on every task. Blue (System 2) keeps stepping until a stop signal fires:
   an answer key, an external checker that exists only for math and code, no check at all (a budget T fixed in advance),
   or its own energy settling. Everything is computed live (8 tasks × 24 random starts). The last step shows the paper's
   three headline results from the digitized figures in data/scaling.json. */
(function () {
  'use strict';
  // ---------------- toy math (illustration, not a trained model) ----------------
  const NT = 8, KMAX = 12, R_TOL = 0.15, EPS_E = 1e-4, CAP = 40, NSTART = 24, EXT = 3;
  const YSTAR = [0.8, -0.4];
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
  const REASON = { checked: 'stopped: answer checked', settled: 'stopped: energy settled', budget: 'stopped: budget used up', cap: 'stopped: 40-step cap' };

  EBT.panel({
    id: 'system2',
    nav: 'System 1, System 2',
    title: 'Thinking fast, thinking slow, and the question this paper asks',
    lede: 'Some predictions deserve more thought than others. Today\'s models either spend the same compute on every token, or need a reward, an answer checker or a second model to think harder. The paper asks whether a model can learn to think harder from unlabeled data alone.',
    text: `
      <p>Psychology splits thinking into two modes (p.1). <b>System 1</b> is fast, intuitive and automatic. <b>System 2</b> is slow, deliberate and analytical: the mode for "mathematics, programming, multistep reasoning, or novel out-of-distribution situations" (p.2). The paper's example: deciding what to eat for lunch takes a moment, deciding whether to change careers takes much longer (p.3).</p>
      <p>The paper means this per <em>prediction</em>, for example per token (footnote 3, p.3). A feed-forward Transformer has fixed depth and width, so the word "the" and the last digit of a long sum get exactly the same forward pass.</p>
      <p>The figure makes "difficulty" precise with a toy. Each task asks for the minimum $y^*$ of a stretched valley</p>
      <div class="eq">$$E_\\kappa(\\hat y) = \\tfrac12\\Big(\\frac{u^2}{\\kappa} + v^2\\Big),$$<span class="why">u, v: coordinates of ŷ − y* along and across the valley. κ ≥ 1: how stretched it is (the condition number).</span></div>
      <p>One gradient step with step size 1 removes the error across the valley and shrinks the error along it by $1-1/\\kappa$. Reaching tolerance $r$ from an initial error $u_0$ therefore takes</p>
      <div class="eq">$$n(\\kappa) = \\Big\\lceil \\frac{\\ln(r/|u_0|)}{\\ln(1-1/\\kappa)} \\Big\\rceil \\;\\approx\\; \\kappa \\ln\\frac{|u_0|}{r}$$<span class="why">steps grow linearly with difficulty, so any fixed budget K fails once n(κ) &gt; K</span></div>`,
    steps: [
      { label: 'System 1: the same budget for every task', html: '<p>Gray is a System 1 model. It takes $K = 3$ steps on every task, the way a Transformer spends one forward pass on every token. Black ticks mark the steps each task actually needs. Gray solves the first two tasks and misses most starts on the rest. Drag the K slider: a bigger K rescues harder tasks but charges every easy task the same price.</p>' },
      { label: 'System 2: spend where it is needed', html: '<p>Blue keeps stepping until its guess is within tolerance (cap 40). Its bars follow the black ticks, from 1 step at $\\kappa = 1$ to about 21 at $\\kappa = 12$, and it solves every task. But it cheats: it stops by comparing against the <b>answer key</b>. Real thinking needs its own way to know when to stop.</p>' },
      { label: 'Today\'s recipes bring the check from outside', html: '<p>Reasoning models such as o1, R1, Grok3 and Claude 3.7 Sonnet are trained with RL on rewards that a rule can verify, which "only works in domains where rule-based rewards can easily verify answers, such as math and coding". It "often deteriorates performance in other tasks such as writing" and "may not induce new reasoning patterns" (p.2). Best-of-N needs a reward model or external verifier, which means a second model and extra supervision (p.3, p.8). Chain-of-thought lives in discrete tokens and does not "seamlessly transfer to continuous modalities" (p.16). In the figure, the checker now exists only for math and code; essays, images and video fall back to a budget fixed in advance ($T = 12$).</p>' },
      { label: 'The question', html: '<p>Remove the outside checker entirely. Blue must now fix $T$ before it starts: 12 steps wasted on the easiest task, too few from $\\kappa \\approx 6$ on. The paper\'s core question (p.2) is whether the stop-and-choose signal itself can be learned:</p><p class="s2-quote">"Can we rely entirely on unsupervised learning to develop System 2 Thinking?"</p><p>That would extend System 2 Thinking "to any problem, any modality, and avoid the reliance on external human, reward, or model supervision" (p.2).</p>' },
      { label: 'The answer: learn the verifier, then optimize against it', html: '<p>The paper\'s answer is yes, "by learning to explicitly verify the compatibility between inputs and candidate-predictions, and then re-framing prediction problems as optimization with respect to this verifier" (p.1). Blue now stops when its <b>own energy</b> stops falling ($E_i - E_{i+1} < 10^{-4}$). It solves every task with almost the answer key\'s allocation, paying a few extra steps to confirm convergence. The figure adds the three headline results.</p>' },
    ],
    after: `
      <h3>The three headline numbers, read carefully</h3>
      <ul>
        <li><b>35%</b> faster scaling is the best of six axes: data, 35.98% (Fig 4a). Batch size gives 28.46%, depth 5.29%, parameters 2.91%, FLOPs 2.92%, width 0.02% (Figs 4, 5). "Faster" means a steeper log-log slope. EBT perplexity is lower than Transformer++ only on the data and batch-size axes, past a crossover (Figs 4a, 4b).</li>
        <li><b>29%</b> is measured from EBT's own no-thinking point (Fig 6a, approx. 44.8 → 31.8). That point is worse than Transformer++ (≈38.4); one extra step already beats it. The Transformer++ line is flat because more passes cannot change its prediction.</li>
        <li><b>99% fewer forward passes</b> means 3 EBT passes against 300 DiT passes for image denoising (p.35). EBT is clearly ahead only at 3 vs 300; at 1 vs 100 and 2 vs 200 the two are about equal (Fig 12). Each EBT pass also needs a backward pass, so the compute saving is smaller than 100×.</li>
      </ul>
      <p class="note">The difficulty dial is an illustration with a hand-written energy, not a trained model: its "own energy" is exact, and only the stopping rule can mislead it. A learned energy can also be wrong. "Thinking" in this paper means refining one continuous prediction step by step, not chain-of-thought; at this model size chain-of-thought did not help (footnote 9, p.10).</p>`,
    source: [{ kind: 'concept', note: 'difficulty dial: hand-written energies, real gradient descent' }, { kind: 'paper', note: 'p.1–3, p.16; Figs 4a, 6a, 12 digitized (approx.)' }],

    figure(stage, ctx) {
      const { lib } = ctx, h = lib.h, C = lib.C, T_ = lib.text;
      const S = { mode: 'key', showBlue: false, showDom: false, K: 3, T: 12, task: 6, y0: null, tick: 999, swap: null };
      const rs = lib.rng(3); const STARTS = Array.from({ length: NSTART }, () => [rs.normal() * 1.3, rs.normal() * 1.3]);
      let startIdx = 0; S.y0 = [-2.1, 1.6];
      const canvases = [];
      // auto-sized canvas: logical width = CSS width, so text renders at its true size on every screen
      function autoCanvas(parent, o) {
        const box = h('div', { class: 'canvas-box' }); parent.appendChild(box);
        const c = h('canvas', { role: 'img', 'aria-label': o.label || '' }); box.appendChild(c);
        const g = c.getContext('2d'); const st = { w: 0, h: 0, ctx: g, canvas: c, box };
        const resize = () => { const w = Math.max(200, Math.round(box.clientWidth || 0)); if (!box.clientWidth) return false; const hh = Math.round(o.height(w)), dpr = Math.min(2, window.devicePixelRatio || 1); if (w === st.w && hh === st.h) return false; st.w = w; st.h = hh; c.width = Math.round(w * dpr); c.height = Math.round(hh * dpr); c.style.height = hh + 'px'; g.setTransform(dpr, 0, 0, dpr, 0, 0); return true; };
        st.draw = () => { resize(); if (!st.w) return; g.clearRect(0, 0, st.w, st.h); o.draw(g, st.w, st.h); };
        st.toLocal = (ev) => { const r = c.getBoundingClientRect(); return [(ev.clientX - r.left) / r.width * st.w, (ev.clientY - r.top) / r.height * st.h]; };
        if (window.ResizeObserver) new ResizeObserver(() => { if (resize()) st.draw(); }).observe(box);
        canvases.push(st); return st;
      }

      // ---------- statistics over the 24 starts ----------
      let stats = null;
      const bluePol = (T) => MODES[S.mode].pol(T, S);
      function computeStats() {
        stats = TASKS.map(T => {
          const key = STARTS.map(y => run(T, y, { kind: 'key' })), gr = STARTS.map(y => run(T, y, { kind: 'fixed', n: S.K })), bl = STARTS.map(y => run(T, y, bluePol(T)));
          const mean = (a, f) => a.reduce((s, r) => s + f(r), 0) / a.length;
          return { needed: mean(key, r => r.n), gray: { steps: mean(gr, r => r.n), rate: mean(gr, r => r.hit ? 1 : 0) }, blue: { steps: mean(bl, r => r.n), rate: mean(bl, r => r.hit ? 1 : 0), fixed: bluePol(T).kind === 'fixed' } };
        });
      }
      let cur = null;
      function computeCurrent() { const T = TASKS[S.task]; cur = { gray: run(T, S.y0, { kind: 'fixed', n: S.K }), blue: run(T, S.y0, bluePol(T)) }; }

      // ---------- layout ----------
      const row = h('div', { class: 'fig-row' }); stage.appendChild(row);
      const A = lib.frame(row, { label: 'Difficulty dial', sub: 'steps per task (mean of 24 starts) · click a task' });
      A.wrap.style.flex = '1 1 330px';
      const dial = autoCanvas(A.frame, { height: (w) => Math.round(Math.max(300, Math.min(340, w * 0.95))), label: 'Steps spent and share of starts solved for eight tasks of rising difficulty, for a fixed-budget model and a thinking model', draw: drawDial });
      const roA = h('div', { class: 'readout s2-ro' }); A.wrap.appendChild(roA);
      const B = lib.frame(row, { label: 'One task, up close', sub: 'E(ŷ) over ŷ ∈ ℝ² · click to move ŷ<sub>0</sub>' });
      B.wrap.style.flex = '1 1 230px';
      const land = autoCanvas(B.frame, { height: (w) => w, label: 'Energy landscape of the selected task with both models\' gradient-descent paths', draw: drawLand });
      const roB = h('div', { class: 'readout s2-ro s2-roB' }); B.wrap.appendChild(roB);

      const controls = h('div', { class: 'controls' }); stage.appendChild(controls);
      controls.appendChild(h('span', { class: 'fig-label' }, 'blue stops when'));
      const seg = lib.segmented({ label: 'Stop signal of the thinking model', options: Object.entries(MODES).map(([k, m]) => [k, m.label]), value: S.mode, onchange: (v) => { S.mode = v; S.showBlue = true; S.showDom = S.showDom || v === 'checker'; refresh(true); } });
      controls.appendChild(seg.el);
      const controls2 = h('div', { class: 'controls' }); stage.appendChild(controls2);
      const slK = lib.slider({ id: 's2-K', label: 'gray budget K', min: 1, max: 16, step: 1, value: S.K, oninput: (v) => { S.K = v; refresh(true); } });
      const slT = lib.slider({ id: 's2-T', label: 'blue budget T when it cannot check', min: 2, max: 40, step: 1, value: S.T, oninput: (v) => { S.T = v; refresh(true); } });
      controls2.append(slK.el, slT.el, lib.button('new start', () => { startIdx = (startIdx + 1) % NSTART; S.y0 = STARTS[startIdx].slice(); refresh(true, true); }), lib.button('replay', () => animate()));

      // swap area (steps 3–5)
      const swapWrap = h('div', { class: 's2-swap' }); stage.appendChild(swapWrap);
      const SW = lib.frame(swapWrap, { label: '·', sub: '·' });
      const swLabel = SW.wrap.querySelector('.fig-label'), swSub = SW.wrap.querySelector('.fig-sub');
      const panes = {};
      // recipes pane
      panes.recipes = h('div', { class: 's2-recipes' });
      const RECIPES = [
        { n: 'RL with verifiable rewards', e: 'o1, R1, Grok3, Claude 3.7 Sonnet', pipe: ['pretrain', '+ RL on a rule-checked reward'], lim: 'math and code only; "often deteriorates performance in other tasks such as writing" (p.2)' },
        { n: 'Best-of-N with a verifier', e: 'reward model or external checker', pipe: ['pretrain', '+ train a second model on extra labels'], lim: 'a second model and extra supervision (p.3, p.8)' },
        { n: 'Chain-of-thought', e: 'more tokens before the answer', pipe: ['pretrain', '+ reasoning tokens'], lim: 'discrete text; does not "seamlessly transfer to continuous modalities" (p.16)' },
        { n: 'Diffusion, more steps', e: 'denoising schedule', pipe: ['pretrain', '+ longer schedule'], lim: 'no gain beyond trained steps; needs an external verifier (p.2)' },
        { n: 'EBT (this paper)', e: 'learned energy E(x, ŷ)', pipe: ['pretrain on unlabeled data'], lim: 'the question in step 4', ebt: true },
      ];
      panes.recipes.appendChild(h('div', { class: 's2-rh' }, h('span', {}, 'recipe'), h('span', {}, 'where the check comes from'), h('span', {}, 'limit (paper)')));
      RECIPES.forEach(r => panes.recipes.appendChild(h('div', { class: 's2-r' + (r.ebt ? ' ebt' : '') },
        h('div', { class: 's2-rn' }, h('b', {}, r.n), h('span', {}, r.e)),
        h('div', { class: 's2-pipe' }, r.pipe.map((p, i) => h('span', { class: 's2-box' + (i > 0 ? ' extra' : '') + (r.ebt ? ' ebt' : '') }, p))),
        h('div', { class: 's2-lim' }, r.lim))));
      // question pane
      panes.question = h('div', { class: 's2-question' },
        h('p', { class: 's2-big' }, '“Can we rely entirely on unsupervised learning to develop System 2 Thinking?”'),
        h('div', { class: 's2-need' }, [['any problem', 'not only math and code'], ['any modality', 'text, images, video'], ['no extra supervision', 'no reward, no verifier model, no labels']].map(([a, b]) => h('div', {}, h('b', {}, a), h('span', {}, b)))),
        h('p', { class: 's2-qsrc' }, 'core research question, p.2 · the abstract asks the same in other words, p.1'));
      // results pane
      panes.results = h('div', { class: 's2-results' });
      const sc = lib.data('scaling'); const plotById = (id) => sc && sc.plots ? sc.plots.find(p => p.id === id) : null;
      const RES = [
        { id: 'fig4a', big: '35.98%', t: 'faster data scaling', fig: 'Fig 4a', note: 'Best of six axes; params 2.91%, FLOPs 2.92%, width 0.02%. EBT starts worse and crosses near 3B tokens.' },
        { id: 'fig6a', big: '29%', t: 'gain from thinking, text', fig: 'Fig 6a', note: 'Measured from EBT\'s own no-thinking point, which is worse than Transformer++. Lower is better.' },
        { id: 'fig12', big: '3 vs 300', t: 'forward passes, denoising', fig: 'Fig 12', note: 'EBT is clearly ahead only at 3 vs 300 passes. Each EBT pass also needs a backward pass.' },
      ];
      const resCv = [];
      RES.forEach(R => {
        const col = h('div', { class: 's2-res' }); panes.results.appendChild(col);
        col.appendChild(h('div', { class: 's2-resh' }, h('b', {}, R.big), h('span', {}, R.t + ' · ' + R.fig)));
        resCv.push(autoCanvas(col, { height: (w) => Math.round(Math.max(130, Math.min(190, w * 0.68))), label: R.fig + ' digitized: ' + R.t, draw: (g, w, hh) => drawRes(R, g, w, hh) }));
        col.appendChild(h('p', { class: 's2-resn' }, R.note));
      });
      const answer = h('p', { class: 's2-answer', html: '<b>The answer (p.1):</b> learn to verify the compatibility of an input and a candidate prediction, then predict by optimizing against that verifier. Thinking becomes gradient descent on a learned energy.' });
      panes.results.prepend(answer);
      Object.values(panes).forEach(p => { p.style.display = 'none'; SW.frame.appendChild(p); });
      function setSwap(name) {
        S.swap = name; swapWrap.style.display = name ? '' : 'none';
        Object.entries(panes).forEach(([k, p]) => { p.style.display = k === name ? '' : 'none'; });
        const lab = { recipes: ['Where today\'s recipes get their check', 'concept · our summary of p.2, p.3, p.8, p.16'], question: ['The core question', 'p.2, verbatim'], results: ['Headline results, with context', 'from the paper · digitized, approx.'] }[name];
        if (lab) { swLabel.textContent = lab[0]; swSub.innerHTML = lab[1]; }
        if (name === 'results') resCv.forEach(c => c.draw());
      }

      // ---------- drawing helpers ----------
      const tick = (g, x1, y1, x2, y2, col, lw, dash) => { g.save(); g.strokeStyle = col; g.lineWidth = lw || 1; if (dash) g.setLineDash(dash); g.beginPath(); g.moveTo(x1, y1); g.lineTo(x2, y2); g.stroke(); g.restore(); };
      function hatch(g, x, y, w, hh, col) { g.save(); g.beginPath(); g.rect(x, y, w, hh); g.clip(); g.strokeStyle = col; g.lineWidth = 1; for (let k = -hh; k < w + hh; k += 4) { g.beginPath(); g.moveTo(x + k, y + hh); g.lineTo(x + k + hh, y); g.stroke(); } g.restore(); g.strokeStyle = col; g.lineWidth = 1; g.strokeRect(x + 0.5, y + 0.5, Math.max(0, w - 1), Math.max(0, hh - 1)); }

      function dialGeom(w, hh) {
        const L = 34, Rr = w - 6, top0 = 20, top1 = Math.round(hh * 0.47), lab0 = top1 + 6;
        const bot0 = top1 + (S.showDom ? 62 : 46), bot1 = hh - 22;
        const sw = (Rr - L) / NT;
        return { L, Rr, top0, top1, lab0, bot0, bot1, sw, cx: (k) => L + sw * (k + 0.5) };
      }
      function drawDial(g, w, hh) {
        if (!stats) return;
        const G = dialGeom(w, hh), blueOn = S.showBlue;
        let ymax = Math.max(...stats.map(s => s.needed), S.K, blueOn ? Math.max(...stats.map(s => s.blue.steps)) : 0);
        ymax = Math.max(10, Math.ceil(ymax / 10) * 10);
        const Y = (v) => G.top1 - (v / ymax) * (G.top1 - G.top0);
        // selected column
        g.fillStyle = C.blue4; g.fillRect(G.L + G.sw * S.task + 1, G.top0 - 14, G.sw - 2, G.bot1 - G.top0 + 14);
        // top axis
        T_(g, 'steps spent', G.L - 30, 2, { size: 10.5, kind: 'mono', color: C.muted });
        [0, ymax / 2, ymax].forEach(v => { tick(g, G.L, Y(v), G.Rr, Y(v), C.rule, 1); T_(g, String(v), G.L - 6, Y(v), { size: 10, kind: 'mono', color: C.muted, align: 'right', baseline: 'middle' }); });
        tick(g, G.L, G.top1 + 0.5, G.Rr, G.top1 + 0.5, C.ink, 1);
        const bw = Math.max(5, Math.min(13, G.sw * 0.3));
        stats.forEach((s, k) => {
          const cx = G.cx(k);
          // gray bar: fixed K
          const gx = blueOn ? cx - bw - 1 : cx - bw / 2;
          g.fillStyle = '#d4d4d8'; g.fillRect(gx, Y(s.gray.steps), bw, G.top1 - Y(s.gray.steps)); g.strokeStyle = C.ink; g.lineWidth = 0.8; g.strokeRect(gx + 0.4, Y(s.gray.steps) + 0.4, bw - 0.8, G.top1 - Y(s.gray.steps) - 0.8);
          if (blueOn) {
            const bx = cx + 1, bs = s.blue.steps;
            if (s.blue.fixed) { // budget fixed in advance: solid up to what was needed, hatched beyond (waste)
              const used = Math.min(bs, s.needed);
              hatch(g, bx, Y(bs), bw, G.top1 - Y(bs), C.blue);
              g.fillStyle = C.blue; g.fillRect(bx, Y(used), bw, G.top1 - Y(used));
            } else { g.fillStyle = C.blue; g.fillRect(bx, Y(bs), bw, G.top1 - Y(bs)); }
          }
          // needed tick
          const ny = Y(s.needed); tick(g, cx - bw - 5, ny, cx + bw + 5, ny, C.ink, 1.6);
        });
        // labels under the bars
        stats.forEach((s, k) => {
          const cx = G.cx(k), T = TASKS[k];
          T_(g, 'κ ' + (T.kap < 10 ? T.kap.toFixed(1) : T.kap.toFixed(0)), cx, G.lab0, { size: 10, kind: 'mono', color: k === S.task ? C.ink : C.muted, align: 'center', weight: k === S.task ? 700 : 400 });
          if (S.showDom) {
            const chk = CHECKABLE(T.dom);
            T_(g, T.dom, cx, G.lab0 + 14, { size: 10, kind: 'mono', color: chk ? C.ink : C.muted, align: 'center' });
            if (S.mode === 'checker') T_(g, chk ? 'check' : 'none', cx, G.lab0 + 27, { size: 9.5, kind: 'mono', color: chk ? C.blue : C.faint, align: 'center' });
          }
        });
        // bottom: solved rate
        const Yb = (v) => G.bot1 - v * (G.bot1 - G.bot0);
        T_(g, 'solved (share of 24 starts)', G.L - 30, G.bot0 - 15, { size: 10.5, kind: 'mono', color: C.muted });
        [0, 0.5, 1].forEach(v => { tick(g, G.L, Yb(v), G.Rr, Yb(v), C.rule, 1); T_(g, Math.round(v * 100) + '%', G.L - 6, Yb(v), { size: 10, kind: 'mono', color: C.muted, align: 'right', baseline: 'middle' }); });
        const line = (key, col, dash, lw) => { const pts = stats.map((s, k) => [G.cx(k), Yb(s[key].rate)]); lib.line(g, pts, { color: col, width: lw, dash }); pts.forEach(p => lib.dot(g, p[0], p[1], key === 'blue' ? 3.2 : 2.8, key === 'blue' ? col : '#fff', { stroke: col, lw: 1.3 })); };
        line('gray', C.ink, [4, 3], 1.3);
        if (blueOn) line('blue', C.blue, null, 2);
        T_(g, 'easy · lunch', G.L, hh - 13, { size: 10, kind: 'mono', color: C.muted });
        T_(g, 'career change · hard', G.Rr, hh - 13, { size: 10, kind: 'mono', color: C.muted, align: 'right' });
      }

      // landscape grids, one per task
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
          return px[px.length - 1];
        };
        const pg = drawPath(cur.gray, C.ink, 1.3, [4, 3], false);
        let pb = null; if (S.showBlue) pb = drawPath(cur.blue, C.blue, 2, null, true);
        const p0 = toPx(S.y0, w); g.save(); g.strokeStyle = C.ink; g.lineWidth = 1.2; g.fillStyle = '#fff'; g.beginPath(); g.arc(p0[0], p0[1], 5, 0, 7); g.fill(); g.stroke(); g.restore();
        T_(g, 'ŷ₀', p0[0] + 8, p0[1] - 16, { size: 12, kind: 'mono', color: C.ink });
        // task tag
        const T = TASKS[S.task], tag = `task ${S.task + 1} · κ ${T.kap.toFixed(1)}` + (S.showDom ? ' · ' + T.dom : '');
        g.font = lib.font(10.5, 'mono'); const tw = g.measureText(tag).width; g.fillStyle = 'rgba(255,255,255,0.9)'; g.fillRect(4, 4, tw + 10, 17);
        T_(g, tag, 9, 7, { size: 10.5, kind: 'mono', color: C.ink });
        void pg; void pb;
      }

      // ---------- headline result mini charts (digitized paper data) ----------
      function drawRes(R, g, w, hh) {
        const P = plotById(R.id); if (!P) { T_(g, 'scaling.json missing', 8, 8, { size: 11, kind: 'mono', color: C.bad }); return; }
        const L = 30, Rr = w - 8, top = 8, bot = hh - 26;
        const lx = (v) => P.x.log ? Math.log10(v) : v, ly = (v) => P.y.log ? Math.log10(v) : v;
        let xl = P.x.lim, yl = P.y.lim; if (R.id === 'fig6a') xl = [1.6, 38];
        const X = (v) => L + (lx(v) - lx(xl[0])) / (lx(xl[1]) - lx(xl[0])) * (Rr - L);
        const Y = (v) => bot - (ly(v) - ly(yl[0])) / (ly(yl[1]) - ly(yl[0])) * (bot - top);
        const lx6 = R.id === 'fig6a';
        const xt = R.id === 'fig4a' ? [1, 3, 5, 7] : R.id === 'fig6a' ? [2, 3, 6, 15, 30] : [1, 3, 10, 100, 300];
        const yt = R.id === 'fig4a' ? [40, 50, 60, 70] : R.id === 'fig6a' ? [32, 36, 40, 44] : [14, 17, 20, 23];
        if (lx6) { /* log-spaced categories */ }
        yt.forEach(v => { tick(g, L, Y(v), Rr, Y(v), C.rule, 1); T_(g, String(v), L - 4, Y(v), { size: 9.5, kind: 'mono', color: C.muted, align: 'right', baseline: 'middle' }); });
        tick(g, L, bot + 0.5, Rr, bot + 0.5, C.ink, 1);
        const X6 = (v) => lx6 ? L + (Math.log10(v) - Math.log10(xl[0])) / (Math.log10(xl[1]) - Math.log10(xl[0])) * (Rr - L) : X(v);
        xt.forEach(v => T_(g, String(v), X6(v), bot + 4, { size: 9.5, kind: 'mono', color: C.muted, align: 'center' }));
        const xlab = R.id === 'fig4a' ? 'tokens (B)' : R.id === 'fig6a' ? 'forward passes' : 'forward passes (log)';
        T_(g, xlab, Rr, bot + 15, { size: 9.5, kind: 'mono', color: C.muted, align: 'right' });
        const ylab = R.id === 'fig4a' ? 'val. perplexity ↓' : R.id === 'fig6a' ? 'ppl increase, OOD ↓' : 'PSNR ↑';
        T_(g, ylab, L + 4, top, { size: 9.5, kind: 'mono', color: C.muted });
        P.series.forEach(s => {
          const isE = /EBT/.test(s.model), col = isE ? C.blue : C.ink;
          const pts = s.points.map(p => [X6(p[0]), Y(p[1])]);
          if (R.id === 'fig4a' && s.fit) { // straight power-law fit from the paper's rate computation (log-log OLS)
            const fpts = []; for (let i = 0; i <= 40; i++) { const xv = 0.45 + (7 - 0.45) * i / 40; fpts.push([X(xv), Y(Math.pow(10, s.fit.intercept) * Math.pow(xv, s.fit.slope))]); }
            lib.line(g, fpts, { color: col, width: isE ? 1.6 : 1.2, dash: isE ? null : [4, 3] });
          } else lib.line(g, pts, { color: col, width: isE ? 1.6 : 1.2, dash: isE ? null : [4, 3] });
          pts.forEach(p => lib.dot(g, p[0], p[1], 2.4, isE ? col : '#fff', { stroke: col, lw: 1 }));
        });
        // labels
        const lab = (s, x, y, col, al) => T_(g, s, x, y, { size: 9.5, kind: 'mono', color: col, align: al || 'left' });
        if (R.id === 'fig4a') { lab('T++', X(0.55) + 4, Y(64.4) - 2, C.ink); lab('EBT', X(0.55) + 4, Y(73.7) - 4, C.blue); }
        if (R.id === 'fig6a') { lab('T++ flat', X6(30), Y(38.41) - 12, C.ink, 'right'); lab('EBT, no thinking', X6(2) + 6, Y(44.77) - 3, C.blue); lab('BoN', X6(30), Y(31.76) - 12, C.blue, 'right'); }
        if (R.id === 'fig12') { lab('EBT', X(3) + 5, Y(23.29) - 2, C.blue); lab('DiT', X(100) - 5, Y(14.31) - 2, C.ink, 'right'); }
      }

      // ---------- readouts ----------
      function readouts() {
        const tg = stats.reduce((a, s) => a + s.gray.steps, 0), tb = stats.reduce((a, s) => a + s.blue.steps, 0);
        const rg = stats.reduce((a, s) => a + s.gray.rate, 0) / NT, rb = stats.reduce((a, s) => a + s.blue.rate, 0) / NT;
        roA.innerHTML = `<span><i class="s2-k gray"></i>System 1 · K = ${S.K}</span>` + (S.showBlue ? `<span><i class="s2-k blue"></i>System 2 · ${MODES[S.mode].label}</span>` : '') + `<span><i class="s2-k tickk"></i>needed</span>` +
          `<span>all tasks: gray <b>${tg.toFixed(0)}</b> steps, solves <b>${Math.round(rg * 100)}%</b>` + (S.showBlue ? ` · blue <b>${tb.toFixed(0)}</b> steps, solves <b>${Math.round(rb * 100)}%</b>` : '') + `</span>`;
        const f = (r, nm) => `<span><i class="s2-k ${nm}"></i>${r.n} step${r.n === 1 ? '' : 's'} · ${r.hit ? '<b>solved</b>' : 'ends ' + dist(r.path[r.path.length - 1]).toFixed(2) + ' from y* (tolerance ' + R_TOL + ')'} · ${REASON[r.reason]}</span>`;
        roB.innerHTML = f(cur.gray, 'gray') + (S.showBlue ? f(cur.blue, 'blue') : '');
      }

      // ---------- animation ----------
      let timer = null;
      const stopAnim = () => { if (timer) clearTimeout(timer); timer = null; };
      function animate() {
        stopAnim(); const N = Math.max(cur.gray.n, S.showBlue ? cur.blue.n : 0);
        if (lib.reducedMotion) { S.tick = 999; land.draw(); return; }
        S.tick = 0; land.draw();
        const go = () => { S.tick++; land.draw(); if (S.tick < N) timer = setTimeout(go, S.tick < 6 ? 120 : 55); else timer = null; };
        timer = setTimeout(go, 200);
      }
      function refresh(stats_, anim) {
        if (stats_) computeStats(); computeCurrent();
        if (!anim) S.tick = 999;
        dial.draw(); land.draw(); readouts();
        const tUsed = S.showBlue && (S.mode === 'checker' || S.mode === 'none'); slT.el.style.opacity = tUsed ? '' : '0.4'; slT.el.title = tUsed ? '' : 'used only when blue cannot check (modes: checker, no check)';
        if (anim) animate();
      }
      // interaction
      dial.canvas.addEventListener('click', (ev) => { const [px] = dial.toLocal(ev), G = dialGeom(dial.w, dial.h); const k = Math.max(0, Math.min(NT - 1, Math.floor((px - G.L) / G.sw))); S.task = k; refresh(false, true); });
      land.canvas.addEventListener('click', (ev) => { const [px, py] = land.toLocal(ev); S.y0 = fromPx(px, py, land.w).map(v => Math.max(-EXT + 0.05, Math.min(EXT - 0.05, v))); refresh(false, true); });
      ctx.setCaption('Left: steps each model spends on eight tasks of rising difficulty κ (bars), steps a task actually needs (black ticks), and the share of 24 random starts that end within tolerance (lines). Hatched blue: a budget fixed in advance, beyond what was needed. Right: the selected task\'s energy (darker = lower), the answer (crosshair, dashed circle = tolerance) and both paths from the same start.');

      const setCtl = (o) => { Object.assign(S, o); seg.set(S.mode); slK.set(S.K); slT.set(S.T); };
      setSwap(null); computeStats(); computeCurrent(); readouts();
      const kick = () => canvases.forEach(c => c.draw());
      kick(); if (window.EBTV && EBTV.fontsReady) EBTV.fontsReady.then(kick);
      return {
        step(i) {
          stopAnim(); S.y0 = [-2.1, 1.6];
          if (i === 0) { setCtl({ mode: 'key', K: 3, T: 12, task: 6 }); S.showBlue = false; S.showDom = false; setSwap(null); }
          if (i === 1) { setCtl({ mode: 'key', K: 3, T: 12, task: 6 }); S.showBlue = true; S.showDom = false; setSwap(null); }
          if (i === 2) { setCtl({ mode: 'checker', K: 3, T: 12, task: 7 }); S.showBlue = true; S.showDom = true; setSwap('recipes'); }
          if (i === 3) { setCtl({ mode: 'none', K: 3, T: 12, task: 7 }); S.showBlue = true; S.showDom = true; setSwap('question'); }
          if (i === 4) { setCtl({ mode: 'energy', K: 3, T: 12, task: 7 }); S.showBlue = true; S.showDom = true; setSwap('results'); }
          refresh(true, true);
        },
        show() { kick(); },
        hide() { stopAnim(); if (S.tick < 999) { S.tick = 999; land.draw(); } },
      };
    },
  });
})();
