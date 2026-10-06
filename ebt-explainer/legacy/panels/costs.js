/* Panel: what EBTs cost (training FLOPs per the paper's D.5 accounting, inference per NFE), what that cost does to the
   equal-compute comparison (digitized Fig 5b), what thinking costs at inference (Fig 6a, Fig 12 re-plotted against an
   estimate of forward-pass equivalents), a budget calculator, and a map of the scales the paper actually tested. */
(function () {
  const SIZES = [ // Table D.1 (p.33), non-embedding params
    { id: 'xxs', P: 6.18e6 }, { id: 'xs', P: 12.4e6 }, { id: 'small', P: 48.8e6 },
    { id: 'medium', P: 176e6 }, { id: 'large', P: 396e6 }, { id: 'xl', P: 708e6 },
  ];
  const PRESETS = {
    fig5b: { label: 'Fig 5b largest run', size: 4, tokens: 256 * 256 * 105000, note: 'large, batch 256 × 256 tokens × 105k steps (p.34)' },
    think: { label: 'thinking runs', size: 0, tokens: 128 * 256 * 1e6, note: 'xxs, batch 128 × 256 tokens × 1M steps (p.34); these are S2 runs, whose FLOPs vary (p.36), so the S1 rule is only a guide' },
    llama: { label: 'Llama 3 data on xl', size: 5, tokens: 15e12, note: 'xl on 15T tokens, the data scale the paper cites (p.10)' },
  };
  const WARN = '#d4421c';
  const SUP = (s) => String(s).split('').map(ch => ({ '-': '⁻', '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹' })[ch] || ch).join('');
  const sci = (v, d = 2) => { if (!isFinite(v) || v <= 0) return '–'; const e = Math.floor(Math.log10(v)), m = v / Math.pow(10, e); return m.toFixed(d) + '×10' + SUP(e); };
  const tok = (v) => v >= 1e12 ? (v / 1e12).toFixed(v >= 1e13 ? 0 : 1) + 'T' : v >= 1e9 ? (v / 1e9).toFixed(v >= 1e11 ? 0 : 1) + 'B' : (v / 1e6).toFixed(0) + 'M';
  const trunc2 = (r) => Math.floor(r * 100 + 1e-9) / 100; // the paper truncates: 10/6 -> 1.66, 20/6 -> 3.33, 40/6 -> 6.66
  const ratioStr = (r) => trunc2(r).toFixed(2) + '×';
  const flops = (N) => N >= 1e9 ? (N / 1e9).toFixed(N >= 1e10 ? 1 : 2) + ' GFLOP' : (N / 1e6).toFixed(0) + ' MFLOP';

  EBT.panel({
    id: 'costs',
    nav: 'Costs and limits',
    title: 'What it costs, and what is still missing',
    lede: 'Thinking is not free. One EBT optimization step is a forward pass plus two backward-sized passes over a sequence twice as long, and every result in the paper was measured below 10²¹ training FLOPs. This section prices both and maps what remains untested.',
    text: `
      <p>The paper prices compute with the standard rule for dense Transformers (p.35, citing Casson's "Transformer FLOPs" [150]): per token, a forward pass costs about $2N$ FLOPs and a backward pass about $4N$, where $N$ is the number of non-embedding parameters. A Transformer++ training step therefore costs $6N$ per token.</p>
      <p>An EBT training step does three passes per optimization step: a forward pass for $E$, a backward pass for $\\nabla_{\\hat y}E$ kept in the graph, and a second backward pass that differentiates that gradient with respect to $\\theta$. The last one is a Hessian-vector product, which "has the same theoretical complexity as a gradient computation" (p.35, following [76]):</p>
      <div class="eq">$$\\underbrace{F}_{2N} + \\underbrace{B}_{4N} + \\underbrace{B_{\\text{HVP}}}_{4N} = 10N$$<span class="why">FLOPs per token, per optimization step. App. D.5 (p.35–36). The paper calls these estimates "approximate" and welcomes corrections (p.36).</span></div>
      <p>Two multipliers follow. The autoregressive layout doubles the sequence to $2S-2$ positions, and S1 pretraining takes $n$ optimization steps with a loss at each, so the costs add:</p>
      <div class="eq">$$\\frac{\\text{EBT}}{\\text{T++}} = \\frac{10N \\cdot 2 \\cdot n}{6N} \\approx 3.33\\,n$$<span class="why">$n$ = optimization steps (Algorithm 1 calls it $N$; here $N$ stays the parameter count). 1.66× for one step without the doubling (p.41), 3.33× per autoregressive step, 6.66× for the two-step pretraining runs (p.36). The paper truncates 10/6 to 1.66.</span></div>
      <p>The figure prices these passes per token on top, then shows what the price does to the paper's own comparisons.</p>`,
    steps: [
      { label: 'The yardstick: 6N per token', html: '<p>The gray bar is a Transformer++ step: $F = 2N$ plus $B = 4N$. On the chart are its five Fig 5b runs, xxs to large, each trained for 105k steps (p.34), with a power-law fit to the digitized points. A log-log least-squares fit reproduces the paper\'s printed 2.92% rate, so it is very likely the fit the paper used.</p>' },
      { label: 'One EBT step: F + B + B', html: '<p>Getting $\\nabla_{\\hat y}E$ is a backward pass; training through it needs a second one. $10N$ per token, 1.66× the baseline (p.41). On the chart, the blue points are a what-if: the paper\'s EBT perplexities as if each run had cost only 1.66× its Transformer++ twin. The real runs cost more and sit at the hollow circles.</p>' },
      { label: 'Twice the sequence', html: '<p>Context and guesses both pass through the blocks, $2S-2$ positions. Thanks to the C.3 mask this doubles the cost instead of quadrupling it (p.35–36): $20N$, 3.33× per step. The blue points slide further right.</p><p>Why "instead of quadrupling"? The $6N$ rule counts only weight multiplications, which grow linearly with positions anyway. The quadratic part is the attention scores: full $2S \\times 2S$ attention computes $4S^2$, the C.3 scheme about $2S^2$ (see <a href="#architecture">Inside an EBT</a>) [derived reading].</p>' },
      { label: 'Two steps: the 6.66× inside Fig 5b', html: '<p>S1 pretraining takes 2 steps with a loss at each, so costs add: $40N$, 6.66× (p.36). The blue points now sit exactly where the paper plots them. At every measured budget EBT perplexity is higher; its fitted slope is only 2.92% steeper. Extending both fits, they would meet near $10^{52}$ FLOPs [derived]. That is not a forecast. It shows how a small slope advantage fights a constant cost factor: set the steps to 1 and the meeting point moves ten orders of magnitude closer, still far beyond anything trained. S2 models vary: truncation saves FLOPs, extra steps and the replay buffer add them (p.36).</p>' },
      { label: 'Price a run', html: '<p>The calculator applies the same accounting to a model size from Table D.1 and a token count. The preset for the largest Fig 5b run (large, 6.88B tokens) returns 1.63×10¹⁹ FLOPs for the Transformer++ and 1.09×10²⁰ for the EBT, matching the plotted points. The Llama 3 preset (15T tokens, p.10) on xl lands hundreds of times past the paper\'s $10^{21}$ ceiling.</p>' },
      { label: 'Inference: NFEs are not forward passes', html: '<p>Fig 6a counts one function evaluation (NFE) per optimization step (p.8), but each step is a forward pass plus a backward pass to the input. We estimate 2 to 3 forward-pass equivalents per NFE [derived]; the full $2S$ layout is not needed when generating (p.36). Re-plotted against that cost, EBT needs about 6 to 9 forward-pass equivalents per token to beat a Transformer++ that spends 1, and the full 29% gain (measured against EBT\'s own no-thinking point) costs 60 to 90.</p>' },
      { label: 'Images: 3 versus 300', html: '<p>For denoising, "99% fewer forward passes" is 3 EBT steps against 300 DiT passes (p.35). With the same correction, 3 NFEs are about 6 to 9 forward-pass equivalents: still 33 to 50 times less compute than the DiT, not 100 times [derived]. The win is at the third pass only: at 1 vs 100 passes EBT is slightly lower and at 2 vs 200 about equal (Fig 12, approx.). Both models re-denoise their own output three times, which is how the DiT reaches 300 (p.35).</p>' },
      { label: 'What was not tested', html: '<p>Everything above sits below $10^{21}$ FLOPs (≈1300 A100 GPU hours, p.27) and 708M parameters, trained with 2 or 3 optimization steps. The paper\'s own reference points, models and data "1,000×" larger (p.9) and Llama 3\'s 15T tokens (p.10), are far to the right. Claims about that regime are extrapolations of slopes.</p>' },
    ],
    after: `
      <h3>What is still missing</h3>
      <ul>
        <li><b>Scale.</b> "EBTs scale well up to 800M parameters" (p.17); text goes up to large (396M), video to xl (708M). No run exceeds $10^{21}$ FLOPs (p.27).</li>
        <li><b>Depth of thinking in training.</b> Only "two or three optimization steps"; more was unstable (p.26). The uncertainty plots run inference out to iteration 11 (Figs 8 and 11), far past anything trained.</li>
        <li><b>Stability.</b> The step size α "was one of the primary factors affecting the stability" and had to be large (5 to 500 for text, 30,000 for video); otherwise gradient magnitudes kept growing (p.42). More steps mean longer gradient chains and less stability (p.42).</li>
        <li><b>Hyperparameters.</b> Step size, step count, Langevin noise and the replay buffer add knobs that are "extremely important and can often be highly sensitive" (p.43), and the base hyperparameters were tuned for feed-forward Transformers (p.33).</li>
        <li><b>When thinking pays.</b> Thinking gains "emerge with a sufficiently large data scale" (p.9). The Fig 6b trend appears only on Dyck (p.34). Models this small did not benefit from chain-of-thought (footnote 9, p.10).</li>
        <li><b>Evaluation.</b> Text results are teacher-forced perplexities with no generated text, from one seed (33) and no error bars (p.33).</li>
        <li><b>Many modes.</b> Training pushes toward one convex basin, so many-mode targets come out blurred (p.17, p.29).</li>
        <li><b>Latency.</b> For "low-latency LLM serving ... the added inference overhead of gradients with EBTs would not be worth the extra computation" (p.27).</li>
      </ul>
      <h3>Loose ends to keep in mind when reading the paper</h3>
      <ul>
        <li>The thinking runs are "xxs" in D.1.2 (p.34) but "Small" in Table D.4 (p.36). The learning rate (0.0012) is the xxs value (p.34).</li>
        <li>"Up to 800M parameters" (p.17): the largest configuration is xl with 708M non-embedding parameters (Table D.1), trained only on video.</li>
        <li>Table 3, Fig 6a and Fig 7 cannot be reconciled from their printed values, so do not mix numbers across them [derived].</li>
        <li>"3.5 in PSNR" (p.14) is 3.73 in Table 4; "around 10×" accuracy is 17.2× top-1 and 9.7× top-5, on accuracies of a few percent (p.13) [derived].</li>
        <li>The many-modes failure is described as "class conditional" generation on p.17 and "unconditional" on p.27.</li>
        <li>The video encoder is called the "frozen SD-XL VAE" (p.12), but its reference links to sd-vae-ft-mse (p.23).</li>
        <li>The S2 truncation scheme is not given; the paper refers to its source code (p.43). Listing 1 is simplified: it covers S2 without the landscape regularizers (p.43).</li>
      </ul>
      <p class="note">Chart points are digitized from the paper's vector figures (approx.); fits are refitted to those points and reproduce the printed scaling rates. Forward-pass equivalents per NFE (2 to 3), the A100-hour conversion (10²¹ FLOPs ≈ 1300 h, from p.27), crossover points and token counts are our arithmetic, marked [derived].</p>`,
    source: [{ kind: 'paper', note: 'D.5 accounting (p.35–36, p.41), Figs 5b, 6a, 12 digitized (approx.)' }, { kind: 'ext', note: 'per-NFE cost estimate, fit crossovers, calculator (derived)' }],

    figure(stage, ctx) {
      const { lib } = ctx, h = lib.h, C = lib.C;
      const SC = lib.data('scaling');
      const plot = (id) => SC && SC.plots.find(p => p.id === id);
      const P5 = plot('fig5b'), P6 = plot('fig6a'), P12 = plot('fig12');
      if (!P5 || !P6 || !P12) { stage.appendChild(h('p', { class: 'callout warn' }, 'Digitized figure data missing (data/scaling.json).')); return {}; }
      const ser = (P, m) => P.series.find(s => s.model === m);
      const T5 = ser(P5, 'Transformer++'), E5 = ser(P5, 'EBT');
      const M0 = E5.points[0][0] / T5.points[0][0]; // ≈6.67: where the paper put the EBT runs
      const st = { steps: 2, ar: true, showEBT: true, view: 'f5b', xm: 1, xmT: 1, size: 4, tokens: PRESETS.fig5b.tokens, nfe: 30 };
      let anim = null;

      function rcanvas(parent, hfn, label) {
        const box = h('div', { class: 'canvas-box' }); parent.appendChild(box);
        const cv = h('canvas', { role: 'img', 'aria-label': label }); box.appendChild(cv);
        const o = { canvas: cv, box, ctx: cv.getContext('2d'), w: 0, h: 0, draw: null };
        o.size = () => {
          const w = Math.max(260, Math.round(box.getBoundingClientRect().width || 600)), hh = Math.round(hfn(w));
          if (w === o.w && hh === o.h) return false;
          const dpr = Math.min(2, window.devicePixelRatio || 1);
          cv.width = Math.round(w * dpr); cv.height = Math.round(hh * dpr); o.ctx.setTransform(dpr, 0, 0, dpr, 0, 0); o.w = w; o.h = hh; return true;
        };
        o.clear = () => { o.ctx.save(); o.ctx.setTransform(1, 0, 0, 1, 0, 0); o.ctx.fillStyle = '#fff'; o.ctx.fillRect(0, 0, cv.width, cv.height); o.ctx.restore(); };
        o.size();
        if (window.ResizeObserver) new ResizeObserver(() => { if (o.size() && o.draw) o.draw(); }).observe(box);
        return o;
      }
      const T = (c, s, x, y, o = {}) => lib.text(c, s, x, y, Object.assign({ size: 11, kind: 'mono', color: C.ink }, o));
      const perStep = () => 10 * (st.ar ? 2 : 1);
      const ebtUnits = () => perStep() * st.steps;
      const mult = () => ebtUnits() / 6;

      // ---------- frame A: cost tape ----------
      const FA = lib.frame(stage, { label: 'Training FLOPs per token', sub: 'in units of N, the non-embedding parameters · App. D.5 (p.35–36)' });
      const tape = rcanvas(FA.frame, (w) => w < 460 ? 234 : 150, 'Bars comparing training FLOPs per token: Transformer++ forward and backward versus EBT forward, backward and Hessian-vector product per optimization step');
      const ca = h('div', { class: 'controls' }); stage.appendChild(ca);
      const stepSeg = lib.segmented({ label: 'Optimization steps', options: [[1, '1'], [2, '2'], [3, '3']], value: st.steps, onchange: (v) => { st.steps = v; st.showEBT = true; redraw(); } });
      const arSeg = lib.segmented({ label: 'Layout', options: [[true, 'autoregressive · 2S−2'], [false, 'one sequence']], value: st.ar, onchange: (v) => { st.ar = v; st.showEBT = true; redraw(); } });
      ca.appendChild(h('span', { class: 'fig-label' }, 'steps')); ca.appendChild(stepSeg.el);
      ca.appendChild(h('span', { class: 'fig-label' }, 'layout')); ca.appendChild(arSeg.el);

      function drawTape() {
        const c = tape.ctx, W = tape.w; tape.clear();
        const narrow = W < 460, lw = narrow ? 0 : 108, rw = narrow ? 56 : 64;
        const x0 = narrow ? 12 : lw + 2, ppn = (W - x0 - rw - 4) / 20;
        const rowH = narrow ? 44 : 27, bh = 19;
        const rows = [{ name: 'Transformer++', blocks: [['F', 2], ['B', 4]], tpp: true }];
        for (let s = 0; s < 3; s++) rows.push({ name: 'EBT · step ' + (s + 1), used: s < st.steps, blocks: [['F', 2], ['B', 4], ['B·HVP', 4]] });
        rows.forEach((r, i) => {
          const y = 6 + i * rowH + (narrow ? 16 : 0);
          if (narrow) T(c, r.name, 2, y - 15, { size: 10.5, color: r.tpp ? C.ink : C.blue, alpha: r.tpp || (st.showEBT && r.used) ? 1 : 0.35 });
          else T(c, r.name, 0, y + bh / 2, { size: 10.5, baseline: 'middle', color: r.tpp ? C.ink : C.blue, alpha: r.tpp || (st.showEBT && r.used) ? 1 : 0.35 });
          if (!r.tpp && (!st.showEBT || !r.used)) {
            c.save(); c.strokeStyle = C.rule; c.setLineDash([3, 3]); c.strokeRect(x0 + .5, y + .5, 10 * ppn * (st.ar ? 2 : 1), bh); c.restore();
            if (!st.showEBT && i === 1) T(c, '?', x0 + 8, y + bh / 2, { size: 11, baseline: 'middle', color: C.faint });
            return;
          }
          let x = x0; const reps = r.tpp ? 1 : (st.ar ? 2 : 1);
          for (let rep = 0; rep < reps; rep++) {
            r.blocks.forEach(([nm, u]) => {
              const w = u * ppn;
              if (r.tpp) { c.fillStyle = '#ececf0'; c.fillRect(x, y, w, bh); c.strokeStyle = C.ink; }
              else if (rep === 0) { c.fillStyle = nm === 'B·HVP' ? C.blue3 : C.blue4; c.fillRect(x, y, w, bh); c.strokeStyle = C.blue; }
              else {
                c.save(); c.beginPath(); c.rect(x, y, w, bh); c.clip(); c.fillStyle = '#fff'; c.fillRect(x, y, w, bh);
                c.strokeStyle = C.blue2; c.lineWidth = 1; c.beginPath(); for (let t = -bh; t < w; t += 5) { c.moveTo(x + t, y + bh); c.lineTo(x + t + bh, y); } c.stroke(); c.restore(); c.strokeStyle = C.blue;
              }
              c.lineWidth = 1; c.strokeRect(x + .5, y + .5, w - 1, bh - 1);
              const lab = w > 52 ? nm + ' ' + u + 'N' : (w > 22 ? nm.replace('·HVP', '') : '');
              if (lab && rep) { const tw = lib.measure(c, lab, { size: 10, kind: 'mono' }).w; c.fillStyle = '#fff'; c.fillRect(x + w / 2 - tw / 2 - 2, y + 4, tw + 4, bh - 8); }
              if (lab) T(c, lab, x + w / 2, y + bh / 2 + .5, { size: 10, align: 'center', baseline: 'middle', color: r.tpp ? C.ink : (rep ? C.blue : C.ink), weight: rep ? 600 : 400 });
              x += w;
            });
          }
          const tot = r.tpp ? 6 : perStep();
          T(c, '= ' + tot + 'N', W - rw + 4, y + bh / 2, { size: 11, baseline: 'middle', color: r.tpp ? C.ink : C.blue, weight: 600 });
        });
        // axis
        const ya = 6 + 4 * rowH + (narrow ? 16 : 0) - 2;
        c.strokeStyle = C.faint; c.lineWidth = 1; c.beginPath(); c.moveTo(x0, ya); c.lineTo(x0 + 20 * ppn, ya); c.stroke();
        [0, 2, 6, 10, 20].forEach(u => { const x = x0 + u * ppn; c.beginPath(); c.moveTo(x, ya); c.lineTo(x, ya + 4); c.stroke(); T(c, u + 'N', x, ya + 6, { size: 10, align: 'center', color: C.muted }); });
        if (st.showEBT && st.ar) T(c, 'hatched = the same work again: 2S−2 positions', x0 + 20 * ppn, ya + 22, { size: 10, color: C.blue, align: 'right' });
        else T(c, 'FLOPs per token, units of N', x0 + 20 * ppn, ya + 22, { size: 10, color: C.muted, align: 'right' });
      }
      const roA = h('div', { class: 'readout' }); stage.appendChild(roA);
      function readA() {
        const r = mult(), paperCase = !st.ar && st.steps === 1 ? ' (p.41)' : st.ar && st.steps === 1 ? ' per step (p.36)' : st.ar && st.steps === 2 ? ' (p.36, S1 pretraining)' : '';
        roA.innerHTML = `<span>Transformer++ <b>6N</b></span>` + (st.showEBT ? `<span>EBT <b>${ebtUnits()}N</b></span><span>ratio <b>${ratioStr(r)}</b>${paperCase}</span>` + (st.steps === 3 ? '<span>S2 runs draw 2–3 steps, truncate backprop and add a replay buffer: FLOPs vary (p.36)</span>' : '') : '<span>EBT <b>?</b></span>');
      }

      // ---------- frame C: views ----------
      const FC = lib.frame(stage, { label: 'What the price does', sub: 'paper figures (digitized, approx.) at the price above' });
      const cv = h('div', { class: 'controls' }); FC.wrap.insertBefore(cv, FC.frame);
      const VIEWS = [['f5b', 'equal FLOPs'], ['budget', 'budget'], ['f6a', 'thinking cost'], ['f12', '3 vs 300'], ['scale', 'scale tested']];
      const viewSeg = lib.segmented({ label: 'View', options: VIEWS, value: st.view, onchange: (v) => { st.view = v; redraw(); } });
      cv.appendChild(viewSeg.el);
      const chart = rcanvas(FC.frame, (w) => w < 460 ? (st.view === 'scale' ? 326 : 300) : 272, 'Chart for the selected view');
      const budget = h('div', { class: 'costs-budget' }); FC.frame.appendChild(budget);
      const cx2 = h('div', { class: 'controls' }); stage.appendChild(cx2);
      const xmSeg = lib.segmented({ label: 'x axis', options: [[0, 'x: NFEs (paper)'], [1, 'x: forward-pass equivalents']], value: 1, onchange: (v) => animateX(v) });
      cx2.appendChild(xmSeg.el);
      const roC = h('div', { class: 'readout' }); stage.appendChild(roC);

      // budget calculator (DOM)
      const bc = h('div', { class: 'controls' }); budget.appendChild(bc);
      const sizeSeg = lib.segmented({ label: 'Model size', options: SIZES.map((s, i) => [i, s.id]), value: st.size, onchange: (v) => { st.size = v; drawBudget(); } });
      bc.appendChild(h('span', { class: 'fig-label' }, 'model · D.1')); bc.appendChild(sizeSeg.el);
      const tokSl = lib.slider({ id: 'costs-tokens', label: 'training tokens', min: 9, max: 13.2, step: 0.01, value: Math.log10(st.tokens), fmt: (v) => tok(Math.pow(10, v)), oninput: (v) => { st.tokens = Math.pow(10, v); drawBudget(); } });
      const bc2 = h('div', { class: 'controls' }); budget.appendChild(bc2);
      bc2.appendChild(tokSl.el);
      const nfeSeg = lib.segmented({ label: 'Inference NFEs per token', options: [1, 2, 3, 6, 15, 30].map(v => [v, String(v)]), value: st.nfe, onchange: (v) => { st.nfe = v; drawBudget(); } });
      bc2.appendChild(h('span', { class: 'fig-label' }, 'NFEs')); bc2.appendChild(nfeSeg.el);
      const bc3 = h('div', { class: 'controls' }); budget.appendChild(bc3);
      Object.entries(PRESETS).forEach(([k, p]) => bc3.appendChild(lib.button(p.label, () => preset(k))));
      const tblBox = h('div', { class: 'tbl' }); budget.appendChild(tblBox);
      function preset(k) { const p = PRESETS[k]; st.size = p.size; st.tokens = p.tokens; sizeSeg.set(p.size); tokSl.set(Math.log10(p.tokens)); st.preset = k; drawBudget(); }
      const rng = (a, b) => b >= 1e9 ? (a / 1e9).toFixed(a >= 1e10 ? 0 : 1) + '–' + (b / 1e9).toFixed(b >= 1e10 ? 0 : 1) + ' GFLOP' : (a / 1e6).toFixed(0) + '–' + (b / 1e6).toFixed(0) + ' MFLOP';
      function drawBudget() {
        const N = SIZES[st.size].P, D = st.tokens, r = mult();
        const tT = 6 * N * D, tE = ebtUnits() * N * D, hT = tT / 1e21 * 1300, hE = tE / 1e21 * 1300;
        const iT = 2 * N, iE0 = st.nfe * 2 * 2 * N, iE1 = st.nfe * 3 * 2 * N;
        const pc = (v) => v / 1e21 >= 10 ? Math.round(v / 1e21) + '×' : (100 * v / 1e21).toFixed(v / 1e21 < 0.01 ? 2 : 1) + '%';
        const hrs = (x) => x >= 1e5 ? sci(x, 1) : x >= 10 ? Math.round(x).toLocaleString('en-US') : x.toFixed(1);
        tblBox.innerHTML = `<table><thead><tr><th></th><th class="num">Transformer++</th><th class="num">EBT</th></tr></thead><tbody>
          <tr><td>train / token</td><td class="num">${flops(6 * N)}</td><td class="num">${flops(ebtUnits() * N)}</td></tr>
          <tr><td>run, ${tok(D)} tokens</td><td class="num">${sci(tT)}</td><td class="num"><b>${sci(tE)}</b></td></tr>
          <tr><td>A100 hours [derived]</td><td class="num">${hrs(hT)}</td><td class="num">${hrs(hE)}</td></tr>
          <tr class="${tE > 1e21 ? 'over' : ''}"><td>of the 10²¹ ceiling</td><td class="num">${pc(tT)}</td><td class="num">${pc(tE)}</td></tr>
          <tr><td>infer / token, ${st.nfe} NFE${st.nfe > 1 ? 's' : ''} [derived]</td><td class="num">${flops(iT)}</td><td class="num">${rng(iE0, iE1)}</td></tr>
        </tbody></table>`;
        const p = st.preset ? PRESETS[st.preset] : null;
        const match = p && st.size === p.size && Math.abs(st.tokens - p.tokens) / p.tokens < 1e-6 ? p.note : null;
        roC.innerHTML = (match ? `<span>${match}</span>` : `<span>EBT/T++ <b>${ratioStr(r)}</b> with the steps and layout above</span>`) + (st.preset === 'fig5b' && match && st.steps === 2 && st.ar ? `<span>Fig 5b plots this pair at ≈${sci(T5.points[4][0])} and ≈${sci(E5.points[4][0])} FLOPs</span>` : '');
      }

      // ---- Fig 5b: equal FLOPs ----
      function drawF5b() {
        const c = chart.ctx, W = chart.w, H = chart.h; chart.clear();
        const m = mult(), sh = m / M0, ml = W < 460 ? 40 : 46;
        const ax = lib.axes(c, { x: ml, y: 26, w: W - ml - 12, h: H - 26 - 46, xlim: [1e16, 1e21], ylim: [14, 64], xlog: true, ylog: true, xticks: [1e16, 1e17, 1e18, 1e19, 1e20, 1e21], yticks: [20, 30, 40, 60], xfmt: (v) => '10' + SUP(Math.round(Math.log10(v))), yfmt: (v) => String(v), xlabel: 'training FLOPs (log)', size: 10.5 });
        T(c, 'validation perplexity (log)', ml - 4, 6, { size: 10.5, color: C.muted });
        // ceiling
        const xc = ax.X(1e21); c.save(); c.strokeStyle = C.ink; c.setLineDash([2, 3]); c.beginPath(); c.moveTo(xc, ax.o.y); c.lineTo(xc, ax.o.y + ax.o.h); c.stroke(); c.restore();
        c.save(); c.translate(xc - 5, ax.o.y + ax.o.h - 4); c.rotate(-Math.PI / 2); T(c, '10²¹ ceiling (p.27)', 0, 0, { size: 10, color: C.ink, baseline: 'bottom' }); c.restore();
        const fitLine = (a, s, xa, xb, col, dash) => { const pts = []; for (let k = 0; k <= 40; k++) { const lx = Math.log10(xa) + (Math.log10(xb) - Math.log10(xa)) * k / 40; pts.push([Math.pow(10, lx), Math.pow(10, a + s * lx)]); } c.save(); c.beginPath(); c.rect(ax.o.x, ax.o.y, ax.o.w, ax.o.h); c.clip(); lib.plot(c, ax, pts, { color: col, width: 1.4, dash }); c.restore(); };
        const fT = T5.fit, fE = E5.fit;
        const tx = T5.points.map(p => p[0]);
        fitLine(fT.intercept, fT.slope, tx[0], tx[4], C.ink); fitLine(fT.intercept, fT.slope, tx[4], 1e21, C.ink, [4, 4]);
        // twin connectors and names
        const names = ['xxs', 'xs', 'small', 'medium', 'large'];
        T5.points.forEach((p, i) => {
          const e = E5.points[i], xe = p[0] * m;
          if (st.showEBT) { c.save(); c.strokeStyle = C.faint; c.setLineDash([1, 3]); c.beginPath(); c.moveTo(ax.X(p[0]), ax.Y(p[1])); c.lineTo(ax.X(xe), ax.Y(e[1])); c.stroke(); c.restore(); }
          lib.dot(c, ax.X(p[0]), ax.Y(p[1]), 3.6, C.ink);
          T(c, names[i], ax.X(p[0]) - 6, ax.Y(p[1]) + 5, { size: 9.5, color: C.muted, align: 'right' });
        });
        if (st.showEBT) {
          // where the paper put the EBT runs
          if (Math.abs(sh - 1) > 0.02) E5.points.forEach(e => lib.dot(c, ax.X(e[0]), ax.Y(e[1]), 4, '#fff', { stroke: C.blue2, lw: 1.2 }));
          const aE = fE.intercept - fE.slope * Math.log10(sh);
          const ex = E5.points.map(e => e[0] * sh);
          fitLine(aE, fE.slope, ex[0], ex[4], C.blue); fitLine(aE, fE.slope, ex[4], 1e21, C.blue, [4, 4]);
          T5.points.forEach((p, i) => lib.dot(c, ax.X(p[0] * m), ax.Y(E5.points[i][1]), 4.2, C.blue));
        }
        // legend
        const lx = ax.o.x + ax.o.w - 4; let ly = ax.o.y + 4;
        const leg = (txt, col, hollow) => { const w = lib.measure(c, txt, { size: 10, kind: 'mono' }).w; lib.dot(c, lx - w - 10, ly + 6, 3.6, hollow ? '#fff' : col, hollow ? { stroke: col, lw: 1.2 } : {}); T(c, txt, lx, ly, { size: 10, align: 'right', color: hollow ? C.muted : col }); ly += 15; };
        leg('Transformer++', C.ink);
        if (st.showEBT) { leg(Math.abs(sh - 1) > 0.02 ? 'EBT at ' + ratioStr(m) + ' cost (what-if)' : 'EBT at ' + ratioStr(m) + ' (as in Fig 5b)', C.blue); if (Math.abs(sh - 1) > 0.02) leg('EBT as published (6.66×)', C.blue2, true); }
        // readout
        const L = (fT.intercept - aEfor(m)) / (fE.slope - fT.slope);
        const at19T = Math.pow(10, fT.intercept + fT.slope * 19), at19E = Math.pow(10, aEfor(m) + fE.slope * 19);
        roC.innerHTML = (st.showEBT ? `<span>log-log slopes: T++ <b>${fT.slope.toFixed(3)}</b> · EBT <b>${fE.slope.toFixed(3)}</b> (2.92% steeper)</span>` : `<span>log-log slope: T++ <b>${fT.slope.toFixed(3)}</b></span>`) + (st.showEBT ? `<span>at 10¹⁹ FLOPs: T++ <b>${at19T.toFixed(1)}</b> vs EBT <b>${at19E.toFixed(1)}</b> ppl</span><span>fits meet near <b>10${SUP(Math.round(L))}</b> FLOPs [derived extrapolation]</span>` : '<span>EBT points appear once the EBT step is priced</span>');
      }
      const aEfor = (m) => E5.fit.intercept - E5.fit.slope * Math.log10(m / M0);

      // ---- Fig 6a / Fig 12: inference cost ----
      function drawInf(which) {
        const c = chart.ctx, W = chart.w, H = chart.h; chart.clear();
        const t = st.xmT, ml = W < 460 ? 40 : 46, f6 = which === 'f6a';
        const P = f6 ? P6 : P12, base = ser(P, f6 ? 'Transformer++' : 'DiT'), E = ser(P, 'EBT');
        const ax = lib.axes(c, { x: ml, y: 26, w: W - ml - 12, h: H - 26 - 46, xlim: f6 ? [0.8, 130] : [0.8, 500], ylim: f6 ? [28, 48] : [12, 25.5], xlog: true,
          xticks: f6 ? [1, 2, 3, 6, 15, 30, 90] : [1, 2, 3, 9, 100, 300], yticks: f6 ? [30, 35, 40, 45] : [14, 17, 20, 23], xfmt: (v) => String(v), yfmt: (v) => String(v),
          xlabel: t > 0.5 ? (W < 460 ? 'forward-pass equivalents (log)' : 'forward-pass equivalents per prediction (log)') : (W < 460 ? 'forward passes, as plotted (log)' : 'number of forward passes, as plotted (log)'), size: 10.5 });
        T(c, f6 ? (W < 460 ? 'ppl increase, OOD (lower is better)' : 'perplexity increase on OOD data (lower is better)') : 'PSNR on OOD σ = 0.2 (higher is better)', ml - 4, 6, { size: 10.5, color: C.muted });
        const lg = (a, b) => Math.pow(10, Math.log10(a) + (Math.log10(b) - Math.log10(a)) * t);
        if (f6) {
          const yT = base.points[0][1];
          // Transformer++: plotted across all x in the paper; it spends 1 pass per token
          const xs = base.points.map(p => lg(p[0], 1));
          c.save(); c.strokeStyle = C.ink; c.setLineDash([5, 4]); c.lineWidth = 1.3; c.beginPath(); c.moveTo(ax.X(Math.min(...xs)), ax.Y(yT)); c.lineTo(ax.X(t > 0.5 ? 130 : 30), ax.Y(yT)); c.stroke(); c.restore();
          xs.forEach(x => lib.dot(c, ax.X(x), ax.Y(yT), 3.4, C.ink));
          T(c, t > 0.5 ? (W < 460 ? 'T++: 1 pass, cannot spend more' : 'Transformer++: 1 pass per token, cannot spend more') : 'Transformer++ (flat)', ax.X(t > 0.5 ? 1.15 : 2), ax.Y(yT) - 16, { size: 10, color: C.ink });
        } else {
          lib.plot(c, ax, base.points, { color: C.ink, width: 1.3, markers: 3.6 });
          T(c, 'DiT (DDIM, recursive)', ax.X(100), ax.Y(base.points[0][1]) + 8, { size: 10, color: C.ink, align: 'right' });
        }
        // EBT: marker at NFE × lerp(1, 2), bar to NFE × 3 fading in with t
        const ep = E.points.map(p => [p[0] * (1 + t), p[1], p[0]]);
        lib.plot(c, ax, ep, { color: C.blue, width: 1.3, line: true });
        ep.forEach(p => {
          if (t > 0.02) { c.save(); c.globalAlpha = t; c.strokeStyle = C.blue; c.lineWidth = 2; c.beginPath(); c.moveTo(ax.X(p[0]), ax.Y(p[1])); c.lineTo(ax.X(p[2] * 3 * t + p[0] * (1 - t)), ax.Y(p[1])); c.stroke();
            c.beginPath(); const xe = ax.X(p[2] * 3 * t + p[0] * (1 - t)); c.moveTo(xe, ax.Y(p[1]) - 4); c.lineTo(xe, ax.Y(p[1]) + 4); c.stroke(); c.restore(); }
          lib.dot(c, ax.X(p[0]), ax.Y(p[1]), 4.2, C.blue);
        });
        if (f6) {
          const labs = ['no thinking (2 NFEs)', 'thinking longer (3)', 'self-verification (6, 15, 30)'];
          T(c, labs[0], ax.X(ep[0][0]) + 6, ax.Y(ep[0][1]) - 18, { size: 10, color: C.blue });
          T(c, labs[1], ax.X(ep[1][0]) - 8, ax.Y(ep[1][1]) + 4, { size: 10, color: C.blue, align: 'right' });
          T(c, labs[2], ax.X(ep[3][0]), ax.Y(ep[3][1]) + 10, { size: 10, color: C.blue, align: 'center' });
          // the 29% bracket
          const xb = ax.o.x + ax.o.w - 8, y0 = ax.Y(E.points[0][1]), y1 = ax.Y(E.points[4][1]);
          c.save(); c.strokeStyle = C.blue2; c.lineWidth = 1; c.beginPath(); c.moveTo(xb - 5, y0); c.lineTo(xb, y0); c.lineTo(xb, y1); c.lineTo(xb - 5, y1); c.stroke(); c.restore();
          const g = 100 * (E.points[0][1] - E.points[4][1]) / E.points[0][1];
          T(c, '−' + g.toFixed(0) + '% vs EBT\'s own start', xb - 8, y0 + 2, { size: 10, color: C.blue, align: 'right' });
          const lo = ep[1], hi = ep[4];
          roC.innerHTML = t > 0.5
            ? `<span>beats T++ (≈${base.points[0][1].toFixed(1)}) from 3 NFEs ≈ <b>${(lo[2] * 2)}–${lo[2] * 3}</b> forward-pass equivalents</span><span>full gain at 30 NFEs ≈ <b>${hi[2] * 2}–${hi[2] * 3}</b>, vs 1 for T++ [derived]</span>`
            : `<span>the paper's x axis counts NFEs, one per optimization step (p.8)</span><span>Transformer++ is drawn flat: it "cannot improve performance at all" per token (p.10)</span>`;
        } else {
          T(c, 'EBT (1, 2, 3 steps)', ax.X(ep[0][0]) + 8, ax.Y(ep[0][1]) + 6, { size: 10, color: C.blue });
          const e3 = E.points[2], d3 = base.points[2];
          roC.innerHTML = t > 0.5
            ? `<span>EBT 3 NFEs ≈ <b>6–9</b> forward-pass equivalents vs DiT <b>300</b>: <b>${Math.round(300 / 9)}–${Math.round(300 / 6)}×</b> less compute [derived]</span><span>PSNR ${e3[1].toFixed(1)} vs ${d3[1].toFixed(1)} (Table 4: 23.29 vs 19.56)</span>`
            : `<span>as plotted: 1, 2, 3 EBT passes vs 100, 200, 300 DiT passes, "99% fewer" (p.4, p.35)</span><span>at 1 vs 100 EBT is slightly lower, at 2 vs 200 about equal; it wins at 3 vs 300</span>`;
        }
      }

      // ---- scale map ----
      function drawScale() {
        const c = chart.ctx, W = chart.w, H = chart.h; chart.clear();
        const narrow = W < 460, lw = narrow ? 0 : 124, x0 = lw + (narrow ? 18 : 10), x1 = W - 16;
        const rowsY = narrow ? [52, 132, 212, 292] : [36, 98, 160, 222];
        const axis = (y, lo, hi, ticks, fmt, name, sub) => {
          if (narrow) T(c, name + ' · ' + sub, 2, y - 44, { size: 10.5, weight: 600 }); else { T(c, name, 0, y - 10, { size: 10.5, weight: 600 }); T(c, sub, 0, y + 4, { size: 10, color: C.muted }); }
          const X = (v) => x0 + (Math.log10(v) - lo) / (hi - lo) * (x1 - x0);
          c.strokeStyle = C.faint; c.lineWidth = 1; c.beginPath(); c.moveTo(x0, y); c.lineTo(x1, y); c.stroke();
          ticks.forEach(v => { c.beginPath(); c.moveTo(X(v), y - 3); c.lineTo(X(v), y + 3); c.stroke(); T(c, fmt(v), X(v), y + 6, { size: 9.5, align: 'center', color: C.muted }); });
          return X;
        };
        const tickE = (v) => '10' + SUP(Math.round(Math.log10(v)));
        // params
        let X = axis(rowsY[0], 6, 12.2, [1e6, 1e8, 1e10, 1e12], (v) => v >= 1e9 ? (v / 1e9) + 'B' : (v / 1e6) + 'M', 'parameters', 'non-embedding');
        SIZES.forEach((s) => lib.dot(c, X(s.P), rowsY[0], 3.6, C.blue));
        bracket(X(6.18e6), X(708e6), rowsY[0], 'trained: 6.18M–708M (xl: video only)');
        refMark(X(396e9), rowsY[0], '1000× large (p.9)');
        // tokens
        X = axis(rowsY[1], 9, 13.4, [1e9, 1e10, 1e11, 1e12, 1e13], (v) => ({ 9: '1B', 10: '10B', 11: '100B', 12: '1T', 13: '10T' })[Math.round(Math.log10(v))], 'training tokens', 'per run');
        [[6.881e9, 'Fig 4a'], [32.8e9, 'thinking'], [131e9, 'FineWeb B.3']].forEach(([v, l], i) => { lib.dot(c, X(v), rowsY[1], 3.6, C.blue); T(c, l, X(v), rowsY[1] + (narrow && i === 1 ? 20 : -17), { size: 9.5, color: C.blue, align: 'center' }); });
        refMark(X(15e12), rowsY[1], 'Llama 3: 15T (p.10)');
        // FLOPs
        X = axis(rowsY[2], 16, 25, [1e16, 1e18, 1e20, 1e22, 1e24], tickE, 'training FLOPs', 'per run');
        E5.points.forEach(e => lib.dot(c, X(e[0]), rowsY[2], 3.2, C.blue));
        T5.points.forEach(p => lib.dot(c, X(p[0]), rowsY[2], 2.6, C.ink));
        c.save(); c.strokeStyle = C.ink; c.setLineDash([2, 3]); c.beginPath(); c.moveTo(X(1e21), rowsY[2] - 14); c.lineTo(X(1e21), rowsY[2] + 14); c.stroke(); c.restore();
        const N = SIZES[st.size].P, fE = ebtUnits() * N * st.tokens, over = fE > 1e21;
        T(c, '10²¹ ceiling', X(1e21) + (over ? -4 : 4), rowsY[2] - 27, { size: 9.5, align: over ? 'right' : 'left' });
        if (fE > 1e16 && fE < 1e25) { const xx = X(fE); c.fillStyle = C.blue; c.beginPath(); c.moveTo(xx, rowsY[2] - 5); c.lineTo(xx - 5, rowsY[2] - 13); c.lineTo(xx + 5, rowsY[2] - 13); c.closePath(); c.fill(); T(c, 'calculator run', xx + (over ? 7 : -7), rowsY[2] - 20, { size: 9.5, color: C.blue, align: over ? 'left' : 'right' }); }
        // steps
        X = axis(rowsY[3], 0, 1.6, [1, 2, 3, 6, 11, 30], (v) => String(v), 'NFEs', 'per prediction');
        c.save(); c.fillStyle = C.blue3; c.fillRect(X(2), rowsY[3] - 4, X(3) - X(2), 8); c.restore();
        T(c, 'trained: 2–3 steps (p.26)', X(2.45), rowsY[3] - 18, { size: 9.5, color: C.blue, align: 'center' });
        const ring = (v) => { c.save(); c.strokeStyle = C.blue; c.lineWidth = 1.4; c.beginPath(); c.arc(X(v), rowsY[3], 4.2, 0, 7); c.stroke(); c.restore(); };
        ring(11); T(c, 'Figs 8 & 11: iterations 0–11', X(11) + (narrow ? 0 : 6), rowsY[3] - 18, { size: 9.5, align: narrow ? 'right' : 'center', color: C.blue });
        ring(30); T(c, 'Fig 6a: 30 NFEs (10 × 3, derived)', X(30) + 2, rowsY[3] + 20, { size: 9.5, align: 'right', color: C.blue });
        roC.innerHTML = '<span>blue dots = runs the paper trained (black: their Transformer++ twins) · open circles = scales it cites but never trained · rings = inference-time step counts</span><span>the foundation-scale claims are slope extrapolations (p.9, p.12)</span>';
        function bracket(xa, xb, y, txt) { c.save(); c.strokeStyle = C.blue; c.lineWidth = 1; c.beginPath(); c.moveTo(xa, y - 9); c.lineTo(xa, y - 12); c.lineTo(xb, y - 12); c.lineTo(xb, y - 9); c.stroke(); c.restore(); T(c, txt, (xa + xb) / 2, y - 26, { size: 9.5, color: C.blue, align: 'center' }); }
        function refMark(x, y, txt) { lib.dot(c, x, y, 4, '#fff', { stroke: C.ink, lw: 1.4 }); T(c, txt, Math.min(x, x1 - 2), y - 20, { size: 9.5, align: 'right' }); }
      }

      function drawChart() {
        const v = st.view;
        chart.box.style.display = v === 'budget' ? 'none' : ''; chart.size();
        budget.style.display = v === 'budget' ? '' : 'none';
        cx2.style.display = v === 'f6a' || v === 'f12' ? '' : 'none';
        if (v === 'f5b') drawF5b(); else if (v === 'f6a' || v === 'f12') drawInf(v); else if (v === 'scale') drawScale(); else drawBudget();
      }
      function animateX(target) {
        if (anim) anim.stop(); st.xm = target; xmSeg.set(target);
        if (lib.reducedMotion) { st.xmT = target; drawChart(); return; }
        const from = st.xmT; let e = 0;
        anim = lib.loop((dt) => { e += dt / 0.6; st.xmT = from + (target - from) * lib.ease(Math.min(1, e)); drawChart(); return e < 1; });
        anim.start();
      }
      function redraw() {
        stepSeg.set(st.steps); arSeg.set(st.ar); viewSeg.set(st.view); xmSeg.set(st.xm);
        drawTape(); readA(); drawChart();
      }
      tape.draw = () => drawTape();
      chart.draw = () => drawChart();
      drawBudget();
      ctx.setCaption('Top: training FLOPs per token (gray: Transformer++; blue: one EBT optimization step per row; hatched: the doubled sequence). Bottom: the paper\'s figures at that price, a budget calculator, and the scales tested.');
      const set = (o) => { if (anim) anim.stop(); Object.assign(st, o); redraw(); };
      return {
        step(i) {
          if (i === 0) set({ showEBT: false, steps: 1, ar: false, view: 'f5b' });
          if (i === 1) set({ showEBT: true, steps: 1, ar: false, view: 'f5b' });
          if (i === 2) set({ showEBT: true, steps: 1, ar: true, view: 'f5b' });
          if (i === 3) set({ showEBT: true, steps: 2, ar: true, view: 'f5b' });
          if (i === 4) { set({ showEBT: true, steps: 2, ar: true, view: 'budget', nfe: 30 }); nfeSeg.set(30); preset('fig5b'); }
          if (i === 5) { set({ showEBT: true, steps: 2, ar: true, view: 'f6a', xmT: 0, xm: 0 }); animateX(1); }
          if (i === 6) { set({ showEBT: true, steps: 2, ar: true, view: 'f12', xmT: 0, xm: 0 }); animateX(1); }
          if (i === 7) set({ showEBT: true, steps: 2, ar: true, view: 'scale' });
        },
        hide() { if (anim) anim.stop(); },
      };
    },
  });
})();
