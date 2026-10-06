/* Panel: exploring with Langevin noise, choosing by energy (Eq. 2, Algorithm 2, Table 2).
   Main instrument: a real 2-D slice of the toy character-level text EBT (data/text.json weights, decoded and run in the browser):
   two logits move, the other 52 are frozen at 0. M candidates descend with Langevin noise; the lowest final energy wins.
   Bottom instrument (tabs): a live Monte Carlo sweep over sigma (slice or full 54-D on held-out text),
   the toy's stored full-vocabulary best-of-M results, and paper Table 2. */
(function () {
  'use strict';

  // ---------------------------------------------------------------------------------------------
  // Toy text EBT, exact port of data/text.json "arch" (same math as src/text_ebt_ref.js, verified there against numpy).
  //   h = Linear(SiLU(Linear(concat 40 char embeddings)))     context code (128)
  //   p = softmax(ŷ); e = p·Wpe; z = [h, e, h⊙e]; E = MLP(z)    scalar energy
  // h is fixed while thinking, so z·W1 = c1 + p·Q with Q = Wpe·(W1e + diag(h)·W1he), precomputed per context.
  // The same guarded factory lives in tokens.js; whichever panel runs first decodes the weights once.
  // ---------------------------------------------------------------------------------------------
  function getTextModel(lib) {
    const EBT = window.EBT;
    if (EBT.__textEBT !== undefined) return EBT.__textEBT;
    EBT.__textEBT = null;
    const D = lib.data('text'), Wt = D && D.weights;
    if (!Wt || !Wt.data || !Wt.order) return null;
    let all;
    try { const bin = atob(Wt.data), u8 = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i); all = new Float32Array(u8.buffer); }
    catch (e) { console.warn('text EBT: weight decode failed', e); return null; }
    const T = {}, S = {};
    for (const k of Wt.order) { const t = Wt.tensors[k], n = t.shape.reduce((a, b) => a * b, 1); T[k] = all.subarray(t.offset, t.offset + n); S[k] = t.shape; }
    const V = S.Wpe[0], Dh = S.Wpe[1], dc = S.emb[1], nIn = S.We1[0], E1 = S.We1[1], H1 = S.W1[1], H2 = S.W2[1], Cn = nIn / dc;
    const emb = T.emb, We1 = T.We1, be1 = T.be1, We2 = T.We2, be2 = T.be2, Wpe = T.Wpe, W1 = T.W1, b1 = T.b1, W2 = T.W2, b2 = T.b2, W3 = T.W3, b3 = T.b3[0];
    const sig = (z) => 1 / (1 + Math.exp(-z));
    const stoi = new Map(D.vocab.map((c, i) => [c, i]));
    const mvAdd = (out, x, W, nI, nO) => { for (let i = 0; i < nI; i++) { const xi = x[i]; if (xi === 0) continue; const o = i * nO; for (let j = 0; j < nO; j++) out[j] += xi * W[o + j]; } };
    // corpus normalization (text.json corpus.normalization): lowercase, ASCII quotes/dashes, other symbols -> '#', non-ASCII dropped, space runs collapsed
    function norm(s) {
      s = String(s).toLowerCase().replace(/[‘’]/g, "'").replace(/[“”]/g, '"').replace(/[–—]/g, '-').replace(/\r/g, '').replace(/\t/g, ' ').replace(/[^\x00-\x7f]/g, '').replace(/ +/g, ' ');
      return Array.from(s).map(ch => stoi.has(ch) ? ch : '#').join('');
    }
    function ids(s) { const t = Array.from(norm(s)).slice(-Cn), r = t.map(ch => stoi.get(ch)); while (r.length < Cn) r.unshift(stoi.get(' ')); return r; }
    const cache = new Map();
    function context(s) {
      const key = ids(s).join(','); if (cache.has(key)) return cache.get(key);
      const id = key.split(',').map(Number), x = new Float64Array(nIn);
      for (let t = 0; t < Cn; t++) for (let d = 0; d < dc; d++) x[t * dc + d] = emb[id[t] * dc + d];
      const a = Float64Array.from(be1); mvAdd(a, x, We1, nIn, E1); for (let j = 0; j < E1; j++) a[j] = a[j] * sig(a[j]);
      const h = Float64Array.from(be2); mvAdd(h, a, We2, E1, Dh);
      const c1 = Float64Array.from(b1); mvAdd(c1, h, W1, Dh, H1);
      const Wp = new Float64Array(Dh * H1);
      for (let d = 0; d < Dh; d++) { const o1 = (Dh + d) * H1, o2 = (2 * Dh + d) * H1, hd = h[d]; for (let j = 0; j < H1; j++) Wp[d * H1 + j] = W1[o1 + j] + hd * W1[o2 + j]; }
      const Q = new Float64Array(V * H1);
      for (let k = 0; k < V; k++) { const qo = k * H1; for (let d = 0; d < Dh; d++) { const w = Wpe[k * Dh + d], o = d * H1; for (let j = 0; j < H1; j++) Q[qo + j] += w * Wp[o + j]; } }
      const cx = { h, c1, Q, ids: id };
      if (cache.size > 400) cache.delete(cache.keys().next().value);
      cache.set(key, cx); return cx;
    }
    const a1 = new Float64Array(H1), s1 = new Float64Array(H1), a2 = new Float64Array(H2), da1 = new Float64Array(H1), da2 = new Float64Array(H2);
    // shared tail: a1 (pre-activation of layer 1) -> E, and (if grad) da1 = dE/da1
    function tail(grad) {
      for (let j = 0; j < H1; j++) s1[j] = a1[j] * sig(a1[j]);
      a2.set(b2); mvAdd(a2, s1, W2, H1, H2);
      let E = b3; for (let j = 0; j < H2; j++) E += a2[j] * sig(a2[j]) * W3[j];
      if (!grad) return E;
      for (let j = 0; j < H2; j++) { const s = sig(a2[j]); da2[j] = W3[j] * s * (1 + a2[j] * (1 - s)); }
      for (let i = 0; i < H1; i++) { let acc = 0; const o = i * H2; for (let j = 0; j < H2; j++) acc += W2[o + j] * da2[j]; const s = sig(a1[i]); da1[i] = acc * s * (1 + a1[i] * (1 - s)); }
      return E;
    }
    // full 54-D: energy at logits y; if g is given it receives dE/dy (exact). p (optional) receives softmax(y); dp (optional) receives dE/dp.
    function evalE(cx, y, g, p, dp) {
      p = p || new Float64Array(V);
      let m = -Infinity; for (let k = 0; k < V; k++) if (y[k] > m) m = y[k];
      let Z = 0; for (let k = 0; k < V; k++) { p[k] = Math.exp(y[k] - m); Z += p[k]; } for (let k = 0; k < V; k++) p[k] /= Z;
      a1.set(cx.c1); const Q = cx.Q;
      for (let k = 0; k < V; k++) { const pk = p[k], o = k * H1; for (let j = 0; j < H1; j++) a1[j] += pk * Q[o + j]; }
      const E = tail(!!g); if (!g) return E;
      let sp = 0;
      for (let k = 0; k < V; k++) { let acc = 0; const o = k * H1; for (let j = 0; j < H1; j++) acc += Q[o + j] * da1[j]; g[k] = acc; sp += p[k] * acc; }
      if (dp) dp.set(g);
      for (let k = 0; k < V; k++) g[k] = p[k] * (g[k] - sp);
      return E;
    }
    // 2-D slice: only logits A and B move, the other V-2 logits stay at 0. ev(a, b, grad, out) -> out = [E, pA, pB, pRest(each), dE/da, dE/db]
    function plane(cx, A, B) {
      const Qa = cx.Q.subarray(A * H1, (A + 1) * H1), Qb = cx.Q.subarray(B * H1, (B + 1) * H1), Qr = new Float64Array(H1), nR = V - 2, c1 = cx.c1;
      for (let k = 0; k < V; k++) { if (k === A || k === B) continue; const o = k * H1; for (let j = 0; j < H1; j++) Qr[j] += cx.Q[o + j]; }
      return function (a, b, grad, out) {
        out = out || new Array(6);
        const m = Math.max(a, b, 0), ea = Math.exp(a - m), eb = Math.exp(b - m), er = Math.exp(-m), Z = ea + eb + nR * er, pa = ea / Z, pb = eb / Z, pr = er / Z;
        for (let j = 0; j < H1; j++) a1[j] = c1[j] + pa * Qa[j] + pb * Qb[j] + pr * Qr[j];
        const E = tail(grad); out[0] = E; out[1] = pa; out[2] = pb; out[3] = pr;
        if (!grad) { out[4] = out[5] = 0; return out; }
        let gA = 0, gB = 0, gR = 0; for (let j = 0; j < H1; j++) { gA += Qa[j] * da1[j]; gB += Qb[j] * da1[j]; gR += Qr[j] * da1[j]; }
        const sp = pa * gA + pb * gB + pr * gR; out[4] = pa * (gA - sp); out[5] = pb * (gB - sp);
        return out;
      };
    }
    EBT.__textEBT = { V, Cn, vocab: D.vocab, disp: D.vocab_display || D.vocab, stoi, norm, ids, context, evalE, plane, alpha0: (D.hparams && D.hparams.alpha0) || 10, nParams: Wt.n_floats };
    return EBT.__textEBT;
  }

  // ---------------------------------------------------------------------------------------------
  const SLICES = [
    { ctx: ' educated at highgate school in london, ', A: 't', B: 'a', truth: 'a', ext: [-4, 12], seg: '…in london, _', what: 'two kinds of basin' },
    { ctx: 'd and harvard universities. he was in th', A: 'e', B: 'a', truth: 'e', ext: [-4, 12], seg: '…he was in th_', what: 'one valley' },
  ];
  const TABLE2 = [ // Table 2 (p.10), percent perplexity improvement, OOD BigBench Dyck
    { name: 'No random step size', tl: -1.47, sv: 0.19, why: 'α fixed in training. Thinking longer now hurts slightly (−1.47%) and verification barely helps (0.19%): the paper calls random step size critical.' },
    { name: 'No random num. steps', tl: 0.00, sv: 9.65, why: 'Fixed step count in training. Extra steps give exactly 0.00%: the model never learned to use them. Verification still gives 9.65%.' },
    { name: 'No Langevin dynamics', tl: 17.2, sv: 17.0, why: 'No noise in training. The best single-path gain (17.2%), but adding verification no longer helps (17.0%): less exploration, a landscape tuned to the direct path.' },
    { name: 'No replay buffer', tl: 14.8, sv: 17.8, why: 'No replayed trajectories in training. 14.8% thinking longer, 17.8% with verification.' },
    { name: 'Full System 2 config', tl: 7.19, sv: 18.7, full: true, why: 'All four regularizers. Weakest single-path gain of the useful rows (7.19%), but the best once candidates are verified (18.7%).' },
  ];
  const DSN = { val: 'held-out web text', ood_shakespeare: 'Shakespeare (OOD)', ood_code: 'Python code (far OOD)', train: 'train text' };

  EBT.panel({
    id: 'langevin-bon',
    nav: 'Noise and self-verification',
    title: 'Exploring with noise, choosing by energy',
    lede: 'Descent from one start finds one answer: the nearest basin. Run several noisy descents, keep the candidate the model itself scores lowest, and extra compute becomes a search, with no outside judge.',
    text: `
      <p>Thinking longer refines one guess. The paper's second way to spend compute is <b>self-verification</b>: draw $M$ random starting guesses, run $N$ descent steps on each, and return the candidate with the lowest energy (Algorithm 2, p.7):</p>
      <div class="eq">$$\\hat y^{*} \\;=\\; \\arg\\min_{j=1..M}\\; E_\\theta\\big(x,\\hat y_{N,j}\\big), \\qquad \\hat y_{0,j}\\sim\\mathcal N(0,I)$$<span class="why">Algorithm 2 (p.7). Cost: M·N function evaluations, each a forward pass plus a backward pass to the input.</span></div>
      <p>Picking only helps if the candidates differ. Random starts give some diversity. During training the paper adds more with a variant of <b>Langevin dynamics</b>, a random kick in every update (Sec 3.3):</p>
      <div class="eq">$$\\hat y_{i+1} \\;=\\; \\hat y_i - \\alpha\\,\\nabla_{\\hat y}E_\\theta(x,\\hat y_i) + \\eta_i, \\qquad \\eta_i\\sim\\mathcal N(0,\\sigma)$$<span class="why">Eq. 2 (p.7). σ is "the magnitude of the noise"; here it is the standard deviation of each coordinate.</span></div>
      <p>The landscape on the left is a real one. It belongs to our toy character-level EBT (54 symbols), given the context "…school in london, _", where the true next character is 'a'. Its prediction $\\hat y$ is a vector of 54 logits, so we freeze 52 of them at 0 and let two move: the logits of 't' and 'a'. Every energy and gradient is computed in your browser from the trained weights.</p>`,
    steps: [
      { label: 'One path, no noise', html: '<p>$M=1$, $\\sigma=0$: plain descent from one random start. It slides into the shallow basin near the origin, where \'t\' and \'a\' get about 11% each and most of the mass stays on the other 52 symbols. The energy stops falling, so by its own rule the model has converged, on a poor answer: $-\\log p(\\text{a}) \\approx 2.2$ nats.</p>' },
      { label: 'Add Langevin noise', html: '<p>$\\sigma = 1$. Each update is the gradient step (dashed arrow) plus a random kick $\\eta_i$ (blue arrow). The path wanders. Now and then a kick carries it over a rim into a deeper valley. The single answer has become a random variable: sometimes much better, sometimes worse. Press <b>[ new starts ]</b> a few times.</p>' },
      { label: 'Many candidates', html: '<p>Algorithm 2 with $M = 8$ starts and no noise. All eight fall into the same shallow basin, their energies agree, and there is nothing to choose between. Diversity from the starting point alone does not help when one basin captures almost every start.</p>' },
      { label: 'Choose by energy', html: '<p>$M = 8$, $\\sigma = 1$. Now the candidates spread across basins. The lowest final energy (blue ring, $\\hat y^{*}$) sits in a deep valley where \'a\' wins. The model never saw the answer: it ranked its own candidates. The readout compares the pick with the average candidate and with an oracle that cheats by using the true label.</p>' },
      { label: 'What noise buys, measured', html: '<p>The bottom chart repeats the experiment many times for every σ (live Monte Carlo, the same starts and kicks reused at every σ). One path (dashed) degrades as σ grows. The energy-picked best of $M$ (blue) drops sharply once noise lets candidates escape, then stays low. The oracle (dotted) shows how much diversity exists; the gap between blue and dotted is what the verifier leaves on the table. Try the one-valley context: there noise has nothing to find.</p>' },
      { label: 'The full 54-D model', html: '<p>The slice shows the mechanism. The 54-D sweep runs the same experiment on 60 held-out positions of real web text, with every logit free, the toy\'s trained $\\alpha = 10$ and 8 steps. Here noise does not pay: the oracle keeps improving, but the energy cannot single out the better candidates, so the noisy picks are no better than the noise-free ones. The stored results (tab "full vocabulary") show what noise-free best-of-8 does buy: 2.329 → 2.250 nats on held-out text, against 1.869 for the oracle.</p>' },
      { label: 'Why the paper adds noise in training', html: '<p>In the paper Langevin noise is a <em>training</em> regularizer: "Without this random noise term, exploration is often limited to paths leading directly to the energy minimum, leaving other regions poorly defined" (p.7). A verifier can only rank candidates in regions it was shaped on. Table 2 (p.10) shows the trade: without Langevin noise, thinking longer gains more (17.2% vs 7.19%), but self-verification gains less (17.0% vs 18.7%).</p>' },
    ],
    after: `
      <h3>Why noise helps a verifier, and when it does not</h3>
      <p>Deterministic descent is a function: the same start always gives the same answer. Best-of-$M$ needs two things. The candidates must land in different places, and the energy must rank those places correctly. Noise supplies the first, at a price: every candidate is shaken, so the single-path answer degrades. Verification wins that price back only where the energy is reliable. That is why the paper puts the noise into training. It makes the model visit, and therefore shape, the regions off the direct path. That is also why the full configuration in Table 2 is best only in the column with verification.</p>
      <p>The verifier is imperfect even in the paper. Comparing best-of-10 with best-of-2, "verifying 10 samples occasionally leads to worse performance than verifying 2 samples, likely because the EBT found an adversarial sample (a sample with low energy that is in fact not a good prediction)". This happened in models trained on less data and faded as data grew (App. B.1, p.28). The gap between our blue and dotted curves is the same effect in miniature.</p>
      <p class="note">Caveats. Algorithm 2 has no noise term, and the paper does not say whether Langevin noise is used at inference. Table 2 comes from one OOD benchmark (BigBench Dyck) and one seed, and the reference point of its "percent perplexity improvement" is not stated. The slice freezes 52 of 54 logits at 0, so its basins are not the minima of the full model: it shows the mechanism, and the 54-D sweep shows the effect size. In the toy, noise is added on every step except the last, as in its training. The "annealed" schedule is our extension; the paper lists annealed Langevin dynamics only as future work (p.27). Self-verification is per prediction (here per character) and uses the model's own energy, with no reward model (p.8).</p>`,
    source: [
      { kind: 'toy', note: 'toy text EBT (54 symbols): 2-D slice and 54-D sweeps computed live' },
      { kind: 'toy', note: 'stored best-of-M results, data/text.json' },
      { kind: 'paper', note: 'Eq. 2, Alg. 2 (p.7), Table 2 (p.10)' },
      { kind: 'ext', note: 'annealed noise schedule' },
    ],
    figure(stage, ctx) {
      const { lib } = ctx, h = lib.h, C = lib.C;
      const D = lib.data('text');
      if (!D || !D.weights) { stage.appendChild(h('p', { class: 'callout warn' }, 'Toy text model missing (data/text.json). Run python3 src/bundle_data.py.')); return {}; }
      let model = null; const M0 = () => model || (model = getTextModel(lib));
      const fx = (v, d = 2) => lib.fmt(v, d);
      const reduced = !!lib.reducedMotion;
      const Vn = D.vocab.length;

      // ---------------- state ----------------
      const S = { si: 0, alpha: 5, N: 20, M: 8, sigma: 1.0, sched: 'const', seed: 1, t: 20, playing: false, inspect: -1, tab: 'sweep', sweepMode: 'slice', bonDs: 'val', bonN: 0, hoverRow: -1 };
      let sim = null;                   // {cands:[{path,E,PT,G,Nz}], N}
      const grids = {}, planes = {};
      const sl = () => SLICES[S.si];
      const planeFor = (k) => planes[k] || (planes[k] = M0().plane(M0().context(SLICES[k].ctx), M0().stoi.get(SLICES[k].A), M0().stoi.get(SLICES[k].B)));
      const noiseAt = (i, N, sigma, sched) => sched === 'anneal' ? sigma * Math.max(0, 1 - i / (0.6 * N)) : (i < N - 1 ? sigma : 0);
      const pTrue = (o, k) => { const s = SLICES[k]; return s.truth === s.A ? o[1] : s.truth === s.B ? o[2] : o[3]; };

      // ---------------- layout ----------------
      const row = h('div', { class: 'fig-row lb-row' }); stage.appendChild(row);
      const FL = lib.frame(row, { label: 'Energy landscape · 2-D slice', sub: 'E<sub>θ</sub>(x, ŷ) on 2 of 54 logits · click a dot' });
      FL.wrap.classList.add('lb-land');
      const LS = 400, cv = lib.canvas(FL.frame, LS, LS, { label: 'Energy landscape of the toy text EBT on a two-logit slice, with the candidate predictions and their descent paths' });
      const FR = lib.frame(row, { label: 'Energy per candidate', sub: 'lowest final E wins' });
      FR.wrap.classList.add('lb-side');
      const pv = lib.canvas(FR.frame, 300, 250, { label: 'Energy of every candidate at each step' });
      const ro = h('div', { class: 'readout lb-ro', 'aria-live': 'polite' }); FR.wrap.appendChild(ro);
      const c3 = h('div', { class: 'controls lb-btns' }); FR.wrap.appendChild(c3);
      c3.appendChild(lib.button('new starts', () => { S.seed++; S.inspect = -1; recompute(true); }));
      c3.appendChild(lib.button('run', () => { recompute(true); }, { primary: true }));
      c3.appendChild(lib.button('step', () => { S.playing = false; if (S.t >= S.N) S.t = 0; else S.t++; draw(); }));
      c3.appendChild(lib.button('end', () => { S.playing = false; S.t = S.N; draw(); }));

      const c1 = h('div', { class: 'controls' }); stage.appendChild(c1);
      const segCtx = lib.segmented({ label: 'Context', options: SLICES.map((s, i) => [i, s.seg]), value: 0, onchange: (v) => { S.si = v; S.inspect = -1; recompute(true); restartSweep(); } });
      c1.appendChild(h('span', { class: 'fig-label' }, 'context')); c1.appendChild(segCtx.el);
      const segSch = lib.segmented({ label: 'Noise schedule', options: [['const', 'constant σ'], ['anneal', 'annealed (ext)']], value: 'const', onchange: (v) => { S.sched = v; recompute(false); restartSweep(); } });
      c1.appendChild(h('span', { class: 'fig-label' }, 'noise')); c1.appendChild(segSch.el);

      const c2 = h('div', { class: 'controls lb-sliders' }); stage.appendChild(c2);
      const slS = lib.slider({ id: 'lb-sigma', label: 'noise σ', min: 0, max: 3, step: 0.05, value: S.sigma, fmt: (v) => v.toFixed(2), oninput: (v) => { S.sigma = v; recompute(false); } });
      const slM = lib.slider({ id: 'lb-M', label: 'candidates M', min: 1, max: 16, step: 1, value: S.M, oninput: (v) => { S.M = v; S.inspect = -1; recompute(false); restartSweep(); } });
      const slN = lib.slider({ id: 'lb-N', label: 'steps N', min: 1, max: 40, step: 1, value: S.N, oninput: (v) => { S.N = v; recompute(false); restartSweep(); } });
      const slA = lib.slider({ id: 'lb-alpha', label: 'step size α', min: 0.5, max: 12, step: 0.5, value: S.alpha, fmt: (v) => v.toFixed(1), oninput: (v) => { S.alpha = v; recompute(false); restartSweep(); } });
      [slS, slM, slN, slA].forEach(s => c2.appendChild(s.el));

      // bottom tabbed instrument
      const tabRow = h('div', { class: 'controls lb-tabs' }); stage.appendChild(tabRow);
      const segTab = lib.segmented({ label: 'View', options: [['sweep', 'noise sweep · live'], ['full', 'full vocabulary · toy data'], ['table2', 'paper · Table 2']], value: 'sweep', onchange: (v) => { S.tab = v; syncTab(); } });
      tabRow.appendChild(h('span', { class: 'fig-label' }, 'what it buys')); tabRow.appendChild(segTab.el);
      const FB = lib.frame(stage, {});
      const subRow = h('div', { class: 'controls lb-sub' }); FB.frame.appendChild(subRow);
      const segMode = lib.segmented({ label: 'Sweep space', options: [['slice', 'this 2-D slice'], ['full', '54-D, 60 held-out positions']], value: 'slice', onchange: (v) => { S.sweepMode = v; restartSweep(); drawBottom(); } });
      const segDs = lib.segmented({ label: 'Dataset', options: Object.keys(DSN).filter(k => D.bon && D.bon.settings[0].datasets[k]).map(k => [k, DSN[k]]), value: 'val', onchange: (v) => { S.bonDs = v; drawBottom(); } });
      const segBN = lib.segmented({ label: 'Steps per candidate', options: (D.bon ? D.bon.settings : []).map((s, i) => [i, 'N = ' + s.N]), value: 0, onchange: (v) => { S.bonN = v; drawBottom(); } });
      const BW = 640, BH = 196, bv = lib.canvas(FB.frame, BW, BH, { label: 'Results chart for the selected view' });
      const bro = h('div', { class: 'readout lb-bro' }); FB.frame.appendChild(bro);
      bv.canvas.addEventListener('mousemove', (ev) => { if (S.tab !== 'table2') return; const [, py] = bv.toLocal(ev); const r = t2Row(py); if (r !== S.hoverRow) { S.hoverRow = r; drawBottom(); } });
      bv.canvas.addEventListener('mouseleave', () => { if (S.hoverRow !== -1 && S.tab === 'table2') { S.hoverRow = -1; drawBottom(); } });
      bv.canvas.addEventListener('click', (ev) => { if (S.tab !== 'table2') return; const [, py] = bv.toLocal(ev); S.hoverRow = t2Row(py); drawBottom(); });

      ctx.setCaption('Darker blue = lower energy (real toy text EBT, one context). Hollow circles: starts ŷ<sub>0,j</sub>. Gray: candidate paths. Blue ring: lowest-energy candidate ŷ*. Dashed curve: true character at probability ½.');

      // ---------------- simulation (common random numbers: same starts and kicks for every σ, α, N) ----------------
      function simulate() {
        const ev = planeFor(S.si), k = S.si, out = new Array(6), cands = [];
        for (let j = 0; j < S.M; j++) {
          const r = lib.rng(S.seed * 7919 + j * 104729 + 13);
          let a = r.normal(), b = r.normal();
          const path = [[a, b]], E = [], PT = [], G = [], Nz = [];
          for (let i = 0; i <= S.N; i++) {
            ev(a, b, i < S.N, out); E.push(out[0]); PT.push(pTrue(out, k));
            if (i === S.N) break;
            const s = noiseAt(i, S.N, S.sigma, S.sched), na = r.normal(), nb = r.normal();
            const ga = -S.alpha * out[4], gb = -S.alpha * out[5];
            G.push([ga, gb]); Nz.push([s * na, s * nb]);
            a = Math.max(-40, Math.min(40, a + ga + s * na)); b = Math.max(-40, Math.min(40, b + gb + s * nb));
            path.push([a, b]);
          }
          cands.push({ path, E, PT, G, Nz });
        }
        sim = { cands, N: S.N };
      }
      function recompute(animate) {
        if (!ctx.visible() && !sim) { pendingAnim = animate; return; }
        simulate(); S.t = animate && !reduced ? 0 : S.N; S.playing = animate && !reduced; kick();
      }
      let pendingAnim = true;
      const winnerAt = (t) => { let w = 0; sim.cands.forEach((c, j) => { if (c.E[t] < sim.cands[w].E[t]) w = j; }); return w; };

      // ---------------- landscape grid (computed once per context, in chunks) ----------------
      const GN = 60;
      function gridStep(k, budget) {
        const g = grids[k] || (grids[k] = { E: Array.from({ length: GN }, () => new Array(GN)), row: 0, done: false });
        if (g.done) return true;
        const ev = planeFor(k), ext = SLICES[k].ext, out = new Array(6), t0 = performance.now();
        while (g.row < GN && performance.now() - t0 < budget) {
          const r = g.row, bv_ = ext[1] - (ext[1] - ext[0]) * r / (GN - 1);
          for (let c = 0; c < GN; c++) g.E[r][c] = ev(ext[0] + (ext[1] - ext[0]) * c / (GN - 1), bv_, false, out)[0];
          g.row++;
        }
        if (g.row >= GN) { g.done = true; let lo = Infinity; g.E.forEach(rw => rw.forEach(v => { if (v < lo) lo = v; })); g.lo = lo; g.q = lib.quantile(g.E, 0.97); }
        return g.done;
      }

      // ---------------- drawing: landscape ----------------
      const PX = { x: 40, y: 6, s: LS - 46 };
      const toPx = (p) => { const e = sl().ext; return [PX.x + (p[0] - e[0]) / (e[1] - e[0]) * PX.s, PX.y + PX.s - (p[1] - e[0]) / (e[1] - e[0]) * PX.s]; };
      const sub = (n) => String(n).split('').map(d => '₀₁₂₃₄₅₆₇₈₉'[+d] || d).join('');
      function drawLand() {
        const c = cv.ctx, s = sl(), g = grids[S.si]; cv.clear();
        c.save(); c.beginPath(); c.rect(PX.x, PX.y, PX.s, PX.s); c.clip();
        if (g && g.done) {
          lib.heatmap(c, g.E, PX.x, PX.y, PX.s, PX.s, { key: 'lb-' + S.si, range: [g.lo, g.q], gamma: 0.9 });
          const lv = Array.from({ length: 18 }, (_, i) => g.lo + (g.q - g.lo) * (i + 0.5) / 18);
          lib.contours(c, g.E, PX.x, PX.y, PX.s, PX.s, lv, { color: 'rgba(17,17,17,0.2)', width: 1 });
        } else {
          c.fillStyle = C.blue4; c.fillRect(PX.x, PX.y, PX.s, PX.s);
          lib.text(c, 'computing the landscape… ' + (g ? Math.round(100 * g.row / GN) : 0) + '%', PX.x + PX.s / 2, PX.y + PX.s / 2, { size: 13, kind: 'mono', color: C.muted, align: 'center', baseline: 'middle' });
        }
        // p(true) = 1/2 boundary
        const e = s.ext, nR = Vn - 2, ptsB = [];
        if (s.truth === s.B) { for (let i = 0; i <= 80; i++) { const a = e[0] + (e[1] - e[0]) * i / 80; ptsB.push(toPx([a, Math.log(Math.exp(a) + nR)])); } }
        else if (s.truth === s.A) { for (let i = 0; i <= 80; i++) { const b = e[0] + (e[1] - e[0]) * i / 80; ptsB.push(toPx([Math.log(Math.exp(b) + nR), b])); } }
        if (ptsB.length) { lib.line(c, ptsB, { color: 'rgba(255,255,255,0.75)', width: 3.2 }); lib.line(c, ptsB, { color: C.ink, width: 1.2, dash: [5, 4] }); }
        c.restore();
        if (ptsB.length) {
          const lab = s.truth === s.B ? `p('${s.truth}') > ½ above` : `p('${s.truth}') > ½ right`;
          const lp = s.truth === s.B ? toPx([e[0] + 0.4, Math.log(Math.exp(e[0]) + nR) + 1.9]) : toPx([Math.log(Math.exp(e[0]) + nR) + 0.4, e[0] + 1.6]);
          const m = lib.measure(c, lab, { size: 12, kind: 'mono' });
          c.save(); c.fillStyle = 'rgba(255,255,255,0.85)'; c.fillRect(lp[0] - 3, lp[1] - 2, m.w + 6, 17); c.restore();
          lib.text(c, lab, lp[0], lp[1], { size: 12, kind: 'mono', color: C.ink });
        }
        // frame + ticks
        c.strokeStyle = C.ink; c.lineWidth = 1; c.strokeRect(PX.x + 0.5, PX.y + 0.5, PX.s - 1, PX.s - 1);
        for (let v = Math.ceil(e[0] / 4) * 4; v <= e[1]; v += 4) {
          const [x] = toPx([v, 0]), [, y] = toPx([0, v]);
          lib.text(c, String(v), x, PX.y + PX.s + 5, { size: 12, kind: 'mono', color: C.muted, align: 'center' });
          lib.text(c, String(v), PX.x - 6, y, { size: 12, kind: 'mono', color: C.muted, align: 'right', baseline: 'middle' });
        }
        lib.text(c, `logit of '${s.A}' →`, PX.x + PX.s, PX.y + PX.s + 20, { size: 12, kind: 'mono', color: C.ink, align: 'right' });
        c.save(); c.translate(9, PX.y + 2); c.rotate(-Math.PI / 2); lib.text(c, `logit of '${s.B}' →`, 0, 0, { size: 12, kind: 'mono', color: C.ink, align: 'right', baseline: 'middle' }); c.restore();
        if (!sim) return;
        // candidates
        c.save(); c.beginPath(); c.rect(PX.x, PX.y, PX.s, PX.s); c.clip();
        const t = Math.min(S.t, sim.N), w = winnerAt(t);
        sim.cands.forEach((cd, j) => {
          if (j === w || j === S.inspect) return;
          const px = cd.path.slice(0, t + 1).map(toPx);
          lib.line(c, px, { color: 'rgba(17,17,17,0.38)', width: 1.1 });
        });
        const hl = (j, col, wd) => { const px = sim.cands[j].path.slice(0, t + 1).map(toPx); lib.line(c, px, { color: 'rgba(255,255,255,0.85)', width: wd + 2.6 }); lib.line(c, px, { color: col, width: wd }); };
        if (S.inspect >= 0 && S.inspect < sim.cands.length && S.inspect !== w) hl(S.inspect, C.ink, 1.8);
        hl(w, C.blue, 2.2);
        // starts
        sim.cands.forEach((cd) => { const p = toPx(cd.path[0]); lib.dot(c, p[0], p[1], 3.2, '#fff', { stroke: C.ink, lw: 1.1 }); });
        // update decomposition for a single (or inspected) candidate
        const showJ = sim.cands.length <= 2 ? w : (S.inspect >= 0 ? S.inspect : -1);
        if (showJ >= 0 && t < sim.N) {
          const cd = sim.cands[showJ], y = cd.path[t], gs = cd.G[t], nz = cd.Nz[t];
          const p0 = toPx(y), p1 = toPx([y[0] + gs[0], y[1] + gs[1]]), p2 = toPx([y[0] + gs[0] + nz[0], y[1] + gs[1] + nz[1]]);
          if (Math.hypot(p1[0] - p0[0], p1[1] - p0[1]) > 4) { lib.arrow(c, p0[0], p0[1], p1[0], p1[1], { color: C.ink, width: 1.5, head: 8, dash: [4, 3] }); lib.text(c, '−α∇E', (p0[0] + p1[0]) / 2 + 6, (p0[1] + p1[1]) / 2 - 16, { size: 12, kind: 'mono', color: C.ink }); }
          if (Math.hypot(p2[0] - p1[0], p2[1] - p1[1]) > 4) { lib.arrow(c, p1[0], p1[1], p2[0], p2[1], { color: C.blue, width: 1.8, head: 8 }); lib.text(c, 'η' + sub(t), (p1[0] + p2[0]) / 2 + 6, (p1[1] + p2[1]) / 2 + 2, { size: 13, kind: 'mono', color: C.blue }); }
        }
        // current positions
        sim.cands.forEach((cd, j) => {
          if (j === w) return; const p = toPx(cd.path[t]);
          lib.dot(c, p[0], p[1], j === S.inspect ? 5 : 4, j === S.inspect ? C.ink : '#fff', { stroke: C.ink, lw: 1.3 });
        });
        const pw = toPx(sim.cands[w].path[t]);
        c.save(); c.strokeStyle = '#fff'; c.lineWidth = 3; c.beginPath(); c.arc(pw[0], pw[1], 10, 0, 7); c.stroke(); c.strokeStyle = C.blue; c.lineWidth = 1.8; c.beginPath(); c.arc(pw[0], pw[1], 10, 0, 7); c.stroke(); c.restore();
        lib.dot(c, pw[0], pw[1], 5, C.blue, { stroke: '#fff', lw: 1.5 });
        c.restore();
        const lbl = t === sim.N ? 'ŷ*' : 'lowest now';
        const lx = Math.min(PX.x + PX.s - 70, pw[0] + 13), ly = Math.max(PX.y + 4, Math.min(PX.y + PX.s - 18, pw[1] - 22));
        const m = lib.measure(c, lbl, { size: 13, kind: 'mono' });
        c.save(); c.fillStyle = 'rgba(255,255,255,0.88)'; c.fillRect(lx - 3, ly - 1, m.w + 6, 18); c.restore();
        lib.text(c, lbl, lx, ly, { size: 13, kind: 'mono', color: C.blue, weight: 600 });
      }

      // ---------------- drawing: energy per candidate ----------------
      function niceTicks(lo, hi, n) { const span = hi - lo; if (!(span > 0)) return [lo]; const mag = Math.pow(10, Math.floor(Math.log10(span / n))); const st = [1, 2, 2.5, 5, 10].map(s => s * mag).find(s => span / s <= n) || 10 * mag; const out = []; for (let v = Math.ceil(lo / st) * st; v <= hi + 1e-9; v += st) out.push(+v.toFixed(6)); return out; }
      function drawSide() {
        const c = pv.ctx; pv.clear(); if (!sim) return;
        const t = Math.min(S.t, sim.N), w = winnerAt(t);
        let lo = Infinity, hi = -Infinity; sim.cands.forEach(cd => cd.E.forEach(v => { lo = Math.min(lo, v); hi = Math.max(hi, v); }));
        const pad = (hi - lo) * 0.06 || 0.1; lo -= pad; hi += pad;
        const ax = lib.axes(c, { x: 46, y: 10, w: 240, h: 200, xlim: [0, sim.N], ylim: [lo, hi], xticks: niceTicks(0, sim.N, 4), yticks: niceTicks(lo, hi, 4), yfmt: (v) => v.toFixed(1), size: 12 });
        lib.text(c, 'step i', 46 + 240, 236, { size: 12, kind: 'mono', color: C.muted, align: 'right' });
        lib.text(c, 'E', 6, 6, { size: 12, kind: 'mono', color: C.muted });
        c.save(); c.strokeStyle = C.blue3; c.lineWidth = 1; c.setLineDash([3, 3]); c.beginPath(); c.moveTo(ax.X(t), 10); c.lineTo(ax.X(t), 210); c.stroke(); c.restore();
        sim.cands.forEach((cd, j) => { if (j === w || j === S.inspect) return; lib.plot(c, ax, cd.E.slice(0, t + 1).map((e, i) => [i, e]), { color: 'rgba(17,17,17,0.35)', width: 1.1 }); });
        if (S.inspect >= 0 && S.inspect < sim.cands.length && S.inspect !== w) lib.plot(c, ax, sim.cands[S.inspect].E.slice(0, t + 1).map((e, i) => [i, e]), { color: C.ink, width: 1.8 });
        lib.plot(c, ax, sim.cands[w].E.slice(0, t + 1).map((e, i) => [i, e]), { color: C.blue, width: 2.4 });
        const we = sim.cands[w].E[t]; lib.dot(c, ax.X(t), ax.Y(we), 4, C.blue);
      }
      function readout() {
        if (!sim) { ro.innerHTML = ''; return; }
        const t = Math.min(S.t, sim.N), w = winnerAt(t), s = sl();
        const ce = sim.cands.map(cd => -Math.log(Math.max(1e-12, cd.PT[t])));
        const avg = ce.reduce((a, b) => a + b, 0) / ce.length, orc = Math.min(...ce);
        const insp = S.inspect >= 0 && S.inspect < sim.cands.length ? `<span>clicked #${S.inspect + 1} · E <b>${fx(sim.cands[S.inspect].E[t])}</b> · −log p <b>${fx(ce[S.inspect])}</b></span>` : '';
        ro.innerHTML = `<span>step <b>${t}</b>/${sim.N} · NFEs <b>${sim.cands.length * t}</b></span>` +
          `<span>${t === sim.N ? 'ŷ*' : 'lowest now'} #${w + 1} · E <b>${fx(sim.cands[w].E[t])}</b> · p('${s.truth}') <b>${fx(sim.cands[w].PT[t], 2)}</b></span>` +
          `<span>−log p('${s.truth}') · picked <b>${fx(ce[w])}</b></span><span>average <b>${fx(avg)}</b> · oracle <b>${fx(orc)}</b></span>` + insp;
      }

      // ---------------- bottom: live sweep ----------------
      const SIG = { slice: [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3], full: [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2] };
      const RMAX = { slice: 40, full: 60 };
      let SW = null, heldOut = null;
      function heldOutPositions() {
        if (heldOut) return heldOut;
        const txt = (D.corpus && D.corpus.samples && D.corpus.samples.val) || '', md = M0(), out = [];
        for (let i = 40; i < txt.length && out.length < 60; i += 6) { const ch = md.norm(txt[i]); if (!ch || !md.stoi.has(ch)) continue; out.push({ ctx: txt.slice(i - 40, i), t: md.stoi.get(ch) }); }
        return (heldOut = out);
      }
      function restartSweep() {
        const mode = S.sweepMode, sig = SIG[mode];
        SW = { key: [mode, S.si, S.alpha, S.N, S.M, S.sched].join('|'), mode, si: S.si, alpha: S.alpha, N: S.N, M: S.M, sched: S.sched, sig, acc: sig.map(() => ({ n: 0, single: 0, pick: 0, orc: 0 })), r: 0, k: 0, done: false, R: RMAX[mode] };
        kick();
      }
      function sweepUnit() {
        const W = SW, md = M0(), sigma = W.sig[W.k], ce = [], E = [];
        if (W.mode === 'slice') {
          const ev = planeFor(W.si), out = new Array(6);
          for (let j = 0; j < W.M; j++) {
            const r = lib.rng(W.r * 104729 + j * 7919 + 101); let a = r.normal(), b = r.normal();
            for (let i = 0; i < W.N; i++) { ev(a, b, true, out); const s = noiseAt(i, W.N, sigma, W.sched); a = Math.max(-40, Math.min(40, a - W.alpha * out[4] + s * r.normal())); b = Math.max(-40, Math.min(40, b - W.alpha * out[5] + s * r.normal())); }
            ev(a, b, false, out); E.push(out[0]); ce.push(-Math.log(Math.max(1e-12, pTrue(out, W.si))));
          }
        } else {
          const pos = heldOutPositions()[W.r], cx = md.context(pos.ctx), N = 8, al = md.alpha0, y = new Float64Array(Vn), g = new Float64Array(Vn), p = new Float64Array(Vn);
          for (let j = 0; j < W.M; j++) {
            const r = lib.rng(W.r * 104729 + j * 7919 + 202); for (let v = 0; v < Vn; v++) y[v] = r.normal();
            for (let i = 0; i < N; i++) { md.evalE(cx, y, g); const s = noiseAt(i, N, sigma, W.sched); for (let v = 0; v < Vn; v++) y[v] = y[v] - al * g[v] + (s > 0 ? s * r.normal() : 0); }
            E.push(md.evalE(cx, y, null, p)); ce.push(-Math.log(Math.max(1e-12, p[pos.t])));
          }
        }
        let w = 0; E.forEach((e, j) => { if (e < E[w]) w = j; });
        const A = W.acc[W.k]; A.n++; A.single += ce[0]; A.pick += ce[w]; A.orc += Math.min(...ce);
        W.k++; if (W.k >= W.sig.length) { W.k = 0; W.r++; if (W.r >= Math.min(W.R, W.mode === 'full' ? heldOutPositions().length : W.R)) W.done = true; }
      }
      function sweepWork(ms) { if (!SW || SW.done || S.tab !== 'sweep') return false; const t0 = performance.now(); while (!SW.done && performance.now() - t0 < ms) sweepUnit(); return true; }
      function drawSweep() {
        const c = bv.ctx; bv.clear(); if (!SW) return;
        const W = SW, sig = W.sig, ser = ['single', 'pick', 'orc'].map(k => sig.map((s, i) => W.acc[i].n ? [s, W.acc[i][k] / W.acc[i].n] : null).filter(Boolean));
        const all = ser.flat().map(p => p[1]).filter(v => v > 0);
        const ymin = W.mode === 'slice' ? 0.01 : 1, ymax = W.mode === 'slice' ? 20 : 20;
        const ax = lib.axes(c, { x: 50, y: 18, w: BW - 250, h: BH - 58, xlim: [0, sig[sig.length - 1]], ylim: [ymin, ymax], ylog: true, xticks: W.mode === 'slice' ? [0, 0.5, 1, 1.5, 2, 2.5, 3] : [0, 0.5, 1, 1.5, 2], yticks: W.mode === 'slice' ? [0.01, 0.1, 1, 10] : [1, 2, 5, 10, 20], yfmt: (v) => String(v), size: 12 });
        lib.text(c, 'noise σ', 50 + BW - 250, BH - 20, { size: 12, kind: 'mono', color: C.muted, align: 'right' });
        lib.text(c, '−log p(true), nats (log)', 4, 0, { size: 11, kind: 'mono', color: C.muted });
        const clipY = (pts) => pts.map(p => [p[0], Math.max(ymin, Math.min(ymax, p[1]))]);
        c.save(); c.strokeStyle = C.blue3; c.lineWidth = 1; c.setLineDash([3, 3]); const xs = ax.X(Math.min(S.sigma, sig[sig.length - 1])); c.beginPath(); c.moveTo(xs, 18); c.lineTo(xs, BH - 40); c.stroke(); c.restore();
        lib.plot(c, ax, clipY(ser[0]), { color: C.ink, width: 1.6, dash: [6, 4], markers: 2.5 });
        lib.plot(c, ax, clipY(ser[2]), { color: C.muted, width: 1.4, dash: [1.5, 3.5], markers: 2.2 });
        lib.plot(c, ax, clipY(ser[1]), { color: C.blue, width: 2.4, markers: 3.2 });
        // legend
        const lx = BW - 184; let ly = 14;
        const leg = (col, dash, label, sub_) => { c.save(); c.strokeStyle = col; c.lineWidth = dash ? 1.6 : 2.4; if (dash) c.setLineDash(dash); c.beginPath(); c.moveTo(lx, ly + 7); c.lineTo(lx + 22, ly + 7); c.stroke(); c.restore(); lib.text(c, label, lx + 28, ly, { size: 12, kind: 'mono', color: C.ink }); if (sub_) lib.text(c, sub_, lx + 28, ly + 15, { size: 11, kind: 'mono', color: C.muted }); ly += sub_ ? 36 : 22; };
        leg(C.ink, [6, 4], 'one path', 'M = 1');
        leg(C.blue, null, 'energy pick', 'best of M by E');
        leg(C.muted, [1.5, 3.5], 'oracle', 'best of M by label');
        const nDone = Math.min(...W.acc.map(a => a.n)), R = W.mode === 'full' ? Math.min(W.R, heldOutPositions().length) : W.R;
        lib.text(c, W.done ? 'done' : 'running…', lx, ly + 2, { size: 11, kind: 'mono', color: W.done ? C.muted : C.blue });
        lib.text(c, nDone + '/' + R + (W.mode === 'full' ? ' positions' : ' runs') + ' per σ', lx, ly + 16, { size: 11, kind: 'mono', color: C.muted });
        if (all.some(v => v > ymax)) lib.text(c, 'values > 20 clipped', lx, ly + 30, { size: 11, kind: 'mono', color: C.muted });
        const md = M0();
        bro.innerHTML = W.mode === 'slice'
          ? `<span>slice "${sl().seg}" · M <b>${W.M}</b> · N <b>${W.N}</b> · α <b>${W.alpha}</b> · ${W.sched === 'anneal' ? 'annealed' : 'constant'} noise</span>`
          : `<span>54-D · M <b>${W.M}</b> · N <b>8</b> · α <b>${md ? md.alpha0 : 10}</b> (trained) · ${W.sched === 'anneal' ? 'annealed' : 'constant'} noise · positions from held-out RedPajama text</span>`;
      }

      // ---------------- bottom: stored toy best-of-M ----------------
      let live54 = null;
      function liveFull() {
        const key = [S.si, S.M].join('|'); if (live54 && live54.key === key) return live54;
        const md = M0(), s = sl(), cx = md.context(s.ctx), y = new Float64Array(Vn), g = new Float64Array(Vn), p = new Float64Array(Vn), out = [];
        for (let j = 0; j < Math.min(S.M, 12); j++) {
          const r = lib.rng(9000 + j * 31); for (let v = 0; v < Vn; v++) y[v] = r.normal();
          for (let i = 0; i < 8; i++) { md.evalE(cx, y, g); for (let v = 0; v < Vn; v++) y[v] -= md.alpha0 * g[v]; }
          const E = md.evalE(cx, y, null, p); let top = 0; for (let v = 1; v < Vn; v++) if (p[v] > p[top]) top = v;
          out.push({ E, top: md.disp[top], pt: p[md.stoi.get(s.truth)] });
        }
        let w = 0; out.forEach((o, j) => { if (o.E < out[w].E) w = j; });
        return (live54 = { key, out, w });
      }
      function drawFull() {
        const c = bv.ctx; bv.clear(); const B = D.bon; if (!B) return;
        const st = B.settings[S.bonN], d = st.datasets[S.bonDs] || st.datasets.val, Ms = B.M;
        const vals = [].concat(d.energy_select_ce, d.oracle_ce, d.mean_ce);
        let lo = Math.min(...vals), hi = Math.max(...vals); const pad = (hi - lo) * 0.12; lo -= pad; hi += pad;
        const ax = lib.axes(c, { x: 50, y: 18, w: BW - 250, h: BH - 58, xlim: [Ms[0], Ms[Ms.length - 1]], xlog: true, ylim: [lo, hi], xticks: Ms, yticks: niceTicks(lo, hi, 4), yfmt: (v) => v.toFixed(2), size: 12 });
        lib.text(c, 'candidates M (log scale)', 50 + BW - 250, BH - 20, { size: 12, kind: 'mono', color: C.muted, align: 'right' });
        lib.text(c, 'held-out loss, nats/char', 4, 0, { size: 11, kind: 'mono', color: C.muted });
        lib.plot(c, ax, Ms.map((m, i) => [m, d.mean_ce[i]]), { color: C.ink, width: 1.6, dash: [6, 4], markers: 2.5 });
        lib.plot(c, ax, Ms.map((m, i) => [m, d.oracle_ce[i]]), { color: C.muted, width: 1.4, dash: [1.5, 3.5], markers: 2.2 });
        lib.plot(c, ax, Ms.map((m, i) => [m, d.energy_select_ce[i]]), { color: C.blue, width: 2.4, markers: 3.2 });
        const last = Ms.length - 1;
        lib.text(c, d.energy_select_ce[last].toFixed(3), ax.X(Ms[last]) - 4, ax.Y(d.energy_select_ce[last]) - 18, { size: 12, kind: 'mono', color: C.blue, align: 'right' });
        lib.text(c, d.oracle_ce[last].toFixed(3), ax.X(Ms[last]) - 4, ax.Y(d.oracle_ce[last]) + 6, { size: 12, kind: 'mono', color: C.muted, align: 'right' });
        const lx = BW - 184; let ly = 14;
        const leg = (col, dash, label, sub_) => { c.save(); c.strokeStyle = col; c.lineWidth = dash ? 1.6 : 2.4; if (dash) c.setLineDash(dash); c.beginPath(); c.moveTo(lx, ly + 7); c.lineTo(lx + 22, ly + 7); c.stroke(); c.restore(); lib.text(c, label, lx + 28, ly, { size: 12, kind: 'mono', color: C.ink }); if (sub_) lib.text(c, sub_, lx + 28, ly + 15, { size: 11, kind: 'mono', color: C.muted }); ly += sub_ ? 36 : 22; };
        leg(C.blue, null, 'energy pick', 'Algorithm 2');
        leg(C.ink, [6, 4], 'average', 'a random candidate');
        leg(C.muted, [1.5, 3.5], 'oracle', 'best by true label');
        lib.text(c, `N = ${st.N} steps, no noise`, lx, ly + 4, { size: 11, kind: 'mono', color: C.muted });
        const L = liveFull(), s = sl();
        bro.innerHTML = `<span>this context, live in 54-D (α ${M0().alpha0}, 8 steps, σ 0): candidates → ` + L.out.map((o, j) => `<b style="${j === L.w ? '' : 'color:var(--ink2);font-weight:400'}">${o.top.replace(/</g, '&lt;')}</b>`).join(' ') +
          ` · ŷ* = '${L.out[L.w].top}' (E ${fx(L.out[L.w].E)}) · true '${s.truth}', p at ŷ* <b>${fx(L.out[L.w].pt)}</b></span>`;
      }

      // ---------------- bottom: Table 2 ----------------
      const T2 = { x0: 186, y0: 30, rh: 34 };
      const t2Row = (py) => { const r = Math.floor((py - T2.y0) / T2.rh); return r >= 0 && r < TABLE2.length ? r : -1; };
      function drawT2() {
        const c = bv.ctx; bv.clear();
        const x0 = T2.x0, w = BW - x0 - 70, X = (v) => x0 + (v + 2) / 22 * w;
        // header + legend
        c.fillStyle = C.ink; c.fillRect(x0, 9, 12, 8); lib.text(c, 'thinking longer + self-verification', x0 + 18, 6, { size: 12, kind: 'mono', color: C.ink });
        c.strokeStyle = C.ink; c.lineWidth = 1; c.strokeRect(x0 + 300.5, 9.5, 12, 7); lib.text(c, 'thinking longer', x0 + 318, 6, { size: 12, kind: 'mono', color: C.ink });
        TABLE2.forEach((r, i) => {
          const y = T2.y0 + i * T2.rh, hl = i === S.hoverRow || (S.hoverRow < 0 && (r.full || i === 2) && S.tab === 'table2' && emphT2);
          if (hl) { c.fillStyle = C.blue4; c.fillRect(0, y - 2, BW, T2.rh - 2); }
          lib.text(c, r.name, x0 - 10, y + 6, { size: 12, kind: 'mono', color: r.full ? C.ink : C.ink2 || C.ink, align: 'right', weight: r.full ? 700 : 400 });
          // thinking longer: outlined bar
          const z = X(0);
          c.strokeStyle = C.ink; c.lineWidth = 1; c.fillStyle = '#fff';
          const tl0 = Math.min(z, X(r.tl)), tlw = Math.max(1.5, Math.abs(X(r.tl) - z));
          c.fillRect(tl0, y + 1, tlw, 11); c.strokeRect(tl0 + 0.5, y + 1.5, tlw - 1, 10);
          lib.text(c, r.tl.toFixed(2), Math.max(z, X(r.tl)) + 5, y, { size: 11, kind: 'mono', color: C.ink });
          // + self-verification: filled blue
          const sv0 = Math.min(z, X(r.sv)), svw = Math.max(1.5, Math.abs(X(r.sv) - z));
          c.fillStyle = C.blue; c.fillRect(sv0, y + 14, svw, 11);
          lib.text(c, r.sv.toFixed(2), Math.max(z, X(r.sv)) + 5, y + 13, { size: 11, kind: 'mono', color: C.blue, weight: 600 });
        });
        const yb = T2.y0 + TABLE2.length * T2.rh;
        c.strokeStyle = C.faint; c.lineWidth = 1; c.beginPath(); c.moveTo(X(0) + 0.5, T2.y0 - 4); c.lineTo(X(0) + 0.5, yb); c.stroke();
        [0, 5, 10, 15, 20].forEach(v => lib.text(c, v + '%', X(v), yb + 2, { size: 11, kind: 'mono', color: C.muted, align: 'center' }));
        const r = TABLE2[S.hoverRow >= 0 ? S.hoverRow : 2];
        bro.innerHTML = `<span><b>${r.name}</b>: ${r.why}</span><span style="color:var(--muted)">% perplexity improvement on OOD BigBench Dyck (p.10) · hover or tap a row</span>`;
      }
      let emphT2 = true;

      function drawBottom() { if (S.tab === 'sweep') drawSweep(); else if (S.tab === 'full') drawFull(); else drawT2(); }
      function syncTab() {
        segTab.set(S.tab); subRow.innerHTML = '';
        if (S.tab === 'sweep') { subRow.appendChild(h('span', { class: 'fig-label' }, 'space')); subRow.appendChild(segMode.el); segMode.set(S.sweepMode); }
        if (S.tab === 'full') { subRow.appendChild(segDs.el); subRow.appendChild(segBN.el); }
        if (S.tab === 'table2') subRow.appendChild(h('span', { class: 'fig-sub' }, 'Table 2 · removing one regularizer from training at a time'));
        if (S.tab === 'sweep' && !SW) restartSweep();
        drawBottom(); kick();
      }

      // ---------------- loop ----------------
      let acc = 0;
      const loop = lib.loop((dt) => {
        if (!ctx.visible()) return false;
        let busy = false;
        // landscape grid first
        if (!(grids[S.si] && grids[S.si].done)) { gridStep(S.si, 10); drawLand(); return true; }
        if (S.playing && sim) {
          acc += dt; const rate = Math.max(8, sim.N / 2.2);
          while (acc > 1 / rate) { acc -= 1 / rate; S.t++; }
          if (S.t >= sim.N) { S.t = sim.N; S.playing = false; }
          draw(); busy = true;
        }
        if (sweepWork(7)) { drawBottom(); busy = true; }
        return busy || S.playing;
      });
      function kick() { draw(); if (ctx.visible()) loop.start(); }
      function draw() { drawLand(); drawSide(); readout(); }

      // inspect a candidate by clicking near its current position
      cv.canvas.addEventListener('click', (ev) => {
        if (!sim) return; const [px, py] = cv.toLocal(ev), t = Math.min(S.t, sim.N);
        let best = -1, bd = 18; sim.cands.forEach((cd, j) => { const p = toPx(cd.path[t]); const d = Math.hypot(p[0] - px, p[1] - py); if (d < bd) { bd = d; best = j; } });
        S.inspect = best === S.inspect ? -1 : best; draw();
      });

      function setAll(o) {
        if (o.si != null) { S.si = o.si; segCtx.set(o.si); }
        if (o.M != null) { S.M = o.M; slM.set(o.M); }
        if (o.sigma != null) { S.sigma = o.sigma; slS.set(o.sigma); }
        if (o.N != null) { S.N = o.N; slN.set(o.N); }
        if (o.alpha != null) { S.alpha = o.alpha; slA.set(o.alpha); }
        if (o.sched) { S.sched = o.sched; segSch.set(o.sched); }
        if (o.tab) S.tab = o.tab;
        if (o.sweepMode) S.sweepMode = o.sweepMode;
        if (o.seed != null) S.seed = o.seed;
        S.inspect = -1;
      }
      const STEPS = [
        { M: 1, sigma: 0, seed: 3, tab: 'sweep', sweepMode: 'slice' },
        { M: 1, sigma: 1, seed: 4, tab: 'sweep', sweepMode: 'slice' },
        { M: 8, sigma: 0, seed: 1, tab: 'sweep', sweepMode: 'slice' },
        { M: 8, sigma: 1, seed: 1, tab: 'sweep', sweepMode: 'slice' },
        { M: 8, sigma: 1, seed: 1, tab: 'sweep', sweepMode: 'slice' },
        { M: 8, sigma: 0.5, seed: 1, tab: 'sweep', sweepMode: 'full' },
        { M: 8, sigma: 1, seed: 1, tab: 'table2' },
      ];
      let curStep = 0;
      return {
        step(i) {
          curStep = i; const o = STEPS[i] || STEPS[0];
          const prevKey = SW && SW.key;
          setAll(Object.assign({ si: i === 5 ? S.si : 0, N: 20, alpha: 5, sched: 'const' }, o));
          if (i < 5 && S.si !== 0) { S.si = 0; segCtx.set(0); }
          emphT2 = true; S.hoverRow = -1;
          if (!ctx.visible()) { pendingAnim = true; sim = null; SW = null; syncTabSilently(); return; }
          recompute(true);
          const key = [S.sweepMode, S.si, S.alpha, S.N, S.M, S.sched].join('|');
          if (!SW || SW.key !== key || prevKey !== key) restartSweep();
          syncTab();
        },
        show() {
          if (!sim) { simulate(); S.t = pendingAnim && !reduced ? 0 : S.N; S.playing = pendingAnim && !reduced; }
          if (!SW) restartSweep();
          syncTab(); kick();
        },
        hide() { loop.stop(); },
      };
      function syncTabSilently() { segTab.set(S.tab); }
    },
  });
})();
