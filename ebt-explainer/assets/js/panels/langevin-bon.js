/* Panel: exploring with Langevin noise, choosing by energy (Eq. 2, Algorithm 2, Table 2).
   Figure: a real 2-D slice of the toy character-level text EBT (data/text.json weights, decoded and run in the browser):
   two logits move, the other 52 are frozen at 0. M candidates descend with Langevin noise; the lowest final energy wins.
   Beside it, the energy of every candidate per step. */
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
      if (cache.size > 48) cache.delete(cache.keys().next().value);
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
    { ctx: ' educated at highgate school in london, ', A: 't', B: 'a', truth: 'a', ext: [-4, 14], seg: '…in london, _' },
    { ctx: 'd and harvard universities. he was in th', A: 'e', B: 'a', truth: 'e', ext: [-4, 12], seg: '…he was in th_' },
  ];
  EBT.panel({
    id: 'langevin-bon',
    nav: 'Noise and self-verification',
    title: 'Exploring with noise, choosing by energy',
    lede: 'Descent from one start finds the nearest basin. Run several noisy descents and keep the candidate the model itself scores lowest: extra compute becomes a search, with no outside judge.',
    text: `
      <p><b>Self-verification</b> (Algorithm 2, p.7): draw $M$ random starts, run $N$ descent steps on each, return the lowest-energy candidate.</p>
      <div class="eq">$$\\hat y^{*} \\;=\\; \\arg\\min_{j=1..M}\\; E_\\theta\\big(x,\\hat y_{N,j}\\big)$$<span class="why">Cost: M·N function evaluations.</span></div>
      <p>Choosing only helps if candidates differ. <b>Langevin dynamics</b> adds a random kick to every update (Eq. 2):</p>
      <div class="eq">$$\\hat y_{i+1} \\;=\\; \\hat y_i - \\alpha\\,\\nabla_{\\hat y}E_\\theta(x,\\hat y_i) + \\eta_i$$<span class="why">η<sub>i</sub> ~ N(0, σ): Gaussian noise of size σ.</span></div>
      <p>The landscape is real: our toy character-level EBT on "…school in london, _" (true next character 'a'). Its prediction is 54 logits; we freeze 52 at 0 and move the logits of 't' and 'a'.</p>`,
    steps: [
      { label: 'One path, no noise', html: '<p>Plain descent slides into a shallow basin near the origin where \'a\' gets about 11%. The energy stops falling, so the model has converged, on a poor answer.</p>' },
      { label: 'Add Langevin noise', html: '<p>Each update is the gradient step (dashed) plus a kick $\\eta_i$ (blue). Sometimes a kick carries the path over a rim into a deeper valley. Press <b>[ new starts ]</b>.</p>' },
      { label: 'Many starts, no noise', html: '<p>$M = 8$, $\\sigma = 0$. All eight fall into the same basin: random starts alone give nothing to choose between.</p>' },
      { label: 'Choose by energy', html: '<p>$M = 8$, $\\sigma = 1$. Candidates spread across basins and the lowest energy (blue ring) lands where \'a\' wins. The model never saw the answer; it ranked its own guesses. The readout adds an oracle that cheats with the label.</p>' },
    ],
    after: `
      <p>In the paper, noise is a <em>training</em> regularizer: without it, "exploration is often limited to paths leading directly to the energy minimum, leaving other regions poorly defined" (p.7). A verifier can only rank regions it was shaped on. Table 2 (p.10, perplexity gains on BigBench Dyck) shows the trade: without Langevin noise, thinking longer gains more (17.2% vs 7.19%) but verification gains less (17.0% vs 18.7%).</p>
      <p class="note">Caveats. The verifier can be fooled: at small data, best-of-10 was occasionally worse than best-of-2, an "adversarial sample" with low energy (p.28). In our full 54-D toy, noise does not help the energy pick. Algorithm 2 has no noise term; whether inference uses noise is unspecified.</p>`,
    source: [
      { kind: 'toy', note: 'text EBT run live · data/text.json' },
      { kind: 'paper', note: 'Eq. 2, Alg. 2, Table 2' },
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
      const S = { si: 0, alpha: 5, N: 20, M: 8, sigma: 1.0, sched: 'const', seed: 1, t: 20, playing: false, inspect: -1 };
      let sim = null;                   // {cands:[{path,E,PT,G,Nz}], N}
      const grids = {}, planes = {};
      const sl = () => SLICES[S.si];
      const planeFor = (k) => planes[k] || (planes[k] = M0().plane(M0().context(SLICES[k].ctx), M0().stoi.get(SLICES[k].A), M0().stoi.get(SLICES[k].B)));
      const noiseAt = (i, N, sigma, sched) => sched === 'anneal' ? sigma * Math.max(0, 1 - i / (0.6 * N)) : (i < N - 1 ? sigma : 0);
      const pTrue = (o, k) => { const s = SLICES[k]; return s.truth === s.A ? o[1] : s.truth === s.B ? o[2] : o[3]; };

      // ---------------- layout ----------------
      const SWD = Math.max(300, stage.clientWidth || 630), wide = SWD >= 500, sideW = Math.round(Math.max(200, Math.min(360, wide ? SWD - 352 : SWD)));   // logical canvas sizes follow the available width so text stays legible on phones
      const row = h('div', { class: 'fig-row lb-row' }); stage.appendChild(row);
      const FL = lib.frame(row, { label: 'Energy landscape · 2-D slice', sub: 'E<sub>θ</sub>(x, ŷ) on 2 of 54 logits · click a dot' });
      FL.wrap.classList.add('lb-land');
      const LS = 400, cv = lib.canvas(FL.frame, LS, LS, { label: 'Energy landscape of the toy text EBT on a two-logit slice, with the candidate predictions and their descent paths' });
      const FR = lib.frame(row, { label: 'Energy per candidate', sub: 'E per step · lowest final value wins' });
      FR.wrap.classList.add('lb-side');
      const PVW = sideW, PVH = wide ? 222 : 200, pv = lib.canvas(FR.frame, PVW, PVH, { label: 'Energy of every candidate at each step' });
      const ro = h('div', { class: 'readout lb-ro', 'aria-live': 'polite' }); FR.wrap.appendChild(ro);
      const c3 = h('div', { class: 'controls lb-btns' }); FR.wrap.appendChild(c3);
      c3.appendChild(lib.button('new starts', () => { S.seed++; S.inspect = -1; S.stopAt = null; recompute(true); }));
      c3.appendChild(lib.button('run', () => { S.stopAt = null; recompute(true); }, { primary: true }));
      c3.appendChild(lib.button('step', () => { S.playing = false; if (S.t >= S.N) S.t = 0; else S.t++; draw(); }));

      const c1 = h('div', { class: 'controls' }); stage.appendChild(c1);
      const segCtx = lib.segmented({ label: 'Context', options: SLICES.map((s, i) => [i, s.seg]), value: 0, onchange: (v) => { S.si = v; S.inspect = -1; S.stopAt = null; recompute(true); } });
      c1.appendChild(h('span', { class: 'fig-label' }, 'context')); c1.appendChild(segCtx.el);
      const c2 = h('div', { class: 'controls lb-sliders' }); stage.appendChild(c2);
      const slS = lib.slider({ id: 'lb-sigma', label: 'noise σ', min: 0, max: 3, step: 0.05, value: S.sigma, fmt: (v) => v.toFixed(2), oninput: (v) => { S.sigma = v; recompute(false); } });
      const slM = lib.slider({ id: 'lb-M', label: 'candidates M', min: 1, max: 16, step: 1, value: S.M, oninput: (v) => { S.M = v; S.inspect = -1; recompute(false); } });
      const slN = lib.slider({ id: 'lb-N', label: 'steps N', min: 1, max: 40, step: 1, value: S.N, oninput: (v) => { S.N = v; recompute(false); } });
      const slA = lib.slider({ id: 'lb-alpha', label: 'step size α', min: 0.5, max: 12, step: 0.5, value: S.alpha, fmt: (v) => v.toFixed(1), oninput: (v) => { S.alpha = v; recompute(false); } });
      [slS, slM, slN, slA].forEach(s => c2.appendChild(s.el));

      ctx.setCaption('Darker blue = lower energy. Circles: starts ŷ<sub>0,j</sub>. Gray: paths. Blue ring: lowest-energy candidate ŷ*. Dashed: true character at p = ½.');

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
        if (!animate) S.stopAt = null;
        if (!ctx.visible() && !sim) { pendingAnim = animate; return; }
        simulate(); S.t = animate && !reduced ? 0 : (S.stopAt != null ? S.stopAt : S.N); S.playing = animate && !reduced; kick();
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
          lib.heatmap(c, g.E, PX.x, PX.y, PX.s, PX.s, { key: 'langevin-bon-slice-' + S.si, range: [g.lo, g.q], gamma: 0.9 });
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
        // current positions
        sim.cands.forEach((cd, j) => {
          if (j === w) return; const p = toPx(cd.path[t]);
          lib.dot(c, p[0], p[1], j === S.inspect ? 5 : 4, j === S.inspect ? C.ink : '#fff', { stroke: C.ink, lw: 1.3 });
        });
        const pw = toPx(sim.cands[w].path[t]);
        c.save(); c.strokeStyle = '#fff'; c.lineWidth = 3; c.beginPath(); c.arc(pw[0], pw[1], 10, 0, 7); c.stroke(); c.strokeStyle = C.blue; c.lineWidth = 1.8; c.beginPath(); c.arc(pw[0], pw[1], 10, 0, 7); c.stroke(); c.restore();
        lib.dot(c, pw[0], pw[1], 5, C.blue, { stroke: '#fff', lw: 1.5 });
        // update decomposition for a single (or clicked) candidate: gradient step (dashed ink) + Langevin kick (blue)
        const showJ = sim.cands.length <= 2 ? w : (S.inspect >= 0 ? S.inspect : -1);
        if (showJ >= 0 && t < sim.N) {
          const cd = sim.cands[showJ], y = cd.path[t], gs = cd.G[t], nz = cd.Nz[t];
          const p0 = toPx(y), p1 = toPx([y[0] + gs[0], y[1] + gs[1]]), p2 = toPx([y[0] + gs[0] + nz[0], y[1] + gs[1] + nz[1]]);
          const tag = (txt, x, y_, col) => { const m = lib.measure(c, txt, { size: 12, kind: 'mono' }); c.save(); c.fillStyle = 'rgba(255,255,255,0.9)'; c.fillRect(x - 2, y_ - 1, m.w + 4, 16); c.restore(); lib.text(c, txt, x, y_, { size: 12, kind: 'mono', color: col }); };
          if (Math.hypot(p1[0] - p0[0], p1[1] - p0[1]) > 3) { lib.arrow(c, p0[0], p0[1], p1[0], p1[1], { color: '#fff', width: 3.4, head: 9 }); lib.arrow(c, p0[0], p0[1], p1[0], p1[1], { color: C.ink, width: 1.5, head: 8, dash: [4, 3] }); tag('−α∇E', Math.min(p0[0], p1[0]) - 46, Math.max(p0[1], p1[1]) + 6, C.ink); }
          if (Math.hypot(p2[0] - p1[0], p2[1] - p1[1]) > 3) { lib.arrow(c, p1[0], p1[1], p2[0], p2[1], { color: '#fff', width: 3.6, head: 9 }); lib.arrow(c, p1[0], p1[1], p2[0], p2[1], { color: C.blue, width: 1.8, head: 8 }); tag('η' + sub(t), Math.max(p0[0], p1[0], p2[0]) + 14, Math.max(p0[1], p1[1], p2[1]) + 6, C.blue); }
        }
        c.restore();
        const lbl = sim.cands.length === 1 ? 'ŷ' + sub(t) : t === sim.N ? 'ŷ*' : 'lowest now';
        const m = lib.measure(c, lbl, { size: 13, kind: 'mono' });
        const lx = sim.cands.length === 1 ? Math.max(PX.x + 4, pw[0] - 14 - m.w) : Math.min(PX.x + PX.s - 70, pw[0] + 13), ly = Math.max(PX.y + 4, Math.min(PX.y + PX.s - 18, pw[1] - 24));
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
        const aw = PVW - 60, ah = PVH - 50;
        const ax = lib.axes(c, { x: 46, y: 10, w: aw, h: ah, xlim: [0, sim.N], ylim: [lo, hi], xticks: niceTicks(0, sim.N, 4), yticks: niceTicks(lo, hi, 4), yfmt: (v) => v.toFixed(1), size: 12 });
        lib.text(c, 'step i', 46 + aw, PVH - 14, { size: 12, kind: 'mono', color: C.muted, align: 'right' });
        c.save(); c.strokeStyle = C.blue3; c.lineWidth = 1; c.setLineDash([3, 3]); c.beginPath(); c.moveTo(ax.X(t), 10); c.lineTo(ax.X(t), 10 + ah); c.stroke(); c.restore();
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

      // ---------------- loop ----------------
      let acc = 0;
      const loop = lib.loop((dt) => {
        if (!ctx.visible()) return false;
        let busy = false;
        // landscape grid first
        if (!(grids[S.si] && grids[S.si].done)) { gridStep(S.si, 10); drawLand(); return true; }
        if (S.playing && sim) {
          acc += dt; const rate = Math.max(8, sim.N / 2.2), stop = S.stopAt != null ? Math.min(S.stopAt, sim.N) : sim.N;
          while (acc > 1 / rate) { acc -= 1 / rate; S.t++; }
          if (S.t >= stop) { S.t = stop; S.playing = false; }
          draw(); busy = true;
        }
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
        if (o.seed != null) S.seed = o.seed;
        S.inspect = -1;
      }
      const STEPS = [
        { M: 1, sigma: 0, seed: 3 },
        { M: 1, sigma: 1, seed: 4 },
        { M: 8, sigma: 0, seed: 1 },
        { M: 8, sigma: 1, seed: 1 },
      ];
      return {
        step(i) {
          setAll(Object.assign({ si: 0, N: 20, alpha: 5 }, STEPS[i] || STEPS[0]));
          S.stopAt = i === 1 ? 7 : null;
          if (!ctx.visible()) { pendingAnim = true; sim = null; return; }
          recompute(true);
        },
        show() {
          if (!sim) { simulate(); S.t = pendingAnim && !reduced ? 0 : (S.stopAt != null ? S.stopAt : S.N); S.playing = pendingAnim && !reduced; }
          kick();
        },
        hide() { loop.stop(); },
      };
    },
  });
})();
