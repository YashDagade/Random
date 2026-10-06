/* Panel: thinking over a whole vocabulary (text EBTs).
   Top instrument: the toy character-level text EBT (data/text.json weights, run live) thinking about one next character:
   probability bars for the top symbols, the logit update -α∂E/∂ŷ under them (zero-sum through the softmax), energy / entropy / p(true) per step.
   Bottom instrument (tabs): a held-out sentence coloured by energy (live, easy/hard groups, raw vs per-context relative energy)
   and paper Fig 8 (digitized). */
(function () {
  'use strict';

  // Same guarded factory as langevin-bon.js (whichever panel runs first decodes the weights once). See that file for the derivation.
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
    function tail(grad) {
      for (let j = 0; j < H1; j++) s1[j] = a1[j] * sig(a1[j]);
      a2.set(b2); mvAdd(a2, s1, W2, H1, H2);
      let E = b3; for (let j = 0; j < H2; j++) E += a2[j] * sig(a2[j]) * W3[j];
      if (!grad) return E;
      for (let j = 0; j < H2; j++) { const s = sig(a2[j]); da2[j] = W3[j] * s * (1 + a2[j] * (1 - s)); }
      for (let i = 0; i < H1; i++) { let acc = 0; const o = i * H2; for (let j = 0; j < H2; j++) acc += W2[o + j] * da2[j]; const s = sig(a1[i]); da1[i] = acc * s * (1 + a1[i] * (1 - s)); }
      return E;
    }
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

  const SHORT = ['…and q → u', '…in th → e', '…informatio → n', 'new word', 'year digit', 'after a comma'];

  EBT.panel({
    id: 'tokens',
    nav: 'Thinking over a vocabulary',
    title: 'Thinking over a whole vocabulary',
    lede: 'For text, the guess is not a token but a whole distribution over the vocabulary. It starts as noise and is reshaped step by step, with the energy scoring every version.',
    text: `
      <p>In a language EBT, $\\hat y$ holds one logit per vocabulary item (50,277 in the paper, 54 characters in our toy, which runs live here), drawn from $\\mathcal N(0, I)$. A Transformer reads embeddings, so the guess enters as a soft mixture of token embeddings:</p>
      <div class="eq">$$p = \\mathrm{softmax}(\\hat y), \\qquad e = \\sum_{v} p_v\\, W_v$$<span class="why">Fig 2's "Linear Projector"; Listing 1 (p.43–44). Without the softmax, training was unstable (p.43).</span></div>
      <p>Training scores the optimized distribution with cross-entropy against the true token. The energy itself is never supervised (p.37).</p>`,
    steps: [
      { label: 'A guess is a distribution', html: '<p>Step 0: random logits give a lumpy, uninformative distribution. Dashed box: the true next character. Ink ticks: a same-size one-pass model.</p>' },
      { label: 'What the Transformer sees', html: '<p>The guess becomes one embedding mixture $e$, which the network scores against the context. At step 0 it blends dozens of embeddings; a few steps later, nearly one.</p>' },
      { label: 'The gradient through the softmax', html: '<div class="eq">$$\\frac{\\partial E}{\\partial \\hat y_v} = p_v\\Big(g_v - \\sum_u p_u\\, g_u\\Big), \\quad g = \\frac{\\partial E}{\\partial p}$$</div><p>The updates (strip under the bars) sum to zero, so thinking only moves mass between symbols. Each is scaled by $p_v$: a symbol with almost no mass barely moves, so different starts can end on different answers.</p>' },
      { label: 'Think', html: '<p>Press play. Mass collapses onto a few symbols within two steps as energy and entropy fall. Try the "year digit" context and <b>[ new start ]</b>: easy contexts reach the same answer from any start, hard ones do not.</p>' },
      { label: 'Energy levels and uncertainty', html: '<p>Training touches $E$ only through $\\nabla_{\\hat y}E$, so adding any $c(x)$ changes no step and no loss. Energies compare cleanly within one context, which is all Best-of-N needs; across contexts their levels are free. Still, in Fig 8 (digitized) hard tokens such as "fox" stay higher than easy ones such as "the" (p.10–11): emergent, not guaranteed. The normalization is unstated and most change happens at iteration 1. In our toy (sentence tab), raw levels put word-initial characters lower; subtracting each context\'s energy at a uniform guess restores the paper\'s order.</p>' },
    ],
    after: `
      <p>Every intermediate distribution is a usable prediction with a score attached, which lets each token get its own amount of thinking. The price: our same-size one-pass baseline still predicts better, as the paper's Transformer++ does in pretraining perplexity (Table 3).</p>`,
    source: [
      { kind: 'toy', note: 'text EBT run live · data/text.json' },
      { kind: 'paper', note: 'Listing 1 (p.43–44), Fig 8 digitized (approx.), Table 3' },
    ],
    figure(stage, ctx) {
      const { lib } = ctx, h = lib.h, C = lib.C;
      const D = lib.data('text');
      if (!D || !D.weights) { stage.appendChild(h('p', { class: 'callout warn' }, 'Toy text model missing (data/text.json). Run python3 src/bundle_data.py.')); return {}; }
      let model = null; const M0 = () => model || (model = getTextModel(lib));
      const fx = (v, d = 2) => lib.fmt(v, d), reduced = !!lib.reducedMotion;
      const V = D.vocab.length, DISP = D.vocab_display || D.vocab, VIDX = new Map(D.vocab.map((c, i) => [c, i]));
      const disp = (i) => DISP[i];
      const EX = D.examples || [];
      const HM = (lib.data('paper_heatmaps') || {}).heatmaps || [];
      const SWD = Math.max(300, stage.clientWidth || 630), wide = SWD >= 500;
      const DPR = Math.min(2, window.devicePixelRatio || 1);
      // resizable HiDPI canvas (lib.canvas has a fixed logical size; the sentence view needs a height that depends on the text)
      function mkCanvas(parent, w, hh, label) {
        const box = h('div', { class: 'canvas-box' }); parent.appendChild(box);
        const c = h('canvas', { role: 'img', 'aria-label': label }); box.appendChild(c);
        const o = { canvas: c, ctx: c.getContext('2d'), w, h: hh, box,
          size(w2, h2) { if (w2 === o.w && h2 === o.h && c.width) return; o.w = w2; o.h = h2; c.width = Math.round(w2 * DPR); c.height = Math.round(h2 * DPR); o.ctx.setTransform(DPR, 0, 0, DPR, 0, 0); },
          clear() { o.ctx.save(); o.ctx.setTransform(DPR, 0, 0, DPR, 0, 0); o.ctx.fillStyle = '#fff'; o.ctx.fillRect(0, 0, o.w, o.h); o.ctx.restore(); },
          toLocal(ev) { const r = c.getBoundingClientRect(), p = ev.touches ? ev.touches[0] : ev; return [(p.clientX - r.left) / r.width * o.w, (p.clientY - r.top) / r.height * o.h]; } };
        c.width = 0; o.size(w, hh); return o;
      }

      // ---------------- state ----------------
      const S = { ex: 0, custom: 'the capital of france is paris. the capital of italy is r', seed: 1, N: 12, t: 12, playing: false, showGrad: true,
        tab: 'chars', sent: 0, emode: 'raw', st: 8, f8: 0, hover: -1 };
      let tr = null;          // current trace
      const ctxStr = () => S.ex === 'custom' ? S.custom : EX[S.ex].context;
      const trueIdx = () => S.ex === 'custom' ? -1 : VIDX.get(EX[S.ex].true_next);

      // ---------------- layout: context ----------------
      const cRow = h('div', { class: 'controls tk-ctx' }); stage.appendChild(cRow);
      const sel = h('select', { id: 'tk-ex', 'aria-label': 'Context' });
      EX.forEach((e, i) => sel.appendChild(h('option', { value: String(i) }, SHORT[i] || e.label)));
      sel.appendChild(h('option', { value: 'custom' }, 'your own text…'));
      cRow.appendChild(h('label', { class: 'fig-label', for: 'tk-ex' }, 'context')); cRow.appendChild(sel);
      const strip = h('div', { class: 'tk-strip', 'aria-live': 'polite' }); cRow.appendChild(strip);
      const inp = h('input', { type: 'text', id: 'tk-in', 'aria-label': 'Your own context text', spellcheck: 'false', value: S.custom, maxlength: '200' });
      const inpWrap = h('div', { class: 'tk-in', hidden: true }, inp); stage.appendChild(inpWrap);
      sel.addEventListener('change', () => { S.ex = sel.value === 'custom' ? 'custom' : +sel.value; inpWrap.hidden = S.ex !== 'custom'; recompute(true); });
      let inTimer = null; inp.addEventListener('input', () => { clearTimeout(inTimer); inTimer = setTimeout(() => { S.custom = inp.value; recompute(false); }, 180); });

      // ---------------- layout: bars + side plots ----------------
      const row = h('div', { class: 'fig-row tk-row' }); stage.appendChild(row);
      const FB = lib.frame(row, { label: 'The guess p = softmax(ŷᵢ)', sub: 'top symbols · update −α∂E/∂ŷ below' });
      FB.wrap.classList.add('tk-bars');
      const BW = wide ? 400 : 360, BH = 250, bars = mkCanvas(FB.frame, BW, BH, 'Probability of the top next-character candidates at the current thinking step, with the logit update underneath');
      const pipe = h('div', { class: 'readout tk-pipe' }); FB.wrap.appendChild(pipe);
      const FS = lib.frame(row, { label: 'Per step', sub: 'energy · entropy · p(true)' });
      FS.wrap.classList.add('tk-side');
      const sideW = wide ? Math.round(Math.max(190, Math.min(300, SWD - 430))) : 360, SH = wide ? 214 : 210;
      const side = mkCanvas(FS.frame, sideW, SH, 'Energy, entropy and probability of the true character at each thinking step');
      const ro = h('div', { class: 'readout tk-ro', 'aria-live': 'polite' }); FS.wrap.appendChild(ro);

      const c2 = h('div', { class: 'controls tk-play' }); FB.wrap.appendChild(c2);
      const c2b = c2;
      const bPlay = lib.button('play', () => { if (S.playing) { S.playing = false; syncPlay(); return; } if (S.t >= S.N) S.t = 0; S.playing = true; syncPlay(); kick(); }, { primary: true });
      c2b.appendChild(bPlay);
      c2b.appendChild(lib.button('new start', () => { S.seed++; recompute(true); }));
      const slT = lib.slider({ id: 'tk-t', label: 'step i', min: 0, max: S.N, step: 1, value: S.t, oninput: (v) => { S.t = v; S.playing = false; syncPlay(); draw(); } });
      c2.appendChild(slT.el); slT.el.classList.add('tk-tsl');
      function syncPlay() { bPlay.textContent = S.playing ? 'pause' : 'play'; slT.input.max = S.N; slT.set(Math.min(S.t, S.N)); }

      // ---------------- layout: bottom tabs ----------------
      const tabRow = h('div', { class: 'controls tk-tabs' }); stage.appendChild(tabRow);
      const segTab = lib.segmented({ label: 'View', options: [['chars', 'toy sentence · live'], ['fig8', 'paper · Fig 8']], value: 'chars', onchange: (v) => { S.tab = v; syncTab(); } });
      tabRow.appendChild(h('span', { class: 'fig-label' }, 'energy levels, easy vs hard')); tabRow.appendChild(segTab.el);
      const FT = lib.frame(stage, {});
      const subRow = h('div', { class: 'controls tk-sub' }); FT.frame.appendChild(subRow);
      const BWB = wide ? Math.min(640, SWD) : 360;
      const bot = mkCanvas(FT.frame, BWB, 200, 'Bottom chart for the selected view');
      const bro = h('div', { class: 'readout tk-bro' }); FT.frame.appendChild(bro);

      const SENTS = [];
      (D.uncertainty && D.uncertainty.sentences || []).forEach((s, i) => SENTS.push({ name: 'web ' + (i + 1), prefix: s.context_before, text: s.text }));
      const PA = 'the quick brown fox jumps over the lazy dog.', PB = 'system 2 thinking is a challenging but interesting research problem.';
      SENTS.push({ name: 'Fig 8a', prefix: (PB + ' ').slice(-40), text: PA });
      SENTS.push({ name: 'Fig 8b', prefix: (PA + ' ').slice(-40), text: PB });
      const segSent = lib.segmented({ label: 'Sentence', options: SENTS.map((s, i) => [i, s.name]), value: 0, onchange: (v) => { S.sent = v; drawBottom(); kick(); } });
      const segMode = lib.segmented({ label: 'Energy shown', options: [['raw', 'raw E'], ['rel', 'E − E(x, uniform)']], value: 'raw', onchange: (v) => { S.emode = v; drawBottom(); } });
      const segF8 = lib.segmented({ label: 'Fig 8 panel', options: [[0, 'Fig 8a'], [1, 'Fig 8b']], value: 0, onchange: (v) => { S.f8 = v; drawBottom(); } });

      ctx.setCaption('Gray: other symbols · dashed box: true character · ink tick: one-pass baseline');

      // ---------------- thinking trace (live) ----------------
      function y0For(seed) { const r = lib.rng(seed * 2654435761 % 4294967296 + 17), y = new Float64Array(V); for (let v = 0; v < V; v++) y[v] = r.normal(); return y; }
      function think(cxs, y0, N, al) {
        const md = M0(), cx = md.context(cxs); let y = Float64Array.from(y0);
        const out = { E: [], P: [], U: [], H: [], DP: [] };
        for (let i = 0; i <= N; i++) {
          const g = new Float64Array(V), p = new Float64Array(V), dp = new Float64Array(V);
          const E = md.evalE(cx, y, g, p, dp); let H = 0; for (let v = 0; v < V; v++) if (p[v] > 0) H -= p[v] * Math.log(p[v]);
          out.E.push(E); out.P.push(p); out.H.push(H); out.U.push(g.map(x => -al * x)); out.DP.push(dp);
          if (i === N) break;
          const yn = new Float64Array(V); for (let v = 0; v < V; v++) yn[v] = y[v] - al * g[v]; y = yn;
        }
        return out;
      }
      let robust = null;
      function computeRobust() {
        const ti = trueIdx(); if (ti < 0) { robust = null; return; }
        const md = M0(), cx = md.context(ctxStr()), al = md.alpha0, K = 32; let hit = 0;
        const y = new Float64Array(V), g = new Float64Array(V), p = new Float64Array(V);
        for (let k = 0; k < K; k++) {
          const r = lib.rng(5000 + k * 97); for (let v = 0; v < V; v++) y[v] = r.normal();
          for (let i = 0; i < S.N; i++) { md.evalE(cx, y, g); for (let v = 0; v < V; v++) y[v] -= al * g[v]; }
          md.evalE(cx, y, null, p); let top = 0; for (let v = 1; v < V; v++) if (p[v] > p[top]) top = v; if (top === ti) hit++;
        }
        robust = { K, hit, N: S.N };
      }
      let order = [];
      function recompute(animate) {
        if (!M0()) return;
        tr = think(ctxStr(), y0For(S.seed), S.N, M0().alpha0);
        const fin = tr.P[S.N], ti = trueIdx();
        order = Array.from({ length: V }, (_, i) => i).sort((a, b) => fin[b] - fin[a]).slice(0, 10);
        if (ti >= 0 && !order.includes(ti)) order[9] = ti;
        order.sort((a, b) => fin[b] - fin[a]);
        computeRobust();
        S.t = animate && !reduced ? 0 : S.N; S.playing = animate && !reduced; syncPlay(); renderStrip(); kick();
      }
      function renderStrip() {
        const md = M0(), cs = Array.from(md.norm(ctxStr())).slice(-40).join(''), ti = trueIdx();
        const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/\n/g, '⏎');
        strip.innerHTML = `<span class="tk-c">…${esc(cs.slice(-34))}</span><span class="tk-q">${ti >= 0 ? esc(disp(ti)) : '?'}</span>`;
      }

      // ---------------- drawing: bars ----------------
      function drawBars() {
        const c = bars.ctx; bars.clear(); if (!tr) return;
        const t = Math.min(S.t, S.N), p = tr.P[t], u = tr.U[Math.min(t, S.N)], ti = trueIdx();
        const L = 34, R = BW - 6, n = order.length + 1, cw = (R - L) / n, y0 = 28, y1 = 162, Hb = y1 - y0;
        const Y = (v) => y1 - v * Hb;
        // grid
        [0, 0.25, 0.5, 0.75, 1].forEach(v => { c.strokeStyle = v === 0 ? C.faint : C.rule; c.lineWidth = 1; c.beginPath(); c.moveTo(L, Math.round(Y(v)) + 0.5); c.lineTo(R, Math.round(Y(v)) + 0.5); c.stroke(); lib.text(c, v === 0 ? '0' : v === 1 ? '1' : String(v).replace('0.', '.'), L - 6, Y(v), { size: 11, kind: 'mono', color: C.muted, align: 'right', baseline: 'middle' }); });
        lib.text(c, 'p', 6, y0 - 22, { size: 12, kind: 'mono', color: C.muted });
        let rest = 1; order.forEach(k => { rest -= p[k]; }); rest = Math.max(0, rest);
        const base = S.ex !== 'custom' && EX[S.ex] ? new Map(EX[S.ex].baseline_top.map(([ch, q]) => [VIDX.get(ch), q])) : new Map();
        const cols = order.map(k => ({ k, p: p[k], lab: disp(k), u: u[k], base: base.get(k) })).concat([{ k: -1, p: rest, lab: 'other', u: order.reduce((s, k) => s - u[k], 0) }]);
        cols.forEach((col, i) => {
          const cx = L + cw * (i + 0.5), bw = Math.min(22, cw * 0.62);
          const fill = col.k < 0 ? C.faint : C.blue;
          c.fillStyle = fill; const yv = Y(Math.max(0, Math.min(1, col.p))); c.fillRect(cx - bw / 2, yv, bw, y1 - yv);
          const pct = col.p >= 0.995 ? '1' : col.p >= 0.1 ? (col.p * 100).toFixed(0) + '%' : (col.p * 100).toFixed(1) + '%';
          if (yv - 14 < y0 - 2) lib.text(c, pct, cx, yv + 3, { size: 10.5, kind: 'mono', color: '#fff', align: 'center' });
          else lib.text(c, pct, cx, yv - 14, { size: 10.5, kind: 'mono', color: col.k < 0 ? C.muted : C.ink, align: 'center' });
          if (col.base != null) { c.strokeStyle = C.ink; c.lineWidth = 2; const yb = Y(col.base); c.beginPath(); c.moveTo(cx - bw / 2 - 4, yb); c.lineTo(cx + bw / 2 + 4, yb); c.stroke(); }
          lib.text(c, col.k < 0 ? '+' + (V - order.length) : col.lab, cx, y1 + 6, { size: col.k < 0 ? 11 : 15, kind: 'mono', color: col.k < 0 ? C.muted : C.ink, align: 'center' });
          if (col.k === ti && ti >= 0) { c.save(); c.strokeStyle = C.ink; c.lineWidth = 1.2; c.setLineDash([4, 3]); c.strokeRect(cx - cw / 2 + 1.5, y0 - 26, cw - 3, y1 - y0 + 44); c.restore(); lib.text(c, 'true', cx, y0 - 24, { size: 10.5, kind: 'mono', color: C.ink, align: 'center' }); }
        });
        // step-specific emphasis: 'embed' brackets the bars as one embedding mixture, 'grad' enlarges the update strip
        if (S.focus === 'embed') {
          // bracket over all columns, then the path the guess takes into the network, drawn where the update strip usually is
          const yb = y1 + 22; c.strokeStyle = C.blue; c.lineWidth = 1.4; c.beginPath(); c.moveTo(L + 2, yb - 5); c.lineTo(L + 2, yb); c.lineTo(R - 2, yb); c.lineTo(R - 2, yb - 5); c.stroke();
          const boxes = [['p', '54 probs'], ['·W', '54×128'], ['e', '128-d'], ['MLP', 'with h(x)'], ['E', tr.E[t].toFixed(2)]];
          const bw2 = (R - L - 4 * 12) / 5, by = yb + 12;
          boxes.forEach(([a, b], k) => {
            const bx = L + k * (bw2 + 12);
            c.strokeStyle = k === 4 ? C.blue : C.ink; c.lineWidth = 1; c.setLineDash(k === 1 || k === 3 ? [3, 2] : []); c.strokeRect(bx + 0.5, by + 0.5, bw2 - 1, 32); c.setLineDash([]);
            lib.text(c, a, bx + bw2 / 2, by + 3, { size: 12, kind: 'mono', color: k === 4 ? C.blue : C.ink, align: 'center', weight: 600 });
            lib.text(c, b, bx + bw2 / 2, by + 18, { size: 10, kind: 'mono', color: C.muted, align: 'center' });
            if (k < 4) lib.arrow(c, bx + bw2 + 1, by + 16, bx + bw2 + 11, by + 16, { color: C.ink, width: 1.2, head: 5 });
          });
          pipeLine(p, u, t); return;
        }
        // update strip
        const uy = S.focus === 'grad' ? 221 : 225, uh = S.focus === 'grad' ? 25 : 21, umax = Math.max(1e-9, ...cols.map(cc => Math.abs(cc.u)));
        c.strokeStyle = C.faint; c.lineWidth = 1; c.beginPath(); c.moveTo(L, uy + 0.5); c.lineTo(R, uy + 0.5); c.stroke();
        lib.text(c, `next update −α∂E/∂ŷ at step ${t}`, L, 186, { size: 11, kind: 'mono', color: C.muted });
        lib.text(c, 'Δŷ', 6, uy - 7, { size: 12, kind: 'mono', color: C.muted });
        cols.forEach((col, i) => {
          const cx = L + cw * (i + 0.5), bw = Math.min(S.focus === 'grad' ? 20 : 16, cw * 0.5), hgt = col.u / umax * uh;
          c.fillStyle = col.u >= 0 ? C.blue : C.ink; if (col.k < 0) c.fillStyle = col.u >= 0 ? C.blue2 : C.muted;
          if (Math.abs(hgt) >= 0.5) c.fillRect(cx - bw / 2, hgt >= 0 ? uy - hgt : uy, bw, Math.abs(hgt));
          if (S.focus === 'grad' && Math.abs(col.u) / umax > 0.12) lib.text(c, (col.u >= 0 ? '+' : '−') + Math.abs(col.u).toFixed(2), cx, hgt >= 0 ? uy + 3 : uy - 15, { size: 10, kind: 'mono', color: C.ink, align: 'center' });
        });
        if (S.focus === 'grad') { c.save(); c.strokeStyle = C.blue; c.lineWidth = 1.2; c.strokeRect(2.5, 182.5, BW - 5, BH - 185); c.restore(); }
        pipeLine(p, u, t);
      }
      function pipeLine(p, u, t) {
        const md = M0(); const idx = Array.from({ length: V }, (_, i) => i).sort((a, b) => p[b] - p[a]);
        let cover = 0; const parts = idx.slice(0, 3).map(k => { cover += p[k]; return `${p[k].toFixed(2).replace(/^0/, '')}·W[${disp(k).replace(/</g, '&lt;')}]`; });
        const nEff = Math.exp(tr.H[t]);
        let sumU = 0; for (let v = 0; v < V; v++) sumU += u[v];
        pipe.innerHTML = `<span>e = ${parts.join(' + ')} + … <span style="color:var(--muted)">(${(100 * (1 - cover)).toFixed(0)}% rest)</span></span><span>effective symbols e<sup>H</sup> <b>${nEff.toFixed(1)}</b> · Σ<sub>v</sub> Δŷ<sub>v</sub> = <b>${Math.abs(sumU) < 5e-7 ? '0.000' : sumU.toFixed(3)}</b></span>`;
      }

      // ---------------- drawing: side plots ----------------
      function niceTicks(lo, hi, n) { const span = hi - lo; if (!(span > 0)) return [lo]; const mag = Math.pow(10, Math.floor(Math.log10(span / n))); const st = [1, 2, 2.5, 5, 10].map(s => s * mag).find(s => span / s <= n) || 10 * mag; const out = []; for (let v = Math.ceil(lo / st) * st; v <= hi + 1e-9; v += st) out.push(+v.toFixed(6)); return out; }
      function drawSide() {
        const c = side.ctx; side.clear(); if (!tr) return;
        const t = Math.min(S.t, S.N), ti = trueIdx(), N = S.N, x = 40, w = sideW - 52;
        const panels = [
          { name: 'energy E', vals: tr.E, col: C.blue, fmt: (v) => v.toFixed(1) },
          { name: 'entropy H (nats)', vals: tr.H, col: C.ink, fmt: (v) => v.toFixed(0), lo: 0, hi: Math.log(V) },
          ti >= 0 ? { name: `p('${disp(ti)}')`, vals: tr.P.map(pp => pp[ti]), col: C.ink, dash: [5, 3], fmt: (v) => v.toFixed(1), lo: 0, hi: 1 } : null,
        ].filter(Boolean);
        const ph = (SH - 30) / panels.length;
        panels.forEach((pn, k) => {
          const top = 4 + k * ph, hh = ph - 30;
          let lo = pn.lo != null ? pn.lo : Math.min(...pn.vals), hi = pn.hi != null ? pn.hi : Math.max(...pn.vals);
          if (pn.lo == null) { const pd = (hi - lo) * 0.1 || 0.1; lo -= pd; hi += pd; }
          const yt = pn.lo != null ? [pn.lo, pn.hi] : [Math.min(...pn.vals), Math.max(...pn.vals)];
          lib.text(c, pn.name, x, top, { size: 11, kind: 'mono', color: C.muted });
          const ax = lib.axes(c, { x, y: top + 16, w, h: hh, xlim: [0, N], ylim: [lo, hi], xticks: k === panels.length - 1 ? niceTicks(0, N, 4) : [], yticks: yt, yfmt: pn.fmt, size: 11 });
          c.save(); c.strokeStyle = C.blue3; c.lineWidth = 1; c.setLineDash([3, 3]); c.beginPath(); c.moveTo(ax.X(t), top + 16); c.lineTo(ax.X(t), top + 16 + hh); c.stroke(); c.restore();
          lib.plot(c, ax, pn.vals.slice(0, t + 1).map((v, i) => [i, v]), { color: pn.col, width: 1.8, dash: pn.dash, markers: N <= 16 ? 2.2 : 0 });
          lib.dot(c, ax.X(t), ax.Y(pn.vals[t]), 3.5, pn.col);
        });
        lib.text(c, 'step i', x + w, SH - 13, { size: 11, kind: 'mono', color: C.muted, align: 'right' });
      }
      function readout() {
        if (!tr) { ro.innerHTML = ''; return; }
        const t = Math.min(S.t, S.N), p = tr.P[t], ti = trueIdx(); let top = 0; for (let v = 1; v < V; v++) if (p[v] > p[top]) top = v;
        const ex = S.ex !== 'custom' ? EX[S.ex] : null;
        ro.innerHTML = `<span>step <b>${t}</b>/${S.N} · E <b>${fx(tr.E[t])}</b></span><span>top '${disp(top).replace(/</g, '&lt;')}' <b>${fx(p[top])}</b>${ti >= 0 ? ` · p('${disp(ti)}') <b>${fx(p[ti])}</b>` : ''}</span>` +
          (ti >= 0 ? `<span>${ex ? `one-pass <b>${fx(ex.baseline_p_true)}</b> · ` : ''}${robust ? `<b>${robust.hit}</b>/32 right` : ''}</span>` : '<span>typed text: no true next character</span>');
      }

      // ---------------- bottom: sentence (live) ----------------
      const sentCache = {};
      function groupOf(full, j) {
        const isL = (ch) => ch >= 'a' && ch <= 'z', ch = full[j], prev = full[j - 1] || ' ';
        if (ch === 'u' && prev === 'q') return 'easy';
        if (!isL(ch)) return 'other';
        let k = 0; while (j - 1 - k >= 0 && isL(full[j - 1 - k])) k++;
        return k === 0 ? 'hard' : k >= 2 ? 'easy' : 'other';
      }
      function sentJob(si) {
        if (sentCache[si]) return sentCache[si];
        const md = M0(), s = SENTS[si], pre = Array.from(md.norm(s.prefix)).slice(-40).join('').padStart(40, ' '), txt = md.norm(s.text), full = pre + txt;
        return (sentCache[si] = { si, txt, full, n: txt.length, j: 0, E: [], Eu: [], pt: [], grp: [], done: false });
      }
      const Y0S = y0For(777);
      function sentWork(ms) {
        if (S.tab !== 'chars') return false; const J = sentJob(S.sent); if (J.done) return false;
        const md = M0(), al = md.alpha0, t0 = performance.now(), zero = new Float64Array(V);
        while (J.j < J.n && performance.now() - t0 < ms) {
          const j = J.j, cx = md.context(J.full.slice(j, j + 40)), ti = md.stoi.get(J.txt[j]);
          const r = think(J.full.slice(j, j + 40), Y0S, 8, al);
          J.E.push(r.E); J.Eu.push(md.evalE(cx, zero)); J.pt.push(r.P[8][ti]); J.grp.push(groupOf(J.full, 40 + j)); J.j++;
        }
        if (J.j >= J.n) J.done = true;
        return true;
      }
      let cellGeo = null;
      function drawChars() {
        const J = sentJob(S.sent), md = M0();
        const cw = wide ? 15.5 : 16, lineH = 26, perLine = Math.floor((BWB - 12) / cw), lines = Math.ceil(J.n / perLine);
        const textH = lines * lineH + 4, plotH = 78, Hh = textH + plotH + 8 + (wide ? 0 : 22);
        bot.size(BWB, Hh); bot.clear(); const c = bot.ctx;
        const val = (j, i) => S.emode === 'rel' ? J.E[j][i] - J.Eu[j] : J.E[j][i];
        let lo = Infinity, hi = -Infinity; for (let j = 0; j < J.j; j++) for (let i = 0; i <= 8; i++) { const v = val(j, i); if (v < lo) lo = v; if (v > hi) hi = v; }
        // colour = energy after the last step (8), robust range over this sentence
        const vs = []; for (let j = 0; j < J.j; j++) vs.push(val(j, 8)); vs.sort((a, b) => a - b);
        const rlo = vs.length ? vs[Math.floor((vs.length - 1) * 0.03)] : 0, rhi = vs.length ? vs[Math.floor((vs.length - 1) * 0.97)] : 1;
        cellGeo = { cw, lineH, perLine, x0: 6, y0: 4 };
        for (let j = 0; j < J.n; j++) {
          const li = Math.floor(j / perLine), x = 6 + (j % perLine) * cw, y = 4 + li * lineH, ch = J.txt[j];
          if (j < J.j) {
            const v = Math.max(0, Math.min(1, (val(j, S.st) - rlo) / (rhi - rlo || 1)));
            const col = lib.cmap(v); c.fillStyle = `rgb(${col[0]},${col[1]},${col[2]})`; c.fillRect(x, y, cw - 1, 21);
            const dark = v < 0.55;
            lib.text(c, ch === ' ' ? '·' : ch, x + (cw - 1) / 2, y + 10.5, { size: 14, kind: 'mono', color: dark ? '#fff' : C.ink, align: 'center', baseline: 'middle' });
            if (J.grp[j] === 'hard') { c.strokeStyle = C.ink; c.lineWidth = 1.4; c.strokeRect(x + 0.5, y + 0.5, cw - 2, 20); }
            if (j === S.hover) { c.strokeStyle = C.blue; c.lineWidth = 2; c.strokeRect(x - 0.5, y - 0.5, cw, 23); }
          } else {
            c.fillStyle = C.tint || '#f7f7f9'; c.fillRect(x, y, cw - 1, 21);
            lib.text(c, ch === ' ' ? '·' : ch, x + (cw - 1) / 2, y + 11, { size: 14, kind: 'mono', color: C.faint, align: 'center', baseline: 'middle' });
          }
        }
        // group means vs step
        const G = { easy: [], hard: [], other: [] }; for (let j = 0; j < J.j; j++) G[J.grp[j]].push(j);
        const mean = (arr, i) => arr.reduce((s, j) => s + val(j, i), 0) / Math.max(1, arr.length);
        const py = textH + 22, pw = wide ? BWB - 230 : BWB - 60, px = 46;
        const gs = [['hard', C.blue, null, 2.2], ['easy', C.ink, [6, 4], 1.6], ['other', C.muted, [1.5, 3], 1.4]];
        let glo = Infinity, ghi = -Infinity; gs.forEach(([g]) => { if (!G[g].length) return; for (let i = 2; i <= 8; i++) { const m = mean(G[g], i); glo = Math.min(glo, m); ghi = Math.max(ghi, m); } });
        if (!isFinite(glo)) { glo = 0; ghi = 1; } const pd = (ghi - glo) * 0.15 || 0.1; glo -= pd; ghi += pd;
        lib.text(c, (S.emode === 'rel' ? 'mean E − E(x, uniform)' : 'mean raw E') + ' by group, steps 2–8', px - 40, py - 17, { size: 11, kind: 'mono', color: C.muted });
        const ax = lib.axes(c, { x: px, y: py, w: pw, h: plotH - 42, xlim: [2, 8], ylim: [glo, ghi], xticks: [2, 4, 6, 8], yticks: niceTicks(glo, ghi, 3), yfmt: (v) => v.toFixed(1), size: 11 });
        gs.forEach(([g, col, dash, wd]) => { if (G[g].length) lib.plot(c, ax, [2, 3, 4, 5, 6, 7, 8].map(i => [i, mean(G[g], i)]), { color: col, width: wd, dash, markers: 2 }); });
        const lx = wide ? px + pw + 22 : px - 40, ly = wide ? py - 4 : py + plotH - 18;
        gs.forEach(([g, col, dash, wd], k) => {
          const xx = wide ? lx : lx + k * 100, yy = wide ? ly + k * 18 : ly;
          c.save(); c.strokeStyle = col; c.lineWidth = wd; if (dash) c.setLineDash(dash); c.beginPath(); c.moveTo(xx, yy + 7); c.lineTo(xx + 18, yy + 7); c.stroke(); c.restore();
          lib.text(c, `${g} (${G[g].length})`, xx + 24, yy, { size: 11, kind: 'mono', color: C.ink });
        });
        if (wide) lib.text(c, 'boxed = word start', lx, ly + 56, { size: 11, kind: 'mono', color: C.muted });
        // readout
        const fin = (arr) => arr.length ? mean(arr, 8).toFixed(2) : '–';
        let hov = '';
        if (S.hover >= 0 && S.hover < J.j) { const j = S.hover; hov = `<span>'${J.txt[j] === ' ' ? '␣' : J.txt[j]}' ${J.grp[j]} · E at 0,2,4,6,8: ${J.E[j].filter((_, i) => i % 2 === 0).map(v => v.toFixed(2)).join(' ')} · p(true) <b>${fx(J.pt[j])}</b></span>`; }
        bro.innerHTML = `<span>${J.done ? '' : `<span style="color:var(--blue)">computing ${J.j}/${J.n}…</span> `}after 8 steps: hard <b>${fin(G.hard)}</b> · easy <b>${fin(G.easy)}</b> · other <b>${fin(G.other)}</b></span>` +
          (hov || '<span style="color:var(--muted)">hover a character for its energies</span>');
      }
      bot.canvas.addEventListener('mousemove', (ev) => {
        if (S.tab !== 'chars' || !cellGeo) return; const [px, py] = bot.toLocal(ev), g = cellGeo;
        const li = Math.floor((py - g.y0) / g.lineH), ci = Math.floor((px - g.x0) / g.cw), j = li * g.perLine + ci;
        const ok = ci >= 0 && ci < g.perLine && li >= 0 && (py - g.y0 - li * g.lineH) < 22 && j < sentJob(S.sent).n;
        const nh = ok ? j : -1; if (nh !== S.hover) { S.hover = nh; drawChars(); }
      });
      bot.canvas.addEventListener('click', (ev) => { bot.canvas.dispatchEvent(new MouseEvent('mousemove', { clientX: ev.clientX, clientY: ev.clientY })); });
      bot.canvas.addEventListener('mouseleave', () => { if (S.tab === 'chars' && S.hover !== -1) { S.hover = -1; drawChars(); } });

      // ---------------- bottom: paper Fig 8 (digitized) ----------------
      function drawFig8() {
        const hm = HM.filter(x => /^fig8/.test(x.id));
        if (!hm.length) { bot.size(BWB, 60); bot.clear(); lib.text(bot.ctx, 'paper_heatmaps.json not bundled', 10, 20, { size: 12, kind: 'mono', color: C.bad }); return; }
        const showAll = wide, list = showAll ? hm : [hm[S.f8] || hm[0]];
        const colW = showAll ? (BWB - 12) / list.length : BWB - 6, labW = 92, nIt = 12;
        const cellW = Math.min(17, (colW - labW - 12) / nIt), rowH = 14;
        const maxRows = Math.max(...list.map(x => x.rows.length)), Hh = 26 + maxRows * rowH + 34;
        bot.size(BWB, Hh); bot.clear(); const c = bot.ctx;
        list.forEach((m, k) => {
          const ox = 6 + k * colW, oy = 22;
          lib.text(c, `${m.figure} · approx., digitized`, ox, 2, { size: 11, kind: 'mono', color: C.muted });
          m.rows.forEach((tok, r) => {
            lib.text(c, tok, ox + labW - 6, oy + r * rowH + rowH / 2, { size: 11.5, kind: 'mono', color: C.ink, align: 'right', baseline: 'middle' });
            m.energy[r].forEach((v, i) => { const col = lib.cmap(v); c.fillStyle = `rgb(${col[0]},${col[1]},${col[2]})`; c.fillRect(ox + labW + i * cellW, oy + r * rowH, cellW - 0.5, rowH - 0.5); });
          });
          const yb = oy + m.rows.length * rowH + 3;
          [0, 5, 11].forEach(i => lib.text(c, String(i), ox + labW + (i + 0.5) * cellW, yb, { size: 10.5, kind: 'mono', color: C.muted, align: 'center' }));
          lib.text(c, 'iteration', ox + labW + nIt * cellW, yb + 13, { size: 10.5, kind: 'mono', color: C.muted, align: 'right' });
        });
        // colour key
        const kx = 8, ky = Hh - 14; for (let i = 0; i < 60; i++) { const col = lib.cmap(i / 59); c.fillStyle = `rgb(${col[0]},${col[1]},${col[2]})`; c.fillRect(kx + 64 + i * 1.5, ky, 1.6, 8); }
        lib.text(c, 'low 0', kx, ky - 3, { size: 10.5, kind: 'mono', color: C.muted }); lib.text(c, '1 high', kx + 64 + 96, ky - 3, { size: 10.5, kind: 'mono', color: C.muted });
        bro.innerHTML = `<span>"Normalized Energy" 0–1 re-coloured: darker = lower (paper: yellow = low, purple = high; normalization not stated). Iteration 0 = random guess. <a href="media/paper/fig08.png" target="_blank" rel="noopener">original ↗</a></span>`;
      }

      function drawBottom() { if (S.tab === 'chars') drawChars(); else drawFig8(); }
      function syncTab() {
        segTab.set(S.tab); subRow.innerHTML = '';
        if (S.tab === 'chars') { subRow.appendChild(segSent.el); subRow.appendChild(segMode.el); segSent.set(S.sent); segMode.set(S.emode); }
        if (S.tab === 'fig8') { if (!wide) subRow.appendChild(segF8.el); else subRow.appendChild(h('span', { class: 'fig-sub' }, 'Fig 8 · token energies across thinking steps')); }
        drawBottom(); kick();
      }

      // ---------------- loop ----------------
      let acc = 0;
      const loop = lib.loop((dt) => {
        if (!ctx.visible()) return false;
        let busy = false;
        if (S.playing && tr) {
          acc += dt; const rate = 2.6;
          while (acc > 1 / rate) { acc -= 1 / rate; S.t++; }
          if (S.t >= S.N) { S.t = S.N; S.playing = false; }
          syncPlay(); draw(); busy = true;
        }
        if (sentWork(8)) { drawBottom(); busy = true; }
        return busy || S.playing;
      });
      function kick() { draw(); if (ctx.visible()) loop.start(); }
      function draw() { drawBars(); drawSide(); readout(); }

      const STEPS = [{ t: 0 }, { t: 0 }, { t: 0 }, { play: true }, { tab: 'fig8' }];
      let booted = false;
      function boot() { if (booted || !M0()) return; booted = true; recompute(false); }
      return {
        step(i) {
          const o = STEPS[i] || STEPS[0];
          if (o.tab) S.tab = o.tab;
          segTab.set(S.tab);
          if (!ctx.visible()) { S.pendingStep = i; return; }
          applyStep(i);
        },
        show() { boot(); if (S.pendingStep != null) { applyStep(S.pendingStep); S.pendingStep = null; } syncTab(); kick(); },
        hide() { loop.stop(); },
      };
      function applyStep(i) {
        boot(); const o = STEPS[i] || STEPS[0];
        if (i <= 3 && S.ex === 'custom') { S.ex = 0; sel.value = '0'; inpWrap.hidden = true; recompute(false); }
        S.focus = ['bars', 'embed', 'grad'][i] || null; pipe.classList.toggle('on', S.focus === 'embed'); FT.frame.classList.toggle('tk-hl', i === 4);
        if (o.t != null) { S.t = o.t; S.playing = false; syncPlay(); }
        if (o.play) { S.t = 0; S.playing = !reduced; if (reduced) S.t = S.N; syncPlay(); }
        syncTab(); kick();
      }
    },
  });
})();
