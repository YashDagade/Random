/* Chapter: Text thinking lab.
   A toy character-level EBT (data/text.json, trained for this explainer) runs live in the browser:
   its weights are decoded from base64, and energies and exact gradients dE/dŷ are computed here in JS.
   Labs: (1) Algorithm 2 live on one next character, (2) a 2D slice of the 54-D energy landscape,
   (3) easy vs hard characters next to paper Fig 8, (4) toy thinking/BoN curves next to digitized Figs 6a, 6b, 7.
   Every number shown is read from the data files at runtime; nothing about the toy is hard-coded. */
(function () {
  'use strict';

  EBT.section({
    id: 'thinking',
    nav: 'Text thinking',
    kicker: 'Lab · language',
    title: 'Text thinking lab: watch an EBT think about the next character',
    lede: 'A tiny character-level EBT, trained for this explainer on RedPajama-V2 web text, runs live in your browser. Its guess for the next character is a whole probability distribution that starts as noise and is pushed downhill on a learned energy. Step through that descent, compare it with a model that answers in one pass, and then hold the toy up against what the paper measured at scale.',
    mount,
  });

  // ---------------------------------------------------------------------------------------------
  // Model: exact port of the toy EBT described in text.json "arch".
  //   h = Linear(SiLU(Linear(concat 40 char embeddings)))            context encoding (128)
  //   p = softmax(ŷ); e = p·Wpe; z = [h, e, h⊙e]; E = MLP(z)           scalar energy
  // Because h is fixed while thinking, z·W1 = c1 + p·Q with Q = Wpe·(W1e + diag(h)·W1he), precomputed per context.
  // ---------------------------------------------------------------------------------------------
  function makeModel(D, lib) {
    const Wt = D && D.weights;
    if (!Wt || !Wt.data || !Wt.tensors || !Wt.order) return null;
    let all;
    try {
      const bin = atob(Wt.data); const u8 = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
      all = new Float32Array(u8.buffer);
    } catch (e) { console.warn('thinking: weights decode failed', e); return null; }
    const T = {}, S = {};
    for (const k of Wt.order) { const t = Wt.tensors[k]; const n = t.shape.reduce((a, b) => a * b, 1); T[k] = all.subarray(t.offset, t.offset + n); S[k] = t.shape; }
    const V = S.Wpe[0], Dh = S.Wpe[1], dc = S.emb[1], nIn = S.We1[0], E1 = S.We1[1], H1 = S.W1[1], H2 = S.W2[1], Cn = nIn / dc;
    if (S.W1[0] !== 3 * Dh || S.We2[1] !== Dh) { console.warn('thinking: unexpected weight shapes'); return null; }
    const sig = (z) => 1 / (1 + Math.exp(-z));
    const mvAdd = (out, x, W, nI, nO) => { for (let i = 0; i < nI; i++) { const xi = x[i]; if (xi === 0) continue; const o = i * nO; for (let j = 0; j < nO; j++) out[j] += xi * W[o + j]; } };
    function encode(ids) {
      const x = new Float64Array(nIn);
      for (let t = 0; t < Cn; t++) { const id = ids[t]; for (let d = 0; d < dc; d++) x[t * dc + d] = T.emb[id * dc + d]; }
      const a = Float64Array.from(T.be1); mvAdd(a, x, T.We1, nIn, E1);
      for (let j = 0; j < E1; j++) a[j] = a[j] * sig(a[j]);
      const h = Float64Array.from(T.be2); mvAdd(h, a, T.We2, E1, Dh); return h;
    }
    function prep(h) {
      const c1 = Float64Array.from(T.b1); mvAdd(c1, h, T.W1, Dh, H1);
      const Wp = new Float64Array(Dh * H1);
      for (let d = 0; d < Dh; d++) { const o1 = (Dh + d) * H1, o2 = (2 * Dh + d) * H1, hd = h[d]; for (let j = 0; j < H1; j++) Wp[d * H1 + j] = T.W1[o1 + j] + hd * T.W1[o2 + j]; }
      const Q = new Float64Array(V * H1);
      for (let k = 0; k < V; k++) { const qo = k * H1; for (let d = 0; d < Dh; d++) { const w = T.Wpe[k * Dh + d]; const o = d * H1; for (let j = 0; j < H1; j++) Q[qo + j] += w * Wp[o + j]; } }
      return { h, c1, Q };
    }
    const a1 = new Float64Array(H1), s1 = new Float64Array(H1), a2 = new Float64Array(H2), da1 = new Float64Array(H1), da2 = new Float64Array(H2);
    // energy at logits y; if g is given, also fills g = dE/dy (exact backprop). p (optional) receives softmax(y).
    function evalE(cx, y, g, p) {
      p = p || new Float64Array(V);
      let m = -Infinity; for (let k = 0; k < V; k++) if (y[k] > m) m = y[k];
      let Z = 0; for (let k = 0; k < V; k++) { p[k] = Math.exp(y[k] - m); Z += p[k]; } for (let k = 0; k < V; k++) p[k] /= Z;
      a1.set(cx.c1); const Q = cx.Q;
      for (let k = 0; k < V; k++) { const pk = p[k], o = k * H1; for (let j = 0; j < H1; j++) a1[j] += pk * Q[o + j]; }
      for (let j = 0; j < H1; j++) s1[j] = a1[j] * sig(a1[j]);
      a2.set(T.b2); mvAdd(a2, s1, T.W2, H1, H2);
      let E = T.b3[0]; for (let j = 0; j < H2; j++) E += a2[j] * sig(a2[j]) * T.W3[j];
      if (!g) return E;
      for (let j = 0; j < H2; j++) { const s = sig(a2[j]); da2[j] = T.W3[j] * s * (1 + a2[j] * (1 - s)); }
      for (let i = 0; i < H1; i++) { let acc = 0; const o = i * H2; for (let j = 0; j < H2; j++) acc += T.W2[o + j] * da2[j]; const s = sig(a1[i]); da1[i] = acc * s * (1 + a1[i] * (1 - s)); }
      let sp = 0; for (let k = 0; k < V; k++) { let acc = 0; const o = k * H1; for (let j = 0; j < H1; j++) acc += Q[o + j] * da1[j]; g[k] = acc; sp += p[k] * acc; }
      for (let k = 0; k < V; k++) g[k] = p[k] * (g[k] - sp);
      return E;
    }
    // Algorithm 2 inner loop for one candidate. Langevin noise (if any) is added on every step except the last, as in training.
    function think(cx, y0, o) {
      const N = o.steps, al = o.alpha, sg = o.sigma || 0, r = lib.rng(o.seed || 1);
      let y = Float64Array.from(y0); const Y = [], P = [], E = [], G = [];
      for (let i = 0; i <= N; i++) {
        const g = new Float64Array(V), p = new Float64Array(V); const e = evalE(cx, y, g, p);
        Y.push(y); P.push(p); E.push(e); G.push(g);
        if (i === N) break;
        const yn = new Float64Array(V);
        for (let k = 0; k < V; k++) yn[k] = y[k] - al * g[k] + (sg > 0 && i < N - 1 ? sg * r.normal() : 0);
        y = yn;
      }
      return { Y, P, E, G };
    }
    return { V, Cn, Dh, encode, prep, evalE, think, nParams: Wt.n_floats };
  }

  // ---------------------------------------------------------------------------------------------
  function mount(root, lib) {
    const h = lib.h, C = lib.C;
    const D = lib.data('text');
    if (!D) { root.appendChild(lib.callout('warn', 'Data missing', 'data/text.json is not in data/bundle.js. Run <code>python3 src/bundle_data.py</code>.')); return; }
    const model = makeModel(D, lib);
    const V = D.vocab.length, VD = D.vocab_display || D.vocab, vidx = new Map(D.vocab.map((c, i) => [c, i]));
    const HP = D.hparams || {}, alpha0 = HP.alpha0 || 10, CL = (D.arch && D.arch.context_len) || 40;
    const REF = D.weights_reference_trace && D.weights_reference_trace.yhat0 && D.weights_reference_trace.yhat0.length === V ? D.weights_reference_trace.yhat0 : null;
    const reduced = !!lib.reducedMotion;
    const fx = (v, d = 2) => lib.fmt(v, d);
    const pct = (v, d = 1) => (v == null || !isFinite(v)) ? '–' : v.toFixed(d) + '%';
    const disp = (ch) => VD[vidx.get(ch)] != null ? VD[vidx.get(ch)] : ch;
    const lnV = Math.log(V);
    const TC = D.thinking_curve || null, N3 = (TC && TC.datasets && TC.datasets.val && TC.datasets.val.train_like_N) || 3;

    // ---------- shared drawing kit ----------
    const canvases = [];
    function rcanvas(parent, o) {
      const box = h('div', { class: 'thinking-cbox' }); parent.appendChild(box);
      const cv = document.createElement('canvas'); box.appendChild(cv);
      cv.setAttribute('role', 'img'); cv.setAttribute('aria-label', o.label || '');
      const st = { box, cv, ctx: cv.getContext('2d'), w: 0, h: 0 };
      st.redraw = () => {
        const w = Math.max(220, Math.floor(box.clientWidth || 600)); const hh = Math.round(o.height(w));
        const dpr = Math.min(2, window.devicePixelRatio || 1);
        if (w !== st.w || hh !== st.h) { cv.width = Math.round(w * dpr); cv.height = Math.round(hh * dpr); cv.style.height = hh + 'px'; st.w = w; st.h = hh; }
        st.ctx.setTransform(dpr, 0, 0, dpr, 0, 0); st.ctx.clearRect(0, 0, w, hh);
        try { o.draw(st.ctx, w, hh); } catch (e) { console.error(e); }
      };
      st.toLocal = (ev) => { const r = cv.getBoundingClientRect(); return [(ev.clientX - r.left) * st.w / r.width, (ev.clientY - r.top) * st.h / r.height]; };
      if (window.ResizeObserver) new ResizeObserver(() => st.redraw()).observe(box); else window.addEventListener('resize', st.redraw);
      canvases.push(st); Promise.resolve().then(st.redraw); return st;
    }
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => canvases.forEach(c => c.redraw()));
    const T = (ctx, s, x, y, o) => lib.text(ctx, s, x, y, Object.assign({ size: 12, color: C.muted }, o || {}));
    function screen(ctx, w, hh) { ctx.save(); ctx.fillStyle = C.bg; lib.rr(ctx, 0, 0, w, hh, 10); ctx.fill(); ctx.restore(); }
    function niceTicks(lo, hi, n = 4) {
      const span = hi - lo; if (!(span > 0)) return [lo];
      const mag = Math.pow(10, Math.floor(Math.log10(span / n)));
      const step = [1, 2, 2.5, 5, 10].map(s => s * mag).find(s => span / s <= n) || 10 * mag;
      const out = []; for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10)); return out;
    }
    const tickFmt = (ticks) => { const st = ticks.length > 1 ? Math.abs(ticks[1] - ticks[0]) : 1; const d = st >= 1 ? 0 : st >= 0.1 ? 1 : 2; return (v) => v.toFixed(d); };
    function ax(ctx, o) { const r = lib.axes(ctx, Object.assign({ size: 11, xfmt: tickFmt(o.xticks || []), yfmt: tickFmt(o.yticks || []) }, o)); return r; }
    function legendRow(items) { return h('div', { class: 'legend' }, items.map(([col, label, style]) => h('span', {}, h('i', { style: `background:${col};${style || ''}` }), label))); }
    function headRow(title, ...badges) { return h('div', { class: 'thinking-head' }, h('h4', {}, title), h('span', { class: 'thinking-badges' }, ...badges)); }
    const lum = (rgb) => (0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]) / 255;
    function shape(ctx, kind, x, y, r, col) {
      ctx.save(); ctx.fillStyle = col; ctx.beginPath();
      if (kind === 'star') { for (let i = 0; i < 10; i++) { const a = -Math.PI / 2 + i * Math.PI / 5, rr = i % 2 ? r * 0.45 : r * 1.15; ctx.lineTo(x + rr * Math.cos(a), y + rr * Math.sin(a)); } }
      else if (kind === 'tri') { ctx.moveTo(x, y - r * 1.1); ctx.lineTo(x + r, y + r * 0.75); ctx.lineTo(x - r, y + r * 0.75); }
      else if (kind === 'sq') { ctx.rect(x - r * 0.8, y - r * 0.8, r * 1.6, r * 1.6); }
      else ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.closePath(); ctx.fill(); ctx.restore();
    }
    const entropy = (p) => { let s = 0; for (let k = 0; k < p.length; k++) if (p[k] > 0) s -= p[k] * Math.log(p[k]); return s; };
    const argmax = (a) => { let b = 0; for (let k = 1; k < a.length; k++) if (a[k] > a[b]) b = k; return b; };
    const topK = (p, k) => Array.from(p, (v, i) => i).sort((a, b) => p[b] - p[a]).slice(0, k);
    const lerpArr = (a, b, f) => { if (f <= 1e-6 || !b) return a; const o = new Float64Array(a.length); for (let k = 0; k < a.length; k++) o[k] = a[k] + (b[k] - a[k]) * f; return o; };
    const pearson = (x, y) => { const n = x.length; const mx = x.reduce((a, b) => a + b, 0) / n, my = y.reduce((a, b) => a + b, 0) / n; let sxy = 0, sxx = 0, syy = 0; for (let i = 0; i < n; i++) { sxy += (x[i] - mx) * (y[i] - my); sxx += (x[i] - mx) ** 2; syy += (y[i] - my) ** 2; } return sxy / Math.sqrt(sxx * syy || 1); };

    // ---------- text helpers ----------
    function normalize(s) {
      s = String(s || '').toLowerCase()
        .replace(/[‘’‚‛′]/g, "'").replace(/[“”„″]/g, '"')
        .replace(/[‐-―−]/g, '-').replace(/…/g, '...').replace(/[\t\r]/g, ' ');
      let out = ''; for (const ch of s) { if (vidx.has(ch)) out += ch; else { const c = ch.charCodeAt(0); if (c < 128 && c >= 32) out += '#'; } }
      return out.replace(/ {2,}/g, ' ');
    }
    const padCtx = (s) => { s = s.slice(-CL); return ' '.repeat(Math.max(0, CL - s.length)) + s; };
    const ctxCache = new Map();
    function ctxFor(str) {
      if (ctxCache.has(str)) return ctxCache.get(str);
      const ids = Array.from(padCtx(str), ch => vidx.has(ch) ? vidx.get(ch) : vidx.get('#'));
      const cx = model.prep(model.encode(ids));
      if (ctxCache.size > 400) ctxCache.clear();
      ctxCache.set(str, cx); return cx;
    }
    function y0For(seed, j) {
      if (seed === 0 && j === 0 && REF) return Float64Array.from(REF);
      const r = lib.rng(9973 * (seed + 1) + 131 * j + 7); const y = new Float64Array(V); for (let k = 0; k < V; k++) y[k] = r.normal(); return y;
    }

    // ===============================================================================================
    // 0. Intro: what the toy is, and the honest scoreboard
    // ===============================================================================================
    const val = TC && TC.datasets && TC.datasets.val;
    (function intro() {
      const wrap = h('div', { class: 'grid2 thinking-intro' }); root.appendChild(wrap);
      const nE = (D.train_curves && D.train_curves.ebt && D.train_curves.ebt.step) || [], nB = (D.train_curves && D.train_curves.baseline && D.train_curves.baseline.step) || [];
      const wall = HP.wall_clock_s || {};
      const nParE = (D.arch && D.arch.n_params_ebt) || (model && model.nParams), nParB = D.arch && D.arch.n_params_baseline;
      const trN = (HP.train_steps_N || [2, 3]).join('–');
      const card = h('div', { class: 'card stack' },
        headRow('The toy model', lib.badge('toy')),
        h('ul', { class: 'thinking-facts' },
          h('li', { html: `<b>Context x</b>: the previous ${CL} characters of lowercased RedPajama-V2 web text, ${V} symbols in all.` }),
          h('li', { html: `<b>Prediction ŷ</b>: ${V} numbers (logits), one per symbol. It starts as random noise ŷ₀ ~ N(0, I).` }),
          h('li', { html: `<b>Energy E(x, ŷ)</b>: one number from a small MLP that sees the context code h and the guess, softmax(ŷ) projected to an embedding, as in the paper (p.43–44).` }),
          h('li', { html: `<b>Thinking</b>: ŷ ← ŷ − α∇<sub>ŷ</sub>E, α = ${alpha0}. Trained with ${trN} steps, random α, Langevin noise σ = ${HP.langevin_sigma} and a replay buffer, the paper's System 2 recipe (Sec 3.3).` }),
          h('li', { html: `<b>Size</b>: ${nParE ? (nParE / 1000).toFixed(0) + 'k' : '?'} parameters, ${nE.length ? (nE[nE.length - 1] + 1).toLocaleString() : '?'} updates, ${wall.ebt ? Math.round(wall.ebt) + ' s' : '?'} on one CPU thread. Baseline: the same context encoder plus an MLP that outputs the logits in one pass (${nParB ? (nParB / 1000).toFixed(0) + 'k' : '?'} parameters, ${nB.length ? (nB[nB.length - 1] + 1).toLocaleString() : '?'} updates, ${wall.baseline ? Math.round(wall.baseline) + ' s' : '?'}).` }),
        ),
        h('p', { class: 'thinking-note', html: model ? 'The weights ship with the page, so every bar and curve in labs 1 to 3 is the real model computed in your browser, with exact gradients.' : 'Weights are not bundled, so the labs below replay stored outputs of the model.' }));
      wrap.appendChild(card);

      // scoreboard: validation loss, nats per character
      const sc = h('div', { class: 'card stack' }, headRow('Scoreboard: validation loss', lib.badge('toy')));
      if (val) {
        const ce = val.ebt_fixed_alpha_ce, steps = TC.steps; let bi = 0; ce.forEach((v, i) => { if (v < ce[bi]) bi = i; });
        const rows = [['One-pass baseline', val.baseline_ce, C.ar, '1 forward pass'], [`EBT, ${N3} thinking steps (as trained)`, ce[steps.indexOf(N3)], C.ebt, `${N3} steps`], [`EBT, ${steps[bi]} thinking steps (its best)`, ce[bi], C.ebt, `${steps[bi]} steps`], ['EBT, no thinking (ŷ₀ = noise)', ce[0], C.faint, '0 steps']];
        const mx = Math.max(...rows.map(r => r[1]));
        const list = h('div', { class: 'thinking-score' });
        rows.forEach(([lab, v, col, sub]) => list.appendChild(h('div', { class: 'thinking-score-row' },
          h('span', { class: 'l' }, lab), h('span', { class: 'b' }, h('i', { style: `width:${(v / mx * 100).toFixed(1)}%;background:${col}` })), h('span', { class: 'v' }, v.toFixed(3)))));
        sc.appendChild(list);
        sc.appendChild(h('p', { class: 'thinking-note', html: `Nats per character, lower is better (perplexity per character = e<sup>loss</sup>: ${Math.exp(val.baseline_ce).toFixed(1)} vs ${Math.exp(ce[bi]).toFixed(1)}). Thinking longer helps the EBT: ${ce[steps.indexOf(N3)].toFixed(2)} → ${ce[bi].toFixed(2)}. But even at its best it stays well behind the same-size model that answers in one pass, which also cost ${wall.ebt && wall.baseline ? (wall.ebt / wall.baseline).toFixed(1) + '× less' : 'less'} training time. Treat this toy as a microscope on the mechanism, not as evidence for or against the paper.` }));
        sc.appendChild(lib.callout('insight', 'Same direction as the paper, much larger gap', 'The paper\'s EBT also had worse pretraining perplexity than its Transformer++ (33.43 vs 31.36, Table 3, p.12), yet did better on 3 of 4 downstream tasks. Its claims are about scaling trends at 6M to 800M parameters. Our per-character numbers cannot be compared with its per-token perplexities.'));
      } else sc.appendChild(h('p', {}, 'thinking_curve missing from text.json.'));
      wrap.appendChild(sc);
    })();

    // ===============================================================================================
    // 1. Algorithm 2, live
    // ===============================================================================================
    const bus = { fns: [], on(f) { this.fns.push(f); }, emit(v) { this.fns.forEach(f => f(v)); } };
    const S1 = { src: 'ex', ex: Math.min(1, (D.examples || []).length - 1), custom: '', N: 12, ak: 0, sigma: 0, M: 1, seed: 0, t: 2, show: -1, playing: false };
    let R1 = null; // current run
    const lab1 = h('div', { class: 'thinking-lab' }); root.appendChild(lab1);
    lab1.appendChild(h('div', { class: 'thinking-labhead' },
      h('h3', {}, '1 · One prediction, step by step'),
      h('p', {}, 'Pick a context. The EBT starts from a random guess and takes gradient steps that lower the energy. Each step reshapes the whole distribution over the 54 symbols. Add candidates to see self-verification: the guess with the lowest final energy wins, with no outside judge.')));
    const bench1 = h('div', { class: 'bench' }); lab1.appendChild(bench1);
    const ctl1 = h('div', { class: 'card stack thinking-ctl' }); bench1.appendChild(ctl1);
    const ins1 = h('div', { class: 'card stack thinking-ins' }); bench1.appendChild(ins1);

    // controls
    ctl1.appendChild(h('h4', {}, 'Context'));
    const exList = h('div', { class: 'thinking-exlist', role: 'group', 'aria-label': 'Example contexts' }); ctl1.appendChild(exList);
    const exBtns = (D.examples || []).map((ex, i) => {
      const b = h('button', { type: 'button', class: 'thinking-ex', 'aria-pressed': 'false' }, h('span', { class: 'lab' }, ex.label), h('span', { class: 'tag ' + ex.kind }, ex.kind));
      b.addEventListener('click', () => { S1.src = 'ex'; S1.ex = i; S1.show = -1; recompute1(true); });
      exList.appendChild(b); return b;
    });
    const typed = h('textarea', { id: 'thinking-typed', rows: 2, spellcheck: 'false', placeholder: 'Or type your own text; the model predicts what comes after it', 'aria-label': 'Your own context text' });
    const typedWrap = h('div', { class: 'ctl' }, h('label', { for: 'thinking-typed' }, h('span', {}, 'Your text (last ' + CL + ' characters are used)')), typed);
    if (model) {
      ctl1.appendChild(typedWrap);
      let tmo = null;
      typed.addEventListener('input', () => { clearTimeout(tmo); tmo = setTimeout(() => { if (!typed.value.trim()) { S1.src = 'ex'; } else { S1.src = 'typed'; S1.custom = typed.value; } S1.show = -1; recompute1(false); }, 180); });
    }
    ctl1.appendChild(h('h4', {}, 'Thinking'));
    const slN = lib.slider({ id: 'thinking-N', label: 'Steps N', min: 1, max: 24, step: 1, value: S1.N, oninput: v => { S1.N = v; recompute1(false, true); } });
    const slA = lib.slider({ id: 'thinking-alpha', label: 'Step size α', min: -2, max: 2, step: 0.5, value: 0, fmt: v => (alpha0 * Math.pow(2, v)).toFixed(alpha0 * Math.pow(2, v) < 10 ? 1 : 0) + (v === 0 ? ' (trained)' : ''), oninput: v => { S1.ak = v; recompute1(false, true); } });
    const slS = lib.slider({ id: 'thinking-sigma', label: 'Langevin noise σ', min: 0, max: 1, step: 0.05, value: 0, fmt: v => v.toFixed(2), oninput: v => { S1.sigma = v; recompute1(false, true); } });
    const slM = lib.slider({ id: 'thinking-M', label: 'Candidates M (self-verification)', min: 1, max: 8, step: 1, value: 1, oninput: v => { S1.M = v; S1.show = -1; recompute1(false, true); } });
    [slN, slA, slS, slM].forEach(s => ctl1.appendChild(s.el));
    if (!model) [slN, slA, slS, slM].forEach(s => { s.input.disabled = true; });
    ctl1.appendChild(h('h4', {}, 'Playback'));
    const slT = lib.slider({ id: 'thinking-t', label: 'Show step i', min: 0, max: S1.N, step: 1, value: S1.t, oninput: v => { S1.t = v; S1.playing = false; syncPlay(); kick1(); } });
    ctl1.appendChild(slT.el);
    const bStep = h('button', { type: 'button', class: 'btn' }, 'Step ▸');
    const bPlay = h('button', { type: 'button', class: 'btn primary' }, 'Play');
    const bReset = h('button', { type: 'button', class: 'btn' }, 'Reset');
    const bNoise = h('button', { type: 'button', class: 'btn', title: 'Draw new random starting guesses ŷ₀' }, 'New ŷ₀');
    ctl1.appendChild(h('div', { class: 'row' }, bStep, bPlay, bReset, model ? bNoise : null));
    bStep.addEventListener('click', () => { const N = nSteps(); S1.playing = false; S1.t = Math.min(N, Math.floor(S1.t + 1e-6) + 1); syncPlay(); kick1(); });
    bPlay.addEventListener('click', () => { const N = nSteps(); if (S1.playing) S1.playing = false; else { if (S1.t >= N - 1e-6) S1.t = 0; S1.playing = true; } syncPlay(); kick1(); });
    bReset.addEventListener('click', () => { S1.playing = false; S1.t = 0; syncPlay(); kick1(); });
    bNoise.addEventListener('click', () => { S1.seed += 1; S1.show = -1; recompute1(true); });
    ctl1.appendChild(h('p', { class: 'thinking-note', html: 'Keys: focus a slider and use the arrow keys. α is the gradient step size, σ adds random kicks during the descent, M runs that many independent descents and keeps the lowest-energy result (Algorithm 2).' }));

    // instrument
    ins1.appendChild(headRow('Algorithm 2 on one character', lib.badge('toy', model ? 'live in your browser' : 'stored outputs')));
    const ctxLine = h('div', { class: 'thinking-ctxline', 'aria-live': 'polite' }); ins1.appendChild(ctxLine);
    const algo = h('pre', { class: 'thinking-algo', 'aria-label': 'Algorithm 2 pseudocode with the current line highlighted' }); ins1.appendChild(algo);
    const ALGO = ['for j = 1 … M:', '  ŷ₀ ~ N(0, I)              random 54 logits', '  for i = 0 … N−1:', '    ŷᵢ₊₁ = ŷᵢ − α∇E(x, ŷᵢ)     one thinking step', 'return argminⱼ E(x, ŷ_N)       self-verification'];
    const algoLines = ALGO.map(s => { const sp = h('span', {}, s + '\n'); algo.appendChild(sp); return sp; });
    const bars = rcanvas(ins1, { label: 'Top 8 next-character probabilities at the current thinking step', height: () => 300, draw: drawBars });
    const traj = rcanvas(ins1, { label: 'Energy and probability of the true character across thinking steps', height: (w) => w >= 620 ? 230 : 440, draw: drawTraj });
    const strip = rcanvas(ins1, { label: 'The 54 logits of the guess and the next gradient update', height: (w) => w >= 620 ? 170 : 190, draw: drawStrip });
    const ro1 = h('div', { class: 'readout', 'aria-live': 'polite' }); ins1.appendChild(ro1);
    const candBox = h('div', { class: 'tbl thinking-cands', hidden: true }); ins1.appendChild(candBox);
    const note1 = h('div', { class: 'callout insight thinking-dyn' }); ins1.appendChild(note1);

    // continuation writer (live model only)
    const genBox = h('div', { class: 'thinking-gen stack' });
    if (model) {
      ins1.appendChild(genBox);
      const genMode = lib.segmented({ label: 'Decoding', options: [['greedy', 'Greedy'], ['sample', 'Sample']], value: 'greedy' });
      const bGen = h('button', { type: 'button', class: 'btn' }, 'Write 60 more characters');
      const genOut = h('div', { class: 'thinking-genout', 'aria-live': 'polite' }, h('span', { class: 'thinking-note' }, 'Let the EBT keep writing with the settings above: for every new character it thinks N steps (and verifies M candidates), then appends its choice. Background = final energy of that choice.'));
      genBox.appendChild(h('div', { class: 'row' }, bGen, genMode.el));
      genBox.appendChild(genOut);
      let genJob = 0;
      bGen.addEventListener('click', () => {
        const job = ++genJob; const start = S1.src === 'ex' ? (D.examples[S1.ex] || {}).context || '' : normalize(S1.custom);
        let text = start; const outs = []; const r = lib.rng(4242 + S1.seed); const mode = genMode.value();
        genOut.innerHTML = ''; const pre = h('span', { class: 'ctx' }, start.slice(-CL)); genOut.appendChild(pre);
        const holder = h('span', {}); genOut.appendChild(holder);
        const al = alpha0 * Math.pow(2, S1.ak);
        const stepOne = () => {
          if (job !== genJob) return;
          for (let rep = 0; rep < 4 && outs.length < 60; rep++) {
            const cx = ctxFor(padCtx(text));
            let best = null;
            for (let j = 0; j < S1.M; j++) { const res = model.think(cx, y0For(S1.seed, j), { steps: S1.N, alpha: al, sigma: S1.sigma, seed: 17 + j + 31 * outs.length }); const e = res.E[S1.N]; if (!best || e < best.e) best = { e, p: res.P[S1.N] }; }
            let k; if (mode === 'greedy') k = argmax(best.p); else { let u = r(), acc = 0; k = V - 1; for (let q = 0; q < V; q++) { acc += best.p[q]; if (u <= acc) { k = q; break; } } }
            text += D.vocab[k]; outs.push({ ch: D.vocab[k], e: best.e });
          }
          const es = outs.map(o => o.e), lo = Math.min(...es), hi = Math.max(...es);
          holder.innerHTML = '';
          outs.forEach(o => { const c = lib.cmap(hi > lo ? (o.e - lo) / (hi - lo) : 0.5); const sp = h('span', { class: 'g', title: 'energy ' + o.e.toFixed(3), style: `background:rgb(${c.join(',')});color:${lum(c) > 0.5 ? C.bg : C.ink}` }, o.ch === '\n' ? '⏎' : o.ch); holder.appendChild(sp); });
          if (outs.length < 60) setTimeout(stepOne, 0);
          else holder.appendChild(h('span', { class: 'thinking-note' }, `  (energy range ${lo.toFixed(2)} to ${hi.toFixed(2)}; dark = low)`));
        };
        stepOne();
      });
    }

    function nSteps() { return R1 ? R1.N : S1.N; }
    function syncPlay() { bPlay.textContent = S1.playing ? 'Pause' : 'Play'; slT.set(Math.round(Math.min(S1.t, nSteps()))); }
    function shownCand() { return R1 ? (S1.show >= 0 && S1.show < R1.cands.length ? S1.show : R1.winner) : 0; }

    function recompute1(playFromStart, keepEnd) {
      exBtns.forEach((b, i) => b.setAttribute('aria-pressed', String(S1.src === 'ex' && S1.ex === i)));
      const ex = S1.src === 'ex' ? D.examples[S1.ex] : null;
      if (!model) {
        // stored fallback: top-8 only, M = 1
        const P = ex.steps.map(s => { const p = new Float64Array(V); let used = 0; s.top.forEach(([c, v]) => { p[vidx.get(c)] = v; used += v; }); const rest = Math.max(0, 1 - used) / (V - s.top.length); for (let k = 0; k < V; k++) if (!p[k]) p[k] = rest; return p; });
        const cand = { E: ex.steps.map(s => s.energy), P, Y: P.map(p => Float64Array.from(p, v => Math.log(v))), G: null };
        R1 = { N: ex.steps.length - 1, ctxStr: ex.context, trueIdx: vidx.get(ex.true_next), cands: [cand], winner: 0, ex, alpha: alpha0 };
      } else {
        const ctxStr = ex ? ex.context : padCtx(normalize(S1.custom));
        const cx = ctxFor(ctxStr);
        const al = alpha0 * Math.pow(2, S1.ak);
        const cands = [];
        for (let j = 0; j < S1.M; j++) cands.push(Object.assign(model.think(cx, y0For(S1.seed, j), { steps: S1.N, alpha: al, sigma: S1.sigma, seed: 101 + j + 1000 * S1.seed }), { j }));
        let winner = 0; cands.forEach((c, j) => { if (c.E[S1.N] < cands[winner].E[S1.N]) winner = j; });
        R1 = { N: S1.N, ctxStr, trueIdx: ex ? vidx.get(ex.true_next) : null, cands, winner, ex, alpha: al, cx };
      }
      slT.input.max = R1.N;
      if (playFromStart) { S1.t = 0; S1.playing = !reduced; } else if (keepEnd) { S1.t = R1.N; S1.playing = false; } else S1.t = Math.min(S1.t, R1.N);
      syncPlay(); renderCands(); kick1(); bus.emit(R1);
    }

    // bar tween state
    const barDisp = new Map();
    let lastBarTop = [];
    function barTargets() {
      const c = R1.cands[shownCand()], N = R1.N, i = Math.min(N, Math.floor(S1.t + 1e-9)), f = S1.t - i;
      const p = lerpArr(c.P[i], c.P[i + 1], reduced ? 0 : f); const top = topK(p, 8);
      return { p, top };
    }
    function easeBars(dt) {
      const { p, top } = barTargets(); lastBarTop = top; let settled = true;
      const k = reduced ? 1 : 1 - Math.exp(-dt * 14);
      const want = new Map(top.map((ix, r) => [ix, r]));
      want.forEach((r, ix) => { if (!barDisp.has(ix)) barDisp.set(ix, { y: r + 0.6, w: 0, a: 0 }); });
      barDisp.forEach((d, ix) => {
        const tr = want.has(ix) ? want.get(ix) : d.y, tw = p[ix], ta = want.has(ix) ? 1 : 0;
        d.y += (tr - d.y) * k; d.w += (tw - d.w) * k; d.a += (ta - d.a) * k;
        if (Math.abs(tr - d.y) > 0.01 || Math.abs(tw - d.w) > 0.002 || Math.abs(ta - d.a) > 0.02) settled = false;
        if (!want.has(ix) && d.a < 0.02) barDisp.delete(ix);
      });
      return settled;
    }
    function drawBars(ctx, w, hh) {
      screen(ctx, w, hh); if (!R1) return;
      const c = R1.cands[shownCand()], N = R1.N, i = Math.min(N, Math.floor(S1.t + 1e-9));
      const { p } = barTargets();
      T(ctx, `softmax(ŷ${sub(i)}): next-character probabilities, step ${i} of ${N}`, 14, 12, { size: 12.5, color: C.ink, maxWidth: w - 28 });
      const x0 = 50, x1 = w - 64, top = 40, rowH = 25.5, bw = x1 - x0;
      // grid
      ctx.save(); ctx.strokeStyle = C.rule; ctx.lineWidth = 1;
      [0, 0.25, 0.5, 0.75, 1].forEach(v => { const x = x0 + v * bw; ctx.beginPath(); ctx.moveTo(x, top - 4); ctx.lineTo(x, top + 8 * rowH); ctx.stroke(); T(ctx, String(v), x, top + 8 * rowH + 4, { size: 10, kind: 'mono', color: C.faint, align: 'center' }); });
      ctx.restore();
      const base = R1.ex && R1.ex.baseline_top ? new Map(R1.ex.baseline_top.map(([ch, v]) => [vidx.get(ch), v])) : null;
      const arg = argmax(p);
      const sorted = [...barDisp.entries()].sort((a, b) => b[1].a - a[1].a);
      sorted.forEach(([ix, d]) => {
        const y = top + d.y * rowH; if (d.a < 0.02) return;
        ctx.save(); ctx.globalAlpha = Math.min(1, d.a);
        const isTrue = ix === R1.trueIdx, isArg = ix === arg;
        // label box
        lib.rr(ctx, 12, y + 2, 28, rowH - 5, 5); ctx.fillStyle = C.panel2; ctx.fill();
        if (isTrue) { ctx.strokeStyle = C.truth; ctx.lineWidth = 2; ctx.stroke(); }
        T(ctx, VD[ix], 26, y + rowH / 2 - 1, { size: 14, kind: 'mono', color: isTrue ? C.truth : C.ink, align: 'center', baseline: 'middle' });
        const bwid = Math.max(1.5, d.w * bw);
        lib.rr(ctx, x0, y + 4, bwid, rowH - 9, 3); ctx.fillStyle = isArg ? C.ebt : lib.rgba(C.ebt, 0.5); ctx.fill();
        if (isTrue) { ctx.strokeStyle = C.truth; ctx.lineWidth = 1.5; ctx.stroke(); }
        if (base && base.has(ix)) { const bx = x0 + base.get(ix) * bw; ctx.fillStyle = C.ar; ctx.fillRect(bx - 1.5, y + 1, 3, rowH - 3); }
        T(ctx, p[ix].toFixed(3), w - 12, y + rowH / 2 - 1, { size: 12, kind: 'mono', color: isArg ? C.ebt : C.muted, align: 'right', baseline: 'middle' });
        ctx.restore();
      });
      // true char outside the top 8
      const yT = top + 8 * rowH + 22;
      if (R1.trueIdx != null) {
        const rank = Array.from(p).filter(v => v > p[R1.trueIdx]).length + 1;
        const inTop = lastBarTop.includes(R1.trueIdx);
        T(ctx, inTop ? `True next character “${VD[R1.trueIdx]}” is rank ${rank} of ${V}.` : `True next character “${VD[R1.trueIdx]}”: p = ${p[R1.trueIdx].toFixed(4)}, rank ${rank} of ${V}.`, 14, yT, { size: 12, color: C.truth, maxWidth: w - 28 });
      } else T(ctx, 'Your own text: the true next character is unknown.', 14, yT, { size: 12, color: C.faint });
      if (base) {
        const bp = R1.ex.baseline_p_true;
        T(ctx, `▍ one-pass baseline${bp != null ? `, p(true) = ${bp.toFixed(3)}` : ''}`, 14, yT + 18, { size: 12, color: C.ar, maxWidth: w - 28 });
      }
    }
    const SUBD = '₀₁₂₃₄₅₆₇₈₉';
    function sub(n) { return String(n).split('').map(d => SUBD[+d]).join(''); }

    function drawTraj(ctx, w, hh) {
      screen(ctx, w, hh); if (!R1) return;
      const wide = w >= 620, N = R1.N, t = Math.min(S1.t, N), sc = shownCand();
      const pw = wide ? (w - 30) / 2 : w, ph = wide ? hh : hh / 2;
      // panel 1: energy
      const all = R1.cands.flatMap(c => c.E); let lo = Math.min(...all), hi = Math.max(...all); const pad = (hi - lo) * 0.08 || 0.1; lo -= pad; hi += pad;
      const yt = niceTicks(lo, hi, 4), xt = niceTicks(0, N, Math.min(6, N));
      const A = ax(ctx, { x: 52, y: 36, w: pw - 70, h: ph - 76, xlim: [0, N], ylim: [lo, hi], xticks: xt, yticks: yt, xlabel: 'thinking step i' });
      T(ctx, 'Energy E(x, ŷᵢ)', 14, 10, { size: 12.5, color: C.ink });
      if (R1.cands.length > 1) T(ctx, `bold = lowest final energy (kept)`, pw - 14, 11, { size: 11, color: C.faint, align: 'right' });
      R1.cands.forEach((c, j) => { if (j === sc) return; lib.plot(ctx, A, c.E.map((e, i) => [i, e]), { color: lib.rgba(C.ebt, 0.35), width: 1.5 }); });
      const cs = R1.cands[sc];
      lib.plot(ctx, A, cs.E.map((e, i) => [i, e]), { color: lib.rgba(C.ebt, 0.28), width: 3 });
      const upto = cs.E.slice(0, Math.floor(t) + 1).map((e, i) => [i, e]); const fi = t - Math.floor(t);
      if (fi > 0 && Math.floor(t) < N) upto.push([t, cs.E[Math.floor(t)] + (cs.E[Math.floor(t) + 1] - cs.E[Math.floor(t)]) * fi]);
      lib.plot(ctx, A, upto, { color: C.ebt, width: 3 });
      cs.E.forEach((e, i) => { if (i <= t + 1e-6) lib.dot(ctx, A.X(i), A.Y(e), 2.6, C.ebt); });
      const cur = upto[upto.length - 1]; lib.dot(ctx, A.X(cur[0]), A.Y(cur[1]), 5, C.ebt, { stroke: C.bg, lw: 2 });
      ctx.save(); ctx.strokeStyle = lib.rgba(C.ink, 0.35); ctx.setLineDash([3, 4]); ctx.beginPath(); ctx.moveTo(A.X(t), 36); ctx.lineTo(A.X(t), 36 + ph - 76); ctx.stroke(); ctx.restore();
      // panel 2: p(true) on log axis, or entropy for typed text
      const ox = wide ? pw + 30 : 0, oy = wide ? 0 : ph;
      if (R1.trueIdx != null) {
        const lo2 = 1e-3, hi2 = 1;
        const B = ax(ctx, { x: ox + 52, y: oy + 36, w: pw - 70, h: ph - 76, xlim: [0, N], ylim: [lo2, hi2], ylog: true, xticks: xt, yticks: [0.001, 0.01, 0.1, 1], yfmt: v => String(v), xlabel: 'thinking step i' });
        T(ctx, `p(true next “${VD[R1.trueIdx]}”), log scale`, ox + 14, oy + 10, { size: 12.5, color: C.ink });
        const clampP = (v) => Math.max(lo2, Math.min(1, v));
        R1.cands.forEach((c, j) => { if (j === sc) return; lib.plot(ctx, B, c.P.map((p, i) => [i, clampP(p[R1.trueIdx])]), { color: lib.rgba(C.truth, 0.3), width: 1.5 }); });
        const pts = cs.P.map((p, i) => [i, clampP(p[R1.trueIdx])]);
        lib.plot(ctx, B, pts, { color: lib.rgba(C.truth, 0.28), width: 3 });
        lib.plot(ctx, B, pts.filter(q => q[0] <= t + 1e-6), { color: C.truth, width: 3 });
        if (R1.ex && R1.ex.baseline_p_true != null) {
          const by = B.Y(clampP(R1.ex.baseline_p_true));
          ctx.save(); ctx.strokeStyle = C.ar; ctx.lineWidth = 2; ctx.setLineDash([6, 4]); ctx.beginPath(); ctx.moveTo(B.X(0), by); ctx.lineTo(B.X(N), by); ctx.stroke(); ctx.restore();
          T(ctx, 'one-pass baseline', B.X(N) - 4, by + (by < oy + 60 ? 6 : -18), { size: 11, color: C.ar, align: 'right' });
        }
        const ip = Math.min(N, Math.floor(t)); lib.dot(ctx, B.X(ip), B.Y(clampP(cs.P[ip][R1.trueIdx])), 5, C.truth, { stroke: C.bg, lw: 2 });
      } else {
        const B = ax(ctx, { x: ox + 52, y: oy + 36, w: pw - 70, h: ph - 76, xlim: [0, N], ylim: [0, 4.2], xticks: xt, yticks: [0, 1, 2, 3, 4], xlabel: 'thinking step i' });
        T(ctx, 'Entropy of the guess (nats)', ox + 14, oy + 10, { size: 12.5, color: C.ink });
        const by = B.Y(lnV); ctx.save(); ctx.strokeStyle = C.faint; ctx.setLineDash([4, 4]); ctx.beginPath(); ctx.moveTo(B.X(0), by); ctx.lineTo(B.X(N), by); ctx.stroke(); ctx.restore();
        T(ctx, `uniform over ${V} = ${lnV.toFixed(2)}`, B.X(N) - 4, by - 16, { size: 11, color: C.faint, align: 'right' });
        R1.cands.forEach((c, j) => { lib.plot(ctx, B, c.P.map((p, i) => [i, entropy(p)]).filter(q => j !== sc || q[0] <= t + 1e-6), { color: j === sc ? C.muted : lib.rgba(C.muted, 0.3), width: j === sc ? 3 : 1.5 }); });
      }
    }

    function drawStrip(ctx, w, hh) {
      screen(ctx, w, hh); if (!R1) return;
      const c = R1.cands[shownCand()], N = R1.N, t = Math.min(S1.t, N), i = Math.floor(t + 1e-9), f = t - i;
      const y = lerpArr(c.Y[i], c.Y[i + 1], f);
      T(ctx, `ŷ${sub(i)} itself: ${V} logits (bars) and the next update ŷᵢ₊₁ − ŷᵢ (amber ticks)`, 14, 10, { size: 12.5, color: C.ink, maxWidth: w - 28 });
      let mx = 1; c.Y.forEach(v => v.forEach(q => { if (Math.abs(q) > mx) mx = Math.abs(q); }));
      const x0 = 14, x1 = w - 14, cw = (x1 - x0) / V, top = 36, bot = hh - 30, zero = (top + bot) / 2, sy = (bot - top) / 2 / mx;
      ctx.save(); ctx.strokeStyle = C.rule; ctx.beginPath(); ctx.moveTo(x0, zero); ctx.lineTo(x1, zero); ctx.stroke(); ctx.restore();
      const p = lerpArr(c.P[i], c.P[i + 1], f), arg = argmax(p);
      for (let k = 0; k < V; k++) {
        const x = x0 + k * cw, v = y[k];
        const col = k === R1.trueIdx ? C.truth : k === arg ? C.ebt : lib.rgba(C.muted, 0.55);
        ctx.fillStyle = col; ctx.fillRect(x + cw * 0.15, Math.min(zero, zero - v * sy), cw * 0.7, Math.max(1, Math.abs(v * sy)));
        if (i < N) { const dy = c.Y[i + 1][k] - c.Y[i][k]; if (Math.abs(dy * sy) > 1) { const ya = zero - v * sy, yb = zero - (c.Y[i][k] + dy) * sy; ctx.strokeStyle = C.ebt; ctx.lineWidth = Math.max(1.2, cw * 0.25); ctx.beginPath(); ctx.moveTo(x + cw / 2, ya); ctx.lineTo(x + cw / 2, yb); ctx.stroke(); } }
        if (cw >= 10 || k === arg || k === R1.trueIdx) T(ctx, VD[k], x + cw / 2, bot + 6, { size: cw >= 12 ? 11 : 9.5, kind: 'mono', color: k === R1.trueIdx ? C.truth : k === arg ? C.ebt : C.faint, align: 'center' });
      }
    }

    function renderCands() {
      if (!R1 || R1.cands.length < 2) { candBox.hidden = true; return; }
      candBox.hidden = false; candBox.innerHTML = '';
      const N = R1.N, tb = h('table', {}, h('thead', {}, h('tr', {}, h('th', {}, 'Candidate'), h('th', {}, 'Final energy'), h('th', {}, 'Its guess'), R1.trueIdx != null ? h('th', {}, 'p(true)') : null, h('th', {}, ''))));
      const body = h('tbody', {}); tb.appendChild(body);
      let bestTrue = 0; if (R1.trueIdx != null) R1.cands.forEach((c, j) => { if (c.P[N][R1.trueIdx] > R1.cands[bestTrue].P[N][R1.trueIdx]) bestTrue = j; });
      R1.cands.forEach((c, j) => {
        const g = argmax(c.P[N]);
        const tr = h('tr', { class: (j === shownCand() ? 'on ' : '') + (j === R1.winner ? 'win' : ''), tabindex: '0', role: 'button', 'aria-label': `Show candidate ${j + 1}` },
          h('td', { class: 'num' }, 'ŷ' + sub(N) + ',' + (j + 1)), h('td', { class: 'num' }, c.E[N].toFixed(3)), h('td', { class: 'num' }, `“${VD[g]}” ${c.P[N][g].toFixed(2)}`),
          R1.trueIdx != null ? h('td', { class: 'num' }, c.P[N][R1.trueIdx].toFixed(3) + (j === bestTrue ? ' ★' : '')) : null,
          h('td', {}, j === R1.winner ? 'kept (lowest E)' : ''));
        const pick = () => { S1.show = j; renderCands(); kick1(); };
        tr.addEventListener('click', pick); tr.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pick(); } });
        body.appendChild(tr);
      });
      candBox.appendChild(tb);
      candBox.appendChild(h('p', { class: 'thinking-note', style: 'padding:6px 12px' }, R1.trueIdx != null ? '★ = candidate that gives the true character the most probability (an oracle that peeks at the answer). Click a row to show that candidate above.' : 'Click a row to show that candidate above.'));
    }

    function renderText1() {
      if (!R1) return;
      const N = R1.N, t = Math.min(S1.t, N), i = Math.floor(t + 1e-9), c = R1.cands[shownCand()], p = c.P[i], g = argmax(p);
      // context line
      ctxLine.innerHTML = '';
      const shown = R1.ctxStr;
      ctxLine.appendChild(h('span', { class: 'lbl' }, 'context x'));
      ctxLine.appendChild(h('span', { class: 'ctx' }, shown.replace(/\n/g, '⏎')));
      ctxLine.appendChild(h('span', { class: 'guess', title: 'EBT\'s current most likely next character' }, VD[g]));
      if (R1.trueIdx != null) ctxLine.appendChild(h('span', { class: 'truth', title: 'true next character' }, 'true: ' + VD[R1.trueIdx]));
      // algorithm highlight
      const line = i === 0 && t < 0.5 ? 1 : (t >= N - 1e-6 ? (R1.cands.length > 1 ? 4 : 3) : 3);
      algoLines.forEach((sp, k) => sp.classList.toggle('on', k === line));
      // readouts
      const E = c.E[i], dE = i > 0 ? c.E[i] - c.E[i - 1] : null, H = entropy(p);
      const items = [['step', `${i} / ${N}`], ['E', E.toFixed(3)], ['ΔE', dE == null ? '–' : (dE > 0 ? '+' : '') + dE.toFixed(3)], ['entropy', `${H.toFixed(2)} nats`], ['guess', `“${VD[g]}” ${p[g].toFixed(2)}`]];
      if (R1.trueIdx != null) { items.push(['p(true)', p[R1.trueIdx].toFixed(3)]); items.push(['loss', (-Math.log(p[R1.trueIdx])).toFixed(2) + ' nats']); }
      if (model) items.push(['α', fx(R1.alpha, R1.alpha < 10 ? 1 : 0)]);
      ro1.innerHTML = ''; items.forEach(([k, v]) => ro1.appendChild(h('span', {}, k + ' ', h('b', {}, v))));
      // dynamic note (about the whole run)
      const cs = R1.cands[shownCand()], E0 = cs.E[0], EN = cs.E[N];
      const parts = [];
      parts.push(`Over ${N} steps the energy of this guess went from <b>${E0.toFixed(2)}</b> to <b>${EN.toFixed(2)}</b>.`);
      if (R1.trueIdx != null) {
        const a = cs.P[0][R1.trueIdx], b = cs.P[N][R1.trueIdx];
        parts.push(`The probability of the true “${VD[R1.trueIdx]}” went from ${a.toFixed(3)} to <b>${b.toFixed(3)}</b>${b > a * 1.5 ? ': thinking found it.' : b < a ? ': lower energy did not mean a better answer here.' : '.'}`);
        if (R1.ex && R1.ex.baseline_p_true != null) {
          const bp = R1.ex.baseline_p_true;
          parts.push(bp > b * 1.3 ? `The one-pass baseline is more confident in the truth (${bp.toFixed(3)}). This tiny EBT is weaker than its baseline overall (see the scoreboard).` : bp < b / 1.3 ? `The one-pass baseline gives it only ${bp.toFixed(3)}.` : `The one-pass baseline gives it a similar ${bp.toFixed(3)}.`);
        }
      }
      let ups = 0; cs.E.forEach((e, k) => { if (k > 0 && e > cs.E[k - 1] + 1e-4) ups++; });
      if (ups) parts.push(`Energy went <i>up</i> on ${ups} step${ups > 1 ? 's' : ''}: the step size α overshoots the valley${S1.sigma > 0 ? ', or the Langevin noise kicked the guess uphill' : ''}.`);
      if (R1.cands.length > 1 && R1.trueIdx != null) {
        let bestTrue = 0; R1.cands.forEach((c2, j) => { if (c2.P[N][R1.trueIdx] > R1.cands[bestTrue].P[N][R1.trueIdx]) bestTrue = j; });
        parts.push(bestTrue === R1.winner ? 'Self-verification kept the candidate that is also best on the true character.' : 'Self-verification kept a candidate that is not the best on the true character: the energy is an imperfect judge.');
      }
      if (S1.src === 'ex' && i === 0 && t < 0.5) parts.push('At step 0 every context shows the same random distribution: ŷ₀ is pure noise. Only the gradient steps bring the context in.');
      note1.innerHTML = parts.join(' ');
    }

    const loop1 = lib.loop((dt) => {
      if (!R1) return false;
      if (S1.playing) {
        const N = R1.N;
        if (reduced) { S1._acc = (S1._acc || 0) + dt; if (S1._acc > 0.45) { S1._acc = 0; S1.t = Math.min(N, Math.floor(S1.t) + 1); } }
        else S1.t = Math.min(N, S1.t + dt * Math.max(1.6, N / 5));
        if (S1.t >= N) { S1.t = N; S1.playing = false; }
        syncPlay();
      }
      const settled = easeBars(dt);
      bars.redraw(); traj.redraw(); strip.redraw(); renderText1();
      return S1.playing || !settled;
    });
    function kick1() { if (!R1) return; loop1.start(); if (reduced || !loop1.running) { easeBars(1); bars.redraw(); traj.redraw(); strip.redraw(); renderText1(); } }

    // ===============================================================================================
    // 2. Energy landscape slice: 2 of the 54 logits
    // ===============================================================================================
    const lab2 = h('div', { class: 'thinking-lab' }); root.appendChild(lab2);
    lab2.appendChild(h('div', { class: 'thinking-labhead' },
      h('h3', {}, '2 · The landscape behind each step'),
      h('p', { html: 'The guess lives in 54 dimensions, so we look at a slice: two logits move, the other 52 stay at 0. The colour is the real energy of the model for the context from lab 1. <b>This is the energy landscape over predictions ŷ</b> for one context and fixed weights, not the training loss over weights θ. Click the map to drop a starting guess and watch gradient descent roll downhill.' })));
    const bench2 = h('div', { class: 'bench' }); lab2.appendChild(bench2);
    const ctl2 = h('div', { class: 'card stack thinking-ctl' }); bench2.appendChild(ctl2);
    const ins2 = h('div', { class: 'card stack thinking-ins' }); bench2.appendChild(ins2);
    const S2 = { A: 0, B: 1, ak: 0, N: 20, start: [-4, 8], t: 20, playing: false, grid: null, path: null, hover: null, slice: 0, auto: true };
    const selA = h('select', { id: 'thinking-selA', 'aria-label': 'Character on the x axis' }), selB = h('select', { id: 'thinking-selB', 'aria-label': 'Character on the y axis' });
    D.vocab.forEach((c, k) => { selA.appendChild(h('option', { value: k }, VD[k])); selB.appendChild(h('option', { value: k }, VD[k])); });
    const sliceSel = h('select', { id: 'thinking-slice', 'aria-label': 'Stored slice' });
    if (model) {
      ctl2.appendChild(h('p', { class: 'thinking-note', id: 'thinking-ctx2' }));
      ctl2.appendChild(h('div', { class: 'row' }, h('label', { for: 'thinking-selA', class: 'thinking-sm' }, 'x: logit of'), selA, h('label', { for: 'thinking-selB', class: 'thinking-sm' }, 'y: logit of'), selB));
      const onSel = () => { S2.A = +selA.value; S2.B = +selB.value; S2.auto = false; if (S2.A === S2.B) { S2.B = (S2.A + 1) % V; selB.value = S2.B; } rebuild2(); };
      selA.addEventListener('change', onSel); selB.addEventListener('change', onSel);
    } else {
      (D.energy_slices || []).forEach((s, i) => sliceSel.appendChild(h('option', { value: i }, s.label)));
      ctl2.appendChild(h('div', { class: 'ctl' }, h('label', { for: 'thinking-slice' }, h('span', {}, 'Stored slice')), sliceSel));
      sliceSel.addEventListener('change', () => { S2.slice = +sliceSel.value; rebuild2(); });
    }
    const sl2A = lib.slider({ id: 'thinking-alpha2', label: 'Step size α', min: -2, max: 2, step: 0.5, value: 0, fmt: v => (alpha0 * Math.pow(2, v)).toFixed(alpha0 * Math.pow(2, v) < 10 ? 1 : 0), oninput: v => { S2.ak = v; runPath2('end'); } });
    const sl2N = lib.slider({ id: 'thinking-N2', label: 'Steps', min: 1, max: 40, step: 1, value: S2.N, oninput: v => { S2.N = v; runPath2('end'); } });
    ctl2.appendChild(sl2A.el); ctl2.appendChild(sl2N.el);
    const b2Step = h('button', { type: 'button', class: 'btn' }, 'Step ▸'), b2Play = h('button', { type: 'button', class: 'btn primary' }, 'Play'), b2Reset = h('button', { type: 'button', class: 'btn' }, 'Reset');
    ctl2.appendChild(h('div', { class: 'row' }, b2Step, b2Play, b2Reset));
    ctl2.appendChild(h('p', { class: 'thinking-note', html: 'Click (or focus the map and use the arrow keys, Enter to run) to choose ŷ₀. Contours are lines of equal energy; gradient descent crosses them at right angles.' }));
    const ro2 = h('div', { class: 'readout', 'aria-live': 'polite' }); ctl2.appendChild(ro2);

    ins2.appendChild(headRow('A 2D slice of the 54-D energy landscape', lib.badge('toy', model ? 'live, exact gradients' : 'stored grid')));
    const two = h('div', { class: 'thinking-two' }); ins2.appendChild(two);
    const mapCell = h('div', {}), sideCell = h('div', { class: 'stack' }); two.appendChild(mapCell); two.appendChild(sideCell);
    const EXT = [-6, 10];
    const map = rcanvas(mapCell, { label: 'Energy heatmap over two logits with the gradient-descent path', height: (w) => Math.min(w, 520) + 0, draw: drawMap });
    map.cv.tabIndex = 0;
    const pbars = h('div', { class: 'thinking-pbars' }); sideCell.appendChild(pbars);
    const eplot = rcanvas(sideCell, { label: 'Energy along the path', height: () => 170, draw: drawE2 });
    sideCell.appendChild(h('div', { class: 'row' }, h('span', { class: 'cbar', style: 'flex:1;min-width:120px' }), h('span', { class: 'thinking-sm' }, 'low → high energy')));
    const note2 = h('p', { class: 'thinking-note' }); sideCell.appendChild(note2);
    let grid2Img = null;

    function mapGeom(w, hh) { const l = 44, b = 34, tp = 10, side = Math.min(w - l - 10, hh - b - tp); return { x: l, y: tp, s: side }; }
    const toPx = (g, a, b) => [g.x + (a - EXT[0]) / (EXT[1] - EXT[0]) * g.s, g.y + g.s - (b - EXT[0]) / (EXT[1] - EXT[0]) * g.s];
    const fromPx = (g, px, py) => [EXT[0] + (px - g.x) / g.s * (EXT[1] - EXT[0]), EXT[0] + (g.y + g.s - py) / g.s * (EXT[1] - EXT[0])];
    function energy2(a, b, wantG) {
      if (model && R1 && R1.cx) {
        const y = new Float64Array(V); y[S2.A] = a; y[S2.B] = b; const g = wantG ? new Float64Array(V) : null; const p = new Float64Array(V);
        const E = model.evalE(R1.cx, y, g, p); return { E, ga: g ? g[S2.A] : 0, gb: g ? g[S2.B] : 0, pa: p[S2.A], pb: p[S2.B] };
      }
      // stored-grid fallback: bilinear interpolation and its exact gradient
      const s = D.energy_slices[S2.slice], n = s.grid_n, lo = s.extent[0], hi = s.extent[1], st = (hi - lo) / (n - 1), G = s.energy;
      const fa = Math.max(0, Math.min(n - 1.0001, (a - lo) / st)), fb = Math.max(0, Math.min(n - 1.0001, (b - lo) / st));
      const c = Math.floor(fa), r = Math.floor(fb), u = fa - c, v = fb - r;
      const e00 = G[r][c], e01 = G[r][c + 1], e10 = G[r + 1][c], e11 = G[r + 1][c + 1];
      const E = (1 - u) * (1 - v) * e00 + u * (1 - v) * e01 + (1 - u) * v * e10 + u * v * e11;
      const ea = Math.exp(a), eb = Math.exp(b), Z = ea + eb + (V - 2);
      return { E, ga: ((1 - v) * (e01 - e00) + v * (e11 - e10)) / st, gb: ((1 - u) * (e10 - e00) + u * (e11 - e01)) / st, pa: ea / Z, pb: eb / Z };
    }
    function rebuild2() {
      if (model && !R1) return;
      if (model) {
        if (S2.auto) {
          const c = R1.cands[shownCand()], pN = c.P[R1.N]; const top = topK(pN, 2);
          S2.A = top[0]; S2.B = R1.trueIdx != null && R1.trueIdx !== top[0] ? R1.trueIdx : top[1];
          selA.value = S2.A; selB.value = S2.B;
        }
        const el = document.getElementById('thinking-ctx2');
        if (el) el.innerHTML = `Context from lab 1: <span class="mono">…${R1.ctxStr.slice(-24).replace(/\n/g, '⏎').replace(/</g, '&lt;')}</span>. ${S2.auto ? 'Axes follow its top guess' + (R1.trueIdx != null && R1.trueIdx !== S2.A ? ' and the true character.' : ' and runner-up.') : 'Axes chosen by you.'}`;
        const n = 41, E = [];
        for (let r = 0; r < n; r++) { const b = EXT[1] - (EXT[1] - EXT[0]) * r / (n - 1); const row = []; for (let c2 = 0; c2 < n; c2++) row.push(energy2(EXT[0] + (EXT[1] - EXT[0]) * c2 / (n - 1), b, false).E); E.push(row); }
        S2.grid = E;
      } else {
        const s = D.energy_slices[S2.slice]; S2.A = vidx.get(s.token_a); S2.B = vidx.get(s.token_b); S2.grid = s.energy.slice().reverse();
      }
      grid2Img = null; runPath2('end');
    }
    // recompute the descent path from S2.start; tMode: 'end' shows the finished path, 'start' rewinds to ŷ₀
    function runPath2(tMode) {
      const al = alpha0 * Math.pow(2, S2.ak); let a = S2.start[0], b = S2.start[1]; const path = [];
      for (let i = 0; i <= S2.N; i++) { const r = energy2(a, b, true); path.push({ a, b, E: r.E, ga: r.ga, gb: r.gb, pa: r.pa, pb: r.pb }); if (i < S2.N) { a -= al * r.ga; b -= al * r.gb; } }
      S2.path = path; S2.t = tMode === 'start' ? 0 : S2.N;
      draw2();
    }
    function makeGridImg(g) {
      const E = S2.grid, n = E.length; let lo = Infinity, hi = -Infinity; E.forEach(r => r.forEach(v => { lo = Math.min(lo, v); hi = Math.max(hi, v); }));
      const off = document.createElement('canvas'); off.width = Math.round(g.s * 2); off.height = Math.round(g.s * 2); const oc = off.getContext('2d');
      lib.heatmap(oc, E, 0, 0, off.width, off.height, { range: [lo, hi], gamma: 0.8 });
      const lv = []; for (let q = 1; q <= 9; q++) lv.push(lo + (hi - lo) * Math.pow(q / 10, 1.25));
      lib.contours(oc, E, 0, 0, off.width, off.height, lv, { color: 'rgba(233,238,246,0.30)', width: 1.4 });
      return { off, lo, hi, s: g.s };
    }
    function drawMap(ctx, w, hh) {
      screen(ctx, w, hh); if (!S2.grid || !S2.path) return;
      const g = mapGeom(w, hh);
      if (!grid2Img || grid2Img.s !== g.s) grid2Img = makeGridImg(g);
      ctx.drawImage(grid2Img.off, g.x, g.y, g.s, g.s);
      // ticks
      [-6, -2, 2, 6, 10].forEach(v => { const [px] = toPx(g, v, 0), [, py] = toPx(g, 0, v); T(ctx, String(v), px, g.y + g.s + 5, { size: 10, kind: 'mono', color: C.faint, align: 'center' }); T(ctx, String(v), g.x - 6, py, { size: 10, kind: 'mono', color: C.faint, align: 'right', baseline: 'middle' }); });
      T(ctx, `logit of “${VD[S2.A]}” →`, g.x + g.s / 2, g.y + g.s + 19, { size: 11.5, color: S2.A === (R1 && R1.trueIdx) ? C.truth : C.muted, align: 'center' });
      ctx.save(); ctx.translate(13, g.y + g.s / 2); ctx.rotate(-Math.PI / 2); T(ctx, `logit of “${VD[S2.B]}” →`, 0, 0, { size: 11.5, color: S2.B === (R1 && R1.trueIdx) ? C.truth : C.muted, align: 'center', baseline: 'middle' }); ctx.restore();
      // path
      ctx.save(); ctx.beginPath(); ctx.rect(g.x, g.y, g.s, g.s); ctx.clip();
      const tN = Math.min(S2.N, S2.t), it = Math.floor(tN), f = tN - it;
      const pts = S2.path.slice(0, it + 1).map(q => toPx(g, q.a, q.b));
      if (f > 0 && it < S2.N) { const q0 = S2.path[it], q1 = S2.path[it + 1]; pts.push(toPx(g, q0.a + (q1.a - q0.a) * f, q0.b + (q1.b - q0.b) * f)); }
      lib.line(ctx, S2.path.map(q => toPx(g, q.a, q.b)), { color: lib.rgba(C.ebt, 0.25), width: 2, dash: [3, 4] });
      lib.line(ctx, pts, { color: C.ebt, width: 2.5 });
      pts.forEach((p, k) => { if (k < pts.length - 1 || f === 0) lib.dot(ctx, p[0], p[1], 2.4, C.ebt); });
      const s0 = toPx(g, S2.start[0], S2.start[1]); ctx.strokeStyle = C.ink; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(s0[0], s0[1], 6, 0, Math.PI * 2); ctx.stroke();
      const cur = pts[pts.length - 1]; lib.dot(ctx, cur[0], cur[1], 6, C.ebt, { stroke: C.bg, lw: 2 });
      if (S2.hover) { const hp = toPx(g, S2.hover[0], S2.hover[1]); ctx.strokeStyle = lib.rgba(C.ink, 0.5); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(hp[0] - 7, hp[1]); ctx.lineTo(hp[0] + 7, hp[1]); ctx.moveTo(hp[0], hp[1] - 7); ctx.lineTo(hp[0], hp[1] + 7); ctx.stroke(); }
      ctx.restore();
      T(ctx, 'ŷ₀', s0[0] + 9, s0[1] - 16, { size: 12, color: C.ink });
    }
    function drawE2(ctx, w, hh) {
      screen(ctx, w, hh); if (!S2.path) return;
      const es = S2.path.map(q => q.E); let lo = Math.min(...es), hi = Math.max(...es); const pad = (hi - lo) * 0.1 || 0.05; lo -= pad; hi += pad;
      const A = ax(ctx, { x: 48, y: 30, w: w - 62, h: hh - 66, xlim: [0, S2.N], ylim: [lo, hi], xticks: niceTicks(0, S2.N, 4), yticks: niceTicks(lo, hi, 3), xlabel: 'step' });
      T(ctx, 'Energy along the path', 12, 8, { size: 12, color: C.ink });
      lib.plot(ctx, A, es.map((e, i) => [i, e]), { color: lib.rgba(C.ebt, 0.3), width: 2 });
      const it = Math.floor(Math.min(S2.t, S2.N)); lib.plot(ctx, A, es.slice(0, it + 1).map((e, i) => [i, e]), { color: C.ebt, width: 2.5 });
      lib.dot(ctx, A.X(it), A.Y(es[it]), 4.5, C.ebt, { stroke: C.bg, lw: 2 });
    }
    function draw2() {
      map.redraw(); eplot.redraw();
      if (!S2.path) return;
      const it = Math.floor(Math.min(S2.t, S2.N)), q = S2.path[it];
      const rest = Math.max(0, 1 - q.pa - q.pb);
      pbars.innerHTML = '';
      [[`“${VD[S2.A]}”`, q.pa, S2.A === (R1 && R1.trueIdx) ? C.truth : C.ebt], [`“${VD[S2.B]}”`, q.pb, S2.B === (R1 && R1.trueIdx) ? C.truth : C.ebt], [`other ${V - 2}`, rest, C.faint]].forEach(([l, v, col]) =>
        pbars.appendChild(h('div', { class: 'thinking-score-row' }, h('span', { class: 'l mono' }, l), h('span', { class: 'b' }, h('i', { style: `width:${(v * 100).toFixed(1)}%;background:${col}` })), h('span', { class: 'v' }, v.toFixed(3)))));
      const gn = Math.hypot(q.ga, q.gb);
      ro2.innerHTML = '';
      [['step', `${it} / ${S2.N}`], ['logits', `(${q.a.toFixed(2)}, ${q.b.toFixed(2)})`], ['E', q.E.toFixed(3)], ['|∇E|', gn.toFixed(3)]].forEach(([k, v]) => ro2.appendChild(h('span', {}, k + ' ', h('b', {}, v))));
      if (S2.hover) { const r = energy2(S2.hover[0], S2.hover[1], false); ro2.appendChild(h('span', {}, 'cursor E ', h('b', {}, r.E.toFixed(3)))); }
      const end = S2.path[S2.path.length - 1], s0 = S2.path[0];
      note2.innerHTML = `From ŷ₀ = (${s0.a.toFixed(1)}, ${s0.b.toFixed(1)}) the descent ${end.E < s0.E - 1e-3 ? `lowers the energy by ${(s0.E - end.E).toFixed(2)}` : 'barely moves: the slope here is almost flat'}. ${end.pa > end.pb ? `It ends favouring “${VD[S2.A]}”` : `It ends favouring “${VD[S2.B]}”`} (${Math.max(end.pa, end.pb).toFixed(2)} vs ${Math.min(end.pa, end.pb).toFixed(2)}). Flat regions far from the valley give tiny gradients, so a random start can stall: one reason the paper randomizes the step size and keeps a replay buffer during training.`;
    }
    map.cv.addEventListener('click', (ev) => { const [px, py] = map.toLocal(ev); const g = mapGeom(map.w, map.h); if (px < g.x || px > g.x + g.s || py < g.y || py > g.y + g.s) return; S2.start = fromPx(g, px, py); runPath2(reduced ? 'end' : 'start'); S2.playing = !reduced; kick2(); });
    map.cv.addEventListener('pointermove', (ev) => { const [px, py] = map.toLocal(ev); const g = mapGeom(map.w, map.h); if (px < g.x || px > g.x + g.s || py < g.y || py > g.y + g.s) { if (S2.hover) { S2.hover = null; draw2(); } return; } S2.hover = fromPx(g, px, py); if (!loop2.running) draw2(); });
    map.cv.addEventListener('pointerleave', () => { S2.hover = null; if (!loop2.running) draw2(); });
    map.cv.addEventListener('keydown', (ev) => {
      const d = 0.5, m = { ArrowLeft: [-d, 0], ArrowRight: [d, 0], ArrowUp: [0, d], ArrowDown: [0, -d] }[ev.key];
      if (m) { ev.preventDefault(); S2.start = [Math.max(EXT[0], Math.min(EXT[1], S2.start[0] + m[0])), Math.max(EXT[0], Math.min(EXT[1], S2.start[1] + m[1]))]; runPath2('end'); }
      else if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); S2.t = 0; S2.playing = !reduced; if (reduced) S2.t = S2.N; kick2(); }
    });
    const loop2 = lib.loop((dt) => { if (S2.playing) { S2.t = Math.min(S2.N, S2.t + dt * Math.max(3, S2.N / 4)); if (S2.t >= S2.N) S2.playing = false; } b2Play.textContent = S2.playing ? 'Pause' : 'Play'; draw2(); return S2.playing; });
    function kick2() { b2Play.textContent = S2.playing ? 'Pause' : 'Play'; if (S2.playing) loop2.start(); else draw2(); }
    b2Step.addEventListener('click', () => { S2.playing = false; S2.t = Math.min(S2.N, Math.floor(S2.t + 1e-6) + 1); kick2(); });
    b2Play.addEventListener('click', () => { if (S2.playing) S2.playing = false; else { if (S2.t >= S2.N) S2.t = 0; S2.playing = true; } kick2(); });
    b2Reset.addEventListener('click', () => { S2.playing = false; S2.t = 0; kick2(); });
    let lab2Ready = false;
    let lastCtx2 = null;
    bus.on((r) => {
      if (!model) return;
      if (r.ctxStr !== lastCtx2) { lastCtx2 = r.ctxStr; S2.auto = true; }   // a new context re-picks the axes
      if (!lab2Ready) return;
      clearTimeout(bus._t2); bus._t2 = setTimeout(rebuild2, 60);
    });
    lib.whenVisible(lab2, () => { lab2Ready = true; rebuild2(); });

    // ===============================================================================================
    // 3. Easy vs hard characters (paper Fig 8)
    // ===============================================================================================
    const U3 = D.uncertainty;
    const GCOL = { easy: C.truth, hard: C.bad, other: C.muted };
    const lab3 = h('div', { class: 'thinking-lab' }); root.appendChild(lab3);
    lab3.appendChild(h('div', { class: 'thinking-labhead' },
      h('h3', {}, '3 · Easy and hard characters: does the energy know?'),
      h('p', { html: 'The paper\'s Fig 8 shows token energies over 12 thinking iterations: easy tokens such as “the” or “.” settle low, hard ones such as “quick” or “problem” stay high (p.10–11). Here is the same picture for our toy, one character at a time. All characters start from the same ŷ₀, as in the toy\'s evaluation.' })));
    const bench3 = h('div', { class: 'bench' }); lab3.appendChild(bench3);
    const ctl3 = h('div', { class: 'card stack thinking-ctl' }); bench3.appendChild(ctl3);
    const ins3 = h('div', { class: 'card stack thinking-ins' }); bench3.appendChild(ins3);
    const nSt3 = U3 ? U3.steps.length - 1 : 8;
    const S3 = { sent: 0, step: nSt3, mode: 'rank', sel: 0, custom: null };
    let sentences = U3 ? U3.sentences.slice() : [];
    const sentSeg = lib.segmented({ label: 'Sentence', options: sentences.map((s, i) => [i, 'Sentence ' + (i + 1)]).concat(model ? [['custom', 'Your text']] : []), value: 0, onchange: v => { if (v === 'custom') { customBox.hidden = false; analyzeCustom(); } else { customBox.hidden = true; S3.sent = v; S3.sel = 0; render3(); } } });
    ctl3.appendChild(h('h4', {}, 'Sentence')); ctl3.appendChild(sentSeg.el);
    const customIn = h('textarea', { id: 'thinking-custom3', rows: 3, spellcheck: 'false', 'aria-label': 'Your sentence' }, 'the quick brown fox jumps over the lazy dog. system 2 thinking is a challenging but interesting research problem.');
    const customBtn = h('button', { type: 'button', class: 'btn' }, 'Analyze');
    const customBox = h('div', { class: 'stack', hidden: true }, h('label', { for: 'thinking-custom3', class: 'thinking-sm' }, 'Up to 120 characters; each one is predicted from the 40 before it'), customIn, customBtn);
    ctl3.appendChild(customBox);
    customBtn.addEventListener('click', analyzeCustom);
    ctl3.appendChild(h('h4', {}, 'Colour by'));
    const modeSeg = lib.segmented({ label: 'Colour mode', options: [['rank', 'Energy, this step'], ['abs', 'Energy, all steps'], ['loss', 'True loss']], value: 'rank', onchange: v => { S3.mode = v; render3(); } });
    ctl3.appendChild(modeSeg.el);
    const sl3 = lib.slider({ id: 'thinking-step3', label: 'Thinking step', min: 0, max: nSt3, step: 1, value: nSt3, oninput: v => { S3.step = v; render3(); } });
    ctl3.appendChild(sl3.el);
    const b3Play = h('button', { type: 'button', class: 'btn primary' }, 'Play steps');
    ctl3.appendChild(h('div', { class: 'row' }, b3Play));
    ctl3.appendChild(h('p', { class: 'thinking-note', html: '<b>Energy, this step</b> rescales the colours within the chosen step so you can see which characters sit relatively high. <b>All steps</b> uses one scale, so you see everything cool down at once. <b>True loss</b> (−ln p of the real character after thinking) shows which characters were actually hard. Compare the first and third.' }));
    let play3 = null;
    b3Play.addEventListener('click', () => {
      if (play3) { clearInterval(play3); play3 = null; b3Play.textContent = 'Play steps'; return; }
      S3.step = 0; sl3.set(0); render3(); b3Play.textContent = 'Stop';
      play3 = setInterval(() => { S3.step++; if (S3.step > nSt3) { clearInterval(play3); play3 = null; S3.step = nSt3; b3Play.textContent = 'Play steps'; } sl3.set(S3.step); render3(); }, reduced ? 700 : 450);
    });

    ins3.appendChild(headRow('Per-character energy across thinking steps', lib.badge('toy')));
    const heat = h('div', { class: 'thinking-heat', tabindex: '0', role: 'group', 'aria-label': 'Sentence coloured by energy. Use left and right arrow keys to inspect characters.' }); ins3.appendChild(heat);
    ins3.appendChild(h('div', { class: 'legend' }, h('span', {}, h('i', { class: 'thinking-u hard' }), 'hard: first letter of a word'), h('span', {}, h('i', { class: 'thinking-u easy' }), 'easy: 3rd+ letter of a word, or u after q'), h('span', {}, h('i', { class: 'cbar', style: 'width:60px' }), 'low → high')));
    const ro3 = h('div', { class: 'readout', 'aria-live': 'polite' }); ins3.appendChild(ro3);
    const gplot = rcanvas(ins3, { label: 'Mean energy per thinking step for easy, hard and other characters, plus the selected character', height: (w) => w >= 560 ? 250 : 300, draw: drawGroups });
    const note3 = h('div', { class: 'callout insight' }); ins3.appendChild(note3);
    heat.addEventListener('keydown', (ev) => { const s = curSent(); if (!s) return; if (ev.key === 'ArrowRight') { S3.sel = Math.min(s.chars.length - 1, S3.sel + 1); ev.preventDefault(); render3(); } else if (ev.key === 'ArrowLeft') { S3.sel = Math.max(0, S3.sel - 1); ev.preventDefault(); render3(); } });

    function groupOf(text, j) {
      const isL = (c) => c >= 'a' && c <= 'z';
      const c = text[j]; if (!isL(c)) return 'other';
      if (c === 'u' && j > 0 && text[j - 1] === 'q') return 'easy';
      let k = j; while (k > 0 && isL(text[k - 1])) k--; const pos = j - k;
      return pos === 0 ? 'hard' : pos >= 2 ? 'easy' : 'other';
    }
    function analyzeCustom() {
      if (!model) return;
      const txt = normalize(customIn.value).slice(0, 120); if (!txt.length) return;
      const before = ' '.repeat(CL); const full = before + txt; const al = U3 ? U3.alpha : alpha0; const y0 = y0For(0, 0);
      const energy = [], p_true_final = [], ce_final = [], group = [], chars = [];
      for (let j = 0; j < txt.length; j++) {
        const cx = ctxFor(full.slice(j, j + CL)); const res = model.think(cx, y0, { steps: nSt3, alpha: al });
        const ti = vidx.get(txt[j]); energy.push(res.E.map(e => +e.toFixed(4))); const pt = res.P[nSt3][ti]; p_true_final.push(pt); ce_final.push(-Math.log(pt)); group.push(groupOf(txt, j)); chars.push(txt[j]);
      }
      S3.custom = { text: txt, chars, energy, p_true_final, ce_final, group, custom: true }; S3.sent = 'custom'; S3.sel = 0; render3();
    }
    function curSent() { return S3.sent === 'custom' ? S3.custom : sentences[S3.sent]; }
    function render3() {
      const s = curSent(); heat.innerHTML = ''; if (!s) return;
      const n = s.chars.length, st = Math.min(S3.step, s.energy[0].length - 1);
      let lo = Infinity, hi = -Infinity;
      if (S3.mode === 'abs') s.energy.forEach(r => r.forEach(v => { lo = Math.min(lo, v); hi = Math.max(hi, v); }));
      else if (S3.mode === 'rank') s.energy.forEach(r => { lo = Math.min(lo, r[st]); hi = Math.max(hi, r[st]); });
      else s.ce_final.forEach(v => { lo = Math.min(lo, v); hi = Math.max(hi, v); });
      const val3 = (j) => S3.mode === 'loss' ? s.ce_final[j] : s.energy[j][st];
      // group characters into words so lines wrap between words
      let word = null;
      for (let j = 0; j < n; j++) {
        if (!word || s.chars[j - 1] === ' ') { word = h('span', { class: 'w' }); heat.appendChild(word); }
        const v = (val3(j) - lo) / (hi - lo || 1); const col = lib.cmap(v);
        const span = h('span', { class: 'c ' + s.group[j] + (j === S3.sel ? ' sel' : ''), style: `background:rgb(${col.join(',')});color:${lum(col) > 0.5 ? C.bg : C.ink}`, title: `“${disp(s.chars[j])}” (${s.group[j]}) energy ${s.energy[j][st].toFixed(3)} at step ${st}` }, s.chars[j] === ' ' ? ' ' : s.chars[j]);
        span.addEventListener('mouseenter', () => { S3.sel = j; renderSel3(); heat.querySelectorAll('.c.sel').forEach(e => e.classList.remove('sel')); span.classList.add('sel'); });
        span.addEventListener('click', () => { S3.sel = j; render3(); });
        word.appendChild(span);
      }
      renderSel3();
      // honest summary for this sentence
      const fin = s.energy.map(r => r[r.length - 1]); const r = pearson(fin, s.ce_final);
      const mean = (g) => { const v = fin.filter((_, j) => s.group[j] === g); return v.length ? v.reduce((a, b) => a + b, 0) / v.length : NaN; };
      const mce = (g) => { const v = s.ce_final.filter((_, j) => s.group[j] === g); return v.length ? v.reduce((a, b) => a + b, 0) / v.length : NaN; };
      const G = U3 && U3.groups, gE = G ? (g) => G[g].mean_energy[G[g].mean_energy.length - 1] : null;
      const parts = [];
      if (G) {
        const he = gE('hard'), ee = gE('easy');
        parts.push(`<b>Over ${(G.easy.n + G.hard.n + G.other.n).toLocaleString()} validation characters</b>, hard characters end at mean energy ${he.toFixed(3)} and easy ones at ${ee.toFixed(3)} after ${nSt3} steps, while their true losses are ${G.hard.mean_ce_final.toFixed(2)} vs ${G.easy.mean_ce_final.toFixed(2)} nats.`);
        parts.push(he > ee ? 'So, like the paper\'s Fig 8, harder characters keep higher energy.' : '<b>So our toy does not reproduce the paper\'s Fig 8 by group:</b> the harder word-initial characters settle at <i>lower</i> energy than easy mid-word ones.');
      }
      parts.push(`Within ${s.custom ? 'your text' : 'this sentence'}, final energy and true loss correlate at r = <b>${r.toFixed(2)}</b> (${Math.abs(r) < 0.15 ? 'no real relation' : r > 0 ? 'higher energy goes with a worse guess, weakly' : 'the wrong sign'}).`);
      parts.push('Why this can happen: no loss ever touches the energy value itself, only the guess after the last step (paper p.37), so nothing forces energies of different contexts onto one scale. The paper reports that this calibration emerges in its far larger models; in a 0.26M-parameter toy it is at best partial. Inside one context the energy is more useful: it still ranks candidates (lab 1 and the self-verification chart below).');
      note3.innerHTML = parts.join(' ');
      gplot.redraw();
    }
    function renderSel3() {
      const s = curSent(); if (!s) return; const j = Math.min(S3.sel, s.chars.length - 1); const e = s.energy[j];
      ro3.innerHTML = '';
      [['char', `“${disp(s.chars[j])}” #${j + 1}`], ['group', s.group[j]], ['E step 0', e[0].toFixed(3)], [`E step ${e.length - 1}`, e[e.length - 1].toFixed(3)], ['p(true) after', s.p_true_final[j].toFixed(3)], ['loss', s.ce_final[j].toFixed(2) + ' nats']].forEach(([k, v]) => ro3.appendChild(h('span', {}, k + ' ', h('b', {}, v))));
      gplot.redraw();
    }
    function drawGroups(ctx, w, hh) {
      screen(ctx, w, hh); if (!U3) return;
      const G = U3.groups, steps = U3.steps; const s = curSent(); const j = s ? Math.min(S3.sel, s.chars.length - 1) : -1;
      let lo = Infinity, hi = -Infinity;
      Object.values(G).forEach(g => g.mean_energy.forEach((m, i) => { lo = Math.min(lo, m - g.std_energy[i]); hi = Math.max(hi, m + g.std_energy[i]); }));
      if (s) s.energy[j].forEach(v => { lo = Math.min(lo, v); hi = Math.max(hi, v); });
      const pad = (hi - lo) * 0.06; lo -= pad; hi += pad;
      const wide = w >= 560;
      const A = ax(ctx, { x: 52, y: 34, w: (wide ? w - 190 : w - 66), h: hh - (wide ? 74 : 124), xlim: [0, steps[steps.length - 1]], ylim: [lo, hi], xticks: steps, yticks: niceTicks(lo, hi, 4), xlabel: 'thinking step' });
      T(ctx, 'Mean energy by group (± 1 std), and the selected character', 14, 10, { size: 12.5, color: C.ink, maxWidth: w - 28 });
      ['other', 'easy', 'hard'].forEach(k => {
        const g = G[k]; ctx.save(); ctx.fillStyle = lib.rgba(GCOL[k], 0.12); ctx.beginPath();
        g.mean_energy.forEach((m, i) => { const x = A.X(steps[i]), y = A.Y(m + g.std_energy[i]); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
        for (let i = g.mean_energy.length - 1; i >= 0; i--) ctx.lineTo(A.X(steps[i]), A.Y(g.mean_energy[i] - g.std_energy[i]));
        ctx.closePath(); ctx.fill(); ctx.restore();
        lib.plot(ctx, A, g.mean_energy.map((m, i) => [steps[i], m]), { color: GCOL[k], width: 2.5 });
      });
      if (s) lib.plot(ctx, A, s.energy[j].map((v, i) => [i, v]), { color: C.ebt, width: 2, dash: [5, 4], markers: 2.5 });
      const sx = A.X(Math.min(S3.step, steps[steps.length - 1])); ctx.save(); ctx.strokeStyle = lib.rgba(C.ink, 0.3); ctx.setLineDash([3, 4]); ctx.beginPath(); ctx.moveTo(sx, 34); ctx.lineTo(sx, A.Y(lo)); ctx.stroke(); ctx.restore();
      // legend
      const items = [['hard', `hard (n=${G.hard.n})`], ['easy', `easy (n=${G.easy.n})`], ['other', `other (n=${G.other.n})`]];
      const lx = wide ? w - 128 : 52, ly = wide ? 44 : hh - 52;
      items.forEach(([k, l], i) => { const x = wide ? lx : lx + i * Math.min(110, (w - 70) / 3.2), y = wide ? ly + i * 20 : ly; ctx.fillStyle = GCOL[k]; ctx.fillRect(x, y + 4, 12, 4); T(ctx, l, x + 17, y, { size: 11, color: C.muted }); });
      if (s) { const x = wide ? lx : 52, y = wide ? ly + 64 : ly + 20; ctx.strokeStyle = C.ebt; ctx.setLineDash([5, 4]); ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(x, y + 6); ctx.lineTo(x + 12, y + 6); ctx.stroke(); ctx.setLineDash([]); T(ctx, `selected “${disp(s.chars[j])}”`, x + 17, y, { size: 11, color: C.ebt }); }
    }

    // paper Fig 8 alongside
    const fig8 = h('div', { class: 'card stack' }); lab3.appendChild(fig8);
    fig8.appendChild(headRow('What the paper shows: Fig 8', lib.badge('paper', 'Fig 8, p.12')));
    fig8.appendChild(h('figure', {}, h('div', { class: 'paper-fig' }, h('img', { src: 'media/paper/fig08.png', alt: 'Paper Figure 8: two heatmaps of normalized token energy (rows: tokens of two sentences; columns: iterations 0 to 11). Hard tokens such as quick, brown, research, problem stay darker (higher energy); easy tokens such as is, but, the period become yellow (low).', loading: 'lazy' })),
      h('figcaption', { html: '“for easier to predict tokens, such as ‘the’ or ‘but’, EBTs optimize to lower energies faster, whereas for harder to predict tokens, such as ‘fox’ or ‘problem’ EBTs have higher energy that does not converge across steps” (p.10–11). Colours there: yellow = low, dark purple = high “Normalized Energy”; the normalisation is not stated. In our toy colours, dark = low.' })));
    const PH = lib.data('paper_heatmaps');
    if (PH && PH.heatmaps) {
      const hm = PH.heatmaps.filter(x => /^fig8/.test(x.id));
      if (hm.length) {
        const rows = []; hm.forEach(x => x.rows.forEach((tok, i) => { const e = x.energy[i]; const tail = e.slice(1); rows.push({ tok, v: tail.reduce((a, b) => a + b, 0) / tail.length, fig: x.figure }); }));
        rows.sort((a, b) => b.v - a.v);
        const fig8c = rcanvas(fig8, { label: 'Paper Fig 8 digitized: mean normalized energy over iterations 1 to 11 for each token', height: (w) => w >= 700 ? 200 : 21 * rows.length + 50, draw: (ctx, w, hh) => {
          screen(ctx, w, hh);
          T(ctx, 'Fig 8 read as numbers: mean normalized energy over iterations 1–11 (approx., digitized)', 14, 10, { size: 12.5, color: C.ink, maxWidth: w - 28 });
          const wide = w >= 700;
          if (wide) {
            const x0 = 20, cw = (w - 40) / rows.length, base = hh - 40, top = 40, mx = Math.max(...rows.map(r => r.v));
            rows.forEach((r, i) => { const bh = r.v / mx * (base - top); ctx.fillStyle = lib.cmapCss(r.v / mx); ctx.fillRect(x0 + i * cw + 2, base - bh, cw - 4, bh); ctx.save(); ctx.translate(x0 + i * cw + cw / 2, base + 6); ctx.rotate(-0.6); T(ctx, r.tok, 0, 0, { size: 10.5, color: C.muted, align: 'right' }); ctx.restore(); });
          } else {
            const x0 = 96, bw = w - x0 - 50, mx = Math.max(...rows.map(r => r.v));
            rows.forEach((r, i) => { const y = 36 + i * 21; T(ctx, r.tok, x0 - 8, y + 3, { size: 11, color: C.muted, align: 'right' }); ctx.fillStyle = lib.cmapCss(r.v / mx); ctx.fillRect(x0, y + 3, r.v / mx * bw, 13); T(ctx, r.v.toFixed(2), x0 + r.v / mx * bw + 4, y + 3, { size: 10, kind: 'mono', color: C.faint }); });
          }
        } });
        fig8.appendChild(h('p', { class: 'thinking-note', html: `Highest: ${rows.slice(0, 4).map(r => '“' + r.tok + '”').join(', ')}. Lowest: ${rows.slice(-4).map(r => '“' + r.tok + '”').join(', ')}. Source: data/paper_heatmaps.json, colours read back through the figure's colour bar (about ±0.005).` }));
      }
    }

    // ===============================================================================================
    // 4. Thinking longer and self-verification: toy curves next to the paper's
    // ===============================================================================================
    const lab4 = h('div', { class: 'thinking-lab' }); root.appendChild(lab4);
    lab4.appendChild(h('div', { class: 'thinking-labhead' },
      h('h3', {}, '4 · Does thinking pay? Our toy next to the paper'),
      h('p', { html: 'Two ways to spend more compute on one prediction (p.8): <b>think longer</b> (more gradient steps) and <b>self-verify</b> (optimize M candidates, keep the lowest energy). Top row: our toy on thousands of held-out characters. Bottom row: the paper\'s results, digitized from its figures. Compare shapes, not levels: toy losses are nats per character, paper numbers are perplexities per token.' })));
    const DS = TC ? Object.keys(TC.datasets) : [];
    const DSN = { val: 'Validation (web text)', train: 'Training text', ood_shakespeare: 'OOD: Shakespeare', ood_code: 'OOD: Python code' };
    const S4 = { ds: DS.includes('val') ? 'val' : DS[0], N: 16, sweep: false, bonN: 0 };
    const top4 = h('div', { class: 'grid2' }); lab4.appendChild(top4);
    // (a) thinking curve
    const cA = h('div', { class: 'card stack' }); top4.appendChild(cA);
    cA.appendChild(headRow('Thinking longer (toy)', lib.badge('toy')));
    const dsSeg = lib.segmented({ label: 'Dataset', options: DS.map(k => [k, DSN[k] || k]), value: S4.ds, onchange: v => { S4.ds = v; redraw4(); } });
    cA.appendChild(dsSeg.el);
    const sweepBtn = h('button', { type: 'button', class: 'btn', 'aria-pressed': 'false' }, 'Compare step sizes α');
    sweepBtn.addEventListener('click', () => { S4.sweep = !S4.sweep; sweepBtn.setAttribute('aria-pressed', String(S4.sweep)); if (S4.sweep) { S4.ds = 'val'; dsSeg.set('val'); } redraw4(); });
    if (D.alpha_sweep) cA.appendChild(h('div', { class: 'row' }, sweepBtn));
    const steps4 = TC ? TC.steps : [];
    const sl4 = lib.slider({ id: 'thinking-N4', label: 'Read out at N steps', min: 0, max: steps4[steps4.length - 1] || 16, step: 1, value: S4.N, oninput: v => { S4.N = v; redraw4(); } });
    cA.appendChild(sl4.el);
    const chA = rcanvas(cA, { label: 'Toy validation loss versus number of thinking steps, with the one-pass baseline', height: (w) => w >= 520 ? 270 : 250, draw: drawA });
    const roA = h('div', { class: 'readout' }); cA.appendChild(roA);
    const noteA = h('p', { class: 'thinking-note' }); cA.appendChild(noteA);
    // (b) BoN
    const cB = h('div', { class: 'card stack' }); top4.appendChild(cB);
    cB.appendChild(headRow('Self-verification, Best-of-M (toy)', lib.badge('toy')));
    const BON = D.bon;
    const bonSeg = lib.segmented({ label: 'Steps per candidate', options: (BON ? BON.settings : []).map((s, i) => [i, `N = ${s.N} steps`]), value: 0, onchange: v => { S4.bonN = v; redraw4(); } });
    cB.appendChild(bonSeg.el);
    cB.appendChild(h('p', { class: 'thinking-sm' }, 'Dataset follows the left chart.'));
    const chB = rcanvas(cB, { label: 'Toy loss versus number of candidates M for energy-based selection, oracle selection and an average candidate', height: (w) => w >= 520 ? 270 : 250, draw: drawB });
    const roB = h('div', { class: 'readout' }); cB.appendChild(roB);
    const noteB = h('p', { class: 'thinking-note' }); cB.appendChild(noteB);
    // paper row
    const SC = lib.data('scaling'); const plot = (id) => SC && SC.plots ? SC.plots.find(p => p.id === id) : null;
    const f6a = plot('fig6a'), f6b = plot('fig6b'), f7 = plot('fig7');
    const bot4 = h('div', { class: 'grid3' }); lab4.appendChild(bot4);
    const cP1 = h('div', { class: 'card stack' }); bot4.appendChild(cP1);
    cP1.appendChild(headRow('Thinking at scale', lib.badge('paper', 'Fig 6a, approx.')));
    rcanvas(cP1, { label: 'Paper Fig 6a digitized: perplexity increase on OOD data versus forward passes, Transformer++ flat, EBT falling', height: () => 250, draw: drawP1 });
    cP1.appendChild(h('p', { class: 'thinking-note', html: f6a ? `Mean over four OOD datasets (p.11). EBT: ${f6a.series[1].points[0][1].toFixed(1)} with no extra thinking → ${f6a.series[1].points[4][1].toFixed(1)} at 30 forward passes, ${(100 * (1 - f6a.series[1].points[4][1] / f6a.series[1].points[0][1])).toFixed(0)}% lower: the paper's “up to 29%”. Transformer++ stays at ${f6a.series[0].points[0][1].toFixed(1)}: more passes cannot change its answer. EBT with no thinking starts <i>worse</i> than Transformer++ (p.10).` : 'scaling.json missing.' }));
    const cP2 = h('div', { class: 'card stack' }); bot4.appendChild(cP2);
    cP2.appendChild(headRow('Verification grows with training', lib.badge('paper', 'Fig 6b, approx.')));
    rcanvas(cP2, { label: 'Paper Fig 6b digitized: BoN-5 perplexity improvement versus tokens trained on', height: () => 250, draw: drawP2 });
    cP2.appendChild(h('p', { class: 'thinking-note', html: 'BoN-5 gain over no verification on BigBench Dyck, “increasing up to 10%−14% from 4%−8%” (p.10). The paper saw this trend on Dyck only (p.34). Points are noisy; the line is a least-squares fit.' }));
    const cP3 = h('div', { class: 'card stack' }); bot4.appendChild(cP3);
    cP3.appendChild(headRow('Thinking helps more out of distribution', lib.badge('paper', 'Fig 7, approx.'), lib.badge('toy', 'overlay')));
    const ovBtn = h('button', { type: 'button', class: 'btn', 'aria-pressed': 'true' }, 'Overlay our toy');
    let overlay = true; ovBtn.addEventListener('click', () => { overlay = !overlay; ovBtn.setAttribute('aria-pressed', String(overlay)); chP3.redraw(); });
    cP3.appendChild(h('div', { class: 'row' }, ovBtn));
    const chP3 = rcanvas(cP3, { label: 'Paper Fig 7 digitized: perplexity improvement from thinking versus OOD shift, with our toy points overlaid', height: () => 250, draw: drawP3 });
    const noteP3 = h('p', { class: 'thinking-note' }); cP3.appendChild(noteP3);

    function curve(ds) { return TC.datasets[ds]; }
    function drawA(ctx, w, hh) {
      screen(ctx, w, hh); if (!TC) return;
      const d = curve(S4.ds), st = TC.steps, xs = st[st.length - 1];
      const series = S4.sweep && D.alpha_sweep ? D.alpha_sweep.val_ce.map((row, i) => ({ y: row, label: 'α = ' + D.alpha_sweep.alphas[i], col: lib.mix(C.ebt, C.diff, i / Math.max(1, D.alpha_sweep.alphas.length - 1)), dash: null })) : [{ y: d.ebt_fixed_alpha_ce, label: `EBT, fixed α = ${alpha0}`, col: C.ebt }, { y: d.ebt_random_alpha_ce, label: 'EBT, random α', col: lib.rgba(C.ebt, 0.75), dash: [6, 4] }];
      const allY = series.flatMap(s => s.y.slice(1)).concat([d.baseline_ce]); let lo = Math.min(...allY), hi = Math.max(...allY, series[0].y[1]); const pad = (hi - lo) * 0.12; lo -= pad; hi += pad;
      const A = ax(ctx, { x: 52, y: 34, w: w - 68, h: hh - 74, xlim: [0, xs], ylim: [lo, hi], xticks: niceTicks(0, xs, 8), yticks: niceTicks(lo, hi, 4), xlabel: 'thinking steps N per character' });
      T(ctx, `Loss (nats/char) · ${DSN[S4.ds] || S4.ds}`, 14, 10, { size: 12.5, color: C.ink, maxWidth: w - 28 });
      // training-N band
      const tn = HP.train_steps_N || [2, 3]; ctx.save(); ctx.fillStyle = lib.rgba(C.ebt, 0.08); ctx.fillRect(A.X(tn[0]), 34, A.X(tn[1]) - A.X(tn[0]), hh - 74); ctx.restore();
      T(ctx, 'trained', (A.X(tn[0]) + A.X(tn[1])) / 2, 38, { size: 10, color: C.faint, align: 'center' });
      // baseline
      const by = A.Y(d.baseline_ce); ctx.save(); ctx.strokeStyle = C.ar; ctx.lineWidth = 2.5; ctx.beginPath(); ctx.moveTo(A.X(0), by); ctx.lineTo(A.X(xs), by); ctx.stroke(); ctx.restore();
      T(ctx, 'one-pass baseline (cannot think longer)', A.X(xs) - 4, by - 16, { size: 11, color: C.ar, align: 'right' });
      series.forEach(s => lib.plot(ctx, A, s.y.map((v, i) => [st[i], v]).filter(p => p[1] <= hi), { color: s.col, width: 2.5, dash: s.dash }));
      series.forEach((s, i) => { ctx.fillStyle = s.col; ctx.fillRect(A.X(xs) - 150, 50 + i * 17, 12, 3); T(ctx, s.label, A.X(xs) - 133, 44 + i * 17, { size: 11, color: C.muted }); });
      T(ctx, `step 0 (noise): ${series[0].y[0].toFixed(2)} ↑`, A.X(0) + 6, 48, { size: 10.5, color: C.faint });
      const ni = st.indexOf(S4.N); if (ni >= 0) { const v = series[0].y[ni]; ctx.save(); ctx.strokeStyle = lib.rgba(C.ink, 0.35); ctx.setLineDash([3, 4]); ctx.beginPath(); ctx.moveTo(A.X(S4.N), 34); ctx.lineTo(A.X(S4.N), hh - 40); ctx.stroke(); ctx.restore(); if (v <= hi) lib.dot(ctx, A.X(S4.N), A.Y(v), 5, series[0].col, { stroke: C.bg, lw: 2 }); }
    }
    function drawB(ctx, w, hh) {
      screen(ctx, w, hh); if (!BON) return;
      const st = BON.settings[S4.bonN], d = st.datasets[S4.ds] || st.datasets.val, Ms = BON.M;
      const ys = [].concat(d.energy_select_ce, d.oracle_ce, d.mean_ce); let lo = Math.min(...ys), hi = Math.max(...ys); const pad = (hi - lo) * 0.12; lo -= pad; hi += pad;
      const A = ax(ctx, { x: 52, y: 34, w: w - 68, h: hh - 74, xlim: [Ms[0], Ms[Ms.length - 1]], xlog: true, ylim: [lo, hi], xticks: Ms, xfmt: v => String(v), yticks: niceTicks(lo, hi, 4), xlabel: 'candidates M (log scale)' });
      T(ctx, `Loss (nats/char) · ${DSN[S4.ds] || S4.ds} · N = ${st.N}`, 14, 10, { size: 12.5, color: C.ink, maxWidth: w - 28 });
      const ser = [[d.mean_ce, 'average candidate', C.muted, [4, 4]], [d.oracle_ce, 'oracle (peeks at the answer)', C.truth, [6, 4]], [d.energy_select_ce, 'lowest energy (self-verification)', C.ebt, null]];
      ser.forEach(([y, l, col, dash]) => lib.plot(ctx, A, y.map((v, i) => [Ms[i], v]), { color: col, width: 2.5, dash, markers: 3.5 }));
      ser.slice().reverse().forEach(([, l, col], i) => { ctx.fillStyle = col; ctx.fillRect(A.X(Ms[0]) + 10, hh - 92 - i * 17 + 6, 12, 3); T(ctx, l, A.X(Ms[0]) + 27, hh - 92 - i * 17, { size: 11, color: C.muted }); });
    }
    function drawP1(ctx, w, hh) {
      screen(ctx, w, hh); if (!f6a) return;
      const A = ax(ctx, { x: 52, y: 34, w: w - 68, h: hh - 74, xlim: [0, 31], ylim: [30, 46], xticks: [2, 6, 15, 30], yticks: [32, 36, 40, 44], xlabel: 'forward passes per prediction' });
      T(ctx, 'Perplexity increase on OOD data (lower is better)', 14, 10, { size: 12, color: C.ink, maxWidth: w - 28 });
      const tp = f6a.series[0].points, ep = f6a.series[1].points;
      lib.plot(ctx, A, tp, { color: C.ar, width: 2.5 }); tp.forEach(p => lib.dot(ctx, A.X(p[0]), A.Y(p[1]), 4, C.ar));
      lib.plot(ctx, A, ep, { color: C.ebt, width: 2.5 });
      const kinds = ['circle', 'star', 'tri', 'tri', 'tri']; ep.forEach((p, i) => shape(ctx, kinds[i], A.X(p[0]), A.Y(p[1]), 5.5, C.ebt));
      T(ctx, 'Transformer++', A.X(30), A.Y(tp[0][1]) - 18, { size: 11, color: C.ar, align: 'right' });
      T(ctx, '● no thinking  ★ longer  ▲ + self-verify', A.X(30), A.Y(ep[4][1]) - 20, { size: 10.5, color: C.ebt, align: 'right' });
    }
    function drawP2(ctx, w, hh) {
      screen(ctx, w, hh); if (!f6b) return;
      const s = f6b.series[0];
      const A = ax(ctx, { x: 46, y: 34, w: w - 62, h: hh - 74, xlim: [0, 34], ylim: [3, 15], xticks: [0, 10, 20, 30], yticks: [4, 8, 12], xlabel: 'tokens trained on (billions)' });
      T(ctx, '% perplexity improvement, BoN-5 vs none', 14, 10, { size: 12, color: C.ink, maxWidth: w - 28 });
      s.points.forEach(p => lib.dot(ctx, A.X(p[0]), A.Y(p[1]), 3.6, lib.rgba(C.ebt, 0.85)));
      if (s.fit) { const f = (x) => s.fit.slope * x + s.fit.intercept; lib.plot(ctx, A, [[1, f(1)], [33, f(33)]], { color: C.ink, width: 2 }); }
    }
    function toyOOD() {
      if (!TC) return [];
      const ref = TC.datasets.val; const iN = TC.steps.indexOf(N3);
      return Object.entries(TC.datasets).filter(([k]) => k !== 'train').map(([k, d]) => {
        const ce = d.ebt_fixed_alpha_ce; const best = Math.min(...ce.slice(iN));
        return { k, x: Math.exp(ce[iN]) / Math.exp(ref.ebt_fixed_alpha_ce[iN]), y: 100 * (1 - Math.exp(best - ce[iN])) };
      });
    }
    function drawP3(ctx, w, hh) {
      screen(ctx, w, hh); if (!f7) return;
      const s = f7.series[0];
      const A = ax(ctx, { x: 46, y: 34, w: w - 62, h: hh - 74, xlim: [0.8, 4.6], ylim: [0, 25], xticks: [1, 2, 3, 4], yticks: [0, 5, 10, 15, 20, 25], xlabel: 'OOD shift (downstream ppl ÷ pretraining ppl)' });
      T(ctx, '% perplexity improvement from thinking', 14, 10, { size: 12, color: C.ink, maxWidth: w - 28 });
      if (s.fit) { const f = (x) => s.fit.slope * x + s.fit.intercept; lib.plot(ctx, A, [[1, f(1)], [4.4, f(4.4)]], { color: lib.rgba(C.ebt, 0.8), width: 2 }); }
      s.points.forEach(p => lib.dot(ctx, A.X(p[0]), A.Y(p[1]), 5, C.ebt));
      T(ctx, 'paper EBT, max thinking', A.X(4.4), A.Y(s.points[s.points.length - 1][1]) + 10, { size: 10.5, color: C.ebt, align: 'right' });
      if (overlay) {
        const tp = toyOOD();
        tp.forEach(p => { shape(ctx, 'sq', A.X(p.x), A.Y(p.y), 5, C.truth); T(ctx, (DSN[p.k] || p.k).replace('OOD: ', '').replace('Validation (web text)', 'val'), A.X(p.x) + 8, A.Y(p.y) - 6, { size: 10, color: C.truth }); });
      }
    }
    function redraw4() {
      chA.redraw(); chB.redraw();
      if (TC) {
        const d = curve(S4.ds), st = TC.steps, i = st.indexOf(S4.N), i3 = st.indexOf(N3), v = d.ebt_fixed_alpha_ce[i], v3 = d.ebt_fixed_alpha_ce[i3];
        let bi = i3; d.ebt_fixed_alpha_ce.forEach((x, k) => { if (k >= i3 && x < d.ebt_fixed_alpha_ce[bi]) bi = k; });
        roA.innerHTML = '';
        [['N', S4.N], ['loss', v.toFixed(3)], ['ppl/char', Math.exp(v).toFixed(2)], [`vs N=${N3}`, (v - v3 > 0 ? '+' : '') + (v - v3).toFixed(3)], ['baseline', d.baseline_ce.toFixed(3)]].forEach(([k, x]) => roA.appendChild(h('span', {}, k + ' ', h('b', {}, String(x)))));
        const g = 100 * (1 - Math.exp(d.ebt_fixed_alpha_ce[bi] - v3));
        const tail = d.ebt_fixed_alpha_ce[st.length - 1] > d.ebt_fixed_alpha_ce[bi] + 0.003;
        noteA.innerHTML = S4.sweep && D.alpha_sweep ? `Step size matters (a limitation the paper names, p.17): with α too small the guess crawls, with α too large it overshoots and plateaus higher. The trained value α = ${alpha0} is best here.` :
          `Going from the trained ${N3} steps to ${st[bi]} steps lowers perplexity per character by ${g.toFixed(1)}%${tail ? `, and thinking past ${st[bi]} steps starts to hurt` : ''}. The flat baseline is ${d.baseline_ce < d.ebt_fixed_alpha_ce[bi] ? 'still better than the EBT\'s best' : 'beaten by the EBT'} (${d.baseline_ce.toFixed(2)} vs ${d.ebt_fixed_alpha_ce[bi].toFixed(2)}). Random α (as in training) is slightly worse at test time than the fixed one.`;
      }
      if (BON) {
        const st = BON.settings[S4.bonN], d = st.datasets[S4.ds] || st.datasets.val, last = BON.M.length - 1;
        const gE = d.energy_select_ce[0] - d.energy_select_ce[last], gO = d.oracle_ce[0] - d.oracle_ce[last];
        roB.innerHTML = '';
        [['M=1', d.energy_select_ce[0].toFixed(3)], [`M=${BON.M[last]} energy`, d.energy_select_ce[last].toFixed(3)], ['oracle', d.oracle_ce[last].toFixed(3)], ['corr(E, loss)', d.energy_loss_corr_within_example.toFixed(2)]].forEach(([k, x]) => roB.appendChild(h('span', {}, k + ' ', h('b', {}, x))));
        noteB.innerHTML = `Picking the lowest-energy of ${BON.M[last]} candidates lowers loss by ${gE.toFixed(3)} nats (${(100 * (1 - Math.exp(-gE))).toFixed(1)}% lower perplexity), about ${(100 * gE / gO).toFixed(0)}% of what an oracle that knows the answer could get. A random candidate gains nothing. Within one context, energy and loss of the candidates correlate at r = ${d.energy_loss_corr_within_example.toFixed(2)}: a weak but real verifier, with no reward model. ${S4.bonN === 0 && BON.settings.length > 1 ? `With more steps per candidate (N = ${BON.settings[1].N}) the candidates agree more and verification adds less.` : ''}`;
      }
      const tp = toyOOD();
      if (tp.length) {
        const vv = tp.find(p => p.k === 'val'), far = tp.slice().sort((a, b) => b.x - a.x)[0];
        noteP3.innerHTML = `Paper: thinking helps more as data moves further from pretraining, from ~12% to ~23% (p.11; the shift is a perplexity ratio). Mint squares: our toy (thinking longer only, ${N3} → best N). ${far && vv && far.y < vv.y ? `The toy shows the opposite: its farthest shift (${DSN[far.k] || far.k}, ×${far.x.toFixed(2)}) gains only ${far.y.toFixed(1)}% vs ${vv.y.toFixed(1)}% in distribution. Different OOD sets, no BoN in these points, and a model thousands of times smaller; the paper also notes thinking emerges only with enough data (p.9).` : 'The toy follows the same direction.'}`;
      }
    }
    lib.whenVisible(lab4, redraw4);

    // initial state
    recompute1(false);
    render3();
    redraw4();
  }
})();
