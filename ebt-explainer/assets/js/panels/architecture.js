/* Panel: inside an EBT. How a candidate prediction enters a Transformer, why the naive autoregressive mask leaks,
   the paper's attention scheme (Eq. 3, App. C.3) and its two-matmul implementation, one energy per guess.
   The dots in the mask are measured live: a tiny random-weight 2-layer attention network is perturbed input by input
   (central finite differences) and the sensitivity of every output to every input is recorded. */
(function () {
  const TOKENS = ['The', 'dog', 'caught', 'the', 'frisbee', 'in', 'mid', 'air', '.'];
  // Table D.1 (p.33), non-embedding params in millions
  const SIZES = [
    { id: 'xxs', P: 6.18, L: 6, D: 384, H: 6 }, { id: 'xs', P: 12.4, L: 12, D: 384, H: 6 },
    { id: 'small', P: 48.8, L: 12, D: 768, H: 12 }, { id: 'medium', P: 176, L: 24, D: 1024, H: 16 },
    { id: 'large', P: 396, L: 24, D: 1536, H: 16 }, { id: 'xl', P: 708, L: 24, D: 2048, H: 32 },
  ];
  const MODES = [['tpp', 'Transformer++'], ['concat', 'naive: concat'], ['inter', 'naive: interleave'], ['ebt', 'EBT · Eq. 3']];
  const SUB = (n) => String(n).split('').map(d => '₀₁₂₃₄₅₆₇₈₉'[+d] || d).join('');
  const WARN = '#d4421c';

  // ---------- masks over the grouped order [z_1..z_N | ẑ_2..ẑ_{N+1}] ----------
  // index i < N: observed z_{i+1}; index N+p: prediction ẑ_{p+2} (guess for token p+2, made from z_1..z_{p+1})
  function buildMask(mode, N) {
    const n = mode === 'tpp' ? N : 2 * N;
    const pos = (i) => mode === 'inter' ? (i < N ? 2 * i : 2 * (i - N) + 1) : i;
    const M = [];
    for (let q = 0; q < n; q++) {
      const row = [];
      for (let k = 0; k < n; k++) {
        let a;
        if (mode === 'ebt') a = q < N ? (k < N && k <= q) : (k < N ? k <= q - N : k === q);
        else a = pos(k) <= pos(q);
        row.push(a);
      }
      M.push(row);
    }
    return M;
  }
  // what kind of edge is (query q -> key k)?  ctx | self | target | future | couple
  function kind(q, k, N) {
    if (q < N) return k < N ? 'ctx' : 'couple';
    const p = q - N;
    if (k < N) return k <= p ? 'ctx' : k === p + 1 ? 'target' : 'future';
    return k === q ? 'self' : 'couple';
  }

  // ---------- a tiny attention network (random weights) ----------
  const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
  const vm = (v, W) => { const o = new Array(W[0].length).fill(0); for (let i = 0; i < v.length; i++) { const vi = v[i], Wi = W[i]; for (let j = 0; j < o.length; j++) o[j] += vi * Wi[j]; } return o; };
  const rms = (v) => { let s = 0; for (const x of v) s += x * x; const r = 1 / Math.sqrt(s / v.length + 1e-6); return v.map(x => x * r); };
  const silu = (z) => z / (1 + Math.exp(-z));
  function makeNet(rngF, D = 8, L = 2) {
    const r = rngF;
    const mat = (a, b, s) => Array.from({ length: a }, () => Array.from({ length: b }, () => r.normal() * s));
    const layers = [];
    for (let l = 0; l < L; l++) layers.push({ Wq: mat(D, D, 1.4 / Math.sqrt(D)), Wk: mat(D, D, 1.4 / Math.sqrt(D)), Wv: mat(D, D, 1 / Math.sqrt(D)), Wo: mat(D, D, 1 / Math.sqrt(D)), W1: mat(D, 2 * D, 1 / Math.sqrt(D)), W2: mat(2 * D, D, 1 / Math.sqrt(2 * D)) });
    const vec = (s) => Array.from({ length: D }, () => r.normal() * s);
    return { D, L, layers, we: vec(1 / Math.sqrt(D)), tok: TOKENS.map(() => vec(1)), guess: TOKENS.map(() => vec(1)), pos: TOKENS.map(() => vec(0.5)) };
  }
  function inputs(net, N, mode) {
    const X = [];
    for (let i = 0; i < N; i++) X.push(net.tok[i].map((x, d) => x + net.pos[i][d]));
    if (mode !== 'tpp') for (let p = 0; p < N; p++) X.push(net.guess[p + 1].map((x, d) => x + net.pos[p + 1][d]));
    return X;
  }
  // attention for one layer; returns per-row pre-Wo outputs (used to verify the C.3 scheme too)
  function attend(A, ly, M) {
    const n = A.length, D = A[0].length, sc = 1 / Math.sqrt(D);
    const Q = A.map(a => vm(a, ly.Wq)), K = A.map(a => vm(a, ly.Wk)), V = A.map(a => vm(a, ly.Wv));
    const out = [];
    for (let i = 0; i < n; i++) {
      const s = new Array(n); let mx = -Infinity;
      for (let j = 0; j < n; j++) if (M[i][j]) { s[j] = dot(Q[i], K[j]) * sc; if (s[j] > mx) mx = s[j]; }
      let z = 0; for (let j = 0; j < n; j++) if (M[i][j]) { s[j] = Math.exp(s[j] - mx); z += s[j]; }
      const o = new Array(D).fill(0);
      for (let j = 0; j < n; j++) if (M[i][j]) { const w = s[j] / z; for (let d = 0; d < D; d++) o[d] += w * V[j][d]; }
      out.push(o);
    }
    return out;
  }
  function forward(net, X, M) {
    let H = X.map(v => v.slice());
    for (const ly of net.layers) {
      const O = attend(H.map(rms), ly, M).map(o => vm(o, ly.Wo));
      H = H.map((hv, i) => hv.map((x, d) => x + O[i][d]));
      const A2 = H.map(rms);
      H = H.map((hv, i) => { const m = vm(vm(A2[i], ly.W1).map(silu), ly.W2); return hv.map((x, d) => x + m[d]); });
    }
    return H.map(hv => dot(rms(hv), net.we));
  }
  // J[r][c] = || d out_r / d x_c ||, central differences; rows = output rows (energies or T++ logits proxy)
  function dependence(net, N, mode) {
    const M = buildMask(mode, N), X = inputs(net, N, mode), n = X.length, D = net.D, eps = 1e-5;
    const rows = mode === 'tpp' ? [...Array(N).keys()] : [...Array(N).keys()].map(p => N + p);
    const J = rows.map(() => new Array(n).fill(0));
    for (let c = 0; c < n; c++) for (let d = 0; d < D; d++) {
      const x0 = X[c][d];
      X[c][d] = x0 + eps; const fp = forward(net, X, M);
      X[c][d] = x0 - eps; const fm = forward(net, X, M);
      X[c][d] = x0;
      rows.forEach((r, ri) => { const g = (fp[r] - fm[r]) / (2 * eps); J[ri][c] += g * g; });
    }
    return { rows, J: J.map(r => r.map(Math.sqrt)), M };
  }
  // Appendix C.3, first layer, real numbers: two matmuls + a superdiagonal swap, checked against masked attention
  function c3(net, N) {
    const X = inputs(net, N, 'ebt'), A = X.map(rms), ly = net.layers[0], D = net.D, sc = 1 / Math.sqrt(D);
    const Ao = A.slice(0, N), Ap = A.slice(N);
    const Qp = Ap.map(a => vm(a, ly.Wq)), Kp = Ap.map(a => vm(a, ly.Wk)), Vp = Ap.map(a => vm(a, ly.Wv));
    const Ko = Ao.map(a => vm(a, ly.Wk)), Vo = Ao.map(a => vm(a, ly.Wv));
    const raw = Qp.map(q => Ko.map(k => dot(q, k) * sc)); // N x N, includes superdiagonal = score vs own target
    const target = raw.map((r, i) => i + 1 < N ? r[i + 1] : null);
    const self = Qp.map((q, i) => dot(q, Kp[i]) * sc);
    const S = raw.map((r, i) => { const row = r.concat([0]); row[i + 1] = self[i]; return row; }); // N x (N+1)
    const P = S.map((row, i) => { const m = Math.max(...row.slice(0, i + 2)); const e = row.map((v, j) => j <= i + 1 ? Math.exp(v - m) : 0); const z = e.reduce((a, b) => a + b, 0); return e.map(v => v / z); });
    const diag = P.map((row, i) => row[i + 1]);
    const zp = P.map((row, i) => { const o = new Array(D).fill(0); for (let j = 0; j <= i; j++) for (let d = 0; d < D; d++) o[d] += row[j] * Vo[j][d]; for (let d = 0; d < D; d++) o[d] += diag[i] * Vp[i][d]; return o; });
    const direct = attend(A, ly, buildMask('ebt', N)).slice(N);
    let err = 0; zp.forEach((o, i) => o.forEach((v, d) => { err = Math.max(err, Math.abs(v - direct[i][d])); }));
    return { raw, target, self, S, P, diag, err };
  }

  EBT.panel({
    id: 'architecture',
    nav: 'Inside an EBT',
    title: 'Inside an EBT: the guess goes in, an energy comes out',
    lede: 'A Transformer++ reads its prediction off the output layer. An EBT feeds a candidate prediction into the input, beside the context, and gets back one number per position. Doing that for every position of a sequence at once takes a carefully built attention mask.',
    text: `
      <p>Take a training sequence of $S$ tokens. A Transformer++ embeds them, runs causal blocks, and maps each final hidden state to logits over the vocabulary: the prediction lives at the <em>output</em>.</p>
      <p>An EBT moves it to the <em>input</em>. Every position keeps a guess $\\hat y$ for the next token: a vector of logits over all $V = 50{,}277$ GPT-NeoX tokens, initialized as Gaussian noise. The guess is normalized and projected into embedding space so the blocks can read it:</p>
      <div class="eq">$$p = \\mathrm{softmax}(\\hat y), \\qquad \\hat z = p^{\\top} W_{\\text{proj}} \\in \\mathbb R^{D}$$<span class="why">Listing 1, p.43–44 ("vocab_to_embed"); Fig 2 calls it the Linear Projector. Without the softmax, runs had "extreme activations" and loss spikes (p.43).</span></div>
      <p>The context embeddings $z_o$ and the guess embeddings $z_p$ (each $S \\times D$) are concatenated into one $2S \\times D$ sequence. The network returns one scalar per guess, $E_{i+1} = E_\\theta(z_{\\le i}, \\hat z_{i+1})$, and thinking is one gradient step on all guesses at once:</p>
      <div class="eq">$$\\hat y \\;\\leftarrow\\; \\hat y \\;-\\; \\alpha\\, \\nabla_{\\hat y} \\textstyle\\sum_{i} E_{i+1}$$<span class="why">Listing 1 calls autograd once on predicted_energies.sum(). This gives each guess its own gradient only if energy $i{+}1$ depends on guess $i{+}1$ alone. The mask is what guarantees that.</span></div>`,
    steps: [
      { label: 'Baseline: the Transformer++ mask', html: '<p>Rows are queries, columns are keys. In a causal Transformer++, token $z_i$ attends to $z_1,\\dots,z_i$: a lower triangle. The dots are measured, not drawn: we perturb each input of a small random-weight network and record how far each output moves, $\\lVert \\partial\\,\\mathrm{out}_i / \\partial z_k \\rVert$. They fill exactly the triangle.</p>' },
      { label: 'Feed the guess in, mask naively', html: '<p>Append the guesses after the context and apply the usual causal mask to all $2S$ positions. Every guess now sits after every context token, so $\\hat z_{i+1}$ can attend to $z_{i+1}$, the very token it should predict, and to later tokens (orange). The probe shows its energy moving with them. Training would learn "low energy when the guess matches the next input", a shortcut that vanishes at generation time. The paper names this leak (p.8); this mechanism is our reading.</p>' },
      { label: 'Interleave instead?', html: '<p>Placing each guess right after its context, $z_1, \\hat z_2, z_2, \\hat z_3, \\dots$, removes the leak but adds coupling (hatched): guesses read earlier guesses, and context tokens read guesses, so the context now depends on noise. Footnote 11 (p.31) explains why the paper avoids this: each prediction would carry stochasticity "not only by its initial value but also by all initial values of all previous predictions".</p>' },
      { label: 'The paper\'s mask (Eq. 3)', html: '<p>The context stream is computed exactly as in a Transformer++: lower triangle, empty top-right block (p.31–32). Each guess $\\hat z_{i+1}$ attends to $z_1,\\dots,z_i$ and to itself, the diagonal of the bottom-right block. Leak and coupling are now exactly zero. Press <b>[ resample weights ]</b>: the zeros come from the wiring, not from lucky weights.</p>' },
      { label: 'Build it with two matmuls (C.3)', html: '<p>A per-row "self" entry cannot come out of one $QK^\\top$. Appendix C.3 computes $\\tilde S = Q_p K_o^\\top/\\sqrt{d_k}$, whose superdiagonal holds each guess\'s score against its own target (orange). That entry is zeroed and overwritten with the self-scores $\\mathrm{sum}(Q_p \\ast K_p)/\\sqrt{d_k}$. A causal softmax shifted by one follows, then a split: $z_p = S\\,V_o + \\mathrm{diag}(S) \\ast V_p$. The readout checks the result against plain masked attention over all $2S$ positions.</p>' },
      { label: 'One energy and one gradient per guess', html: '<p>The head maps each guess position to one scalar. In the dashed block, $\\partial E_{i+1}/\\partial \\hat z_k$ is non-zero only for $k = i+1$, so the gradient of $\\sum_i E_{i+1}$ hands every guess exactly its own gradient, and one autograd call per thinking step suffices. The readout measures how much of that gradient would leak between guesses under each scheme. The dashed loop is the thinking step itself: a forward pass, then a backward pass to the input.</p>' },
    ],
    after: `
      <h3>The rest is a Transformer++</h3>
      <p>EBTs reuse the Llama 2 Transformer++ code (p.33) with RMSNorm, SwiGLU, RoPE and Xavier initialization (p.42). The Q/K/V weights are shared between the context and guess streams "to enable a one-to-one comparison to existing feed-forward transformers" (footnote 12, p.32). S1 models prepend a learnable embedding of the optimization step index, credited with "enabling the accumulation of attention mass" (p.42). The bidirectional EBT used for images is built on DiT code and needs no special mask (p.8, p.33); our reading is that the clean image is never an input, so there is nothing to copy.</p>
      <h3>Model sizes (Table D.1, p.33)</h3>
      <div class="tbl"><table><thead><tr><th>size</th><th class="num">non-emb. params</th><th class="num">layers</th><th class="num">D</th><th class="num">heads</th></tr></thead><tbody>
        <tr><td>xxs</td><td class="num">6.18M</td><td class="num">6</td><td class="num">384</td><td class="num">6</td></tr>
        <tr><td>xs</td><td class="num">12.4M</td><td class="num">12</td><td class="num">384</td><td class="num">6</td></tr>
        <tr><td>small</td><td class="num">48.8M</td><td class="num">12</td><td class="num">768</td><td class="num">12</td></tr>
        <tr><td>medium</td><td class="num">176M</td><td class="num">24</td><td class="num">1024</td><td class="num">16</td></tr>
        <tr><td>large</td><td class="num">396M</td><td class="num">24</td><td class="num">1536</td><td class="num">16</td></tr>
        <tr><td>xl</td><td class="num">708M</td><td class="num">24</td><td class="num">2048</td><td class="num">32</td></tr>
      </tbody></table></div>
      <p>With no FFN multiplier (p.34) a block holds about $7D^2$ weights (4 for attention, 3 for SwiGLU), so $N \\approx 7LD^2$: 6.19M for xxs, 396M for large [derived].</p>
      <h3>What the doubled sequence costs</h3>
      <p>The sequence is $2S-2$ long. The C.3 scheme keeps that a two-fold rather than four-fold cost increase, while full $2S \\times 2S$ attention with a generalized causal mask (C.4) costs "around double" the C.3 version (p.33, p.35–36). Because the context stream never reads a guess, it is identical at every thinking step when the step condition is shared (S2, p.45) and could in principle be computed once; the paper's accounting does not assume this [derived]. The <a href="#costs">costs</a> section prices it all.</p>
      <p class="note">The probe is not a trained EBT: 2 layers, 1 head, width 8, random weights, an energy head on each guess, wired with the selected mask. It measures the wiring, which is the same for any weights. Sensitivities are central finite differences computed in your browser.</p>`,
    source: [{ kind: 'paper', note: 'Eq. 3 and App. C.3 (p.31–32), Listing 1, Table D.1' }, { kind: 'concept', note: 'leak mechanism, interleaved variant (our reading)' }, { kind: 'ext', note: 'random-weight probe network, computed live' }],

    figure(stage, ctx) {
      const { lib } = ctx, h = lib.h, C = lib.C;
      const st = { mode: 'tpp', N: 5, size: 0, view: 'mask', op: 4, hover: null, pin: null, def: 2, hl: [], seed: 11, blockHL: false };
      let net = makeNet(lib.rng(st.seed)), cache = {}, c3c = null, timer = null;

      // responsive HiDPI canvas: logical width = CSS width, so text stays at true pixel size on phones
      function rcanvas(parent, hfn, label) {
        const box = h('div', { class: 'canvas-box' }); parent.appendChild(box);
        const cv = h('canvas', { role: 'img', 'aria-label': label }); box.appendChild(cv);
        const o = { canvas: cv, ctx: cv.getContext('2d'), w: 0, h: 0, draw: null };
        o.size = () => {
          const w = Math.max(260, Math.round(box.getBoundingClientRect().width || 600)), hh = Math.round(hfn(w));
          if (w === o.w && hh === o.h) return false;
          const dpr = Math.min(2, window.devicePixelRatio || 1);
          cv.width = Math.round(w * dpr); cv.height = Math.round(hh * dpr); o.ctx.setTransform(dpr, 0, 0, dpr, 0, 0); o.w = w; o.h = hh; return true;
        };
        o.clear = () => { o.ctx.save(); o.ctx.setTransform(1, 0, 0, 1, 0, 0); o.ctx.fillStyle = '#fff'; o.ctx.fillRect(0, 0, cv.width, cv.height); o.ctx.restore(); };
        o.local = (ev) => { const r = cv.getBoundingClientRect(); const p = ev.touches ? ev.touches[0] : ev; return [(p.clientX - r.left) * o.w / r.width, (p.clientY - r.top) * o.h / r.height]; };
        o.size();
        if (window.ResizeObserver) new ResizeObserver(() => { if (o.size() && o.draw) o.draw(); }).observe(box);
        return o;
      }
      const T = (c, s, x, y, o = {}) => lib.text(c, s, x, y, Object.assign({ size: 11, kind: 'mono', color: C.ink }, o));

      // ---------- frame 1: schematic ----------
      const F1 = lib.frame(stage, { label: 'One thinking step, end to end', sub: 'text EBT · Listing 1 (p.43–44) · shapes per sequence' });
      const sch = rcanvas(F1.frame, w => w < 560 ? 304 : 200, 'Schematic of an EBT forward pass: context and guess streams concatenated, Transformer blocks, energy head, and the gradient loop back to the guess');
      const c1 = h('div', { class: 'controls' }); stage.appendChild(c1);
      const sizeSeg = lib.segmented({ label: 'Model size (Table D.1)', options: SIZES.map((s, i) => [i, s.id]), value: 0, onchange: (v) => { st.size = v; drawSch(); sizeRO(); } });
      c1.appendChild(h('span', { class: 'fig-label' }, 'size · Table D.1')); c1.appendChild(sizeSeg.el);
      const sro = h('div', { class: 'readout' }); stage.appendChild(sro);
      function sizeRO() {
        const S = SIZES[st.size], est = 7 * S.L * S.D * S.D / 1e6;
        sro.innerHTML = `<span>${S.L} blocks · D <b>${S.D}</b> · ${S.H} heads</span><span>N <b>${S.P}M</b> non-emb. (7·L·D² ≈ ${est < 100 ? est.toFixed(2) : est.toFixed(0)}M)</span><span>train FLOPs/token/step <b>20N</b> = ${(20 * S.P / 1e3).toFixed(2)} GFLOP vs 6N for T++</span>`;
      }

      function drawSch() {
        const c = sch.ctx, W = sch.w, H = sch.h; sch.clear();
        const tpp = st.mode === 'tpp', hl = new Set(st.hl), S = SIZES[st.size];
        const narrow = W < 560, X = (f) => Math.round(f * W);
        let N;
        if (!narrow) N = {
          tok: [X(.012), 14, X(.158), 60], emb: [X(.19), 14, X(.315), 60],
          ylog: [X(.012), 98, X(.158), 144], sm: [X(.19), 98, X(.315), 144], proj: [X(.35), 98, X(.49), 144],
          cat: [X(.53), 56, X(.64), 102], tf: [X(.675), 44, X(.84), 114], head: [X(.87), 56, X(.988), 102],
        };
        else N = {
          tok: [X(.06), 10, X(.33), 56], emb: [X(.38), 10, X(.64), 56],
          ylog: [X(.06), 96, X(.33), 142], sm: [X(.38), 96, X(.64), 142], proj: [X(.70), 96, X(.985), 142],
          cat: [X(.06), 200, X(.29), 246], tf: [X(.34), 190, X(.68), 256], head: [X(.73), 200, X(.985), 246],
        };
        const cx = (b) => (b[0] + b[2]) / 2, cy = (b) => (b[1] + b[3]) / 2;
        const maskName = { tpp: 'causal mask', concat: 'causal over 2S', inter: 'interleaved', ebt: 'mask: Eq. 3' }[st.mode];
        const lab = {
          tok: ['CONTEXT', 'x₁ … x_S'], emb: ['EMBED', 'S×D'],
          ylog: ['GUESS ŷ', 'S×V logits', 'init N(0, I)'], sm: ['SOFTMAX', 'p = σ(ŷ)'], proj: ['PROJECT', 'p·W → S×D'],
          cat: ['CONCAT', tpp ? '(none)' : '2S×D'], tf: ['BLOCKS ×' + S.L, 'D ' + S.D + ' · ' + S.H + ' heads', maskName],
          head: tpp ? ['LM HEAD', 'S×V logits'] : ['ENERGY', 'E: S×1'],
        };
        const faint = (id) => tpp && (id === 'ylog' || id === 'sm' || id === 'proj' || id === 'cat');
        const col = (id) => hl.has(id) ? C.blue : C.ink;
        const edge = (pts, id, o = {}) => {
          const on = hl.has(id), a = faint(id) ? 0.22 : 1;
          c.save(); c.globalAlpha = a; c.strokeStyle = c.fillStyle = on ? C.blue : C.ink; c.lineWidth = on ? 1.6 : 1;
          if (o.dash) c.setLineDash(o.dash);
          c.beginPath(); c.moveTo(pts[0][0], pts[0][1]); for (let i = 1; i < pts.length; i++) c.lineTo(pts[i][0], pts[i][1]); c.stroke(); c.setLineDash([]);
          const [p, q] = [pts[pts.length - 2], pts[pts.length - 1]], ang = Math.atan2(q[1] - p[1], q[0] - p[0]), hs = 6;
          c.beginPath(); c.moveTo(q[0], q[1]); c.lineTo(q[0] - hs * Math.cos(ang - 0.45), q[1] - hs * Math.sin(ang - 0.45)); c.lineTo(q[0] - hs * Math.cos(ang + 0.45), q[1] - hs * Math.sin(ang + 0.45)); c.closePath(); c.fill();
          c.restore();
        };
        const boxF = (id) => {
          const b = N[id], on = hl.has(id), a = faint(id) ? 0.22 : 1;
          c.save(); c.globalAlpha = a;
          if (on) { c.fillStyle = C.blue4; c.fillRect(b[0], b[1], b[2] - b[0], b[3] - b[1]); }
          c.strokeStyle = on ? C.blue : C.ink; c.lineWidth = on ? 1.5 : 1;
          if (id === 'tf') c.setLineDash([]);
          c.strokeRect(b[0] + .5, b[1] + .5, b[2] - b[0] - 1, b[3] - b[1] - 1);
          const L = lab[id], w = b[2] - b[0];
          T(c, L[0], b[0] + 7, b[1] + 6, { size: 10.5, color: col(id), weight: 600, spacing: 0.6, maxWidth: w - 10 });
          for (let k = 1; k < L.length; k++) T(c, L[k], b[0] + 7, b[1] + 6 + 14 * k, { size: 10.5, color: on ? C.blue : C.muted });
          c.restore();
        };
        // edges
        edge([[N.tok[2], cy(N.tok)], [N.emb[0], cy(N.emb)]], 'e1');
        if (!narrow) {
          if (tpp) edge([[N.emb[2], cy(N.emb)], [N.tf[0] - 14, cy(N.emb)], [N.tf[0] - 14, cy(N.tf)], [N.tf[0], cy(N.tf)]], 'zo');
          else edge([[N.emb[2], cy(N.emb)], [cx(N.cat), cy(N.emb)], [cx(N.cat), N.cat[1]]], 'zo');
          edge([[N.ylog[2], cy(N.ylog)], [N.sm[0], cy(N.sm)]], 'ylog');
          edge([[N.sm[2], cy(N.sm)], [N.proj[0], cy(N.proj)]], 'sm');
          edge([[N.proj[2], cy(N.proj)], [cx(N.cat), cy(N.proj)], [cx(N.cat), N.cat[3]]], 'proj');
          if (!tpp) edge([[N.cat[2], cy(N.cat)], [N.tf[0], cy(N.tf)]], 'cat');
          edge([[N.tf[2], cy(N.tf)], [N.head[0], cy(N.head)]], 'tf');
          T(c, tpp ? 'z  S×D' : 'z_o  S×D', (N.emb[2] + cx(N.cat)) / 2 + (tpp ? 18 : 0), cy(N.emb) - 15, { size: 10.5, color: C.muted, align: 'center' });
          if (!tpp) T(c, 'z_p  S×D', (N.proj[2] + cx(N.cat)) / 2, cy(N.proj) + 5, { size: 10.5, color: hl.has('proj') ? C.blue : C.muted, align: 'center' });
        } else {
          const bx = X(.67), by = 172;
          if (tpp) edge([[N.emb[2], cy(N.emb)], [bx, cy(N.emb)], [bx, by], [cx(N.tf), by], [cx(N.tf), N.tf[1]]], 'zo');
          else edge([[N.emb[2], cy(N.emb)], [bx, cy(N.emb)], [bx, by], [cx(N.cat), by], [cx(N.cat), N.cat[1]]], 'zo');
          edge([[N.ylog[2], cy(N.ylog)], [N.sm[0], cy(N.sm)]], 'ylog');
          edge([[N.sm[2], cy(N.sm)], [N.proj[0], cy(N.proj)]], 'sm');
          edge([[cx(N.proj), N.proj[3]], [cx(N.proj), by + 6], [cx(N.cat) + 8, by + 6], [cx(N.cat) + 8, N.cat[1]]], 'proj');
          if (!tpp) edge([[N.cat[2], cy(N.cat)], [N.tf[0], cy(N.cat)]], 'cat');
          edge([[N.tf[2], cy(N.head)], [N.head[0], cy(N.head)]], 'tf');
          T(c, tpp ? 'z' : 'z_o', bx + 5, 70, { size: 10.5, color: C.muted });
          if (!tpp) T(c, 'z_p', cx(N.proj) + 5, 150, { size: 10.5, color: hl.has('proj') ? C.blue : C.muted });
        }
        Object.keys(N).forEach(boxF);
        // the thinking loop
        const loopOn = hl.has('loop');
        if (!tpp) {
          const yb = narrow ? 276 : 172, xl = narrow ? 8 : cx(N.ylog);
          if (!narrow) edge([[cx(N.head), N.head[3]], [cx(N.head), yb], [xl, yb], [xl, N.ylog[3]]], loopOn ? 'loop' : 'loopx', { dash: [5, 4] });
          else edge([[cx(N.head), N.head[3]], [cx(N.head), yb], [xl, yb], [xl, cy(N.ylog)], [N.ylog[0], cy(N.ylog)]], loopOn ? 'loop' : 'loopx', { dash: [5, 4] });
          const msg = 'ŷ ← ŷ − α ∇ŷ Σ E   (backward pass to the input)';
          T(c, msg, narrow ? W / 2 : (xl + cx(N.head)) / 2, yb + 5, { size: 10.5, color: loopOn ? C.blue : C.muted, align: 'center', maxWidth: W - 20 });
        } else {
          T(c, 'Transformer++: the prediction is read from the output, one forward pass per token', W / 2, narrow ? 276 : 168, { size: 10.5, color: C.muted, align: 'center', maxWidth: W - 20 });
        }
      }

      // ---------- frame 2: mask + probe ----------
      const row = h('div', { class: 'fig-row' }); stage.appendChild(row);
      const F2 = lib.frame(row, { label: 'Attention mask · measured dependence', sub: 'cell = may attend (one layer) · dot = ‖∂out/∂input‖ after 2 layers' });
      F2.wrap.style.flex = '1 1 340px';
      const mx = rcanvas(F2.frame, (w) => { const n = st.view === 'c3' ? st.N : (st.mode === 'tpp' ? st.N : 2 * st.N); const lw = labelW(w), cell = Math.min(st.view === 'c3' ? 54 : 30, (w - lw - 8) / (st.view === 'c3' ? st.N + 1 : 2 * st.N)); return 44 + cell * n + (st.view === 'c3' ? 60 : (st.blockHL ? 40 : 10)); }, 'Attention mask matrix with measured input sensitivities');
      const F3 = lib.frame(row, { label: 'Readout', sub: 'hover or tap a row, click to pin', dashed: true, pad: 12 });
      F3.wrap.style.flex = '1 1 200px';
      const ro = h('div', { class: 'arch-ro' }); F3.frame.appendChild(ro);
      const legend = h('div', { class: 'arch-legend' }); stage.appendChild(legend);
      legend.innerHTML = '<span><i class="sw ctx"></i>context</span><span><i class="sw self"></i>guess sees itself</span><span><i class="sw leak"></i>leak: own target or future token</span><span><i class="sw couple"></i>coupling to another guess</span><span><i class="sw dot"></i>measured sensitivity</span>';

      const c2 = h('div', { class: 'controls' }); stage.appendChild(c2);
      const modeSeg = lib.segmented({ label: 'Attention scheme', options: MODES, value: st.mode, onchange: (v) => { st.mode = v; if (v !== 'ebt' && st.view === 'c3') { st.view = 'mask'; viewSeg.set('mask'); } stopOps(); redraw(); } });
      c2.appendChild(h('span', { class: 'fig-label' }, 'scheme')); c2.appendChild(modeSeg.el);
      const c3row = h('div', { class: 'controls' }); stage.appendChild(c3row);
      const nSl = lib.slider({ id: 'arch-n', label: 'context tokens N', min: 3, max: 8, step: 1, value: st.N, fmt: (v) => String(v), oninput: (v) => { st.N = v; st.pin = null; st.def = Math.min(st.def, v - 1); redraw(); } });
      c3row.appendChild(nSl.el);
      c3row.appendChild(lib.button('resample weights', () => { st.seed += 1; net = makeNet(lib.rng(st.seed * 7919)); cache = {}; redraw(); }));
      const viewSeg = lib.segmented({ label: 'View', options: [['mask', 'mask + probe'], ['c3', 'C.3 build']], value: 'mask', onchange: (v) => { st.view = v; if (v === 'c3') { st.mode = 'ebt'; modeSeg.set('ebt'); } stopOps(); redraw(); } });
      c3row.appendChild(viewSeg.el);
      const opRow = h('div', { class: 'controls' }); stage.appendChild(opRow);
      const OPS = ['① QₚKₒᵀ', '② zero superdiag.', '③ write self-scores', '④ softmax', '⑤ split'];
      const opSeg = lib.segmented({ label: 'C.3 operation', options: OPS.map((s, i) => [i, s]), value: st.op, onchange: (v) => { stopOps(); st.op = v; redraw(); } });
      opRow.appendChild(h('span', { class: 'fig-label' }, 'C.3 op')); opRow.appendChild(opSeg.el);
      opRow.appendChild(lib.button('play', () => playOps(), { primary: true }));

      const info = h('div', { class: 'readout' }); stage.appendChild(info);

      function getDep() {
        const key = st.mode + ':' + st.N;
        if (!cache[key]) cache[key] = dependence(net, st.N, st.mode);
        return cache[key];
      }
      function labelW(w) { return w < 360 ? 70 : 84; }
      const nameOf = (i, N) => i < N ? 'z' + SUB(i + 1) : 'ẑ' + SUB(i - N + 2);
      const wordOf = (i, N) => i < N ? TOKENS[i] : TOKENS[i - N + 1];
      let geo = null;

      function drawMask() {
        const c = mx.ctx, W = mx.w; mx.clear();
        const N = st.N, tpp = st.mode === 'tpp', n = tpp ? N : 2 * N;
        const lw = labelW(W), cell = Math.min(30, (W - lw - 8) / n), x0 = lw, y0 = 40;
        geo = { x0, y0, cell, n };
        const dep = getDep(), M = dep.M;
        const hov = st.hover ? st.hover.r : (st.pin != null ? st.pin : (tpp ? Math.min(st.def, N - 1) : N + Math.min(st.def, N - 1)));
        // group headers
        if (!tpp) {
          T(c, 'context z_o', x0 + cell * N / 2, 2, { size: 10, color: C.muted, align: 'center' });
          T(c, 'guesses z_p', x0 + cell * N * 1.5, 2, { size: 10, color: C.blue, align: 'center' });
        } else T(c, 'tokens z (keys)', x0 + cell * N / 2, 2, { size: 10, color: C.muted, align: 'center' });
        // column labels
        for (let k = 0; k < n; k++) T(c, nameOf(k, N), x0 + (k + .5) * cell, 20, { size: cell < 20 ? 9.5 : 10.5, color: k < N ? C.ink : C.blue, align: 'center' });
        // cells
        for (let q = 0; q < n; q++) {
          const yy = y0 + q * cell;
          T(c, nameOf(q, N) + ' ' + wordOf(q, N), 4, yy + cell / 2, { size: cell < 20 ? 9.5 : 10.5, baseline: 'middle', color: q === hov ? C.blue : (q < N ? C.ink : C.blue), weight: q === hov ? 700 : 400, maxWidth: lw - 6 });
          for (let k = 0; k < n; k++) {
            const xx = x0 + k * cell;
            c.strokeStyle = C.rule; c.lineWidth = 1; c.strokeRect(xx + .5, yy + .5, cell, cell);
            if (!M[q][k]) continue;
            const kd = tpp ? 'ctx' : kind(q, k, N);
            if (kd === 'ctx') { c.fillStyle = q < N ? '#e7e7ec' : C.blue4; c.fillRect(xx + 1, yy + 1, cell - 1, cell - 1); }
            else if (kd === 'self') { c.fillStyle = C.blue3; c.fillRect(xx + 1, yy + 1, cell - 1, cell - 1); }
            else if (kd === 'target' || kd === 'future') { c.fillStyle = 'rgba(212,66,28,0.16)'; c.fillRect(xx + 1, yy + 1, cell - 1, cell - 1); c.strokeStyle = WARN; c.lineWidth = kd === 'target' ? 1.6 : 1; c.strokeRect(xx + 1.5, yy + 1.5, cell - 2, cell - 2); }
            else if (kd === 'couple') {
              c.save(); c.beginPath(); c.rect(xx + 1, yy + 1, cell - 1, cell - 1); c.clip(); c.fillStyle = '#f2f2f4'; c.fillRect(xx, yy, cell, cell);
              c.strokeStyle = '#a3a3a8'; c.lineWidth = 1; c.beginPath(); for (let t = -cell; t < cell; t += 5) { c.moveTo(xx + t, yy + cell); c.lineTo(xx + t + cell, yy); } c.stroke(); c.restore();
            }
          }
        }
        // block separators
        c.strokeStyle = C.ink; c.lineWidth = 1;
        c.strokeRect(x0 + .5, y0 + .5, cell * n, cell * n);
        if (!tpp) { c.beginPath(); c.moveTo(x0 + cell * N + .5, y0); c.lineTo(x0 + cell * N + .5, y0 + cell * n); c.moveTo(x0, y0 + cell * N + .5); c.lineTo(x0 + cell * n, y0 + cell * N + .5); c.stroke(); }
        // measured dependence dots
        let vmax = 0; dep.J.forEach(r => r.forEach(v => { if (v > vmax) vmax = v; }));
        dep.rows.forEach((q, ri) => {
          for (let k = 0; k < n; k++) {
            const v = dep.J[ri][k]; if (!(v > 0)) continue;
            const kd = tpp ? 'ctx' : kind(q, k, N);
            const r = Math.max(1.6, cell * 0.36 * Math.sqrt(v / vmax));
            const colr = kd === 'target' || kd === 'future' ? WARN : kd === 'couple' ? '#6b6b70' : (q < N && !tpp ? C.ink : (tpp ? C.ink : C.blue));
            lib.dot(c, x0 + (k + .5) * cell, y0 + (q + .5) * cell, r, colr);
          }
        });
        if (!tpp && st.mode !== 'inter') T(c, 'context rows: no energy', x0 + cell * N * 1.5, y0 + cell * N / 2, { size: 10, color: C.faint, align: 'center', baseline: 'middle', maxWidth: cell * N - 6 });
        // block highlight for the gradient-independence step
        if (st.blockHL && !tpp) {
          c.save(); c.strokeStyle = C.blue; c.lineWidth = 2; c.setLineDash([4, 3]); c.strokeRect(x0 + cell * N - 1, y0 + cell * N - 1, cell * N + 2, cell * N + 2); c.restore();
          T(c, 'dashed block = ∂E(ẑᵢ)/∂ẑₖ, the Jacobian that ∇ŷ Σ E uses', x0, y0 + cell * n + 8, { size: 10.5, color: C.blue, maxWidth: W - x0 - 4 });
        }
        // hovered row / col
        if (hov != null && hov < n) {
          c.save(); c.strokeStyle = C.blue; c.lineWidth = 1.6; c.strokeRect(x0 - 2, y0 + hov * cell - 1, cell * n + 4, cell + 2); c.restore();
          if (st.hover && st.hover.c != null) { c.save(); c.strokeStyle = C.ink; c.lineWidth = 1.4; c.strokeRect(x0 + st.hover.c * cell + 1, y0 + hov * cell + 1, cell - 1, cell - 1); c.restore(); }
        }
        readout(hov, dep);
      }

      function drawC3() {
        const c = mx.ctx, W = mx.w; mx.clear();
        const N = st.N; if (!c3c || c3c.N !== N || c3c.seed !== st.seed) { c3c = c3(net, N); c3c.N = N; c3c.seed = st.seed; }
        const R = c3c, op = st.op, n1 = N + 1;
        const lw = labelW(W), cell = Math.min(54, (W - lw - 8) / n1), x0 = lw, y0 = 40;
        geo = null;
        const fmt = (v) => (Math.abs(v) < 0.005 ? '0' : v.toFixed(Math.abs(v) >= 10 ? 0 : (cell > 40 ? 2 : 1)));
        T(c, ['① S̃ = QₚKₒᵀ/√d  (N×N, every entry computed)', '② append a column, zero the superdiagonal', '③ write sum(Qₚ∗Kₚ)/√d on the superdiagonal', '④ softmax, causal mask shifted by one', '⑤ diag → ∗Vₚ, the rest → ·Vₒ'][op], 4, 2, { size: 10.5, color: C.blue, maxWidth: W - 8 });
        for (let k = 0; k < n1; k++) {
          const lab = k < N ? 'z' + SUB(k + 1) : '+col';
          if (op === 0 && k === N) continue;
          T(c, lab, x0 + (k + .5) * cell, 22, { size: 10, color: k < N ? C.ink : C.muted, align: 'center' });
        }
        for (let i = 0; i < N; i++) {
          const yy = y0 + i * cell;
          T(c, 'ẑ' + SUB(i + 2) + ' ' + TOKENS[i + 1], 4, yy + cell / 2, { size: 10.5, baseline: 'middle', color: C.blue, maxWidth: lw - 6 });
          for (let k = 0; k < n1; k++) {
            if (op === 0 && k === N) continue;
            const xx = x0 + k * cell, sup = k === i + 1;
            c.strokeStyle = C.rule; c.lineWidth = 1; c.strokeRect(xx + .5, yy + .5, cell, cell);
            let txt = null, colr = C.ink, fill = null, outline = null;
            if (op === 0) {
              txt = fmt(R.raw[i][k]); colr = k <= i ? C.ink : C.faint;
              if (k <= i) fill = C.blue4;
              if (sup) { fill = 'rgba(212,66,28,0.14)'; outline = WARN; colr = WARN; }
            } else if (op === 1 || op === 2) {
              const v = k === N ? 0 : R.raw[i][k];
              if (sup) { if (op === 1) { txt = '0'; outline = WARN; colr = WARN; } else { txt = fmt(R.self[i]); fill = C.blue3; colr = C.blue; outline = C.blue; } }
              else { txt = k === N ? '0' : fmt(v); colr = k <= i ? C.ink : C.faint; if (k <= i) fill = C.blue4; }
            } else {
              if (k > i + 1) { txt = null; }
              else if (sup) { txt = fmt(R.P[i][k]); fill = C.blue3; colr = C.blue; outline = op === 4 ? C.blue : null; }
              else { txt = fmt(R.P[i][k]); fill = C.blue4; }
              if (op === 4 && sup) { fill = '#fff'; txt = '→'; colr = C.blue; }
            }
            if (fill) { c.fillStyle = fill; c.fillRect(xx + 1, yy + 1, cell - 1, cell - 1); }
            if (outline) { c.save(); c.strokeStyle = outline; c.lineWidth = 1.5; c.strokeRect(xx + 1.5, yy + 1.5, cell - 2, cell - 2); c.restore(); }
            if (txt != null && cell >= 22) T(c, txt, xx + cell / 2, yy + cell / 2, { size: cell > 40 ? 11 : 9.5, align: 'center', baseline: 'middle', color: colr });
            if (txt == null && op >= 3 && k > i + 1) { c.fillStyle = '#f7f7f9'; c.fillRect(xx + 1, yy + 1, cell - 1, cell - 1); }
          }
        }
        const wN = cell * (op === 0 ? N : n1);
        c.strokeStyle = C.ink; c.lineWidth = 1; c.strokeRect(x0 + .5, y0 + .5, wN, cell * N);
        // footer
        const yf = y0 + cell * N + 10;
        if (op === 0) T(c, 'orange superdiagonal = each guess vs its own target token: must not survive', x0, yf, { size: 10.5, color: WARN, maxWidth: W - x0 - 4 });
        if (op === 1) T(c, 'masking by elementwise multiplication keeps it differentiable (p.32)', x0, yf, { size: 10.5, color: C.muted, maxWidth: W - x0 - 4 });
        if (op === 2) T(c, 'self-scores: one dot product per row, a Hadamard product summed over features', x0, yf, { size: 10.5, color: C.blue, maxWidth: W - x0 - 4 });
        if (op === 3) T(c, 'each row sums to 1 over z₁…zᵢ and the guess itself (Eq. 3)', x0, yf, { size: 10.5, color: C.muted, maxWidth: W - x0 - 4 });
        if (op === 4) T(c, 'diag(S) = [' + R.diag.map(v => v.toFixed(2)).join(', ') + ']  scales Vₚ; the rest multiplies Vₒ', x0, yf, { size: 10.5, color: C.blue, maxWidth: W - x0 - 4 });
        readoutC3();
      }

      function readout(hov, dep) {
        const N = st.N, tpp = st.mode === 'tpp', n = tpp ? N : 2 * N, M = dep.M;
        let html = '';
        if (hov != null && hov < n) {
          const isP = hov >= N;
          const seen = [], leaks = [], couples = [];
          for (let k = 0; k < n; k++) if (M[hov][k]) {
            const kd = tpp ? 'ctx' : kind(hov, k, N), nm = (k === hov && isP) ? 'itself' : (k < N ? TOKENS[k] : nameOf(k, N));
            if (kd === 'ctx' || kd === 'self') seen.push(nm); else if (kd === 'target') leaks.push(TOKENS[k] + ' (its own target)'); else if (kd === 'future') leaks.push(TOKENS[k] + ' (future)'); else couples.push(nameOf(k, N));
          }
          const title = isP ? `<b>${nameOf(hov, N)}</b> · guess for “${TOKENS[hov - N + 1]}” after “${TOKENS.slice(0, hov - N + 1).join(' ')}”` : `<b>${nameOf(hov, N)}</b> · observed “${TOKENS[hov]}”`;
          html += `<p>${title}</p><p>attends to: ${seen.join(' · ') || 'nothing'}</p>`;
          html += leaks.length ? `<p class="bad">leak: ${leaks.join(' · ')}</p>` : (isP || tpp ? '<p>leak: none</p>' : '');
          if (!tpp) html += couples.length ? `<p class="cpl">${isP ? 'coupled to guesses' : 'reads guesses'}: ${couples.join(' · ')}</p>` : '<p>coupling: none</p>';
          if (st.hover && st.hover.c != null) {
            const k = st.hover.c, ri = dep.rows.indexOf(hov);
            const nm = nameOf(k, N);
            if (ri >= 0) { const v = dep.J[ri][k]; html += `<p class="cell">‖∂${tpp ? 'out' : 'E'}(${nameOf(hov, N)}) / ∂${nm}‖ = <b>${v > 0 ? v.toFixed(3) : 'exactly 0'}</b></p>`; }
            else html += `<p class="cell">${nm}: ${M[hov][k] ? 'may attend' : 'masked'} · no output at this row</p>`;
          }
        }
        if (st.blockHL && !tpp) {
          const xt = (m) => { const key = m + ':' + N; if (!cache[key]) cache[key] = dependence(net, N, m); const d = cache[key]; let on = 0, off = 0; d.rows.forEach((q, ri) => { for (let k = N; k < 2 * N; k++) { if (k === q) on += d.J[ri][k]; else off += d.J[ri][k]; } }); return 100 * off / (on + off); };
          html += `<p class="cell">share of ∂(Σ E)/∂ẑ that crosses between guesses:<br>naive concat <b>${xt('concat').toFixed(1)}%</b> · interleave <b>${xt('inter').toFixed(1)}%</b> · Eq. 3 <b>${xt('ebt').toFixed(1)}%</b></p>`;
        }
        ro.innerHTML = html;
        // mode summary
        let tot = 0, bad = 0, cpl = 0;
        dep.rows.forEach((q, ri) => dep.J[ri].forEach((v, k) => { tot += v; if (!tpp) { const kd = kind(q, k, N); if (kd === 'target' || kd === 'future') bad += v; if (kd === 'couple') cpl += v; } }));
        const scores = tpp ? `N² = ${N * N}` : st.mode === 'ebt' ? `2N²+N = ${2 * N * N + N} (C.3)` : `(2N)² = ${4 * N * N}`;
        info.innerHTML = `<span>sequence <b>${tpp ? N : 2 * N}</b> positions</span><span>scores computed <b>${scores}</b></span>` + (tpp ? '' : `<span>sensitivity from forbidden tokens <b>${(100 * bad / tot).toFixed(1)}%</b></span><span>from other guesses <b>${(100 * cpl / tot).toFixed(1)}%</b></span>`);
      }
      function readoutC3() {
        const R = c3c, N = st.N;
        const tg = R.target.filter(v => v != null);
        ro.innerHTML = `<p><b>Appendix C.3</b>, layer 1 of the probe, N = ${N}</p>` +
          `<p class="bad">QₚKₒᵀ superdiagonal (guess vs its target): ${tg.map(v => v.toFixed(2)).join(', ')}</p>` +
          `<p>self-scores sum(Qₚ∗Kₚ)/√d: ${R.self.map(v => v.toFixed(2)).join(', ')}</p>` +
          `<p class="cell">max |two-matmul result − masked attention over 2N| = <b>${R.err === 0 ? '0 (bit-identical)' : R.err.toExponential(1)}</b></p>`;
        info.innerHTML = `<span>scores: T++ <b>N² = ${N * N}</b></span><span>C.3 <b>2N²+N = ${2 * N * N + N}</b></span><span>C.4 full mask <b>(2N)² = ${4 * N * N}</b></span>`;
      }

      function redraw() {
        mx.size();
        opRow.style.display = st.view === 'c3' ? '' : 'none';
        legend.style.display = st.view === 'c3' ? 'none' : '';
        if (st.view === 'c3') drawC3(); else drawMask();
        drawSch();
      }
      mx.draw = () => (st.view === 'c3' ? drawC3() : drawMask());
      sch.draw = drawSch;

      // hover / tap on the mask
      const hit = (ev) => {
        if (!geo) return null; const [x, y] = mx.local(ev);
        const r = Math.floor((y - geo.y0) / geo.cell), cc = Math.floor((x - geo.x0) / geo.cell);
        if (r < 0 || r >= geo.n) return null;
        return { r, c: cc >= 0 && cc < geo.n ? cc : null };
      };
      mx.canvas.addEventListener('mousemove', (ev) => { if (st.view !== 'mask') return; const hh = hit(ev); const key = hh ? hh.r + ',' + hh.c : ''; if (key !== (st.hover ? st.hover.r + ',' + st.hover.c : '')) { st.hover = hh; drawMask(); } });
      mx.canvas.addEventListener('mouseleave', () => { if (st.view !== 'mask') return; st.hover = null; drawMask(); });
      mx.canvas.setAttribute('tabindex', '0');
      mx.canvas.addEventListener('keydown', (ev) => {
        if (st.view !== 'mask' || (ev.key !== 'ArrowDown' && ev.key !== 'ArrowUp')) return;
        const n = st.mode === 'tpp' ? st.N : 2 * st.N, cur = st.pin != null ? st.pin : (st.mode === 'tpp' ? st.def : st.N + st.def);
        st.pin = Math.max(0, Math.min(n - 1, cur + (ev.key === 'ArrowDown' ? 1 : -1))); st.hover = null; drawMask(); ev.preventDefault(); ev.stopPropagation();
      });
      mx.canvas.addEventListener('click', (ev) => { if (st.view !== 'mask') return; const hh = hit(ev); if (hh) { st.pin = hh.r; st.hover = hh; drawMask(); } });

      function stopOps() { if (timer) clearInterval(timer); timer = null; }
      function playOps() {
        stopOps(); if (st.view !== 'c3') { st.view = 'c3'; viewSeg.set('c3'); st.mode = 'ebt'; modeSeg.set('ebt'); }
        st.op = 0; opSeg.set(0); redraw();
        if (lib.reducedMotion) { st.op = 4; opSeg.set(4); redraw(); return; }
        timer = setInterval(() => { if (st.op >= 4) { stopOps(); return; } st.op += 1; opSeg.set(st.op); drawC3(); }, 1300);
      }

      sizeRO();
      ctx.setCaption('Top: data flow of one thinking step (blue = the part the current step is about). Bottom: rows are queries and columns are keys, in grouped order [context | guesses]. Shaded cells may be attended in one layer. Dots are measured sensitivities of each output to each input in a 2-layer random-weight network wired with that mask; orange marks a guess reaching its own target or a later token.');
      const set = (o) => {
        stopOps(); Object.assign(st, { hover: null, pin: null, blockHL: false }, o);
        modeSeg.set(st.mode); viewSeg.set(st.view); opSeg.set(st.op); redraw();
      };
      return {
        step(i) {
          if (i === 0) set({ mode: 'tpp', view: 'mask', hl: [], def: 2 });
          if (i === 1) set({ mode: 'concat', view: 'mask', hl: ['ylog', 'sm', 'proj', 'cat', 'zo'], def: 1 });
          if (i === 2) set({ mode: 'inter', view: 'mask', hl: ['cat'], def: 2 });
          if (i === 3) set({ mode: 'ebt', view: 'mask', hl: ['tf'], def: 2 });
          if (i === 4) { set({ mode: 'ebt', view: 'c3', hl: ['tf'], op: 0 }); playOps(); }
          if (i === 5) set({ mode: 'ebt', view: 'mask', hl: ['head', 'loop', 'loopx'], def: 3, blockHL: true });
        },
        hide() { stopOps(); },
      };
    },
  });
})();
