/* Panel 16 · text-data: the corpus, the targets, and perplexity.
   Live math: GPT-NeoX token statistics of real RedPajama-V2 documents, the causal next-token layout, token budgets from the
   paper's batch sizes and step counts, perplexity of real text scored by the toy char-level EBT (data/text.json weights,
   run in the browser with N thinking steps), the Dyck bracket stack, and the Table 3 conversions. */
(function () {
  'use strict';
  const VOCAB = 50277;
  const PAIRS = { '(': ')', '[': ']', '{': '}', '<': '>' };
  const T3 = { cols: ['Pretrain', 'GSM8K', 'SQuAD', 'BB Math QA', 'BB Dyck'], tpp: [31.36, 49.6, 52.3, 79.8, 131.5], ebt: [33.43, 43.3, 53.1, 72.6, 125.3] };

  // ---------- toy char-level EBT (same math as src/text_ebt_ref.js; weights from data/text.json) ----------
  function makeCharEBT(text) {
    if (!text || !text.weights || !text.weights.data) return null;
    const bin = atob(text.weights.data), bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    const all = new Float32Array(bytes.buffer), T = {};
    for (const k of text.weights.order) { const t = text.weights.tensors[k], n = t.shape.reduce((a, b) => a * b, 1); T[k] = { d: all.subarray(t.offset, t.offset + n), shape: t.shape }; }
    const V = text.vocab.length, Cx = text.arch.context_len, dc = T.emb.shape[1], Hh = T.We1.shape[1], dh = T.We2.shape[1], H = T.W1.shape[1];
    const stoi = {}; text.vocab.forEach((c, i) => { stoi[c] = i; });
    const sig = (x) => 1 / (1 + Math.exp(-x)), silu = (x) => x * sig(x), dsilu = (x) => { const s = sig(x); return s * (1 + x * (1 - s)); };
    function lin(x, W, b, nin, nout) { const y = new Float32Array(nout); if (b) y.set(b.d); for (let i = 0; i < nin; i++) { const xi = x[i]; if (xi === 0) continue; const off = i * nout, Wd = W.d; for (let j = 0; j < nout; j++) y[j] += xi * Wd[off + j]; } return y; }
    function linT(g, W, nin, nout) { const x = new Float32Array(nin), Wd = W.d; for (let i = 0; i < nin; i++) { let s = 0; const off = i * nout; for (let j = 0; j < nout; j++) s += Wd[off + j] * g[j]; x[i] = s; } return x; }
    function encode(ids) { const x = new Float32Array(Cx * dc); ids.forEach((id, t) => { for (let k = 0; k < dc; k++) x[t * dc + k] = T.emb.d[id * dc + k]; }); return lin(lin(x, T.We1, T.be1, Cx * dc, Hh).map(silu), T.We2, T.be2, Hh, dh); }
    function softmax(y) { let m = -Infinity; for (const v of y) m = Math.max(m, v); const p = new Float32Array(y.length); let s = 0; for (let i = 0; i < y.length; i++) { p[i] = Math.exp(y[i] - m); s += p[i]; } for (let i = 0; i < y.length; i++) p[i] /= s; return p; }
    function energyGrad(h, yhat) {
      const p = softmax(yhat), e = lin(p, T.Wpe, null, V, dh), z = new Float32Array(3 * dh);
      for (let k = 0; k < dh; k++) { z[k] = h[k]; z[dh + k] = e[k]; z[2 * dh + k] = h[k] * e[k]; }
      const u1 = lin(z, T.W1, T.b1, 3 * dh, H), a1 = u1.map(silu), u2 = lin(a1, T.W2, T.b2, H, H), a2 = u2.map(silu);
      let E = T.b3.d[0]; for (let j = 0; j < H; j++) E += a2[j] * T.W3.d[j];
      const du2 = new Float32Array(H); for (let j = 0; j < H; j++) du2[j] = T.W3.d[j] * dsilu(u2[j]);
      const da1 = linT(du2, T.W2, H, H), du1 = new Float32Array(H); for (let j = 0; j < H; j++) du1[j] = da1[j] * dsilu(u1[j]);
      const dz = linT(du1, T.W1, 3 * dh, H), de = new Float32Array(dh); for (let k = 0; k < dh; k++) de[k] = dz[dh + k] + h[k] * dz[2 * dh + k];
      const dp = linT(de, T.Wpe, V, dh); let pd = 0; for (let v = 0; v < V; v++) pd += p[v] * dp[v];
      const grad = new Float32Array(V); for (let v = 0; v < V; v++) grad[v] = p[v] * (dp[v] - pd);
      return { E, grad, p };
    }
    const ids = (s) => { const out = []; for (const ch of s.slice(-Cx)) out.push(stoi[ch] !== undefined ? stoi[ch] : stoi['#']); while (out.length < Cx) out.unshift(stoi[' ']); return out; };
    return { V, Cx, stoi, vocab: text.vocab, encode, energyGrad, ids, alpha: (text.hparams && text.hparams.alpha0) || 10 };
  }
  // the toy's text normalization (src/text_prep.py), applied per character so offsets stay aligned
  const TRANS = { '’': "'", '‘': "'", '“': '"', '”': '"', '–': '-', '—': '-', '…': '...', ' ': ' ', '\t': ' ', '\r': '' };
  function normChar(ch, stoi) { let out = ''; for (const c0 of (TRANS[ch] !== undefined ? TRANS[ch] : ch)) { const c = c0.toLowerCase(); if (stoi[c] !== undefined) out += c; else if (c.charCodeAt(0) < 128) out += '#'; } return out; }

  EBT.panel({
    id: 'text-data',
    nav: 'Text data and perplexity',
    title: 'Text: the corpus, the targets, and perplexity',
    lede: 'Every language result in the paper rests on one pipeline: web text from RedPajama-V2, cut into GPT-NeoX tokens, turned into next-token targets, and scored by perplexity. Here it is on real documents, with a small model doing the scoring live.',
    text: `
      <p>All language models are pretrained from scratch on the RedPajama-V2 "100B sample" with the GPT-NeoX tokenizer, on a manual split of 66 million training and 33 thousand validation samples (p.8). The context is 256 tokens and the vocabulary has 50,277 entries (Tables D.2 to D.4). None of this is specific to EBTs. What changes is what the model does at each position.</p>
      <p>A Transformer++ maps the context to logits in one pass. An EBT keeps a candidate for the next token as an extra <em>input</em>, a full logit vector, and improves it by descending its own energy:</p>
      <div class="eq">$$\\hat y^{(i+1)} = \\hat y^{(i)} - \\alpha\\,\\nabla_{\\hat y}E_\\theta\\big(x_{\\le t},\\,\\hat y^{(i)}\\big)$$<span class="why">ŷ is the guess for token t+1, a logit vector in ℝ^50,277 that starts as ŷ⁽⁰⁾ ~ N(0, I) and is softmaxed and linearly projected before entering the Transformer (p.43–44). i counts thinking steps: 2 in pretraining (Table D.3), 2–3 in the thinking runs (Table D.4).</span></div>
      <p>Both models are trained with the same loss, the cross-entropy of the token that actually came next, $J=-\\ln p(x_{t+1})$. For the EBT, $p=\\mathrm{softmax}(\\hat y^{(N)})$, taken after the last step in the S2 models and after every step in the S1 models (p.30, p.36). Both are scored the same way too. The steps build that shared yardstick from raw text up.</p>`,
    steps: [
      { label: 'Web text becomes token ids', html: '<p>Pick a document. GPT-NeoX\'s byte-level BPE cuts it into pieces: a frequent word with its leading space is one token (<code>·the</code> is id 253), a rare word splits (<code>Enceladus</code> becomes <code>En</code> <code>cel</code> <code>ad</code> <code>us</code>). Hover or tap a chip for its id and raw BPE string. Over our 12 documents a token covers about 4.2 characters. Perplexity appears even before training: RedPajama-V2 sorts documents into quality buckets by the perplexity a Wikipedia-trained 5-gram model gives them (readout under the chips).</p>' },
      { label: 'Every position is a training example', html: '<p>A sequence of $S$ tokens holds $S$ training pairs: position $t$ reads $x_{\\le t}$ and is scored on $x_{t+1}$. The causal mask (triangle) lets all of them train in one pass. In the EBT each row also carries its own guess $\\hat y_{t+1}$, optimized for a few steps before the loss is taken, "enabling each prediction (e.g., a token for LLMs) to have its own thinking process" (p.5). Drag the position.</p>' },
      { label: 'Batches and token budgets', html: '<p>An optimizer step consumes a $B\\times S$ block of token ids. The paper changes $B$ and the number of steps from experiment to experiment (p.34); the figure multiplies them out. The longest RedPajama run, 1M steps of $128\\times256$ tokens, sees about 33B tokens, a third of the 100B sample; in the paper\'s words, the text models "see each sample only once due to the dataset size" (p.12). Video is the opposite regime (panel 18).</p>' },
      { label: 'Perplexity, computed live', html: '<p>A small character-level EBT (toy, trained for this explainer) scores the first 20 tokens in your browser. A token\'s probability is the product of its characters\' probabilities, so its loss is the sum of theirs: each bar is stacked from its characters (click one to open it). Drag $N$: every probability is read off $\\mathrm{softmax}(\\hat y^{(N)})$, so the EBT\'s perplexity drops as it thinks, most of it in the first two steps. A Transformer++ has no such dial. The toy is weak, though: a same-size one-pass network reaches 4.95 per character on held-out text, against 10.3 for the toy EBT at $N=3$. Read it for the mechanism, not for the paper\'s result. Then switch the unit to tokens.</p>' },
      { label: 'Four downstream benchmarks', html: '<p>GSM8K, SQuAD, BIG-bench Elementary Math QA and BIG-bench Dyck Languages, "ordered roughly by increasing perplexity difficulty" (p.9). They are scored by perplexity because these small models "do not achieve high accuracies" and "perplexity often functions as a more linear metric than accuracy" (p.9). Step through a Dyck example: the closing brackets need a stack, which a next-token predictor has to keep implicitly.</p>' },
      { label: 'Reading Table 3 honestly', html: '<p>EBT has the worse pretraining perplexity (33.43 vs 31.36, +6.6%) and still the better downstream perplexity on GSM8K (−12.7%), BB Math QA (−9.0%) and BB Dyck (−4.7%). On SQuAD it is slightly worse (+1.5%). The paper\'s claim is lower perplexity "on most downstream tasks" (p.12), not all, from one seed per model. Dividing by each model\'s own pretraining perplexity gives the paper\'s measure of distribution shift (p.11). These ratios do not reproduce the x-values of Fig 7, whose points are unlabeled, so do not map Fig 7 onto these benchmarks (derived).</p>' },
    ],
    after: `
      <h3>Perplexity, precisely</h3>
      <div class="eq">$$\\mathrm{PPL}=\\exp\\Big(-\\frac1T\\sum_{t=1}^{T}\\ln p_\\theta(x_t\\mid x_{\\lt t})\\Big)$$<span class="why">T scored tokens; p is the probability the model gave to the token that actually came next.</span></div>
      <p>Perplexity is the exponential of the mean cross-entropy, which makes it the geometric mean of $1/p$. A perplexity of 31.36 means the model is, on average, as unsure as a fair pick among about 31 tokens. Because of the logarithm, a single token given near-zero probability can dominate the average.</p>
      <p>The number also depends on the unit being predicted. A text has one total loss whether you divide it by tokens or by characters, so $\\mathrm{PPL}_{\\text{tok}}=\\mathrm{PPL}_{\\text{char}}^{\\,C/T}$, with $C/T$ the characters per token. At our 4.17 characters per token, Transformer++'s 31.36 is about 2.3 per character (derived; the paper's validation text may differ), against about 11 for the toy. The toy is far weaker than even the paper's smallest models, which is why its numbers are never set beside the paper's.</p>
      <p>For an EBT, $p=\\mathrm{softmax}(\\hat y^{(N)})$, so perplexity is a function of the thinking budget. Fig 6a plots how the mean perplexity increase on the four benchmarks falls as the EBT spends more forward passes, first on longer thinking and then on best-of-$M$ verification. Table 3 reports a single setting, and the paper does not say which.</p>
      <p class="note">Not specified in the paper: which tokens of each benchmark are scored (question and answer, or the answer only), the prompt format and split, how documents are cut into 256-token windows, whether Table 3's EBT numbers use extra thinking, and how the "best checkpoints" were chosen (p.9, p.34). All text evaluation is teacher-forced next-token scoring; no generated text is evaluated.</p>`,
    source: [{ kind: 'paper', note: 'p.8–9, Table 3, p.34, Tables D.2–D.4' }, { kind: 'ext', note: 'real documents, tokens, benchmark rows' }, { kind: 'toy', note: 'char-level EBT scores text (step 4)' }],
    figure(stage, ctx) {
      const { lib } = ctx, h = lib.h, C = lib.C;
      const S = lib.data('samples');
      if (!S || !S.text) { stage.appendChild(h('p', { class: 'callout warn' }, 'Sample data missing (data/samples.json).')); return {}; }
      const docs = S.text.docs, EV = S.eval || {};
      const model = makeCharEBT(lib.data('text'));
      const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
      const fmtN = (n) => Math.round(n).toLocaleString('en-US');
      const fmtTok = (n) => n >= 1e12 ? (n / 1e12).toFixed(n >= 1e13 ? 0 : 1) + 'T' : n >= 1e9 ? (n / 1e9).toFixed(n >= 1e10 ? 1 : 2) + 'B' : n >= 1e6 ? (n / 1e6).toFixed(0) + 'M' : fmtN(n);
      const fmtPpl = (v) => !isFinite(v) ? '–' : v >= 1e5 ? v.toExponential(1).replace('e+', 'e') : v >= 100 ? fmtN(v) : v.toFixed(2);
      const pct = (v) => (v > 0 ? '+' : v < 0 ? '−' : '') + Math.abs(v * 100).toFixed(1) + '%';
      const tokShow = (t) => t.replace(/\n/g, '↵').replace(/^ /, '·');
      const st = { doc: docs.find(d => d.id === 'rpv2_12') || docs[0], tv: 'text', hover: 5, pos: 6, exp: 'think', expIdx: { data: 13, batch: 3, params: 0 }, N: 3, mode: 'ebt', unit: 'char', sel: 13, bench: 'bb_dyck', ex: 1, dk: null, t3: 1, view: 'tokens' };
      const canv = [];

      // responsive canvas: logical width = CSS width so text stays at true pixel size on phones
      function rcanvas(parent, heightFor, label, draw) {
        const box = h('div', { class: 'canvas-box' }); parent.appendChild(box);
        const c = h('canvas', { role: 'img', 'aria-label': label }); box.appendChild(c);
        const g = c.getContext('2d'), R = { c, g, box, w: 0, h: 0, dpr: 1 };
        const paint = () => { if (!R.w) return; g.setTransform(R.dpr, 0, 0, R.dpr, 0, 0); g.fillStyle = '#fff'; g.fillRect(0, 0, R.w, R.h); draw(g, R.w, R.h); };
        R.resize = () => { const w = Math.round(box.clientWidth); if (!w) return; const hh = Math.round(heightFor(w)), d = Math.min(2, window.devicePixelRatio || 1); if (w !== R.w || hh !== R.h || d !== R.dpr) { c.width = Math.round(w * d); c.height = Math.round(hh * d); R.w = w; R.h = hh; R.dpr = d; } paint(); };
        let q = false; R.redraw = () => { if (q) return; q = true; requestAnimationFrame(() => { q = false; R.resize(); }); };
        R.toLocal = (ev) => { const r = c.getBoundingClientRect(), p = ev.touches ? ev.touches[0] : ev; return [(p.clientX - r.left) / r.width * R.w, (p.clientY - r.top) / r.height * R.h]; };
        if (window.ResizeObserver) new ResizeObserver(() => R.resize()).observe(box); else window.addEventListener('resize', R.resize);
        canv.push(R); return R;
      }
      const T = (g, s, x, y, o = {}) => lib.text(g, s, x, y, Object.assign({ size: 12, kind: 'mono', color: C.ink }, o));

      // ---------- pipeline strip + document picker ----------
      const NODES = [['raw', 'web text'], ['tok', 'tokens'], ['pairs', '(x, y) pairs'], ['batch', 'B × S batch'], ['model', 'p(next | x)'], ['ppl', 'perplexity'], ['bench', 'benchmarks']];
      const ACTIVE = [['raw', 'tok'], ['pairs'], ['batch'], ['model', 'ppl'], ['bench', 'ppl'], ['bench', 'ppl']];
      const pipe = h('ol', { class: 'td-pipe', 'aria-label': 'Pipeline' }, ...NODES.map(([k, l]) => h('li', { 'data-k': k }, h('span', {}, l))));
      stage.appendChild(pipe);
      const docSel = h('select', { id: 'td-doc', 'aria-label': 'Document' }, ...docs.map(d => h('option', { value: d.id }, `${d.id} · ${d.topic} · ${d.source_domain}`)));
      docSel.value = st.doc.id;
      const docRow = h('div', { class: 'td-docrow' }, h('label', { for: 'td-doc' }, 'document'), docSel);
      stage.appendChild(docRow);
      const viewWrap = h('div', {}); stage.appendChild(viewWrap);
      const views = {};
      const mkView = (k) => { const v = h('div', { class: 'td-view', hidden: true }); viewWrap.appendChild(v); views[k] = v; return v; };

      // =============== view 1: tokens ===============
      const v1 = mkView('tokens');
      const F1 = lib.frame(v1, { label: 'RedPajama-V2 document → GPT-NeoX tokens', sub: 'first 80 tokens · vocabulary 50,277 · hover a chip', pad: 12 });
      const chips = h('div', { class: 'td-chips', 'aria-label': 'Tokens of the document' }); F1.frame.appendChild(chips);
      const info = h('div', { class: 'readout td-info', 'aria-live': 'polite' }); v1.appendChild(info);
      const ruler = rcanvas(v1, () => 66, 'Where this document\'s token ids fall in the 50,277-entry vocabulary, log scale', drawRuler);
      const tvSeg = lib.segmented({ label: 'Chip display', options: [['text', 'text'], ['ids', 'text + id'], ['bpe', 'raw BPE']], value: st.tv, onchange: (v) => { st.tv = v; renderChips(); } });
      const stats1 = h('div', { class: 'readout' });
      v1.appendChild(h('div', { class: 'controls' }, h('span', { class: 'fig-label' }, 'chips'), tvSeg.el));
      v1.appendChild(stats1);
      const isCont = (i) => { const t = st.doc.tokens.text; return i > 0 && /^[A-Za-z0-9]/.test(t[i]) && /[A-Za-z0-9]$/.test(t[i - 1]); };
      function chipKids(t) {
        if (/^\n+$/.test(t)) return [h('span', { class: 'sp' }, '↵')];
        const m = t.match(/^( +)([\s\S]*)$/);
        return m ? [h('span', { class: 'sp' }, '·'.repeat(m[1].length)), m[2].replace(/\n/g, '↵')] : [t.replace(/\n/g, '↵')];
      }
      function renderChips() {
        chips.textContent = ''; const d = st.doc;
        d.tokens.text.forEach((t, i) => {
          const kids = st.tv === 'bpe' ? [h('span', {}, d.tokens.bpe[i])] : [h('span', {}, ...chipKids(t))];
          if (st.tv === 'ids') kids.push(h('span', { class: 'id' }, String(d.tokens.ids[i])));
          const b = h('button', { type: 'button', class: 'td-chip' + (isCont(i) ? ' cont' : '') + (i === st.hover ? ' sel' : ''), 'data-i': String(i), 'aria-label': `token ${i}: ${JSON.stringify(t)}, id ${d.tokens.ids[i]}` }, ...kids);
          b.addEventListener('mouseenter', () => inspect(i)); b.addEventListener('focus', () => inspect(i)); b.addEventListener('click', () => inspect(i));
          chips.appendChild(b);
          if (/\n$/.test(t)) chips.appendChild(h('span', { class: 'td-br' }));
        });
        renderStats(); inspect(st.hover);
      }
      function inspect(i) {
        st.hover = i; const d = st.doc;
        chips.querySelectorAll('.td-chip').forEach(b => b.classList.toggle('sel', +b.dataset.i === i));
        const t = d.tokens.text[i];
        info.innerHTML = `<span>token <b>#${i}</b></span><span>text <b>"${tokShow(t).replace(/</g, '&lt;')}"</b></span><span>BPE <b>${d.tokens.bpe[i].replace(/</g, '&lt;')}</b></span><span>id <b>${fmtN(d.tokens.ids[i])}</b></span><span>${isCont(i) ? 'continues a word' : (t.length > 1 && t[0] === ' ' ? 'word with its leading space' : '')}</span>`;
        ruler.redraw();
      }
      function renderStats() {
        const d = st.doc, txt = d.tokens.text; let words = 0, split = 0;
        for (let i = 0; i < txt.length; i++) { if (isCont(i) || !/[A-Za-z]/.test(txt[i])) continue; words++; let j = i + 1; while (j < txt.length && isCont(j)) j++; if (j > i + 1) split++; }
        const cpt = d.snippet.length / d.snippet_tokens_total;
        stats1.innerHTML = `<span><b>${fmtN(d.snippet.length)}</b> chars → <b>${d.snippet_tokens_total}</b> tokens</span><span><b>${cpt.toFixed(2)}</b> chars / token</span><span><b>${split}</b> of ${words} words split</span><span title="RedPajama-V2 sorts documents into head, middle and tail by the perplexity a Wikipedia-trained 5-gram model (CCNet) assigns them; head is the most Wikipedia-like">quality bucket <b>${d.bucket}</b> (5-gram perplexity ${fmtN(d.signals.ccnet_perplexity)})</span>`;
      }
      function drawRuler(g, w) {
        const x0 = 8, x1 = w - 8, y = 30, L = Math.log10(VOCAB), X = (id) => x0 + Math.log10(Math.max(1, id)) / L * (x1 - x0), d = st.doc;
        g.strokeStyle = C.ink; g.lineWidth = 1; g.beginPath(); g.moveTo(x0, y); g.lineTo(x1, y); g.stroke();
        [[1, '1'], [10, '10'], [100, '100'], [1000, '1k'], [10000, '10k'], [VOCAB, '50,277']].forEach(([v, s], k, a) => {
          g.beginPath(); g.moveTo(X(v), y); g.lineTo(X(v), y + 4); g.stroke();
          if (k === a.length - 2 && X(VOCAB) - X(v) < 70) return;
          T(g, s, X(v), y + 8, { size: 11, color: C.muted, align: k === 0 ? 'left' : k === a.length - 1 ? 'right' : 'center' });
        });
        g.strokeStyle = 'rgba(17,17,17,0.35)';
        d.tokens.ids.forEach(id => { g.beginPath(); g.moveTo(X(id), y - 8); g.lineTo(X(id), y); g.stroke(); });
        const id = d.tokens.ids[st.hover], xs = X(id);
        g.strokeStyle = C.blue; g.lineWidth = 2; g.beginPath(); g.moveTo(xs, y - 18); g.lineTo(xs, y); g.stroke();
        T(g, `"${tokShow(d.tokens.text[st.hover])}" = id ${fmtN(id)}`, clamp(xs, x0 + 70, x1 - 70), y - 30, { color: C.blue, align: 'center' });
        T(g, w < 520 ? 'token id, log scale (low = frequent)' : 'token id, log scale (low ids: single characters and the earliest, most frequent merges)', x0, y + 24, { size: 11, color: C.muted, kind: 'mono' });
      }

      // =============== view 2: pairs ===============
      const v2 = mkView('pairs');
      const sent = h('div', { class: 'td-sent', 'aria-live': 'polite' });
      const F2 = lib.frame(v2, { label: 'One sequence, one training pair per position', sub: 'row t reads tokens ≤ t and is scored on token t+1 · click a row' });
      const grid = rcanvas(F2.frame, (w) => gLay(w).H, 'Causal next-token layout: which tokens each position reads and which token it is scored on', drawGrid);
      grid.c.style.cursor = 'pointer';
      v2.appendChild(sent);
      const posSl = lib.slider({ id: 'td-pos', label: 'position t', min: 0, max: 78, step: 1, value: st.pos, oninput: (v) => { st.pos = v; updPairs(); } });
      v2.appendChild(h('div', { class: 'controls' }, posSl.el));
      const ebtBox = h('div', { class: 'td-ebtbox' }); v2.appendChild(ebtBox);
      function gLay(w) { const n = w >= 560 ? 8 : 6, LM = 50, cw = Math.floor((w - LM - 6) / (n + 1)), rh = Math.min(26, cw), HH = 44; return { n, LM, cw, rh, HH, H: HH + n * rh + 34 }; }
      const winStart = (n) => clamp(st.pos - (n - 3), 0, Math.max(0, st.doc.tokens.ids.length - 1 - n));
      function drawGrid(g, w) {
        const L = gLay(w), s = winStart(L.n), d = st.doc, t = d.tokens.text, NT = t.length;
        const maxc = Math.max(3, Math.floor((L.cw - 4) / 7.3));
        for (let c = 0; c <= L.n; c++) {
          const q = s + c; if (q >= NT) break; const x = L.LM + c * L.cw;
          const isT = q === st.pos + 1, isC = q <= st.pos;
          g.strokeStyle = isT ? C.ink : isC ? C.blue : C.rule; g.lineWidth = 1;
          if (isT) g.setLineDash([3, 2]); g.strokeRect(x + 1.5, 4.5, L.cw - 3, 22); g.setLineDash([]);
          let s0 = tokShow(t[q]); if (s0.length > maxc) s0 = s0.slice(0, maxc - 1) + '…';
          T(g, s0, x + L.cw / 2, 9, { align: 'center', color: isT ? C.ink : isC ? C.blue : C.muted, weight: isT ? 700 : 400 });
          T(g, String(d.tokens.ids[q]), x + L.cw / 2, 30, { size: 10, align: 'center', color: C.faint });
        }
        for (let r = 0; r < L.n; r++) {
          const p = s + r; if (p > NT - 2) break; const y = L.HH + r * L.rh, sel = p === st.pos;
          T(g, 't=' + p, L.LM - 6, y + L.rh / 2, { size: 11, align: 'right', baseline: 'middle', color: sel ? C.blue : C.muted, weight: sel ? 700 : 400 });
          for (let c = 0; c <= L.n; c++) {
            const q = s + c; if (q >= NT) break; const x = L.LM + c * L.cw;
            if (q <= p) { g.fillStyle = sel ? C.blue3 : 'rgba(17,17,17,0.08)'; g.fillRect(x + 1, y + 1, L.cw - 2, L.rh - 2); }
            else if (q === p + 1) {
              g.strokeStyle = sel ? C.ink : 'rgba(17,17,17,0.45)'; g.lineWidth = sel ? 1.4 : 1; g.setLineDash([3, 2]); g.strokeRect(x + 2.5, y + 2.5, L.cw - 5, L.rh - 5); g.setLineDash([]);
              const cx = x + L.cw / 2, cy = y + L.rh / 2, k = Math.min(5, L.rh / 4);
              g.beginPath(); g.moveTo(cx - k, cy); g.lineTo(cx + k, cy); g.moveTo(cx, cy - k); g.lineTo(cx, cy + k); g.stroke();
            }
          }
          if (sel) { g.strokeStyle = C.blue; g.lineWidth = 1.2; g.strokeRect(L.LM + 0.5, y + 0.5, Math.min(L.n + 1, NT - s) * L.cw - 1, L.rh - 1); }
        }
        const ly = L.HH + L.n * L.rh + 12;
        g.fillStyle = 'rgba(17,17,17,0.08)'; g.fillRect(L.LM, ly, 12, 12); T(g, 'read (context)', L.LM + 18, ly, { size: 11, color: C.muted });
        const x2 = L.LM + 18 + 14 * 7.3 + 14; g.strokeStyle = C.ink; g.setLineDash([3, 2]); g.strokeRect(x2 + 0.5, ly + 0.5, 11, 11); g.setLineDash([]);
        T(g, 'scored on (target)', x2 + 18, ly, { size: 11, color: C.muted });
      }
      grid.c.addEventListener('click', (ev) => { const [, y] = grid.toLocal(ev), L = gLay(grid.w), r = Math.floor((y - L.HH) / L.rh); if (r >= 0 && r < L.n) { st.pos = clamp(winStart(L.n) + r, 0, st.doc.tokens.ids.length - 2); posSl.set(st.pos); updPairs(); } });
      function updPairs() {
        const d = st.doc, t = d.tokens.text, p = st.pos, a = Math.max(0, p - 11);
        const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/\n/g, '↵');
        sent.innerHTML = `${a > 0 ? '<span class="f">…</span>' : ''}<span class="c">${esc(t.slice(a, p + 1).join(''))}</span><span class="t">${esc(t[p + 1])}</span><span class="f">${esc(t.slice(p + 2, p + 6).join(''))}…</span><br><span style="color:var(--muted)">position t = ${p} reads ${p + 1} token${p ? 's' : ''} and is scored on id ${fmtN(d.tokens.ids[p + 1])}; the next row reads one more.</span>`;
        ebtBox.innerHTML = `Transformer++ at row ${p}: one pass → 50,277 logits → loss −ln p(id ${d.tokens.ids[p + 1]}).<br>EBT at row ${p}: guess <b>ŷ</b> = 50,277 logits from N(0, I) → softmax → linear projector → its own input slot → energy <b>E</b> → step ŷ down ∇E (2 steps in pretraining) → loss −ln softmax(ŷ<sup>(N)</sup>)[id ${d.tokens.ids[p + 1]}].<br>A 256-token window gives 256 such rows at once; the EBT doubles the sequence to hold the guesses (2S, p.31).`;
        grid.redraw();
      }

      // =============== view 3: batches and budgets ===============
      const v3 = mkView('batch');
      const EXP = {
        data: { label: 'Fig 4a · data', model: 'xxs (6.18M)', S: 256, list: Array.from({ length: 14 }, (_, k) => ({ B: 128, steps: 15000 * (k + 1), tag: `checkpoint ${k + 1} of 14` })), note: 'Data scaling: xxs, batch 128 (p.34). The 14 points of Fig 4a sit 0.49B tokens apart, i.e. every 15,000 steps of 128 × 256 tokens [derived from the digitized figure].' },
        batch: { label: 'Fig 4b · batch', model: 'xxs (6.18M)', S: 256, list: [16, 32, 64, 128, 256].map(B => ({ B, steps: 105000, tag: `${fmtN(B * 256)} tokens per batch` })), note: 'Batch scaling: 105k steps each (p.34). Fig 4b\'s x-axis is batch size in tokens, 4K to 65K, i.e. 16 to 256 sequences of 256 tokens [derived].' },
        params: { label: 'Fig 5 · model size', model: '', S: 256, list: [['xxs', 6.18, 32], ['xs', 12.4, 46], ['small', 48.8, 90], ['medium', 176, 170], ['large', 396, 256]].map(([n, P, B]) => ({ B, steps: 105000, tag: `${n} · ${P}M non-embedding params` })), note: 'Parameter and FLOP scaling: 105k steps, batch 32, 46, 90, 170, 256 for xxs to large, "scaling the batch size with the square root of the number of parameters" (p.34).' },
        think: { label: 'Fig 6 · thinking runs', model: 'xxs (6.18M) S2-EBT and Transformer++', S: 256, list: [{ B: 128, steps: 1e6, tag: 'the longest text run' }], note: 'Thinking experiments: xxs models, batch 128, 1M steps (p.34). Figs 6, 7, B.1 and Table 3 use the best checkpoints of these two models.' },
        fineweb: { label: 'Fig B.3 · FineWeb', model: 'small (48.8M)', S: 1024, list: [{ B: 256, steps: 5e5, tag: 'a different corpus (FineWeb)' }], note: 'Larger-scale data scaling on FineWeb: small models, batch 256, context 1024, 500k steps (p.28).' },
      };
      const expSeg = lib.segmented({ label: 'Experiment', options: Object.entries(EXP).map(([k, e]) => [k, e.label]), value: st.exp, onchange: (v) => { st.exp = v; syncExpSlider(); updBatch(); } });
      v3.appendChild(h('div', { class: 'controls' }, expSeg.el));
      const F3 = lib.frame(v3, { label: 'One optimizer step', sub: 'a B × S block of token ids (texture: ids of our 12 documents, repeated)' });
      const bat = rcanvas(F3.frame, (w) => 330, 'A batch drawn as a matrix of token ids, and the total tokens seen on a log axis', drawBatch);
      const expSl = lib.slider({ id: 'td-exp', label: 'setting', min: 0, max: 13, step: 1, value: 13, fmt: (v) => String(v + 1), oninput: (v) => { st.expIdx[st.exp] = v; updBatch(); } });
      const bro = h('div', { class: 'readout' }), bnote = h('p', { class: 'note' });
      v3.appendChild(h('div', { class: 'controls' }, expSl.el)); v3.appendChild(bro); v3.appendChild(bnote);
      function syncExpSlider() { const e = EXP[st.exp], n = e.list.length; expSl.el.style.display = n > 1 ? '' : 'none'; expSl.input.max = String(n - 1); expSl.set(clamp(st.expIdx[st.exp] || 0, 0, n - 1)); }
      const curRun = () => { const e = EXP[st.exp]; return e.list[clamp(st.expIdx[st.exp] || 0, 0, e.list.length - 1)]; };
      let batchImg = null, batchKey = '';
      function batchBitmap(B, Sx) {
        const key = B + 'x' + Sx; if (key === batchKey) return batchImg;
        const cv = document.createElement('canvas'); cv.width = Sx; cv.height = B; const gg = cv.getContext('2d'), im = gg.createImageData(Sx, B), LV = Math.log(VOCAB + 1);
        for (let r = 0; r < B; r++) { const dd = docs[r % docs.length].tokens.ids, off = (r * 13) % dd.length; for (let c = 0; c < Sx; c++) { const id = dd[(off + c) % dd.length], v = Math.round(250 - 190 * Math.log(id + 1) / LV), i = 4 * (r * Sx + c); im.data[i] = v; im.data[i + 1] = v; im.data[i + 2] = v; im.data[i + 3] = 255; } }
        gg.putImageData(im, 0, 0); batchImg = cv; batchKey = key; return cv;
      }
      function drawBatch(g, w, hh) {
        const e = EXP[st.exp], r = curRun(), B = r.B, Sx = e.S, LM = 30, mx = LM, my = 26, mw = w - LM - 10, mhMax = 150;
        const mh = Math.max(16, Math.round(mhMax * Math.sqrt(B / 256)));
        g.imageSmoothingEnabled = false; g.drawImage(batchBitmap(B, Sx), mx, my, mw, mh); g.imageSmoothingEnabled = true;
        g.strokeStyle = C.ink; g.lineWidth = 1; g.strokeRect(mx + 0.5, my + 0.5, mw - 1, mh - 1);
        const rowH = mh / B; g.strokeStyle = C.blue; g.lineWidth = 1.5; g.strokeRect(mx + 0.5, my + 0.5, mw - 1, Math.max(2, rowH));
        T(g, `S = ${Sx} tokens per sequence (the context)`, mx, 8, { size: 11, color: C.muted });
        g.save(); g.translate(14, my + mh / 2); g.rotate(-Math.PI / 2); T(g, `B = ${B}`, 0, 0, { size: 11, align: 'center', baseline: 'middle', color: C.muted }); g.restore();
        T(g, 'blue outline = one row = one sequence', mx, my + mh + 6, { size: 11, color: C.blue });
        // budget axis
        const ay = my + mhMax + 76, ax0 = 22, ax1 = w - 22, lo = 8, hi = 13.4, X = (v) => ax0 + (Math.log10(v) - lo) / (hi - lo) * (ax1 - ax0);
        T(g, 'total tokens seen in training (log scale)', ax0, ay - 46, { size: 11, color: C.muted });
        g.strokeStyle = C.ink; g.lineWidth = 1; g.beginPath(); g.moveTo(ax0, ay); g.lineTo(ax1, ay); g.stroke();
        [[1e8, '100M'], [1e9, '1B'], [1e10, '10B'], [1e11, '100B'], [1e12, '1T'], [1e13, '10T']].forEach(([v, s]) => { g.beginPath(); g.moveTo(X(v), ay); g.lineTo(X(v), ay + 4); g.stroke(); T(g, s, X(v), ay + 8, { size: 11, color: C.muted, align: 'center' }); });
        const ref = (v, s, col, up) => { g.save(); g.strokeStyle = col; g.setLineDash([3, 3]); g.beginPath(); g.moveTo(X(v), ay - 34); g.lineTo(X(v), ay); g.stroke(); g.restore(); T(g, s, X(v) + (up ? -4 : 4), ay - 34, { size: 11, color: col, align: up ? 'right' : 'left' }); };
        ref(1e11, '100B sample', C.ink, true); if (w >= 420) ref(1.5e13, 'Llama 3: 15T', C.muted, true);
        Object.values(EXP).forEach(ee => ee.list.forEach(rr => { const tot = rr.B * ee.S * rr.steps; g.fillStyle = 'rgba(17,17,17,0.25)'; g.fillRect(X(tot) - 0.5, ay - 9, 1.5, 9); }));
        const tot = B * Sx * r.steps, xt = X(tot);
        lib.dot(g, xt, ay, 5, C.blue); T(g, fmtTok(tot), clamp(xt, ax0 + 20, ax1 - 20), ay + 26, { color: C.blue, align: 'center', weight: 700 });
      }
      function updBatch() {
        const e = EXP[st.exp], r = curRun(), tps = r.B * e.S, tot = tps * r.steps;
        const share = st.exp === 'fineweb' ? 'FineWeb, not RedPajama' : `${(tot / 1e11 * 100).toFixed(tot < 1e10 ? 1 : 0)}% of the 100B sample`;
        bro.innerHTML = `<span>${r.tag}${e.model ? ' · ' + e.model : ''}</span><span>tokens / step <b>${fmtN(tps)}</b></span><span>steps <b>${fmtN(r.steps)}</b></span><span>total <b>${fmtTok(tot)}</b></span><span><b>${share}</b></span>`;
        bnote.textContent = e.note; bat.redraw();
      }

      // =============== view 4: perplexity, scored live ===============
      const v4 = mkView('ppl');
      const TXT = lib.data('text'), VAL = (TXT && TXT.thinking_curve && TXT.thinking_curve.datasets && TXT.thinking_curve.datasets.val) || {};
      const CPT = ((S.text._meta || {}).chars_per_token) || 4.167;          // chars per GPT-NeoX token over our 12 documents
      const TPP_CHAR = Math.pow(31.36, 1 / CPT);                              // Table 3's 31.36 per token, re-expressed per character [derived]
      const TOY_BASE = VAL.baseline_ppl || 4.95, TOY_SYMS = model ? model.V : 54;
      const nSl = lib.slider({ id: 'td-n', label: 'thinking steps N', min: 0, max: 12, step: 1, value: st.N, oninput: (v) => { st.N = v; updPpl(); } });
      const modeSeg = lib.segmented({ label: 'Scoring model', options: [['ebt', 'toy EBT, N steps'], ['tpp', 'every p = 1/31.36'], ['uni', 'uniform 1/50,277']], value: st.mode, onchange: (v) => { st.mode = v; updPpl(); } });
      const unitSeg = lib.segmented({ label: 'Perplexity unit', options: [['char', 'per character'], ['tok', 'per token']], value: st.unit, onchange: (v) => { st.unit = v; updPpl(); } });
      v4.appendChild(h('div', { class: 'controls' }, nSl.el, modeSeg.el));
      v4.appendChild(h('div', { class: 'controls' }, h('span', { class: 'fig-label' }, 'perplexity per'), unitSeg.el));
      const F4 = lib.frame(v4, { label: 'Scoring the first 20 tokens · click a bar' });
      const pplC = rcanvas(F4.frame, (w) => pLay(w).H, 'Per-token loss bars split into characters, running perplexity with reference levels, and the characters of the selected token', drawPpl);
      pplC.c.style.cursor = 'pointer';
      const prog = h('div', { class: 'td-prog', 'aria-live': 'polite' }); v4.appendChild(prog);
      const pro = h('div', { class: 'readout' }); v4.appendChild(pro);
      const NMAX = 12;
      let score = null; // {docId, toks:[{text, chars:[{c, k}]}], str, nll[N][k], done}
      function prepScore() {
        if (!model) return null;
        const d = st.doc, toks = []; let s = '';
        const nT = Math.min(20, d.tokens.text.length);
        for (let i = 0; i < nT; i++) { const t = d.tokens.text[i], ch = []; for (const c of t) { const n = normChar(c, model.stoi); for (const cc of n) { ch.push({ c: cc, k: s.length }); s += cc; } } toks.push({ text: t, id: d.tokens.ids[i], chars: ch }); }
        return { docId: d.id, toks, str: s, nll: Array.from({ length: NMAX + 1 }, () => new Float32Array(s.length)), done: 0 };
      }
      let job = null, jobScore = null;
      function startScoring() {
        if (!model) { prog.textContent = 'Toy model weights missing (data/text.json); only the reference scorers work.'; return; }
        if (score && score.docId === st.doc.id && score.done >= score.str.length) return;
        if (!score || score.docId !== st.doc.id) score = prepScore();
        if (job && jobScore === score) return;
        if (job) { clearTimeout(job); job = null; }
        const sc = score; jobScore = sc;
        const work = () => {
          job = null; if (sc !== score) return;
          const t0 = performance.now();
          while (sc.done < sc.str.length && performance.now() - t0 < 8) {
            // one character: encode its 40-character context, draw ŷ⁽⁰⁾ ~ N(0, I), take NMAX gradient steps, record −ln p(true char) after every step
            const k = sc.done, trueId = model.stoi[sc.str[k]], hv = model.encode(model.ids(sc.str.slice(0, k)));
            const r = lib.rng(9001 + k * 7919); let y = new Float32Array(model.V); for (let v = 0; v < model.V; v++) y[v] = r.normal();
            for (let i = 0; i <= NMAX; i++) { const o = model.energyGrad(hv, y); sc.nll[i][k] = -Math.log(Math.max(1e-12, o.p[trueId])); if (i === NMAX) break; const nx = new Float32Array(model.V); for (let v = 0; v < model.V; v++) nx[v] = y[v] - model.alpha * o.grad[v]; y = nx; }
            sc.done++;
          }
          prog.textContent = sc.done < sc.str.length ? `scoring with the toy EBT in your browser… ${sc.done} / ${sc.str.length} characters` : `scored live: ${sc.str.length} characters × N = 0…${NMAX} (α = ${model.alpha}, ŷ⁽⁰⁾ ~ N(0, I) seeded per character)`;
          updPpl();
          if (sc.done < sc.str.length && st.view === 'ppl') job = setTimeout(work, 0);
        };
        job = setTimeout(work, 0);
      }
      // −ln p of one character / one token under the current scorer (NaN while not yet scored)
      const charNll = (c) => score && c.k < score.done ? score.nll[st.N][c.k] : NaN;
      function tokNll(tk) {
        if (st.mode === 'tpp') return Math.log(31.36);
        if (st.mode === 'uni') return Math.log(VOCAB);
        if (!score) return NaN; let s = 0;
        for (const c of tk.chars) { const v = charNll(c); if (!isFinite(v)) return NaN; s += v; }
        return s;
      }
      function pplStats() {
        const toks = score ? score.toks : [], nl = toks.map(tokNll); let s = 0, n = 0, cs = 0; const runTok = [], runChar = [];
        nl.forEach((v, i) => { if (!isFinite(v)) { runTok.push(NaN); runChar.push(NaN); return; } s += v; n++; cs += toks[i].chars.length; runTok.push(Math.exp(s / n)); runChar.push(cs ? Math.exp(s / cs) : NaN); });
        return { toks, nl, run: st.unit === 'tok' ? runTok : runChar, sum: s, n, chars: cs };
      }
      function pLay(w) { const LM = 46, RM = 8, y0 = 20, bh = 76, lab = 52, rh = 98, cy = y0 + bh + 6 + lab + 10 + rh + 4; return { LM, RM, y0, bh, lab, rh, cy, H: cy + 92 }; }
      function drawPpl(g, w) {
        if (!score) score = prepScore();
        const L = pLay(w), P = pplStats(), n = P.toks.length, bw = (w - L.LM - L.RM) / Math.max(1, n);
        const maxNl = Math.max(12, ...P.nl.filter(isFinite)), top = Math.ceil(maxNl / 4) * 4;
        const Yb = (v) => L.y0 + L.bh - v / top * L.bh, base = L.y0 + L.bh;
        g.strokeStyle = C.rule; g.lineWidth = 1;
        [0, top / 2, top].forEach(v => { g.beginPath(); g.moveTo(L.LM, Yb(v)); g.lineTo(w - L.RM, Yb(v)); g.stroke(); T(g, String(v), L.LM - 6, Yb(v), { size: 11, align: 'right', baseline: 'middle', color: C.muted }); });
        T(g, '−ln p, nats per token (stacked: its characters)', L.LM, 2, { size: 11, color: C.muted });
        P.nl.forEach((v, i) => {
          const x = L.LM + i * bw + bw * 0.16, ww = bw * 0.68, on = i === st.sel;
          if (!isFinite(v)) { g.strokeStyle = C.rule; g.setLineDash([2, 2]); g.strokeRect(x, Yb(top * 0.15), ww, base - Yb(top * 0.15)); g.setLineDash([]); return; }
          if (st.mode !== 'ebt') { g.fillStyle = on ? C.ink : 'rgba(17,17,17,0.45)'; g.fillRect(x, Yb(v), ww, base - Yb(v)); return; }
          // one segment per character: the token's loss is literally the sum of its characters' losses
          g.fillStyle = on ? C.blue : C.blue2; g.fillRect(x, Yb(v), ww, base - Yb(v));
          let acc = 0; g.strokeStyle = '#fff'; g.lineWidth = 1; P.toks[i].chars.forEach((c, j) => { acc += charNll(c); if (j < P.toks[i].chars.length - 1) { g.beginPath(); g.moveTo(x, Yb(acc)); g.lineTo(x + ww, Yb(acc)); g.stroke(); } });
        });
        const ly = base + 6;
        P.toks.forEach((tk, i) => {
          const x = L.LM + (i + 0.5) * bw; let s = tokShow(tk.text); if (s.length > 8) s = s.slice(0, 7) + '…';
          g.save(); g.translate(x + 3, ly); g.rotate(-Math.PI / 3.2); T(g, s, 0, 0, { size: 11, align: 'right', color: i === st.sel ? C.blue : C.ink, weight: i === st.sel ? 700 : 400 }); g.restore();
        });
        // running perplexity on a log axis, in the chosen unit, with reference levels in the same unit
        const ry = ly + L.lab + 12, rh = L.rh - 22, tok = st.unit === 'tok';
        const refs = tok ? [[31.36, 'Transformer++ 31.36 (Table 3)', C.ink, 'left'], [VOCAB, 'uniform over 50,277 tokens', C.muted, 'right']]
          : [[TPP_CHAR, `Transformer++ ≈ ${TPP_CHAR.toFixed(2)} (derived)`, C.ink, 'left'], [TOY_BASE, `toy one-pass baseline ${TOY_BASE.toFixed(2)} (held-out)`, C.muted, 'right'], [TOY_SYMS, `uniform over ${TOY_SYMS} symbols`, C.muted, 'right']];
        const vals = P.run.filter(isFinite), hiExp = Math.max(2, Math.ceil(Math.log10(Math.max(10, ...vals, ...refs.map(r => r[0]))) + 0.05)), Yr = (v) => ry + rh - Math.log10(Math.max(1, v)) / hiExp * rh;
        T(g, `perplexity ${tok ? 'per token' : 'per character'} of the text so far (log scale)`, L.LM, ry - 14, { size: 11, color: C.muted });
        const stepE = hiExp > 6 ? 2 : 1;
        for (let e = 0; e <= hiExp; e += stepE) { g.strokeStyle = C.rule; g.beginPath(); g.moveTo(L.LM, Yr(Math.pow(10, e))); g.lineTo(w - L.RM, Yr(Math.pow(10, e))); g.stroke(); T(g, e === 0 ? '1' : e < 4 ? String(Math.pow(10, e)) : '1e' + e, L.LM - 6, Yr(Math.pow(10, e)), { size: 11, align: 'right', baseline: 'middle', color: C.muted }); }
        refs.forEach(([v, s, col, al]) => {
          g.save(); g.strokeStyle = col; g.setLineDash([4, 3]); g.beginPath(); g.moveTo(L.LM, Yr(v)); g.lineTo(w - L.RM, Yr(v)); g.stroke(); g.restore();
          const tw = s.length * 6.7 + 6, tx = al === 'left' ? L.LM + 4 : w - L.RM - tw - 2; g.fillStyle = 'rgba(255,255,255,0.9)'; g.fillRect(tx, Yr(v) + 1, tw + 2, 13);
          T(g, s, al === 'left' ? tx + 2 : w - L.RM - 2, Yr(v) + 2, { size: 10.5, align: al, color: col });
        });
        const pts = []; P.run.forEach((v, i) => { if (isFinite(v)) pts.push([L.LM + (i + 0.5) * bw, Yr(v)]); });
        const lc = st.mode === 'ebt' ? C.blue : C.ink;
        if (pts.length > 1) lib.line(g, pts, { color: lc, width: 2 });
        pts.forEach((p, i) => lib.dot(g, p[0], p[1], i === pts.length - 1 ? 4 : 2.2, lc));
        g.strokeStyle = C.rule; g.beginPath(); g.moveTo(0, L.cy - 2); g.lineTo(w, L.cy - 2); g.stroke();
        const tk = P.toks[st.sel]; T(g, `inside "${tk ? tokShow(tk.text) : ''}": p(token) = product of its characters' p`, 6, L.cy + 3, { size: 11, color: C.muted, maxWidth: w - 12 });
        g.save(); g.translate(0, L.cy + 16); drawChars(g, w); g.restore();
      }
      function drawChars(g, w) {
        const P = pplStats(), tk = P.toks[st.sel];
        if (!tk) return;
        if (st.mode !== 'ebt') { T(g, st.mode === 'tpp' ? 'This reference scorer gives every true token p = 1/31.36: −ln p = 3.45 nats per token,' : 'This reference scorer spreads p evenly over all 50,277 tokens: −ln p = 10.83 nats per token,', 6, 10, { size: 12, color: C.muted, maxWidth: w - 12 }); T(g, 'so its per-token perplexity is that constant, whatever the text.', 6, 46, { size: 12, color: C.muted, maxWidth: w - 12 }); return; }
        if (!tk.chars.length) { T(g, 'This token has no characters the toy model can score.', 6, 10, { size: 12, color: C.muted }); return; }
        const n = tk.chars.length, cw = Math.min(64, (w - 12) / Math.max(n, 4)), x0 = 6, top = 14, bh = 28;
        const vals = tk.chars.map(charNll), mx = Math.max(4, ...vals.filter(isFinite));
        tk.chars.forEach((c, i) => {
          const v = vals[i], x = x0 + i * cw, y1 = top + bh;
          if (isFinite(v)) { g.fillStyle = C.blue2; const hh = v / mx * bh; g.fillRect(x + cw * 0.2, y1 - hh, cw * 0.6, hh); T(g, v.toFixed(1), x + cw / 2, y1 - hh - 13, { size: 10, align: 'center', color: C.muted }); }
          g.strokeStyle = C.rule; g.strokeRect(x + 1.5, y1 + 4.5, cw - 3, 22);
          T(g, c.c === ' ' ? '·' : c.c === '\n' ? '↵' : c.c, x + cw / 2, y1 + 9, { size: 13, align: 'center' });
          if (isFinite(v)) T(g, (cw >= 46 ? 'p ' : '') + Math.exp(-v).toFixed(2).replace(/^0/, ''), x + cw / 2, y1 + 31, { size: 10, align: 'center', color: C.muted });
        });
        const tot = vals.reduce((a, b) => a + b, 0), xr = x0 + n * cw + 12;
        if (isFinite(tot) && xr < w - 120) T(g, `Σ = ${tot.toFixed(2)} nats\np(token) = ${Math.exp(-tot).toExponential(1)}`, xr, top + 10, { size: 12, color: C.blue, lh: 1.5 });
      }
      pplC.c.addEventListener('click', (ev) => { const [x] = pplC.toLocal(ev), L = pLay(pplC.w), n = score ? score.toks.length : 0, bw = (pplC.w - L.LM - L.RM) / Math.max(1, n), i = Math.floor((x - L.LM) / bw); if (i >= 0 && i < n) { st.sel = i; updPpl(); } });
      const CAP4 = {
        char: 'Bars: −ln p per token; hairlines split it by character. Line: perplexity per character, the toy\'s unit. Dashed: uniform guess, the toy\'s one-pass baseline, and Transformer++\'s 31.36 per token at 4.17 characters per token (derived).',
        tok: 'Same losses divided by tokens instead of characters. The toy spreads probability over every string of characters, not only the 50,277 tokens, so a weak character model can land above the uniform-token line.',
      };
      function updPpl() {
        if (!score || score.docId !== st.doc.id) { score = prepScore(); if (st.view === 'ppl') startScoring(); }
        if (score) st.sel = clamp(st.sel, 0, score.toks.length - 1);
        const P = pplStats(), pplT = Math.exp(P.sum / P.n), pplCh = Math.exp(P.sum / P.chars), cpt = P.chars / P.n;
        nSl.el.style.opacity = st.mode === 'ebt' ? '1' : '0.4';
        let byN = '';
        if (st.mode === 'ebt' && score && score.done >= score.str.length) { const Lc = score.str.length; byN = '<span>per-char PPL by N: ' + [0, 1, 2, 3, 6, 12].map(n => { let t = 0; for (let k = 0; k < Lc; k++) t += score.nll[n][k]; return `${n} → <b${n === st.N ? '' : ' style="color:var(--ink2)"'}>${Math.exp(t / Lc).toFixed(1)}</b>`; }).join(' · ') + '</span>'; }
        pro.innerHTML = P.n ? `<span>Σ −ln p = <b>${P.sum.toFixed(1)}</b> nats over T = <b>${P.n}</b> tokens, C = <b>${P.chars}</b> characters</span><span>per token e<sup>${(P.sum / P.n).toFixed(2)}</sup> = <b>${fmtPpl(pplT)}</b></span><span>per character e<sup>${(P.sum / P.chars).toFixed(2)}</sup> = <b>${pplCh.toFixed(2)}</b></span><span>check ${pplCh.toFixed(2)}<sup>C/T = ${cpt.toFixed(2)}</sup> = ${fmtPpl(Math.pow(pplCh, cpt))}</span>` + byN : '<span>waiting for the first scores…</span>';
        if (st.view === 'ppl') ctx.setCaption(CAP4[st.unit]);
        pplC.redraw();
      }

      // =============== view 5: benchmarks ===============
      const v5 = mkView('bench');
      const BENCH = [
        { key: 'gsm8k', label: 'GSM8K', col: 1, what: 'grade-school math word problems with worked solutions (the &lt;&lt;…&gt;&gt; parts are calculator annotations in the raw data)' },
        { key: 'squad', label: 'SQuAD', col: 2, what: 'reading comprehension: the answer is a span of a Wikipedia paragraph' },
        { key: 'bb_elementary_math_qa', label: 'BB Math QA', col: 3, what: 'BIG-bench elementary math QA: arithmetic chains, nested calls and word problems, multiple choice' },
        { key: 'bb_dyck', label: 'BB Dyck', col: 4, what: 'BIG-bench Dyck languages: close every open bracket in the right order' },
      ];
      const bSeg = lib.segmented({ label: 'Benchmark', options: BENCH.map(b => [b.key, b.label]), value: st.bench, onchange: (v) => { st.bench = v; st.ex = 0; exSeg.set(0); st.dk = null; renderBench(); } });
      const exSeg = lib.segmented({ label: 'Example', options: [[0, 'ex 1'], [1, 'ex 2'], [2, 'ex 3']], value: st.ex, onchange: (v) => { st.ex = v; st.dk = null; renderBench(); } });
      v5.appendChild(h('div', { class: 'controls' }, bSeg.el, exSeg.el));
      const F5 = lib.frame(v5, { label: 'A real row of the benchmark', sub: '&nbsp;', pad: 12 });
      const exBox = h('div', { class: 'td-ex' }); F5.frame.appendChild(exBox);
      const dyckCtl = h('div', { class: 'controls' }); v5.appendChild(dyckCtl);
      const benchBars = rcanvas(v5, () => 92, 'Table 3 perplexity of this benchmark for Transformer++ and EBT', drawBenchBars);
      function renderBench() {
        const B = BENCH.find(b => b.key === st.bench), rows = EV[st.bench] || [], r = rows[st.ex] || rows[0];
        const src = ((EV._meta || {}).sources || {})[st.bench] || {};
        F5.wrap.querySelector('.fig-sub').innerHTML = `${src.dataset_id || ''}${src.config ? ' / ' + src.config : ''}${src.split ? ' / ' + src.split : ''} · row ${r ? r.row_idx : ''} · ${B.what}`;
        exBox.textContent = ''; dyckCtl.textContent = '';
        if (!r) { exBox.appendChild(h('p', {}, 'No rows for this benchmark in samples.json.')); return; }
        const field = (lab, ...kids) => h('div', {}, h('span', { class: 'lab' }, lab), ...kids);
        if (st.bench === 'gsm8k') {
          const sol = h('div', {});
          r.solution.split('\n').forEach(line => {
            if (line.startsWith('####')) { sol.appendChild(h('p', {}, 'final answer: ', h('b', {}, line.replace('####', '').trim()))); return; }
            const p = h('p', {}); line.split(/(<<[^>]*>>)/).forEach(part => p.appendChild(part.startsWith('<<') ? h('span', { class: 'calc' }, part) : document.createTextNode(part))); sol.appendChild(p);
          });
          exBox.append(field('question', h('p', {}, r.question)), field('reference solution', sol));
        } else if (st.bench === 'squad') {
          const ans = r.answers[0], k = r.context.indexOf(ans), p = h('p', {});
          if (k >= 0) p.append(r.context.slice(0, k), h('mark', {}, ans), r.context.slice(k + ans.length)); else p.textContent = r.context;
          exBox.append(field('passage · ' + r.title.replace(/_/g, ' '), p), field('question', h('p', {}, r.question)), field('reference answer', h('p', {}, h('b', {}, [...new Set(r.answers)].join(' / ')))));
        } else if (st.bench === 'bb_elementary_math_qa') {
          const parts = r.prompt.split(/\n choice:/), q = parts[0], ch = parts.slice(1).map(s => s.replace(/\nA:\s*$/, '').trim()), m = q.match(/^(.*?\?):\s*([\s\S]*)$/);
          exBox.append(field('instruction', h('p', {}, m ? m[1] : q)), m ? field('problem', h('p', { class: 'mono' }, m[2])) : null, field('choices (dashed = target)', h('div', {}, ...ch.map(c => h('span', { class: 'td-choice' + (c === r.target ? ' ok' : '') }, c)))));
        } else renderDyck(r, field);
        benchBars.redraw();
      }
      function renderDyck(r, field) {
        const toks = r.input_sequence.split(/\s+/).filter(Boolean);
        if (st.dk == null) st.dk = toks.length;
        const seq = h('div', { class: 'td-dyck', 'aria-label': 'Input brackets' }), stackEl = h('div', { class: 'td-stack' }), out = h('div', { class: 'readout' });
        exBox.append(field('prompt', h('p', { class: 'mono' }, 'Complete the rest of the sequence, making sure that the parentheses are closed properly.')), field('input, read left to right', seq), field('stack after reading (bottom → top)', stackEl), out);
        const ksl = lib.slider({ id: 'td-dk', label: 'symbols read', min: 0, max: toks.length, step: 1, value: st.dk, oninput: (v) => { st.dk = v; upd(); } });
        dyckCtl.appendChild(ksl.el);
        dyckCtl.appendChild(lib.button('read all', () => { st.dk = toks.length; ksl.set(st.dk); upd(); }));
        function upd() {
          const k = st.dk, stack = []; let err = false;
          for (let i = 0; i < k; i++) { const c = toks[i]; if (PAIRS[c]) stack.push(c); else if (stack.length && PAIRS[stack[stack.length - 1]] === c) stack.pop(); else err = true; }
          seq.textContent = ''; toks.forEach((c, i) => seq.appendChild(h('span', { class: 'td-br-chip' + (i < k ? ' done' : '') + (i === k - 1 ? ' cur' : '') }, c)));
          const need = stack.slice().reverse().map(o => PAIRS[o]);
          if (k === toks.length) need.forEach(c => seq.appendChild(h('span', { class: 'td-br-chip need' }, c)));
          stackEl.textContent = ''; if (!stack.length) stackEl.appendChild(h('span', {}, 'empty')); stack.forEach(c => stackEl.appendChild(h('span', { class: 'td-br-chip done' }, c)));
          const ok = need.join(' ') === r.target.trim();
          out.innerHTML = `<span>read <b>${k}/${toks.length}</b></span><span>open <b>${stack.length}</b></span>` + (k === toks.length ? `<span>closers needed <b>${need.join(' ') || '(none)'}</b></span><span>dataset target <b>${r.target}</b> ${ok ? 'matches' : 'differs'}</span>` : `<span>next symbol <b>${toks[k]}</b></span>`) + (err ? '<span style="color:var(--warn)">unbalanced</span>' : '');
        }
        upd();
      }
      function drawBenchBars(g, w) {
        const B = BENCH.find(b => b.key === st.bench), j = B.col, a = T3.tpp[j], b = T3.ebt[j], LM = 104, mx = 140, X = (v) => LM + v / mx * (w - LM - 70);
        T(g, `Table 3 perplexity on ${B.label} (lower is better)`, 0, 0, { size: 11, color: C.muted });
        [['Transformer++', a, C.ink], ['EBT', b, C.blue]].forEach(([nm, v, col], i) => {
          const y = 22 + i * 24; T(g, nm, LM - 8, y + 7, { size: 12, align: 'right', baseline: 'middle', color: col });
          g.fillStyle = col; g.fillRect(LM, y, X(v) - LM, 14); T(g, v.toFixed(1), X(v) + 6, y + 7, { size: 12, baseline: 'middle', color: col, weight: 700 });
        });
        const d = (b - a) / a; T(g, `EBT vs Transformer++: ${pct(d)} (${d < 0 ? 'better' : 'worse'})`, w < 520 ? 0 : LM, 74, { size: 12, color: d < 0 ? C.blue : C.bad });
      }

      // =============== view 6: Table 3 ===============
      const v6 = mkView('table');
      const F6 = lib.frame(v6, { label: 'Table 3 · perplexity, lower is better', sub: 'Transformer++ (ink) vs EBT (blue) · same training setup (p.11) · click a column' });
      const t3c = rcanvas(F6.frame, (w) => w < 520 ? 286 : 250, 'Table 3: pretraining and downstream perplexity for Transformer++ and EBT', drawT3);
      t3c.c.style.cursor = 'pointer';
      const t3tbl = h('div', { class: 'tbl td-t3' }); v6.appendChild(t3tbl);
      function t3Lay(w) { return { LM: 34, y0: 30, ph: w < 520 ? 150 : 140 }; }
      function drawT3(g, w) {
        const L = t3Lay(w), n = T3.cols.length, gw = (w - L.LM - 4) / n, Y = (v) => L.y0 + L.ph - v / 140 * L.ph;
        g.strokeStyle = C.rule; g.lineWidth = 1;
        [0, 35, 70, 105, 140].forEach(v => { g.beginPath(); g.moveTo(L.LM, Y(v)); g.lineTo(w - 4, Y(v)); g.stroke(); T(g, String(v), L.LM - 5, Y(v), { size: 11, align: 'right', baseline: 'middle', color: C.muted }); });
        T3.cols.forEach((c, j) => {
          const gx = L.LM + j * gw, on = j === st.t3, bw = Math.min(26, gw * 0.3), a = T3.tpp[j], b = T3.ebt[j], d = (b - a) / a;
          if (on) { g.fillStyle = C.blue4; g.fillRect(gx + 2, L.y0 - 26, gw - 4, L.ph + 26); }
          g.fillStyle = 'rgba(17,17,17,0.8)'; g.fillRect(gx + gw / 2 - bw - 2, Y(a), bw, L.y0 + L.ph - Y(a));
          g.fillStyle = C.blue; g.fillRect(gx + gw / 2 + 2, Y(b), bw, L.y0 + L.ph - Y(b));
          T(g, pct(d), gx + gw / 2, Math.min(Y(a), Y(b)) - 16, { size: 11, align: 'center', color: d < 0 ? C.blue : C.bad, weight: 700 });
          T(g, c, gx + gw / 2, L.y0 + L.ph + 6, { size: w < 520 ? 10 : 11, align: 'center', color: on ? C.ink : C.muted, maxWidth: gw - 2 });
          if (w < 520) { T(g, (a / T3.tpp[0]).toFixed(2), gx + gw / 2, L.y0 + L.ph + 36, { size: 10.5, align: 'center', color: C.ink }); T(g, (b / T3.ebt[0]).toFixed(2), gx + gw / 2, L.y0 + L.ph + 50, { size: 10.5, align: 'center', color: C.blue }); }
          else T(g, `÷pre ${(a / T3.tpp[0]).toFixed(2)} / ${(b / T3.ebt[0]).toFixed(2)}`, gx + gw / 2, L.y0 + L.ph + 24, { size: 10, align: 'center', color: C.muted, maxWidth: gw - 2 });
        });
        T(g, w < 520 ? 'above: EBT vs T++ (blue = EBT better) · below: ratio to own pretraining perplexity, T++ (ink) and EBT (blue)' : 'EBT vs T++ above each pair (blue = EBT better) · ÷pre = ratio to the model\'s own pretraining perplexity, T++ / EBT', w < 520 ? 4 : L.LM, w < 520 ? L.y0 + L.ph + 68 : L.y0 + L.ph + 46, { size: 10.5, color: C.muted, maxWidth: w - (w < 520 ? 8 : L.LM + 4) });
      }
      t3c.c.addEventListener('click', (ev) => { const [x] = t3c.toLocal(ev), L = t3Lay(t3c.w), gw = (t3c.w - L.LM - 4) / T3.cols.length, j = Math.floor((x - L.LM) / gw); if (j >= 0 && j < T3.cols.length) { st.t3 = j; updT3(); } });
      function updT3() {
        const cell = (v, j, cls) => `<td class="num${j === st.t3 ? ' b' : ''}${cls ? ' ' + cls : ''}">${v}</td>`;
        const row = (name, f) => `<tr><th scope="row">${name}</th>${T3.cols.map((_, j) => f(j)).join('')}</tr>`;
        t3tbl.innerHTML = `<table><thead><tr><th></th>${T3.cols.map((c, j) => `<th class="num"${j === st.t3 ? ' style="color:var(--blue)"' : ''}>${c}</th>`).join('')}</tr></thead><tbody>` +
          row('Transformer++', (j) => cell(T3.tpp[j].toFixed(2), j)) + row('EBT', (j) => cell(T3.ebt[j].toFixed(2), j)) +
          row('EBT vs T++', (j) => { const d = (T3.ebt[j] - T3.tpp[j]) / T3.tpp[j]; return cell(pct(d), j, d < 0 ? 'b' : 'w'); }) +
          row('Δ nats / token', (j) => { const d = Math.log(T3.ebt[j] / T3.tpp[j]); return cell((d > 0 ? '+' : '−') + Math.abs(d).toFixed(3), j); }) +
          row('geo-mean p, EBT', (j) => cell((100 / T3.ebt[j]).toFixed(2) + '%', j)) +
          row('÷ own pretrain, T++', (j) => cell((T3.tpp[j] / T3.tpp[0]).toFixed(2), j)) + row('÷ own pretrain, EBT', (j) => cell((T3.ebt[j] / T3.ebt[0]).toFixed(2), j)) +
          `</tbody></table>`;
        t3c.redraw();
      }

      // ---------- wiring ----------
      docSel.addEventListener('change', () => { st.doc = docs.find(d => d.id === docSel.value) || docs[0]; st.hover = Math.min(st.hover, st.doc.tokens.ids.length - 1); st.pos = Math.min(st.pos, st.doc.tokens.ids.length - 2); posSl.set(st.pos); renderChips(); updPairs(); score = null; st.sel = 0; updPpl(); });
      const CAPS = [
        'Real RedPajama-V2 text cut by the real GPT-NeoX tokenizer. Joined chips are pieces of one word; "·" is a leading space, "↵" a newline.',
        'Rows are positions, columns are tokens. Shaded: what a row reads. Dashed: the token it is scored on. All rows train at once.',
        'The batch matrix is drawn to scale in B and S. The axis shows the tokens a whole run sees; ticks mark every run in the selector.',
        CAP4.char,
        'A real row of each benchmark, with its Table 3 perplexities. The paper does not say which part of each row is scored.',
        'Table 3 (p.12) with derived conversions: relative difference, nats per token, geometric-mean p of the true token, ratio to own pretraining perplexity.',
      ];
      const VIEW_OF = ['tokens', 'pairs', 'batch', 'ppl', 'bench', 'table'];
      function go(i) {
        st.view = VIEW_OF[i];
        Object.entries(views).forEach(([k, v]) => { v.hidden = k !== st.view; });
        pipe.querySelectorAll('li').forEach(li => li.classList.toggle('on', ACTIVE[i].includes(li.dataset.k)));
        docRow.style.display = (i <= 1 || i === 3) ? '' : 'none';
        ctx.setCaption(CAPS[i]);
        if (st.view === 'ppl') startScoring();
        requestAnimationFrame(() => canv.forEach(R => { if (R.box.offsetParent) R.resize(); }));
      }
      renderChips(); updPairs(); syncExpSlider(); updBatch(); updPpl(); renderBench(); updT3();
      if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => canv.forEach(R => R.redraw()));
      go(0);
      return {
        step(i) { go(i); if (i === 1) { st.pos = 6; posSl.set(6); updPairs(); } if (i === 3) { st.mode = 'ebt'; modeSeg.set('ebt'); st.unit = 'char'; unitSeg.set('char'); updPpl(); } },
        show() { if (st.view === 'ppl') startScoring(); },
        hide() { if (job) { clearTimeout(job); job = null; } },
      };
    },
  });
})();
