/* Chapter "data": Data and evaluation.
   Five tabs: Pretraining text | Evaluation | Images | Video | Splits & setups.
   Provenance: real dataset samples from data/samples.json carry a "Real data" badge; paper numbers and figures carry
   the paper badge; our illustrations carry the ext badge. Live math in this chapter: BPE statistics, causal and EBT
   (Eq. 3) attention masks, perplexity from per-token probabilities, a unigram model, the Dyck bracket stack, the
   linear-beta noise schedule, patchification, Smooth L1 and its gradient. */
(function () {
  'use strict';
  const VOCAB = 50277;
  const PAIRS = { '(': ')', '[': ']', '{': '}', '<': '>' };
  // Paper Fig 11 (p.14): "Normalized Energy" per frame (16 rows) x iteration (12 cols), read off the figure by
  // inverting its reversed-viridis colormap at each cell centre (color error < 2/255). approx., read from Fig 11.
  const FIG11 = [
    [0.67, 0.37, 0.34, 0.34, 0.34, 0.34, 0.34, 0.34, 0.34, 0.34, 0.34, 0.34], [0.66, 0.38, 0.36, 0.36, 0.36, 0.36, 0.36, 0.36, 0.36, 0.36, 0.36, 0.36],
    [0.68, 0.39, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37], [0.77, 0.36, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.34, 0.34, 0.34, 0.34],
    [0.93, 0.21, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18], [1.0, 0.06, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
    [0.94, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.88, 0.14, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08],
    [0.91, 0.09, 0.03, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02], [0.86, 0.13, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08],
    [0.89, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05], [0.92, 0.09, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03],
    [0.86, 0.17, 0.11, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [0.86, 0.19, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14],
    [0.84, 0.16, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11], [0.84, 0.2, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13],
  ];
  // geometry of media/paper/fig11.png (550 x 508): heatmap and the frame-thumbnail column
  const F11 = { W: 550, H: 508, x0: 69, x1: 451, y0: 26, y1: 464, tx: 29, tw: 38 };

  EBT.section({
    id: 'data',
    nav: 'Data',
    kicker: 'Lab',
    title: 'Data and evaluation: what the models read, and how they are scored',
    lede: 'Real samples from every dataset the paper trains or tests on, pushed through the same preprocessing: web text becomes GPT-NeoX tokens and next-token pairs, benchmarks become perplexity, COCO images become noisy 16-pixel patches, and video frames become VAE latents. Each tab is a small bench you can step through.',
    mount,
  });

  function mount(el, lib) {
    const h = lib.h, C = lib.C;
    const S = lib.data('samples');
    if (!S) { el.appendChild(lib.callout('warn', 'Data missing', 'data/samples.json is not in data/bundle.js. Run <code>python3 src/bundle_data.py</code>.')); return; }
    const RM = !!lib.reducedMotion;
    const canvases = [];

    // ---------------- helpers ----------------
    const dataBadge = (label) => h('span', { class: 'badge data-badge', title: 'A real sample from a public dataset. Not a paper figure and not a model output.' }, 'Real data' + (label ? ' · ' + label : ''));
    const badgeRow = (...b) => h('div', { class: 'data-badges' }, ...b);
    const head = (title, ...b) => h('div', { class: 'data-head' }, h('h3', {}, title), badgeRow(...b));
    const small = (html) => h('p', { class: 'data-small', html });
    const fmtN = (n) => Math.round(n).toLocaleString('en-US');
    const fmtP = (p) => p >= 0.1 ? p.toFixed(2) : p >= 0.001 ? p.toFixed(3) : p.toExponential(1).replace('e-', 'e−');
    const fmtX = (v) => v >= 1000 ? fmtN(v) : v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v.toFixed(2);
    const signPct = (v, d = 1) => (v > 0 ? '+' : v < 0 ? '−' : '') + Math.abs(v * 100).toFixed(d) + '%';
    const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
    const add = (node, ...kids) => { node.append(...kids.filter(k => k != null && k !== false)); return node; };
    const pipe = (steps) => h('ol', { class: 'data-pipe', 'aria-label': 'Pipeline' }, ...steps.map(([a, b]) => h('li', {}, h('b', {}, a), h('span', {}, b))));
    const field = (label, ...kids) => h('div', { class: 'data-field' }, h('div', { class: 'data-flabel' }, label), ...kids);
    const tokShow = (t) => t === '\n' ? '↵' : t.replace(/\n/g, '↵').replace(/^ +/, (m) => '·'.repeat(m.length));
    function tokKids(t) { // chip content: leading spaces as faint dots, newlines as ↵
      if (t === '\n' || /^\n+$/.test(t)) return [h('span', { class: 'sp' }, '↵')];
      const m = t.match(/^( +)([\s\S]*)$/);
      return m ? [h('span', { class: 'sp' }, '·'.repeat(m[1].length)), m[2].replace(/\n/g, '↵')] : [t.replace(/\n/g, '↵')];
    }

    // responsive canvas: logical width = container CSS width, so text is drawn at true CSS px
    function rcanvas(parent, heightFor, label, draw) {
      const box = h('div', { class: 'canvas-box' }); parent.appendChild(box);
      const c = h('canvas', { role: 'img', 'aria-label': label }); box.appendChild(c);
      const ctx = c.getContext('2d');
      const st = { c, ctx, box, w: 0, h: 0 };
      let queued = false;
      function paint() { if (!st.w) return; const d = st.dpr; ctx.setTransform(d, 0, 0, d, 0, 0); ctx.clearRect(0, 0, st.w, st.h); draw(ctx, st.w, st.h); }
      st.resize = function () {
        const w = Math.round(box.clientWidth); if (!w) return;
        const hh = Math.max(20, Math.round(heightFor(w))); const d = Math.min(2, window.devicePixelRatio || 1);
        if (w !== st.w || hh !== st.h || d !== st.dpr) { c.width = Math.round(w * d); c.height = Math.round(hh * d); st.w = w; st.h = hh; st.dpr = d; }
        paint();
      };
      st.redraw = () => { if (queued) return; queued = true; requestAnimationFrame(() => { queued = false; paint(); }); };
      st.now = paint;
      st.toLocal = (ev) => { const r = c.getBoundingClientRect(); return [(ev.clientX - r.left) / r.width * st.w, (ev.clientY - r.top) / r.height * st.h]; };
      if (window.ResizeObserver) new ResizeObserver(() => st.resize()).observe(box); else window.addEventListener('resize', st.resize);
      canvases.push(st);
      return st;
    }
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => canvases.forEach(s => s.redraw()));
    const loadAll = (paths, st) => Promise.all(paths.map(p => lib.loadImg(p))).then(() => st && st.redraw());

    // Step / Play / Reset
    function player({ step, reset, canStep, interval = 0.9, label }) {
      const bStep = h('button', { type: 'button', class: 'btn' }, 'Step');
      const bPlay = h('button', { type: 'button', class: 'btn primary', 'aria-pressed': 'false' }, 'Play');
      const bReset = h('button', { type: 'button', class: 'btn' }, 'Reset');
      let acc = 0;
      const loop = lib.loop((dt) => {
        acc += dt; if (acc < interval) return true; acc = 0;
        if (!canStep()) { setPlay(false); return false; }
        step(); return true;
      });
      function setPlay(on) {
        if (on) { if (!canStep()) reset(); acc = interval * 0.6; loop.start(); } else loop.stop();
        bPlay.textContent = on ? 'Pause' : 'Play'; bPlay.setAttribute('aria-pressed', String(on));
      }
      bPlay.addEventListener('click', () => setPlay(!loop.running));
      bStep.addEventListener('click', () => { setPlay(false); if (!canStep()) reset(); else step(); });
      bReset.addEventListener('click', () => { setPlay(false); reset(); });
      return { el: h('div', { class: 'row data-player', role: 'group', 'aria-label': label }, bStep, bPlay, bReset), stop: () => setPlay(false), play: () => setPlay(true) };
    }
    // a short tween driver (for flying patches etc.)
    function tweener(onFrame) {
      let t = 1, dur = 0.35;
      const loop = lib.loop((dt) => { t = Math.min(1, t + dt / dur); onFrame(t); return t < 1; });
      return { start(d) { dur = d || 0.35; if (RM) { t = 1; onFrame(1); return; } t = 0; loop.start(); }, get t() { return t; }, stop() { loop.stop(); t = 1; } };
    }

    // ---------------- tabs ----------------
    const TABS = [
      ['text', 'Pretraining text', buildText],
      ['eval', 'Evaluation', buildEval],
      ['images', 'Images', buildImages],
      ['video', 'Video', buildVideo],
      ['setups', 'Splits & setups', buildSetups],
    ];
    el.appendChild(h('p', { class: 'data-note', html: 'Labels in this chapter: <span class="badge data-badge">Real data</span> is a real sample from a public dataset the paper uses (not a model output). <span class="badge paper">From the paper</span> marks the paper\'s numbers and figures, and <span class="badge ext">Beyond the paper</span> marks our illustrations.' }));
    const bar = h('div', { class: 'data-tabs', role: 'tablist', 'aria-label': 'Data and evaluation topics' });
    const panels = h('div', { class: 'data-panels' });
    el.appendChild(bar); el.appendChild(panels);
    const TS = {}; let cur = null;
    const btns = TABS.map(([id, label, build], i) => {
      const b = h('button', { type: 'button', role: 'tab', id: 'data-tab-' + id, 'aria-controls': 'data-panel-' + id, 'aria-selected': 'false', tabindex: '-1', class: 'data-tab' }, h('span', { class: 'n' }, String(i + 1)), label);
      const p = h('div', { role: 'tabpanel', id: 'data-panel-' + id, 'aria-labelledby': 'data-tab-' + id, class: 'data-panel', hidden: true });
      b.addEventListener('click', () => select(id));
      bar.appendChild(b); panels.appendChild(p);
      TS[id] = { b, p, build, inst: null, i };
      return b;
    });
    bar.addEventListener('keydown', (e) => {
      const i = btns.indexOf(document.activeElement); if (i < 0) return;
      const j = e.key === 'ArrowRight' ? (i + 1) % btns.length : e.key === 'ArrowLeft' ? (i - 1 + btns.length) % btns.length : e.key === 'Home' ? 0 : e.key === 'End' ? btns.length - 1 : null;
      if (j != null) { e.preventDefault(); btns[j].focus(); select(TABS[j][0]); }
    });
    function select(id, scroll) {
      if (cur === id) return;
      if (cur) { const s = TS[cur]; s.p.hidden = true; s.b.setAttribute('aria-selected', 'false'); s.b.tabIndex = -1; if (s.inst && s.inst.stop) s.inst.stop(); }
      cur = id; const s = TS[id]; s.p.hidden = false; s.b.setAttribute('aria-selected', 'true'); s.b.tabIndex = 0;
      if (!s.inst) {
        try { s.inst = s.build(s.p) || {}; } catch (e) { console.error(e); s.p.appendChild(lib.callout('warn', 'This tab failed to load', String(e.message))); s.inst = {}; }
        const nx = TABS[s.i + 1];
        if (nx) { const nb = h('button', { type: 'button', class: 'btn data-next' }, 'Next: ' + nx[1] + ' →'); nb.addEventListener('click', () => { select(nx[0]); TS[nx[0]].b.focus({ preventScroll: true }); bar.scrollIntoView({ behavior: RM ? 'auto' : 'smooth', block: 'start' }); }); s.p.appendChild(h('div', { class: 'row' }, nb)); }
      }
      canvases.forEach(c => { if (s.p.contains(c.box)) c.resize(); });
    }
    select('text');

    // =====================================================================
    // 1. Pretraining text
    // =====================================================================
    function buildText(P) {
      const T = S.text, docs = T.docs;
      P.appendChild(h('div', { class: 'prose' }, h('p', { html: 'Every language model in the paper is pretrained from scratch on <b>RedPajama-V2</b> web text (the 100B sample on Hugging Face), cut into tokens by the <b>GPT-NeoX</b> tokenizer and trained to predict the next token (p.8). The context is 256 tokens and the vocabulary has 50,277 entries (Table D.2). The authors split the corpus into 66 million training and 33 thousand validation samples (p.8). Below are 12 real documents from the same corpus, split by the real tokenizer.' })));
      P.appendChild(pipe([['Raw web text', 'RedPajama-V2'], ['BPE tokens', 'GPT-NeoX, 50,277'], ['Context', '256 tokens'], ['Target', 'the next token'], ['Loss', 'cross-entropy'], ['Score', 'perplexity']]));

      let doc = docs.find(d => d.id === 'rpv2_01') || docs[0];
      let view = 'tokens', pos = 3, mode = 'ebt', inspected = 4;
      const N = () => doc.tokens.ids.length;

      // ---- bench 1: tokenizer ----
      const sel = h('select', { id: 'data-doc' }, ...docs.map(d => h('option', { value: d.id }, `${d.topic} · ${d.source_domain}`)));
      sel.value = doc.id;
      const meta = h('div', { class: 'data-meta' });
      const stats = h('div', { class: 'readout' });
      const viewSeg = lib.segmented({ options: [['tokens', 'Tokens'], ['ids', 'Token ids'], ['bpe', 'Raw BPE']], value: view, label: 'Token display', onchange: (v) => { view = v; renderChips(); } });
      const ctl1 = h('div', { class: 'card stack' },
        h('label', { for: 'data-doc', class: 'data-h4' }, 'Document'), sel, meta,
        h('div', { class: 'data-h4' }, 'Show'), viewSeg.el, stats,
        small('A token is usually a word with its leading space (<code>·looks</code>), a piece of a rarer word (<code>En</code> <code>cel</code> <code>ad</code> <code>us</code>), or punctuation. Pieces of one word are drawn joined, with an amber edge. <b>Click any token</b> to make it the target in the bench below.'));
      const chips = h('div', { class: 'data-chips', 'aria-label': 'Tokens of the document' });
      const rest = h('p', { class: 'data-rest' });
      const insp = h('div', { class: 'readout data-insp', 'aria-live': 'polite' });
      const ins1 = h('div', { class: 'card stack' }, head('Real GPT-NeoX tokens', dataBadge('RedPajama-V2'), dataBadge('tokenizer EleutherAI/gpt-neox-20b')), chips, rest, insp);
      const ruler = rcanvas(ins1, () => 70, 'Where the token ids of this document fall in the 50,277-entry vocabulary, log scale', drawRuler);
      ins1.appendChild(small('Ids after the byte symbols are numbered in BPE merge order, and early merges are usually frequent strings. So <code>·the</code> has id 253 while a rare word like <code>·Mercedes</code> sits at 35,512.'));
      P.appendChild(h('div', { class: 'bench' }, ctl1, ins1));

      function isCont(i) { const t = doc.tokens.text; return i > 0 && /^[A-Za-z0-9]/.test(t[i]) && /[A-Za-z0-9]$/.test(t[i - 1]); }
      function renderChips() {
        chips.textContent = '';
        doc.tokens.text.forEach((t, i) => {
          const kids = view === 'bpe' ? [h('span', { class: 't' }, doc.tokens.bpe[i])] : [h('span', { class: 't' }, ...tokKids(t))];
          if (view === 'ids') kids.push(h('span', { class: 'id' }, String(doc.tokens.ids[i])));
          const b = h('button', { type: 'button', class: 'data-tok' + (isCont(i) ? ' cont' : ''), 'data-i': String(i), 'aria-label': `token ${i}, ${JSON.stringify(t)}, id ${doc.tokens.ids[i]}` }, ...kids);
          b.addEventListener('mouseenter', () => inspect(i));
          b.addEventListener('focus', () => inspect(i));
          b.addEventListener('click', () => { setPos(Math.max(0, i - 1)); inspect(i); });
          chips.appendChild(b);
          if (/\n$/.test(t)) chips.appendChild(h('span', { class: 'data-br', 'aria-hidden': 'true' }));
        });
        marks();
      }
      function marks() {
        chips.querySelectorAll('.data-tok').forEach(b => { const i = +b.dataset.i; b.classList.toggle('ctx', i <= pos); b.classList.toggle('tgt', i === pos + 1); b.classList.toggle('sel', i === inspected); });
      }
      function inspect(i) {
        inspected = i; marks();
        const t = doc.tokens.text[i], o = doc.tokens.char_offsets[i];
        insp.innerHTML = '';
        insp.append(h('span', {}, 'token ', h('b', {}, '#' + i)), h('span', {}, 'text ', h('b', {}, '"' + tokShow(t) + '"')), h('span', {}, 'BPE ', h('b', {}, doc.tokens.bpe[i])), h('span', {}, 'id ', h('b', {}, fmtN(doc.tokens.ids[i]))), h('span', {}, 'chars ', h('b', {}, o[0] + '–' + o[1])), h('span', {}, isCont(i) ? 'piece of a longer word' : i === pos + 1 ? 'current target' : ''));
        ruler.redraw();
      }
      function drawRuler(ctx, w, hh) {
        const x0 = 14, x1 = w - 14, y = 30, L = Math.log10(VOCAB);
        const X = (id) => x0 + Math.log10(Math.max(1, id)) / L * (x1 - x0);
        ctx.strokeStyle = C.faint; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x1, y); ctx.stroke();
        [[1, '1'], [10, '10'], [100, '100'], [1000, '1k'], [10000, '10k'], [VOCAB, '50,277']].forEach(([v, s], k, arr) => {
          ctx.beginPath(); ctx.moveTo(X(v), y - 4); ctx.lineTo(X(v), y + 4); ctx.stroke();
          if (k === arr.length - 2 && X(VOCAB) - X(v) < 80) return; // keep '10k' clear of '50,277' on phones
          lib.text(ctx, s, X(v), y + 10, { size: 13, kind: 'mono', color: C.faint, align: k === 0 ? 'left' : k === arr.length - 1 ? 'right' : 'center' });
        });
        lib.text(ctx, 'token id (log scale)', x0, 2, { size: 13, color: C.muted });
        ctx.strokeStyle = lib.rgba(C.ar, 0.55); ctx.lineWidth = 1.5;
        doc.tokens.ids.forEach(id => { ctx.beginPath(); ctx.moveTo(X(id), y - 9); ctx.lineTo(X(id), y); ctx.stroke(); });
        const id = doc.tokens.ids[inspected]; const xs = X(id);
        ctx.strokeStyle = C.ebt; ctx.lineWidth = 3; ctx.beginPath(); ctx.moveTo(xs, y - 16); ctx.lineTo(xs, y + 2); ctx.stroke();
        lib.text(ctx, '"' + tokShow(doc.tokens.text[inspected]) + '" = ' + fmtN(id), clamp(xs, x0 + 60, x1 - 60), y - 30, { size: 13, kind: 'mono', color: C.ebt, align: 'center' });
      }
      function renderMeta() {
        const s = doc.signals || {};
        meta.innerHTML = '';
        meta.append(h('p', { class: 'data-doctitle' }, doc.title),
          h('div', { class: 'readout' }, h('span', {}, 'CC snapshot ', h('b', {}, doc.snapshot)), h('span', {}, 'bucket ', h('b', {}, doc.bucket)), h('span', {}, 'CCNet ppl ', h('b', {}, String(s.ccnet_perplexity))), h('span', {}, 'words ', h('b', {}, fmtN(s.rps_doc_word_count || 0)))),
          small('Bucket and CCNet perplexity are quality signals shipped with RedPajama-V2: the perplexity of a Wikipedia-trained n-gram model on this page (lower = more Wikipedia-like, "head").'));
        const txt = doc.tokens.text; let words = 0, split = 0;
        for (let i = 0; i < txt.length; i++) {
          if (isCont(i) || !/[A-Za-z]/.test(txt[i])) continue;
          words++; let j = i + 1; while (j < txt.length && isCont(j)) j++; if (j > i + 1) split++;
        }
        const cpt = doc.snippet.length / doc.snippet_tokens_total;
        stats.innerHTML = '';
        stats.append(h('span', {}, h('b', {}, fmtN(doc.snippet.length)), ' chars → ', h('b', {}, String(doc.snippet_tokens_total)), ' tokens'), h('span', {}, h('b', {}, cpt.toFixed(2)), ' chars per token'), h('span', {}, h('b', {}, split + ' of ' + words), ' words split into pieces (first 80 tokens)'));
        const shown = txt.join('');
        const tail = doc.snippet.slice(shown.length);
        rest.textContent = '';
        if (tail) rest.append(h('span', { class: 'data-flabel' }, 'Rest of the 600-character snippet (not tokenized here): '), tail.length > 260 ? tail.slice(0, 260) + '…' : tail);
      }
      sel.addEventListener('change', () => { doc = docs.find(d => d.id === sel.value); pl.stop(); posSl.input.max = String(N() - 2); renderMeta(); renderChips(); setPos(Math.min(pos, N() - 2)); inspect(pos + 1); });

      // ---- bench 2: next-token pairs and attention layout ----
      const posSl = lib.slider({ id: 'data-pos', label: 'Position i (target = token i+1)', min: 0, max: N() - 2, step: 1, value: pos, oninput: (v) => { pl.stop(); setPos(v, true); } });
      const pl = player({ label: 'Step through next-token training pairs', interval: 1.1, canStep: () => pos < N() - 2, step: () => setPos(pos + 1), reset: () => setPos(0) });
      const modeSeg = lib.segmented({ options: [['tpp', 'Transformer++'], ['ebt', 'EBT']], value: mode, label: 'Model whose attention layout is shown', onchange: (v) => { mode = v; updPairs(); } });
      const expl = h('div', { class: 'data-small stack' });
      const ctl2 = h('div', { class: 'card stack' }, h('div', { class: 'data-h4' }, 'Training pairs'), posSl.el, pl.el, h('div', { class: 'data-h4' }, 'Attention layout of one row'), modeSeg.el, expl);
      const pairLine = h('div', { class: 'data-pair', 'aria-live': 'polite' });
      const ins2 = h('div', { class: 'card stack' }, head('One document gives one training example per position', dataBadge('tokens above'), lib.badge('paper', 'Eq. 3, p.31')), pairLine);
      const mat = rcanvas(ins2, (w) => matLayout(w).H, 'Attention mask: which tokens each prediction may look at', drawMat);
      const legend = h('div', { class: 'legend' });
      const ro2 = h('div', { class: 'readout' });
      ins2.append(legend, ro2);
      P.appendChild(h('div', { class: 'bench' }, ctl2, ins2));

      function matLayout(w) {
        const narrow = w < 560; const LM = narrow ? 70 : 104, TM = narrow ? 62 : 74, n = narrow ? 10 : 16;
        const cs = Math.max(14, Math.min(28, Math.floor((w - LM - 8) / (n + 1))));
        return { LM, TM, n, cs, H: TM + n * cs + 8 };
      }
      function win(n) { return clamp(pos - (n - 3), 0, Math.max(0, N() - 1 - n)); }
      function drawMat(ctx, w) {
        const L = matLayout(w), n = L.n, cs = L.cs, s = win(n), ebt = mode === 'ebt';
        const t = doc.tokens.text, trunc = (x, k) => x.length > k ? x.slice(0, k - 1) + '…' : x;
        for (let r = 0; r < n; r++) {
          const p = s + r; if (p > N() - 2) break; const hl = p === pos;
          for (let c = 0; c <= n; c++) {
            const q = s + c; if (q > N() - 1) continue;
            const x = L.LM + c * cs, y = L.TM + r * cs;
            let fill;
            if (q <= p) fill = lib.rgba(C.ar, hl ? 0.9 : 0.32);
            else if (q === p + 1 && ebt) fill = lib.rgba(C.ebt, hl ? 1 : 0.42);
            else fill = lib.rgba(C.rule, hl ? 0.9 : 0.45);
            ctx.fillStyle = fill; ctx.fillRect(x + 1, y + 1, cs - 2, cs - 2);
            if (q === p + 1 && ebt && cs >= 17) lib.text(ctx, 'ŷ', x + cs / 2, y + cs / 2 + 1, { size: Math.min(15, cs - 4), kind: 'mono', color: '#1b1407', align: 'center', baseline: 'middle', weight: 700 });
            if (q === p + 1 && hl && !ebt) { ctx.save(); ctx.strokeStyle = C.truth; ctx.lineWidth = 2; ctx.setLineDash([4, 3]); ctx.strokeRect(x + 2, y + 2, cs - 4, cs - 4); ctx.restore(); lib.text(ctx, '✕', x + cs / 2, y + cs / 2 + 1, { size: Math.min(14, cs - 6), color: C.truth, align: 'center', baseline: 'middle' }); }
          }
          const lab = ebt ? 'ŷ' + (p + 1) : trunc(tokShow(t[p]), L.LM > 90 ? 11 : 8);
          lib.text(ctx, lab, L.LM - 6, L.TM + r * cs + cs / 2, { size: 13, kind: 'mono', color: hl ? (ebt ? C.ebt : C.ink) : C.muted, align: 'right', baseline: 'middle', weight: hl ? 700 : 400 });
        }
        // row outline
        if (pos >= s && pos < s + n) { ctx.strokeStyle = ebt ? C.ebt : C.ink; ctx.lineWidth = 1.5; ctx.strokeRect(L.LM - 0.5, L.TM + (pos - s) * cs - 0.5, Math.min(n + 1, N() - s) * cs + 1, cs + 1); }
        // column labels (rotated)
        for (let c = 0; c <= n; c++) {
          const q = s + c; if (q > N() - 1) break;
          ctx.save(); ctx.translate(L.LM + c * cs + cs / 2 + 4, L.TM - 6); ctx.rotate(-Math.PI / 3.2);
          lib.text(ctx, trunc(tokShow(t[q]), 9), 0, 0, { size: 13, kind: 'mono', color: q === pos + 1 ? C.truth : q <= pos ? C.ink : C.faint, baseline: 'middle', weight: q === pos + 1 ? 700 : 400 });
          ctx.restore();
        }
      }
      mat.c.addEventListener('click', (ev) => { const [, y] = mat.toLocal(ev); const L = matLayout(mat.w); const r = Math.floor((y - L.TM) / L.cs); if (r >= 0 && r < L.n) { pl.stop(); setPos(clamp(win(L.n) + r, 0, N() - 2)); } });
      mat.c.style.cursor = 'pointer';

      function updPairs() {
        const t = doc.tokens.text, ids = doc.tokens.ids, ebt = mode === 'ebt';
        pairLine.textContent = '';
        const a = Math.max(0, pos - 7);
        pairLine.appendChild(h('span', { class: 'data-flabel' }, 'context'));
        if (a > 0) pairLine.appendChild(h('span', { class: 'data-ell' }, '…'));
        for (let i = a; i <= pos; i++) pairLine.appendChild(h('span', { class: 'data-tok ctx sm' }, h('span', { class: 't' }, ...tokKids(t[i]))));
        pairLine.appendChild(h('span', { class: 'data-arrow' }, '→'));
        if (ebt) pairLine.append(h('span', { class: 'data-yhat' }, 'ŷ: 50,277 logits'), h('span', { class: 'data-arrow' }, '≈'));
        pairLine.appendChild(h('span', { class: 'data-tok tgt sm' }, h('span', { class: 't' }, ...tokKids(t[pos + 1])), h('span', { class: 'id' }, 'id ' + fmtN(ids[pos + 1]))));
        expl.innerHTML = ebt
          ? '<p><b>EBT.</b> The guess ŷ for token i+1 is an <i>input</i>. It starts as 50,277 random logits (N(0, I)), is softmaxed and projected to an embedding, and gets its own slot. Row i of Eq. 3 lets that slot see tokens 0 to i and itself, never the real token i+1, which would leak the answer.</p><p>The model outputs one energy for the slot. Gradient steps on ŷ lower it (2 steps in pretraining, Table D.3), then cross-entropy scores softmax(ŷ<sub>N</sub>) against the target.</p>'
          : '<p><b>Transformer++.</b> The ordinary causal mask: row i sees tokens 0 to i. Its hidden state goes through the output head to 50,277 logits, and the loss is −ln p(target). One forward pass per token, however hard the token is.</p><p>The target cell (✕) is masked: the true next token is never an input.</p>';
        legend.innerHTML = '';
        legend.append(h('span', {}, h('i', { style: `background:${lib.rgba(C.ar, 0.9)}` }), 'observed token (visible)'),
          ebt ? h('span', {}, h('i', { style: `background:${C.ebt}` }), 'prediction slot ŷ (its own input)') : h('span', {}, h('i', { style: `border:2px dashed ${C.truth}` }), 'target (masked)'),
          h('span', {}, h('i', { style: `background:${lib.rgba(C.rule, 0.9)}` }), 'masked future'));
        ro2.innerHTML = '';
        ro2.append(h('span', {}, 'row ', h('b', {}, 'i = ' + pos)), h('span', {}, 'context ', h('b', {}, (pos + 1) + ' tokens')), h('span', {}, 'target id ', h('b', {}, fmtN(ids[pos + 1]))),
          h('span', {}, 'input length for a 256-token context: ', h('b', {}, ebt ? '2N = 512 (observed + predicted, p.31)' : 'N = 256')));
        mat.redraw();
      }
      function setPos(v, fromSlider) {
        pos = clamp(Math.round(v), 0, N() - 2);
        if (!fromSlider) posSl.set(pos);
        inspected = pos + 1; marks(); inspect(pos + 1); updPairs();
      }
      renderMeta(); renderChips(); setPos(pos);
      P.appendChild(lib.callout('insight', 'Why this matters for EBTs', 'Every position of every sequence is one prediction, and all of them train in parallel. A Transformer++ spends one forward pass on each. An EBT runs a short optimization on each, with the candidate fed back in as an input. That is why it needs the 2N layout and the special mask of Eq. 3 (p.31–32), and why each EBT training step costs about 3.33× a Transformer++ step (p.36).'));
      return { stop: () => pl.stop() };
    }

    // =====================================================================
    // 2. Evaluation
    // =====================================================================
    function buildEval(P) {
      const E = S.eval, T3 = E.paper_table3, src = (E._meta && E._meta.sources) || {};
      P.appendChild(h('div', { class: 'prose' }, h('p', { html: 'Language models are scored on the RedPajama-V2 validation split and on four downstream benchmarks chosen to lean on reasoning, "ordered roughly by increasing perplexity difficulty" (p.9). These small models trained from scratch get near-zero accuracy, so every score is <b>perplexity</b>, where lower is better (p.9). The paper does not say exactly how each example is formatted or which part is scored, so below are the raw dataset fields.' })));
      P.appendChild(pipe([['Benchmark text', '4 datasets'], ['Model', 'p(next token)'], ['Per token', '−ln p'], ['Average', 'mean NLL'], ['Perplexity', 'e^(mean NLL)']]));

      const BENCH = [
        { key: 'gsm8k', label: 'GSM8K', col: 1, what: 'Grade-school math word problems with worked solutions. The <code>&lt;&lt;…&gt;&gt;</code> parts are calculator annotations in the raw data.' },
        { key: 'squad', label: 'SQuAD', col: 2, what: 'Reading comprehension: the answer is a span of a Wikipedia paragraph.' },
        { key: 'bb_elementary_math_qa', label: 'BB Math QA', col: 3, what: 'BIG-bench elementary math QA: arithmetic chains, nested function calls and word problems, as multiple choice.' },
        { key: 'bb_dyck', label: 'BB Dyck', col: 4, what: 'BIG-bench Dyck languages: close every open bracket in the right order. You have to keep a stack in your head. Step through it on the right.' },
      ];
      let bk = 'bb_dyck', ex = 1, subPlayer = null;
      // ---- bench 1: benchmarks ----
      const bSeg = lib.segmented({ options: BENCH.map(b => [b.key, b.label]), value: bk, label: 'Benchmark', onchange: (v) => { bk = v; ex = 0; exSeg.set(0); renderEx(); } });
      const exSeg = lib.segmented({ options: [[0, 'Example 1'], [1, 'Example 2'], [2, 'Example 3']], value: ex, label: 'Example', onchange: (v) => { ex = v; renderEx(); } });
      const what = h('div', { class: 'stack' });
      const ctl1 = h('div', { class: 'card stack' }, h('div', { class: 'data-h4' }, 'Benchmark'), bSeg.el, exSeg.el, what);
      const exBox = h('div', { class: 'stack' });
      const exHead = h('div', {});
      const ins1 = h('div', { class: 'card stack' }, exHead, exBox);
      P.appendChild(h('div', { class: 'bench data-bench-wrap' }, ctl1, ins1));

      function renderEx() {
        if (subPlayer) { subPlayer.stop(); subPlayer = null; }
        const B = BENCH.find(b => b.key === bk), rows = E[bk] || [], r = rows[ex] || rows[0];
        const s = src[bk] || {};
        exHead.textContent = '';
        exHead.appendChild(head(B.label + ' · row ' + r.row_idx, dataBadge(`${s.dataset_id || ''}${s.config ? ' / ' + s.config : ''}${s.split ? ' / ' + s.split : ''}`)));
        const tp = T3['Transformer++'][B.col], te = T3.EBT[B.col];
        what.innerHTML = '';
        add(what, small(B.what),
          h('div', { class: 'data-h4' }, 'Table 3 perplexity, lower is better'), lib.badge('paper', 'Table 3, p.12'),
          h('div', { class: 'data-kv' }, h('span', { class: 'k' }, 'Transformer++'), h('b', { style: `color:var(--ar)` }, tp.toFixed(1)),
            h('span', { class: 'k' }, 'EBT'), h('b', { style: `color:var(--ebt)` }, te.toFixed(1)),
            h('span', { class: 'k' }, 'EBT vs T++'), h('b', {}, signPct((te - tp) / tp) + (te < tp ? ' (better)' : ' (worse)'))),
          s.note ? small(s.note) : null);
        exBox.textContent = '';
        if (bk === 'gsm8k') exBox.appendChild(renderGSM(r));
        else if (bk === 'squad') exBox.appendChild(renderSquad(r));
        else if (bk === 'bb_elementary_math_qa') exBox.appendChild(renderMath(r));
        else exBox.appendChild(renderDyck(r));
      }
      function renderGSM(r) {
        const sol = h('div', { class: 'data-sol' });
        r.solution.split('\n').forEach(line => {
          if (line.startsWith('####')) { sol.appendChild(h('p', { class: 'data-final' }, 'Final answer: ', h('b', {}, line.replace('####', '').trim()))); return; }
          const p = h('p', {});
          line.split(/(<<[^>]*>>)/).forEach(part => p.appendChild(part.startsWith('<<') ? h('span', { class: 'data-calc', title: 'calculator annotation in the raw GSM8K data' }, part) : document.createTextNode(part)));
          sol.appendChild(p);
        });
        return h('div', { class: 'stack' }, field('Question', h('p', {}, r.question)), field('Reference solution', sol));
      }
      function renderSquad(r) {
        const ans = r.answers[0], k = r.context.indexOf(ans);
        const ctxEl = h('p', { class: 'data-ctxp' });
        if (k >= 0) ctxEl.append(r.context.slice(0, k), h('mark', { class: 'data-ans' }, ans), r.context.slice(k + ans.length)); else ctxEl.textContent = r.context;
        return h('div', { class: 'stack' }, field('Passage · ' + r.title.replace(/_/g, ' '), ctxEl), field('Question', h('p', {}, r.question)), field('Reference answer', h('p', { class: 'data-final' }, [...new Set(r.answers)].join(' / '))));
      }
      function renderMath(r) {
        const parts = r.prompt.split(/\n choice:/);
        const q = parts[0];
        const choices = parts.slice(1).map(s => s.replace(/\nA:\s*$/, '').trim());
        const qm = q.match(/^(.*?\?):\s*([\s\S]*)$/);
        return h('div', { class: 'stack' },
          field('Instruction', h('p', {}, qm ? qm[1] : q)),
          qm ? field('Problem', h('p', { class: 'mono data-prob' }, qm[2])) : null,
          field('Choices (the target is highlighted)', h('div', { class: 'row' }, ...choices.map(c => h('span', { class: 'data-choice' + (c === r.target ? ' ok' : '') }, c)))),
          small('The raw prompt ends with <code>A:</code>, and the target string is <code>' + r.target + '</code>.'));
      }
      function renderDyck(r) {
        const toks = r.input_sequence.split(/\s+/).filter(Boolean);
        let k = toks.length;
        const seqEl = h('div', { class: 'data-dyck', 'aria-label': 'Input brackets' });
        const stEl = h('div', { class: 'data-dyck' });
        const outEl = h('div', { class: 'readout', 'aria-live': 'polite' });
        const pl = player({ label: 'Step through the bracket stack', interval: 0.55, canStep: () => k < toks.length, step: () => { k++; upd(); }, reset: () => { k = 0; upd(); } });
        subPlayer = pl;
        function upd() {
          const stack = []; let err = false;
          for (let i = 0; i < k; i++) { const c = toks[i]; if (PAIRS[c]) stack.push(c); else if (stack.length && PAIRS[stack[stack.length - 1]] === c) stack.pop(); else err = true; }
          seqEl.textContent = '';
          let depth = 0;
          toks.forEach((c, i) => {
            const open = !!PAIRS[c]; if (!open) depth--;
            const d = Math.max(0, depth);
            seqEl.appendChild(h('span', { class: 'data-br-chip d' + (d % 4) + (i < k ? ' done' : '') + (i === k - 1 ? ' cur' : '') }, c));
            if (open) depth++;
          });
          stEl.textContent = '';
          stEl.appendChild(h('span', { class: 'data-flabel' }, 'stack (bottom → top)'));
          if (!stack.length) stEl.appendChild(h('span', { class: 'data-small' }, 'empty'));
          stack.forEach((c, i) => stEl.appendChild(h('span', { class: 'data-br-chip done d' + (i % 4) }, c)));
          const completion = stack.slice().reverse().map(o => PAIRS[o]).join(' ');
          outEl.innerHTML = '';
          outEl.append(h('span', {}, 'read ', h('b', {}, k + ' of ' + toks.length), ' symbols'), h('span', {}, 'still open ', h('b', {}, String(stack.length))));
          if (k === toks.length) {
            const ok = completion === r.target.trim();
            outEl.append(h('span', {}, 'needed closing sequence ', h('b', { style: 'color:var(--truth)' }, completion || '(none)')), h('span', {}, 'dataset target ', h('b', {}, r.target), ok ? ' ✓ matches' : ' ✗ differs'));
          } else outEl.append(h('span', {}, 'next symbol ', h('b', {}, toks[k])));
          if (err) outEl.append(h('span', { style: 'color:var(--bad)' }, 'unbalanced input'));
        }
        upd();
        return h('div', { class: 'stack' }, field('Raw prompt', h('pre', { class: 'data-pre' }, r.prompt)), field('Input, colored by nesting depth', seqEl), stEl, pl.el, outEl,
          small('Popping the stack in reverse gives exactly the closers the dataset expects. A language model has to do this implicitly, token by token, which is why the paper treats Dyck as a reasoning-flavored, far out-of-distribution test.'));
      }
      renderEx();

      // ---- bench 2: perplexity widget ----
      const demo = S.text.pairs_demo, pairs = demo.pairs;
      const targets = pairs.map(p => ({ text: p.target_text, id: p.target_id }));
      const T = targets.length;
      // hand-set illustration (not a model output)
      const ILLUS_MAP = { ' Thursday': 0.004, ' afternoon': 0.1, ' I': 0.08, ' received': 0.03, ' a': 0.35, ' che': 0.0015, 'ery': 0.8, ' phone': 0.06, ' call': 0.85, ' from': 0.55, ' the': 0.35, ' Mercedes': 0.0006, ' dealership': 0.2, ' in': 0.15, ' town': 0.04, '.': 0.3 };
      const ILLUS = targets.map(t => ILLUS_MAP[t.text] != null ? ILLUS_MAP[t.text] : 0.1);
      // a real (tiny) unigram model: counts of GPT-NeoX ids in the OTHER 11 documents, mixed 90/10 with uniform
      const counts = new Map(); let total = 0;
      S.text.docs.forEach(d => { if (d.id === demo.doc) return; d.tokens.ids.forEach(id => { counts.set(id, (counts.get(id) || 0) + 1); total++; }); });
      const UNI = targets.map(t => 0.9 * (counts.get(t.id) || 0) / total + 0.1 / VOCAB);
      const PRESETS = {
        illus: { p: ILLUS, note: 'Illustration: probabilities set by hand to look like a decent small model (easy function words high, rare words low). No model is run here; the arithmetic is exact.' },
        unigram: { p: UNI, note: `A real but tiny model: how often each token id appears in the other 11 documents (${fmtN(total)} tokens), mixed 90/10 with a uniform guess so unseen tokens keep p = 0.1/50,277. It ignores context, so it is terrible.` },
        uniform: { p: targets.map(() => 1 / VOCAB), note: 'A uniform guess over all 50,277 tokens. Perplexity equals the vocabulary size: the worst sensible model.' },
        tpp: { p: targets.map(() => 1 / 31.36), note: 'Every target gets p = 1/31.36. Perplexity is then exactly 31.36, the Transformer++ pretraining perplexity in Table 3. Real models are uneven, but the geometric mean is what counts.' },
        surprise: { p: targets.map(t => t.text === ' Mercedes' ? 1e-5 : 0.5), note: 'Every token gets p = 0.5 except one at 0.00001. That single surprise adds 11.5 nats and multiplies perplexity by about 2 (= e^(11.5 − 0.69)/16).' },
      };
      let probs = ILLUS.slice(), kk = T, preset = 'illus';
      const presetSeg = lib.segmented({ options: [['illus', 'Illustration'], ['unigram', 'Unigram'], ['uniform', 'Uniform'], ['tpp', '1/31.36'], ['surprise', 'One surprise']], value: preset, label: 'Probability preset', onchange: (v) => { preset = v; probs = PRESETS[v].p.slice(); syncSliders(); upd(); } });
      const presetNote = h('p', { class: 'data-small' });
      const pl2 = player({ label: 'Accumulate the per-token loss', interval: 0.7, canStep: () => kk < T, step: () => { kk++; upd(); }, reset: () => { kk = 0; upd(); } });
      const big = h('div', { class: 'data-big' });
      const formula = h('div', { class: 'data-formula' });
      const ctl2 = h('div', { class: 'card stack' }, h('div', { class: 'data-h4' }, 'Probabilities of the true next token'), presetSeg.el, presetNote,
        h('div', { class: 'data-h4' }, 'Add up the loss one token at a time'), pl2.el, big, formula,
        small('Drag any slider to change how much probability the model gives the true token. Perplexity is e to the mean negative log-likelihood: the geometric mean of 1/p.'));
      const rowsBox = h('div', { class: 'data-prows', role: 'group', 'aria-label': 'Per-token probability sliders' });
      rowsBox.appendChild(h('div', { class: 'data-pr hd', 'aria-hidden': 'true' }, h('span', { class: 'tk' }, 'target token'), h('span', { class: 'sl' }, 'p(target | context), log scale'), h('span', { class: 'p' }, 'p'), h('span', { class: 'bar' }, '−ln p'), h('span', { class: 'n' }, 'nats')));
      const rows = targets.map((t, i) => {
        const sl = h('input', { type: 'range', min: -5, max: 0, step: 0.01, value: Math.log10(probs[i]), 'aria-label': `probability of token ${JSON.stringify(t.text)}` });
        const pO = h('output', { class: 'p' }), bi = h('i', {}), nO = h('output', { class: 'n' });
        const row = h('div', { class: 'data-pr' }, h('span', { class: 'tk' }, h('span', { class: 'data-tok sm tgt' }, h('span', { class: 't' }, ...tokKids(t.text)))), h('span', { class: 'sl' }, sl), pO, h('span', { class: 'bar' }, bi), nO);
        sl.addEventListener('input', () => { probs[i] = Math.pow(10, +sl.value); preset = null; presetSeg.set(null); upd(); });
        rowsBox.appendChild(row);
        return { row, sl, pO, bi, nO };
      });
      const ctxLine = h('p', { class: 'data-small' }, 'Text: “' + demo.pairs[T - 1].context_text + '.” (', h('code', {}, demo.doc), '). Each row is one next-token prediction; its context is everything before it.');
      const ins2 = h('div', { class: 'card stack' }, head('How perplexity is computed', dataBadge('real GPT-NeoX tokens'), lib.badge('ext', 'probabilities: illustration or tiny unigram')), ctxLine, rowsBox);
      const chart = rcanvas(ins2, (w) => w < 560 ? 230 : 250, 'Per-token surprise 1/p as bars and running perplexity as a line, log scale', drawPpl);
      const choices = rcanvas(ins2, (w) => w < 560 ? 150 : 96, 'Perplexity as an equivalent number of equally likely choices', drawChoices);
      P.appendChild(h('div', { class: 'bench' }, ctl2, ins2));

      function syncSliders() { rows.forEach((r, i) => { r.sl.value = String(Math.log10(probs[i])); }); }
      function stats() { const inc = probs.slice(0, kk); const sum = inc.reduce((a, p) => a - Math.log(p), 0); const mean = kk ? sum / kk : NaN; return { sum, mean, ppl: Math.exp(mean) }; }
      function upd() {
        const st = stats();
        rows.forEach((r, i) => {
          const p = probs[i], nl = -Math.log(p);
          r.pO.textContent = fmtP(p); r.nO.textContent = nl.toFixed(2);
          r.bi.style.width = (nl / Math.log(1e5) * 100).toFixed(1) + '%';
          r.row.classList.toggle('off', i >= kk); r.row.classList.toggle('cur', i === kk - 1 && kk < T);
        });
        presetNote.textContent = preset ? PRESETS[preset].note : 'Custom probabilities (you moved a slider).';
        big.innerHTML = kk ? `<span>perplexity</span><b>${fmtX(st.ppl)}</b>` : '<span>perplexity</span><b>–</b>';
        formula.innerHTML = kk
          ? `Σ −ln p over <b>${kk}</b> token${kk > 1 ? 's' : ''} = <b>${st.sum.toFixed(2)}</b> nats<br>mean = ${st.sum.toFixed(2)} / ${kk} = <b>${st.mean.toFixed(3)}</b> nats (${(st.mean / Math.LN2).toFixed(2)} bits) per token<br>perplexity = e<sup>${st.mean.toFixed(3)}</sup> = <b>${fmtX(st.ppl)}</b>`
          : 'Press Step to add the first token.';
        chart.redraw(); choices.redraw();
      }
      function drawPpl(ctx, w, hh) {
        const narrow = w < 560;
        const ax = { x: narrow ? 44 : 56, y: 26, w: w - (narrow ? 44 : 56) - (narrow ? 8 : 116), h: hh - 26 - (narrow ? 34 : 40) };
        const Y = (v) => ax.y + ax.h - Math.log10(clamp(v, 1, 1e5)) / 5 * ax.h;
        const bw = ax.w / T;
        ctx.strokeStyle = C.rule; ctx.lineWidth = 1;
        [1, 10, 100, 1000, 1e4, 1e5].forEach(v => { ctx.beginPath(); ctx.moveTo(ax.x, Y(v)); ctx.lineTo(ax.x + ax.w, Y(v)); ctx.stroke(); lib.text(ctx, v >= 1000 ? (v / 1000) + 'k' : String(v), ax.x - 6, Y(v), { size: 13, kind: 'mono', color: C.faint, align: 'right', baseline: 'middle' }); });
        lib.text(ctx, narrow ? '1/p (bars), running perplexity (line)' : '1/p per token (bars) and running perplexity (line), log scale', narrow ? 0 : ax.x, 4, { size: 13, color: C.muted });
        // reference lines (paper Table 3 + uniform)
        [[31.36, 'T++ 31.36', C.ar], [VOCAB, 'uniform 50,277', C.faint]].forEach(([v, s, col]) => {
          ctx.save(); ctx.strokeStyle = col; ctx.setLineDash([5, 4]); ctx.lineWidth = 1.2; ctx.beginPath(); ctx.moveTo(ax.x, Y(v)); ctx.lineTo(ax.x + ax.w, Y(v)); ctx.stroke(); ctx.restore();
          if (!narrow) lib.text(ctx, s, ax.x + ax.w + 6, Y(v), { size: 13, kind: 'mono', color: col, baseline: 'middle' });
          else lib.text(ctx, s, ax.x + ax.w - 2, Y(v) - 9, { size: 12, kind: 'mono', color: col, align: 'right', baseline: 'middle' });
        });
        // bars
        probs.forEach((p, i) => {
          const x = ax.x + i * bw + bw * 0.18, y = Y(1 / p), on = i < kk;
          ctx.fillStyle = on ? lib.rgba(C.diff, i === kk - 1 ? 0.95 : 0.6) : lib.rgba(C.faint, 0.25);
          ctx.fillRect(x, y, bw * 0.64, ax.y + ax.h - y);
        });
        // running perplexity line
        let s = 0; const pts = [];
        for (let i = 0; i < kk; i++) { s -= Math.log(probs[i]); pts.push([ax.x + (i + 0.5) * bw, Y(Math.exp(s / (i + 1)))]); }
        if (pts.length > 1) lib.line(ctx, pts, { color: C.ebt, width: 2.5 });
        pts.forEach(([x, y], i) => lib.dot(ctx, x, y, i === pts.length - 1 ? 5 : 3, C.ebt));
        if (pts.length && !narrow) { const [x, y] = pts[pts.length - 1]; lib.text(ctx, 'ppl ' + fmtX(Math.exp(s / kk)), x, y - 20, { size: 13, kind: 'mono', color: C.ebt, align: 'center', baseline: 'middle' }); }
        // x labels
        targets.forEach((t, i) => {
          const x = ax.x + (i + 0.5) * bw;
          if (narrow) { if (i % 3 === 0 || i === T - 1) lib.text(ctx, String(i + 1), x, ax.y + ax.h + 6, { size: 12, kind: 'mono', color: C.faint, align: 'center' }); }
          else { ctx.save(); ctx.translate(x, ax.y + ax.h + 6); ctx.rotate(-0.5); lib.text(ctx, tokShow(t.text).slice(0, 9), 0, 0, { size: 12, kind: 'mono', color: i < kk ? C.muted : C.faint, align: 'right', baseline: 'top' }); ctx.restore(); }
        });
        if (narrow) lib.text(ctx, 'token number', ax.x + ax.w / 2, hh - 14, { size: 12, color: C.faint, align: 'center' });
      }
      function drawChoices(ctx, w, hh) {
        const st = stats(); if (!kk) { lib.text(ctx, 'No tokens added yet.', 0, 8, { size: 14, color: C.muted }); return; }
        const n = Math.max(1, Math.round(st.ppl)), shown = Math.min(n, 2400), narrow = w < 560;
        const msg = `As unsure as a fair pick among ${fmtX(st.ppl)} tokens` + (n > shown ? ` (showing ${fmtN(shown)} squares)` : '') + '. Mint = the true one.';
        const txtW = narrow ? 0 : 260, gy = narrow ? 44 : 0;
        if (narrow) lib.text(ctx, msg, 0, 2, { size: 13, color: C.ink, maxWidth: w - 4, lh: 1.35 });
        else lib.text(ctx, msg, w - txtW + 14, 6, { size: 14, color: C.ink, maxWidth: txtW - 16, lh: 1.4 });
        const gw = w - txtW, gh = hh - gy - 2, area = gw * gh;
        const sz = clamp(Math.floor(Math.sqrt(area / shown)), 2, 24), cols = Math.max(1, Math.floor(gw / sz));
        for (let i = 0; i < shown; i++) {
          const x = (i % cols) * sz, y = gy + Math.floor(i / cols) * sz; if (y + sz > hh) break;
          const d = Math.max(1, sz - (sz > 4 ? 2 : 1));
          ctx.fillStyle = i === 0 ? C.truth : lib.rgba(C.muted, 0.35); ctx.fillRect(x, y, d, d);
        }
      }
      upd();
      P.appendChild(lib.callout('insight', 'Where thinking enters', 'For an EBT, p(target) is read from softmax(ŷ<sub>N</sub>), the logits after N gradient steps on the energy. More steps, or picking the lowest-energy of several candidates, changes those probabilities, and that is how EBT perplexity drops with thinking (Fig 6a). A Transformer++ produces its probabilities in one pass and cannot change them.'));

      // ---- card 3: Table 3 ----
      const cols = T3.columns; let tsel = 1;
      const tSeg = lib.segmented({ options: cols.map((c, i) => [i, c]), value: tsel, label: 'Table 3 column', onchange: (v) => { tsel = v; updT3(); } });
      const t3ro = h('div', { class: 'data-t3' });
      const card3 = h('div', { class: 'card stack' }, head('Table 3, unpacked', lib.badge('paper', 'Table 3, p.12'), lib.badge('ext', 'conversions derived')), tSeg.el);
      const t3c = rcanvas(card3, (w) => w < 560 ? 210 : 230, 'Table 3 perplexities for Transformer++ and EBT on pretraining and four benchmarks', drawT3);
      card3.appendChild(t3ro);
      card3.appendChild(small('EBT is worse on pretraining perplexity (33.43 vs 31.36) yet better on GSM8K, BB Math QA and BB Dyck; it is slightly worse on SQuAD. The models were trained identically (p.11). The ratio column is the paper\'s measure of how far out of distribution a dataset is (p.11); our ratios come from Table 3, while Fig 7 was computed by the authors in their own thinking setup.'));
      P.appendChild(card3);
      t3c.c.addEventListener('click', (ev) => { const [x] = t3c.toLocal(ev); const L = t3Lay(t3c.w); const j = Math.floor((x - L.x) / (L.w / cols.length)); if (j >= 0 && j < cols.length) { tsel = j; tSeg.set(j); updT3(); } });
      t3c.c.style.cursor = 'pointer';
      function t3Lay(w) { return { x: 40, y: 18, w: w - 48, h: (w < 560 ? 210 : 230) - 18 - 30 }; }
      function drawT3(ctx, w) {
        const L = t3Lay(w), gw = L.w / cols.length, Y = (v) => L.y + L.h - v / 140 * L.h;
        ctx.strokeStyle = C.rule; ctx.lineWidth = 1;
        [0, 35, 70, 105, 140].forEach(v => { ctx.beginPath(); ctx.moveTo(L.x, Y(v)); ctx.lineTo(L.x + L.w, Y(v)); ctx.stroke(); lib.text(ctx, String(v), L.x - 6, Y(v), { size: 12, kind: 'mono', color: C.faint, align: 'right', baseline: 'middle' }); });
        cols.forEach((c, j) => {
          const gx = L.x + j * gw, on = j === tsel;
          if (on) { ctx.fillStyle = lib.rgba(C.ink, 0.06); ctx.fillRect(gx + 2, L.y - 14, gw - 4, L.h + 14); }
          [['Transformer++', C.ar], ['EBT', C.ebt]].forEach(([m, col], k) => {
            const v = T3[m][j], bw = Math.min(30, gw * 0.3), x = gx + gw / 2 + (k ? 3 : -3 - bw);
            ctx.fillStyle = lib.rgba(col, on ? 1 : 0.65); ctx.fillRect(x, Y(v), bw, L.y + L.h - Y(v));
            if (gw > 60) lib.text(ctx, v.toFixed(1), x + bw / 2, Y(v) - 15, { size: 12, kind: 'mono', color: on ? C.ink : C.muted, align: 'center' });
          });
          lib.text(ctx, c, gx + gw / 2, L.y + L.h + 8, { size: w < 560 ? 12 : 13, color: on ? C.ink : C.muted, align: 'center', maxWidth: gw - 4 });
        });
      }
      function updT3() {
        const j = tsel, a = T3['Transformer++'][j], b = T3.EBT[j];
        const row = (name, v, col) => h('tr', {}, h('th', { scope: 'row', style: `color:var(--${col})` }, name), h('td', {}, v.toFixed(2)), h('td', {}, Math.log(v).toFixed(3)), h('td', {}, Math.log2(v).toFixed(2)), h('td', {}, (100 / v).toFixed(2) + '%'), h('td', {}, j ? (v / T3[name][0]).toFixed(2) + '×' : '1 (itself)'));
        t3ro.innerHTML = '';
        t3ro.append(h('div', { class: 'tbl' }, h('table', {}, h('thead', {}, h('tr', {}, ...['model', 'perplexity', 'nats / token', 'bits / token', 'geo-mean p', 'ratio to pretrain'].map(x => h('th', { scope: 'col' }, x)))),
          h('tbody', {}, row('Transformer++', a, 'ar'), row('EBT', b, 'ebt')))),
          h('p', { class: 'data-small' }, `${cols[j]}: EBT is ${signPct((b - a) / a)} vs Transformer++ in perplexity, ${(Math.log(b) - Math.log(a) >= 0 ? '+' : '−') + Math.abs(Math.log(b) - Math.log(a)).toFixed(3)} nats per token. Lower is better. nats = ln(ppl); geo-mean p = 1/ppl, the geometric-mean probability given to the true token.`));
        t3c.redraw();
      }
      updT3();
      P.appendChild(lib.callout('', 'Why perplexity, not accuracy', '"we focus on reporting perplexity as our relatively small models trained from scratch ... do not achieve high accuracies on many of these benchmarks. Furthermore, perplexity often functions as a more linear metric than accuracy" (p.9).'));
      return { stop: () => { pl2.stop(); if (subPlayer) subPlayer.stop(); } };
    }

    // =====================================================================
    // 3. Images
    // =====================================================================
    function buildImages(P) {
      const I = S.images, coco = I.coco, NZ = I.noise;
      // linear beta schedule, 1000 steps, computed live
      const betas = Array.from({ length: 1000 }, (_, t) => 1e-4 + (2e-2 - 1e-4) * t / 999);
      const abarK = [1]; for (let t = 0; t < 1000; t++) abarK.push(abarK[t] * (1 - betas[t]));
      const abar = (sg) => { const k = clamp(sg, 0, 1) * 1000, k0 = Math.floor(k); return k0 >= 1000 ? abarK[1000] : abarK[k0] * Math.pow(1 - betas[k0], k - k0); };
      let sig = 's10', selId = '01', kTok = 24, hover = -1, hoverSig = null;
      const SIG = { clean: 0, s10: 0.1, s20: 0.2 };
      const fileFor = (c) => sig === 'clean' ? c.file : c.noisy[sig].file;

      P.appendChild(h('div', { class: 'prose' }, h('p', { html: 'The bidirectional EBT learns to denoise <b>COCO 2014</b> images at 128×128 pixels, cut into 16×16 patches (p.13, p.34). Noise follows the DiT linear variance schedule (β from 1e-4 to 2e-2), and σ is the <i>fraction of that schedule</i> applied, not a standard deviation: σ = 0.1 for training and in-distribution tests, σ = 0.2 for out-of-distribution tests (p.13). Representations are then probed on ImageNet-1k (p.14).' })));
      P.appendChild(pipe([['COCO image', '128 × 128'], ['Add noise', 'σ = 0.1 or 0.2'], ['Patchify', '64 patches, 16 px'], ['Bidirectional EBT', 'every patch sees all'], ['Loss', 'MSE'], ['Score', 'PSNR, pixel MSE']]));

      // ---- bench 1: grid + schedule ----
      const sigSeg = lib.segmented({ options: [['clean', 'Clean'], ['s10', 'σ = 0.1 (train)'], ['s20', 'σ = 0.2 (OOD)']], value: sig, label: 'Noise level', onchange: (v) => { sig = v; updImgs(); } });
      const sro = h('div', { class: 'data-kv2' });
      const ctl1 = h('div', { class: 'card stack' }, h('div', { class: 'data-h4' }, 'Noise level'), sigSeg.el, sro);
      const sched = rcanvas(ctl1, () => 190, 'Signal and noise scale of the linear beta schedule as a function of sigma', drawSched);
      ctl1.appendChild(small('x<sub>σ</sub> = √ᾱ · x + √(1 − ᾱ) · ε, with pixels in [−1, 1] and ε ~ N(0, I). ᾱ is the product of (1 − β<sub>t</sub>) over the first σ × 1000 steps, computed here live. Hover the chart to read any σ.'));
      const grid = h('div', { class: 'data-imgs' });
      const capEl = h('div', { class: 'readout', 'aria-live': 'polite' });
      const ins1 = h('div', { class: 'card stack' }, head('12 COCO 2014 images, as the model sees them', dataBadge('COCO 2014 via AbdoTW/COCO_2014'), lib.badge('paper', 'Table 4')), grid, capEl,
        small('Images 01–08 are from the train split, 09–12 from validation. Cropped exactly as the authors\' code does (center crop to 128 px). The noisy versions use fixed seeds. Click an image to patchify it below.'));
      P.appendChild(h('div', { class: 'bench' }, ctl1, ins1));
      const figs = coco.map(c => {
        const im = h('img', { src: c.file, alt: c.caption, width: '128', height: '128', loading: 'lazy', decoding: 'async' });
        const b = h('button', { type: 'button', class: 'data-img', 'aria-pressed': String(c.id === selId), title: c.caption }, im, h('span', { class: 'tag ' + (c.split === 'train' ? 'tr' : 'va') }, c.split === 'train' ? 'train' : 'val'));
        b.addEventListener('click', () => { selId = c.id; figs.forEach(f => f.b.setAttribute('aria-pressed', String(f.c.id === selId))); updCap(); patch.redraw(); });
        grid.appendChild(b);
        return { c, im, b };
      });
      function updCap() {
        const c = coco.find(x => x.id === selId);
        capEl.innerHTML = '';
        add(capEl, h('span', {}, h('b', {}, c.id), ' · "' + c.caption + '"'), h('span', {}, 'COCO id ', h('b', {}, String(c.coco_id))), h('span', {}, c.split), h('span', {}, c.license),
          sig !== 'clean' ? h('span', {}, 'noisy-input PSNR ', h('b', {}, c.noisy[sig].psnr_db_vs_clean.toFixed(2) + ' dB')) : null);
      }
      function updImgs() {
        figs.forEach(f => { f.im.src = fileFor(f.c); });
        const sg = SIG[sig], ab = abar(sg);
        const meanPsnr = sig === 'clean' ? null : NZ.noisy_input_mean_psnr_db[sig];
        const t4 = sig === 's10' ? ['27.25', '26.58'] : sig === 's20' ? ['23.29', '19.56'] : null;
        sro.innerHTML = '';
        const kv = (k, v) => [h('span', { class: 'k' }, k), h('b', {}, v)];
        sro.append(...kv('steps applied', fmtN(sg * 1000) + ' / 1,000'), ...kv('ᾱ', ab.toFixed(3)), ...kv('signal √ᾱ', Math.sqrt(ab).toFixed(3)), ...kv('noise std √(1−ᾱ)', Math.sqrt(1 - ab).toFixed(3)));
        if (meanPsnr != null) sro.append(...kv('noisy-input PSNR (our 12)', meanPsnr.toFixed(2) + ' dB'));
        if (t4) sro.append(h('span', { class: 'k full' }, 'After denoising, paper Table 4: ', h('b', { style: 'color:var(--ebt)' }, 'EBT ' + t4[0] + ' dB'), ', ', h('b', { style: 'color:var(--diff)' }, 'DiT ' + t4[1] + ' dB'), ' (σ ' + sg + (sg === 0.2 ? ', OOD' : ', in-distribution') + ').'));
        else sro.append(h('span', { class: 'k full' }, 'Clean image: no noise. The paper trains at σ = 0.1 and tests OOD at σ = 0.2.'));
        updCap(); sched.redraw(); patch.redraw();
      }
      function drawSched(ctx, w, hh) {
        const ax = { x: 34, y: 14, w: w - 44, h: hh - 14 - 34 };
        const X = (s) => ax.x + s * ax.w, Y = (v) => ax.y + ax.h - v * ax.h;
        ctx.strokeStyle = C.rule; ctx.lineWidth = 1;
        [0, 0.5, 1].forEach(v => { ctx.beginPath(); ctx.moveTo(ax.x, Y(v)); ctx.lineTo(ax.x + ax.w, Y(v)); ctx.stroke(); lib.text(ctx, String(v), ax.x - 6, Y(v), { size: 12, kind: 'mono', color: C.faint, align: 'right', baseline: 'middle' }); });
        [0, 0.25, 0.5, 0.75, 1].forEach(v => lib.text(ctx, String(v), X(v), ax.y + ax.h + 5, { size: 12, kind: 'mono', color: C.faint, align: 'center' }));
        lib.text(ctx, 'σ (fraction of schedule)', ax.x + ax.w, hh - 13, { size: 12, color: C.muted, align: 'right' });
        [[0.1, 'train', 'right'], [0.2, 'OOD', 'left']].forEach(([s, l, al]) => { ctx.save(); ctx.strokeStyle = lib.rgba(C.ebt, SIG[sig] === s ? 1 : 0.45); ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(X(s), ax.y); ctx.lineTo(X(s), ax.y + ax.h); ctx.stroke(); ctx.restore(); lib.text(ctx, l, X(s) + (al === 'left' ? 4 : -4), ax.y + ax.h * 0.42, { size: 12, color: C.ebt, align: al }); });
        const sig1 = [], noi = [];
        for (let i = 0; i <= 100; i++) { const s = i / 100, ab = abar(s); sig1.push([X(s), Y(Math.sqrt(ab))]); noi.push([X(s), Y(Math.sqrt(1 - ab))]); }
        lib.line(ctx, sig1, { color: C.truth, width: 2.2 }); lib.line(ctx, noi, { color: C.bad, width: 2.2 });
        lib.text(ctx, 'signal √ᾱ', X(0.62), Y(0.3), { size: 13, color: C.truth });
        lib.text(ctx, 'noise std', X(0.62), Y(0.88), { size: 13, color: C.bad });
        const s0 = hoverSig != null ? hoverSig : SIG[sig];
        const ab0 = abar(s0);
        lib.dot(ctx, X(s0), Y(Math.sqrt(ab0)), 4.5, C.truth); lib.dot(ctx, X(s0), Y(Math.sqrt(1 - ab0)), 4.5, C.bad);
        if (hoverSig != null) lib.text(ctx, `σ ${s0.toFixed(2)}: signal ${Math.sqrt(ab0).toFixed(2)}, noise ${Math.sqrt(1 - ab0).toFixed(2)}`, ax.x + 4, ax.y + ax.h - 18, { size: 12, kind: 'mono', color: C.ink });
      }
      sched.c.addEventListener('pointermove', (ev) => { const [x] = sched.toLocal(ev); hoverSig = clamp((x - 34) / (sched.w - 44), 0, 1); sched.redraw(); });
      sched.c.addEventListener('pointerleave', () => { hoverSig = null; sched.redraw(); });

      // ---- bench 2: patchify ----
      const kSl = lib.slider({ id: 'data-ktok', label: 'Patches turned into tokens', min: 0, max: 64, step: 1, value: kTok, oninput: (v) => { pl.stop(); kTok = v; tw.stop(); patch.redraw(); updK(); } });
      const tw = tweener(() => patch.redraw());
      const pl = player({ label: 'Patchify step by step', interval: RM ? 0.25 : 0.3, canStep: () => kTok < 64, step: () => { kTok++; kSl.set(kTok); tw.start(0.28); updK(); }, reset: () => { kTok = 0; kSl.set(0); tw.stop(); patch.redraw(); updK(); } });
      const kro = h('div', { class: 'readout' });
      const ctl2 = h('div', { class: 'card stack' }, h('div', { class: 'data-h4' }, 'Patchify'), kSl.el, pl.el, kro,
        small('A 128×128 image is cut into an 8×8 grid of 16×16 patches, read row by row. Each patch is 16 × 16 × 3 = 768 numbers, linearly projected to one token of the model\'s width (1,536 for the large size used here, Table D.1).'),
        small('The EBT is bidirectional: every patch token attends to all 64, unlike the causal text mask. The noisy patches are the context x; the prediction ŷ is the clean image. Hover a patch to inspect it.'));
      const ins2 = h('div', { class: 'card stack' }, head('From image to 64 tokens', dataBadge('the selected COCO image'), lib.badge('paper', 'p.13, patch 16')));
      const patch = rcanvas(ins2, (w) => pLay(w).H, 'The selected image cut into 64 patches that become a token sequence', drawPatch);
      P.appendChild(h('div', { class: 'bench' }, ctl2, ins2));
      function updK() { kro.innerHTML = ''; kro.append(h('span', {}, 'tokens ', h('b', {}, kTok + ' / 64')), h('span', {}, 'numbers so far ', h('b', {}, fmtN(kTok * 768))), h('span', {}, 'whole image ', h('b', {}, '128 × 128 × 3 = 49,152'))); }
      function pLay(w) {
        const side = w >= 640;
        const S0 = side ? Math.min(300, Math.floor(w * 0.36)) : Math.min(w, 288);
        const ix = side ? 0 : Math.floor((w - S0) / 2), iy = 22;
        const rx = side ? S0 + 30 : 0, rw = side ? w - S0 - 30 : w;
        const per = 16, gap = side ? 3 : 2, ts = Math.floor((rw - gap * (per - 1)) / per);
        const ry = side ? 44 : iy + S0 + 40;
        const rh = 4 * (ts + gap) + 16;
        const insY = ry + rh + 18, insH = side ? 130 : 170;
        return { side, S0, ix, iy, rx, rw, per, gap, ts, ry, insY, H: Math.max(side ? iy + S0 + 10 : 0, insY + insH) };
      }
      function srcImg() { const c = coco.find(x => x.id === selId); return { path: fileFor(c), c }; }
      function drawPatch(ctx, w) {
        const L = pLay(w), { path } = srcImg(), im = lib.img(path);
        if (!im) { lib.loadImg(path).then(() => patch.redraw()); lib.text(ctx, 'loading image…', 8, 8, { size: 14, color: C.muted }); return; }
        const cell = L.S0 / 8, tq = tw.t, flying = kTok > 0 && tq < 1 ? kTok - 1 : -1;
        lib.text(ctx, '128 × 128 image, 8 × 8 patches', L.ix, 2, { size: 13, color: C.muted });
        ctx.imageSmoothingEnabled = false;
        ctx.drawImage(im, L.ix, L.iy, L.S0, L.S0);
        for (let j = 0; j < 64; j++) {
          const r = Math.floor(j / 8), c = j % 8, x = L.ix + c * cell, y = L.iy + r * cell;
          if (j < kTok && j !== flying) { ctx.fillStyle = lib.rgba(C.bg, 0.62); ctx.fillRect(x, y, cell, cell); }
        }
        ctx.strokeStyle = lib.rgba(C.ink, 0.35); ctx.lineWidth = 1;
        for (let g = 0; g <= 8; g++) { ctx.beginPath(); ctx.moveTo(L.ix + g * cell, L.iy); ctx.lineTo(L.ix + g * cell, L.iy + L.S0); ctx.moveTo(L.ix, L.iy + g * cell); ctx.lineTo(L.ix + L.S0, L.iy + g * cell); ctx.stroke(); }
        const focus = hover >= 0 ? hover : Math.max(0, kTok - 1);
        { const r = Math.floor(focus / 8), c = focus % 8; ctx.strokeStyle = C.ebt; ctx.lineWidth = 2.5; ctx.strokeRect(L.ix + c * cell, L.iy + r * cell, cell, cell); }
        // token ribbon
        lib.text(ctx, 'token sequence (raster order)', L.rx, L.ry - 20, { size: 13, color: C.muted });
        const slot = (j) => [L.rx + (j % L.per) * (L.ts + L.gap), L.ry + Math.floor(j / L.per) * (L.ts + L.gap)];
        for (let j = 0; j < 64; j++) {
          const [x, y] = slot(j);
          if (j < kTok && j !== flying) ctx.drawImage(im, (j % 8) * 16, Math.floor(j / 8) * 16, 16, 16, x, y, L.ts, L.ts);
          else { ctx.strokeStyle = lib.rgba(C.faint, 0.6); ctx.lineWidth = 1; ctx.strokeRect(x + 0.5, y + 0.5, L.ts - 1, L.ts - 1); }
          if (j === focus) { ctx.strokeStyle = C.ebt; ctx.lineWidth = 2; ctx.strokeRect(x - 1, y - 1, L.ts + 2, L.ts + 2); }
        }
        [0, 16, 32, 48].forEach(j => { const [x, y] = slot(j); lib.text(ctx, String(j), x - 4, y + L.ts / 2, { size: 12, kind: 'mono', color: C.faint, align: 'right', baseline: 'middle' }); });
        if (flying >= 0) {
          const e = lib.ease(tq), r = Math.floor(flying / 8), c = flying % 8, [tx, ty] = slot(flying);
          const x = lib.lerp(L.ix + c * cell, tx, e), y = lib.lerp(L.iy + r * cell, ty, e), s = lib.lerp(cell, L.ts, e);
          ctx.drawImage(im, c * 16, r * 16, 16, 16, x, y, s, s); ctx.strokeStyle = C.ebt; ctx.lineWidth = 2; ctx.strokeRect(x, y, s, s);
        }
        // inspector (under the ribbon on wide screens, under everything on phones)
        const ox = L.side ? L.rx : 0, aw = w - ox;
        const y0 = L.insY, r = Math.floor(focus / 8), c = focus % 8, big = L.side ? 92 : 80;
        lib.text(ctx, `token ${focus} = patch (row ${r}, col ${c})${hover >= 0 ? '' : ', the latest'}`, ox, y0, { size: 14, kind: 'mono', color: C.ebt });
        ctx.drawImage(im, c * 16, r * 16, 16, 16, ox, y0 + 24, big, big);
        const chX = ox + big + 18;
        const chW = L.side ? 52 : clamp(Math.floor((aw - big - 18) / 3) - 6, 24, 64);
        ['#ff0000', '#00ff00', '#0000ff'].forEach((col, k) => {
          const x = chX + k * (chW + 6);
          ctx.drawImage(im, c * 16, r * 16, 16, 16, x, y0 + 24, chW, chW);
          ctx.save(); ctx.globalCompositeOperation = 'multiply'; ctx.fillStyle = col; ctx.fillRect(x, y0 + 24, chW, chW); ctx.restore();
          lib.text(ctx, ['R', 'G', 'B'][k], x + chW / 2, y0 + 28 + chW, { size: 12, kind: 'mono', color: C.muted, align: 'center' });
        });
        const tx = L.side ? chX + 3 * (chW + 6) + 12 : 0, ty = L.side ? y0 + 26 : y0 + 24 + Math.max(big, chW + 18) + 10;
        lib.text(ctx, '16 × 16 × 3 = 768 numbers → linear patch embedding → one 1,536-d token', tx, ty, { size: 13, color: C.ink, lh: 1.45, maxWidth: w - tx - 2 });
      }
      patch.c.addEventListener('pointermove', (ev) => {
        const [x, y] = patch.toLocal(ev), L = pLay(patch.w), cell = L.S0 / 8;
        let j = -1;
        if (x >= L.ix && x < L.ix + L.S0 && y >= L.iy && y < L.iy + L.S0) j = Math.floor((y - L.iy) / cell) * 8 + Math.floor((x - L.ix) / cell);
        else { const cc = Math.floor((x - L.rx) / (L.ts + L.gap)), rr = Math.floor((y - L.ry) / (L.ts + L.gap)); if (cc >= 0 && cc < L.per && rr >= 0 && rr < 4 && x >= L.rx) j = rr * L.per + cc; }
        if (j !== hover) { hover = j; patch.redraw(); }
      });
      patch.c.addEventListener('pointerleave', () => { hover = -1; patch.redraw(); });
      loadAll(coco.flatMap(c => [c.file, c.noisy.s10.file, c.noisy.s20.file]), patch);
      updImgs(); updK();

      // ---- card 3: linear probe ----
      const inet = I.imagenet || [];
      const strip = h('div', { class: 'data-inet' }, ...inet.map(x => h('figure', {}, h('img', { src: x.file, alt: x.label, width: '128', height: '128', loading: 'lazy' }), h('figcaption', {}, x.label))));
      const card3 = h('div', { class: 'card stack' }, head('Linear probe on ImageNet-1k', dataBadge('ImageNet-1k val, 128 px mirror'), lib.badge('paper', 'Table 4')),
        small('To test whether denoising taught the model useful features, the authors freeze it, average all final patch tokens into one vector, and train only a linear classifier over the 1,000 ImageNet classes (p.14). For the DiT they feed timestep T = 0. Eight real validation images:'), strip);
      const probe = rcanvas(card3, (w) => w < 560 ? 200 : 180, 'ImageNet linear-probe accuracy: chance, DiT and EBT, top-1 and top-5', drawProbe);
      card3.appendChild(small('Both are far from a real classifier. Still, EBT top-1 is 5.32% (about 1 in 19 images, 53× chance) against 0.31% for DiT (about 1 in 323, 3× chance). The paper calls this "around 10×"; it is 17× for top-1 and 9.7× for top-5 [derived].'));
      P.appendChild(card3);
      function drawProbe(ctx, w, hh) {
        const groups = [['top-1', 0.1, 0.31, 5.32], ['top-5', 0.5, 1.36, 13.2]];
        const narrow = w < 560, lx = narrow ? 64 : 80, ax = { x: lx, y: 8, w: w - lx - 60, h: hh - 30 };
        const X = (v) => ax.x + v / 14 * ax.w; const gh = ax.h / 2;
        ctx.strokeStyle = C.rule; [0, 5, 10].forEach(v => { ctx.beginPath(); ctx.moveTo(X(v), ax.y); ctx.lineTo(X(v), ax.y + ax.h); ctx.stroke(); lib.text(ctx, v + '%', X(v), ax.y + ax.h + 4, { size: 12, kind: 'mono', color: C.faint, align: 'center' }); });
        groups.forEach(([g, ch, dit, ebt], k) => {
          const gy = ax.y + k * gh; lib.text(ctx, g, ax.x - 10, gy + gh / 2, { size: 13, kind: 'mono', color: C.muted, align: 'right', baseline: 'middle' });
          [['chance', ch, C.faint], ['DiT', dit, C.diff], ['EBT', ebt, C.ebt]].forEach(([nm, v, col], i) => {
            const bh = (gh - 14) / 3, y = gy + 4 + i * bh;
            ctx.fillStyle = col; ctx.fillRect(ax.x, y + 1, Math.max(2, X(v) - ax.x), bh - 3);
            lib.text(ctx, `${nm} ${v}%`, X(v) + 6, y + bh / 2, { size: 12, kind: 'mono', color: col === C.faint ? C.muted : col, baseline: 'middle' });
          });
        });
      }
      return { stop: () => { pl.stop(); tw.stop(); } };
    }

    // =====================================================================
    // 4. Video
    // =====================================================================
    function buildVideo(P) {
      const V = S.video, clips = (V.ssv2 && V.ssv2.clips) || [];
      let ci = 0, t = 4, aspect = 'model', it = 1;
      P.appendChild(h('div', { class: 'prose' }, h('p', { html: 'The video EBT predicts the next frame of <b>Something-Something V2</b> clips, conditioned on all previous frames (p.12). Frames are sampled 0.25 s apart, resized to 224×224 and encoded by a frozen <b>SD-XL VAE</b> into 3,136-dimensional features; the model sees 16 frames of context and is trained with a Smooth L1 loss (p.12, p.34, Table D.2). Here SSV2 data is too small to see each clip once, so this experiment asks how well models fit a fixed dataset (p.12).' })));
      P.appendChild(pipe([['SSV2 clip', 'frames 0.25 s apart'], ['Resize', '224 × 224'], ['Frozen VAE', '3,136-d latent'], ['Context', '16 frames'], ['Target', 'next latent'], ['Loss', 'Smooth L1']]));
      if (!clips.length) { P.appendChild(lib.callout('warn', 'No video frames', 'samples.json has no SSV2 clips.')); return {}; }

      // ---- bench 1: next-frame setup ----
      const csel = h('select', { id: 'data-clip' }, ...clips.map((c, i) => h('option', { value: String(i) }, `“${c.label}” (video ${c.video_id})`)));
      const tSl = lib.slider({ id: 'data-tctx', label: 'Context frames t (predict frame t+1)', min: 1, max: 8, step: 1, value: t, oninput: (v) => { pl.stop(); t = v; upd(); } });
      const pl = player({ label: 'Slide the next-frame window', interval: 0.9, canStep: () => t < 8, step: () => { t++; tSl.set(t); upd(); }, reset: () => { t = 1; tSl.set(1); upd(); } });
      const aSeg = lib.segmented({ options: [['model', '224 × 224 (as fed)'], ['raw', 'As recorded']], value: aspect, label: 'Frame shape', onchange: (v) => { aspect = v; vid.resize(); } });
      const vro = h('div', { class: 'readout' });
      const ctl1 = h('div', { class: 'card stack' }, h('label', { class: 'data-h4', for: 'data-clip' }, 'Clip'), csel, tSl.el, pl.el, h('div', { class: 'data-h4' }, 'Frame shape'), aSeg.el, vro,
        small('The authors\' code resizes each frame to 224×224 without keeping the aspect ratio, so wide clips are squashed. The VAE then downsamples 8× per side into 4 channels: 4 × 28 × 28 = 3,136 numbers per frame, 48× fewer than the 150,528 pixel values.'));
      const ins1 = h('div', { class: 'card stack' }, head('Next-frame prediction in latent space', dataBadge('SSV2 frames via unofficial mirror moondream/ssv2-3x3'), lib.badge('ext', 'latent preview = illustration')));
      const vid = rcanvas(ins1, (w) => vLay(w).H, 'Film strip of the clip with the context window and the next-frame target, and the frame-to-latent pipeline', drawVid);
      ins1.appendChild(small('The mirror stores 9 frames per video and does not document their spacing, so these frames are not necessarily 0.25 s apart. The 28 × 28 preview is the frame average-pooled to the latent\'s grid size, to show its resolution; the real SD-XL latent channels are not color images.'));
      P.appendChild(h('div', { class: 'bench' }, ctl1, ins1));
      csel.addEventListener('change', () => { ci = +csel.value; pl.stop(); loadAll(clips[ci].frames, vid); upd(); });
      const pooled = new Map();
      function pool(path) { // 28x28 average pool via downscaling (drawn only, never read back)
        if (pooled.has(path)) return pooled.get(path);
        const im = lib.img(path); if (!im) return null;
        const a = document.createElement('canvas'); a.width = 112; a.height = 112; const ga = a.getContext('2d'); ga.imageSmoothingEnabled = true; ga.imageSmoothingQuality = 'high'; ga.drawImage(im, 0, 0, 112, 112);
        const b = document.createElement('canvas'); b.width = 28; b.height = 28; const gb = b.getContext('2d'); gb.imageSmoothingEnabled = true; gb.imageSmoothingQuality = 'high'; gb.drawImage(a, 0, 0, 28, 28);
        pooled.set(path, b); return b;
      }
      function vLay(w) {
        const clip = clips[ci], ar = aspect === 'model' ? 1 : clip.frame_size[1] / clip.frame_size[0];
        const gap = w < 560 ? 3 : 6, fw = Math.floor((w - 8 * gap) / 9), fh = Math.round(fw * ar);
        const stripY = 42, stripH = fh + 22;
        const narrow = w < 640;
        const bigW = narrow ? Math.floor(w * 0.4) : 180, bigH = Math.round(bigW * ar);
        const latS = narrow ? Math.floor(w * 0.3) : 132;
        const py = stripY + stripH + 52;
        const H = py + Math.max(bigH, latS + 30) + (narrow ? 104 : 12);
        return { ar, gap, fw, fh, stripY, narrow, bigW, bigH, latS, py, H };
      }
      function drawVid(ctx, w) {
        const clip = clips[ci], L = vLay(w);
        lib.text(ctx, 'frames 1 to 9 of the clip', 0, 0, { size: 13, color: C.muted });
        { const tx = t * (L.fw + L.gap) + L.fw / 2; lib.text(ctx, 'target ↓', clamp(tx, 34, w - 34), 20, { size: 13, color: C.truth, align: 'center' }); }
        clip.frames.forEach((f, i) => {
          const x = i * (L.fw + L.gap), y = L.stripY, im = lib.img(f), n = i + 1;
          if (im) ctx.drawImage(im, x, y, L.fw, L.fh); else { ctx.fillStyle = C.panel2; ctx.fillRect(x, y, L.fw, L.fh); }
          if (n > t + 1) { ctx.fillStyle = lib.rgba(C.bg, 0.72); ctx.fillRect(x, y, L.fw, L.fh); }
          if (n <= t) { ctx.strokeStyle = C.ar; ctx.lineWidth = 2; ctx.strokeRect(x + 1, y + 1, L.fw - 2, L.fh - 2); }
          if (n === t + 1) { ctx.save(); ctx.strokeStyle = C.truth; ctx.lineWidth = 2.5; ctx.setLineDash([5, 3]); ctx.strokeRect(x + 1, y + 1, L.fw - 2, L.fh - 2); ctx.restore(); }
          lib.text(ctx, String(n), x + L.fw / 2, y + L.fh + 4, { size: 12, kind: 'mono', color: n <= t ? C.ar : n === t + 1 ? C.truth : C.faint, align: 'center' });
        });
        // brackets
        const bx1 = t * (L.fw + L.gap) - L.gap, by = L.stripY + L.fh + 20;
        ctx.strokeStyle = C.ar; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(0, by - 3); ctx.lineTo(0, by); ctx.lineTo(bx1, by); ctx.lineTo(bx1, by - 3); ctx.stroke();
        lib.text(ctx, 'context x: frames 1–' + t, 2, by + 3, { size: 13, color: C.ar });
        // pipeline for the target frame
        const f = clip.frames[t], im = lib.img(f), py = L.py;
        lib.text(ctx, 'frame ' + (t + 1) + (aspect === 'model' ? ': 224 × 224 × 3 = 150,528 values' : ': ' + clip.frame_size[0] + ' × ' + clip.frame_size[1] + ' px in the mirror'), 0, py - 22, { size: 13, color: C.truth });
        if (im) ctx.drawImage(im, 0, py, L.bigW, L.bigH);
        ctx.strokeStyle = C.truth; ctx.lineWidth = 2; ctx.strokeRect(1, py + 1, L.bigW - 2, L.bigH - 2);
        const ax0 = L.bigW + 8, latS = L.latS;
        const lx = L.narrow ? w - latS - 20 : ax0 + 150, ly = py + 4;
        const ay = py + Math.min(L.bigH, latS) / 2;
        lib.arrow(ctx, ax0, ay, lx - 8, ay, { color: C.muted, width: 2, head: 9 });
        lib.text(ctx, L.narrow ? 'VAE' : 'frozen VAE', (ax0 + lx - 8) / 2, ay - 22, { size: 13, color: C.muted, align: 'center' });
        const pim = pool(f);
        for (let k = 3; k >= 0; k--) {
          const ox = lx + k * 6, oy = ly + k * 6;
          ctx.fillStyle = C.panel2; ctx.fillRect(ox, oy, latS, latS); ctx.strokeStyle = lib.rgba(C.ebt, 0.7); ctx.lineWidth = 1; ctx.strokeRect(ox + 0.5, oy + 0.5, latS - 1, latS - 1);
          if (k === 0 && pim) { ctx.imageSmoothingEnabled = false; ctx.drawImage(pim, ox, oy, latS, latS); ctx.imageSmoothingEnabled = true; ctx.strokeRect(ox + 0.5, oy + 0.5, latS - 1, latS - 1); }
        }
        const note = 'The EBT starts its guess ŷ for this latent as N(0, I) noise and descends the energy given the context latents. Smooth L1 compares the final ŷ to the true latent.';
        if (L.narrow) {
          const yb = Math.max(py + L.bigH, ly + latS + 18) + 8;
          lib.text(ctx, '4 × 28 × 28 = 3,136 numbers', w, yb, { size: 13, kind: 'mono', color: C.ebt, align: 'right' });
          lib.text(ctx, note, 0, yb + 26, { size: 13, color: C.muted, maxWidth: w - 4, lh: 1.35 });
        } else {
          lib.text(ctx, '4 × 28 × 28 = 3,136 numbers', lx, ly + latS + 28, { size: 13, kind: 'mono', color: C.ebt });
          lib.text(ctx, note, lx + latS + 40, py + 6, { size: 14, color: C.muted, maxWidth: Math.max(140, w - (lx + latS + 40)), lh: 1.4 });
        }
      }
      function upd() {
        vro.innerHTML = '';
        vro.append(h('span', {}, 'context ', h('b', {}, t + ' frame' + (t > 1 ? 's' : '') + ' → ' + fmtN(t * 3136) + ' numbers')), h('span', {}, 'target ', h('b', {}, 'frame ' + (t + 1) + ' → 3,136')), h('span', {}, 'paper context ', h('b', {}, '16 frames = 4 s')));
        vid.resize();
      }
      loadAll(clips[ci].frames, vid); clips.forEach(c => c.frames.forEach(f => lib.loadImg(f)));
      upd();

      // ---- bench 2: reading Fig 11 ----
      const itSl = lib.slider({ id: 'data-it', label: 'Thinking iteration', min: 0, max: 11, step: 1, value: it, oninput: (v) => { pl2.stop(); it = v; upd2(); } });
      const pl2 = player({ label: 'Step through thinking iterations', interval: 0.8, canStep: () => it < 11, step: () => { it++; itSl.set(it); upd2(); }, reset: () => { it = 0; itSl.set(0); upd2(); } });
      const ro2 = h('div', { class: 'data-kv2' });
      const ctl2 = h('div', { class: 'card stack' }, h('div', { class: 'data-h4' }, 'Energy per frame (Fig 11)'), itSl.el, pl2.el, ro2,
        small('Paper: early frames are mostly empty, so the next frame is hard to predict and the energy stays high. Once the blue garment is in view, the energy falls, and it rises a little again as the garment leaves (p.13–14). The energy is never trained directly; this uncertainty signal emerges (Facet 2).'));
      const ins2 = h('div', { class: 'card stack' }, head('Energy tracks uncertainty across a clip', lib.badge('paper', 'Fig 11, p.14'), lib.badge('paper', 'values approx., read from Fig 11')));
      const fbars = rcanvas(ins2, (w) => w < 560 ? 220 : 240, 'Normalized energy of each of 16 frames at the chosen thinking iteration, read from paper Figure 11', drawF11);
      const ov = h('div', { class: 'data-f11ov' });
      const figImg = h('img', { src: 'media/paper/fig11.png', alt: 'Paper Figure 11: frame energies across thinking steps, heatmap of 16 frames by 12 iterations', width: '550', height: '508' });
      ins2.appendChild(h('div', { class: 'data-f11' }, h('figure', { class: 'paper-fig' }, h('div', { class: 'data-f11wrap' }, figImg, ov)), small('The original figure. Dark purple = high energy, yellow = low. The amber box marks the iteration you picked. Bars above use this chapter\'s energy colors (bright = high).')));
      P.appendChild(h('div', { class: 'bench' }, ctl2, ins2));
      lib.loadImg('media/paper/fig11.png').then(() => fbars.redraw());
      function drawF11(ctx, w, hh) {
        const ax = { x: 36, y: 42, w: w - 44, h: hh - 42 - 62 }, n = 16, bw = ax.w / n;
        const Y = (v) => ax.y + ax.h - v * ax.h;
        ctx.strokeStyle = C.rule; ctx.lineWidth = 1;
        [0, 0.5, 1].forEach(v => { ctx.beginPath(); ctx.moveTo(ax.x, Y(v)); ctx.lineTo(ax.x + ax.w, Y(v)); ctx.stroke(); lib.text(ctx, String(v), ax.x - 6, Y(v), { size: 12, kind: 'mono', color: C.faint, align: 'right', baseline: 'middle' }); });
        lib.text(ctx, `normalized energy per frame at iteration ${it} (ghost = iteration 0)`, ax.x, 2, { size: 13, color: C.muted });
        const fig = lib.img('media/paper/fig11.png'), rowH = (F11.y1 - F11.y0) / 16;
        for (let r = 0; r < n; r++) {
          const v = FIG11[r][it], v0 = FIG11[r][0], x = ax.x + r * bw;
          ctx.strokeStyle = lib.rgba(C.muted, 0.5); ctx.setLineDash([3, 3]); ctx.strokeRect(x + bw * 0.15, Y(v0), bw * 0.7, ax.y + ax.h - Y(v0)); ctx.setLineDash([]);
          ctx.fillStyle = lib.cmapCss(v); ctx.fillRect(x + bw * 0.15, Y(v), bw * 0.7, Math.max(1.5, ax.y + ax.h - Y(v)));
          if (fig) { const th = Math.min(42, bw * 0.95 * rowH / F11.tw); const tw2 = th * F11.tw / rowH; ctx.drawImage(fig, F11.tx, F11.y0 + r * rowH + 1, F11.tw, rowH - 2, x + (bw - tw2) / 2, ax.y + ax.h + 6, tw2, th); }
          if (bw > 22 && (r % (bw > 40 ? 1 : 3) === 0)) lib.text(ctx, String(r + 1), x + bw / 2, hh - 14, { size: 12, kind: 'mono', color: C.faint, align: 'center' });
        }
        ctx.save(); ctx.strokeStyle = lib.rgba(C.truth, 0.7); ctx.lineWidth = 1.5; ctx.setLineDash([4, 3]); ctx.beginPath(); ctx.moveTo(ax.x + 4 * bw, ax.y - 22); ctx.lineTo(ax.x + 4 * bw, ax.y + ax.h + 48); ctx.stroke(); ctx.restore();
        lib.text(ctx, w < 560 ? 'empty' : 'empty scene', ax.x + 2 * bw, ax.y - 20, { size: 13, color: C.truth, align: 'center' });
        lib.text(ctx, 'garment in view', ax.x + 10 * bw, ax.y - 20, { size: 13, color: C.truth, align: 'center' });
      }
      function upd2() {
        const mean = (a, b) => { let s = 0; for (let r = a; r < b; r++) s += FIG11[r][it]; return s / (b - a); };
        const e = mean(0, 4), g = mean(4, 16);
        ro2.innerHTML = '';
        const kv = (k, v) => [h('span', { class: 'k' }, k), h('b', {}, v)];
        ro2.append(...kv('iteration', String(it)), ...kv('empty-scene frames 1–4, mean', e.toFixed(2)), ...kv('garment frames 5–16, mean', g.toFixed(2)), ...kv('ratio', (g > 0.005 ? (e / g).toFixed(1) + '×' : '≫')));
        ov.style.left = ((F11.x0 + it * (F11.x1 - F11.x0) / 12) / F11.W * 100) + '%';
        ov.style.width = ((F11.x1 - F11.x0) / 12 / F11.W * 100) + '%';
        ov.style.top = (F11.y0 / F11.H * 100) + '%';
        ov.style.height = ((F11.y1 - F11.y0) / F11.H * 100) + '%';
        fbars.redraw();
      }
      upd2();

      // ---- bench 3: Smooth L1 ----
      let err = 1.8, beta = 1;
      const eSl = lib.slider({ id: 'data-err', label: 'Error e = ŷ − y (one latent dimension)', min: -3, max: 3, step: 0.01, value: err, fmt: (v) => v.toFixed(2), oninput: (v) => { err = v; upd3(); } });
      const bSl = lib.slider({ id: 'data-beta', label: 'β (paper: 1.0)', min: 0.25, max: 2, step: 0.05, value: beta, fmt: (v) => v.toFixed(2), oninput: (v) => { beta = v; upd3(); } });
      const ro3 = h('div', { class: 'data-kv2' });
      const ctl3 = h('div', { class: 'card stack' }, h('div', { class: 'data-h4' }, 'Smooth L1 loss'), eSl.el, bSl.el, ro3,
        small('Smooth L1 is quadratic for |e| < β and linear beyond. Its gradient is e/β near zero and capped at ±1 for big errors, so a few badly predicted latent dimensions cannot dominate the update. The loss is averaged over all 3,136 dimensions.'));
      const ins3 = h('div', { class: 'card stack' }, head('The video loss, and its gradient', lib.badge('paper', 'β = 1.0, p.12')));
      const sl1 = rcanvas(ins3, (w) => w < 560 ? 420 : 260, 'Smooth L1, L1 and squared error and their gradients as a function of the error', drawSL1);
      sl1.c.addEventListener('pointerdown', (ev) => { const f = (e2) => { const [x] = sl1.toLocal(e2); const L = slLay(sl1.w); const pn = x < L.p2.x - 10 ? L.p1 : L.p2; err = clamp(-3 + (x - pn.x) / pn.w * 6, -3, 3); eSl.set(Math.round(err * 100) / 100); upd3(); }; f(ev); const mv = (e2) => f(e2); const up = () => { window.removeEventListener('pointermove', mv); window.removeEventListener('pointerup', up); }; window.addEventListener('pointermove', mv); window.addEventListener('pointerup', up); });
      sl1.c.style.cursor = 'ew-resize'; sl1.c.style.touchAction = 'pan-y';
      P.appendChild(h('div', { class: 'bench' }, ctl3, ins3));
      const sm = (e) => Math.abs(e) < beta ? 0.5 * e * e / beta : Math.abs(e) - 0.5 * beta;
      const dsm = (e) => Math.abs(e) < beta ? e / beta : Math.sign(e);
      function slLay(w) {
        const narrow = w < 560; const pw = narrow ? w - 46 : (w - 34 - 46 - 14) / 2, ph = narrow ? 170 : 200;
        return { narrow, p1: { x: 34, y: 22, w: pw, h: ph }, p2: narrow ? { x: 34, y: 22 + ph + 46, w: pw, h: ph } : { x: 34 + pw + 46, y: 22, w: pw, h: ph } };
      }
      function drawSL1(ctx, w) {
        const L = slLay(w);
        const panel = (p, title, fns, ylim, yt) => {
          const X = (e) => p.x + (e + 3) / 6 * p.w, Y = (v) => p.y + p.h - (v - ylim[0]) / (ylim[1] - ylim[0]) * p.h;
          ctx.strokeStyle = C.rule; ctx.lineWidth = 1;
          yt.forEach(v => { ctx.beginPath(); ctx.moveTo(p.x, Y(v)); ctx.lineTo(p.x + p.w, Y(v)); ctx.stroke(); lib.text(ctx, String(v), p.x - 6, Y(v), { size: 12, kind: 'mono', color: C.faint, align: 'right', baseline: 'middle' }); });
          [-3, 0, 3].forEach(v => lib.text(ctx, String(v), X(v), p.y + p.h + 4, { size: 12, kind: 'mono', color: C.faint, align: 'center' }));
          ctx.save(); ctx.strokeStyle = lib.rgba(C.ebt, 0.35); ctx.setLineDash([3, 3]); [-beta, beta].forEach(b => { ctx.beginPath(); ctx.moveTo(X(b), p.y); ctx.lineTo(X(b), p.y + p.h); ctx.stroke(); }); ctx.restore();
          lib.text(ctx, title, p.x, p.y - 20, { size: 13, color: C.muted });
          ctx.save(); ctx.beginPath(); ctx.rect(p.x, p.y, p.w, p.h); ctx.clip();
          fns.forEach(([f, col, wd, dash]) => { const pts = []; for (let i = 0; i <= 240; i++) { const e = -3 + i / 40; pts.push([X(e), Y(f(e))]); } lib.line(ctx, pts, { color: col, width: wd, dash }); });
          ctx.restore();
          ctx.strokeStyle = lib.rgba(C.ink, 0.5); ctx.beginPath(); ctx.moveTo(X(err), p.y); ctx.lineTo(X(err), p.y + p.h); ctx.stroke();
          fns.forEach(([f, col]) => { const v = f(err); if (v >= ylim[0] && v <= ylim[1]) lib.dot(ctx, X(err), Y(v), 4.5, col); });
        };
        panel(L.p1, 'loss', [[(e) => e * e, C.muted, 1.6, [5, 4]], [(e) => Math.abs(e), C.faint, 1.6, [2, 3]], [sm, C.ebt, 3]], [0, 4], [0, 2, 4]);
        panel(L.p2, 'gradient d loss / d e', [[(e) => 2 * e, C.muted, 1.6, [5, 4]], [(e) => Math.sign(e), C.faint, 1.6, [2, 3]], [dsm, C.ebt, 3]], [-3, 3], [-3, 0, 3]);
        const ly = L.narrow ? L.p2.y + L.p2.h + 24 : L.p1.y + L.p1.h + 24;
        let lx = L.p1.x;
        [['Smooth L1', C.ebt], ['squared error e²', C.muted], ['L1 |e|', C.faint]].forEach(([s, col]) => { ctx.fillStyle = col; ctx.fillRect(lx, ly + 4, 14, 4); lib.text(ctx, s, lx + 20, ly - 2, { size: 13, color: C.muted }); lx += 30 + lib.measure(ctx, s, { size: 13 }).w + 10; });
      }
      function upd3() {
        ro3.innerHTML = '';
        const kv = (k, v, col) => [h('span', { class: 'k' }, k), h('b', { style: col ? `color:var(--${col})` : '' }, v)];
        ro3.append(...kv('Smooth L1', sm(err).toFixed(3), 'ebt'), ...kv('its gradient', dsm(err).toFixed(3), 'ebt'), ...kv('squared error / gradient', (err * err).toFixed(3) + ' / ' + (2 * err).toFixed(2)), ...kv('L1 / gradient', Math.abs(err).toFixed(3) + ' / ' + Math.sign(err).toFixed(0)));
        sl1.redraw();
      }
      upd3();
      return { stop: () => { pl.stop(); pl2.stop(); } };
    }

    // =====================================================================
    // 5. Splits & setups
    // =====================================================================
    function buildSetups(P) {
      const FIELDS = [['data', 'Data'], ['split', 'Split / size'], ['input', 'Input x'], ['pred', 'Prediction ŷ'], ['loss', 'Training loss'], ['metric', 'Reported metric'], ['base', 'Baseline'], ['size', 'Models']];
      const ROWS = [
        { task: 'Text pretraining', mod: 'text', data: 'RedPajama-V2, 100B sample on Hugging Face (p.8)', split: '66M train / 33K validation samples, split by the authors (p.8)', input: 'GPT-NeoX tokens, context 256, vocabulary 50,277 (Table D.2)', pred: '50,277 logits per position from N(0, I), softmaxed and projected to an embedding (p.43–44)', loss: 'Cross-entropy (p.7)', metric: 'Validation perplexity (Figs 4, 5)', base: 'Transformer++ (Llama 2 recipe, p.8, p.33)', size: 'xxs to large, 6.18M to 396M non-embedding parameters (Table D.1, p.34)' },
        { task: 'Text downstream', mod: 'text', data: 'GSM8K, SQuAD, BIG-bench Elementary Math QA, BIG-bench Dyck Languages (p.9)', split: 'Evaluation only; the split used is not stated', input: 'Same tokenizer and context as pretraining', pred: 'Next-token distributions, as in pretraining', loss: 'None (evaluation only)', metric: 'Perplexity (Table 3, Figs 6a, 7); Fig 6b is Dyck only (p.34)', base: 'Transformer++', size: 'xxs S2-EBT and Transformer++, batch 128 for 1M steps (p.34)' },
        { task: 'Video', mod: 'video', data: 'Something-Something V2 (p.12)', split: 'Standard SSV2 train / validation split (p.34)', input: 'Frames 0.25 s apart, 224×224, frozen SD-XL VAE features (3,136-d), context 16 frames (p.12, p.34, Table D.2)', pred: 'Next frame\'s 3,136-d feature, starting from N(0, I) (p.44)', loss: 'Smooth L1, β = 1.0 (p.12)', metric: 'Minimum validation loss (p.12)', base: 'Transformer++', size: 'Up to xl (708M non-embedding), batch 256 (Fig 9, p.34)' },
        { task: 'Image denoising', mod: 'image', data: 'COCO 2014, 128×128 (p.13)', split: 'COCO 2014 train / validation (p.34)', input: 'Noisy image: σ = 0.1 of the linear β schedule in training, 0.2 for OOD tests; 16 px patches (p.13)', pred: 'The denoised image (bidirectional EBT); how ŷ0 is initialized is not stated', loss: 'MSE (p.7)', metric: 'PSNR and pixel MSE (Table 4)', base: 'DiT, best with DDIM applied recursively to its own output (p.35)', size: 'large, learning rate 1e-4, 100k steps (p.34–35)' },
        { task: 'Linear probe', mod: 'image', data: 'ImageNet-1k (p.14)', split: 'Not stated', input: 'Average of all final patch tokens; DiT is fed T = 0 (p.14)', pred: 'Class label from a linear classifier', loss: 'Linear classifier only (details not stated)', metric: 'Top-1 and top-5 accuracy (Table 4)', base: 'DiT', size: 'The denoising models above' },
      ];
      P.appendChild(h('div', { class: 'prose' }, h('p', { html: 'Five setups, one table. Pick a column to compare all of them on one question; the full table is below. Page numbers point into the paper; "not stated" means the paper does not say.' })));
      let fld = 'loss';
      const chooser = h('div', { class: 'data-chooser', role: 'group', 'aria-label': 'Compare setups by' });
      const cmp = h('div', { class: 'data-cmp' });
      FIELDS.forEach(([k, l]) => { const b = h('button', { type: 'button', class: 'btn', 'aria-pressed': String(k === fld) }, l); b.addEventListener('click', () => { fld = k; chooser.querySelectorAll('button').forEach((x, i) => x.setAttribute('aria-pressed', String(FIELDS[i][0] === fld))); updCmp(); }); chooser.appendChild(b); });
      function updCmp() {
        cmp.innerHTML = '';
        ROWS.forEach(r => cmp.appendChild(h('div', { class: 'data-cmprow m-' + r.mod }, h('b', {}, r.task), h('span', {}, r[fld]))));
      }
      updCmp();
      P.appendChild(h('div', { class: 'card stack' }, head('Compare the setups', lib.badge('paper', 'Sec 4, App. D')), chooser, cmp));
      const tbl = h('table', {}, h('thead', {}, h('tr', {}, h('th', { scope: 'col' }, 'Setup'), ...FIELDS.map(([, l]) => h('th', { scope: 'col' }, l)))),
        h('tbody', {}, ...ROWS.map(r => h('tr', {}, h('th', { scope: 'row' }, r.task), ...FIELDS.map(([k]) => h('td', {}, r[k]))))));
      P.appendChild(h('details', { class: 'data-details' }, h('summary', {}, 'Full table (scrolls sideways on small screens)'), h('div', { class: 'tbl' }, tbl)));
      P.appendChild(h('div', { class: 'grid2' },
        lib.callout('warn', 'Sigma means three things here', '<b>Image σ</b> is the fraction of the diffusion schedule (0.1 train, 0.2 OOD), not a standard deviation; it means a noise std of about 0.32 and 0.58 on [−1, 1] pixels. <b>Langevin σ</b> is the size of the noise η added to each thinking step (Eq. 2, p.7). <b>σ<sub>t</sub></b> in Fig E.1 is the diffusion sampler\'s noise scale.'),
        lib.callout('', 'Compute is not matched across setups', 'Scaling plots for text mostly use S1 EBTs with no landscape regularization; thinking results use S2 EBTs (p.30). One AR-EBT training step costs about 3.33× a Transformer++ step, 6.66× with two optimization steps (p.36). Image denoising compares 3 EBT forward passes with 300 DiT passes (p.35).')));
      const jepa = lib.callout('insight', 'For a JEPA reader', 'In the video setup the target is a <i>frozen</i> VAE latent, and the prediction ŷ lives in that same space. The energy scores (context latents, candidate latent) pairs, so the predictor is an optimizer over a fixed target space, not a learned joint embedding with an EMA target encoder. Collapse is not an issue because the targets never move. Swapping the SD-XL VAE for a learned JEPA encoder is an open experiment the paper does not run (the paper never mentions JEPA).');
      jepa.insertBefore(lib.badge('ext'), jepa.firstChild.nextSibling);
      P.appendChild(jepa);
      return {};
    }
  }
})();
