/* Chapter: takeaways ("What to take away").
   1) Claims, with receipts: each headline claim redrawn from the paper's own numbers (notes/paper_facts.md),
      with a Headline / Fine print toggle and a link to the lab that explores it.
   2) Costs and limits: the paper's FLOP accounting (p.35-36, p.41) as a live calculator, plus the limitations list.
   3) Open-questions lab (beyond the paper): can an EBT's energy tell a world model that its world changed?
      Runs the real toy 2D EBT (data/toy2d.json, final checkpoint) in the browser. Thinking = gradient descent on ŷ
      with exact gradients; surprise = E(x, y_obs) - E(x, ŷ*).
   4) Open questions for JEPA / world-model / continual-learning research (beyond the paper).
   5) Paper map: every figure, table, algorithm and listing with page, one-line meaning and chapter link.
   6) Searchable glossary (data/glossary.json when present, merged with a built-in list). */
(function () {
  'use strict';
  const redrawAll = []; // redraw hooks (after web fonts load)

  EBT.section({
    id: 'takeaways',
    nav: 'Takeaways',
    kicker: 'Wrap-up',
    title: 'What to take away',
    lede: 'What the evidence supports, what it costs, and what is still open if you build JEPA-style world models that keep learning. Every claim links to its figure or table and to the lab where you can test it. A lab on energy as a surprise signal, a map of every figure and table, and a searchable glossary close the page.',
    mount(el, lib) { build(el, lib); },
  });

  function build(el, lib) {
    const h = lib.h, C = lib.C;
    const GCOL = [C.truth, C.muted, C.rnn];               // context groups in the lab: precise, mid, noisy
    const GNAME = ['precise', 'mid', 'noisy'];
    const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

    // ---------- small helpers ----------
    // Canvas sized to its container in CSS pixels (so text is always drawn at its true size), redrawn on resize.
    function autoCanvas(parent, o) {
      const box = h('div', { class: 'takeaways-cbox' }); parent.appendChild(box);
      const c = h('canvas', { role: 'img', 'aria-label': o.label || '' }); box.appendChild(c);
      const ctx = c.getContext('2d');
      let W = 0, H = 0;
      function size() {
        const w = Math.max(240, Math.round(box.clientWidth || 600));
        const hh = Math.round(clamp(o.height ? o.height(w) : w * (o.aspect || 0.6), o.minH || 120, o.maxH || 2000));
        if (w === W && hh === H) return false;
        const dpr = Math.min(2, window.devicePixelRatio || 1);
        W = w; H = hh; c.width = Math.round(w * dpr); c.height = Math.round(hh * dpr); c.style.height = hh + 'px';
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0); return true;
      }
      const api = {
        canvas: c, ctx, box,
        get w() { return W; }, get h() { return H; },
        draw() { size(); ctx.save(); ctx.clearRect(0, 0, W, H); try { o.draw(ctx, W, H); } catch (e) { console.error(e); } ctx.restore(); },
        toLocal(ev) { const r = c.getBoundingClientRect(); return [(ev.clientX - r.left) / r.width * W, (ev.clientY - r.top) / r.height * H]; },
      };
      if (window.ResizeObserver) new ResizeObserver(() => { if (size()) api.draw(); }).observe(box);
      return api;
    }
    const T = (ctx, s, x, y, o) => lib.text(ctx, s, x, y, Object.assign({ size: 12, kind: 'mono', color: C.muted }, o || {}));
    const box = (w, hh, m) => ({ x: m.l, y: m.t, w: Math.max(40, w - m.l - m.r), h: Math.max(40, hh - m.t - m.b) });
    const go = (id, label) => h('a', { href: '#' + id, class: 'takeaways-go' }, label + ' →');
    const pct = (v) => (v == null || !isFinite(v)) ? '–' : Math.round(v * 100) + '%';
    function subsection(key, n, title, blurb) {
      const s = h('section', { class: 'takeaways-sub', id: 'takeaways-' + key, 'aria-labelledby': 'takeaways-' + key + '-h' });
      s.appendChild(h('header', {}, h('h3', { id: 'takeaways-' + key + '-h' }, h('span', { class: 'n' }, n), title), blurb ? h('p', { html: blurb }) : null));
      el.appendChild(s); return s;
    }
    function hline(ctx, x1, x2, y, col, o = {}) { ctx.save(); ctx.strokeStyle = col; ctx.lineWidth = o.w || 1.5; if (o.dash) ctx.setLineDash(o.dash); ctx.globalAlpha = o.alpha == null ? 1 : o.alpha; ctx.beginPath(); ctx.moveTo(x1, y); ctx.lineTo(x2, y); ctx.stroke(); ctx.restore(); }
    function vline(ctx, x, y1, y2, col, o = {}) { ctx.save(); ctx.strokeStyle = col; ctx.lineWidth = o.w || 1.5; if (o.dash) ctx.setLineDash(o.dash); ctx.globalAlpha = o.alpha == null ? 1 : o.alpha; ctx.beginPath(); ctx.moveTo(x, y1); ctx.lineTo(x, y2); ctx.stroke(); ctx.restore(); }
    function pill(ctx, s, x, y, col, o = {}) { // small label with a dark backing so it stays legible over lines
      const size = o.size || 12; const m = lib.measure(ctx, s, { size, kind: o.kind || 'mono' });
      const pw = m.w + 10, ph = size + 8; const xx = o.align === 'right' ? x - pw : o.align === 'center' ? x - pw / 2 : x;
      ctx.save(); ctx.fillStyle = lib.rgba(C.bg, 0.86); lib.rr(ctx, xx, y - ph / 2, pw, ph, 5); ctx.fill(); ctx.restore();
      T(ctx, s, xx + 5, y, { color: col, size, baseline: 'middle', kind: o.kind || 'mono' });
    }

    // ---------- in-chapter navigation ----------
    const subnav = h('nav', { class: 'takeaways-subnav', 'aria-label': 'Sections of this chapter' });
    [['claims', '1', 'Claims with receipts'], ['limits', '2', 'Costs and limits'], ['lab', '3', 'Surprise lab'], ['open', '4', 'Open questions'], ['map', '5', 'Paper map'], ['glossary', '6', 'Glossary']]
      .forEach(([k, n, t]) => subnav.appendChild(h('a', { href: '#takeaways-' + k }, h('b', {}, n), t)));
    el.appendChild(subnav);

    buildClaims();
    buildLimits();
    buildLab();
    buildOpen();
    buildMap();
    buildGlossary();
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => redrawAll.forEach(f => f()));

    // =====================================================================================
    // 1. CLAIMS WITH RECEIPTS
    // =====================================================================================
    function buildClaims() {
      const s = subsection('claims', '1', 'Claims, with receipts', 'Pick a claim. <b>Headline</b> shows what the paper reports. <b>Fine print</b> adds the context that keeps it honest. Numbers come from the paper\'s text and tables; values marked <i>approx.</i> were read off its figures.');
      s.appendChild(lib.callout('insight', 'Why it might work, in one paragraph', 'The authors attribute the gains to EBTs <b>learning to verify, not only to predict</b> (p.15). Checking an answer is easier than producing one, and a verifier transfers better to harder problems (Sec 2.1 to 2.2, p.5 to 6). Turning prediction into descent on a learned verifier then gives three things a feed-forward net lacks (Table 1, p.3): compute that can grow per prediction, an energy that signals uncertainty, and a built-in way to pick the best of several guesses.'));

      const CL = claimData();
      let cur = 'think', mode = 'head';
      const bench = h('div', { class: 'bench takeaways-claims' });
      const list = h('div', { class: 'takeaways-claimlist', role: 'group', 'aria-label': 'Claims' });
      const card = h('div', { class: 'card takeaways-evidence' });
      bench.append(list, card); s.appendChild(bench);

      const btns = {};
      CL.forEach((c, i) => {
        const b = h('button', { type: 'button', 'aria-pressed': String(c.id === cur) }, h('span', { class: 'k' }, String(i + 1).padStart(2, '0')), h('span', { class: 't' }, c.title), h('small', {}, c.short));
        b.addEventListener('click', () => { cur = c.id; Object.entries(btns).forEach(([k, bb]) => bb.setAttribute('aria-pressed', String(k === cur))); render(); });
        btns[c.id] = b; list.appendChild(b);
      });

      const title = h('h3', { class: 'takeaways-ev-title' });
      const badges = h('div', { class: 'row takeaways-ev-badges' });
      const seg = lib.segmented({ label: 'Headline or fine print', options: [['head', 'Headline'], ['fine', 'Fine print']], value: mode, onchange: (v) => { mode = v; render(); } });
      const visual = h('div', { class: 'takeaways-ev-visual' });
      const cv = autoCanvas(visual, { label: 'Evidence chart for the selected claim', height: (w) => w < 520 ? w * 0.86 : w * 0.46, minH: 260, maxH: 400, draw: (ctx, w, hh) => { const c = CL.find(k => k.id === cur); if (c.draw) c.draw(ctx, w, hh, mode === 'fine'); } });
      const fig = h('figure', { class: 'takeaways-ev-fig', hidden: true }, h('div', { class: 'paper-fig' }, h('img', { alt: '' })), h('figcaption', {}));
      visual.appendChild(fig);
      const txt = h('p', { class: 'takeaways-ev-text', 'aria-live': 'polite' });
      const why = h('p', { class: 'takeaways-ev-why' });
      const foot = h('div', { class: 'row takeaways-ev-foot' });
      card.append(h('div', { class: 'takeaways-ev-head' }, title, seg.el), badges, visual, txt, why, foot);

      function render() {
        const c = CL.find(k => k.id === cur);
        title.textContent = c.title;
        badges.replaceChildren(lib.badge('paper', c.ev + (c.approx ? ', approx.' : '')), h('span', { class: 'takeaways-chip' }, c.pages));
        if (c.img) {
          cv.box.hidden = true; fig.hidden = false;
          const im = mode === 'fine' && c.imgFine ? c.imgFine : c.img;
          fig.querySelector('img').src = im.src; fig.querySelector('img').alt = im.alt; fig.querySelector('figcaption').textContent = im.cap;
        } else { cv.box.hidden = false; fig.hidden = true; cv.draw(); }
        txt.replaceChildren(h('b', {}, mode === 'fine' ? 'Fine print. ' : 'What the paper reports. '), document.createTextNode(mode === 'fine' ? c.fine : c.head));
        why.replaceChildren(h('b', {}, 'For a world-model builder. '), document.createTextNode(c.why));
        foot.replaceChildren(h('span', {}, 'Explore: '), go(c.link[0], c.link[1]));
      }
      render();
      redrawAll.push(() => { if (!cv.box.hidden) cv.draw(); });
    }

    function claimData() {
      const fs = 12;
      return [
        { id: 'think', title: 'Thinking longer helps. Transformer++ cannot.', short: 'Up to 29% lower OOD perplexity from extra passes', ev: 'Fig 6a', approx: true, pages: 'p.10 to 11', link: ['thinking', 'Text thinking lab'],
          head: 'On four out-of-distribution datasets, extra forward passes per token cut the EBT\'s perplexity increase by up to 29%: first one more optimization step, then Best-of-N over candidates. A Transformer++ spends exactly one pass per token, so its line is flat (p.10).',
          fine: 'The 29% is measured against the EBT\'s own no-thinking start, which is worse than Transformer++ (about 44.8 vs 38.4). One extra step already beats it (about 35.9). Values are read off Fig 6a. Each EBT pass also needs a backward pass for ∇ŷE, so passes are not equal in FLOPs.',
          why: 'Inference compute becomes a dial you can turn per prediction instead of a fixed cost set by depth.',
          draw(ctx, w, hh, fine) {
            const b = box(w, hh, { l: 76, r: 16, t: 26, b: 52 });
            const ax = lib.axes(ctx, Object.assign({}, b, { xlim: [1.7, 40], ylim: [28, 48], xlog: true, xticks: [2, 3, 6, 15, 30], yticks: [30, 35, 40, 45], xlabel: 'Forward passes per token (log scale)', ylabel: 'OOD ppl increase ↓', size: fs }));
            T(ctx, w < 520 ? 'Fig 6a, OOD mean (approx.)' : 'Fig 6a, mean over four OOD datasets (approx.)', b.x, 4, { size: 12 });
            hline(ctx, b.x, b.x + b.w, ax.Y(38.4), C.ar, { w: 2.5 });
            pill(ctx, w < 520 ? 'T++ ≈38.4, 1 pass' : 'Transformer++ ≈38.4, one pass only', ax.X(30) - 10, ax.Y(38.4) + 14, C.ar, { align: 'right', size: 12 });
            const P = [[2, 44.8], [3, 35.9], [6, 32.5], [15, 31.9], [30, 31.8]];
            lib.plot(ctx, ax, P, { color: C.ebt, width: 2.5, markers: 4.5 });
            pill(ctx, 'EBT, no thinking', ax.X(2) + 8, ax.Y(44.8), C.ebt, { size: 12 });
            pill(ctx, 'think longer', ax.X(3) + 8, ax.Y(35.9) + 2, C.ebt, { size: 12 });
            pill(ctx, '+ self-verify (BoN)', ax.X(6) - 4, ax.Y(32.5) + 18, C.ebt, { size: 12 });
            if (!fine) {
              const x = ax.X(30) + 2; vline(ctx, x, ax.Y(44.8), ax.Y(31.8), C.ink, { w: 1.5 }); hline(ctx, ax.X(2), x, ax.Y(44.8), C.ink, { dash: [3, 4], alpha: 0.5 });
              pill(ctx, '−29%', x - 6, ax.Y(42), C.ink, { align: 'right', size: 14 });
            } else {
              const x = ax.X(2); ctx.save(); ctx.fillStyle = lib.rgba(C.bad, 0.18); ctx.fillRect(x - 7, ax.Y(44.8), 14, ax.Y(38.4) - ax.Y(44.8)); ctx.restore();
              vline(ctx, x, ax.Y(44.8), ax.Y(38.4), C.bad, { w: 2 });
              pill(ctx, 'starts worse than T++', x + 10, (ax.Y(44.8) + ax.Y(38.4)) / 2 + 8, C.bad, { size: 12 });
              lib.dot(ctx, ax.X(3), ax.Y(35.9), 8, 'transparent', { stroke: C.truth, lw: 2 });
              pill(ctx, '1 extra step beats T++', ax.X(3) + 10, ax.Y(35.9) + 22, C.truth, { size: 12 });
            }
          } },
        { id: 'ood', title: 'Thinking helps most on unfamiliar data', short: 'Gain grows roughly linearly with distribution shift', ev: 'Fig 7', approx: true, pages: 'p.11', link: ['thinking', 'Text thinking lab'],
          head: 'Across the pretraining set and four downstream sets, the improvement from maximum thinking (longer plus self-verification) rises from about 12% to about 23% as the data moves further from pretraining. The authors call it a strong linear trend (p.11).',
          fine: 'There are five points, one per dataset, and the figure does not label which is which. The shift = 1.0 point is the pretraining data itself. "OOD shift" is a perplexity ratio (downstream ÷ pretraining), not a distance between distributions (p.11).',
          why: 'This is the regime a deployed world model lives in. Which kinds of shift thinking can absorb is open; the surprise lab below shows one kind it cannot.',
          draw(ctx, w, hh, fine) {
            const b = box(w, hh, { l: 76, r: 18, t: 26, b: 52 });
            const ax = lib.axes(ctx, Object.assign({}, b, { xlim: [0.7, 4.7], ylim: [8, 26], xticks: [1, 2, 3, 4], yticks: [10, 15, 20, 25], xlabel: 'OOD shift = downstream ppl ÷ pretraining ppl', ylabel: '% ppl gain ↑', yfmt: v => v + '%', size: fs }));
            T(ctx, 'Fig 7, EBT with max thinking (approx.)', b.x, 4, { size: 12 });
            const P = [[1.0, 11.8], [1.31, 12.4], [1.77, 16.3], [2.32, 16.1], [4.37, 23.0]];
            const n = P.length, mx = P.reduce((a, p) => a + p[0], 0) / n, my = P.reduce((a, p) => a + p[1], 0) / n;
            const sl = P.reduce((a, p) => a + (p[0] - mx) * (p[1] - my), 0) / P.reduce((a, p) => a + (p[0] - mx) ** 2, 0), ic = my - sl * mx;
            lib.plot(ctx, ax, [[0.8, ic + sl * 0.8], [4.6, ic + sl * 4.6]], { color: C.ebt, width: 1.5, dash: [6, 5], alpha: 0.7 });
            P.forEach(p => lib.dot(ctx, ax.X(p[0]), ax.Y(p[1]), 6, C.ebt, { stroke: C.bg, lw: 2 }));
            pill(ctx, 'fit: +' + sl.toFixed(1) + ' points per unit of shift (our fit)', b.x + 6, b.y + 14, C.ebt, { size: 12 });
            if (fine) {
              lib.dot(ctx, ax.X(1), ax.Y(11.8), 11, 'transparent', { stroke: C.truth, lw: 2 });
              pill(ctx, 'pretraining data itself', ax.X(1) + 14, ax.Y(11.8) + 18, C.truth, { size: 12 });
              pill(ctx, '4 downstream sets, unlabeled', ax.X(2.6), ax.Y(19.5), C.muted, { size: 12, align: 'center' });
            }
          } },
        { id: 'learn', title: 'Learns faster as data and batch grow', short: 'Up to 35% higher scaling rate than Transformer++', ev: 'Figs 4, 5, 9', approx: false, pages: 'p.9 to 13', link: ['scaling', 'Scaling laws explorer'],
          head: 'On every axis tried, EBT perplexity or loss falls faster with scale than Transformer++. The largest gaps are data (35.98%) and batch size (28.46%) for text, and more than 33% for video width and parameters (figure titles, Figs 4, 5, 9).',
          fine: 'Faster means a steeper fitted slope, not lower loss. At the sizes actually trained, EBT is behind on depth, parameters, FLOPs and both video axes and ahead only on data and batch. "Would win at foundation scale" extrapolates these slopes (p.9, p.12). The paper never formally defines scaling rate.',
          why: 'The paper argues data efficiency matters because data is becoming the bottleneck (p.14 to 15). For embodied world models it already is.',
          draw(ctx, w, hh, fine) {
            const rows = [['Text · data', 35.98, '4a', 'ahead'], ['Text · batch', 28.46, '4b', 'ahead'], ['Text · depth', 5.29, '4c', 'behind'], ['Text · FLOPs', 2.92, '5b', 'behind'], ['Text · params', 2.91, '5a', 'behind'], ['Text · width', 0.02, '5c', '≈ tie'], ['Video · width', 33.66, '9a', 'behind'], ['Video · params', 34.28, '9b', 'behind']];
            const lw = w < 460 ? 100 : 128, rw = fine ? (w < 460 ? 62 : 120) : 8;
            const b = box(w, hh, { l: lw, r: rw, t: 30, b: 30 });
            T(ctx, 'Scaling-rate gain over Transformer++ (figure titles)', 0, 4, { size: 12 });
            const X = (v) => b.x + v / 40 * b.w, rh = b.h / rows.length;
            [0, 10, 20, 30, 40].forEach(v => { vline(ctx, X(v), b.y, b.y + b.h, C.rule, { w: 1 }); T(ctx, v + '%', X(v), b.y + b.h + 8, { align: 'center', size: 11.5 }); });
            if (fine) T(ctx, w < 460 ? 'lower loss?' : 'lower loss at largest size?', w - 2, 4 + 15, { align: 'right', size: 11.5, color: C.faint });
            rows.forEach((r, i) => {
              const y = b.y + i * rh, bh = Math.min(18, rh * 0.62), yc = y + rh / 2;
              T(ctx, r[0], b.x - 10, yc, { align: 'right', baseline: 'middle', color: C.ink, size: w < 460 ? 11.5 : 12.5 });
              ctx.fillStyle = r[0].startsWith('Video') ? lib.rgba(C.ebt, 0.6) : C.ebt; ctx.fillRect(b.x, yc - bh / 2, Math.max(1.5, X(r[1]) - b.x), bh);
              T(ctx, r[1].toFixed(2) + '%', Math.max(X(r[1]), b.x) + 6, yc, { baseline: 'middle', size: 12, color: C.ink });
              if (fine) T(ctx, r[3], w - 4, yc, { align: 'right', baseline: 'middle', size: 12, color: r[3] === 'ahead' ? C.truth : r[3] === 'behind' ? C.bad : C.muted });
            });
          } },
        { id: 'general', title: 'Generalizes better despite worse pretraining', short: 'Lower perplexity on 3 of 4 downstream tasks', ev: 'Table 3', approx: false, pages: 'p.11 to 12', link: ['data', 'Data and evaluation'],
          head: 'With identical training setups, EBT has lower perplexity than Transformer++ on GSM8K (43.3 vs 49.6), BigBench Math QA (72.6 vs 79.8) and Dyck languages (125.3 vs 131.5), although its pretraining perplexity is higher (33.43 vs 31.36).',
          fine: 'It is slightly worse on SQuAD (53.1 vs 52.3), so "most tasks", not all. Relative differences are modest: −4.7% to −12.7% where EBT wins (our arithmetic). The paper does not say whether these EBT numbers use extra thinking steps or Best-of-N.',
          why: 'Fitting the training data a little worse but transferring better is usually the trade you want in a model that must face new situations.',
          draw(ctx, w, hh, fine) {
            const rows = [['Pretraining', 31.36, 33.43], ['GSM8K', 49.6, 43.3], ['SQuAD', 52.3, 53.1], ['BB Math QA', 79.8, 72.6], ['BB Dyck', 131.5, 125.3]];
            const lw = w < 460 ? 88 : 110; const b = box(w, hh, { l: lw, r: 64, t: 30, b: 30 });
            const rh = b.h / rows.length;
            if (!fine) {
              T(ctx, 'Table 3 perplexity, lower is better', 0, 4, { size: 12 });
              const X = (v) => b.x + v / 140 * b.w;
              [0, 35, 70, 105, 140].forEach(v => { vline(ctx, X(v), b.y, b.y + b.h, C.rule, { w: 1 }); T(ctx, String(v), X(v), b.y + b.h + 8, { align: 'center', size: 11.5 }); });
              rows.forEach((r, i) => {
                const yc = b.y + i * rh + rh / 2, bh = Math.min(13, rh * 0.3);
                T(ctx, r[0], b.x - 10, yc, { align: 'right', baseline: 'middle', color: C.ink, size: 12.5 });
                [[r[1], C.ar, -1], [r[2], C.ebt, 1]].forEach(([v, col, s]) => { const y = yc + s * (bh / 2 + 1); ctx.fillStyle = col; ctx.fillRect(b.x, y - bh / 2, X(v) - b.x, bh); T(ctx, String(v), X(v) + 5, y, { baseline: 'middle', size: 11.5, color: col }); });
              });
              const lx = b.x + b.w + 60; T(ctx, '■ T++', lx, 4, { align: 'right', color: C.ar, size: 12 }); T(ctx, '■ EBT', lx - 64, 4, { align: 'right', color: C.ebt, size: 12 });
            } else {
              T(ctx, 'EBT vs T++, relative difference (our arithmetic)', 0, 4, { size: 12 });
              const X = (v) => b.x + (v + 15) / 25 * b.w;
              [-15, -10, -5, 0, 5, 10].forEach(v => { vline(ctx, X(v), b.y, b.y + b.h, v === 0 ? C.faint : C.rule, { w: v === 0 ? 2 : 1 }); T(ctx, (v > 0 ? '+' : '') + v + '%', X(v), b.y + b.h + 8, { align: 'center', size: 11.5 }); });
              rows.forEach((r, i) => {
                const yc = b.y + i * rh + rh / 2, bh = Math.min(18, rh * 0.5), d = (r[2] - r[1]) / r[1] * 100;
                T(ctx, r[0], b.x - 10, yc, { align: 'right', baseline: 'middle', color: C.ink, size: 12.5 });
                ctx.fillStyle = d < 0 ? C.ebt : C.bad; const x0 = X(0), x1 = X(d); ctx.fillRect(Math.min(x0, x1), yc - bh / 2, Math.abs(x1 - x0), bh);
                T(ctx, (d > 0 ? '+' : '') + d.toFixed(1) + '%' + (d > 0 ? ' worse' : ''), d < 0 ? x0 + 6 : x1 + 6, yc, { baseline: 'middle', size: 12, color: d < 0 ? C.ink : C.bad });
              });
              T(ctx, '← EBT better', X(-15), b.y - 14, { size: 11.5, color: C.ebt }); T(ctx, 'EBT worse →', X(10), b.y - 14, { size: 11.5, color: C.bad, align: 'right' });
            }
          } },
        { id: 'images', title: 'Beats a diffusion model with 1% of the passes', short: 'Denoising: 3 EBT passes vs 300 DiT passes', ev: 'Table 4, Fig 12', approx: true, pages: 'p.13 to 15, p.35', link: ['images', 'Denoising lab'],
          head: 'On out-of-distribution noise (σ = 0.2), EBT reaches 23.29 PSNR in 3 forward passes; DiT reaches 19.56 in 300 (Table 4, Fig 12). In-distribution (σ = 0.1) EBT also wins, narrowly: 27.25 vs 26.58.',
          fine: 'Each EBT pass is a forward plus a backward pass for ∇ŷE, so the compute saving is smaller than 100×. Both models were applied to their own output three times at test time (p.35). The linear-probe gap (5.32% vs 0.31% top-1) is large as a ratio but tiny in absolute accuracy.',
          why: 'A bidirectional EBT scores a whole image at once, the same shape of problem as scoring a predicted latent frame.',
          draw(ctx, w, hh, fine) {
            const b = box(w, hh, { l: 70, r: 18, t: 26, b: 52 });
            const ax = lib.axes(ctx, Object.assign({}, b, { xlim: [0.8, 420], ylim: [12, 29], xlog: true, xticks: [1, 3, 10, 30, 100, 300], yticks: [14, 18, 22, 26], xlabel: 'Forward passes (log scale)', ylabel: 'PSNR ↑', size: fs }));
            T(ctx, 'Fig 12 (approx.) and Table 4, OOD σ = 0.2', b.x, 4, { size: 12 });
            const E = [[1, 13.9], [2, 18.8], [3, 23.0]], Dd = [[100, 14.3], [200, 18.9], [300, 19.6]];
            lib.plot(ctx, ax, Dd, { color: C.diff, width: 2.5, markers: 4.5 });
            lib.plot(ctx, ax, E, { color: C.ebt, width: 2.5, markers: 4.5 });
            pill(ctx, 'EBT, 3 passes: 23.29', ax.X(3) + 8, ax.Y(23.0) - 2, C.ebt, { size: 12 });
            pill(ctx, 'DiT, 300 passes: 19.56', ax.X(300) - 4, ax.Y(19.6) - 18, C.diff, { size: 12, align: 'right' });
            if (fine) {
              hline(ctx, b.x, b.x + b.w, ax.Y(27.25), C.ebt, { dash: [5, 4], alpha: 0.8 }); hline(ctx, b.x, b.x + b.w, ax.Y(26.58), C.diff, { dash: [5, 4], alpha: 0.8 });
              pill(ctx, 'in-distribution σ = 0.1: EBT 27.25 vs DiT 26.58', b.x + 6, ax.Y(27.25) - 12, C.muted, { size: 12 });
              pill(ctx, 'each EBT pass = forward + backward', ax.X(1.05), ax.Y(15.4), C.bad, { size: 12 });
            }
          } },
        { id: 'verify', title: 'Self-verification improves with training', short: 'Best-of-5 gain grows from 4–8% to 10–14%', ev: 'Fig 6b, Fig B.1', approx: true, pages: 'p.10, p.28, p.34', link: ['scaling', 'Scaling laws explorer'],
          head: 'As the EBT trains on more tokens, choosing the lowest-energy of 5 candidates helps more: the gain grows from 4–8% to 10–14% (p.10). The verifier gets better at telling good predictions from bad ones.',
          fine: 'Measured on Dyck only; other benchmarks did not show the trend (p.34). With little data, Best-of-10 was sometimes worse than Best-of-2: the model found low-energy samples that were wrong (p.28). The >1000% point in Fig B.1b is an extrapolation to 15T tokens.',
          why: 'Any planner that ranks futures by energy inherits this: a weak verifier can be gamed (open question 6).',
          draw(ctx, w, hh, fine) {
            const b = box(w, hh, { l: 70, r: 18, t: 26, b: 52 });
            const ax = lib.axes(ctx, Object.assign({}, b, { xlim: [0, 34], ylim: [0, 16], xticks: [0, 10, 20, 30], yticks: [0, 4, 8, 12, 16], xlabel: 'Tokens trained on (billions)', ylabel: '% ppl gain from BoN', yfmt: v => v + '%', size: fs }));
            T(ctx, 'Fig 6b, BoN-5 on Dyck (bands from text, line approx.)', b.x, 4, { size: 12 });
            ctx.save(); ctx.fillStyle = lib.rgba(C.ebt, 0.14); ctx.fillRect(ax.X(0.5), ax.Y(8), ax.X(8) - ax.X(0.5), ax.Y(4) - ax.Y(8)); ctx.fillRect(ax.X(25), ax.Y(14), ax.X(33) - ax.X(25), ax.Y(10) - ax.Y(14)); ctx.restore();
            pill(ctx, 'early: 4–8%', ax.X(0.8), ax.Y(8) - 12, C.ebt, { size: 12 }); pill(ctx, 'late: 10–14%', ax.X(33), ax.Y(14) - 12, C.ebt, { size: 12, align: 'right' });
            lib.plot(ctx, ax, [[0.5, 7.9], [33, 10.6]], { color: C.ebt, width: 2.5 });
            pill(ctx, 'fit ≈7.9% → 10.6%', ax.X(12), ax.Y(9.4) + 18, C.ebt, { size: 12 });
            if (fine) {
              lib.plot(ctx, ax, [[3, 1.1], [33, 2.2]], { color: C.truth, width: 2, dash: [6, 4] });
              pill(ctx, 'Fig B.1a: BoN-10 vs BoN-2, RedPajama val ≈1.1% → 2.2%', ax.X(3), ax.Y(2.2) - 16, C.truth, { size: 11.5 });
              pill(ctx, 'Dyck only (p.34)', ax.X(33), ax.Y(5.5), C.bad, { size: 12, align: 'right' });
            }
          } },
        { id: 'regs', title: 'The landscape regularizers make thinking work', short: 'Random step size is the one you cannot drop', ev: 'Table 2', approx: false, pages: 'p.7 to 8, p.10', link: ['training', 'Training lab'],
          head: 'On OOD Dyck, the full recipe (replay buffer, Langevin noise, random step size, random number of steps) gives an 18.7% gain from thinking longer plus self-verification. Without a random step size the gain nearly vanishes (−1.47% and 0.19%).',
          fine: 'The full recipe wins only in the combined column. For thinking longer alone it gives 7.19%, below "no Langevin" (17.2%) and "no replay buffer" (14.8%): noise trades single-path quality for a landscape self-verification can exploit (p.10). Without random step counts, thinking longer gives exactly 0.00.',
          why: 'Exploration during training shapes the regions that inference, or a planner, will later visit. Regions never visited stay undefined.',
          draw(ctx, w, hh, fine) {
            const rows = [['No random step size', -1.47, 0.19], ['No random # steps', 0.0, 9.65], ['No Langevin', 17.2, 17.0], ['No replay buffer', 14.8, 17.8], ['Full S2 recipe', 7.19, 18.7]];
            const lw = w < 460 ? 118 : 150; const b = box(w, hh, { l: lw, r: 50, t: 40, b: 30 });
            const X = (v) => b.x + (v + 2) / 22 * b.w, rh = b.h / rows.length;
            T(ctx, 'Table 2: % ppl improvement on OOD Dyck', 0, 4, { size: 12 });
            T(ctx, '■ thinking longer', 0, 21, { size: 12, color: lib.rgba(C.ebt, 0.95) }); T(ctx, '■ longer + self-verify', w < 460 ? 140 : 150, 21, { size: 12, color: C.truth });
            [0, 5, 10, 15, 20].forEach(v => { vline(ctx, X(v), b.y, b.y + b.h, v === 0 ? C.faint : C.rule, { w: v === 0 ? 2 : 1 }); T(ctx, v + '%', X(v), b.y + b.h + 8, { align: 'center', size: 11.5 }); });
            rows.forEach((r, i) => {
              const yc = b.y + i * rh + rh / 2, bh = Math.min(13, rh * 0.32), full = i === 4;
              T(ctx, r[0], b.x - 10, yc, { align: 'right', baseline: 'middle', color: full ? C.ink : C.muted, size: w < 460 ? 11.5 : 12.5, weight: full ? 700 : 400, kind: 'body' });
              const hl = fine ? (i === 2 || i === 4) : (i === 0 || i === 4);
              [[r[1], C.ebt, -1], [r[2], C.truth, 1]].forEach(([v, col, s]) => {
                const y = yc + s * (bh / 2 + 1); ctx.save(); ctx.globalAlpha = hl ? 1 : 0.45; ctx.fillStyle = col; const x0 = X(0), x1 = X(v); ctx.fillRect(Math.min(x0, x1), y - bh / 2, Math.max(1.5, Math.abs(x1 - x0)), bh); ctx.restore();
                T(ctx, v === 0 ? '0.00' : String(v).replace('-', '−'), Math.max(x1, X(0)) + 5, y, { baseline: 'middle', size: 11.5, color: hl ? col : C.faint });
              });
            });
          } },
        { id: 'uncert', title: 'Uncertainty comes for free', short: 'Hard tokens and unpredictable frames keep high energy', ev: 'Figs 8, 11, B.2', approx: false, pages: 'p.10 to 14, p.28 to 29', link: ['thinking', 'Text thinking lab'],
          img: { src: 'media/paper/fig08.png', alt: 'Paper Figure 8: heatmaps of normalized token energy across 12 thinking iterations for two sentences', cap: 'Paper Fig 8 (p.12): normalized energy per token (rows) across thinking iterations (columns). Yellow = low energy.' },
          imgFine: { src: 'media/paper/figB2.png', alt: 'Paper Figure B.2: energy of a familiar sentence versus a random token sequence across iterations', cap: 'Paper Fig B.2 (p.29): a familiar sentence (left) keeps lower energy than random tokens (right). Yellow = low energy.' },
          head: 'Without any uncertainty labels, easy tokens ("the", ".", "is") reach low energy within one iteration while hard ones ("quick", "brown", "problem") stay high (Fig 8). In video, frames have high energy while the scene is unpredictable (Fig 11).',
          fine: 'These are hand-picked sequences shown as "normalized energy" with an unspecified normalization. Fig B.2 (shown) adds an epistemic check: random tokens stay higher than familiar text. In our toy model, the energy level after thinking tracks data noise only weakly (r ≈ 0.45 across contexts), and the basin width barely changes at all (surprise lab, below).',
          why: 'This is the property a continual learner wants as a novelty or "learn now" signal. The surprise lab below tests how far it goes.' },
      ];
    }

    // =====================================================================================
    // 2. COSTS AND LIMITS
    // =====================================================================================
    function buildLimits() {
      const s = subsection('limits', '2', 'What it costs, and where it breaks', 'The paper\'s own limitations (Sec 7, p.17, and the appendix), and its FLOP accounting turned into a calculator.');
      const grid = h('div', { class: 'takeaways-limits' }); s.appendChild(grid);

      // calculator
      const calc = h('div', { class: 'card stack takeaways-calc' });
      calc.append(h('div', { class: 'row', style: 'justify-content:space-between' }, h('h4', { style: 'margin:0' }, 'Training compute per example'), lib.badge('paper', 'p.35 to 36, p.41')));
      calc.appendChild(h('p', { class: 'takeaways-small', html: 'Per token, a Transformer++ step costs forward <b>2N</b> + backward <b>4N</b> FLOPs (N = parameters). An EBT optimization step adds a Hessian-vector product, about one more backward: <b>2N + 4N + 4N</b>. The autoregressive EBT also doubles the sequence (observed + predicted states), and S1 models pay this for every step (p.35 to 36).' }));
      let steps = 2, mode = 'ar';
      const sl = lib.slider({ id: 'takeaways-steps', label: 'Optimization steps per training example', min: 1, max: 6, step: 1, value: steps, fmt: v => String(v), oninput: v => { steps = v; upd(); } });
      const seg = lib.segmented({ label: 'Sequence layout', options: [['ar', 'Autoregressive (2× sequence)'], ['single', 'Single sequence']], value: mode, onchange: v => { mode = v; upd(); } });
      calc.append(sl.el, h('div', { class: 'row' }, seg.el));
      const cv = autoCanvas(calc, { label: 'Stacked bars comparing FLOPs of one Transformer++ training step with one EBT training step', height: (w) => w < 440 ? 200 : 176, minH: 170, maxH: 210, draw: drawBill });
      const big = h('div', { class: 'takeaways-bill' });
      const note = h('p', { class: 'takeaways-small', 'aria-live': 'polite' });
      calc.append(big, note);
      calc.appendChild(h('p', { class: 'takeaways-small', html: 'S2 (thinking) models truncate backprop and add random step counts and a replay buffer, so their cost varies (p.36). Fig 5b already charges EBT for this: its points sit about 6 to 7× further right, and EBT is still behind at equal FLOPs in the measured range.' }));
      const PAPER = { '1single': '1.66× (p.41)', '1ar': '3.33× (p.36)', '2ar': '6.66× (p.36)' };
      function mult() { return steps * 10 * (mode === 'ar' ? 2 : 1) / 6; }
      function upd() {
        const m = Math.floor(mult() * 100 + 1e-9) / 100; const key = steps + mode;
        big.replaceChildren(h('b', {}, '≈' + m.toFixed(2) + '×'), h('span', {}, ' the FLOPs of a Transformer++ step'));
        note.innerHTML = PAPER[key] ? 'The paper states this value: <b>' + PAPER[key] + '</b>.' : 'Not stated in the paper for this setting. Same accounting as p.36 (cost multiplies by steps for S1): our arithmetic.';
        cv.draw();
      }
      function drawBill(ctx, w, hh) {
        const narrow = w < 440, seq = mode === 'ar' ? 2 : 1, tot = steps * 10 * seq, lw = narrow ? 44 : 104;
        const COL = { F: lib.CMAP[2], B: lib.CMAP[3], H: lib.CMAP[4] };
        const legend = [['F', 'forward 2N'], ['B', 'backward 4N'], ['H', 'Hessian-vector 4N']]; let lx = 0, ly = 4;
        legend.forEach(([k, t]) => { const tw = lib.measure(ctx, t, { size: 11.5, kind: 'mono' }).w; if (lx > 0 && lx + 14 + tw > w) { lx = 0; ly += 18; } ctx.fillStyle = COL[k]; ctx.fillRect(lx, ly + 2, 10, 10); lx += 14; T(ctx, t, lx, ly, { size: 11.5 }); lx += tw + 14; });
        const b = box(w, hh, { l: lw, r: 52, t: ly + 26, b: 30 }), unit = b.w / Math.max(tot, 6);
        const rowY = [b.y + 4, b.y + b.h * 0.5], bh = Math.min(30, b.h * 0.3);
        T(ctx, narrow ? 'T++' : 'Transformer++', b.x - 10, rowY[0] + bh / 2, { align: 'right', baseline: 'middle', color: C.ar, size: 12.5, kind: 'body', weight: 700 });
        T(ctx, 'EBT', b.x - 10, rowY[1] + bh / 2, { align: 'right', baseline: 'middle', color: C.ebt, size: 12.5, kind: 'body', weight: 700 });
        let x = b.x; [['F', 2], ['B', 4]].forEach(([k, v]) => { ctx.fillStyle = COL[k]; ctx.fillRect(x, rowY[0], v * unit, bh); x += v * unit; });
        T(ctx, '6N', x + 6, rowY[0] + bh / 2, { baseline: 'middle', size: 12, color: C.ink });
        x = b.x;
        for (let i = 0; i < steps; i++) for (let q = 0; q < seq; q++) {
          [['F', 2], ['B', 4], ['H', 4]].forEach(([k, v]) => { ctx.fillStyle = COL[k]; ctx.fillRect(x, rowY[1], v * unit, bh); x += v * unit; });
          ctx.fillStyle = C.bg; ctx.fillRect(x - 1, rowY[1], 2, bh);
        }
        T(ctx, tot + 'N', Math.min(x + 6, w - 46), rowY[1] + bh / 2, { baseline: 'middle', size: 12, color: C.ink });
        T(ctx, steps + ' step' + (steps > 1 ? 's' : '') + (seq > 1 ? ' × 2 sequence halves' : ''), b.x, rowY[1] + bh + 8, { size: 11.5, color: C.faint });
      }
      upd();
      redrawAll.push(() => cv.draw());

      // limitations list
      const LIM = [
        ['Stability', 'Extra hyperparameters', 'Step size and number of steps must be tuned; poor values often make training unstable.', 'p.17'],
        ['Stability', 'Only 2 to 3 training steps', 'More optimization steps per training example were not stable, so the optimization seen in training stays short.', 'p.26'],
        ['Compute', 'Every step costs more', 'Second-order training is about 3.33× a Transformer++ step per optimization step, 6.66× with 2 steps. Thinking at inference also needs a gradient per step.', 'p.17, p.27, p.36'],
        ['Scale', 'Small by today\'s standards', 'Up to about 800M parameters (largest config 708M non-embedding) and under 10^21 FLOPs (about 1300 A100 hours). Wins at foundation scale are extrapolated.', 'p.17, p.27, p.33'],
        ['Modeling', 'Many modes blur together', 'Training makes the basin convex around each target, so data with many modes per condition gets averaged, as in blurry small-scale text-to-image samples.', 'p.17, p.27, p.29'],
        ['Data', 'Thinking needs data', 'Thinking gains emerge only with enough training data. With little data, Best-of-10 can be worse than Best-of-2 (wrong samples with low energy).', 'p.9, p.28'],
        ['Evidence', 'Some trends are narrow', 'The growing Best-of-N gain (Fig 6b) appeared only on Dyck; FLOP estimates are approximate; baseline hyperparameters were tuned for Transformers, not EBTs.', 'p.33, p.34, p.36'],
        ['Framing', 'Not human System 2', 'No chain-of-thought benefit at this size, and the authors do not claim the models replicate human System 2 thinking.', 'p.10, p.39'],
      ];
      const ul = h('ul', { class: 'takeaways-limlist' });
      LIM.forEach(([tag, t, d, p]) => ul.appendChild(h('li', {}, h('span', { class: 'takeaways-tag', 'data-tag': tag }, tag), h('div', {}, h('b', {}, t + '. '), d, ' ', h('span', { class: 'takeaways-pg' }, p)))));
      const limCard = h('div', { class: 'card' }, h('div', { class: 'row', style: 'justify-content:space-between;margin-bottom:6px' }, h('h4', { style: 'margin:0' }, 'Limitations the paper states'), lib.badge('paper', 'Sec 7 and appendix')), ul);
      grid.append(limCard, calc);
    }

    // =====================================================================================
    // 3. SURPRISE LAB (beyond the paper, toy model)
    // =====================================================================================
    function buildLab() {
      const s = subsection('lab', '3', 'Surprise lab: can energy tell a world model that its world changed?', 'A model that keeps learning must first notice that something is new. The paper shows energy tracks uncertainty (Figs 8, 11, B.2) but never tests it as a novelty detector. Here the real toy EBT from the <a href="#landscape">Energy landscape lab</a> watches a stream of observations, and at t = 80 the world changes.');
      s.appendChild(h('div', { class: 'row' }, lib.badge('ext'), lib.badge('toy', 'final checkpoint')));
      const T2 = EBT.toy2d;
      if (!T2 || !T2.ready) { s.appendChild(lib.callout('warn', 'Toy model missing', 'data/toy2d.json is not in data/bundle.js. Run <code>python3 src/bundle_data.py</code>.')); return; }

      // ---------- fast exact evaluator of the toy EBT (same weights and formula as EBT.toy2d) ----------
      const D2 = T2.data, ck = D2.weights.length - 1, LAY = D2.weights[ck].layers;
      const LAM = (D2.arch.quadratic_term && D2.arch.quadratic_term.lambda_) || 0.02, KF = (D2.arch.feature_map && D2.arch.feature_map.K) || 3, NF = 2 * KF;
      const L = LAY.map(l => ({ W: Float64Array.from([].concat(...l.W)), b: Float64Array.from(l.b), nOut: l.W.length, nIn: l.W[0].length }));
      const nL = L.length, ZS = L.map(l => new Float64Array(l.nOut)), HS = L.map(l => new Float64Array(l.nOut)), GH = L.map(l => new Float64Array(l.nOut));
      const sigm = (z) => 1 / (1 + Math.exp(-z));
      function ctxBias(x) {
        const f = []; for (let k = 1; k <= KF; k++) f.push(Math.sin(2 * Math.PI * k * x), Math.cos(2 * Math.PI * k * x));
        const l = L[0], zb = new Float64Array(l.nOut);
        for (let i = 0; i < l.nOut; i++) { let s2 = l.b[i]; for (let j = 0; j < NF; j++) s2 += l.W[i * l.nIn + j] * f[j]; zb[i] = s2; }
        return zb;
      }
      function fwd(zb, a, c) {
        const l0 = L[0];
        for (let i = 0; i < l0.nOut; i++) { const z = zb[i] + l0.W[i * l0.nIn + NF] * a + l0.W[i * l0.nIn + NF + 1] * c; ZS[0][i] = z; HS[0][i] = z * sigm(z); }
        let out = 0;
        for (let k = 1; k < nL; k++) {
          const l = L[k], hin = HS[k - 1];
          for (let i = 0; i < l.nOut; i++) { let s2 = l.b[i]; const base = i * l.nIn; for (let j = 0; j < l.nIn; j++) s2 += l.W[base + j] * hin[j]; if (k < nL - 1) { ZS[k][i] = s2; HS[k][i] = s2 * sigm(s2); } else out = s2; }
        }
        return out + LAM * (a * a + c * c);
      }
      let G0 = 0, G1 = 0;
      function fwdGrad(zb, a, c) {
        const E = fwd(zb, a, c); const last = L[nL - 1];
        let gh = GH[nL - 2]; for (let j = 0; j < last.nIn; j++) gh[j] = last.W[j];
        for (let k = nL - 2; k >= 0; k--) {
          const z = ZS[k], g = GH[k]; for (let i = 0; i < g.length; i++) { const sg = sigm(z[i]); g[i] = g[i] * sg * (1 + z[i] * (1 - sg)); }
          const l = L[k];
          if (k > 0) { const gp = GH[k - 1]; gp.fill(0); for (let i = 0; i < l.nOut; i++) { const gi = g[i], base = i * l.nIn; for (let j = 0; j < l.nIn; j++) gp[j] += l.W[base + j] * gi; } }
          else { let s0 = 0, s1 = 0; for (let i = 0; i < l.nOut; i++) { s0 += l.W[i * l.nIn + NF] * g[i]; s1 += l.W[i * l.nIn + NF + 1] * g[i]; } G0 = s0 + 2 * LAM * a; G1 = s1 + 2 * LAM * c; }
        }
        return E;
      }
      // self-check against the shared model helper
      try { [[0.2, [0.3, -0.4]], [0.71, [-1.2, 0.5]]].forEach(([x, y]) => { const r = T2.energyGrad(ck, x, y), E = fwdGrad(ctxBias(x), y[0], y[1]); if (Math.abs(r.E - E) > 1e-6 || Math.abs(r.g[0] - G0) > 1e-6 || Math.abs(r.g[1] - G1) > 1e-6) console.warn('takeaways: evaluator mismatch', r, E, G0, G1); }); } catch (e) { console.warn(e); }

      // ---------- the toy world (formulas from toy2d.json dataset) ----------
      const TAU = 2 * Math.PI;
      const mu = (x) => [1.3 * Math.sin(TAU * x), 0.9 * Math.sin(2 * TAU * x)];
      const sig = (x) => 0.03 + 0.27 * Math.sin(Math.PI * (x - 0.25)) ** 2;
      const grp = (x) => { const s2 = sig(x); return s2 < 0.1 ? 0 : s2 > 0.2 ? 2 : 1; };
      const KINDS = {
        none: { label: 'No change', min: 0, max: 1, step: 1, def: 0, fmt: () => 'n/a', blurb: 'Nothing changes. Every alarm is a false alarm.' },
        rotate: { label: 'Rotates', min: 0, max: 45, step: 1, def: 15, fmt: v => v + '°', blurb: 'The whole loop of outcomes rotates about the origin.' },
        drift: { label: 'Drifts', min: 0, max: 0.6, step: 0.02, def: 0.2, fmt: v => '+' + v.toFixed(2), blurb: 'Every outcome shifts right, like a sensor offset.' },
        noisier: { label: 'Noisier', min: 1, max: 4, step: 0.1, def: 2, fmt: v => 'noise ×' + v.toFixed(1), blurb: 'Same means, more noise: the world got less predictable.' },
        dynamics: { label: 'New dynamics', min: 0, max: 0.2, step: 0.01, def: 0.05, fmt: v => 'phase +' + v.toFixed(2), blurb: 'Outcomes still lie on the same loop, but each context now leads to a different point on it. Only a detector that conditions on x can see this.' },
      };
      function observe(kind, amt, x, e0, e1) {
        let m = mu(x), s2 = sig(x);
        if (kind === 'rotate') { const a = amt * Math.PI / 180, c = Math.cos(a), sn = Math.sin(a); m = [m[0] * c - m[1] * sn, m[0] * sn + m[1] * c]; }
        else if (kind === 'drift') m = [m[0] + amt, m[1]];
        else if (kind === 'noisier') s2 *= amt;
        else if (kind === 'dynamics') { const x2 = (x + amt) % 1; m = mu(x2); s2 = sig(x2); }
        return [m[0] + s2 * e0, m[1] + s2 * e1];
      }
      const meanOf = (kind, amt, x) => observe(kind, amt, x, 0, 0);

      // ---------- data: stream, calibration set ("memory") and test set ----------
      const TT = 240, TC = 80, NMAX = 12, ALPHA = 1.0, NB = 16, PER = 32;
      const S = { kind: 'rotate', amt: KINDS.rotate.def, N: 6, score: 'raw', budget: 0.05, t: 120, seed: 11 };
      // Thinking from ŷ0 for up to NMAX steps; keep the energy (and path) after every step, so any N is free afterwards.
      function thinkAll(zb, a, c, keepPath) {
        const E = new Float64Array(NMAX + 1), P = keepPath ? new Float64Array(2 * (NMAX + 1)) : null;
        for (let k = 0; k < NMAX; k++) { if (P) { P[2 * k] = a; P[2 * k + 1] = c; } E[k] = fwdGrad(zb, a, c); a -= ALPHA * G0; c -= ALPHA * G1; }
        if (P) { P[2 * NMAX] = a; P[2 * NMAX + 1] = c; } E[NMAX] = fwd(zb, a, c);
        return { E, P };
      }
      function makeDraws(n, seed, strat) {
        const r = lib.rng(seed), D = [];
        for (let i = 0; i < n; i++) { const x = strat ? (Math.floor(i / PER) + r()) / NB : r(); D.push({ x, g: grp(x), e0: r.normal(), e1: r.normal(), a0: r.normal(), c0: r.normal() }); }
        D.forEach(d => { d.zb = ctxBias(d.x); });
        return D;
      }
      let stream = null, cal = null, test = null;
      function initData() {
        cal = makeDraws(NB * PER, 101, true); test = makeDraws(NB * PER, 202, true);
        cal.concat(test).forEach(d => { d.th = thinkAll(d.zb, d.a0, d.c0, false); const y = observe('none', 0, d.x, d.e0, d.e1); d.Ein = fwd(d.zb, y[0], y[1]); });
        newStream();
      }
      function newStream() {
        stream = makeDraws(TT, S.seed, false);
        stream.forEach(d => { d.th = thinkAll(d.zb, d.a0, d.c0, true); });
      }
      const memo = {};
      function memoryFor(N) {
        if (memo[N]) return memo[N];
        const by = Array.from({ length: NB }, () => []);
        cal.forEach(d => by[Math.min(NB - 1, Math.floor(d.x * NB))].push(d.Ein - d.th.E[N]));
        const m = by.map(a => { a.sort((p, q) => p - q); return Math.max(1e-6, a[Math.floor(a.length / 2)]); });
        return (memo[N] = m);
      }
      const binOf = (x) => Math.min(NB - 1, Math.floor(x * NB));
      function scoreOf(dE, x, mem) { return S.score === 'raw' ? dE : dE / mem[binOf(x)]; }
      function quantileHigh(arr, q) { const a = Float64Array.from(arr).sort(); const i = clamp(Math.ceil(q * a.length) - 1, 0, a.length - 1); return a[i]; }

      // recompute everything that depends on the settings (cheap: no thinking here)
      let R = null; // {thr, mem, recs[], rates}
      function recompute() {
        const N = S.N, mem = memoryFor(N);
        const calScores = cal.map(d => scoreOf(d.Ein - d.th.E[N], d.x, mem));
        const thr = quantileHigh(calScores, 1 - S.budget);
        const rates = { fa: [0, 0, 0, 0], hit: [0, 0, 0, 0], n: [0, 0, 0, 0] };
        test.forEach(d => {
          const Es = d.th.E[N], s0 = scoreOf(d.Ein - Es, d.x, mem);
          const y = observe(S.kind, S.amt, d.x, d.e0, d.e1), s1 = scoreOf(fwd(d.zb, y[0], y[1]) - Es, d.x, mem);
          [d.g, 3].forEach(g => { rates.n[g]++; if (s0 > thr) rates.fa[g]++; if (s1 > thr) rates.hit[g]++; });
        });
        for (let g = 0; g < 4; g++) { rates.fa[g] /= rates.n[g]; rates.hit[g] /= rates.n[g]; }
        const recs = stream.map((d, t) => {
          const changed = t >= TC && S.kind !== 'none';
          const y = observe(changed ? S.kind : 'none', S.amt, d.x, d.e0, d.e1);
          const Es = d.th.E[N], Eo = fwd(d.zb, y[0], y[1]), dE = Eo - Es, sc = scoreOf(dE, d.x, mem);
          const yh = [d.th.P[2 * N], d.th.P[2 * N + 1]];
          return { t, x: d.x, g: d.g, y, yh, Es, Eo, dE, sc, alarm: sc > thr, dist: Math.hypot(y[0] - yh[0], y[1] - yh[1]), changed };
        });
        // share of in-distribution false alarms that land on noisy contexts (test set)
        let faN = 0, faAll = 0; test.forEach(d => { const s0 = scoreOf(d.Ein - d.th.E[N], d.x, mem); if (s0 > thr) { faAll++; if (d.g === 2) faN++; } });
        R = { thr, mem, recs, rates, faShareNoisy: faAll ? faN / faAll : 0 };
      }

      // ---------- layout ----------
      const bench = h('div', { class: 'bench takeaways-lab' }); s.appendChild(bench);
      const left = h('div', { class: 'stack takeaways-labctl' });
      const inst = h('div', { class: 'takeaways-inst' });
      bench.append(left, inst);

      const steps4 = h('ol', { class: 'takeaways-steps', 'aria-label': 'How the instrument works' },
        h('li', { html: '<b>Context.</b> The world shows a context x (a phase on the toy\'s loop of outcomes), and later an outcome y.' }),
        h('li', { html: '<b>Think.</b> Before seeing y, the EBT takes N gradient steps on its energy from random noise ŷ0 to its guess <b style="color:var(--ebt)">ŷ*</b> (α = 1, exact gradients).' }),
        h('li', { html: '<b>Surprise.</b> When y arrives: ΔE = E(x, y) − E(x, ŷ*), the energy of what happened minus the energy of what it expected.' }),
        h('li', { html: '<b>Alarm.</b> An <b style="color:var(--bad)">alarm</b> fires when surprise crosses a line set so only the chosen share of normal observations would cross it.' }));
      s.insertBefore(steps4, bench);

      const ctl = h('div', { class: 'card stack' }); left.appendChild(ctl);
      const kindSeg = lib.segmented({ label: 'How the world changes at t = 80', options: Object.keys(KINDS).map(k => [k, KINDS[k].label]), value: S.kind, onchange: v => { S.kind = v; const K = KINDS[v]; amtS.input.min = K.min; amtS.input.max = K.max; amtS.input.step = K.step; S.amt = K.def; amtS.set(K.def); amtS.input.disabled = v === 'none'; kindNote.textContent = K.blurb; settingsChanged(); } });
      kindSeg.el.classList.add('takeaways-wrapseg');
      const kindNote = h('p', { class: 'takeaways-small' }, KINDS[S.kind].blurb);
      const amtS = lib.slider({ id: 'takeaways-amt', label: 'Size of the change', min: KINDS.rotate.min, max: KINDS.rotate.max, step: KINDS.rotate.step, value: S.amt, fmt: v => KINDS[S.kind].fmt(v), oninput: v => { S.amt = v; settingsChanged(); } });
      const nS = lib.slider({ id: 'takeaways-n', label: 'Thinking steps N before judging', min: 0, max: NMAX, step: 1, value: S.N, fmt: v => String(v), oninput: v => { S.N = v; settingsChanged(); } });
      const scSeg = lib.segmented({ label: 'Surprise score', options: [['raw', 'Raw ΔE'], ['norm', 'ΔE ÷ memory']], value: S.score, onchange: v => { S.score = v; settingsChanged(); } });
      const bS = lib.slider({ id: 'takeaways-budget', label: 'False-alarm budget on normal data', min: 0.01, max: 0.2, step: 0.01, value: S.budget, fmt: v => Math.round(v * 100) + '%', oninput: v => { S.budget = v; settingsChanged(); } });
      const bStep = h('button', { type: 'button', class: 'btn' }, 'Step');
      const bPlay = h('button', { type: 'button', class: 'btn primary', 'aria-pressed': 'false' }, 'Play');
      const bEnd = h('button', { type: 'button', class: 'btn' }, 'Skip to end');
      const bReset = h('button', { type: 'button', class: 'btn' }, 'Reset');
      const bNew = h('button', { type: 'button', class: 'btn' }, 'New stream');
      ctl.append(
        h('div', { class: 'ctl' }, h('label', {}, h('span', {}, 'How the world changes at t = 80')), kindSeg.el), kindNote, amtS.el, nS.el,
        h('div', { class: 'ctl' }, h('label', {}, h('span', {}, 'Surprise score')), scSeg.el),
        h('p', { class: 'takeaways-small', html: '<b>Memory</b> = the typical ΔE the model saw for this kind of context in 512 earlier normal observations (median per 1/16 of the loop).' }),
        bS.el, h('div', { class: 'row' }, bStep, bPlay, bEnd, bReset, bNew));

      const ro = h('div', { class: 'card stack takeaways-ro' });
      const roNow = h('div', { class: 'readout', 'aria-live': 'polite' });
      const tbl = h('table', { class: 'takeaways-rt' });
      ro.append(h('h4', { style: 'margin:0' }, 'Readouts'), roNow, h('div', { class: 'tbl' }, tbl), h('p', { class: 'takeaways-small', html: 'Rates over 512 fresh normal draws and the same draws after the change; <span class="mono">(k/n)</span> = this stream so far.' }));
      const insight = h('div', { class: 'callout insight takeaways-insight', 'aria-live': 'polite' });

      // instrument cards
      const card = (cls, title, ...kids) => h('div', { class: 'card takeaways-icard ' + cls }, h('div', { class: 'takeaways-ihead' }, h('h4', {}, title), lib.badge('toy')), ...kids);
      const cPlane = card('takeaways-a-plane', 'Energy landscape over ŷ for the current context');
      const cTime = card('takeaways-a-time', 'Surprise over time');
      const cScat = card('takeaways-a-scat', 'Why raw energy over-alarms');
      const cMem = card('takeaways-a-mem', 'Memory: typical surprise per context');
      inst.append(cPlane, cTime, cScat, cMem);
      const legend = h('div', { class: 'legend takeaways-legend' },
        h('span', {}, h('i', { style: 'background:var(--truth)' }), 'precise context (σ < 0.1)'), h('span', {}, h('i', { style: 'background:var(--muted)' }), 'mid'), h('span', {}, h('i', { style: 'background:var(--rnn)' }), 'noisy context (σ > 0.2)'),
        h('span', {}, h('i', { style: 'background:transparent;border:2px solid var(--bad);border-radius:50%' }), 'alarm'), h('span', {}, h('i', { style: 'background:var(--ebt)' }), 'thinking path to ŷ*'));
      inst.appendChild(legend);
      inst.append(ro, insight);

      const plane = autoCanvas(cPlane, { label: 'Energy landscape of the toy EBT over the 2D prediction plane for the current context, with the thinking path, the guess and the observed outcome', height: (w) => Math.min(w, 520), minH: 240, maxH: 520, draw: drawPlane });
      const time = autoCanvas(cTime, { label: 'Timeline of surprise relative to the alarm line for every observation so far', height: (w) => w < 520 ? 230 : 210, minH: 200, maxH: 260, draw: drawTime });
      const scat = autoCanvas(cScat, { label: 'Scatter of surprise versus distance between outcome and guess, colored by context noise', height: (w) => w < 520 ? w * 0.7 : w * 0.62, minH: 200, maxH: 300, draw: drawScat });
      const memc = autoCanvas(cMem, { label: 'Bar chart of the remembered typical surprise for each sixteenth of the context loop', height: (w) => w < 520 ? w * 0.6 : w * 0.5, minH: 170, maxH: 250, draw: drawMem });
      cPlane.appendChild(h('p', { class: 'takeaways-cap', html: 'Dashed white: the outcomes\' mean loop before the change; dashed lilac: after. <b style="color:var(--ebt)">Amber</b>: thinking from ŷ0 to ŷ*. The big dot is the real outcome y.' }));
      cScat.appendChild(h('p', { class: 'takeaways-cap', html: 'Each dot is one observation: distance d = ‖y − ŷ*‖ vs ΔE. All contexts share one bowl, ΔE ≈ c·d². Noisy contexts simply land further out on it.' }));
      cMem.appendChild(h('p', { class: 'takeaways-cap', html: 'Bars: remembered median ΔE per context (log scale). Line: the true data noise σ(x). Used only by the ΔE ÷ memory score.' }));

      // ---------- heatmap cache for the plane ----------
      const EXT = [-2.8, 2.8, -2.8, 2.8], HN = 60;
      let heat = { x: null };
      function heatFor(x) {
        if (heat.x === x) return heat;
        const zb = ctxBias(x), grid = []; let lo = Infinity;
        for (let r = 0; r < HN; r++) { const yv = EXT[3] - (EXT[3] - EXT[2]) * r / (HN - 1), row = new Array(HN); for (let c = 0; c < HN; c++) { const xv = EXT[0] + (EXT[1] - EXT[0]) * c / (HN - 1); const e = fwd(zb, xv, yv); row[c] = e; if (e < lo) lo = e; } grid.push(row); }
        const span = 4.2, off = document.createElement('canvas'); off.width = HN; off.height = HN; const g = off.getContext('2d'), img = g.createImageData(HN, HN);
        for (let r = 0; r < HN; r++) for (let c = 0; c < HN; c++) { const v = Math.pow(clamp((grid[r][c] - lo) / span, 0, 1), 0.62), col = lib.cmap(v), i = 4 * (r * HN + c); img.data[i] = col[0]; img.data[i + 1] = col[1]; img.data[i + 2] = col[2]; img.data[i + 3] = 255; }
        g.putImageData(img, 0, 0);
        heat = { x, grid, lo, off, levels: [0.03, 0.1, 0.25, 0.5, 0.85, 1.3, 1.9, 2.7, 3.6].map(v => lo + v) };
        return heat;
      }

      // ---------- drawing ----------
      let anim = 1; // reveal progress of the latest observation (thinking path, then outcome)
      const curIdx = () => Math.max(0, S.t - 1);
      function drawPlane(ctx, w, hh) {
        if (!R) { T(ctx, 'computing…', 12, 12); return; }
        const Sz = Math.min(w, hh), bx = (w - Sz) / 2, by = (hh - Sz) / 2;
        const toPx = (p) => [bx + (p[0] - EXT[0]) / (EXT[1] - EXT[0]) * Sz, by + Sz - (p[1] - EXT[2]) / (EXT[3] - EXT[2]) * Sz];
        const i = curIdx(), rec = R.recs[i], d = stream[i];
        const hm = heatFor(d.x);
        ctx.save(); lib.rr(ctx, bx, by, Sz, Sz, 10); ctx.clip();
        ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = 'high'; ctx.drawImage(hm.off, bx, by, Sz, Sz);
        lib.contours(ctx, hm.grid, bx, by, Sz, Sz, hm.levels, { color: 'rgba(233,238,246,0.2)', width: 1 });
        // mean loops before / after the change
        const loop = (kind, amt) => { const pts = []; for (let k = 0; k <= 160; k++) pts.push(toPx(meanOf(kind, amt, k / 160))); return pts; };
        lib.line(ctx, loop('none', 0), { color: 'rgba(233,238,246,0.55)', width: 1.5, dash: [5, 5] });
        const showNew = S.kind === 'rotate' || S.kind === 'drift';
        if (showNew && S.t > TC) lib.line(ctx, loop(S.kind, S.amt), { color: lib.rgba(C.rnn, 0.85), width: 1.6, dash: [5, 5] });
        // trail of recent observations
        for (let t = Math.max(0, S.t - 41); t < S.t - 1; t++) { const r = R.recs[t], p = toPx(r.y), a = 0.25 + 0.6 * (t - (S.t - 41)) / 40; lib.dot(ctx, p[0], p[1], r.alarm ? 3.6 : 2.6, GCOL[r.g], { alpha: a, stroke: r.alarm ? C.bad : null, lw: 1.5 }); }
        if (S.t > 0) {
          // thinking path, revealed step by step
          const N = S.N, kShow = anim >= 1 ? N : Math.floor(Math.min(1, anim / 0.65) * N + 1e-9), path = [];
          for (let k = 0; k <= kShow; k++) path.push(toPx([d.th.P[2 * k], d.th.P[2 * k + 1]]));
          const p0 = toPx([d.a0, d.c0]);
          lib.dot(ctx, p0[0], p0[1], 5, 'transparent', { stroke: C.ebt, lw: 2 });
          if (path.length > 1) { lib.line(ctx, path, { color: C.ebt, width: 2.2 }); path.slice(1, -1).forEach(p => lib.dot(ctx, p[0], p[1], 2.2, C.ebt)); }
          const ph = path[path.length - 1]; lib.dot(ctx, ph[0], ph[1], 6.5, C.ebt, { stroke: C.bg, lw: 2 });
          if (anim >= 0.7 || anim >= 1) {
            const py = toPx(rec.y), col = rec.alarm ? C.bad : GCOL[rec.g];
            ctx.save(); ctx.setLineDash([4, 4]); ctx.strokeStyle = lib.rgba(C.ink, 0.7); ctx.lineWidth = 1.3; ctx.beginPath(); ctx.moveTo(ph[0], ph[1]); ctx.lineTo(py[0], py[1]); ctx.stroke(); ctx.restore();
            lib.dot(ctx, py[0], py[1], 8, col, { stroke: rec.alarm ? C.ink : C.bg, lw: 2 });
            if (rec.alarm) lib.dot(ctx, py[0], py[1], 13, 'transparent', { stroke: C.bad, lw: 2 });
            const lx = py[0] + (py[0] > bx + Sz * 0.7 ? -14 : 14);
            pill(ctx, 'y  ΔE ' + rec.dE.toFixed(3), lx, py[1] - 16, rec.alarm ? C.bad : C.ink, { size: 12, align: py[0] > bx + Sz * 0.7 ? 'right' : 'left' });
          }
          pill(ctx, 'ŷ*', ph[0] + 10, ph[1] + 14, C.ebt, { size: 12 });
          // true means for this context (only the instrument shows these; the model never sees them)
          const m0 = toPx(meanOf('none', 0, d.x)); cross(ctx, m0, 'rgba(233,238,246,0.9)');
          if (rec.changed && S.kind !== 'noisier') { const m1 = toPx(meanOf(S.kind, S.amt, d.x)); cross(ctx, m1, C.rnn); }
        }
        ctx.restore();
        const sx = d.x, sg = sig(sx);
        pill(ctx, 'context x = ' + sx.toFixed(3) + ' · σ = ' + sg.toFixed(2) + ' (' + GNAME[grp(sx)] + ')', bx + 8, by + 16, C.ink, { size: 12 });
        pill(ctx, 't = ' + S.t + ' / ' + TT + (S.t > TC && S.kind !== 'none' ? ' · changed world' : ' · normal world'), bx + 8, by + 38, S.t > TC && S.kind !== 'none' ? C.rnn : C.muted, { size: 12 });
        pill(ctx, '+ true mean for this x', bx + Sz - 8, by + Sz - 16, C.muted, { size: 11.5, align: 'right' });
      }
      function cross(ctx, p, col) { ctx.save(); ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(p[0] - 6, p[1]); ctx.lineTo(p[0] + 6, p[1]); ctx.moveTo(p[0], p[1] - 6); ctx.lineTo(p[0], p[1] + 6); ctx.stroke(); ctx.restore(); }

      function drawTime(ctx, w, hh) {
        if (!R) return;
        const b = box(w, hh, { l: 58, r: 12, t: 26, b: 40 });
        const lo = -3, hi = Math.log10(40), thr = R.thr, bad = !(thr > 0);
        const Y = (r) => b.y + b.h - (clamp(Math.log10(Math.max(r, 1e-9)), lo, hi) - lo) / (hi - lo) * b.h;
        const X = (t) => b.x + (t + 0.5) / TT * b.w;
        if (S.kind !== 'none') { ctx.fillStyle = lib.rgba(C.rnn, 0.07); ctx.fillRect(X(TC - 0.5), b.y, b.x + b.w - X(TC - 0.5), b.h); }
        [0.001, 0.01, 0.1, 1, 10].forEach(v => { hline(ctx, b.x, b.x + b.w, Y(v), C.rule, { w: 1 }); T(ctx, (v >= 1 ? v : v.toString()) + '×', b.x - 6, Y(v), { align: 'right', baseline: 'middle', size: 11 }); });
        [0, 40, 80, 120, 160, 200, 240].forEach(v => T(ctx, String(v), X(v - 0.5), b.y + b.h + 6, { align: 'center', size: 11 }));
        T(ctx, 'observation t', b.x + b.w / 2, b.y + b.h + 22, { align: 'center', size: 11.5, kind: 'body' });
        T(ctx, 'surprise ÷ alarm line (log)', 0, 4, { size: 11.5, kind: 'body' });
        hline(ctx, b.x, b.x + b.w, Y(1), C.bad, { w: 1.5, dash: [6, 4] });
        pill(ctx, 'alarm line', b.x + 4, Y(1) - 11, C.bad, { size: 11 });
        if (S.kind !== 'none') { vline(ctx, X(TC - 0.5), b.y, b.y + b.h, C.rnn, { dash: [4, 4] }); pill(ctx, 'world changes', X(TC - 0.5) + 4, b.y + 10, C.rnn, { size: 11 }); }
        for (let t = 0; t < S.t; t++) {
          const r = R.recs[t], ratio = bad ? (r.sc - thr) / Math.max(1e-9, Math.abs(thr)) + 1 : r.sc / thr, x = X(t), y = Y(ratio);
          const last = t === S.t - 1 && anim < 0.7;
          if (last) continue;
          if (r.alarm) lib.dot(ctx, x, y, 4.2, GCOL[r.g], { stroke: C.bad, lw: 1.8 }); else lib.dot(ctx, x, y, 2.3, GCOL[r.g], { alpha: 0.9 });
        }
        if (S.t > 0) vline(ctx, X(S.t - 1), b.y, b.y + b.h, lib.rgba(C.ink, 0.35), { w: 1 });
        if (bad) pill(ctx, 'N too small: ŷ* is still noise, so ΔE is meaningless', b.x + b.w / 2, b.y + b.h - 14, C.bad, { size: 11.5, align: 'center' });
      }

      function drawScat(ctx, w, hh) {
        if (!R) return;
        const pts = R.recs.slice(0, S.t);
        let md = 0, me = 0; pts.forEach(p => { if (p.dist > md) md = p.dist; if (p.dE > me) me = p.dE; });
        const nice = (v) => [0.2, 0.4, 0.5, 0.8, 1, 1.2, 1.6, 2, 2.5, 3, 4].find(q => q >= v) || 4;
        const dmax = nice(clamp(md * 1.08, 0.4, 4)), emax = nice(clamp(Math.max(me * 1.1, (R.thr > 0 && S.score === 'raw') ? R.thr * 1.6 : 0), 0.2, 4));
        const tk = (m) => [0, 0.25, 0.5, 0.75, 1].map(f => +(m * f).toFixed(3));
        const b = box(w, hh, { l: 60, r: 12, t: 24, b: 44 });
        const ax = lib.axes(ctx, Object.assign({}, b, { xlim: [0, dmax], ylim: [-emax * 0.06, emax], xticks: tk(dmax), yticks: tk(emax), yfmt: v => String(v), xfmt: v => String(v), xlabel: 'd = ‖y − ŷ*‖', ylabel: 'ΔE', size: 11 }));
        let num = 0, den = 0; pts.forEach(p => { if (p.dist > 0.02) { num += p.dE * p.dist * p.dist; den += p.dist ** 4; } });
        const c = den > 0 ? num / den : null;
        if (c != null && S.N >= 2) {
          const curve = []; for (let k = 0; k <= 60; k++) { const dd = dmax * k / 60; curve.push([dd, c * dd * dd]); }
          ctx.save(); ctx.beginPath(); ctx.rect(b.x, b.y, b.w, b.h); ctx.clip(); lib.plot(ctx, ax, curve, { color: C.ebt, width: 1.8, dash: [6, 4] }); ctx.restore();
          pill(ctx, 'fit: ΔE ≈ ' + c.toFixed(2) + '·d²  (1/(2α) = 0.50)', b.x + 6, b.y + 10, C.ebt, { size: 11.5 });
        }
        if (S.score === 'raw' && R.thr > 0 && R.thr < emax) { hline(ctx, b.x, b.x + b.w, ax.Y(R.thr), C.bad, { dash: [6, 4] }); pill(ctx, 'alarm line', b.x + b.w - 4, ax.Y(R.thr) - 11, C.bad, { size: 11, align: 'right' }); }
        else if (S.score === 'norm') pill(ctx, 'alarm line differs per context', b.x + b.w - 4, b.y + 30, C.muted, { size: 11, align: 'right' });
        ctx.save(); ctx.beginPath(); ctx.rect(b.x - 4, b.y - 4, b.w + 8, b.h + 8); ctx.clip();
        pts.forEach(p => { const x = ax.X(Math.min(p.dist, dmax)), y = ax.Y(clamp(p.dE, -emax * 0.06, emax)); if (p.alarm) lib.dot(ctx, x, y, 4, GCOL[p.g], { stroke: C.bad, lw: 1.6 }); else lib.dot(ctx, x, y, 2.6, GCOL[p.g], { alpha: 0.85 }); });
        ctx.restore();
      }

      function drawMem(ctx, w, hh) {
        if (!R) return;
        const b = box(w, hh, { l: 60, r: 12, t: 24, b: 44 }), mem = R.mem;
        const lo = -4, hi = -1, Y = (v) => b.y + b.h - (clamp(Math.log10(Math.max(v, 1e-9)), lo, hi) - lo) / (hi - lo) * b.h;
        const X = (x) => b.x + x * b.w;
        ctx.save(); ctx.globalAlpha = S.score === 'raw' ? 0.55 : 1;
        [1e-4, 1e-3, 1e-2, 1e-1].forEach(v => { hline(ctx, b.x, b.x + b.w, Y(v), C.rule, { w: 1 }); T(ctx, v >= 0.01 ? String(v) : v.toExponential(0), b.x - 6, Y(v), { align: 'right', baseline: 'middle', size: 11 }); });
        for (let k = 0; k < NB; k++) { const xc = (k + 0.5) / NB, x0 = X(k / NB) + 1.5, x1 = X((k + 1) / NB) - 1.5, y = Y(mem[k]); ctx.fillStyle = GCOL[grp(xc)]; ctx.globalAlpha = (S.score === 'raw' ? 0.5 : 0.9); ctx.fillRect(x0, y, x1 - x0, b.y + b.h - y); }
        ctx.globalAlpha = S.score === 'raw' ? 0.55 : 1;
        const sp = []; for (let k = 0; k <= 100; k++) { const x = k / 100; sp.push([X(x), b.y + b.h - sig(x) / 0.32 * b.h]); }
        lib.line(ctx, sp, { color: C.ink, width: 1.5, alpha: 0.7 });
        ctx.restore();
        [0, 0.25, 0.5, 0.75, 1].forEach(v => T(ctx, String(v), X(v), b.y + b.h + 6, { align: 'center', size: 11 }));
        T(ctx, 'context x (position on the loop)', b.x + b.w / 2, b.y + b.h + 22, { align: 'center', size: 11.5, kind: 'body' });
        T(ctx, 'typical ΔE (log)', 0, 4, { size: 11.5, kind: 'body' });
        T(ctx, 'σ(x)', X(0.75), b.y + b.h - sig(0.75) / 0.32 * b.h - 16, { size: 11.5, color: C.ink, align: 'center' });
        if (S.t > 0) { const x = stream[curIdx()].x; vline(ctx, X(x), b.y, b.y + b.h, C.ebt, { w: 2 }); }
        if (S.score === 'raw') pill(ctx, 'not used by raw ΔE', b.x + 6, b.y + 10, C.muted, { size: 11 });
      }

      // ---------- readouts ----------
      function updReadouts() {
        if (!R) return;
        const i = curIdx(), r = R.recs[i];
        const fmt3 = (v) => (v >= 0 ? '' : '−') + Math.abs(v).toFixed(3);
        roNow.innerHTML = S.t === 0 ? '<span>t = 0. Press Step or Play.</span>' :
          `<span>t <b>${S.t}</b>/${TT}${S.t > TC && S.kind !== 'none' ? ' · after change' : ''}</span><span>x <b>${r.x.toFixed(3)}</b> (${GNAME[r.g]})</span>` +
          `<span>E(x,y) <b>${fmt3(r.Eo)}</b></span><span>E(x,ŷ*) <b>${fmt3(r.Es)}</b></span><span>ΔE <b>${fmt3(r.dE)}</b></span>` +
          `<span>score <b>${S.score === 'raw' ? fmt3(r.sc) : r.sc.toFixed(1)}</b> vs line <b>${S.score === 'raw' ? fmt3(R.thr) : R.thr.toFixed(1)}</b></span><span style="color:${r.alarm ? 'var(--bad)' : 'var(--muted)'}"><b style="color:inherit">${r.alarm ? 'ALARM' : 'no alarm'}</b></span>`;
        const cnt = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]; // [g] -> [fa, nPre, hit, nPost]
        for (let t = 0; t < S.t; t++) { const q = R.recs[t]; [q.g, 3].forEach(g => { if (t < TC) { cnt[g][1]++; if (q.alarm) cnt[g][0]++; } else { cnt[g][3]++; if (q.alarm) cnt[g][2]++; } }); }
        const none = S.kind === 'none';
        const rowsH = ['<tr><th>Context</th><th>False alarms before</th><th>' + (none ? 'Alarms after (no change)' : 'Hits after change') + '</th></tr>'];
        ['precise σ < 0.1', 'mid', 'noisy σ > 0.2', 'all'].forEach((nm, g) => {
          const sw = g < 3 ? `<i style="background:${GCOL[g]}"></i>` : '';
          rowsH.push(`<tr${g === 3 ? ' class="all"' : ''}><td>${sw}${nm}</td><td class="num"><b>${pct(R.rates.fa[g])}</b> <span>(${cnt[g][0]}/${cnt[g][1]})</span></td><td class="num"><b>${pct(R.rates.hit[g])}</b> <span>(${cnt[g][2]}/${cnt[g][3]})</span></td></tr>`);
        });
        tbl.innerHTML = rowsH.join('');
        // insight
        let ttl, msg;
        const rt = R.rates;
        if (!(R.thr > 0) || S.N < 2) { ttl = 'Think before you judge'; msg = `With N = ${S.N}, ŷ* is still close to the random start, so E(x, ŷ*) is high and ΔE is often negative: the outcome is compared with noise. Raise N to 3 or more; the toy\'s thinking converges in about 3 steps.`; }
        else if (none) { ttl = 'Nothing changed'; msg = `Every alarm here is false. With the raw score, ${pct(R.faShareNoisy)} of them land on noisy contexts. Pick a change above to test detection.`; }
        else if (S.score === 'raw') { ttl = 'Raw energy confuses noise with novelty'; msg = `${pct(R.faShareNoisy)} of false alarms land on noisy contexts, while precise contexts catch only ${pct(rt.hit[0])} of the change. The scatter shows why: in this toy, ΔE is close to one shared bowl c·d² in every context, so the energy basin has the same width whether the world is precise or noisy. Switch the score to <b>ΔE ÷ memory</b>.`; }
        else { ttl = 'A memory of normal surprise calibrates the alarm'; msg = `False alarms are now spread evenly (precise ${pct(rt.fa[0])}, noisy ${pct(rt.fa[2])}) and precise contexts catch ${pct(rt.hit[0])} of the change (all contexts: ${pct(rt.hit[3])}). The energy says how far off the outcome is; the memory says how far off is normal for this context.` + (S.kind === 'dynamics' ? ' Note that the outcomes stay on the same loop: only a detector conditioned on x can see this change.' : S.kind === 'noisier' ? ' A pure noise increase is hard: most new outcomes still look plausible one at a time.' : ''); }
        insight.innerHTML = `<b class="t">${ttl}</b><div>${msg}</div>`;
        bPlay.textContent = playing ? 'Pause' : 'Play'; bPlay.setAttribute('aria-pressed', String(playing));
      }
      function drawAll() { plane.draw(); time.draw(); scat.draw(); memc.draw(); updReadouts(); }
      function settingsChanged() { if (!stream) return; recompute(); drawAll(); }

      // ---------- animation ----------
      let playing = false, hold = 0;
      const STEP = 0.36;
      const loop = lib.loop((dt) => {
        if (anim < 1) { const before = anim; anim = Math.min(1, anim + dt / (playing ? STEP * 0.75 : 0.6)); plane.draw(); if (before < 0.7 && anim >= 0.7) { time.draw(); } if (anim >= 1) drawAll(); return true; }
        if (playing) { hold += dt; if (hold >= STEP * 0.25) { hold = 0; if (!advance()) { playing = false; updReadouts(); return false; } } return true; }
        return false;
      });
      function advance() { if (!stream || S.t >= TT) return false; S.t++; anim = lib.reducedMotion ? 1 : 0; scat.draw(); memc.draw(); updReadouts(); plane.draw(); if (anim >= 1) drawAll(); loop.start(); return true; }
      bStep.addEventListener('click', () => { playing = false; advance(); updReadouts(); });
      bPlay.addEventListener('click', () => { if (!stream) return; if (S.t >= TT) S.t = 0; playing = !playing; hold = STEP; if (playing) loop.start(); updReadouts(); });
      bEnd.addEventListener('click', () => { playing = false; S.t = TT; anim = 1; drawAll(); });
      bReset.addEventListener('click', () => { playing = false; S.t = 0; anim = 1; drawAll(); });
      bNew.addEventListener('click', () => { playing = false; S.seed += 1; newStream(); recompute(); anim = 1; drawAll(); });

      // ---------- init: one-time thinking for 1,264 draws (~16k gradient evaluations) ----------
      let inited = false;
      function init() { if (inited) return; inited = true; const t0 = performance.now(); initData(); recompute(); drawAll(); if (window.EBT_DEBUG) console.log('takeaways lab init ms', performance.now() - t0); }
      setTimeout(init, 200);
      lib.whenVisible(s, init);
      redrawAll.push(() => { if (R) drawAll(); });
      // expose for headless tests
      window.EBT_TAKEAWAYS_LAB = { S, set(o) { Object.assign(S, o); if (o.kind) { kindSeg.set(o.kind); kindNote.textContent = KINDS[o.kind].blurb; } if (o.score) scSeg.set(o.score); if (o.N != null) nS.set(o.N); if (o.amt != null) amtS.set(o.amt); settingsChanged(); }, get R() { return R; } };
    }

    // =====================================================================================
    // 4. OPEN QUESTIONS (beyond the paper)
    // =====================================================================================
    function buildOpen() {
      const s = subsection('open', '4', 'Open questions for a world-model builder', 'Written for research on JEPA-style world models, continual learning and memory. <b>None of this is in the paper</b>: each card says what the paper does give, where the gap is, and a first experiment.');
      const Q = [
        ['Energy over JEPA latents: who prevents collapse?', 'JEPA · latent prediction', null,
          'EBTs already predict in a latent space for video: a 3136-d frozen SD-XL VAE latent (p.12). The encoder is frozen, so nothing can collapse. JEPA is never mentioned in the paper.',
          'EBT training only puts a loss on the final prediction, J(ŷ_N, y) (Alg 1, p.7). If the target encoder learns jointly, a constant embedding drives J to zero. You would still need JEPA\'s anti-collapse machinery (EMA target, stop-gradient, variance terms), and the energy would inherit that geometry.',
          'Freeze a pretrained JEPA encoder, as the paper froze the VAE, and train an EBT predictor in its latent space. Compare 1-step, N-step and Best-of-N latent error. Then unfreeze with an EMA target and watch for collapse.'],
        ['What counts as surprising? Energy as a learning trigger', 'Continual learning · novelty', ['takeaways-lab', 'Surprise lab'],
          'Energies are unnormalized and only relative comparisons are meaningful (p.41). Energy stays higher for hard tokens, unpredictable frames and random text (Figs 8, 11, B.2). Novelty detection is never tested.',
          'In the surprise lab, raw ΔE behaves like a squared distance with the same curvature in every context, so it mistakes noise for novelty; a per-context memory of typical surprise fixes that. Do large EBTs behave the same way? And does the energy scale drift as weights update, moving every threshold?',
          'In a non-stationary stream, gate weight updates or memory writes on normalized surprise. Report detection delay against false alarms, versus a plain prediction-error detector, separately for noise changes and dynamics changes.'],
        ['Replay of predictions vs replay of experience', 'Continual learning · memory', null,
          'The replay buffer stores past predictions and restarts optimization from them to simulate long trajectories, so the landscape stays well shaped near its minima (p.7). The buffer size and contents are unspecified; our toy stores (x, y, ŷ_N) triples.',
          'Experience replay stores (x, y) to fight forgetting. When new data reshapes the landscape, do old basins move, flatten or vanish? A prediction buffer only revisits regions the model already finds, so it may keep a basin\'s shape without keeping its location.',
          'Train on world A, then world B. Track E(x_A, y_A) − E(x_A, ŷ*) for old contexts with no replay, experience replay, prediction replay, and both. Measure forgetting in energy space, not only in prediction error.'],
        ['Thinking as test-time adaptation, and where it fails', 'Distribution shift · OOD', ['thinking', 'Text thinking lab'],
          'Thinking gains grow roughly linearly with OOD shift (Fig 7, p.11), and Transformer++ cannot think at all (Fig 6a).',
          'Thinking can only repair errors of the generator, not of the verifier. If the dynamics change, the old energy still has its minimum at the old answer, and thinking converges to it confidently. The lab\'s "New dynamics" setting shows exactly this: perfect convergence, wrong world.',
          'Split shifts into context shifts (unusual x, verifier still right) and dynamics shifts (verifier wrong). Measure the thinking gain and the energy gap for each, and use the gap to choose between "think longer" and "update the weights".'],
        ['The second-order bill in an online learner', 'Compute · continual updates', ['takeaways-limits', 'Compute calculator'],
          'About 3.33× a Transformer++ step per optimization step (autoregressive), 6.66× with 2 steps (p.36). Only 2 to 3 training steps were stable (p.26). S1 models detach between steps for stability (p.30).',
          'A continual learner pays this on every update, indefinitely. Which part of the second-order signal actually keeps the landscape well shaped?',
          'Compare full unrolling, S1-style detaching, truncated backprop, and a warm start from a fast JEPA predictor (the paper suggests EBTs as verifiers of feed-forward predictions, p.26 to 27). Check landscape quality far from the warm start: Langevin noise and the replay buffer exist to shape regions plain descent never visits (p.7).'],
        ['An EBT as the verifier inside a planner', 'Planning · world models', ['planning', 'Planning lab'],
          'Appendix A.3 (p.26) is the only world-model passage: hold the past fixed and descend the energy with respect to future actions and states, so the world model acts as a policy. There are no experiments.',
          'A planner is an adversary for its verifier. With little data, Best-of-10 sometimes picked low-energy wrong samples (p.28). An optimizer searching over plans will find such holes faster than random candidates do.',
          'In a latent planner (DINO-WM [138] is the paper\'s own reference), add an energy E(s_t, a_t:t+H, ẑ_t+1:t+H) next to the goal cost. Rank sampled plans by energy, refine the best by gradient, and check the plans the energy likes against the true simulator.'],
        ['Branching futures vs the convex basin', 'Multimodality · stochastic worlds', null,
          'Training pushes the landscape to be convex around each target, so data with many modes per condition is averaged into blur (p.17, p.29). The authors expect more conditioning to help (p.29).',
          'World models must keep several futures alive ("a pedestrian might emerge from behind a parked vehicle" is the paper\'s own example, p.3). JEPA answers with a latent variable z; EBT training as published has none.',
          'Condition the energy on a latent z that is itself inferred by minimization, a latent-variable EBM in the sense of Dawid and LeCun [42]. Test on a toy with two equally likely futures: one basin per z, or one blurred basin?'],
        ['Memory reads as candidate predictions', 'Memory · retrieval', null,
          'EBTs use no associative memory (p.38). Hopfield networks are listed as implicit EBMs (p.41). Best-of-N picks the lowest-energy candidate per prediction (Alg 2, p.7).',
          'Episodic retrieval is usually scored by similarity to the query, not by how well the retrieved item fits the current context.',
          'Retrieve K stored outcomes for the current context, use them as starting points ŷ0, think briefly, and keep the lowest energy: Best-of-N whose candidates come from memory instead of noise. Compare against similarity-only retrieval.'],
      ];
      const grid = h('div', { class: 'takeaways-qgrid' }); s.appendChild(grid);
      Q.forEach(([t, tag, link, paper, gap, tr], i) => {
        grid.appendChild(h('article', { class: 'card takeaways-q' },
          h('div', { class: 'takeaways-qtop' }, h('span', { class: 'takeaways-qn' }, 'Q' + (i + 1)), h('span', { class: 'takeaways-qtag' }, tag), lib.badge('ext')),
          h('h4', { class: 'takeaways-qt' }, t),
          h('p', {}, h('b', { class: 'lab' }, 'Paper'), paper),
          h('p', {}, h('b', { class: 'lab' }, 'Gap'), gap),
          h('p', {}, h('b', { class: 'lab' }, 'Try'), tr),
          link ? h('p', { class: 'takeaways-qlink' }, go(link[0], link[1])) : null));
      });
    }

    // =====================================================================================
    // 5. PAPER MAP
    // =====================================================================================
    function buildMap() {
      const s = subsection('map', '5', 'Paper map', 'Every figure, table, algorithm and code listing in the paper (arXiv 2507.02092v1), with its page, what it shows in one line, and where to explore it here. Click a figure row to preview the crop.');
      const M = [
        ['Fig 1', 'fig', 2, 'Four ways to predict the next element: AR Transformer, RNN, Diffusion Transformer, EBT. Only the EBT outputs an energy for a candidate.', 'families', 'fig01'],
        ['Table 1', 'tab', 3, 'Three facets of System 2 thinking by architecture: only EBTs have dynamic compute, uncertainty and verification; diffusion has compute only.', 'families'],
        ['Fig 2', 'fig', 4, 'EBT predicting the next token and the next video frame: a random guess sharpens over steps as its energy is minimized.', 'thinking', 'fig02'],
        ['Fig 3', 'fig', 5, 'Thinking as descent on an energy landscape until the energy converges. A schematic adapted from a loss-landscape paper, not a measured landscape.', 'landscape', 'fig03'],
        ['Algorithm 1', 'alg', 7, 'Training: start from noise, take N gradient steps on the energy, put the loss on the final prediction, backpropagate through the whole optimization.', 'training'],
        ['Algorithm 2', 'alg', 7, 'Inference with verification: optimize M candidates and keep the one with the lowest energy.', 'thinking'],
        ['Fig 4', 'fig', 9, 'Text scaling with data (35.98% faster), batch size (28.46%) and depth (5.29%). EBT crosses below Transformer++ only for data and batch.', 'scaling', 'fig04'],
        ['Fig 5', 'fig', 10, 'Text scaling with parameters (2.91%), FLOPs (2.92%) and width (0.02%). EBT perplexity is higher at every measured point.', 'scaling', 'fig05'],
        ['Table 2', 'tab', 10, 'Ablating the landscape regularizers on OOD Dyck. Without random step size thinking nearly vanishes (−1.47 / 0.19); the full recipe is best with self-verification (18.7).', 'training'],
        ['Fig 6', 'fig', 11, '(a) More forward passes cut OOD perplexity by up to 29%; Transformer++ is flat. (b) The Best-of-5 gain grows with training tokens (Dyck only).', 'thinking', 'fig06'],
        ['Fig 7', 'fig', 11, 'The gain from thinking grows roughly linearly with OOD shift (downstream ÷ pretraining perplexity), about 12% to 23% (approx.).', 'thinking', 'fig07'],
        ['Table 3', 'tab', 12, 'Worse pretraining perplexity (33.43 vs 31.36) but better on GSM8K, BB Math QA and Dyck; slightly worse on SQuAD.', 'data'],
        ['Fig 8', 'fig', 12, 'Token energy over 12 thinking iterations: easy tokens settle low, hard tokens stay high. Uncertainty without supervision.', 'thinking', 'fig08'],
        ['Fig 9', 'fig', 13, 'Video next-frame scaling on SSV2: more than 33% faster with width and parameters, but higher loss at every measured size.', 'scaling', 'fig09'],
        ['Table 4', 'tab', 13, 'Denoising PSNR and MSE at σ = 0.1 and OOD σ = 0.2 (EBT 23.29 vs DiT 19.56 PSNR) and ImageNet-1k linear probe (5.32% vs 0.31% top-1).', 'images'],
        ['Fig 10', 'fig', 14, 'OOD denoising examples: EBT after 1, 2, 3 steps vs DiT after 100, 200, 300 steps.', 'images', 'fig10'],
        ['Fig 11', 'fig', 14, 'Frame energies across a video: high while the scene is unpredictable, low once the garment is in view, higher again when it leaves.', 'data', 'fig11'],
        ['Fig 12', 'fig', 15, 'PSNR vs forward passes on OOD noise: EBT with 3 passes (≈23) beats DiT with 300 (≈19.6) (approx.).', 'images', 'fig12'],
        ['Fig B.1', 'fig', 28, '(a) Best-of-10 over Best-of-2 gains grow with data (RedPajama validation). (b) Fig 6b extrapolated to 15T tokens: speculative.', 'scaling', 'figB1'],
        ['Fig B.2', 'fig', 29, 'Familiar text keeps lower energy than a random token sequence: epistemic uncertainty.', 'thinking', 'figB2'],
        ['Fig B.3', 'fig', 29, 'Data scaling on FineWeb up to about 130B tokens: 35.69% faster overall, 51.70% in the zoomed tail where EBT crosses below.', 'scaling', 'figB3'],
        ['Fig C.1', 'fig', 31, 'S2 (thinking-tuned) vs S1 EBTs on tiny models: S2 starts higher but scales 3.30% faster.', 'training', 'figC1'],
        ['Table D.1', 'tab', 33, 'Model sizes xxs to xl: 6.18M to 708M non-embedding parameters, 6 to 24 layers.', 'scaling'],
        ['Table D.2', 'tab', 33, 'Transformer++ hyperparameters for video (CV) and text (NLP): AdamW, context 16 frames or 256 tokens.', 'data'],
        ['Table D.3', 'tab', 35, 'EBT scaling hyperparameters: 2 optimization steps, step size 500 (text) and 30,000 (video), learnable step size.', 'training'],
        ['Table D.4', 'tab', 36, 'S2 thinking hyperparameters: 2 to 3 random steps, step size 5, Langevin noise 3, replay buffer, no detaching.', 'training'],
        ['Fig E.1', 'fig', 37, 'Diffusion is supervised at every denoising step; an EBM only at the end of its optimization.', 'families', 'figE1'],
        ['Fig H.1', 'fig', 40, 'A feed-forward model maps x to ŷ; an EBM scores the pair (x, ŷ) with one energy.', 'families', 'figH1'],
        ['Listing 1', 'code', 44, 'PyTorch pseudocode for autoregressive language EBT training; create_graph=True keeps the graph for second-order gradients.', 'training'],
        ['Listing 2', 'code', 45, 'The same for video: next-frame embeddings with a Smooth L1 loss.', 'training'],
      ];
      const KIND = { fig: 'Figure', tab: 'Table', alg: 'Algorithm', code: 'Listing' };
      let filt = 'all', q = '', sel = 'Fig 3';
      const bar = h('div', { class: 'row takeaways-mapbar' });
      const fseg = lib.segmented({ label: 'Filter paper items', options: [['all', 'All'], ['fig', 'Figures'], ['tab', 'Tables'], ['alg', 'Algorithms and code'], ['main', 'Main text'], ['app', 'Appendix']], value: filt, onchange: v => { filt = v; render(); } });
      fseg.el.classList.add('takeaways-wrapseg');
      const inp = h('input', { type: 'search', class: 'takeaways-search', placeholder: 'Search the map (e.g. OOD, Dyck, PSNR)', 'aria-label': 'Search the paper map' });
      inp.addEventListener('input', () => { q = inp.value.trim().toLowerCase(); render(); });
      const count = h('span', { class: 'takeaways-count' });
      bar.append(fseg.el, inp, count, lib.badge('paper', 'arXiv 2507.02092v1'));
      s.appendChild(bar);
      const lay = h('div', { class: 'takeaways-maplay' });
      const tbody = h('tbody');
      const table = h('table', { class: 'takeaways-map' }, h('thead', {}, h('tr', {}, h('th', {}, 'Item'), h('th', {}, 'Page'), h('th', {}, 'What it shows'), h('th', {}, 'Explore'))), tbody);
      const prev = h('figure', { class: 'card takeaways-prev' });
      lay.append(h('div', { class: 'tbl' }, table), prev); s.appendChild(lay);
      function preview(row) {
        prev.replaceChildren();
        if (!row || !row[5]) { prev.appendChild(h('p', { class: 'takeaways-small' }, 'Tables, algorithms and listings have no image crop. Their key numbers appear in the row and in the linked chapter.')); return; }
        prev.append(h('div', { class: 'row', style: 'justify-content:space-between;margin-bottom:8px' }, h('b', {}, row[0] + ' · p.' + row[2]), lib.badge('paper', row[0])),
          h('a', { href: 'media/paper/' + row[5] + '.png', target: '_blank', rel: 'noopener', class: 'paper-fig', style: 'display:block' }, h('img', { src: 'media/paper/' + row[5] + '.png', alt: 'Paper ' + row[0] + ': ' + row[3] })),
          h('figcaption', {}, row[3], ' ', go(row[4], 'Explore in ' + chapterName(row[4]))));
      }
      function render() {
        tbody.replaceChildren(); let n = 0;
        M.forEach(row => {
          const isApp = row[2] >= 26;
          if (filt === 'main' && isApp) return; if (filt === 'app' && !isApp) return;
          if (filt === 'fig' && row[1] !== 'fig') return; if (filt === 'tab' && row[1] !== 'tab') return; if (filt === 'alg' && !(row[1] === 'alg' || row[1] === 'code')) return;
          if (q && !(row[0] + ' ' + row[3] + ' ' + KIND[row[1]]).toLowerCase().includes(q)) return;
          n++;
          const tr = h('tr', { class: row[0] === sel ? 'sel' : '' },
            h('td', {}, row[5] ? h('button', { type: 'button', class: 'takeaways-mapbtn', 'aria-label': 'Preview ' + row[0] }, row[0]) : h('span', { class: 'takeaways-mapitem' }, row[0])),
            h('td', { class: 'num' }, 'p.' + row[2]), h('td', {}, row[3]), h('td', {}, go(row[4], chapterName(row[4]))));
          const btn = tr.querySelector('button'); if (btn) btn.addEventListener('click', () => { sel = row[0]; preview(row); render(); });
          tbody.appendChild(tr);
        });
        count.textContent = n + ' of ' + M.length;
      }
      render(); preview(M.find(r => r[0] === sel));
    }
    function chapterName(id) { return ({ watch: 'Watch', families: 'Model families', landscape: 'Energy landscape lab', thinking: 'Thinking lab', training: 'Training lab', images: 'Denoising lab', planning: 'Planning lab', scaling: 'Scaling laws', data: 'Data and evaluation', takeaways: 'Takeaways' })[id] || id; }

    // =====================================================================================
    // 6. GLOSSARY
    // =====================================================================================
    function buildGlossary() {
      const s = subsection('glossary', '6', 'Glossary', 'Search any term or definition. Page numbers point into the paper; terms marked <span class="badge ext" style="font-size:10px">beyond the paper</span> are ours.');
      // Built-in entries. k = normalized prefix: if data/glossary.json already has a term starting with k, ours is skipped.
      const BUILTIN = [
        ['energy', 'Energy E_θ(x, ŷ)', 'One scalar the model outputs for a context x and a candidate prediction ŷ. Lower means more compatible. It is an unnormalized negative log-likelihood up to a constant.', 'p.4 to 6, p.41', 'landscape'],
        ['energylandscape', 'Energy landscape', 'The energy as a function of the prediction ŷ, for one fixed context and fixed weights. Thinking descends it.', 'p.5', 'landscape'],
        ['losslandscape', 'Loss landscape', 'The training loss as a function of the weights θ. Learning descends it. A different object from the energy landscape.', 'Fig 3 source', 'landscape'],
        ['system2thinking', 'Thinking (System 2 thinking)', 'Here: refining one continuous prediction by gradient descent on the energy. Not chain-of-thought, which did not help at this model size.', 'p.4, p.10', 'thinking'],
        ['thinkinglonger', 'Thinking longer', 'Running more optimization steps on the same prediction (Facet 1).', 'p.8', 'thinking'],
        ['selfverification', 'Self-verification (Best-of-N)', 'Optimize M candidates and keep the lowest-energy one, per prediction, with no external verifier (Facet 3).', 'Alg 2, p.7 to 8', 'thinking'],
        ['stepsize', 'Step size α', 'How far each gradient step moves ŷ. A main stability knob; randomized during S2 training.', 'p.7, p.42', 'training'],
        ['langevin', 'Langevin dynamics', 'Adds noise η ~ N(0, σ) to each optimization step so training explores more of the landscape.', 'Eq. 2, p.7', 'training'],
        ['replaybuffer', 'Replay buffer', 'Stores past optimized predictions and restarts optimization from them, simulating longer trajectories.', 'p.7', 'training'],
        ['hessianvector', 'Hessian-vector product', 'The second-order term needed to backpropagate through gradient steps. About the cost of one extra backward pass, linear in model size.', 'p.7, p.35', 'training'],
        ['unrolled', 'Unrolled optimization', 'Treat the N gradient steps on ŷ as layers of one computation graph, so the training loss can be backpropagated through them (Algorithm 1).', 'p.7', 'training'],
        ['s1ebt', 'S1-EBT and S2-EBT', 'S1: tuned for stable scaling, detaches between steps, no landscape regularizers. S2: tuned for thinking, no detaching, truncated backprop, all regularizers.', 'p.30', 'training'],
        ['transformer', 'Transformer++', 'The standard Llama-2-style Transformer recipe, the text and video baseline.', 'p.8, p.33', 'families'],
        ['diffusiontransformer', 'DiT (Diffusion Transformer)', 'The image-denoising baseline, run with DDIM applied recursively.', 'p.13, p.35', 'images'],
        ['numberoffunctionevaluations', 'NFE (function evaluations)', 'Count of forward passes. For an EBT, one per optimization step; each step also needs a gradient with respect to ŷ.', 'p.8', 'thinking'],
        ['perplexity', 'Perplexity', 'exp of the average negative log-likelihood per token. Lower is better.', 'p.9', 'data'],
        ['psnr', 'PSNR', 'Peak signal-to-noise ratio, 10·log10(255² / MSE) on 0 to 255 pixels. Higher is better.', 'Table 4', 'images'],
        ['outofdistribution', 'OOD (out-of-distribution)', 'Data unlike the training data. Fig 7 measures the shift as downstream perplexity ÷ pretraining perplexity.', 'p.11', 'thinking'],
        ['scalingrate', 'Scaling rate', 'How fast loss falls as a resource grows; behaves like the slope of a log-log fit. "X% faster" compares slopes. Not formally defined in the paper.', 'p.9', 'scaling'],
        ['partitionfunction', 'Partition function Z(θ)', 'The normalizer ∫ e^(−E) over all predictions. Intractable, so EBTs work with unnormalized energies and only compare energies.', 'p.6', 'landscape'],
        ['contrastive', 'Contrastive training', 'Push energy down on data and up on negative samples. Struggles in high dimensions (curse of dimensionality).', 'p.6 to 7', 'training'],
        ['optimizationbased', 'Optimization-based training', 'The paper\'s alternative: train the energy so that gradient descent on it lands on the target. Shapes a convex basin around the truth.', 'p.7', 'training'],
        ['threecognitivefacets', 'Facets 1, 2, 3', 'Dynamic compute allocation; modeling uncertainty in continuous spaces; verifying predictions.', 'Table 1, p.3', 'families'],
        ['energybasedtransformer', 'Autoregressive vs bidirectional EBT', 'Autoregressive: decoder-only, predicts the next element with a special attention layout. Bidirectional: attends everywhere, used for image denoising.', 'p.8', 'families'],
        ['informationleakage', 'Information leakage', 'In a naive causal EBT, a prediction could attend to its own target token. The paper\'s attention layout (observed and predicted states, size B × 2N × D) prevents it.', 'p.8, p.31', 'families'],
        ['linearprobe', 'Linear probe', 'Train only a linear classifier on frozen features to test representation quality.', 'p.14', 'images'],
        ['sigmathreemeanings', 'σ (three meanings)', 'Image noise level as a fraction of the diffusion schedule (0.1 or 0.2); Langevin noise magnitude in Eq. 2; the diffusion sampler noise in Fig E.1. Context tells which.', 'p.7, p.13, p.37', 'images'],
        ['score', 'Score function', 'The gradient of the log density, −∇E. Diffusion models learn it directly, which makes them implicit EBMs.', 'p.36, p.41', 'families'],
        ['energybasedmodel', 'Explicit vs implicit EBM', 'Explicit: the network outputs the energy (EBTs). Implicit: the energy is only implied, for example diffusion predicts its gradient.', 'p.41', 'families'],
        ['adversarialsample', 'Adversarial sample', 'A prediction with low energy that is actually wrong. Seen with Best-of-10 at small data scale.', 'p.28', 'scaling'],
        ['convexbasin', 'Convex basin assumption', 'Training pushes the landscape to be convex around each target. With many modes per condition it averages them, causing blur.', 'p.17, p.29', 'landscape'],
        ['s1ebt', 'Detach and truncation', 'Stopping gradients between optimization steps (S1) or backpropagating only through the last steps (S2), for stability.', 'p.30, p.43', 'training'],
        ['somethingsomething', 'SD-XL VAE latent', 'Frozen autoencoder features, 3136 numbers per 224×224 frame, in which the video EBT predicts.', 'p.12', 'data'],
        ['redpajama', 'RedPajamaV2', 'Web text corpus. The paper uses a 100B-token sample split into 66 million train and 33 thousand validation samples.', 'p.8', 'data'],
        ['redpajama', 'GPT-NeoX tokenizer', 'The tokenizer for all text models; 50,277-entry vocabulary.', 'p.8, p.35', 'data'],
        ['smoothl1', 'Smooth L1 loss', 'Huber-like loss used for video next-frame prediction, β = 1.0.', 'p.12', 'data'],
        ['jepa', 'JEPA', 'Joint-Embedding Predictive Architecture: predicts the representation of a target from a context in a learned embedding space. Not mentioned in the paper.', '', 'takeaways-open', true],
        ['surprisede', 'Surprise ΔE', 'In this chapter\'s lab: E(x, y_obs) − E(x, ŷ*), the energy of what happened minus the energy of what the model expected after thinking.', '', 'takeaways-lab', true],
        ['memorynormalized', 'Memory-normalized surprise', 'ΔE divided by the typical ΔE remembered for similar contexts. Separates "unusual for here" from "noisy here".', '', 'takeaways-lab', true],
      ].map(([k, term, def, ref, link, ext]) => ({ k, term, def, ref, link, ext: !!ext }));
      const fromData = glossaryFromData(lib.data('glossary'));
      const keys = fromData.map(e => norm(e.term));
      const ALL = fromData.concat(BUILTIN.filter(e => !keys.some(dk => dk.startsWith(e.k)))).sort((a, b) => a.term.localeCompare(b.term));
      const inp = h('input', { type: 'search', class: 'takeaways-search', placeholder: 'Search terms and definitions', 'aria-label': 'Search the glossary' });
      const count = h('span', { class: 'takeaways-count', 'aria-live': 'polite' });
      const src = fromData.length ? h('span', { class: 'takeaways-small' }, fromData.length + ' terms from the shared glossary, ' + (ALL.length - fromData.length) + ' added here') : null;
      s.appendChild(h('div', { class: 'row takeaways-mapbar' }, inp, count, src));
      const dl = h('dl', { class: 'takeaways-gloss' }); s.appendChild(dl);
      const more = h('button', { type: 'button', class: 'btn', hidden: true }); s.appendChild(h('div', { class: 'row' }, more));
      let collapsed = window.innerWidth < 700;
      more.addEventListener('click', () => { collapsed = !collapsed; render(); });
      const esc = (t) => String(t).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);
      const mark = (t, qq) => { const e = esc(t); if (!qq) return e; const q2 = esc(qq), i = e.toLowerCase().indexOf(q2.toLowerCase()); return i < 0 ? e : e.slice(0, i) + '<mark>' + e.slice(i, i + q2.length) + '</mark>' + e.slice(i + q2.length); };
      function render() {
        const qq = inp.value.trim(), ql = qq.toLowerCase(); dl.replaceChildren();
        const hits = ALL.filter(e => !ql || (e.term + ' ' + e.def + ' ' + (e.long || '')).toLowerCase().includes(ql));
        const shown = (collapsed && !ql) ? hits.slice(0, 10) : hits;
        shown.forEach(e => {
          const dt = h('dt', { html: mark(e.term, qq) }); if (e.ext) dt.appendChild(h('span', { class: 'badge ext', style: 'margin-left:8px;font-size:10px' }, 'beyond the paper'));
          const dd = h('dd', {}, h('span', { html: mark(e.def, qq) }), ' ', e.ref && !e.ext ? h('span', { class: 'takeaways-pg' }, /^\d+$/.test(String(e.ref)) ? 'p.' + e.ref : String(e.ref)) : null, e.link ? h('a', { href: '#' + e.link, class: 'takeaways-go', style: 'margin-left:6px', 'aria-label': 'Go to ' + chapterName(e.link.replace('takeaways-', '')) }, '→') : null);
          if (e.long) { const det = h('details', {}, h('summary', {}, 'More'), h('p', { html: mark(e.long, qq) })); if (ql && e.long.toLowerCase().includes(ql) && !(e.term + e.def).toLowerCase().includes(ql)) det.open = true; dd.appendChild(det); }
          dl.append(h('div', { class: 'takeaways-gi' }, dt, dd));
        });
        count.textContent = hits.length + ' of ' + ALL.length + ' terms';
        more.hidden = !!ql || hits.length <= 10; more.textContent = collapsed ? 'Show all ' + ALL.length + ' terms' : 'Show fewer';
        if (!hits.length) dl.appendChild(h('p', { class: 'takeaways-small' }, 'No term matches. Try a shorter word.'));
      }
      inp.addEventListener('input', render); render();
      function norm(t) { return String(t || '').toLowerCase().replace(/[^a-z0-9]+/g, ''); }
    }
    function glossaryFromData(G) {
      if (!G) return [];
      let arr = Array.isArray(G) ? G : (G.terms || G.glossary || G.items || G.entries || null);
      if (!arr && typeof G === 'object') arr = Object.entries(G).filter(([k]) => k !== '_meta').map(([k, v]) => typeof v === 'string' ? { term: k, def: v } : Object.assign({ term: k }, v));
      if (!Array.isArray(arr)) return [];
      return arr.map(e => (!e || typeof e !== 'object') ? null : ({
        term: e.term || e.name || e.title, def: e.short || e.def || e.definition || e.desc || e.description || e.text, long: e.long || null,
        ref: e.ref || e.page || e.pages || e.where || null, link: e.chapter || e.section || e.anchor || null, ext: !!(e.ext || e.beyond || e.beyond_paper),
      })).filter(e => e && e.term && e.def);
    }
  }
})();
