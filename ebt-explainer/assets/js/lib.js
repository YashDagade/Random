/* EBT explainer: panel system + shared helpers.
   Load order: data/bundle.js, assets/js/engine.js (drawing helpers, light theme), this file, toy2d_model.js, panels/manifest.js, panels/*.js, then EBT.boot().

   A panel file calls:
   EBT.panel({
     id: 'descent',                         // unique, used as #anchor and file name
     nav: 'Gradient descent on ŷ',          // short label in the left rail
     title: 'Thinking is gradient descent on the prediction',
     lede: 'One or two sentences under the title (HTML ok, KaTeX ok).',
     text: '<p>Prose shown at the top of the right column (HTML + $math$ + $$display$$).</p>',
     steps: [ { label: 'Start from noise', html: '<p>…</p>' }, … ],   // optional numbered steps; figure.step(i) is called
     after: '<p>Prose after the steps (equations, caveats).</p>',     // optional
     source: [{ kind: 'toy' | 'paper' | 'ext' | 'concept', note: 'Fig 4a, digitized (approx.)' }],
     figure(stage, ctx) {                   // build the interactive figure inside `stage`
       // ctx.lib, ctx.panel (section element), ctx.setCaption(html), ctx.goStep(i), ctx.visible()
       return { step(i) {}, show() {}, hide() {} };   // all optional
     },
   });
*/
(function () {
  const U = window.EBTV ? window.EBTV.U : {};
  const lib = Object.assign({}, U);
  lib.data = (name) => (window.EBT_DATA || {})[name] || null;
  lib.C = U.C; // palette (light theme): bg, ink, muted, faint, rule, blue, blue2, blue3, blue4, bad, ebt (=blue), ar (=ink)

  // ---------- DOM ----------
  lib.h = function (tag, attrs = {}, ...kids) {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (k === 'class') e.className = v; else if (k === 'html') e.innerHTML = v; else if (k === 'style' && typeof v === 'object') Object.assign(e.style, v);
      else if (k.startsWith('on') && typeof v === 'function') e.addEventListener(k.slice(2), v);
      else if (v === true) e.setAttribute(k, ''); else if (v !== false && v != null) e.setAttribute(k, v);
    }
    kids.flat().forEach(k => { if (k == null || k === false) return; e.appendChild(typeof k === 'string' || typeof k === 'number' ? document.createTextNode(String(k)) : k); });
    return e;
  };
  // HiDPI canvas: logical size w×h, scales with container width. Returns {canvas, ctx, w, h, toLocal(ev), clear()}
  lib.canvas = function (parent, w, h, o = {}) {
    const box = lib.h('div', { class: 'canvas-box' }); parent.appendChild(box);
    const c = lib.h('canvas', o.label ? { role: 'img', 'aria-label': o.label } : {}); box.appendChild(c);
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    c.width = Math.round(w * dpr); c.height = Math.round(h * dpr);
    if (o.maxWidth) c.style.maxWidth = o.maxWidth + 'px';
    const ctx = c.getContext('2d'); ctx.scale(dpr, dpr);
    const toLocal = (ev) => { const r = c.getBoundingClientRect(); const p = ev.touches ? ev.touches[0] : ev; return [(p.clientX - r.left) / r.width * w, (p.clientY - r.top) / r.height * h]; };
    return { canvas: c, ctx, w, h, box, dpr, toLocal, clear(col) { ctx.save(); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.fillStyle = col || '#ffffff'; ctx.fillRect(0, 0, w, h); ctx.restore(); } };
  };
  // framed figure block with mono label + subtitle and corner "+" marks
  lib.frame = function (parent, { label, sub, dashed, plain, pad } = {}) {
    const wrap = lib.h('div', { class: 'fig-col' });
    if (label) wrap.appendChild(lib.h('div', { class: 'fig-label' }, label));
    if (sub) wrap.appendChild(lib.h('div', { class: 'fig-sub', html: sub }));
    const fr = lib.h('div', { class: 'frame' + (dashed ? ' dashed' : '') + (plain ? ' plain' : ''), style: pad ? { padding: pad + 'px' } : {} }, lib.h('span', { class: 'cnr' }));
    wrap.appendChild(fr); parent.appendChild(wrap);
    return { wrap, frame: fr };
  };
  lib.slider = function ({ id, label, min, max, step, value, fmt, oninput }) {
    const f = fmt || ((v) => String(v));
    const out = lib.h('output', { for: id }, f(value));
    const input = lib.h('input', { type: 'range', id, min, max, step, value, 'aria-label': label });
    const el = lib.h('div', { class: 'ctl' }, lib.h('label', { for: id }, lib.h('span', {}, label), out), input);
    input.addEventListener('input', () => { const v = parseFloat(input.value); out.textContent = f(v); oninput && oninput(v); });
    return { el, input, value: () => parseFloat(input.value), set(v, fire) { input.value = v; out.textContent = f(v); if (fire) oninput && oninput(v); } };
  };
  lib.segmented = function ({ options, value, onchange, label }) {
    let cur = value; const el = lib.h('div', { class: 'seg', role: 'group', 'aria-label': label || '' });
    const btns = options.map(([v, l]) => { const b = lib.h('button', { type: 'button', 'aria-pressed': String(v === cur) }, l); b.addEventListener('click', () => { cur = v; sync(); onchange && onchange(v); }); el.appendChild(b); return b; });
    const sync = () => btns.forEach((bb, i) => bb.setAttribute('aria-pressed', String(options[i][0] === cur)));
    return { el, value: () => cur, set(v) { cur = v; sync(); } };
  };
  lib.button = (label, onclick, o = {}) => lib.h('button', { type: 'button', class: 'btn' + (o.primary ? ' primary' : ''), onclick, 'aria-label': o.aria || null }, label);
  lib.src = (kind, note) => lib.h('span', { class: 'src ' + kind }, ({ paper: 'from the paper', toy: 'toy model trained for this explainer', ext: 'beyond the paper', concept: 'illustration' })[kind] + (note ? ' · ' + note : ''));
  lib.fmt = (v, d = 2) => (v == null || !isFinite(v)) ? '–' : Math.abs(v) >= 1e4 ? v.toExponential(2) : v.toFixed(d);
  lib.loop = function (cb) {
    let raf = null, last = 0;
    const tick = (now) => { const dt = Math.max(0, Math.min(0.1, (now - last) / 1000)); last = now; if (cb(dt) === false) { raf = null; return; } raf = requestAnimationFrame(tick); };
    return { start() { if (raf) return; last = performance.now(); raf = requestAnimationFrame(tick); }, stop() { if (raf) cancelAnimationFrame(raf); raf = null; }, get running() { return !!raf; } };
  };
  // value at quantile q (0..1) of a 2-D grid: use for heatmap ranges so basins stay visible, e.g. range: [lo, lib.quantile(E, 0.6)]
  lib.quantile = (grid, q) => { const v = grid.flat().filter(Number.isFinite).sort((a, b) => a - b); return v[Math.min(v.length - 1, Math.max(0, Math.round(q * (v.length - 1))))]; };
  lib.reducedMotion = !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);
  lib.math = (el) => { if (window.renderMathInElement) window.renderMathInElement(el, { delimiters: [{ left: '$$', right: '$$', display: true }, { left: '\\[', right: '\\]', display: true }, { left: '$', right: '$', display: false }, { left: '\\(', right: '\\)', display: false }], throwOnError: false }); };
  lib.tex = (s, display) => { try { return window.katex ? window.katex.renderToString(s, { displayMode: !!display, throwOnError: false }) : s; } catch (e) { return s; } };

  // ---------- panel registry ----------
  const EBT = window.EBT = window.EBT || {};
  EBT.lib = lib; EBT.panels = EBT.panels || {};
  EBT.panel = function (def) { EBT.panels[def.id] = def; };

  EBT.boot = function () {
    const order = window.EBT_PANELS || []; // [{part, ids:[...]}]
    const main = document.getElementById('main'), rail = document.getElementById('rail-list'), sel = document.getElementById('jump');
    const flat = []; order.forEach(p => p.ids.forEach(id => flat.push({ id, part: p.part })));
    const total = flat.length; const live = [];
    let partIdx = -1;
    order.forEach((p, pi) => {
      const head = lib.h('div', { class: 'part' }, `${['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX'][pi]} · ${p.part}`);
      rail.appendChild(head); const ol = lib.h('ol'); rail.appendChild(ol);
      const og = lib.h('optgroup', { label: p.part }); sel.appendChild(og);
      p.ids.forEach(id => {
        const n = flat.findIndex(f => f.id === id) + 1; const def = EBT.panels[id];
        const label = def ? (def.nav || def.title) : id;
        ol.appendChild(lib.h('li', {}, lib.h('a', { href: '#' + id, 'data-id': id }, lib.h('span', { class: 'n' }, String(n).padStart(2, '0')), label)));
        og.appendChild(lib.h('option', { value: id }, String(n).padStart(2, '0') + '  ' + label));
      });
    });
    sel.addEventListener('change', () => { const t = document.getElementById(sel.value); if (t) t.scrollIntoView({ behavior: lib.reducedMotion ? 'auto' : 'smooth' }); });
    flat.forEach(({ id, part }, i) => {
      const def = EBT.panels[id]; const n = i + 1;
      const sec = lib.h('section', { class: 'panel', id, 'aria-labelledby': id + '-h' });
      main.appendChild(sec);
      const head = lib.h('header', { class: 'p-head' }, lib.h('div', { class: 'p-num', html: `<b>${String(n).padStart(2, '0')}</b> / ${total} · ${part}` }));
      sec.appendChild(head);
      if (!def) { head.appendChild(lib.h('h2', { id: id + '-h' }, id)); head.appendChild(lib.h('p', { class: 'p-lede' }, 'This panel is still being built.')); return; }
      head.appendChild(lib.h('h2', { id: id + '-h', html: def.title }));
      if (def.lede) head.appendChild(lib.h('p', { class: 'p-lede', html: def.lede }));
      const body = lib.h('div', { class: 'p-body' }); sec.appendChild(body);
      const stage = lib.h('div', { class: 'p-stage' }); const text = lib.h('article', { class: 'p-text' });
      body.appendChild(stage); body.appendChild(text);
      if (def.text) text.appendChild(lib.h('div', { class: 'p-intro', style: { display: 'grid', gap: '12px' }, html: def.text }));
      let stepEls = [], stepBtns = [], cur = 0, fig = null;
      const stepbar = lib.h('div', { class: 'stepbar', role: 'group', 'aria-label': 'Steps' });
      const goStep = (k, fromFig) => {
        if (!def.steps || !def.steps.length) return; k = Math.max(0, Math.min(def.steps.length - 1, k)); cur = k;
        stepEls.forEach((e, j) => e.classList.toggle('on', j === k)); stepBtns.forEach((b, j) => { b.classList.toggle('on', j === k); b.setAttribute('aria-pressed', String(j === k)); });
        if (!fromFig && fig && fig.step) { try { fig.step(k); } catch (e) { console.error(id, e); } }
      };
      if (def.steps && def.steps.length) {
        const st = lib.h('div', { class: 'steps' }); text.appendChild(st);
        def.steps.forEach((s, j) => {
          const e = lib.h('div', { class: 'step', tabindex: '0', role: 'button', 'aria-label': `Step ${j + 1}: ${s.label}` }, lib.h('span', { class: 'sl' }, `[${j + 1}] ${s.label}`), lib.h('div', { html: s.html || '' }));
          e.addEventListener('click', () => goStep(j)); e.addEventListener('keydown', (ev) => { if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); goStep(j); } });
          st.appendChild(e); stepEls.push(e);
          const b = lib.h('button', { type: 'button', 'aria-label': `Step ${j + 1}` }, `[${j + 1}]`); b.addEventListener('click', () => goStep(j)); stepbar.appendChild(b); stepBtns.push(b);
        });
      }
      if (def.after) text.appendChild(lib.h('div', { class: 'p-after', style: { display: 'grid', gap: '12px' }, html: def.after }));
      const cap = lib.h('div', { class: 'caption' });
      const srcRow = lib.h('div', { class: 'readout', style: { gap: '4px 16px' } });
      (def.source || []).forEach(s => srcRow.appendChild(lib.src(s.kind, s.note)));
      let visible = false;
      const ctx = { lib, panel: sec, stage, setCaption: (html) => { cap.innerHTML = html; lib.math(cap); }, goStep: (k) => goStep(k, true), visible: () => visible, get step() { return cur; } };
      try { fig = def.figure ? def.figure(stage, ctx) || {} : {}; } catch (e) { console.error(id, e); stage.appendChild(lib.h('div', { class: 'callout warn' }, 'Figure failed to load: ' + e.message)); fig = {}; }
      if (def.steps && def.steps.length) stage.appendChild(stepbar);
      stage.appendChild(cap); stage.appendChild(srcRow);
      lib.math(sec);
      goStep(0);
      live.push({ id, sec, fig, def, goStep: (k) => goStep(k), get cur() { return cur; }, setVisible(v) { if (v === visible) return; visible = v; try { v ? fig.show && fig.show() : fig.hide && fig.hide(); } catch (e) { console.error(id, e); } } });
    });
    // visibility + rail highlight
    const links = [...document.querySelectorAll('#rail a[data-id]')];
    let current = null;
    const io = new IntersectionObserver(es => es.forEach(e => { const L = live.find(l => l.sec === e.target); if (L) L.setVisible(e.isIntersecting); }), { rootMargin: '10% 0px 10% 0px' });
    const io2 = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) { current = e.target.id; links.forEach(a => a.classList.toggle('on', a.dataset.id === current)); sel.value = current; const a = links.find(x => x.dataset.id === current); if (a && a.scrollIntoViewIfNeeded) a.scrollIntoViewIfNeeded(false); } }), { rootMargin: '-40% 0px -55% 0px' });
    live.forEach(L => { io.observe(L.sec); io2.observe(L.sec); });
    // keyboard: j/k or PageDown/PageUp = next/prev panel, 1-9 = step, ←/→ = prev/next step
    document.addEventListener('keydown', (ev) => {
      const t = ev.target; if (t && (t.tagName === 'INPUT' || t.tagName === 'SELECT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) return;
      if (ev.metaKey || ev.ctrlKey || ev.altKey) return;
      const i = live.findIndex(L => L.id === current); const L = live[i];
      if (ev.key === 'j' || ev.key === 'k') { const nxt = live[Math.max(0, Math.min(live.length - 1, (i < 0 ? 0 : i) + (ev.key === 'j' ? 1 : -1)))]; if (nxt) nxt.sec.scrollIntoView({ behavior: lib.reducedMotion ? 'auto' : 'smooth' }); ev.preventDefault(); return; }
      if (!L || !L.def.steps || !L.def.steps.length) return;
      if (/^[1-9]$/.test(ev.key)) { L.goStep(parseInt(ev.key) - 1); ev.preventDefault(); }
      else if (ev.key === 'ArrowRight') { L.goStep(L.cur + 1); ev.preventDefault(); }
      else if (ev.key === 'ArrowLeft') { L.goStep(L.cur - 1); ev.preventDefault(); }
    });
    if (location.hash) { const t = document.getElementById(location.hash.slice(1)); if (t) setTimeout(() => t.scrollIntoView(), 50); }
    EBT.live = live;
  };
})();
