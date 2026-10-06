/* Shared helpers for the EBT explainer site. Loaded after video/engine.js, so EBTV.U drawing helpers are available.
   Sections register with EBT.section({id, nav, kicker, title, lede, mount(el, lib)}) and are mounted in EBT_SECTIONS order. */
(function () {
  const U = window.EBTV ? window.EBTV.U : {};
  const lib = Object.assign({}, U);
  lib.data = (name) => (window.EBT_DATA || {})[name] || null;

  // HiDPI canvas inside a container. Logical size w×h (CSS scales to container width). Returns {canvas, ctx, w, h, resize}
  lib.canvas = function (parent, w, h, o = {}) {
    const box = document.createElement('div'); box.className = 'canvas-box'; parent.appendChild(box);
    const c = document.createElement('canvas'); box.appendChild(c);
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    c.width = Math.round(w * dpr); c.height = Math.round(h * dpr);
    if (o.label) { c.setAttribute('role', 'img'); c.setAttribute('aria-label', o.label); }
    const ctx = c.getContext('2d'); ctx.scale(dpr, dpr);
    // map a pointer event to logical coords
    const toLocal = (ev) => { const r = c.getBoundingClientRect(); return [(ev.clientX - r.left) / r.width * w, (ev.clientY - r.top) / r.height * h]; };
    return { canvas: c, ctx, w, h, box, toLocal, clear(col) { ctx.save(); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.fillStyle = col || '#0f1724'; ctx.fillRect(0, 0, w, h); ctx.restore(); } };
  };

  // DOM helpers
  lib.h = function (tag, attrs = {}, ...kids) {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (k === 'class') e.className = v; else if (k === 'html') e.innerHTML = v; else if (k.startsWith('on') && typeof v === 'function') e.addEventListener(k.slice(2), v);
      else if (v === true) e.setAttribute(k, ''); else if (v !== false && v != null) e.setAttribute(k, v);
    }
    kids.flat().forEach(k => { if (k == null || k === false) return; e.appendChild(typeof k === 'string' || typeof k === 'number' ? document.createTextNode(String(k)) : k); });
    return e;
  };
  // slider control: returns {el, input, value()}
  lib.slider = function ({ id, label, min, max, step, value, fmt, oninput }) {
    const out = lib.h('output', {}, fmt ? fmt(value) : String(value));
    const input = lib.h('input', { type: 'range', id, min, max, step, value });
    const el = lib.h('div', { class: 'ctl' }, lib.h('label', { for: id }, lib.h('span', {}, label), out), input);
    input.addEventListener('input', () => { const v = parseFloat(input.value); out.textContent = fmt ? fmt(v) : String(v); oninput && oninput(v); });
    return { el, input, out, value: () => parseFloat(input.value), set(v) { input.value = v; out.textContent = fmt ? fmt(v) : String(v); } };
  };
  // segmented toggle: options [[value,label]], returns {el, value(), set(v)}
  lib.segmented = function ({ options, value, onchange, label }) {
    let cur = value; const el = lib.h('div', { class: 'seg', role: 'group', 'aria-label': label || '' });
    const btns = options.map(([v, l]) => { const b = lib.h('button', { type: 'button', 'aria-pressed': String(v === cur) }, l); b.addEventListener('click', () => { cur = v; btns.forEach((bb, i) => bb.setAttribute('aria-pressed', String(options[i][0] === cur))); onchange && onchange(v); }); el.appendChild(b); return b; });
    return { el, value: () => cur, set(v) { cur = v; btns.forEach((bb, i) => bb.setAttribute('aria-pressed', String(options[i][0] === cur))); } };
  };
  lib.badge = (kind, label) => lib.h('span', { class: 'badge ' + kind }, (kind === 'paper' ? 'From the paper' : kind === 'toy' ? 'Toy model trained for this explainer' : 'Beyond the paper') + (label ? ' · ' + label : ''));
  lib.callout = (kind, title, html) => lib.h('div', { class: 'callout ' + (kind || '') }, title ? lib.h('b', { class: 't' }, title) : null, lib.h('div', { html }));
  lib.fmt = (v, d = 2) => (v == null || !isFinite(v)) ? '–' : Math.abs(v) >= 1000 ? v.toLocaleString(undefined, { maximumFractionDigits: 0 }) : v.toFixed(d);
  // animation loop helper: returns {start, stop, running}; cb(dt) returns false to stop
  lib.loop = function (cb) {
    let raf = null, last = 0;
    const tick = (now) => { const dt = Math.min(0.1, (now - last) / 1000); last = now; if (cb(dt) === false) { raf = null; return; } raf = requestAnimationFrame(tick); };
    return { start() { if (raf) return; last = performance.now(); raf = requestAnimationFrame(tick); }, stop() { if (raf) cancelAnimationFrame(raf); raf = null; }, get running() { return !!raf; } };
  };
  // run fn when element first becomes visible (to avoid heavy work offscreen)
  lib.whenVisible = function (el, fn) { const io = new IntersectionObserver(es => { if (es.some(e => e.isIntersecting)) { io.disconnect(); fn(); } }, { rootMargin: '200px' }); io.observe(el); };
  lib.reducedMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  // section registry
  const EBT = window.EBT = window.EBT || {};
  EBT.lib = lib; EBT.sections = EBT.sections || {};
  EBT.section = function (def) { EBT.sections[def.id] = def; };
})();
