/* Chapter: Scaling laws explorer (id: scaling).
   Data: window.EBT_DATA.scaling (data/scaling.json): points digitized from the paper's figures (approx.).
   Labs:
     1. Refit a scaling law: pick an axis, walk through the fit (points -> log-log -> OLS -> slope -> rate -> extrapolation),
        warp the axes from linear to log, drop points from the fit, extrapolate the fits and find where they cross.
     2. What a steeper slope buys: catch-up arithmetic, decades = log10(1 + gap) / (|m| * gain).
     3. The compute caveat: the paper's FLOP accounting (6N vs 10N, x2, x2) and a Fig 5b what-if on the cost multiplier.
     4. Thinking scalability: Figs 6a, 6b, B.1a, 7, B.1b, 12.
     5. All axes at a glance + what the fits cannot tell you. */
(function () {
  'use strict';
  const L10 = Math.log10, P10 = (v) => Math.pow(10, v);
  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
  const MINUS = '−';
  const SUP = { '-': '⁻', '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹' };
  const sup = (s) => String(s).split('').map(c => SUP[c] || c).join('');
  const neg = (s) => String(s).replace(/-/g, MINUS);

  // ---------- number formatting ----------
  function sci(v, d = 1) {
    if (v == null || !isFinite(v)) return '–';
    if (v === 0) return '0';
    let e = Math.floor(L10(Math.abs(v)));
    let ms = (v / P10(e)).toFixed(d);
    if (Math.abs(parseFloat(ms)) >= 10) { e += 1; ms = (v / P10(e)).toFixed(d); }
    ms = ms.replace(/\.0+$/, '');
    return (ms === '1' ? '' : neg(ms) + '×') + '10' + sup(e);
  }
  function num(v, sig = 3) {
    if (v == null || !isFinite(v)) return '–';
    const a = Math.abs(v);
    if (a !== 0 && (a >= 1e6 || a < 1e-3)) return sci(v, 1);
    if (a >= 1000) return neg(Math.round(v).toLocaleString('en-US'));
    return neg(String(+v.toPrecision(sig)));
  }
  const pct = (g, d = 1) => (g >= 0 ? '+' : MINUS) + Math.abs(g * 100).toFixed(d) + '%';
  const pctPlain = (g, d = 2) => neg((g * 100).toFixed(d)) + '%';
  const mult = (k) => k < 1e4 ? '×' + num(k) : '×' + sci(k, 1);
  const KIND = {
    tokB: v => v >= 1000 ? num(v / 1000) + 'T' : v >= 1 ? num(v) + 'B' : num(v, 2) + 'B',
    tokK: v => v >= 1000 ? num(v / 1000) + 'M' : num(v) + 'K',
    count: v => num(v),
    paramM: v => v >= 1e9 ? sci(v * 1e6, 1) : v >= 1e6 ? num(v / 1e6) + 'T' : v >= 1000 ? num(v / 1000) + 'B' : v >= 1 ? num(v) + 'M' : num(v, 2) + 'M',
    flops: v => sci(v, 1),
    tokens: v => v >= 1e12 ? num(v / 1e12) + 'T' : v >= 1e9 ? num(v / 1e9) + 'B' : num(v / 1e6) + 'M',
    pct: v => num(v) + '%',
    plain: v => num(v),
  };

  // ---------- fitting ----------
  // OLS in log10-log10 space: log10 y = b + m log10 x
  function ols(pts) {
    const n = pts.length; if (n < 2) return null;
    let sx = 0, sy = 0, sxx = 0, sxy = 0, syy = 0;
    pts.forEach(([x, y]) => { const a = L10(x), c = L10(y); sx += a; sy += c; sxx += a * a; sxy += a * c; syy += c * c; });
    const mx = sx / n, my = sy / n, vxx = sxx / n - mx * mx, vxy = sxy / n - mx * my, vyy = syy / n - my * my;
    if (!(vxx > 0)) return null;
    const m = vxy / vxx;
    return { m, b: my - m * mx, r2: vyy > 0 ? (vxy * vxy) / (vxx * vyy) : 1, n };
  }
  // ordinary linear fit y = b + m x
  function linfit(pts) {
    const n = pts.length; let sx = 0, sy = 0, sxx = 0, sxy = 0, syy = 0;
    pts.forEach(([x, y]) => { sx += x; sy += y; sxx += x * x; sxy += x * y; syy += y * y; });
    const mx = sx / n, my = sy / n, vxx = sxx / n - mx * mx, vxy = sxy / n - mx * my, vyy = syy / n - my * my;
    const m = vxy / vxx; return { m, b: my - m * mx, r2: (vxy * vxy) / (vxx * vyy) };
  }
  const fitY = (f, x) => P10(f.b + f.m * L10(x));
  const gainOf = (fA, fB) => (fA && fB) ? Math.abs(fB.m) / Math.abs(fA.m) - 1 : NaN;
  const crossLog = (fA, fB) => (fA && fB && fA.m !== fB.m) ? (fB.b - fA.b) / (fA.m - fB.m) : NaN;

  // ---------- axes, ticks, mapping ----------
  function linTicks(a, b, n) {
    const span = b - a; if (!(span > 0)) return [a];
    const s0 = span / n, mag = P10(Math.floor(L10(s0))), r = s0 / mag;
    const step = (r < 1.5 ? 1 : r < 3 ? 2 : r < 7 ? 5 : 10) * mag; const out = [];
    for (let v = Math.ceil(a / step) * step; v <= b + 1e-9 * span; v += step) out.push(+v.toPrecision(12));
    return out;
  }
  function logTicks(la, lb) {
    const d0 = Math.ceil(la - 1e-9), d1 = Math.floor(lb + 1e-9), dec = [];
    for (let d = d0; d <= d1; d++) dec.push(d);
    if (dec.length >= 3) { const st = Math.ceil(dec.length / 7); return dec.filter(d => (d - d0) % st === 0).map(P10); }
    for (const ms of [[1, 2, 5], [1, 2, 3, 4, 5, 6, 7, 8, 9]]) {
      const t = [];
      for (let d = Math.floor(la); d <= Math.ceil(lb); d++) ms.forEach(m => { const v = m * P10(d), lv = L10(v); if (lv >= la - 1e-9 && lv <= lb + 1e-9) t.push(v); });
      if (t.length >= 3 && t.length <= 9) return t;
    }
    return linTicks(P10(la), P10(lb), 4).filter(v => v > 0);
  }
  function range(vals, pad, padLog) {
    const v = vals.filter(isFinite), lo = Math.min(...v), hi = Math.max(...v);
    const sp = (hi - lo) || Math.abs(hi) * 0.1 || 1;
    const out = { lin: [lo - sp * pad, hi + sp * pad], log: null };
    if (lo > 0) { const a = L10(lo), b = L10(hi), s = (b - a) || 0.1, pl = padLog == null ? pad : padLog; out.log = [a - s * pl, b + s * pl]; }
    return out;
  }
  // tx, ty in [0,1]: 0 = linear axis, 1 = log axis, in between = blend (used to morph the axes)
  function mapper(rect, xr, yr, tx, ty) {
    const nx = (v) => { let u = 0; if (tx < 1) u += (1 - tx) * (v - xr.lin[0]) / (xr.lin[1] - xr.lin[0]); if (tx > 0) u += tx * (L10(v) - xr.log[0]) / (xr.log[1] - xr.log[0]); return u; };
    const ny = (v) => { let u = 0; if (ty < 1) u += (1 - ty) * (v - yr.lin[0]) / (yr.lin[1] - yr.lin[0]); if (ty > 0) u += ty * (L10(v) - yr.log[0]) / (yr.log[1] - yr.log[0]); return u; };
    return { rect, xr, yr, tx, ty, X: (v) => rect.x + nx(v) * rect.w, Y: (v) => rect.y + rect.h - ny(v) * rect.h };
  }

  EBT.section({
    id: 'scaling',
    nav: 'Scaling laws',
    kicker: 'Lab',
    title: 'Scaling laws: who improves faster, and by how much?',
    lede: 'The abstract reports an <b>“up to 35% higher scaling rate”</b> than Transformer++. That number is a ratio of two slopes, and it comes from one axis. Here you refit every axis from the paper’s figures, see where the lines would cross, what the extra training compute costs, and what thinking buys at inference time.',
    mount(el, lib) {
      const h = lib.h, C = lib.C;
      const D = lib.data('scaling');
      if (!D || !D.plots) { el.appendChild(lib.callout('warn', 'Data missing', 'data/scaling.json is not in data/bundle.js. Run <code>python3 src/bundle_data.py</code>.')); return; }
      const P = {}; D.plots.forEach(p => { P[p.id] = p; });
      const RM = lib.reducedMotion;
      const monoF = (s, w = 400) => `${w} ${s}px "JetBrains Mono", Menlo, monospace`;
      const bodyF = (s, w = 400) => `${w} ${s}px "Atkinson Hyperlegible", "Segoe UI", sans-serif`;
      const redraws = [];

      // ---------- shared canvas plumbing ----------
      // Responsive canvas: logical width = CSS width, so text stays legible on phones.
      function rcanvas(parent, o) {
        const box = h('div', { class: 'canvas-box scaling-cbox' }); parent.appendChild(box);
        const c = h('canvas', { role: 'img', 'aria-label': o.label || '' }); box.appendChild(c);
        const S = { canvas: c, ctx: c.getContext('2d'), w: 0, h: 0, box, draw: o.draw || (() => {}) };
        S.fit = () => {
          const w = Math.max(260, Math.round(box.clientWidth || 600));
          const asp = typeof o.aspect === 'function' ? o.aspect(w) : o.aspect;
          const hh = Math.round(clamp(w * asp, o.minH, o.maxH));
          if (w === S.w && hh === S.h) return false;
          const dpr = Math.min(2, window.devicePixelRatio || 1);
          S.w = w; S.h = hh; c.width = Math.round(w * dpr); c.height = Math.round(hh * dpr);
          S.ctx.setTransform(dpr, 0, 0, dpr, 0, 0); return true;
        };
        S.toLocal = (ev) => { const r = c.getBoundingClientRect(); return [(ev.clientX - r.left) / r.width * S.w, (ev.clientY - r.top) / r.height * S.h]; };
        S.fit();
        if (window.ResizeObserver) new ResizeObserver(() => { if (S.fit()) S.draw(); }).observe(box);
        redraws.push(() => S.draw());
        return S;
      }
      // tween helper: named values ease toward targets inside one rAF loop
      function animator(draw) {
        const tw = {}; let extra = null;
        const loop = lib.loop((dt) => {
          let active = false;
          for (const k in tw) {
            const t = tw[k];
            if (t.v !== t.target) { t.v += (t.target - t.v) * (1 - Math.exp(-dt * t.rate)); if (Math.abs(t.target - t.v) < t.eps) t.v = t.target; else active = true; }
          }
          if (extra && extra(dt)) active = true;
          draw(); return active;
        });
        return {
          add(k, v, rate = 6, eps = 1e-3) { tw[k] = { v, target: v, rate, eps }; },
          get: (k) => tw[k].v,
          target: (k) => tw[k].target,
          set(k, v) { tw[k].v = v; tw[k].target = v; },
          to(k, target) { tw[k].target = target; if (RM) tw[k].v = target; },
          extra(fn) { extra = fn; },
          kick() { if (RM) { for (const k in tw) tw[k].v = tw[k].target; if (extra) extra(1e3); draw(); return; } loop.start(); },
          stop() { loop.stop(); },
        };
      }
      // draws grid, ticks (cross-faded between linear and log sets) and axis titles; returns the data->pixel mapping
      function frame(ctx, W, H, o) {
        const fs = W < 480 ? 12 : 12.5;
        const sets = (rg, t, n) => { const out = []; if (t < 0.999) out.push({ ticks: linTicks(rg.lin[0], rg.lin[1], n), a: 1 - t }); if (t > 0.001 && rg.log) out.push({ ticks: logTicks(rg.log[0], rg.log[1]), a: t }); return out; };
        const xs = sets(o.xr, o.tx, W < 480 ? 4 : 6), ys = sets(o.yr, o.ty, H < 300 ? 4 : 5);
        ctx.save(); ctx.font = monoF(fs);
        let yw = 0; ys.forEach(s => s.ticks.forEach(v => { yw = Math.max(yw, ctx.measureText(o.yfmt(v)).width); }));
        const y0 = o.y0 || 0, y1 = o.y1 != null ? o.y1 : H;
        const rect = { x: o.left != null ? o.left : Math.ceil(yw) + 30, y: y0 + (o.top != null ? o.top : 14), w: 0, h: 0 };
        rect.w = W - rect.x - (o.right != null ? o.right : 14); rect.h = y1 - rect.y - (o.noX ? 10 : 46);
        const M = mapper(rect, o.xr, o.yr, o.tx, o.ty); M.W = W; M.H = H; M.fs = fs;
        const inX = (x) => x >= rect.x - 0.5 && x <= rect.x + rect.w + 0.5, inY = (y) => y >= rect.y - 0.5 && y <= rect.y + rect.h + 0.5;
        ctx.lineWidth = 1;
        xs.forEach(s => { ctx.globalAlpha = s.a; ctx.strokeStyle = C.rule; s.ticks.forEach(v => { const x = M.X(v); if (!inX(x)) return; ctx.beginPath(); ctx.moveTo(x, rect.y); ctx.lineTo(x, rect.y + rect.h); ctx.stroke(); }); });
        ys.forEach(s => { ctx.globalAlpha = s.a; ctx.strokeStyle = C.rule; s.ticks.forEach(v => { const y = M.Y(v); if (!inY(y)) return; ctx.beginPath(); ctx.moveTo(rect.x, y); ctx.lineTo(rect.x + rect.w, y); ctx.stroke(); }); });
        ctx.fillStyle = C.muted;
        if (!o.noX) xs.forEach(s => {
          ctx.globalAlpha = s.a; ctx.textAlign = 'center'; ctx.textBaseline = 'top'; let lastR = -1e9;
          s.ticks.forEach(v => { const x = M.X(v); if (!inX(x)) return; const lab = o.xfmt(v), w = ctx.measureText(lab).width; let cx = Math.min(x, W - 2 - w / 2); if (cx - w / 2 < lastR + 8) return; ctx.fillText(lab, cx, rect.y + rect.h + 6); lastR = cx + w / 2; });
        });
        ys.forEach(s => {
          ctx.globalAlpha = s.a; ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; let lastY = 1e9;
          s.ticks.forEach(v => { const y = M.Y(v); if (!inY(y)) return; if (Math.abs(lastY - y) < fs + 3) return; ctx.fillText(o.yfmt(v), rect.x - 6, y); lastY = y; });
        });
        ctx.globalAlpha = 1; ctx.strokeStyle = C.faint; ctx.lineWidth = 1.2;
        ctx.beginPath(); ctx.moveTo(rect.x, rect.y); ctx.lineTo(rect.x, rect.y + rect.h); ctx.lineTo(rect.x + rect.w, rect.y + rect.h); ctx.stroke();
        ctx.fillStyle = C.muted; ctx.font = bodyF(fs + 0.5); ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
        if (!o.noX && o.xlabel) ctx.fillText(fitStr(ctx, o.xlabel, rect.w), rect.x + rect.w / 2, y1 - 4);
        ctx.save(); ctx.translate(4, rect.y + rect.h / 2); ctx.rotate(-Math.PI / 2); ctx.textBaseline = 'top'; ctx.fillText(fitStr(ctx, o.ylabel, rect.h), 0, 0); ctx.restore();
        ctx.restore();
        return M;
      }
      function fitStr(ctx, s, maxW) { if (ctx.measureText(s).width <= maxW) return s; let t = s; while (t.length > 4 && ctx.measureText(t + '…').width > maxW) t = t.slice(0, -1); return t + '…'; }
      function hatch(ctx) { const c = document.createElement('canvas'); c.width = c.height = 9; const g = c.getContext('2d'); g.strokeStyle = lib.rgba(C.ebt, 0.17); g.lineWidth = 1; g.beginPath(); g.moveTo(-1, 10); g.lineTo(10, -1); g.stroke(); return ctx.createPattern(c, 'repeat'); }
      function polyline(ctx, pts, col, o = {}) {
        if (pts.length < 2) return; ctx.save(); ctx.strokeStyle = col; ctx.lineWidth = o.w || 2; ctx.globalAlpha = o.alpha == null ? 1 : o.alpha;
        if (o.dash) ctx.setLineDash(o.dash); ctx.lineJoin = 'round'; ctx.lineCap = 'round';
        ctx.beginPath(); ctx.moveTo(pts[0][0], pts[0][1]); for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]); ctx.stroke(); ctx.restore();
      }
      function dot(ctx, x, y, r, fill, o = {}) {
        ctx.save(); ctx.globalAlpha = o.alpha == null ? 1 : o.alpha; ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2);
        if (fill) { ctx.fillStyle = fill; ctx.fill(); }
        if (o.stroke) { ctx.strokeStyle = o.stroke; ctx.lineWidth = o.lw || 1.5; ctx.stroke(); }
        ctx.restore();
      }
      function star(ctx, x, y, r, col) { ctx.save(); ctx.beginPath(); for (let i = 0; i < 10; i++) { const a = -Math.PI / 2 + i * Math.PI / 5, rr = i % 2 ? r * 0.45 : r; ctx.lineTo(x + rr * Math.cos(a), y + rr * Math.sin(a)); } ctx.closePath(); ctx.fillStyle = col; ctx.fill(); ctx.strokeStyle = C.bg; ctx.lineWidth = 1.2; ctx.stroke(); ctx.restore(); }
      // text with a dark halo so it stays readable over lines
      function label(ctx, s, x, y, o = {}) {
        ctx.save(); ctx.font = o.mono ? monoF(o.size || 12.5, o.weight || 400) : bodyF(o.size || 13, o.weight || 400);
        ctx.textAlign = o.align || 'left'; ctx.textBaseline = o.baseline || 'middle'; ctx.globalAlpha = o.alpha == null ? 1 : o.alpha;
        ctx.lineJoin = 'round'; ctx.strokeStyle = lib.rgba(C.bg, 0.85); ctx.lineWidth = 4; ctx.strokeText(s, x, y);
        ctx.fillStyle = o.color || C.ink; ctx.fillText(s, x, y); const w = ctx.measureText(s).width; ctx.restore(); return w;
      }
      function boxText(ctx, lines, x, y, o = {}) { // lines: [{t, color, size, mono, weight}] ; x,y = anchor (top-left unless align right)
        ctx.save(); const pad = 8; let w = 0, hh = 0;
        lines.forEach(l => { ctx.font = l.mono ? monoF(l.size || 12.5, l.weight || 400) : bodyF(l.size || 13, l.weight || 400); w = Math.max(w, ctx.measureText(l.t).width); hh += (l.size || 13) * 1.35; });
        let bx = o.alignRight ? x - w - 2 * pad : x, by = y;
        if (o.clampW) bx = clamp(bx, 2, o.clampW - w - 2 * pad - 2);
        if (o.clampH) by = clamp(by, 2, o.clampH - hh - 2 * pad - 2);
        ctx.globalAlpha = o.alpha == null ? 1 : o.alpha;
        ctx.fillStyle = lib.rgba(C.panel, 0.94); ctx.strokeStyle = o.stroke || C.rule; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.roundRect(bx, by, w + 2 * pad, hh + 2 * pad - 4, 8); ctx.fill(); ctx.stroke();
        let cy = by + pad; ctx.textBaseline = 'top'; ctx.textAlign = 'left';
        lines.forEach(l => { ctx.font = l.mono ? monoF(l.size || 12.5, l.weight || 400) : bodyF(l.size || 13, l.weight || 400); ctx.fillStyle = l.color || C.ink; ctx.fillText(l.t, bx + pad, cy); cy += (l.size || 13) * 1.35; });
        ctx.restore(); return { x: bx, y: by, w: w + 2 * pad, h: hh + 2 * pad };
      }
      function figCrop(src, cap) {
        return h('figure', { class: 'scaling-crop' }, h('div', { class: 'paper-fig' }, h('img', { src, alt: cap })), h('figcaption', { html: cap }));
      }
      function roCell(k) { const v = h('b', { class: 'v' }), s = h('span', { class: 's' }); const el = h('div', { class: 'scaling-ro-cell' }, h('span', { class: 'k' }, k), v, s); return { el, set(val, sub) { v.innerHTML = val; s.innerHTML = sub || ''; } }; }
      const btn = (txt, o = {}) => h('button', Object.assign({ type: 'button', class: 'btn' + (o.primary ? ' primary' : '') }, o.attrs || {}), txt);

      // ================================================================
      // Intro
      // ================================================================
      el.appendChild(h('div', { class: 'prose' },
        h('p', { html: 'The paper asks two different scaling questions. <b>Learning scalability</b>: how fast does validation perplexity fall when a model gets more data, bigger batches, more depth, more parameters, more FLOPs or more width? <b>Thinking scalability</b>: how much does extra compute on a single prediction help? The “35%” answers the first question, for the data axis.' }),
      ));
      el.appendChild(lib.callout('insight', 'Read “up to 35%” carefully', '<ul class="scaling-ul">' +
        '<li>It is the <b>best</b> axis (data, 35.98%, Fig 4a). Batch size 28.46%, depth 5.29%, parameters 2.91%, FLOPs 2.92%, width 0.02%. For parameters and FLOPs the paper itself says EBTs <i>“slightly out-scale”</i> Transformer++ (p.10).</li>' +
        '<li>It compares <b>slopes, not heights</b>. At the measured sizes EBT has the higher perplexity on depth, parameters, FLOPs and video. It crosses below Transformer++ only on data and batch size.</li>' +
        '<li>Everything here is read off the paper’s plots (approx.). The printed percentages are exact; refits are ours and match them to rounding.</li></ul>'));

      // ================================================================
      // LAB 1: refit a scaling law
      // ================================================================
      const AX = [
        { key: 'data', id: 'fig4a', group: 'Text · RedPajamaV2', name: 'Data', kind: 'tokB', what: 'training data', per: 'tokens',
          note: 'xxs models (6.18M non-embedding parameters), batch 128; only the number of training tokens changes (p.34). The paper draws smooth trend curves in this panel; its rate comes from straight log–log fits.' },
        { key: 'batch', id: 'fig4b', group: 'Text · RedPajamaV2', name: 'Batch', kind: 'tokK', what: 'batch size', per: 'batch size',
          note: 'xxs models, 105k steps each, only the batch size changes (p.34). With a fixed step count, a bigger batch also means more training tokens [derived].' },
        { key: 'depth', id: 'fig4c', group: 'Text · RedPajamaV2', name: 'Depth', kind: 'count', what: 'depth (Transformer blocks)', per: 'depth',
          note: 'Only the number of Transformer blocks changes (p.34). EBT stays above Transformer++ at every measured depth; the fits meet just past 16 blocks.' },
        { key: 'params', id: 'fig5a', group: 'Text · RedPajamaV2', name: 'Params', kind: 'paramM', what: 'non-embedding parameters', per: 'parameters',
          note: 'xxs to large (6.18M to 396M non-embedding), 105k steps, batch 32 to 256 (p.34). The paper: EBTs “slightly out-scale” here (p.10).' },
        { key: 'flops', id: 'fig5b', group: 'Text · RedPajamaV2', name: 'FLOPs', kind: 'flops', what: 'training compute', per: 'FLOPs',
          note: 'The same runs as Parameters, plotted against training FLOPs. EBT points sit about 6.66× further right because each EBT training step costs more (p.36). At equal FLOPs EBT is worse across the whole measured range.' },
        { key: 'width', id: 'fig5c', group: 'Text · RedPajamaV2', name: 'Width', kind: 'count', what: 'embedding width', per: 'width',
          note: 'Only the embedding dimension changes; these widths are not listed in Table D.1. The two fits are practically the same line (0.02%).' },
        { key: 'vwidth', id: 'fig9a', group: 'Video · Something-Something V2', name: 'Width', kind: 'count', what: 'embedding width', per: 'width',
          note: 'Next-frame prediction in SD-XL VAE latents; y is the minimum validation Smooth-L1 loss (p.12). The paper shows width and parameters because these axes “behaved the most linearly” (p.12).' },
        { key: 'vparams', id: 'fig9b', group: 'Video · Something-Something V2', name: 'Params', kind: 'paramM', what: 'non-embedding parameters', per: 'parameters',
          note: 'xxs to xl (6.18M to 708M non-embedding), batch 256 (p.34). EBT loss is higher at every size; its slope is about 34% steeper.' },
        { key: 'fw', id: 'figB3a', group: 'Bigger run · FineWeb', name: 'Data', kind: 'tokB', what: 'training data', per: 'tokens',
          note: 'Appendix run: FineWeb, small models, batch 256, context 1024, 500k steps (p.28). EBT crosses below Transformer++ near 90B tokens.' },
        { key: 'fwz', id: 'figB3b', group: 'Bigger run · FineWeb', name: 'Zoomed', long: 'Data from 51B (zoomed)', kind: 'tokB', what: 'training data', per: 'tokens',
          note: 'The paper’s zoomed panel of the same FineWeb runs, fitted only from about 51B tokens on. Same models, narrower window, 51.70%: the rate depends on where you fit.' },
        { key: 's2', id: 'figC1', group: 'Variant · two EBT recipes', name: 'S2 vs S1', kind: 'paramM', what: 'non-embedding parameters', per: 'parameters',
          note: 'Both are EBTs. S1 is tuned for stable pretraining, S2 for thinking (landscape regularization) (p.30). Tiny models (0.9M to 12.4M). Many of the scaling runs on the other axes use S1 (p.30).' },
      ];
      AX.forEach(a => { a.p = P[a.id]; });
      const AXOK = AX.filter(a => a.p && a.p.series && a.p.series.length === 2);
      const axByKey = {}; AXOK.forEach(a => { axByKey[a.key] = a; });
      const isLoss = (ax) => /loss/i.test(ax.p.y.label);
      const metric = (ax) => isLoss(ax) ? 'loss' : 'perplexity';
      const metricShort = (ax) => isLoss(ax) ? 'loss' : 'ppl';
      const names = (ax) => ax.key === 's2' ? ['EBT-S1', 'EBT-S2'] : ['Transformer++', 'EBT'];
      const shortNames = (ax) => ax.key === 's2' ? ['S1', 'S2'] : ['T++', 'EBT'];
      const colA = (ax) => ax.key === 's2' ? C.muted : C.ar;
      const cleanLabel = (s) => s.replace(/\s*\(log scale\)/i, '').replace(/,\s*log scale/i, '');
      const paperWarp = (ax) => (ax.p.x.log ? 1 : 0);

      function analyze(ax, drop) {
        const p = ax.p, A = p.series[0], B = p.series[1], n = Math.min(A.points.length, B.points.length);
        const act = (i) => !drop.has(i);
        const aPts = A.points.filter((_, i) => act(i)), bPts = B.points.filter((_, i) => act(i));
        const fA = ols(aPts), fB = ols(bPts), gain = gainOf(fA, fB);
        const idx = []; for (let i = 0; i < n; i++) if (act(i)) idx.push(i);
        let jk = null;
        if (idx.length >= 4) {
          let lo = Infinity, hi = -Infinity;
          idx.forEach(j => { const g = gainOf(ols(A.points.filter((_, i) => act(i) && i !== j)), ols(B.points.filter((_, i) => act(i) && i !== j))); lo = Math.min(lo, g); hi = Math.max(hi, g); });
          jk = [lo, hi];
        }
        const xsAll = A.points.concat(B.points).map(q => q[0]);
        const xmin = Math.min(...xsAll), xmax = Math.max(...xsAll);
        const lastA = Math.max(...aPts.map(q => q[0])), lastB = Math.max(...bPts.map(q => q[0]));
        const xc = Math.min(lastA, lastB), sameX = Math.abs(lastA - lastB) / lastA < 0.02;
        const ya = sameX ? aPts.find(q => q[0] === lastA)[1] : fitY(fA, xc), yb = sameX ? bPts.find(q => q[0] === lastB)[1] : fitY(fB, xc);
        const yaF = fitY(fA, xc), ybF = fitY(fB, xc);
        return { ax, p, A, B, n, aPts, bPts, fA, fB, gain, jk, xmin, xmax, Lc: crossLog(fA, fB), xc, ya, yb, yaF, ybF, sameX, nAct: idx.length };
      }
      // describe where the two fits cross, relative to the measured range
      function crossInfo(R) {
        const [na, nb] = shortNames(R.ax), fx = KIND[R.ax.kind];
        if (!isFinite(R.gain) || !isFinite(R.Lc)) return { v: 'parallel', s: 'The fitted lines are parallel.', inRange: false, beyond: Infinity };
        if (Math.abs(R.gain) < 0.005) return { v: 'no real crossing', s: `The slopes differ by ${pctPlain(R.gain)}: the lines are practically parallel, so a crossing point means little.`, inRange: false, beyond: Infinity, flat: true };
        const xs = P10(R.Lc), lmin = L10(R.xmin), lmax = L10(R.xmax);
        const steeperB = R.gain > 0;
        if (R.Lc < lmin) return { v: steeperB ? `${nb} lower everywhere` : `${na} lower everywhere`, s: `The fits cross at ≈${fx(xs)}, below the measured range.`, inRange: false, beyond: -Infinity, xs };
        if (R.Lc <= lmax) return { v: '≈' + fx(xs), s: `Inside the measured range. Beyond it the ${steeperB ? nb : na} fit is lower.`, inRange: true, beyond: 0, xs };
        const k = xs / R.xmax;
        return { v: '≈' + fx(xs), s: `${mult(k)} beyond the largest run: an extrapolation${k > 1e6 ? ', far outside any plausible range' : ''}.`, inRange: false, beyond: L10(k), xs };
      }

      const S1 = { axis: 'data', stage: 4, drop: new Set(), hover: null, curve: false, playing: null };
      let R1 = analyze(axByKey[S1.axis], S1.drop);

      el.appendChild(h('h3', { class: 'scaling-h' }, h('span', { class: 'n' }, '1'), 'Refit a scaling law yourself'));
      el.appendChild(h('p', { class: 'scaling-sub', html: 'Pick an axis. Each button’s bar is the gain printed in that figure’s title. Then walk through how a “% faster” number is made, drop points to test how robust it is, and extend the lines to see where they would cross.' }));
      const bench1 = h('div', { class: 'bench scaling-bench' }); el.appendChild(bench1);
      const ctl1 = h('div', { class: 'card stack scaling-ctl' }); bench1.appendChild(ctl1);
      const ins1 = h('div', { class: 'card stack scaling-ins' }); bench1.appendChild(ins1);

      // axis picker with printed-gain bars
      ctl1.appendChild(h('h4', {}, 'Scaling axis'));
      const picker = h('div', { class: 'scaling-picker', role: 'group', 'aria-label': 'Scaling axis' }); ctl1.appendChild(picker);
      const gmax = Math.max(...AXOK.map(a => a.p.rate_gain_printed || 0));
      const axBtns = {}; let grpEl = null, grpName = null;
      AXOK.forEach(a => {
        if (a.group !== grpName) { grpName = a.group; grpEl = h('div', { class: 'opts' }); picker.appendChild(h('div', { class: 'scaling-group' }, h('div', { class: 'gl' }, a.group), grpEl)); }
        const g = a.p.rate_gain_printed || 0;
        const b = h('button', { type: 'button', class: 'scaling-ax', 'aria-pressed': String(a.key === S1.axis), title: a.p.printed_title || '' },
          h('span', { class: 'nm' }, a.name, h('span', { class: 'fig' }, a.p.figure.replace('Fig ', '').replace('.', ''))),
          h('span', { class: 'val' }, (g * 100).toFixed(2) + '%'),
          h('span', { class: 'bar', 'aria-hidden': 'true' }, h('i', { style: `width:${Math.max(1.5, g / gmax * 100).toFixed(1)}%` })));
        b.addEventListener('click', () => selectAxis(a.key));
        axBtns[a.key] = b; grpEl.appendChild(b);
      });

      // walkthrough
      const STAGES = [
        { t: 'The measurements', d: (ax) => `Each dot is one trained model, read off ${ax.p.figure}. x is how much ${ax.what} it got; y is its validation ${metric(ax)}. Lower is better.` },
        { t: 'Log both axes', d: () => 'A scaling law is a power law, y = a·xᵐ. On log–log axes every power law becomes a straight line, and the exponent m is its slope.' },
        { t: 'Fit one line per model', d: () => 'Least squares on (log₁₀ x, log₁₀ y) gives each model its exponent m. This is the fit behind the paper’s “X% faster” titles; it reproduces every printed number to rounding.' },
        { t: 'Read the slope', d: (ax) => `m says what 10× more ${ax.per} buys: y gets multiplied by 10ᵐ. The triangles show one step of the fit. A more negative m means faster improvement.` },
        { t: 'Compare the slopes', d: (ax) => `Scaling-rate gain = |m ${shortNames(ax)[1]}| ÷ |m ${shortNames(ax)[0]}| − 1. It compares slopes, not heights: a model can scale faster and still be worse at every measured point.` },
        { t: 'Extrapolate, carefully', d: () => 'Extend both lines past the last measurement. If both power laws kept holding, the steeper model would lead after the crossing. Nobody has measured that region.' },
      ];
      ctl1.appendChild(h('h4', { style: 'margin-top:6px' }, 'Walk through the fit'));
      const bStep = btn('Step ▸', { primary: true }), bPlay = btn('Play', { attrs: { 'aria-pressed': 'false' } }), bReset = btn('Reset');
      const dots = h('div', { class: 'scaling-dots', role: 'group', 'aria-label': 'Walkthrough stage' });
      const dotBtns = STAGES.map((s, i) => { const b = h('button', { type: 'button', 'aria-label': `Stage ${i + 1}: ${s.t}` }, String(i + 1)); b.addEventListener('click', () => { stopPlay(); setStage(i); }); dots.appendChild(b); return b; });
      ctl1.appendChild(h('div', { class: 'row' }, bStep, bPlay, bReset));
      ctl1.appendChild(dots);
      const capT = h('b', { class: 'ct' }), capD = h('p', {});
      ctl1.appendChild(h('div', { class: 'scaling-cap', 'aria-live': 'polite' }, capT, capD));

      // sliders
      const warpS = lib.slider({ id: 'scaling-warp', label: 'Axes: linear ↔ log–log', min: 0, max: 1, step: 0.01, value: 1, fmt: v => v < 0.02 ? 'linear' : v > 0.98 ? 'log–log' : Math.round(v * 100) + '% log', oninput: v => { anim1.set('warp', v); anim1.kick(); } });
      ctl1.appendChild(warpS.el);
      const warpHint = h('p', { class: 'scaling-hint' }); ctl1.appendChild(warpHint);
      const extS = lib.slider({ id: 'scaling-ext', label: 'Extend the fits', min: 0, max: 3, step: 0.01, value: 0, fmt: v => v < 0.005 ? 'off (×1)' : mult(P10(v)) + ' largest run', oninput: v => { setExt(v, true); } });
      ctl1.appendChild(extS.el);
      const bCross = btn('Go to the crossing'); bCross.classList.add('scaling-small');
      const b1000 = btn('Try ×1000'); b1000.classList.add('scaling-small');
      bCross.addEventListener('click', () => { stopPlay(); setExt(crossEk(), false); });
      b1000.addEventListener('click', () => { stopPlay(); setExt(3, false); });
      ctl1.appendChild(h('div', { class: 'row' }, bCross, b1000));
      ctl1.appendChild(h('p', { class: 'scaling-hint', html: 'The paper expects EBTs to be “significantly better” at “1,000× more data with models 1,000× larger” (p.9). Try ×1000 on each axis.' }));
      const curveId = 'scaling-curve';
      const curveCb = h('input', { type: 'checkbox', id: curveId });
      const curveRow = h('label', { class: 'scaling-check', for: curveId }, curveCb, h('span', {}, 'Show the paper’s drawn trend curve'));
      curveCb.addEventListener('change', () => { S1.curve = curveCb.checked; draw1(); });
      ctl1.appendChild(curveRow);

      // instrument
      const head1 = h('div', { class: 'scaling-head' });
      const t1 = h('h3', { class: 'scaling-ptitle' }), sub1 = h('div', { class: 'scaling-psub' });
      const badge1 = h('span', {});
      head1.appendChild(h('div', {}, t1, sub1)); head1.appendChild(badge1);
      ins1.appendChild(head1);
      const cv1 = rcanvas(ins1, { aspect: (w) => w < 560 ? 1.15 : 0.6, minH: 340, maxH: 480, label: 'Scaling-law chart: validation perplexity versus the chosen resource for Transformer++ and EBT, with least-squares power-law fits.', draw: () => draw1() });
      const legend1 = h('div', { class: 'legend' }); ins1.appendChild(legend1);
      const chipsWrap = h('div', { class: 'scaling-chips-wrap' });
      const chips = h('div', { class: 'scaling-chips', role: 'group', 'aria-label': 'Model sizes used in the fit' });
      chipsWrap.appendChild(h('span', { class: 'scaling-chips-l' }, 'In the fit (tap a dot or a chip to drop it):'));
      chipsWrap.appendChild(chips); ins1.appendChild(chipsWrap);
      const ro1 = h('div', { class: 'scaling-ro' }); ins1.appendChild(ro1);
      const c1A = roCell('Exponent m, baseline'), c1B = roCell('Exponent m, EBT'), c1G = roCell('Scaling-rate gain'), c1L = roCell('At the largest measured point'), c1X = roCell('Where the fits cross'), c1E = roCell('Extrapolated');
      [c1A, c1B, c1G, c1L, c1X, c1E].forEach(c => ro1.appendChild(c.el));
      const foot1 = h('div', { class: 'scaling-foot' }); ins1.appendChild(foot1);
      const crop1 = h('div', {}), note1 = h('div', { class: 'scaling-note' });
      foot1.appendChild(crop1); foot1.appendChild(note1);

      const anim1 = animator(() => draw1());
      anim1.add('warp', 1, 4.5); anim1.add('ek', 0, 4.5); anim1.add('fit', 1, 4); anim1.add('tri', 1, 5); anim1.add('rate', 1, 5);
      let M1 = null, hatch1 = null;

      function crossEk() {
        const ci = crossInfo(R1);
        if (ci.inRange || ci.beyond === -Infinity || ci.flat) return 1;
        if (!isFinite(ci.beyond)) return 1;
        return Math.min(3, ci.beyond + 0.25);
      }
      function setExt(v, fromSlider) {
        if (!fromSlider) extS.set(v);
        anim1.to('ek', v);
        if (v > 0.004) anim1.to('fit', 1); else if (S1.stage < 2) anim1.to('fit', 0);
        anim1.kick(); updateReadouts1();
      }
      function setStage(s) {
        const prev = S1.stage; S1.stage = s;
        anim1.to('warp', s >= 1 ? 1 : paperWarp(axByKey[S1.axis]));
        if (s >= 2 && prev < 2 && anim1.target('ek') < 0.004) anim1.set('fit', 0);
        anim1.to('fit', s >= 2 || anim1.target('ek') > 0.004 ? 1 : 0);
        if (s >= 3 && prev < 3) anim1.set('tri', 0);
        anim1.to('tri', s >= 3 ? 1 : 0);
        if (s >= 4 && prev < 4) anim1.set('rate', 0);
        anim1.to('rate', s >= 4 ? 1 : 0);
        if (s >= 5) { const v = crossEk(); extS.set(v); anim1.to('ek', v); }
        else if (prev >= 5) { extS.set(0); anim1.to('ek', 0); }
        warpS.set(anim1.target('warp'));
        anim1.kick(); updateStageUI(); updateReadouts1();
      }
      function updateStageUI() {
        const ax = axByKey[S1.axis];
        capT.textContent = `Stage ${S1.stage + 1} of ${STAGES.length} · ${STAGES[S1.stage].t}`;
        capD.textContent = STAGES[S1.stage].d(ax);
        dotBtns.forEach((b, i) => { b.setAttribute('aria-current', i === S1.stage ? 'step' : 'false'); b.classList.toggle('done', i < S1.stage); });
        bStep.textContent = S1.stage >= STAGES.length - 1 ? 'Start over' : 'Step ▸';
      }
      function stopPlay() { if (S1.playing) { clearTimeout(S1.playing); S1.playing = null; } bPlay.textContent = 'Play'; bPlay.setAttribute('aria-pressed', 'false'); }
      bStep.addEventListener('click', () => { stopPlay(); setStage(S1.stage >= STAGES.length - 1 ? 0 : S1.stage + 1); });
      bPlay.addEventListener('click', () => {
        if (S1.playing) { stopPlay(); return; }
        if (S1.stage >= STAGES.length - 1) { S1.drop.clear(); refresh1(); setStage(0); }
        bPlay.textContent = 'Pause'; bPlay.setAttribute('aria-pressed', 'true');
        const tick = () => { if (S1.stage >= STAGES.length - 1) { stopPlay(); return; } setStage(S1.stage + 1); S1.playing = S1.stage >= STAGES.length - 1 ? null : setTimeout(tick, 2700); if (!S1.playing) stopPlay(); };
        S1.playing = setTimeout(tick, S1.stage === 0 ? 1200 : 400);
      });
      bReset.addEventListener('click', () => { stopPlay(); S1.drop.clear(); refresh1(); extS.set(0); anim1.to('ek', 0); setStage(0); });

      function selectAxis(key) {
        S1.axis = key; S1.drop = new Set(); S1.hover = null;
        Object.entries(axBtns).forEach(([k, b]) => b.setAttribute('aria-pressed', String(k === key)));
        if (S1.stage === 0) { anim1.to('warp', paperWarp(axByKey[key])); warpS.set(paperWarp(axByKey[key])); }
        if (S1.stage >= 5) { const v = crossEk2(key); extS.set(v); anim1.to('ek', v); }
        refresh1(); updateStageUI(); anim1.kick();
      }
      function crossEk2(key) { R1 = analyze(axByKey[key], S1.drop); return crossEk(); }
      function refresh1() {
        const ax = axByKey[S1.axis]; R1 = analyze(ax, S1.drop);
        const p = ax.p, [na, nb] = names(ax);
        t1.textContent = '“' + (p.printed_title || p.title) + '”';
        sub1.textContent = `${p.figure} (p.${p.page}) · ${p.title}`;
        badge1.replaceChildren(lib.badge('paper', 'approx., read from ' + p.figure));
        legend1.innerHTML = `<span><i style="background:${colA(ax)}"></i>${na}</span><span><i style="background:${C.ebt}"></i>${nb}</span>` +
          `<span><i class="scaling-li-line"></i>least-squares fit (log–log)</span><span><i class="scaling-li-dash"></i>extended fit</span><span><i class="scaling-li-hollow"></i>dropped from fit</span>`;
        // chips: one per model size (pairs share an index)
        chips.replaceChildren();
        const labAt = (i) => ax.key === 'flops' ? KIND.paramM(P.fig5a.series[0].points[i][0]) + ' model' : KIND[ax.kind](R1.A.points[i][0]);
        for (let i = 0; i < R1.n; i++) {
          const on = !S1.drop.has(i);
          const c = h('button', { type: 'button', class: 'scaling-chip', 'aria-pressed': String(on), 'aria-label': `${labAt(i)}: ${on ? 'in the fit, press to drop' : 'dropped, press to restore'}` }, labAt(i));
          c.addEventListener('click', () => toggleDrop(i)); chips.appendChild(c);
        }
        const hasQuad = p.series.some(s => s.paper_curve && /c0/.test(s.paper_curve.form || ''));
        curveRow.hidden = !hasQuad; if (!hasQuad) { curveCb.checked = false; S1.curve = false; }
        warpHint.textContent = `The paper draws ${p.figure} on ${p.x.log ? 'log–log' : 'linear'} axes.`;
        crop1.replaceChildren(figCrop(p.image, `Original ${p.figure} (p.${p.page}). In the paper Transformer++ is orange and EBT blue${ax.key === 's2' ? ' (here S1 orange, S2 blue)' : ''}; this site uses periwinkle and amber.`));
        note1.innerHTML = `<b class="t">About these runs</b><p>${ax.note}</p>` + (p.note ? `<p class="scaling-faint">${p.note}</p>` : '');
        updateReadouts1(); draw1();
      }
      function toggleDrop(i) {
        if (S1.drop.has(i)) S1.drop.delete(i);
        else { if (R1.n - S1.drop.size <= 3) { flashChips(); return; } S1.drop.add(i); }
        R1 = analyze(axByKey[S1.axis], S1.drop);
        [...chips.children].forEach((c, j) => { const on = !S1.drop.has(j); c.setAttribute('aria-pressed', String(on)); });
        if (S1.stage < 2) setStage(4); else { updateReadouts1(); draw1(); }
      }
      function flashChips() { chipsWrap.classList.remove('scaling-flash'); void chipsWrap.offsetWidth; chipsWrap.classList.add('scaling-flash'); }

      function updateReadouts1() {
        const R = R1, ax = R.ax, [na, nb] = shortNames(ax), ms = metricShort(ax), p = ax.p;
        const per = (f) => `10× ${ax.per} → ${ms} ×${num(P10(f.m), 3)} (${pct(P10(f.m) - 1, 0)}) · r² ${f.r2.toFixed(3)}`;
        c1A.el.querySelector('.k').textContent = `Exponent m, ${names(ax)[0]}`;
        c1B.el.querySelector('.k').textContent = `Exponent m, ${names(ax)[1]}`;
        c1A.set(R.fA ? 'm = ' + neg(R.fA.m.toFixed(3)) : '–', R.fA ? per(R.fA) : '');
        c1B.set(R.fB ? 'm = ' + neg(R.fB.m.toFixed(3)) : '–', R.fB ? per(R.fB) : '');
        const printed = p.rate_gain_printed;
        const jk = R.jk ? ` · drop any one size: ${pctPlain(R.jk[0], 1)} to ${pctPlain(R.jk[1], 1)}` : '';
        const dropped = S1.drop.size ? ` · ${S1.drop.size} size${S1.drop.size > 1 ? 's' : ''} dropped` : '';
        c1G.set(`<span class="amber">${pct(R.gain, 2)}</span>`, `refit from ${R.nAct} sizes${dropped}. Printed in the title: ${(printed * 100).toFixed(2)}%${jk}`);
        const diff = R.yb / R.ya - 1;
        const where = R.sameX ? `${KIND[ax.kind](R.xc)} (measured)` : `${KIND[ax.kind](R.xc)} (from the fits, equal ${ax.per})`;
        c1L.set(`${nb} ${num(R.yb, 4)} · ${na} ${num(R.ya, 4)}`, `at ${where}: ${nb} is ${diff < 0 ? 'lower' : 'higher'} by ${Math.abs(diff * 100).toFixed(1)}%`);
        const ci = crossInfo(R); c1X.set(ci.v, ci.s);
        const ek = extS.value();
        if (ek > 0.004 && R.fA && R.fB) {
          const xk = R.xmax * P10(ek), ya = fitY(R.fA, xk), yb = fitY(R.fB, xk);
          c1E.set(`${nb} ${num(yb, 3)} · ${na} ${num(ya, 3)}`, `fits at ${mult(P10(ek))} the largest run (${KIND[ax.kind](xk)}). <span class="scaling-warn">Extrapolation, not a result.</span>`);
          c1E.el.classList.remove('off');
        } else { c1E.set('off', 'Use “Extend the fits” to read the lines beyond the data.'); c1E.el.classList.add('off'); }
        const ce = crossEk(); const ciB = crossInfo(R);
        bCross.disabled = !(isFinite(ciB.beyond) && ciB.beyond > 0 && ciB.beyond <= 3);
        bCross.title = bCross.disabled ? (ciB.inRange ? 'The fits already cross inside the measured range' : 'The crossing is beyond ×1000 or not meaningful') : `Extend to ${mult(P10(ce))}`;
      }

      function draw1() {
        const { ctx, w: W, h: H } = cv1; const R = R1, ax = R.ax;
        if (!hatch1) hatch1 = hatch(ctx);
        ctx.save(); ctx.fillStyle = C.bg; ctx.fillRect(0, 0, W, H);
        const warp = anim1.get('warp'), ek = anim1.get('ek'), fitP = anim1.get('fit'), triP = anim1.get('tri'), rateP = anim1.get('rate');
        const k = P10(ek), xRight = R.xmax * k, ext = ek > 0.004;
        const yv = R.A.points.concat(R.B.points).map(q => q[1]);
        if (ext && R.fA && R.fB) { yv.push(fitY(R.fA, xRight), fitY(R.fB, xRight)); }
        const xr = range([R.xmin, xRight], 0.05), yr = range(yv, 0.08);
        const xl = cleanLabel(ax.p.x.label) + (warp > 0.5 ? ' · log scale' : ''), yl = cleanLabel(ax.p.y.label) + (warp > 0.5 ? ' · log scale' : '');
        const M = frame(ctx, W, H, { xr, yr, tx: warp, ty: warp, xfmt: KIND[ax.kind], yfmt: KIND.plain, xlabel: xl, ylabel: yl, top: 14, right: 16 });
        M1 = M; const r = M.rect;
        ctx.save(); ctx.beginPath(); ctx.rect(r.x, r.y - 2, r.w + 2, r.h + 4); ctx.clip();
        // extrapolation zone
        if (ext) {
          const zx = M.X(R.xmax), a = clamp(ek * 6, 0, 1);
          ctx.globalAlpha = a; ctx.fillStyle = hatch1; ctx.fillRect(zx, r.y, r.x + r.w - zx, r.h);
          ctx.strokeStyle = lib.rgba(C.ebt, 0.5); ctx.setLineDash([4, 4]); ctx.beginPath(); ctx.moveTo(zx, r.y); ctx.lineTo(zx, r.y + r.h); ctx.stroke(); ctx.setLineDash([]);
          ctx.globalAlpha = 1;
          if (r.x + r.w - zx > 70) { label(ctx, 'EXTRAPOLATION', zx + 8, r.y + 12, { mono: true, size: 11.5, color: C.ebt, alpha: a }); label(ctx, 'not measured', zx + 8, r.y + 28, { size: 12.5, color: C.muted, alpha: a }); }
        }
        const sers = [[R.A, R.fA, colA(ax)], [R.B, R.fB, C.ebt]];
        // paper's drawn display curve
        if (S1.curve) sers.forEach(([s, , col]) => { if (s.paper_curve && s.paper_curve.points) polyline(ctx, s.paper_curve.points.map(q => [M.X(q[0]), M.Y(q[1])]), col, { w: 1.5, dash: [1.5, 4], alpha: 0.9 }); });
        // fits
        if (fitP > 0.002) sers.forEach(([s, f, col], si) => {
          if (!f) return;
          const act = s.points.filter((_, i) => !S1.drop.has(i)).map(q => q[0]);
          const xa = Math.min(...act), xb = Math.max(...act), lb = L10(xa) + (L10(xb) - L10(xa)) * clamp(fitP, 0, 1);
          const pts = []; for (let i = 0; i <= 80; i++) { const x = P10(L10(xa) + (lb - L10(xa)) * i / 80); pts.push([M.X(x), M.Y(fitY(f, x))]); }
          polyline(ctx, pts, col, { w: 2.2, alpha: 0.95 });
          if (ext && fitP > 0.98) {
            const pe = []; for (let i = 0; i <= 80; i++) { const x = P10(L10(xb) + (L10(xRight) - L10(xb)) * i / 80); pe.push([M.X(x), M.Y(fitY(f, x))]); }
            polyline(ctx, pe, col, { w: 2, dash: [7, 6], alpha: clamp(ek * 6, 0, 1) });
          }
        });
        // slope triangles
        if (triP > 0.01 && R.fA && R.fB) {
          const a = triP * clamp((warp - 0.5) * 3, 0, 1);
          if (a > 0.01) {
            const lx0 = L10(R.xmin), span = L10(R.xmax) - lx0, run = Math.min(1, span * 0.24);
            // one step of each fit: run (×10 or less) and rise (×10^(m·run)); drawn on the side away from the other line
            const tri = (f, other, pos, col) => {
              const la = lx0 + span * pos - run / 2, xa = P10(la), xb = P10(la + run), ya = fitY(f, xa), yb = fitY(f, xb);
              const below = fitY(f, P10(la + run / 2)) <= fitY(other, P10(la + run / 2));
              const A0 = [M.X(xa), M.Y(ya)], A2 = [M.X(xb), M.Y(yb)], A1 = below ? [M.X(xa), M.Y(yb)] : [M.X(xb), M.Y(ya)];
              ctx.save(); ctx.globalAlpha = a; ctx.fillStyle = lib.rgba(col, 0.12); ctx.beginPath(); ctx.moveTo(...A0); ctx.lineTo(...A1); ctx.lineTo(...A2); ctx.closePath(); ctx.fill(); ctx.restore();
              polyline(ctx, [A0, A1, A2], col, { w: 1.4, alpha: a });
              const runLab = run >= 0.999 ? '×10' : '×' + P10(run).toFixed(1), riseLab = '×' + P10(f.m * run).toFixed(3);
              if (below) {
                label(ctx, runLab, (A1[0] + A2[0]) / 2, A1[1] + 11, { mono: true, size: 12, color: col, alpha: a, align: 'center' });
                ctx.font = monoF(12); const rw = ctx.measureText(riseLab).width, outside = A1[0] - 6 - rw > M.rect.x + 2;
                label(ctx, riseLab, outside ? A1[0] - 6 : A1[0] + 5, outside ? (A0[1] + A1[1]) / 2 : A1[1] - 9, { mono: true, size: 12, color: col, alpha: a, align: outside ? 'right' : 'left' });
              } else {
                label(ctx, runLab, (A0[0] + A1[0]) / 2, A1[1] - 10, { mono: true, size: 12, color: col, alpha: a, align: 'center' });
                label(ctx, riseLab, A1[0] + 6, (A1[1] + A2[1]) / 2, { mono: true, size: 12, color: col, alpha: a, align: 'left' });
              }
            };
            tri(R.fA, R.fB, 0.2, colA(ax)); tri(R.fB, R.fA, 0.66, C.ebt);
          }
        }
        // points
        sers.forEach(([s, , col], si) => s.points.forEach((q, i) => {
          const x = M.X(q[0]), y = M.Y(q[1]), off = S1.drop.has(i), hv = S1.hover && S1.hover.i === i;
          if (off) { dot(ctx, x, y, 4.5, null, { stroke: col, alpha: 0.6, lw: 1.5 }); }
          else dot(ctx, x, y, hv ? 6 : 4.6, col, { stroke: C.bg, lw: 1.5 });
          if (hv && S1.hover.s === si) dot(ctx, x, y, 9, null, { stroke: C.ink, lw: 1.5 });
        }));
        // crossing marker
        const ci = crossInfo(R);
        if (fitP > 0.9 && ci.xs && !ci.flat && ci.xs >= R.xmin && ci.xs <= xRight * 1.0001) {
          const cx = M.X(ci.xs), cy = M.Y(fitY(R.fA, ci.xs));
          if (cx >= r.x && cx <= r.x + r.w) {
            dot(ctx, cx, cy, 8, null, { stroke: C.ink, lw: 2 }); dot(ctx, cx, cy, 2.5, C.ink);
            // lines fall to the right, so the space below-left of the crossing is empty
            const room = cx - r.x > 110;
            const lx = room ? cx - 10 : cx + 12, al = room ? 'right' : 'left', ly = room ? cy + 16 : cy + 20;
            label(ctx, 'fits cross', lx, ly, { size: 13, weight: 700, align: al });
            label(ctx, '≈' + KIND[ax.kind](ci.xs), lx, ly + 16, { mono: true, size: 12, color: C.muted, align: al });
          }
        }
        ctx.restore();
        // rate box
        if (rateP > 0.01 && R.fA && R.fB) {
          const [na, nb] = shortNames(ax), narrow = W < 520;
          const lines = [];
          if (!narrow) lines.push({ t: 'scaling-rate gain', color: C.muted, size: 12.5 });
          lines.push({ t: `|${neg(R.fB.m.toFixed(3))}| ÷ |${neg(R.fA.m.toFixed(3))}| − 1`, mono: true, size: narrow ? 11.5 : 12.5, color: C.ink });
          lines.push({ t: `= ${pct(R.gain, 1)}`, mono: true, size: narrow ? 15 : 18, weight: 600, color: C.ebt });
          lines.push({ t: `printed: ${(ax.p.rate_gain_printed * 100).toFixed(2)}%`, size: 12, color: C.muted });
          const zoneW = ext ? r.x + r.w - M.X(R.xmax) : 0;
          boxText(ctx, lines, r.x + r.w - 6, r.y + 6 + (zoneW > 70 ? 38 : 0), { alignRight: true, alpha: rateP });
        }
        // hover tooltip
        if (S1.hover) {
          const { s, i } = S1.hover, ser = s ? R.B : R.A, q = ser.points[i], x = M.X(q[0]), y = M.Y(q[1]);
          const nm = names(ax)[s];
          boxText(ctx, [
            { t: `${nm} · ${KIND[ax.kind](q[0])}`, size: 13, weight: 700 },
            { t: `${metricShort(ax)} ${num(q[1], 4)}  (approx.)`, mono: true, size: 12 },
            { t: S1.drop.has(i) ? 'click: put this size back' : 'click: drop this size', size: 12, color: C.muted },
          ], x + 12, y + 10, { clampW: W, clampH: H });
        }
        ctx.restore();
      }
      // pointer: hover + click to drop a size from the fit
      function nearest1(ev) {
        if (!M1) return null; const [px, py] = cv1.toLocal(ev); let best = null, bd = 16;
        [R1.A, R1.B].forEach((s, si) => s.points.forEach((q, i) => { const d = Math.hypot(M1.X(q[0]) - px, M1.Y(q[1]) - py); if (d < bd) { bd = d; best = { s: si, i }; } }));
        return best;
      }
      cv1.canvas.addEventListener('pointermove', (ev) => { const n = nearest1(ev); const same = (n && S1.hover && n.i === S1.hover.i && n.s === S1.hover.s) || (!n && !S1.hover); if (!same) { S1.hover = n; cv1.canvas.style.cursor = n ? 'pointer' : 'default'; draw1(); } });
      cv1.canvas.addEventListener('pointerleave', () => { if (S1.hover) { S1.hover = null; draw1(); } });
      cv1.canvas.addEventListener('click', (ev) => { const n = nearest1(ev); if (n) { toggleDrop(n.i); S1.hover = n; draw1(); } });

      // ================================================================
      // LAB 2: what a steeper slope buys
      // ================================================================
      el.appendChild(h('h3', { class: 'scaling-h' }, h('span', { class: 'n' }, '2'), 'What a steeper slope buys you'));
      el.appendChild(h('p', { class: 'scaling-sub', html: 'A slope gain only pays off after the lines cross. How far away that is depends on two numbers: how far behind the steeper model starts, and how much steeper it is. On log–log axes the gap shrinks by the same amount every decade, so the catch-up distance is a single division.' }));
      const bench2 = h('div', { class: 'bench scaling-bench' }); el.appendChild(bench2);
      const ctl2 = h('div', { class: 'card stack scaling-ctl' }); bench2.appendChild(ctl2);
      const ins2 = h('div', { class: 'card stack scaling-ins' }); bench2.appendChild(ins2);
      const S2 = { m: 0.2685, g: 0.0291, d: 0.052, preset: 'params' };
      ctl2.appendChild(h('h4', {}, 'Load the numbers from a figure'));
      const pre2 = h('div', { class: 'scaling-presets' }); ctl2.appendChild(pre2);
      const PRE = ['data', 'depth', 'params', 'flops', 'vwidth', 'vparams', 's2'].filter(k => axByKey[k]);
      const preBtns = {};
      PRE.forEach(k => { const ax = axByKey[k]; const b = h('button', { type: 'button', class: 'scaling-chip', 'aria-pressed': String(k === S2.preset) }, `${ax.key.startsWith('v') ? 'Video ' + ax.name.toLowerCase() : ax.name} · ${ax.p.figure.replace('Fig ', '')}`); b.addEventListener('click', () => loadPreset(k)); preBtns[k] = b; pre2.appendChild(b); });
      ctl2.appendChild(h('h4', { style: 'margin-top:6px' }, 'Or set them yourself'));
      const mS = lib.slider({ id: 'scaling-m', label: 'Baseline exponent m', min: 0.02, max: 0.5, step: 0.001, value: S2.m, fmt: v => 'm = ' + neg((-v).toFixed(3)), oninput: v => { S2.m = v; clearPreset(); upd2(); } });
      const gS = lib.slider({ id: 'scaling-g', label: 'Steeper model: rate gain', min: 0, max: 0.6, step: 0.001, value: S2.g, fmt: v => '+' + (v * 100).toFixed(1) + '%', oninput: v => { S2.g = v; clearPreset(); upd2(); } });
      const dS = lib.slider({ id: 'scaling-d', label: 'Its gap at the largest run', min: -0.2, max: 0.6, step: 0.001, value: S2.d, fmt: v => (v >= 0 ? '+' : MINUS) + Math.abs(v * 100).toFixed(1) + (v >= 0 ? '% worse' : '% better'), oninput: v => { S2.d = v; clearPreset(); upd2(); } });
      [mS, gS, dS].forEach(s => ctl2.appendChild(s.el));
      ctl2.appendChild(h('div', { class: 'scaling-formula', html: 'decades to catch up<br><span>= log₁₀(1 + gap) ÷ (|m| × gain)</span>' }));
      const head2 = h('div', { class: 'scaling-head' }, h('div', {}, h('h3', { class: 'scaling-ptitle' }, 'Two power laws and the gap between them'), h('div', { class: 'scaling-psub' }, 'Top: the two power laws, which look almost identical on log–log axes. Bottom: the gap between them, which shrinks by a constant amount per decade.')), h('span', {}, lib.badge('ext', 'arithmetic on the paper’s fits')));
      ins2.appendChild(head2);
      const cv2 = rcanvas(ins2, { aspect: (w) => w < 560 ? 1.55 : 0.62, minH: 400, maxH: 540, label: 'Top: two power laws on log-log axes, a baseline and a steeper model that starts behind. Bottom: the percentage gap between them, which shrinks steadily and crosses zero at the catch-up point.', draw: () => draw2() });
      const ro2 = h('div', { class: 'scaling-ro' }); ins2.appendChild(ro2);
      const c2P = roCell('Per 10× more resource'), c2C = roCell('Catch-up distance'), c2T = roCell('At ×1000 (the paper’s scenario)');
      [c2P, c2C, c2T].forEach(c => ro2.appendChild(c.el));
      const presetNote = h('p', { class: 'scaling-hint' }); ins2.appendChild(presetNote);
      function clearPreset() { S2.preset = null; Object.values(preBtns).forEach(b => b.setAttribute('aria-pressed', 'false')); presetNote.textContent = 'Custom numbers.'; }
      function loadPreset(k) {
        const R = analyze(axByKey[k], new Set());
        S2.m = Math.abs(R.fA.m); S2.g = clamp(R.gain, 0, 0.6); S2.d = clamp(R.ybF / R.yaF - 1, -0.2, 0.6); S2.preset = k;
        mS.set(S2.m); gS.set(S2.g); dS.set(S2.d);
        Object.entries(preBtns).forEach(([kk, b]) => b.setAttribute('aria-pressed', String(kk === k)));
        const ax = axByKey[k];
        presetNote.innerHTML = `${ax.p.figure}: m and gain from the refit; gap = ${shortNames(ax)[1]} fit ÷ ${shortNames(ax)[0]} fit − 1 at ${KIND[ax.kind](R.xc)}, ${pct(R.ybF / R.yaF - 1, 1)} (approx.).`;
        upd2();
      }
      function calc2() {
        const m = S2.m, g = S2.g, d = S2.d, gap = L10(1 + d), sd = m * g;
        const Ls = d <= 0 ? 0 : sd > 0 ? gap / sd : Infinity;
        return { m, g, d, gap, sd, Ls };
      }
      function upd2() {
        const c = calc2();
        c2P.set(`×${num(P10(-c.m), 3)} vs ×${num(P10(-c.m * (1 + c.g)), 3)}`, `baseline vs steeper model: the gap closes by ${num(c.sd, 3)} decades of loss per decade of resource`);
        if (c.d <= 0) c2C.set('already ahead', 'The steeper model is not behind at the largest run.');
        else if (!isFinite(c.Ls)) c2C.set('never', 'Equal slopes: the gap never closes.');
        else c2C.set(`${num(c.gap, 3)} ÷ ${num(c.sd, 3)} = ${num(c.Ls, 3)} decades`, `the lines cross at ${mult(P10(c.Ls))} the largest run`);
        const r3 = (1 + c.d) * P10(-c.m * c.g * 3) - 1;
        c2T.set(r3 <= 0 ? `${Math.abs(r3 * 100).toFixed(1)}% better` : `${(r3 * 100).toFixed(1)}% worse`, 'the steeper model at 1000× the largest run, if both lines held');
        draw2();
      }
      let hatch2 = null;
      function draw2() {
        const { ctx, w: W, h: H } = cv2; if (!hatch2) hatch2 = hatch(ctx);
        ctx.save(); ctx.fillStyle = C.bg; ctx.fillRect(0, 0, W, H);
        const c = calc2();
        const Dmax = isFinite(c.Ls) && c.d > 0 ? clamp(Math.ceil(c.Ls * 1.25 + 0.6), 3, 12) : 4;
        const yT = (L) => P10(-c.m * L), yE = (L) => (1 + c.d) * P10(-c.m * (1 + c.g) * L);
        const gp = (L) => 100 * (yE(L) / yT(L) - 1);
        const xr = { lin: [0, 1], log: [-1, Dmax] }, xf = (v) => v < 1 ? '×' + num(v, 2) : mult(v);
        const split = Math.round(H * 0.5), left = W < 520 ? 58 : 66;
        const Mt = frame(ctx, W, H, { xr, yr: range([yT(-1), yE(-1), yT(Dmax), yE(Dmax)], 0.06, 0.08), tx: 1, ty: 1, xfmt: xf, yfmt: (v) => num(v, 3), xlabel: '', ylabel: 'loss, relative', y0: 0, y1: split, noX: true, left, top: 12, right: 18 });
        const gv = [gp(-1), gp(Dmax), 0];
        const Mb = frame(ctx, W, H, { xr, yr: range(gv, 0.14), tx: 1, ty: 0, xfmt: xf, yfmt: (v) => (v > 0 ? '+' : '') + num(v, 3) + '%', xlabel: 'resource ÷ largest measured run (log)', ylabel: 'gap', y0: split, y1: H, left, top: 10, right: 18 });
        const shade = (M) => {
          const r = M.rect, z = M.X(1);
          ctx.fillStyle = hatch2; ctx.fillRect(z, r.y, r.x + r.w - z, r.h);
          ctx.fillStyle = lib.rgba(C.truth, 0.05); ctx.fillRect(r.x, r.y, z - r.x, r.h);
        };
        const curve = (M, fn, col) => { const pts = [], pe = []; for (let i = 0; i <= 140; i++) { const L = -1 + (Dmax + 1) * i / 140; const pt = [M.X(P10(L)), M.Y(fn(L))]; if (L <= 0) pts.push(pt); else { if (!pe.length && pts.length) pe.push(pts[pts.length - 1]); pe.push(pt); } } polyline(ctx, pts, col, { w: 2.4 }); polyline(ctx, pe, col, { w: 2.2, dash: [7, 6] }); };
        // top: the two power laws
        ctx.save(); ctx.beginPath(); ctx.rect(Mt.rect.x, Mt.rect.y - 2, Mt.rect.w + 2, Mt.rect.h + 4); ctx.clip();
        shade(Mt);
        if (Mt.rect.x + Mt.rect.w - Mt.X(1) > 100) label(ctx, 'EXTRAPOLATION', Mt.rect.x + Mt.rect.w - 6, Mt.rect.y + 11, { mono: true, size: 11.5, color: C.ebt, align: 'right' });
        curve(Mt, yT, C.ar); curve(Mt, yE, C.ebt);
        // legend in the empty lower-left corner (both lines fall to the right)
        label(ctx, '— baseline', Mt.rect.x + 8, Mt.rect.y + Mt.rect.h - 30, { size: 12.5, color: C.ar });
        label(ctx, '— steeper model', Mt.rect.x + 8, Mt.rect.y + Mt.rect.h - 12, { size: 12.5, color: C.ebt });
        ctx.restore();
        // bottom: the gap between them, which closes at a constant rate per decade
        const rb = Mb.rect;
        ctx.save(); ctx.beginPath(); ctx.rect(rb.x, rb.y - 2, rb.w + 2, rb.h + 4); ctx.clip();
        shade(Mb);
        const y0 = Mb.Y(0);
        ctx.fillStyle = lib.rgba(C.truth, 0.07); ctx.fillRect(rb.x, y0, rb.w, rb.y + rb.h - y0);
        polyline(ctx, [[rb.x, y0], [rb.x + rb.w, y0]], C.muted, { w: 1.2 });
        // the gap line starts above zero on the left, so the area just above and below zero on the left is free
        label(ctx, '↑ behind', rb.x + 6, Math.max(rb.y + 10, y0 - 11), { size: 12, color: C.muted });
        if (rb.y + rb.h - y0 > 24) label(ctx, '↓ ahead', rb.x + 6, Math.min(rb.y + rb.h - 10, y0 + 12), { size: 12, color: C.truth });
        curve(Mb, gp, C.ebt);
        const x1 = Mb.X(1), g0 = Mb.Y(gp(0));
        dot(ctx, x1, g0, 5, C.ebt, { stroke: C.bg });
        label(ctx, `today ${c.d >= 0 ? '+' : MINUS}${Math.abs(c.d * 100).toFixed(1)}%`, x1 + 9, g0 - 12, { mono: true, size: 12, color: C.ink });
        if (c.d > 0 && isFinite(c.Ls) && c.Ls <= Dmax) {
          const cx = Mb.X(P10(c.Ls));
          ctx.save(); ctx.strokeStyle = lib.rgba(C.ink, 0.45); ctx.setLineDash([3, 4]); ctx.beginPath(); ctx.moveTo(cx, 0); ctx.lineTo(cx, rb.y + rb.h); ctx.stroke(); ctx.restore();
          dot(ctx, cx, y0, 8, null, { stroke: C.ink, lw: 2 }); dot(ctx, cx, y0, 2.5, C.ink);
          // after the crossing the line runs below zero, so put the label above the zero line, right of the crossing
          ctx.font = bodyF(13, 700);
          const opts = ['caught up at ' + mult(P10(c.Ls)), mult(P10(c.Ls))], t = opts.find(o => cx + 12 + ctx.measureText(o).width < rb.x + rb.w - 2);
          if (t) label(ctx, t, cx + 12, y0 - 14, { size: 13, weight: 700 });
          else label(ctx, opts[1], cx - 10, y0 - 14, { size: 13, weight: 700, align: 'right' });
        } else if (c.d > 0) label(ctx, isFinite(c.Ls) ? `catches up at ${mult(P10(c.Ls))} →` : 'never catches up', rb.x + rb.w - 6, y0 - 30, { size: 12.5, weight: 700, align: 'right' });
        ctx.restore(); ctx.restore();
      }

      // ================================================================
      // LAB 3: the compute caveat
      // ================================================================
      el.appendChild(h('h3', { class: 'scaling-h' }, h('span', { class: 'n' }, '3'), 'The compute caveat: an EBT training step costs more'));
      el.appendChild(h('p', { class: 'scaling-sub', html: 'Scaling faster with FLOPs is not the same as being cheaper per FLOP. Here is the paper’s own accounting (Appendix D.5), one line at a time. N is the number of non-embedding parameters; costs are per training token.' }));
      const card3 = h('div', { class: 'card stack' }); el.appendChild(card3);
      card3.appendChild(h('div', { class: 'scaling-head' }, h('div', {}, h('h3', { class: 'scaling-ptitle' }, 'Training FLOPs per token'), h('div', { class: 'scaling-psub' }, 'Appendix D.5 (p.35 to 36) and FAQ H.3 (p.41)')), h('span', {}, lib.badge('paper', 'p.35–36, p.41'))));
      const FL = [
        { lab: 'Transformer++ step', segs: [['F 2N', 2, 'ar'], ['B 4N', 4, 'ar2']], tot: '6N', x: '1×', note: 'a forward pass (2N) and a backward pass (4N): the standard 6N estimate (p.35)' },
        { lab: 'EBT, one optimization step', segs: [['F 2N', 2, 'e1'], ['B 4N', 4, 'e2'], ['B 4N', 4, 'e3']], tot: '10N', x: '≈1.66×', note: '“FLOPs = F + B + B”: a forward pass, a backward pass to get ∇ŷE, and a Hessian-vector product for training through the update, which costs about one more backward pass (p.35). With a single step: “about a constant 1.66×” (p.41)' },
        { lab: 'Autoregressive EBT: the sequence doubles', segs: [['10N', 10, 'e1'], ['+10N', 10, 'e2']], tot: '20N', x: '≈3.33×', note: 'predictions are fed in next to the context, effective length 2S − 2, which costs about 2× (p.35 to 36)' },
        { lab: 'Two optimization steps (the S1 scaling runs)', segs: [['step 1 · 20N', 20, 'e1'], ['step 2 · 20N', 20, 'e2']], tot: '40N', x: '6.66×', note: '“EBTs used 6.66× the FLOPs of a comparable Transformer++ during training” (p.36)' },
      ];
      const bars = h('div', { class: 'scaling-flops' }); card3.appendChild(bars);
      const rowsEl = FL.map((f) => {
        const segs = f.segs.map(([t, v, cl]) => h('span', { class: 'seg ' + cl + (v <= 4 ? ' sm' : ''), style: `flex:${v} 1 0px`, title: t }, h('span', {}, t)));
        const row = h('div', { class: 'frow' },
          h('div', { class: 'fl' }, h('b', {}, f.lab), h('span', { class: 'fx' }, `${f.tot} · ${f.x}`)),
          h('div', { class: 'ftrack' }, h('div', { class: 'fbar', style: `width:${(f.segs.reduce((a, s) => a + s[1], 0) / 40 * 100).toFixed(2)}%` }, segs)),
          h('div', { class: 'fn' }, f.note));
        bars.appendChild(row); return row;
      });
      const S3 = { shown: FL.length, timer: null };
      const b3Step = btn('Step ▸', { primary: true }), b3Play = btn('Play'), b3Reset = btn('Reset');
      const cap3 = h('p', { class: 'scaling-hint', 'aria-live': 'polite' });
      card3.appendChild(h('div', { class: 'row' }, b3Step, b3Play, b3Reset, cap3));
      function show3(n) { S3.shown = n; rowsEl.forEach((r, i) => r.classList.toggle('hide', i >= n)); cap3.textContent = n >= FL.length ? 'Total: about 6.66× a Transformer++ step for the 2-step runs behind Figs 4 and 5.' : n === 0 ? 'Press Step to build the cost.' : `${n} of ${FL.length}: ${FL[n - 1].lab}.`; b3Step.textContent = n >= FL.length ? 'Start over' : 'Step ▸'; }
      b3Step.addEventListener('click', () => { clearInterval(S3.timer); S3.timer = null; show3(S3.shown >= FL.length ? 0 : S3.shown + 1); });
      b3Reset.addEventListener('click', () => { clearInterval(S3.timer); S3.timer = null; show3(0); });
      b3Play.addEventListener('click', () => { clearInterval(S3.timer); show3(0); S3.timer = setInterval(() => { if (S3.shown >= FL.length) { clearInterval(S3.timer); S3.timer = null; return; } show3(S3.shown + 1); }, 1100); });
      show3(FL.length);
      card3.appendChild(h('p', { class: 'scaling-faint', html: 'The paper calls these estimates approximate (“We welcome corrections”, p.36). The thinking-tuned S2 models vary: truncated backprop saves FLOPs, extra steps and the replay buffer add them (p.36).' }));

      // what-if on Fig 5b
      el.appendChild(h('p', { class: 'scaling-sub', html: 'How much of the FLOPs gap is that 6.66×? Slide the cost per token and watch the EBT points of Fig 5b move. This is a what-if: it keeps each run’s perplexity and only changes its price.' }));
      const bench3 = h('div', { class: 'bench scaling-bench' }); el.appendChild(bench3);
      const ctl3 = h('div', { class: 'card stack scaling-ctl' }); bench3.appendChild(ctl3);
      const ins3 = h('div', { class: 'card stack scaling-ins' }); bench3.appendChild(ins3);
      const S3w = { c: 6.66 };
      ctl3.appendChild(h('h4', {}, 'EBT cost per training token'));
      const costS = lib.slider({ id: 'scaling-cost', label: 'relative to Transformer++', min: 0, max: 1, step: 0.001, value: L10(6.66), fmt: v => '×' + P10(v).toFixed(2), oninput: v => { S3w.c = P10(v); upd3(); } });
      ctl3.appendChild(costS.el);
      const cp = h('div', { class: 'scaling-presets' }); ctl3.appendChild(cp);
      [[1, '1× (free)'], [1.66, '1.66× (one step)'], [3.33, '3.33× (one AR step)'], [6.66, '6.66× (measured)']].forEach(([v, t]) => { const b = h('button', { type: 'button', class: 'scaling-chip' }, t); b.addEventListener('click', () => { S3w.c = v; costS.set(L10(v)); upd3(); }); cp.appendChild(b); });
      ctl3.appendChild(h('p', { class: 'scaling-hint', html: 'Even at 1×, EBT would still be behind: at equal size it has higher perplexity (Fig 5a). The steeper slope has to pay back both gaps.' }));
      ins3.appendChild(h('div', { class: 'scaling-head' }, h('div', {}, h('h3', { class: 'scaling-ptitle' }, 'Fig 5b, with a different price per EBT step'), h('div', { class: 'scaling-psub' }, 'validation perplexity vs training FLOPs, log–log')), h('div', { class: 'row', style: 'gap:6px' }, lib.badge('paper', 'approx., read from Fig 5b'), lib.badge('ext', 'what-if'))));
      const cv3 = rcanvas(ins3, { aspect: (w) => w < 560 ? 0.9 : 0.52, minH: 300, maxH: 420, label: 'Fig 5b points with EBT points shifted by a chosen cost multiplier, and the power-law fits extended to 10^25 FLOPs.', draw: () => draw3() });
      const ro3 = h('div', { class: 'scaling-ro' }); ins3.appendChild(ro3);
      const c3X = roCell('Where the fits cross'), c3E = roCell('At 10²¹ FLOPs (the paper’s ceiling)');
      [c3X, c3E].forEach(c => ro3.appendChild(c.el));
      function calc3() {
        const p = P.fig5b, A = p.series[0].points, k = S3w.c / 6.66;
        const B = p.series[1].points.map(q => [q[0] * k, q[1]]);
        const fA = ols(A), fB = ols(B); return { A, B, fA, fB, Lc: crossLog(fA, fB), xmax: Math.max(...A.concat(B).map(q => q[0])) };
      }
      function upd3() {
        const c = calc3();
        const xs = P10(c.Lc);
        c3X.set('≈' + sci(xs, 1) + ' FLOPs', `${mult(xs / c.xmax)} beyond the largest run. ${c.Lc > 21 ? 'Above the 10²¹ FLOPs the paper could train (p.27).' : ''}`);
        const ya = fitY(c.fA, 1e21), yb = fitY(c.fB, 1e21);
        c3E.set(`EBT ${num(yb, 3)} · T++ ${num(ya, 3)}`, `fits extended to 10²¹ FLOPs: EBT ${yb > ya ? 'higher' : 'lower'} by ${Math.abs((yb / ya - 1) * 100).toFixed(1)}%. <span class="scaling-warn">Extrapolation.</span>`);
        draw3();
      }
      let hatch3 = null;
      function draw3() {
        const { ctx, w: W, h: H } = cv3; if (!hatch3) hatch3 = hatch(ctx);
        ctx.save(); ctx.fillStyle = C.bg; ctx.fillRect(0, 0, W, H);
        const c = calc3(), X1 = 1e25;
        const yv = c.A.concat(c.B).map(q => q[1]).concat([fitY(c.fA, X1), fitY(c.fB, X1)]);
        const xr = range([1e16, X1], 0.02), yr = range(yv, 0.06);
        const M = frame(ctx, W, H, { xr, yr, tx: 1, ty: 1, xfmt: KIND.flops, yfmt: KIND.plain, xlabel: 'training FLOPs · log scale', ylabel: 'validation perplexity · log', top: 12, right: 16 });
        const r = M.rect;
        ctx.save(); ctx.beginPath(); ctx.rect(r.x, r.y - 2, r.w + 2, r.h + 4); ctx.clip();
        const zx = M.X(c.xmax); ctx.fillStyle = hatch3; ctx.fillRect(zx, r.y, r.x + r.w - zx, r.h);
        if (r.x + r.w - zx > 90) label(ctx, 'EXTRAPOLATION', zx + 8, r.y + 12, { mono: true, size: 11.5, color: C.ebt });
        const cx = M.X(1e21); ctx.save(); ctx.strokeStyle = lib.rgba(C.bad, 0.7); ctx.setLineDash([5, 4]); ctx.beginPath(); ctx.moveTo(cx, r.y); ctx.lineTo(cx, r.y + r.h); ctx.stroke(); ctx.restore();
        { const t = W < 520 ? '10²¹ ceiling' : '10²¹: the paper’s compute ceiling'; ctx.font = bodyF(12); const tw = ctx.measureText(t).width; const fitsRight = cx + 6 + tw < r.x + r.w - 2; label(ctx, t, fitsRight ? cx + 6 : cx - 6, r.y + r.h - 14, { size: 12, color: C.bad, align: fitsRight ? 'left' : 'right' }); }
        [[c.A, c.fA, C.ar], [c.B, c.fB, C.ebt]].forEach(([pts, f, col]) => {
          const xa = Math.min(...pts.map(q => q[0])), xb = Math.max(...pts.map(q => q[0]));
          const s = [], e = []; for (let i = 0; i <= 60; i++) { const x = P10(L10(xa) + (L10(xb) - L10(xa)) * i / 60); s.push([M.X(x), M.Y(fitY(f, x))]); const x2 = P10(L10(xb) + (L10(X1) - L10(xb)) * i / 60); e.push([M.X(x2), M.Y(fitY(f, x2))]); }
          polyline(ctx, s, col, { w: 2.2 }); polyline(ctx, e, col, { w: 2, dash: [7, 6] });
        });
        // ghost of the measured EBT points when shifted
        if (Math.abs(S3w.c - 6.66) > 0.05) P.fig5b.series[1].points.forEach((q, i) => { const x0 = M.X(q[0]), x1 = M.X(c.B[i][0]), y = M.Y(q[1]); dot(ctx, x0, y, 4.5, null, { stroke: C.ebt, alpha: 0.45 }); polyline(ctx, [[x0, y], [x1, y]], C.ebt, { w: 1, alpha: 0.4, dash: [2, 3] }); });
        c.A.forEach(q => dot(ctx, M.X(q[0]), M.Y(q[1]), 4.6, C.ar, { stroke: C.bg }));
        c.B.forEach(q => dot(ctx, M.X(q[0]), M.Y(q[1]), 4.6, C.ebt, { stroke: C.bg }));
        if (c.Lc <= L10(X1)) { const xx = M.X(P10(c.Lc)), yy = M.Y(fitY(c.fA, P10(c.Lc))); dot(ctx, xx, yy, 8, null, { stroke: C.ink, lw: 2 }); label(ctx, 'fits cross', xx - 12, yy - 16, { size: 13, weight: 700, align: 'right' }); }
        else label(ctx, `fits meet at ≈${sci(P10(c.Lc), 0)} →`, r.x + r.w - 8, r.y + 34, { size: 12.5, align: 'right', color: C.ink });
        ctx.restore();
        const lg = W < 520 ? 0 : 1;
        if (lg) { label(ctx, '● Transformer++', r.x + 10, r.y + r.h - 34, { size: 12.5, color: C.ar }); label(ctx, '● EBT at ×' + S3w.c.toFixed(2) + ' cost', r.x + 10, r.y + r.h - 16, { size: 12.5, color: C.ebt }); }
        ctx.restore();
      }

      // ================================================================
      // LAB 4: thinking scalability
      // ================================================================
      el.appendChild(h('h3', { class: 'scaling-h' }, h('span', { class: 'n' }, '4'), 'Thinking scalability: what extra inference compute buys'));
      el.appendChild(h('p', { class: 'scaling-sub', html: 'The second kind of scaling. Fix a trained model and spend more forward passes on each prediction: more optimization steps (thinking longer) and several candidates ranked by their own energy (self-verification, best-of-N). These runs use small thinking-tuned (S2) models.' }));
      const bench4 = h('div', { class: 'bench scaling-bench' }); el.appendChild(bench4);
      const ctl4 = h('div', { class: 'card stack scaling-ctl' }); bench4.appendChild(ctl4);
      const ins4 = h('div', { class: 'card stack scaling-ins' }); bench4.appendChild(ins4);
      const ser = (id, i = 0) => (P[id] && P[id].series[i]) ? P[id].series[i].points : [];
      const f6b = linfit(ser('fig6b')), fB1a = linfit(ser('figB1a')), f7 = linfit(ser('fig7'));
      const TT = [];
      if (P.fig6a) TT.push({
        key: '6a', id: 'fig6a', tab: 'More passes', disc: 5,
        text: 'Lower is better: y is the paper’s “perplexity increase” on four out-of-distribution datasets (mean; the exact formula is not given). Transformer++ makes one forward pass per token and cannot use more, so its line is flat. EBT can: more optimization steps (thinking longer), then several candidates with the lowest-energy one kept (self-verification). Note that EBT without thinking starts worse than Transformer++.',
        frame: { xr: { lin: [0, 32], log: null }, yr: { lin: [30, 46.5], log: null }, tx: 0, ty: 0, xfmt: KIND.plain, yfmt: KIND.plain, xlabel: 'forward passes per prediction', ylabel: 'perplexity increase on OOD data' },
        clabel: (c) => `${ser('fig6a', 1)[Math.round(c)][0]} passes`,
        draw(ctx, M, c) {
          const T = ser('fig6a', 0), E = ser('fig6a', 1), labs = P.fig6a.series[1].point_labels || [];
          const yT = T[0][1]; polyline(ctx, [[M.X(0.6), M.Y(yT)], [M.X(31.4), M.Y(yT)]], C.ar, { w: 2.2, dash: [8, 5] });
          label(ctx, M.W < 520 ? 'Transformer++ (1 pass)' : 'Transformer++: one pass per token, cannot think longer', M.X(5), M.Y(yT) - 12, { size: 12.5, color: C.ar });
          const ci = Math.floor(c), fr = c - ci, pts = [];
          for (let i = 0; i <= Math.min(ci, E.length - 1); i++) pts.push([M.X(E[i][0]), M.Y(E[i][1])]);
          if (ci < E.length - 1 && fr > 0) pts.push([M.X(E[ci][0] + (E[ci + 1][0] - E[ci][0]) * fr), M.Y(E[ci][1] + (E[ci + 1][1] - E[ci][1]) * fr)]);
          polyline(ctx, pts, C.ebt, { w: 2.4 });
          E.forEach((q, i) => dot(ctx, M.X(q[0]), M.Y(q[1]), i === Math.round(c) ? 6.5 : 4.5, i <= c + 0.01 ? C.ebt : null, { stroke: i <= c + 0.01 ? C.bg : C.ebt, alpha: i <= c + 0.01 ? 1 : 0.5 }));
          const k = Math.round(c), q = E[k], x = M.X(q[0]), y = M.Y(q[1]);
          dot(ctx, x, y, 10, null, { stroke: C.ink, lw: 1.5 });
          const right = x < M.rect.x + M.rect.w * 0.55;
          label(ctx, (labs[k] || '').replace('EBT ', ''), x + (right ? 14 : -14), y + 16, { size: 13, weight: 700, align: right ? 'left' : 'right' });
          if (k > 0) {
            // bracket: drop from "no thinking" down to the current setting
            const y0 = M.Y(E[0][1]), bx = Math.min(x + 14, M.rect.x + M.rect.w - 3);
            polyline(ctx, [[M.X(E[0][0]) + 8, y0], [bx, y0]], C.truth, { w: 1.2, dash: [2, 3] });
            polyline(ctx, [[bx, y0], [bx, y - 9]], C.truth, { w: 1.6 });
            polyline(ctx, [[bx - 4, y - 14], [bx, y - 8], [bx + 4, y - 14]], C.truth, { w: 1.6 });
            const tl = '−' + ((1 - q[1] / E[0][1]) * 100).toFixed(1) + '%', toRight = bx + 70 < M.rect.x + M.rect.w;
            label(ctx, tl, toRight ? bx + 7 : bx - 7, y0 + (y - y0) * 0.28, { mono: true, size: 13, weight: 600, color: C.truth, align: toRight ? 'left' : 'right' });
          }
        },
        readout(c) {
          const E = ser('fig6a', 1), T = ser('fig6a', 0), k = Math.round(c), q = E[k], labs = P.fig6a.series[1].point_labels || [];
          return [['EBT at ' + q[0] + ' passes', num(q[1], 4), (labs[k] || '') + ' (approx.)'],
            ['vs EBT without thinking', k ? '−' + ((1 - q[1] / E[0][1]) * 100).toFixed(1) + '%' : '0%', 'the paper: “as much as 29%” (p.10)'],
            ['vs Transformer++ (' + num(T[0][1], 4) + ')', (q[1] < T[0][1] ? 'better by ' : 'worse by ') + Math.abs(q[1] - T[0][1]).toFixed(2), 'flat by construction: one pass per token']];
        },
      });
      if (P.fig6b) TT.push({
        key: '6b', id: 'fig6b', tab: 'Verification vs training', cont: [0.98, 32.65],
        text: 'Each dot is a checkpoint of one small EBT during training. y is the perplexity gain from keeping the lowest-energy of 5 candidates (BoN-5) instead of 1. The fitted line rises from about 8% to about 11%; the paper’s “4%−8%” to “10%−14%” (p.10) describes the spread of early and late points, which is wide. Measured on BigBench Dyck only; “We did not observe this trend in other benchmarks” (p.34).',
        frame: { xr: { lin: [-0.6, 34.2], log: null }, yr: { lin: [3.4, 14.8], log: null }, tx: 0, ty: 0, xfmt: (v) => num(v) + 'B', yfmt: (v) => num(v) + '%', xlabel: 'tokens trained on (billions)', ylabel: '% ppl gain over no verification' },
        clabel: (c) => num(c, 3) + 'B tokens',
        draw(ctx, M, c) { drawScatterFit(ctx, M, ser('fig6b'), f6b, c, C.ebt); },
        readout(c) { const y = f6b.b + f6b.m * c; return [['the fitted line at ' + num(c, 3) + 'B', num(y, 3) + '%', `+${num(f6b.m, 2)} points per billion tokens`], ['spread around the line', 'r² ' + f6b.r2.toFixed(2), 'a noisy trend; the paper draws the same line'], ['checkpoints so far', String(ser('fig6b').filter(q => q[0] <= c + 1e-9).length) + ' / ' + ser('fig6b').length, 'approx., read from Fig 6b']]; },
      });
      if (P.figB1a) TT.push({
        key: 'b1a', id: 'figB1a', tab: 'BoN-10 vs BoN-2', cont: [2.95, 32.65],
        text: 'Same training run, RedPajamaV2 validation (p.34). y compares keeping the best of 10 candidates with the best of 2. Early in training, more candidates sometimes hurt (below zero): the model found “an adversarial sample (a sample with low energy that is in fact not a good prediction)” (p.28). With more data the gain grows and those dips fade.',
        frame: { xr: { lin: [1.4, 34.2], log: null }, yr: { lin: [-0.5, 2.8], log: null }, tx: 0, ty: 0, xfmt: (v) => num(v) + 'B', yfmt: (v) => num(v) + '%', xlabel: 'tokens trained on (billions)', ylabel: '% ppl gain, BoN-10 over BoN-2' },
        clabel: (c) => num(c, 3) + 'B tokens',
        draw(ctx, M, c) { polyline(ctx, [[M.rect.x, M.Y(0)], [M.rect.x + M.rect.w, M.Y(0)]], C.faint, { w: 1.2 }); drawScatterFit(ctx, M, ser('figB1a'), fB1a, c, C.ebt, true); },
        readout(c) { const y = fB1a.b + fB1a.m * c, neg0 = ser('figB1a').filter(q => q[1] < 0); return [['the fitted line at ' + num(c, 3) + 'B', num(y, 3) + '%', 'gain of 10 candidates over 2'], ['checkpoints below zero', String(neg0.length), neg0.length ? 'at ' + neg0.map(q => num(q[0], 2) + 'B').join(', ') + ': more candidates hurt' : ''], ['spread', 'r² ' + fB1a.r2.toFixed(2), 'approx., read from Fig B.1a']]; },
      });
      if (P.fig7) TT.push({
        key: '7', id: 'fig7', tab: 'Farther out of distribution', cont: [1, 4.374],
        text: 'Five datasets, one dot each: the pretraining data (shift 1.0) and four downstream sets. x is downstream perplexity ÷ pretraining perplexity, a measure of how out-of-distribution the data is (p.11). y is the gain from max thinking (thinking longer plus self-verification). “as the data becomes more OOD, thinking leads to greater performance improvements” (p.11). The figure does not say which dot is which dataset.',
        frame: { xr: { lin: [0.8, 4.6], log: null }, yr: { lin: [10, 24.5], log: null }, tx: 0, ty: 0, xfmt: (v) => num(v), yfmt: (v) => num(v) + '%', xlabel: 'OOD shift (downstream ppl ÷ pretraining ppl)', ylabel: '% ppl gain from thinking' },
        clabel: (c) => 'shift ' + c.toFixed(2),
        draw(ctx, M, c) { drawScatterFit(ctx, M, ser('fig7'), f7, c, C.ebt); },
        readout(c) { const y = f7.b + f7.m * c; return [['the fitted line at shift ' + c.toFixed(2), num(y, 3) + '%', `+${num(f7.m, 3)} points per unit of shift`], ['in distribution (shift 1.0)', num(ser('fig7')[0][1], 3) + '%', 'thinking already helps on the pretraining data'], ['fit quality', 'r² ' + f7.r2.toFixed(2), '5 points, approx., read from Fig 7']]; },
      });
      if (P.figB1b) TT.push({
        key: 'b1b', id: 'figB1b', tab: 'Projected to 15T tokens', cont: [9, L10(1.5e13)], logc: true,
        text: 'The paper extends the Fig 6b line to 15T tokens, the data scale of Llama 3, about 1000× the data here (p.10, Fig B.1). Both axes are log, so the straight line looks bent. The star is the paper’s projected point, not a measurement: a noisy trend (r² ≈ 0.18) extended about 450× past the last checkpoint. A gain above 100% only makes sense as a ratio, and the paper does not define it [derived].',
        frame: { xr: { lin: [0, 1], log: [8.75, 13.45] }, yr: { lin: [0, 1], log: [L10(3), L10(2200)] }, tx: 1, ty: 1, xfmt: KIND.tokens, yfmt: (v) => num(v) + '%', xlabel: 'tokens trained on · log scale', ylabel: '% ppl gain over no verification · log' },
        clabel: (c) => KIND.tokens(P10(c)) + ' tokens',
        draw(ctx, M, c) {
          const pts = ser('figB1b'), f = { m: f6b.m / 1e9, b: f6b.b }, xmax = pts[pts.length - 1][0], xc = P10(c);
          const zx = M.X(xmax); ctx.save(); ctx.fillStyle = hatch4 || (hatch4 = hatch(ctx)); ctx.fillRect(zx, M.rect.y, M.rect.x + M.rect.w - zx, M.rect.h); ctx.restore();
          if (M.rect.x + M.rect.w - zx > 90) label(ctx, 'EXTRAPOLATION', zx + 8, M.rect.y + 12, { mono: true, size: 11.5, color: C.ebt });
          pts.forEach(q => dot(ctx, M.X(q[0]), M.Y(q[1]), 4, C.ebt, { stroke: C.bg, alpha: 0.85 }));
          const s = [], e = [];
          for (let i = 0; i <= 100; i++) { const x = P10(9 + (L10(xmax) - 9) * i / 100); s.push([M.X(x), M.Y(f.b + f.m * x)]); }
          if (xc > xmax) for (let i = 0; i <= 100; i++) { const x = P10(L10(xmax) + (c - L10(xmax)) * i / 100); e.push([M.X(x), M.Y(f.b + f.m * x)]); }
          polyline(ctx, s, C.ebt, { w: 2.2 }); polyline(ctx, e, C.ebt, { w: 2.2, dash: [7, 6] });
          const yc = f.b + f.m * xc; dot(ctx, M.X(xc), M.Y(yc), 6, C.ink, { stroke: C.bg });
          const ex = P.figB1b.series[0].extrapolated_point;
          if (ex && c >= L10(ex[0]) - 0.02) { star(ctx, M.X(ex[0]), M.Y(ex[1]), 10, C.ebt); label(ctx, 'paper’s projection, not a result', M.X(ex[0]) - 14, M.Y(ex[1]) + 4, { size: 12.5, align: 'right', color: C.ink }); }
        },
        readout(c) { const xc = P10(c), y = f6b.b + f6b.m * xc / 1e9, last = ser('figB1b').slice(-1)[0][0]; return [['the line at ' + KIND.tokens(xc), num(y, 3) + '%', xc > last ? `${mult(xc / last)} the last checkpoint: <span class="scaling-warn">extrapolation</span>` : 'inside the measured range'], ['the same line', num(f6b.b, 3) + '% + ' + num(f6b.m, 3) + '% per 1B tokens', 'Fig 6b’s fit, unchanged'], ['the paper’s star', P.figB1b.series[0].extrapolated_point ? '≈' + num(P.figB1b.series[0].extrapolated_point[1], 3) + '%' : '–', 'at 15T tokens, approx., read from Fig B.1b']]; },
      });
      if (P.fig12) TT.push({
        key: '12', id: 'fig12', tab: 'Images: 3 vs 300 passes', disc: 3,
        text: 'Image denoising at out-of-distribution noise (σ = 0.2 of the schedule). EBT takes 1 to 3 forward passes; DiT takes 100 to 300. At 1 vs 100 and 2 vs 200 passes the two are about equal; EBT pulls ahead at 3 vs 300, where the values match Table 4: EBT 23.29 dB vs DiT 19.56 dB (p.13). One EBT pass also needs a backward pass for ∇ŷE, so the compute saving is smaller than 100× [derived]. More in the denoising lab.',
        frame: { xr: { lin: [0, 1], log: [L10(0.75), L10(420)] }, yr: { lin: [0, 1], log: [L10(13), L10(25)] }, tx: 1, ty: 1, xfmt: KIND.plain, yfmt: KIND.plain, xlabel: 'forward passes · log scale', ylabel: 'PSNR (dB), higher is better' },
        clabel: (c) => `EBT ${Math.round(c) + 1} · DiT ${(Math.round(c) + 1) * 100}`,
        draw(ctx, M, c) {
          const E = ser('fig12', 1), Dt = ser('fig12', 0), k = Math.round(c);
          [[Dt, C.diff, 'DiT'], [E, C.ebt, 'EBT']].forEach(([pts, col, nm]) => {
            polyline(ctx, pts.slice(0, k + 1).map(q => [M.X(q[0]), M.Y(q[1])]), col, { w: 2.4 });
            pts.forEach((q, i) => dot(ctx, M.X(q[0]), M.Y(q[1]), i === k ? 6.5 : 4.5, i <= k ? col : null, { stroke: i <= k ? C.bg : col, alpha: i <= k ? 1 : 0.5 }));
            const q = pts[k]; label(ctx, `${nm} ${num(q[1], 4)} dB`, M.X(q[0]) + (nm === 'EBT' ? 10 : -10), M.Y(q[1]) - 14, { mono: true, size: 12.5, color: col, align: nm === 'EBT' ? 'left' : 'right' });
          });
        },
        readout(c) { const E = ser('fig12', 1), Dt = ser('fig12', 0), k = Math.round(c); return [[`EBT, ${k + 1} pass${k ? 'es' : ''}`, num(E[k][1], 4) + ' dB', 'approx., read from Fig 12'], [`DiT, ${(k + 1) * 100} passes`, num(Dt[k][1], 4) + ' dB', 'approx., read from Fig 12'], ['Table 4 (σ = 0.2)', 'EBT 23.29 · DiT 19.56', 'exact, p.13']]; },
      });
      let hatch4 = null;
      function drawScatterFit(ctx, M, pts, f, c, col, keepNeg) {
        const xs = pts.map(q => q[0]), x0 = Math.min(...xs), x1 = Math.max(...xs);
        pts.forEach(q => { const seen = q[0] <= c + 1e-9; dot(ctx, M.X(q[0]), M.Y(q[1]), seen ? 4.5 : 3.5, seen ? (keepNeg && q[1] < 0 ? C.bad : col) : null, { stroke: seen ? C.bg : col, alpha: seen ? 0.95 : 0.3 }); });
        const cc = clamp(c, x0, x1);
        polyline(ctx, [[M.X(x0), M.Y(f.b + f.m * x0)], [M.X(cc), M.Y(f.b + f.m * cc)]], C.ink, { w: 2, alpha: 0.9 });
        ctx.save(); ctx.strokeStyle = lib.rgba(C.ink, 0.35); ctx.setLineDash([3, 4]); ctx.beginPath(); ctx.moveTo(M.X(cc), M.rect.y); ctx.lineTo(M.X(cc), M.rect.y + M.rect.h); ctx.stroke(); ctx.restore();
        dot(ctx, M.X(cc), M.Y(f.b + f.m * cc), 6, C.ink, { stroke: C.bg });
      }
      const S4 = { tab: TT.length ? TT[0].key : null, c: 0, playing: false };
      const ttBy = {}; TT.forEach(t => { ttBy[t.key] = t; });
      ctl4.appendChild(h('h4', {}, 'Figure'));
      const tabs4 = h('div', { class: 'scaling-tabs', role: 'group', 'aria-label': 'Thinking figure' }); ctl4.appendChild(tabs4);
      const tabBtns = {};
      TT.forEach(t => { const b = h('button', { type: 'button', class: 'scaling-chip', 'aria-pressed': String(t.key === S4.tab) }, h('span', { class: 'fig' }, P[t.id].figure.replace('Fig ', '')), t.tab); b.addEventListener('click', () => selectTab(t.key)); tabBtns[t.key] = b; tabs4.appendChild(b); });
      const b4Step = btn('Step ▸', { primary: true }), b4Play = btn('Play', { attrs: { 'aria-pressed': 'false' } }), b4Reset = btn('Reset');
      ctl4.appendChild(h('div', { class: 'row' }, b4Step, b4Play, b4Reset));
      const curS = lib.slider({ id: 'scaling-cur', label: 'Cursor', min: 0, max: 1, step: 0.001, value: 1, fmt: () => '', oninput: v => { stopPlay4(); setCur(fromU(v), true); } });
      ctl4.appendChild(curS.el);
      const text4 = h('p', { class: 'scaling-text' }); ctl4.appendChild(text4);
      const head4t = h('h3', { class: 'scaling-ptitle' }), head4s = h('div', { class: 'scaling-psub' }), badge4 = h('span', {});
      ins4.appendChild(h('div', { class: 'scaling-head' }, h('div', {}, head4t, head4s), badge4));
      const cv4 = rcanvas(ins4, { aspect: (w) => w < 560 ? 0.95 : 0.55, minH: 310, maxH: 430, label: 'Thinking-scalability chart for the selected paper figure, with a movable cursor.', draw: () => draw4() });
      const ro4 = h('div', { class: 'scaling-ro' }); ins4.appendChild(ro4);
      const c4 = [roCell(''), roCell(''), roCell('')]; c4.forEach(c => ro4.appendChild(c.el));
      const crop4 = h('div', { class: 'scaling-foot single' }); ins4.appendChild(crop4);
      const anim4 = animator(() => draw4()); anim4.add('c', 0, 7, 1e-4);
      const dom = (t) => t.disc ? [0, t.disc - 1] : t.cont;
      const toU = (t, c) => { const [a, b] = dom(t); return (c - a) / (b - a); };
      function fromU(u) { const t = ttBy[S4.tab], [a, b] = dom(t); const c = a + (b - a) * u; return t.disc ? Math.round(c) : c; }
      function setCur(c, now) {
        const t = ttBy[S4.tab]; S4.c = c; curS.input.value = toU(t, c); curS.out.textContent = t.clabel(c);
        anim4.to('c', c); if (now && !t.disc) anim4.set('c', c); anim4.kick(); upd4();
      }
      function upd4() { const t = ttBy[S4.tab]; t.readout(S4.c).forEach((r, i) => { c4[i].el.querySelector('.k').textContent = r[0]; c4[i].set(r[1], r[2]); }); }
      function selectTab(key) {
        stopPlay4(); S4.tab = key; const t = ttBy[key], p = P[t.id];
        Object.entries(tabBtns).forEach(([k, b]) => b.setAttribute('aria-pressed', String(k === key)));
        head4t.textContent = '“' + (p.printed_title || p.title) + '”'; head4s.textContent = `${p.figure} (p.${p.page}) · ${p.title}`;
        badge4.replaceChildren(lib.badge('paper', 'approx., read from ' + p.figure));
        text4.innerHTML = t.text;
        curS.input.step = t.disc ? String(1 / (t.disc - 1)) : '0.001';
        crop4.replaceChildren(figCrop(p.image, `Original ${p.figure} (p.${p.page}). Paper colors differ from this site’s.`));
        const end = dom(t)[1]; anim4.set('c', end); setCur(end, true);
        cv4.canvas.setAttribute('aria-label', `Chart of ${p.figure}: ${p.title}. Points approx., read from the figure.`);
      }
      function stopPlay4() { S4.playing = false; anim4.extra(null); b4Play.textContent = 'Play'; b4Play.setAttribute('aria-pressed', 'false'); }
      b4Step.addEventListener('click', () => { stopPlay4(); const t = ttBy[S4.tab], [a, b] = dom(t); let c = t.disc ? S4.c + 1 : S4.c + (b - a) / 6; if (S4.c >= b - 1e-6) c = a; setCur(Math.min(b, c)); });
      b4Reset.addEventListener('click', () => { stopPlay4(); const [a] = dom(ttBy[S4.tab]); setCur(a); });
      b4Play.addEventListener('click', () => {
        if (S4.playing) { stopPlay4(); return; }
        const t = ttBy[S4.tab], [a, b] = dom(t); if (S4.c >= b - 1e-6) { anim4.set('c', a); setCur(a, true); }
        S4.playing = true; b4Play.textContent = 'Pause'; b4Play.setAttribute('aria-pressed', 'true');
        let acc = 0;
        anim4.extra((dt) => {
          if (!S4.playing) return false;
          if (t.disc) { acc += dt; if (acc > 1.1) { acc = 0; if (S4.c >= b) { stopPlay4(); return false; } setCur(S4.c + 1); } return true; }
          const c = Math.min(b, S4.c + (b - a) * dt / 4.5); S4.c = c; anim4.set('c', c); curS.input.value = toU(t, c); curS.out.textContent = t.clabel(c); upd4();
          if (c >= b) { stopPlay4(); return false; } return true;
        });
        anim4.kick();
      });
      function draw4() {
        const { ctx, w: W, h: H } = cv4; ctx.save(); ctx.fillStyle = C.bg; ctx.fillRect(0, 0, W, H);
        const t = ttBy[S4.tab]; if (!t) { ctx.restore(); return; }
        const M = frame(ctx, W, H, Object.assign({ top: 14, right: 16 }, t.frame));
        ctx.save(); ctx.beginPath(); ctx.rect(M.rect.x, M.rect.y - 4, M.rect.w + 4, M.rect.h + 8); ctx.clip();
        t.draw(ctx, M, anim4.get('c'));
        ctx.restore(); ctx.restore();
      }

      // ================================================================
      // All axes at a glance
      // ================================================================
      el.appendChild(h('h3', { class: 'scaling-h' }, h('span', { class: 'n' }, '5'), 'All learning axes at a glance'));
      const tblCard = h('div', { class: 'stack' }); el.appendChild(tblCard);
      tblCard.appendChild(h('div', { class: 'row' }, lib.badge('paper', 'printed gains exact; other columns from our refits, approx.')));
      const tbl = h('div', { class: 'tbl scaling-tbl' }); tblCard.appendChild(tbl);
      const rowsHtml = AXOK.map(ax => {
        const R = analyze(ax, new Set()), ci = crossInfo(R), [na, nb] = shortNames(ax), diff = R.yb / R.ya - 1;
        return { ax, cells: [
          `<button type="button" class="scaling-link" data-k="${ax.key}">${ax.group.split(' · ')[0]}: ${ax.long || ax.name}</button><span class="scaling-faint"> ${ax.p.figure}</span>`,
          `<b class="amber">${(ax.p.rate_gain_printed * 100).toFixed(2)}%</b>`,
          `${(R.gain * 100).toFixed(2)}%`,
          R.jk ? `${(R.jk[0] * 100).toFixed(1)} to ${(R.jk[1] * 100).toFixed(1)}%` : '–',
          `${nb} ${diff < 0 ? '<span class="scaling-good">lower</span>' : '<span class="scaling-bad">higher</span>'} by ${Math.abs(diff * 100).toFixed(1)}%`,
          ci.inRange ? `${ci.v} <span class="scaling-faint">(in range)</span>` : ci.flat ? 'parallel' : isFinite(ci.beyond) && ci.beyond > 0 ? `${ci.v} <span class="scaling-faint">(${mult(P10(ci.beyond))} out)</span>` : ci.v,
        ] };
      });
      tbl.innerHTML = '<table><thead><tr><th>Axis</th><th>Printed gain</th><th>Our refit</th><th>Drop any one size</th><th>At the largest measured point</th><th>Fits cross at</th></tr></thead><tbody>' +
        rowsHtml.map(r => '<tr>' + r.cells.map((c, i) => `<td class="${i > 0 ? 'num' : ''}">${c}</td>`).join('') + '</tr>').join('') + '</tbody></table>';
      tbl.querySelectorAll('.scaling-link').forEach(b => b.addEventListener('click', () => { selectAxis(b.dataset.k); bench1.scrollIntoView({ behavior: RM ? 'auto' : 'smooth', block: 'start' }); }));
      tblCard.appendChild(h('p', { class: 'scaling-faint', html: '“At the largest measured point” compares the last measured pair (for FLOPs: the two fits at Transformer++’s largest budget). “Drop any one size” refits with each model size left out in turn, so it shows how much a single run moves the gain.' }));

      el.appendChild(lib.callout('warn', 'What these fits cannot tell you', '<ul class="scaling-ul">' +
        '<li><b>Small scale.</b> Text parameter scaling stops at 396M non-embedding parameters; the largest model is 708M (video); the paper says EBTs “scale well up to 800M parameters” and could not train above 10²¹ FLOPs, about 1300 A100 GPU hours (p.17, p.27, p.33).</li>' +
        '<li><b>Cost.</b> Each EBT pretraining step used about 6.66× the FLOPs of a Transformer++ step (p.36). Inference also pays for gradients: for low-latency serving with a single pass, the extra cost “would not be worth” it (p.27).</li>' +
        '<li><b>Recipe.</b> Hyperparameters were tuned for feed-forward Transformers, not EBTs (p.33). Many scaling runs use S1 EBTs without landscape regularization (p.30).</li>' +
        '<li><b>Few points, one seed.</b> Most fits use 4 or 5 models, every run uses seed 33, and there are no error bars (p.33). Small gains (parameters, FLOPs, depth, width) move a lot when one run is left out.</li>' +
        '<li><b>Metric.</b> Perplexity and Smooth-L1 loss, not task accuracy (p.9).</li>' +
        '<li><b>Extrapolation.</b> Anything past the last measured point is a straight line extended, not a result. The paper’s own foundation-scale expectations (p.9, p.12 to 13) are extrapolations too.</li></ul>'));

      // ---------- initial state ----------
      refresh1(); updateStageUI(); loadPreset(axByKey.params ? 'params' : PRE[0]); upd3(); show3(FL.length);
      if (TT.length) selectTab(TT[0].key);
      if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => redraws.forEach(f => f()));
    },
  });
})();
