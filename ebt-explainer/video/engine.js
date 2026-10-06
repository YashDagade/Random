/* EBT explainer video engine.
   Scenes register with EBTV.scene({id, title, dur, assets, draw(ctx, t, U)}).
   draw() must be a pure function of local time t (seconds) and repaint the full 1920x1080 frame. */
(function () {
  const W = 1920, H = 1080, FPS = 30;
  const ROOT = (window.EBTV_ROOT || '../'); // project root relative to the page

  const C = {
    bg: '#0d131f', panel: '#141c2b', panel2: '#1a2437', rule: '#283449',
    ink: '#e9eef6', muted: '#93a1b8', faint: '#5d6b82',
    ar: '#8ea7ff', rnn: '#c99bff', diff: '#ff8f7a', ebt: '#ffc95c', truth: '#74e0c0', bad: '#ff6b81',
  };
  const CMAP = ['#0b1a33', '#15466b', '#2b7a8a', '#6fae84', '#e3b75a', '#f7e6bd'];
  const F = { display: '"Source Serif 4", Georgia, serif', body: '"Atkinson Hyperlegible", "Segoe UI", sans-serif', mono: '"JetBrains Mono", Menlo, monospace' };

  // ---------- math / timing ----------
  const clamp = (x, a = 0, b = 1) => Math.max(a, Math.min(b, x));
  const lerp = (a, b, t) => a + (b - a) * t;
  const inv = (a, b, x) => clamp((x - a) / (b - a));
  const ease = (x) => { x = clamp(x); return x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2; };
  const easeOut = (x) => 1 - Math.pow(1 - clamp(x), 3);
  const easeIn = (x) => Math.pow(clamp(x), 3);
  const seg = (t, a, b, e = ease) => e(inv(a, b, t));
  // visible window [a,b] with fade-in fi and fade-out fo -> alpha
  const fade = (t, a, b, fi = 0.5, fo = 0.5) => {
    if (t < a || t > b) return 0;
    return Math.min(fi > 0 ? easeOut((t - a) / fi) : 1, fo > 0 ? easeOut((b - t) / fo) : 1, 1);
  };
  function rng(seed) { // mulberry32
    let a = seed >>> 0;
    const f = () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
    f.normal = () => { const u = Math.max(1e-9, f()), v = f(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
    return f;
  }
  const hex = (h) => { h = h.replace('#', ''); if (h.length === 3) h = h.split('').map(c => c + c).join(''); return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)]; };
  const rgba = (h, a = 1) => { const [r, g, b] = hex(h); return `rgba(${r},${g},${b},${a})`; };
  const mix = (h1, h2, t) => { const a = hex(h1), b = hex(h2); return `rgb(${a.map((v, i) => Math.round(lerp(v, b[i], t))).join(',')})`; };
  const CM = CMAP.map(hex);
  function cmap(v) { // v in [0,1] -> [r,g,b]
    v = clamp(v) * (CM.length - 1); const i = Math.min(CM.length - 2, Math.floor(v)), f = v - i;
    return [0, 1, 2].map(k => Math.round(lerp(CM[i][k], CM[i + 1][k], f)));
  }
  const cmapCss = (v, a = 1) => { const c = cmap(v); return `rgba(${c[0]},${c[1]},${c[2]},${a})`; };

  // ---------- caches ----------
  const _cache = new Map();
  function cache(key, w, h, drawFn) {
    if (_cache.has(key)) return _cache.get(key);
    const c = document.createElement('canvas'); c.width = Math.max(1, Math.round(w)); c.height = Math.max(1, Math.round(h));
    drawFn(c.getContext('2d'), c.width, c.height); _cache.set(key, c); return c;
  }

  // ---------- text ----------
  function font(size, kind = 'body', weight = 400, italic = false) { return `${italic ? 'italic ' : ''}${weight} ${size}px ${F[kind] || kind}`; }
  function wrap(ctx, s, maxWidth) {
    const out = [];
    String(s).split('\n').forEach(par => {
      if (!maxWidth) { out.push(par); return; }
      const words = par.split(' '); let line = '';
      words.forEach(w => { const test = line ? line + ' ' + w : w; if (ctx.measureText(test).width > maxWidth && line) { out.push(line); line = w; } else line = test; });
      out.push(line);
    });
    return out;
  }
  // o: size, kind, weight, italic, color, align, baseline ('top'|'middle'|'alphabetic'), alpha, lh (line height multiplier), maxWidth, spacing (px letter spacing)
  function text(ctx, s, x, y, o = {}) {
    const size = o.size || 32;
    ctx.save();
    ctx.font = font(size, o.kind || 'body', o.weight || 400, o.italic);
    ctx.fillStyle = o.color || C.ink; ctx.globalAlpha *= (o.alpha == null ? 1 : o.alpha);
    ctx.textAlign = o.align || 'left'; ctx.textBaseline = o.baseline || 'top';
    if (o.spacing) ctx.letterSpacing = o.spacing + 'px';
    const lines = wrap(ctx, s, o.maxWidth); const lh = size * (o.lh || 1.3);
    let w = 0; lines.forEach((ln, i) => { ctx.fillText(ln, x, y + i * lh); w = Math.max(w, ctx.measureText(ln).width); });
    ctx.restore();
    return { w, h: lines.length * lh, lines: lines.length };
  }
  function measure(ctx, s, o = {}) { ctx.save(); ctx.font = font(o.size || 32, o.kind || 'body', o.weight || 400, o.italic); if (o.spacing) ctx.letterSpacing = o.spacing + 'px'; const lines = wrap(ctx, s, o.maxWidth); const w = Math.max(...lines.map(l => ctx.measureText(l).width)); ctx.restore(); return { w, h: lines.length * (o.size || 32) * (o.lh || 1.3), lines: lines.length }; }
  // inline segments: [{t, color, kind, weight, italic, size}] on one line
  function rich(ctx, segs, x, y, o = {}) {
    ctx.save(); ctx.globalAlpha *= (o.alpha == null ? 1 : o.alpha); ctx.textBaseline = o.baseline || 'top';
    let total = 0; const size = o.size || 32;
    segs.forEach(s => { ctx.font = font(s.size || size, s.kind || o.kind || 'body', s.weight || o.weight || 400, s.italic); total += ctx.measureText(s.t).width; });
    let cx = o.align === 'center' ? x - total / 2 : o.align === 'right' ? x - total : x;
    segs.forEach(s => { ctx.font = font(s.size || size, s.kind || o.kind || 'body', s.weight || o.weight || 400, s.italic); ctx.fillStyle = s.color || o.color || C.ink; ctx.fillText(s.t, cx, y); cx += ctx.measureText(s.t).width; });
    ctx.restore(); return total;
  }
  const typewriter = (s, p) => String(s).slice(0, Math.round(String(s).length * clamp(p)));

  // ---------- shapes ----------
  function rr(ctx, x, y, w, h, r = 12) { ctx.beginPath(); ctx.roundRect(x, y, w, h, r); }
  function panel(ctx, x, y, w, h, o = {}) {
    ctx.save(); ctx.globalAlpha *= (o.alpha == null ? 1 : o.alpha);
    rr(ctx, x, y, w, h, o.r == null ? 14 : o.r);
    if (o.fill !== false) { ctx.fillStyle = o.fill || C.panel; ctx.fill(); }
    if (o.stroke) { ctx.strokeStyle = o.stroke; ctx.lineWidth = o.lw || 2; if (o.dash) ctx.setLineDash(o.dash); ctx.stroke(); }
    ctx.restore();
  }
  function arrow(ctx, x1, y1, x2, y2, o = {}) {
    const p = o.progress == null ? 1 : clamp(o.progress); if (p <= 0) return;
    const ex = lerp(x1, x2, p), ey = lerp(y1, y2, p);
    ctx.save(); ctx.globalAlpha *= (o.alpha == null ? 1 : o.alpha);
    ctx.strokeStyle = ctx.fillStyle = o.color || C.muted; ctx.lineWidth = o.width || 3; ctx.lineCap = 'round';
    if (o.dash) ctx.setLineDash(o.dash);
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(ex, ey); ctx.stroke(); ctx.setLineDash([]);
    const a = Math.atan2(y2 - y1, x2 - x1), hs = o.head || 14;
    ctx.beginPath(); ctx.moveTo(ex, ey); ctx.lineTo(ex - hs * Math.cos(a - 0.45), ey - hs * Math.sin(a - 0.45)); ctx.lineTo(ex - hs * Math.cos(a + 0.45), ey - hs * Math.sin(a + 0.45)); ctx.closePath(); ctx.fill();
    ctx.restore();
  }
  // polyline in pixel coords with draw-on progress
  function line(ctx, pts, o = {}) {
    if (!pts || pts.length < 2) return;
    const p = o.progress == null ? 1 : clamp(o.progress); if (p <= 0) return;
    let total = 0; const L = [0]; for (let i = 1; i < pts.length; i++) { total += Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]); L.push(total); }
    const lim = total * p;
    ctx.save(); ctx.globalAlpha *= (o.alpha == null ? 1 : o.alpha);
    ctx.strokeStyle = o.color || C.ink; ctx.lineWidth = o.width || 3; ctx.lineJoin = 'round'; ctx.lineCap = 'round'; if (o.dash) ctx.setLineDash(o.dash);
    ctx.beginPath(); ctx.moveTo(pts[0][0], pts[0][1]);
    for (let i = 1; i < pts.length; i++) {
      if (L[i] <= lim) ctx.lineTo(pts[i][0], pts[i][1]);
      else { const f = (lim - L[i - 1]) / Math.max(1e-9, L[i] - L[i - 1]); ctx.lineTo(lerp(pts[i - 1][0], pts[i][0], f), lerp(pts[i - 1][1], pts[i][1], f)); break; }
    }
    ctx.stroke(); ctx.restore();
  }
  function dot(ctx, x, y, r, color, o = {}) { ctx.save(); ctx.globalAlpha *= (o.alpha == null ? 1 : o.alpha); ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fillStyle = color; ctx.fill(); if (o.stroke) { ctx.strokeStyle = o.stroke; ctx.lineWidth = o.lw || 2; ctx.stroke(); } ctx.restore(); }

  // ---------- axes & plots ----------
  // o: {x,y,w,h, xlim:[a,b], ylim:[a,b], xlog, ylog, xticks:[..], yticks:[..], xfmt, yfmt, xlabel, ylabel, grid, alpha, size}
  function axes(ctx, o) {
    const tx = (v) => o.xlog ? Math.log10(v) : v, ty = (v) => o.ylog ? Math.log10(v) : v;
    const [xa, xb] = o.xlim.map(tx), [ya, yb] = o.ylim.map(ty);
    const X = (v) => o.x + (tx(v) - xa) / (xb - xa) * o.w;
    const Y = (v) => o.y + o.h - (ty(v) - ya) / (yb - ya) * o.h;
    const al = o.alpha == null ? 1 : o.alpha, fs = o.size || 22;
    if (al > 0 && !o.hidden) {
      ctx.save(); ctx.globalAlpha *= al;
      ctx.strokeStyle = C.rule; ctx.lineWidth = 1.5;
      if (o.grid !== false) {
        (o.xticks || []).forEach(v => { ctx.beginPath(); ctx.moveTo(X(v), o.y); ctx.lineTo(X(v), o.y + o.h); ctx.stroke(); });
        (o.yticks || []).forEach(v => { ctx.beginPath(); ctx.moveTo(o.x, Y(v)); ctx.lineTo(o.x + o.w, Y(v)); ctx.stroke(); });
      }
      ctx.strokeStyle = C.faint; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(o.x, o.y); ctx.lineTo(o.x, o.y + o.h); ctx.lineTo(o.x + o.w, o.y + o.h); ctx.stroke();
      const xf = o.xfmt || ((v) => String(v)), yf = o.yfmt || ((v) => String(v));
      (o.xticks || []).forEach(v => text(ctx, xf(v), X(v), o.y + o.h + 10, { size: fs, kind: 'mono', color: C.muted, align: 'center' }));
      (o.yticks || []).forEach(v => text(ctx, yf(v), o.x - 12, Y(v), { size: fs, kind: 'mono', color: C.muted, align: 'right', baseline: 'middle' }));
      if (o.xlabel) text(ctx, o.xlabel, o.x + o.w / 2, o.y + o.h + 16 + fs * 1.3, { size: fs + 2, color: C.muted, align: 'center' });
      if (o.ylabel) { ctx.save(); ctx.translate(o.x - 30 - fs * 3.2, o.y + o.h / 2); ctx.rotate(-Math.PI / 2); text(ctx, o.ylabel, 0, 0, { size: fs + 2, color: C.muted, align: 'center', baseline: 'middle' }); ctx.restore(); }
      ctx.restore();
    }
    return { X, Y, o };
  }
  // pts in data coords
  function plot(ctx, ax, pts, o = {}) {
    const px = pts.filter(p => p && isFinite(p[0]) && isFinite(p[1])).map(p => [ax.X(p[0]), ax.Y(p[1])]);
    if (o.line !== false) line(ctx, px, o);
    if (o.markers) { const n = Math.ceil(px.length * (o.progress == null ? 1 : clamp(o.progress))); for (let i = 0; i < n; i++) dot(ctx, px[i][0], px[i][1], o.markers, o.color || C.ink, { alpha: o.alpha, stroke: o.markerStroke }); }
    return px;
  }

  // ---------- heatmaps / contours / surfaces ----------
  function gridRange(grid) { let lo = Infinity, hi = -Infinity; grid.forEach(r => r.forEach(v => { if (v < lo) lo = v; if (v > hi) hi = v; })); return [lo, hi]; }
  // grid[row][col], row 0 = top unless o.flipY
  function heatmap(ctx, grid, x, y, w, h, o = {}) {
    const key = o.key || null; const [lo, hi] = o.range || gridRange(grid);
    const make = (g, cw, ch) => {
      const rows = grid.length, cols = grid[0].length; const img = g.createImageData(cols, rows);
      for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) { const rr0 = o.flipY ? rows - 1 - r : r; let v = (grid[rr0][c] - lo) / (hi - lo || 1); if (o.gamma) v = Math.pow(clamp(v), o.gamma); const col = cmap(v); const i = 4 * (r * cols + c); img.data[i] = col[0]; img.data[i + 1] = col[1]; img.data[i + 2] = col[2]; img.data[i + 3] = 255; }
      const tmp = document.createElement('canvas'); tmp.width = cols; tmp.height = rows; tmp.getContext('2d').putImageData(img, 0, 0);
      g.imageSmoothingEnabled = o.smooth !== false; g.imageSmoothingQuality = 'high'; g.drawImage(tmp, 0, 0, cw, ch);
    };
    const c = key ? cache('hm:' + key, w, h, make) : (() => { const cc = document.createElement('canvas'); cc.width = Math.round(w); cc.height = Math.round(h); make(cc.getContext('2d'), cc.width, cc.height); return cc; })();
    ctx.save(); ctx.globalAlpha *= (o.alpha == null ? 1 : o.alpha); ctx.drawImage(c, x, y, w, h); ctx.restore();
  }
  // marching squares contour lines; levels: array of values
  function contours(ctx, grid, x, y, w, h, levels, o = {}) {
    const rows = grid.length, cols = grid[0].length, sx = w / (cols - 1), sy = h / (rows - 1);
    const G = (r, c) => grid[o.flipY ? rows - 1 - r : r][c];
    ctx.save(); ctx.globalAlpha *= (o.alpha == null ? 1 : o.alpha); ctx.strokeStyle = o.color || 'rgba(233,238,246,0.35)'; ctx.lineWidth = o.width || 1.5;
    ctx.beginPath();
    levels.forEach(L => {
      for (let r = 0; r < rows - 1; r++) for (let c = 0; c < cols - 1; c++) {
        const v = [G(r, c), G(r, c + 1), G(r + 1, c + 1), G(r + 1, c)];
        const P = [[c, r], [c + 1, r], [c + 1, r + 1], [c, r + 1]]; const pts = [];
        for (let k = 0; k < 4; k++) { const a = v[k], b = v[(k + 1) % 4]; if ((a < L) !== (b < L)) { const f = (L - a) / (b - a); const pa = P[k], pb = P[(k + 1) % 4]; pts.push([x + lerp(pa[0], pb[0], f) * sx, y + lerp(pa[1], pb[1], f) * sy]); } }
        if (pts.length >= 2) { ctx.moveTo(pts[0][0], pts[0][1]); ctx.lineTo(pts[1][0], pts[1][1]); if (pts.length === 4) { ctx.moveTo(pts[2][0], pts[2][1]); ctx.lineTo(pts[3][0], pts[3][1]); } }
      }
    });
    ctx.stroke(); ctx.restore();
  }
  // Pseudo-3D surface of grid[row][col] (values = height). o: {cx, cy, size (px width of base), yaw (rad), pitch (rad), zscale (px for full range), range:[lo,hi], step (subsample), alpha, wire}
  // Returns project(u, v, z) where u,v in [0,1] grid coords (u = col fraction, v = row fraction) and z = raw value.
  function surface(ctx, grid, o = {}) {
    const rows = grid.length, cols = grid[0].length, st = o.step || 1;
    const [lo, hi] = o.range || gridRange(grid);
    const yaw = o.yaw == null ? -0.7 : o.yaw, pitch = o.pitch == null ? 0.95 : o.pitch;
    const size = o.size || 700, zs = o.zscale || 260, cx = o.cx || W / 2, cy = o.cy || H / 2;
    const cyw = Math.cos(yaw), syw = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
    const project = (u, v, z) => { // u,v in [0,1]
      const X0 = (u - 0.5) * size, Y0 = (v - 0.5) * size; const zn = ((z - lo) / (hi - lo || 1)) * zs;
      const xr = X0 * cyw - Y0 * syw, yr = X0 * syw + Y0 * cyw; // rotate in plane
      return [cx + xr, cy + yr * cp - zn * sp, yr * sp + zn * cp];
    };
    const quads = [];
    for (let r = 0; r < rows - st; r += st) for (let c = 0; c < cols - st; c += st) {
      const z = [grid[r][c], grid[r][c + st], grid[r + st][c + st], grid[r + st][c]];
      const p = [project(c / (cols - 1), r / (rows - 1), z[0]), project((c + st) / (cols - 1), r / (rows - 1), z[1]), project((c + st) / (cols - 1), (r + st) / (rows - 1), z[2]), project(c / (cols - 1), (r + st) / (rows - 1), z[3])];
      const depth = (p[0][2] + p[1][2] + p[2][2] + p[3][2]) / 4; const zm = (z[0] + z[1] + z[2] + z[3]) / 4;
      quads.push({ p, depth, v: (zm - lo) / (hi - lo || 1) });
    }
    quads.sort((a, b) => a.depth - b.depth);
    ctx.save(); ctx.globalAlpha *= (o.alpha == null ? 1 : o.alpha); ctx.lineJoin = 'round';
    quads.forEach(q => {
      ctx.beginPath(); ctx.moveTo(q.p[0][0], q.p[0][1]); for (let k = 1; k < 4; k++) ctx.lineTo(q.p[k][0], q.p[k][1]); ctx.closePath();
      const col = cmap(q.v); ctx.fillStyle = `rgb(${col[0]},${col[1]},${col[2]})`; ctx.fill();
      ctx.strokeStyle = o.wire || `rgba(13,19,31,0.35)`; ctx.lineWidth = 0.8; ctx.stroke();
    });
    ctx.restore();
    return project;
  }

  // ---------- images ----------
  const _imgs = new Map();
  function loadImg(path) {
    if (_imgs.has(path)) return _imgs.get(path).p;
    const im = new Image(); const p = new Promise(res => { im.onload = () => res(im); im.onerror = () => { console.warn('missing image', path); res(null); }; });
    im.src = ROOT + path; _imgs.set(path, { im, p, ok: false }); p.then(r => { _imgs.get(path).ok = !!r; }); return p;
  }
  const img = (path) => { const e = _imgs.get(path); return e && e.ok ? e.im : null; };
  // o: alpha, r (corner radius), border color, pixelated, fit ('cover'|'contain'|'fill')
  function drawImage(ctx, path, x, y, w, h, o = {}) {
    const im = img(path);
    ctx.save(); ctx.globalAlpha *= (o.alpha == null ? 1 : o.alpha);
    if (o.r) { rr(ctx, x, y, w, h, o.r); ctx.clip(); }
    if (!im) { ctx.fillStyle = C.panel2; ctx.fillRect(x, y, w, h); text(ctx, 'missing: ' + path, x + 8, y + 8, { size: 16, color: C.bad, kind: 'mono', maxWidth: w - 16 }); ctx.restore(); return; }
    ctx.imageSmoothingEnabled = !o.pixelated;
    let sx = 0, sy = 0, sw = im.naturalWidth, sh = im.naturalHeight, dx = x, dy = y, dw = w, dh = h;
    if (o.fit === 'cover') { const s = Math.max(w / sw, h / sh); const cw = w / s, ch = h / s; sx = (sw - cw) / 2; sy = (sh - ch) / 2; sw = cw; sh = ch; }
    else if (o.fit === 'contain') { const s = Math.min(w / sw, h / sh); dw = sw * s; dh = sh * s; dx = x + (w - dw) / 2; dy = y + (h - dh) / 2; }
    ctx.drawImage(im, sx, sy, sw, sh, dx, dy, dw, dh);
    ctx.restore();
    if (o.border) { ctx.save(); ctx.globalAlpha *= (o.alpha == null ? 1 : o.alpha); rr(ctx, x, y, w, h, o.r || 0); ctx.strokeStyle = o.border; ctx.lineWidth = o.lw || 2; ctx.stroke(); ctx.restore(); }
  }

  const data = (name) => (window.EBT_DATA || {})[name] || null;

  // ---------- standard scene furniture ----------
  // eyebrow + title, top-left, fade in at 0.2s and out before dur
  function header(ctx, t, dur, eyebrow, title, o = {}) {
    const a = fade(t, 0.15, dur - 0.3, 0.7, 0.5) * (o.alpha == null ? 1 : o.alpha);
    const dy = (1 - easeOut(inv(0.15, 0.9, t))) * 14;
    text(ctx, eyebrow.toUpperCase(), 96, 70 + dy, { size: 24, kind: 'mono', color: o.color || C.ebt, alpha: a, spacing: 3 });
    text(ctx, title, 96, 106 + dy, { size: o.size || 58, kind: 'display', weight: 600, color: C.ink, alpha: a, maxWidth: o.maxWidth || 1500 });
  }
  // lower caption panel; visible during [a,b]
  function caption(ctx, s, t, a, b, o = {}) {
    const al = fade(t, a, b, 0.5, 0.5); if (al <= 0) return;
    const size = o.size || 34; const maxW = o.maxWidth || 1400;
    const m = measure(ctx, s, { size, maxWidth: maxW });
    const pw = m.w + 64, ph = m.h + 40, x = (o.x != null ? o.x : W / 2) - (o.align === 'left' ? 0 : pw / 2), y = o.y != null ? o.y : H - 70 - ph;
    panel(ctx, x, y, pw, ph, { fill: rgba(C.panel, 0.92), stroke: o.stroke || C.rule, alpha: al, r: 16 });
    text(ctx, s, x + 32, y + 20, { size, color: o.color || C.ink, alpha: al, maxWidth: maxW, lh: 1.32 });
  }
  // provenance badge: kind 'paper' | 'toy' | 'ext'
  function badge(ctx, kind, label, x, y, o = {}) {
    const col = kind === 'paper' ? C.ar : kind === 'toy' ? C.truth : C.rnn;
    const txt = (kind === 'paper' ? 'FROM THE PAPER' : kind === 'toy' ? 'TOY MODEL TRAINED FOR THIS EXPLAINER' : 'BEYOND THE PAPER') + (label ? ' · ' + label : '');
    const m = measure(ctx, txt, { size: 18, kind: 'mono', spacing: 1 });
    const xx = o.align === 'right' ? x - m.w - 28 : x;
    panel(ctx, xx, y, m.w + 28, 36, { fill: rgba(col, 0.13), stroke: rgba(col, 0.6), alpha: o.alpha, r: 8, lw: 1.5 });
    text(ctx, txt, xx + 14, y + 8, { size: 18, kind: 'mono', color: col, alpha: o.alpha, spacing: 1 });
  }
  function background(ctx) {
    ctx.fillStyle = C.bg; ctx.fillRect(0, 0, W, H);
    // faint vignette
    const g = cache('vignette', W, H, (g, w, h) => { const gr = g.createRadialGradient(w / 2, h / 2, h * 0.2, w / 2, h / 2, h * 0.95); gr.addColorStop(0, 'rgba(30,45,70,0.20)'); gr.addColorStop(1, 'rgba(0,0,0,0.0)'); g.fillStyle = gr; g.fillRect(0, 0, w, h); });
    ctx.drawImage(g, 0, 0);
  }

  const U = { W, H, FPS, C, CMAP, F, clamp, lerp, inv, ease, easeOut, easeIn, seg, fade, rng, hex, rgba, mix, cmap, cmapCss,
    cache, font, wrap, text, measure, rich, typewriter, rr, panel, arrow, line, dot, axes, plot, gridRange, heatmap, contours, surface,
    loadImg, img, drawImage, data, header, caption, badge, background };

  // ---------- scene registry & timeline ----------
  const registry = new Map();
  const EBTV = window.EBTV = {
    W, H, FPS, U, registry, order: [],
    scene(def) { registry.set(def.id, def); (def.assets || []).forEach(loadImg); },
    timeline() {
      let t0 = 0; return EBTV.order.map(id => { const d = registry.get(id) || { id, title: id + ' (missing)', dur: 5, draw: placeholder(id) }; const e = { id, def: d, t0, t1: t0 + d.dur }; t0 += d.dur; return e; });
    },
    total() { const tl = EBTV.timeline(); return tl.length ? tl[tl.length - 1].t1 : 0; },
    at(T) { const tl = EBTV.timeline(); for (const e of tl) if (T < e.t1) return { e, t: T - e.t0 }; const e = tl[tl.length - 1]; return { e, t: e.def.dur - 1e-3 }; },
    render(ctx, T) {
      const { e, t } = EBTV.at(T);
      ctx.save(); ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.globalAlpha = 1; background(ctx);
      try { e.def.draw(ctx, t, U); } catch (err) { console.error(err); text(ctx, 'scene error: ' + e.id + ': ' + err.message, 40, 40, { size: 24, color: C.bad, kind: 'mono', maxWidth: 1800 }); }
      // global scene fade in/out (0.35s) to soften cuts
      const fa = Math.min(inv(0, 0.35, t), inv(e.def.dur, e.def.dur - 0.45, t));
      if (fa < 1) { ctx.globalAlpha = 1 - fa; ctx.fillStyle = C.bg; ctx.fillRect(0, 0, W, H); }
      ctx.restore();
    },
    renderScene(ctx, id, t) { const d = registry.get(id); ctx.save(); background(ctx); d.draw(ctx, t, U); ctx.restore(); },
    ready: null,
  };
  function placeholder(id) {
    return (ctx, t, U) => { U.text(ctx, 'Scene ' + id + ' is not built yet', W / 2, H / 2, { size: 48, kind: 'display', align: 'center', baseline: 'middle', color: C.muted }); };
  }
  // fonts
  const fonts = [
    ['Source Serif 4', 'assets/fonts/SourceSerif4.ttf', { weight: '200 900' }],
    ['Source Serif 4', 'assets/fonts/SourceSerif4-Italic.ttf', { weight: '200 900', style: 'italic' }],
    ['Atkinson Hyperlegible', 'assets/fonts/Atkinson-Regular.ttf', { weight: '400' }],
    ['Atkinson Hyperlegible', 'assets/fonts/Atkinson-Bold.ttf', { weight: '700' }],
    ['Atkinson Hyperlegible', 'assets/fonts/Atkinson-Italic.ttf', { weight: '400', style: 'italic' }],
    ['JetBrains Mono', 'assets/fonts/JetBrainsMono.ttf', { weight: '100 800' }],
  ];
  EBTV.fontsReady = Promise.all(fonts.map(([fam, url, desc]) => new FontFace(fam, `url(${ROOT}${url})`, desc).load().then(f => { document.fonts.add(f); }).catch(e => console.warn('font', url, e))));
  // load scene scripts listed in EBTV_SCENES (set by scenes/manifest.js) sequentially
  EBTV.loadScenes = function (list) {
    EBTV.order = list.slice();
    return list.reduce((p, id) => p.then(() => new Promise(res => { const s = document.createElement('script'); s.src = 'scenes/' + id + '.js'; s.onload = res; s.onerror = () => { console.warn('missing scene file', id); res(); }; document.head.appendChild(s); })), Promise.resolve());
  };
  EBTV.init = function (list) {
    EBTV.ready = EBTV.fontsReady.then(() => EBTV.loadScenes(list)).then(() => Promise.all([..._imgs.values()].map(e => e.p))).then(() => true);
    return EBTV.ready;
  };
  // helpers for the headless renderer
  EBTV.frameData = function (canvas, T, type = 'image/jpeg', q = 0.92) { EBTV.render(canvas.getContext('2d'), T); return canvas.toDataURL(type, q); };
})();
