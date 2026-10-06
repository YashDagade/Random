// s01_intro: "Thinking, fast and slow" (50 s)
// Beats: A title card (0-6.8) | B System 1 vs System 2 (6.8-18.3) | C today's System 2 recipes and their limits (18.3-27.2)
//        D the core question (27.2-33.4) | E the paper's answer: verify, then optimize (33.4-46.0) | F roadmap (46.0-50)
// Facts: abstract p.1, intro p.1-4, Table 1 p.3, Eq. 1 p.7, Algorithms 1-2 p.7.
(function () {
  const DUR = 50;

  // ---------- timing of beats ----------
  const A0 = 0, A1 = 6.8;
  const B0 = 6.8, B1 = 18.3;
  const C0 = 18.3, C1 = 27.2;
  const D0 = 27.2, D1 = 33.4;
  const E0 = 33.4, E1 = 46.0;
  const F0 = 46.0, F1 = 50.0;

  // ---------- procedural energy field for the background (world coords in px) ----------
  // A gentle bowl with several wells. Purely decorative; it is a picture of "an energy landscape".
  const WELLS = [
    [1460, 600, 1.75, 245],
    [1170, 300, 0.55, 130],
    [430, 830, 1.0, 240],
    [720, 240, 0.5, 170],
    [1730, 980, 0.45, 150],
    [170, 330, 0.55, 190],
    [960, 700, 0.35, 160],
  ];
  function field(x, y, smooth) {
    const qx = (x - 960) / 1000, qy = (y - 540) / 760;
    let e = 0.9 * (qx * qx + qy * qy);
    for (let k = 0; k < WELLS.length; k++) {
      const w = WELLS[k], dx = x - w[0], dy = y - w[1];
      e -= w[2] * Math.exp(-(dx * dx + dy * dy) / (2 * w[3] * w[3]));
    }
    if (!smooth) e += 0.045 * Math.sin(x / 85 + 0.7 * Math.sin(y / 130)) + 0.035 * Math.cos(y / 70 - x / 210);
    return e;
  }
  const PAN = (t) => [-4.5 * t, -1.4 * t]; // screen = world + PAN(t)

  // contour levels, fixed for the whole scene (no popping)
  let LEVELS = null;
  function levels() {
    if (LEVELS) return LEVELS;
    let lo = Infinity, hi = -Infinity;
    for (let y = -40; y <= 1200; y += 20) for (let x = -40; x <= 2200; x += 20) { const v = field(x, y); if (v < lo) lo = v; if (v > hi) hi = v; }
    const n = 17; LEVELS = [];
    for (let k = 0; k < n; k++) LEVELS.push(lo + (hi - lo) * (0.03 + 0.9 * k / (n - 1)));
    return LEVELS;
  }

  const GS = 16, GC = 121, GR = 69; // 120*16 = 1920, 68*16 = 1088
  const GRID = new Float32Array(GC * GR);
  function drawField(ctx, t, U, alpha) {
    if (alpha <= 0.003) return;
    const L = levels(), nL = L.length;
    const [ox, oy] = PAN(t);
    for (let r = 0; r < GR; r++) for (let c = 0; c < GC; c++) GRID[r * GC + c] = field(c * GS - ox, r * GS - oy);
    const segs = L.map(() => []);
    for (let r = 0; r < GR - 1; r++) for (let c = 0; c < GC - 1; c++) {
      const i = r * GC + c;
      const a = GRID[i], b = GRID[i + 1], d = GRID[i + GC], e = GRID[i + GC + 1];
      const mn = Math.min(a, b, d, e), mx = Math.max(a, b, d, e);
      const x0 = c * GS, y0 = r * GS;
      for (let k = 0; k < nL; k++) {
        const lv = L[k]; if (lv <= mn || lv > mx) continue;
        const s = segs[k];
        // edges: top a-b, right b-e, bottom d-e, left a-d
        let n = 0, px0 = 0, py0 = 0;
        const push = (x, y) => { if (n === 0) { px0 = x; py0 = y; n = 1; } else if (n === 1) { s.push(px0, py0, x, y); n = 2; } else if (n === 2) { px0 = x; py0 = y; n = 3; } else { s.push(px0, py0, x, y); n = 4; } };
        if ((a < lv) !== (b < lv)) push(x0 + GS * (lv - a) / (b - a), y0);
        if ((b < lv) !== (e < lv)) push(x0 + GS, y0 + GS * (lv - b) / (e - b));
        if ((d < lv) !== (e < lv)) push(x0 + GS * (lv - d) / (e - d), y0 + GS);
        if ((a < lv) !== (d < lv)) push(x0, y0 + GS * (lv - a) / (d - a));
      }
    }
    ctx.save(); ctx.lineCap = 'round';
    for (let k = 0; k < nL; k++) {
      const s = segs[k]; if (!s.length) continue;
      const v = k / (nL - 1);
      const major = k % 4 === 0;
      ctx.strokeStyle = U.cmapCss(0.3 + 0.62 * v, alpha * (major ? 0.42 : 0.24));
      ctx.lineWidth = major ? 1.9 : 1.2;
      ctx.beginPath();
      for (let j = 0; j < s.length; j += 4) { ctx.moveTo(s[j], s[j + 1]); ctx.lineTo(s[j + 2], s[j + 3]); }
      ctx.stroke();
    }
    ctx.restore();
  }

  // gradient descent path of the marble on the smooth field (world coords)
  let PATH = null;
  function marblePath() {
    if (PATH) return PATH;
    let x = 1300, y = 90; const pts = [[x, y]]; const h = 1.5, eta = 5200;
    for (let i = 0; i < 75; i++) {
      const gx = (field(x + h, y, true) - field(x - h, y, true)) / (2 * h);
      const gy = (field(x, y + h, true) - field(x, y - h, true)) / (2 * h);
      x -= eta * gx; y -= eta * gy; pts.push([x, y]);
    }
    PATH = pts; return PATH;
  }

  const KEYS = [[0, 0.0], [0.5, 1], [6.3, 1], [7.4, 0.42], [17.9, 0.42], [18.7, 0.34], [26.9, 0.34], [27.9, 0.78], [32.8, 0.78], [33.8, 0.26], [45.6, 0.26], [46.4, 0.5], [49.3, 0.5], [50, 0]];
  function fieldAlpha(t, U) {
    for (let i = 1; i < KEYS.length; i++) if (t <= KEYS[i][0]) { const [ta, va] = KEYS[i - 1], [tb, vb] = KEYS[i]; return U.lerp(va, vb, U.ease(U.inv(ta, tb, t))); }
    return 0;
  }

  // ---------- small helpers ----------
  function hdr(ctx, U, t, a, b, eyebrow, title, o = {}) {
    const al = U.fade(t, a, b, 0.7, 0.5); if (al <= 0) return;
    const dy = (1 - U.easeOut(U.inv(a, a + 0.9, t))) * 14;
    U.text(ctx, eyebrow.toUpperCase(), 96, 70 + dy, { size: 24, kind: 'mono', color: o.color || U.C.ebt, alpha: al, spacing: 3 });
    if (Array.isArray(title)) U.rich(ctx, title, 96, 106 + dy, { size: o.size || 56, kind: 'display', weight: 600, color: U.C.ink, alpha: al });
    else U.text(ctx, title, 96, 106 + dy, { size: o.size || 56, kind: 'display', weight: 600, color: U.C.ink, alpha: al, maxWidth: o.maxWidth || 1728 });
  }
  function check(ctx, x, y, s, col, a) {
    if (a <= 0) return; ctx.save(); ctx.globalAlpha *= a; ctx.strokeStyle = col; ctx.lineWidth = Math.max(2, s * 0.15); ctx.lineCap = 'round'; ctx.lineJoin = 'round';
    ctx.beginPath(); ctx.moveTo(x - s * 0.42, y + s * 0.02); ctx.lineTo(x - s * 0.12, y + s * 0.32); ctx.lineTo(x + s * 0.45, y - s * 0.36); ctx.stroke(); ctx.restore();
  }
  function cross(ctx, x, y, s, col, a) {
    if (a <= 0) return; ctx.save(); ctx.globalAlpha *= a; ctx.strokeStyle = col; ctx.lineWidth = Math.max(2, s * 0.15); ctx.lineCap = 'round';
    ctx.beginPath(); ctx.moveTo(x - s * 0.34, y - s * 0.34); ctx.lineTo(x + s * 0.34, y + s * 0.34); ctx.moveTo(x + s * 0.34, y - s * 0.34); ctx.lineTo(x - s * 0.34, y + s * 0.34); ctx.stroke(); ctx.restore();
  }
  // nabla drawn as a path (no bundled font has U+2207)
  function nabla(ctx, x, y, size, col, a) {
    const w = size * 0.62, h = size * 0.66, top = y + size * 0.2;
    ctx.save(); ctx.globalAlpha *= a; ctx.strokeStyle = col; ctx.lineWidth = Math.max(1.5, size * 0.07); ctx.lineJoin = 'miter';
    ctx.beginPath(); ctx.moveTo(x + size * 0.04, top); ctx.lineTo(x + size * 0.04 + w, top); ctx.lineTo(x + size * 0.04 + w / 2, top + h); ctx.closePath(); ctx.stroke(); ctx.restore();
    return size * 0.72;
  }
  // math line: parts = [{t, sub, nabla, color}], mono font; returns width. align: 'left'|'center'
  function mathLine(ctx, U, parts, x, y, o = {}) {
    const size = o.size || 30, col = o.color || U.C.ink, al = o.alpha == null ? 1 : o.alpha;
    const kind = o.kind || 'display';
    const wOf = (p) => p.sp != null ? p.sp : p.nabla ? size * 0.72 : U.measure(ctx, p.t, { size: p.sub ? size * 0.66 : size, kind: p.kind || kind }).w;
    const total = parts.reduce((s, p) => s + wOf(p), 0);
    let cx = o.align === 'center' ? x - total / 2 : x;
    parts.forEach(p => {
      const c = p.color || col;
      if (p.sp != null) { cx += p.sp; return; }
      if (p.nabla) { nabla(ctx, cx, y, size, c, al); cx += size * 0.72; return; }
      const sz = p.sub ? size * 0.66 : size;
      U.text(ctx, p.t, cx, y + (p.sub ? size * 0.42 : 0) + (p.dy || 0), { size: sz, kind: p.kind || kind, color: c, alpha: al });
      cx += wOf(p);
    });
    return total;
  }
  function pill(ctx, U, label, x, y, w, h, o = {}) {
    const al = o.alpha == null ? 1 : o.alpha; if (al <= 0) return;
    U.panel(ctx, x, y, w, h, { fill: o.fill || U.C.panel2, stroke: o.stroke || U.C.rule, r: h / 2, alpha: al, lw: o.lw || 2 });
    U.text(ctx, label, x + w / 2, y + h / 2 + 1, { size: o.size || 28, kind: o.kind || 'body', color: o.color || U.C.ink, align: 'center', baseline: 'middle', alpha: al, italic: o.italic });
  }
  // point at fraction f along a polyline
  function along(pts, f) {
    let total = 0; const L = [0];
    for (let i = 1; i < pts.length; i++) { total += Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]); L.push(total); }
    const lim = total * Math.max(0, Math.min(1, f));
    for (let i = 1; i < pts.length; i++) if (L[i] >= lim) { const g = (lim - L[i - 1]) / Math.max(1e-9, L[i] - L[i - 1]); return [pts[i - 1][0] + (pts[i][0] - pts[i - 1][0]) * g, pts[i - 1][1] + (pts[i][1] - pts[i - 1][1]) * g]; }
    return pts[pts.length - 1];
  }
  function glowDot(ctx, U, x, y, r, col, a) {
    if (a <= 0) return;
    ctx.save(); ctx.globalAlpha *= a;
    const g = ctx.createRadialGradient(x, y, 0, x, y, r * 3.2); g.addColorStop(0, U.rgba(col, 0.55)); g.addColorStop(1, U.rgba(col, 0));
    ctx.fillStyle = g; ctx.beginPath(); ctx.arc(x, y, r * 3.2, 0, Math.PI * 2); ctx.fill();
    ctx.restore();
    U.dot(ctx, x, y, r, col, { alpha: a });
  }

  // ========== Beat A: title card ==========
  function beatA(ctx, U, t) {
    const { C } = U;
    const al = U.fade(t, 0.2, A1, 0.9, 0.7); if (al <= 0) return;
    // soft dark wash behind the text so contour lines never fight the title
    const wash = U.cache('s01-wash', 1300, 1080, (g, w, h) => { const gr = g.createLinearGradient(0, 0, w, 0); gr.addColorStop(0, 'rgba(13,19,31,0.86)'); gr.addColorStop(0.62, 'rgba(13,19,31,0.6)'); gr.addColorStop(1, 'rgba(13,19,31,0)'); g.fillStyle = gr; g.fillRect(0, 0, w, h); });
    ctx.save(); ctx.globalAlpha *= al; ctx.drawImage(wash, 0, 0); ctx.restore();

    const x = 150, rise = (a) => (1 - U.easeOut(U.inv(a, a + 1.1, t))) * 18;
    const a1 = al * U.easeOut(U.inv(0.3, 1.2, t)), a2 = al * U.easeOut(U.inv(0.7, 1.7, t)), a3 = al * U.easeOut(U.inv(1.3, 2.3, t)), a4 = al * U.easeOut(U.inv(1.9, 2.9, t));
    U.text(ctx, 'ARXIV 2507.02092 · JULY 2025', x, 262 + rise(0.3), { size: 26, kind: 'mono', color: C.ebt, spacing: 3, alpha: a1 });
    U.text(ctx, 'Energy-Based', x, 312 + rise(0.7), { size: 128, kind: 'display', weight: 650, color: C.ink, alpha: a2, lh: 1.0 });
    U.text(ctx, 'Transformers', x, 440 + rise(0.8), { size: 128, kind: 'display', weight: 650, color: C.ink, alpha: a2, lh: 1.0 });
    U.text(ctx, 'are Scalable Learners and Thinkers', x, 596 + rise(1.3), { size: 60, kind: 'display', weight: 400, italic: true, color: C.ebt, alpha: a3 });
    // hairline grows
    const hw = 140 * U.ease(U.inv(1.6, 2.6, t));
    if (hw > 1) { ctx.save(); ctx.globalAlpha *= al; ctx.fillStyle = C.ebt; ctx.fillRect(x, 702, hw, 3); ctx.restore(); }
    U.text(ctx, 'A. Gladstone, G. Nanduru, M. M. Islam, P. Han, H. Ha, A. Chadha, Y. Du, H. Ji, J. Li, T. Iqbal', x, 736 + rise(1.9), { size: 27, color: C.muted, alpha: a4, maxWidth: 1240, lh: 1.35 });
    U.text(ctx, 'UVA · UIUC · Amazon GenAI · Stanford · Harvard', x, 790 + rise(2.1), { size: 23, kind: 'mono', color: C.faint, alpha: a4 * 0.95 });

    // marble: gradient descent on the background landscape, foreshadowing "thinking"
    const P = marblePath(), [ox, oy] = PAN(t);
    const ma = U.fade(t, 1.4, A1, 0.6, 0.8);
    if (ma > 0) {
      const s = U.clamp((t - 1.6) / 4.4) * (P.length - 1); // linear in GD steps: fast on steep slopes, slow near the minimum
      const i0 = Math.floor(s), f = s - i0, i1 = Math.min(P.length - 1, i0 + 1);
      const mx = U.lerp(P[i0][0], P[i1][0], f) + ox, my = U.lerp(P[i0][1], P[i1][1], f) + oy;
      const trail = P.slice(0, i0 + 1).map(p => [p[0] + ox, p[1] + oy]); trail.push([mx, my]);
      U.line(ctx, trail, { color: U.rgba(C.ebt, 0.55), width: 3, alpha: ma, dash: [2, 9] });
      // minimum marker
      const [bx, by] = [P[P.length - 1][0] + ox, P[P.length - 1][1] + oy];
      const ra = U.fade(t, 3.6, A1, 0.8, 0.7) * ma;
      if (ra > 0) {
        ctx.save(); ctx.globalAlpha *= ra; ctx.strokeStyle = U.rgba(C.ebt, 0.5); ctx.lineWidth = 2; ctx.setLineDash([6, 8]);
        ctx.beginPath(); ctx.arc(bx, by, 34 + 4 * Math.sin(t * 2.2), 0, Math.PI * 2); ctx.stroke(); ctx.restore();
        U.text(ctx, 'low energy', bx + 50, by + 30, { size: 24, kind: 'mono', color: C.muted, alpha: ra });
      }
      glowDot(ctx, U, mx, my, 11, C.ebt, ma);
      U.text(ctx, 'ŷ', mx + 18, my - 44, { size: 30, kind: 'mono', color: C.ebt, alpha: ma });
    }
  }

  // ========== Beat B: System 1 vs System 2 ==========
  function beatB(ctx, U, t) {
    const { C } = U;
    const al = U.fade(t, B0, B1, 0.6, 0.5); if (al <= 0) return;
    hdr(ctx, U, t, B0 + 0.1, B1, 'Why this paper', 'Thinking, fast and slow');

    const panels = [
      { x: 96, a: B0 + 0.5, acc: C.muted, eb: 'SYSTEM 1', ti: 'Thinking fast', de: 'quick · intuitive · automatic', q: '“What should I eat for lunch?”' },
      { x: 996, a: B0 + 2.4, acc: C.ebt, eb: 'SYSTEM 2', ti: 'Thinking slow', de: 'slow · deliberate · analytical', q: '“Should I change careers?”' },
    ];
    const py = 236, pw = 828, ph = 590;
    panels.forEach((p, idx) => {
      const pa = al * U.easeOut(U.inv(p.a, p.a + 0.8, t)); if (pa <= 0) return;
      const dy = (1 - U.easeOut(U.inv(p.a, p.a + 0.9, t))) * 20;
      const y = py + dy;
      U.panel(ctx, p.x, y, pw, ph, { fill: U.rgba(C.panel, 0.94), stroke: U.rgba(p.acc, idx ? 0.55 : 0.35), alpha: pa, r: 18 });
      U.text(ctx, p.eb, p.x + 44, y + 38, { size: 24, kind: 'mono', color: p.acc, spacing: 3, alpha: pa });
      U.text(ctx, p.ti, p.x + 44, y + 76, { size: 54, kind: 'display', weight: 600, color: C.ink, alpha: pa });
      U.text(ctx, p.de, p.x + 44, y + 152, { size: 30, color: C.muted, alpha: pa });
      U.text(ctx, p.q, p.x + 44, y + 214, { size: 32, italic: true, color: C.ink, alpha: pa * U.easeOut(U.inv(p.a + 0.4, p.a + 1.2, t)) });
    });

    // --- System 1 mechanism: one straight pass, repeated every 3 s ---
    {
      const p = panels[0], y = py, ma = al * U.easeOut(U.inv(p.a + 0.9, p.a + 1.6, t));
      if (ma > 0) {
        const cy = y + 400;
        pill(ctx, U, 'lunch?', p.x + 60, cy - 36, 190, 72, { alpha: ma, size: 30 });
        const x1 = p.x + 270, x2 = p.x + 540;
        U.arrow(ctx, x1, cy, x2, cy, { color: U.rgba(C.muted, 0.9), width: 3, alpha: ma, progress: U.seg(t, p.a + 1.2, p.a + 1.9) });
        U.text(ctx, 'one quick pass', (x1 + x2) / 2, cy - 58, { size: 26, color: C.muted, align: 'center', alpha: ma });
        const ansA = ma * U.easeOut(U.inv(p.a + 1.9, p.a + 2.4, t));
        pill(ctx, U, 'a sandwich', p.x + 560, cy - 36, 210, 72, { alpha: ansA, size: 30, stroke: U.rgba(C.muted, 0.8) });
        // a pulse that repeats: the answer is immediate every time
        const ph0 = t - (p.a + 1.2); if (ph0 > 0) { const u = (ph0 % 3.0) / 0.7; if (u < 1) glowDot(ctx, U, U.lerp(x1, x2 - 10, U.ease(u)), cy, 8, C.ink, ma * (1 - 0.3 * u)); }
        U.text(ctx, 'passes: 1', p.x + 415, y + 520, { size: 26, kind: 'mono', color: C.muted, align: 'center', alpha: ma });
      }
    }
    // --- System 2 mechanism: a propose -> check -> refine loop ---
    {
      const p = panels[1], y = py, ma = al * U.easeOut(U.inv(p.a + 0.9, p.a + 1.6, t));
      if (ma > 0) {
        const cy = y + 384, cx = p.x + 430, R = 92;
        pill(ctx, U, 'career?', p.x + 40, cy - 36, 160, 72, { alpha: ma, size: 30 });
        U.arrow(ctx, p.x + 212, cy, cx - R - 14, cy, { color: U.rgba(C.ebt, 0.8), width: 3, alpha: ma, progress: U.seg(t, p.a + 1.1, p.a + 1.6) });
        // ring
        const L0 = p.a + 1.6, per = 1.1, nLoops = 5, Lend = L0 + per * nLoops;
        ctx.save(); ctx.globalAlpha *= ma; ctx.strokeStyle = U.rgba(C.ebt, 0.45); ctx.lineWidth = 3; ctx.beginPath(); ctx.arc(cx, cy, R, 0, Math.PI * 2 * U.ease(U.inv(p.a + 1.2, p.a + 1.8, t))); ctx.stroke(); ctx.restore();
        const nodes = [['propose', -Math.PI / 2], ['check', Math.PI / 6], ['refine', 5 * Math.PI / 6]];
        const ang = -Math.PI / 2 + 2 * Math.PI * U.clamp((t - L0) / per, 0, nLoops);
        const running = t > L0 && t < Lend;
        nodes.forEach(([lab, th], k) => {
          const nx = cx + R * Math.cos(th), ny = cy + R * Math.sin(th);
          // light up node when the dot passes
          let d = ((ang - th) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI); const hot = running ? Math.max(0, 1 - d / 0.9) : 0;
          pill(ctx, U, lab, nx - 62, ny - 21, 124, 42, { alpha: ma * U.easeOut(U.inv(p.a + 1.3 + 0.15 * k, p.a + 1.8 + 0.15 * k, t)), size: 24, stroke: U.mix(C.rule, C.ebt, 0.35 + 0.65 * hot), fill: U.mix(C.panel2, '#3a3220', hot * 0.8), color: U.mix(C.muted, C.ink, 0.4 + 0.6 * hot) });
        });
        if (running) glowDot(ctx, U, cx + R * Math.cos(ang), cy + R * Math.sin(ang), 8, C.ebt, ma);
        const passes = Math.max(0, Math.min(nLoops, Math.floor((t - L0) / per) + (t > L0 ? 1 : 0)));
        U.text(ctx, 'passes: ' + (t > L0 ? passes : 0), cx, y + 520, { size: 26, kind: 'mono', color: C.ebt, align: 'center', alpha: ma });
        // decision
        const da = ma * U.easeOut(U.inv(Lend, Lend + 0.6, t));
        U.arrow(ctx, cx + R + 16, cy, p.x + 620, cy, { color: U.rgba(C.ebt, 0.8), width: 3, alpha: ma, progress: U.seg(t, Lend - 0.1, Lend + 0.4) });
        pill(ctx, U, 'decision', p.x + 630, cy - 36, 160, 72, { alpha: da, size: 28, stroke: U.rgba(C.ebt, 0.9) });
      }
    }
    U.caption(ctx, 'Hard choices deserve more thought. But a standard Transformer spends the same compute on every token.', t, B0 + 6.0, B1 - 0.2, { maxWidth: 1680 });
  }

  // ========== Beat C: today's System 2 recipes and their limits ==========
  function beatC(ctx, U, t) {
    const { C } = U;
    const al = U.fade(t, C0, C1, 0.6, 0.6); if (al <= 0) return;
    hdr(ctx, U, t, C0 + 0.1, C1, 'System 2 in AI today', 'Reasoning models think longer. Three catches.');
    const cw = 544, ch = 470, cy0 = 300, gap = 48;
    const cards = [
      { ti: 'Only checkable problems', bo: 'RL needs answers a rule can check, like math and code.', a: C0 + 0.9 },
      { ti: 'Extra supervision', bo: 'A separate verifier or reward model must be trained.', a: C0 + 2.4 },
      { ti: 'One modality', bo: 'Often works only in text.', a: C0 + 3.9 },
    ];
    cards.forEach((cd, i) => {
      const x = 96 + i * (cw + gap);
      const ca = al * U.easeOut(U.inv(cd.a, cd.a + 0.8, t)); if (ca <= 0) return;
      const y = cy0 + (1 - U.easeOut(U.inv(cd.a, cd.a + 0.9, t))) * 22;
      U.panel(ctx, x, y, cw, ch, { fill: U.rgba(C.panel, 0.95), stroke: C.rule, alpha: ca, r: 18 });
      const ix = x + cw / 2, iy = y + 120;
      if (i === 0) {
        [['math', 1], ['code', 1], ['writing', 0]].forEach(([w, ok], k) => {
          const ra = ca * U.easeOut(U.inv(cd.a + 0.3 + 0.25 * k, cd.a + 0.8 + 0.25 * k, t));
          U.text(ctx, w, ix - 30, iy - 58 + k * 48, { size: 30, kind: 'mono', color: ok ? C.ink : C.muted, align: 'right', alpha: ra });
          (ok ? check : cross)(ctx, ix + 30, iy - 40 + k * 48, 28, ok ? C.truth : C.bad, ra);
        });
      } else if (i === 1) {
        const bw = 176, bh = 74;
        pill(ctx, U, 'model', ix - bw - 36, iy - bh / 2, bw, bh, { alpha: ca, size: 28 });
        U.text(ctx, '+', ix, iy, { size: 44, kind: 'display', color: C.muted, align: 'center', baseline: 'middle', alpha: ca });
        const va = ca * U.easeOut(U.inv(cd.a + 0.5, cd.a + 1.1, t));
        U.panel(ctx, ix + 36, iy - bh / 2, bw, bh, { fill: U.rgba(C.bad, 0.08), stroke: U.rgba(C.bad, 0.75), dash: [8, 6], r: bh / 2, alpha: va });
        U.text(ctx, 'verifier', ix + 36 + bw / 2, iy + 1, { size: 28, color: C.ink, align: 'center', baseline: 'middle', alpha: va });
      } else {
        const sp = 150;
        // text glyph
        U.text(ctx, 'Aa', ix - sp, iy - 6, { size: 50, kind: 'display', weight: 600, color: C.ink, align: 'center', baseline: 'middle', alpha: ca });
        check(ctx, ix - sp, iy + 52, 24, C.truth, ca);
        // image glyph
        ctx.save(); ctx.globalAlpha *= ca * 0.9; ctx.strokeStyle = C.muted; ctx.lineWidth = 3; ctx.lineJoin = 'round';
        ctx.strokeRect(ix - 40, iy - 36, 80, 60); ctx.beginPath(); ctx.moveTo(ix - 34, iy + 18); ctx.lineTo(ix - 10, iy - 8); ctx.lineTo(ix + 6, iy + 8); ctx.lineTo(ix + 18, iy - 2); ctx.lineTo(ix + 34, iy + 18); ctx.stroke();
        ctx.beginPath(); ctx.arc(ix + 20, iy - 20, 6, 0, Math.PI * 2); ctx.stroke();
        // video glyph
        const vx = ix + sp; ctx.strokeRect(vx - 40, iy - 36, 80, 60);
        for (let k = 0; k < 4; k++) { ctx.strokeRect(vx - 33 + k * 19, iy - 31, 8, 7); ctx.strokeRect(vx - 33 + k * 19, iy + 12, 8, 7); }
        ctx.restore();
        const xa = ca * U.easeOut(U.inv(cd.a + 0.4, cd.a + 0.9, t));
        cross(ctx, ix, iy + 52, 24, C.bad, xa); cross(ctx, vx, iy + 52, 24, C.bad, xa);
      }
      U.text(ctx, cd.ti, x + 40, y + 226, { size: 38, kind: 'display', weight: 600, color: C.ink, alpha: ca, maxWidth: cw - 80 });
      U.text(ctx, cd.bo, x + 40, y + 290, { size: 30, color: C.muted, alpha: ca, maxWidth: cw - 80, lh: 1.32 });
      // soft "limited" stamp
      const sa = C0 + 5.6 + 0.45 * i, sp2 = U.inv(sa, sa + 0.5, t);
      if (sp2 > 0) {
        const s = 1.35 - 0.35 * U.easeOut(sp2);
        ctx.save(); ctx.translate(x + cw - 128, y + ch - 62); ctx.rotate(-0.14); ctx.scale(s, s); ctx.globalAlpha *= ca * U.easeOut(sp2) * 0.9;
        U.panel(ctx, -86, -25, 172, 50, { fill: U.rgba(C.bad, 0.1), stroke: U.rgba(C.bad, 0.85), r: 8, lw: 2.5 });
        U.text(ctx, 'LIMITED', 0, 1, { size: 26, kind: 'mono', weight: 700, color: C.bad, align: 'center', baseline: 'middle', spacing: 4 });
        ctx.restore();
      }
    });
    U.badge(ctx, 'paper', 'Abstract, p.1–2', 96, cy0 + ch + 34, { alpha: al * U.easeOut(U.inv(C0 + 1.2, C0 + 2, t)) });
  }

  // ========== Beat D: the core question ==========
  function beatD(ctx, U, t) {
    const { C } = U;
    const al = U.fade(t, D0, D1, 0.7, 0.6); if (al <= 0) return;
    const cx = 960;
    const a1 = al * U.easeOut(U.inv(D0 + 0.2, D0 + 1.0, t)), a2 = al * U.easeOut(U.inv(D0 + 0.5, D0 + 1.4, t)), a3 = al * U.easeOut(U.inv(D0 + 0.9, D0 + 1.8, t));
    U.text(ctx, "THE PAPER'S CORE QUESTION", cx, 330 - 10 * (1 - a1), { size: 26, kind: 'mono', color: C.ebt, align: 'center', spacing: 4, alpha: a1 });
    const hl = U.ease(U.inv(D0 + 1.9, D0 + 2.7, t));
    const hcol = U.mix(C.ink, C.ebt, hl);
    const size = 70, opt = { size, kind: 'display', weight: 500, align: 'center' };
    const L1 = [{ t: '“Can we rely entirely on ' }, { t: 'unsupervised learning', color: hcol }];
    const L2 = [{ t: 'to develop ' }, { t: 'System 2 Thinking', color: hcol }, { t: '?”' }];
    const y1 = 420 + 14 * (1 - a2 / Math.max(al, 1e-6)), y2 = 520 + 14 * (1 - a3 / Math.max(al, 1e-6));
    const w1 = U.rich(ctx, L1, cx, y1, { ...opt, alpha: a2 });
    const w2 = U.rich(ctx, L2, cx, y2, { ...opt, alpha: a3 });
    // underline sweeps under the highlighted phrases
    ctx.save(); ctx.font = U.font(size, 'display', 500, false);
    const pre1 = ctx.measureText(L1[0].t).width, ph1 = ctx.measureText(L1[1].t).width;
    const pre2 = ctx.measureText(L2[0].t).width, ph2 = ctx.measureText(L2[1].t).width;
    ctx.restore();
    const ul = U.ease(U.inv(D0 + 2.1, D0 + 3.0, t));
    if (ul > 0) {
      ctx.save(); ctx.globalAlpha *= al; ctx.fillStyle = C.ebt;
      ctx.fillRect(cx - w1 / 2 + pre1, y1 + size * 1.17, ph1 * ul, 3);
      ctx.fillRect(cx - w2 / 2 + pre2, y2 + size * 1.17, ph2 * ul, 3);
      ctx.restore();
    }
    const a4 = al * U.easeOut(U.inv(D0 + 2.6, D0 + 3.4, t));
    U.text(ctx, 'Any problem, any modality, no external supervision.', cx, 676, { size: 36, color: C.muted, align: 'center', alpha: a4 });
    const bw = U.measure(ctx, 'FROM THE PAPER · p.2', { size: 18, kind: 'mono', spacing: 1 }).w + 28;
    U.badge(ctx, 'paper', 'p.2', cx - bw / 2, 760, { alpha: a4 });
  }

  // ========== Beat E: the paper's answer ==========
  const TOK = ['fox', 'dog', 'cat', 'car', 'sky'];
  const Z0 = [-0.25, 0.55, -0.45, 0.35, 0.05];
  const ZT = [3.4, 0.7, 0.2, -0.6, -0.9];
  const probs = (s) => { const f = 1 - Math.pow(0.55, s); const z = Z0.map((v, i) => v + (ZT[i] - v) * f); const m = Math.max(...z); const e = z.map(v => Math.exp(v - m)); const S = e.reduce((a, b) => a + b, 0); return e.map(v => v / S); };
  const energy = (s) => 0.42 + 4.18 * Math.pow(0.55, s);
  const IT0 = E0 + 3.7, PER = 1.45, NIT = 6;

  function beatE(ctx, U, t) {
    const { C } = U;
    const al = U.fade(t, E0, E1, 0.6, 0.5); if (al <= 0) return;
    hdr(ctx, U, t, E0 + 0.1, E1, "The paper's answer", [{ t: 'Learn to ' }, { t: 'verify', color: C.ebt }, { t: '. Then think by ' }, { t: 'optimizing', color: C.ebt }, { t: ' against the verifier.' }], { size: 50 });

    // iteration state
    const u = (t - IT0) / PER; // iterations completed (fractional)
    const k = Math.max(0, Math.min(NIT, Math.floor(u)));
    const ph = u - Math.floor(u); const inIt = u >= 0 && u < NIT;
    // step shown on the candidate: morphs during last 30% of each iteration
    let sCand = k; if (inIt) sCand = k + U.ease(U.inv(0.7, 1.0, ph)); if (u >= NIT) sCand = NIT;
    const evalIdx = u < 0 ? -1 : (inIt ? (ph >= 0.42 ? k : k - 1) : NIT - 1); // energy of ŷ_k shown after the forward pass of iteration k

    const ba = (a) => al * U.easeOut(U.inv(a, a + 0.6, t));
    // --- context chip ---
    const xa = ba(E0 + 0.5);
    U.text(ctx, 'context x', 120, 290, { size: 24, kind: 'mono', color: C.muted, alpha: xa });
    U.panel(ctx, 120, 326, 340, 88, { fill: C.panel2, stroke: C.rule, alpha: xa, r: 14 });
    U.text(ctx, 'The quick brown', 145, 370, { size: 30, kind: 'mono', color: C.ink, baseline: 'middle', alpha: xa });
    // --- candidate panel ---
    const ya = ba(E0 + 0.8);
    U.text(ctx, 'candidate ŷ', 120, 460, { size: 24, kind: 'mono', color: C.ebt, alpha: ya });
    U.panel(ctx, 120, 496, 340, 300, { fill: C.panel2, stroke: U.rgba(C.ebt, 0.45), alpha: ya, r: 14 });
    const stepLab = u < 0 ? 'random start' : 'step ' + Math.min(NIT, Math.round(sCand >= k + 0.999 ? k + 1 : k));
    U.text(ctx, stepLab, 440, 512, { size: 22, kind: 'mono', color: C.muted, align: 'right', alpha: ya });
    const pr = probs(sCand);
    TOK.forEach((w, i) => {
      const by = 560 + i * 46;
      U.text(ctx, w, 146, by, { size: 26, kind: 'mono', color: i === 0 ? C.ink : C.muted, baseline: 'middle', alpha: ya });
      ctx.save(); ctx.globalAlpha *= ya; ctx.fillStyle = U.rgba(C.rule, 0.8); ctx.fillRect(220, by - 12, 210, 24);
      ctx.fillStyle = i === 0 ? C.ebt : U.rgba(C.ebt, 0.5); ctx.fillRect(220, by - 12, 210 * pr[i], 24); ctx.restore();
    });
    // --- EBT box ---
    const bx = 580, by0 = 420, bw = 230, bh = 250, bcx = bx + bw / 2, bcy = by0 + bh / 2;
    const ea = ba(E0 + 1.3);
    const glow = inIt ? Math.max(0, 1 - Math.abs(ph - 0.3) / 0.18) : 0;
    U.arrow(ctx, 462, 370, bx - 6, bcy - 50, { color: U.rgba(C.muted, 0.85), width: 3, alpha: al, progress: U.seg(t, E0 + 1.0, E0 + 1.5) });
    U.arrow(ctx, 462, 646, bx - 6, bcy + 50, { color: U.rgba(C.ebt, 0.85), width: 3, alpha: al, progress: U.seg(t, E0 + 1.1, E0 + 1.6) });
    U.panel(ctx, bx, by0, bw, bh, { fill: U.mix(C.panel, '#3a3220', 0.35 + 0.65 * glow), stroke: U.rgba(C.ebt, 0.7 + 0.3 * glow), alpha: ea, r: 18, lw: 2.5 });
    U.text(ctx, 'EBT', bcx, bcy - 40, { size: 60, kind: 'display', weight: 700, color: C.ebt, align: 'center', baseline: 'middle', alpha: ea });
    mathLine(ctx, U, [{ t: 'E' }, { t: 'θ', sub: true }, { sp: 2 }, { t: '(x, ŷ)' }], bcx, bcy + 14, { size: 38, color: C.ink, alpha: ea, align: 'center' });
    U.text(ctx, 'verifier', bcx, bcy + 82, { size: 24, kind: 'mono', color: C.muted, align: 'center', alpha: ea });
    // forward pulses
    if (inIt && ph < 0.42) {
      const f = U.ease(U.inv(0.0, 0.3, ph));
      if (ph < 0.3) { glowDot(ctx, U, U.lerp(462, bx - 6, f), U.lerp(370, bcy - 50, f), 7, C.ink, al); glowDot(ctx, U, U.lerp(462, bx - 6, f), U.lerp(646, bcy + 50, f), 7, C.ebt, al); }
      else { const g = U.ease(U.inv(0.3, 0.42, ph)); glowDot(ctx, U, U.lerp(bx + bw + 4, 858, g), bcy, 7, C.ebt, al); }
    }
    // --- energy readout ---
    const ra = ba(E0 + 1.9);
    U.arrow(ctx, bx + bw + 4, bcy, 866, bcy, { color: U.rgba(C.ebt, 0.85), width: 3, alpha: al, progress: U.seg(t, E0 + 1.8, E0 + 2.2) });
    U.text(ctx, 'energy', 884, 410, { size: 24, kind: 'mono', color: C.muted, alpha: ra });
    const Ev = evalIdx >= 0 ? energy(evalIdx) : null;
    U.text(ctx, Ev == null ? '?' : Ev.toFixed(2), 884, bcy, { size: 72, kind: 'display', weight: 600, color: Ev == null ? C.faint : C.ink, baseline: 'middle', alpha: ra });
    // gauge
    const gv = Ev == null ? 0 : Ev / 5;
    ctx.save(); ctx.globalAlpha *= ra; ctx.fillStyle = C.rule; ctx.fillRect(884, 600, 190, 12); ctx.fillStyle = U.cmapCss(0.25 + 0.75 * gv); ctx.fillRect(884, 600, 190 * gv, 12); ctx.restore();
    U.text(ctx, 'low = good fit', 884, 622, { size: 22, kind: 'mono', color: C.faint, alpha: ra });
    // sparkline of energy per step
    if (evalIdx >= 0) {
      const sx = 884, sy = 680, sw = 190, sh = 110;
      ctx.save(); ctx.globalAlpha *= ra; ctx.strokeStyle = C.faint; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(sx, sy); ctx.lineTo(sx, sy + sh); ctx.lineTo(sx + sw, sy + sh); ctx.stroke(); ctx.restore();
      const pts = []; for (let j = 0; j <= evalIdx; j++) pts.push([sx + 8 + j * (sw - 16) / NIT, sy + sh - (energy(j) / 5) * sh]);
      U.line(ctx, pts, { color: C.ebt, width: 2.5, alpha: ra });
      pts.forEach((p, j) => U.dot(ctx, p[0], p[1], j === evalIdx ? 6 : 4, C.ebt, { alpha: ra }));
      U.text(ctx, 'step', sx + sw, sy + sh + 8, { size: 22, kind: 'mono', color: C.faint, align: 'right', alpha: ra });
    }
    // --- the loop: gradient step back to the candidate ---
    const loop = [[bcx, by0 + bh + 4], [bcx, 880], [290, 880], [290, 802]];
    const la = U.seg(t, E0 + 2.5, E0 + 3.3);
    U.line(ctx, loop, { color: U.rgba(C.ebt, 0.8), width: 3, progress: la, alpha: al });
    if (la >= 1) U.arrow(ctx, 290, 830, 290, 802, { color: U.rgba(C.ebt, 0.8), width: 3, alpha: al, head: 16 });
    const rla = ba(E0 + 3.0);
    mathLine(ctx, U, [{ t: 'ŷ' }, { sp: 14 }, { t: '←', kind: 'mono', dy: 2 }, { sp: 14 }, { t: 'ŷ' }, { sp: 12 }, { t: '−' }, { sp: 12 }, { t: 'α' }, { sp: 3 }, { nabla: true }, { t: 'ŷ', sub: true }, { t: 'E' }], (290 + bcx) / 2, 896, { size: 40, color: C.ebt, alpha: rla, align: 'center' });
    U.text(ctx, 'each gradient step is one step of thinking', (290 + bcx) / 2, 958, { size: 24, color: C.muted, align: 'center', alpha: rla });
    if (inIt && ph >= 0.42) { const g = U.ease(U.inv(0.42, 0.95, ph)); const [qx, qy] = along(loop, g); glowDot(ctx, U, qx, qy, 7, C.ebt, al); }

    // --- headline results ---
    const chips = [
      { pre: 'UP TO', n: '35%', tx: 'higher pretraining scaling rate than Transformer++', b: 'Abstract', a: E0 + 6.2 },
      { pre: 'UP TO', n: '29%', tx: 'more gain from thinking than Transformer++ on language', b: 'Fig 6a', a: E0 + 7.4 },
      { pre: '', n: '99%', tx: 'fewer forward passes than DiT, with better image denoising', b: 'Table 4', a: E0 + 8.6 },
    ];
    const cx0 = 1150, cwid = 674, chh = 172;
    chips.forEach((c, i) => {
      const ca = al * U.easeOut(U.inv(c.a, c.a + 0.7, t)); if (ca <= 0) return;
      const x = cx0 + (1 - U.easeOut(U.inv(c.a, c.a + 0.8, t))) * 40, y = 300 + i * (chh + 24);
      U.panel(ctx, x, y, cwid, chh, { fill: U.rgba(C.panel, 0.95), stroke: U.rgba(C.ebt, 0.35), alpha: ca, r: 16 });
      if (c.pre) U.text(ctx, c.pre, x + 28, y + 26, { size: 22, kind: 'mono', color: C.muted, spacing: 2, alpha: ca });
      U.text(ctx, c.n, x + 26, y + 54, { size: 76, kind: 'display', weight: 700, color: C.ebt, alpha: ca });
      U.text(ctx, c.tx, x + 206, y + 26, { size: 30, color: C.ink, alpha: ca, maxWidth: cwid - 236, lh: 1.25 });
      U.badge(ctx, 'paper', c.b, x + 206, y + chh - 56, { alpha: ca });
    });
    U.text(ctx, 'schematic · numbers in the loop are illustrative', 120, 1010, { size: 22, kind: 'mono', color: C.faint, alpha: ba(E0 + 3.4) });
  }

  // ========== Beat F: roadmap ==========
  const RM = ['Model families', 'Energy', 'Data', 'Thinking', 'Training', 'Evaluation', 'Scaling', 'Takeaways'];
  function beatF(ctx, U, t) {
    const { C } = U;
    const al = U.fade(t, F0, F1, 0.5, 0.6); if (al <= 0) return;
    hdr(ctx, U, t, F0 + 0.05, F1, 'Roadmap', 'Next: how it works, piece by piece');
    const w = 390, h = 124, gap = 56;
    RM.forEach((name, i) => {
      const r = Math.floor(i / 4), c = i % 4;
      const x = 96 + c * (w + gap), y = 400 + r * (h + 48);
      const a0 = F0 + 0.3 + 0.09 * i, ca = al * U.easeOut(U.inv(a0, a0 + 0.45, t)); if (ca <= 0) return;
      const dy = (1 - U.easeOut(U.inv(a0, a0 + 0.6, t))) * 16;
      // highlight sweep
      const hs = F0 + 1.5 + 0.21 * i, hot = Math.max(0, 1 - Math.abs(t - hs) / 0.35);
      U.panel(ctx, x, y + dy, w, h, { fill: U.mix(C.panel, '#3a3220', hot * 0.6), stroke: U.mix(C.rule, C.ebt, 0.25 + 0.75 * hot), alpha: ca, r: 16 });
      U.text(ctx, String(i + 2).padStart(2, '0'), x + 30, y + dy + h / 2, { size: 28, kind: 'mono', color: C.ebt, baseline: 'middle', alpha: ca });
      U.text(ctx, name, x + 92, y + dy + h / 2 + 2, { size: 38, kind: 'display', weight: 600, color: C.ink, baseline: 'middle', alpha: ca });
    });
  }

  EBTV.scene({
    id: 's01_intro',
    title: 'Thinking, fast and slow',
    dur: DUR,
    assets: [],
    draw(ctx, t, U) {
      drawField(ctx, t, U, fieldAlpha(t, U));
      beatA(ctx, U, t);
      beatB(ctx, U, t);
      beatC(ctx, U, t);
      beatD(ctx, U, t);
      beatE(ctx, U, t);
      beatF(ctx, U, t);
    },
  });
})();
