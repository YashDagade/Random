// s02_families: "Four ways to make a prediction"
// AR Transformer, RNN, Diffusion Transformer and EBT fill the same blank; then Table 1 (three facets);
// then the takeaway: diffusion learns the gradient of an energy, EBT learns the energy itself.
// Sources: paper Fig 1 + caption (p.2), Intro p.2, Table 1 (p.3), Fig 2 (p.4), Sec 6.2, 6.4 (p.16), App. E.2 (p.37).
(function () {
  const DUR = 75;
  const PY0 = 270, PY1 = 846;                 // lane panel top / bottom
  const LANES = [
    { key: 'ar',   name: 'AR Transformer',        row: 'FF Transformers',        sub: 'feed-forward',    x: 90,   w: 380, t0: 6.0,  t1: 13.6 },
    { key: 'rnn',  name: 'RNN',                   row: 'RNNs',                   sub: 'recurrent state', x: 490,  w: 380, t0: 13.6, t1: 21.6 },
    { key: 'diff', name: 'Diffusion Transformer', row: 'Diffusion Transformers', sub: 'denoising',       x: 890,  w: 440, t0: 21.6, t1: 30.0 },
    { key: 'ebt',  name: 'EBT',                   row: 'EBTs',                   sub: 'energy-based',    x: 1350, w: 480, t0: 30.0, t1: 42.4 },
  ];
  const T_LANES_OUT = 42.4, T_ALL_BRIGHT = 38.6;

  // ---------- table (paper Table 1) ----------
  const TB = { x: 140, labelW: 470, colW: 390, headY: 252, rowY: 392, rowH: 96 };
  const FACETS = [
    { tag: 'FACET 1', title: 'Dynamic compute allocation' },
    { tag: 'FACET 2', title: 'Modeling uncertainty (continuous spaces)' },
    { tag: 'FACET 3', title: 'Prediction verification' },
  ];
  const TABLE = [[0, 0, 0], [0, 0, 0], [1, 0, 0], [1, 1, 1]];
  const REASON = { '2,0': 'more denoising steps', '3,0': 'more gradient steps', '3,1': 'unnormalized likelihood', '3,2': 'energy scores the guess' };
  const FW = [[44.3, 49.3], [49.3, 54.3], [54.3, 59.3]];
  const T_SUM = 59.3, T_TABLE_OUT = 62.7;

  // ---------- takeaway (illustrative 2D landscape) ----------
  const TK = { y: 226, h: 420, w: 790, lx: 135, rx: 995, t0: 63.0 };
  const UMAX = 2.56;
  function Efun(u, v) { return 0.06 * u * u + 0.35 * v * v + 1.0 - 0.9 * Math.exp(-((u - 1.1) ** 2 / 0.55 + (v + 0.15) ** 2 / 0.3)) - 0.55 * Math.exp(-((u + 1.3) ** 2 / 0.4 + (v - 0.25) ** 2 / 0.25)); }
  function Egrad(u, v) { const h = 1e-4; return [(Efun(u + h, v) - Efun(u - h, v)) / (2 * h), (Efun(u, v + h) - Efun(u, v - h)) / (2 * h)]; }
  function gdPath(u, v, n, a) { const P = [[u, v, Efun(u, v)]]; for (let i = 0; i < n; i++) { const g = Egrad(u, v); u -= a * g[0]; v -= a * g[1]; P.push([u, v, Efun(u, v)]); } return P; }
  let RUNS = null;
  function runs() {
    if (!RUNS) RUNS = [
      { t0: 64.1, t1: 67.3, starts: [[-2.3, -0.75], [2.4, 0.8]] },
      { t0: 69.6, t1: 72.8, starts: [[-0.3, 0.85], [0.0, 0.9]] },
    ].map(r => ({ ...r, paths: r.starts.map(s => gdPath(s[0], s[1], 80, 0.12)) }));
    return RUNS;
  }

  // ---------- small drawing helpers ----------
  function chip(ctx, U, x, y, w, h, label, o = {}) {
    U.panel(ctx, x, y, w, h, { fill: o.fill || U.C.panel2, stroke: o.stroke || U.C.rule, lw: o.lw || 2, r: o.r == null ? 10 : o.r, alpha: o.alpha, dash: o.dash });
    if (label) U.text(ctx, label, x + w / 2, y + h / 2 + 1, { size: o.size || 24, kind: o.kind || 'body', weight: o.weight || 400, italic: o.italic, color: o.color || U.C.ink, align: 'center', baseline: 'middle', alpha: o.alpha });
  }
  function glow(ctx, U, x, y, color, a = 1, r = 6) {
    if (a <= 0) return;
    U.dot(ctx, x, y, r * 3.4, U.rgba(color, 0.10), { alpha: a });
    U.dot(ctx, x, y, r * 2.0, U.rgba(color, 0.28), { alpha: a });
    U.dot(ctx, x, y, r, color, { alpha: a });
  }
  // polyline with draw-on progress and an arrow head at the drawn tip
  function polyArrow(ctx, U, pts, o = {}) {
    const p = o.progress == null ? 1 : U.clamp(o.progress); if (p <= 0) return;
    U.line(ctx, pts, { color: o.color, width: o.width || 3, progress: p, alpha: o.alpha, dash: o.dash });
    const tip = along(pts, p), prev = along(pts, Math.max(0, p - 0.02));
    const a = Math.atan2(tip[1] - prev[1], tip[0] - prev[0]), hs = o.head || 12;
    ctx.save(); ctx.globalAlpha *= (o.alpha == null ? 1 : o.alpha); ctx.fillStyle = o.color || U.C.muted;
    ctx.beginPath(); ctx.moveTo(tip[0], tip[1]); ctx.lineTo(tip[0] - hs * Math.cos(a - 0.45), tip[1] - hs * Math.sin(a - 0.45)); ctx.lineTo(tip[0] - hs * Math.cos(a + 0.45), tip[1] - hs * Math.sin(a + 0.45)); ctx.closePath(); ctx.fill(); ctx.restore();
  }
  function along(pts, f) {
    let total = 0; const L = [0];
    for (let i = 1; i < pts.length; i++) { total += Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]); L.push(total); }
    const lim = total * Math.max(0, Math.min(1, f));
    for (let i = 1; i < pts.length; i++) if (L[i] >= lim) { const g = (lim - L[i - 1]) / Math.max(1e-9, L[i] - L[i - 1]); return [pts[i - 1][0] + (pts[i][0] - pts[i - 1][0]) * g, pts[i - 1][1] + (pts[i][1] - pts[i - 1][1]) * g]; }
    return pts[pts.length - 1].slice();
  }
  function check(ctx, U, cx, cy, s, color, p, a = 1) {
    U.line(ctx, [[cx - s * 0.5, cy + s * 0.02], [cx - s * 0.14, cy + s * 0.38], [cx + s * 0.55, cy - s * 0.4]], { color, width: Math.max(3, s * 0.15), progress: p, alpha: a });
  }
  function cross(ctx, U, cx, cy, s, color, p, a = 1) {
    if (p <= 0) return;
    U.line(ctx, [[cx - s * 0.36, cy - s * 0.36], [cx + s * 0.36, cy + s * 0.36]], { color, width: Math.max(3, s * 0.12), progress: U.clamp(p * 2), alpha: a });
    U.line(ctx, [[cx + s * 0.36, cy - s * 0.36], [cx - s * 0.36, cy + s * 0.36]], { color, width: Math.max(3, s * 0.12), progress: U.clamp(p * 2 - 1), alpha: a });
  }
  function softmax(z) { const m = Math.max(...z); const e = z.map(v => Math.exp(v - m)); const s = e.reduce((a, b) => a + b, 0); return e.map(v => v / s); }
  function counter(ctx, U, L, value, qual, color, a, qa) {
    const cx = L.x + L.w / 2;
    U.text(ctx, 'forward passes for this token', cx, 722, { size: 22, color: U.C.muted, align: 'center', alpha: a });
    U.text(ctx, value, cx, 752, { size: 44, kind: 'mono', weight: 600, color, align: 'center', alpha: a });
    if (qual && qa > 0) U.text(ctx, qual, cx, 808, { size: 22, color: U.C.muted, align: 'center', alpha: a * qa });
  }

  // ---------- lanes ----------
  function laneAlpha(U, i, t) {
    const L = LANES[i];
    let a = U.easeOut(U.inv(L.t0, L.t0 + 0.7, t));
    const next = LANES[i + 1];
    if (next) a *= 1 - 0.5 * U.seg(t, next.t0, next.t0 + 0.6) * (1 - U.seg(t, T_ALL_BRIGHT, T_ALL_BRIGHT + 0.9));
    return a * (1 - U.seg(t, T_LANES_OUT, T_LANES_OUT + 0.8));
  }
  function lanePanel(ctx, U, L, t, col) {
    const active = (t >= L.t0 && t < L.t1) ? 1 : 0;
    const on = Math.max(U.fade(t, L.t0, L.t1, 0.6, 0.6), 0.6 * U.seg(t, T_ALL_BRIGHT, T_ALL_BRIGHT + 0.9));
    const breathe = 0.5 + 0.12 * Math.sin(t * 2.2) * active;
    U.panel(ctx, L.x, PY0, L.w, PY1 - PY0, { fill: U.C.panel, stroke: U.mix(U.C.rule, col, on * breathe * 1.4), lw: 2, r: 18 });
    // lane name (hidden while it flies to the table)
    const fly = U.seg(t, 42.6, 44.0);
    if (fly <= 0) {
      U.text(ctx, L.name, L.x + L.w / 2, PY0 + 18, { size: 32, weight: 700, color: col, align: 'center' });
    }
    U.text(ctx, L.sub, L.x + L.w / 2, PY0 + 62, { size: 22, color: U.C.muted, align: 'center' });
  }

  function drawAR(ctx, U, L, t) {
    const { C } = U; const u = t - L.t0, col = C.ar;
    const cx = L.x + L.w / 2, sx = cx - 18;            // stack center (leave room for the bracket)
    const chipY = 392, layerY = [472, 514, 556, 598], lh = 34, outY = 660;
    chip(ctx, U, sx - 112, chipY, 224, 48, 'The quick brown', { alpha: U.seg(u, 0.3, 0.8) });
    U.arrow(ctx, sx, chipY + 48, sx, layerY[0] - 4, { color: C.faint, progress: U.seg(u, 0.6, 1.0), head: 10, width: 2.5 });
    const pulseF = U.seg(u, 2.2, 4.0, (x) => x);         // linear travel
    const pulseY = U.lerp(chipY + 48, outY, pulseF);
    layerY.forEach((y, k) => {
      const a = U.seg(u, 0.8 + k * 0.16, 1.2 + k * 0.16);
      const lit = (u > 2.2 && u < 4.4) ? Math.max(0, 1 - Math.abs(pulseY - (y + lh / 2)) / 46) : 0;
      U.panel(ctx, sx - 100, y, 200, lh, { fill: U.rgba(col, 0.10 + 0.35 * lit), stroke: U.rgba(col, 0.55 + 0.45 * lit), lw: 2, r: 8, alpha: a });
      U.text(ctx, 'layer ' + (k + 1), sx, y + lh / 2 + 1, { size: 22, color: lit > 0.3 ? C.ink : C.muted, align: 'center', baseline: 'middle', alpha: a });
    });
    // bracket "fixed depth"
    const ba = U.seg(u, 1.6, 2.2);
    if (ba > 0) {
      const bx = sx + 112, y0 = layerY[0], y1 = layerY[3] + lh;
      U.line(ctx, [[bx, y0], [bx + 8, y0], [bx + 8, y1], [bx, y1]], { color: C.muted, width: 2, alpha: ba });
      U.text(ctx, 'fixed', bx + 18, (y0 + y1) / 2 - 26, { size: 22, color: C.muted, alpha: ba });
      U.text(ctx, 'depth', bx + 18, (y0 + y1) / 2 + 2, { size: 22, color: C.muted, alpha: ba });
    }
    U.arrow(ctx, sx, layerY[3] + lh + 2, sx, outY - 4, { color: C.faint, progress: U.seg(u, 1.4, 1.8), head: 10, width: 2.5 });
    if (u > 2.2 && u < 4.1) glow(ctx, U, sx, pulseY, col, U.fade(u, 2.2, 4.1, 0.15, 0.15));
    const oa = U.seg(u, 3.9, 4.4);
    const pop = 1 + 0.12 * Math.sin(Math.PI * U.clamp((u - 3.9) / 0.5));
    if (oa > 0) {
      const ow = 110 * pop, oh = 48 * pop;
      chip(ctx, U, sx - ow / 2, outY + 24 - oh / 2, ow, oh, 'fox', { alpha: oa, stroke: col, fill: U.rgba(col, 0.16), weight: 700, size: 26 });
    }
    counter(ctx, U, L, u >= 4.0 ? '1' : '0', 'always one, easy or hard', col, U.seg(u, 0.8, 1.3), U.seg(u, 4.6, 5.2));
  }

  function drawRNN(ctx, U, L, t) {
    const { C } = U; const u = t - L.t0, col = C.rnn;
    const toks = ['The', 'quick', 'brown'], rowY = [384, 450, 516], rh = 44;
    const tx = L.x + 18, tw = 108, cxL = L.x + 152, cw = 150, ccx = cxL + cw / 2;
    toks.forEach((s, i) => {
      const ti = 0.6 + i * 1.1;
      const a = U.seg(u, ti, ti + 0.4), slide = (1 - U.easeOut(U.inv(ti, ti + 0.5, u))) * -24;
      chip(ctx, U, tx + slide, rowY[i], tw, rh, s, { alpha: a, size: 24 });
      U.arrow(ctx, tx + tw + 4, rowY[i] + rh / 2, cxL - 4, rowY[i] + rh / 2, { color: C.faint, progress: U.seg(u, ti + 0.25, ti + 0.55), head: 9, width: 2.5 });
      const ca = U.seg(u, ti + 0.1, ti + 0.5);
      const hit = U.fade(u, ti + 0.5, ti + 1.3, 0.1, 0.6);
      U.panel(ctx, cxL, rowY[i], cw, rh, { fill: U.rgba(col, 0.10 + 0.3 * hit), stroke: U.rgba(col, 0.6 + 0.4 * hit), r: 9, alpha: ca });
      U.text(ctx, 'RNN', cxL + 18, rowY[i] + rh / 2 + 1, { size: 24, weight: 700, color: C.ink, baseline: 'middle', alpha: ca });
      // hidden-state dot: brightness = how much it has absorbed
      const hv = U.seg(u, ti + 0.5, ti + 0.9);
      U.dot(ctx, cxL + cw - 26, rowY[i] + rh / 2, 11, U.mix(C.panel2, col, 0.25 + 0.75 * hv), { alpha: ca, stroke: col, lw: 1.5 });
      if (i > 0) {
        U.arrow(ctx, ccx, rowY[i - 1] + rh + 2, ccx, rowY[i] - 3, { color: col, progress: U.seg(u, ti + 0.3, ti + 0.6), head: 9, width: 2.5 });
        if (i === 1) U.text(ctx, 'h', ccx + 12, rowY[0] + rh + 1, { size: 22, kind: 'mono', color: col, alpha: U.seg(u, ti + 0.5, ti + 0.9) });
        if (u > ti + 0.3 && u < ti + 0.75) glow(ctx, U, ccx, U.lerp(rowY[i - 1] + rh, rowY[i], U.inv(ti + 0.3, ti + 0.7, u)), col, 1, 5);
      }
    });
    // output
    const outY = 600;
    U.arrow(ctx, ccx, rowY[2] + rh + 2, ccx, outY - 4, { color: C.faint, progress: U.seg(u, 3.8, 4.2), head: 10, width: 2.5 });
    const oa = U.seg(u, 4.1, 4.6);
    if (oa > 0) chip(ctx, U, ccx - 55, outY, 110, 48, 'fox', { alpha: oa, stroke: col, fill: U.rgba(col, 0.16), weight: 700, size: 26 });
    // try to think longer: self loop on the last cell, crossed out
    const lp = U.seg(u, 5.0, 5.8);
    if (lp > 0) {
      const lx = cxL + cw + 4, ly = rowY[2] + rh / 2, R = 20;
      ctx.save(); ctx.strokeStyle = U.rgba(col, 0.85); ctx.lineWidth = 2.5; ctx.setLineDash([6, 5]);
      const a0 = -Math.PI * 0.62, a1 = a0 + lp * Math.PI * 1.24;
      ctx.beginPath(); ctx.arc(lx, ly, R, a0, a1); ctx.stroke(); ctx.restore();
      const cx2 = lx + 34, cy2 = ly;
      cross(ctx, U, lx + R, ly, 26, C.bad, U.seg(u, 5.8, 6.2), 0.9);
      void cx2; void cy2;
      U.text(ctx, 'no new token, no update', ccx - 4, 664, { size: 22, color: C.muted, align: 'center', alpha: U.seg(u, 6.0, 6.5) });
    }
    counter(ctx, U, L, u >= 4.1 ? '1' : '0', 'cannot think longer', col, U.seg(u, 0.8, 1.3), U.seg(u, 6.2, 6.8));
  }

  // diffusion: clean target pattern for the 6 cells of the noisy guess
  const CLEAN = [0.9, 0.25, 0.7, 0.12, 0.95, 0.45];
  function drawDiff(ctx, U, L, t) {
    const { C } = U; const u = t - L.t0, col = C.diff;
    const NSTEP = 8, sT0 = 2.0, sDt = 0.55;
    const k = Math.max(0, Math.min(NSTEP, Math.floor((u - sT0) / sDt) + 1));      // completed steps
    const ph = u >= sT0 && k <= NSTEP ? ((u - sT0) / sDt) % 1 : 0;                 // phase inside current step
    const running = u >= sT0 && u < sT0 + NSTEP * sDt;
    const cX = L.x + 18, cW = 224, rowY = 396, rh = 48;
    const nX = L.x + 260, nW = 128;
    const bX = L.x + 30, bW = 350, bY = 500, bH = 60;
    const loopX = L.x + 410;
    const oX = L.x + 166, oW = 170, oY = 590, oH = 48, gX = L.x + 30, gW = 116;
    chip(ctx, U, cX, rowY, cW, rh, 'The quick brown', { alpha: U.seg(u, 0.3, 0.8) });
    // noisy guess y-hat
    const na = U.seg(u, 0.4, 0.9);
    U.panel(ctx, nX, rowY, nW, rh, { fill: C.panel2, stroke: U.rgba(col, 0.7), r: 10, alpha: na });
    U.text(ctx, 'ŷ', nX + 12, rowY + rh / 2, { size: 26, kind: 'display', italic: true, color: C.ink, baseline: 'middle', alpha: na });
    const done = U.seg(u, sT0 + NSTEP * sDt + 0.2, sT0 + NSTEP * sDt + 0.8);
    const r = U.rng(9001 + Math.min(k, NSTEP) * 17);
    const s = Math.min(k, NSTEP) / NSTEP;
    for (let i = 0; i < 6; i++) {
      const noise = r();
      const v = (1 - s) * noise + s * CLEAN[i];
      ctx.save(); ctx.globalAlpha *= na * (1 - done);
      ctx.fillStyle = U.mix(C.panel2, col, 0.15 + 0.85 * v); ctx.fillRect(nX + 36 + i * 14, rowY + 10, 12, rh - 20); ctx.restore();
    }
    if (done > 0) U.text(ctx, 'fox', nX + 36 + 42, rowY + rh / 2 + 1, { size: 26, weight: 700, color: col, align: 'center', baseline: 'middle', alpha: done });
    // arrows into the block, block, output
    U.arrow(ctx, cX + cW / 2, rowY + rh + 2, cX + cW / 2, bY - 4, { color: C.faint, progress: U.seg(u, 0.6, 1.0), head: 10, width: 2.5 });
    U.arrow(ctx, nX + nW / 2, rowY + rh + 2, nX + nW / 2, bY - 4, { color: C.faint, progress: U.seg(u, 0.7, 1.1), head: 10, width: 2.5 });
    const lit = running ? Math.max(0, 1 - Math.abs(ph - 0.3) / 0.3) : 0;
    const ba = U.seg(u, 0.8, 1.3);
    U.panel(ctx, bX, bY, bW, bH, { fill: U.rgba(col, 0.10 + 0.3 * lit), stroke: U.rgba(col, 0.65 + 0.35 * lit), r: 10, alpha: ba });
    U.text(ctx, 'Diffusion Transformer', bX + bW / 2, bY + bH / 2 + 1, { size: 24, weight: 700, color: C.ink, align: 'center', baseline: 'middle', alpha: ba });
    U.arrow(ctx, oX + oW / 2, bY + bH + 2, oX + oW / 2, oY - 4, { color: C.faint, progress: U.seg(u, 1.1, 1.4), head: 10, width: 2.5 });
    const oa = U.seg(u, 1.2, 1.6);
    const flash = running ? Math.max(0, 1 - Math.abs(ph - 0.55) / 0.25) : 0;
    chip(ctx, U, oX, oY, oW, oH, 'noise ε̂', { alpha: oa, stroke: col, fill: U.rgba(col, 0.10 + 0.3 * flash), size: 24, weight: 700 });
    // loop: subtract the predicted noise from the guess
    const loop = [[oX + oW + 2, oY + oH / 2], [loopX, oY + oH / 2], [loopX, rowY + rh / 2], [nX + nW + 3, rowY + rh / 2]];
    polyArrow(ctx, U, loop, { color: U.rgba(col, 0.8), width: 2.5, progress: U.seg(u, 1.4, 1.9), head: 10 });
    if (running && ph > 0.55) glow(ctx, U, ...along(loop, U.inv(0.55, 1, ph)), col, 1, 5);
    if (running && ph < 0.5) glow(ctx, U, nX + nW / 2, U.lerp(rowY + rh, oY, U.inv(0, 0.5, ph)), col, 1, 5);
    // fixed schedule
    const sa = U.seg(u, 1.6, 2.0);
    U.text(ctx, 'fixed schedule', L.x + 18, 660, { size: 22, color: C.muted, alpha: sa });
    const tk0 = L.x + 196, tk1 = L.x + L.w - 30;
    for (let i = 0; i < NSTEP; i++) {
      const x = U.lerp(tk0, tk1, i / (NSTEP - 1)); const on = i < k;
      U.dot(ctx, x, 674, on ? 7 : 5, on ? col : C.faint, { alpha: sa });
    }
    U.text(ctx, 'T', tk0, 690, { size: 22, kind: 'mono', color: C.muted, align: 'center', alpha: sa });
    U.text(ctx, '1', tk1, 690, { size: 22, kind: 'mono', color: C.muted, align: 'center', alpha: sa });
    // after the run: there is no score for its own guess
    const ga = U.seg(u, 6.3, 6.8);
    if (ga > 0) {
      chip(ctx, U, gX, oY, gW, oH, null, { alpha: ga * 0.9, stroke: C.faint, fill: 'rgba(0,0,0,0)', dash: [6, 5] });
      cross(ctx, U, gX + 22, oY + oH / 2, 20, C.bad, U.seg(u, 6.7, 7.1), 0.9);
      U.text(ctx, 'score', gX + 40, oY + oH / 2 + 1, { size: 24, color: C.muted, baseline: 'middle', alpha: ga });
    }
    const val = u < sT0 ? '0' : (k >= NSTEP ? 'T' : k + ' / T');
    counter(ctx, U, L, val, 'set by the schedule', col, U.seg(u, 0.8, 1.3), U.seg(u, 6.4, 7.0));
  }

  // EBT: the guess is a distribution over a few vocabulary items, refined by gradient steps on the energy
  const VOC = ['fox', 'dog', 'cat', 'car', 'sun'];
  const Z0 = [0.2, 0.9, -0.1, 0.6, 0.35], Z1 = [3.4, 0.5, 0.2, -0.4, -0.6];
  const NE = 6;                       // gradient steps shown (7 forward passes: E at steps 0..6)
  const EV = Array.from({ length: NE + 1 }, (_, k) => 0.14 + 0.86 * Math.exp(-0.62 * k));
  function drawEBT(ctx, U, L, t) {
    const { C } = U; const u = t - L.t0, col = C.ebt;
    const sT0 = 1.9, sDt = 0.72;
    const kf = U.clamp((u - sT0) / sDt, 0, NE + 0.999);          // continuous step index
    const k = Math.floor(kf), ph = kf - k;                       // step index, phase
    const running = u >= sT0 && u < sT0 + (NE + 1) * sDt;
    const settled = U.seg(u, sT0 + (NE + 1) * sDt, sT0 + (NE + 1) * sDt + 0.5);
    const cX = L.x + 18, cW = 224, rowY = 376, rh = 94;
    const yX = L.x + 264, yW = 152;
    const bX = L.x + 30, bW = 360, bY = 500, bH = 60;
    const eX = bX, eW = bW, eY = 590, eH = 80;
    const loopX = L.x + 442;
    chip(ctx, U, cX, rowY + (rh - 48) / 2, cW, 48, 'The quick brown', { alpha: U.seg(u, 0.3, 0.8) });
    // guess y-hat: bars
    const ya = U.seg(u, 0.4, 0.9);
    U.panel(ctx, yX, rowY, yW, rh, { fill: C.panel2, stroke: U.rgba(col, 0.75), r: 10, alpha: ya });
    U.text(ctx, 'ŷ', yX + 12, rowY + 10, { size: 26, kind: 'display', italic: true, color: C.ink, alpha: ya });
    // mix factor: guess moves after each gradient step (second half of each step)
    const stepDone = running ? k + U.ease(U.inv(0.55, 0.95, ph)) : (u >= sT0 ? NE : 0);
    const f = 1 - Math.exp(-0.62 * Math.min(stepDone, NE));
    const p = softmax(Z0.map((z, i) => U.lerp(z, Z1[i], f)));
    const bx0 = yX + 40, bw = 16, bg = 6, bBase = rowY + rh - 10, bMax = 52;
    p.forEach((v, i) => {
      const h = Math.max(3, Math.sqrt(v) * bMax);
      ctx.save(); ctx.globalAlpha *= ya; ctx.fillStyle = i === 0 ? col : U.rgba(col, 0.45); ctx.fillRect(bx0 + i * (bw + bg), bBase - Math.min(h, bMax), bw, Math.min(h, bMax)); ctx.restore();
    });
    if (settled > 0) U.text(ctx, VOC[0], bx0 + bw / 2, rowY + 12, { size: 22, weight: 700, color: col, align: 'left', alpha: settled });
    // arrows in, block
    U.arrow(ctx, cX + cW / 2, rowY + (rh + 48) / 2 + 2, cX + cW / 2, bY - 4, { color: C.faint, progress: U.seg(u, 0.6, 1.0), head: 10, width: 2.5 });
    U.arrow(ctx, yX + yW / 2 - 20, rowY + rh + 2, yX + yW / 2 - 20, bY - 4, { color: C.faint, progress: U.seg(u, 0.7, 1.1), head: 10, width: 2.5 });
    const lit = running ? Math.max(0, 1 - Math.abs(ph - 0.22) / 0.22) : 0;
    const ba = U.seg(u, 0.8, 1.3);
    U.panel(ctx, bX, bY, bW, bH, { fill: U.rgba(col, 0.10 + 0.3 * lit), stroke: U.rgba(col, 0.65 + 0.35 * lit), r: 10, alpha: ba });
    U.text(ctx, 'Energy-Based Transformer', bX + bW / 2, bY + bH / 2 + 1, { size: 24, weight: 700, color: C.ink, align: 'center', baseline: 'middle', alpha: ba });
    U.arrow(ctx, bX + bW / 2, bY + bH + 2, bX + bW / 2, eY - 4, { color: C.faint, progress: U.seg(u, 1.1, 1.4), head: 10, width: 2.5 });
    if (running && ph < 0.4) glow(ctx, U, yX + yW / 2 - 20, U.lerp(rowY + rh, eY, U.inv(0, 0.4, ph)), col, 1, 5);
    // energy readout: one scalar per step, plotted against step
    const ea = U.seg(u, 1.0, 1.4);
    U.panel(ctx, eX, eY, eW, eH, { fill: C.panel2, stroke: U.rgba(col, 0.75), r: 10, alpha: ea });
    U.text(ctx, 'energy', eX + 14, eY + 10, { size: 22, color: C.muted, alpha: ea });
    U.text(ctx, 'E', eX + 14, eY + 40, { size: 26, kind: 'display', italic: true, weight: 600, color: col, alpha: ea });
    const px0 = eX + 104, px1 = eX + eW - 20, py0 = eY + 14, py1 = eY + eH - 14;
    const PX = (i) => U.lerp(px0, px1, i / NE), PY = (e) => U.lerp(py1, py0, e);
    const shown = u < sT0 ? -1 : (running ? (ph >= 0.4 ? k : k - 1) : NE);
    if (shown >= 0) {
      const pts = []; for (let i = 0; i <= shown; i++) pts.push([PX(i), PY(EV[i])]);
      if (pts.length > 1) U.line(ctx, pts, { color: col, width: 3, alpha: ea });
      pts.forEach((q, i) => U.dot(ctx, q[0], q[1], i === shown ? 6 : 4, col, { alpha: ea }));
      if (settled > 0) {
        const br = 0.6 + 0.4 * Math.sin(t * 2.4);
        glow(ctx, U, PX(NE), PY(EV[NE]), col, settled * br, 5);
        U.line(ctx, [[px0, PY(EV[NE])], [px1, PY(EV[NE])]], { color: C.truth, width: 2, dash: [5, 6], alpha: settled * 0.8 });
        U.text(ctx, 'settled', px1, PY(EV[NE]) - 30, { size: 22, color: C.truth, align: 'right', alpha: settled });
      }
    }
    // gradient loop back to the guess
    const loop = [[eX + eW + 2, eY + eH / 2], [loopX, eY + eH / 2], [loopX, rowY + rh / 2], [yX + yW + 3, rowY + rh / 2]];
    polyArrow(ctx, U, loop, { color: U.rgba(col, 0.85), width: 2.5, progress: U.seg(u, 1.3, 1.8), head: 10 });
    if (running && ph > 0.45 && k < NE) glow(ctx, U, ...along(loop, U.inv(0.45, 0.95, ph)), col, 1, 5);
    U.text(ctx, 'ŷ  ←  ŷ − α ∇ŷ E', bX + bW / 2, eY + eH + 10, { size: 28, kind: 'display', italic: true, color: col, align: 'center', alpha: U.seg(u, 1.4, 1.9) });
    const passes = u < sT0 ? 0 : (running ? k + (ph >= 0.22 ? 1 : 0) : NE + 1);
    counter(ctx, U, L, String(passes), 'until the energy settles', col, U.seg(u, 0.8, 1.3), settled);
  }

  // ---------- table ----------
  function rowLabelPos(ctx, U, r) { return [TB.x + 30, TB.rowY + r * TB.rowH + TB.rowH / 2]; }
  function drawTable(ctx, U, t) {
    const { C } = U;
    const a = U.seg(t, 42.9, 43.9) * (1 - U.seg(t, T_TABLE_OUT, T_TABLE_OUT + 0.7));
    if (a <= 0) return;
    ctx.save(); ctx.globalAlpha *= a;
    const x0 = TB.x, x1 = TB.x + TB.labelW + 3 * TB.colW, colX = (k) => TB.x + TB.labelW + k * TB.colW;
    const yEnd = TB.rowY + 4 * TB.rowH;
    // active column backdrop
    FACETS.forEach((F, k) => {
      const ca = U.fade(t, FW[k][0], FW[k][1], 0.5, 0.5);
      if (ca > 0) U.panel(ctx, colX(k) + 6, TB.headY - 12, TB.colW - 12, yEnd - TB.headY + 22, { fill: U.rgba(C.ink, 0.05), stroke: U.rgba(C.ebt, 0.35), lw: 1.5, r: 14, alpha: ca });
    });
    // header
    FACETS.forEach((F, k) => {
      const active = U.fade(t, FW[k][0], FW[k][1], 0.4, 0.4), past = U.seg(t, FW[k][1] - 0.2, FW[k][1] + 0.4);
      const ink = U.mix(C.muted, C.ink, Math.max(active, past * 0.85));
      U.text(ctx, F.tag, colX(k) + 26, TB.headY, { size: 22, kind: 'mono', color: active > 0.5 ? C.ebt : C.muted, spacing: 2 });
      U.text(ctx, F.title, colX(k) + 26, TB.headY + 32, { size: 30, weight: 700, color: ink, maxWidth: TB.colW - 50, lh: 1.2 });
    });
    // rules
    for (let r = 0; r <= 4; r++) {
      const y = TB.rowY + r * TB.rowH;
      U.line(ctx, [[x0, y], [x1, y]], { color: r === 0 ? C.faint : C.rule, width: r === 0 ? 2 : 1.5, progress: U.seg(t, 43.0 + r * 0.1, 43.9 + r * 0.1) });
    }
    // cells
    const SUM = U.seg(t, T_SUM, T_SUM + 0.8);
    for (let r = 0; r < 4; r++) for (let k = 0; k < 3; k++) {
      const tc = FW[k][0] + 1.2 + r * 0.45, p = U.seg(t, tc, tc + 0.45);
      if (p <= 0) continue;
      const cx = colX(k) + 64, cy = TB.rowY + r * TB.rowH + TB.rowH / 2, mcol = [C.ar, C.rnn, C.diff, C.ebt][r];
      if (TABLE[r][k]) {
        U.dot(ctx, cx, cy, 30, U.rgba(mcol, 0.14), { alpha: p });
        check(ctx, U, cx, cy + 2, 36, mcol, p);
        const why = REASON[r + ',' + k];
        if (why) U.text(ctx, why, cx + 48, cy, { size: 22, color: C.muted, baseline: 'middle', alpha: U.seg(t, tc + 0.3, tc + 0.8), maxWidth: TB.colW - 120 });
      } else cross(ctx, U, cx, cy, 30, C.bad, p, 0.65);
    }
    // summary highlights
    if (SUM > 0) {
      const ry = TB.rowY + 3 * TB.rowH;
      U.panel(ctx, x0 - 10, ry + 6, x1 - x0 + 20, TB.rowH - 12, { fill: U.rgba(C.ebt, 0.07), stroke: U.rgba(C.ebt, 0.75 + 0.2 * Math.sin(t * 2.5)), lw: 2.5, r: 14, alpha: SUM, dash: null });
      const dy = TB.rowY + 2 * TB.rowH + TB.rowH / 2;
      ctx.save(); ctx.globalAlpha *= SUM; ctx.strokeStyle = U.rgba(C.diff, 0.85); ctx.lineWidth = 2.5; ctx.setLineDash([6, 6]);
      ctx.beginPath(); ctx.arc(colX(0) + 64, dy + 1, 38, 0, Math.PI * 2); ctx.stroke(); ctx.restore();
    }
    U.badge(ctx, 'paper', 'Table 1', x0, yEnd + 20);
    ctx.restore();
  }
  // lane names fly from lane tops to table row labels
  function drawNames(ctx, U, t) {
    const { C } = U; const fly = U.seg(t, 42.6, 44.0);
    if (fly <= 0 || t > T_TABLE_OUT + 0.8) return;
    const a = 1 - U.seg(t, T_TABLE_OUT, T_TABLE_OUT + 0.7);
    LANES.forEach((L, r) => {
      const col = C[L.key];
      const s = U.lerp(32, 34, fly);
      const w0 = U.measure(ctx, L.name, { size: 32, weight: 700 }).w;
      const xa = L.x + L.w / 2 - w0 / 2, ya = PY0 + 18 + 16;
      const [xb, yb] = rowLabelPos(ctx, U, r);
      const x = U.lerp(xa, xb, fly), y = U.lerp(ya, yb, fly);
      // swap to the paper's Table 1 names after landing (old fades out fully before new fades in)
      const fo = U.seg(t, 44.0, 44.3), fi = U.seg(t, 44.3, 44.7);
      if (fo < 1) U.text(ctx, L.name, x, y, { size: s, weight: 700, color: col, baseline: 'middle', alpha: a * (1 - fo) });
      if (fi > 0) U.text(ctx, L.row, x, y, { size: s, weight: 700, color: col, baseline: 'middle', alpha: a * fi });
    });
  }

  // ---------- takeaway ----------
  function fieldCanvas(U, w, h, kind) {
    return U.cache('s02-field-' + kind + '-' + w + 'x' + h, w, h, (g, cw, ch) => {
      const { C } = U;
      const toPx = (uu, vv) => [(uu + UMAX) / (2 * UMAX) * cw, (1 - (vv + 1) / 2) * ch];
      if (kind === 'ebt') {
        const cols = 128, rows = Math.round(cols * ch / cw);
        const grid = []; let lo = Infinity, hi = -Infinity;
        for (let r = 0; r < rows; r++) { const row = []; for (let c = 0; c < cols; c++) { const uu = -UMAX + 2 * UMAX * c / (cols - 1), vv = 1 - 2 * r / (rows - 1); const e = Efun(uu, vv); row.push(e); lo = Math.min(lo, e); hi = Math.max(hi, e); } grid.push(row); }
        const img = g.createImageData(cols, rows);
        for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) { const v = Math.pow((grid[r][c] - lo) / (hi - lo), 0.8); const cc = U.cmap(v); const i = 4 * (r * cols + c); img.data[i] = cc[0]; img.data[i + 1] = cc[1]; img.data[i + 2] = cc[2]; img.data[i + 3] = 255; }
        const tmp = document.createElement('canvas'); tmp.width = cols; tmp.height = rows; tmp.getContext('2d').putImageData(img, 0, 0);
        g.imageSmoothingEnabled = true; g.imageSmoothingQuality = 'high'; g.drawImage(tmp, 0, 0, cw, ch);
        const levels = []; for (let i = 1; i <= 9; i++) levels.push(lo + (hi - lo) * Math.pow(i / 10, 1.25));
        U.contours(g, grid, 0, 0, cw, ch, levels, { color: 'rgba(233,238,246,0.30)', width: 1.3 });
      } else {
        g.fillStyle = C.panel2; g.fillRect(0, 0, cw, ch);
        const nx = 19, ny = 8;
        for (let i = 0; i < nx; i++) for (let j = 0; j < ny; j++) {
          const uu = -UMAX + 2 * UMAX * (i + 0.5) / nx, vv = -1 + 2 * (j + 0.5) / ny;
          const gr = Egrad(uu, vv), m = Math.hypot(gr[0], gr[1]);
          const [px, py] = toPx(uu, vv);
          // direction of -grad in pixel space (v up)
          const dx = -gr[0] / (2 * UMAX) * cw, dy = gr[1] / 2 * ch, dm = Math.hypot(dx, dy) || 1;
          const len = 10 + 14 * Math.min(1, m / 0.9);
          const ex = px + dx / dm * len, ey = py + dy / dm * len, sx = px - dx / dm * len * 0.5, sy = py - dy / dm * len * 0.5;
          g.globalAlpha = 0.35 + 0.5 * Math.min(1, m / 0.6);
          U.arrow(g, sx, sy, ex, ey, { color: C.diff, width: 2, head: 7 });
        }
        g.globalAlpha = 1;
      }
    });
  }
  function drawTakeaway(ctx, U, t) {
    const { C } = U;
    const a = U.seg(t, TK.t0, TK.t0 + 0.9);
    if (a <= 0) return;
    ctx.save(); ctx.globalAlpha *= a;
    const PADX = 24, TOP = 70, BOT = 20;
    const pw = TK.w - 2 * PADX, ph = TK.h - TOP - BOT;
    const panels = [
      { x: TK.lx, kind: 'diff', col: C.diff, title: 'Diffusion learns arrows (∇E)' },
      { x: TK.rx, kind: 'ebt', col: C.ebt, title: 'EBT learns heights (E)' },
    ];
    const R = runs();
    panels.forEach((P, pi) => {
      const rise = (1 - U.easeOut(U.inv(TK.t0 + pi * 0.25, TK.t0 + 0.9 + pi * 0.25, t))) * 18;
      const y = TK.y + rise;
      U.panel(ctx, P.x, y, TK.w, TK.h, { fill: C.panel, stroke: U.rgba(P.col, 0.45), r: 18 });
      U.text(ctx, P.title, P.x + 26, y + 20, { size: 30, weight: 700, color: P.col });
      U.text(ctx, 'illustration', P.x + TK.w - 26, y + 26, { size: 22, italic: true, color: C.faint, align: 'right' });
      const fx = P.x + PADX, fy = y + TOP;
      ctx.save(); U.rr(ctx, fx, fy, pw, ph, 10); ctx.clip();
      ctx.drawImage(fieldCanvas(U, pw, ph, P.kind), fx, fy, pw, ph);
      ctx.restore();
      const toPx = (uu, vv) => [fx + (uu + UMAX) / (2 * UMAX) * pw, fy + (1 - (vv + 1) / 2) * ph];
      R.forEach((run, ri) => {
        const vis = U.fade(t, run.t0 - 0.3, ri === 0 ? R[1].t0 - 0.35 : DUR, 0.4, 0.5);
        if (vis <= 0) return;
        const prog = U.seg(t, run.t0, run.t1, U.easeOut);
        const lab = U.seg(t, run.t1 - 0.2, run.t1 + 0.4);
        const ends = run.paths.map(pp => pp[pp.length - 1][2]);
        const best = ends[0] < ends[1] ? 0 : 1;
        run.paths.forEach((pp, j) => {
          const pts = pp.map(q => toPx(q[0], q[1]));
          U.line(ctx, pts, { color: P.col, width: 3, progress: prog, alpha: vis * 0.9 });
          U.dot(ctx, pts[0][0], pts[0][1], 6, C.ink, { alpha: vis * 0.8, stroke: C.bg, lw: 2 });
          const head = along(pts, prog);
          glow(ctx, U, head[0], head[1], P.col, vis, 6);
          if (lab > 0) {
            const lx = head[0] + (head[0] > fx + pw - 170 ? -150 : 18), ly = head[1] - 54;
            if (P.kind === 'ebt') {
              const txt = 'E = ' + ends[j].toFixed(2);
              chip(ctx, U, lx, ly, 132, 40, txt, { alpha: vis * lab, stroke: j === best ? C.truth : C.faint, fill: U.rgba(C.bg, 0.85), kind: 'mono', size: 22, color: j === best ? C.truth : C.ink });
              if (j === best) { U.dot(ctx, lx + 158, ly + 20, 17, U.rgba(C.bg, 0.85), { alpha: vis * lab }); check(ctx, U, lx + 158, ly + 21, 20, C.truth, U.seg(t, run.t1 + 0.3, run.t1 + 0.8), vis); }
            } else {
              chip(ctx, U, lx, ly, 70, 40, '?', { alpha: vis * lab, stroke: C.faint, fill: U.rgba(C.bg, 0.85), kind: 'mono', size: 24, color: C.muted });
            }
          }
        });
      });
    });
    U.badge(ctx, 'paper', 'Sec 6.4, App. E.2', TK.lx, TK.y + TK.h + 18);
    const la = U.fade(t, 63.7, DUR - 0.5, 0.6, 0.4);
    U.text(ctx, 'Diffusion is the closest relative: it also iterates, but it learns the gradient of an energy, not the energy itself.', 960, 744, { size: 40, kind: 'display', weight: 500, color: C.ink, align: 'center', maxWidth: 1480, alpha: la, lh: 1.25 });
    const lb = U.fade(t, 68.4, DUR - 0.5, 0.6, 0.4);
    U.text(ctx, 'EBT learns the energy itself, so every step comes with a score it can check.', 960, 872, { size: 36, weight: 700, color: C.ebt, align: 'center', maxWidth: 1500, alpha: lb });
    ctx.restore();
  }

  // ---------- intro: the shared task ----------
  function drawPrompt(ctx, U, t) {
    const { C } = U;
    const a = U.fade(t, 0.5, 43.0, 0.8, 0.6);
    if (a <= 0) return;
    const pf = U.seg(t, 5.2, 6.4);
    const size = U.lerp(92, 40, pf);
    const sEnd = 40;
    const word = 'The quick brown';
    const w1 = U.measure(ctx, word, { size, kind: 'display', weight: 500 }).w;
    const bw = size * 2.3, gap = size * 0.35, total = w1 + gap + bw;
    const totalEnd = U.measure(ctx, word, { size: sEnd, kind: 'display', weight: 500 }).w + sEnd * 2.65;
    const cx = U.lerp(960, 1824 - totalEnd / 2, pf), cy = U.lerp(440, 142, pf);
    const x0 = cx - total / 2;
    U.text(ctx, word, x0, cy, { size, kind: 'display', weight: 500, color: C.ink, baseline: 'middle', alpha: a });
    const blink = 0.55 + 0.45 * Math.sin(t * 3.2);
    ctx.save(); ctx.globalAlpha *= a; ctx.fillStyle = U.rgba(C.ebt, 0.35 + 0.55 * blink);
    ctx.fillRect(x0 + w1 + gap, cy + size * 0.36, bw, Math.max(3, size * 0.07)); ctx.restore();
    U.text(ctx, 'NEXT TOKEN?', 1824, 92, { size: 22, kind: 'mono', color: C.muted, align: 'right', spacing: 2, alpha: a * pf });
    U.text(ctx, 'Four model families fill the same blank.', 960, 590, { size: 40, color: C.ink, align: 'center', alpha: U.fade(t, 1.3, 5.6, 0.6, 0.5) });
    U.text(ctx, 'Count the computation each one spends.', 960, 646, { size: 40, color: C.ebt, align: 'center', alpha: U.fade(t, 2.4, 5.6, 0.6, 0.5) });
  }

  EBTV.scene({
    id: 's02_families',
    title: 'Model families',
    dur: DUR,
    assets: [],
    draw(ctx, t, U) {
      const { C } = U;
      const endA = 1 - U.seg(t, DUR - 0.7, DUR - 0.1);
      ctx.save(); ctx.globalAlpha *= endA;
      U.header(ctx, t, DUR, 'Model families', 'Four ways to make a prediction');
      drawPrompt(ctx, U, t);
      // lanes
      if (t > LANES[0].t0 && t < T_LANES_OUT + 0.9) {
        U.badge(ctx, 'paper', 'redrawn from Fig 1', 96, 206, { alpha: U.fade(t, 6.4, T_LANES_OUT + 0.6, 0.6, 0.6) });
        const fns = [drawAR, drawRNN, drawDiff, drawEBT];
        LANES.forEach((L) => {   // dashed placeholders for lanes still to come
          const pa = U.seg(t, 6.0, 6.9) * (1 - U.seg(t, L.t0 - 0.2, L.t0 + 0.4));
          if (pa <= 0) return;
          U.panel(ctx, L.x, PY0, L.w, PY1 - PY0, { fill: false, stroke: C.rule, dash: [8, 8], lw: 2, r: 18, alpha: pa });
          U.text(ctx, L.name, L.x + L.w / 2, PY0 + 18, { size: 32, weight: 700, color: C[L.key], align: 'center', alpha: pa * 0.32 });
        });
        LANES.forEach((L, i) => {
          const a = laneAlpha(U, i, t); if (a <= 0) return;
          const dy = (1 - U.easeOut(U.inv(L.t0, L.t0 + 0.8, t))) * 16;
          ctx.save(); ctx.globalAlpha *= a; ctx.translate(0, dy);
          lanePanel(ctx, U, L, t, C[L.key]);
          fns[i](ctx, U, L, t);
          ctx.restore();
        });
      }
      drawNames(ctx, U, t);
      drawTable(ctx, U, t);
      drawTakeaway(ctx, U, t);
      // captions (one at a time)
      const CAP = [
        ['One pass through a fixed stack of layers. Easy token or hard token, the compute is exactly the same.', 6.8, 13.4, C.ar],
        ['The hidden state h updates once per incoming token. With no new input, there is no way to think longer.', 14.2, 21.4, C.rnn],
        ['Start from noise, then denoise over a fixed schedule of T steps. Each step predicts the noise, not a score for the guess.', 22.2, 29.8, C.diff],
        ['Guess, score the guess with one number (the energy), then step downhill along its gradient. Repeat until the energy settles.', 30.6, 36.9, C.ebt],
        ['Harder tokens can take more steps. And any two guesses can be compared by their energy.', 37.1, 42.3, C.ebt],
        ['Facet 1 · Dynamic compute: can it spend more computation on a harder prediction?', 44.4, 49.2, C.rule],
        ['Facet 2 · Uncertainty: can it say how unsure it is about a continuous prediction?', 49.4, 54.2, C.rule],
        ['Facet 3 · Verification: can it score its own prediction, with no extra model?', 54.4, 59.2, C.rule],
        ['Only EBTs have all three. Diffusion has Facet 1 only.', 59.4, 62.7, C.ebt],
      ];
      CAP.forEach(([s, a, b, stroke]) => U.caption(ctx, s, t, a, b, { size: 34, maxWidth: 1240, stroke: U.rgba(stroke, 0.6) }));
      ctx.restore();
    },
  });
})();
