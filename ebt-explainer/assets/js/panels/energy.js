/* Panel: what an energy function is.
   View A: a 1-D energy E(x, ŷ) for one fixed context (sculptable, or a real slice of the toy EBT) and the Boltzmann
   distribution it implies, with temperature, a constant shift, the partition function and basin masses.
   View B: a maze where generating a path needs search and verifying one needs a single pass.
   Everything is computed live: spline / toy-model energies, log-sum-exp normalization, Wilson mazes, BFS.
   Consolidated from the legacy energy panel (thin-data step dropped). */
EBT.panel({
  id: 'energy',
  nav: 'What an energy is',
  title: 'What an energy function is',
  lede: 'An energy-based model does not output an answer. It outputs one number scoring how well a candidate answer fits the context; lower is better.',
  text: `
    <p>An EBT takes a context $x$ and a candidate prediction $\\hat y$ and returns one scalar, the energy $E_\\theta(x,\\hat y)$ (p.5). Fix the context and the energy is just a function of $\\hat y$: one-dimensional here, thousands of dimensions in a real EBT. The Boltzmann distribution reads it as a probability:</p>
    <div class="eq energy-eqs">$$p_\\theta(\\hat y\\mid x)=\\frac{e^{-E_\\theta(x,\\hat y)}}{Z_\\theta(x)}$$$$Z_\\theta(x)=\\int e^{-E_\\theta(x,\\hat y')}\\,d\\hat y'$$<span class="why">The minus sign makes low energy mean high probability; Z makes it integrate to one (p.6).</span></div>`,
  steps: [
    { label: 'One number per candidate', html: '<p>Drag candidates <b>a</b> and <b>b</b>, or reshape the curve by its handles. Each candidate costs one forward pass and gets one number. The energy scores answers but does not produce them: to predict, the model must walk downhill, along $-\\partial E/\\partial\\hat y$ (blue arrow). The <b>toy EBT slice</b> is a cut through a trained model.</p>' },
    { label: 'Energy becomes probability', html: '<p>The lower frame shows $p(\\hat y\\mid x)$. An energy gap $\\Delta E$ makes one candidate $e^{\\Delta E}$ times likelier. The animation divides $E$ by a temperature $T$ ($T = 1$ in the paper): cold piles mass into the deepest well, warm spreads it out.</p>' },
    { label: 'Z is intractable, and not needed', html: '<p>Now the frame shows $e^{-E}$; its area is $Z$. Over a 50,277-dimensional $\\hat y$ it is hopeless: "intractable" (p.6). Watch the shift $c$ added to every energy: $Z$ changes, but the ratio $p(a)/p(b)=e^{-(E(a)-E(b))}$, the ranking and the gradient do not. So EBTs stay <b>unnormalized</b>: "you only ever need sample relative likelihood comparison" (p.41). The catch: raw energies from different contexts are not directly comparable.</p>' },
    { label: 'Checking is cheaper than finding', html: '<p>"Verifying the correctness of a given path is significantly easier than discovering such a path" (p.5). <b>Search</b> expands most of the maze; <b>verify</b> walks the given path once. Draw a path through a wall and the verifier catches it. An EBT is such a verifier, turned into a generator by descending its own score (p.6).</p>' },
  ],
  after: `
    <p class="note">The curves are illustrations except the toy slice (trained toy EBT, x = 0.25). The paper's "exponentially easier" (p.3) is a complexity claim; in mazes the gap is polynomial.</p>`,
  source: [
    { kind: 'concept', note: '1-D energies, maze' },
    { kind: 'toy', note: '"toy EBT slice" = trained toy 2-D EBT, x = 0.25, ŷ₂ = 0' },
    { kind: 'paper', note: 'Sec 2.1, Sec 3.1, App. H.3' },
  ],
  figure(stage, ctx) {
    const { lib } = ctx, h = lib.h, C = lib.C;
    const T2 = window.EBT && window.EBT.toy2d;
    const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
    const fx = (v, d = 2) => !isFinite(v) ? '–' : (v < 0 && Math.abs(v) >= 0.5 * Math.pow(10, -d) ? '−' : '') + Math.abs(v).toFixed(d);
    const big = (v) => Math.round(v).toLocaleString('en-US');
    const pct = (v) => (v >= 0.995 ? '>99' : v < 0.005 ? '<1' : Math.round(100 * v)) + '%';

    // ---------------------------------------------------------------- 1-D energies
    const XMIN = -3, XMAX = 3, NG = 1201, DX = (XMAX - XMIN) / (NG - 1);
    const GX = Float64Array.from({ length: NG }, (_, i) => XMIN + i * DX);
    const KX = [-3, -2.25, -1.5, -0.75, 0, 0.75, 1.5, 2.25, 3], KH = 0.75;
    // a narrow deep well (left) and a wide shallower basin (right): which one holds more mass depends on T
    const KY0 = [5, 3.2, -1.4, 2.4, 1.0, -0.2, -0.4, 1.0, 5];
    const SRC = {
      sculpt: { yr: [-3, 6], cand: [-1.33, 1.3] },
      toy: { yr: [-2.5, 10], cand: [0.5, 1.3] },
    };
    const S = { view: 'energy', src: 'sculpt', T: 1, c: 0, a: -1.33, b: 1.3, pmode: 'norm', dimP: false, showCost: false, ky: KY0.slice(), hover: -1 };
    let EB = new Float64Array(NG); // base energy (no shift)
    let toyCache = null;

    function spline(v) {
      let i = clamp(Math.floor((v - KX[0]) / KH), 0, KX.length - 2); const t = (v - KX[i]) / KH, y = S.ky, L = y.length;
      const p1 = y[i], p2 = y[i + 1], p0 = i > 0 ? y[i - 1] : 2 * p1 - p2, p3 = i + 2 < L ? y[i + 2] : 2 * p2 - p1;
      return 0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t + (-p0 + 3 * p1 - 3 * p2 + p3) * t * t * t);
    }
    function rebuild() {
      if (S.src === 'sculpt') for (let i = 0; i < NG; i++) EB[i] = spline(GX[i]);
      else {
        if (!toyCache) { toyCache = new Float64Array(NG); const K = T2.nCkpt - 1; for (let i = 0; i < NG; i++) toyCache[i] = T2.energy(K, 0.25, [GX[i], 0]); }
        EB = toyCache.slice();
      }
      // basins: intervals between consecutive local maxima of the energy (where gradient descent would end up)
      const mx = [0]; for (let i = 1; i < NG - 1; i++) if (EB[i] > EB[i - 1] && EB[i] >= EB[i + 1]) mx.push(i); mx.push(NG - 1);
      S.maxima = mx;
    }
    const idx = (v) => clamp((v - XMIN) / DX, 0, NG - 1);
    const Ebase = (v) => { const f = idx(v), i = Math.min(NG - 2, Math.floor(f)), t = f - i; return EB[i] * (1 - t) + EB[i + 1] * t; };
    const Eat = (v) => Ebase(v) + S.c;
    const dEdy = (v) => { const e = 0.01; return (Ebase(v + e) - Ebase(v - e)) / (2 * e); };
    // probabilities on the grid (log-sum-exp; Riemann sum with spacing DX)
    function probs() {
      const T = S.T; let m = -Infinity; for (let i = 0; i < NG; i++) { const q = -(EB[i] + S.c) / T; if (q > m) m = q; }
      let s = 0; for (let i = 0; i < NG; i++) s += Math.exp(-(EB[i] + S.c) / T - m);
      const logZ = m + Math.log(s * DX), p = new Float64Array(NG); for (let i = 0; i < NG; i++) p[i] = Math.exp(-(EB[i] + S.c) / T - logZ);
      // unnormalized scale is fixed by the unshifted curve so that the shift visibly changes e^{-E/T}
      let mb = -Infinity; for (let i = 0; i < NG; i++) mb = Math.max(mb, -EB[i] / T);
      return { p, logZ, uMaxBase: Math.exp(mb) };
    }
    // the basin [lo, hi) (grid indices) that contains ŷ = v, its probability mass and its peak
    function basinOf(v, P) {
      const i = Math.round(idx(v)), mx = S.maxima; let k = 0; while (k < mx.length - 2 && mx[k + 1] <= i) k++;
      const lo = mx[k], hi = mx[k + 1]; let m = 0, pk = lo; for (let j = lo; j < hi; j++) { m += P.p[j] * DX; if (P.p[j] > P.p[pk]) pk = j; }
      return { lo, hi, mass: m, peak: pk };
    }

    // ---------------------------------------------------------------- DOM
    const viewRow = h('div', { class: 'controls' }); stage.appendChild(viewRow);
    viewRow.appendChild(h('span', { class: 'fig-label' }, 'view'));
    const viewSeg = lib.segmented({ label: 'View', options: [['energy', 'energy ↔ probability'], ['maze', 'verify vs generate']], value: 'energy', onchange: (v) => setView(v) });
    viewRow.appendChild(viewSeg.el);
    const viewA = h('div', { style: { display: 'grid', gap: '16px' } }); stage.appendChild(viewA);
    const viewB = h('div', { style: { display: 'none', gap: '16px' } }); stage.appendChild(viewB);

    const FE = lib.frame(viewA, { label: 'Energy', sub: '&nbsp;' });
    // logical canvas width = displayed width, so canvas labels keep their pixel size on phones
    const EW = clamp(Math.floor(FE.frame.clientWidth || 600), 300, 660), narrow = EW < 480;
    const EH = narrow ? 214 : 236, cvE = lib.canvas(FE.frame, EW, EH, { label: 'Energy of every candidate prediction for one fixed context, with two candidates a and b' });
    const FP = lib.frame(viewA, { label: 'Probability', sub: '&nbsp;' });
    const PH = narrow ? 160 : 178, cvP = lib.canvas(FP.frame, EW, PH, { label: 'The Boltzmann probability implied by the energy above' });

    const c1 = h('div', { class: 'controls' }); viewA.appendChild(c1);
    c1.appendChild(h('span', { class: 'fig-label' }, 'curve'));
    const srcSeg = lib.segmented({ label: 'Energy curve', options: [['sculpt', 'sculpt'], ['toy', 'toy EBT slice']].filter(o => o[0] !== 'toy' || (T2 && T2.ready)), value: 'sculpt', onchange: (v) => { setSrc(v); draw(); } });
    c1.appendChild(srcSeg.el);
    const resetBtn = lib.button('reset curve', () => { S.ky = KY0.slice(); rebuild(); draw(); }); c1.appendChild(resetBtn);
    const c1b = h('div', { class: 'controls' }); viewA.appendChild(c1b);
    c1b.appendChild(h('span', { class: 'fig-label' }, 'lower frame'));
    const pSeg = lib.segmented({ label: 'Lower frame', options: [['norm', 'p, normalized'], ['unnorm', 'e^(−E/T), area Z']], value: 'norm', onchange: (v) => { S.pmode = v; S.dimP = false; draw(); } });
    c1b.appendChild(pSeg.el);
    const c2 = h('div', { class: 'controls' }); viewA.appendChild(c2);
    const slT = lib.slider({ id: 'energy-T', label: 'temperature T', min: 0.25, max: 3, step: 0.05, value: 1, fmt: (v) => v.toFixed(2), oninput: (v) => { S.T = v; S.dimP = false; draw(); } });
    const slC = lib.slider({ id: 'energy-c', label: 'shift c (added to E)', min: -1.5, max: 1.5, step: 0.05, value: 0, fmt: (v) => (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(2), oninput: (v) => { S.c = v; draw(); } });
    c2.append(slT.el, slC.el);
    const roA = h('div', { class: 'readout', 'aria-live': 'polite' }); viewA.appendChild(roA);
    const roCost = h('div', { class: 'readout', style: { display: 'none' } }); viewA.appendChild(roCost);

    // ---------------------------------------------------------------- energy frame drawing
    const PL = narrow ? 40 : 52, PR = 14, PT = 12, PB = 28;
    const X = (v) => PL + (v - XMIN) / (XMAX - XMIN) * (EW - PL - PR);
    const Xinv = (px) => XMIN + (px - PL) / (EW - PL - PR) * (XMAX - XMIN);
    const yr = () => SRC[S.src].yr;
    const Y = (e) => { const [lo, hi] = yr(); return PT + (1 - (e - lo) / (hi - lo)) * (EH - PT - PB); };
    const Yinv = (py) => { const [lo, hi] = yr(); return lo + (1 - (py - PT) / (EH - PT - PB)) * (hi - lo); };
    const niceStep = (span) => { const s0 = span / 4, m = Math.pow(10, Math.floor(Math.log10(s0))), e = s0 / m; return (e >= 5 ? 5 : e >= 2 ? 2 : 1) * m; };
    function ticks(lo, hi) { const st = niceStep(hi - lo), o = []; for (let v = Math.ceil(lo / st) * st; v <= hi + 1e-9; v += st) o.push(+v.toFixed(6)); return o; }
    const tickDec = (lo, hi) => Math.max(0, -Math.floor(Math.log10(niceStep(hi - lo)) + 1e-9));
    // mono label with a white halo so it stays legible over curves
    function label(c, s, x, y, o = {}) { c.save(); c.font = (o.weight || 500) + ' ' + (o.size || 11.5) + 'px ' + lib.F.mono; c.textAlign = o.align || 'left'; c.textBaseline = o.base || 'middle'; c.lineWidth = 3; c.strokeStyle = 'rgba(255,255,255,0.92)'; c.lineJoin = 'round'; c.strokeText(s, x, y); c.fillStyle = o.color || C.ink; c.fillText(s, x, y); c.restore(); }

    function curvePath(c, arr, Yf, top, bot) {
      c.beginPath(); let pen = false;
      for (let i = 0; i < NG; i += 2) { const px = X(GX[i]), py = Yf(arr(i)); if (!isFinite(py)) { pen = false; continue; } const cy = clamp(py, top - 4, bot + 4); if (!pen) { c.moveTo(px, cy); pen = true; } else c.lineTo(px, cy); }
    }
    function drawEnergy() {
      const c = cvE.ctx; cvE.clear(); const [lo, hi] = yr();
      lib.axes(c, { x: PL, y: PT, w: EW - PL - PR, h: EH - PT - PB, xlim: [XMIN, XMAX], ylim: [lo, hi], xticks: [-3, -2, -1, 0, 1, 2, 3], yticks: ticks(lo, hi), xfmt: (v) => fx(v, 0), yfmt: (v) => fx(v, 0), size: 11 });
      lib.text(c, 'ŷ', EW - PR, EH - PB - 16, { size: 13, kind: 'mono', color: C.ink, align: 'right' });
      lib.text(c, 'E', PL + 6, PT + 2, { size: 13, kind: 'mono', color: C.ink });
      c.save(); c.beginPath(); c.rect(PL, PT, EW - PL - PR, EH - PT - PB); c.clip();
      // ghost of the unshifted curve
      if (Math.abs(S.c) > 0.004) {
        curvePath(c, (i) => EB[i], Y, PT, EH - PB); c.strokeStyle = 'rgba(17,17,17,0.45)'; c.lineWidth = 1; c.setLineDash([4, 4]); c.stroke(); c.setLineDash([]);
        const xm = 2.55, y0 = Y(Ebase(xm)), y1 = Y(Ebase(xm) + S.c);
        lib.arrow(c, X(xm), y0, X(xm), y1, { color: C.ink, width: 1.2, head: 6 });
        label(c, (S.c > 0 ? '+' : '−') + 'c', X(xm) - 6, (y0 + y1) / 2, { size: 12, align: 'right' });
      }
      curvePath(c, (i) => EB[i] + S.c, Y, PT, EH - PB); c.strokeStyle = C.blue; c.lineWidth = 2; c.stroke();
      // candidates (drawn clipped to the plot; off-scale ones sit on the top edge)
      [['b', S.b], ['a', S.a]].forEach(([nm, v]) => {
        const e = Eat(v), py = clamp(Y(e), PT + 6, EH - PB);
        c.strokeStyle = 'rgba(17,17,17,0.35)'; c.lineWidth = 1; c.setLineDash([2, 3]); c.beginPath(); c.moveTo(X(v), EH - PB); c.lineTo(X(v), py); c.stroke(); c.setLineDash([]);
        lib.dot(c, X(v), py, 5.5, nm === 'a' ? C.blue : '#fff', { stroke: nm === 'a' ? '#fff' : C.ink, lw: 1.6 });
        label(c, nm + (Y(e) < PT ? ' ↑' : ''), X(v) + 9, py + (py < PT + 20 ? 12 : -12), { size: 13, weight: 700, color: nm === 'a' ? C.blue : C.ink });
      });
      // downhill arrow at a
      {
        const g = dEdy(S.a), py = Y(Eat(S.a)), L = clamp(Math.abs(g) * 22, 0, 70);
        if (L > 6) { const d = -Math.sign(g); lib.arrow(c, X(S.a), py + 14, X(S.a) + d * L, py + 14, { color: C.blue, width: 1.5, head: 7 }); label(c, '−∂E/∂ŷ', X(S.a) + d * (L + 4), py + 14, { size: 11, color: C.blue, align: d > 0 ? 'left' : 'right' }); }
        else label(c, 'flat: ∂E/∂ŷ ≈ 0', X(S.a) + 10, py + 14, { size: 11, color: C.blue });
      }
      c.restore();
      // handles
      if (S.src === 'sculpt') KX.forEach((kx, k) => { const py = Y(S.ky[k] + S.c); if (py < PT - 2 || py > EH - PB + 2) return; lib.dot(c, X(kx), py, S.hover === k ? 6.5 : 4.5, '#fff', { stroke: C.blue, lw: 1.5 }); });
    }
    function drawProb(P) {
      const c = cvP.ctx; cvP.clear();
      const top = 14, bot = PH - 26, norm = S.pmode === 'norm';
      let ymax;
      if (norm) { let m = 0; for (let i = 0; i < NG; i++) m = Math.max(m, P.p[i]); ymax = m * 1.25; }
      else ymax = P.uMaxBase * 1.25;
      const Yp = (v) => top + (1 - v / ymax) * (bot - top);
      c.save(); if (S.dimP) c.globalAlpha = 0.22;
      const yt = ticks(0, ymax).filter(v => v <= ymax * 0.97), dec = tickDec(0, ymax);
      lib.axes(c, { x: PL, y: top, w: EW - PL - PR, h: bot - top, xlim: [XMIN, XMAX], ylim: [0, ymax], xticks: [-3, -2, -1, 0, 1, 2, 3], yticks: yt, xfmt: (v) => fx(v, 0), yfmt: (v) => v.toFixed(dec), size: 11 });
      const val = (i) => norm ? P.p[i] : Math.exp(-(EB[i] + S.c) / S.T);
      c.save(); c.beginPath(); c.rect(PL, top, EW - PL - PR, bot - top); c.clip();
      c.beginPath(); c.moveTo(X(XMIN), Yp(0)); for (let i = 0; i < NG; i += 2) c.lineTo(X(GX[i]), Math.max(top - 2, Yp(val(i)))); c.lineTo(X(XMAX), Yp(0)); c.closePath();
      c.fillStyle = norm ? C.blue4 : C.blue3; c.fill();
      c.beginPath(); for (let i = 0; i < NG; i += 2) { const py = Math.max(top - 2, Yp(val(i))); i ? c.lineTo(X(GX[i]), py) : c.moveTo(X(GX[i]), py); } c.strokeStyle = C.blue; c.lineWidth = 1.8; c.stroke();
      if (!norm && Math.abs(S.c) > 0.004) { c.beginPath(); for (let i = 0; i < NG; i += 2) { const py = Math.max(top - 2, Yp(Math.exp(-EB[i] / S.T))); i ? c.lineTo(X(GX[i]), py) : c.moveTo(X(GX[i]), py); } c.strokeStyle = 'rgba(17,17,17,0.45)'; c.lineWidth = 1; c.setLineDash([4, 4]); c.stroke(); c.setLineDash([]); }
      [['b', S.b], ['a', S.a]].forEach(([nm, v]) => { const f = idx(v), i = Math.round(f); const py = Math.max(top, Yp(val(i))); c.strokeStyle = 'rgba(17,17,17,0.35)'; c.lineWidth = 1; c.setLineDash([2, 3]); c.beginPath(); c.moveTo(X(v), bot); c.lineTo(X(v), py); c.stroke(); c.setLineDash([]); lib.dot(c, X(v), py, 4.5, nm === 'a' ? C.blue : '#fff', { stroke: nm === 'a' ? '#fff' : C.ink, lw: 1.5 }); });
      c.restore();
      // basin masses (normalized view, sculpt and toy curves): the probability that descent-to-the-bottom would land in each basin
      if (norm && !S.dimP && S.maxima.length > 2) {
        const ba = basinOf(S.a, P), bb = basinOf(S.b, P), list = ba.lo === bb.lo ? [['a, b', ba]] : [['a', ba], ['b', bb]];
        list.forEach(([nm, B]) => { const px = X(GX[B.peak]), py = Math.max(top + 10, Yp(P.p[B.peak]) - 10); label(c, 'basin ' + nm + ': ' + pct(B.mass), clamp(px, PL + 60, EW - PR - 60), py, { size: 11.5, align: 'center', weight: 600, color: C.ink }); });
      }
      const lab = norm ? 'area = 1' : 'area = Z = ' + (Math.exp(P.logZ) < 1e4 ? Math.exp(P.logZ).toFixed(3) : Math.exp(P.logZ).toExponential(2));
      label(c, lab, EW - PR - 6, top + 6, { size: 12, align: 'right' });
      if (!norm && Math.abs(S.c) > 0.004) label(c, 'dashed: c = 0 · same shape × ' + Math.exp(-S.c / S.T).toFixed(2), EW - PR - 6, top + 22, { size: 11, align: 'right' });
      label(c, norm ? 'p(ŷ|x)' : 'e^(−E/T)', PL + 6, top + 6, { size: 12 });
      lib.text(c, 'ŷ', EW - PR, bot - 16, { size: 13, kind: 'mono', color: C.ink, align: 'right' });
      c.restore();
      if (S.dimP) lib.text(c, 'step 2 turns these energies into probabilities', EW / 2, (top + bot) / 2, { size: 13, kind: 'mono', color: C.muted, align: 'center', baseline: 'middle' });
    }
    function readoutA(P) {
      const Ea = Eat(S.a), Eb = Eat(S.b), dE = Ea - Eb;
      const pa = Math.exp(-Ea / S.T - P.logZ), pb = Math.exp(-Eb / S.T - P.logZ);
      const viaE = Math.exp(-dE / S.T), viaP = pa / pb;
      const r = (v) => v >= 1e4 || v < 1e-3 ? v.toExponential(2) : v.toFixed(3);
      const Z = Math.exp(P.logZ);
      let html = `<span>E(a) <b>${fx(Ea)}</b></span><span>E(b) <b>${fx(Eb)}</b></span>` +
        `<span>p(a)/p(b) <b>${r(viaP)}</b> = e<sup>−ΔE/T</sup> <b>${r(viaE)}</b> (no Z needed)</span>` +
        `<span>Z on [−3, 3] <b>${Z < 1e5 ? Z.toFixed(3) : Z.toExponential(2)}</b></span><span>log Z <b>${fx(P.logZ, 3)}</b>${Math.abs(S.c) > 0.004 ? ' (shifted by −c/T = ' + fx(-S.c / S.T) + ')' : ''}</span>`;
      html += `<span>∂E/∂ŷ at a <b>${fx(dEdy(S.a))}</b></span>`;
      roA.innerHTML = html;
      roCost.innerHTML = `<span>Z here: <b>1,201</b> energy evaluations (1-D grid)</span><span>100 points per axis in d dims: 100<sup>d</sup> = 10<sup>2d</sup></span><span>d = 2: <b>10<sup>4</sup></b></span><span>d = 3,136 (video latent): <b>10<sup>6,272</sup></b></span><span>d = 50,277 (text logits): <b>10<sup>100,554</sup></b></span>`;
      roCost.style.display = S.showCost ? 'flex' : 'none';
    }
    function draw() {
      if (S.view !== 'energy') return; const P = probs(); drawEnergy(); drawProb(P); readoutA(P);
      resetBtn.style.display = S.src === 'sculpt' ? '' : 'none';
      FP.wrap.querySelector('.fig-sub').innerHTML = S.pmode === 'norm' ? 'p(ŷ | x) = e<sup>−E/T</sup> / Z(x) · total area 1' : 'unnormalized e<sup>−E/T</sup> · total area Z(x)';
    }
    function setSrc(v) {
      S.src = v; srcSeg.set(v); rebuild(); [S.a, S.b] = SRC[v].cand; S.hover = -1;
      FE.wrap.querySelector('.fig-sub').innerHTML = v === 'sculpt' ? 'E<sub>θ</sub>(x, ŷ) for one fixed context x · drag a, b and the handles'
        : 'trained toy EBT: E(x = 0.25, (ŷ, 0)) · a real cross-section · drag a, b';
    }

    // energy / probability pointer interaction
    let drag = null;
    function pick(px, py, allowHandles) {
      if (allowHandles && S.src === 'sculpt') { for (let k = 0; k < KX.length; k++) if (Math.abs(X(KX[k]) - px) < 13 && Math.abs(Y(S.ky[k] + S.c) - py) < 15) return { kind: 'h', k }; }
      const da = Math.abs(X(S.a) - px), db = Math.abs(X(S.b) - px);
      return { kind: da <= db ? 'a' : 'b' };
    }
    function moveTo(d, px, py) {
      if (d.kind === 'h') { S.ky[d.k] = clamp(Yinv(py) - S.c, -2.8, 5.8); rebuild(); }
      else S[d.kind] = clamp(Xinv(px), XMIN + 0.02, XMAX - 0.02);
      draw();
    }
    [[cvE, true], [cvP, false]].forEach(([cv, handles]) => {
      cv.canvas.style.cursor = 'pointer';
      cv.canvas.addEventListener('pointerdown', (ev) => { const [px, py] = cv.toLocal(ev); drag = pick(px, py, handles); S.dimP = false; stopAll(); try { cv.canvas.setPointerCapture(ev.pointerId); } catch (_) { } moveTo(drag, px, py); });
      cv.canvas.addEventListener('pointermove', (ev) => {
        const [px, py] = cv.toLocal(ev);
        if (drag) { moveTo(drag, px, py); return; }
        if (handles && S.src === 'sculpt') { const p = pick(px, py, true), k = p.kind === 'h' ? p.k : -1; if (k !== S.hover) { S.hover = k; draw(); } cv.canvas.style.cursor = k >= 0 ? 'ns-resize' : 'ew-resize'; }
        else cv.canvas.style.cursor = 'ew-resize';
      });
      const up = () => { drag = null; }; cv.canvas.addEventListener('pointerup', up); cv.canvas.addEventListener('pointercancel', up);
    });

    // keyboard: ←/→ move candidate a (b with Shift) on the energy canvas
    cvE.canvas.tabIndex = 0; cvE.canvas.setAttribute('aria-label', 'Energy curve. Left and right arrow keys move candidate a; with Shift they move b.');
    cvE.canvas.addEventListener('keydown', (ev) => {
      const d = ev.key === 'ArrowLeft' ? -0.05 : ev.key === 'ArrowRight' ? 0.05 : 0; if (!d) return;
      ev.preventDefault(); ev.stopPropagation(); const k = ev.shiftKey ? 'b' : 'a'; S[k] = clamp(S[k] + d, XMIN + 0.02, XMAX - 0.02); S.dimP = false; draw();
    });
    // keyframed tween for the step demos: frames = [[from, to, seconds], ...]
    let tw = null;
    const tween = lib.loop((dt) => {
      if (!tw) return false; tw.t += dt; const [a, b, d] = tw.frames[tw.k]; const f = lib.ease(Math.min(1, tw.t / d)); tw.set(a + (b - a) * f);
      if (tw.t >= d) { tw.k++; tw.t = 0; if (tw.k >= tw.frames.length) { tw = null; return false; } }
      return true;
    });
    function animate(frames, set) { if (lib.reducedMotion) { set(frames[frames.length - 1][1]); return; } tw = { frames, k: 0, t: 0, set }; tween.start(); }

    // ---------------------------------------------------------------- view B: maze
    const mrow = h('div', { class: 'fig-row' }); viewB.appendChild(mrow);
    const FM = lib.frame(mrow, { label: 'Maze', sub: 'S → G · drag across cells to draw a path' }); FM.wrap.style.flex = '1 1 300px';
    const MW = 400, cvM = lib.canvas(FM.frame, MW, MW, { label: 'A random maze with the cells expanded by search or the path being verified' });
    const FC = lib.frame(mrow, { label: 'Cost vs maze size', sub: 'operations per maze · log–log' }); FC.wrap.style.flex = '1 1 230px';
    const CW = 300, cvC = lib.canvas(FC.frame, CW, 300, { label: 'Operations needed to search versus to verify, as the maze grows' });
    const mc1 = h('div', { class: 'controls' }); viewB.appendChild(mc1);
    mc1.appendChild(lib.button('search', () => runSearch(), { primary: true }));
    mc1.appendChild(lib.button('verify', () => runVerify()));
    const mc1b = h('div', { class: 'controls' }); viewB.appendChild(mc1b);
    mc1b.appendChild(h('span', { class: 'fig-label' }, 'candidate'));
    const candSeg = lib.segmented({ label: 'Candidate path', options: [['correct', 'correct path'], ['wall', 'through a wall'], ['user', 'your path']], value: 'correct', onchange: (v) => { MZ.cand = v; MZ.anim = null; MZ.verdict = null; drawMaze(); readoutB(); } });
    mc1b.appendChild(candSeg.el);
    const mc2 = h('div', { class: 'controls' }); viewB.appendChild(mc2);
    const slN = lib.slider({ id: 'energy-n', label: 'maze size n × n', min: 5, max: 41, step: 2, value: 15, fmt: (v) => v + ' × ' + v, oninput: (v) => { MZ.n = v; makeMaze(); } });
    mc2.appendChild(slN.el);
    mc2.appendChild(lib.button('new maze', () => { MZ.seed++; makeMaze(); }));
    const roB = h('div', { class: 'readout', 'aria-live': 'polite' }); viewB.appendChild(roB);

    function mrng(seed) { let a = seed >>> 0; return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
    // Wilson's algorithm: a uniformly random spanning tree of the n×n grid. open[c] bits: 1 east, 2 south, 4 west, 8 north
    function wilson(n, seed) {
      const r = mrng(seed), N = n * n, inT = new Uint8Array(N), open = new Uint8Array(N), nxt = new Int32Array(N), dir = new Int32Array(N);
      const nb = (c) => { const x = c % n, y = (c / n) | 0, o = []; if (x < n - 1) o.push([c + 1, 1, 4]); if (y < n - 1) o.push([c + n, 2, 8]); if (x > 0) o.push([c - 1, 4, 1]); if (y > 0) o.push([c - n, 8, 2]); return o; };
      inT[(r() * N) | 0] = 1;
      for (let s = 0; s < N; s++) {
        if (inT[s]) continue; let c = s;
        while (!inT[c]) { const o = nb(c), k = o[(r() * o.length) | 0]; nxt[c] = k[0]; dir[c] = k[1] * 16 + k[2]; c = k[0]; }
        c = s; while (!inT[c]) { open[c] |= dir[c] >> 4; open[nxt[c]] |= dir[c] & 15; inT[c] = 1; c = nxt[c]; }
      }
      return open;
    }
    function bfs(n, open) {
      const N = n * n, prev = new Int32Array(N).fill(-1), q = [0], order = []; prev[0] = 0; let qi = 0;
      while (qi < q.length) { const c = q[qi++]; order.push(c); if (c === N - 1) break; const o = open[c]; for (const [bit, d] of [[1, 1], [2, n], [4, -1], [8, -n]]) if (o & bit) { const t = c + d; if (prev[t] < 0) { prev[t] = c; q.push(t); } } }
      const path = [N - 1]; while (path[path.length - 1] !== 0) path.push(prev[path[path.length - 1]]); path.reverse();
      return { order, path };
    }
    const adjBit = (n, a, b) => b === a + 1 && a % n !== n - 1 ? 1 : b === a + n ? 2 : b === a - 1 && a % n !== 0 ? 4 : b === a - n ? 8 : 0;
    function wallShortcut(n, open, path) {
      let best = null;
      for (let i = 0; i < path.length; i++) for (let j = i + 6; j < path.length; j++) { const bit = adjBit(n, path[i], path[j]); if (bit && !(open[path[i]] & bit) && (!best || j - i > best[1] - best[0])) best = [i, j]; }
      if (!best) return path.slice(0, Math.max(2, path.length - 3)); // fall back: a path that stops short of G
      return path.slice(0, best[0] + 1).concat(path.slice(best[1]));
    }
    // verifier: one pass, one local check per move, then "ends at G"
    function verify(n, open, p) {
      const checks = [];
      for (let i = 1; i < p.length; i++) { const bit = adjBit(n, p[i - 1], p[i]); const ok = !!bit && !!(open[p[i - 1]] & bit); checks.push(ok); if (!ok) return { checks, ok: false, why: 'wall between move ' + (i - 1) + ' and ' + i, at: i }; }
      if (p[0] !== 0) return { checks, ok: false, why: 'does not start at S', at: 0 };
      if (p[p.length - 1] !== n * n - 1) return { checks, ok: false, why: 'does not end at G', at: p.length - 1 };
      return { checks, ok: true, at: -1 };
    }
    const MZ = { n: 15, seed: 3, open: null, sol: null, wall: null, cand: 'correct', user: [0], anim: null, verdict: null, searched: null };
    function candPath() { return MZ.cand === 'correct' ? MZ.sol.path : MZ.cand === 'wall' ? MZ.wall : MZ.user; }
    function makeMaze() {
      MZ.open = wilson(MZ.n, MZ.seed * 7919 + MZ.n); MZ.sol = bfs(MZ.n, MZ.open); MZ.wall = wallShortcut(MZ.n, MZ.open, MZ.sol.path);
      MZ.user = [0]; MZ.anim = null; MZ.verdict = null; MZ.searched = null; drawMaze(); drawCost(); readoutB();
    }
    const geo = () => { const pad = 14, cs = (MW - 2 * pad) / MZ.n; return { pad, cs, cx: (c) => pad + (c % MZ.n + 0.5) * cs, cy: (c) => pad + (((c / MZ.n) | 0) + 0.5) * cs }; };
    function drawMaze() {
      if (S.view !== 'maze') return;
      const c = cvM.ctx, n = MZ.n, { pad, cs, cx, cy } = geo(); cvM.clear();
      // expanded cells
      const A = MZ.anim;
      const nExp = A && A.kind === 'search' ? Math.floor(A.k) : MZ.searched ? MZ.sol.order.length : 0;
      for (let i = 0; i < nExp; i++) { const cell = MZ.sol.order[i], recent = A && A.kind === 'search' && i > nExp - Math.max(4, n); c.fillStyle = recent ? C.blue3 : C.blue4; c.fillRect(pad + (cell % n) * cs, pad + ((cell / n) | 0) * cs, cs + 0.5, cs + 0.5); }
      // walls
      c.strokeStyle = C.ink; c.lineWidth = n > 31 ? 1 : 1.3; c.lineCap = 'square'; c.beginPath();
      for (let cell = 0; cell < n * n; cell++) {
        const x0 = pad + (cell % n) * cs, y0 = pad + ((cell / n) | 0) * cs;
        if (cell % n < n - 1 && !(MZ.open[cell] & 1)) { c.moveTo(x0 + cs, y0); c.lineTo(x0 + cs, y0 + cs); }
        if (((cell / n) | 0) < n - 1 && !(MZ.open[cell] & 2)) { c.moveTo(x0, y0 + cs); c.lineTo(x0 + cs, y0 + cs); }
      }
      c.stroke(); c.lineWidth = 1.6; c.strokeRect(pad, pad, MW - 2 * pad, MW - 2 * pad);
      const poly = (p, col, w, dash) => { if (p.length < 2) return; c.save(); c.strokeStyle = col; c.lineWidth = w; c.lineJoin = 'round'; c.lineCap = 'round'; if (dash) c.setLineDash(dash); c.beginPath(); p.forEach((q, i) => i ? c.lineTo(cx(q), cy(q)) : c.moveTo(cx(q), cy(q))); c.stroke(); c.restore(); };
      // found path after search
      if (MZ.searched && !(A && A.kind === 'search')) poly(MZ.sol.path, C.blue, Math.max(1.6, Math.min(3, cs * 0.25)));
      // candidate being verified (or shown)
      const showCand = MZ.cand === 'user' || (A && A.kind === 'verify') || MZ.verdict;
      if (showCand) {
        const p = candPath(), V = MZ.verdict, upto = A && A.kind === 'verify' ? Math.floor(A.k) : (V ? V.checks.length : -1);
        poly(p, 'rgba(17,17,17,0.35)', Math.max(1.2, cs * 0.14), [3, 3]);
        if (upto >= 0) {
          const ok = p.slice(0, Math.min(p.length, upto + 1)); const failAt = V && !V.ok && V.at > 0 && upto >= V.at ? V.at : -1;
          poly(failAt > 0 ? p.slice(0, failAt) : ok, C.blue, Math.max(1.8, Math.min(3, cs * 0.25)));
          if (failAt > 0) { const a = p[failAt - 1], b = p[failAt], mx = (cx(a) + cx(b)) / 2, my = (cy(a) + cy(b)) / 2, r = Math.max(4, cs * 0.3); c.save(); c.strokeStyle = C.bad; c.lineWidth = 2.2; c.beginPath(); c.moveTo(mx - r, my - r); c.lineTo(mx + r, my + r); c.moveTo(mx + r, my - r); c.lineTo(mx - r, my + r); c.stroke(); c.restore(); }
          const head = p[Math.min(p.length - 1, failAt > 0 ? failAt - 1 : upto)]; if (head !== n * n - 1) lib.dot(c, cx(head), cy(head), Math.max(3, cs * 0.22), C.blue, { stroke: '#fff', lw: 1.2 });
        } else if (MZ.cand === 'user') { const e = p[p.length - 1]; lib.dot(c, cx(e), cy(e), Math.max(3, cs * 0.22), C.ink); }
      }
      const fs = clamp(cs * 0.55, 9, 14);
      label(c, 'S', cx(0), cy(0), { size: fs, weight: 700, align: 'center' });
      label(c, 'G', cx(n * n - 1), cy(n * n - 1), { size: fs, weight: 700, align: 'center' });
    }
    // cost curves: computed live on 6 random mazes per size
    const SIZES = [5, 7, 9, 13, 17, 21, 25, 33, 41]; let costs = null;
    function computeCosts() {
      costs = SIZES.map(n => { let e = 0, l = 0; for (let s = 0; s < 6; s++) { const o = wilson(n, 1000 + s * 31 + n), r = bfs(n, o); e += r.order.length; l += r.path.length - 1; } return [n, e / 6, l / 6]; });
      const fit = (k) => { const xs = costs.map(r => Math.log(r[0])), ys = costs.map(r => Math.log(r[k])), mx = xs.reduce((a, b) => a + b) / xs.length, my = ys.reduce((a, b) => a + b) / ys.length; let a = 0, b = 0; xs.forEach((x, i) => { a += (x - mx) * (ys[i] - my); b += (x - mx) * (x - mx); }); return a / b; };
      costs.slopes = [fit(1), fit(2)];
    }
    function drawCost() {
      if (S.view !== 'maze') return; if (!costs) computeCosts();
      const c = cvC.ctx; cvC.clear();
      const ax = lib.axes(c, { x: 48, y: 30, w: CW - 62, h: 210, xlim: [4.5, 46], ylim: [3, 3000], xlog: true, ylog: true, xticks: [5, 10, 20, 40], yticks: [10, 100, 1000], yfmt: (v) => v >= 1000 ? '1k' : String(v), xlabel: 'maze side n', size: 11 });
      lib.plot(c, ax, costs.map(r => [r[0], r[1]]), { color: C.ink, width: 1.6, markers: 2.6 });
      lib.plot(c, ax, costs.map(r => [r[0], r[2]]), { color: C.blue, width: 1.8, markers: 2.6 });
      label(c, 'search (cells) · slope ' + costs.slopes[0].toFixed(2), ax.X(5) + 2, 42, { size: 11 });
      label(c, 'verify (checks) · slope ' + costs.slopes[1].toFixed(2), ax.X(5) + 2, 58, { size: 11, color: C.blue });
      c.save(); c.strokeStyle = 'rgba(17,17,17,0.4)'; c.setLineDash([3, 3]); c.lineWidth = 1; c.beginPath(); c.moveTo(ax.X(MZ.n), 30); c.lineTo(ax.X(MZ.n), 240); c.stroke(); c.restore();
      lib.dot(c, ax.X(MZ.n), ax.Y(MZ.sol.order.length), 4, '#fff', { stroke: C.ink, lw: 1.5 });
      lib.dot(c, ax.X(MZ.n), ax.Y(Math.max(3, MZ.sol.path.length - 1)), 4, '#fff', { stroke: C.blue, lw: 1.5 });
      lib.text(c, 'this maze', ax.X(MZ.n) + 4, 226, { size: 10.5, kind: 'mono', color: C.muted });
      lib.text(c, 'operations', 6, 8, { size: 11, kind: 'mono', color: C.muted });
    }
    function readoutB() {
      const n = MZ.n, N = n * n, A = MZ.anim, parts = [];
      const nExp = A && A.kind === 'search' ? Math.floor(A.k) : MZ.searched ? MZ.sol.order.length : null;
      parts.push(nExp == null ? `<span>search: <b>not run</b></span>` : `<span>search (BFS): <b>${big(nExp)}</b> of ${big(N)} cells expanded${nExp >= MZ.sol.order.length ? ' (' + Math.round(100 * nExp / N) + '%)' : ''}</span>`);
      const p = candPath();
      if (A && A.kind === 'verify') parts.push(`<span>verify: <b>${Math.floor(A.k)}</b> of ${p.length - 1} checks</span>`);
      else if (MZ.verdict) parts.push(`<span>verify: <b>${MZ.verdict.checks.length}</b> check${MZ.verdict.checks.length === 1 ? '' : 's'} in one pass → <b style="color:${MZ.verdict.ok ? 'var(--blue)' : 'var(--warn)'}">${MZ.verdict.ok ? 'valid' : 'invalid: ' + MZ.verdict.why}</b></span>`);
      else parts.push(`<span>verify: <b>not run</b> (candidate has ${p.length - 1} moves)</span>`);
      if (MZ.searched && MZ.verdict && MZ.cand === 'correct') parts.push(`<span>search / verify <b>${(MZ.sol.order.length / Math.max(1, MZ.verdict.checks.length)).toFixed(1)}×</b></span>`);
      roB.innerHTML = parts.join('');
    }
    const ops = () => Math.max(110, MZ.n * MZ.n / 2.6); // same operations-per-second for both, so animation time ∝ work
    const mloop = lib.loop((dt) => {
      const A = MZ.anim; if (!A) return false;
      A.k += dt * ops();
      if (A.kind === 'search' && A.k >= MZ.sol.order.length) { MZ.searched = true; MZ.anim = null; if (A.then) { A.then(); return true; } }
      else if (A.kind === 'verify' && A.k >= A.total) { MZ.anim = null; }
      drawMaze(); readoutB(); drawCost(); return !!MZ.anim;
    });
    function runSearch(then) { MZ.searched = false; MZ.anim = { kind: 'search', k: 0, then }; if (lib.reducedMotion) { MZ.anim = null; MZ.searched = true; if (then) then(); drawMaze(); readoutB(); return; } mloop.start(); }
    function runVerify() {
      const p = candPath(); MZ.verdict = verify(MZ.n, MZ.open, p);
      MZ.anim = { kind: 'verify', k: 0, total: MZ.verdict.ok ? p.length - 1 : MZ.verdict.checks.length };
      if (lib.reducedMotion) { MZ.anim = null; drawMaze(); readoutB(); return; } mloop.start();
    }
    // draw your own path: drag across cells (moves to any 4-neighbour, walls are not enforced while drawing)
    let mdrag = false;
    const cellAt = (ev) => { const [px, py] = cvM.toLocal(ev), { pad, cs } = geo(), i = Math.floor((px - pad) / cs), j = Math.floor((py - pad) / cs); return i >= 0 && j >= 0 && i < MZ.n && j < MZ.n ? j * MZ.n + i : -1; };
    function extend(cell) {
      if (cell < 0) return; const p = MZ.user, last = p[p.length - 1]; if (cell === last) return;
      if (p.length > 1 && cell === p[p.length - 2]) p.pop(); else if (adjBit(MZ.n, last, cell) || adjBit(MZ.n, cell, last)) p.push(cell); else return;
      if (MZ.anim && MZ.anim.kind === 'search') MZ.searched = true; MZ.anim = null; MZ.verdict = null; drawMaze(); readoutB();
    }
    cvM.canvas.style.cursor = 'crosshair';
    cvM.canvas.addEventListener('pointerdown', (ev) => { mdrag = true; if (MZ.cand !== 'user') { MZ.cand = 'user'; candSeg.set('user'); MZ.verdict = null; } try { cvM.canvas.setPointerCapture(ev.pointerId); } catch (_) { } const cell = cellAt(ev); if (cell === 0) { MZ.user = [0]; } extend(cell); drawMaze(); readoutB(); });
    cvM.canvas.addEventListener('pointermove', (ev) => { if (mdrag) extend(cellAt(ev)); });
    const mup = () => { if (!mdrag) return; mdrag = false; if (MZ.user.length > 1) runVerify(); };
    cvM.canvas.addEventListener('pointerup', mup); cvM.canvas.addEventListener('pointercancel', mup);

    // ---------------------------------------------------------------- views, captions, steps
    const CAP_A = 'Top: energy of each candidate ŷ for one fixed context (blue curve; lower = more compatible). Bottom: the probability it implies. Candidate a is filled, b is hollow; "basin" percentages are the probability mass between the two ridges around each candidate.';
    const CAP_B = 'Maze: light blue cells were expanded by breadth-first search; blue line: a path. Verification walks a candidate once and stops at the first illegal move (red ×). Right: work needed by each, averaged over 6 fresh random mazes per size.';
    function setView(v) {
      S.view = v; viewSeg.set(v);
      viewA.style.display = v === 'energy' ? 'grid' : 'none'; viewB.style.display = v === 'maze' ? 'grid' : 'none';
      ctx.setCaption(v === 'energy' ? CAP_A : CAP_B);
      if (v === 'maze') { if (!MZ.open) makeMaze(); else { drawMaze(); drawCost(); readoutB(); } } else draw();
    }
    function stopAll() { tw = null; tween.stop(); }
    const setT = (v) => { S.T = v; slT.set(+v.toFixed(2)); draw(); };
    const setC = (v) => { S.c = v; slC.set(+v.toFixed(2)); draw(); };
    const fresh = () => { S.ky = KY0.slice(); setSrc('sculpt'); };
    setSrc('sculpt'); setView('energy');
    return {
      step(i) {
        stopAll();
        if (i <= 2) setView('energy');
        if (i === 0) { fresh(); S.T = 1; slT.set(1); S.c = 0; slC.set(0); S.pmode = 'norm'; pSeg.set('norm'); S.dimP = true; S.showCost = false; draw(); }
        if (i === 1) {
          fresh(); S.c = 0; slC.set(0); S.pmode = 'norm'; pSeg.set('norm'); S.dimP = false; S.showCost = false; S.T = 1; slT.set(1); draw();
          animate([[1, 0.35, 1.2], [0.35, 0.35, 0.7], [0.35, 2.5, 1.6], [2.5, 2.5, 0.7], [2.5, 1, 1.1]], setT);
        }
        if (i === 2) { S.c = 0; slC.set(0); S.T = 1; slT.set(1); S.pmode = 'unnorm'; pSeg.set('unnorm'); S.dimP = false; S.showCost = true; draw(); animate([[0, 0, 0.6], [0, 1, 1.1]], setC); }
        if (i === 3) { setView('maze'); MZ.cand = 'correct'; candSeg.set('correct'); MZ.verdict = null; runSearch(() => setTimeout(() => { if (ctx.step === 3 && !MZ.anim) runVerify(); }, 500)); }
      },
      show() { if (S.view === 'maze') drawCost(); else draw(); },
      hide() { stopAll(); mloop.stop(); if (MZ.anim) { if (MZ.anim.kind === 'search') MZ.searched = true; MZ.anim = null; } },
    };
  },
});
