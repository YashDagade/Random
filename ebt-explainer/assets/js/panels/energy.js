/* Panel 03: what an energy function is.
   View A: a 1-D energy E(x, ŷ) for one fixed context (sculptable, a real slice of the toy EBT, or a maximum-likelihood
   fit to thin data) and the Boltzmann distribution it implies, with temperature, a constant shift, the partition function
   and the cost of computing it. View B: a maze where generating a path needs search and verifying one needs a single pass.
   Everything is computed live: spline / toy-model energies, log-sum-exp normalization, exact KDE gradients, Wilson mazes, BFS. */
EBT.panel({
  id: 'energy',
  nav: 'What an energy is',
  title: 'What an energy function is',
  lede: 'An energy-based model does not output an answer. It outputs one number that scores how well a candidate answer fits the context. Lower is better, and everything else in the paper is built on that scalar.',
  text: `
    <p>An Energy-Based Transformer takes two inputs: a context $x$ (the tokens or frames seen so far) and a candidate prediction $\\hat y$ (a guess at what comes next). It returns one real number, the <b>energy</b> $E_\\theta(x,\\hat y)\\in\\mathbb R$, where $\\theta$ are the network weights. Lower energy means $x$ and $\\hat y$ are more compatible (p.5). The figure fixes one context, so the energy is an ordinary function of $\\hat y$, drawn with a one-dimensional $\\hat y$.</p>
    <p>Energies can be read as probabilities through the Boltzmann distribution, which exponentiates and normalizes them:</p>
    <div class="eq energy-eqs">$$p_\\theta(\\hat y\\mid x)=\\frac{e^{-E_\\theta(x,\\hat y)/T}}{Z_\\theta(x)}$$$$Z_\\theta(x)=\\int e^{-E_\\theta(x,\\hat y')/T}\\,d\\hat y'$$<span class="why">Sec 3.1 (p.6) and App. H.3 (p.41), written for one context x. The paper uses T = 1; T is added here to show what the exponent does.</span></div>
    <p>$Z_\\theta(x)$ is the <b>partition function</b>: the total mass under $e^{-E/T}$. Dividing by it makes the probabilities integrate to one. The minus sign is why low energy means high probability.</p>`,
  steps: [
    { label: 'One number per candidate', html: '<p>Drag the candidates <b>a</b> and <b>b</b> along the axis, or reshape the curve with its handles. Each candidate gets one energy, from one forward pass. The energy scores answers but does not produce one: to predict, the model has to search for a low point. The arrow at <b>a</b> is $-\\partial E/\\partial\\hat y$, the downhill direction that EBT thinking follows. Switch the curve to <b>toy EBT slice</b> to see a cross-section of a real trained model (the next section shows all of it).</p>' },
    { label: 'Energy becomes probability', html: '<p>The lower frame shows $p(\\hat y\\mid x)$. Only differences in energy, measured in units of $T$, shape it: a gap $\\Delta E$ makes one candidate $e^{\\Delta E/T}$ times more likely than another. Lower $T$ piles the mass into the deepest basin; higher $T$ flattens it. A basin that is low but narrow can hold less mass than a shallower wide one.</p>' },
    { label: 'The partition function', html: '<p>To normalize, you need $Z$: the shaded area. On this 1-D grid it took 1,201 energy evaluations. A grid with only 100 points per axis needs $100^d$ evaluations in $d$ dimensions. An EBT\'s text prediction is a logit vector with $d=50{,}277$ (Table D.3) and a video prediction is a 3,136-dimensional latent (p.12). The paper calls $Z$ "intractable" (p.6), and the readout shows why.</p>' },
    { label: 'Relative energies are enough', html: '<p>Slide <b>shift c</b>. Adding a constant to every energy multiplies $Z$ by $e^{-c/T}$, but the probabilities, the ratio $p(a)/p(b)=e^{-(E(a)-E(b))/T}$ and the gradient $\\nabla_{\\hat y}E$ do not change. Ranking candidates and descending the landscape never need $Z$. So EBTs are <b>unnormalized</b>: they "dispense of the partition function in favor of representing relative unnormalized probabilities" (p.6). The flip side: one context\'s energies can sit at any offset, so raw energies from two different contexts are not directly comparable (panel 06).</p>' },
    { label: 'Thin data breaks likelihood', html: '<p>Real data occupy a thin manifold. <b>thin data</b> fits $p$ to four targets by maximum likelihood with a kernel of width $w$; likelihood keeps rewarding smaller $w$. Off the data the energy then heads to $+\\infty$, and the walls get so steep (curvature $1/w^2$) that gradient descent is stable only for step sizes $\\alpha<2w^2$. Shrink $w$ and watch descent from <b>a</b> bounce out of control. The paper avoids this by asking only for low energy on the data and higher energy elsewhere, with walls that stay traversable (p.6, p.41).</p>' },
    { label: 'Checking is cheaper than finding', html: '<p>"Verifying the correctness of a given path is significantly easier than discovering such a path" (Sec 2.1, p.5). Press <b>search</b>: breadth-first search expands most of the maze before it reaches G. Press <b>verify</b>: checking a given path walks it once, one local test per move. Draw your own path, through walls if you like, and the verifier catches it. The same three-line check works on any maze size, which is the paper\'s point about verifiers generalizing (p.6).</p>' },
  ],
  after: `
    <h3>Why the verifier view matters</h3>
    <p>An EBT is a verifier: one forward pass scores one candidate. It becomes a generator by optimizing against that score, so "the generator is defined implicitly by the gradient of the verifier" (p.6). Generation costs many passes (thinking), checking costs one (self-verification). Because the energy is never normalized, training cannot be maximum likelihood; the paper instead trains the energy through the very optimization it will be used for (Algorithm 1, panels 10 and 11).</p>
    <p>Read literally, the toy slice's $e^{-E}$ is a bump about 1 unit wide, yet the targets for that context have noise $\\sigma=0.03$. Nothing in its training asked the energy to be a calibrated density. Energies are learned "implicitly" (p.41); what they encode is a separate question (panel 06).</p>
    <p class="note">The 1-D curves are illustrations, except "toy EBT slice": the trained toy model of the next section along ŷ₂ = 0 for context x = 0.25. Mazes are uniform random spanning trees (Wilson's algorithm). The paper's "exponentially easier" (p.3) refers to complexity theory, where checking can be polynomial while finding is not known to be. In mazes the gap is polynomial: search expands a fixed fraction of the n² cells, while the solution length grows roughly like n^1.25 (a classical result for loop-erased random walks), and the fitted slopes in the cost plot show the same thing.</p>`,
  source: [
    { kind: 'concept', note: '1-D energies, thin-data fit, maze' },
    { kind: 'toy', note: '"toy EBT slice" = trained toy 2-D EBT, x = 0.25, ŷ₂ = 0' },
    { kind: 'paper', note: 'Sec 2.1, Sec 3.1, App. H.3, Table D.3' },
  ],
  figure(stage, ctx) {
    const { lib } = ctx, h = lib.h, C = lib.C;
    const T2 = window.EBT && window.EBT.toy2d;
    const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
    const fx = (v, d = 2) => !isFinite(v) ? '–' : (v < 0 && Math.abs(v) >= 0.5 * Math.pow(10, -d) ? '−' : '') + Math.abs(v).toFixed(d);
    const big = (v) => Math.round(v).toLocaleString('en-US');

    // ---------------------------------------------------------------- 1-D energies
    const XMIN = -3, XMAX = 3, NG = 1201, DX = (XMAX - XMIN) / (NG - 1);
    const GX = Float64Array.from({ length: NG }, (_, i) => XMIN + i * DX);
    const KX = [-3, -2.25, -1.5, -0.75, 0, 0.75, 1.5, 2.25, 3], KH = 0.75;
    const KY0 = [5, 1.5, -0.95, -0.7, 1.8, 0.5, -0.25, 1.4, 5];
    const THIN = [-1.7, -0.6, 0.3, 1.5], ALPHA_THIN = 0.05, THIN_STEPS = 14;
    const SRC = {
      sculpt: { name: 'sculpt', yr: [-4, 7], cand: [-1.15, 1.33] },
      toy: { name: 'toy EBT slice', yr: [-2.5, 10], cand: [1.3, 0.25] },
      thin: { name: 'thin data', yr: [-3.5, 12], cand: [-2.25, 0.9] },
    };
    const S = { view: 'energy', src: 'sculpt', T: 1, c: 0, w: 0.12, a: -1.15, b: 1.33, pmode: 'norm', dimP: false, showCost: false, ky: KY0.slice(), hover: -1 };
    let EB = new Float64Array(NG); // base energy (no shift)
    let toyCache = null;

    function spline(v) {
      let i = clamp(Math.floor((v - KX[0]) / KH), 0, KX.length - 2); const t = (v - KX[i]) / KH, y = S.ky, L = y.length;
      const p1 = y[i], p2 = y[i + 1], p0 = i > 0 ? y[i - 1] : 2 * p1 - p2, p3 = i + 2 < L ? y[i + 2] : 2 * p2 - p1;
      return 0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t + (-p0 + 3 * p1 - 3 * p2 + p3) * t * t * t);
    }
    // maximum-likelihood energy of a Gaussian kernel density with width w: E = −log p_w(ŷ) (exact, with log-sum-exp)
    function thinE(v) {
      const w = S.w; let m = -Infinity; const z = THIN.map(t => -((v - t) * (v - t)) / (2 * w * w)); z.forEach(q => { if (q > m) m = q; });
      const lse = m + Math.log(z.reduce((s, q) => s + Math.exp(q - m), 0));
      return -(lse - Math.log(THIN.length) - Math.log(Math.sqrt(2 * Math.PI) * w));
    }
    function thinG(v) {
      const w = S.w; const z = THIN.map(t => -((v - t) * (v - t)) / (2 * w * w)); const m = Math.max(...z);
      const wt = z.map(q => Math.exp(q - m)), sw = wt.reduce((a, b) => a + b, 0);
      return THIN.reduce((s, t, k) => s + wt[k] * (v - t), 0) / (sw * w * w);
    }
    function rebuild() {
      if (S.src === 'sculpt') for (let i = 0; i < NG; i++) EB[i] = spline(GX[i]);
      else if (S.src === 'thin') for (let i = 0; i < NG; i++) EB[i] = thinE(GX[i]);
      else {
        if (!toyCache) { toyCache = new Float64Array(NG); const K = T2.nCkpt - 1; for (let i = 0; i < NG; i++) toyCache[i] = T2.energy(K, 0.25, [GX[i], 0]); }
        EB = toyCache.slice();
      }
    }
    const idx = (v) => clamp((v - XMIN) / DX, 0, NG - 1);
    const Ebase = (v) => { if (S.src === 'thin') return thinE(v); const f = idx(v), i = Math.min(NG - 2, Math.floor(f)), t = f - i; return EB[i] * (1 - t) + EB[i + 1] * t; };
    const Eat = (v) => Ebase(v) + S.c;
    const dEdy = (v) => { if (S.src === 'thin') return thinG(v); const e = 0.01; return (Ebase(v + e) - Ebase(v - e)) / (2 * e); };
    // probabilities on the grid (log-sum-exp; Riemann sum with spacing DX)
    function probs() {
      const T = S.T; let m = -Infinity; for (let i = 0; i < NG; i++) { const q = -(EB[i] + S.c) / T; if (q > m) m = q; }
      let s = 0; for (let i = 0; i < NG; i++) s += Math.exp(-(EB[i] + S.c) / T - m);
      const logZ = m + Math.log(s * DX), p = new Float64Array(NG); for (let i = 0; i < NG; i++) p[i] = Math.exp(-(EB[i] + S.c) / T - logZ);
      // unnormalized scale is fixed by the unshifted curve so that the shift visibly changes e^{-E/T}
      let mb = -Infinity; for (let i = 0; i < NG; i++) mb = Math.max(mb, -EB[i] / T);
      return { p, logZ, uMaxBase: Math.exp(mb) };
    }
    function thinDescent() {
      let y = S.a; const pts = [y];
      for (let i = 0; i < THIN_STEPS; i++) { y = y - ALPHA_THIN * thinG(y); if (!isFinite(y) || Math.abs(y) > 60) { pts.push(y > 0 ? 60 : -60); break; } pts.push(y); }
      return pts;
    }

    // ---------------------------------------------------------------- DOM
    const viewRow = h('div', { class: 'controls' }); stage.appendChild(viewRow);
    viewRow.appendChild(h('span', { class: 'fig-label' }, 'view'));
    const viewSeg = lib.segmented({ label: 'View', options: [['energy', 'energy ↔ probability'], ['maze', 'verify vs generate']], value: 'energy', onchange: (v) => setView(v) });
    viewRow.appendChild(viewSeg.el);
    const viewA = h('div', { style: { display: 'grid', gap: '16px' } }); stage.appendChild(viewA);
    const viewB = h('div', { style: { display: 'none', gap: '16px' } }); stage.appendChild(viewB);

    const FE = lib.frame(viewA, { label: 'Energy', sub: 'E<sub>θ</sub>(x, ŷ) for one fixed context x · drag a, b and the handles' });
    // logical canvas width = displayed width, so canvas labels keep their pixel size on phones
    const EW = clamp(Math.floor(FE.frame.clientWidth || 600), 300, 660), narrow = EW < 480;
    const EH = narrow ? 214 : 236, cvE = lib.canvas(FE.frame, EW, EH, { label: 'Energy of every candidate prediction for one fixed context, with two candidates a and b' });
    const FP = lib.frame(viewA, { label: 'Probability', sub: 'p(ŷ | x) = e<sup>−E/T</sup> / Z(x)' });
    const PH = narrow ? 160 : 178, cvP = lib.canvas(FP.frame, EW, PH, { label: 'The Boltzmann probability implied by the energy above' });

    const c1 = h('div', { class: 'controls' }); viewA.appendChild(c1);
    c1.appendChild(h('span', { class: 'fig-label' }, 'energy'));
    const srcSeg = lib.segmented({ label: 'Energy source', options: [['sculpt', 'sculpt'], ['toy', 'toy EBT slice'], ['thin', 'thin data']].filter(o => o[0] !== 'toy' || (T2 && T2.ready)), value: 'sculpt', onchange: (v) => { setSrc(v); draw(); } });
    c1.appendChild(srcSeg.el);
    c1.appendChild(h('span', { class: 'fig-label', style: { marginLeft: '6px' } }, 'lower frame'));
    const pSeg = lib.segmented({ label: 'Probability frame', options: [['norm', 'p (normalized)'], ['unnorm', 'e^(−E/T), area Z']], value: 'norm', onchange: (v) => { S.pmode = v; S.dimP = false; draw(); } });
    c1.appendChild(pSeg.el);
    const c2 = h('div', { class: 'controls' }); viewA.appendChild(c2);
    const slT = lib.slider({ id: 'energy-T', label: 'temperature T', min: 0.25, max: 3, step: 0.05, value: 1, fmt: (v) => v.toFixed(2), oninput: (v) => { S.T = v; S.dimP = false; draw(); } });
    const slC = lib.slider({ id: 'energy-c', label: 'shift c (add to every E)', min: -1.5, max: 1.5, step: 0.05, value: 0, fmt: (v) => (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(2), oninput: (v) => { S.c = v; draw(); } });
    const slW = lib.slider({ id: 'energy-w', label: 'data width w', min: 0.04, max: 0.5, step: 0.01, value: S.w, fmt: (v) => v.toFixed(2), oninput: (v) => { S.w = v; rebuild(); draw(); } });
    c2.append(slT.el, slC.el, slW.el);
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

    function curvePath(c, arr, Yf, top, bot) {
      c.beginPath(); let pen = false;
      for (let i = 0; i < NG; i += 2) { const px = X(GX[i]), py = Yf(arr(i)); if (!isFinite(py)) { pen = false; continue; } const cy = clamp(py, top - 4, bot + 4); if (!pen) { c.moveTo(px, cy); pen = true; } else c.lineTo(px, cy); }
    }
    function drawEnergy() {
      const c = cvE.ctx; cvE.clear(); const [lo, hi] = yr();
      const ax = lib.axes(c, { x: PL, y: PT, w: EW - PL - PR, h: EH - PT - PB, xlim: [XMIN, XMAX], ylim: [lo, hi], xticks: [-3, -2, -1, 0, 1, 2, 3], yticks: ticks(lo, hi), xfmt: (v) => fx(v, 0), yfmt: (v) => fx(v, 0), size: 11 });
      void ax;
      lib.text(c, 'ŷ', EW - PR, EH - PB - 16, { size: 13, kind: 'mono', color: C.ink, align: 'right' });
      lib.text(c, 'E', PL + 6, PT + 2, { size: 13, kind: 'mono', color: C.ink });
      c.save(); c.beginPath(); c.rect(PL, PT, EW - PL - PR, EH - PT - PB); c.clip();
      // ghost of the unshifted curve
      if (Math.abs(S.c) > 0.004) {
        curvePath(c, (i) => EB[i], Y, PT, EH - PB); c.strokeStyle = 'rgba(17,17,17,0.45)'; c.lineWidth = 1; c.setLineDash([4, 4]); c.stroke(); c.setLineDash([]);
        const xm = 2.55, y0 = Y(Ebase(xm)), y1 = Y(Ebase(xm) + S.c);
        lib.arrow(c, X(xm), y0, X(xm), y1, { color: C.ink, width: 1.2, head: 6 });
        lib.text(c, (S.c > 0 ? '+' : '−') + 'c', X(xm) - 6, (y0 + y1) / 2, { size: 12, kind: 'mono', color: C.ink, align: 'right', baseline: 'middle' });
      }
      // thin data: the data points
      if (S.src === 'thin') THIN.forEach(t => { c.strokeStyle = C.ink; c.lineWidth = 1.2; c.beginPath(); c.moveTo(X(t), EH - PB); c.lineTo(X(t), EH - PB - 9); c.stroke(); });
      curvePath(c, (i) => EB[i] + S.c, Y, PT, EH - PB); c.strokeStyle = C.blue; c.lineWidth = 2; c.stroke();
      // candidates
      [['b', S.b], ['a', S.a]].forEach(([nm, v]) => {
        const e = Eat(v), py = clamp(Y(e), PT, EH - PB);
        c.strokeStyle = 'rgba(17,17,17,0.35)'; c.lineWidth = 1; c.setLineDash([2, 3]); c.beginPath(); c.moveTo(X(v), EH - PB); c.lineTo(X(v), py); c.stroke(); c.setLineDash([]);
        lib.dot(c, X(v), py, 5.5, nm === 'a' ? C.blue : '#fff', { stroke: nm === 'a' ? '#fff' : C.ink, lw: 1.6 });
        lib.text(c, nm, X(v) + 8, py - 18, { size: 13, kind: 'mono', color: nm === 'a' ? C.blue : C.ink, weight: 700 });
      });
      // downhill arrow at a, or the descent iterates in thin mode
      if (S.src !== 'thin') {
        const g = dEdy(S.a), py = Y(Eat(S.a)), L = clamp(Math.abs(g) * 22, 0, 70);
        if (L > 6) { const d = -Math.sign(g); lib.arrow(c, X(S.a), py + 14, X(S.a) + d * L, py + 14, { color: C.blue, width: 1.5, head: 7 }); lib.text(c, '−∂E/∂ŷ', X(S.a) + d * (L + 4), py + 14, { size: 11, kind: 'mono', color: C.blue, align: d > 0 ? 'left' : 'right', baseline: 'middle' }); }
        else lib.text(c, 'flat: ∂E/∂ŷ ≈ 0', X(S.a) + 10, py + 10, { size: 11, kind: 'mono', color: C.blue });
      } else {
        const pts = thinDescent().slice(0, 9), top = PT + 4;
        const Q = pts.map(p => [X(clamp(p, XMIN, XMAX)), clamp(Y(Eat(p)), top, EH - PB - 3), p < XMIN || p > XMAX, Y(Eat(p)) < top]);
        c.lineWidth = 1; c.strokeStyle = 'rgba(47,60,255,0.6)';
        for (let i = 1; i < Q.length; i++) { const a0 = Q[i - 1], a1 = Q[i], mx = (a0[0] + a1[0]) / 2, lift = Math.min(46, 8 + 0.18 * Math.abs(a1[0] - a0[0])); c.beginPath(); c.moveTo(a0[0], a0[1]); c.quadraticCurveTo(mx, Math.max(top, Math.min(a0[1], a1[1]) - lift), a1[0], a1[1]); c.stroke(); }
        Q.forEach((q, i) => { const bad = q[2] || q[3]; lib.dot(c, q[0], q[1], i === 0 ? 0 : 3.2, bad ? C.bad : '#fff', { stroke: bad ? C.bad : C.blue, lw: 1.3 }); if (i > 0) lib.text(c, String(i), q[0] + 5, q[1] + 3, { size: 10.5, kind: 'mono', color: bad ? C.bad : C.blue }); });
        if (Q.some(q => q[3])) lib.text(c, 'red = above the plotted range', PL + 8, EH - PB - 18, { size: 10.5, kind: 'mono', color: C.bad });
      }
      c.restore();
      // handles
      if (S.src === 'sculpt') KX.forEach((kx, k) => { const py = Y(S.ky[k] + S.c); if (py < PT - 2 || py > EH - PB + 2) return; lib.dot(c, X(kx), py, S.hover === k ? 6.5 : 4.5, '#fff', { stroke: C.blue, lw: 1.5 }); });
    }
    function drawProb(P) {
      const c = cvP.ctx; cvP.clear();
      const top = 14, bot = PH - 26, norm = S.pmode === 'norm';
      let ymax;
      if (norm) { let m = 0; for (let i = 0; i < NG; i++) m = Math.max(m, P.p[i]); ymax = m * 1.18; }
      else ymax = P.uMaxBase * 1.18;
      const Yp = (v) => top + (1 - v / ymax) * (bot - top);
      c.save(); if (S.dimP) c.globalAlpha = 0.22;
      const yt = ticks(0, ymax).filter(v => v <= ymax * 0.97);
      lib.axes(c, { x: PL, y: top, w: EW - PL - PR, h: bot - top, xlim: [XMIN, XMAX], ylim: [0, ymax], xticks: [-3, -2, -1, 0, 1, 2, 3], yticks: yt, xfmt: (v) => fx(v, 0), yfmt: (v) => v >= 10 ? v.toFixed(0) : v >= 1 ? v.toFixed(1) : v.toFixed(2), size: 11 });
      const val = (i) => norm ? P.p[i] : Math.exp(-(EB[i] + S.c) / S.T);
      c.save(); c.beginPath(); c.rect(PL, top, EW - PL - PR, bot - top); c.clip();
      c.beginPath(); c.moveTo(X(XMIN), Yp(0)); for (let i = 0; i < NG; i += 2) c.lineTo(X(GX[i]), Math.max(top - 2, Yp(val(i)))); c.lineTo(X(XMAX), Yp(0)); c.closePath();
      c.fillStyle = norm ? C.blue4 : C.blue3; c.fill();
      c.beginPath(); for (let i = 0; i < NG; i += 2) { const py = Math.max(top - 2, Yp(val(i))); i ? c.lineTo(X(GX[i]), py) : c.moveTo(X(GX[i]), py); } c.strokeStyle = C.blue; c.lineWidth = 1.8; c.stroke();
      if (!norm && Math.abs(S.c) > 0.004) { c.beginPath(); for (let i = 0; i < NG; i += 2) { const py = Math.max(top - 2, Yp(Math.exp(-EB[i] / S.T))); i ? c.lineTo(X(GX[i]), py) : c.moveTo(X(GX[i]), py); } c.strokeStyle = 'rgba(17,17,17,0.45)'; c.lineWidth = 1; c.setLineDash([4, 4]); c.stroke(); c.setLineDash([]); }
      [['b', S.b], ['a', S.a]].forEach(([nm, v]) => { const f = idx(v), i = Math.round(f); const py = Math.max(top, Yp(val(i))); c.strokeStyle = 'rgba(17,17,17,0.35)'; c.lineWidth = 1; c.setLineDash([2, 3]); c.beginPath(); c.moveTo(X(v), bot); c.lineTo(X(v), py); c.stroke(); c.setLineDash([]); lib.dot(c, X(v), py, 4.5, nm === 'a' ? C.blue : '#fff', { stroke: nm === 'a' ? '#fff' : C.ink, lw: 1.5 }); });
      c.restore();
      // clipped indicator in unnormalized mode
      let maxU = 0; if (!norm) for (let i = 0; i < NG; i++) maxU = Math.max(maxU, val(i));
      const lab = norm ? 'area = 1' : 'area = Z = ' + (Math.exp(P.logZ) < 1e4 ? Math.exp(P.logZ).toFixed(3) : Math.exp(P.logZ).toExponential(2));
      lib.text(c, lab, EW - PR - 6, top + 4, { size: 12, kind: 'mono', color: C.ink, align: 'right' });
      if (!norm && Math.abs(S.c) > 0.004) lib.text(c, 'dashed: c = 0 · same shape × e^(−c/T) = ' + Math.exp(-S.c / S.T).toFixed(2) + ' → same p after ÷ Z', EW - PR - 6, top + 20, { size: 11, kind: 'mono', color: C.ink, align: 'right' });
      else if (!norm && maxU > ymax) lib.text(c, 'peak off scale (×' + (maxU / P.uMaxBase).toFixed(1) + ')', EW - PR - 6, top + 20, { size: 11, kind: 'mono', color: C.bad, align: 'right' });
      lib.text(c, norm ? 'p(ŷ|x)' : 'e^(−E/T)', PL + 6, top, { size: 12, kind: 'mono', color: C.ink });
      lib.text(c, 'ŷ', EW - PR, bot - 16, { size: 13, kind: 'mono', color: C.ink, align: 'right' });
      c.restore();
      if (S.dimP) lib.text(c, 'step 2 turns these energies into probabilities', EW / 2, (top + bot) / 2, { size: 13, kind: 'mono', color: C.muted, align: 'center', baseline: 'middle' });
    }
    function readoutA(P) {
      const Ea = Eat(S.a), Eb = Eat(S.b), dE = Ea - Eb;
      const pa = P.p[Math.round(idx(S.a))], pb = P.p[Math.round(idx(S.b))];
      const viaE = Math.exp(-dE / S.T), viaP = pa / pb;
      const r = (v) => v >= 1e4 || v < 1e-3 ? v.toExponential(2) : v.toFixed(3);
      let html = `<span>E(a) <b>${fx(Ea)}</b></span><span>E(b) <b>${fx(Eb)}</b></span>` +
        `<span>p(a)/p(b) <b>${r(viaP)}</b> = e<sup>−ΔE/T</sup> <b>${r(viaE)}</b> (no Z needed)</span>` +
        `<span>Z <b>${Math.exp(P.logZ) < 1e5 ? Math.exp(P.logZ).toFixed(3) : Math.exp(P.logZ).toExponential(2)}</b></span><span>log Z <b>${fx(P.logZ, 3)}</b></span>`;
      if (S.src === 'thin') {
        const pts = thinDescent(), last = pts[pts.length - 1], ok = 2 * S.w * S.w > ALPHA_THIN;
        const settled = Math.abs(pts[pts.length - 1] - pts[pts.length - 2]) < 1e-3;
        html += `<span>wall curvature 1/w² <b>${big(1 / (S.w * S.w))}</b></span><span>descent stable only if α &lt; 2w² = <b>${(2 * S.w * S.w).toFixed(3)}</b></span>` +
          `<span>here α = ${ALPHA_THIN} → <b>${ok && settled ? 'settles at ŷ = ' + fx(last) : (Math.abs(last) >= 60 ? 'diverges' : 'still bouncing after ' + THIN_STEPS + ' steps')}</b></span>`;
      } else html += `<span>∂E/∂ŷ at a <b>${fx(dEdy(S.a))}</b></span>`;
      roA.innerHTML = html;
      roCost.innerHTML = `<span>Z here: <b>1,201</b> energy evaluations (1-D grid)</span><span>100 points per axis in d dims: 100<sup>d</sup> = 10<sup>2d</sup></span><span>d = 2: <b>10<sup>4</sup></b></span><span>d = 3,136 (video latent): <b>10<sup>6,272</sup></b></span><span>d = 50,277 (text logits): <b>10<sup>100,554</sup></b></span>`;
      roCost.style.display = S.showCost ? 'flex' : 'none';
    }
    function draw() { if (S.view !== 'energy') return; const P = probs(); drawEnergy(); drawProb(P); readoutA(P); slW.el.style.display = S.src === 'thin' ? '' : 'none'; }
    function setSrc(v) {
      S.src = v; srcSeg.set(v); rebuild(); [S.a, S.b] = SRC[v].cand; S.hover = -1;
      FE.wrap.querySelector('.fig-sub').innerHTML = v === 'sculpt' ? 'E<sub>θ</sub>(x, ŷ) for one fixed context x · drag a, b and the handles'
        : v === 'toy' ? 'trained toy EBT: E(x = 0.25, (ŷ, 0)) · a real cross-section · drag a, b'
          : 'maximum-likelihood energy −log p<sub>w</sub>(ŷ) of four data points (ticks) · descent from a, α = 0.05';
    }

    // energy / probability pointer interaction
    let drag = null;
    function pick(px, py, allowHandles) {
      if (allowHandles && S.src === 'sculpt') { for (let k = 0; k < KX.length; k++) if (Math.abs(X(KX[k]) - px) < 13 && Math.abs(Y(S.ky[k] + S.c) - py) < 15) return { kind: 'h', k }; }
      const da = Math.abs(X(S.a) - px), db = Math.abs(X(S.b) - px);
      return { kind: da <= db ? 'a' : 'b', jump: Math.min(da, db) > 18 };
    }
    function moveTo(d, px, py) {
      if (d.kind === 'h') { S.ky[d.k] = clamp(Yinv(py) - S.c, -3.3, 6.3); rebuild(); }
      else S[d.kind] = clamp(Xinv(px), XMIN + 0.02, XMAX - 0.02);
      draw();
    }
    [[cvE, true], [cvP, false]].forEach(([cv, handles]) => {
      cv.canvas.style.cursor = 'pointer';
      cv.canvas.addEventListener('pointerdown', (ev) => { const [px, py] = cv.toLocal(ev); drag = pick(px, py, handles); S.dimP = false; try { cv.canvas.setPointerCapture(ev.pointerId); } catch (_) { } moveTo(drag, px, py); });
      cv.canvas.addEventListener('pointermove', (ev) => {
        const [px, py] = cv.toLocal(ev);
        if (drag) { moveTo(drag, px, py); return; }
        if (handles && S.src === 'sculpt') { const p = pick(px, py, true), k = p.kind === 'h' ? p.k : -1; if (k !== S.hover) { S.hover = k; draw(); } cv.canvas.style.cursor = k >= 0 ? 'ns-resize' : 'ew-resize'; }
        else cv.canvas.style.cursor = 'ew-resize';
      });
      const up = () => { drag = null; }; cv.canvas.addEventListener('pointerup', up); cv.canvas.addEventListener('pointercancel', up);
    });

    // small tween helper (for the step demos)
    let tw = null;
    const tween = lib.loop((dt) => { if (!tw) return false; tw.t += dt; const f = lib.ease(Math.min(1, tw.t / tw.dur)); tw.set(tw.from + (tw.to - tw.from) * f); if (tw.t >= tw.dur) { tw = null; return false; } return true; });
    function animate(from, to, dur, set) { if (lib.reducedMotion) { set(to); return; } tw = { from, to, dur, t: 0, set }; tween.start(); }

    // ---------------------------------------------------------------- view B: maze
    const mrow = h('div', { class: 'fig-row' }); viewB.appendChild(mrow);
    const FM = lib.frame(mrow, { label: 'Maze', sub: 'S → G · drag across cells to draw a path' }); FM.wrap.style.flex = '1 1 300px';
    const MW = 400, cvM = lib.canvas(FM.frame, MW, MW, { label: 'A random maze with the cells expanded by search or the path being verified' });
    const FC = lib.frame(mrow, { label: 'Cost vs maze size', sub: 'operations per maze · mean of 6 mazes · log–log' }); FC.wrap.style.flex = '1 1 230px';
    const CW = 300, cvC = lib.canvas(FC.frame, CW, 300, { label: 'Operations needed to search versus to verify, as the maze grows' });
    const mc1 = h('div', { class: 'controls' }); viewB.appendChild(mc1);
    mc1.appendChild(lib.button('search', () => runSearch(), { primary: true }));
    mc1.appendChild(lib.button('verify', () => runVerify()));
    mc1.appendChild(h('span', { class: 'fig-label' }, 'candidate'));
    const candSeg = lib.segmented({ label: 'Candidate path', options: [['correct', 'correct path'], ['wall', 'through a wall'], ['user', 'your path']], value: 'correct', onchange: (v) => { MZ.cand = v; MZ.anim = null; MZ.verdict = null; drawMaze(); readoutB(); } });
    mc1.appendChild(candSeg.el);
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
          const head = p[Math.min(p.length - 1, failAt > 0 ? failAt - 1 : upto)]; lib.dot(c, cx(head), cy(head), Math.max(3, cs * 0.22), C.blue, { stroke: '#fff', lw: 1.2 });
        } else if (MZ.cand === 'user') { const e = p[p.length - 1]; lib.dot(c, cx(e), cy(e), Math.max(3, cs * 0.22), C.ink); }
      }
      const fs = clamp(cs * 0.55, 9, 14);
      lib.text(c, 'S', cx(0), cy(0), { size: fs, kind: 'mono', weight: 700, color: C.ink, align: 'center', baseline: 'middle' });
      lib.text(c, 'G', cx(n * n - 1), cy(n * n - 1), { size: fs, kind: 'mono', weight: 700, color: C.ink, align: 'center', baseline: 'middle' });
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
      lib.text(c, 'search · slope ' + costs.slopes[0].toFixed(2), ax.X(5) + 2, 36, { size: 11, kind: 'mono', color: C.ink });
      lib.text(c, 'verify · slope ' + costs.slopes[1].toFixed(2), ax.X(44), ax.Y(32), { size: 11, kind: 'mono', color: C.blue, align: 'right' });
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
      else if (MZ.verdict) parts.push(`<span>verify: <b>${MZ.verdict.checks.length}</b> checks in one pass → <b style="color:${MZ.verdict.ok ? 'var(--blue)' : 'var(--warn)'}">${MZ.verdict.ok ? 'valid' : 'invalid: ' + MZ.verdict.why}</b></span>`);
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
      MZ.anim = null; MZ.verdict = null; drawMaze(); readoutB();
    }
    cvM.canvas.style.cursor = 'crosshair';
    cvM.canvas.addEventListener('pointerdown', (ev) => { mdrag = true; if (MZ.cand !== 'user') { MZ.cand = 'user'; candSeg.set('user'); MZ.verdict = null; } try { cvM.canvas.setPointerCapture(ev.pointerId); } catch (_) { } const cell = cellAt(ev); if (cell === 0) { MZ.user = [0]; } extend(cell); drawMaze(); readoutB(); });
    cvM.canvas.addEventListener('pointermove', (ev) => { if (mdrag) extend(cellAt(ev)); });
    const mup = () => { if (!mdrag) return; mdrag = false; if (MZ.user.length > 1) runVerify(); };
    cvM.canvas.addEventListener('pointerup', mup); cvM.canvas.addEventListener('pointercancel', mup);

    // ---------------------------------------------------------------- views, captions, steps
    const CAP_A = 'Top: energy of each candidate ŷ for one fixed context (blue curve; lower = more compatible). Bottom: the probability it implies. Candidate a is filled, b is hollow.';
    const CAP_B = 'Maze: light blue cells were expanded by breadth-first search; blue line: a path. Verification walks a candidate once and stops at the first illegal move (red ×). Right: work needed by each, measured on fresh random mazes.';
    function setView(v) {
      S.view = v; viewSeg.set(v);
      viewA.style.display = v === 'energy' ? 'grid' : 'none'; viewB.style.display = v === 'maze' ? 'grid' : 'none';
      ctx.setCaption(v === 'energy' ? CAP_A : CAP_B);
      if (v === 'maze') { if (!MZ.open) makeMaze(); else { drawMaze(); drawCost(); readoutB(); } } else draw();
    }
    function stopAll() { tw = null; tween.stop(); }
    rebuild(); setView('energy');
    return {
      step(i) {
        stopAll();
        if (i <= 4) { setView('energy'); slW.el.style.display = 'none'; }
        if (i === 0) { setSrc('sculpt'); S.T = 1; slT.set(1); S.c = 0; slC.set(0); S.pmode = 'norm'; pSeg.set('norm'); S.dimP = true; S.showCost = false; draw(); }
        if (i === 1) {
          if (S.src === 'thin') setSrc('sculpt'); S.c = 0; slC.set(0); S.pmode = 'norm'; pSeg.set('norm'); S.dimP = false; S.showCost = false;
          animate(1, 0.45, 0.9, (v) => { S.T = v; slT.set(+v.toFixed(2)); draw(); });
          setTimeout(() => { if (ctx.step === 1) animate(0.45, 1, 0.9, (v) => { S.T = v; slT.set(+v.toFixed(2)); draw(); }); }, 1300);
        }
        if (i === 2) { if (S.src === 'thin') setSrc('sculpt'); S.c = 0; slC.set(0); S.T = 1; slT.set(1); S.pmode = 'unnorm'; pSeg.set('unnorm'); S.dimP = false; S.showCost = true; draw(); }
        if (i === 3) { if (S.src === 'thin') setSrc('sculpt'); S.T = 1; slT.set(1); S.pmode = 'unnorm'; pSeg.set('unnorm'); S.dimP = false; S.showCost = false; animate(S.c, 1, 1.1, (v) => { S.c = v; slC.set(+v.toFixed(2)); draw(); }); }
        if (i === 4) { setSrc('thin'); S.w = 0.12; slW.set(0.12); S.T = 1; slT.set(1); S.c = 0; slC.set(0); S.pmode = 'norm'; pSeg.set('norm'); S.dimP = false; S.showCost = false; draw(); }
        if (i === 5) { setView('maze'); MZ.cand = 'correct'; candSeg.set('correct'); MZ.verdict = null; runSearch(() => setTimeout(() => { if (ctx.step === 5 && !MZ.anim) runVerify(); }, 500)); }
      },
      show() { if (S.view === 'maze') drawCost(); },
      hide() { stopAll(); mloop.stop(); if (MZ.anim) { if (MZ.anim.kind === 'search') MZ.searched = true; MZ.anim = null; } },
    };
  },
});
