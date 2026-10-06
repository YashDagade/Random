/* Panel: why EBTs are trained by optimizing instead of contrasting (Sec 3.2, p.6–7; H.1, p.40; B.2, p.29; limitation p.17).
   Figure: (1) two 2-D energies trained live in the browser on the same data stream, one with a contrastive (InfoNCE) loss and
   random negatives, one with Algorithm 1 (unrolled gradient descent + MSE, exact second-order backprop); a verifier test and a
   thinking test on both. (2) The curse of dimensionality: exact chi-square probabilities and Monte Carlo samples of how far
   random negatives land from the data as the dimension grows. */
EBT.panel({
  id: 'contrastive',
  nav: 'Optimizing, not contrasting',
  title: 'Why train by optimizing instead of contrasting',
  lede: 'Data only says where energy should be <em>low</em>; something must push it up everywhere else. Contrastive training does that with sampled wrong answers and needs exponentially many of them as dimension grows. EBTs train the thinking process instead.',
  text: `
    <p>An energy function $E_\\theta(x,\\hat y)$ is useful only if it is low on compatible pairs and high elsewhere. A training set contains only the compatible pairs $(x, y)$. Lowering the energy there is easy; the hard part is making sure nothing <em>else</em> ends up just as low. A perfectly flat landscape assigns the data the lowest energy too, and it is useless. The paper names the two classic answers (p.6, citing LeCun): <b>contrastive</b> and <b>regularized</b> training.</p>
    <p><b>Contrastive</b> training samples $K$ negatives $\\hat y^-_1,\\dots,\\hat y^-_K$ and pushes their energy up while pushing the data's down. The left landscape in the figure is trained with this InfoNCE form:</p>
    <div class="eq">$$\\begin{gathered}\\mathcal L_{\\text{con}}(\\theta) = -\\log \\frac{e^{-E_\\theta(x,y)}}{Z(\\theta)}\\\\ Z(\\theta) = e^{-E_\\theta(x,y)} + \\textstyle\\sum_{j=1}^{K} e^{-E_\\theta(x,\\hat y^-_j)}\\end{gathered}$$<span class="why">Its gradient lowers $E$ at $y$ and raises $E$ at each negative, weighted by how likely the model currently finds it.</span></div>
    <p><b>Optimization-based</b> training (the paper's choice, Sec 3.2) uses no negatives. It runs the model's own descent from noise and scores only where it lands:</p>
    <div class="eq">$$\\begin{gathered}\\mathcal L_{\\text{opt}}(\\theta) = J\\big(\\hat y_N(\\theta),\\,y\\big)\\\\ \\hat y_{i+1} = \\hat y_i - \\alpha\\nabla_{\\hat y}E_\\theta(x,\\hat y_i),\\quad \\hat y_0\\sim\\mathcal N(0,I)\\end{gathered}$$<span class="why">Algorithm 1 (p.7). $J$ is MSE here; the right landscape is trained with exactly this, backpropagating through all $N$ steps.</span></div>`,
  steps: [
    { label: 'Same data, two recipes', html: '<p>Both landscapes start from the same small random weights and see the same targets (crosshair). Each is a real energy over a 2-D prediction $\\hat y$: 169 Gaussian bumps with learnable heights plus a weak bowl $0.12\\lVert\\hat y\\rVert^2$. Only the loss differs. Watch 300 Adam steps of each (press <b>[ train 300 ]</b> to repeat). The contrastive model digs a deep, narrow well at the data and raises a plateau wherever its negatives (×) fell. The Algorithm 1 model forms a wide funnel. Both end up good <b>verifiers</b>: the data has lower energy than about 100% of random candidates (readout).</p>' },
    { label: 'Now think on them', html: '<p>Start 24 guesses from $\\mathcal N(0,I)$ and run plain gradient descent, with whichever of three step sizes works best for each landscape. On the funnel, every start lands. On the contrastive landscape most starts slide off the plateau and never find the well. Contrastive training fixed energy <em>values</em> at sampled points; nothing asked the <em>slopes</em> in between to lead anywhere. A good verifier is not automatically a landscape you can think on.</p>' },
    { label: 'The curse of dimensionality', html: '<p>In 2-D a few thousand negatives blanket the square. In $d$ dimensions, random negatives land on a thin shell far from the data: the chance that one falls within half the typical distance shrinks roughly like $10^{-0.14\\,d}$. For one video-frame latent ($d=3136$, p.12) you would need about $10^{435}$ random negatives per hit near the data; for one token\'s logits ($d=50{,}277$, Table D.3) about $10^{6949}$. This is the paper\'s argument: contrastive methods "must increase the energy of an exponentially higher number of negative samples" (p.7).</p>' },
    { label: 'Where the optimization signal lands', html: '<p>Blue paths are the latest training unrolls of Algorithm 1. Its loss reaches $\\theta$ only through $\\nabla_{\\hat y}E$ at the points these paths visit (dots), so the landscape is shaped exactly where thinking will walk: from noise to the data. The paper calls this <em>implicitly regularizing</em> the landscape, which pushes it "to be convex surrounding the ground truth solution" (p.7). The cost per example is $N$ steps (2 to 3 in the paper, Table D.4), whatever the dimension.</p>' },
    { label: 'The price: one basin per context', html: '<p>Now each context has two equally likely answers. The contrastive model keeps two wells. Algorithm 1 with MSE sends every start to the point that minimizes expected loss, the <em>average</em> of the two answers, so it learns a single basin between them. This is the failure the paper reports: blurry, averaged images in text-to-image generation (B.2, p.29), and "EBTs currently struggle with data distributions that have many modes" (p.17).</p>' },
  ],
  after: `
    <h3>What "implicit regularization" means here</h3>
    <p>Every term of Algorithm 1's weight gradient touches the landscape only through $\\nabla_\\theta\\nabla_{\\hat y}E_\\theta(x,\\hat y_i)$ at a visited point $\\hat y_i$. No energy value is ever supervised (Fig E.1, p.37). So no negatives are needed, but regions that no training path visits stay unconstrained. The next panel covers the four tricks the paper uses to send training paths to more places.</p>
    <h3>Smarter negatives move the cost, they do not remove it</h3>
    <p>Practical contrastive EBMs draw negatives from the model itself with MCMC, or from a generator network as in a GAN (p.40). That aims the push-up at the model's own spurious low regions, but in high dimension it needs long sampling chains (the "long training times" of p.4) or an adversarial generator. The paper's GAN analogy (p.7) fits Algorithm 1 as well: the forward pass is the discriminator, the descent is the generator. Here one network plays both roles and one supervised loss trains both, which the authors credit with avoiding "adversarial issues" (p.6). The appendix also likens this training to denoising score matching (p.38).</p>
    <p class="note">Lab details: energy $E_\\theta(\\hat y)=\\sum_{k=1}^{169}\\theta_k e^{-\\lVert\\hat y-c_k\\rVert^2/2w^2}+0.12\\lVert\\hat y\\rVert^2$, centers $c_k$ on a 13×13 grid over [−3, 3]², $w=0.5$, Adam (lr 0.03, batch 16), targets with noise 0.08. Contrastive: $K$ negatives uniform in the square shown. Algorithm 1: $N$ steps, $\\alpha=0.5$, exact backprop through the unroll with 2×2 Hessians. Verifier test: 200 random candidates. Thinking test: 24 starts, up to 150 steps, best α of {0.5, 0.15, 0.04}. Dimension figure: target at the origin, negatives drawn like $\\hat y_0\\sim\\mathcal N(0,I_d)$; "near" = within $c\\sqrt d$; exact χ² probabilities. Illustrations, not the paper's models.</p>`,
  source: [{ kind: 'concept', note: '2-D energies trained live in your browser' }, { kind: 'concept', note: 'exact χ² volumes' }, { kind: 'paper', note: 'Sec 3.2 p.6–7, H.1 p.40, B.2 p.29, p.17' }],
  figure(stage, ctx) {
    const { lib } = ctx, h = lib.h, C = lib.C;
    // ---------------------------------------------------------------- RBF energy (local, real math)
    const G = 13, CE = 3, W = 0.5, LAM = 0.12, iw2 = 1 / (W * W), KB = G * G, EXT = 2.5;
    const CX = new Float64Array(KB), CY = new Float64Array(KB);
    for (let i = 0; i < G; i++) for (let j = 0; j < G; j++) { CX[i * G + j] = -CE + 2 * CE * j / (G - 1); CY[i * G + j] = -CE + 2 * CE * i / (G - 1); }
    const PH = new Float64Array(KB);
    const phi = (a, b) => { for (let k = 0; k < KB; k++) { const dx = a - CX[k], dy = b - CY[k]; PH[k] = Math.exp(-(dx * dx + dy * dy) * 0.5 * iw2); } return PH; };
    const energy = (th, a, b) => { phi(a, b); let s = LAM * (a * a + b * b); for (let k = 0; k < KB; k++) s += th[k] * PH[k]; return s; };
    const grad = (th, a, b) => { phi(a, b); let gx = 2 * LAM * a, gy = 2 * LAM * b; for (let k = 0; k < KB; k++) { const q = th[k] * PH[k] * iw2; gx -= q * (a - CX[k]); gy -= q * (b - CY[k]); } return [gx, gy]; };
    const hess = (th, a, b) => { let hxx = 2 * LAM, hyy = 2 * LAM, hxy = 0; for (let k = 0; k < KB; k++) { const q = th[k] * PH[k]; if (q === 0) continue; const dx = a - CX[k], dy = b - CY[k]; hxx += q * (dx * dx * iw2 * iw2 - iw2); hyy += q * (dy * dy * iw2 * iw2 - iw2); hxy += q * dx * dy * iw2 * iw2; } return [hxx, hxy, hyy]; }; // uses PH from the last phi()
    const clampY = (v) => Math.max(-4, Math.min(4, v));
    // precomputed features on the display grid and on fixed verifier candidates
    const NG = 44, GP = new Float32Array(NG * NG * KB);
    for (let r = 0; r < NG; r++) for (let c = 0; c < NG; c++) { const a = -EXT + 2 * EXT * c / (NG - 1), b = EXT - 2 * EXT * r / (NG - 1); phi(a, b); const o = (r * NG + c) * KB; for (let k = 0; k < KB; k++) GP[o + k] = PH[k]; }
    const vr = lib.rng(4242), CAND = Array.from({ length: 200 }, () => [(2 * vr() - 1) * EXT, (2 * vr() - 1) * EXT]);
    const T1 = [1.2, 0.6], MA = [-1.1, 0.9], MB = [1.2, -0.7], MID = [(MA[0] + MB[0]) / 2, (MA[1] + MB[1]) / 2];
    const targets = () => (S.mode === 'one' ? [T1] : [MA, MB]);
    const sampleY = (r) => { const t = S.mode === 'one' ? T1 : (r() < 0.5 ? MA : MB); return [t[0] + 0.08 * r.normal(), t[1] + 0.08 * r.normal()]; };

    // ---------------------------------------------------------------- models
    const S = { mode: 'one', K: 8, N: 4, alpha: 0.5, todo: 0, showThink: false, showSignal: false, d: 2, c: 0.5, dAnim: null, B: 16 };
    function makeModel(kind) {
      const th = new Float64Array(KB), r0 = lib.rng(5); for (let k = 0; k < KB; k++) th[k] = 0.05 * r0.normal();
      return { kind, th, m: new Float64Array(KB), v: new Float64Array(KB), t: 0, it: 0, loss: NaN, negs: [], pos: [], paths: [], test: null, verif: null, rng: lib.rng(kind === 'con' ? 11 : 12), E: null };
    }
    let Mc = makeModel('con'), Mo = makeModel('opt');
    function adam(M, g, lr) { M.t++; const b1 = 0.9, b2 = 0.999; for (let k = 0; k < KB; k++) { const gk = g[k] + 1e-4 * M.th[k]; M.m[k] = b1 * M.m[k] + (1 - b1) * gk; M.v[k] = b2 * M.v[k] + (1 - b2) * gk * gk; M.th[k] -= lr * (M.m[k] / (1 - Math.pow(b1, M.t))) / (Math.sqrt(M.v[k] / (1 - Math.pow(b2, M.t))) + 1e-8); } }
    function stepCon(M) { // InfoNCE with K uniform negatives
      const r = M.rng, g = new Float64Array(KB); let loss = 0; const negs = [], pos = [];
      for (let b = 0; b < S.B; b++) {
        const ys = [sampleY(r)]; for (let j = 0; j < S.K; j++) ys.push([(2 * r() - 1) * EXT, (2 * r() - 1) * EXT]);
        const es = ys.map(q => energy(M.th, q[0], q[1])); const mn = Math.min(...es); const ws = es.map(e => Math.exp(-(e - mn))); const Z = ws.reduce((s, v) => s + v, 0);
        loss += -Math.log(ws[0] / Z);
        ys.forEach((q, j) => { const cf = j === 0 ? (1 - ws[0] / Z) : -ws[j] / Z; phi(q[0], q[1]); for (let k = 0; k < KB; k++) g[k] += cf * PH[k] / S.B; });
        pos.push(ys[0]); for (let j = 1; j < ys.length; j++) negs.push(ys[j]);
      }
      adam(M, g, 0.03); M.it++; M.loss = loss / S.B;
      M.negs = M.negs.concat(negs).slice(-72); M.pos = M.pos.concat(pos).slice(-24);
    }
    function stepOpt(M) { // Algorithm 1: unroll N steps, MSE at the end, exact backprop through the unroll
      const r = M.rng, g = new Float64Array(KB); let loss = 0; const paths = [];
      for (let b = 0; b < S.B; b++) {
        const y = sampleY(r); let a = r.normal(), c = r.normal(); const path = [[a, c]];
        for (let i = 0; i < S.N; i++) { const gr = grad(M.th, a, c); a = clampY(a - S.alpha * gr[0]); c = clampY(c - S.alpha * gr[1]); path.push([a, c]); }
        let gy0 = 2 * (a - y[0]), gy1 = 2 * (c - y[1]); loss += (a - y[0]) ** 2 + (c - y[1]) ** 2;
        for (let i = S.N - 1; i >= 0; i--) {
          const p = path[i]; phi(p[0], p[1]);
          for (let k = 0; k < KB; k++) g[k] += S.alpha * PH[k] * iw2 * (gy0 * (p[0] - CX[k]) + gy1 * (p[1] - CY[k])) / S.B;
          const [hxx, hxy, hyy] = hess(M.th, p[0], p[1]);
          const n0 = gy0 - S.alpha * (hxx * gy0 + hxy * gy1), n1 = gy1 - S.alpha * (hxy * gy0 + hyy * gy1); gy0 = n0; gy1 = n1;
        }
        paths.push(path);
      }
      adam(M, g, 0.03); M.it++; M.loss = loss / S.B; M.paths = paths.slice(0, 10);
    }
    function gridE(M) { const E = []; let lo = Infinity, hi = -Infinity; const vals = new Float64Array(NG * NG);
      for (let i = 0; i < NG * NG; i++) { const o = i * KB; let s = 0; for (let k = 0; k < KB; k++) s += M.th[k] * GP[o + k]; const r = Math.floor(i / NG), c = i % NG; const a = -EXT + 2 * EXT * c / (NG - 1), b = EXT - 2 * EXT * r / (NG - 1); s += LAM * (a * a + b * b); vals[i] = s; if (s < lo) lo = s; if (s > hi) hi = s; }
      for (let r = 0; r < NG; r++) E.push(Array.from(vals.subarray(r * NG, (r + 1) * NG)));
      const sorted = Array.from(vals).sort((x, y) => x - y); M.E = { E, lo, hi, q: sorted[Math.floor(0.7 * (sorted.length - 1))] }; return M.E; }
    function verifierTest(M) { const T = targets(); let ok = 0, n = 0; T.forEach(t => { const et = energy(M.th, t[0], t[1]); CAND.forEach(q => { n++; if (et < energy(M.th, q[0], q[1])) ok++; }); }); M.verif = ok / n; }
    function thinkTest(M) { // 24 starts from N(0,I); best step size of three; up to 150 steps
      const T = targets(); let best = null;
      [0.5, 0.15, 0.04].forEach(al => {
        const r = lib.rng(99); const runs = [];
        for (let s = 0; s < 24; s++) {
          let a = r.normal(), c = r.normal(); const path = [[a, c]];
          for (let i = 0; i < 150; i++) { const gr = grad(M.th, a, c); const na = clampY(a - al * gr[0]), nc = clampY(c - al * gr[1]); const mv = Math.hypot(na - a, nc - c); a = na; c = nc; if (i % 3 === 2 || i < 6) path.push([a, c]); if (mv < 1e-5) break; }
          path.push([a, c]);
          const dT = Math.min(...T.map(t => Math.hypot(a - t[0], c - t[1]))), dM = Math.hypot(a - MID[0], c - MID[1]);
          runs.push({ path, ok: dT < 0.3, mid: S.mode === 'two' && dM < 0.3, d: dT });
        }
        const ok = runs.filter(q => q.ok).length, md = runs.reduce((s, q) => s + q.d, 0) / runs.length;
        if (!best || ok > best.ok || (ok === 0 && best.ok === 0 && md < best.md)) best = { alpha: al, runs, ok, md, mid: runs.filter(q => q.mid).length };
      });
      M.test = best;
    }

    // ---------------------------------------------------------------- layout: lab
    const row = h('div', { class: 'fig-row' }); stage.appendChild(row);
    const panes = [Mc, Mo].map((M, i) => {
      const fr = lib.frame(row, { label: i === 0 ? 'Contrastive' : 'Optimization · Algorithm 1', sub: i === 0 ? 'push data down, push K negatives up' : 'unroll N steps from noise, MSE at ŷ<sub>N</sub>' });
      fr.wrap.style.flex = '1 1 250px';
      const cv = lib.canvas(fr.frame, 300, 300, { label: i === 0 ? 'Energy landscape trained with a contrastive loss' : 'Energy landscape trained with Algorithm 1' });
      const ro = h('div', { class: 'readout ctr-ro' }); fr.wrap.appendChild(ro);
      return { cv, ro, fr };
    });
    const box = { x: 0, y: 0, w: 300, h: 300 };
    const P = (q) => [box.x + (q[0] + EXT) / (2 * EXT) * box.w, box.y + (EXT - q[1]) / (2 * EXT) * box.h];
    const controls = h('div', { class: 'controls' }); stage.appendChild(controls);
    const modeSeg = lib.segmented({ label: 'Targets', options: [['one', 'one answer'], ['two', 'two modes']], value: 'one', onchange: (v) => { S.mode = v; resetModels(); S.showSignal = true; queueTrain(300, think); draw(); } });
    const bTrain = lib.button('train 300', () => { resetModels(); S.showSignal = true; queueTrain(300); draw(); }, { primary: true });
    const bThink = lib.button('think', () => ensureTrained(think));
    const bReset = lib.button('reset', () => { resetModels(); draw(); });
    controls.append(bTrain, bThink, bReset, h('span', { class: 'fig-label' }, 'targets'), modeSeg.el);
    const kSl = lib.slider({ id: 'ctr-k', label: 'negatives per example K', min: 1, max: 32, step: 1, value: S.K, fmt: (v) => String(v), oninput: (v) => { S.K = v; } });
    const nSl = lib.slider({ id: 'ctr-n', label: 'unrolled steps N', min: 1, max: 6, step: 1, value: S.N, fmt: (v) => String(v), oninput: (v) => { S.N = v; } });
    const controls2 = h('div', { class: 'controls' }); stage.appendChild(controls2);
    controls2.append(kSl.el, nSl.el);

    // ---------------------------------------------------------------- layout: curse of dimensionality
    const row2 = h('div', { class: 'fig-row' }); stage.appendChild(row2);
    const F1 = lib.frame(row2, { label: 'Where random negatives land', sub: 'distance to the data ÷ √d, 2000 samples' }); F1.wrap.style.flex = '1 1 250px';
    const hc = lib.canvas(F1.frame, 300, 180, { label: 'Histogram of distances from random negatives to the data in d dimensions' });
    const F2 = lib.frame(row2, { label: 'Negatives needed per hit', sub: 'random draws per one that lands near' }); F2.wrap.style.flex = '1 1 250px';
    const lc = lib.canvas(F2.frame, 300, 180, { label: 'Number of random negatives needed per hit near the data, versus dimension' });
    const fmtInt = (n) => n.toLocaleString('en-US');
    const controls3 = h('div', { class: 'controls' }); stage.appendChild(controls3);
    const dSl = lib.slider({ id: 'ctr-d', label: 'dimension d', min: 0, max: Math.log10(50277), step: 0.005, value: Math.log10(2), fmt: (v) => fmtInt(Math.round(Math.pow(10, v))), oninput: (v) => { S.dAnim = null; S.d = Math.round(Math.pow(10, v)); drawCurse(); } });
    const cSl = lib.slider({ id: 'ctr-c', label: '"near" = within c·√d, c', min: 0.2, max: 0.9, step: 0.05, value: S.c, fmt: (v) => v.toFixed(2), oninput: (v) => { S.c = v; drawCurse(); } });
    controls3.append(dSl.el, cSl.el);
    const ro3 = h('div', { class: 'readout' }); stage.appendChild(ro3);

    // ---------------------------------------------------------------- math for the curse figure
    function lgamma(z) { // Lanczos, z > 0
      const g = 7, c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
      if (z < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * z)) - lgamma(1 - z);
      z -= 1; let x = c[0]; for (let i = 1; i < g + 2; i++) x += c[i] / (z + i); const t = z + g + 0.5;
      return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x);
    }
    function log10ChiCdf(d, t) { // log10 P(chi2_d <= t) via the series of the regularized lower incomplete gamma
      const a = d / 2, x = t / 2; let term = 1, sum = 1;
      for (let n = 1; n < 5000; n++) { term *= x / (a + n); sum += term; if (term < sum * 1e-15) break; }
      return (a * Math.log(x) - x - lgamma(a + 1) + Math.log(sum)) / Math.LN10;
    }
    function gammaSample(r, a) { // Marsaglia–Tsang, scale 1
      if (a < 1) return gammaSample(r, a + 1) * Math.pow(Math.max(1e-12, r()), 1 / a);
      const d = a - 1 / 3, c = 1 / Math.sqrt(9 * d);
      for (;;) { let x, v; do { x = r.normal(); v = 1 + c * x; } while (v <= 0); v = v * v * v; const u = r(); if (u < 1 - 0.0331 * x * x * x * x) return d * v; if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v; }
    }
    let mc = { d: -1, r: [] };
    function samples(d) { if (mc.d === d) return mc.r; const r = lib.rng(777 + d); const out = new Float64Array(2000); for (let i = 0; i < 2000; i++) out[i] = Math.sqrt(2 * gammaSample(r, d / 2) / d); mc = { d, r: out }; return out; }
    const SUP = (n) => String(n).split('').map(ch => ({ '-': '⁻', '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹' })[ch] || ch).join('');

    // ---------------------------------------------------------------- drawing
    function drawTargets(c) {
      targets().forEach(t => { const p = P(t); c.save(); c.lineWidth = 3; c.strokeStyle = 'rgba(17,17,17,0.35)'; c.beginPath(); c.moveTo(p[0] - 8, p[1]); c.lineTo(p[0] + 8, p[1]); c.moveTo(p[0], p[1] - 8); c.lineTo(p[0], p[1] + 8); c.stroke();
        c.lineWidth = 1.4; c.strokeStyle = '#fff'; c.stroke(); c.setLineDash([3, 3]); c.beginPath(); c.arc(p[0], p[1], 13, 0, 7); c.stroke(); c.restore(); });
      if (S.mode === 'two') { const p = P(MID); c.save(); c.strokeStyle = 'rgba(17,17,17,0.55)'; c.setLineDash([2, 3]); c.lineWidth = 1; c.beginPath(); c.arc(p[0], p[1], 9, 0, 7); c.stroke(); c.restore(); plate(c, 'average', p[0] + 11, p[1] - 8); }
    }
    function drawPane(M, pn) {
      const c = pn.cv.ctx; pn.cv.clear();
      const g = M.E || gridE(M);
      lib.heatmap(c, g.E, 0, 0, 300, 300, { range: [g.lo, g.q], gamma: 0.75 });
      const levels = Array.from({ length: 10 }, (_, i) => g.lo + (g.q - g.lo) * Math.pow((i + 1) / 10, 1.3));
      lib.contours(c, g.E, 0, 0, 300, 300, levels, { color: 'rgba(17,17,17,0.22)', width: 1 });
      const isOpt = M.kind === 'opt', col = isOpt ? C.blue : '#111';
      const training = S.todo > 0;
      if (!S.showThink && (training || S.showSignal)) {
        if (!isOpt) {
          M.negs.forEach((q, i) => { const p = P(q); const al = 0.25 + 0.6 * i / M.negs.length; c.save(); c.globalAlpha = al; c.strokeStyle = '#111'; c.lineWidth = 1.2; c.beginPath(); c.moveTo(p[0] - 3, p[1] - 3); c.lineTo(p[0] + 3, p[1] + 3); c.moveTo(p[0] - 3, p[1] + 3); c.lineTo(p[0] + 3, p[1] - 3); c.stroke(); c.restore(); });
          M.pos.forEach(q => { const p = P(q); lib.dot(c, p[0], p[1], 2, '#fff', { alpha: 0.9 }); });
        } else {
          M.paths.forEach(path => { const px = path.map(P); lib.line(c, px, { color: C.blue, width: 1.3, alpha: 0.8 }); px.slice(0, -1).forEach(p => lib.dot(c, p[0], p[1], 2.4, '#fff', { stroke: C.blue, lw: 1.2 })); const e = px[px.length - 1]; lib.dot(c, e[0], e[1], 3, C.blue); });
        }
      }
      if (S.showThink && M.test) {
        M.test.runs.forEach(run => { const px = run.path.map(P); lib.line(c, px, { color: col, width: 1.1, alpha: 0.55 }); const e0 = px[px.length - 1], e = [Math.max(5, Math.min(295, e0[0])), Math.max(5, Math.min(295, e0[1]))];
          if (run.ok) lib.dot(c, e[0], e[1], 3.4, col, { stroke: '#fff', lw: 1 });
          else { c.save(); c.strokeStyle = isOpt ? C.blue : '#111'; c.lineWidth = 1.6; c.beginPath(); c.moveTo(e[0] - 4, e[1] - 4); c.lineTo(e[0] + 4, e[1] + 4); c.moveTo(e[0] - 4, e[1] + 4); c.lineTo(e[0] + 4, e[1] - 4); c.stroke(); c.restore(); }
          lib.dot(c, px[0][0], px[0][1], 2, '#fff', { stroke: col, lw: 1 }); });
      }
      drawTargets(c);
      plate(c, 'iter ' + M.it, 6, 294 - 16);
      // readout
      const parts = ['<span>iter <b>' + M.it + '</b></span>', '<span>loss <b>' + (isFinite(M.loss) ? M.loss.toFixed(3) : '–') + '</b></span>'];
      if (M.verif != null) parts.push('<span>verifier <b>' + Math.round(M.verif * 100) + '%</b></span>');
      if (M.test) { parts.push('<span>thinking <b>' + M.test.ok + '/24</b> ' + (S.mode === 'two' ? 'on a mode' : 'reach it') + ' (α ' + M.test.alpha + ')</span>'); if (S.mode === 'two') parts.push('<span>at the average <b>' + M.test.mid + '/24</b></span>'); }
      pn.ro.innerHTML = parts.join('');
    }
    function plate(c, s, x, y) { c.save(); c.font = '11px "JetBrains Mono", monospace'; const w = c.measureText(s).width + 10; c.fillStyle = 'rgba(255,255,255,0.85)'; c.fillRect(x, y, w, 16); c.restore(); lib.text(c, s, x + 5, y + 2, { size: 11, kind: 'mono', color: '#111' }); }
    function draw() { drawPane(Mc, panes[0]); drawPane(Mo, panes[1]); }

    function drawCurse() {
      const d = S.d, c0 = S.c, rs = samples(d);
      // histogram
      { const c = hc.ctx; hc.clear(); const nb = 40, bins = new Array(nb).fill(0); let inside = 0;
        let beyond = 0; rs.forEach(v => { if (v <= c0) inside++; if (v >= 2) { beyond++; return; } bins[Math.floor(v / 2 * nb)]++; });
        const mx = Math.max(...bins);
        const ax = lib.axes(c, { x: 34, y: 12, w: 254, h: 128, xlim: [0, 2], ylim: [0, mx * 1.08], xticks: [0, 0.5, 1, 1.5, 2], yticks: [], xfmt: (v) => String(v), size: 11 });
        c.fillStyle = C.blue4; c.fillRect(ax.X(0), 12, ax.X(c0) - ax.X(0), 128);
        bins.forEach((n, i) => { if (!n) return; const x0 = ax.X(i * 2 / nb), x1 = ax.X((i + 1) * 2 / nb), y = ax.Y(n); c.fillStyle = (i + 0.5) * 2 / nb <= c0 ? C.blue : '#9a9aa0'; c.fillRect(x0 + 0.5, y, x1 - x0 - 1, ax.Y(0) - y); });
        c.strokeStyle = C.blue; c.lineWidth = 1.2; c.setLineDash([4, 3]); c.beginPath(); c.moveTo(ax.X(c0), 12); c.lineTo(ax.X(c0), 140); c.stroke(); c.setLineDash([]);
        lib.text(c, 'near the data', ax.X(0) + 4, 14, { size: 11, kind: 'mono', color: C.blue });
        c.fillStyle = 'rgba(255,255,255,0.85)'; c.fillRect(170, 12, 118, 32);
        lib.text(c, inside + ' of 2000 inside', 286, 14, { size: 11, kind: 'mono', color: '#111', align: 'right' });
        lib.text(c, 'd = ' + fmtInt(d) + (beyond ? ' · ' + beyond + ' beyond 2' : ''), 286, 30, { size: 11, kind: 'mono', color: '#6b6b70', align: 'right' });
        lib.text(c, '‖ŷ⁻ − y‖ / √d', 161, 160, { size: 11, kind: 'mono', color: '#6b6b70', align: 'center' });
        S._inside = inside;
      }
      // negatives needed per hit (log10 of the expected count), on a log axis of exponents
      { const c = lc.ctx; lc.clear();
        const yl = [0.1, 30000];
        const ax = lib.axes(c, { x: 62, y: 12, w: 218, h: 128, xlim: [1, 1e5], ylim: yl, xlog: true, ylog: true, xticks: [1, 10, 100, 1e3, 1e4, 1e5], yticks: [1, 10, 100, 1000, 10000], xfmt: (v) => (v >= 1000 ? (v / 1000) + 'k' : String(v)), yfmt: (v) => '10' + SUP(v), size: 11 });
        const ex = (dd) => Math.max(yl[0], -log10ChiCdf(dd, c0 * c0 * dd));
        const pts = []; for (let s = 0; s <= 5.0001; s += 0.05) { const dd = Math.pow(10, s); pts.push([dd, ex(dd)]); }
        lib.plot(c, ax, pts, { color: '#111', width: 1.6 });
        // Algorithm 1: N = 2–3 steps per example, flat in d
        c.save(); c.strokeStyle = C.blue; c.setLineDash([5, 4]); c.lineWidth = 1.4; c.beginPath(); c.moveTo(ax.X(1), ax.Y(Math.log10(3))); c.lineTo(ax.X(1e5), ax.Y(Math.log10(3))); c.stroke(); c.restore();
        lib.text(c, 'Alg 1: N = 2–3 steps, any d', ax.X(1e5) - 2, ax.Y(Math.log10(3)) - 14, { size: 10.5, kind: 'mono', color: C.blue, align: 'right' });
        [[3136, 'video frame'], [50277, 'text token']].forEach(([dd, lab]) => { const X = ax.X(dd); c.save(); c.strokeStyle = 'rgba(17,17,17,0.3)'; c.setLineDash([2, 3]); c.beginPath(); c.moveTo(X, 12); c.lineTo(X, 140); c.stroke(); c.restore(); const e = ex(dd); lib.dot(c, X, ax.Y(e), 2.6, '#111'); lib.text(c, lab, X - 6, ax.Y(e) - 5, { size: 10.5, kind: 'mono', color: '#33333a', align: 'right' }); });
        const e = ex(d); lib.dot(c, ax.X(d), ax.Y(e), 4.5, C.blue, { stroke: '#fff', lw: 1.5 });
        lib.text(c, 'dimension d', 175, 160, { size: 11, kind: 'mono', color: '#6b6b70', align: 'center' });
      }
      const L = -log10ChiCdf(d, c0 * c0 * d);
      const pTxt = L < 3 ? Math.pow(10, -L).toPrecision(2) : '10' + SUP('-' + Math.round(L));
      const nTxt = L < 3 ? Math.max(1, Math.round(Math.pow(10, L))).toLocaleString('en-US') : '10' + SUP(Math.round(L));
      ro3.innerHTML = `<span>d <b>${fmtInt(d)}</b></span><span>P(one negative lands near) <b>${pTxt}</b></span><span>negatives per hit ≈ <b>${nTxt}</b></span><span>sampled: <b>${S._inside}</b> of 2000 near</span>`;
    }

    // ---------------------------------------------------------------- control flow
    function resetModels() { Mc = makeModel('con'); Mo = makeModel('opt'); S.todo = 0; S.after = null; S.showThink = false; gridE(Mc); gridE(Mo); }
    function runTests() { verifierTest(Mc); verifierTest(Mo); thinkTest(Mc); thinkTest(Mo); }
    const kick = () => { if (ctx.visible()) loop.start(); };
    function queueTrain(n, after) { S.todo = n; S.showThink = false; S.after = after || null; kick(); }
    function setD(d) { S.d = d; dSl.set(Math.log10(d)); drawCurse(); }
    const loop = lib.loop((dt) => {
      let busy = false;
      if (S.todo > 0) {
        const n = Math.min(S.todo, lib.reducedMotion ? S.todo : 6);
        for (let i = 0; i < n; i++) { stepCon(Mc); stepOpt(Mo); }
        S.todo -= n; gridE(Mc); gridE(Mo);
        if (S.todo <= 0) { verifierTest(Mc); verifierTest(Mo); Mc.test = null; Mo.test = null; if (S.after) { const f = S.after; S.after = null; f(); } }
        draw(); busy = true;
      }
      if (S.dAnim) {
        const A = S.dAnim; A.t = Math.min(1, A.t + dt / A.dur); const s = A.s0 + (A.s1 - A.s0) * lib.ease(A.t);
        S.d = Math.round(Math.pow(10, s)); dSl.set(s); drawCurse(); if (A.t >= 1) S.dAnim = null; else busy = true;
      }
      if (!busy) return false;
    });
    function animateD(d1, dur) { S.dAnim = { s0: Math.log10(S.d), s1: Math.log10(d1), t: lib.reducedMotion ? 1 : 0, dur }; kick(); }
    const trained = () => Mc.it >= 300 && Mo.it >= 300;
    function ensureTrained(then) { if (trained() && S.todo === 0) { then(); draw(); return; } if (S.todo === 0) { resetModels(); S.todo = 300; } S.after = then; kick(); }
    const think = () => { S.showThink = true; S.showSignal = false; runTests(); draw(); };
    function setMode(m) { if (S.mode !== m) { S.mode = m; modeSeg.set(m); resetModels(); } }

    resetModels(); draw(); drawCurse();
    const caps = [
      'Two energies over the same 2-D prediction space, trained live in your browser on the same targets (crosshair). Darker blue = lower energy. Left (ink): contrastive; × = recent negatives pushed up, white dots = data pushed down. Right (blue): Algorithm 1; paths = recent training unrolls from noise.',
      'Thinking test: 24 starts from N(0, I), gradient descent with the best step size for each landscape. Filled dot = landed on the target, × = stuck elsewhere.',
      'Bottom: random negatives in d dimensions, drawn like ŷ₀ ~ N(0, I). Blue bars fall within c·√d of the data. The curve counts the negatives needed per hit, as a power of ten (both axes logarithmic).',
      'Right: the latest Algorithm 1 training unrolls. The dots are the only places where its loss touches the landscape, through ∇ŷE.',
      'Two equally likely answers per context (crosshairs); dotted ring = their average. After training, the thinking test shows where 24 starts end up on each landscape.',
    ];
    return {
      step(i) {
        ctx.setCaption(caps[i]); S.dAnim = null;
        if (i === 0) { setMode('one'); resetModels(); setD(2); S.showSignal = true; queueTrain(300); draw(); }
        if (i === 1) { setMode('one'); ensureTrained(think); }
        if (i === 2) { setMode('one'); ensureTrained(think); setD(2); animateD(3136, 2.6); }
        if (i === 3) { setMode('one'); S.showThink = false; S.showSignal = true; ensureTrained(() => { S.showThink = false; S.showSignal = true; }); draw(); }
        if (i === 4) { setMode('two'); resetModels(); S.showSignal = true; queueTrain(300, think); draw(); }
      },
      show() { if (S.todo > 0 || S.dAnim) loop.start(); },
      hide() { loop.stop(); },
    };
  },
});
