/* Panel: shaping the landscape (merged from legacy contrastive + regularizers; Sec 3.2–3.3 p.6–8, Table 2 p.10).
   Figure: (1) two 2-D energies trained live on the same data stream, one with a contrastive (InfoNCE) loss and random
   negatives, one with Algorithm 1 (unrolled descent + MSE, exact second-order backprop), plus a verifier test and a
   thinking test; the Algorithm 1 side can add the four Sec 3.3 regularizers (random α, random N, Langevin noise, replay).
   (2) The curse of dimensionality: exact chi-square probability that a random negative lands near the data. */
EBT.panel({
  id: 'shaping',
  nav: 'Shaping the landscape',
  title: 'Shaping a landscape you can think on',
  lede: 'Data only says where energy should be low. Training must also keep it high everywhere else, and make the slopes lead somewhere.',
  text: `
    <p><b>Contrastive</b> training samples wrong answers (negatives) and pushes their energy up while pushing the data's down. <b>Optimization-based</b> training, the paper's choice, samples none: it runs the model's own descent from noise and scores where it lands (previous panel). The two landscapes learn from the same targets; only the loss differs.</p>`,
  steps: [
    { label: 'Same data, two recipes', html: '<p>Contrastive training digs a narrow well at the data and raises plateaus where negatives (×) fell. Algorithm 1 forms a wide funnel. Both become good <b>verifiers</b>: the data scores lower than nearly all of 200 random candidates (readout).</p>' },
    { label: 'Now think on them', html: '<p>24 descents from noise. On the funnel nearly all land. On the contrastive landscape most stall in dips between past negatives: it fixed energy values at sampled points, and nothing asked the slopes in between to lead anywhere.</p>' },
    { label: 'The curse of dimensionality', html: '<p>In 2-D a few thousand negatives blanket the square. In $d$ dimensions random negatives almost never land near the data: for one video latent ($d=3136$) about $10^{435}$ draws per hit. Contrastive methods "must increase the energy of an exponentially higher number of negative samples" (p.6). Algorithm 1 costs $N$ steps in any dimension. Its forward pass acts as a GAN discriminator and its descent as the generator (p.7), but one network plays both.</p>' },
    { label: 'Vary the strides', html: '<p>The loss reaches the landscape only at points training descents visit (dots); elsewhere it changes by accident. Sec 3.3 widens that set. A random step size and a random number of steps per descent make paths probe the basin at many radii and, in our reading, teach it to tolerate the extra steps thinking will take.</p>' },
    { label: 'Add noise and replay', html: '<p>Langevin noise on each step lets paths wander off the direct route; a replay buffer restarts some descents from earlier predictions (dashed), simulating longer trajectories (p.7). Table 2 (readout) prices each trick: without random step size, thinking longer stops helping; without noise, a single path improves more but self-verification less.</p>' },
  ],
  after: `
    <p class="note">Toy: 169 Gaussian bumps with learnable heights plus a weak bowl, 300 Adam updates in your browser. Contrastive: InfoNCE with K uniform negatives. Algorithm 1: N = 4, exact backprop. Table 2 is one out-of-distribution benchmark (Dyck), one seed.</p>`,
  source: [{ kind: 'concept', note: '2-D energies trained live in your browser' }, { kind: 'concept', note: 'exact χ² volumes' }, { kind: 'paper', note: 'Sec 3.2–3.3 p.6–7, Table 2 p.10' }],
  figure(stage, ctx) {
    const { lib } = ctx, h = lib.h, C = lib.C;
    // ---------------------------------------------------------------- RBF energy (local, real math)
    const G = 13, CE = 3, W = 0.5, LAM = 0.12, iw2 = 1 / (W * W), KB = G * G, EXT = 2.5;
    const CX = new Float64Array(KB), CY = new Float64Array(KB);
    for (let i = 0; i < G; i++) for (let j = 0; j < G; j++) { CX[i * G + j] = -CE + 2 * CE * j / (G - 1); CY[i * G + j] = -CE + 2 * CE * i / (G - 1); }
    const PH = new Float64Array(KB), EXX = new Float64Array(G), EXY = new Float64Array(G);
    // separable Gaussian features: exp(-(dx²+dy²)/2w²) = exp(-dx²/2w²)·exp(-dy²/2w²), 26 exps instead of 169
    const phi = (a, b) => { for (let j = 0; j < G; j++) { const dx = a - CX[j], dy = b - CY[j * G]; EXX[j] = Math.exp(-dx * dx * 0.5 * iw2); EXY[j] = Math.exp(-dy * dy * 0.5 * iw2); } for (let i = 0; i < G; i++) { const ey = EXY[i], o = i * G; for (let j = 0; j < G; j++) PH[o + j] = ey * EXX[j]; } return PH; };
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
    const S = { mode: 'one', K: 8, N: 4, alpha: 0.5, todo: 0, showThink: false, showSignal: false, d: 2, c: 0.5, dAnim: null, B: 16, rA: false, rN: false, lang: false, replay: false, sigma: 0.15, pRep: 0.25 };
    const KEYS = ['rA', 'rN', 'lang', 'replay'];
    function makeModel(kind) {
      const th = new Float64Array(KB), r0 = lib.rng(5); for (let k = 0; k < KB; k++) th[k] = 0.05 * r0.normal();
      return { kind, th, m: new Float64Array(KB), v: new Float64Array(KB), t: 0, it: 0, loss: NaN, negs: [], pos: [], paths: [], buf: [], test: null, verif: null, rng: lib.rng(kind === 'con' ? 11 : 12), E: null };
    }
    let Mc = makeModel('con'), Mo = makeModel('opt');
    function adam(M, g, lr) { M.t++; const b1 = 0.9, b2 = 0.999; for (let k = 0; k < KB; k++) { const gk = g[k] + 1e-4 * M.th[k]; M.m[k] = b1 * M.m[k] + (1 - b1) * gk; M.v[k] = b2 * M.v[k] + (1 - b2) * gk * gk; M.th[k] -= lr * (M.m[k] / (1 - Math.pow(b1, M.t))) / (Math.sqrt(M.v[k] / (1 - Math.pow(b2, M.t))) + 1e-8); } }
    function stepCon(M) { // InfoNCE with K uniform negatives
      const r = M.rng, g = new Float64Array(KB); let loss = 0; const negs = [], pos = [];
      const nY = S.K + 1; if (!M.F || M.F.length < nY * KB) M.F = new Float64Array(nY * KB); const F = M.F, es = new Float64Array(nY);
      for (let b = 0; b < S.B; b++) {
        const ys = [sampleY(r)]; for (let j = 0; j < S.K; j++) ys.push([(2 * r() - 1) * EXT, (2 * r() - 1) * EXT]);
        let mn = Infinity;
        for (let j = 0; j < nY; j++) { const q = ys[j]; phi(q[0], q[1]); let s = LAM * (q[0] * q[0] + q[1] * q[1]); const o = j * KB; for (let k = 0; k < KB; k++) { F[o + k] = PH[k]; s += M.th[k] * PH[k]; } es[j] = s; if (s < mn) mn = s; }
        let Z = 0; for (let j = 0; j < nY; j++) { es[j] = Math.exp(-(es[j] - mn)); Z += es[j]; } // es now holds the unnormalized weights
        loss += -Math.log(es[0] / Z);
        for (let j = 0; j < nY; j++) { const cf = (j === 0 ? (1 - es[0] / Z) : -es[j] / Z) / S.B, o = j * KB; for (let k = 0; k < KB; k++) g[k] += cf * F[o + k]; }
        pos.push(ys[0]); for (let j = 1; j < nY; j++) negs.push(ys[j]);
      }
      adam(M, g, 0.03); M.it++; M.loss = loss / S.B;
      M.negs = M.negs.concat(negs).slice(-72); M.pos = M.pos.concat(pos).slice(-24);
    }
    function stepOpt(M) { // Algorithm 1: unroll N steps, MSE at the end, exact backprop through the unroll; optional Sec 3.3 regularizers
      const r = M.rng, g = new Float64Array(KB); let loss = 0; const paths = [];
      const N = S.rN ? 2 + Math.floor(r() * 5) : S.N; // random number of steps: 2..6 per batch
      for (let b = 0; b < S.B; b++) {
        const y = sampleY(r), al = S.rA ? S.alpha * Math.exp((2 * r() - 1) * Math.LN2) : S.alpha; // random step size in [α/2, 2α]
        const rep = S.replay && M.buf.length > 0 && r() < S.pRep, st = rep ? M.buf[Math.floor(r() * M.buf.length)] : [r.normal(), r.normal()];
        let a = st[0], c = st[1]; const path = [[a, c]];
        for (let i = 0; i < N; i++) { const gr = grad(M.th, a, c), e0 = S.lang ? S.sigma * r.normal() : 0, e1 = S.lang ? S.sigma * r.normal() : 0; a = clampY(a - al * gr[0] + e0); c = clampY(c - al * gr[1] + e1); path.push([a, c]); }
        let gy0 = 2 * (a - y[0]), gy1 = 2 * (c - y[1]); loss += (a - y[0]) ** 2 + (c - y[1]) ** 2;
        for (let i = N - 1; i >= 0; i--) { // noise does not depend on θ or ŷ, so the backward pass is unchanged
          const p = path[i]; phi(p[0], p[1]);
          for (let k = 0; k < KB; k++) g[k] += al * PH[k] * iw2 * (gy0 * (p[0] - CX[k]) + gy1 * (p[1] - CY[k])) / S.B;
          const [hxx, hxy, hyy] = hess(M.th, p[0], p[1]);
          const n0 = gy0 - al * (hxx * gy0 + hxy * gy1), n1 = gy1 - al * (hxy * gy0 + hyy * gy1); gy0 = n0; gy1 = n1;
        }
        path.replay = rep; paths.push(path); M.buf.push([a, c]);
      }
      if (M.buf.length > 64) M.buf = M.buf.slice(-64);
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
    const bTrain = lib.button('train 300', () => { resetModels(); S.showSignal = true; queueTrain(300); draw(); }, { primary: true });
    const bThink = lib.button('think', () => ensureTrained(think));
    const bReset = lib.button('reset', () => { resetModels(); draw(); });
    const kSl = lib.slider({ id: 'shp-k', label: 'negatives per example K', min: 1, max: 32, step: 1, value: S.K, fmt: (v) => String(v), oninput: (v) => { S.K = v; } });
    kSl.input.addEventListener('change', () => { resetModels(); S.showSignal = true; queueTrain(300, think); draw(); });
    controls.append(bTrain, bThink, bReset, kSl.el);
    // Sec 3.3 regularizers, applied to the Algorithm 1 side; toggling retrains both from scratch
    const ctlR = h('div', { class: 'controls shp-tg' }); stage.appendChild(ctlR);
    ctlR.appendChild(h('span', { class: 'fig-label' }, 'Alg 1 recipe'));
    const tg = {};
    [['rA', 'random α'], ['rN', 'random N'], ['lang', 'Langevin'], ['replay', 'replay']].forEach(([k, lab]) => {
      const b = lib.button(lab, () => { S[k] = !S[k]; syncTg(); resetModels(); S.showSignal = true; queueTrain(300); draw(); }); b.classList.add('tg'); tg[k] = b; ctlR.appendChild(b);
    });
    const roT = h('div', { class: 'readout shp-t2' }); stage.appendChild(roT);
    const T2 = { rA: ['no random step size', '−1.47', '0.19'], rN: ['no random num. steps', '0.00', '9.65'], lang: ['no Langevin dynamics', '17.2', '17.0'], replay: ['no replay buffer', '14.8', '17.8'], all: ['full System 2 config', '7.19', '18.7'] };
    function syncTg() {
      KEYS.forEach(k => tg[k].setAttribute('aria-pressed', String(S[k])));
      const off = KEYS.filter(k => !S[k]);
      const row = off.length === 0 ? T2.all : off.length === 1 ? T2[off[0]] : null;
      roT.innerHTML = row ? `<span>Table 2 (paper, % ppl gain, Dyck OOD): <b>${row[0]}</b></span><span>thinking longer <b>${row[1]}</b></span><span>+ self-verification <b>${row[2]}</b></span>`
        : `<span>Table 2 (paper): ${off.length === 4 ? 'plain Algorithm 1, no regularizers (like the paper\'s S1 models)' : 'not a row; the paper removed one trick at a time'}</span>`;
    }

    // ---------------------------------------------------------------- layout: curse of dimensionality
    const row2 = h('div', { class: 'fig-row' }); stage.appendChild(row2);
    const F2 = lib.frame(row2, { label: 'Negatives needed per hit', sub: 'random draws per one landing within 0.5·√d of the data' }); F2.wrap.style.flex = '1 1 280px';
    const lc = lib.canvas(F2.frame, 300, 180, { label: 'Number of random negatives needed per hit near the data, versus dimension', maxWidth: 520 });
    const fmtInt = (n) => n.toLocaleString('en-US');
    const col3 = h('div', { class: 'fig-col shp-dcol' }); col3.style.flex = '1 1 200px'; row2.appendChild(col3);
    const dSl = lib.slider({ id: 'shp-d', label: 'dimension d', min: 0, max: Math.log10(50277), step: 0.005, value: Math.log10(2), fmt: (v) => fmtInt(Math.round(Math.pow(10, v))), oninput: (v) => { S.dAnim = null; S.d = Math.round(Math.pow(10, v)); drawCurse(); } });
    const ro3 = h('div', { class: 'readout' });
    col3.append(dSl.el, ro3);

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
          M.paths.forEach(path => { const px = path.map(P); lib.line(c, px, { color: path.replay ? '#111' : C.blue, width: 1.3, alpha: 0.8, dash: path.replay ? [3, 2] : null }); px.slice(0, -1).forEach(p => lib.dot(c, p[0], p[1], 2.4, '#fff', { stroke: C.blue, lw: 1.2 })); const e = px[px.length - 1]; lib.dot(c, e[0], e[1], 3, C.blue); });
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
      const d = S.d, c0 = S.c;
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
      ro3.innerHTML = `<span>d <b>${fmtInt(d)}</b></span><span>P(one negative lands near) <b>${pTxt}</b></span><span>negatives per hit ≈ <b>${nTxt}</b></span>`;
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
        // time-budgeted: up to 8 training iterations per frame, stop early after ~6 ms (K = 32 is the slow case)
        const t0 = performance.now(); let n = 0;
        while (S.todo > 0 && (lib.reducedMotion || (n < 8 && (n === 0 || performance.now() - t0 < 6)))) { stepCon(Mc); stepOpt(Mo); S.todo--; n++; }
        gridE(Mc); gridE(Mo);
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
    function setRecipe(o) { const ch = KEYS.some(k => S[k] !== !!o[k]); KEYS.forEach(k => { S[k] = !!o[k]; }); syncTg(); if (ch) resetModels(); }

    syncTg(); resetModels(); draw(); drawCurse();
    const caps = [
      'Two energies over the same 2-D prediction space, trained live in your browser on the same target (crosshair). Darker = lower energy. Left: contrastive; × = recent negatives pushed up, white dots = data pushed down. Right: Algorithm 1; blue paths = recent training descents from noise.',
      'Thinking test: 24 starts from N(0, I), gradient descent with the best of three step sizes for each landscape. Filled dot = landed on the target, × = stuck elsewhere (on the border = left the square shown).',
      'Bottom: random negatives drawn like ŷ₀ ~ N(0, I) in d dimensions. The curve counts draws needed per one that lands within 0.5·√d of the data, as a power of ten (both axes logarithmic). Blue dashed: Algorithm 1 needs N steps whatever d is.',
      'Right: training descents with random step size and random step count, retrained from scratch. White dots are the only places where the loss touches the landscape, through ∇ŷE.',
      'All four Sec 3.3 tricks. Dashed ink paths restart from stored earlier predictions (replay buffer). The readout shows the matching row of the paper\'s Table 2; toggle the recipe to see the other rows.',
    ];
    const signal = () => { S.showThink = false; S.showSignal = true; };
    return {
      step(i) {
        ctx.setCaption(caps[i]); S.dAnim = null;
        if (i === 0) { setRecipe({}); resetModels(); setD(2); S.showSignal = true; queueTrain(300); draw(); }
        if (i === 1) { setRecipe({}); ensureTrained(think); }
        if (i === 2) { setRecipe({}); ensureTrained(think); setD(2); animateD(3136, 2.6); }
        if (i === 3) { setRecipe({ rA: 1, rN: 1 }); signal(); ensureTrained(signal); draw(); }
        if (i === 4) { setRecipe({ rA: 1, rN: 1, lang: 1, replay: 1 }); signal(); ensureTrained(signal); draw(); }
      },
      show() { if (S.todo > 0 || S.dAnim) loop.start(); },
      hide() { loop.stop(); },
    };
  },
});
