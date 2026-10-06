/* Panel: energy-landscape regularizers (Sec 3.3, p.7–8; Table 2, p.10; Table D.4, p.36).
   Figure: (A) training unrolls sampled live on real checkpoints of the toy 2-D EBT under the four regularizers (toggle each),
   showing every point where the loss reaches the landscape; (B) exact stride analysis: the curvature a quadratic basin learns
   under fixed vs randomized step size, against the toy's measured Hessian eigenvalues; (B') the toy's replay-buffer chain;
   (C) Table 2 of the paper as bars (click a row to load that ablation). */
EBT.panel({
  id: 'regularizers',
  nav: 'Landscape regularizers',
  title: 'Shaping a landscape you can think on',
  lede: 'In plain Algorithm 1, training descents are two or three steps long and always start from fresh noise. Thinking asks for more steps, other starting points and fair energy comparisons. Four tricks from Sec 3.3 change where training descents go and how far they stride, so the landscape stays well shaped where thinking will need it.',
  text: `
    <p>Algorithm 1 trains with 2 to 3 descent steps (Table D.4), but at inference the model may be asked to take more (the paper's uncertainty plots run iterations 0 to 11) and to rank several candidates by their energy (Best-of-N). "Because y is high-dimensional, the energy landscape spans a high-dimensional space, and must remain well-shaped throughout" (p.7).</p>
    <p>Where does training actually shape it? Write one training unroll with a per-example step size $\\alpha_i$ and the optional Langevin noise $\\eta_i$ of Eq. 2, $\\hat y_{i+1}=\\hat y_i-\\alpha_i\\nabla_{\\hat y}E_\\theta(x,\\hat y_i)+\\eta_i$, and backpropagate the loss $J(\\hat y_N, y)$ through it:</p>
    <div class="eq">$$\\begin{gathered}\\frac{\\partial \\mathcal L}{\\partial\\theta} = -\\sum_{i=0}^{N-1}\\alpha_i\\,\\big[\\partial_\\theta\\nabla_{\\hat y}E_\\theta(x,\\hat y_i)\\big]^{\\!\\top} g_{i+1}\\\\[4pt] g_{i}=\\big(I-\\alpha_i H_i\\big)\\,g_{i+1},\\quad g_N=\\nabla_{\\hat y}J(\\hat y_N,y)\\\\[4pt] H_i=\\nabla^2_{\\hat y}E_\\theta(x,\\hat y_i)\\end{gathered}$$<span class="why">Every term evaluates the landscape at one visited point ŷᵢ: its slope's sensitivity to θ, and its curvature H. g carries the loss back along the path.</span></div>
    <p>So the training signal enters the landscape only at the dots in the figure. Elsewhere the landscape changes only as a side effect of shared weights. The four regularizers of Sec 3.3 all work by changing where those dots fall and how far apart they are. The figure samples real training unrolls on a checkpoint of the toy EBT, with each regularizer switched on or off.</p>`,
  steps: [
    { label: 'Plain Algorithm 1', html: '<p>All four off. Every unroll starts from fresh noise and takes $N=4$ steps of the same size $\\alpha_0=1$, with no noise. White dots mark where the loss reaches the landscape. On this trained checkpoint the first step already lands in the basin, so later steps pile onto the same few spots (zoom): many points on the floor, few distinct places trained, and every path is a straight radial line. Nothing trains the landscape for a second look from a slightly different angle.</p>' },
    { label: 'Randomize the step size', html: '<p>Each example draws its own $\\alpha\\in[\\alpha_0/2,\\,2\\alpha_0]$ (toy: log-uniform; paper: "multiplicative random factor 2", distribution unspecified). Strides under- and overshoot, so paths probe the basin at many radii: the area near the basin that gets signal about doubles (readout, versus plain; 2.0× averaged over 20 resamples).</p><p>The stride plot shows a deeper effect (our analysis, not the paper\'s). Near its minimum a basin is a bowl $E\\approx E^*+\\tfrac{k}{2}\\lVert\\hat y-\\hat y^*\\rVert^2$, and one step multiplies the remaining error by $1-\\alpha k$. Trained with one $\\alpha_0$, the best bowl has $k=1/\\alpha_0$: the first step lands exactly, extra steps have nothing left to do, and any $\\alpha>2\\alpha_0$ diverges. Trained over a range, no bowl lands every stride. The best compromise is $k^*\\approx0.8/\\alpha_0$ (the value $2/(\\alpha_{\\min}+\\alpha_{\\max})$; 0.75 if α is uniform), which leaves 20% of the error after each step at $\\alpha_0$. So every stride in the range converges, and each extra step keeps shrinking the error. The trained toy\'s measured curvatures are 0.78 to 0.92. Table 2 fits this picture: without random step size, thinking longer gains −1.47% and thinking with self-verification 0.19%.</p>' },
    { label: 'Randomize the number of steps', html: '<p>$N$ varies (toy: 2 to 6 per batch; paper: "2-3 (randomized)", per batch or per example unspecified). It barely moves the dots (readout): it changes how long the loss waits, not where the steps go. The loss sees only $\\hat y_N$, so with a fixed $N$ nothing in training says what a step $N+1$ should do. Randomizing $N$ asks every $\\hat y_n$ in the range to be good, which favors landscapes where the answer is a fixed point that further steps keep or improve. Table 2: without it, thinking longer gains exactly 0.00 (with self-verification, 9.65). Our reading: the model only learns to use extra steps if training showed it varied step counts.</p>' },
    { label: 'Add Langevin noise', html: '<p>Eq. 2 adds $\\eta_i\\sim\\mathcal N(0,\\sigma)$ to every training step. Paths wander, so the neighborhood of each path is shaped too: "Without this random noise term, exploration is often limited to paths leading directly to the energy minimum, leaving other regions poorly defined" (p.7). Even the toy\'s small σ = 0.05 spreads the dots over about 60% more distinct floor cells (20 resamples at step 400); the figure uses 0.12 to make it visible. Table 2: removing it <em>raises</em> thinking longer from 7.19 to 17.2 but lowers thinking + self-verification from 18.7 to 17.0.</p>' },
    { label: 'Add the replay buffer', html: '<p>Some examples restart from a stored earlier prediction instead of noise (toy: 25% of each batch, restarted from the same example\'s last ŷ<sub>N</sub>; paper: what is stored and how often it is used are unspecified). Chained short unrolls act as one long trajectory, "enabling energy landscapes to be well defined near their minimum" (p.7). In the toy it matters most early in training (2.4× more dots on the floor at step 50, only 1.1× at step 400), so the figure jumps to step 50: four short steps from noise rarely reach the still-shallow basin, and replayed starts (dashed) put far more dots on its floor. The chain plot shows four real toy unrolls ($N$ = 2, 3, 6, 4) at step 400 that form one 15-step descent. Table 2: without it, 14.8 (thinking longer) and 17.8 (with self-verification).</p>' },
    { label: 'All four: the System 2 recipe', html: '<p>Read Table 2 as a trade-off. All four are needed for the best thinking + verification score (18.7). Random step size is indispensable. But the full recipe is <em>not</em> best for single-path thinking: 7.19 versus 17.2 without Langevin noise. The paper\'s caption reads this as exploration: less of it "improves single path performance (thinking longer) at the expense of self-verification performance" (p.10). Noise makes energies trustworthy across many candidates at the cost of a sharper single path, and the text calls the choice "a performance-compute tradeoff" (p.10).</p>' },
  ],
  after: `
    <h3>The recipes side by side</h3>
    <div class="tbl"><table>
      <thead><tr><th></th><th>S1-EBT</th><th>S2-EBT</th><th>our toy</th></tr></thead>
      <tbody>
        <tr><td>used for</td><td>many scaling runs</td><td>thinking runs</td><td>this site</td></tr>
        <tr><td>steps N</td><td>2</td><td>2–3, random</td><td>2–6, random per batch</td></tr>
        <tr><td>step size α</td><td>learnable (text 500)</td><td>5, random factor 2</td><td>1, random factor 2 per example</td></tr>
        <tr><td>Langevin noise</td><td>none</td><td>3</td><td>0.05</td></tr>
        <tr><td>replay buffer</td><td>no</td><td>yes</td><td>yes, 25% of batch</td></tr>
        <tr><td>detach between steps</td><td>yes</td><td>no</td><td>no</td></tr>
        <tr><td>loss</td><td>every step</td><td>last step, truncated</td><td>last step</td></tr>
        <tr><td>step conditioning</td><td>step embedding</td><td>one shared landscape</td><td>none</td></tr>
      </tbody></table></div>
    <p>S1 and S2 values are from Tables D.3 and D.4 and p.30, 36, 42, 45. S1 models use no landscape regularization at all; they are tuned "for stability and learning convergence" (p.30). The appendix recipe for going from S1 to S2 ends with "Then add a replay buffer, Langevin Dynamics, and eventually a randomized alpha" (p.45). The paper does not specify the replay buffer's size or sampling probability, the distribution behind "random factor 2", whether the noise "3" is a standard deviation or a variance, or whether noise is used at inference.</p>
    <p class="note">Table 2 is one out-of-distribution benchmark (BigBench Dyck), one seed, and its reference point for "percent perplexity improvement" is unspecified. Our toy was trained once with all four regularizers on. The toggles resample training unrolls on its saved weights; they do not retrain it. The stride plot is exact math for a quadratic basin: $k^*=\\arg\\min_k \\mathbb E_{\\alpha,N}(1-\\alpha k)^{2N}$ with α log-uniform over the shaded range and $N\\in\\{2,3\\}$ (for log-uniform α the minimizer is the same for any $N$). Toy statistics quoted in the steps are averages over 20 resamples.</p>`,
  source: [{ kind: 'toy', note: 'toy 2-D EBT checkpoints, unrolls sampled live' }, { kind: 'concept', note: 'stride analysis, exact for a quadratic basin' }, { kind: 'paper', note: 'Sec 3.3 p.7, Table 2 p.10, Table D.4 p.36' }],
  figure(stage, ctx) {
    const { lib } = ctx, h = lib.h, C = lib.C, M = window.EBT.toy2d;
    if (!M || !M.ready) { stage.appendChild(h('p', { class: 'callout warn' }, 'Toy model data missing (data/toy2d.json).')); return {}; }
    const D = M.data, HP = D.hparams, STEPS = M.steps, EXT = M.extent;
    const S = { rA: false, rN: false, lang: false, replay: false, sigma: HP.langevin, ck: 6, ctx: 0, seed: 3, view: 'stride', F: 1, Ftarget: 1, chain: false };
    const RF = 0.6; // zoom half-width around the minimum
    const T2 = [['No random step size', -1.47, 0.19, 'rA'], ['No random num. steps', 0.0, 9.65, 'rN'], ['No Langevin dynamics', 17.2, 17.0, 'lang'], ['No replay buffer', 14.8, 17.8, 'replay'], ['Full System 2 config', 7.19, 18.7, null]];
    const T2S = [['−1.47', '0.19'], ['0.00', '9.65'], ['17.2', '17.0'], ['14.8', '17.8'], ['7.19', '18.7']]; // exactly as printed (p.10)
    const KEYS = ['rA', 'rN', 'lang', 'replay'];

    // ---------------------------------------------------------------- toggles + selectors
    const ctl = h('div', { class: 'controls reg-tg' }); stage.appendChild(ctl);
    ctl.appendChild(h('span', { class: 'fig-label' }, 'recipe'));
    const tg = {};
    [['rA', 'random α'], ['rN', 'random N'], ['lang', 'Langevin'], ['replay', 'replay']].forEach(([k, lab]) => {
      const b = lib.button(lab, () => { S[k] = !S[k]; sync(); update(); }); b.classList.add('tg'); tg[k] = b; ctl.appendChild(b);
    });
    const ctl1 = h('div', { class: 'controls' }); stage.appendChild(ctl1);
    const sigSl = lib.slider({ id: 'reg-sig', label: 'Langevin σ (toy: 0.05)', min: 0, max: 0.3, step: 0.01, value: S.sigma, fmt: (v) => v.toFixed(2), oninput: (v) => { S.sigma = v; update(); } });
    ctl1.append(lib.button('all on', () => { KEYS.forEach(k => S[k] = true); sync(); update(); }), lib.button('all off', () => { KEYS.forEach(k => S[k] = false); sync(); update(); }), lib.button('resample', () => { S.seed++; update(); }), sigSl.el);
    const ctl2 = h('div', { class: 'controls' }); stage.appendChild(ctl2);
    const ckSeg = lib.segmented({ label: 'Checkpoint', options: [[3, 'step 50'], [6, 'step 400'], [11, 'step 12k']], value: S.ck, onchange: (v) => { S.ck = v; update(); } });
    const cxSeg = lib.segmented({ label: 'Context', options: [[0, 'x 0.25 · σ .03'], [1, 'x 0.75 · σ .30']], value: S.ctx, onchange: (v) => { S.ctx = v; update(); } });
    ctl2.append(h('span', { class: 'fig-label' }, 'checkpoint'), ckSeg.el, h('span', { class: 'fig-label' }, 'context'), cxSeg.el);
    function sync() { KEYS.forEach(k => tg[k].setAttribute('aria-pressed', String(S[k]))); sigSl.el.style.opacity = S.lang ? 1 : 0.4; }

    // ---------------------------------------------------------------- A: landscape + zoom
    const row = h('div', { class: 'fig-row' }); stage.appendChild(row);
    const FA = lib.frame(row, { label: 'Training unrolls', sub: 'dots = where ∇<sub>ŷ</sub>E gets training signal' }); FA.wrap.style.flex = '1 1 250px';
    const ca = lib.canvas(FA.frame, 300, 300, { label: 'Toy energy landscape with sampled training unrolls; dots mark points where the energy gradient receives training signal' });
    const FZ = lib.frame(row, { label: 'Zoom: the basin floor', sub: '±0.6 around the minimum · ring r = 0.25' }); FZ.wrap.style.flex = '1 1 250px';
    const cz = lib.canvas(FZ.frame, 300, 300, { label: 'Zoom on the basin floor with the same training unrolls' });
    const ro = h('div', { class: 'readout' }); stage.appendChild(ro);

    // ---------------------------------------------------------------- B + C
    const row2 = h('div', { class: 'fig-row' }); stage.appendChild(row2);
    const FB = lib.frame(row2, { label: 'Strides', sub: '|1 − αk| per step, quadratic basin' }); FB.wrap.style.flex = '1 1 250px';
    const cb = lib.canvas(FB.frame, 300, 230, { label: 'Per-step contraction factor versus inference step size for a basin trained with fixed or randomized step size' });
    const FC = lib.frame(row2, { label: 'Table 2 · paper', sub: '% ppl gain, OOD Dyck · click a row' });
    const roC = h('div', { class: 'readout' }); FC.wrap.appendChild(roC); FC.wrap.style.flex = '1 1 250px';
    const cc = lib.canvas(FC.frame, 300, 230, { label: 'Table 2 of the paper as bars: percent perplexity improvement from thinking longer and from thinking longer plus self-verification, per ablation' });
    const roB = h('div', { class: 'readout' }); FB.wrap.appendChild(roB);
    const fSl = lib.slider({ id: 'reg-f', label: 'random factor F', min: 1, max: 4, step: 0.05, value: S.F, fmt: (v) => v.toFixed(2), oninput: (v) => { S.F = v; S.Ftarget = v; drawB(); } });
    FB.wrap.appendChild(fSl.el);

    // ---------------------------------------------------------------- sampling of training unrolls (separate streams per component)
    const zCache = new Map();
    function zoomData() {
      const key = S.ck + ':' + S.ctx; if (zCache.has(key)) return zCache.get(key);
      const grid = M.storedGrid(S.ctx, STEPS[S.ck]), n = grid.length, x = M.contexts[S.ctx].x; let best = Infinity, at = [0, 0];
      grid.forEach((r, i) => r.forEach((v, j) => { if (v < best) { best = v; at = [EXT[0] + j * (EXT[1] - EXT[0]) / (n - 1), EXT[3] - i * (EXT[3] - EXT[2]) / (n - 1)]; } }));
      const mn = M.descend(S.ck, x, at, { alpha: 0.5, steps: 60 }).path.pop();
      const zext = [mn[0] - RF, mn[0] + RF, mn[1] - RF, mn[1] + RF];
      const zg = M.grid(S.ck, x, 48, zext);
      const sorted = grid.flat().sort((a, b) => a - b), zs = zg.E.flat().sort((a, b) => a - b);
      const z = { grid, q: sorted[Math.floor(0.6 * sorted.length)], lo: sorted[0], min: mn, zext, zgrid: zg.E, zlo: zs[0], zq: zs[Math.floor(0.8 * zs.length)] };
      zCache.set(key, z); return z;
    }
    let sampled = null;
    function sampleAll(R) {
      R = R || S; const x = M.contexts[S.ctx].x, base = 7919 * S.seed + 31 * S.ctx;
      const rS = lib.rng(base + 1), rA = lib.rng(base + 2), rN = lib.rng(base + 3), rE = lib.rng(base + 4), rR = lib.rng(base + 5), rW = lib.rng(base + 6);
      const buf = [];
      const alphaOf = () => { const u = rA(); return R.rA ? HP.alpha0 * Math.exp((2 * u - 1) * Math.log(HP.alpha_rand_factor)) : HP.alpha0; };
      const doOne = (y0, N, rep) => { const a = alphaOf(); const res = M.descend(S.ck, x, y0, { alpha: a, steps: N, sigma: R.lang ? S.sigma : 0, seed: Math.floor(rE() * 1e9) + 1 }); const end = res.path[N].map(v => Math.max(-4, Math.min(4, v))); buf.push(end); return { path: res.path, alpha: a, N, replay: rep }; };
      for (let j = 0; j < 12; j++) doOne([rW.normal(), rW.normal()], 2 + (j % 5), false); // warm the buffer (not drawn)
      const Ns = [], unrolls = [];
      for (let b = 0; b < 5; b++) {
        const u = rN(); const N = R.rN ? HP.n_min + Math.floor(u * (HP.n_max - HP.n_min + 1)) : 4; Ns.push(N);
        for (let j = 0; j < 8; j++) { const s0 = [rS.normal(), rS.normal()]; const rep = R.replay && rR() < HP.p_replay; const y0 = rep ? buf[Math.floor(rR() * buf.length)].slice() : s0; unrolls.push(doOne(y0, N, rep)); }
      }
      const z = zoomData(); let pts = 0, near = 0; const fine = new Set(), mid = new Set();
      unrolls.forEach(u => { for (let i = 0; i < u.N; i++) { const q = u.path[i]; pts++; const d = Math.hypot(q[0] - z.min[0], q[1] - z.min[1]); if (d < 0.25) { near++; fine.add(Math.floor((q[0] - z.min[0]) / 0.05) + ':' + Math.floor((q[1] - z.min[1]) / 0.05)); } if (d < RF) mid.add(Math.floor((q[0] - z.min[0]) / 0.1) + ':' + Math.floor((q[1] - z.min[1]) / 0.1)); } });
      const al = unrolls.map(u => u.alpha);
      return { unrolls, Ns, z, pts, near, fine: fine.size, mid: mid.size, amin: Math.min(...al), amax: Math.max(...al), nrep: unrolls.filter(u => u.replay).length };
    }
    let plain = null;

    // ---------------------------------------------------------------- drawing helpers
    function land(c, E, box, lo, hi, nlev) {
      lib.heatmap(c, E, box.x, box.y, box.w, box.h, { range: [lo, hi], gamma: 0.75 });
      const lv = Array.from({ length: nlev }, (_, i) => lo + (hi - lo) * Math.pow((i + 1) / nlev, 1.3));
      lib.contours(c, E, box.x, box.y, box.w, box.h, lv, { color: 'rgba(17,17,17,0.2)', width: 1 });
    }
    function cross(c, p, r = 7) { c.save(); c.lineWidth = 3; c.strokeStyle = 'rgba(17,17,17,0.35)'; c.beginPath(); c.moveTo(p[0] - r, p[1]); c.lineTo(p[0] + r, p[1]); c.moveTo(p[0], p[1] - r); c.lineTo(p[0], p[1] + r); c.stroke(); c.lineWidth = 1.4; c.strokeStyle = '#fff'; c.stroke(); c.restore(); }
    function plate(c, s, x, y, o = {}) { c.save(); c.font = '11px "JetBrains Mono", monospace'; const w = c.measureText(s).width + 10; const xx = o.align === 'right' ? x - w : x; c.fillStyle = 'rgba(255,255,255,0.88)'; c.fillRect(xx, y, w, 16); c.restore(); lib.text(c, s, xx + 5, y + 2, { size: 11, kind: 'mono', color: o.color || '#111' }); }
    const box = { x: 0, y: 0, w: 300, h: 300 };
    function drawA() {
      const z = sampled.z, c = ca.ctx; ca.clear();
      land(c, z.grid, box, z.lo, z.q, 10);
      const P = (q) => M.toPx(q, box);
      c.save(); c.beginPath(); c.rect(0, 0, 300, 300); c.clip();
      sampled.unrolls.forEach(u => lib.line(c, u.path.map(P), { color: u.replay ? '#111' : C.blue, width: 1.2, alpha: 0.6, dash: u.replay ? [3, 2] : null }));
      sampled.unrolls.forEach(u => { for (let i = 0; i < u.N; i++) { const q = P(u.path[i]); lib.dot(c, q[0], q[1], 2.2, '#fff', { stroke: '#111', lw: 0.9 }); } const e = P(u.path[u.N]); lib.dot(c, e[0], e[1], 2.6, u.replay ? '#111' : C.blue); });
      if (S.chain) drawChainA(c, P);
      c.restore();
      const a = P([z.zext[0], z.zext[3]]), b = P([z.zext[1], z.zext[2]]);
      c.save(); c.strokeStyle = '#111'; c.lineWidth = 1.2; c.setLineDash([4, 3]); c.strokeRect(a[0], a[1], b[0] - a[0], b[1] - a[1]); c.restore();
      cross(c, P(M.contexts[S.ctx].mu));
      plate(c, 'step ' + STEPS[S.ck].toLocaleString('en-US') + ' · x = ' + M.contexts[S.ctx].x, 6, 278);
      // zoom
      const g = cz.ctx; cz.clear();
      land(g, z.zgrid, box, z.zlo, z.zq, 12);
      const Pz = (q) => M.toPx(q, box, z.zext);
      g.save(); g.beginPath(); g.rect(0, 0, 300, 300); g.clip();
      const mz = Pz(z.min), rr = 0.25 / (2 * RF) * 300;
      g.save(); g.strokeStyle = 'rgba(17,17,17,0.6)'; g.setLineDash([4, 4]); g.lineWidth = 1.2; g.beginPath(); g.arc(mz[0], mz[1], rr, 0, 7); g.stroke(); g.restore();
      sampled.unrolls.forEach(u => { for (let i = 1; i <= u.N; i++) lib.line(g, [Pz(u.path[i - 1]), Pz(u.path[i])], { color: u.replay ? '#111' : C.blue, width: 1.3, alpha: 0.7, dash: u.replay ? [3, 2] : null }); });
      sampled.unrolls.forEach(u => { for (let i = 0; i < u.N; i++) { const q = Pz(u.path[i]); lib.dot(g, q[0], q[1], 3, '#fff', { stroke: '#111', lw: 1 }); } const e = Pz(u.path[u.N]); lib.dot(g, e[0], e[1], 3.2, u.replay ? '#111' : C.blue); });
      if (S.chain) drawChainA(g, Pz);
      cross(g, Pz(M.contexts[S.ctx].mu), 8);
      g.restore();
      plate(g, sampled.near + ' signal points inside the ring', 6, 278);
      const z2 = sampled;
      const vs = (k) => (KEYS.some(q => S[q]) ? ' <span class="reg-vs">(plain ' + plain[k] + ')</span>' : '');
      ro.innerHTML = `<span>N per batch <b>${z2.Ns.join(', ')}</b></span><span>α <b>${z2.amin.toFixed(2)}–${z2.amax.toFixed(2)}</b></span><span>replay starts <b>${z2.nrep}/40</b></span><span>signal points <b>${z2.pts}</b></span><span>on the floor (r &lt; 0.25) <b>${z2.near}</b>${vs('near')}</span><span>floor cells 0.05² <b>${z2.fine}</b>${vs('fine')}</span><span>basin cells 0.1² (r &lt; 0.6) <b>${z2.mid}</b>${vs('mid')}</span>`;
    }
    // replay chain from the toy's own buffer demo
    function chainEx() { const ck = S.ck === 11 ? 11 : 6; return D.replay_buffer_demo.examples.find(e => e.ctx === S.ctx && e.ckpt === ck); }
    function drawChainA(c, P) {
      const ex = chainEx(); if (!ex || ex.ckpt !== S.ck) return;
      ex.segments.forEach((sg, k) => { const px = sg.path.map(P); lib.line(c, px, { color: '#111', width: 2.2, alpha: 0.9 }); lib.line(c, px, { color: k % 2 ? C.blue2 : C.blue, width: 1.4 }); px.forEach(p => lib.dot(c, p[0], p[1], 2.4, '#fff', { stroke: '#111', lw: 1 })); });
      const s0 = P(ex.segments[0].path[0]); lib.dot(c, s0[0], s0[1], 4, '#fff', { stroke: '#111', lw: 1.5 });
    }

    // ---------------------------------------------------------------- B: strides (exact quadratic analysis) / replay chain energies
    function kStar(F) { // argmin_k E_{alpha ~ logU[1/F, F], N in {2,3}} (1 - alpha k)^(2N), alpha0 = 1
      if (F <= 1.0001) return 1;
      const us = 241, al = Array.from({ length: us }, (_, i) => Math.exp(-Math.log(F) + 2 * Math.log(F) * i / (us - 1)));
      const L = (k) => { let s = 0; al.forEach(a => { const e = 1 - a * k; const e2 = e * e; s += e2 * e2 + e2 * e2 * e2; }); return s; };
      let a = 0.05, b = 1.5; for (let it = 0; it < 80; it++) { const m1 = a + (b - a) / 3, m2 = b - (b - a) / 3; if (L(m1) < L(m2)) b = m2; else a = m1; } return (a + b) / 2;
    }
    const bs = D.basin_stats, kLo = Math.min(...bs.hess_eig_small), kHi = Math.max(...bs.hess_eig_large);
    function drawB() {
      const c = cb.ctx; cb.clear();
      if (S.view === 'replay') return drawChainB();
      FB.wrap.querySelector('.fig-label').textContent = 'Strides'; FB.wrap.querySelector('.fig-sub').innerHTML = '|1 − αk| per step, quadratic basin';
      fSl.el.style.display = '';
      const ax = lib.axes(c, { x: 36, y: 10, w: 252, h: 170, xlim: [0.25, 4], ylim: [0, 1.6], xlog: true, xticks: [0.25, 0.5, 1, 2, 4], yticks: [0, 0.5, 1, 1.5], xfmt: (v) => v + '', yfmt: (v) => v + '', size: 11 });
      const F = S.F, ks = kStar(F);
      c.fillStyle = C.blue4; c.fillRect(ax.X(1 / F), 10, ax.X(F) - ax.X(1 / F), 170);
      c.save(); c.strokeStyle = 'rgba(17,17,17,0.5)'; c.setLineDash([2, 3]); c.beginPath(); c.moveTo(ax.X(0.25), ax.Y(1)); c.lineTo(ax.X(4), ax.Y(1)); c.stroke(); c.restore();
      lib.text(c, 'diverges above 1', ax.X(0.27), ax.Y(1) - 14, { size: 10.5, kind: 'mono', color: '#6b6b70' });
      const curve = (k) => { const pts = []; for (let s = Math.log10(0.25); s <= Math.log10(4) + 1e-9; s += 0.01) { const a = Math.pow(10, s); pts.push([a, Math.min(1.6, Math.abs(1 - a * k))]); } return pts; };
      // toy measured band
      const lo = curve(kLo), hi = curve(kHi); c.save(); c.beginPath(); lo.forEach((p, i) => { const X = ax.X(p[0]), Y = ax.Y(p[1]); i ? c.lineTo(X, Y) : c.moveTo(X, Y); }); hi.slice().reverse().forEach(p => c.lineTo(ax.X(p[0]), ax.Y(p[1]))); c.closePath(); c.fillStyle = 'rgba(138,147,255,0.35)'; c.fill(); c.restore();
      lib.plot(c, ax, curve(1), { color: '#111', width: 1.4, dash: [5, 4] });
      lib.plot(c, ax, curve(ks), { color: C.blue, width: 2 });
      lib.text(c, 'α / α₀ at inference', 162, 202, { size: 11, kind: 'mono', color: '#6b6b70', align: 'center' });
      if (F > 1.05) lib.text(c, 'training range', ax.X(1 / F) + 3, 13, { size: 10, kind: 'mono', color: C.blue });
      roB.innerHTML = `<span>k*·α₀ <b>${ks.toFixed(2)}</b> (fixed α: 1.00)</span><span>error left per step at α₀ <b>${Math.abs(1 - ks).toFixed(2)}</b></span><span>stable for α &lt; <b>${(2 / ks).toFixed(2)}α₀</b></span><span>toy measured k·α₀ <b>${kLo.toFixed(2)}–${kHi.toFixed(2)}</b></span>`;
    }
    function drawChainB() {
      const c = cb.ctx, ex = chainEx();
      FB.wrap.querySelector('.fig-label').textContent = 'Replay chain'; FB.wrap.querySelector('.fig-sub').innerHTML = 'energy along 4 chained training unrolls';
      fSl.el.style.display = 'none';
      if (!ex) return;
      const Es = []; const bounds = [0]; ex.segments.forEach((sg, k) => { (k === 0 ? sg.E : sg.E.slice(1)).forEach(e => Es.push(e)); bounds.push(Es.length - 1); });
      const lo = Math.min(...Es), hi = Math.max(...Es);
      const ax = lib.axes(c, { x: 44, y: 10, w: 244, h: 160, xlim: [0, Es.length - 1], ylim: [lo - 0.05 * (hi - lo), hi], xticks: bounds, yticks: [lo, hi], yfmt: (v) => v.toFixed(2), size: 11 });
      // shading first, labels second, so a later segment's band never covers an earlier label
      ex.segments.forEach((sg, k) => { if (k % 2) { c.fillStyle = C.blue4; c.fillRect(ax.X(bounds[k]), 10, ax.X(bounds[k + 1]) - ax.X(bounds[k]), 160); } });
      ex.segments.forEach((sg, k) => { const x0 = bounds[k], x1 = bounds[k + 1]; const cx = Math.max(ax.X(0) + 22, (ax.X(x0) + ax.X(x1)) / 2), yy = 40 + (k % 2) * 28; lib.text(c, 'N=' + sg.N, cx, yy, { size: 10, kind: 'mono', color: '#33333a', align: 'center' }); lib.text(c, 'α ' + sg.alpha.toFixed(2), cx, yy + 12, { size: 10, kind: 'mono', color: '#6b6b70', align: 'center' }); });
      lib.plot(c, ax, Es.map((e, i) => [i, e]), { color: C.blue, width: 1.8, markers: 2.6 });
      lib.text(c, 'cumulative descent step', 166, 196, { size: 11, kind: 'mono', color: '#6b6b70', align: 'center' });
      roB.innerHTML = `<span>context x <b>${M.contexts[S.ctx].x}</b></span><span>checkpoint <b>step ${STEPS[ex.ckpt].toLocaleString('en-US')}</b></span><span>steps chained <b>${Es.length - 1}</b> (each unroll ≤ 6)</span><span>E: <b>${Es[0].toFixed(2)} → ${Es[Es.length - 1].toFixed(3)}</b></span>`;
    }

    // ---------------------------------------------------------------- C: Table 2
    const rowH = 38, top = 26;
    function activeRow() { const off = KEYS.filter(k => !S[k]); if (off.length === 0) return 4; if (off.length === 1) return T2.findIndex(r => r[3] === off[0]); return -1; }
    function drawC() {
      const c = cc.ctx; cc.clear(); const x0 = 38, x1 = 262, X = (v) => x0 + (v + 2) / 22 * (x1 - x0), act = activeRow();
      c.save(); c.strokeStyle = C.blue; c.lineWidth = 1.2; c.strokeRect(0.5, 4.5, 12, 9); c.restore(); lib.text(c, 'thinking longer', 18, 2, { size: 11, kind: 'mono', color: '#33333a' });
      c.fillStyle = C.blue; c.fillRect(140, 4, 12, 10); lib.text(c, '+ self-verification', 158, 2, { size: 11, kind: 'mono', color: '#33333a' });
      [0, 5, 10, 15, 20].forEach(v => { c.strokeStyle = v === 0 ? '#a3a3a8' : '#eeeef0'; c.lineWidth = 1; for (let i = 0; i < 5; i++) { c.beginPath(); c.moveTo(X(v), top + i * rowH + 15); c.lineTo(X(v), top + i * rowH + rowH - 2); c.stroke(); } lib.text(c, String(v), X(v), top + 5 * rowH + 3, { size: 10.5, kind: 'mono', color: '#a3a3a8', align: 'center' }); });
      T2.forEach((r, i) => {
        const y = top + i * rowH, on = i === act;
        if (on) { c.fillStyle = C.blue4; c.fillRect(0, y, 300, rowH - 2); }
        lib.text(c, r[0], 4, y + 2, { size: 11, kind: 'mono', color: on ? '#111' : '#6b6b70', weight: i === 4 ? 700 : 400 });
        [[r[1], false], [r[2], true]].forEach(([v, solid], j) => {
          const yy = y + 17 + j * 9, xa = X(Math.min(0, v)), xb = X(Math.max(0, v)), bw = Math.max(1.5, xb - xa);
          if (solid) { c.fillStyle = on ? C.blue : 'rgba(47,60,255,0.55)'; c.fillRect(xa, yy, bw, 7); }
          else { c.strokeStyle = on ? C.blue : 'rgba(47,60,255,0.55)'; c.lineWidth = 1.1; c.strokeRect(xa + 0.5, yy + 0.5, Math.max(1, bw - 1), 6); }
          lib.text(c, T2S[i][j], v >= 0 ? xb + 4 : xa - 4, yy - 2, { size: 10, kind: 'mono', color: on ? '#111' : '#6b6b70', align: v >= 0 ? 'left' : 'right' });
        });
      });
      roC.innerHTML = act >= 0 ? '<span>your recipe = <b>' + T2[act][0].charAt(0).toLowerCase() + T2[act][0].slice(1) + '</b></span><span>thinking longer <b>' + T2S[act][0] + '</b></span><span>+ verification <b>' + T2S[act][1] + '</b></span>' : '<span>your recipe removes ' + KEYS.filter(k => !S[k]).length + ' regularizers: not a Table 2 row (the paper removed one at a time)</span>';
    }
    cc.canvas.style.cursor = 'pointer';
    cc.canvas.addEventListener('click', (ev) => { const [, py] = cc.toLocal(ev); const i = Math.floor((py - top) / rowH); if (i < 0 || i > 4) return; KEYS.forEach(k => S[k] = T2[i][3] !== k); sync(); update(); });

    // ---------------------------------------------------------------- update + animation of F
    function update() { sampled = sampleAll(); plain = sampleAll({}); drawA(); drawB(); drawC(); }
    const loop = lib.loop((dt) => { const d = S.Ftarget - S.F; if (Math.abs(d) < 0.01) { S.F = S.Ftarget; fSl.set(S.F); drawB(); return false; } S.F += Math.sign(d) * Math.min(Math.abs(d), dt * 1.2); fSl.set(S.F); drawB(); });
    function animF(F) { S.Ftarget = F; if (lib.reducedMotion || !ctx.visible()) { S.F = F; fSl.set(F); drawB(); } else loop.start(); }
    function setRecipe(o) { KEYS.forEach(k => S[k] = !!o[k]); sync(); }
    const caps = [
      'Toy 2-D EBT at one checkpoint and one context (darker = lower energy). Lines: 40 training unrolls (5 batches × 8) sampled live with the recipe above. White dots: ŷᵢ where ∇ŷE is evaluated, the only places training shapes. Blue dot: ŷN. Crosshair: μ(x).',
      'Random α: strides vary per example. Bottom left: exact analysis of a quadratic basin. Dashed = trained with one α (k = 1/α₀); blue = trained over the shaded range; light band = curvatures measured on the trained toy.',
      'Random N: the number of unrolled steps varies per batch (readout). Longer unrolls keep stepping on the floor of the basin.',
      'Langevin noise added to every training step. Drag σ to exaggerate; the toy was trained with σ = 0.05.',
      'Replay buffer at training step 50: dashed ink paths restart from stored earlier predictions. Bottom left: energy along one real chain of four short training unrolls from the toy\'s buffer at step 400 (switch the checkpoint to step 400 to see it on the map).',
      'The paper\'s System 2 recipe: all four on. Click a row of Table 2 to load that ablation in the figure.',
    ];
    sync(); update();
    return {
      step(i) {
        ctx.setCaption(caps[i]); S.chain = false; S.view = 'stride';
        if (i === 0) { setRecipe({}); S.ck = 6; ckSeg.set(6); S.sigma = HP.langevin; sigSl.set(S.sigma); animF(1); }
        if (i === 1) { setRecipe({ rA: 1 }); S.ck = 6; ckSeg.set(6); animF(2); }
        if (i === 2) { setRecipe({ rA: 1, rN: 1 }); S.ck = 6; ckSeg.set(6); animF(2); }
        if (i === 3) { setRecipe({ rA: 1, rN: 1, lang: 1 }); S.ck = 6; ckSeg.set(6); S.sigma = 0.12; sigSl.set(0.12); animF(2); }
        if (i === 4) { setRecipe({ rA: 1, rN: 1, lang: 1, replay: 1 }); S.sigma = HP.langevin; sigSl.set(S.sigma); S.ck = 3; ckSeg.set(3); S.chain = true; S.view = 'replay'; }
        if (i === 5) { setRecipe({ rA: 1, rN: 1, lang: 1, replay: 1 }); S.sigma = HP.langevin; sigSl.set(S.sigma); S.ck = 6; ckSeg.set(6); animF(2); }
        update();
      },
      hide() { loop.stop(); },
    };
  },
});
