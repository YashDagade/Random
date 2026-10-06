/* Panel 10 · Algorithm 1: unroll the optimization, then backpropagate.
   Figure, all live and linked:
   (1) the unrolled computation graph of Algorithm 1 for one highlighted training example, animated forward (blue) and
       backward (ink), with the real energies, error vectors, Jacobian factors 1 − αH and the per-step pieces of ∂L/∂θ;
   (2) the energy of a 3-parameter EBT over (context x, prediction ŷ): every vertical slice is one energy landscape;
       the batch's unrolled paths, the data, the valley floor, and a slice at the highlighted x;
   (3) the outer loop: batch loss and the three parameters as you train.
   Toy energy: E_θ(x, ŷ) = (a/2)(ŷ − u·sin 2πx)² + c·cos 3ŷ, θ = (a, u, c). Exact gradients through the unroll (adjoint
   recursion), checked against central finite differences of the batch loss after every update. */
EBT.panel({
  id: 'alg1',
  nav: 'Algorithm 1: training',
  title: 'Training: unroll the optimization, then backpropagate',
  lede: 'An EBT is trained by letting it think. Start from noise, take N gradient steps on its own energy, score only where the prediction lands, and backpropagate that one loss through every step into the weights.',
  text: `
    <p>Algorithm 1 (p.7) nests two optimizations, and most confusion about EBT training comes from merging them. The <b>inner loop</b> is thinking: gradient descent on the prediction $\\hat y$ with the weights frozen. The <b>outer loop</b> is learning: one gradient step on the weights $\\theta$, using a loss measured only after the inner loop has finished.</p>
    <div class="eq">$$\\begin{aligned}\\hat y_0&\\sim\\mathcal N(0,I)\\\\ \\hat y_{i+1}&=\\hat y_i-\\alpha\\,\\nabla_{\\hat y}E_\\theta(x,\\hat y_i)\\\\ L&=J(\\hat y_N,y)\\end{aligned}$$<span class="why">Alg. 1 lines 1 to 4 and Eq. 1 (p.7); then θ ← θ − η ∇θL (AdamW in the paper).</span></div>
    <p>Here $x$ is the context, $y$ the true target (next token, next frame or clean image), $\\hat y_i$ the prediction after $i$ thinking steps, $\\alpha$ the inner step size, $N$ the number of unrolled steps and $J$ an ordinary supervised loss. Notice what is missing: a target for the energy. No term ever says what $E$ should equal (Fig. E.1, p.37). The weights learn only from where descending their own energy ends up.</p>`,
  steps: [
    { label: 'One iteration at a glance', html: '<p>The graph is Algorithm 1 for the highlighted example: noise in, N blue forward steps, one loss, and the ink backward pass returning through every step to the shared weights θ. The heatmap shows the same model for every context at once: each vertical slice is the energy landscape over $\\hat y$ for one $x$, and the panel on its right is the slice at the highlighted $x$.</p>' },
    { label: 'Sample ŷ₀', html: '<p>Line 1: the prediction starts as noise, $\\hat y_0\\sim\\mathcal N(0,I)$, whatever the context. For text, $\\hat y$ is a vector of 50,277 logits per position (p.43); here it is one number, a hollow dot in its column. Click the heatmap to start from any $(x,\\hat y_0)$.</p>' },
    { label: 'Unroll N thinking steps', html: '<p>Lines 2 and 3. Each box runs a forward pass for $E_\\theta(x,\\hat y_i)$ and a backward pass to the input for $\\nabla_{\\hat y}E$, then moves the prediction. Every box reads the same θ (dashed lines). PyTorch takes this gradient with <code>create_graph=True</code> (Listing 1, p.44), which keeps the gradient\'s own graph so the loss can later be differentiated through it. The paper trains with 2 or 3 steps (p.26).</p><p>Read the energies above the nodes: before training they do not fall. The untrained bumps curve more sharply than this step size can handle, so the highlighted path overshoots and its energy rises. A gradient step is only guaranteed to lower the energy when the step is small for the local curvature, roughly $\\alpha H&lt;2$. Training will have to fix this too.</p>' },
    { label: 'Score only the end', html: '<p>Line 4: $L=J(\\hat y_N,y)$, here $(\\hat y_N-y)^2$. Intermediate guesses are not scored in Algorithm 1 as written, nor in the paper\'s S2 models (below). The backward pass starts from one number per output entry, $\\partial J/\\partial\\hat y_N=2(\\hat y_N-y)$, written under $\\hat y_N$. Everything modality-specific lives in this starting error (table below).</p>' },
    { label: 'Backpropagate through every step', html: '<p>The error $a_i=\\partial L/\\partial\\hat y_i$ travels right to left and does two things at each box. It deposits a piece of the weight gradient, $c_i=-\\alpha\\,[\\partial_\\theta\\nabla_{\\hat y}E_\\theta(\\hat y_{i-1})]^{\\top}a_i$, whose share is printed on that box\'s θ link. And it moves one step back, $a_{i-1}=(1-\\alpha H_{i-1})\\,a_i$ with $H_{i-1}=\\partial^2E/\\partial\\hat y^2$ at $\\hat y_{i-1}$: the factor under each arc, red when it amplifies. Because every box shares θ, $\\nabla_\\theta L=\\sum_i c_i$, as in backpropagation through time. Both pieces differentiate a gradient; the next panel derives them.</p><p>Before training the chain amplifies: the red ×3.53 on the arc from $\\hat y_3$ back to $\\hat y_2$ makes the error arriving at the middle box 3.5 times the one the loss sent, so most of the learning signal (64%) comes from the middle step, not the last. The faint arc back to the noise $\\hat y_0$ carries an error nobody uses, since no weights sit upstream of $\\hat y_0$.</p>' },
    { label: 'Update θ, then repeat', html: '<p>Each update averages $\\nabla_\\theta L$ over a fresh batch of 16 examples, each with its own noise start, and takes an Adam step. Watch three things in the outer-loop chart: the valley slides onto the data ($u\\to1.6$), the bumps that trapped the descent or threw it off course flatten ($c\\to0$), and the curvature settles where N steps of size α reach the floor ($\\alpha a$ enters the shaded band). The loss falls to the noise floor: $y$ itself is noisy, so no prediction can do better than $\\sigma^2=0.15^2$ on average.</p>' },
    { label: 'What the landscape learned', html: '<p>Nothing told the model which energies to output, yet the floor of its landscape now traces $\\mathbb E[y\\mid x]$, every step lowers the energy, and the curvature is tuned to the optimizer. With N = 1 one step must land exactly, which forces $\\alpha a=1$. With more steps any $a$ with $|1-\\alpha a|^N\\approx0$ works (shaded band), because the start is forgotten at rate $(1-\\alpha a)^N$. A side effect shows on the graph: every factor under the arcs is now about $1-\\alpha a\\approx0.1$, so the error dies on its way back and over 90% of $\\nabla_\\theta L$ comes from the last step.</p><p>Change N or α, then <b>[ reset θ ]</b> and <b>[ train ×60 ]</b>: the landscape adapts to how it will be used. Not every setting trains. With N = 6 and α = 1, or N = 3 and α = 1.5, red factors multiply, the gradient becomes erratic and the loss stalls far above the floor. That is the "longer gradient chains" instability that kept the paper at 2 or 3 steps (p.26, p.42); the next panel takes it apart. Shaping the landscape only where the optimizer travels is what Sec. 3.2 calls implicitly regularizing it (p.7).</p>' },
  ],
  after: `
    <h3>Only the loss is modality-specific</h3>
    <div class="tbl"><table>
      <tr><th>modality</th><th>ŷ and J</th><th>starting error ∂J/∂ŷ<sub>N</sub></th></tr>
      <tr><td>text (p.7, p.43)</td><td>50,277 logits per position; cross-entropy</td><td>$\\mathrm{softmax}(\\hat y_N)-e_y$</td></tr>
      <tr><td>images (p.7)</td><td>denoised image; MSE</td><td>$\\tfrac2D(\\hat y_N-y)$</td></tr>
      <tr><td>video (p.12)</td><td>3136-d latent of the next frame; Smooth L1, β = 1</td><td>$\\tfrac1D\\,\\mathrm{clip}(\\hat y_N-y,\\pm1)$</td></tr>
    </table></div>
    <p>The paper reuses each field's standard loss "to simplify experiments" (p.7); $D$ is the number of entries and $e_y$ the one-hot target. For text the logits are softmaxed and linearly projected into the Transformer's embedding space at every step (p.44). The unroll and the backward pass are identical across modalities.</p>
    <h3>What the paper actually runs</h3>
    <p>"Backpropagated through the entire optimization process" (p.7) is the idealized version, and it is what the toy above does. S1 models (many of the scaling runs) detach $\\hat y$ between steps and put the loss on every step. S2 models (the thinking results) keep the chain, truncate backpropagation and score only the last step (p.30, p.36, p.43). Training used N = 2 (Table D.3) or 2 to 3 random steps (Table D.4); the next panel shows what detaching and truncating do to the gradient.</p>
    <p class="note">Toy: $E_\\theta(x,\\hat y)=\\tfrac a2(\\hat y-u\\sin2\\pi x)^2+c\\cos3\\hat y$, θ = (a, u, c), trained in your browser on 72 points $y=1.6\\sin2\\pi x+0.15\\,\\varepsilon$ with $\\varepsilon\\sim\\mathcal N(0,1)$. Full backpropagation through all N steps, loss on the last step only, Adam with learning rate 0.05 and the gradient clipped to norm 1 (the paper lists a gradient clip value of 1, Table D.4). Gradients are exact and checked against central finite differences of the batch loss after every update (readout).</p>`,
  source: [{ kind: 'paper', note: 'Algorithm 1, Eq. 1, p.7; losses p.7, p.12' }, { kind: 'toy', note: '3-parameter EBT trained live, exact gradients' }],
  figure(stage, ctx) {
    const { lib } = ctx, h = lib.h, C = lib.C;
    const BLUE = C.blue || '#2f3cff', BLUE2 = C.blue2 || '#8a93ff', BLUE3 = C.blue3 || '#c9cdff', BLUE4 = C.blue4 || '#eef0ff';
    const INK = C.ink, MUTED = C.muted, FAINT = C.faint, RULE = C.rule, WARN = C.bad || '#d4421c';
    const SUBS = '₀₁₂₃₄₅₆₇₈₉', sub = (n) => String(n).split('').map((d) => SUBS[+d] || d).join('');
    const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
    const fv = (v, d = 2) => (!isFinite(v) ? '–' : (Math.abs(v) >= 1000 ? v.toExponential(1) : v.toFixed(d)).replace('-', '−'));
    // like fv, but tiny non-zero values keep their sign and size (−8e−5) instead of collapsing to −0.00
    const fs = (v) => { if (!isFinite(v)) return '–'; const a = Math.abs(v); if (a === 0) return '0'; if (a < 0.005) return v.toExponential(0).replace('e-', 'e−').replace('-', '−'); return fv(v); };
    const ease = (t) => lib.ease(t);
    const T = (g, s, x, y, o = {}) => lib.text(g, s, x, y, Object.assign({ size: 12, kind: 'mono', color: INK, baseline: 'middle' }, o));

    // ---------------- the 3-parameter EBT (θ = [a, u, c]) ----------------
    const OM = 3, AMP = 1.6, NOISE = 0.15, B = 16, YL = [-3, 3];
    const sx = (x) => Math.sin(2 * Math.PI * x);
    const En = (th, x, y) => 0.5 * th[0] * (y - th[1] * sx(x)) ** 2 + th[2] * Math.cos(OM * y);
    const gE = (th, x, y) => th[0] * (y - th[1] * sx(x)) - th[2] * OM * Math.sin(OM * y);
    const hE = (th, x, y) => th[0] - th[2] * OM * OM * Math.cos(OM * y);
    const mE = (th, x, y) => [y - th[1] * sx(x), -th[0] * sx(x), -OM * Math.sin(OM * y)]; // ∂(∇ŷE)/∂(a, u, c)
    const TH0 = [0.3, -0.4, 0.45];
    function unroll(th, ex, N, al) { const p = [ex.y0]; let y = ex.y0; for (let i = 0; i < N; i++) { y -= al * gE(th, ex.x, y); p.push(y); } return p; }
    // forward + exact backward for one example (MSE loss on the last step)
    function trace(th, ex, N, al) {
      const p = unroll(th, ex, N, al), E = p.map((y) => En(th, ex.x, y));
      const adj = new Array(N + 1).fill(0), pieces = new Array(N + 1).fill(null), fac = new Array(N + 1).fill(1);
      let a = 2 * (p[N] - ex.y); adj[N] = a; const G = [0, 0, 0];
      for (let i = N - 1; i >= 0; i--) {
        const m = mE(th, ex.x, p[i]), c = m.map((v) => -al * a * v);
        pieces[i + 1] = c; for (let k = 0; k < 3; k++) G[k] += c[k];
        fac[i + 1] = 1 - al * hE(th, ex.x, p[i]); a *= fac[i + 1]; adj[i] = a;
      }
      // share of the learning signal carried by each step: ‖c_i‖ / Σ‖c_j‖
      const nrm = pieces.map((c) => (c ? Math.hypot(c[0], c[1], c[2]) : 0)), tot = nrm.reduce((q, v) => q + v, 0);
      const share = nrm.map((v) => (tot > 0 ? v / tot : 0));
      return { p, E, L: (p[N] - ex.y) ** 2, adj, pieces, fac, G, share };
    }
    function batchLG(th, batch, N, al) {
      let L = 0; const G = [0, 0, 0];
      batch.forEach((ex) => { const r = trace(th, ex, N, al); L += r.L; for (let k = 0; k < 3; k++) G[k] += r.G[k]; });
      return { L: L / batch.length, G: G.map((v) => v / batch.length) };
    }
    const batchL = (th, batch, N, al) => batch.reduce((s, ex) => { const p = unroll(th, ex, N, al); return s + (p[N] - ex.y) ** 2; }, 0) / batch.length;
    function fdCheck(th, batch, N, al, G) {
      const hh = 1e-6, fd = [0, 1, 2].map((k) => { const tp = th.slice(), tm = th.slice(); tp[k] += hh; tm[k] -= hh; return (batchL(tp, batch, N, al) - batchL(tm, batch, N, al)) / (2 * hh); });
      const num = Math.hypot(G[0] - fd[0], G[1] - fd[1], G[2] - fd[2]), den = Math.hypot(...G) + Math.hypot(...fd);
      return den > 0 ? num / den : 0;
    }

    // ---------------- data, batches, state ----------------
    const rD = lib.rng(2507); const DATA = [];
    for (let j = 0; j < 72; j++) { const x = rD(); DATA.push({ x, y: AMP * sx(x) + NOISE * rD.normal() }); }
    const SEL0 = { x: 0.27, y: AMP * sx(0.27) + 0.06, y0: -1.15 };
    const S = { N: 3, alpha: 1.0, lr: 0.05, th: TH0.slice(), m: [0, 0, 0], v: [0, 0, 0], t: 0, hist: [], thHist: [], fd: null,
      sel: Object.assign({}, SEL0), batch: null, seed: 1, stage: 0, prog: 1, target: 0, trainLeft: 0, acc: 0 };
    function newBatch() {
      const r = lib.rng(9001 + S.seed * 7919); S.seed++;
      S.batch = Array.from({ length: B }, () => { const d = DATA[Math.floor(r() * DATA.length)]; return { x: d.x, y: d.y, y0: r.normal() }; });
    }
    newBatch();
    let played = false; // the one-time autoplay on first view has happened (or the reader acted first)
    let R = null, BL = null, grid = null, heat = null, floor = null, floor0 = null;
    const LAST = () => 2 * S.N + 1;
    function recompute() { R = trace(S.th, S.sel, S.N, S.alpha); BL = batchLG(S.th, S.batch, S.N, S.alpha); S.fd = fdCheck(S.th, S.batch, S.N, S.alpha, BL.G); buildHeat(); }
    function resetTheta() { S.th = TH0.slice(); S.m = [0, 0, 0]; S.v = [0, 0, 0]; S.t = 0; S.hist = []; S.thHist = []; S.seed = 1; newBatch(); }
    function adamStep() {
      newBatch();
      const { L, G } = batchLG(S.th, S.batch, S.N, S.alpha);
      if (!isFinite(L) || !G.every(isFinite)) return;
      if (S.t === 0) { S.hist.push([0, L]); S.thHist.push([0, S.th.slice()]); }
      const gn = Math.hypot(...G), sc = Math.min(1, 1 / (gn + 1e-12)); S.t++;
      for (let k = 0; k < 3; k++) {
        const g = G[k] * sc; S.m[k] = 0.9 * S.m[k] + 0.1 * g; S.v[k] = 0.999 * S.v[k] + 0.001 * g * g;
        const mh = S.m[k] / (1 - Math.pow(0.9, S.t)), vh = S.v[k] / (1 - Math.pow(0.999, S.t));
        S.th[k] -= S.lr * mh / (Math.sqrt(vh) + 1e-8);
      }
      S.th[0] = clamp(S.th[0], 0.03, 4); S.th[2] = clamp(S.th[2], -1, 1);
      const L2 = batchL(S.th, S.batch, S.N, S.alpha);
      S.hist.push([S.t, L2]); S.thHist.push([S.t, S.th.slice()]);
    }

    // energy over (x, ŷ): grid sampled at cell centres, rows top-down (row 0 = ŷ max)
    const NX = 96, NY = 80, NF = 241;
    function floorOf(th) {
      const pts = []; for (let c = 0; c <= 120; c++) { const x = c / 120; let best = Infinity, by = 0;
        for (let r = 0; r < NF; r++) { const y = YL[0] - 0.2 + (YL[1] - YL[0] + 0.4) * r / (NF - 1), e = En(th, x, y); if (e < best) { best = e; by = y; } }
        pts.push([x, by]); }
      return pts;
    }
    function buildHeat() {
      grid = []; const vals = [];
      for (let r = 0; r < NY; r++) { const y = YL[1] - (YL[1] - YL[0]) * (r + 0.5) / NY, row = new Array(NX);
        for (let c = 0; c < NX; c++) { const e = En(S.th, (c + 0.5) / NX, y); row[c] = e; vals.push(e); } grid.push(row); }
      vals.sort((a, b) => a - b); const lo = vals[0], hi = vals[Math.floor(0.72 * (vals.length - 1))];
      const cv = document.createElement('canvas'); cv.width = NX; cv.height = NY; const gg = cv.getContext('2d'), img = gg.createImageData(NX, NY);
      for (let r = 0; r < NY; r++) for (let c = 0; c < NX; c++) { let v = clamp((grid[r][c] - lo) / (hi - lo || 1), 0, 1); v = Math.pow(v, 0.75); const col = lib.cmap(0.12 + 0.88 * v), i = 4 * (r * NX + c); img.data[i] = col[0]; img.data[i + 1] = col[1]; img.data[i + 2] = col[2]; img.data[i + 3] = 255; }
      gg.putImageData(img, 0, 0); heat = cv; floor = floorOf(S.th);
    }
    floor0 = floorOf(TH0);

    // ---------------- responsive canvas helper ----------------
    function rcv(parent, o) {
      const box = h('div', { class: 'canvas-box' }); parent.appendChild(box);
      const cv = h('canvas', { role: 'img', 'aria-label': o.label }); box.appendChild(cv);
      const g = cv.getContext('2d'); const Rc = { box, cv, g, w: 0, h: 0, dpr: 1 };
      Rc.render = () => {
        const w = Math.max(240, Math.floor(box.clientWidth || parent.clientWidth || 600)), hh = Math.round(o.height(w)), dpr = Math.min(2, window.devicePixelRatio || 1);
        if (w !== Rc.w || hh !== Rc.h || dpr !== Rc.dpr) { cv.width = Math.round(w * dpr); cv.height = Math.round(hh * dpr); cv.style.height = hh + 'px'; Rc.w = w; Rc.h = hh; Rc.dpr = dpr; }
        g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, w, hh); g.fillStyle = '#fff'; g.fillRect(0, 0, w, hh); o.draw(g, w, hh);
      };
      Rc.toLocal = (ev) => { const r = cv.getBoundingClientRect(), p = ev.touches ? ev.touches[0] : ev; return [(p.clientX - r.left) * Rc.w / r.width, (p.clientY - r.top) * Rc.h / r.height]; };
      if (window.ResizeObserver) { let pw = 0; new ResizeObserver(() => { const w = Math.floor(box.clientWidth); if (w && w !== pw) { pw = w; Rc.render(); fitSticky(); } }).observe(box); }
      return Rc;
    }

    // ---------------- layout ----------------
    const F1 = lib.frame(stage, { label: 'Algorithm 1, unrolled · forward blue, backward ink', sub: 'E above each ŷ<sub>i</sub> · ∂L/∂ŷ<sub>i</sub> below · 1 − αH under arcs · share of ∂L/∂θ on θ links' });
    const graphC = rcv(F1.frame, { label: 'Unrolled computation graph of Algorithm 1: noise y-hat-0, N gradient steps sharing the weights theta, the loss J, and the backward pass through every step.', height: (w) => graphLayout(w, S.N).h, draw: drawGraph });
    const row = h('div', { class: 'fig-row' }); stage.appendChild(row);
    const F2 = lib.frame(row, { label: 'Energy over context and prediction', sub: 'E<sub>θ</sub>(x, ŷ) · click to start a path' });
    F2.wrap.style.flex = '1 1 100%';
    const heatC = rcv(F2.frame, { label: 'Heatmap of the toy energy over context x (horizontal) and prediction y-hat (vertical), with the data, the valley floor, the unrolled paths of the batch, and a slice of the energy at the highlighted context.', height: (w) => (w >= 540 ? 204 : Math.round(w * 0.86)), draw: drawHeat });
    const row2 = h('div', { class: 'fig-row' }); stage.appendChild(row2);
    const F3 = lib.frame(row2, { label: 'Outer loop', sub: 'θ ← θ − η∇<sub>θ</sub>L · batch of 16 per update' });
    F3.wrap.style.flex = '1 1 330px';
    const trainC = rcv(F3.frame, { label: 'Training loss per update and the three parameters of the toy energy, with their targets.', height: (w) => (w >= 520 ? 150 : 214), draw: drawTrain });
    const ctlCol = h('div', { class: 'fig-col alg1-ctl' }); ctlCol.style.flex = '1 1 220px'; row2.appendChild(ctlCol);

    // ---------------- drawing: computation graph ----------------
    function graphLayout(w, N) {
      if (w >= 540) {
        const cy = 97, x0 = 32, xJ = w - 40, dx = (xJ - 62 - x0) / N;
        const nodes = [], ops = [null];
        for (let i = 0; i <= N; i++) nodes.push([x0 + i * dx, cy]);
        for (let i = 1; i <= N; i++) ops.push([(nodes[i - 1][0] + nodes[i][0]) / 2, cy]);
        const ow = Math.min(58, dx - 40), oh = 24;
        return {
          horiz: true, h: 174, nodes, ops, ow, oh, r: 15, J: [xJ, cy], Y: [xJ, cy + 58],
          thetaBox: [12, 8, w - 24, 30], link: (i) => [[ops[i][0], 38], [ops[i][0], cy - oh / 2]], shareAt: (i) => [ops[i][0] + 5, 50, 'left'],
          fwd: (i) => [[nodes[i - 1][0] + 16, cy], [ops[i][0] - ow / 2 - 2, cy], [ops[i][0] + ow / 2 + 2, cy], [nodes[i][0] - 17, cy]],
          arc: (i) => [[nodes[i][0], cy + 16], [(nodes[i][0] + nodes[i - 1][0]) / 2, cy + 50], [nodes[i - 1][0], cy + 16]],
          facAt: (i) => [(nodes[i][0] + nodes[i - 1][0]) / 2, cy + 47, 'center'],
          eAt: (i) => [nodes[i][0], cy - 25, 'center'], aAt: (i) => [nodes[i][0], cy + 66, 'center'],
          toJ: [[nodes[N][0] + 16, cy], [xJ - 24, cy]], fromY: [[xJ, cy + 46], [xJ, cy + 15]], lAt: [xJ, cy - 25, 'center'],
          seed: [[xJ - 12, cy + 15], [(xJ + nodes[N][0]) / 2, cy + 50], [nodes[N][0] + 4, cy + 16]],
          legendY: 0,
        };
      }
      const DY = 70, cx = 44, top = 62, nodes = [], ops = [null];
      for (let i = 0; i <= N; i++) nodes.push([cx, top + i * DY]);
      for (let i = 1; i <= N; i++) ops.push([cx, (nodes[i - 1][1] + nodes[i][1]) / 2]);
      const jy = nodes[N][1] + 62, ow = 46, oh = 22, tx = w - 26;
      return {
        horiz: false, h: jy + 52, nodes, ops, ow, oh, r: 15, J: [cx, jy], Y: [cx + 76, jy], tx,
        thetaBox: [tx - 12, 34, 24, jy - 34 + 12], link: (i) => [[cx + ow / 2, ops[i][1]], [tx - 12, ops[i][1]]], shareAt: (i) => [tx - 18, ops[i][1] - 9, 'right'],
        fwd: (i) => [[cx, nodes[i - 1][1] + 16], [cx, ops[i][1] - oh / 2 - 2], [cx, ops[i][1] + oh / 2 + 2], [cx, nodes[i][1] - 17]],
        arc: (i) => [[cx - 16, nodes[i][1]], [cx - 46, ops[i][1]], [cx - 16, nodes[i - 1][1]]],
        facAt: (i) => [cx + ow / 2 + 8, ops[i][1] - 9, 'left'],
        eAt: (i) => [cx + 24, nodes[i][1] - 8, 'left'], aAt: (i) => [cx + 24, nodes[i][1] + 9, 'left'],
        toJ: [[cx, nodes[N][1] + 16], [cx, jy - 15]], fromY: [[cx + 64, jy], [cx + 25, jy]], lAt: [cx + 96, jy, 'left'],
        seed: [[cx - 24, jy - 6], [cx - 50, (jy + nodes[N][1]) / 2], [cx - 16, nodes[N][1] + 2]],
        legendY: jy + 40,
      };
    }
    const bez = (P, t) => [0, 1].map((k) => (1 - t) * (1 - t) * P[0][k] + 2 * (1 - t) * t * P[1][k] + t * t * P[2][k]);
    function arcLine(g, P, prog, color, lw) {
      const n = 24, pts = []; for (let q = 0; q <= n; q++) pts.push(bez(P, q / n));
      lib.line(g, pts, { color, width: lw, progress: prog });
      if (prog >= 1) { const a = bez(P, 0.9), b = P[2]; lib.arrow(g, a[0], a[1], b[0], b[1], { color, width: lw, head: 6 }); }
      else if (prog > 0) { const d = bez(P, prog); lib.dot(g, d[0], d[1], 3.5, color); }
    }
    function drawGraph(g, w) {
      const N = S.N, L = graphLayout(w, N), st = S.stage, p = ease(S.prog), k = st - (N + 1), anim = S.prog < 1;
      // θ bar (shared weights)
      const [bx, by, bw, bh] = L.thetaBox;
      g.save(); g.fillStyle = BLUE4; g.strokeStyle = INK; g.lineWidth = 1; g.fillRect(bx, by, bw, bh); g.strokeRect(bx + 0.5, by + 0.5, bw - 1, bh - 1); g.restore();
      const thTxt = 'a ' + fv(S.th[0]) + '  u ' + fv(S.th[1]) + '  c ' + fv(S.th[2]);
      const done = k >= N;
      if (L.horiz) {
        T(g, 'θ', bx + 10, by + bh / 2, { kind: 'display', size: 17, weight: 600 });
        T(g, 'shared weights  ' + thTxt, bx + 26, by + bh / 2, { color: MUTED });
        if (done) T(g, '∂L/∂θ  ' + fs(R.G[0]) + ', ' + fs(R.G[1]) + ', ' + fs(R.G[2]), bx + bw - 10, by + bh / 2, { align: 'right', weight: 700 });
      } else {
        T(g, 'θ', bx + bw / 2, by + 13, { kind: 'display', size: 15, weight: 600, align: 'center' });
        T(g, 'θ shared: ' + thTxt, 8, 14, { color: MUTED, size: 11.5 });
        if (done) T(g, '∂L/∂θ = (' + fs(R.G[0]) + ', ' + fs(R.G[1]) + ', ' + fs(R.G[2]) + ')', 8, L.legendY, { weight: 700, size: 11.5 });
      }
      // θ links: each step reads the same weights; on the way back each step deposits its piece
      for (let i = 1; i <= N; i++) {
        const [a, b] = L.link(i), used = st >= i, dep = k >= N - i + 1, fwdAnim = st === i && anim, bwdAnim = k === N - i + 1 && anim;
        g.save(); g.lineWidth = dep ? 1.4 : 1; g.strokeStyle = dep ? INK : used ? BLUE2 : RULE; if (!dep) g.setLineDash([3, 3]);
        g.beginPath(); g.moveTo(a[0], a[1]); g.lineTo(b[0], b[1]); g.stroke(); g.restore();
        if (fwdAnim) lib.dot(g, lib.lerp(a[0], b[0], p), lib.lerp(a[1], b[1], p), 3.5, BLUE);
        if (bwdAnim) lib.dot(g, lib.lerp(b[0], a[0], p), lib.lerp(b[1], a[1], p), 3.5, INK);
        if (dep && !bwdAnim) { const [sx0, sy0, al] = L.shareAt(i); T(g, Math.round(R.share[i] * 100) + '%', sx0, sy0, { size: 11, align: al, weight: 600 }); }
      }
      // forward arrows and op boxes
      for (let i = 1; i <= N; i++) {
        const f = L.fwd(i), vis = st >= i, pr = st === i ? p : 1, o = L.ops[i];
        if (vis) {
          lib.arrow(g, f[0][0], f[0][1], f[1][0], f[1][1], { color: BLUE, width: 1.5, head: 6, progress: clamp(pr * 2, 0, 1) });
          if (pr > 0.5) lib.arrow(g, f[2][0], f[2][1], f[3][0], f[3][1], { color: BLUE, width: 1.5, head: 6, progress: (pr - 0.5) * 2 });
        }
        const bdone = k >= N - i + 1, active = (st === i && anim) || (k === N - i + 1 && anim);
        g.save(); g.fillStyle = active ? (k === N - i + 1 ? '#f2f2f4' : BLUE4) : '#fff'; g.strokeStyle = bdone ? INK : vis ? BLUE : RULE; g.lineWidth = 1.2;
        g.fillRect(o[0] - L.ow / 2, o[1] - L.oh / 2, L.ow, L.oh); g.strokeRect(o[0] - L.ow / 2 + 0.5, o[1] - L.oh / 2 + 0.5, L.ow - 1, L.oh - 1); g.restore();
        T(g, L.ow >= 50 ? '−α∇E' : '∇E', o[0], o[1] + 0.5, { size: 11.5, align: 'center', color: vis ? INK : FAINT });
      }
      // backward arcs with the Jacobian factor 1 − αH
      for (let i = N; i >= 1; i--) {
        const kk = N - i + 1; if (k < kk) continue;
        // the arc back to ŷ0 is faint: ŷ0 is noise with no weights upstream, so its error is never used
        const pr = k === kk ? p : 1, toNoise = i === 1; arcLine(g, L.arc(i), pr, toNoise ? FAINT : INK, toNoise ? 1 : 1.3);
        if (pr >= 1) { const [fx, fy, al] = L.facAt(i), f = R.fac[i]; T(g, '×' + fv(f), fx, fy, { size: 11, align: al, color: toNoise ? FAINT : Math.abs(f) > 1 ? WARN : INK }); }
      }
      // loss
      const lossVis = st >= N + 1, lp = st === N + 1 ? p : 1, J = L.J, Y = L.Y;
      if (lossVis) {
        lib.arrow(g, L.toJ[0][0], L.toJ[0][1], L.toJ[1][0], L.toJ[1][1], { color: BLUE, width: 1.5, head: 6, progress: lp });
        lib.arrow(g, L.fromY[0][0], L.fromY[0][1], L.fromY[1][0], L.fromY[1][1], { color: INK, width: 1.2, head: 6, progress: lp });
        if (lp >= 1) arcLine(g, L.seed, 1, INK, 1.3);
      }
      g.save(); g.fillStyle = '#fff'; g.strokeStyle = lossVis ? INK : RULE; g.lineWidth = 1.2; g.fillRect(J[0] - 22, J[1] - 14, 44, 28); g.strokeRect(J[0] - 21.5, J[1] - 13.5, 43, 27); g.restore();
      T(g, 'J', J[0], J[1] + 0.5, { size: 14, align: 'center', color: lossVis ? INK : FAINT, weight: 600 });
      // target y: crosshair in a dashed circle (shape, not colour)
      g.save(); g.strokeStyle = lossVis ? INK : FAINT; g.lineWidth = 1; g.setLineDash([2, 2]); g.beginPath(); g.arc(Y[0], Y[1], 11, 0, 7); g.stroke(); g.restore();
      T(g, 'y', Y[0], Y[1] + 0.5, { size: 12, align: 'center', color: lossVis ? INK : FAINT });
      if (lossVis && lp >= 1) { const [lx, ly, al] = L.lAt; T(g, 'L ' + fv(R.L), lx, ly, { size: 12, align: al, weight: 700 }); }
      // nodes ŷ_i with energies (above) and errors ∂L/∂ŷ_i (below)
      for (let i = 0; i <= N; i++) {
        const n = L.nodes[i], vis = i <= st, fresh = st === i && anim && i > 0, al = vis ? (fresh ? clamp((p - 0.5) * 2, 0.15, 1) : 1) : 0.4;
        g.save(); g.globalAlpha = al;
        lib.dot(g, n[0], n[1], L.r, i === 0 ? '#fff' : vis ? BLUE4 : '#fff', { stroke: i === 0 ? INK : vis ? BLUE : RULE, lw: 1.3 });
        T(g, 'ŷ' + sub(i), n[0], n[1] + 0.5, { size: 12, align: 'center', color: vis ? INK : FAINT });
        g.restore();
        if (vis && (!fresh || p > 0.7)) { const [ex, ey, a2] = L.eAt(i); T(g, 'E ' + fv(R.E[i]), ex, ey, { size: 11, align: a2, color: MUTED }); }
        const known = k >= N - i && !(k === N - i && anim && i < N);
        if (known) { const [ax, ay, a2] = L.aAt(i); T(g, (L.horiz ? '' : '∂L/∂ŷ ') + fs(R.adj[i]), ax, ay, { size: 11, align: a2, weight: i === 0 ? 400 : 600, color: i === 0 ? FAINT : INK }); }
      }
    }

    // ---------------- drawing: energy over (x, ŷ) + slice ----------------
    function heatLayout(w, hh) {
      const left = 36, bottom = 34, top = 10, gap = w >= 540 ? 22 : 14;
      const avail = w - left - 8 - gap, hw = Math.round(avail * (w >= 540 ? 0.66 : 0.62));
      return { hb: { x: left, y: top, w: hw, h: hh - top - bottom }, sb: { x: left + hw + gap, y: top, w: avail - hw, h: hh - top - bottom } };
    }
    const Yp = (b, y) => b.y + (YL[1] - y) / (YL[1] - YL[0]) * b.h;
    const Xp = (b, x) => b.x + x * b.w;
    let lastHeatLayout = null;
    function plate(g, s, x, y, o = {}) { const m = lib.measure(g, s, { size: 11, kind: 'mono' }); g.save(); g.fillStyle = 'rgba(255,255,255,0.88)'; g.fillRect(x - 3, y - 8, m.w + 6, 16); g.restore(); T(g, s, x, y, Object.assign({ size: 11, weight: 600 }, o)); }
    function haloLine(g, pts, color, wdt, dash) { lib.line(g, pts, { color: 'rgba(255,255,255,0.9)', width: wdt + 2.6 }); lib.line(g, pts, { color, width: wdt, dash }); }
    function cross(g, x, y, col) { g.save(); g.strokeStyle = col; g.lineWidth = 1.3; g.beginPath(); g.moveTo(x - 6, y); g.lineTo(x + 6, y); g.moveTo(x, y - 6); g.lineTo(x, y + 6); g.stroke(); g.setLineDash([2, 2]); g.beginPath(); g.arc(x, y, 9, 0, 7); g.stroke(); g.restore(); }
    function drawHeat(g, w, hh) {
      const { hb, sb } = heatLayout(w, hh); lastHeatLayout = { hb, sb };
      const N = S.N, st = S.stage, upto = Math.min(st, N), pr = S.prog < 1 && st >= 1 && st <= N ? ease(S.prog) : 1;
      g.save(); g.imageSmoothingEnabled = true; g.imageSmoothingQuality = 'high'; g.drawImage(heat, hb.x, hb.y, hb.w, hb.h); g.restore();
      g.save(); g.strokeStyle = INK; g.lineWidth = 1; g.strokeRect(hb.x + 0.5, hb.y + 0.5, hb.w - 1, hb.h - 1); g.restore();
      g.save(); g.beginPath(); g.rect(hb.x, hb.y, hb.w, hb.h); g.clip();
      // valley floor now (white) and at the start of training (dashed)
      const seg = (pts, dash, col, wd) => { let cur = []; pts.forEach((q, j) => { if (j && Math.abs(q[1] - pts[j - 1][1]) > 0.5) { if (cur.length > 1) lib.line(g, cur, { color: col, width: wd, dash }); cur = []; } cur.push([Xp(hb, q[0]), Yp(hb, q[1])]); }); if (cur.length > 1) lib.line(g, cur, { color: col, width: wd, dash }); };
      if (S.t > 0) seg(floor0, [4, 4], 'rgba(17,17,17,0.55)', 1.1);
      seg(floor, null, 'rgba(255,255,255,0.95)', 1.6);
      // data
      DATA.forEach((d) => lib.dot(g, Xp(hb, d.x), Yp(hb, d.y), 2.2, '#fff', { stroke: INK, lw: 0.8 }));
      // batch paths (last state of the batch used for training)
      if (st >= 0) S.batch.forEach((ex) => {
        const p = unroll(S.th, ex, N, S.alpha), xp = Xp(hb, ex.x), last = p[Math.max(0, upto)];
        if (upto > 0) haloLine(g, [[xp, Yp(hb, p[0])], [xp, Yp(hb, clamp(last, -4, 4))]], BLUE, 1.1);
        lib.dot(g, xp, Yp(hb, p[0]), 2.6, '#fff', { stroke: BLUE, lw: 1.1 });
        if (upto > 0) lib.dot(g, xp, Yp(hb, clamp(last, -4, 4)), 2.6, BLUE);
      });
      // highlighted example
      const xs = Xp(hb, S.sel.x);
      g.save(); g.strokeStyle = 'rgba(17,17,17,0.7)'; g.setLineDash([2, 3]); g.lineWidth = 1; g.beginPath(); g.moveTo(xs, hb.y); g.lineTo(xs, hb.y + hb.h); g.stroke(); g.restore();
      cross(g, xs, Yp(hb, S.sel.y), INK);
      const P = R.p;
      lib.dot(g, xs, Yp(hb, P[0]), 4.5, '#fff', { stroke: INK, lw: 1.5 });
      if (upto > 0) { const yy = upto === st ? lib.lerp(P[upto - 1], P[upto], pr) : P[upto]; haloLine(g, [[xs, Yp(hb, P[0])], [xs, Yp(hb, clamp(yy, -4, 4))]], BLUE, 2); lib.dot(g, xs, Yp(hb, clamp(yy, -4, 4)), 4.5, BLUE, { stroke: '#fff', lw: 1.2 }); }
      g.restore();
      // axes
      [0, 0.5, 1].forEach((v) => T(g, String(v), Xp(hb, v), hb.y + hb.h + 11, { size: 11, align: 'center', color: MUTED }));
      T(g, 'context x', hb.x + hb.w / 2, hb.y + hb.h + 26, { size: 11.5, align: 'center', color: MUTED });
      [-2, 0, 2].forEach((v) => T(g, fv(v, 0), hb.x - 6, Yp(hb, v), { size: 11, align: 'right', color: MUTED }));
      T(g, 'ŷ', 10, hb.y + 8, { size: 13, color: INK });
      plate(g, 'x = ' + fv(S.sel.x), clamp(xs + 5, hb.x + 3, hb.x + hb.w - 62), hb.y + 12);

      // slice at the highlighted x: E on the horizontal axis, ŷ shared with the heatmap
      const ys = [], es = []; for (let r = 0; r < 181; r++) { const y = YL[0] + (YL[1] - YL[0]) * r / 180; ys.push(y); es.push(En(S.th, S.sel.x, y)); }
      const lo = Math.min(...es), sorted = es.slice().sort((a, b) => a - b), hi = lo + Math.max(1.2, Math.min(7, sorted[Math.floor(0.8 * (sorted.length - 1))] - lo));
      const Xe = (e) => sb.x + 4 + (e - lo) / (hi - lo) * (sb.w - 8);
      g.save(); g.strokeStyle = RULE; g.lineWidth = 1; g.strokeRect(sb.x + 0.5, sb.y + 0.5, sb.w - 1, sb.h - 1);
      [-2, 0, 2].forEach((v) => { g.beginPath(); g.moveTo(sb.x, Yp(sb, v) + 0.5); g.lineTo(sb.x + sb.w, Yp(sb, v) + 0.5); g.stroke(); });
      g.beginPath(); g.rect(sb.x, sb.y, sb.w, sb.h); g.clip();
      // target
      g.strokeStyle = INK; g.setLineDash([3, 3]); g.beginPath(); g.moveTo(sb.x, Yp(sb, S.sel.y)); g.lineTo(sb.x + sb.w, Yp(sb, S.sel.y)); g.stroke(); g.setLineDash([]);
      lib.line(g, ys.map((y, j) => [Xe(es[j]), Yp(sb, y)]), { color: BLUE, width: 1.8 });
      // path on the curve
      const pp = []; for (let i = 0; i <= upto; i++) { let y = P[i]; if (i === upto && upto === st && i > 0) y = lib.lerp(P[i - 1], P[i], pr); pp.push([Xe(En(S.th, S.sel.x, y)), Yp(sb, y)]); }
      for (let i = 1; i < pp.length; i++) lib.arrow(g, pp[i - 1][0], pp[i - 1][1], pp[i][0], pp[i][1], { color: INK, width: 1.1, head: 5 });
      pp.forEach((q, i) => lib.dot(g, q[0], q[1], i === 0 ? 4 : 3.5, i === 0 ? '#fff' : BLUE, { stroke: i === 0 ? INK : '#fff', lw: 1.2 }));
      g.restore();
      T(g, 'y', sb.x + sb.w - 8, Yp(sb, S.sel.y) - 9, { size: 11.5, align: 'right' });
      if (pp.length) T(g, 'ŷ' + sub(0), pp[0][0] + 7, pp[0][1] + (pp.length > 1 && P[1] > P[0] ? 9 : -9), { size: 11.5, color: MUTED });
      if (upto > 0 && (st > N || S.prog >= 1)) { const q = pp[pp.length - 1]; T(g, 'ŷ' + sub(upto), clamp(q[0] + 8, sb.x + 2, sb.x + sb.w - 22), q[1] + 10, { size: 11.5, color: BLUE, weight: 600 }); }
      T(g, 'E at x = ' + fv(S.sel.x) + ' →', sb.x + sb.w / 2, sb.y + sb.h + 11, { size: 11, align: 'center', color: MUTED });
      T(g, 'slice', sb.x + sb.w / 2, sb.y + sb.h + 26, { size: 11.5, align: 'center', color: MUTED });
    }

    // ---------------- drawing: outer loop ----------------
    function drawTrain(g, w, hh) {
      const wide = w >= 520, tmax = Math.max(60, S.t);
      const lb = wide ? { x: 46, y: 12, w: Math.round(w * 0.5) - 60, h: hh - 44 } : { x: 46, y: 10, w: w - 58, h: 66 };
      const L0 = Math.log10(0.005), L1 = Math.log10(20);
      const X = (t) => lb.x + t / tmax * lb.w, Y = (v) => lb.y + lb.h - (Math.log10(clamp(v, 0.005, 20)) - L0) / (L1 - L0) * lb.h;
      g.save(); g.strokeStyle = RULE; g.lineWidth = 1;
      [0.01, 0.1, 1, 10].forEach((v) => { g.beginPath(); g.moveTo(lb.x, Y(v) + 0.5); g.lineTo(lb.x + lb.w, Y(v) + 0.5); g.stroke(); T(g, String(v), lb.x - 6, Y(v), { size: 10.5, align: 'right', color: MUTED }); });
      g.strokeStyle = FAINT; g.beginPath(); g.moveTo(lb.x + 0.5, lb.y); g.lineTo(lb.x + 0.5, lb.y + lb.h + 0.5); g.lineTo(lb.x + lb.w, lb.y + lb.h + 0.5); g.stroke();
      g.strokeStyle = INK; g.setLineDash([4, 3]); g.beginPath(); g.moveTo(lb.x, Y(NOISE * NOISE)); g.lineTo(lb.x + lb.w, Y(NOISE * NOISE)); g.stroke(); g.restore();
      T(g, 'noise floor σ²', lb.x + 6, Y(NOISE * NOISE) + 9, { size: 10.5, color: MUTED });
      T(g, 'update ' + S.t + ' · L ' + fv(S.hist.length ? S.hist[S.hist.length - 1][1] : BL.L, 3), lb.x + lb.w, lb.y + lb.h + 12, { size: 10.5, align: 'right', color: INK });
      T(g, 'batch loss (log)', lb.x, lb.y + lb.h + 12, { size: 10.5, color: MUTED });
      const pts = S.hist.length ? S.hist.map(([t, v]) => [X(t), Y(v)]) : [[X(0), Y(BL.L)]];
      if (pts.length > 1) lib.line(g, pts, { color: BLUE, width: 1.6 });
      const lp = pts[pts.length - 1]; lib.dot(g, lp[0], lp[1], 3.2, BLUE);
      // parameters
      const pb = wide ? { x: lb.x + lb.w + 30, y: 8, w: w - (lb.x + lb.w + 30) - 10, h: hh - 22 } : { x: 8, y: lb.y + lb.h + 22, w: w - 16, h: hh - (lb.y + lb.h + 22) - 2 };
      const rh = pb.h / 3, N = S.N, al = S.alpha, band = Math.pow(0.05, 1 / N);
      const rows = [
        { name: 'α·a', get: (th) => al * th[0], lo: 0, hi: 2.2, tgt: 1, band: [1 - band, 1 + band], note: N === 1 ? 'one step: αa = 1' : 'band: |1−αa|ᴺ < 0.05' },
        { name: 'u', get: (th) => th[1], lo: -0.8, hi: 2.2, tgt: AMP, note: 'data amplitude 1.6' },
        { name: 'c', get: (th) => th[2], lo: -0.3, hi: 0.6, tgt: 0, note: 'no bumps: 0' },
      ];
      const hist = S.thHist.length ? S.thHist : [[0, S.th]];
      rows.forEach((r, j) => {
        const y0 = pb.y + j * rh, bx = pb.x + 54, bw = pb.w - 54, ry = (v) => y0 + rh - 6 - (clamp(v, r.lo, r.hi) - r.lo) / (r.hi - r.lo) * (rh - 12);
        const Xt = (t) => bx + t / tmax * bw;
        if (r.band) { g.save(); g.fillStyle = BLUE4; g.fillRect(bx, ry(Math.min(r.hi, r.band[1])), bw, ry(Math.max(r.lo, r.band[0])) - ry(Math.min(r.hi, r.band[1]))); g.restore(); }
        g.save(); g.strokeStyle = INK; g.setLineDash([3, 3]); g.lineWidth = 1; g.beginPath(); g.moveTo(bx, ry(r.tgt)); g.lineTo(bx + bw, ry(r.tgt)); g.stroke(); g.restore();
        g.save(); g.strokeStyle = RULE; g.strokeRect(bx + 0.5, y0 + 2.5, bw - 1, rh - 5); g.restore();
        const ps = hist.map(([t, th]) => [Xt(t), ry(r.get(th))]);
        if (ps.length > 1) lib.line(g, ps, { color: BLUE, width: 1.4 });
        lib.dot(g, ps[ps.length - 1][0], ps[ps.length - 1][1], 2.8, BLUE);
        T(g, r.name, pb.x, y0 + rh / 2 - 7, { size: 12, weight: 600 });
        T(g, fv(r.get(S.th)), pb.x, y0 + rh / 2 + 8, { size: 11.5, weight: 600, color: BLUE });
        const ny = ry(r.tgt) < y0 + rh / 2 ? ry(r.tgt) + 9 : ry(r.tgt) - 8;
        plate(g, r.note, bx + bw - 4 - lib.measure(g, r.note, { size: 10.5, kind: 'mono' }).w, ny, { size: 10.5, weight: 400, color: MUTED });
      });
    }

    // ---------------- controls and readouts ----------------
    const ctl1 = h('div', { class: 'controls' }); ctlCol.appendChild(ctl1);
    const bNew = lib.button('new ŷ₀', () => { act(); S.sel.y0 = lib.rng(31 + S.seed++ * 17).normal() * 1.1; recompute(); S.stage = 0; S.prog = 1; drawAll(); });
    const bFwd = lib.button('forward', () => { played = true; stopTrain(); playTo(S.N + 1, 0); }, { primary: true });
    const bBwd = lib.button('backward', () => { played = true; stopTrain(); playTo(LAST(), S.stage >= S.N + 1 && S.stage < LAST() ? S.stage : S.N + 1); });
    const bTrain1 = lib.button('train step', () => { act(); adamStep(); recompute(); S.stage = LAST(); S.prog = 1; drawAll(); });
    const bTrainN = lib.button('train ×60', () => { act(); startTrain(60); });
    const bReset = lib.button('reset θ', () => { act(); resetTheta(); recompute(); S.stage = LAST(); S.prog = 1; drawAll(); });
    [bNew, bFwd, bBwd, bTrain1, bTrainN, bReset].forEach((b) => ctl1.appendChild(b));
    const ctl2 = h('div', { class: 'controls' }); ctlCol.appendChild(ctl2);
    const slN = lib.slider({ id: 'alg1-n', label: 'unrolled steps N', min: 1, max: 6, step: 1, value: S.N, oninput: (v) => { act(); S.N = v; recompute(); S.stage = LAST(); S.prog = 1; drawAll(); } });
    const slA = lib.slider({ id: 'alg1-alpha', label: 'inner step size α', min: 0.25, max: 1.5, step: 0.05, value: S.alpha, fmt: (v) => v.toFixed(2), oninput: (v) => { act(); S.alpha = v; recompute(); S.stage = LAST(); S.prog = 1; drawAll(); } });
    ctl2.appendChild(slN.el); ctl2.appendChild(slA.el);
    const ro = h('div', { class: 'readout', 'aria-live': 'polite' }); ctlCol.appendChild(ro);
    function readout() {
      ro.innerHTML = `<span>batch ∇θL <b>(${BL.G.map((v) => fs(v)).join(', ')})</b></span><span title="relative error of the exact batch gradient against central finite differences">vs finite diff. <b>${S.fd == null ? '–' : S.fd.toExponential(0).replace('e-', 'e−')}</b></span>`;
    }

    // ---------------- animation ----------------
    const anim = lib.loop((dt) => {
      if (S.trainLeft > 0) {
        S.acc += dt; let did = false;
        while (S.acc >= 0.07 && S.trainLeft > 0) { S.acc -= 0.07; adamStep(); S.trainLeft--; did = true; }
        if (did) { recompute(); S.stage = LAST(); S.prog = 1; drawAll(); }
        return S.trainLeft > 0 || S.stage < S.target || S.prog < 1;
      }
      S.prog = Math.min(1, S.prog + dt / 0.55);
      if (S.prog >= 1 && S.stage < S.target) { S.stage++; S.prog = 0; }
      drawFast();
      return S.stage < S.target || S.prog < 1;
    });
    function playTo(target, from) {
      S.target = Math.min(target, LAST());
      if (from != null) { S.stage = from; S.prog = 1; }
      if (lib.reducedMotion) { S.stage = S.target; S.prog = 1; drawAll(); return; }
      if (S.stage < S.target) { S.stage++; S.prog = 0; }
      anim.start();
    }
    function startTrain(n) {
      if (lib.reducedMotion) { for (let i = 0; i < n; i++) adamStep(); recompute(); S.stage = LAST(); S.prog = 1; drawAll(); return; }
      S.trainLeft = n; S.acc = 0; S.stage = LAST(); S.prog = 1; S.target = S.stage; anim.start();
    }
    function stopTrain() { S.trainLeft = 0; }
    function stopAnim() { S.trainLeft = 0; anim.stop(); S.prog = 1; S.target = S.stage; }
    function act() { played = true; stopAnim(); }
    function drawFast() { graphC.render(); heatC.render(); readout(); }
    function drawAll() { graphC.render(); heatC.render(); trainC.render(); readout(); fitSticky(); }

    // click on the heatmap: start a new highlighted example at that context and ŷ0
    heatC.cv.addEventListener('click', (ev) => {
      const Lh = lastHeatLayout; if (!Lh) return; const [px, py] = heatC.toLocal(ev), hb = Lh.hb;
      if (px < hb.x || px > hb.x + hb.w || py < hb.y || py > hb.y + hb.h) return;
      const x = clamp((px - hb.x) / hb.w, 0.01, 0.99), y0 = YL[1] - (py - hb.y) / hb.h * (YL[1] - YL[0]);
      const r = lib.rng(Math.floor(x * 1e6) + 7); act();
      S.sel = { x, y: AMP * sx(x) + NOISE * r.normal(), y0 }; recompute(); S.stage = 0; S.prog = 1; drawAll(); playTo(LAST(), 0);
    });

    // keep the sticky figure fully visible when it is taller than the window
    function fitSticky() {
      try {
        stage.style.position = ''; if (getComputedStyle(stage).position !== 'sticky') { stage.style.top = ''; return; }
        const hh = stage.offsetHeight, vh = window.innerHeight;
        if (hh > vh * 1.2) { stage.style.position = 'static'; stage.style.top = ''; return; } // far taller than the window: let it scroll
        stage.style.top = Math.min(24, vh - hh - 6) + 'px'; // a bit taller: stick with its bottom edge in view
      } catch (e) { /* ignore */ }
    }
    window.addEventListener('resize', fitSticky);

    ctx.setCaption('Top: Algorithm 1 for the highlighted example (crosshair). Heatmap: darker = lower energy; white line: valley floor (dashed: before training); dots: data; blue: the batch\'s predictions after N steps (hollow: ŷ₀). Click the heatmap to unroll another example.');
    recompute(); S.stage = LAST(); S.prog = 1; drawAll();
    setTimeout(fitSticky, 300);

    return {
      step(i) {
        stopAnim(); if (i > 0) played = true;
        S.sel = Object.assign({}, SEL0); // the steps narrate the canonical example (N = 3, α = 1); clicks and sliders explore others
        const changed = S.N !== 3 || S.alpha !== 1;
        if (changed) { S.N = 3; S.alpha = 1; slN.set(3); slA.set(1); }
        if ((i <= 4 || changed) && S.t > 0) { resetTheta(); }
        if (i <= 4) recompute();
        if (i === 0) { S.stage = LAST(); S.prog = 1; drawAll(); }
        if (i === 1) { S.stage = 0; S.prog = 1; drawAll(); }
        if (i === 2) { S.stage = 0; S.prog = 1; drawAll(); playTo(S.N, 0); }
        if (i === 3) { S.stage = S.N; S.prog = 1; drawAll(); playTo(S.N + 1, S.N); }
        if (i === 4) { S.stage = S.N + 1; S.prog = 1; drawAll(); playTo(LAST(), S.N + 1); }
        if (i === 5) { recompute(); S.stage = LAST(); S.prog = 1; drawAll(); startTrain(S.t < 60 ? 60 - S.t : 30); }
        if (i === 6) { if (S.t < 120) { const n = 120 - S.t; for (let j = 0; j < n; j++) adamStep(); } recompute(); S.stage = LAST(); S.prog = 1; drawAll(); }
      },
      show() {
        fitSticky();
        // first time in view: play one full iteration (forward, loss, backward) unless the reader has already acted
        if (!played && !lib.reducedMotion && ctx.step === 0 && S.t === 0 && S.stage === LAST() && !anim.running) { played = true; setTimeout(() => { if (ctx.visible() && S.t === 0 && !anim.running) playTo(LAST(), 0); }, 500); }
      },
      hide() { stopAnim(); },
    };
  },
});
