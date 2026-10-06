/* Panel 11 · Gradients of gradients.
   A 1-D energy E_θ(ŷ) = (a/2)(ŷ − m)² + c·cos 3ŷ with three weights θ = (m, a, c). Pick one weight and drag it:
   (1) the energy curve and the unrolled path, with a ghost at θ + δ showing how a nudge to the weight moves the path;
   (2) the training loss L(θ) along that weight, with the exact tangent dL/dθ (analytic, second order), a finite-difference
       check, and the gradient you get when backpropagation is truncated to the last K steps (K = 1: detached, as in S1;
       K = 0: stop-gradient on ∇ŷE);
   (3) the backward chain: per step the Jacobian factor 1 − αH, the error arriving there, and the step's term of dL/dθ.
   All derivatives are analytic: ∇E, H = ∂²E/∂ŷ², and the mixed ∂(∇ŷE)/∂θ. */
EBT.panel({
  id: 'second-order',
  nav: 'Gradients of gradients',
  title: 'Gradients of gradients',
  lede: 'The loss reaches the weights only through $\\nabla_{\\hat y}E$, the slope the model followed while thinking. So training has to differentiate a gradient. Here is that derivative, checked live, and what breaks when you cut it.',
  text: `
    <p>Take one thinking step and score it: $\\hat y_1=\\hat y_0-\\alpha\\nabla_{\\hat y}E_\\theta(\\hat y_0)$, $L=J(\\hat y_1,y)$. The start $\\hat y_0$ is noise and $J$ contains no θ, so a weight can only act through the gradient that moved the prediction:</p>
    <div class="eq">$$\\frac{dL}{d\\theta}=\\Big(\\frac{\\partial \\hat y_1}{\\partial\\theta}\\Big)^{\\!\\top}\\frac{\\partial J}{\\partial \\hat y_1},\\qquad \\frac{\\partial \\hat y_1}{\\partial \\theta}=-\\alpha\\,\\nabla_\\theta\\nabla_{\\hat y}E_\\theta(\\hat y_0)$$<span class="why">a mixed second derivative: how the slope in ŷ changes when a weight moves [derived from Eq. 1, p.7]</span></div>
    <p>With N steps, a change made at step $k$ must also pass through every later step, and step $k$ maps a perturbation of its input through the Jacobian $\\partial\\hat y_{k+1}/\\partial\\hat y_k=I-\\alpha H(\\hat y_k)$, where $H=\\nabla^2_{\\hat y}E_\\theta$ is the Hessian of the energy in prediction space. Backpropagation evaluates this right to left with the error $a_k=\\partial L/\\partial\\hat y_k$:</p>
    <div class="eq">$$\\begin{aligned}a_N&=\\nabla_{\\hat y_N}J\\\\ a_k&=a_{k+1}-\\alpha\\,H(\\hat y_k)\\,a_{k+1}\\\\ \\frac{dL}{d\\theta}&=-\\alpha\\sum_{k=0}^{N-1}\\nabla_\\theta\\big[\\nabla_{\\hat y}E_\\theta(\\hat y_k)^{\\top}a_{k+1}\\big]\\end{aligned}$$<span class="why">H a is a Hessian-vector product; the last line is a mixed one. No matrix is ever formed.</span></div>
    <p>Unrolled, with the loss on the last step only, the error that step $k$ receives has passed through every later factor:</p>
    <div class="eq">$$a_{k+1}=(I-\\alpha H_{k+1})\\cdots(I-\\alpha H_{N-1})\\,a_N,\\qquad H_j=H(\\hat y_j)$$</div>
    <p>Factors inside ±1 (for vectors: eigenvalues of $I-\\alpha H$) shrink the error on its way back; larger ones make it grow geometrically with N. The figure shows every piece of this for a 1-D energy with three weights.</p>`,
  steps: [
    { label: 'One step, one weight', html: '<p>N = 1. The weight $m$ sets where the valley sits. Nudge it and the landscape shifts (dashed ghost), the slope under $\\hat y_0$ changes, and $\\hat y_1$ lands elsewhere (ring). Here $\\partial_m\\nabla_{\\hat y}E=-a$, so $\\partial\\hat y_1/\\partial m=\\alpha a$: the blue arrow on the ŷ axis is that first-order prediction. Drag on the loss curve: the blue tangent is $dL/dm$ from the formula, not from finite differences.</p>' },
    { label: 'Chain through N steps', html: '<p>N = 4. A nudge to $m$ now also changes where every later step starts. The chain table shows, per step $k\\to k{+}1$, the factor $1-\\alpha H(\\hat y_k)$, the error $a_{k+1}$ arriving at $\\hat y_{k+1}$, and that step\'s term of $dL/dm$. A factor inside ±1 shrinks the error on its way back; a negative one means the step overshot the minimum; red means it grows. The terms sum to the slope (Σ). The first factor is gray: it only carries the error back to the noise $\\hat y_0$, which has no weights upstream.</p><p>Try the other weights: only the mixed term changes. For the curvature, $\\partial_a\\nabla E=\\hat y-m$, which fades as the path nears the bottom of the bowl; for the bumps, $\\partial_c\\nabla E=-3\\sin3\\hat y$.</p>' },
    { label: 'Check it numerically', html: '<p>Two ink points on the loss curve at $\\theta\\pm h$ and the dashed secant through them. Drag $h$ down: the secant turns into the tangent. The readout repeats the check with $h=10^{-6}$. Across all three weights, N ≤ 6 and α ≤ 1.3 the analytic and numerical slopes agree to about $10^{-10}$ (relative) in the typical case; only explosive settings (N = 6, large bumps) lose digits, down to about $10^{-3}$, because the loss changes violently within $h$. Finite differences cost two unrolls per weight: fine for one weight, hopeless for the paper\'s 6M to 708M non-embedding parameters (Table D.1). Backpropagation gets all of them at once.</p>' },
    { label: 'Cut the chain: detach', html: '<p>Window K = 1 detaches $\\hat y$ before every step, as the paper\'s S1 models do (p.30). The $1-\\alpha H$ factors drop out (shaded) and only the last step\'s term survives (gray tangent). That term is still a mixed second derivative, so S1 training is second order too; only the Hessian chain is gone. With the loss on the last step only, this can point the wrong way: here −1.23 against the exact +0.54.</p><p>Switch the loss to <b>[ every step (S1) ]</b>. The objective changes, so the exact slope changes too (to −1.25), but now each step carries its own error and the detached gradient lands within 1% of it (−1.27). Drag $m$ across its range: with a loss at every step the detached slope keeps the right sign almost everywhere and stays close in size; with the loss at the end only it is often far off and sometimes points the wrong way. This fits the paper\'s recipes: S1 pairs detaching with a loss at every step, and "if one is calculating the loss only at the last step, then one should not detach between steps" (p.43).</p>' },
    { label: 'Stop the gradient of the gradient', html: '<p>K = 0 treats $\\nabla_{\\hat y}E$ as a constant (in PyTorch: take it without <code>create_graph=True</code>). Then $\\hat y_N$ no longer depends on θ and $dL/d\\theta=0$ exactly: a flat tangent. Compare first-order MAML, where dropping second derivatives still leaves a usable gradient because the loss touches the adapted weights directly. An EBT has no such shortcut: its weights touch the loss only through a gradient [our comparison].</p>' },
    { label: 'When the chain explodes', html: '<p>α = 1.3, N = 5. Wherever $\\alpha H>2$ (overshoot) or $H<0$ (the crest of a bump) near the path, factors leave [−1, 1] and their product grows geometrically with N. The loss curve turns jagged and the exact gradient becomes huge (−35.8), though it still matches finite differences. It is exact but only very locally true: for the nudge δ = 0.3 the first-order arrow on the ŷ axis predicts $\\hat y_5\\approx9$, far off the plot, while the ring shows it really moves from 0.02 to 0.72. A weight update sized by such a slope lands somewhere the slope says nothing about. The paper reports this failure mode: more steps mean "longer gradient chains" and less stability, so training stayed at 2 or 3 steps with a carefully tuned α (p.26, p.42).</p>' },
    { label: 'Hessian-vector products', html: '<p>The top row of the chain now shows each Hessian-vector product $H(\\hat y_k)\\,a_{k+1}$, the piece that turns $a_{k+1}$ into $a_k=a_{k+1}-\\alpha\\,Ha_{k+1}$. In 1-D it is one product of two numbers. In the paper $\\hat y$ has 50,277 entries per text position, so $H$ would hold about $2.5\\times10^9$ numbers per position [derived]. It is never built: autograd differentiates the scalar $\\nabla_{\\hat y}E^{\\top}a$ once more, which costs about one extra backward pass and, in the paper\'s words, "scale[s] linearly with model size" (p.7). The same backward pass also yields the mixed term for θ. The paper cites a 2024 ICLR blog post on computing HVPs for this [76]; the trick itself is Pearlmutter\'s (1994), which the paper does not cite. The bill per step is below.</p>' },
  ],
  after: `
    <h3>In code</h3>
    <pre class="so-code"><code>y = randn_like(target).requires_grad_()  # ŷ0
for _ in range(N):
    # y = y.detach().requires_grad_()  (S1)
    E = model(x, y).sum()
    g = grad(E, y, create_graph=True)[0]
    y = y - alpha * g      # Eq. 1
loss = loss_fn(y, target)  # end or every step
loss.backward()            # θ only via each g</code></pre>
    <p>This is the skeleton of Listing 1 (p.44), including its optional detach line (<code>grad</code> is <code>torch.autograd.grad</code>). <code>create_graph=True</code> records how <code>g</code> was computed, so <code>loss.backward()</code> can differentiate it again. At step $k$ the backward pass arrives with $a=\\partial L/\\partial\\hat y_{k+1}$ and differentiates the scalar $g^{\\top}a$ once more: with respect to $\\hat y_k$ that gives the Hessian-vector product $Ha$, with respect to θ the mixed term, both from one extra backward pass, whatever the size of $H$.</p>
    <h3>Cost per training step, in the paper's accounting</h3>
    <div class="so-cost" role="img" aria-label="Bar chart: training FLOPs per token per step. Transformer++ 6P; EBT one step 10P (1.66x); autoregressive EBT one step 20P (3.33x); S1 with two steps 40P (6.66x).">
      <div class="so-r"><span class="so-l">Transformer++</span><span class="so-b"><i class="f" style="width:5%"></i><i class="g" style="width:10%"></i></span><span class="so-x">6P · 1×</span></div>
      <div class="so-r"><span class="so-l">EBT, 1 step</span><span class="so-b"><i class="f" style="width:5%"></i><i class="g" style="width:10%"></i><i class="hv" style="width:10%"></i></span><span class="so-x">10P · 1.66×</span></div>
      <div class="so-r"><span class="so-l">AR EBT, 1 step</span><span class="so-b"><i class="f" style="width:5%"></i><i class="g" style="width:10%"></i><i class="hv" style="width:10%"></i><i class="f" style="width:5%"></i><i class="g" style="width:10%"></i><i class="hv" style="width:10%"></i></span><span class="so-x">20P · 3.33×</span></div>
      <div class="so-r"><span class="so-l">AR EBT, S1 ×2</span><span class="so-b"><i class="f" style="width:5%"></i><i class="g" style="width:10%"></i><i class="hv" style="width:10%"></i><i class="f" style="width:5%"></i><i class="g" style="width:10%"></i><i class="hv" style="width:10%"></i><i class="f" style="width:5%"></i><i class="g" style="width:10%"></i><i class="hv" style="width:10%"></i><i class="f" style="width:5%"></i><i class="g" style="width:10%"></i><i class="hv" style="width:10%"></i></span><span class="so-x">40P · 6.66×</span></div>
      <div class="so-k"><span><i class="f"></i>F forward, 2P</span><span><i class="g"></i>B backward, 4P</span><span><i class="hv"></i>B through the gradient (HVP), 4P</span></div>
    </div>
    <p>FLOPs per token for $P$ non-embedding parameters (the paper writes N): forward ≈ 2P, backward ≈ 4P (p.35). An EBT step is F + B + B = 10P against 6P for a Transformer++ step, about 1.66× (p.41). The autoregressive EBT also doubles the sequence (observed plus predicted tokens, 2S − 2), so each optimization step costs ≈3.33×, and S1 pretraining with two steps 6.66× (p.36). The authors call these estimates approximate (p.36). Fig. 5b plots training FLOPs, so its EBT points sit about 6 to 7× to the right of the matching Transformer++ points [approx., read from Fig. 5b].</p>
    <p class="note">Toy: $E_\\theta(\\hat y)=\\tfrac a2(\\hat y-m)^2+c\\cos3\\hat y$, $J=\\tfrac12(\\hat y_N-y)^2$ or its mean over steps. Analytic derivatives: $\\nabla E=a(\\hat y-m)-3c\\sin3\\hat y$, $H=a-9c\\cos3\\hat y$, $\\partial_m\\nabla E=-a$, $\\partial_a\\nabla E=\\hat y-m$, $\\partial_c\\nabla E=-3\\sin3\\hat y$. The window K is a generic truncation; the paper does not specify its S2 truncation scheme (p.43).</p>`,
  source: [{ kind: 'paper', note: 'Sec. 3.2 p.7, Listing 1 p.44, FLOPs p.35–36, p.41' }, { kind: 'concept', note: '1-D energy, analytic second derivatives' }],
  figure(stage, ctx) {
    const { lib } = ctx, h = lib.h, C = lib.C;
    const BLUE = C.blue || '#2f3cff', BLUE2 = C.blue2 || '#8a93ff', BLUE3 = C.blue3 || '#c9cdff', BLUE4 = C.blue4 || '#eef0ff';
    const INK = C.ink, MUTED = C.muted, FAINT = C.faint, RULE = C.rule, WARN = C.bad || '#d4421c';
    const SUBS = '₀₁₂₃₄₅₆₇₈₉', sub = (n) => String(n).split('').map((d) => SUBS[+d] || d).join('');
    const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
    const fv = (v, d = 2) => (!isFinite(v) ? '–' : (Math.abs(v) >= 1000 ? v.toExponential(1) : v.toFixed(d)).replace('-', '−'));
    const f3 = (v) => { if (!isFinite(v)) return '–'; const a = Math.abs(v); const s = a >= 1000 ? v.toExponential(1) : a >= 100 ? v.toFixed(0) : a >= 10 ? v.toFixed(1) : v.toFixed(2); return s.replace('-', '−'); };
    const T = (g, s, x, y, o = {}) => lib.text(g, s, x, y, Object.assign({ size: 12, kind: 'mono', color: INK, baseline: 'middle' }, o));

    // ---------------- model and exact derivatives ----------------
    const OM = 3, YR = [-3.5, 3.5], ER = [-1, 6];
    const W = { m: { name: 'm', what: 'valley position', range: [-2.5, 3], d: 0.3 }, a: { name: 'a', what: 'curvature', range: [0.1, 3], d: 0.25 }, c: { name: 'c', what: 'bump height', range: [-0.8, 0.8], d: 0.15 } };
    const P0 = { m: -0.4, a: 1.2, c: 0.3 };
    const En = (P, y) => 0.5 * P.a * (y - P.m) ** 2 + P.c * Math.cos(OM * y);
    const gE = (P, y) => P.a * (y - P.m) - P.c * OM * Math.sin(OM * y);
    const hE = (P, y) => P.a - P.c * OM * OM * Math.cos(OM * y);
    const mix = (P, y, k) => (k === 'm' ? -P.a : k === 'a' ? y - P.m : -OM * Math.sin(OM * y));
    const S = { P: Object.assign({}, P0), which: 'm', N: 1, alpha: 0.5, K: 1, every: false, y0: -2.4, tgt: 1.2, showFD: false, hRel: 0.07, hvp: false };
    function fwd(P, N, al) { const p = [S.y0]; let y = S.y0; for (let i = 0; i < N; i++) { y -= al * gE(P, y); p.push(y); } return p; }
    function lossOf(P) { const N = S.N, p = fwd(P, N, S.alpha); if (!S.every) return 0.5 * (p[N] - S.tgt) ** 2; let s = 0; for (let j = 1; j <= N; j++) s += 0.5 * (p[j] - S.tgt) ** 2; return s / N; }
    function analyze(P) {
      const N = S.N, al = S.alpha, k = S.which, p = fwd(P, N, al), w = S.every ? 1 / N : 1;
      const direct = (j) => (S.every ? (j >= 1 ? w * (p[j] - S.tgt) : 0) : (j === N ? p[N] - S.tgt : 0));
      const fac = [], mx = []; for (let i = 0; i < N; i++) { fac.push(1 - al * hE(P, p[i])); mx.push(mix(P, p[i], k)); }
      // exact adjoint
      const adj = new Array(N + 1).fill(0); adj[N] = direct(N);
      for (let i = N - 1; i >= 0; i--) adj[i] = direct(i) + fac[i] * adj[i + 1];
      const termEx = mx.map((m, i) => -al * adj[i + 1] * m), Gex = termEx.reduce((s, v) => s + v, 0);
      const hv = fac.map((f, i) => hE(P, p[i]) * adj[i + 1]); // Hessian-vector products H(ŷ_k)·a_{k+1}: a_k = direct_k + a_{k+1} − α·hv_k
      // truncated: each loss term backpropagates through at most K updates
      const termAp = new Array(N).fill(0), js = S.every ? Array.from({ length: N }, (_, j) => j + 1) : [N];
      js.forEach((j) => { let b = direct(j); for (let i = j - 1; i >= Math.max(0, j - S.K); i--) { termAp[i] += -al * b * mx[i]; b *= fac[i]; } });
      const Gap = termAp.reduce((s, v) => s + v, 0);
      // forward-mode tangent dŷ_i/dθ (independent cross-check of the adjoint result)
      const tan = [0]; for (let i = 0; i < N; i++) tan.push(fac[i] * tan[i] - al * mx[i]);
      let Gtan = 0; for (let j = 0; j <= N; j++) Gtan += direct(j) * tan[j];
      // central finite difference, h = 1e-6 (relative): an independent, naive check of the analytic result
      const th = P[k], hh = 1e-6 * Math.max(1, Math.abs(th)), Lat = (d) => { const Q = Object.assign({}, P); Q[k] = th + d; return lossOf(Q); };
      const fd = (Lat(hh) - Lat(-hh)) / (2 * hh);
      return { p, L: lossOf(P), fac, mx, adj, hv, termEx, termAp, Gex, Gap, tan, Gtan, fd, rel: Math.abs(Gex - fd) / Math.max(1e-12, Math.abs(Gex) + Math.abs(fd)), relTan: Math.abs(Gex - Gtan) / Math.max(1e-300, Math.abs(Gex) + Math.abs(Gtan)) };
    }

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
      Rc.toLocal = (ev) => { const r = cv.getBoundingClientRect(), q = ev.touches ? ev.touches[0] : ev; return [(q.clientX - r.left) * Rc.w / r.width, (q.clientY - r.top) * Rc.h / r.height]; };
      if (window.ResizeObserver) { let pw = 0; new ResizeObserver(() => { const w = Math.floor(box.clientWidth); if (w && w !== pw) { pw = w; Rc.render(); fitSticky(); } }).observe(box); }
      return Rc;
    }

    // ---------------- layout ----------------
    const F1 = lib.frame(stage, { label: 'Energy and the unrolled path', sub: 'E<sub>θ</sub>(ŷ) solid · dashed: the same energy with the chosen weight nudged by +δ' });
    const eC = rcv(F1.frame, { label: 'Energy curve over the prediction, the unrolled gradient-descent path and the target, plus a dashed ghost of the energy and path after nudging the chosen weight.', height: (w) => (w >= 540 ? 184 : 200), draw: drawEnergy });
    const row = h('div', { class: 'fig-row' }); stage.appendChild(row);
    const F2 = lib.frame(row, { label: 'Loss along one weight', sub: 'L(θ) · drag to move θ' }); F2.wrap.style.flex = '1 1 270px';
    const lC = rcv(F2.frame, { label: 'Training loss as a function of the chosen weight, with the exact tangent, a finite-difference secant and the truncated-gradient tangent.', height: () => 214, draw: drawLoss });
    const F3 = lib.frame(row, { label: 'Backward chain', sub: 'per step k: ŷ<sub>k</sub> → ŷ<sub>k+1</sub>' }); F3.wrap.style.flex = '1 1 300px';
    const cC = rcv(F3.frame, { label: 'Per-step factors one minus alpha times the Hessian, the error arriving at each step, and each step\'s term of the gradient, exact and truncated.', height: () => 214, draw: drawChain });

    // ---------------- drawing: energy ----------------
    function drawEnergy(g, w, hh) {
      const b = { x: 40, y: 10, w: w - 52, h: hh - 40 }, X = (y) => b.x + (y - YR[0]) / (YR[1] - YR[0]) * b.w, Y = (e) => b.y + b.h - (e - ER[0]) / (ER[1] - ER[0]) * b.h;
      const R = cur, P = S.P, k = S.which, d = W[k].d, Pg = Object.assign({}, P); Pg[k] = P[k] + d;
      g.save(); g.strokeStyle = RULE; g.lineWidth = 1;
      [0, 2, 4, 6].forEach((e) => { g.beginPath(); g.moveTo(b.x, Y(e) + 0.5); g.lineTo(b.x + b.w, Y(e) + 0.5); g.stroke(); T(g, String(e), b.x - 6, Y(e), { size: 10.5, align: 'right', color: MUTED }); });
      g.strokeStyle = FAINT; g.beginPath(); g.moveTo(b.x + 0.5, b.y); g.lineTo(b.x + 0.5, b.y + b.h + 0.5); g.lineTo(b.x + b.w, b.y + b.h + 0.5); g.stroke(); g.restore();
      [-3, -2, -1, 0, 1, 2, 3].forEach((v) => T(g, fv(v, 0), X(v), b.y + b.h + 10, { size: 10.5, align: 'center', color: MUTED }));
      T(g, 'ŷ', b.x + b.w, b.y + b.h + 24, { size: 12, align: 'right' }); T(g, 'E', 12, b.y + 6, { size: 12 });
      g.save(); g.beginPath(); g.rect(b.x, b.y, b.w, b.h); g.clip();
      // target
      const xt = X(S.tgt); g.save(); g.strokeStyle = INK; g.setLineDash([3, 3]); g.lineWidth = 1; g.beginPath(); g.moveTo(xt, b.y); g.lineTo(xt, b.y + b.h); g.stroke(); g.restore();
      // curves
      const cur2 = (PP) => { const pts = []; for (let i = 0; i <= 280; i++) { const y = YR[0] + (YR[1] - YR[0]) * i / 280; pts.push([X(y), Y(clamp(En(PP, y), ER[0] - 1, ER[1] + 1))]); } return pts; };
      lib.line(g, cur2(Pg), { color: FAINT, width: 1.2, dash: [4, 4] });
      lib.line(g, cur2(P), { color: BLUE, width: 1.8 });
      // ghost path
      const pg = fwd(Pg, S.N, S.alpha);
      pg.forEach((y, i) => { if (i) lib.dot(g, X(y), Y(En(Pg, y)), 3.2, '#fff', { stroke: FAINT, lw: 1.2 }); });
      // path
      const pp = R.p.map((y) => [X(clamp(y, YR[0] - 1, YR[1] + 1)), Y(clamp(En(P, y), ER[0], ER[1] + 0.5))]);
      for (let i = 1; i < pp.length; i++) lib.arrow(g, pp[i - 1][0], pp[i - 1][1], pp[i][0], pp[i][1], { color: INK, width: 1.1, head: 5 });
      pp.forEach((q, i) => lib.dot(g, q[0], q[1], i === 0 ? 4.2 : 3.8, i === 0 ? '#fff' : BLUE, { stroke: i === 0 ? INK : '#fff', lw: 1.3 }));
      g.restore();
      // where ŷ_N goes when θ moves: linearized (blue arrow on the axis) vs actual (ghost)
      const N = S.N, yN = R.p[N], yL = yN + d * R.tan[N], ax = b.y + b.h - 7;
      if (Math.abs(X(yL) - X(yN)) > 3) lib.arrow(g, X(clamp(yN, YR[0], YR[1])), ax, X(clamp(yL, YR[0], YR[1])), ax, { color: BLUE, width: 1.6, head: 6 });
      // first-order prediction off the plot: say where it points
      if (yL > YR[1] || yL < YR[0]) {
        const s = 'first order: ŷ' + sub(N) + ' ≈ ' + f3(yL), right = yL > YR[1], tw = lib.measure(g, s, { size: 10.5, kind: 'mono', weight: 600 }).w, tx = right ? b.x + b.w - 4 : b.x + 4;
        g.save(); g.fillStyle = 'rgba(255,255,255,0.92)'; g.fillRect(right ? tx - tw - 3 : tx - 3, ax - 18, tw + 6, 15); g.restore();
        T(g, s, tx, ax - 10.5, { size: 10.5, align: right ? 'right' : 'left', color: BLUE, weight: 600 });
      }
      lib.dot(g, X(clamp(pg[N], YR[0], YR[1])), ax, 3, '#fff', { stroke: INK, lw: 1.1 });
      T(g, 'y', xt + 5, b.y + 8, { size: 12 });
      T(g, 'ŷ' + sub(0), clamp(pp[0][0], b.x + 6, b.x + b.w - 20) + 7, pp[0][1] - 10, { size: 11.5, color: MUTED });
      const qN = pp[N]; T(g, 'ŷ' + sub(N), clamp(qN[0] + 8, b.x + 4, b.x + b.w - 24), clamp(qN[1] - 12, b.y + 8, b.y + b.h - 8), { size: 11.5, color: BLUE, weight: 700 });
    }

    // ---------------- drawing: loss along the chosen weight ----------------
    let lossBox = null, curveCache = { key: '', pts: null, hi: 1 };
    function lossCurve() {
      const k = S.which, rg = W[k].range, others = ['m', 'a', 'c'].filter((q) => q !== k).map((q) => S.P[q]);
      const kk = [k, S.N, S.alpha, S.every, S.y0, S.tgt].concat(others).join('|'); if (curveCache.key === kk) return curveCache;
      const P = Object.assign({}, S.P), pts = []; for (let i = 0; i <= 360; i++) { const v = rg[0] + (rg[1] - rg[0]) * i / 360; P[k] = v; pts.push([v, lossOf(P)]); }
      const ys = pts.map((q) => q[1]).filter(isFinite).sort((a, b) => a - b); const hi = Math.max(0.5, ys[Math.floor(0.97 * (ys.length - 1))] * 1.08);
      curveCache = { key: kk, pts, hi }; return curveCache;
    }
    function drawLoss(g, w, hh) {
      const k = S.which, rg = W[k].range, R = cur, cc = lossCurve();
      const b = { x: 42, y: 12, w: w - 54, h: hh - 44 }; lossBox = b;
      const X = (v) => b.x + (v - rg[0]) / (rg[1] - rg[0]) * b.w, Y = (v) => b.y + b.h - clamp(v / cc.hi, -0.05, 1.05) * b.h;
      g.save(); g.strokeStyle = RULE; g.lineWidth = 1;
      const tick = cc.hi > 20 ? Math.pow(10, Math.floor(Math.log10(cc.hi))) : cc.hi > 4 ? 2 : cc.hi > 1.5 ? 0.5 : 0.25;
      const tf = (v) => (tick >= 1 ? v.toFixed(0) : tick >= 0.5 ? v.toFixed(1) : v.toFixed(2));
      for (let v = 0; v <= cc.hi + 1e-9; v += tick) { g.beginPath(); g.moveTo(b.x, Y(v) + 0.5); g.lineTo(b.x + b.w, Y(v) + 0.5); g.stroke(); T(g, tf(v), b.x - 6, Y(v), { size: 10.5, align: 'right', color: MUTED }); }
      g.strokeStyle = FAINT; g.beginPath(); g.moveTo(b.x + 0.5, b.y); g.lineTo(b.x + 0.5, b.y + b.h + 0.5); g.lineTo(b.x + b.w, b.y + b.h + 0.5); g.stroke(); g.restore();
      [rg[0], (rg[0] + rg[1]) / 2, rg[1]].forEach((v) => T(g, fv(v, 1), X(v), b.y + b.h + 10, { size: 10.5, align: 'center', color: MUTED }));
      T(g, 'weight ' + k + ' (' + W[k].what + ')', b.x + b.w / 2, b.y + b.h + 24, { size: 11, align: 'center', color: MUTED });
      g.save(); g.beginPath(); g.rect(b.x, b.y - 2, b.w, b.h + 4); g.clip();
      lib.line(g, cc.pts.map(([v, L]) => [X(v), Y(L)]), { color: INK, width: 1.3 });
      const th = S.P[k], L = R.L, span = (rg[1] - rg[0]) * 0.16;
      const tline = (s, col, wdt, dash) => lib.line(g, [[X(th - span), Y(L - s * span)], [X(th + span), Y(L + s * span)]], { color: col, width: wdt, dash });
      g.save(); g.strokeStyle = 'rgba(17,17,17,0.35)'; g.setLineDash([2, 3]); g.beginPath(); g.moveTo(X(th), b.y); g.lineTo(X(th), b.y + b.h); g.stroke(); g.restore();
      const approx = S.K < S.N;
      if (approx) tline(R.Gap, MUTED, 1.6, [5, 4]);
      tline(R.Gex, BLUE, 2.2);
      if (S.showFD) {
        const hv = (rg[1] - rg[0]) * S.hRel, Pp = Object.assign({}, S.P), Pm = Object.assign({}, S.P); Pp[k] = th + hv; Pm[k] = th - hv;
        const Lp = lossOf(Pp), Lm = lossOf(Pm), s = (Lp - Lm) / (2 * hv);
        lib.line(g, [[X(th - span), Y(L - s * span)], [X(th + span), Y(L + s * span)]], { color: INK, width: 1.2, dash: [3, 3] });
        lib.dot(g, X(th - hv), Y(Lm), 3.5, '#fff', { stroke: INK, lw: 1.4 }); lib.dot(g, X(th + hv), Y(Lp), 3.5, '#fff', { stroke: INK, lw: 1.4 });
      }
      lib.dot(g, X(th), Y(L), 4.8, BLUE, { stroke: '#fff', lw: 1.5 });
      g.restore();
      // legend
      let ly = b.y + 6; const lx = b.x + b.w - 4;
      T(g, 'exact ' + f3(R.Gex), lx, ly, { size: 11, align: 'right', color: BLUE, weight: 700 }); ly += 15;
      if (approx) { T(g, 'K = ' + S.K + ': ' + f3(R.Gap), lx, ly, { size: 11, align: 'right', color: MUTED, weight: 600 }); ly += 15; }
      if (S.showFD) T(g, 'secant, h = ' + hTxt(), lx, ly, { size: 11, align: 'right', color: INK });
    }

    // ---------------- drawing: backward chain ----------------
    function drawChain(g, w, hh) {
      const R = cur, N = S.N, lw = 62, cols = N + 1, cw = (w - lw - 6) / cols, rows = [{ y: 22, h: 46 }, { y: 84, h: 46 }, { y: 146, h: 46 }];
      for (let i = 0; i < N; i++) {
        const x = lw + i * cw + cw / 2;
        T(g, i + '→' + (i + 1), x, 9, { size: 11, align: 'center', color: MUTED });
      }
      T(g, 'Σ', lw + N * cw + cw / 2, 10, { size: 12, align: 'center', weight: 700 });
      const label = (r, a, b2) => { T(g, a, 4, r.y + r.h / 2 - 7, { size: 11.5, weight: 600 }); T(g, b2, 4, r.y + r.h / 2 + 8, { size: 10, color: MUTED }); };
      label(rows[0], S.hvp ? 'H·aₖ₊₁' : '1 − αH', S.hvp ? 'HVP' : 'factor');
      label(rows[1], 'aₖ₊₁', 'error in');
      label(rows[2], 'term', 'of dL/d' + S.which);
      // shared bar drawer: signed bar from the row's zero line
      const bar = (r, x, v, vmax, o = {}) => {
        const zy = r.y + r.h / 2, bh = clamp(v / vmax, -1, 1) * (r.h / 2 - 4), bw = Math.min(26, cw * 0.56);
        g.save(); if (o.fill) { g.fillStyle = o.fill; g.fillRect(x - bw / 2, Math.min(zy, zy - bh), bw, Math.max(1.2, Math.abs(bh))); }
        if (o.stroke) { g.strokeStyle = o.stroke; g.lineWidth = 1.2; g.strokeRect(x - bw / 2 + 0.5, Math.min(zy, zy - bh) + 0.5, bw - 1, Math.max(1, Math.abs(bh) - 1)); }
        g.restore();
      };
      rows.forEach((r) => { g.save(); g.strokeStyle = RULE; g.beginPath(); g.moveTo(lw - 4, r.y + r.h / 2 + 0.5); g.lineTo(w - 4, r.y + r.h / 2 + 0.5); g.stroke(); g.restore(); });
      // row 0: factors, scale ±3 with ±1 guides
      const r0 = rows[0], fmax = 3;
      g.save(); g.strokeStyle = 'rgba(17,17,17,0.35)'; g.setLineDash([2, 3]);
      if (!S.hvp) [1, -1].forEach((v) => { const y = r0.y + r0.h / 2 - v / fmax * (r0.h / 2 - 4); g.beginPath(); g.moveTo(lw - 4, y); g.lineTo(lw + N * cw, y); g.stroke(); });
      g.restore();
      // window shading: steps whose terms are kept by truncation (loss at the end: the last K steps)
      if (S.K < N && !S.every && N - S.K > 0) {
        const kept0 = N - S.K;
        g.save(); g.fillStyle = 'rgba(17,17,17,0.05)'; g.fillRect(lw, rows[0].y - 4, kept0 * cw, rows[2].y + rows[2].h - rows[0].y + 16); g.restore();
        T(g, 'cut by K', lw + kept0 * cw / 2, rows[0].y + 2, { size: 10, align: 'center', color: MUTED });
      }
      const amax = Math.max(1e-9, ...R.adj.slice(1).map(Math.abs)), hvmax = Math.max(1e-9, ...R.hv.slice(1).map(Math.abs));
      const tmax = Math.max(1e-9, ...R.termEx.map(Math.abs), ...R.termAp.map(Math.abs), Math.abs(R.Gex), Math.abs(R.Gap));
      for (let i = 0; i < N; i++) {
        // the factor of step 0 only maps the error to ŷ0, which has no weights upstream: never used for dL/dθ
        const x = lw + i * cw + cw / 2, f = R.fac[i], hot = Math.abs(f) > 1, unused = i === 0;
        if (S.hvp) { bar(r0, x, R.hv[i], hvmax, { fill: unused ? RULE : BLUE }); T(g, f3(R.hv[i]), x, r0.y + r0.h + 6, { size: 10, align: 'center', color: unused ? FAINT : BLUE }); }
        else { bar(r0, x, f, fmax, { fill: unused ? RULE : hot ? WARN : INK }); T(g, f3(f), x, r0.y + r0.h + 6, { size: 10, align: 'center', color: unused ? FAINT : hot ? WARN : INK }); }
        bar(rows[1], x, R.adj[i + 1], amax, { fill: INK });
        T(g, f3(R.adj[i + 1]), x, rows[1].y + rows[1].h + 6, { size: 10, align: 'center' });
        bar(rows[2], x, R.termEx[i], tmax, { stroke: BLUE });
        if (S.K < N) bar(rows[2], x, R.termAp[i], tmax, { fill: 'rgba(107,107,112,0.55)' });
        else bar(rows[2], x, R.termEx[i], tmax, { fill: BLUE });
        T(g, f3(R.termEx[i]), x, rows[2].y + rows[2].h + 6, { size: 10, align: 'center', color: BLUE });
      }
      const xs = lw + N * cw + cw / 2;
      bar(rows[2], xs, R.Gex, tmax, { fill: BLUE });
      if (S.K < N) bar(rows[2], xs + Math.min(10, cw * 0.18), R.Gap, tmax, { fill: 'rgba(107,107,112,0.7)' });
      T(g, f3(R.Gex), xs, rows[2].y + rows[2].h + 6, { size: 10, align: 'center', color: BLUE, weight: 700 });
      if (S.K < N) T(g, f3(R.Gap), xs, rows[2].y - 4, { size: 10, align: 'center', color: MUTED, weight: 700 });
      g.save(); g.strokeStyle = RULE; g.beginPath(); g.moveTo(lw + N * cw + 0.5, 20); g.lineTo(lw + N * cw + 0.5, hh - 6); g.stroke(); g.restore();
    }

    // ---------------- controls ----------------
    const ctl1 = h('div', { class: 'controls' }); stage.appendChild(ctl1);
    ctl1.appendChild(h('span', { class: 'fig-label' }, 'weight'));
    const segW = lib.segmented({ label: 'Weight to differentiate', options: [['m', 'm · position'], ['a', 'a · curvature'], ['c', 'c · bumps']], value: S.which, onchange: (v) => { S.which = v; syncTheta(); update(); } });
    ctl1.appendChild(segW.el);
    const ctl1b = h('div', { class: 'controls' }); stage.appendChild(ctl1b);
    ctl1b.appendChild(h('span', { class: 'fig-label' }, 'loss on'));
    const segL = lib.segmented({ label: 'Where the loss is applied', options: [[false, 'last step (S2)'], [true, 'every step (S1)']], value: S.every, onchange: (v) => { S.every = v; update(); } });
    ctl1b.appendChild(segL.el);
    const ctl2 = h('div', { class: 'controls' }); stage.appendChild(ctl2);
    const slT = lib.slider({ id: 'so-theta', label: 'weight value', min: -2.5, max: 3, step: 0.01, value: S.P.m, fmt: (v) => v.toFixed(2), oninput: (v) => { S.P[S.which] = v; update(); } });
    const slN = lib.slider({ id: 'so-n', label: 'unrolled steps N', min: 1, max: 6, step: 1, value: S.N, oninput: (v) => { const full = S.K >= S.N; S.N = v; if (full || S.K > v) S.K = v; syncK(); update(); } });
    const slA = lib.slider({ id: 'so-alpha', label: 'step size α', min: 0.1, max: 1.5, step: 0.05, value: S.alpha, fmt: (v) => v.toFixed(2), oninput: (v) => { S.alpha = v; update(); } });
    const kName = (v) => (v >= S.N ? v + ' · exact' : v === 0 ? '0 · stop-grad' : v === 1 ? '1 · detached' : v + ' · truncated');
    const slK = lib.slider({ id: 'so-k', label: 'backprop window K', min: 0, max: 6, step: 1, value: S.K, fmt: kName, oninput: (v) => { S.K = Math.min(v, S.N); if (v > S.N) slK.set(S.N); update(); } });
    [slT, slN, slA, slK].forEach((s) => ctl2.appendChild(s.el));
    // finite-difference secant: toggle and half-width h (log scale, as a fraction of the weight's range)
    const hTxt = () => { const v = (W[S.which].range[1] - W[S.which].range[0]) * S.hRel; return v >= 0.1 ? v.toFixed(2) : v.toPrecision(2); };
    const ctl3 = h('div', { class: 'controls' }); stage.appendChild(ctl3);
    ctl3.appendChild(h('span', { class: 'fig-label' }, 'check'));
    const bFD = lib.button('finite-difference secant', () => { S.showFD = !S.showFD; syncFD(); update(); });
    ctl3.appendChild(bFD);
    const slH = lib.slider({ id: 'so-h', label: 'secant half-width h', min: -3, max: -1, step: 0.05, value: Math.log10(S.hRel), fmt: () => hTxt(), oninput: (v) => { S.hRel = Math.pow(10, v); slH.set(v); update(); } });
    ctl3.appendChild(slH.el);
    function syncFD() { bFD.setAttribute('aria-pressed', String(S.showFD)); slH.el.style.display = S.showFD ? '' : 'none'; slH.set(Math.log10(S.hRel)); }
    const ro = h('div', { class: 'readout', 'aria-live': 'polite' }); stage.appendChild(ro);
    function syncTheta() { const rg = W[S.which].range; slT.input.min = rg[0]; slT.input.max = rg[1]; slT.set(S.P[S.which]); slH.set(Math.log10(S.hRel)); }
    function syncK() { slK.input.max = S.N; slK.set(S.K); }

    // drag on the loss plot to move the weight
    let dragging = false;
    const setFromEv = (ev) => { if (!lossBox) return; const [px] = lC.toLocal(ev), rg = W[S.which].range; S.P[S.which] = clamp(rg[0] + (px - lossBox.x) / lossBox.w * (rg[1] - rg[0]), rg[0], rg[1]); slT.set(S.P[S.which]); update(); };
    lC.cv.addEventListener('pointerdown', (ev) => { dragging = true; try { lC.cv.setPointerCapture(ev.pointerId); } catch (e) { /* ignore */ } setFromEv(ev); });
    lC.cv.addEventListener('pointermove', (ev) => { if (dragging) setFromEv(ev); });
    ['pointerup', 'pointercancel'].forEach((t) => lC.cv.addEventListener(t, () => { dragging = false; }));
    lC.cv.style.cursor = 'ew-resize';

    let cur = null;
    function readout() {
      const R = cur, k = S.which;
      let s = `<span>L <b>${fv(R.L, 3)}</b></span><span>dL/d${k} exact <b>${f3(R.Gex)}</b></span><span>finite diff. (h = 10⁻⁶) <b>${f3(R.fd)}</b> · rel. error <b>${R.rel.toExponential(0).replace('e-', 'e−')}</b></span>`;
      if (S.K < S.N) s += `<span>window K = ${S.K} <b>${f3(R.Gap)}</b>${Math.abs(R.Gex) > 1e-9 && Math.abs(R.Gap) > 1e-12 ? (R.Gap * R.Gex < 0 ? ' (opposite sign)' : ' (' + Math.round(R.Gap / R.Gex * 100) + '% of exact)') : ''}</span>`;
      if (S.hvp) s += `<span>backward through ∇E: <b>${S.N}</b> extra passes, each giving H·a and the mixed term</span>`;
      if (S.N > 1) s += `<span>chain gain ∏ |1−αH| over k ≥ 1 <b>${f3(R.fac.slice(1).reduce((q, f) => q * Math.abs(f), 1))}</b></span>`;
      ro.innerHTML = s;
    }
    function update() { cur = analyze(S.P); eC.render(); lC.render(); cC.render(); readout(); fitSticky(); }
    function fitSticky() {
      try {
        stage.style.position = ''; if (getComputedStyle(stage).position !== 'sticky') { stage.style.top = ''; return; }
        const hh = stage.offsetHeight, vh = window.innerHeight;
        if (hh > vh * 1.2) { stage.style.position = 'static'; stage.style.top = ''; return; } // far taller than the window: let it scroll
        stage.style.top = Math.min(24, vh - hh - 6) + 'px'; // a bit taller: stick with its bottom edge in view
      } catch (e) { /* ignore */ }
    }
    window.addEventListener('resize', fitSticky);
    ctx.setCaption('Hollow dot: noise start ŷ₀; blue dots: the unrolled steps; dashed line: target y. The arrow on the ŷ axis is where ŷ<sub>N</sub> moves to first order when the weight moves by δ; the ring is where it actually lands. Blue tangent: exact dL/dθ; gray: truncated backprop (K < N); ink secant: finite differences.');
    syncFD(); update();
    setTimeout(fitSticky, 300);

    function setup(o) {
      Object.assign(S, { showFD: false, hRel: 0.07, hvp: false }, o);
      if (o.P) S.P = Object.assign({}, o.P);
      segW.set(S.which); segL.set(S.every); slN.set(S.N); slA.set(S.alpha); syncK(); syncTheta(); syncFD(); update();
    }
    return {
      step(i) {
        if (i === 0) setup({ P: P0, which: 'm', N: 1, K: 1, alpha: 0.5, every: false });
        if (i === 1) setup({ P: P0, which: 'm', N: 4, K: 4, alpha: 0.5, every: false });
        if (i === 2) setup({ P: { m: 0.3, a: 1.2, c: 0.3 }, which: 'm', N: 4, K: 4, alpha: 0.5, every: false, showFD: true });
        if (i === 3) setup({ P: P0, which: 'm', N: 4, K: 1, alpha: 0.5, every: false });
        if (i === 4) setup({ P: P0, which: 'm', N: 4, K: 0, alpha: 0.5, every: false });
        if (i === 5) setup({ P: P0, which: 'm', N: 5, K: 5, alpha: 1.3, every: false });
        if (i === 6) setup({ P: P0, which: 'm', N: 3, K: 3, alpha: 0.5, every: false, hvp: true });
      },
      show() { fitSticky(); },
    };
  },
});
