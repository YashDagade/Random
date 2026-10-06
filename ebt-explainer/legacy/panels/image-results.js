/* Panel 21: images, denoising and representations vs DiT.
   Paper data: Table 4 (exact), Fig 12 (lib.data('scaling').plots fig12, digitized approx.), Fig 10 crop (media/paper/fig10.png).
   Toy data: lib.data('image') (our 32x32 CIFAR-10 denoising EBT vs same-size FF and DDIM baselines) + media/toy/denoise/*.png.
   Live computations: PSNR <-> MSE <-> RMS conversions, budget-limited best PSNR, forward-pass-equivalent cost ratios. */
(function () {
  'use strict';
  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
  const L10 = Math.log10, P10 = (v) => Math.pow(10, v);
  const MINUS = '−';
  const num = (v, sig = 3) => (v == null || !isFinite(v)) ? '–' : String(+v.toPrecision(sig)).replace(/-/g, MINUS);
  const psnrOf = (mse) => 10 * L10(255 * 255 / mse);
  const mseOf = (psnr) => 255 * 255 / P10(psnr / 10);
  function rcanvas(lib, parent, o) {
    const box = lib.h('div', { class: 'canvas-box' }); parent.appendChild(box);
    const c = lib.h('canvas', { role: 'img', 'aria-label': o.label || '' }); box.appendChild(c);
    const g = c.getContext('2d'); const R = { canvas: c, ctx: g, w: 0, h: 0, dpr: 1, box };
    R.fit = function () {
      const w = Math.max(260, Math.round(box.clientWidth || 600)), hh = Math.round(o.aspect(w));
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      if (w === R.w && hh === R.h && dpr === R.dpr) return false;
      R.w = w; R.h = hh; R.dpr = dpr; c.width = Math.round(w * dpr); c.height = Math.round(hh * dpr); c.style.height = hh + 'px';
      g.setTransform(dpr, 0, 0, dpr, 0, 0); return true;
    };
    R.fit();
    if (window.ResizeObserver) new ResizeObserver(() => { if (box.clientWidth && R.fit() && o.draw) o.draw(); }).observe(box);
    R.local = (ev) => { const r = c.getBoundingClientRect(); const p = ev.touches && ev.touches.length ? ev.touches[0] : ev; return [(p.clientX - r.left) / r.width * R.w, (p.clientY - r.top) / r.height * R.h]; };
    R.clear = () => { g.save(); g.setTransform(R.dpr, 0, 0, R.dpr, 0, 0); g.fillStyle = '#ffffff'; g.fillRect(0, 0, R.w, R.h); g.restore(); };
    return R;
  }
  // Table 4 (p.13), exact
  const T4 = {
    DiT: { s1: { psnr: 26.58, mse: 142.98 }, s2: { psnr: 19.56, mse: 718.7 }, top1: 0.31, top5: 1.36 },
    EBT: { s1: { psnr: 27.25, mse: 122.55 }, s2: { psnr: 23.29, mse: 305.2 }, top1: 5.32, top5: 13.2 },
  };

  EBT.panel({
    id: 'image-results',
    nav: 'Images vs DiT',
    title: 'Images: denoising and representations vs DiT',
    lede: 'On image denoising a bidirectional EBT beats a Diffusion Transformer: by a little at the noise level both were trained on, by a lot at a noisier level neither saw, and with 3 forward passes where DiT uses 300. Its features also classify ImageNet better, at low absolute accuracy.',
    text: `
      <p>Setup (Sec 4.3, p.13; App. D.3, p.34–35): COCO 2014 at 128×128, patch size 16, both models the "large" size of Table D.1, trained 100k steps. Noise follows DiT's linear variance schedule, $\\beta_t$ from $10^{-4}$ to $2\\cdot10^{-2}$:</p>
      <div class="eq">$$\\begin{gathered}x_t=\\sqrt{\\bar\\alpha_t}\\,x_0+\\sqrt{1-\\bar\\alpha_t}\\,\\epsilon\\\\ \\bar\\alpha_t=\\textstyle\\prod_{s\\le t}(1-\\beta_s)\\end{gathered}$$<span class="why">x₀ clean image, ε ~ N(0, I), t = σ·T</span></div>
      <p>Here σ is a fraction of the schedule, not a standard deviation: σ = 0.1 for training and the in-distribution test, σ = 0.2 for the out-of-distribution (OOD) test. With the usual T = 1000 [assumed], that is a noise std of ≈0.32 and ≈0.58 on images scaled to [−1, 1] [derived]. The EBT is trained as in Alg. 1 with MSE on its optimized prediction; the bidirectional EBT is built on the DiT codebase (p.33).</p>
      <p>Quality is peak signal-to-noise ratio on 0–255 pixels,</p>
      <div class="eq">$$\\mathrm{PSNR}=10\\log_{10}\\frac{255^2}{\\mathrm{MSE}}\\ \\text{dB},$$<span class="why">a log scale: +Δ dB means MSE divided by 10^(Δ/10)</span></div>
      <p>so a 3 dB gain halves the squared error, and 10 dB divides it by ten.</p>`,
    steps: [
      { label: 'In distribution: a small gap', html: '<p>At σ = 0.1 (Table 4, p.13): EBT 27.25 dB, DiT 26.58 dB. That is +0.67 dB, or 14.3% lower MSE (122.55 vs 142.98). Recomputing each PSNR from its printed MSE reproduces the table to 0.01 dB, which is how we know the MSE is on the 0–255 scale [derived]. Drag along the ruler to convert dB, MSE and RMS error.</p>' },
      { label: 'Out of distribution: a big gap', html: '<p>At σ = 0.2: EBT 23.29 dB, DiT 19.56 dB. That is +3.73 dB, MSE 305.2 vs 718.7, 2.36× lower (the text rounds it to "as much as 3.5"). In RMS terms EBT\'s pixels are off by ≈17.5 gray levels, DiT\'s by ≈26.8. Neither model saw this noise level in training, so this is a generalization result, and it is where nearly all of the advantage is.</p>' },
      { label: 'Three passes against three hundred', html: '<p>Fig 12 (p.15) plots σ = 0.2 PSNR against forward passes. Both models did best when re-applied to their own output three times (p.35); DiT runs 100 DDIM steps per application, EBT apparently one optimization step [derived]. After one application EBT is slightly behind (≈13.9 vs ≈14.3 dB), after two level (≈18.8 vs ≈18.9), after three clearly ahead (≈23.3 vs ≈19.6). Hence "99% fewer forward passes": 3 vs 300. Slide the budget to see who wins at each cost.</p>' },
      { label: 'What one pass costs', html: '<p>An EBT step is a forward pass plus a backward pass to the input to get $\\nabla_{\\hat y}E$, but the paper counts it as one function evaluation (p.8). Set the cost of one EBT step in forward-pass equivalents. At 3×, EBT\'s 3 steps cost ≈9 forward passes: still ≈33× less than DiT\'s 300 [derived, rough]. The saving is real but smaller than 100×.</p>' },
      { label: 'What the outputs look like', html: '<p>Fig 10 (p.14): out-of-distribution denoising of two COCO images. Top row EBT after 1, 2, 3 steps; bottom row DiT after 100, 200, 300 steps. The paper\'s reading is that EBT outputs are less blurry. These are two examples; the paper does not say how they were chosen.</p>' },
      { label: 'Do the features know what is in the image?', html: '<p>A linear classifier on the mean of the final patch tokens, ImageNet-1k, 1,000 classes; DiT is queried with T = 0 (p.14, p.35). EBT 5.32% vs DiT 0.31% top-1 (17.2×), 13.2% vs 1.36% top-5 (9.7×) [derived]; the paper rounds both to "around 10×". On a 0–100% axis both bars are tiny. Chance is 0.1% top-1: DiT\'s features carry almost no class information, EBT\'s carry some. Switch to the log axis to see the ratio.</p>' },
      { label: 'Our toy: thinking on real pixels', html: '<p>A 69k-parameter convolutional EBT we trained on 5,000 CIFAR-10 images (32×32) with the paper\'s recipe: start from pure noise, 2–3 descent steps in training, MSE on the last step (toy, not the paper). Scrub the steps: the clean image is built entirely by descending the learned energy. Over the 300-image eval set PSNR goes from ≈7 dB (noise) to ≈19 dB after 2 steps and ≈21.5 dB by 8, then stays flat while the energy settles.</p>' },
      { label: 'Our toy: where the baselines win', html: '<p>At this tiny scale the baselines win. A same-size one-shot denoiser gets 23.2 dB at σ = 0.1 and 19.3 dB at σ = 0.2, against the EBT\'s best 21.6 and 17.7; DDIM at its best step count is also ahead. What does carry over: EBT quality saturates instead of degrading with more steps, its energy is higher on the OOD noise (mean 389 vs 153 after 16 steps), and long DDIM chains collapse at σ = 0.2 (5.3 dB at 100 passes). The paper\'s result is at 128×128 rather than 32×32, with models thousands of times larger and 100k training steps; a CPU toy can neither confirm nor refute it.</p>' },
    ],
    after: `
      <h3>What to take from this</h3>
      <ul>
        <li>The headline gains are out of distribution: +0.67 dB in distribution vs +3.73 dB at σ = 0.2.</li>
        <li>"99% fewer forward passes" compares 3 EBT steps with 300 DDIM steps, both applied recursively three times (p.35). After one or two applications the two models are level.</li>
        <li>The denoising EBT did not need the System 2 recipe: "it was not necessary to train EBTs with the S2 hyperparameters for System 2 Capabilities to emerge" (p.35). Its step size, step count and batch size are not given.</li>
        <li>One seed, no error bars. The DiT baseline keeps the hyperparameters of its original codebase except batch 128 (p.34–35).</li>
        <li>Generation is a different story. In a small text-to-image test EBTs "generated blurred images similar to the training distribution", which the authors attribute to training that makes the landscape convex around each target and so merges nearby modes (App. B.2, p.29).</li>
      </ul>
      <p class="note">Table 4 values are exact (p.13). Fig 12 points are digitized from the PDF (approx.); its 3- and 300-pass values match Table 4's OOD column. Fig 10 is a crop of the paper. The toy tabs use data/image.json: a CIFAR-10 denoising EBT, a same-size one-shot denoiser and a same-size DDIM model trained for this explainer on a CPU.</p>`,
    source: [{ kind: 'paper', note: 'Table 4 (exact), Fig 12 (digitized, approx.), Fig 10 (crop)' }, { kind: 'toy', note: 'CIFAR-10 32×32 denoising EBT vs FF and DDIM (toy tabs)' }],
    figure(stage, ctx) {
      const { lib } = ctx, h = lib.h, C = lib.C;
      const SD = lib.data('scaling'), IM = lib.data('image');
      const f12 = SD && SD.plots.find(p => p.id === 'fig12');
      const S = { tab: 't4', sig: 's1', cursor: 27.25, budget: 300, cost: 1, scale: 'lin', img: 0, tsig: '0.1', init: 'ebt', tstep: 16, nsig: '0.2', hover: null };
      const tabs = [['t4', 'Table 4'], ['12', 'Fig 12 · passes'], ['10', 'Fig 10 · examples'], ['probe', 'linear probe'], ['toy', 'toy · thinking steps'], ['toynfe', 'toy · vs baselines']];
      const tabRow = h('div', { class: 'controls ir-tabs' }); stage.appendChild(tabRow);
      tabRow.appendChild(h('span', { class: 'fig-label' }, 'exhibit'));
      const tabSeg = lib.segmented({ label: 'Exhibit', options: tabs, value: S.tab, onchange: v => setTab(v) }); tabRow.appendChild(tabSeg.el);
      const fr = lib.frame(stage, { label: 'Table 4', sub: '&nbsp;' });
      const labEl = fr.wrap.querySelector('.fig-label'), subEl = fr.wrap.querySelector('.fig-sub');
      const dom = h('div', { class: 'ir-dom' }); fr.frame.appendChild(dom);
      const cv = rcanvas(lib, fr.frame, { aspect: (w) => aspectFor(w), label: 'Image denoising results chart', draw: () => draw() });
      const tctl = h('div', { class: 'controls' }); stage.appendChild(tctl);
      const ro = h('div', { class: 'readout', 'aria-live': 'polite' }); stage.appendChild(ro);
      function aspectFor(w) {
        const n = w < 520;
        if (S.tab === 't4') return n ? Math.round(w * 0.95) : 300;
        if (S.tab === '12') return n ? Math.round(w * 0.85) : clamp(Math.round(w * 0.56), 300, 360);
        if (S.tab === 'probe') return n ? Math.round(w * 0.85) : 300;
        if (S.tab === 'toy') return n ? Math.round(w * 0.95) : 230;
        if (S.tab === 'toynfe') return n ? Math.round(w * 0.9) : clamp(Math.round(w * 0.56), 300, 360);
        return 10;
      }
      const T = (g, s, x, y, o = {}) => lib.text(g, s, x, y, Object.assign({ size: 11, kind: 'mono', color: C.ink }, o));
      function poly(g, pts, col, o = {}) { if (pts.length < 2) return; g.save(); g.strokeStyle = col; g.lineWidth = o.w || 1.5; g.globalAlpha *= (o.alpha == null ? 1 : o.alpha); g.lineJoin = 'round'; if (o.dash) g.setLineDash(o.dash); g.beginPath(); g.moveTo(pts[0][0], pts[0][1]); for (let i = 1; i < pts.length; i++) g.lineTo(pts[i][0], pts[i][1]); g.stroke(); g.restore(); }
      function mk(g, x, y, r, fill, stroke, kind) { g.save(); g.beginPath(); if (kind === 'sq') g.rect(x - r, y - r, 2 * r, 2 * r); else if (kind === 'x') { g.moveTo(x - r, y - r); g.lineTo(x + r, y + r); g.moveTo(x + r, y - r); g.lineTo(x - r, y + r); g.strokeStyle = stroke || fill; g.lineWidth = 1.6; g.stroke(); g.restore(); return; } else g.arc(x, y, r, 0, 7); if (fill) { g.fillStyle = fill; g.fill(); } if (stroke) { g.strokeStyle = stroke; g.lineWidth = 1.3; g.stroke(); } g.restore(); }
      let HITS = [];
      const hit = (x, y, lines) => HITS.push({ x, y, lines });
      function tooltip(g) {
        if (!S.hover) return; const { x, y, lines } = S.hover;
        const tw = Math.max(...lines.map(t => lib.measure(g, t, { size: 11, kind: 'mono' }).w)) + 14, th = lines.length * 14 + 10;
        let bx = x + 12, by = y - th - 6; if (bx + tw > cv.w - 4) bx = x - 12 - tw; if (by < 2) by = y + 12;
        g.save(); g.fillStyle = 'rgba(255,255,255,0.96)'; g.fillRect(bx, by, tw, th); g.strokeStyle = C.ink; g.lineWidth = 1; g.strokeRect(bx + 0.5, by + 0.5, tw - 1, th - 1); g.beginPath(); g.arc(x, y, 7, 0, 7); g.stroke(); g.restore();
        lines.forEach((t, i) => T(g, t, bx + 7, by + 6 + i * 14, { color: i === 0 ? C.ink : C.muted }));
      }
      function axesBox(g, o) {
        const W = o.W || cv.w, H = o.H || cv.h, rect = { x: o.left || 54, y: o.top || 16, w: W - (o.left || 54) - (o.right || 14), h: H - (o.top || 16) - (o.bottom || 44) };
        const tx = (v) => o.xlog ? L10(v) : v; const [xa, xb] = o.xlim.map(tx), [ya, yb] = o.ylim.map(o.ylog ? L10 : (v) => v);
        const X = (v) => rect.x + (tx(v) - xa) / (xb - xa) * rect.w, Y = (v) => rect.y + rect.h - ((o.ylog ? L10(v) : v) - ya) / (yb - ya) * rect.h;
        g.save(); g.strokeStyle = C.rule; g.lineWidth = 1;
        (o.xticks || []).forEach(v => { g.beginPath(); g.moveTo(X(v), rect.y); g.lineTo(X(v), rect.y + rect.h); g.stroke(); });
        (o.yticks || []).forEach(v => { g.beginPath(); g.moveTo(rect.x, Y(v)); g.lineTo(rect.x + rect.w, Y(v)); g.stroke(); });
        g.strokeStyle = C.ink; g.beginPath(); g.moveTo(rect.x, rect.y); g.lineTo(rect.x, rect.y + rect.h); g.lineTo(rect.x + rect.w, rect.y + rect.h); g.stroke(); g.restore();
        let lastR = -1e9;
        (o.xticks || []).forEach(v => { const s = (o.xfmt || num)(v), w = lib.measure(g, s, { size: 11, kind: 'mono' }).w, x = X(v); if (x - w / 2 < lastR + 4) return; lastR = x + w / 2; T(g, s, x, rect.y + rect.h + 6, { align: 'center', color: C.muted }); });
        (o.yticks || []).forEach(v => T(g, (o.yfmt || num)(v), rect.x - 6, Y(v), { align: 'right', baseline: 'middle', color: C.muted }));
        if (o.xlabel) T(g, o.xlabel, rect.x + rect.w / 2, rect.y + rect.h + 24, { align: 'center', size: 11.5, maxWidth: rect.w });
        if (o.ylabel) { g.save(); g.translate(13, rect.y + rect.h / 2); g.rotate(-Math.PI / 2); T(g, o.ylabel, 0, 0, { align: 'center', baseline: 'middle', size: 11.5 }); g.restore(); }
        return { rect, X, Y };
      }

      const TABS = {};
      // ---------- Table 4 + dB ruler ----------
      const tblEl = h('div', { class: 'tbl ir-t4' });
      tblEl.innerHTML = `<table><thead><tr><th>Model</th><th class="num" data-c="s1">σ=0.1 PSNR ↑</th><th class="num" data-c="s1">MSE ↓</th><th class="num" data-c="s2">σ=0.2 PSNR ↑</th><th class="num" data-c="s2">MSE ↓</th><th class="num" data-c="p">top-1 ↑</th><th class="num" data-c="p">top-5 ↑</th></tr></thead><tbody>
        ${['DiT', 'EBT'].map(m => `<tr class="${m === 'EBT' ? 'ebt' : ''}"><td>${m}</td><td class="num" data-c="s1">${T4[m].s1.psnr.toFixed(2)}</td><td class="num" data-c="s1">${T4[m].s1.mse}</td><td class="num" data-c="s2">${T4[m].s2.psnr.toFixed(2)}</td><td class="num" data-c="s2">${T4[m].s2.mse}</td><td class="num" data-c="p">${T4[m].top1}%</td><td class="num" data-c="p">${T4[m].top5}%</td></tr>`).join('')}
        <tr class="re"><td>from MSE</td><td class="num" data-c="s1" colspan="2">${psnrOf(T4.DiT.s1.mse).toFixed(2)} · ${psnrOf(T4.EBT.s1.mse).toFixed(2)}</td><td class="num" data-c="s2" colspan="2">${psnrOf(T4.DiT.s2.mse).toFixed(2)} · ${psnrOf(T4.EBT.s2.mse).toFixed(2)}</td><td class="num" data-c="p" colspan="2">ratio ×${(5.32 / 0.31).toFixed(1)} · ×${(13.2 / 1.36).toFixed(1)}</td></tr></tbody></table>`;
      TABS.t4 = {
        label: 'Table 4 · image denoising and classification', sub: 'COCO 2014 128×128 denoising, ImageNet-1k linear probe · exact values, p.13', src: 'paper',
        dom(el) { el.appendChild(tblEl); },
        controls(el) {
          el.appendChild(lib.segmented({ label: 'Noise level', options: [['s1', 'σ = 0.1 · in-dist.'], ['s2', 'σ = 0.2 · OOD']], value: S.sig, onchange: v => { S.sig = v; S.cursor = T4.EBT[v].psnr; draw(); readout(); } }).el);
          el.appendChild(h('span', { class: 'mono', style: { color: 'var(--muted)', fontSize: '12px' } }, 'drag on the ruler'));
        },
        geo() { const W = cv.w, H = cv.h, n = W < 520; const x0 = n ? 16 : 150, x1 = W - 22; return { W, H, n, x0, x1, X: (p) => x0 + (p - 15) / (30 - 15) * (x1 - x0), P: (x) => 15 + (x - x0) / (x1 - x0) * 15, yTop: 34, y1: n ? 104 : 104, y2: n ? 190 : 178, yBot: H - 46 }; },
        draw(g) {
          const G = this.geo(), { X } = G; tblEl.dataset.sig = S.sig;
          // PSNR axis on top, MSE axis at the bottom
          poly(g, [[G.x0, G.yTop], [G.x1, G.yTop]], C.ink, { w: 1 });
          for (let p = 15; p <= 30; p += 1) { const x = X(p); poly(g, [[x, G.yTop], [x, G.yTop + (p % 5 === 0 ? 7 : 4)]], C.ink, { w: 1 }); if (p % (G.n ? 5 : 2) === 0 || p === 15) T(g, String(p), x, G.yTop - 15, { align: 'center', color: C.muted }); }
          T(g, 'PSNR, dB (higher is better) →', G.x1, 4, { align: 'right', color: C.muted, size: 10.5 });
          poly(g, [[G.x0, G.yBot], [G.x1, G.yBot]], C.ink, { w: 1 });
          [2000, 1000, 500, 300, 200, 100, 50, 30, 20].forEach(m => { const p = psnrOf(m); if (p < 15 || p > 30) return; const x = X(p); poly(g, [[x, G.yBot], [x, G.yBot - 5]], C.ink, { w: 1 }); T(g, String(m), x, G.yBot + 5, { align: 'center', color: C.muted }); });
          T(g, '← MSE on 0–255 pixels (lower is better)', G.x1, G.yBot + 22, { align: 'right', color: C.muted, size: 10.5 });
          // rows
          [['s1', G.y1, 'σ = 0.1', 'in distribution'], ['s2', G.y2, 'σ = 0.2', 'out of distribution']].forEach(([k, y, a, b]) => {
            const on = S.sig === k, al = on ? 1 : 0.35;
            g.save(); g.globalAlpha = al;
            if (G.n) { T(g, `${a} · ${b}`, G.x0, y - 34, { weight: 600 }); } else { T(g, a, 12, y - 9, { weight: 600, size: 12 }); T(g, b, 12, y + 6, { color: C.muted, size: 10.5 }); }
            poly(g, [[G.x0, y], [G.x1, y]], C.rule, { w: 1 });
            const d = T4.DiT[k], e = T4.EBT[k], xd = X(d.psnr), xe = X(e.psnr);
            lib.arrow(g, xd + 6, y, xe - 7, y, { color: C.blue, width: 1.4, head: 6 });
            mk(g, xd, y, 5, '#fff', C.ink); mk(g, xe, y, 5.5, C.blue, '#fff');
            T(g, `DiT ${d.psnr}`, xd - 4, y + 10, { align: 'right' });
            T(g, `EBT ${e.psnr}`, xe + 4, y + 10, { color: C.blue, weight: 600 });
            const dd = e.psnr - d.psnr, ratio = d.mse / e.mse;
            T(g, `+${dd.toFixed(2)} dB = MSE ÷${ratio.toFixed(2)}`, (xd + xe) / 2, y - 18, { align: 'center', color: C.blue, size: 10.5 });
            hit(xd, y, ['DiT, ' + a, `PSNR ${d.psnr} · MSE ${d.mse}`, `RMS error ${Math.sqrt(d.mse).toFixed(1)} gray levels`]);
            hit(xe, y, ['EBT, ' + a, `PSNR ${e.psnr} · MSE ${e.mse}`, `RMS error ${Math.sqrt(e.mse).toFixed(1)} gray levels`]);
            g.restore();
          });
          // cursor
          const xc = X(S.cursor); poly(g, [[xc, G.yTop], [xc, G.yBot]], C.ink, { w: 1, dash: [3, 3] });
          const lab = `${S.cursor.toFixed(2)} dB ↔ MSE ${num(mseOf(S.cursor), 4)}`, tw = lib.measure(g, lab, { size: 11, kind: 'mono' }).w;
          T(g, lab, clamp(xc - tw / 2, 4, G.W - tw - 4), G.yBot - 22, { weight: 600 });
        },
        drag(px) { const G = this.geo(); S.cursor = clamp(G.P(px), 15, 30); },
        readout() {
          const m = mseOf(S.cursor), d = T4.DiT[S.sig], e = T4.EBT[S.sig];
          return [`cursor <b>${S.cursor.toFixed(2)} dB</b> = MSE <b>${num(m, 4)}</b> = RMS ${Math.sqrt(m).toFixed(1)} gray levels`, `${S.sig === 's1' ? 'σ = 0.1' : 'σ = 0.2'}: EBT +${(e.psnr - d.psnr).toFixed(2)} dB, MSE ${((1 - e.mse / d.mse) * 100).toFixed(1)}% lower`, `check: 10·log₁₀(255²/${e.mse}) = ${psnrOf(e.mse).toFixed(2)}`];
        },
      };
      // ---------- Fig 12 ----------
      TABS['12'] = {
        label: 'Fig 12 · PSNR as forward passes increase', sub: 'OOD noise (σ = 0.2), COCO · approx., read from Fig 12 · 3 and 300 match Table 4', src: 'paper',
        controls(el) {
          el.appendChild(lib.slider({ id: 'ir-budget', label: 'budget, forward passes', min: 0, max: L10(300), step: 0.005, value: L10(S.budget), fmt: v => String(Math.round(P10(v))), oninput: v => { S.budget = Math.round(P10(v)); draw(); readout(); } }).el);
          S._costSl = lib.slider({ id: 'ir-cost', label: 'cost of one EBT step', min: 1, max: 4, step: 0.25, value: S.cost, fmt: v => v === 1 ? '1 pass (paper)' : v.toFixed(2) + ' passes', oninput: v => { S.cost = v; draw(); readout(); } });
          el.appendChild(S._costSl.el);
        },
        pts() { const D = f12 ? f12.series[0].points : [[100, 14.31], [200, 18.86], [300, 19.57]], E = f12 ? f12.series[1].points : [[1, 13.92], [2, 18.82], [3, 23.29]]; return { D, E: E.map(q => [q[0] * S.cost, q[1], q[0]]) }; },
        best(arr) { let b = null; arr.forEach(q => { if (q[0] <= S.budget + 1e-9 && (!b || q[1] > b[1])) b = q; }); return b; },
        draw(g) {
          const { D, E } = this.pts();
          const M = axesBox(g, { xlog: true, xlim: [0.7, 450], ylim: [12, 25], xticks: [1, 2, 3, 5, 10, 30, 100, 200, 300], yticks: [13, 15, 17, 19, 21, 23, 25], xlabel: S.cost === 1 ? 'forward passes (NFE) · log' : 'forward-pass equivalents · log', ylabel: 'PSNR, dB (↑)' });
          const r = M.rect;
          // budget shading
          const xb = M.X(clamp(S.budget, 0.7, 450)); g.save(); g.fillStyle = 'rgba(17,17,17,0.04)'; g.fillRect(xb, r.y, r.x + r.w - xb, r.h); g.restore();
          poly(g, [[xb, r.y], [xb, r.y + r.h]], C.ink, { w: 1, dash: [3, 4] });
          { const bl = 'budget ' + S.budget, bw = lib.measure(g, bl, { size: 10.5, kind: 'mono' }).w, rt = xb + 4 + bw > r.x + r.w; T(g, bl, rt ? xb - 4 : xb + 4, r.y + 2, { color: C.muted, size: 10.5, align: rt ? 'right' : 'left' }); }
          // application pairing
          E.forEach((q, i) => { const d = D[i]; if (!d) return; poly(g, [[M.X(q[0]), M.Y(q[1])], [M.X(d[0]), M.Y(d[1])]], C.faint, { w: 1, dash: [2, 4] }); });
          T(g, 'dotted: same number of applications (1, 2, 3)', r.x + 6, r.y + r.h - 16, { color: C.muted, size: 10.5 });
          // Table 4 marks
          [[23.29, C.blue, 'Table 4 EBT 23.29'], [19.56, C.ink, 'Table 4 DiT 19.56']].forEach(([v, col, s]) => { poly(g, [[r.x + r.w - 8, M.Y(v)], [r.x + r.w, M.Y(v)]], col, { w: 2 }); });
          const bE = this.best(E), bD = this.best(D);
          poly(g, D.map(q => [M.X(q[0]), M.Y(q[1])]), C.ink, { w: 1.4 });
          poly(g, E.map(q => [M.X(q[0]), M.Y(q[1])]), C.blue, { w: 2 });
          D.forEach((q, i) => { const on = q[0] <= S.budget; mk(g, M.X(q[0]), M.Y(q[1]), 4.5, on ? '#fff' : '#fff', on ? C.ink : C.faint); hit(M.X(q[0]), M.Y(q[1]), [`DiT · ${q[0]} DDIM steps (${i + 1} application${i ? 's' : ''})`, `${q[1].toFixed(2)} dB (approx.)`]); });
          E.forEach((q, i) => { const on = q[0] <= S.budget; mk(g, M.X(q[0]), M.Y(q[1]), 5, on ? C.blue : '#fff', on ? '#fff' : C.blue2); hit(M.X(q[0]), M.Y(q[1]), [`EBT · ${q[2]} step${q[2] > 1 ? 's' : ''}${S.cost !== 1 ? ' = ' + num(q[0], 3) + ' pass-equivalents' : ''}`, `${q[1].toFixed(2)} dB (approx.)`]); });
          if (bE) { g.save(); g.strokeStyle = C.blue; g.lineWidth = 1.2; g.beginPath(); g.arc(M.X(bE[0]), M.Y(bE[1]), 9, 0, 7); g.stroke(); g.restore(); T(g, `EBT ${bE[1].toFixed(1)}`, M.X(bE[0]) + 12, M.Y(bE[1]) - 14, { color: C.blue, weight: 600 }); }
          if (bD) { g.save(); g.strokeStyle = C.ink; g.lineWidth = 1.2; g.beginPath(); g.arc(M.X(bD[0]), M.Y(bD[1]), 9, 0, 7); g.stroke(); g.restore(); T(g, `DiT ${bD[1].toFixed(1)}`, M.X(bD[0]) - 12, M.Y(bD[1]) - 22, { align: 'right', weight: 600 }); }
        },
        readout() {
          const { D, E } = this.pts(), bE = this.best(E), bD = this.best(D);
          const ratio = 300 / (3 * S.cost);
          return [`budget <b>${S.budget}</b>`, `EBT: ${bE ? `<b>${bE[1].toFixed(2)} dB</b> (${bE[2]} step${bE[2] > 1 ? 's' : ''})` : 'not enough for one step'}`, `DiT: ${bD ? `<b>${bD[1].toFixed(2)} dB</b> (${bD[0]} steps)` : 'needs at least 100 passes'}`, `3 EBT steps vs 300 DiT passes: <b>${num(ratio, 3)}×</b> fewer${S.cost === 1 ? ' (as counted by the paper)' : ' forward-pass equivalents [derived]'}`];
        },
      };
      // ---------- Fig 10 ----------
      const fig10 = h('figure', { class: 'paper-fig ir-fig10' }, h('img', { src: 'media/paper/fig10.png', alt: 'Paper Figure 10: out-of-distribution denoising of a bus and a street scene. Top row EBT after 1, 2, 3 steps and ground truth; bottom row DiT after 100, 200, 300 steps and ground truth.', loading: 'lazy' }));
      TABS['10'] = {
        label: 'Fig 10 · qualitative OOD image denoising', sub: 'crop of the paper figure, p.14', src: 'paper',
        dom(el) { el.appendChild(fig10); },
        controls(el) { el.appendChild(h('span', { class: 'mono', style: { color: 'var(--muted)', fontSize: '12px' } }, 'top: EBT 1, 2, 3 steps · bottom: DiT 100, 200, 300 steps · right: ground truth')); },
        readout() { return ['σ = 0.2 (OOD)', 'one EBT step per 100 DiT steps', 'Fig 10 caption: "less blurry than images denoised by DiT"']; },
      };
      // ---------- linear probe ----------
      TABS.probe = {
        label: 'Linear probe on ImageNet-1k', sub: 'mean of final patch tokens → linear classifier; DiT queried at T = 0 · exact values, Table 4', src: 'paper',
        controls(el) { S._scaleSeg = lib.segmented({ label: 'Scale', options: [['lin', '0–100% axis'], ['log', 'log axis']], value: S.scale, onchange: v => { S.scale = v; draw(); readout(); } }); el.appendChild(S._scaleSeg.el); },
        draw(g) {
          const W = cv.w, H = cv.h, lg = S.scale === 'log';
          const x0 = 70, x1 = W - 20, y0 = 26, y1 = H - 48;
          const vmin = 0.05, vmax = 100;
          const X = (v) => lg ? x0 + (L10(Math.max(v, vmin)) - L10(vmin)) / (L10(vmax) - L10(vmin)) * (x1 - x0) : x0 + v / vmax * (x1 - x0);
          const ticks = lg ? [0.1, 1, 10, 100] : [0, 20, 40, 60, 80, 100];
          ticks.forEach(v => { poly(g, [[X(v), y0 - 6], [X(v), y1]], C.rule, { w: 1 }); T(g, v + '%', X(v), y1 + 6, { align: 'center', color: C.muted }); });
          poly(g, [[x0, y0 - 6], [x0, y1]], C.ink, { w: 1 });
          T(g, 'accuracy' + (lg ? ' · log' : ''), (x0 + x1) / 2, y1 + 24, { align: 'center', size: 11.5 });
          const groups = [['top-1', 'top1', 0.1], ['top-5', 'top5', 0.5]], gh = (y1 - y0) / 2;
          groups.forEach(([nm, key, chance], gi) => {
            const gy = y0 + gi * gh;
            T(g, nm, 10, gy + gh / 2 - 6, { weight: 600, size: 12 });
            const bh = Math.min(18, gh * 0.24);
            [['DiT', T4.DiT[key], null], ['EBT', T4.EBT[key], C.blue]].forEach(([m, v, col], j) => {
              const by = gy + gh * 0.18 + j * (bh + 8);
              g.save(); if (col) { g.fillStyle = col; g.fillRect(x0, by, Math.max(1.5, X(v) - x0), bh); } else { g.strokeStyle = C.ink; g.lineWidth = 1; g.strokeRect(x0 + 0.5, by + 0.5, Math.max(1.5, X(v) - x0), bh - 1); } g.restore();
              T(g, `${m} ${v}%`, Math.max(X(v), x0) + 6, by + bh / 2, { baseline: 'middle', color: col || C.ink, weight: col ? 600 : 400 });
              hit(X(v), by + bh / 2, [`${m} ${nm}: ${v}%`, `chance ${chance}%`, `${(v / chance).toFixed(1)}× chance`]);
            });
            const xc = X(chance); poly(g, [[xc, gy + gh * 0.1], [xc, gy + gh * 0.82]], C.bad, { w: 1, dash: [2, 3] });
            T(g, 'chance ' + chance + '%', xc + 4, gy + gh * 0.82 - 4, { color: C.bad, size: 10.5, baseline: 'alphabetic' });
            const rt = T4.EBT[key] / T4.DiT[key];
            T(g, `×${rt.toFixed(1)}`, x1, gy + gh * 0.18, { align: 'right', color: C.blue, weight: 600, size: 12 });
          });
        },
        readout() { return [`top-1 EBT/DiT <b>×${(5.32 / 0.31).toFixed(1)}</b>`, `top-5 <b>×${(13.2 / 1.36).toFixed(1)}</b>`, 'paper: "around 10×"', `${S.scale === 'lin' ? 'linear axis: both bars are tiny' : 'log axis: ratios become distances'}`]; },
      };
      // ---------- toy: thinking steps ----------
      const toyBox = h('div', { class: 'ir-toy' });
      const tiles = {};
      [['clean', 'clean target'], ['noisy', 'noisy input x'], ['ebt', 'EBT ŷ step 16'], ['ff', 'one-shot FF'], ['ddim', 'DDIM best']].forEach(([k, l]) => {
        const img = h('img', { alt: l, loading: 'lazy' }), cap = h('span', { class: 'cap' }, l), ps = h('span', { class: 'ps' }, '');
        const fig = h('figure', { class: 'tile' + (k === 'ebt' ? ' on' : '') }, img, cap, ps); toyBox.appendChild(fig); tiles[k] = { img, cap, ps, fig };
      });
      const strips = IM && IM.strips ? IM.strips : [];
      const imgNames = [...new Set(strips.map(s => s.test_index + ':' + s.label))];
      const stripFor = () => { const nm = imgNames[S.img]; return strips.find(s => (s.test_index + ':' + s.label) === nm && String(s.sigma) === S.tsig); };
      let playT = null;
      function stopPlay() { if (playT) { clearInterval(playT); playT = null; } }
      function play() { stopPlay(); S.tstep = 0; sync(); if (lib.reducedMotion) { S.tstep = 16; sync(); return; } playT = setInterval(() => { S.tstep++; sync(); if (S.tstep >= 16) stopPlay(); }, 180); }
      function sync() { if (S._stepSl) S._stepSl.set(S.tstep); updateToyTiles(); draw(); readout(); }
      function updateToyTiles() {
        const st = stripFor(); if (!st) return; const v = st.variants[S.init];
        tiles.clean.img.src = st.frames.clean; tiles.noisy.img.src = st.frames.noisy; tiles.ff.img.src = st.frames.ff; tiles.ddim.img.src = st.frames.diff_best;
        tiles.ebt.img.src = v.step_frames[Math.min(S.tstep, v.step_frames.length - 1)];
        tiles.ebt.cap.textContent = `EBT ŷ step ${S.tstep}` + (S.tstep === 0 ? (S.init === 'ebt' ? ' (noise)' : ' (= x)') : '');
        tiles.ddim.cap.textContent = `DDIM best (${st.ddim_best_K} passes)`;
        tiles.clean.ps.textContent = '∞ dB'; tiles.noisy.ps.textContent = st.psnr.noisy_input.toFixed(1) + ' dB';
        tiles.ebt.ps.textContent = v.psnr_steps[Math.min(S.tstep, v.psnr_steps.length - 1)].toFixed(1) + ' dB'; tiles.ff.ps.textContent = st.psnr.ff.toFixed(1) + ' dB'; tiles.ddim.ps.textContent = st.psnr.diff_best.toFixed(1) + ' dB';
      }
      TABS.toy = {
        label: 'Toy · a denoising EBT thinking on CIFAR-10', sub: 'trained for this explainer · 32×32 images shown enlarged · PSNR of this image', src: 'toy',
        dom(el) { el.appendChild(toyBox); updateToyTiles(); },
        controls(el) {
          if (!strips.length) { el.appendChild(h('span', { class: 'mono' }, 'toy data missing')); return; }
          el.appendChild(lib.segmented({ label: 'Image', options: imgNames.map((n, i) => [i, n.split(':')[1]]), value: S.img, onchange: v => { S.img = v; sync(); } }).el);
          el.appendChild(lib.segmented({ label: 'Noise', options: [['0.1', 'σ 0.1'], ['0.2', 'σ 0.2 (OOD)']], value: S.tsig, onchange: v => { S.tsig = v; sync(); } }).el);
          el.appendChild(lib.segmented({ label: 'Start', options: [['ebt', 'start: noise'], ['ebtx', 'start: x']], value: S.init, onchange: v => { S.init = v; sync(); } }).el);
          S._stepSl = lib.slider({ id: 'ir-tstep', label: 'thinking step', min: 0, max: 16, step: 1, value: S.tstep, fmt: v => String(v), oninput: v => { stopPlay(); S.tstep = v; updateToyTiles(); draw(); readout(); } });
          el.appendChild(S._stepSl.el); el.appendChild(lib.button('play', () => play(), { primary: true }));
        },
        draw(g) {
          const st = stripFor(); if (!st) { T(g, 'toy data missing (data/image.json)', 20, 20); return; }
          const v = st.variants[S.init], W = cv.w, n = W < 520;
          const ps = v.psnr_steps, es = v.energy_steps, k = S.tstep;
          const half = n ? W : Math.round(W * 0.55);
          // PSNR chart
          const lo = Math.floor(Math.min(...ps, st.psnr.noisy_input) / 5) * 5, hi = Math.ceil(Math.max(...ps, st.psnr.ff, st.psnr.diff_best) / 5) * 5 + 1;
          const M = axesBox(g, { W: half, H: n ? Math.round(cv.h * 0.55) : cv.h, xlim: [0, 16.4], ylim: [lo, hi], xticks: [0, 4, 8, 12, 16], yticks: [lo, (lo + hi) / 2 | 0, hi - 1].filter((x, i, a) => a.indexOf(x) === i), xlabel: 'thinking step', ylabel: 'PSNR, dB', bottom: 40 });
          const r = M.rect;
          const L2 = [[M.Y(st.psnr.ff), 'one-shot FF', [5, 4]], [M.Y(st.psnr.diff_best), 'DDIM best (' + st.ddim_best_K + ')', [2, 3]]].sort((p, q) => p[0] - q[0]);
          L2.forEach(([yy, , dash]) => poly(g, [[r.x, yy], [r.x + r.w, yy]], C.ink, { w: 1, dash }));
          T(g, L2[0][1], r.x + r.w - 4, L2[0][0] - 12, { size: 10.5, align: 'right' });
          if (L2[1][0] - L2[0][0] >= 13) T(g, L2[1][1], r.x + r.w - 4, L2[1][0] - 12, { size: 10.5, align: 'right' });
          else T(g, L2[1][1], r.x + 4, L2[1][0] + 3, { size: 10.5 });
          poly(g, ps.map((y, i) => [M.X(i), M.Y(y)]), C.blue, { w: 2 });
          ps.forEach((y, i) => { mk(g, M.X(i), M.Y(y), i === k ? 5 : 2.5, i <= k ? C.blue : '#fff', i <= k ? '#fff' : C.blue2); hit(M.X(i), M.Y(y), [`step ${i}`, `PSNR ${y.toFixed(2)} dB · energy ${num(es[i], 4)}`]); });
          // energy chart
          const ex0 = n ? 54 : half + 52, ey0 = n ? Math.round(cv.h * 0.55) + 24 : 16, ew = (n ? W - 54 - 14 : W - half - 52 - 14), eh = n ? cv.h - ey0 - 40 : cv.h - 16 - 40;
          const elo = L10(Math.min(...es)) - 0.05, ehi = L10(Math.max(...es)) + 0.05;
          const EX = (i) => ex0 + i / 16.4 * ew, EY = (e) => ey0 + eh - (L10(e) - elo) / (ehi - elo) * eh;
          poly(g, [[ex0, ey0], [ex0, ey0 + eh], [ex0 + ew, ey0 + eh]], C.ink, { w: 1 });
          [0, 8, 16].forEach(i => T(g, String(i), EX(i), ey0 + eh + 6, { align: 'center', color: C.muted }));
          [Math.min(...es), Math.max(...es)].forEach(e => T(g, num(e, 3), ex0 - 6, EY(e), { align: 'right', baseline: 'middle', color: C.muted }));
          T(g, 'energy E(x, ŷ) · log', ex0 + ew / 2, ey0 + eh + 22, { align: 'center', size: 11.5 });
          poly(g, es.map((e, i) => [EX(i), EY(e)]), C.ink, { w: 1.5 });
          es.forEach((e, i) => mk(g, EX(i), EY(e), i === k ? 4.5 : 2, i <= k ? C.ink : '#fff', C.ink));
        },
        readout() {
          const st = stripFor(); if (!st) return ['toy data missing'];
          const v = st.variants[S.init], k = S.tstep, R = IM.results[S.tsig][S.init];
          const ev = R.curve[Math.min(k, R.curve.length - 1)];
          return [`step <b>${k}</b>: this image ${v.psnr_steps[k].toFixed(2)} dB, energy ${num(v.energy_steps[k], 4)}`, `300-image eval mean at step ${k}: <b>${ev.psnr.toFixed(2)} dB</b>`, `same image: FF ${st.psnr.ff.toFixed(1)} · DDIM best ${st.psnr.diff_best.toFixed(1)} · noisy ${st.psnr.noisy_input.toFixed(1)}`];
        },
      };
      // ---------- toy: PSNR vs NFE against baselines ----------
      TABS.toynfe = {
        label: 'Toy · PSNR vs forward passes, EBT vs baselines', sub: 'mean over 300 CIFAR-10 eval images · trained for this explainer', src: 'toy',
        controls(el) { el.appendChild(lib.segmented({ label: 'Noise', options: [['0.1', 'σ = 0.1 (train level)'], ['0.2', 'σ = 0.2 (OOD)']], value: S.nsig, onchange: v => { S.nsig = v; draw(); readout(); } }).el); },
        draw(g) {
          if (!IM) { T(g, 'toy data missing', 20, 20); return; }
          const R = IM.results[S.nsig];
          const M = axesBox(g, { xlog: true, xlim: [0.8, 130], ylim: [4, 29], xticks: [1, 2, 3, 5, 10, 20, 50, 100], yticks: [5, 10, 15, 20, 25], xlabel: 'network evaluations (NFE) · log', ylabel: 'PSNR, dB (↑)' });
          const r = M.rect;
          const ni = R.noisy_input.psnr; poly(g, [[r.x, M.Y(ni)], [r.x + r.w, M.Y(ni)]], C.faint, { w: 1, dash: [4, 4] }); T(g, 'noisy input ' + ni.toFixed(1), r.x + 4, M.Y(ni) + 3, { color: C.muted, size: 10.5 });
          const e = R.ebt.curve.filter(c => c.nfe >= 1), ex = R.ebtx.curve.filter(c => c.nfe >= 1), d = R.diff_curve;
          poly(g, d.map(c => [M.X(c.nfe), M.Y(c.psnr)]), C.ink, { w: 1.4 });
          d.forEach(c => { mk(g, M.X(c.nfe), M.Y(c.psnr), 3.6, '#fff', C.ink); hit(M.X(c.nfe), M.Y(c.psnr), [`DDIM, ${c.nfe} steps`, `${c.psnr.toFixed(2)} dB`]); });
          poly(g, ex.map(c => [M.X(c.nfe), M.Y(c.psnr)]), C.blue, { w: 1.4, dash: [5, 4] });
          poly(g, e.map(c => [M.X(c.nfe), M.Y(c.psnr)]), C.blue, { w: 2 });
          e.forEach(c => { if ([1, 2, 3, 4, 8, 16, 24].includes(c.nfe)) { mk(g, M.X(c.nfe), M.Y(c.psnr), 3.4, C.blue, '#fff'); hit(M.X(c.nfe), M.Y(c.psnr), [`EBT (start: noise), ${c.nfe} steps`, `${c.psnr.toFixed(2)} dB · mean energy ${num(c.energy_mean, 4)}`]); } });
          ex.forEach(c => { if ([1, 2, 3, 4, 8, 16, 24].includes(c.nfe)) hit(M.X(c.nfe), M.Y(c.psnr), [`EBT (start: noisy x), ${c.nfe} steps`, `${c.psnr.toFixed(2)} dB`]); });
          mk(g, M.X(1), M.Y(R.ff.psnr), 4.5, C.ink, null, 'sq'); hit(M.X(1), M.Y(R.ff.psnr), ['one-shot FF (step-matched training)', `${R.ff.psnr.toFixed(2)} dB, 1 pass`]);
          mk(g, M.X(1), M.Y(R.ff_time.psnr), 4.5, '#fff', C.ink, 'sq'); hit(M.X(1), M.Y(R.ff_time.psnr), ['one-shot FF (same wall-clock as EBT)', `${R.ff_time.psnr.toFixed(2)} dB, 1 pass`]);
          T(g, 'FF', M.X(1) + 8, M.Y(R.ff.psnr) - 5, { size: 10.5, baseline: 'middle' });
          (R.ebtx.recursive_x3 || []).forEach(c => { mk(g, M.X(c.nfe), M.Y(c.psnr), 4, null, C.blue, 'x'); hit(M.X(c.nfe), M.Y(c.psnr), [`EBT (start: x) ×3 recursive, ${c.steps_per_application} step(s) each`, `${c.psnr.toFixed(2)} dB, ${c.nfe} NFE`]); });
          (R.diff_recursive_x3 || []).forEach(c => { mk(g, M.X(c.nfe), M.Y(c.psnr), 4, null, C.ink, 'x'); hit(M.X(c.nfe), M.Y(c.psnr), [`DDIM ×3 recursive, K = ${c.ddim_steps_per_application}`, `${c.psnr.toFixed(2)} dB, ${c.nfe} NFE`]); });
          // legend
          let ly = r.y + 4; const lx = r.x + r.w - 6;
          [['EBT, start from noise', C.blue, null], ['EBT, start from noisy x', C.blue, [5, 4]], ['DDIM (K steps)', C.ink, null], ['× three recursive applications', null, null]].forEach(([s, col, dash]) => {
            if (s.startsWith('×') && !R.diff_recursive_x3) return;
            const tw = lib.measure(g, s, { size: 10.5, kind: 'mono' }).w;
            if (col) poly(g, [[lx - tw - 30, ly + 6], [lx - tw - 8, ly + 6]], col, { w: 2, dash });
            T(g, s, lx, ly, { align: 'right', size: 10.5, color: col || C.muted }); ly += 14;
          });
        },
        readout() {
          if (!IM) return ['toy data missing'];
          const R = IM.results[S.nsig], eb = Math.max(...R.ebt.curve.map(c => c.psnr));
          const e16 = R.ebt.curve.find(c => c.nfe === 16);
          return [`EBT best <b>${eb.toFixed(2)} dB</b>`, `one-shot FF <b>${R.ff.psnr.toFixed(2)}</b> (wall-clock-matched ${R.ff_time.psnr.toFixed(2)})`, `DDIM best ${R.diff_best.psnr.toFixed(2)} at ${R.diff_best.nfe} passes · at 100: ${R.diff_curve[R.diff_curve.length - 1].psnr.toFixed(2)}`, `EBT mean energy after 16 steps: <b>${num(e16.energy_mean, 4)}</b> (σ 0.1: ${num(IM.results['0.1'].ebt.curve[16].energy_mean, 4)}, σ 0.2: ${num(IM.results['0.2'].ebt.curve[16].energy_mean, 4)})`];
        },
      };

      // ---------- generic ----------
      function nearestHit(px, py) { let best = null, bd = 14; HITS.forEach(hh => { const d = Math.hypot(hh.x - px, hh.y - py); if (d < bd) { bd = d; best = hh; } }); return best; }
      function draw() {
        const tb = TABS[S.tab]; if (!tb.draw) return; if (!cv.w) return;
        const g = cv.ctx; cv.clear(); HITS = [];
        try { tb.draw(g); } catch (e) { console.error('image-results', e); }
        if (S.hover) tooltip(g);
      }
      function readout() { const tb = TABS[S.tab]; ro.innerHTML = (tb.readout ? tb.readout() : []).map(s => `<span>${s}</span>`).join(''); }
      const CAP = {
        t4: 'Top ruler: PSNR in dB. Bottom ruler: the same positions in MSE (0–255 pixels), which runs backwards and is logarithmic. Hollow ink circles: DiT. Blue dots: EBT. The dimmed row is the other noise level.',
        '12': 'Blue: EBT after 1, 2, 3 steps. Ink: DiT after 100, 200, 300 DDIM steps. Ringed: the best each model reaches within your budget. Ticks on the right edge: Table 4 values. The y axis is linear here; the paper draws it log-scaled.',
        '10': 'Paper Figure 10, out-of-distribution noise (σ = 0.2).',
        probe: 'Outlined: DiT. Blue: EBT. Red dashed: chance level for 1,000 classes.',
        toy: 'Tiles: the clean target, the noisy input the model is conditioned on, the EBT prediction at the chosen step, and the two baselines. Left chart: PSNR of this image per step (dashed: baselines). Right: the energy of the prediction.',
        toynfe: 'Mean over the eval set. Squares: one-shot feed-forward denoiser (filled: same training steps; hollow: same wall-clock). Crosses: three recursive applications, as in the paper\'s App. D.3.',
      };
      function setTab(v) {
        if (!TABS[v]) return; stopPlay(); S.tab = v; S.hover = null; tabSeg.set(v);
        const tb = TABS[v]; labEl.textContent = tb.label; subEl.innerHTML = tb.sub;
        dom.innerHTML = ''; if (tb.dom) tb.dom(dom);
        cv.box.style.display = tb.draw ? '' : 'none';
        tctl.innerHTML = ''; tb.controls && tb.controls(tctl);
        const tag = { paper: 'from the paper', toy: 'toy model trained for this explainer' }[tb.src] || '';
        ctx.setCaption(`<span class="src ${tb.src}">${tag}</span> ${CAP[v] || ''}`);
        cv.fit(); draw(); readout();
      }
      // interaction
      let dragging = false;
      cv.canvas.addEventListener('pointerdown', (ev) => { if (S.tab !== 't4') return; dragging = true; cv.canvas.setPointerCapture && cv.canvas.setPointerCapture(ev.pointerId); const [px] = cv.local(ev); TABS.t4.drag(px); draw(); readout(); });
      cv.canvas.addEventListener('pointerup', () => { dragging = false; });
      cv.canvas.addEventListener('pointermove', (ev) => {
        const [px, py] = cv.local(ev);
        if (S.tab === 't4' && dragging) { TABS.t4.drag(px); draw(); readout(); return; }
        const n = nearestHit(px, py); const same = n === S.hover || (n && S.hover && n.x === S.hover.x && n.y === S.hover.y);
        S.hover = n; cv.canvas.style.cursor = S.tab === 't4' ? 'ew-resize' : (n ? 'pointer' : 'default'); if (!same) draw();
      });
      cv.canvas.addEventListener('pointerleave', () => { if (S.hover) { S.hover = null; draw(); } });
      const CFG = [
        () => { S.sig = 's1'; S.cursor = 27.25; setTab('t4'); },
        () => { S.sig = 's2'; S.cursor = 23.29; setTab('t4'); },
        () => { S.budget = 300; S.cost = 1; setTab('12'); },
        () => { S.budget = 300; S.cost = 3; setTab('12'); },
        () => { setTab('10'); },
        () => { S.scale = 'lin'; setTab('probe'); },
        () => { S.img = 0; S.tsig = '0.1'; S.init = 'ebt'; setTab('toy'); if (ctx.visible()) play(); else { S.tstep = 16; sync(); } },
        () => { S.nsig = '0.2'; setTab('toynfe'); },
      ];
      setTab('t4');
      return {
        step(i) { CFG[Math.max(0, Math.min(CFG.length - 1, i))](); },
        show() { draw(); },
        hide() { stopPlay(); },
      };
    },
  });
})();
