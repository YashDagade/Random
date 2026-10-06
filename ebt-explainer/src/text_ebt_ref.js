// Reference JS port of the toy character-level EBT in data/text.json (toy model trained for this explainer).
// Exact same math as src/text_ebt.py: encoder MLP -> h; energy MLP over [h, e, h*e] with e = softmax(yhat) @ Wpe;
// thinking = gradient descent on the logits yhat. Works in the browser (window.EBTText) and in Node (module.exports).
// Usage:
//   const m = EBTText.load(EBT_DATA.text);           // decodes the base64 float32 weights
//   const h = m.encode(m.textToIds("...last 40 chars..."));
//   const tr = m.think(h, yhat0, alpha, nSteps);     // {ys:[Float32Array], E:[number], p:[Float32Array]}
(function (root) {
  function b64ToF32(s) {
    let bytes;
    if (typeof atob === "function") {
      const bin = atob(s); bytes = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    } else { bytes = new Uint8Array(Buffer.from(s, "base64")); }
    return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
  }
  const sig = (x) => 1 / (1 + Math.exp(-x));
  const silu = (x) => x * sig(x);
  const dsilu = (x) => { const s = sig(x); return s * (1 + x * (1 - s)); };

  function load(text) {
    const w = text.weights, all = b64ToF32(w.data), T = {};
    for (const k of w.order) {
      const t = w.tensors[k], n = t.shape.reduce((a, b) => a * b, 1);
      T[k] = { d: all.subarray(t.offset, t.offset + n), shape: t.shape };
    }
    const V = text.vocab.length, C = text.arch.context_len, dc = T.emb.shape[1];
    const Hh = T.We1.shape[1], dh = T.We2.shape[1], H = T.W1.shape[1];
    const stoi = {}; text.vocab.forEach((c, i) => (stoi[c] = i));
    // y[j] = sum_i x[i] * W[i][j] + b[j]   (W row-major [in, out])
    function lin(x, W, b, nin, nout) {
      const y = new Float32Array(nout);
      if (b) y.set(b.d);
      for (let i = 0; i < nin; i++) { const xi = x[i]; if (xi === 0) continue; const off = i * nout;
        for (let j = 0; j < nout; j++) y[j] += xi * W.d[off + j]; }
      return y;
    }
    // x[i] = sum_j W[i][j] * g[j]   (backprop through a linear layer)
    function linT(g, W, nin, nout) {
      const x = new Float32Array(nin);
      for (let i = 0; i < nin; i++) { let s = 0; const off = i * nout; for (let j = 0; j < nout; j++) s += W.d[off + j] * g[j]; x[i] = s; }
      return x;
    }
    function textToIds(s) {
      s = s.toLowerCase().slice(-C);
      const ids = []; for (const ch of s) ids.push(stoi[ch] !== undefined ? stoi[ch] : stoi["#"]);
      while (ids.length < C) ids.unshift(stoi[" "]);
      return ids;
    }
    function encode(ids) {
      const x = new Float32Array(C * dc);
      ids.forEach((id, t) => { for (let k = 0; k < dc; k++) x[t * dc + k] = T.emb.d[id * dc + k]; });
      const a = lin(x, T.We1, T.be1, C * dc, Hh).map(silu);
      return lin(a, T.We2, T.be2, Hh, dh);
    }
    function softmax(y) {
      let m = -Infinity; for (const v of y) m = Math.max(m, v);
      const p = new Float32Array(y.length); let s = 0;
      for (let i = 0; i < y.length; i++) { p[i] = Math.exp(y[i] - m); s += p[i]; }
      for (let i = 0; i < y.length; i++) p[i] /= s; return p;
    }
    // returns {E, grad (dE/dyhat), p}
    function energyGrad(h, yhat) {
      const p = softmax(yhat), e = lin(p, T.Wpe, null, V, dh);
      const z = new Float32Array(3 * dh);
      for (let k = 0; k < dh; k++) { z[k] = h[k]; z[dh + k] = e[k]; z[2 * dh + k] = h[k] * e[k]; }
      const u1 = lin(z, T.W1, T.b1, 3 * dh, H), a1 = u1.map(silu);
      const u2 = lin(a1, T.W2, T.b2, H, H), a2 = u2.map(silu);
      let E = T.b3.d[0]; for (let j = 0; j < H; j++) E += a2[j] * T.W3.d[j];
      const du2 = new Float32Array(H); for (let j = 0; j < H; j++) du2[j] = T.W3.d[j] * dsilu(u2[j]);
      const da1 = linT(du2, T.W2, H, H), du1 = new Float32Array(H);
      for (let j = 0; j < H; j++) du1[j] = da1[j] * dsilu(u1[j]);
      const dz = linT(du1, T.W1, 3 * dh, H), de = new Float32Array(dh);
      for (let k = 0; k < dh; k++) de[k] = dz[dh + k] + h[k] * dz[2 * dh + k];
      const dp = linT(de, T.Wpe, V, dh);
      let pd = 0; for (let v = 0; v < V; v++) pd += p[v] * dp[v];
      const grad = new Float32Array(V); for (let v = 0; v < V; v++) grad[v] = p[v] * (dp[v] - pd);
      return { E, grad, p };
    }
    function think(h, yhat0, alpha, n, noiseStd = 0, rand = null) {
      let y = Float32Array.from(yhat0); const ys = [], E = [], P = [];
      for (let i = 0; i <= n; i++) {
        const r = energyGrad(h, y); ys.push(Float32Array.from(y)); E.push(r.E); P.push(r.p);
        if (i === n) break;
        const nx = new Float32Array(V);
        for (let v = 0; v < V; v++) nx[v] = y[v] - alpha * r.grad[v] + (noiseStd > 0 && rand ? noiseStd * rand() : 0);
        y = nx;
      }
      return { ys, E, p: P };
    }
    return { V, C, vocab: text.vocab, textToIds, encode, energyGrad, think, softmax };
  }
  const api = { load };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.EBTText = api;
})(typeof window !== "undefined" ? window : globalThis);

// Self-test in Node: node src/text_ebt_ref.js data/text.json
if (typeof require !== "undefined" && typeof module !== "undefined" && require.main === module) {
  const d = JSON.parse(require("fs").readFileSync(process.argv[2] || "data/text.json", "utf8"));
  const m = module.exports.load(d), ref = d.weights_reference_trace;
  const h = m.encode(ref.context_ids), tr = m.think(h, ref.yhat0, ref.alpha, 4);
  console.log("JS energies ", tr.E.map((x) => x.toFixed(4)).join(" "));
  console.log("PY energies ", ref.energies.join(" "));
  console.log("JS yhat4[:5]", Array.from(tr.ys[4].slice(0, 5)).map((x) => x.toFixed(4)).join(" "));
  console.log("PY yhat4[:5]", ref.yhat4_first5.join(" "));
}
