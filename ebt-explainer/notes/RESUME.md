# Resume plan v2 (user feedback: site too long)

USER REQUEST: whole site ≈ 5,000 words (not much more), focused on INTUITION for the methods and the ARCHITECTURE. Keep the strong interactive figures and animations; cut prose and results/data detail.
Do NOT resume the old critique workflows (they review the long versions). Instead consolidate.

Step 1: measure words per panel (strip HTML/KaTeX from text/steps/after) to see the current total.
Step 2: restructure into ~14 panels, ~300-400 words each (hard cap 450), total ≈ 5,000:
  1 families        ← system2 + families (why a Transformer's per-token compute is fixed; AR / RNN / diffusion / EBT; Table 1 facets)
  2 energy          ← energy (E, Boltzmann, unnormalized; verifying vs generating)
  3 landscape       ← landscape + two-landscapes (energy landscape over ŷ vs loss landscape over θ, toggle)
  4 descent         ← keep (reference panel; trim to ~300 words)
  5 langevin-bon    ← keep (noise + Best-of-N self-verification)
  6 tokens          ← tokens + one paragraph of uncertainty insight (energy defined up to c(x); Fig 8)
  7 training        ← alg1 + second-order (unrolled optimization, backprop through it, Hessian-vector products)
  8 shaping         ← contrastive + regularizers (why optimization-based training scales; replay, Langevin, random α/N)
  9 learning        ← keep (landscape forming over checkpoints)
 10 architecture    ← keep (how ŷ enters the Transformer, AR leakage fix, bidirectional, cost per step)
 11 data            ← text-data + image-data + video-data (compact: what training/eval samples look like, perplexity, PSNR, noise σ)
 12 results         ← scaling + thinking-results + image-results + costs (compact, honest: per-axis rates, 29% context, 3 vs 300 passes, compute cost)
 13 planning        ← planning (energy planning steps + MPC steps, CEM comparison)
 14 world-models    ← world-models trimmed (JEPA connection) + tiny glossary
Use a workflow: ~6 merge agents (each 2-3 target panels; they read the source panel files, keep the best figure(s) with all interactivity working, rewrite prose to the word budget with intuition first, equations kept where they carry the idea), then 2 reviewer agents (accuracy vs notes/paper_facts.md + visual QA desktop/phone via src/shot_panel.js), then a final word-count check (≈5,000).
Update assets/js/panels/manifest.js to the new ids/parts; move retired panel files to legacy/panels/. Update README word/section counts. Rebundle (python3 src/bundle_data.py), test every panel, commit, push.
