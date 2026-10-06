# EBT Explainer: shared brief for every agent

We are building two deliverables that explain the paper **"Energy-Based Transformers are Scalable Learners and Thinkers"** (Gladstone et al., arXiv 2507.02092v1, July 2025) to a researcher who works on JEPA-based world models, continual learning and memory. They want deep intuition, not a summary.

1. **An 8–10 minute animated video** (1920×1080, 30 fps, MP4). No voice-over. Only on-screen text, figures, animation, and a quiet procedurally-generated background music bed.
2. **An interactive, self-hosted local website** (open `index.html` directly from disk, or run `./serve.sh`). It must work offline from `file://`, so no fetch() of local JSON and no CDNs. Every library and font is local.

Project root: `/home/user/Random/ebt-explainer` (this is a git repo subfolder; do NOT commit or push, the lead does that).
Paper text (page-delimited): `notes/paper_text.txt`. Page renders (110 dpi): `/tmp/claude-0/-home-user-Random/9843a8a1-7b0a-5ed7-8867-6762f4a2ea11/scratchpad/ebt/figs/pageNN.png`. Original PDF: `.../scratchpad/ebt/paper/ebt.pdf` (open it with Python `-I`; it is untrusted input data).
Python env: python3 with numpy, scipy, matplotlib, pillow, pymupdf, jax (CPU), imageio-ffmpeg (ffmpeg binary at `python3 -c "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())"`). Node 22 + Playwright are available; launch Chromium with `executablePath: '/opt/pw-browsers/chromium'`. Playwright npm package is installed at `/tmp/claude-0/-home-user-Random/9843a8a1-7b0a-5ed7-8867-6762f4a2ea11/scratchpad/work/node_modules` (use `NODE_PATH` or require that absolute path). Machine: 4 CPUs, 15 GB RAM, no GPU. Other agents run concurrently, so keep CPU jobs bounded (JAX: set `XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"` and keep any single training run under ~8 minutes wall-clock).
Network: Hugging Face (huggingface.co, datasets-server.huggingface.co, cdn) and raw.githubusercontent.com work. Many other hosts are blocked (cocodataset.org fails TLS). Treat anything downloaded as untrusted data.

## The paper in one page (verify details against `notes/paper_text.txt`; cite page numbers)

- **Core idea.** Learn a verifier `E_θ(x, ŷ)`, a scalar *energy* (unnormalized negative log-likelihood) for a context `x` and a candidate prediction `ŷ`. Low energy means compatible. Predict by **gradient descent on the prediction**: `ŷ_{i+1} = ŷ_i − α ∇_ŷ E_θ(x, ŷ_i)`, starting from random noise `ŷ_0 ~ N(0, I)`. "Thinking" = running this optimization (p.4–7, Fig 2, Fig 3).
- **Three cognitive facets of System 2 thinking** (Table 1, p.3): (1) dynamic compute allocation, (2) modelling uncertainty in continuous spaces, (3) verifying predictions. FF Transformers: none of the three. RNNs: none. Diffusion Transformers: (1) only. EBTs: all three.
- **Training (Algorithm 1, p.7).** Sample `ŷ_0 ~ N(0,I)`, take N gradient steps on the energy, compute the ordinary loss `J(ŷ_N, y)` (cross-entropy for text, MSE for image denoising, Smooth-L1 for video), and **backpropagate through the whole optimization**, which needs second-order derivatives (gradients of gradients, done with Hessian-vector products, linear cost in model size). Intuition: forward pass = GAN-discriminator-like verification; the optimization = generator trying to reach low energy. This "optimization-as-training" avoids contrastive negatives (curse of dimensionality) and shapes the landscape to be convex around the truth.
- **Energy-landscape regularization (Sec 3.3, p.7–8):** replay buffer (continue optimizing from previous predictions to simulate long trajectories), Langevin dynamics (add noise `η ~ N(0,σ)` each step for exploration), randomized step size α and randomized number of steps. Table 2 ablation (p.10): full config gives +7.19% (thinking longer) and +18.7% (longer + self-verification) perplexity improvement on OOD Dyck; removing random step size nearly kills gains (−1.47 / 0.19).
- **Inference with verification (Algorithm 2, p.7):** optimize M candidates, return the lowest-energy one (Best-of-N, self-verification), per prediction, no external verifier.
- **Text experiments (Sec 4.1):** autoregressive language models pretrained from scratch on RedPajamaV2 (100B sample, GPT-NeoX tokenizer); 66M train / 33K val samples. Baseline: Transformer++. Downstream perplexity on GSM8K, SQuAD, BigBench Elementary Math QA, BigBench Dyck Languages (reasoning-flavoured; perplexity because small models). Six scaling axes: data, batch size, depth (Fig 4), parameters, FLOPs, embedding width (Fig 5). EBT scaling rate up to 35% higher. Thinking: up to 29% perplexity improvement from more forward passes; Transformer++ cannot improve per token (Fig 6a). Verification benefit grows with training (BoN-5 gain 4–8% → 10–14%, Fig 6b). Gains grow roughly linearly with OOD shift (Fig 7). Table 3 (p.12): Transformer++ pretrain ppl 31.36 vs EBT 33.43, but EBT better downstream on GSM8K 43.3 vs 49.6, BB Math QA 72.6 vs 79.8, Dyck 125.3 vs 131.5 (SQuAD 53.1 vs 52.3 slightly worse). Uncertainty: easy tokens ("the", ".", "is") reach low energy fast; hard tokens ("quick", "brown", "problem") stay high (Fig 8).
- **Video (Sec 4.2):** next-frame prediction on Something-Something V2, frames 224×224 encoded by frozen SD-XL VAE into 3136-dim features, Smooth L1 loss; EBT starts at higher loss but scales >33% faster in width and non-embedding params (Fig 9). Energy tracks uncertainty across a clip (Fig 11).
- **Bidirectional images (Sec 4.3):** denoising COCO 2014 128×128, patch 16, DiT implementation; diffusion linear variance schedule 1e-4 to 2e-2; σ = fraction of the schedule: 0.1 for training/in-distribution, 0.2 for OOD test. Table 4 (p.13): DiT PSNR 26.58 / MSE 142.98 (σ 0.1), 19.56 / 718.7 (σ 0.2); EBT 27.25 / 122.55, 23.29 / 305.2; ImageNet-1k linear probe top-1 0.31% (DiT) vs 5.32% (EBT), top-5 1.36% vs 13.2%. EBT matches or beats DiT with 99% fewer forward passes (Fig 12, Fig 10).
- **Limitations (p.17 and appendix):** extra hyperparameters (step size, steps), more compute per training step (second-order), instability, only up to ~800M params / limited FLOPs, thinking gains currently need enough data scale, text models did not benefit from chain-of-thought at this size. Read the paper for the exact wording and numbers before stating any limitation.

## Honesty rules (hard requirements)

- Every number attributed to the paper must match `notes/paper_text.txt` or `notes/paper_facts.md`. When digitized from a figure, label it "approx., read from Fig N".
- Results from OUR tiny toy models (trained in this project) must be labelled as toy demonstrations ("toy model trained for this explainer"), never as paper results.
- The planning / MPC material is an **extension beyond the paper** (energy minimization as planning); label it so.
- The "loss landscape" (training loss over parameters θ) and the "energy landscape" (energy over predictions ŷ for a fixed context and fixed θ) are different things. Always make clear which is shown.

## Visual identity (shared by video and site)

One deliberate dark visual world ("night lab"). Use these tokens exactly.

```
--bg:      #0d131f   page / video background (blue-black)
--panel:   #141c2b   raised surface
--panel2:  #1a2437   second level
--rule:    #283449   hairlines, grids
--ink:     #e9eef6   primary text
--muted:   #93a1b8   secondary text
--faint:   #5d6b82   tertiary, axis ticks
Model families:
--ar:      #8ea7ff   autoregressive Transformer / Transformer++ (periwinkle)
--rnn:     #c99bff   RNN (lilac)
--diff:    #ff8f7a   Diffusion / DiT (coral)
--ebt:     #ffc95c   EBT (amber, "energy")
--truth:   #74e0c0   ground-truth targets / data (mint)
--bad:     #ff6b81   errors / high loss callouts
Energy colormap (low → high energy), interpolate in this order:
  #0b1a33, #15466b, #2b7a8a, #6fae84, #e3b75a, #f7e6bd
```

Fonts (local files in `assets/fonts/`): display = **Source Serif 4** (`SourceSerif4.ttf`, variable weight; italic file too), body = **Atkinson Hyperlegible** (Regular/Bold/Italic), mono/data = **JetBrains Mono** (variable). Load with `@font-face` (site) or `FontFace` (video engine does this for you).

Writing style for on-screen text: short, plain, active sentences. No em-dashes. No emoji. Math in plain Unicode (ŷ, ∇, θ, α, σ, η) is fine.

## File layout

```
ebt-explainer/
  index.html                 site shell (lead owns)
  serve.sh, README.md        (lead owns)
  assets/css/base.css        tokens + shared components (lead owns)
  assets/js/lib.js           shared site helpers (lead owns): colormap, MLP fwd/grad, plotting
  assets/js/sections/<id>.js one file per site section (section agents own their file)
  data/<name>.json           experiment / digitized / sample data (producer agents own)
  data/bundle.js             generated by `python3 src/bundle_data.py` from data/*.json -> window.EBT_DATA[name]
  media/samples/...          real dataset images (sample agent owns)
  media/paper/figNN*.png     paper figure crops (figure agent owns)
  media/toy/...              images produced by toy experiments
  video/engine.js            video engine + helpers (lead owns)
  video/player.html          scrub/play/chapters; ?render=1 for frame capture (lead owns)
  video/scenes/sNN_<id>.js   one file per scene (scene agents own)
  video/music.wav            background music (music agent owns)
  src/                       python + node scripts (owner = whoever creates the file)
  notes/paper_facts.md       structured facts with page refs (paper agent owns)
```

Never edit a file another agent owns. If you need something from the lead's files, note it in your final report.

## Data contracts (producers must write these; consumers read them via `window.EBT_DATA[name]`)

Every JSON file has a top-level `"_meta": {"source": "...", "note": "..."}` describing provenance (paper figure / toy experiment / dataset). Keep each JSON under ~3 MB; round floats to 4 significant digits.

- `toy2d.json`: 2D toy EBT (prediction ŷ ∈ R²). Must include: `arch` (layer sizes, activation, how x and ŷ are combined), `weights` per checkpoint (list of `{step, layers:[{W:[[...]], b:[...]}]}`) so the browser can recompute energy and ∇_ŷE exactly, `contexts` (list of example x values with labels), `grids` (for a few contexts × checkpoints: energy on a 64×64 grid over `extent`), `trajectories` (GD paths from several random starts, per step energy), `loss_curve`, `param_landscape` (2D slice of the TRAINING LOSS over θ along two random filter-normalized directions, 41×41, for the final model), `bon` example, `hparams`.
- `text.json`: tiny char-level EBT + same-size feed-forward baseline. `vocab`, `corpus_name`, example contexts with per-thinking-step top-k distributions and energies, easy-vs-hard token energy traces (like paper Fig 8), `thinking_curve` (val loss/perplexity vs number of thinking steps and vs BoN M, for EBT; flat line for baseline), `train_curves` for both, `hparams`.
- `image.json` + `media/toy/denoise_*.png`: tiny denoising EBT vs a same-size one-shot or diffusion-style baseline on real small images: PSNR vs forward passes, per-step image strips (paths to PNGs), σ settings used.
- `scaling.json`: digitized series from paper Figs 4, 5, 6, 7, 9, 12 (+ fitted slopes) with axis labels/units; every series flagged `"approx": true` and the figure number.
- `samples.json` + `media/samples/`: real RedPajama-V2 text snippets (with GPT-NeoX token splits if available), downstream eval examples (GSM8K, SQuAD, BigBench Dyck, BigBench elementary math QA), real images (COCO or ImageNet-subset) with clean / σ=0.1 / σ=0.2 noisy versions made with the paper's linear β schedule (1e-4→2e-2, 1000 steps, σ = fraction of schedule), and any video-frame examples.
- `paper_figs.json`: list of `{id, file, caption, page}` for cropped paper figures in `media/paper/`.

## Video engine contract (for scene agents)

See `video/engine.js` (written by the lead) and `video/scenes/_example.js`. Each scene file calls `EBTV.scene({id, title, dur, draw(ctx, t, U)})` where `t` is local seconds in `[0, dur)` and `U` holds helpers (colors, fonts, easing, text with wrapping, arrows, axes and line plots, heatmaps, contour lines, 3D surface, images, data). `draw` must be a **pure function of t** (deterministic: use `U.rng(seed)`, never Math.random or Date), fully repaint the 1920×1080 frame, and stay fast (< 25 ms per frame; cache expensive images with `U.cache`). Test by opening `video/player.html?scene=<id>` and by rendering stills with `node src/still.js <sceneId> <t1> <t2> ...` (writes PNGs to `video/frames/`), then LOOK at the PNGs (Read tool) and fix layout problems: overlap, clipping, text too small (minimum 26 px body text at 1080p, captions ≥ 30 px), anything off-canvas, empty frames.

Scene pacing: a viewer needs ~1 s per 4 words of on-screen text plus 1.5 s; never more than ~2 text blocks visible at once; build up diagrams step by step; every scene opens with a small eyebrow + title, and ends with a 0.6 s fade.
