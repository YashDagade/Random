# Video storyboard (target 600 s, 9 scenes)

General: no narration, so text carries the story. One idea per beat. Build diagrams step by step. Keep at most two text blocks on screen. Every number from the paper gets a small `U.badge(ctx,'paper','Fig N')`; every toy result gets `U.badge(ctx,'toy')`; planning material gets `U.badge(ctx,'ext')`. Use model colors: AR/Transformer++ `C.ar`, RNN `C.rnn`, Diffusion/DiT `C.diff`, EBT `C.ebt`, ground truth `C.truth`. Energy plots use `U.cmap`. Check `notes/paper_facts.md` for every claim.

## s01_intro — "Thinking, fast and slow, for machines" (50 s)
- 0–7 s: Title card. "Energy-Based Transformers" (display, large) / "are Scalable Learners and Thinkers" / authors line + "arXiv 2507.02092 · 2025". Behind it, a slowly drifting faint contour field of an energy landscape (deterministic, cheap).
- 7–19 s: Two columns. System 1: fast, intuitive, automatic, e.g. choosing lunch: one quick pass. System 2: slow, deliberate, analytical, e.g. changing careers: loop of checking and refining. (Paper p.1–3 uses the lunch vs career example.) Animate a single arrow for System 1 and a looping arrow for System 2.
- 19–33 s: Today's System 2 recipes and their limits (p.1–2): RL with verifiable rewards works only where answers can be checked (math, code); extra verifiers or reward models need extra supervision; most are text-only. Show three cards that get a soft "limited" stamp. Then the core question, large serif: "Can System 2 Thinking emerge from unsupervised learning alone?"
- 33–46 s: The paper's answer: "Learn to verify. Then think by optimizing against the verifier." Mini diagram: context x and candidate ŷ → EBT → one number, the energy. An arrow loops back: ŷ ← ŷ − α∇ŷE. Then three headline result chips (badge paper): "Up to 35% higher scaling rate than Transformer++", "Thinking improves language perplexity by up to 29% more", "Beats DiT image denoising with 99% fewer forward passes".
- 46–50 s: Roadmap: 8 small chips (Model families, Energy, Data, Thinking, Training, Evaluation, Scaling, Takeaways). Fade.

## s02_families — "Four ways to make a prediction" (75 s)
- 0–6 s: Header. Prompt to complete: "The quick brown ___" shown at top.
- 6–40 s: Four lanes, revealed one at a time (each ~8 s), each with a small animated mechanism and a compute counter "forward passes for this token: N":
  1. AR Transformer (C.ar): context → stack of layers → softmax → "fox". Exactly one pass. Fixed depth = fixed compute per token.
  2. RNN (C.rnn): a hidden state h is updated once per incoming token. Thinking longer is not possible: the state only changes when new input arrives (p.2, Sec 6.2).
  3. Diffusion Transformer (C.diff): start from noise, a FIXED schedule of T denoising steps; each step predicts the noise (an amortized gradient of the energy), not an energy. Can use more steps, but cannot score its own guess and usually does not benefit past the trained schedule (p.2, Sec 6.4).
  4. EBT (C.ebt): start from a random guess ŷ0. Each step: forward pass gives an energy; its gradient updates the guess. Energy readout falls step by step. Stops when energy converges. More steps for harder tokens; can compare several candidates by energy.
- 40–60 s: Table 1 builds cell by cell: rows FF Transformers, RNNs, Diffusion Transformers, EBTs × columns Facet 1 Dynamic compute, Facet 2 Uncertainty in continuous spaces, Facet 3 Verification. Only EBT has three checks; Diffusion has Facet 1 only.
- 60–75 s: Takeaway card: "Diffusion is the closest relative: it is iterative, but it learns the gradient of an energy, not the energy itself. EBT learns the energy, so every step comes with a score." (Paper Sec 6.4 / Sec E.)

## s03_energy — "Energy: a learned verifier" (75 s)
- 0–14 s: Definition. E_θ(x, ŷ) → one scalar. Low energy = compatible, high = incompatible. p(ŷ | x) ∝ e^(−E): an unnormalized likelihood, no partition function needed (Sec 3.1). Example bars: context "The cat sat on the", candidates "mat" (low), "moon" (high), "the" (high). Label as an illustration.
- 14–24 s: Verification is easier than generation (Sec 2.1): a maze. Checking a drawn path is easy; finding it is hard. Small maze animation.
- 24–48 s: The energy landscape (toy, badge toy). Use `toy2d.json` final checkpoint for one context: a slowly rotating 3D surface (U.surface) beside its 2D heatmap + contours with the true y marked (C.truth). A marble starts at a random point and rolls downhill along a stored GD trajectory; small readout "energy" ticking down.
- 48–64 s: NOT the loss landscape. Split screen. Left: "Loss landscape: training loss as a function of the weights θ. Gradient descent here is learning, and it reshapes the energy landscape." (use `toy2d.param_landscape`). Right: "Energy landscape: energy as a function of the prediction ŷ, for one context and fixed θ. Gradient descent here is thinking." Make the distinction unmistakable.
- 64–75 s: Uncertainty (Facet 2). Two contexts from toy2d side by side: a low-noise context with a sharp narrow basin vs a high-noise context with a wide shallow basin (or whatever the toy data shows; describe it honestly).

## s04_data — "What the model actually sees" (60 s)
- 0–20 s: Text. A RedPajama-V2 snippet from `samples.json` types in, then splits into GPT-NeoX token chips (alternating subtle colors, ids underneath). A sliding window highlights context tokens and the next-token target. Captions: "Pretraining: RedPajama-V2, 100B-token sample, GPT-NeoX tokenizer", "66M training / 33K validation samples (p.8)". For EBT the prediction ŷ is a distribution over the vocabulary that gets optimized (check paper_facts for the exact representation).
- 20–40 s: Images. A grid of real COCO images (from `samples.json`), one enlarged: patchify into 16×16 patches at 128×128 (8×8 = 64 patches). Then the noise: clean → σ=0.1 (training and in-distribution test) → σ=0.2 (out-of-distribution test), with "linear β schedule 1e-4 → 2e-2; σ = fraction of the diffusion schedule".
- 40–52 s: Video. Something-Something V2, next-frame prediction; frames 224×224 → frozen SD-XL VAE → 3136-dim latent per frame; show the paper's Fig 11 crop (badge paper) as the example.
- 52–60 s: Evaluation sets preview: four small cards from `samples.json` (GSM8K, SQuAD, BigBench Elementary Math QA, BigBench Dyck Languages) with one example each, "scored by perplexity, not accuracy".

## s05_thinking — "Thinking = gradient descent on the prediction" (80 s)
- 0–10 s: Algorithm 2 (Inference with verification) as pseudocode, lines lighting up in sync with the animation that follows.
- 10–32 s: Toy 2D (badge toy): heatmap + contours of one context; 6 random starts ŷ0 ~ N(0, I) appear as dots, then step along stored trajectories; a side plot shows energy vs step for each; the truth marker is revealed at the end.
- 32–50 s: Toy text (badge toy, from `text.json`): a context string; a bar chart of the top-8 next-character probabilities morphs step by step from a random distribution to a confident one; an energy-per-step mini plot. Explain "ŷ is a whole probability distribution, refined by gradient steps".
- 50–62 s: Two thinking tools from the paper. Thinking longer: more steps. Self-verification: optimize M candidates (Langevin noise makes them explore differently), keep the one with the lowest energy (Best-of-N, per prediction, no external verifier).
- 62–80 s: Uncertainty emerges. Left: the paper's Fig 8 crop (badge paper): easy tokens like "the", ".", "is" reach low energy fast; hard tokens like "quick", "brown", "problem" stay high. Right: our toy's energy-per-step heatmap over a sentence (badge toy), easy vs hard characters. Closing line: "Energy that will not settle is a signal to think longer."

## s06_training — "Training: learn to verify by learning to optimize" (70 s)
- 0–10 s: Algorithm 1 pseudocode, lines lighting up.
- 10–30 s: Unrolled computation graph: ŷ0 (noise) → [EBT: E, ∇] → ŷ1 → [EBT] → ŷ2 → loss J(ŷ2, y) (cross-entropy for text, MSE for images). Forward arrows draw left to right in C.ebt; then backward arrows flow right to left through every step back to θ in C.bad: "the loss is backpropagated through the whole optimization: gradients of gradients, computed with Hessian-vector products at linear cost (Sec 3.2)".
- 30–50 s: What training does to the landscape (badge toy): the toy2d heatmap for one context morphs across the stored checkpoints (step 0 → final) while the training loss curve draws in sync; marble trajectories at each checkpoint get better.
- 50–62 s: Why this scales: no negative samples needed (contrastive EBMs suffer the curse of dimensionality); optimizing toward the target makes the landscape convex around the truth. GAN analogy: forward pass = discriminator (verify), optimization = generator.
- 62–70 s: Three landscape regularizers (Sec 3.3): replay buffer, Langevin dynamics, randomized step size and number of steps. Table 2 bars (badge paper, Table 2): thinking longer / longer + self-verification for each ablation; full config 7.19 / 18.7; no random step size −1.47 / 0.19.

## s07_eval — "How the paper measures success" (75 s)
- 0–10 s: Two questions. Learning scalability: how fast does the model fit the pretraining data as we scale data, batch, depth, parameters, FLOPs, width? Thinking scalability: how does performance change with more forward passes per prediction (NFE)?
- 10–28 s: Language. Perplexity = exp(average negative log-likelihood); lower is better. Table 3 bars (badge paper): pretraining 31.36 (T++) vs 33.43 (EBT) but downstream GSM8K 49.6 vs 43.3, SQuAD 52.3 vs 53.1, BB Math QA 79.8 vs 72.6, Dyck 131.5 vs 125.3. Message: worse pretraining perplexity, better on most downstream tasks.
- 28–45 s: Thinking on OOD data: digitized Fig 6a (perplexity improvement vs forward passes: T++ flat, EBT rises to ~29% with thinking longer + self-verification) and Fig 7 (improvement grows roughly linearly with OOD shift). Badge paper, approx.
- 45–65 s: Images. Table 4 (PSNR, MSE at σ 0.1 and 0.2), Fig 12 (PSNR vs forward passes, EBT reaches DiT's level with ~1% of the passes), the Fig 10 crop of OOD denoising. Then our toy denoising strip (badge toy) animating step by step.
- 65–75 s: Linear probe on ImageNet-1k: top-1 0.31% (DiT) vs 5.32% (EBT), top-5 1.36% vs 13.2% ("about 10×"). Note small absolute numbers: these are small models, the gap is the point.

## s08_scaling — "Scaling laws: who improves faster?" (70 s)
- 0–14 s: What a scaling law is: loss vs resource, both log scale → roughly a straight line; slope = scaling rate. Two lines with different slopes cross: the steeper one wins at scale.
- 14–46 s: Six axes for language (Figs 4, 5) from `scaling.json` (badge paper, approx.): data, batch size, depth, parameters, FLOPs, width. Cycle through 3–6 of them as small multiples, draw points then fit lines, show each "scaling rate gain" number. Mention "up to 35% higher scaling rate" (abstract).
- 46–56 s: Video (Fig 9): EBT starts with higher loss but scales >33% faster with width and parameters; dashed extrapolation suggests a crossing (label "extrapolation, not a result").
- 56–64 s: Thinking scales with training too: Fig 6b, the gain from BoN self-verification grows from 4–8% to 10–14% as data grows.
- 64–70 s: Honest caveats from paper_facts: small scale (state the paper's model/FLOP limits), each EBT training step costs more compute (state the factor from paper_facts), extra hyperparameters.

## s09_wrap — "What to take away" (45 s)
- 0–16 s: Why EBTs are better, as four short lines with icons: one model is both generator and verifier; thinking works in any modality without rewards; energy gives uncertainty in continuous spaces; scales faster in data and depth so far.
- 16–28 s: Limitations (paper): list 3–4 exactly as in paper_facts.
- 28–41 s: Beyond the paper (badge ext): energy minimization as planning. A 2D world: start, goal, obstacles. A whole plan (sequence of actions) is a prediction ŷ; a learned energy scores how compatible the plan is with the goal and the world; gradient steps bend the plan around obstacles; MPC executes the first action and re-plans. One line connects to JEPA world models: "a JEPA-style predictor in latent space plus an energy verifier would let a world model plan and check its own futures."
- 41–45 s: End card: "Explore the labs: index.html". Fade to bg.
