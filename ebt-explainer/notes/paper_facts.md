# EBT paper facts: single source of truth

Paper: **"Energy-Based Transformers are Scalable Learners and Thinkers"**, Alexi Gladstone, Ganesh Nanduru, Md Mofijul Islam, Peixuan Han, Hyeonjeong Ha, Aman Chadha, Yilun Du, Heng Ji, Jundong Li, Tariq Iqbal (UVA, UIUC, Amazon GenAI, Stanford, Harvard). arXiv:2507.02092v1 [cs.LG], 2 Jul 2025. "Preprint. Under review." Project page energy-based-transformers.github.io, code github.com/alexiglad/EBT (p.1).

How to read this file:
- `(p.N)` = PDF page N. The printed page numbers match the PDF page numbers. Source text: `notes/paper_text.txt`.
- Quotes in "double quotes" are verbatim from the paper (line breaks and hyphenation removed). Figure titles, axis labels and legend text in double quotes come from the figure images, so they are not in `paper_text.txt`. Phrases in 'single quotes' are our own wording (phrasings to avoid, suggested labels).
- **[approx., read from Fig N]** = a value read off a plot by eye (by the paper agent, from 220–300 dpi renders). It is not a number printed in the paper.
- **[derived]** = arithmetic or reasoning by us from paper numbers. It is not stated in the paper.
- **[not in paper]** = context or interpretation added for the explainer.
- **[unspecified]** = the paper does not say. Do not invent a value.
- The per-axis scaling percentages (35.98, 28.46, 5.29, 2.91, 2.92, 0.02, 33.66, 34.28, 35.69, 51.70, 3.30) are exact numbers printed in the **figure titles** (Figs 4, 5, 9, B.3, C.1). They live inside the figure images, so they are absent from `paper_text.txt`. They were transcribed from 260 dpi renders and are not approximations.

---

## 0. Numbers at a glance (all verified against the text)

| Claim | Exact value and context | Page |
|---|---|---|
| Learning scaling rate | "up to 35% higher scaling rate" vs Transformer++. The largest value is data scaling, **35.98%** (Fig 4a title). Others: batch 28.46%, depth 5.29%, params 2.91%, FLOPs 2.92%, embed dim 0.02% (text) | p.1, p.4, p.9–10 |
| Thinking gain (text) | "improve performance by as much as 29% by increasing the amount of forward passes (thinking time), whereas the Transformer++ cannot improve performance at all" | p.10 (Fig 6a), p.1, p.4, p.15 |
| Video scaling | "more than 33% faster rate" (Fig 9a 33.66%, Fig 9b 34.28%) | p.12–13 |
| Image denoising compute | "99% fewer forward passes" = EBT 3 forward passes vs DiT 300 | p.4, p.13, p.14, p.15, p.35 |
| Linear probe | "around 10× higher accuracy". Top-1 5.32% vs 0.31% (that is 17.2× [derived]); top-5 13.2% vs 1.36% (9.7× [derived]) | p.13, p.14, p.15 |
| PSNR gap | "by as much as 3.5 in Peak Signal to Noise Ratio (PSNR)" (Table 4 actual OOD gap 23.29 − 19.56 = 3.73 [derived]; the in-distribution gap is only 27.25 − 26.58 = 0.67 [derived]) | p.14 |
| Verification grows with training | BoN-5 gain "increasing up to 10%−14% from 4%−8%" (Dyck only) | p.10, p.34 |
| Table 2 full config | Thinking Longer **7.19**; Thinking Longer + Self-Verification **18.7** (percent perplexity improvement, OOD BigBench Dyck) | p.10 |
| Table 3 | Pretrain ppl T++ 31.36 vs EBT 33.43 (EBT worse); GSM8K 49.6 vs 43.3; SQuAD 52.3 vs 53.1 (EBT worse); BB Math QA 79.8 vs 72.6; BB Dyck 131.5 vs 125.3 | p.12 |
| Table 4 | DiT PSNR 26.58 / MSE 142.98 (σ=0.1), 19.56 / 718.7 (σ=0.2); EBT 27.25 / 122.55, 23.29 / 305.2; top-1 0.31% vs 5.32%; top-5 1.36% vs 13.2% | p.13 |
| Training FLOPs | One AR-EBT second-order step ≈ **3.33×** a Transformer++ step; with 2 optimization steps EBTs used **6.66×** the FLOPs of a comparable Transformer++ | p.35–36 |
| Single-step cost | "about a constant 1.66× as expensive as a vanilla feed-forward model given everything else remains constant and you use a single step" | p.41 |
| Largest model | "while EBTs scale well up to 800M parameters" (xl = 708M non-embedding params, Table D.1) | p.17, p.33 |
| Compute ceiling | "unable to train models with more than 10^21 FLOPs (≈1300 A100 GPU Hours)" | p.27, p.35 |
| Data | RedPajamaV2 100B sample, GPT-NeoX tokenizer, 66 million train / 33 thousand validation samples | p.8 |
| Training opt. steps | "we primarily trained EBTs with either two or three optimization steps" | p.26 |

---

## 1. One-paragraph thesis

The paper asks: "Can we rely entirely on unsupervised learning to develop System 2 Thinking?" (p.2). Its answer is yes (p.2). The method is "learning to explicitly verify the compatibility between inputs and candidate-predictions, and then re-framing prediction problems as optimization with respect to this verifier" (p.1). An Energy-Based Transformer (EBT) is a Transformer that outputs one scalar **energy** Eθ(x, ŷ) for a context x and a candidate prediction ŷ. The energy is an "unnormalized probability" and lower means more compatible (p.1, p.4). A prediction starts as random noise and is refined "through gradient descent-based energy minimization until convergence" (p.1). This makes "thinking" an optimization procedure with respect to a learned verifier (p.4, p.17). Because the descent runs during pretraining too, the paper says this "simulates the thinking process during pretraining, unlike with traditional models, enabling each prediction (e.g., a token for LLMs) to have its own thinking process" (p.5). Training backpropagates a normal supervised loss through this unrolled optimization, which needs second-order derivatives (Hessian-vector products) (p.7). The paper reports four main results. (1) During pretraining, EBTs reach "an up to 35% higher scaling rate" than Transformer++ "with respect to data, batch size, parameters, FLOPs, and depth" (p.1). (2) At inference, EBTs improve language performance with extra computation "by 29% more than the Transformer++" (p.1). (3) Bidirectional EBTs beat Diffusion Transformers (DiT) at image denoising "while using fewer forward passes" (p.1) ("99% fewer", p.4). (4) Thinking helps more on data "farther out-of-distribution", and EBTs do better on most downstream tasks "given the same or worse pretraining performance" (p.1). The authors conclude that EBTs are "a promising new paradigm for scaling both the learning and thinking capabilities of models" (p.1).

---

## 2. Three cognitive facets and Table 1

The paper calls these "three key Facets of System 2 Thinking". Footnote 1: "We acknowledge that these are not comprehensive for achieving System 2 Thinking, but rather, a good first step" (p.2).

- **Facet 1: Dynamic Allocation of Computation.** "Humans naturally allocate varying amounts of effort to different tasks depending on difficulty" (p.3). Example: "a decision regarding whether to change careers generally takes people much more time than deciding what to eat for lunch" (p.3). Footnote 3: this is meant "at the granularity of each prediction being made". Current LLMs built with AR transformers or RNNs "cannot dynamically allocate compute per token generated as they have a finite depth/width and are only updated with new text tokens" (p.3). Appendix F formalizes Facet 1 as Turing completeness, assuming an infinite external memory (p.38).
- **Facet 2: Modeling Uncertainty in Continuous State Spaces.** LLMs can express uncertainty through token-level probabilities. In continuous spaces such as vision, standard Transformers, RNNs and diffusion models "generally do not provide strong or reliable uncertainty estimates" unless they use Vector Quantization or pseudo-objectives such as ELBO (p.3). "EBMs can naturally model uncertainty without having to model exact likelihoods by modeling the relative unnormalized likelihoods of predictions" (p.3). Example: "when a pedestrian might emerge from behind a parked vehicle" (p.3). Footnote 4 concedes that Mixture Density Networks and score-based diffusion models can also express uncertainty, but have "seen less widespread success and scalability" (p.3).
- **Facet 3: Verification of Predictions.** Verification "can guide decisions about when to stop thinking or to select the most accurate predictions" (p.3). "verifying solutions is exponentially easier than generating solutions [48]" (p.3). An explicit verifier enables "early stopping", "allocating more compute when a problem is difficult", "Monte Carlo Tree Search" or "sampling many times and choosing the best prediction" (p.3).
- Appendix F adds **Facet 4: Compositional Reasoning and Systematicity**. EBM energies "are composable in several different manners", which enables "zero-shot generation of novel combinations without retraining" (p.38).

**Table 1: Architectures and Cognitive Facets (p.3), verbatim**

| Architecture | Dynamic Compute Allocation (Facet 1) | Modeling Uncertainty (Facet 2) | Prediction Verification (Facet 3) |
|---|---|---|---|
| FF Transformers | ✗ | ✗ | ✗ |
| RNNs | ✗ | ✗ | ✗ |
| Diffusion Transformers | ✓ | ✗ | ✗ |
| EBTs | ✓ | ✓ | ✓ |

Table 1 caption: "For each prediction, Feed Forward (FF) Transformers and RNNs generally² have a finite amount of computation. While diffusion models have potentially more computation during inference by increasing the number of denoising steps, they do not learn to explicitly verify or estimate uncertainty for predictions. EBMs can use a dynamic amount of computation during inference by iterating for any number of steps, and give an energy scalar that can be used to evaluate uncertainty and verify the strength of predictions." (p.3). Footnote 2 says recent works enable per-prediction dynamic compute [25, 29, 30], "but these approaches are generally not modality agnostic and have not been widely adopted", and that RNNs "are typically updated only with new state information, except in specialized cases such as in [25]" (p.3).

Why existing approaches fall short (p.2):
- RL-trained reasoning models (O1, R1, Grok3, Claude 3.7 Sonnet) work mainly "in domains where rule-based rewards can easily verify answers, such as math and coding". The approach "often deteriorates performance in other tasks such as writing", and RL "may not induce new reasoning patterns" (p.2).
- Diffusion models "typically fail to benefit from denoising steps beyond what they were trained on [19]" and "require an external verifier to improve System 2 Thinking capabilities" (p.2).
- "most modern RNNs only update their internal state with new information, meaning they cannot be used for thinking longer". RNNs with recurrent depth "still lack mechanisms for explicit verification [25]" (p.2).

---

## 3. EBM background

- **Definition.** EBMs "learn an energy function that assigns a scalar value to each input configuration, with lower energy indicating higher compatibility or likelihood between input variables" (p.5). "the energy function acts as a verifier of input data coherence" (p.5).
- **Boltzmann distribution.** For probabilistic EBMs: pθ(x) = e^(−Eθ(x)) / Z(θ), where Z(θ) = ∫ e^(−Eθ(x)) dx is "the intractable partition function involving an integral over all possible values of x" (p.6). "Due to the negative exponential, lower energy corresponds to higher probability" (p.6).
- **Unnormalized EBMs.** "it is common to work with unnormalized EBMs, which dispense of the partition function in favor of representing relative unnormalized probabilities" (p.6). The goal becomes "simply assigning low energy to the true data manifold and high energy elsewhere" (p.6). This helps where "the true data manifold is thin and therefore a probabilistic EBMs would have an infinite score [42]" (p.6). In supervised or predictive self-supervised learning: pθ(x, ŷ) ∝ e^(−Eθ(x, ŷ)) (p.6). Footnote 5: in self-supervised learning "x is some unmasked portion of the original x and ŷ is the masked portion" (p.6).
- **Energy = unnormalized NLL.** "the energy E is essentially the (unnormalized) negative log-likelihood up to an additive constant. The term compatibility is just a term used for intuition." (FAQ, p.41).
- **Why not maximum likelihood.** Data lie on "a very thin manifold". MLE "would push EBMs to have low energy on the true data manifold and then infinite energy off that manifold". That "would make the score (gradient of the energy function) undefined and the energy landscape untraversable" (p.41).
- **Relative energies are enough.** "for most real-world applications, you only ever need sample relative likelihood comparison". Reward models "can be seen as EBMs (just multiplying the reward by −1)" (p.41).
- **Two training families: contrastive vs regularized** (p.6, p.40). "Contrastive methods increase the energy of negative samples while decreasing the energy of positive samples" (p.6). Negatives can come from a GAN generator ("amortize negative sample generation") or from running MCMC (p.40).
- **Curse of dimensionality.** "where the volume of spaces grows exponentially with their dimension, contrastive methods struggle to scale because they must increase the energy of an exponentially higher number of negative samples [42]" (p.6–7).
- **Optimization-based training (the paper's choice).** "frame EBM learning as an optimization problem [48, 71], which avoids the curse of dimensionality by implicitly regularizing the energy landscape" (p.7). "This pushes the energy landscape to be convex surrounding the ground truth solution, thereby regularizing the energy landscape to only have low energy on the true data manifold" (p.7). The appendix says "locally convex" (p.40). The paper notes that this training "has strong resemblance to Langevin Dynamics" (p.40) and "can be seen as being similar to denoising score matching [154]" (p.40, p.38).
- **Explicit vs implicit EBMs** (p.41). Explicit EBMs "directly map all variables (inputs) to a single scalar energy as the output of the neural network"; EBTs are explicit. In implicit EBMs the energy is only implied. Example, diffusion: ∇x E(x,t) = −sθ(x,t), so E(x,t) = −∫ sθ(u,t) du + C(t). "Other notable examples of implicit EBMs include Hopfield Networks [158], RNNs [25], and Boltzmann machines." (p.41).
- **Why EBMs were hard to scale.** "EBMs have traditionally struggled with scalability [51], with zero publicly known Foundation EBMs", because of "training stability" and "long training times" (p.4). Transformers bring "parallelizability", "stability" and "scalability" (p.8).
- **Verifier intuitions** (Sec 2.1–2.2, p.5–6). Maze example: "verifying the correctness of a given path is significantly easier than discovering such a path" (p.5). Verifiers generalize better: a maze verifier "more easily transfers to larger mazes" (p.6). EBMs combine verifier and generator "into a single model, where the generator is defined implicitly by the gradient of the verifier [51]" (p.6). "The Generative AI Paradox": generative models "frequently lack basic discrimination skills". Because EBMs learn a verifier, "they develop strong discrimination skills [71]" (p.6).

---

## 4. Training algorithm in full detail

### 4.1 Algorithm 1 (p.7), verbatim structure

```
Algorithm 1: Training
Inputs: Context x, Target y, EBM Eθ(x, ŷ)
Hparams: Steps N, Step Size α, Loss J(·)
1 Sample ŷ0 ∼ N(0, I);
2 for i = 0, ..., N−1 do
3     ŷ_{i+1} ← ŷ_i − α ∇_{ŷ_i} Eθ(x, ŷ_i);
4 L ← J(ŷ_N, y);
5 return L, update Eθ;
```

Equation (1): ŷ_{i+1} = ŷ_i − α ∇_{ŷ_i} Eθ(x, ŷ_i), "where α is the step size" (p.7).

Setup in words: "We begin with an EBM Eθ, a prior (initial prediction) ŷ0, an input (context) for the model x, and seek to predict y. We aim to find the minimum energy (most compatible/likely) ŷ given an x, which we search for using gradient descent" (p.7).

**GAN intuition** (p.7): "During the forward pass, EBMs can be seen as a GAN discriminator by giving an energy 'verification'; on the backward pass they can be seen a GAN generator by optimizing predictions through energy minimization to try and fool the discriminator."

**The energy is never directly supervised.** "EBMs only receive supervision at the end of the optimization process" (Fig E.1 caption, p.37). The energy values are shaped only indirectly, through the loss on the optimized prediction. The paper says energies are learned "implicitly as described in Section 3.1" (p.41).

### 4.2 Loss functions per modality

- Language modeling: "categorical cross-entropy" (p.7). Listing 1 uses `nn.CrossEntropyLoss(..., ignore_index=pad)` applied to the predicted distributions vs `next_tokens` (p.43–44).
- Image denoising (bidirectional): "mean squared error" (p.7).
- Video next-frame prediction: "Smooth L1 loss with β = 1.0" (p.12). Listing 2 uses `torch.nn.SmoothL1Loss(beta=1.0)` (p.44).
- "we use the same loss functions as existing papers to simplify experiments" (p.7).

### 4.3 Second-order backprop and Hessian-vector products

- "Importantly, this loss is backpropagated through the entire optimization process, requiring second-order derivatives (i.e., gradients of gradients). These are computed efficiently via Hessian-vector products, which scale linearly with model size [76], similar to standard first-order backpropagation in feed-forward models" (p.7).
- In code, the gradient of the energy w.r.t. the prediction is taken with `torch.autograd.grad(predicted_energies.sum(), predicted_distributions, create_graph=True)`. `create_graph=True` keeps the graph so the outer loss can differentiate through the update (p.44).
- "Does training EBMs require a full Hessian calculation? No—the approach described in the paper only requires Hessian-vector products." (p.41)
- [derived] Why second order: ŷ_N depends on θ through each ∇_ŷ Eθ. So dL/dθ involves ∂(∇_ŷ E)/∂θ, a mixed second derivative, applied to a vector. Without detaching, it also involves ∂²E/∂ŷ² (Hessian-vector products through earlier steps).
- [derived] Why `.sum()` over positions is valid: each position's energy depends only on its own prediction plus observed context, because predictions are made independently (footnote 11, p.31). So the gradient of the summed energy gives each position's own gradient.

### 4.4 Detach / stop-gradient / truncation details (S1 vs S2 EBTs, Sec C.2, p.30; I.2–I.3, p.43–45)

The paper uses two variants of EBT:

| | S1-EBT ("System 1") | S2-EBT ("System 2") |
|---|---|---|
| Purpose | "hyperparameters specifically optimized for stability and learning convergence" (p.30) | "hyperparameters optimized for System 2 Thinking capabilities" (p.30) |
| Detach between steps | "S1 models have the gradient of predictions detached between optimization steps to increase training stability" (p.30) | "following [67], the S2 models truncate backpropagation and avoid detaching prediction tensors between optimization steps" (p.30) |
| Where loss is applied | "the loss is evaluated at every iteration without gradient truncation" (p.36) | "the gradient is truncated, the loss is only calculated at the last step following [48]" (p.36) |
| Energy landscape regularization | none (p.30) | all of Sec 3.3: replay buffer, Langevin dynamics, random step size, random number of steps (p.30) |
| Step size α | learnable (p.42; Table D.3 ✓) | "not be learned"; needs "a different value for the optimization step size" and "a minimum number of optimization steps greater than one" (p.30). Table D.4: learnable ✗ |
| Step conditioning | "learnable 'step' embedding" prepended to sequences, mapping step index to an embedding (p.42). S1 uses "multiple time steps" (p.45) | "all optimization steps are performed along the same energy landscape (same time embedding condition)" (p.45) |
| Used for | "Many of the pretraining scaling experiments ... are with S1 models" (p.30) | thinking experiments (Table D.4, p.36) |

- Guidance (p.43): "Not detaching predictions in between steps allows for more 'Thinking Time' before making predictions, but makes the gradient computation graph longer and therefore increases the likelihood of stability issues". "For S2 models, we found that not detaching between steps was best, and similarly that calculating the loss only at the last step was best. For S1 models, we found the opposite to be most stable." "if one is calculating the loss only at the last step, then one should not detach between steps".
- Fig C.1 (p.31): S2 models "have a higher Y-intercept but scale slightly faster than System 1 models". The title reads "EBT-S2 Scales 3.30% Faster Than EBT-S1".
- [unspecified] The exact truncation scheme for S2, i.e. how many steps receive gradient, is not given in the text ("we refer the reader to the source code", p.43).
- The listings are simplified: "The pseudocode is primarily for S2 models without any energy landscape regularization techniques" (p.43). Their comments say the loss is computed "optionally at each optimization step or only at the end" (p.44–45), so the listings cover both the S1 and the S2 loss placement.
- I.3's recipe for turning an S1 model into an S2 model (p.45): remove detaching and compute the loss only at the final step; tune α but "do not make it learnable"; tune the number of steps (with a minimum and maximum when randomizing); "Then add a replay buffer, Langevin Dynamics, and eventually a randomized alpha (step size)".

### 4.5 How the prediction ŷ is represented

- **Text:** ŷ is a **vector of logits over the vocabulary**, one per position, of shape `B, S, V` with V = 50,277 (Table D.3, p.35). It is initialized as `torch.randn_like(...)`, "initialize predictions as random" (p.43). At each step the logits are softmax-normalized and projected to embedding space: `predicted_embeddings = self.vocab_to_embed(softmax(predicted_distributions)) # B, S, D; need to proj. to embed space for transformer to work in, use linear layer, weighted sum, etc` (p.44). Fig 2 draws this as a "Linear Projector" (p.4). Gradient descent updates the logits. Cross-entropy is computed on the optimized logits vs the true next tokens: at the last step only (S2), or at every step (S1) (p.43–44). Caveat: the `B, S, V` shape comes from the Listing 1 comments. The code line `torch.randn_like(next_embeddings)` would literally give a D-dim tensor, so the pseudocode is loose here [derived]. Fig 3 caption: "the model predicts a distribution over text tokens, progressively shifting from an initial random distribution toward the target distribution" (p.5).
  - "Normalize Input Distribution ✓" for NLP (Table D.3 p.35, Table D.4 p.36). Normalizing was "crucial": without softmax, runs "often had extreme activations as well as large loss spikes" (p.43).
- **Video:** ŷ is a predicted **embedding** of the next frame, shape `B, S, D`, initialized `torch.randn_like(next_embeddings)` (p.44). Frames are 224×224 and "encode[d] ... into 3136 dimensional features with the frozen SD-XL VAE" (p.12). [derived] 3136 = 4 × 28 × 28 (4-channel latent at 8× downsampling). Reference [115] for the VAE is the URL huggingface.co/stabilityai/sd-vae-ft-mse (p.23).
- **Images (denoising):** the prediction is the denoised image, scored with MSE (p.7). [unspecified] Whether ŷ0 for denoising is random noise or the noisy input image. Fig 10 shows EBT outputs after 1, 2, 3 steps (p.14).
- **Initial prediction ŷ0:** "Sample ŷ0 ∼ N(0, I)" in Alg 1 and Alg 2 (p.7). "Starting from an initial prediction, such as random noise" (p.5). H.1: "In practice, ŷ is often initialized as random" (p.40).
- **Energy head:** `predicted_energies = self.transformer(all_embeddings) # B, S, 1; this returns only energies for the predicted_embeddings` (p.44). Each predicted position gets one scalar energy. [unspecified] The exact head (pooling or linear layer) that maps the final hidden state to a scalar.

### 4.6 Autoregressive (decoder-only) EBT and the information-leakage issue (Sec 3.4 p.8; C.3 p.30–32; C.4 p.33)

- Two variants: "a decoder-only EBT inspired by the GPT architecture [55], which parallelizes all predictions simultaneously; and a bidirectional EBT" (p.4, p.8).
- The problem as stated: "the autoregressive EBT presents greater implementation challenges, primarily due to the potential for information leakage in naïve implementations" (p.8).
- [derived explanation of the leak; the paper names the risk but does not spell it out] Suppose you concatenate observed tokens and predictions and apply an ordinary causal mask. Then a prediction for position j+1 could attend to the observed token z_{j+1}, which is exactly its target, or to later observed tokens and other predictions. The C.3 design prevents this.
- **Sequence layout:** for context length N, the first N elements are "zo, or the observed sequence representations, and the final N elements as zp, or the predicted representations (for each next element in the sequence)". "the manipulated tensors within the EBT are of size B × 2N × D" (p.31). Listing 1 makes the same point: `all_embeddings = torch.cat((context_embeddings, predicted_embeddings), dim=1) # B, 2S, D` (p.44).
- **Desired attention (Eq. 3, p.31):** an N × (N+1) score matrix. Row i holds scores to observed z1..zi plus one superdiagonal entry to its own prediction ẑ_{i+1}. "each ẑj along the superdiagonal is unique for its row, as we chose for each prediction to be made independently from each other prediction" (p.31). Because of this, it "cannot be computed with a matrix multiplication (softmax(QKᵀ/√dk))" (p.31).
- Footnote 11: predictions could be made dependent on each other, "however, not making every prediction independent would mean that there is stochasticity for each prediction caused not only by its initial value but also by all initial values of all previous predictions" (p.31).
- **Efficient implementation (C.3, p.31–32):**
  1. Observed states use ordinary causal attention, Attention(Qo,Ko,Vo) = softmax(QoKoᵀ/√dk)Vo, "independent of the representations of the predictions" (p.31–32).
  2. Predicted states compute S̃_{zo←zp} = Qp Koᵀ/√dk (p.32).
  3. Append a column to make the matrix N × (N+1). Zero the superdiagonal by elementwise multiplication with a mask, which keeps it differentiable. Compute self-scores S̃_{zp←zp} = sum(Qp ∗ Kp)/√dk, a Hadamard product summed over features. Write them onto the superdiagonal (p.32).
  4. Apply softmax with an adjusted causal mask, so "all future information is masked out and the superdiagonal we added is not masked out" (p.32).
  5. Clone and extract the superdiagonal, zero it, and drop the extra column. Then zp = S_zp · Vo, plus the Hadamard product of Vp with the extracted diagonal. Finally zo and zp both go through the shared output matrix Wo (p.32).
- Footnote 12: "In practice, we shared the weights for the Q/K/V matrices for both observations and predictions to enable a one-to-one comparison to existing feed-forward transformers" (p.32).
- **Simplified alternative (C.4, p.33):** full attention matrices with a generalized causal mask [28]. It costs "4 times the number of FLOPs as normal attention, which is around double the number of FLOPs of our more efficient implementation".
- **Backbone:** the Llama 2 transformer implementation, used for Transformer++ and as the base for causal EBTs (p.33). The best architecture was "the standard Llama2 [87] architecture and initialization ... RMSNorm, Xavier init [159], SwiGLU MLP [160], and RoPE [161]" (p.42). The authors tried DiT adaptive layer norm, but "the timestep embedding worked better" (p.42). The step embedding "significantly improved scalability and stability, especially for S1 models" (p.42).

### 4.7 Bidirectional EBT

- "a bidirectional EBT with bidirectional attention across sequences, enabling capabilities such as infilling and masked modeling [26, 56]" (p.8). "the bidirectional EBT implementation is relatively straightforward" (p.8).
- Built on the DiT implementation of [26]: "for the bidirectional EBT we build upon this implementation" (p.33). Also: "We based our bidirectional EBT implementation on the code from this repository" (p.35).
- Footnote 6: "Autoregressive and Bidirectional refer to the procedure for generation. It's worth noting that autoregressive models are compatible with bidirectional attention" (p.8).

### 4.8 Stability tricks (Appendix I)

- The step size α "was one of the primary factors affecting the stability of EBTs" (p.42). "a smaller step size results in larger generated gradients, whereas a larger step size results in smaller gradients". It "needs to be tuned per modality". The step size is never weight-decayed (p.42).
- "a relatively high optimization step size was necessary (30,000 for video and between 5 and 500 for text). Without a high optimization step size, gradient magnitudes continued to increase throughout training" (p.42).
- The learnable step size (S1) uses "the model's learning rate by the step size learning rate multiplier". "we recommend keeping its LR multiplier around 3x the value of alpha" (p.42–43). [derived] Table D.3 follows that rule exactly: 90,000 = 3 × 30,000 (CV) and 1,500 = 3 × 500 (NLP).
- Step embedding rationale: "We believe this helped improve stability by enabling the accumulation of attention mass, as well as enabling less steep energy landscapes conditioned on the optimization step" (p.42).
- "using a lower number of optimization steps resulted in more stability, as using more optimization steps necessitates longer gradient chains" (p.42). The authors did not try randomizing the step count per sample: "It's possible that randomizing the number of optimization steps would yield similar results in reducing gradient variance, however, we did not experiment with such a configuration" (p.42; see also p.45). [derived, unverified] So the 2–3 random step count in Table D.4 is probably drawn once per batch, while α is drawn per batch and sequence element.
- Sensitivity: the hyperparameters "are extremely important and can often be highly sensitive towards performance" (p.43). Recommended order: start with S1-EBTs, which are "the easiest and most stable variant", then move "gradually" toward S2 settings (p.43).
- Randomize α "differently for every batch and sequence element". A single α per batch "resulted in issues with training convergence", likely through "high-variance gradients" (p.42).
- Clamping the prediction gradients can help, but "We do not conduct any experiments in the paper with clamped prediction gradients" (p.42).
- High batch size helps "because you initialize predictions from random noise this makes gradients noisier" (p.43).
- Signals to monitor: "gradient magnitudes" and "the gap between the initial and final energy after optimization" (p.43).

---

## 5. Energy landscape regularization (Sec 3.3, p.7–8; Table D.4 p.36)

Motivation: "Because y is high-dimensional, the energy landscape spans a high-dimensional space, and must remain well-shaped throughout." The authors "found three key energy landscape regularization techniques to be essential in ensuring the smoothness and convexity of learned energy landscapes" (p.7).

1. **Replay buffer.** "following existing EBM works [48, 51]". It "helps simulate longer optimization trajectories, enabling energy landscapes to be well defined near their minimum" (p.7). [unspecified] Buffer size, sampling probability, what is stored. [derived] Presumably it stores past optimized predictions and restarts optimization from them.
2. **Langevin dynamics** ("a variant of", p.7). Equation (2): ŷ_{i+1} = ŷ_i − α ∇_{ŷ_i} Eθ(x, ŷ_i) + η_i, with η_i ∼ N(0, σ), "where σ is the magnitude of the noise η". "Without this random noise term, exploration is often limited to paths leading directly to the energy minimum, leaving other regions poorly defined" (p.7).
3. **Randomized step size α and randomized number of optimization steps.** "varying the paths taken towards predicting solutions ... significantly improved generalization" (p.7).

**Exact S2 hyperparameters (Table D.4, p.36, "Hyperparameters for EBT System 2 Thinking experiments", NLP)**, verbatim:

| Hyperparameter | NLP |
|---|---|
| Optimizer | AdamW |
| Optimizer Momentum (β1, β2) | 0.9, 0.999 |
| Model Size | Small |
| Learning Rate (LR) | 0.0012 |
| Learning Rate (LR) Schedule | Linear warm-up + cosine decay |
| Warmup Steps | 1 × 10^4 |
| Minimum LR Scale | 10 |
| Gradient Clip Value | 1 |
| Weight Decay | 0.01 |
| Context Length | 256 |
| Tokenizer | EleutherAI/gpt-neox-20b [148] |
| Vocab Size | 50,277 |
| Normalize Input Distribution | ✓ |
| Optimization Steps | 2-3 (randomized) |
| Optimization Step Size | 5 |
| Optimization Step Size Multiplicative Random Factor | 2 |
| Langevin Dynamics Noise | 3 |
| Learnable Optimization Step Size | ✗ |
| No Detach Between Optimization Steps | ✓ |
| Truncate Optimization | ✓ |
| Replay Buffer | ✓ |

[unspecified] The exact distribution behind "Multiplicative Random Factor 2", for example whether α is drawn from [α/2, 2α] and whether uniformly or log-uniformly. Also unspecified: whether Langevin noise "3" is a std or a variance, and whether it is used at inference.

Note: "Model Size: Small" in Table D.4 conflicts with D.1.2, which says "We train xxs S2-EBT and Transformer++ models". LR 0.0012 is the xxs learning rate (p.34). See section 12.

**Table 2: System 2 Thinking Ablations (p.10), verbatim.** "percent perplexity improvement" on OOD BigBench Dyck Languages. ↑ = higher is better.

| Model | Thinking Longer ↑ | Thinking Longer and Self-Verification ↑ |
|---|---|---|
| No Random Step Size | -1.47 | 0.19 |
| No Random Num. Steps | 0.00 | 9.65 |
| No Langevin Dynamics | **17.2** | 17.0 |
| No Replay Buffer | 14.8 | 17.8 |
| **Full System 2 Configuration** | 7.19 | **18.7** |

(In the paper the bold values are 17.2, the best in the Thinking Longer column, and 18.7, plus the row label "Full System 2 Configuration".) [unspecified] The reference point for "percent perplexity improvement" (Table 2 caption), i.e. which no-thinking setting each row is compared against.

- Caption, key lines: "Bolded highlights the default System 2 Hyperparameters ... This configuration results in the best performance when thinking longer and doing self-verification. Removing regularization, such as Langevin Dynamics, results in less energy landscape exploration, which improves single path performance (thinking longer) at the expense of self-verification performance." (p.10)
- Text: "randomizing the step size is critical—removing it nearly eliminates Thinking gains, while disabling Langevin Dynamics degrades combined performance but improves results without verification, offering a performance-compute tradeoff" (p.10).
- [derived] Without a random number of steps, thinking longer gives exactly 0.00 gain. This suggests the model only uses extra steps if it saw varied step counts in training.

---

## 6. Inference: thinking longer and self-verification

- Two thinking modes (p.7–8, p.9):
  - **Thinking Longer** (Facet 1): "changing the number of steps taken for optimization of a single prediction. This is conceptually similar to increasing the denoising steps performed with a diffusion model" (p.8). "thinking longer, which denotes more optimization steps" (p.9).
  - **Self-Verification** (Facet 3): "we generate N predictions from an EBM and then choose the minimum energy prediction. This is conceptually similar to Best of N (BoN) sampling using language models [77]. However, EBMs generalize this approach to both discrete and continuous modalities, don't require additional supervision (e.g., an external reward/verification model), and perform it on every single prediction, not just to entire sequences" (p.8).
- **Algorithm 2: Inference with Verification (p.7), verbatim structure:**

```
Inputs: Context x, EBM Eθ(x, ŷ)
Hparams: Steps N, Step Size α, Samples M
1 for j = 1, ..., M do
2     Sample ŷ_{0,j} ∼ N(0, I);
3     for i = 0, ..., N−1 do
4         ŷ_{i+1,j} ← ŷ_{i,j} − α ∇_{ŷ_{i,j}} Eθ(x, ŷ_{i,j});
5 return ŷ* = argmin_j Eθ(x, ŷ_{N,j});
```

  Alg 2 has no noise term. [unspecified] Whether Langevin noise is added at inference.
- **Stopping rule:** "This gradient descent update is done iteratively to refine the prediction until convergence of the predicted energy" (Fig 2 caption, p.4). "This scalar's convergence allows the model to determine whether the prediction is adequate or if further thinking is necessary" (Fig 3 caption, p.5). [unspecified] The paper reports fixed step counts per experiment and gives no adaptive early-stopping experiment.
- **NFE definition:** "the Number of Function Evaluations (NFEs) [19, 91], which we deem to be one for every forward pass completed (for EBMs this is one function evaluation per optimization step)" (p.8). [derived] An EBT optimization step also needs a backward pass to get ∇_ŷ E, so one NFE costs more than one plain forward pass.
- **System 2 Thinking definition (Definition C.1, p.30):** "Given a problem with data x, a model θ, and additional computational resources in the form of function evaluations F greater than the minimum number of function evaluations to get a valid prediction from the model F0, System Two Thinking STT(·) quantifies the expected percentage improvement in performance as F increases":
  STT(x, θ, F) = E_x[ P(x, θ, F) / P(x, θ, F0) − 1 ]. "This formalization is compatible with any type of metric (e.g., Accuracy, Perplexity, FID, etc)" (p.30).
- **Steps at train vs test:**
  - Training: "we primarily trained EBTs with either two or three optimization steps"; "because of challenges in stability when training with more steps ... we were unable to successfully increase past two or three steps" (p.26). S1 scaling: 2 steps for both CV and NLP (Table D.3, p.35). S2 NLP: "2-3 (randomized)" (Table D.4, p.36).
  - Text inference (Fig 6a, p.11): x-axis "Number of Forward Passes" with points at 2, 3, 6, 15, 30. EBT No Thinking sits at 2, Thinking Longer at 3, Self-Verification at 6, 15, 30. [derived, not stated] This fits 2 training-length steps, then 3 steps, then BoN over M = 2, 5, 10 candidates × 3 steps.
  - Fig 6b: BoN-5. Fig B.1a: BoN-10 vs BoN-2 (p.11, p.28).
  - Uncertainty heatmaps (Figs 8, 11, B.2) show "Iteration" 0 to 11, i.e. 12 columns (p.12, p.14, p.29). [unspecified] Whether column 0 is the energy of the initial random guess before any update. If it is, the plots show 11 optimization steps. Either way, this is far more steps than the 2–3 used in training.
  - Image denoising: 1, 2, 3 EBT steps vs 100, 200, 300 DiT steps. "For both DiTs and EBTs, we found that models performed best on OOD noise levels when denoising their own outputs twice, that is applying the model to denoise the image three times recursively. This is how we are able to achieve the results in Figure 12 demonstrating the performance for 300 forward passes from DiTs and 3 forward passes from EBTs" (p.35). Footnote 13: "This recursive application was only performed during testing due to the distribution shift" (p.35). [derived, unverified] Fig 10 labels the EBT columns "1 step, 2 steps, 3 steps", and the text speaks of three recursive applications. Together these suggest each recursive application is a single EBT optimization step (one NFE). The paper does not say how many optimization steps EBT uses per application, or whether the in-distribution (σ = 0.1) column of Table 4 also uses 3 vs 300 passes.

---

## 7. Experiments

General (p.8): "All models are pretrained from scratch under a tightly constrained compute budget, as the architecture of EBTs is incompatible with existing foundation models, making them incompatible for adaptation via fine-tuning." There are two kinds of results: "learning scalability, investigating how quickly models can fit the pretraining data" and "thinking scalability, or how the performance of models changes as we scale the System 2 Thinking of models" (p.8). Seed: "We seed all libraries using PyTorch Lightning [147] for all experiments with a seed of 33" (p.33).

### 7.1 Model sizes (Table D.1, p.33), verbatim. "For applicable model sizes we follow Mamba [22]."

| Size | Non-Embedding Params | # layers | embed. dim | # heads |
|---|---|---|---|---|
| xxs | 6.18M | 6 | 384 | 6 |
| xs | 12.4M | 12 | 384 | 6 |
| small | 48.8M | 12 | 768 | 12 |
| medium | 176M | 24 | 1024 | 16 |
| large | 396M | 24 | 1536 | 16 |
| xl | 708M | 24 | 2048 | 32 |

xxs and xs are "two extra model sizes" added "because we are compute-constrained" (p.33). "Because the hyperparameters used in these experiments were tuned for feed-forward Transformers in [85], we broadly expect that hyperparameters tuned for EBTs will further increase the performance gap" (p.33).

### 7.2 Transformer++ hyperparameters (Table D.2, p.33), verbatim

| Hyperparameter | CV | NLP |
|---|---|---|
| Optimizer | AdamW (both) | |
| Optimizer Momentum | β1, β2 = 0.9, 0.999 (both) | |
| LR Schedule | Linear warm up cosine decay (both) | |
| Warmup steps | 1e4 (both) | |
| Minimum LR Scale | 10 (both) | |
| Gradient Clip Value | 1 (both) | |
| Weight Decay | 0.01 (both) | |
| Context Length | 16 | 256 |
| Encoder | SD-XL VAE [114, 115] | - |
| Image Dimension | 224x224 | - |
| Tokenizer | - | EleutherAI/gpt-neox-20b [148] |
| Vocab Size | - | 50277 |

### 7.3 EBT scaling hyperparameters (Table D.3, p.35), verbatim. Rows identical to D.2 (optimizer through vocab size) plus:

| Hyperparameter | CV | NLP |
|---|---|---|
| Normalize Input Distribution | - | ✓ |
| Optimization Steps | 2 | 2 |
| Optimization Step Size | 30,000 | 500 |
| Optimization Step Size LR Multiplier | 90,000 | 1,500 |
| Learnable Optimization Step Size | ✓ (single check centered across both columns) | |

### 7.4 Autoregressive language modeling (Sec 4.1, p.8–12; D.1, p.33–34)

- **Data:** "All language models are pretrained on the RedPajamaV2 text corpus [92, 93] 100B sample from HuggingFace using the GPT-NeoX tokenizer [94] (following [22]) to predict the next token" (p.8). "We manually created a training and validation split of 66 million and 33 thousand samples, respectively" (p.8). Context length 256 (Table D.2/D.3). In NLP "models see each sample only once due to the dataset size" (p.12).
- **Baseline:** "the standard Transformer++ recipe [22, 87, 95]" (p.8), implemented with Llama 2 (p.33).
- **Downstream evaluation (p.9):** "Ordered roughly by increasing perplexity difficulty, these include GSM8K [97], SQuAD [98], BigBench Elementary Math QA [99], and BigBench Dyck Languages [99]". "We intentionally design the evaluation towards reasoning benchmarks due to their alignment with System 2 Thinking." The metric is perplexity: "we focus on reporting perplexity as our relatively small models trained from scratch ... do not achieve high accuracies on many of these benchmarks. Furthermore, perplexity often functions as a more linear metric than accuracy [22, 100]" (p.9).
- **What the eval protocol leaves out (important for 'how is eval done').** The paper does not specify the following [unspecified]:
  - how downstream perplexity is computed: which tokens are scored (question + answer or the answer only), prompt formatting, or which split of each benchmark is used;
  - how the "best checkpoints" (p.34) were selected;
  - how an EBT's perplexity is computed from its optimized prediction. [derived, unverified] Most likely it is exp(cross-entropy of the softmax of the final optimized logits vs the true next token), mirroring the training loss in Listing 1;
  - any free-running text generation or decoding rule (argmax or sampling from the optimized distribution). All text results are next-token perplexities with the true context given (teacher-forced), and no generated text samples are shown. Video results are likewise one-step next-frame validation losses. No multi-frame rollouts are reported.
- **No error bars.** Every experiment uses one seed ("a seed of 33", p.33). No figure or table reports multiple runs, error bars or confidence intervals [derived from a full-text search].
- **Scaling methodology (D.1.1, p.33–34):** The authors "conduct normal scaling experiments over all of these factors at the exact same time (as is standard), as well as ablating over changing just one of these factors at a time", arguing this "allows for controlling a single independent variable at a time" (p.34).
  - Parameter and FLOP scaling: "all models are pretrained for 105k steps". Batch sizes for xxs, xs, small, medium, large: "32, 46, 90, 170, and 256", chosen "by scaling the batch size with the square root of the number of parameters and then rounding to an even number" (p.34). LRs: 0.0006, 0.0003, 0.00025, 0.0002 for small, medium, large, xl (Mamba), and 0.0012 and 0.0009 for xxs and xs (p.34).
  - Data and batch-size scaling (Fig 4): "we use xxs models". "The data scaling experiment used a batch size of 128 and the batch size experiments ran for 105k steps" (p.34).
  - "All models were trained with a context length of 256 and no FFN multiplier (FFN dimension being equal to the embedding dimension) due to limited resources" (p.34).
- **Thinking experiments (D.1.2, p.34):** "We train xxs S2-EBT and Transformer++ models with the same setup as above, with the exception that models are trained with a batch size of 128 for 1M training steps". [derived] 128 × 256 × 1,000,000 ≈ 32.8B tokens, which matches the Fig 6b x-axis range (≈0–33B). "Figures 6, B.1, 7 and Table 3 use the best checkpoints of these two models. Figures 6a, 7 both use all four downstream datasets shown in Table 3. Figure 7 also uses the pretraining dataset in addition to these downstream datasets, where every dataset is represented as a separate OOD point." "Figure 6b was solely from the Dyck Languages benchmark. We did not observe this trend in other benchmarks, possibly due to perplexity not being a completely linear metric. Figure B.1a was on the RedPajamaV2 validation dataset." (p.34)
- **Learning-scalability claims:** "Across all axes EBT consistently out-scales (has a higher scaling rate) the Transformer++ recipe, becoming the first model to achieve such a feat without using a different tokenizer" ([101], Byte Latent Transformer, is the only prior out-scaler, but it uses a different tokenizer and does not out-scale across multiple axes) (p.9). "These results suggest that EBTs are more data efficient, batch size efficient, parameter efficient, depth efficient, and compute efficient than the Transformer++ recipe. Thus, at the scale of modern foundation models trained on 1,000× more data with models 1,000× larger (following [84]), we expect the pretraining performance of EBTs to be significantly better" (p.9).
- **Thinking emerges with data:** "We found that the thinking capabilities of EBT emerge with a sufficiently large data scale, and therefore, due to limited resources, we focus on conducting thinking experiments with smaller models trained on substantial amounts of data" (p.9).
- **Fig 6b claim:** "as EBTs are trained for longer, their ability to achieve improvements from verification improves, increasing up to 10%−14% from 4%−8%". Extrapolation: "EBTs trained at the same scale as modern foundation models, such as the 15T tokens Llama3 [84] was trained on (≈1000× the current data scale), would have significantly more substantial results from self-verifying" (p.10).
- **Chain of thought:** footnote 9: "Because we pretrained language models from scratch, and are unable to train models the size of modern foundation models, we find models did not benefit from inference time techniques such as Chain-of-Thought. However, we expect both EBT and Transformer++ models to benefit equally from existing techniques" (p.10).
- **Generalization (Sec 4.1.3, p.11–12):** "We observe a strong linear trend: as the data becomes more OOD, thinking leads to greater performance improvements" (p.11). OOD shift is "measured as the ratio of downstream task perplexity to pretraining perplexity" (p.11). For Table 3 the authors "compare models with identical training setups, where EBTs have slightly worse pretraining perplexity than Transformer++ models" (p.11).

**Table 3: Language Model Task Generalization Comparison (p.12), verbatim** (perplexity, ↓ lower is better; bold in paper marks the better value):

| Model | Pretrain | GSM8K ↓ | SQuAD ↓ | BB Math QA ↓ | BB Dyck ↓ |
|---|---|---|---|---|---|
| Transformer++ | **31.36** | 49.6 | **52.3** | 79.8 | 131.5 |
| EBT | 33.43 | **43.3** | 53.1 | **72.6** | **125.3** |

Caption: "Despite exhibiting a slightly higher pretraining perplexity, EBTs usually achieve lower perplexity on downstream tasks than the Transformer++. This suggests that EBTs generalize better than the Transformer++. Additionally, because EBTs scale better than the Transformer++ during pretraining (Figure 4), these findings suggest that EBTs would outperform Transformer++ at foundation model scale. BB stands for BigBench." (p.12). [derived] Relative downstream differences: GSM8K −12.7%, SQuAD +1.5% (worse), BB Math QA −9.0%, BB Dyck −4.7%. Pretrain is +6.6% (worse). [unspecified] Whether Table 3 EBT numbers use thinking (extra steps or BoN).
[derived] Table 3, Fig 6a and Fig 7 cannot be reconciled from the printed numbers. Fig 6a's "Perplexity Increase" (T++ ≈38.4, EBT ≈44.8 to ≈31.8) does not match the mean of (downstream − pretrain) from Table 3 (T++ ≈46.9, EBT ≈40.1) or any obvious ratio. Fig 7's x-values (≈1.31, 1.77, 2.32, 4.37) do not match the Table 3 ratios either: EBT gives 1.30, 1.59, 2.17, 3.75 and T++ gives 1.58, 1.67, 2.54, 4.19. So do not mix numbers across these three exhibits, and do not label Fig 7 points with dataset names.

**Uncertainty (p.10–12, Fig 8):** "for easier to predict tokens, such as 'the' or 'but', EBTs optimize to lower energies faster, whereas for harder to predict tokens, such as 'fox' or 'problem' EBTs have higher energy that does not converge across steps. This suggests that during pretraining EBTs learn to capture uncertainty regarding which tokens are harder or easier to predict, achieving Facet 2" (p.10–11).

**Appendix B.1 (p.28):**
- BoN-10 vs BoN-2: "when models are trained on less tokens, there is little performance improvement by verifying 10 samples instead of just 2. In fact, verifying 10 samples occasionally leads to worse performance than verifying 2 samples, likely because the EBT found an adversarial sample (a sample with low energy that is in fact not a good prediction). However, as data scale increases we observe that performance improvements from BoN-10 versus BoN-2 increase, and that these adversarial dynamics decrease" (p.28).
- Epistemic vs aleatoric (Fig B.2): "for a more in-distribution sequence, EBTs have lower energy (less uncertainty), than for an OOD sequence. This suggests that EBTs learn to know what they don't know" (p.28).
- FineWeb larger-scale data scaling (Fig B.3): "training small-sized models with a batch size of 256, a context length of 1024, and for 500,000 training steps ... EBTs still continue to outscale the Transformer++ by as much as 35% in scaling rate" (p.28). [derived] 256 × 1024 × 500,000 ≈ 131B tokens, which matches the x-axis range. "Small-sized" most likely means the Table D.1 "small" config (48.8M non-embedding), since p.34 says "we also report small models in Section B" [derived].
- Failure case (B.2, p.29): small-scale text-to-image generation on COCO. "EBTs did not learn to generate high-quality novel images. Instead, EBTs often generated blurred images similar to the training distribution." The cause: "the training approach pushing the energy landscape to be convex surrounding the training examples (modes). Therefore, when there are many different modes within the same region (same condition), this convex energy landscape 'merges' to one landscape averaged around the different modes, resulting in blurriness." The authors add: "We believe that this is not a fundamental limitation of EBTs ... such as by adding more conditioning" (p.29).

### 7.5 Autoregressive video (Sec 4.2, p.12–14; D.2, p.34)

- Task: "predict the next image in a video conditioned on all previous frames" (p.12).
- Regime: "current popular video datasets are relatively small, requiring models to train repeatedly on the same data. As a result, this setting probes a different question: 'how well can models fit a fixed dataset?'" (p.12).
- Data: Something Something V2 (SSV2) [116], "standard SSV2 train and validation split" (p.34). Videos processed "with 0.25 seconds between frames" (p.34). Context length 16 frames (Table D.2/D.3). [derived] 16 frames 0.25 s apart cover about 4 s of video. 224 × 224 frames → 3136-d features via frozen SD-XL VAE (p.12).
- Loss and metric: Smooth L1, β = 1.0; "we report the minimum validation loss achieved" (p.12).
- Model sizes from Table D.1. "batch size of 256 for all models". Transformer++ uses the NLP learning rates; "For EBTs, we found that it was necessary to use a lower learning rate by a factor of 3 for proper training stability" (p.34).
- Axes: "we report scaling results for the embedding dimension and non-embedding parameter count, as we found these axes behaved the most linearly" (p.12).
- Result: "despite achieving a higher initial loss, EBTs scale at a more than 33% faster rate than the Transformer++" (p.12). The authors link this to EBTs being able "to express uncertainty (Facet 2) through their energy scalar" (p.12–13).
- Uncertainty (Fig 11): "frames earlier on in the video have higher energy (higher uncertainty) due to no large objects being within the frame, and then as the major object in the scene becomes revealed more EBT predicts lower energy (lower uncertainty)". The standard Transformer++ "would require discretization schemes such as Vector Quantization [117] with a categorical loss" (p.13).

### 7.6 Bidirectional image denoising and linear probe (Sec 4.3, p.13–14; D.3, p.34–35)

- Purpose: "These experiments allow for a fairer comparison with diffusion models, which are not commonly trained autoregressively" (p.13).
- Noise: "we follow [26], and use a linear variance schedule ranging from 1 × 10^−4 to 2 × 10^−2. To control the noise level, we use a hyperparameter denoted σ representing the percentage of the diffusion schedule to noise samples; σ was set to 0.1 during training, and 0.2 during testing to test generalization" (p.13). At inference they test σ = 0.1 (in-distribution) and σ = 0.2 ("Out-of-Distribution (OOD) noisier images") (p.13–14).
  - [derived, assuming the DiT default 1000-step schedule (the step count is not stated in the paper) and x_t = √ᾱ_t x0 + √(1−ᾱ_t) ε] σ = 0.1 → t = 100, ᾱ ≈ 0.897, signal scale √ᾱ ≈ 0.947, noise std √(1−ᾱ) ≈ 0.321. σ = 0.2 → t = 200, ᾱ ≈ 0.659, √ᾱ ≈ 0.812, noise std ≈ 0.584 (on images scaled to [−1, 1]).
- Data: "COCO 2014 dataset [119, 120] with 128 by 128 images, a patch size of 16, and the Diffusion Transformer implementation from [26]", using its train/validation split (p.13, p.34). Reference [120] is huggingface.co/datasets/AbdoTW/COCO_2014 (p.23).
- Training: "All models were trained using the large model size described in Table D.1, with a learning rate of 1e−4 for 100,000 steps. For the DiT baseline, we used the same hyperparameters from [26], changing only the batch size to 128 from 256" (p.34–35).
- DiT inference baseline: tried "DDPM, DDIM, increasing the number of diffusion steps at inference, and recursively applying the diffusion model on its own denoised output. Ultimately, we found that the combination of DDIM recursively applied on its own output performed best". Default denoising schedule from the DiT codebase (p.35).
- EBT training: "for image denoising, it was not necessary to train EBTs with the S2 hyperparameters for System 2 Capabilities to emerge, although its possible these would further improve performance" (p.35). [unspecified] Exact EBT optimization step count, α and batch size for this task.
- Linear probe: "linear probe evaluation on ImageNet-1k [122] ... For both models, we take the average of all the final patch tokens, and for DiTs we feed in T = 0" (p.14, p.35).

**Table 4: Image Denoising and Classification Comparison (p.13), verbatim:**

| Model | In-Dist. σ = 0.1 PSNR ↑ | In-Dist. σ = 0.1 MSE Pixel ↓ | OOD σ = 0.2 PSNR ↑ | OOD σ = 0.2 MSE Pixel ↓ | ImageNet-1k Top 1 Acc. ↑ | ImageNet-1k Top 5 Acc. ↑ |
|---|---|---|---|---|---|---|
| DiT | 26.58 | 142.98 | 19.56 | 718.7 | 0.31% | 1.36% |
| EBT | 27.25 | 122.55 | 23.29 | 305.2 | 5.32% | 13.2% |

Caption: "For image denoising, EBTs significantly outperform DiTs [26] in Peak Signal to Noise Ratio (PSNR), as well as MSE, on both in-distribution and Out-Of-Distribution (OOD) data, while using 99% less forward passes. This suggests that EBTs generalize better than DiTs while using less computation. On image classification, EBTs also perform better than DiTs, yielding around 10× higher accuracy, suggesting that EBTs learn better image representations and therefore understand images better than DiTs." (p.13)

[derived check] PSNR = 10·log10(255² / MSE) reproduces every PSNR in Table 4 to 0.01 dB. So "MSE Pixel" is on the 0–255 pixel scale. MSE reductions: −14.3% (σ = 0.1), −57.5% (σ = 0.2). The in-distribution PSNR gap is small (+0.67 dB). The big gap is OOD (+3.73 dB) [derived].

Text claims: EBTs beat DiTs "by as much as 3.5 in Peak Signal to Noise Ratio (PSNR)" (p.14). "EBTs perform better at denoising while using 99% less denoising steps than DiTs, and that the System 2 Thinking scaling rate for EBTs is higher than for DiTs" (p.14). "achieving a Top-1 and Top-5 accuracy around 10× higher than that of DiTs" (p.14).

### 7.7 Every figure (what it shows, axes)

- **Fig 1 (p.2) "Autoregressive Architecture Comparison."** Four block diagrams. (a) AR Transformer: x1…xt → Feed Forward Transformer → x̂_{t+1}. (b) RNN: x1…xt each feed a chain of RNN cells → x̂_{t+1}. (c) Diffusion Transformer: x1…xt plus x̂_{t+1} → Diffusion Transformer → Noise(x̂_{t+1}). (d) EBT: x1…xt plus x̂_{t+1} → Energy Based Transformer → Energy(x̂_{t+1}). A bracket labels (a)–(c) "Existing Autoregressive Approaches". Caption: diffusion transformers "are the most similar to EBT ... but predict the noise rather than the energy. Consequently, diffusion models cannot give unnormalized likelihood estimates at each step of the thinking process, and are not trained as explicit verifiers".
- **Fig 2 (p.4) "EBT for Autoregressive Modeling."** Left panel "Text Pretraining Objective: Predict Next Token". Context "The dog caught the" goes into the Energy Based Transformer → Energy. "Predictions Across Steps" Step 0–4 show vocab distributions (cat … frisbee … ball) sharpening onto "frisbee"; a "Linear Projector" feeds the prediction in. Strip: "Thinking Process: Minimize Energy of Predicted Distribution using Gradient Descent Iteratively", with ball-in-bowl icons descending. Right panel "Video Pretraining Objective: Predict Next Frame". Context is 3 frames of a dog leaping for a frisbee; Step 0 (gray noise) to Step 4 (clear frame). Strip: "Minimize Energy of Predicted Frame using Gradient Descent Iteratively".
- **Fig 3 (p.5) "Thinking Process Visualization."** An illustrative 3D energy landscape for context "The dog caught the ____". The colorbar runs from "High Energy ↑ (Low Compatibility)" (red) to "Low Energy ↓ (High Compatibility)" (blue). A white trajectory from Step 0 to Step 4 descends into a basin marked "Converged ✓", with the label "Continue Thinking Process until Energy Convergence". Distributions on the left sharpen onto "frisbee". "Adapted from [57]" ([57] = Li et al., "Visualizing the loss landscape of neural nets"). This is a schematic, not a plot from a trained model. The caption also gives the intuition for uncertainty: "Uncertainty (Facet 2) can be represented by landscapes that are harder to optimize or by landscapes with many local minima, allowing the model to know when it requires more steps to think (Facet 1)" (p.5). The trajectory is solid from Step 0 to Step 3, then dashed down into a narrow basin at Step 4.
- **Fig 4 (p.9) "Language Learning Scalability—Data, Batch Size, and Depth."** Orange = Transformer++, blue = EBT.
  - (a) Title "EBT Scales 35.98% Faster Than Transformer++". y "Validation Perplexity" (linear), x "Number of Tokens (Billions)" ≈0.5–7, with fitted curves. EBT starts higher (≈74 vs ≈64 at the first point) and crosses below near ≈3B tokens, ending ≈38.8 vs ≈40.0 at ≈7B tokens [approx., read from Fig 4a].
  - (b) Title "EBT Scales 28.46% Faster Than Transformer++". y "Validation Perplexity", x "Batch Size in Tokens (Thousands)", points ≈4, 8, 16, 33, 65K. EBT is worse at 4K and 8K (≈56.3 vs ≈53.2 at 4K). It crosses between 8K and 16K, is slightly lower at 16K (≈42.3 vs ≈43.0), and ends ≈35.9 vs ≈37.3 at ≈65K [approx., read from Fig 4b].
  - (c) Title "EBT Scales 5.29% Faster Than Transformer++". y "Validation Perplexity (log scale)", x "Depth in Transformer Blocks (log scale)", points ≈4, 8, 12, 16. EBT is slightly above T++ at every depth, and the lines converge [approx., read from Fig 4c].
  - Caption: "EBTs out-scale the Transformer++ recipe significantly, indicating improved data efficiency ... the improved depth scaling offers promise for reasoning, where depth is crucial [96]. These results suggest that if these scaling trends persist, EBTs would likely outperform Transformer++ models at foundation model data scale."
- **Fig 5 (p.10) "Language Learning Scalability—Parameters, FLOPs, and Width."**
  - (a) "EBT Scales 2.91% Faster Than Transformer++". y "Validation Perplexity (log scale)", x "Non-Embedding Parameters (M, log scale)", 5 points ≈6, 12, 49, 176, 396M (xxs to large). EBT is above T++ at every size.
  - (b) "EBT Scales 2.92% Faster Than Transformer++". y "Validation Perplexity (log scale)", x "Training FLOPs (log scale)". T++ points span ≈3×10^16 to ≈1.6×10^19; EBT points sit ≈6–7× further right, spanning ≈2×10^17 to ≈1×10^20. At every measured FLOP budget EBT perplexity is higher [approx., read from Fig 5b]. [derived] T++ xxs: 6 × 6.18M × (32 × 256 × 105k tokens) ≈ 3.2×10^16, which matches the first orange point. EBT ≈ 6.66× that.
  - (c) "EBT Scales 0.02% Faster Than Transformer++". y "Validation Perplexity (log scale)", x "Embedding Dimension (log scale)", 4 points spanning roughly 170–350 [approx.]. These widths are not listed in Table D.1. The lines nearly coincide. EBT is about equal at the two ends and slightly *lower* (better) at the two middle widths, e.g. ≈51.5 vs ≈52.3 near 300 [approx., read from Fig 5c].
  - Caption: "EBTs slightly out-scale the Transformer++ in FLOP and parameter scaling, becoming the first approach to achieve a higher scaling rate without modifying the tokenizer [101] to our knowledge."
- **Fig 6 (p.11) "EBT Thinking Analysis."**
  - (a) "OOD Thinking Performance Comparison". Title "Performance Drop on Out of Distribution Data". y "Perplexity Increase (Lower is Better ↓)", x "Number of Forward Passes" (ticks 2, 3, 6, 15, 30). Series: Transformer++ is a flat line ≈38.4. EBT No Thinking ≈44.8 at 2; EBT Thinking Longer ≈35.9 at 3; EBT Self-Verification ≈32.5 at 6, ≈31.9 at 15, ≈31.8 at 30 [approx., read from Fig 6a]. [derived] (44.8 − 31.8)/44.8 ≈ 29%, which matches the "as much as 29%" claim. Caption: "Mean performance degradation of the standard Transformer++ recipe [87] and the Energy-Based Transformer (EBT) on four Out of Distribution (OOD) datasets. While the Transformer++ cannot reduce perplexity at a per-token level, EBTs can by performing more forward passes over a single token/sample (Thinking Longer) as well as generating many samples and choosing the minimum energy one (Self-Verifying/BoN in addition to longer thought)." [unspecified] The exact formula for "Perplexity Increase".
  - (b) "Verification Capabilities as Scale Increases". Title "Self-Verification Capabilities Scale During Training". y "% PPL Improvement Over No Verification" (≈4–14), x "Tokens Trained On (B)" (0–≈33). BoN-5 scatter with a linear fit from ≈7.9% to ≈10.6% [approx., read from Fig 6b]. Dyck only (p.34). The scatter is very noisy. The text's "4%−8% → 10%−14%" describes the spread of the early and late points, not the fit. Early points run ≈4.4–12; late points run ≈5.7–13.9, including one ≈5.7% point near 27B tokens [approx., read from Fig 6b].
- **Fig 7 (p.11) "OOD Thinking Performance."** Title "Thinking Helps More on OOD Data". y "% PPL Improvement By Thinking", x "OOD Magnitude Shift". Series "EBT With Max Thinking". 5 points ≈(1.0, 11.8), (1.31, 12.4), (1.77, 16.3), (2.32, 16.1), (4.37, 23.0) with a linear fit [approx., read from Fig 7]. The figure does not label which point is which dataset. The shift = 1.0 point is the pretraining dataset (by definition of the ratio, p.11, p.34). "Max Thinking denotes combining thinking longer and self-verification" (p.11).
- **Fig 8 (p.12) "Learning Uncertainty on Text Results."** Two heatmaps titled "Token Energies Across Thinking Steps". y "Token", x "Iteration" (0–11), colorbar "Normalized Energy" 0.0–1.0 (dark purple = high, yellow = low). Left sequence: "The quick brown fox jumps over the lazy dog ." Right: "System 2 Thinking is a challenging but interesting research problem ." Every token has its highest energy at iteration 0, but the iteration-0 values already differ: ≈0.45–0.55 for "The", "is", "but", "." and ≈1.0 for "quick", "jumps", "research", "problem". All tokens drop sharply at iteration 1, then stay roughly flat with small fluctuations through iteration 11. Almost all of the change happens in the first step. Hard tokens ("quick", "brown", "fox", "research", "problem") plateau higher (teal, ≈0.35–0.5). Easy tokens (".", "is", "a", "but", "the") plateau near yellow (≈0.05–0.2) [approx., read from Fig 8]. Caption hard-token list: "quick", "brown", "research", "problem". "fox" comes from the main text (p.10). Caption: "EBTs learn to vary uncertainty across text tokens without any explicit supervision ... Inspired by [25]." [unspecified] The normalization procedure.
- **Fig 9 (p.13) "Video Learning Scalability—Width and Parameters."**
  - (a) "EBT Scales 33.66% Faster Than Transformer++". y "Loss (log scale)", x "Embedding Dimension (log scale)", points at 384, 768, 1024, 1536, 2048. EBT ≈1.88→1.51, T++ ≈1.75→1.48 [approx., read from Fig 9a].
  - (b) "EBT Scales 34.28% Faster Than Transformer++". y "Loss (log scale)", x "Non-Embedding Parameters (M, log scale)", points ≈6, 49, 176, 396, 708M. EBT loss is higher at every size.
  - Caption: "The minimum validation loss achieved on the Something Something V2 (SSV2) dataset. While EBTs achieve higher validation loss than the Transformer++ at smaller scales, the scaling rate is more than 33% higher ... scaling with respect to the embedding dimension behaves more linearly than for the number of parameters, likely due to the embedding dimension serving as a bottleneck for the image representation."
  - [derived check] Log-log slopes read from the 9a fit lines give a ratio ≈1.33. So 'X% faster' ≈ (ratio of fitted log-log slopes − 1). The paper does not define the scaling rate formally.
- **Fig 10 (p.14) "Qualitative OOD Image Denoising."** Two examples: a red double-decker bus, and a street with a parked car and truck. Top row "Energy Based Transformer": 1 step, 2 steps, 3 steps, GT Image. Bottom row "Diffusion Transformer": 100 steps, 200 steps, 300 steps, GT Image. Arrow "Forward Passes". Caption: "EBTs achieve better denoising quality during inference while using one step for every 100 denoising steps of a DiT. The overall image quality of EBT denoised images is less blurry than images denoised by DiT."
- **Fig 11 (p.14) "Learning Uncertainty on Video Results."** Title "Frame Energies Across Thinking Steps". y "Frame" (16 frame thumbnails: brown surface, then a hand placing a blue garment), x "Iteration" (0–11), colorbar "Normalized Energy" 0–1. At iteration 0 the garment frames are the darkest (highest, ≈0.9–1.0). The early frames start at ≈0.65–0.75. After iteration 1 the 4 early near-empty frames stay ≈0.3 (teal-green), while garment frames fall to ≈0.0–0.15 (yellow). The last few frames rise only slightly, to ≈0.1–0.15 [approx., read from Fig 11]. The caption's "When the blue garment is removed from the scene" is hard to see in the thumbnails, which still show the garment in the final frames. Caption: "At the start of the video, uncertainty is high (high energy) because the frame is mostly empty and the scene is highly unpredictable. As a blue garment is placed into the frame, uncertainty decreases (low energy) ... When the blue garment is removed from the scene, uncertainty increases again".
- **Fig 12 (p.15) "Image Denoising Thinking Scalability."** Title "Performance as Forward Passes Increases". y "PSNR (Higher is Better ↑)", x "Number of Forward Passes (log scale)". EBT ≈13.9, 18.8, 23.0 at 1, 2, 3 passes. DiT ≈14.3, 18.9, 19.6 at 100, 200, 300 passes [approx., read from Fig 12]. The 3-pass and 300-pass values match the Table 4 OOD PSNR (23.29, 19.56), so this is the σ = 0.2 setting [derived]. At matched recursion depth, EBT is slightly *below* DiT at 1 vs 100 passes (≈13.9 vs ≈14.3) and about equal at 2 vs 200 (≈18.8 vs ≈18.9). EBT is clearly ahead only at 3 vs 300 (≈23.0 vs ≈19.6) [approx., read from Fig 12]. This matches the caption's "comparable or better". Caption: "EBTs require only 1% of the forward passes used by DiT to achieve comparable or better PSNR. Further, the scaling rate of PSNR improvement given more forward passes is much higher for EBTs than it is for DiTs. These results suggest EBTs have superior thinking capabilities than DiTs on OOD data."
- **Fig B.1 (p.28) "EBT Thinking Analysis for Data Scaling."**
  - (a) "Self-verification with BoN-10 versus BoN-2". Title "Self-Verification Capablities Scale During Training" (sic). y "% Improvement in PPL" (≈−0.2 to 2.6), x "Tokens Trained On (B)" (≈3–33). BoN-10 scatter with a rising fit from ≈1.1 to ≈2.2 [approx.]. RedPajamaV2 validation (p.34).
  - (b) "Results in Fig. 6b projected to Llama3 scale [84]". Title "Self-Verification Capabilities Extrapolated". y "% PPL Improvement Over No Verification" (log), x "Tokens Trained on (log scale)" 10^9–≈1.5×10^13. Series "BoN-5 Samples" and "BoN-5 Samples (extrapolated)". The extrapolated star sits above 1000% (≈1300%) at ≈1.5×10^13 tokens [approx.; a speculative extrapolation]. The curve is fitted to the noisy, roughly flat 1–33B scatter of Fig 6b and bends sharply upward beyond ≈10^11 tokens. A perplexity "improvement" above 100% only makes sense as a ratio (e.g. old/new − 1, in the spirit of Definition C.1), and the paper does not define it [derived]. Present it as the authors' extrapolation, not a result.
- **Fig B.2 (p.29) "Epistemic Uncertainty Comparison."** Title "Energy Comparison for Different Sequences". Sequence 1 "The quick brown fox jumps over the lazy dog ." vs Sequence 2 random tokens "NO $ # ) h v - 3 h -". y "Token", x "Iteration" 0–11, "Normalized Energy". Caption: "the sequence on the left, which is a text sequence likely seen during training, has consistently lower energy (uncertainty) for tokens than the sequence on the right, which is a random text sequence not from the training distribution."
- **Fig B.3 (p.29) "EBT Larger Scale Data Scaling."** FineWeb.
  - (a) "EBT Scales 35.69% Faster Than Transformer++". y "Validation Perplexity", x "Number of Tokens (Billions)" ≈4–130.
  - (b) zoomed in, "EBT Scales 51.70% Faster Than Transformer++", x ≈50–130B. EBT crosses below T++ near ≈90B tokens and ends ≈24.6 vs ≈24.9 [approx., read from Fig B.3b].
- **Fig C.1 (p.31) "EBT S1 and S2 Scaling Comparison."** "EBT-S2 Scales 3.30% Faster Than EBT-S1". y "Validation Perplexity (log scale)", x "Non-Embedding Parameters in Millions (log scale)", 4 points ≈0.9–12.4M. These are tiny models, below xxs except the last. S2 sits above S1 (≈79→49 vs ≈69→42) [approx.].
- **Fig E.1 (p.37) "EBM and Diffusion Comparison."** (a) Diffusion Models, "Learned Denoising Path": update ŷ_{t−1} = (1/√α_t)(ŷ_t − ((1−α_t)/√(1−ᾱ_t)) ε_θ(ŷ_t, x, t)) + σ_t ε, with ε ∼ N(0, I). A jagged path from Initial ŷ0 to Target y, with "Supervision" arrows at many points. (b) Energy Based Models, "Learned Energy Landscape": ŷ_{i+1} = ŷ_i − α ∇_{ŷ_i}(Eθ(x, ŷ_i)). A smooth bowl from ŷ0 down to y, with one "Supervision" arrow at the target. Caption: "Diffusion models receive supervision at each step of the denoising process (e.g., for one thousand steps), whereas EBMs only receive supervision at the end of the optimization process ... Learning landscapes in this manner can reduce 'error' accumulation ... diffusion models ... must follow a set denoising schedule ... EBMs ... can be performed for any number of steps. x here refers to some condition (e.g., a class or text) whereas y is the generated prediction."
- **Fig H.1 (p.40) "Feed-Forward and Energy-Based Model Comparison."** (a) x → Feed Forward Model → ŷ. (b) x and ŷ → Energy Based Model → Energy(x, ŷ). Caption: "Low energy corresponds to high probability, and high energy to low probability. In practice, ŷ is often initialized as random."

### 7.8 The headline claims with exact context

- **35%:** Abstract: "achieving an up to 35% higher scaling rate with respect to data, batch size, parameters, FLOPs, and depth" (p.1). Intro: "up to 35% higher scaling rate than the Transformer++ across several axes, including data, batch size, parameters, FLOPs, and depth" (p.4). The source is data scaling, 35.98% (Fig 4a, p.9). FineWeb gives 35.69% (Fig B.3a): "by as much as 35% in scaling rate" (p.28). Discussion: "especially apparent with data and batch scaling for text, as well as width and parameter scaling for video, where the scaling rate was over 30% higher" (p.14). Note that text batch scaling is 28.46%, under 30%.
- **29%:** Abstract: "EBTs improve performance with System 2 Thinking (i.e., extra computation) by 29% more than the Transformer++ on language tasks" (p.1). Intro: "EBTs can improve language modeling performance 29% more than the Transformer++" (p.4). Results: "EBTs are able to improve performance by as much as 29% by increasing the amount of forward passes (thinking time), whereas the Transformer++ cannot improve performance at all" (p.10). Discussion: "System 2 Thinking could improve performance by as much as 29% for text (Figure 6a)" (p.15).
- **33%:** "EBTs scale at a more than 33% faster rate than the Transformer++" for video (p.12). Fig 9 caption: "the scaling rate is more than 33% higher" (p.13).
- **99%:** "improved performance over DiTs with 99% fewer forward passes" (p.4). Table 4: "while using 99% less forward passes" (p.13). "99% less denoising steps than DiTs" (p.14). Fig 12: "only 1% of the forward passes used by DiT" (p.15). Concretely, 3 EBT forward passes vs 300 DiT forward passes (p.35).
- **10×:** Table 4: "yielding around 10× higher accuracy" (p.13). "around 10× higher than that of DiTs" for Top-1 and Top-5 (p.14). Discussion: "achieving up to a 10× improvement in accuracy" (p.15). Unrelated other uses of 1000× and 1,000×: "trained on 1,000× more data with models 1,000× larger" (p.9); "≈1000× the current data scale" (p.10); "1000× data scale of modern foundation models" (p.28).

---

## 8. Compute and FLOPs accounting (D.5, p.35–36; H.3, p.41; D.4, p.35)

- Transformer++: "the standard estimate of 6N FLOPs per token for the AR Transformer++ [150], where N denotes the number of non-embedding parameters" (p.35).
- EBT per step: "Since an HVP has the same theoretical complexity as a gradient computation, we express the per-step FLOPs as FLOPs = F + B + B", with forward ≈ 2N and backward ≈ 4N per token (p.35).
- Sequence doubling: "In the autoregressive EBT implementation, the effective sequence length becomes twice that of the original Transformer (formally 2S − 2 for an original sequence length S). Owing to the efficient scheme of Section C.3, this doubling of sequence length translates roughly into a two-fold increase in FLOPs, rather than a four-fold increase" (p.35–36).
- Result: "each second-order optimization step demands roughly (F + B + B) × 2 = (2N + 4N + 4N) × 2 = 10N × 2, making it ≈3.33× more expensive than a standard feed-forward Transformer step" (p.36).
- S1 with 2 steps: "For S1 models, where the loss is evaluated at every iteration without gradient truncation, the total FLOPs simply multiply by the number of steps. Therefore, for our pretraining experiments using two optimization steps, we get that EBTs used 6.66× the FLOPs of a comparable Transformer++ during training" (p.36).
- S2: "a random number of optimization steps is used, the gradient is truncated, the loss is only calculated at the last step following [48], and a Replay Buffer is used. Therefore, the FLOP count varies and can both decrease (as truncating uses less FLOPs for earlier steps) as well as increase (as using more steps and a replay buffer both use more FLOPs). These numbers also vary during inference, where the full EBT implementation parallelizing all predictions at once is not necessary" (p.36).
- Caveat: "Given the scarcity of published methods for computing higher-order derivative FLOPs and our inability to leverage existing libraries for Hessian-vector products, these estimates remain approximate. We welcome corrections" (p.36).
- Single-step, no sequence doubling: "training only about a constant 1.66× as expensive as a vanilla feed-forward model given everything else remains constant and you use a single step" (p.41). [derived] 10N/6N ≈ 1.67. The 3.33× figure adds the 2× sequence length of AR EBTs.
- How the FLOP plot accounts for it: the Fig 5b x-axis is "Training FLOPs", and EBT points sit ≈6–7× to the right of the matching T++ points [approx., read from Fig 5b]. This is consistent with the 6.66× multiplier. EBT is worse at every measured FLOP budget; it only has a 2.92% steeper slope. Footnote 8: "The FLOP calculation is nuanced and depends on specific hyperparameters" (p.9).
- Hardware: "All experiments were conducted on either Nvidia A100s, H100s or GH200s, with the largest scale experiment requiring approximately ≈1300 A100 GPU Hours" (p.35). "unable to train models with more than 10^21 FLOPs (≈1300 A100 GPU Hours)" (p.27).
- Inference overhead: "There exist several current real-world use cases, such as low-latency LLM serving, where doing a single forward pass is sufficient, and where the added inference overhead of gradients with EBTs would not be worth the extra computation" (p.27).

---

## 9. Relation to other model families

- **Feed-forward / AR Transformers (Sec 6.1, p.15):** "makes predictions directly in the output space with a single forward pass". With "finite depth and width" they cannot "dynamically allocate more computation to each prediction". They cannot model continuous uncertainty "because the normalization process for continuous state spaces is not as well-defined as it is for discrete spaces using softmax [42]". Improving per-prediction quality "often requires external models [50]" (p.15–16).
- **RNNs and recurrent depth (Sec 6.2, p.16; E.1, p.36):** Modern RNNs (Mamba, RWKV) "scaled similarly to Transformers" with better memory and latency, "However, traditional RNNs that update their internal state based solely on new information/data [22, 23] are not capable of allocating additional computation during inference" (p.16). Recurrent-depth models (Universal Transformer [126], recurrent-depth LLMs [25, 127]) reason "in a continuous latent space through the depth of an unrolled RNN. However, like Diffusion models, these models learn to amortize gradient prediction of the energy function [25], meaning they cannot model uncertainty or explicitly verify predictions. Consequently, EBMs generalize these RNN-based architectures by offering explicit prediction verification capabilities" (p.16). E.1: "Both diffusion models [133] and RNNs [127] can be seen as predicting the score, or the gradient of the energy function/data density, ∇xEθ(x) ... which can be seen as implicit EBMs" (p.36). Future work A.5: "recurrent Energy-Based Models, possibly leveraging the Mamba architecture [22], will eventually become common" (p.27).
- **Diffusion (Sec 6.4, p.16; E.1–E.2, p.36–38; H.1, p.40):**
  - Diffusion "implicitly define a likelihood through the reverse process", but "in practice an external verifier is necessary to improve performance at inference time beyond increasing denoising steps [19–21]" (p.16).
  - "diffusion models rely on a fixed denoising schedule, which restricts their ability to adaptively halt or extend computation—unlike EBMs" (p.16).
  - "diffusion models can be seen as predicting the gradient of the data density/energy function [133], and therefore EBMs are a generalization of diffusion models that learn to explicitly verify predictions" (p.16).
  - Supervision: diffusion is supervised at every step; EBMs only at the end (Fig E.1, p.37).
  - All of E.2's advantages are stated as conditional: "Under the assumption that the energy landscape is well formed and that optimization is well behaved, EBMs offer several distinct advantages over diffusion models" (p.37). The paper does not measure likelihood comparability or error accumulation directly.
  - Likelihoods: "two samples can be directly compared to determine which is more likely, in a single forward pass". Diffusion needs the full reverse process with "ELBOs or numerical solvers", giving "incomparable likelihoods" (p.37).
  - Error accumulation: "any approximation errors at each individual step of the Markov Chain (optimization process) do not result in cumulative error, as the minimum of the energy landscape can still be reached" (p.37).
  - The two are compatible: "[67] combines EBMs and diffusion to reason over challenging problems. This can increase stability of the learned energy landscape by adding explicit score supervision" (p.37).
  - H.1: "This gradient can be seen as the opposite of the noise (e.g., denoising)" (p.40).
  - Denoising score matching: EBT training "can be seen as a form of denoising score matching [71, 154]", but "we find the optimization perspective is more intuitive", and DSM "involv[es] multiple levels of noise rather than just one level" (p.38).
- **GANs (p.6, p.7, p.40):** Forward pass = discriminator, energy minimization = generator "optimizing predictions through energy minimization to try and fool the discriminator" (p.7). Decoupled verifier and generator approaches suffer "adversarial dynamics [19] and challenges in scalability [64]"; EBMs "combine the verifier and generator into a single model" and "this coupling resolves scalability and adversarial issues (Figures 6b and B.1a)" (p.6). Contrastive EBMs resemble GAN discriminators; GANs "use a generator to amortize negative sample generation" (p.40).
- **Dynamic compute in LLMs (Sec 6.3, p.16):** CoT and continuous latent reasoning [29] "don't seamlessly transfer to continuous modalities", and CoT "has been shown to be unreliable". RL reasoning "appl[ies] only to discrete domains", works on "a narrow set of problems that are easily verifiable", and needs "additional supervision, typically in the form of reward signals" (p.16). G.1.2 argues CoT works over "a discrete state space", is "not an intrinsic architectural capability", and gives each token "a fixed computational budget": "a chain is only as strong as its weakest link" (p.39).
- **Prior EBM reasoning work (Sec 6.5, p.16):** "The perspective of energy minimization as thinking/reasoning has been known for some time [134]". The closest works [48, 67] (Du et al., "Learning iterative reasoning through energy minimization/diffusion") "only focus on small-scale problems, and did not scale EBMs to high-dimensional real-world problems such as language or video" (p.16).
- **Other EBMs (E.3, p.38):** The Energy Transformer [152] is "very different". It integrates "Modern Hopfield Networks", whereas "in our work the architecture is non-recurrent and does not use associative memories". E-ARM [153] adds an EBM objective to an AR model but lacks two facets. EBR [140] and Residual EBMs [139] train on top of pretrained LMs with "a contrastive objective, which suffers from the curse of dimensionality" (p.38).
- **EBTs as complements (A.4, p.26–27):** EBTs could serve "as the verifier of predictions initialized by standard feed-forward models", "the System 2 Backbone for helping lighter models that perform System 1 thinking" (p.26–27).
- **MCMC (A.6, p.27):** "The EBM thinking algorithms described have strong connections to or are derived from Markov Chain Monte Carlo (MCMC) sampling". Suggested next steps: "Hamiltonian Monte Carlo [142] or annealed Langevin dynamics [51]"; "we did not explore more advanced search algorithms such as Monte Carlo Tree Search".
- **JEPA:** **not mentioned anywhere in the paper.** The only LeCun-related citations: [42] Dawid & LeCun, "Introduction to latent variable energy-based models: a path toward autonomous machine intelligence" (cited for relative unnormalized likelihoods p.3, unnormalized EBMs and the thin manifold p.6, the curse of dimensionality p.7, softmax normalization p.15, and contrastive vs regularizing training p.40); [72] LeCun, "A path towards autonomous machine intelligence" (cited only for "Two primary training approaches exist—contrastive and regularized methods", p.6); [134] the EBM tutorial; [138] DINO-WM (A.3). [not in paper] For a JEPA researcher: EBTs score (x, ŷ) pairs and optimize ŷ in the model's input space (token-distribution space or VAE-latent space). The paper does not discuss joint-embedding or predictor-in-latent-space architectures. [not in paper] The video EBT does predict in a latent space, but that space is a *frozen*, reconstruction-trained VAE (p.12). It is not a target encoder learned jointly with the predictor, so the representation-collapse problem that JEPA's anti-collapse machinery addresses does not arise in the paper's setup. EBT's regularized (non-contrastive) training (p.6–7) is the closest conceptual link to the regularized-EBM branch of [72].
- **Continual learning / memory mentions:** C.1 says the term "Thinking" fits "domains where the line between inference and training is blurry, such as real-world continual learning, domain adaptation, or actual human learning/thinking processes [146]" (p.30). G.1.1 repeats it: "continual learning where terms such as 'inference-time compute' becomes ambiguous" (p.39). Memory comes up only in the RNN and latency context (p.16, p.27), the Turing-completeness note (p.38), and "does not use associative memories" (p.38).
- **Other future directions (Appendix A):** Reversal Curse: "with EBTs the tokens of A and B are within context during gradient updates due to predictions being made in the input space" (A.1, p.26). Multimodal EBMs: "a single energy scalar to represent the alignment between modalities" (A.7, p.27). "Understanding Predictions": feed-forward models "may not understand their predictions in the same way they understand their inputs", whereas EBTs make "predictions ... in the input space, enabling representations of predictions to be developed" (A.10, p.27–28).

---

## 10. Limitations and conclusion (verbatim-close)

**Limitations (Sec 7, p.17):**
1. "because EBTs generate predictions through an optimization process, they introduce additional hyperparameters, such as the optimization step size and the number of optimization steps. Tuning these hyperparameters is crucial for training stability, as we found poorly chosen values often lead to unstable training."
2. "training and inference are more computationally expensive than standard feed-forward models, requiring additional gradient computations."
3. "while EBTs scale well up to 800M parameters, we have not explored larger models due to resource constraints. However, experimental scaling trends demonstrate that EBTs scale faster than existing paradigms during pretraining, suggesting that EBTs would perform better at foundation-model scale."
4. "EBTs currently struggle with data distributions that have many modes, such as class conditional image generation, likely due to the convex energy landscape assumption made during training."

**Additional limitations elsewhere:**
- Optimization depth: only "two or three optimization steps"; stability blocked more (A.2, p.26).
- Compute ceiling: "more than 10^21 FLOPs (≈1300 A100 GPU Hours)"; "training and thinking with EBTs remains untested at larger foundation model scale" (A.8, p.27).
- Multimodal distributions: "struggle to capture distributions with many modes (e.g., unconditional image generation)" (A.9, p.27). Note that p.17 says "class conditional" and p.27 says "unconditional". Text-to-image failure on COCO (B.2, p.29).
- Thinking needs data scale: "the thinking capabilities of EBT emerge with a sufficiently large data scale" (p.9).
- No CoT benefit at this model size (footnote 9, p.10; p.34).
- Fig 6b trend only on Dyck; "We did not observe this trend in other benchmarks" (p.34).
- Adversarial samples at low data: BoN-10 sometimes worse than BoN-2 (p.28).
- FLOP estimates "remain approximate" (p.36).
- Hyperparameters were tuned for feed-forward Transformers, not EBTs (p.33).
- Hyperparameter sensitivity: EBT hyperparameters "are extremely important and can often be highly sensitive towards performance" (p.43).
- Inference cost: for "low-latency LLM serving ... the added inference overhead of gradients with EBTs would not be worth the extra computation" (p.27).
- Single seed (33) and no error bars anywhere (p.33) [derived from a full-text search]. Several trends rest on few points: Fig 7 has 5 points, Fig 4c has 4 depths, Fig 5c has 4 widths, Fig C.1 has 4 sizes.
- Text evaluation is perplexity only: no accuracies and no generated text. The thinking results come from one small (xxs) model pair (p.9, p.34).
- The advantages over diffusion in E.2 are conditional: "Under the assumption that the energy landscape is well formed and that optimization is well behaved" (p.37).
- "Whether the computational effort spent at inference time fully captures what psychologists term System 2 Thinking is still actively debated"; "our work does not claim current models replicate the full spectrum of human System 2 Thinking" (p.39).

**Conclusion (p.17):** "EBTs are the first instance of an approach that scales at a faster rate than the Transformer++ during pretraining across both continuous and discrete modalities. Additionally, our results suggest that EBTs scale better than existing approaches during inference at System 2 Thinking by dynamically allocating computational resources and self-verifying their own predictions; these System 2 Thinking capabilities enable improved generalization to out-of-distribution data. Ultimately, the superior scaling in both training and inference, coupled with improved generalization, positions EBTs as a promising new paradigm shift for advancing the capabilities of future foundation models."

Opening line of Sec 7 (p.17): EBTs are "a new paradigm that frames System 2 Thinking as an optimization procedure with respect to a learned verifier (an Energy-Based Model), enabling System 2 Thinking to emerge across any problem or modality entirely from unsupervised learning."

**Discussion highlights (Sec 5, p.14–15):** "EBTs appear to be the first approach that has better data efficiency than the Transformer++. As data has become one of the major limiting factors in further scaling [123], this makes EBTs especially appealing" (p.14–15). "EBTs become increasingly robust to self-generated errors during verification (Figure B.1a), indicating that their self-verification process scales reliably and sidesteps the adversarial instabilities reported in prior works [19, 124]" (p.15). "We hypothesize that the superior scaling of EBTs compared to the Transformer++ can be attributed to EBTs learning to verify (Facet 3) rather than solely learning to predict" (p.15).

---

## 11. Planning, world models, control, MPC

I searched the full text for plan, control, MPC, model predictive, world model, action, policy, JEPA and tree search.

- **MPC / model predictive control: ABSENT.** The terms never appear.
- **"Control":** appears only in "To control the noise level" (p.13), "controlled experiments" (p.34), and a reference title "flexible cognitive control" [32]. None of these is about control as in robotics.
- **Planning:** p.6: generative models' lack of discrimination skills "impede their ability to engage in reasoning, planning, and decision-making [9, 70]". Reference titles [136] "Planning with diffusion for flexible behavior synthesis" and [138] "Dino-wm: World models on pre-trained visual features enable zero-shot planning" (both cited only in A.3).
- **World models: one paragraph, A.3 (p.26), verbatim:** "In this work, we focus on autoregressive and bidirectional models over just state information (no actions). EBTs offer high promise in modeling states and actions due to the nature of EBMs learning a distribution over all possible inputs. Particularly, given a model trained to estimate the unnormalized joint distribution of the current context, future, as well as future actions, such world models could implicitly be used as policies to generate actions to achieve a specific state, similar to [136–138]. This would involve holding the current context (past states) constant, and minimizing the energy by propagating the gradient back to the action inputs and future state predictions. Thus, world models trained in this manner become capable of more than just predicting the future, but also in decision making to achieve a specific goal state."
- **Monte Carlo Tree Search:** mentioned as an enabled capability (p.3) and as unexplored future work (p.27).
- **No experiments involve actions, rewards, environments or planning.** Any planning or MPC visualization in our explainer is an **extension beyond the paper**. It is grounded only in the A.3 paragraph above and must be labeled that way.

---

## 12. Easy-to-get-wrong list (for narrators, scene writers, verifiers)

1. **Loss landscape ≠ energy landscape.** The *energy landscape* is E_θ(x, ŷ) as a function of the prediction ŷ, for a fixed context x and fixed weights θ. Thinking descends it at inference. The *loss landscape* is the training loss as a function of the weights θ; the optimizer descends it during training. The paper only ever talks about energy landscapes. "Loss landscape" appears only in the title of [57], which Fig 3's graphic is "Adapted from" (p.5, p.20). Fig 3 is a schematic, not a measured landscape.
2. **Training has two nested optimizations.** The inner loop is gradient descent on ŷ (energy); the outer loop is gradient descent on θ (loss through the inner loop). Do not merge them.
3. **The energy is never directly supervised.** No loss is placed on energy values. Only J(ŷ_N, y) is used (Fig E.1, p.37). Energies are unnormalized and meaningful mainly in relative terms (p.41).
4. **Low energy = good.** Lower energy = more compatible / more likely (p.4–6). Do not write 'maximize energy'.
5. **Language results use perplexity, not accuracy.** Lower is better (p.9). Table 2, Fig 6b, Fig 7 and Fig B.1 report *percent perplexity improvement* (higher is better). Fig 6a reports "Perplexity Increase (Lower is Better ↓)".
6. **EBT pretraining perplexity is worse in Table 3** (33.43 vs 31.36), and EBT is worse on SQuAD (53.1 vs 52.3). EBT wins on GSM8K, BB Math QA and BB Dyck: "most downstream tasks", not all (p.12).
7. **'Scales faster' means a steeper slope, not lower loss.** At the measured scales EBT is *worse* at every point in Fig 4c, Fig 5a, 5b, Fig 9a and 9b. Fig 5c (width, 0.02%) is a wash: equal at the ends, EBT slightly lower at the two middle widths [approx.]. EBT crosses over only in data and batch scaling (Fig 4a/4b, Fig B.3). The 'would outperform at foundation scale' claims are extrapolations (p.9, p.12–13).
8. **"Up to 35%" is the maximum (data, 35.98%)**, not the rate on every axis. Batch 28.46%, depth 5.29%, params 2.91%, FLOPs 2.92%, width 0.02% (Figs 4, 5). The abstract's axis list omits width. The paper never formally defines "scaling rate" [unspecified]; it behaves like the ratio of fitted log-log slopes minus 1 [derived from Fig 9a].
9. **Compute cost:** an EBT training step is ≈3.33× a Transformer++ step per optimization step, and **6.66×** for the 2-step S1 pretraining runs (p.36). The FLOP-scaling plot includes this cost. At equal FLOPs EBT is still worse in the measured range (Fig 5b). Do not say 'EBTs are more compute-efficient today'. The paper says this only as a slope or extrapolation claim (p.9).
10. **The 29% is relative to EBT's own no-thinking baseline.** In Fig 6a, EBT with no thinking (≈44.8) is *worse* than Transformer++ (≈38.4). One extra step (≈35.9) already beats it, and BoN reaches ≈31.8 [approx., read from Fig 6a]. The Transformer++ line is flat by construction: it "cannot improve performance at all" per token (p.10).
11. **σ has three different meanings.**
    - Image denoising σ = *fraction of the diffusion schedule* (0.1 train / in-distribution; 0.2 OOD test). It is not a noise standard deviation (p.13). [derived] σ = 0.1 ≈ noise std 0.32, σ = 0.2 ≈ 0.58 under a 1000-step linear schedule.
    - Langevin σ = magnitude of η ∼ N(0, σ) in Eq. 2 (p.7). Its value is "Langevin Dynamics Noise 3" (Table D.4, p.36).
    - In Fig E.1's diffusion update, σ_t is the diffusion sampler's noise scale.
12. **"99% fewer forward passes" = 3 vs 300** (p.35). An EBT NFE counts one optimization step (p.8), and each step also needs a backward pass for ∇_ŷ E. So wall-clock or FLOP savings are smaller than 100× [derived]. A rough estimate, not in the paper: if one EBT step costs a forward pass plus a backward-to-input pass (about 2–3 forward-pass equivalents), then 3 EBT steps ≈ 6–9 forward-equivalents vs 300 DiT forward passes, roughly 33–50× less compute [derived, unverified]. Also, EBT is not ahead at 1 vs 100 or 2 vs 200 passes; it only wins at 3 vs 300 (Fig 12, approx.). The DiT baseline is DDIM applied recursively 3 times; EBT also re-denoises its own output (p.35).
13. **'10× accuracy' is loose.** Top-1 is 5.32% vs 0.31% (≈17×); top-5 is 13.2% vs 1.36% (≈9.7×) [derived]. Absolute accuracies are tiny (a linear probe on denoising features). Do not imply EBT is a good classifier.
14. **"3.5 PSNR"** in the text (p.14) vs 23.29 − 19.56 = 3.73 from Table 4 [derived]. Cite the table numbers.
15. **Few steps in training.** Pretraining uses 2 steps (S1, Table D.3) or 2–3 randomized steps (S2, Table D.4) (p.26). Do not show 'hundreds of steps' as the training procedure. The heatmaps show 12 inference iteration columns (0–11).
16. **Backpropagating "through the entire optimization process" (p.7) is the idealized description.** In practice S1 *detaches* predictions between steps and puts the loss on every step; S2 does *not* detach, *truncates* backprop and puts the loss only on the last step (p.30, p.36, p.43).
17. **S1 vs S2.** "Many of the pretraining scaling experiments" (the paper's words; it does not say which) use S1 models with *no* landscape regularization (p.30). The thinking results use S2. Image denoising EBTs did not need S2 hyperparameters (p.35). In S1 a learnable step embedding conditions each step, so each step has its own landscape. In S2 "all optimization steps are performed along the same energy landscape" (p.42, p.45).
18. **Text predictions are distributions, not token IDs.** ŷ is a 50,277-dim logit vector per position, initialized from N(0, I). It is softmaxed and then linearly projected to the embedding (p.43–44, Fig 2). Gradient descent runs in this continuous space.
19. **ŷ0 is random Gaussian noise** in Alg 1/2 (p.7). It is not the Transformer++ prediction (that is only a future-work idea, A.4).
20. **BoN is per prediction (per token or frame) and uses the model's own energy.** There is no external verifier and no reward model (p.8). Alg 2 calls the number of candidates **M** and the steps **N**. Sec 3.3 says "N predictions"; avoid the clash.
21. **Table 2: the full configuration is best only for the combined column (18.7).** For thinking longer alone it scores 7.19, below "No Langevin" (17.2, which the paper bolds as the best in that column) and "No Replay Buffer" (14.8). Removing random step size nearly kills thinking (−1.47 / 0.19). Removing random step counts gives exactly 0.00 for thinking longer (p.10).
22. **Fig 6b (BoN-5 gains 4–8% → 10–14%) is Dyck only.** The trend did not appear on other benchmarks (p.34). Fig B.1a is RedPajamaV2 validation. Fig B.1b's huge gains are an extrapolation to 15T tokens.
23. **Thinking-experiment model size conflict.** D.1.2 says xxs (p.34); Table D.4 says "Small" (p.36). LR 0.0012 and the ≈32.8B-token count fit xxs at batch 128 × 1M steps [derived]. Say 'a small (xxs, 6.18M non-embedding) model' or just 'small models'.
24. **Where "800M" comes from:** "scale well up to 800M parameters" (p.17). The largest config is xl, 708M non-embedding (Table D.1), used in video (Fig 9). Text parameter scaling goes only up to large (396M) (p.34). Image denoising uses large.
25. **Video metric is minimum validation Smooth-L1 loss** on 3136-d VAE features (p.12). It is not pixel quality or FVD. EBT video loss is higher at every size (Fig 9).
26. **The OOD shift in Fig 7 is a perplexity ratio** (downstream ppl / pretrain ppl), not a distance between distributions (p.11). The 1.0 point is the pretraining data itself.
27. **Diffusion and RNN ✗ marks in Table 1 are "generally" true.** The paper concedes score-based diffusion and Mixture Density Networks can express uncertainty (footnote 4, p.3), and recurrent-depth RNNs [25] can allocate compute (footnote 2, p.3).
28. **Uncertainty heatmaps (Figs 8, 11, B.2) are "Normalized Energy" 0–1** with a reversed viridis map (dark purple = high, yellow = low). The normalization method is unspecified. Every token is at its highest at iteration 0, but iteration-0 values already range from ≈0.45 to 1.0. Nearly all of the drop happens at iteration 1, and the rows are roughly flat after that [approx., read from Fig 8]. The interesting part is which tokens stay high. Do not animate a slow, steady descent over 11 steps as if it were the paper's data.
29. **"Thinking" here is not chain-of-thought.** It is iterative refinement of one continuous prediction. Models at this size did not benefit from CoT (p.10).
30. **The FLOP multiplier differs by setting:** 1.66× (single step, no sequence doubling, p.41) vs 3.33× per step (AR EBT, p.36) vs 6.66× (AR EBT with 2 steps, p.36).
31. **Planning / MPC / control experiments do not exist in the paper.** Only A.3 speculates about world models used as policies (p.26). Label our planning demos as an extension.
32. **The many-modes failure:** blurry, averaged outputs come from the "convex energy landscape assumption" (p.17, p.29). Do not claim EBTs model multimodal output distributions well. They can represent several low-energy y in principle (p.39), but the training pushes toward one convex basin.
33. **"Energy-Based Transformer" ≠ "Energy Transformer" [152]** (p.38).
34. **Video VAE naming:** the text says "frozen SD-XL VAE [114, 115]" (p.12), but reference [115] links to huggingface.co/stabilityai/sd-vae-ft-mse (p.23). Quote it as 'SD-XL VAE (per the paper)'.
35. **Image σ = 0.2 is OOD for both models.** Fig 10 is the OOD (noisier) setting, and Fig 12 matches Table 4's OOD column [derived].
36. **Exact NLP batch/step settings:** data scaling uses batch 128 (xxs). Parameter and FLOP scaling run 105k steps with batch 32/46/90/170/256. Context is always 256. There is no FFN multiplier (p.34).
37. **How eval works (text):** the paper reports teacher-forced next-token perplexity on RedPajamaV2 validation and on four downstream sets. It does not specify which tokens of each benchmark are scored, the prompt format, or how "best checkpoints" were chosen (p.9, p.34) [unspecified]. There is no free-running text generation and no decoding rule. Do not show 'EBT-generated text' as a paper result.
38. **No error bars, one seed (33).** No result has variance estimates (p.33) [derived]. Use 'the paper reports', not 'EBTs reliably …'.
39. **Do not mix Table 3, Fig 6a and Fig 7 numbers.** They cannot be reconciled from the printed values, and Fig 7's points are not labeled by dataset [derived].
40. **Fig 6b is noisy.** The fit rises from ≈7.9% to ≈10.6%. Individual points range from ≈4.4% to ≈13.9%. The text's "increasing up to 10%−14% from 4%−8%" (p.10) describes the spread, not a clean trend (Dyck only).
41. **Who wins at equal compute?** Text: Transformer++ at every measured FLOP budget (Fig 5b). Images: EBT at 3 vs 300 passes. At 1 vs 100 and 2 vs 200 the two are about equal (Fig 12). Video: Transformer++ at every measured size (Fig 9).
42. **Diffusion comparisons in E.2 are conditional.** They hold "Under the assumption that the energy landscape is well formed and that optimization is well behaved" (p.37).

---

## Fact-check log (adversarial pass)

An independent pass re-checked every quoted string against `paper_text.txt` with a script, re-read all 45 pages, and re-rendered Figs 1–12, B.1, B.2, B.3, C.1, E.1 and H.1 from the PDF at 110–300 dpi. All per-axis scaling percentages (35.98, 28.46, 5.29, 2.91, 2.92, 0.02, 33.66, 34.28, 35.69, 51.70, 3.30) match the figure titles. All table values (Tables 1–4, D.1–D.4) match. All [derived] arithmetic was recomputed. Corrections made in this pass:
- Fixed three near-verbatim quotes: "re-framing" (p.1), "often deteriorates performance" (p.2), and "to try and fool the discriminator" (p.7). Expanded "3.5 in PSNR" to the exact wording (p.14).
- Table 2 bolding: 17.2 is also bold in the paper.
- Fig 8: iteration-0 energies are not uniformly high. Most of the drop happens at iteration 1.
- Fig 5c: EBT is not worse at every width; it is slightly better at the middle points.
- Fig 4b: the crossover is between 8K and 16K tokens per batch.
- Fig 12: EBT only beats DiT at 3 vs 300 passes.
- 'Most scaling plots use S1' changed to the paper's "Many".
- Added the missing caveats listed in items 37–42, plus notes on the eval protocol, the single seed, the pseudocode loose ends, step-count randomization, Fig 6b noise, Fig B.1b's speculative extrapolation, Fig 11, and the E.2 assumption.
