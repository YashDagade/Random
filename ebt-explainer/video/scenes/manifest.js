// Scene order for the explainer video. Durations live in each scene file (EBTV.scene({dur})).
window.EBTV_SCENES = [
  's01_intro',      // ~50 s  System 1 vs System 2, the question
  's02_families',   // ~75 s  AR Transformer / RNN / Diffusion vs EBT, the three facets
  's03_energy',     // ~75 s  energy function, energy landscape over predictions vs loss landscape over weights
  's04_data',       // ~60 s  what the data looks like: RedPajama tokens, COCO + noise, SSV2 frames
  's05_thinking',   // ~80 s  sampling = gradient descent on the prediction, Langevin, BoN, uncertainty
  's06_training',   // ~70 s  Algorithm 1, backprop through optimization, regularizers, toy training live
  's07_eval',       // ~75 s  how evaluation works: perplexity downstream, thinking/OOD, PSNR vs NFE, linear probe
  's08_scaling',    // ~70 s  scaling laws: six axes, rates, video, thinking scales
  's09_wrap',       // ~45 s  why better, limitations, energy minimization as planning (beyond the paper)
];
