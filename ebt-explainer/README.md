# Energy-Based Transformers, step by step

An interactive, offline explainer for **Energy-Based Transformers are Scalable Learners and Thinkers** (Gladstone et al., arXiv 2507.02092, 2025).

14 short sections (about 5,000 words in total), each a panel with a live figure, numbered steps and the key equations, focused on intuition for the method and the architecture: how EBTs differ from autoregressive Transformers, RNNs and diffusion; what an energy is; the energy landscape vs the loss landscape; thinking as gradient descent; Langevin noise and best-of-N self-verification; thinking over a vocabulary; training by backpropagating through the optimization; why it trains stably; how the prediction enters the Transformer; what the data and evaluation look like; the results and their costs; and two panels beyond the paper (energy-based planning with MPC steps, and the connection to JEPA world models).

## Run it

Open `index.html` in a browser. Everything is local: fonts, KaTeX, data and images. Or serve it:

```sh
./serve.sh          # http://localhost:8000
```

Keys: `j` / `k` next / previous section, `1`–`9` jump to a step, `←` / `→` previous / next step.

## Provenance

Every figure is tagged:

- **from the paper**: numbers and figures from the paper (digitized plots are approximate and say so).
- **toy model trained for this explainer**: three tiny EBTs trained from scratch on CPU with the paper's recipe (a 2-D toy, a character-level text model on RedPajama text, an image denoiser). They show mechanisms, not the paper's results.
- **beyond the paper**: planning, MPC and JEPA material.
- **illustration**: conceptual diagrams.

## Layout

```
index.html                 page shell (left section rail, panels)
assets/js/lib.js           panel system + helpers
assets/js/engine.js        canvas drawing helpers (light theme)
assets/js/toy2d_model.js   the 2-D toy EBT, runnable in the browser
assets/js/panels/*.js      one file per section (order in panels/manifest.js)
data/*.json, data/bundle.js  experiment data, digitized figures, real samples (bundle.js is generated)
media/                     paper figure crops, real dataset samples, toy outputs
notes/                     paper facts (with page refs), specs
src/                       training/export scripts for the toy models, data tools, screenshot tools
```

Rebuild the data bundle after changing `data/*.json`: `python3 src/bundle_data.py`.
