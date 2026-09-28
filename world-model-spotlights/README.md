# Spotlight Atlas

A browsable list of **main-conference** NeurIPS and ICML **orals and spotlights (2024 onward)** on world models, JEPA and self-supervised learning, action-conditioned learning and planning, robot learning, continual learning, and memory.

Open `index.html` in a browser. It is self-contained; the data is embedded.

- **Papers**: 143 orals and spotlights, plus 10 baseline posters (DINO-WM, PLDM, Temporal Straightening, Rectified LpJEPA, and related JEPA work). Each entry has the PDF link, the abstract, every reviewer score, and how the paper ranks within its venue. It also has a short note on why the paper stood out and how it relates to continual learning and memory for JEPA world models.
- **What earns an oral**: score statistics across about 17,000 reviewed papers (orals vs spotlights vs posters vs rejects), your baseline posters compared with the spotlight bar, a reading list, and a framing checklist.
- **Sources & scales**: the data source, each venue's scoring scale, and caveats.

`papers.json` holds the same data as structured JSON.

All papers are from the main conference track. Workshop papers and the Datasets & Benchmarks and Position tracks are excluded.

Venues covered: NeurIPS 2024 and 2025, ICML 2024, 2025 and 2026. NeurIPS 2026 decisions were not yet public at build time (28 Sep 2026).
The data comes from the [Paper Copilot paper lists](https://github.com/papercopilot/paperlists), which mirror OpenReview. ICML 2024 released no reviews. The ICML 2026 data labels spotlights only, with no separate orals.
The per-paper notes were written from the abstracts and scores, not the full papers.
