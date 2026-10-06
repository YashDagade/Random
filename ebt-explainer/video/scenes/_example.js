// Example scene: shows the helper API. Not part of the manifest.
EBTV.scene({
  id: '_example',
  title: 'Example',
  dur: 12,
  assets: ['media/samples/coco/01.jpg'],          // preloaded before rendering (paths relative to project root)
  draw(ctx, t, U) {
    const { C } = U;
    U.header(ctx, t, 12, 'Example', 'How a scene is drawn');
    // a fading text block, visible 1.0s..11s
    U.text(ctx, 'Text wraps at maxWidth and fades in.', 96, 260, { size: 36, alpha: U.fade(t, 1, 11), maxWidth: 700 });
    // axes + a curve that draws on between 2s and 5s
    const ax = U.axes(ctx, { x: 1000, y: 260, w: 760, h: 420, xlim: [0, 10], ylim: [0, 1], xticks: [0, 5, 10], yticks: [0, 0.5, 1], xlabel: 'thinking step', ylabel: 'energy', alpha: U.fade(t, 1.5, 11) });
    const pts = Array.from({ length: 50 }, (_, i) => [i / 5, Math.exp(-i / 12)]);
    U.plot(ctx, ax, pts, { color: C.ebt, width: 4, progress: U.seg(t, 2, 5) });
    // a heatmap of a bowl with contour lines, cached by key
    const g = Array.from({ length: 48 }, (_, r) => Array.from({ length: 48 }, (_, c) => ((c - 30) / 24) ** 2 + ((r - 20) / 24) ** 2));
    U.heatmap(ctx, g, 96, 420, 420, 420, { key: 'example-bowl', alpha: U.fade(t, 3, 11) });
    U.contours(ctx, g, 96, 420, 420, 420, [0.1, 0.3, 0.6, 1.0], { alpha: U.fade(t, 3.5, 11) });
    U.badge(ctx, 'toy', null, 96, 860, { alpha: U.fade(t, 4, 11) });
    U.caption(ctx, 'Captions sit at the bottom and fade in and out.', t, 6, 11);
  },
});
