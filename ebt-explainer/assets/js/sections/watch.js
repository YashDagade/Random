/* Chapter 00: the explainer video, with chapter jumps and a link to the frame-accurate canvas player. */
EBT.section({
  id: 'watch',
  nav: 'Watch',
  kicker: 'Start here',
  title: 'The 10-minute animated explainer',
  lede: 'No narration, just figures, motion, on-screen text and a quiet music bed. Pause anywhere; every chapter below lets you run that part of the paper yourself.',
  mount(el, lib) {
    const h = lib.h;
    const chapters = (lib.data('video_chapters') || {}).chapters || [];
    const video = h('video', { controls: true, preload: 'metadata', playsinline: true, src: 'video/ebt_explainer.mp4', 'aria-label': 'Animated explainer video of Energy-Based Transformers' });
    const missing = h('div', { class: 'callout warn', hidden: true, html: '<b class="t">Video file not found</b>Render it with <code>node src/render_video.js</code>, or open the <a href="video/player.html">live canvas player</a>, which draws the same animation in real time.' });
    video.addEventListener('error', () => { missing.hidden = false; });
    el.appendChild(h('div', { class: 'video-box' }, video));
    el.appendChild(missing);
    const fmt = (s) => Math.floor(s / 60) + ':' + String(Math.floor(s % 60)).padStart(2, '0');
    if (chapters.length) {
      const bar = h('div', { class: 'chapters', role: 'group', 'aria-label': 'Video chapters' });
      chapters.forEach(c => {
        const b = h('button', { type: 'button' }, h('span', { class: 't' }, fmt(c.t0)), c.title);
        b.addEventListener('click', () => { try { video.currentTime = c.t0 + 0.05; video.play().catch(() => {}); } catch (e) {} });
        bar.appendChild(b);
      });
      el.appendChild(bar);
    }
    el.appendChild(h('p', { class: 'readout', html: 'Prefer to scrub frame by frame? Open the <a href="video/player.html">live canvas player</a> (space = play, arrow keys = ±5 s). It renders the same scenes in your browser, so you can pause on any frame at full resolution.' }));
    const guide = h('div', { class: 'grid3' });
    [
      ['families', 'Four ways to predict', 'Why a Transformer cannot think longer about one token, and what an EBT does differently.'],
      ['landscape', 'Energy landscape lab', 'Drop a guess on a real learned landscape and watch gradient descent think.'],
      ['training', 'Training lab', 'Unroll the optimization and backpropagate through it, one step at a time.'],
      ['planning', 'Planning and MPC', 'The same idea used to plan: optimize a whole action sequence against an energy.'],
      ['scaling', 'Scaling laws', 'Read the paper\'s six scaling axes and what a 35% higher rate means.'],
      ['data', 'Data and evaluation', 'The actual corpora, images and benchmarks, and how perplexity and PSNR are computed.'],
    ].forEach(([id, t, d]) => guide.appendChild(h('a', { href: '#' + id, class: 'card', style: 'text-decoration:none;color:inherit;display:grid;gap:4px' }, h('h3', { style: 'font-size:18px' }, t), h('p', { style: 'color:var(--muted);font-size:15px' }, d))));
    el.appendChild(guide);
  },
});
