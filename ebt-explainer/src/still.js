// Render stills of one scene: node src/still.js <sceneId> <t1> [t2 ...]  -> video/frames/<sceneId>_<t>.png
// Uses the scene alone (player.html?scene=<id>&render=1). Also prints console errors from the page.
const path = require('path');
const { chromium } = require('/tmp/claude-0/-home-user-Random/9843a8a1-7b0a-5ed7-8867-6762f4a2ea11/scratchpad/work/node_modules/playwright');
const fs = require('fs');
(async () => {
  const [id, ...ts] = process.argv.slice(2);
  if (!id || !ts.length) { console.error('usage: node src/still.js <sceneId> <t...>'); process.exit(1); }
  const root = path.resolve(__dirname, '..');
  const b = await chromium.launch({ executablePath: '/opt/pw-browsers/chromium', args: ['--allow-file-access-from-files'] });
  const p = await b.newPage({ viewport: { width: 1920, height: 1080 } });
  p.on('console', m => { if (m.type() === 'error' || m.type() === 'warning') console.log('[page]', m.type(), m.text()); });
  p.on('pageerror', e => console.log('[pageerror]', e.message));
  await p.goto('file://' + root + '/video/player.html?render=1&scene=' + encodeURIComponent(id));
  await p.waitForFunction(() => window.__ready === true, null, { timeout: 60000 });
  fs.mkdirSync(root + '/video/frames', { recursive: true });
  for (const t of ts) {
    const t0 = Date.now();
    const url = await p.evaluate(([id, t]) => window.__sceneFrame(id, parseFloat(t), 'image/png'), [id, t]);
    const f = `${root}/video/frames/${id}_${String(t).replace('.', 'p')}.png`;
    fs.writeFileSync(f, Buffer.from(url.split(',')[1], 'base64'));
    console.log('wrote', f, (Date.now() - t0) + 'ms');
  }
  // timing check: average draw time over 30 frames
  const ms = await p.evaluate((id) => { const e = EBTV.timeline().find(x => x.id === id); const c = document.getElementById('cv').getContext('2d'); const n = 30, t0 = performance.now(); for (let i = 0; i < n; i++) EBTV.render(c, e.t0 + (i / n) * e.def.dur); return (performance.now() - t0) / n; }, id);
  console.log('avg draw ms', ms.toFixed(1), '(target < 25)');
  await b.close();
})();
