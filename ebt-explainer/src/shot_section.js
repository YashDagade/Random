// Screenshot one site chapter at desktop (1280) and phone (390) widths and report console errors.
// usage: node src/shot_section.js <sectionId> [outPrefix] [--full]   -> video/frames/site_<id>_desktop.png, _phone.png
// Optional: set ACTIONS env var to a JS snippet run in the page before the screenshot (e.g. clicking buttons).
const path = require('path'), fs = require('fs');
const { chromium } = require('/tmp/claude-0/-home-user-Random/9843a8a1-7b0a-5ed7-8867-6762f4a2ea11/scratchpad/work/node_modules/playwright');
(async () => {
  const id = process.argv[2]; const root = path.resolve(__dirname, '..');
  const pre = process.argv[3] && !process.argv[3].startsWith('--') ? process.argv[3] : `site_${id}`;
  const b = await chromium.launch({ executablePath: '/opt/pw-browsers/chromium', args: ['--allow-file-access-from-files'] });
  for (const [name, w] of [['desktop', 1280], ['phone', 390]]) {
    const p = await b.newPage({ viewport: { width: w, height: 900 } });
    const errs = [];
    p.on('pageerror', e => errs.push('pageerror: ' + e.message));
    p.on('console', m => { if (m.type() === 'error') errs.push('console: ' + m.text()); });
    await p.goto('file://' + root + '/index.html');
    await p.waitForTimeout(800);
    const el = await p.$('#' + id);
    if (!el) { console.log('no section #' + id); continue; }
    await el.scrollIntoViewIfNeeded(); await p.waitForTimeout(1200);
    if (process.env.ACTIONS) { await p.evaluate(process.env.ACTIONS); await p.waitForTimeout(1500); }
    const ow = await p.evaluate(() => document.documentElement.scrollWidth - innerWidth);
    const f = `${root}/video/frames/${pre}_${name}.png`;
    await el.screenshot({ path: f });
    console.log(name, 'horizontal overflow px:', ow, 'errors:', errs.length ? errs : 'none', '->', f);
    await p.close();
  }
  await b.close();
})();
