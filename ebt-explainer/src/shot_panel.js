// Screenshot one panel of the site at desktop (1440×900) and phone (390×844) and report errors.
// usage: node src/shot_panel.js <panelId> [step(1-based)] [outName]
//   env ACTIONS="js code run in page before the shot" (e.g. click a button inside #<id>)
//   env FULL=1 to screenshot the whole panel (default: whole panel element)
// Writes video/frames/panel_<id>[_s<step>]_{desktop,phone}.png
const path = require('path');
const { chromium } = require('/tmp/claude-0/-home-user-Random/9843a8a1-7b0a-5ed7-8867-6762f4a2ea11/scratchpad/work/node_modules/playwright');
(async () => {
  const [id, step, outName] = process.argv.slice(2); const root = path.resolve(__dirname, '..');
  const name = outName || `panel_${id}${step ? '_s' + step : ''}`;
  const b = await chromium.launch({ executablePath: '/opt/pw-browsers/chromium', args: ['--allow-file-access-from-files'] });
  for (const [vn, w, hgt] of [['desktop', 1440, 900], ['phone', 390, 844]]) {
    const p = await b.newPage({ viewport: { width: w, height: hgt } });
    const errs = [];
    p.on('pageerror', e => errs.push('pageerror: ' + e.message));
    let missing = 0; p.on('console', m => { if (m.type() === 'error') { if (/ERR_FILE_NOT_FOUND/.test(m.text())) missing++; else errs.push('console: ' + m.text()); } });
    await p.goto('file://' + root + '/index.html#' + id);
    await p.waitForTimeout(1200);
    const el = await p.$('#' + id);
    if (!el) { console.log('no panel #' + id); await p.close(); continue; }
    await el.scrollIntoViewIfNeeded(); await p.waitForTimeout(600);
    if (step) { await p.evaluate(([id, s]) => { const L = (EBT.live || []).find(l => l.id === id); if (L) L.goStep(s - 1); }, [id, parseInt(step)]); await p.waitForTimeout(2500); }
    if (process.env.ACTIONS) { await p.evaluate(process.env.ACTIONS); await p.waitForTimeout(2000); }
    const ow = await p.evaluate(() => document.documentElement.scrollWidth - innerWidth);
    const f = `${root}/video/frames/${name}_${vn}.png`;
    await el.screenshot({ path: f });
    const hh = await el.evaluate(e => e.getBoundingClientRect().height);
    console.log('(missing files on page, usually panels not built yet:', missing + ')'); console.log(vn, 'panel height', Math.round(hh), 'overflow px', ow, 'errors', errs.length ? errs : 'none', '->', f);
    await p.close();
  }
  await b.close();
})();
