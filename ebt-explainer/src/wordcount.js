// Count prose words per panel (title, lede, text, steps, after), stripping HTML and KaTeX.
global.window = { EBTV: { U: {} } }; global.document = { };
const fs = require('fs'), path = require('path');
const dir = path.resolve(__dirname, '../assets/js/panels');
const panels = {}; global.EBT = { panel: (d) => { panels[d.id] = d; } }; window.EBT = global.EBT;
eval(fs.readFileSync(path.join(dir, 'manifest.js'), 'utf8').replace('window.EBT_PANELS', 'global.EBT_PANELS'));
const strip = (s) => String(s || '').replace(/\$\$[\s\S]*?\$\$/g, ' EQ ').replace(/\$[^$]*\$/g, ' x ').replace(/<[^>]+>/g, ' ').replace(/&[a-z]+;/g, ' ');
const wc = (s) => strip(s).split(/\s+/).filter(w => /[A-Za-z0-9]/.test(w)).length;
let total = 0;
for (const f of fs.readdirSync(dir).filter(f => f.endsWith('.js') && f !== 'manifest.js')) {
  try { eval(fs.readFileSync(path.join(dir, f), 'utf8')); } catch (e) { console.log('ERR', f, e.message); }
}
const ids = (global.EBT_PANELS || []).flatMap(p => p.ids);
for (const id of (process.argv[2] === 'all' ? Object.keys(panels) : ids)) {
  const d = panels[id]; if (!d) { console.log(id.padEnd(18), 'missing'); continue; }
  const n = wc(d.title) + wc(d.lede) + wc(d.text) + (d.steps || []).reduce((a, s) => a + wc(s.label) + wc(s.html), 0) + wc(d.after);
  total += n; console.log(id.padEnd(18), n);
}
console.log('TOTAL', total);
