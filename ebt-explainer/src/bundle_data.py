"""Bundle every data/*.json into data/bundle.js as window.EBT_DATA[name] so pages work from file://."""
import json, pathlib
root = pathlib.Path(__file__).resolve().parent.parent / 'data'
out = ['window.EBT_DATA = window.EBT_DATA || {};']
for p in sorted(root.glob('*.json')):
    try:
        obj = json.loads(p.read_text())
    except Exception as e:
        print('skip', p.name, e); continue
    out.append(f'window.EBT_DATA[{json.dumps(p.stem)}] = {json.dumps(obj, separators=(",", ":"))};')
(root / 'bundle.js').write_text('\n'.join(out) + '\n')
print('bundled', len(out) - 1, 'files ->', root / 'bundle.js', (root / 'bundle.js').stat().st_size // 1024, 'KB')
