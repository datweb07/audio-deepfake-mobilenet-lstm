"""Bounded official-source metadata inspection; no audio acquisition."""
from pathlib import Path
import json
import urllib.request
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/jaes_v2/datasets/mlaad_tiny_discovery_v1'


def get(url, limit=12_000_000):
    with urllib.request.urlopen(url, timeout=45) as response:
        data = response.read(limit + 1)
    if len(data) > limit:
        raise ValueError('Metadata byte limit exceeded')
    return data


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    info = json.loads(get('https://huggingface.co/api/datasets/mueller91/MLAAD-tiny'))
    (OUT / 'repository.json').write_text(json.dumps(info, indent=2), encoding='utf8')
    revision = info['sha']
    paths = [row['rfilename'] for row in info['siblings']]
    selected = ['README.md', 'LICENSE', 'NOTICE_NO_COMMERCIAL_USE.txt', 'original/LICENSE']
    selected += [path for path in paths if path.endswith('meta.csv') and path.startswith('fake/en/')][:3]
    for path in selected:
        target = OUT / path
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            target.write_bytes(get(f'https://huggingface.co/datasets/mueller91/MLAAD-tiny/resolve/{revision}/{quote(path)}'))
            print(path, target.stat().st_size, flush=True)
        except Exception as error:
            print(path, type(error).__name__, str(error), flush=True)
    print(json.dumps({'revision': revision, 'files':len(paths), 'metadata': [p for p in paths if not p.endswith('.wav')],
                      'groups_en':sorted({p.split('/')[2] for p in paths if p.startswith('fake/en/')})}, indent=2))


if __name__ == '__main__':
    main()
