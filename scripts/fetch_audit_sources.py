"""Fetch the exact paper/code snapshots needed by the full evidence audit."""
from pathlib import Path
import hashlib
import json
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main():
    directory = ROOT / 'research'
    manifest = json.loads((directory / 'paper_source_manifest.json').read_text(encoding='utf-8'))
    for source in manifest['sources']:
        path = directory / source['local_path']
        if path.is_file():
            data = path.read_bytes()
        else:
            with urllib.request.urlopen(source['url'], timeout=90) as response:
                data = response.read()
        actual = hashlib.sha256(data).hexdigest()
        if actual != source['sha256']:
            raise RuntimeError(f'Unexpected source bytes: {source["local_path"]}; expected '
                               f'{source["sha256"]}, received {actual}. No file was overwritten.')
        if not path.exists():
            path.write_bytes(data)
        print(f'Verified {source["local_path"]}', flush=True)


if __name__ == '__main__':
    main()
