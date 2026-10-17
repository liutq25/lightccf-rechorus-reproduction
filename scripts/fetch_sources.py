"""Download and pin publicly available research inputs."""
from pathlib import Path
import concurrent.futures
import hashlib
import json
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
AUTHOR_COMMIT = '45645ef15fc08e858ffc81e1b2530b4b1805ad34'

def fetch(url, target):
    path = ROOT / target
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=90) as response:
            data = response.read()
        path.write_bytes(data)
        record = dict(url=url, path=target, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        path.with_suffix(path.suffix + '.source.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
        print(json.dumps(record), flush=True)
        return str(path)
    except Exception as exc:
        print(url, repr(exc), flush=True)

if __name__ == '__main__':
    jobs = [
        (f'https://codeload.github.com/ZzYUuuu/LightCCF/zip/{AUTHOR_COMMIT}', 'research/LightCCF-pinned.zip'),
        (f'https://raw.githubusercontent.com/ZzYUuuu/LightCCF/{AUTHOR_COMMIT}/LightCCF.py', 'research/official_LightCCF.py'),
    ]
    with concurrent.futures.ThreadPoolExecutor() as pool:
        list(pool.map(lambda job: fetch(*job), jobs))
    path = ROOT / 'research/LightCCF-pinned.zip'
    if path.exists():
        with zipfile.ZipFile(path) as z:
            z.extractall(ROOT / 'vendor')
        extracted = ROOT / 'vendor' / f'LightCCF-{AUTHOR_COMMIT}'
        destination = ROOT / 'vendor' / 'LightCCF-main'
        if not destination.exists():
            extracted.rename(destination)
