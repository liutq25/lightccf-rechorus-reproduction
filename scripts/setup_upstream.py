"""Fetch pinned upstream code and regenerate the two ReChorus course datasets."""
from pathlib import Path
import argparse
import hashlib
import shutil
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
RECHORUS = ROOT / 'framework' / 'ReChorus'
LIGHTCCF = ROOT / 'vendor' / 'LightCCF-main'
MOVIELENS_ZIP = ROOT / 'data' / 'raw' / 'ml-1m.zip'
MOVIELENS_SHA256 = 'a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20'


def git_clone_pin(url, path, commit):
    if path == LIGHTCCF and path.exists() and not (path / '.git').exists():
        # The original local research archive was a ZIP, not a Git checkout.
        expected = '1e52ddaa5634fed2db5a883ec2686428b0ac9a6bb2b446e6c7f69aca8010bd66'
        actual = hashlib.sha256((path / 'LightCCF.py').read_bytes()).hexdigest()
        if actual == expected:
            return
        raise RuntimeError(f'Existing author ZIP snapshot has an unexpected hash: {actual}')
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'clone', '--config', 'core.autocrlf=false', url, str(path)], check=True)
        subprocess.run(['git', '-C', str(path), 'checkout', '--detach', commit], check=True)
    head = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
    if head != commit:
        raise RuntimeError(f'{path} is at {head}, expected {commit}; inspect it manually')


def install_extensions():
    for source, destination in (
        (ROOT / 'extensions' / 'LightCCF.py', RECHORUS / 'src/models/general/LightCCF.py'),
        (ROOT / 'extensions' / 'LightCCFMasked.py', RECHORUS / 'src/models/general/LightCCFMasked.py'),
        (ROOT / 'extensions' / 'ReproductionRunner.py', RECHORUS / 'src/helpers/ReproductionRunner.py'),
    ):
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination.read_bytes() != source.read_bytes():
            raise RuntimeError(f'Extension differs from source: {destination}')
        shutil.copy2(source, destination)


def normalize_grocery_line_endings():
    """Pin the measured dataset bytes independently of the host Git settings.

    The reported Windows run used CRLF; only the line-ending-equivalent upstream
    content is accepted. An altered dataset is rejected before any write.
    """
    expected={
        'train':'026610b38a04930b84468cd6a7407ec9de52393d42eff910d9783856f21adaa6',
        'dev':'524aeba840c7170281902406f394071bb8d4d8ffc4185d80ed7d1bc3ab03de32',
        'test':'9d5591a373a17575f2988f5b5951fa5027fb695c53ce0f857f9497168db51f20',
    }
    for phase,digest in expected.items():
        path=RECHORUS/'data/Grocery_and_Gourmet_Food'/f'{phase}.csv'
        original=path.read_bytes()
        canonical=original.replace(b'\r\n',b'\n').replace(b'\n',b'\r\n')
        if hashlib.sha256(canonical).hexdigest()!=digest:
            raise RuntimeError(f'Unexpected Grocery content: {path}; refusing to overwrite')
        if canonical!=original:
            path.write_bytes(canonical)


def get_movielens():
    MOVIELENS_ZIP.parent.mkdir(parents=True, exist_ok=True)
    if not MOVIELENS_ZIP.exists():
        url = 'https://files.grouplens.org/datasets/movielens/ml-1m.zip'
        with (urllib.request.urlopen(url, timeout=120) as response,
              MOVIELENS_ZIP.open('wb') as output):
            shutil.copyfileobj(response, output)
    digest = hashlib.sha256(MOVIELENS_ZIP.read_bytes()).hexdigest()
    if digest != MOVIELENS_SHA256:
        raise RuntimeError(f'MovieLens zip SHA256 mismatch: {digest}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--with-author', action='store_true',
                        help='Also fetch the pinned author code for the optional Douban anchor')
    args = parser.parse_args()
    git_clone_pin('https://github.com/THUwangcy/ReChorus.git', RECHORUS,
                  'c164ec4303cc20ddcfbd1b57de366a481811d1e5')
    normalize_grocery_line_endings()
    install_extensions()
    get_movielens()
    subprocess.run([sys.executable, str(ROOT / 'scripts/prepare_rechorus_data.py')],
                   cwd=ROOT, check=True)
    if args.with_author:
        git_clone_pin('https://github.com/ZzYUuuu/LightCCF.git', LIGHTCCF,
                      '45645ef15fc08e858ffc81e1b2530b4b1805ad34')
    print('Pinned code, extensions, and course datasets are ready.')


if __name__ == '__main__':
    main()
