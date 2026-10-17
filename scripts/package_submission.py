"""Build the course hand-in ZIP from the finished report and reproducible code."""
from pathlib import Path
import argparse
import hashlib
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', type=int, required=True)
    parser.add_argument('--members', nargs='+', required=True)
    args = parser.parse_args()
    if args.group < 1 or any(any(char in name for char in '/\\<>:"|?*') for name in args.members):
        raise ValueError('Group/member names must form a valid Windows file name')
    prefix = f'期末作业-小组编号({args.group})-' + '-'.join(args.members)
    destination = ROOT.parent / f'{prefix}.zip'
    if destination.exists():
        raise RuntimeError(f'Refusing to overwrite existing submission: {destination}')
    report_files = [ROOT / name for name in ('复现报告.md', '复现报告.docx', '复现报告.pdf')]
    for path in report_files:
        if not path.is_file():
            raise FileNotFoundError(path)
    tracked = subprocess.check_output(['git', '-C', str(ROOT), 'ls-files', '-z']).split(b'\0')
    code_files = [ROOT / p.decode('utf-8') for p in tracked if p]
    if not code_files:
        raise RuntimeError('No tracked code files; stage the reviewed repository first')
    evidence = list((ROOT / 'analysis').glob('*'))
    for group in ('core', 'ablation', 'official_anchor'):
        base = ROOT / 'runs' / group
        if base.exists():
            evidence += [p for p in base.rglob('*') if p.is_file() and p.name in
                         {'result.json', 'config.json', 'history.jsonl', 'train.log',
                          'dev_users.json', 'test_users.json',
                          'author_result.txt', 'author_train.log'}]
    entries = {}
    for path in code_files:
        if path.is_file():
            entries[f'{prefix}/code/{path.relative_to(ROOT).as_posix()}'] = path
    for path in report_files:
        entries[f'{prefix}/{path.name}'] = path
    for path in evidence:
        if path.is_file():
            entries[f'{prefix}/evidence/{path.relative_to(ROOT).as_posix()}'] = path
    digest_lines = []
    with zipfile.ZipFile(destination, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=7) as z:
        for name, path in sorted(entries.items()):
            z.write(path, name)
            digest_lines.append(f'{hashlib.sha256(path.read_bytes()).hexdigest()}  {name}')
        z.writestr(f'{prefix}/MANIFEST_SHA256.txt', '\n'.join(digest_lines) + '\n')
    print(f'{destination}: {len(entries)} files, {destination.stat().st_size / 2**20:.1f} MiB')


if __name__ == '__main__':
    main()
