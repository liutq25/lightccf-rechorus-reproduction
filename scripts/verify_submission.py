"""Verify immutable run evidence, generated LaTeX blocks, and a hand-in ZIP.

Uses the PDF runtime's pypdf. Never trains models or rewrites run records.
"""
from pathlib import Path
import argparse
from collections import Counter
import hashlib
import json
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(zip_path=None):
    checks = {}
    counts = Counter()
    validation_only = 0
    run_records = {}
    for group in ('core', 'ablation', 'official_anchor', 'extension'):
        for path in sorted((ROOT / 'runs' / group).rglob('result.json')):
            result = read_json(path)
            assert result['status'] == 'complete', str(path)
            run_records[path.relative_to(ROOT).as_posix()] = sha(path)
            if result.get('pilot'):
                assert result['test'] is None, str(path)
                validation_only += 1
                continue
            counts[group] += 1
            if group == 'official_anchor':
                assert result['selection_uses_test'] is True
                for phase in ('train', 'test'):
                    data = ROOT / 'vendor/LightCCF-main/dataset/douban-book' / (phase + '.txt')
                    assert sha(data) == result[phase + '_sha256']
                continue
            config = read_json(path.with_name('config.json'))
            for phase, digest in config['data_sha256'].items():
                data = ROOT / 'framework/ReChorus/data' / result['dataset'] / (phase + '.csv')
                assert sha(data) == digest, str(data)
            history = [json.loads(line) for line in path.with_name('history.jsonl').read_text(encoding='utf-8').splitlines()]
            assert len(history) == result['epochs']
            chosen = next(h for h in history if h['epoch'] == result['best_epoch'])
            assert chosen['dev'] == result['dev'], str(path)
            assert chosen['dev']['NDCG@20'] == max(h['dev']['NDCG@20'] for h in history if 'dev' in h)
            assert path.with_name('best.pt').is_file()
            for phase in ('dev','test'):
                users = read_json(path.with_name(phase + '_users.json'))
                for key, value in result[phase].items():
                    assert abs(sum(u[key] for u in users) / len(users) - value) < 1e-12
    assert dict(counts) == {'core':18,'ablation':6,'official_anchor':5,'extension':12}, counts
    assert validation_only == 10
    checks['formal_runs'] = dict(counts)
    checks['formal_runs_total'] = sum(counts.values())
    checks['validation_only_search_runs'] = validation_only
    checks['immutable_result_sha256'] = run_records
    manifest = read_json(ROOT / 'research/paper_source_manifest.json')
    for source in manifest['sources']:
        assert sha(ROOT / 'research' / source['local_path']) == source['sha256']
    checks['paper_and_author_sources_verified'] = len(manifest['sources'])
    for file in ('LightCCF.py','ReproductionRunner.py'):
        directory = 'models/general' if file == 'LightCCF.py' else 'helpers'
        assert (ROOT / 'extensions' / file).read_bytes() == (ROOT / 'framework/ReChorus/src' / directory / file).read_bytes()
    acceptance = read_json(ROOT / 'research/loss_validation_results.json')
    assert sha(ROOT / 'extensions/LightCCF.py') == acceptance['model_sha256_at_test']
    checks['model_matches_original_gradient_acceptance'] = True
    source = ROOT / '复现报告.tex'
    tex = source.read_text(encoding='utf-8')
    for name, value in read_json(ROOT / 'analysis/latex_blocks.json').items():
        match = re.search(r'% BEGIN AUTO ' + name + r'\n(.*?)\n% END AUTO ' + name, tex, re.S)
        assert match and match.group(1) == value, name
    body = tex.split('% MAIN BODY START')[1].split('% MAIN BODY END')[0]
    prose = re.sub(r'\\begin\{(figure|table)\}.*?\\end\{\1\}', '', body, flags=re.S)
    count_han = lambda text: len(re.findall(r'[\u4e00-\u9fff]', text))
    checks['body_prose_chinese_characters'] = count_han(prose)
    checks['body_including_figures_and_tables_chinese_characters'] = count_han(body)
    assert 3000 <= count_han(prose) <= 5000
    checks['figures'] = tex.count(r'\begin{figure}')
    checks['tables'] = tex.count(r'\begin{table}')
    assert checks['figures'] <= 10
    checks['auto_blocks_verified'] = len(read_json(ROOT / 'analysis/latex_blocks.json'))
    checks['tex_sha256'] = sha(source)
    pdf = source.with_suffix('.pdf')
    checks['pdf_sha256'] = sha(pdf)
    from pypdf import PdfReader
    reader = PdfReader(pdf)
    checks['pages'] = len(reader.pages)
    all_text = ''.join(p.extract_text() for p in reader.pages)
    assert 'github.com/liutq25/lightccf-rechorus-reproduction' in all_text
    checks['pdf_text_extractable'] = '邻域聚合' in all_text and 'MovieLens' in all_text
    if zip_path:
        with zipfile.ZipFile(zip_path) as archive:
            assert archive.testzip() is None
            for file in ('复现报告.tex','复现报告.pdf'):
                entry = next(n for n in archive.namelist() if n.endswith('/' + file))
                assert archive.read(entry) == (ROOT / file).read_bytes(), file
            manifest_name = next(n for n in archive.namelist() if n.endswith('/MANIFEST_SHA256.txt'))
            for line in archive.read(manifest_name).decode('utf-8').splitlines():
                digest, name = line.split('  ', 1)
                assert hashlib.sha256(archive.read(name)).hexdigest() == digest, name
            checks['zip_entries'] = len(archive.namelist())
        checks['zip_sha256'] = sha(zip_path)
        checks['zip_bytes'] = zip_path.stat().st_size
    return checks


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zip', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'analysis/final_delivery_audit.json')
    args = parser.parse_args()
    checks = verify(args.zip)
    args.output.write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in checks.items() if k != 'immutable_result_sha256'}, ensure_ascii=False, indent=2))
