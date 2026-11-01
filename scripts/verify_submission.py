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


def verify_archive(path,report_directory=None):
    checks={}
    with zipfile.ZipFile(path) as archive:
        names=archive.namelist()
        assert len(names)==len(set(names)), 'Duplicate archive entries'
        assert archive.testzip() is None
        manifests=[name for name in names if name.endswith('/MANIFEST_SHA256.txt')]
        assert len(manifests)==1
        declared={}
        for line in archive.read(manifests[0]).decode('utf-8').splitlines():
            digest,name=line.split('  ',1)
            assert name not in declared
            declared[name]=digest
            assert hashlib.sha256(archive.read(name)).hexdigest()==digest,name
        assert set(names)==set(declared)|{manifests[0]}, 'Manifest does not cover every file'
        for filename in ('复现报告.tex','复现报告.pdf'):
            matches=[name for name in names if name.endswith('/'+filename)]
            assert len(matches)==1
            data=archive.read(matches[0])
            checks['embedded_'+filename+'_sha256']=hashlib.sha256(data).hexdigest()
            if report_directory is not None:
                assert data==(report_directory/filename).read_bytes(),filename
        checks['zip_entries']=len(names)
        checks['manifest_files_verified']=len(declared)
    checks.update(zip_sha256=sha(path),zip_bytes=path.stat().st_size)
    return checks


def verify(zip_path=None):
    checks = {}
    counts = Counter()
    validation_only = 0
    run_records = {}
    expected_counts={'core':18,'ablation':6,'official_anchor':5,'extension':12}
    expected_validation=10
    followup=ROOT/'analysis/followup_summary.json'
    if followup.exists():
        extra=read_json(followup)
        expected_counts['followup_20261101']=extra['new_runs']['formal']
        expected_validation+=extra['new_runs']['validation_only']
        locked=ROOT/'runs/followup_20261101'
        for name in ('previous_result_sha256.json','source_hashes.json'):
            records=read_json(locked/name)
            for relative,digest in records.items():
                assert sha(ROOT/relative)==digest, relative
            checks[name.removesuffix('.json')+'_verified']=len(records)
        acceptance=read_json(ROOT/'research/followup_acceptance.json')
        for relative,digest in acceptance['test_sources_sha256'].items():
            assert sha(ROOT/relative)==digest, relative
        checks['numerical_evaluator_masking_acceptance_tests']=acceptance['tests_passed']
    elif (ROOT/'runs/followup_20261101').exists():
        raise RuntimeError('Follow-up exists but its completed summary is missing; do not finalize delivery yet.')
    for group in expected_counts:
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
    assert dict(counts) == expected_counts, counts
    assert validation_only == expected_validation
    checks['formal_runs'] = dict(counts)
    checks['formal_runs_total'] = sum(counts.values())
    checks['validation_only_search_runs'] = validation_only
    checks['immutable_result_sha256'] = run_records
    manifest = read_json(ROOT / 'research/paper_source_manifest.json')
    for source in manifest['sources']:
        assert sha(ROOT / 'research' / source['local_path']) == source['sha256']
    checks['paper_and_author_sources_verified'] = len(manifest['sources'])
    for file in ('LightCCF.py','LightCCFMasked.py','ReproductionRunner.py'):
        directory = 'helpers' if file == 'ReproductionRunner.py' else 'models/general'
        assert (ROOT / 'extensions' / file).read_bytes() == (ROOT / 'framework/ReChorus/src' / directory / file).read_bytes()
    acceptance = read_json(ROOT / 'research/loss_validation_results.json')
    assert sha(ROOT / 'extensions/LightCCF.py') == acceptance['model_sha256_at_test']
    checks['model_matches_original_gradient_acceptance'] = True
    source = ROOT / '复现报告.tex'
    tex = source.read_text(encoding='utf-8')
    blocks=read_json(ROOT/'analysis/latex_blocks.json')
    if (ROOT/'analysis/followup_latex_blocks.json').exists():
        blocks.update(read_json(ROOT/'analysis/followup_latex_blocks.json'))
    actual_names=re.findall(r'% BEGIN AUTO (\w+)',tex)
    assert len(actual_names)==len(set(actual_names)), 'Duplicate numerical block'
    for name in actual_names:
        value=blocks[name]
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
    checks['auto_blocks_verified'] = len(actual_names)
    checks['tex_sha256'] = sha(source)
    pdf = source.with_suffix('.pdf')
    checks['pdf_sha256'] = sha(pdf)
    compiled=read_json(ROOT/'analysis/report_compile.json')
    assert compiled['source_name']==source.name
    assert compiled['tex_sha256']==checks['tex_sha256'], 'Source changed since its successful compilation'
    assert compiled['pdf_sha256']==checks['pdf_sha256'], 'PDF changed since its successful compilation'
    checks['report_matches_successful_compile']=True
    from pypdf import PdfReader
    reader = PdfReader(pdf)
    checks['pages'] = len(reader.pages)
    page_texts=[p.extract_text() for p in reader.pages]
    all_text='\n'.join(page_texts)
    assert 'github.com/liutq25/lightccf-rechorus-reproduction' in re.sub(r'\s+','',all_text)
    uris=[]
    for page in reader.pages:
        for reference in page.get('/Annots',[]):
            action=reference.get_object().get('/A',{})
            if '/URI' in action:
                uris.append(str(action['/URI']))
    checks['code_repository_link_clickable']='https://github.com/liutq25/lightccf-rechorus-reproduction' in uris
    assert checks['code_repository_link_clickable']
    visible=[]
    for number,page_text in enumerate(page_texts,1):
        lines=page_text.splitlines()
        cleaned=[]
        for line in lines:
            compact=re.sub(r'\s+','',line)
            if 'LightCCF论文复现与边界分析' in compact and '机器学习课程' in compact:
                continue
            cleaned.append(line)
        if cleaned and cleaned[-1].strip()==str(number):
            cleaned.pop()
        visible.append('\n'.join(cleaned))
    visible_text='\n'.join(visible)
    checks['full_visible_chinese_characters_excluding_repeated_headers']=count_han(visible_text)
    checks['full_visible_mixed_token_estimate']=count_han(visible_text)+len(re.findall(r'[A-Za-z][A-Za-z0-9_@./%-]*|[0-9]+(?:\.[0-9]+)?',visible_text))
    checks['full_text_count_scope']='All extracted visible report text, including abstract, figures, tables, references and appendices; repeated running header and page numbers excluded. Mixed count is a conservative regex estimate, not a Word measurement.'
    assert count_han(visible_text)<=5000, 'Condense the full report, not only the body.'
    assert checks['full_visible_mixed_token_estimate']<=5000, 'Condense the conservative full visible mixed-text estimate.'
    checks['pdf_text_extractable'] = '邻域聚合' in all_text and 'MovieLens' in all_text
    if zip_path:
        checks.update(verify_archive(zip_path,ROOT))
    return checks


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zip', type=Path)
    parser.add_argument('--zip-only',action='store_true',help='Verify all archived hashes using only the standard library; no datasets, checkpoints or PDF runtime needed.')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.zip_only and args.zip is None:
        parser.error('--zip-only requires --zip')
    checks=verify_archive(args.zip) if args.zip_only else verify(args.zip)
    output=args.output or (None if args.zip_only else ROOT/'analysis/final_delivery_audit.json')
    if output is not None:
        output.write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in checks.items() if k != 'immutable_result_sha256'}, ensure_ascii=False, indent=2))
