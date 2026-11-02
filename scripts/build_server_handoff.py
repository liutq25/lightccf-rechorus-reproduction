"""Freeze completed evidence and refresh the detailed Linux-server handoff document.

Does not modify training code, source locks, reports or running checkpoints.
"""
from collections import Counter
from datetime import datetime,timezone,timedelta
from pathlib import Path
import hashlib
import json
import re
import statistics
import subprocess

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'handoff'
GROUPS=('core','ablation','official_anchor','extension','followup_20261101')
SOURCE_FOLDERS=('LightCCF-main','ID-GRec','DiffRec','vae_cf','SIGIR23-VGCL','BIGCF-full-sample')


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b''):
            value.update(chunk)
    return value.hexdigest()


def write(path,value):
    # Explicit LF is useful when moving the frozen files between operating systems.
    with path.open('w',encoding='utf-8',newline='\n') as stream:
        stream.write(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def main():
    OUT.mkdir(exist_ok=True)
    created=datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
    previous=read(ROOT/'runs/paper_expansion/previous_result_sha256.json')
    assert len(previous)==96
    for name,digest in previous.items():
        assert sha(ROOT/name)==digest,name
    for name,digest in read(ROOT/'runs/paper_expansion/plan_sha256.json').items():
        assert sha(ROOT/name)==digest,name
    counts=Counter()
    completed=[]
    folders=[]
    for group in GROUPS:
        for path in sorted((ROOT/'runs'/group).rglob('result.json')):
            value=read(path)
            assert value['status']=='complete',str(path)
            kind='validation_only' if value.get('pilot') else 'formal'
            counts[kind]+=1
            completed.append(dict(path=path.relative_to(ROOT).as_posix(),track=group,kind=kind,
                                  sha256=sha(path)))
        folders.append(ROOT/'runs'/group)
    assert counts==Counter(formal=71,validation_only=25),counts
    datasets={}
    active=[]
    metrics=('Recall@10','NDCG@10','Recall@20','NDCG@20')
    for dataset in ('douban-book','tmall','amazon-book'):
        runs=[]
        for folder in sorted((ROOT/'runs/paper_expansion/main'/dataset).glob('seed_*')):
            result_path=folder/'result.json'
            if not result_path.exists():
                active.append(folder.relative_to(ROOT).as_posix()); continue
            result=read(result_path)
            if result['status']!='complete':
                active.append(folder.relative_to(ROOT).as_posix()); continue
            assert not result['diagnostic']
            assert sha(folder/'config.json')==result['config_sha256']
            assert sha(folder/'best.pt')==result['best_checkpoint_sha256']
            history=[json.loads(s) for s in (folder/'history.jsonl').read_text(encoding='utf-8').splitlines()]
            assert [x['epoch'] for x in history]==list(range(1,result['epochs']+1))
            best=max(history,key=lambda row:row['test']['Recall@20'])
            assert best['epoch']==result['best_epoch'] and best['test']==result['test']
            runs.append(dict(seed=result['seed'],epochs=result['epochs'],best_epoch=result['best_epoch'],
                test=result['test'],wall_seconds=result['wall_seconds'],path=result_path.relative_to(ROOT).as_posix(),
                result_sha256=sha(result_path)))
            folders.append(folder)
            completed.append(dict(path=result_path.relative_to(ROOT).as_posix(),track='paper_expansion/main',
                                  kind='formal',sha256=sha(result_path)))
        datasets[dataset]=dict(completed_seeds=[x['seed'] for x in runs],runs=runs,metrics={})
        if runs:
            datasets[dataset]['metrics']={metric:dict(mean=statistics.mean(x['test'][metric] for x in runs),
                sample_sd=statistics.stdev(x['test'][metric] for x in runs) if len(runs)>1 else None)
                for metric in metrics}
    snapshot=dict(created_at=created,timezone='Asia/Shanghai',
        source_project_root=str(ROOT),code_commit_at_capture=subprocess.check_output(
            ['git','rev-parse','HEAD'],cwd=ROOT,encoding='utf-8').strip(),
        old_stage=dict(formal=71,validation_only=25,result_records_locked=96),
        original_main=datasets,new_formal_completed=sum(len(v['runs']) for v in datasets.values()),
        active_directories_not_frozen=active,windows_queue=read(ROOT/'runs/paper_expansion/queue_status.json'),
        completed_result_records=completed,original_dataset_audit=read(ROOT/'research/original_dataset_audit.json'),
        old_report_audit={k:v for k,v in read(ROOT/'analysis/final_delivery_audit.json').items()
                          if k!='immutable_result_sha256'},
        limitations=['This is a timestamped local snapshot; check current progress before assigning server jobs.',
            'The server GPU count and per-device VRAM have not been inspected by this assistant.',
            'Linux runtime validation and the remaining paper experiments are not declared complete.'])
    snapshot_path=OUT/'交接状态快照_20261102.json'
    write(snapshot_path,snapshot)
    paths=set()
    excluded_mutable={'analysis/original_paper_expansion.json'}
    tracked=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).split(b'\0')
    for name in tracked:
        if not name:
            continue
        relative=name.decode('utf-8')
        path=ROOT/relative
        if (path.is_file() and relative not in excluded_mutable
            and not relative.startswith('handoff/') and relative!='服务器交接文档.md'
            and not relative.startswith('tmp/')):
            paths.add(path)
    for prefix in ('scripts','extensions','tests','configs'):
        paths.update(p for p in (ROOT/prefix).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    for suffix in ('*.pdf','*.tex'):
        paths.update(ROOT.glob(suffix))
    paths.add(ROOT/'LaTeX报告说明.md')
    paths.add(ROOT/'research/course_requirements_full.txt')
    paths.add(ROOT/'research/requirements_checklist.md')
    for source in read(ROOT/'research/paper_source_manifest.json')['sources']:
        path=ROOT/'research'/source['local_path']
        assert path.is_file(),str(path)
        paths.add(path)
    for p in (ROOT/'framework/ReChorus/src').rglob('*.py'):
        paths.add(p)
    for dataset in ('Grocery_and_Gourmet_Food','ML_1MTOPK'):
        paths.update((ROOT/'framework/ReChorus/data'/dataset).glob('*.csv'))
    for name in SOURCE_FOLDERS:
        for p in (ROOT/'vendor'/name).rglob('*'):
            if p.is_file() and '.git' not in p.parts and '__pycache__' not in p.parts and p.suffix.lower() in ('.py','.md','.txt','.ipynb'):
                # ID-GRec's unused example datasets are not required inputs.
                if name!='LightCCF-main' and 'dataset' in p.relative_to(ROOT/'vendor'/name).parts:
                    continue
                paths.add(p)
    for dataset in ('douban-book','tmall','amazon-book'):
        base=ROOT/'vendor/LightCCF-main/dataset'/dataset
        paths.update(base/n for n in ('train.txt','test.txt','pre_Adj.npz'))
    paths.update((ROOT/'data/author_clean_graphs').rglob('*.npz'))
    paths.add(ROOT/'data/raw/ml-1m.zip')
    folders.extend(ROOT/'runs/paper_expansion'/name for name in ('acceptance','resource_current','resource_author'))
    for folder in folders:
        paths.update(p for p in folder.rglob('*') if p.is_file() and p.suffix not in ('.tmp','.lock'))
    for name in ('previous_result_sha256.json','plan_sha256.json'):
        paths.add(ROOT/'runs/paper_expansion'/name)
    paths.add(ROOT/'handoff/verify_handoff.py')
    paths.add(ROOT/'handoff/服务器任务登记模板.csv')
    paths.add(snapshot_path)
    entries=[]
    for path in sorted(paths,key=lambda p:p.relative_to(ROOT).as_posix()):
        assert path.is_file(),str(path)
        before=path.stat()
        digest=sha(path)
        after=path.stat()
        assert before.st_size==after.st_size and before.st_mtime_ns==after.st_mtime_ns,'File changed during inventory: '+str(path)
        entries.append(dict(path=path.relative_to(ROOT).as_posix(),bytes=after.st_size,sha256=digest))
    manifest=dict(created_at=created,scope='Stable project sources, scientific inputs and completed evidence including checkpoints',
        files=entries,total_bytes=sum(v['bytes'] for v in entries),
        excluded_active_directories=active,excluded_live_summaries=sorted(excluded_mutable),
        notes=['Git metadata, Windows wheels, temporary files and active checkpoints are not covered by this scientific-input manifest.',
               'Copy whole repositories including .git separately if source-fetch/git-status tools will be used.',
               'The human handoff document has a generated status block and is not part of its own input inventory.'])
    write(OUT/'输入文件校验_20261102.json',manifest)
    lines=[f"**快照时间：{created}（北京时间）**。代码基点 `{snapshot['code_commit_at_capture']}`。",
        '',f"旧阶段仍为71次正式、25次仅验证；本阶段新增原文主实验已完成 **{snapshot['new_formal_completed']}/15**。",'',
        '| 原数据 | 已完成种子 | R@20均值 ± 样本SD | N@20均值 ± 样本SD |',
        '|---|---|---:|---:|']
    for dataset,record in datasets.items():
        def formatted(metric):
            if not record['runs']:
                return '未完成'
            v=record['metrics'][metric]
            return f"{v['mean']:.5f} ± {v['sample_sd']:.5f}" if v['sample_sd'] is not None else f"{v['mean']:.5f}（仅1次）"
        lines.append('| '+dataset+' | '+(', '.join(map(str,record['completed_seeds'])) or '无')+' | '+formatted('Recall@20')+' | '+formatted('NDCG@20')+' |')
    queue=snapshot['windows_queue']
    current=queue.get('current',{})
    lines+=['',f"本机队列状态：`{queue['status']}`；当前登记任务：`{current.get('dataset','未知')}` / seed `{current.get('seed','未知')}`。",
        '',f"稳定输入清单：**{len(entries)} 个文件，约 {manifest['total_bytes']/1024**3:.2f} GiB**，包含已完成模型权重；不含Windows安装包和活跃checkpoint。",
        '', '注意：Tmall不足5次时上表是阶段性均值；新旧环境同种子重复不合并增加独立样本数。']
    document=ROOT/'服务器交接文档.md'
    content=document.read_text(encoding='utf-8')
    updated,count=re.subn(r'<!-- HANDOFF_SNAPSHOT_START -->.*?<!-- HANDOFF_SNAPSHOT_END -->',
        lambda _: '<!-- HANDOFF_SNAPSHOT_START -->\n'+'\n'.join(lines)+'\n<!-- HANDOFF_SNAPSHOT_END -->',content,flags=re.S)
    assert count==1
    with document.open('w',encoding='utf-8',newline='\n') as stream:
        stream.write(updated)
    print(json.dumps(dict(snapshot_time=created,old_results_locked=96,new_main_completed=snapshot['new_formal_completed'],
                          inventory_files=len(entries),inventory_gib=manifest['total_bytes']/1024**3,active=active),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
