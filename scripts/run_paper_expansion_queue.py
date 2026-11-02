"""Serial, restartable main-result queue with discrepancy gates and provenance locks."""
from pathlib import Path
import argparse
import ctypes
import datetime
import hashlib
import json
import msvcrt
import os
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'runs/paper_expansion'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2),encoding='utf-8')
    temp.replace(path)


def now():
    return datetime.datetime.now().astimezone().isoformat()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--materialize',action='store_true',help='Lock the queue without starting it')
    p.add_argument('--after-pid',type=int,help='Wait for this already-running diagnostic process to exit before allocating CUDA')
    a=p.parse_args()
    BASE.mkdir(exist_ok=True)
    plan=dict(stage='original three datasets LightCCF, five seeds',
        source_sha256=sha(ROOT/'scripts/run_original_lightccf.py'),
        acceptance_sha256=sha(ROOT/'research/original_runner_acceptance.json'),
        discrepancy_gate='Stop for audit when any completed seed differs by more than 5% relative from a paper main metric. This is an operational audit threshold, not a statistical test.',
        jobs=[dict(dataset=d,seed=s,output=f'runs/paper_expansion/main/{d}/seed_{s}')
              for d in ('douban-book','tmall','amazon-book') for s in range(2023,2028)])
    locked=ROOT/'configs/paper_expansion_main.json'
    if locked.exists():
        assert json.loads(locked.read_text(encoding='utf-8'))==plan,'Queue or training source changed; audit before making a new plan'
    else:
        write(locked,plan)
    if a.materialize:
        print('Locked 15 main-result runs and discrepancy gates.'); return
    proof=json.loads((ROOT/'research/original_runner_acceptance.json').read_text(encoding='utf-8'))
    assert proof['status']=='passed' and proof['evidence_sha256']['scripts/run_original_lightccf.py']==plan['source_sha256']
    for manifest in ('previous_result_sha256.json','plan_sha256.json'):
        for path,digest in json.loads((BASE/manifest).read_text(encoding='utf-8')).items():
            assert sha(ROOT/path)==digest,path
    # A kernel file lock is released automatically even if this process crashes.
    with (BASE/'queue.lock').open('a+b') as lock:
        if lock.tell()==0:
            lock.write(b'0'); lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        progress=dict(status='running',pid=os.getpid(),started=now(),queue_sha256=sha(locked),completed=[])
        status=BASE/'queue_status.json'
        write(status,progress)
        if a.after_pid:
            kernel=ctypes.windll.kernel32
            kernel.OpenProcess.restype=ctypes.c_void_p
            handle=kernel.OpenProcess(0x00100000,False,a.after_pid)
            if handle:
                progress.update(status='waiting_for_preflight',preflight_pid=a.after_pid)
                write(status,progress)
                try:
                    while kernel.WaitForSingleObject(ctypes.c_void_p(handle),1000)==258:
                        pass
                finally:
                    kernel.CloseHandle(ctypes.c_void_p(handle))
            progress.update(status='running',updated=now())
            write(status,progress)
        paper=json.loads((ROOT/'research/paper_table5.json').read_text(encoding='utf-8'))
        try:
            for job in plan['jobs']:
                assert sha(ROOT/'scripts/run_original_lightccf.py')==plan['source_sha256']
                folder=ROOT/job['output']; result=folder/'result.json'
                progress.update(current=job,updated=now())
                write(status,progress)
                complete=result.exists() and json.loads(result.read_text(encoding='utf-8')).get('status')=='complete'
                if not complete:
                    cmd=[sys.executable,str(ROOT/'scripts/run_original_lightccf.py'),
                         '--dataset',job['dataset'],'--seed',str(job['seed']),'--output',str(folder)]
                    if folder.exists() and any(folder.iterdir()):
                        cmd+=['--resume']
                    folder.mkdir(parents=True,exist_ok=True)
                    # Keep launcher logs outside the training directory until it
                    # initializes its config, so an empty first run is unambiguous.
                    logs=BASE/'queue_logs'; logs.mkdir(exist_ok=True)
                    with (logs/(job['dataset']+'_'+str(job['seed'])+'.log')).open('a',encoding='utf-8') as log:
                        subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
                value=json.loads(result.read_text(encoding='utf-8'))
                assert value['status']=='complete' and not value['diagnostic']
                config=json.loads((folder/'config.json').read_text(encoding='utf-8'))
                assert config['wrapper_sha256']==plan['source_sha256']
                assert config['torch']=='2.1.0+cu121' and config['numpy']=='1.24.3'
                ref=paper['rows']['LightCCF'][('douban-book','tmall','amazon-book').index(job['dataset'])]
                differences={m:100*(value['test'][m]/x-1) for m,x in zip(paper['metric_order'],ref)}
                progress['completed'].append(dict(**job,result_sha256=sha(result),relative_difference_percent=differences))
                subprocess.run([sys.executable,str(ROOT/'scripts/analyze_original_results.py')],cwd=ROOT,check=True)
                if any(abs(x)>5 for x in differences.values()):
                    progress.update(status='needs_discrepancy_audit',updated=now(),
                        reason='Completed seed exceeded the locked audit threshold; inspect data, graph, loss, selection and source before continuing.')
                    write(status,progress)
                    print(json.dumps(progress,indent=2),flush=True); return
            progress.update(status='main_complete',updated=now())
            write(status,progress)
        except Exception as e:
            progress.update(status='error_requires_audit',error=repr(e),updated=now())
            write(status,progress)
            raise
        finally:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
            lock.seek(0); msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)


if __name__=='__main__':
    main()
