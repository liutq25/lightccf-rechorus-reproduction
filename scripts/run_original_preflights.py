"""Serial full-batch probes, stopping only the probe if host RAM becomes scarce."""
from pathlib import Path
import ctypes
import hashlib
import json
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/paper_expansion/resource_author'
MODELS=('LightCCF','MFBPR','LightGCN','NGCF','SGL','NCL','SimGCL','DirectAU',
        'CVGA','LightGCL','CGCL','RecDCL','BIGCF','SCCF')


class Memory(ctypes.Structure):
    _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong),('total',ctypes.c_ulonglong),
              ('available',ctypes.c_ulonglong),('page_total',ctypes.c_ulonglong),
              ('page_available',ctypes.c_ulonglong),('virtual_total',ctypes.c_ulonglong),
              ('virtual_available',ctypes.c_ulonglong),('extended',ctypes.c_ulonglong)]


def available():
    value=Memory(); value.length=ctypes.sizeof(value)
    assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value))
    return value.available


def main():
    OUT.mkdir(exist_ok=True)
    summary=[]
    for dataset in ('douban-book','tmall','amazon-book'):
        for model in MODELS:
            path=OUT/(dataset+'_'+model+'.json')
            aliases={('amazon-book','LightCCF'):'amazon_lightccf.json',
                     ('douban-book','RecDCL'):'douban_recdcl.json'}
            if (dataset,model) in aliases:
                path=OUT/aliases[dataset,model]
            # Never repeat the deliberately stopped high-memory Amazon probe.
            if dataset=='amazon-book' and model=='RecDCL':
                path=OUT/'amazon_recdcl_attempt.json'
            if path.exists():
                value=json.loads(path.read_text(encoding='utf-8-sig'))
            else:
                cmd=[sys.executable,str(ROOT/'scripts/preflight_original_models.py'),'--dataset',dataset,
                     '--model',model,'--output',str(path)]
                if model=='RecDCL':
                    cmd+=['--attempt-over-budget']
                begin=time.perf_counter(); minimum=available(); low_count=0
                with path.with_suffix('.log').open('w',encoding='utf-8') as log:
                    child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                    reason=None
                    while child.poll() is None:
                        free=available(); minimum=min(minimum,free)
                        low_count=low_count+1 if free<2*1024**3 else 0
                        if low_count>=2:
                            reason='terminated_for_host_memory_pressure'; break
                        if time.perf_counter()-begin>300:
                            reason='probe_time_limit_not_a_feasibility_verdict'; break
                        time.sleep(1)
                    if reason:
                        child.terminate(); child.wait(timeout=30)
                        value=dict(status=reason,probe_only=True,dataset=dataset,model=model,
                            wall_seconds=time.perf_counter()-begin,minimum_available_host_bytes=minimum,
                            reason='Stopped only this diagnostic. This does not prove that CPU, exact memory optimizations, or longer execution are impossible.',
                            command=cmd,source_sha256=hashlib.sha256((ROOT/'scripts/preflight_original_models.py').read_bytes()).hexdigest())
                        path.write_text(json.dumps(value,indent=2),encoding='utf-8')
                    elif path.exists():
                        value=json.loads(path.read_text(encoding='utf-8-sig'))
                    else:
                        value=dict(status='probe_process_failed',exit_code=child.returncode,probe_only=True,
                                   dataset=dataset,model=model,wall_seconds=time.perf_counter()-begin)
                        path.write_text(json.dumps(value,indent=2),encoding='utf-8')
            summary.append(dict(dataset=dataset,model=model,status=value['status'],
                                path=path.relative_to(ROOT).as_posix()))
            (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
            print(json.dumps(summary[-1]),flush=True)


if __name__=='__main__':
    main()
