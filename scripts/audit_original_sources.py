"""Lock primary baseline sources, installed author environment and public issues."""
from pathlib import Path
import datetime
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import urllib.request
import torch

ROOT=Path(__file__).resolve().parents[1]
REPOS={
    'BIGCF-full-sample':('BlueGhostYi/BIGCF-full-sample','90dbf8241e07e8fc0afd9a43d56f2921a7bf58d3'),
    'ID-GRec':('BlueGhostYi/ID-GRec','10974d8d0d932d0018eda84bf46d2e8504b58ca8'),
    'DiffRec':('YiyanXu/DiffRec','d35a6c63c14cc0b376c138eb0b5c35895d1dcffd'),
    'vae_cf':('dawenl/vae_cf','ad103c506c8897a853968e1b98c30f31cf5e30f6'),
    'SIGIR23-VGCL':('yimutianyang/SIGIR23-VGCL','aaefcec14a9955e2a01c19e089f769d2ee61c02f')}


def git(folder,*args):
    return subprocess.check_output(['git','-C',str(folder),*args],encoding='utf-8').strip()


def main():
    manifest=dict(retrieved_utc=datetime.datetime.utcnow().isoformat()+'Z',sources={})
    for name,(repo,commit) in REPOS.items():
        folder=ROOT/'vendor'/name
        assert git(folder,'rev-parse','HEAD')==commit
        files={}
        for pattern in ('*.py','*.txt','*.md','*.ipynb'):
            for path in folder.rglob(pattern):
                if '.git' not in path.parts:
                    files[path.relative_to(folder).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
        manifest['sources'][name]=dict(url='https://github.com/'+repo,commit=commit,
            source_sha256=files,selection_note='Historical published source available before 2025-05-10; not proven to be the LightCCF authors\' exact experiment snapshot.')
    manifest['environment']=dict(executable=sys.executable,python=sys.version,platform=platform.platform(),
        torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),
        gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,
        packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions()})
    manifest['wheel_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (ROOT/'data/author_wheels').glob('*.whl')}
    (ROOT/'research/original_sources_environment.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('Locked '+str(len(REPOS))+' primary-source commits, source file hashes, author environment and local wheel hashes.')


if __name__=='__main__':
    main()
