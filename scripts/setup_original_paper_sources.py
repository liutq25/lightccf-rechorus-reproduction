"""Fetch the fixed primary-source repositories used for original-paper expansion."""
from pathlib import Path
import json
import subprocess

ROOT=Path(__file__).resolve().parents[1]
SPARSE={'ID-GRec':['models','configure','utility'], 'DiffRec':['DiffRec'],
        'vae_cf':[], 'SIGIR23-VGCL':['torch_version'],
        'BIGCF-full-sample':['models','configure','utility']}


def run(*args):
    return subprocess.check_output(args,encoding='utf-8').strip()


def main():
    manifest=json.loads((ROOT/'research/original_sources_environment.json').read_text(encoding='utf-8'))
    for name,entry in manifest['sources'].items():
        path=ROOT/'vendor'/name
        if not path.exists():
            subprocess.run(['git','clone','--config','core.autocrlf=false','--filter=blob:none',
                            '--no-checkout',entry['url']+'.git',str(path)],check=True)
            if SPARSE[name]:
                subprocess.run(['git','-C',str(path),'sparse-checkout','set',*SPARSE[name]],check=True)
            subprocess.run(['git','-C',str(path),'checkout','--detach',entry['commit']],check=True)
        assert run('git','-C',str(path),'rev-parse','HEAD')==entry['commit'],name+' is at a different commit; inspect before changing it'
        assert not run('git','-C',str(path),'status','--porcelain','--untracked-files=no'),name+' has local tracked edits'
        print(name+': '+entry['commit'])


if __name__=='__main__':
    main()
