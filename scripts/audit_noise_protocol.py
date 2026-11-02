"""Search both cited code repositories and their public issue histories."""
from pathlib import Path
import datetime
import hashlib
import json
import re
import subprocess
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
PATTERN=re.compile(r'adversar|poison|attack|noise.?ratio|inject',re.I)


def main():
    output=dict(retrieved_utc=datetime.datetime.utcnow().isoformat()+'Z',repositories={},
        paper_source='https://arxiv.org/abs/2504.10113',
        paper_specifies=['Inject 10%, 20%, 30% adversarial user-item interactions into training; leave test unchanged.'],
        unspecified=['Attack objective and attacker knowledge','How injected users and items are selected',
                     'Whether ratios refer to original or augmented interaction counts',
                     'Duplicate/previous-positive/held-out-positive rejection rules',
                     'Attack optimization, budget constraints and random seeds'],
        conclusion='No claim of exact Figure 3 reproduction is justified without a defined attack protocol. Random corruption would be a separate reconstruction.',
        caveat='A negative search of released sources and public issues does not prove that private experimental code does not exist.')
    for repo,folder in [('ZzYUuuu/LightCCF','LightCCF-main'),('BlueGhostYi/ID-GRec','ID-GRec')]:
        base=ROOT/'vendor'/folder
        hits=[]; scanned={}
        for path in base.rglob('*'):
            if path.suffix not in ('.py','.md','.sh') or '.git' in path.parts:
                continue
            text=path.read_text(encoding='utf-8',errors='replace')
            scanned[path.relative_to(base).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
            for line_number,line in enumerate(text.splitlines(),1):
                if PATTERN.search(line):
                    hits.append(dict(path=path.relative_to(base).as_posix(),line=line_number,text=line))
        record=dict(url='https://github.com/'+repo,scanned_sha256=scanned,source_matches=hits,
                    public_issues=[],issue_search_complete=False)
        try:
            page=1
            while True:
                url='https://api.github.com/repos/'+repo+'/issues?state=all&per_page=100&page='+str(page)
                request=urllib.request.Request(url,headers={'User-Agent':'LightCCF-Course-Reproduction'})
                payload=urllib.request.urlopen(request,timeout=45).read()
                archive=ROOT/'research'/('issues_'+folder+'_'+str(page)+'.json')
                archive.write_bytes(payload)
                issues=json.loads(payload)
                for item in issues:
                    if 'pull_request' in item:
                        continue
                    request=urllib.request.Request(item['comments_url'],headers={'User-Agent':'LightCCF-Course-Reproduction'})
                    comments=json.loads(urllib.request.urlopen(request,timeout=45).read()) if item['comments'] else []
                    record['public_issues'].append(dict(number=item['number'],url=item['html_url'],
                        title=item['title'],body=item['body'],comments=[dict(url=c['html_url'],body=c['body']) for c in comments],
                        keyword_match=bool(PATTERN.search(item['title']+' '+(item['body'] or '')+' '+str(comments)))))
                if len(issues)<100:
                    record['issue_search_complete']=True
                    break
                page+=1
        except Exception as e:
            record['issue_search_error']=repr(e)
        if (base/'.git').exists():
            record['commit_message_matches']=subprocess.check_output(
                ['git','-C',str(base),'log','--all','--oneline','--regexp-ignore-case',
                 '--grep=adversar\\|poison\\|attack\\|noise'],encoding='utf-8').splitlines()
        output['repositories'][repo]=record
    (ROOT/'research/noise_protocol_audit.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:dict(source_matches=len(v['source_matches']),issues=len(v['public_issues']),
        complete=v['issue_search_complete'],issue_matches=[i['url'] for i in v['public_issues'] if i['keyword_match']])
        for k,v in output['repositories'].items()},indent=2))


if __name__=='__main__':
    main()
