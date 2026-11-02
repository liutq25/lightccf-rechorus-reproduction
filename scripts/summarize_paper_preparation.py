"""Publish concise preparation evidence without copying complete issue discussions."""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    raw=ROOT/'research/noise_protocol_audit.json'
    value=json.loads(raw.read_text(encoding='utf-8'))
    for repo,record in value['repositories'].items():
        issues=[]
        for item in record['public_issues']:
            issues.append(dict(number=item['number'],url=item['url'],title=item['title'],
                keyword_match=item['keyword_match'],body_sha256=hashlib.sha256((item['body'] or '').encode()).hexdigest(),
                comments=[dict(url=c['url'],body_sha256=hashlib.sha256(c['body'].encode()).hexdigest()) for c in item['comments']]))
        record['public_issues']=issues
    value['private_raw_audit_sha256']=sha(raw)
    value['html_supplement']=dict(url='https://github.com/BlueGhostYi/ID-GRec/issues/1',
        reviewed='2026-11-02',topic='Dataset format and random 80/20 splitting, not adversarial perturbation',
        comments_reviewed=3,reason='API rate limit interrupted the automated request; the public HTML conversation was then read.')
    (ROOT/'research/noise_protocol_public_summary.json').write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    sources=json.loads((ROOT/'research/original_sources_environment.json').read_text(encoding='utf-8'))
    idroot=ROOT/'vendor/ID-GRec'
    rows=[]
    for model in ('MFBPR','NGCF','LightGCN','SGL','NCL','SimGCL','DirectAU','CVGA','LightGCL','CGCL','RecDCL','BIGCF','SCCF'):
        code=idroot/'models'/(model+'.py'); config=idroot/'configure'/(model+'.txt')
        text=code.read_text(encoding='utf-8')
        rows.append(dict(model=model,source='https://github.com/BlueGhostYi/ID-GRec/blob/'+sources['sources']['ID-GRec']['commit']+'/models/'+model+'.py',
            code_sha256=sha(code),released_config_sha256=sha(config) if config.exists() else None,
            released_config_available=config.exists(),
            trainer='universal' if 'trainer.universal_trainer(' in text else 'custom',
            status='source_audited_formal_original_dataset_baseline_run_pending',
            common_paper_overrides=dict(embedding_size=2048 if model=='RecDCL' else 64,learn_rate=.001,reg_lambda=.0001)))
        if model=='BIGCF' and not config.exists():
            recovered=ROOT/'vendor/BIGCF-full-sample/configure/BIGCF.txt'
            rows[-1]['recovered_config']=dict(source=sources['sources']['BIGCF-full-sample']['url'],
                commit=sources['sources']['BIGCF-full-sample']['commit'],sha256=sha(recovered),
                note='Same model config keys and math in the author full-sample integration; still a Yelp default, not a recovered LightCCF Table 5 dataset-specific optimum.')
    for name,repo in [('Mult-VAE','vae_cf'),('DiffRec','DiffRec'),('VGCL','SIGIR23-VGCL')]:
        rows.append(dict(model=name,repository=sources['sources'][repo]['url'],commit=sources['sources'][repo]['commit'],
                         status='primary_source_fetched_integration_and_training_pending'))
    output=dict(baselines=rows,limitations=[
        'Published ID-GRec configs mainly target Yelp2018, not a complete dataset-specific set of Table 5 settings.',
        'The pinned ID-GRec release omits configure/BIGCF.txt; a compatible config was recovered from the author BIGCF-full-sample repository. It is not evidence of the exact LightCCF experiment settings.',
        'Dedicated LightCCF and ID-GRec use different test Recall selection indexes and early-stop comparisons.',
        'NCL and SGL custom trainers calculate early-stop flags but do not break their loops on those flags in the pinned release.',
        'RecDCL target histories are mutable tensors outside state_dict; any resumable training wrapper must save them explicitly.',
        'VGCL primary repository says its original paper results use TensorFlow and its PyTorch version differs; the backend used by LightCCF Table 5 is not stated.',
        'A resource probe is not model convergence, baseline accuracy validation, or a full paper reproduction.'])
    (ROOT/'research/baseline_protocol_audit.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    print('Recorded 16 primary baseline sources and unresolved protocol/integration details.')


if __name__=='__main__':
    main()
