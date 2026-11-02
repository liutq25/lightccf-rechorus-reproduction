"""Original-author LightCCF math/sampling/evaluation with audited checkpoints.

The loop follows utility/trainer.py, including test Recall@20 selection and
cnt > early_stop. Added recording/resume never reads scores to choose configs.
"""
from pathlib import Path
from types import SimpleNamespace
from contextlib import redirect_stdout,redirect_stderr
import argparse
import hashlib
import json
import math
import random
import sys
import time
import traceback
import numpy as np
import scipy.sparse as sp
import torch

ROOT=Path(__file__).resolve().parents[1]
VENDOR=ROOT/'vendor/LightCCF-main'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_write(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    temp.replace(path)


def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),
                cpu=torch.random.get_rng_state(),cuda=torch.cuda.get_rng_state_all())


def restore_rng(value):
    random.setstate(value['python']); np.random.set_state(value['numpy'])
    torch.random.set_rng_state(value['cpu']); torch.cuda.set_rng_state_all(value['cuda'])


def save(path,value):
    temp=path.with_suffix('.tmp')
    torch.save(value,temp)
    temp.replace(path)


def per_user(model,data,tester,batch_size):
    rows=[]
    with torch.no_grad():
        users=list(data.test_dict)
        for start in range(0,len(users),batch_size):
            batch=users[start:start+batch_size]
            scores=model.get_rating_for_test(torch.tensor(batch,dtype=torch.long,device='cuda'))
            rr,cc=[],[]
            for row,user in enumerate(batch):
                positives=data.train_dict.get(user,[])
                rr.extend([row]*len(positives)); cc.extend(positives)
            scores[rr,cc]=-1
            top=torch.topk(scores,k=20).indices.cpu().numpy()
            truth=[data.test_dict[u] for u in batch]
            labels=tester.pred_to_label(top,truth)
            for row,user in enumerate(batch):
                n=len(truth[row])
                values=dict(user_id=int(user),train_degree=len(data.train_dict.get(user,[])),
                            test_degree=n)
                for k in (10,20):
                    discount=1/np.log2(np.arange(2,k+2))
                    values['Recall@'+str(k)]=float(labels[row,:k].sum()/n)
                    values['NDCG@'+str(k)]=float((labels[row,:k]*discount).sum()/discount[:min(k,n)].sum())
                rows.append(values)
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['douban-book','tmall','amazon-book'],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=2023)
    p.add_argument('--epochs',type=int,default=600)
    p.add_argument('--layers',type=int,choices=[0,1,2,3])
    p.add_argument('--tau',type=float)
    p.add_argument('--alpha',type=float)
    p.add_argument('--na-mode',choices=['full','self','ui'],default='full')
    p.add_argument('--graph-mode',choices=['released','clean'])
    p.add_argument('--diagnostic',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--interrupt-after',type=int,help='Acceptance-only controlled interruption')
    a=p.parse_args()
    a.output=a.output.resolve()
    if a.interrupt_after and not a.diagnostic:
        p.error('--interrupt-after requires --diagnostic')
    config=dict(dataset=a.dataset,seed=a.seed,max_epochs=a.epochs,
        layers=a.layers if a.layers is not None else (3 if a.dataset=='amazon-book' else 0),
        tau=a.tau if a.tau is not None else {'douban-book':.28,'tmall':.24,'amazon-book':.22}[a.dataset],
        alpha=a.alpha if a.alpha is not None else (1. if a.dataset=='douban-book' else 5.),
        na_mode=a.na_mode,graph_mode=a.graph_mode or ('clean' if a.dataset=='amazon-book' else 'released'),
        embedding_size=64,train_batch_size=2048 if a.dataset=='douban-book' else 4096,
        test_batch_size=2048,learn_rate=.001,reg_lambda=1e-4,early_stop=10,
        selection='author test Recall@20',selection_uses_test=True,diagnostic=a.diagnostic,
        author_commit='45645ef15fc08e858ffc81e1b2530b4b1805ad34',
        python=sys.version,torch=torch.__version__,numpy=np.__version__,
        wrapper_sha256=sha(Path(__file__)))
    audit=json.loads((ROOT/'research/original_dataset_audit.json').read_text(encoding='utf-8'))[a.dataset]
    if config['graph_mode']=='released' and not audit['cache_matches_train_only_graph'] and not a.diagnostic:
        raise RuntimeError('Released graph includes nontrain edges; use clean graph or label a diagnostic explicitly')
    for phase in ('train','test'):
        path=VENDOR/'dataset'/a.dataset/(phase+'.txt')
        assert sha(path)==audit['sha256'][phase+'.txt']
    config['data_sha256']={phase:audit['sha256'][phase+'.txt'] for phase in ('train','test')}
    graph=(ROOT/audit['clean_graph_path'] if config['graph_mode']=='clean'
           else VENDOR/'dataset'/a.dataset/'pre_Adj.npz')
    config['graph_sha256']=sha(graph)
    config['author_source_sha256']={str(v.relative_to(VENDOR)).replace('\\','/'):sha(v)
        for v in [VENDOR/'LightCCF.py',VENDOR/'utility/losses.py',VENDOR/'utility/trainer.py',
                  VENDOR/'utility/data_loader.py',VENDOR/'utility/tools.py',VENDOR/'utility/tester.py']}
    if a.output.exists() and any(a.output.iterdir()):
        if not a.resume:
            raise RuntimeError('Output exists; audit it and use --resume explicitly')
        previous=json.loads((a.output/'config.json').read_text(encoding='utf-8'))
        assert previous==config,'Resume configuration or training source changed'
        if (a.output/'result.json').exists():
            old=json.loads((a.output/'result.json').read_text(encoding='utf-8'))
            if old['status']=='complete':
                print('Already complete'); return
    else:
        a.output.mkdir(parents=True,exist_ok=True)
        json_write(a.output/'config.json',config)
    torch.set_num_threads(4)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    sys.path.insert(0,str(VENDOR))
    from utility.data_loader import Data
    from LightCCF import LightCCF
    import utility.losses as losses
    import utility.tools as tools
    import utility.tester as tester
    args=SimpleNamespace(dataset=a.dataset,dataset_path=str(VENDOR/'dataset')+'/',
        dataset_type='.txt',sparsity_test=0,embedding_size=64,reg_lambda=1e-4,
        ssl_lambda=config['alpha'],tau=config['tau'],gcn_layer=config['layers'],
        encoder='MF' if config['layers']==0 else 'LightGCN',top_K='[10,20]',
        train_batch_size=config['train_batch_size'],test_batch_size=2048)
    started=time.perf_counter()
    with (a.output/'reader.log').open('a',encoding='utf-8') as log,redirect_stdout(log):
        data=Data(args)
    if config['graph_mode']=='clean':
        data.sparse_adjacency_matrix=lambda:sp.load_npz(graph)
    if a.na_mode!='full':
        if a.na_mode=='ui':
            losses.get_neighbor_aggregate_loss=lambda u,i,t:losses.get_InfoNCE_loss(u,i,t)
        else:
            losses.get_neighbor_aggregate_loss=lambda u,i,t:losses.get_InfoNCE_loss(u,u,t)+losses.get_InfoNCE_loss(i,i,t)
    model=LightCCF(args,data,torch.device('cuda')).cuda()
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    best_recall=0.; best_epoch=0; best_metrics=None; count=0; start_epoch=0
    accumulated_wall=0.
    if a.resume:
        state=torch.load(a.output/'latest.pt',map_location='cpu')
        model.load_state_dict(state['model_state']); optimizer.load_state_dict(state['optimizer_state'])
        restore_rng(state['rng'])
        best_recall=state['best_recall']; best_epoch=state['best_epoch']
        best_metrics=state['best_metrics']; count=state['count']; start_epoch=state['epoch']
        accumulated_wall=state['accumulated_wall']
    torch.cuda.reset_peak_memory_stats()
    # A completed last epoch may precede an interrupted post-training export.
    # Resume that export without taking another optimizer step.
    stop=count>10
    epoch=start_epoch-1
    for epoch in range(start_epoch,start_epoch if stop else a.epochs):
        begin=time.perf_counter()
        model.train()
        pairs=data.random_create_user_pos_neg()
        users=torch.Tensor(pairs[:,0]).long().cuda()
        pos=torch.Tensor(pairs[:,1]).long().cuda()
        neg=torch.Tensor(pairs[:,2]).long().cuda()
        users,pos,neg=tools.shuffle(users,pos,neg)
        n_batch=len(users)//args.train_batch_size+1
        total=[0.,0.,0.]
        for u,i,j in tools.mini_batch(users,pos,neg,batch_size=args.train_batch_size):
            parts=model(u,i,j)
            loss=0.
            for k,value in enumerate(parts):
                loss+=value
                scalar=float(value.item())
                if not math.isfinite(scalar):
                    raise RuntimeError('Nonfinite training loss')
                total[k]+=scalar
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()
        train_seconds=time.perf_counter()-begin
        evaluation_rng=rng_state()
        before=time.perf_counter()
        with (a.output/'evaluation_progress.log').open('w',encoding='utf-8') as log,redirect_stdout(log):
            values=tester.testing(model,args,data,torch.device('cuda'))
        torch.cuda.synchronize()
        evaluation_seconds=time.perf_counter()-before
        metrics={name+'@'+str(k):float(values[key][index])
                 for name,key in [('Recall','recall'),('NDCG','ndcg')] for index,k in enumerate((10,20))}
        count+=1
        if metrics['Recall@20']>best_recall:
            best_recall=metrics['Recall@20']; best_epoch=epoch+1; best_metrics=metrics; count=0
            save(a.output/'best.pt',dict(model_state=model.state_dict(),rng=evaluation_rng,epoch=best_epoch,metrics=metrics))
        row=dict(epoch=epoch+1,loss=[v/n_batch for v in total],test=metrics,
                 train_seconds=train_seconds,evaluation_seconds=evaluation_seconds,
                 best_epoch=best_epoch,count=count)
        with (a.output/'history.jsonl').open('a',encoding='utf-8') as log:
            log.write(json.dumps(row)+'\n')
        save(a.output/'latest.pt',dict(model_state=model.state_dict(),optimizer_state=optimizer.state_dict(),
             rng=rng_state(),epoch=epoch+1,best_recall=best_recall,best_epoch=best_epoch,best_metrics=best_metrics,
             count=count,accumulated_wall=accumulated_wall+time.perf_counter()-started))
        print(json.dumps(dict(epoch=epoch+1,recall20=metrics['Recall@20'],best_epoch=best_epoch,
                              train_seconds=round(train_seconds,2),test_seconds=round(evaluation_seconds,2))),flush=True)
        if a.interrupt_after and epoch+1==a.interrupt_after:
            json_write(a.output/'result.json',dict(status='interrupted_for_acceptance',epoch=epoch+1,diagnostic=True))
            return
        if count>10:
            stop=True; break
    training_peak=torch.cuda.max_memory_allocated()
    state=torch.load(a.output/'best.pt',map_location='cpu')
    model.load_state_dict(state['model_state']); restore_rng(state['rng']); model.eval()
    rows=per_user(model,data,tester,2048)
    for metric,score in best_metrics.items():
        assert abs(float(np.mean([r[metric] for r in rows]))-score)<1e-10,(metric,score)
    json_write(a.output/'test_users.json',rows)
    result=dict(status='complete',dataset=a.dataset,seed=a.seed,diagnostic=a.diagnostic,
        epochs=epoch+1,best_epoch=best_epoch,test=best_metrics,stopped_by_patience=stop,
        selection_uses_test=True,graph_mode=config['graph_mode'],na_mode=a.na_mode,
        wall_seconds=accumulated_wall+time.perf_counter()-started,
        peak_cuda_allocated_during_training_and_native_evaluation_bytes=training_peak,
        data_sha256=config['data_sha256'],config_sha256=sha(a.output/'config.json'),
        best_checkpoint_sha256=sha(a.output/'best.pt'),gpu=torch.cuda.get_device_name(0))
    json_write(a.output/'result.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
