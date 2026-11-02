"""Measure full-batch feasibility without turning a short probe into a result."""
from pathlib import Path
from types import SimpleNamespace
import argparse
import hashlib
import importlib
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


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['douban-book','tmall','amazon-book'],required=True)
    p.add_argument('--model',default='LightCCF')
    p.add_argument('--steps',type=int,default=3)
    p.add_argument('--attempt-over-budget',action='store_true',help='Measure actual CUDA behavior despite the resident-VRAM lower bound; Windows may use shared system memory')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():
        raise RuntimeError('Refusing to overwrite a completed resource probe')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    random.seed(2023); np.random.seed(2023); torch.manual_seed(2023)
    torch.cuda.manual_seed_all(2023); torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    device=torch.device('cuda')
    record=dict(dataset=a.dataset,model=a.model,status='running',probe_only=True,
                torch=torch.__version__,numpy=np.__version__,python=sys.version,
                gpu=torch.cuda.get_device_name(),gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,
                training_steps=a.steps,test_batch_size=2048,seed=2023,
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    started=time.perf_counter()
    try:
        folder=ROOT/'vendor/LightCCF-main/dataset'/a.dataset
        clean=ROOT/'data/author_clean_graphs'/a.dataset/'pre_Adj.npz'
        if not clean.is_file():
            raise RuntimeError('Run the original dataset audit first')
        record['clean_graph_sha256']=hashlib.sha256(clean.read_bytes()).hexdigest()
        if a.model=='LightCCF':
            upstream=ROOT/'vendor/LightCCF-main'
            sys.path.insert(0,str(upstream))
            from utility.data_loader import Data
            from LightCCF import LightCCF
            batch=2048 if a.dataset=='douban-book' else 4096
            config=dict(dataset=a.dataset,dataset_path=str(folder.parent)+'/',dataset_type='.txt',
                        sparsity_test=0,reg_lambda=1e-4,ssl_lambda=1. if a.dataset=='douban-book' else 5.,
                        tau={'douban-book':.28,'tmall':.24,'amazon-book':.22}[a.dataset],
                        encoder='LightGCN' if a.dataset=='amazon-book' else 'MF',
                        embedding_size=64,gcn_layer=3)
            data=Data(SimpleNamespace(**config))
            data.sparse_adjacency_matrix=lambda:sp.load_npz(clean)
            model=LightCCF(SimpleNamespace(**config),data,device).to(device)
            users,items=data.train_users,data.train_items
            positives=data.train_dict
            record.update(config=config,training_batch_size=batch)
        else:
            upstream=ROOT/'vendor/ID-GRec'
            sys.path.insert(0,str(upstream))
            from utility.utility_data.data_loader import Data
            from utility.utility_function.tools import read_configuration
            import utility.utility_data.data_graph as graph
            config_path=upstream/'configure'/f'{a.model}.txt'
            if a.model=='BIGCF' and not config_path.exists():
                config_path=ROOT/'vendor/BIGCF-full-sample/configure/BIGCF.txt'
                record['config_recovered_from_primary_source']='https://github.com/BlueGhostYi/BIGCF-full-sample/blob/90dbf8241e07e8fc0afd9a43d56f2921a7bf58d3/configure/BIGCF.txt'
            record['released_config_sha256']=hashlib.sha256(config_path.read_bytes()).hexdigest()
            config=read_configuration(str(config_path),a.model)
            config.update(dataset=a.dataset,embedding_size=2048 if a.model=='RecDCL' else 64,
                          learn_rate=.001,reg_lambda=.0001)
            data=Data(str(folder),config)
            batch=int(config['batch_size'])
            record.update(config=config,training_batch_size=batch,
                          idgrec_commit='10974d8d0d932d0018eda84bf46d2e8504b58ca8')
            if a.model=='RecDCL':
                lower=4*(data.num_users+data.num_items)*2048*4
                record['dense_embedding_adam_lower_bound_bytes']=lower
                record['over_resident_vram_budget']=lower>record['gpu_total_bytes']
                record['attempt_over_budget']=a.attempt_over_budget
                if lower>record['gpu_total_bytes'] and not a.attempt_over_budget:
                    record.update(status='exceeds_proven_memory_lower_bound',
                                  reason='FP32 embedding parameters + dense gradients + Adam first/second moments alone exceed GPU VRAM; excludes histories, projector and graph activations.')
                    return
            graph.sparse_adjacency_matrix=lambda dataset:sp.load_npz(clean)
            def rectangular(dataset):
                r=dataset.user_item_net.tocsr()
                dr=np.asarray(r.sum(axis=1)).ravel()
                dc=np.asarray(r.sum(axis=0)).ravel()
                return sp.diags(np.power(np.maximum(dr,1),-.5))@r@sp.diags(np.power(np.maximum(dc,1),-.5))
            def with_self(dataset):
                r=dataset.user_item_net.tocsr()
                adj=sp.bmat([[None,r],[r.T,None]],format='csr')+sp.eye(dataset.num_nodes)
                degree=np.asarray(adj.sum(axis=1)).ravel()
                scale=sp.diags(np.power(degree,-.5))
                return scale@adj@scale
            graph.sparse_adjacency_matrix_R=rectangular
            graph.sparse_adjacency_matrix_with_self=with_self
            cls=getattr(importlib.import_module('models.'+a.model),a.model)
            model=cls(config,data,device).to(device)
            users,items=data.train_user,data.train_item
            positives={u:set(row.indices) for u,row in enumerate(data.user_item_net)}
        record.update(users=int(data.num_users),items=int(data.num_items),
                      train_interactions=len(users),parameters=sum(v.numel() for v in model.parameters()))
        rng=np.random.RandomState(83)
        chosen=rng.choice(len(users),size=batch,replace=False)
        uid=np.asarray(users)[chosen]; pos=np.asarray(items)[chosen]
        neg=rng.randint(0,data.num_items,size=batch)
        for k,u in enumerate(uid):
            while neg[k] in positives[u]:
                neg[k]=rng.randint(0,data.num_items)
        triples=[torch.as_tensor(x,dtype=torch.long,device=device) for x in (uid,pos,neg)]
        forward_args=triples
        if a.model=='CVGA':
            # The published CVGA trainer batches users and dense reconstruction
            # targets, rather than sampled user-positive-negative triples.
            batch_users=torch.arange(batch,device=device)
            targets=torch.as_tensor(data.user_item_net[:batch].toarray(),dtype=torch.float32,device=device)
            forward_args=[batch_users,targets]
        elif a.model=='SGL':
            from utility.utility_function.tools import create_adj_mat,convert_sp_mat_to_sp_tensor
            def augmented():
                if config['aug_type'] in ('nd','ed'):
                    return convert_sp_mat_to_sp_tensor(create_adj_mat(data.user_item_net,config['aug_type'],float(config['ssl_ratio']))).to(device)
                return [convert_sp_mat_to_sp_tensor(create_adj_mat(data.user_item_net,config['aug_type'],float(config['ssl_ratio']))).to(device)
                        for _ in range(int(config['GCN_layer']))]
            forward_args=triples+[augmented(),augmented()]
        elif a.model=='NCL':
            # Include the full cluster loss rather than measuring its warm-up.
            import faiss
            faiss.omp_set_num_threads(4)
            record.update(faiss=faiss.__version__,faiss_gpu_count=faiss.get_num_gpus(),
                          clustering_backend='GPU' if faiss.get_num_gpus() else 'CPU')
            model.E_step()
            forward_args=triples+[20]
        optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        torch.cuda.reset_peak_memory_stats()
        elapsed=[]
        losses=[]
        model.train()
        for step in range(a.steps+1):
            torch.cuda.synchronize(); begin=time.perf_counter()
            optimizer.zero_grad()
            components=model(*forward_args)
            loss=sum(components)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite full-batch loss')
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            if step:
                elapsed.append(time.perf_counter()-begin)
                losses.append(float(loss))
        record.update(training_step_seconds=elapsed,training_step_loss=losses,
                      training_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      training_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      epoch_training_units=data.num_users if a.model=='CVGA' else len(users),
                      epoch_training_unit='users' if a.model=='CVGA' else 'interactions',
                      estimated_compute_seconds_per_epoch=float(np.mean(elapsed))*math.ceil((data.num_users if a.model=='CVGA' else len(users))/batch),
                      epoch_estimate_excludes_sampling_evaluation_and_io=True)
        optimizer.zero_grad(set_to_none=True)
        model.eval()
        torch.cuda.reset_peak_memory_stats()
        test_users=torch.arange(min(2048,data.num_users),device=device)
        torch.cuda.synchronize(); begin=time.perf_counter()
        with torch.no_grad():
            scores=model.get_rating_for_test(test_users)
            torch.topk(scores,k=20)
        torch.cuda.synchronize()
        record.update(status='complete',test_batch_seconds=time.perf_counter()-begin,
                      test_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      test_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except Exception as e:
        record.update(status='cuda_out_of_memory' if 'out of memory' in str(e).lower() else 'error',
                      error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc())
        if torch.cuda.is_initialized():
            record['failure_peak_allocated_bytes']=torch.cuda.max_memory_allocated()
            record['failure_peak_reserved_bytes']=torch.cuda.max_memory_reserved()
    finally:
        record['wall_seconds']=time.perf_counter()-started
        a.output.write_text(json.dumps(record,indent=2),encoding='utf-8')
        print(json.dumps({k:v for k,v in record.items() if k not in {'config','traceback'}},indent=2))


if __name__=='__main__':
    main()
