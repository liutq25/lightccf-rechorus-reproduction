"""ReChorus reproduction entry point; all ranking evaluation is full catalog."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'framework/ReChorus/src'))
import argparse
import hashlib
import json
import logging
import random
import numpy as np
import torch
from helpers.BaseReader import BaseReader
from helpers.ReproductionRunner import ReproductionRunner
from models.general.BPRMF import BPRMF
from models.general.LightGCN import LightGCN
from models.general.LightCCF import LightCCF
from models.general.LightCCFMasked import LightCCFMasked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--pilot', action='store_true')
    cli = parser.parse_args()
    config = json.loads(Path(cli.config).read_text(encoding='utf-8'))
    if cli.seed is not None:
        config['random_seed'] = cli.seed
    if cli.epochs is not None:
        config['epoch'] = cli.epochs
    config['pilot'] = cli.pilot
    out = Path(cli.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'result.json').exists():
        raise RuntimeError(f'Refusing to overwrite completed run: {out}')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', handlers=[logging.FileHandler(out / 'train.log', encoding='utf-8'), logging.StreamHandler(sys.stdout)])
    defaults = dict(path=str(ROOT / 'framework/ReChorus/data'), sep='\t', emb_size=64, num_neg=1,
                    dropout=0., test_all=1, buffer=0, n_layers=3, tau=.28, ssl_weight=1.,
                    reg_weight=1e-4, na_mode='full', random_seed=2023, lr=.001, l2=1e-6,
                    epoch=200, early_stop=20, eval_every=5, batch_size=2048, eval_batch_size=256,
                    optimizer='Adam', num_workers=0, pin_memory=0, topk='10,20', metric='NDCG,Recall',
                    main_metric='NDCG@20', check_epoch=0, test_epoch=-1, train=1, threads=4,
                    log_file=str(out / 'train.log'), model_path=str(out / 'best.pt'), output=str(out))
    defaults.update(config)
    defaults['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = argparse.Namespace(**defaults)
    if args.num_neg != 1:
        raise ValueError('This protocol uses exactly one training negative.')
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.set_num_threads(args.threads)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    environment = dict(python=sys.version, pytorch=torch.__version__, numpy=np.__version__, device=str(args.device),
                       gpu=torch.cuda.get_device_name(0) if args.device.type == 'cuda' else None)
    corpus = BaseReader(args)
    hashes = {phase: hashlib.sha256((Path(args.path) / args.dataset / (phase + '.csv')).read_bytes()).hexdigest() for phase in ['train','dev','test']}
    model_cls = {'BPRMF': BPRMF, 'LightGCN': LightGCN, 'LightCCF': LightCCF, 'LightCCFMasked': LightCCFMasked}[args.model]
    model = model_cls(args, corpus).to(args.device)
    if isinstance(model, LightGCN):
        # Use canonical sparse storage without changing graph values. The
        # pinned upstream LightGCN constructor still requires CUDA.
        model.encoder.sparse_norm_adj = model.encoder.sparse_norm_adj.coalesce()
    metadata = {**vars(args), 'device': str(args.device), 'environment': environment, 'data_sha256': hashes,
                'parameters': model.count_variables(), 'framework_commit': 'c164ec4303cc20ddcfbd1b57de366a481811d1e5'}
    (out / 'config.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    logging.info('Configuration: %s', metadata)
    datasets = {phase: model_cls.Dataset(model, corpus, phase) for phase in ['train','dev','test']}
    runner = ReproductionRunner(args)
    runner.train(datasets)

if __name__ == '__main__':
    main()
