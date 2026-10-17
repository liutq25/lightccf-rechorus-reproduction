"""Run one author-code Douban-book anchor with an explicit random seed.

This diagnostic intentionally follows the author's test-set epoch selection,
so its score must stay separate from the course validation-selected matrix.
"""
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
import argparse
import hashlib
import json
import os
import random
import re
import shutil
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / 'vendor' / 'LightCCF-main'


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--dataset', default='douban-book')
    args = parser.parse_args()
    if args.dataset != 'douban-book':
        raise ValueError('The fixed anchor protocol is for Douban-book only')
    output = ROOT / 'runs' / 'official_anchor' / args.dataset / f'seed_{args.seed}'
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f'Anchor directory is not empty: {output}')
    output.mkdir(parents=True)
    (VENDOR / 'log' / 'LightCCF' / args.dataset).mkdir(parents=True, exist_ok=True)
    log_tag = f'anchor_seed_{args.seed}'
    vendor_log = VENDOR / 'log' / 'LightCCF' / args.dataset / f'{log_tag}.log'
    if vendor_log.exists():
        raise RuntimeError(f'Author-code log already exists: {vendor_log}')
    result_dir = VENDOR / 'results'
    before = set(result_dir.glob('LightCCF_results_*.txt')) if result_dir.exists() else set()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    sys.path.insert(0, str(VENDOR))
    sys.argv = [str(VENDOR / 'main.py'), '--dataset', args.dataset,
                '--encoder', 'MF', '--train_batch_size', '2048', '--tau', '0.28',
                '--ssl_lambda', '1.0', '--gpu', '0', '--seed', str(args.seed),
                '--train_epoch', str(args.epochs), '--early_stop', '10',
                '--top_K', '[10,20]', '--log', log_tag]
    (output / 'config.json').write_text(json.dumps({
        'dataset': args.dataset, 'seed': args.seed, 'max_epochs': args.epochs,
        'encoder': 'MF', 'embedding_size': 64, 'train_batch_size': 2048,
        'test_batch_size': 2048, 'tau': .28, 'ssl_lambda': 1.,
        'reg_lambda': 1e-4, 'learn_rate': .001, 'early_stop': 10,
        'test_frequency': 1, 'top_K': [10, 20],
        'selection': 'author test Recall@20',
        'author_commit': '45645ef15fc08e858ffc81e1b2530b4b1805ad34',
        'entrypoint_compatibility_fix': 'load_models(args) -> load_models() in memory only',
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    old_cwd = Path.cwd()
    started = time.perf_counter()
    # The pinned public main.py defines load_models() with no parameters but
    # calls load_models(args). Correct that call in memory only; retain the
    # unchanged upstream file and disclose this execution compatibility fix.
    entry = VENDOR / 'main.py'
    source = entry.read_text(encoding='utf-8')
    broken_call = 'model_name, Trainer = load_models(args)'
    fixed_call = 'model_name, Trainer = load_models()'
    if source.count(broken_call) != 1:
        raise RuntimeError('Expected author main.py call-site was not found exactly once')
    executable = compile(source.replace(broken_call, fixed_call), str(entry), 'exec')
    try:
        os.chdir(VENDOR)
        with ((output / 'console.log').open('w', encoding='utf-8') as stream,
              redirect_stdout(stream), redirect_stderr(stream)):
            exec(executable, {'__name__': '__main__', '__file__': str(entry),
                              '__package__': None})
    finally:
        os.chdir(old_cwd)
    wall_seconds = time.perf_counter() - started
    shutil.copy2(vendor_log, output / 'author_train.log')
    new_results = set(result_dir.glob('LightCCF_results_*.txt')) - before
    if len(new_results) != 1:
        raise RuntimeError(f'Expected one new author result, found {len(new_results)}')
    source_result = new_results.pop()
    shutil.copy2(source_result, output / 'author_result.txt')
    contents = source_result.read_text(encoding='utf-8')
    def read_number(label, cast=float):
        match = re.search(rf'^{re.escape(label)}: ([^\n]+)', contents, re.M)
        if not match:
            raise RuntimeError(f'Missing {label} in author result')
        return cast(match.group(1))
    result = {
        'status': 'complete', 'source': 'author_code', 'dataset': args.dataset,
        'seed_set_before_model': args.seed, 'max_epochs': args.epochs,
        'best_epoch_by_test_recall20': read_number('Best Recall Epoch', int),
        'test_recall20_at_best': read_number('Best Recall'),
        'test_ndcg20_at_best': read_number('Best NDCG'),
        'selection_uses_test': True,
        'entrypoint_compatibility_fix': 'load_models(args) -> load_models() in memory only',
        'unmodified_author_main_sha256': sha256(entry),
        'wall_seconds': wall_seconds,
        'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
        'torch': torch.__version__, 'numpy': np.__version__,
        'train_sha256': sha256(VENDOR / 'dataset' / args.dataset / 'train.txt'),
        'test_sha256': sha256(VENDOR / 'dataset' / args.dataset / 'test.txt'),
    }
    (output / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
