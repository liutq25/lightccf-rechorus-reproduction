"""Audited ReChorus runner: validation selection and user-level full ranking.

The tensor batching is an exact interaction-wise pass over GeneralModel.Dataset.
It avoids the Windows DataLoader overhead without altering the model objectives.
"""
from pathlib import Path
import json
import logging
import time
import numpy as np
import torch
from helpers.BaseRunner import BaseRunner


def metrics_from_topk(recommendations, truth, topks=(10,20)):
    """One macro-averaged observation per user; multiple positives allowed."""
    rows = []
    for ranked, relevant in zip(recommendations, truth):
        relevant = set(relevant)
        if not relevant:
            raise ValueError('Cannot evaluate a user without relevant items')
        hit = np.array([int(int(item) in relevant) for item in ranked], dtype=float)
        row = {}
        for k in topks:
            n = min(k, len(hit))
            row[f'Recall@{k}'] = float(hit[:n].sum() / len(relevant))
            row[f'HR@{k}'] = float(hit[:n].sum() > 0)
            idcg = np.sum(1 / np.log2(np.arange(2, min(len(relevant), k) + 2)))
            row[f'NDCG@{k}'] = float(np.sum(hit[:n] / np.log2(np.arange(2,n + 2))) / idcg)
        rows.append(row)
    return {key: float(np.mean([r[key] for r in rows])) for key in rows[0]}, rows


def full_embeddings(model):
    if hasattr(model, 'all_embeddings'):
        return model.all_embeddings()
    if hasattr(model, 'u_embeddings'):
        return model.u_embeddings.weight, model.i_embeddings.weight
    enc = model.encoder
    uids = torch.arange(model.user_num, device=model.device)
    iids = torch.arange(model.item_num, device=model.device)
    return enc(uids, iids)


class ReproductionRunner(BaseRunner):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.output = Path(args.output)
        self.eval_every = args.eval_every
        self.output.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def evaluate_full(self, dataset, save=False):
        model, corpus, phase = dataset.model, dataset.corpus, dataset.phase
        model.eval()
        all_u, all_i = full_embeddings(model)
        frame = corpus.data_df[phase]
        target_map = frame.groupby('user_id')['item_id'].agg(lambda x: set(x)).to_dict()
        users = sorted(target_map)
        known = {u: set(corpus.train_clicked_set[u]) for u in users}
        if phase == 'test':
            for u, i in zip(corpus.data_df['dev'].user_id, corpus.data_df['dev'].item_id):
                if u in known:
                    known[u].add(i)
        for u in users:
            if known[u] & target_map[u]:
                raise ValueError(f'{phase}: known/target overlap for user {u}')
        ranks = []
        k = min(max(self.topk), model.item_num - 1)
        for start in range(0, len(users), self.eval_batch_size):
            us = users[start:start + self.eval_batch_size]
            scores = all_u[torch.tensor(us, device=model.device)] @ all_i.T
            scores[:, 0] = -torch.inf
            rows, cols = [], []
            for row, u in enumerate(us):
                cols.extend(known[u])
                rows.extend([row] * len(known[u]))
            if rows:
                scores[rows, cols] = -torch.inf
            # Stable sorting gives a documented ascending-item-ID tie break.
            ranks.extend(torch.argsort(scores, dim=1, descending=True, stable=True)[:, :k].cpu().numpy())
        result, rows = metrics_from_topk(ranks, [target_map[u] for u in users], self.topk)
        if save:
            path = self.output / f'{phase}_users.json'
            path.write_text(json.dumps([dict(user_id=u, positives=len(target_map[u]), **row) for u,row in zip(users,rows)]), encoding='utf-8')
        return result

    def _initialize_training(self, dataset):
        model, corpus = dataset.model, dataset.corpus
        device = model.device
        self.train_u = torch.tensor(dataset.data['user_id'], dtype=torch.long, device=device)
        self.train_i = torch.tensor(dataset.data['item_id'], dtype=torch.long, device=device)
        self.known_keys = torch.unique(self.train_u * model.item_num + self.train_i, sorted=True)
        model.optimizer = self._build_optimizer(model)
        if max(len(x) for x in corpus.train_clicked_set.values()) >= model.item_num - 1:
            raise ValueError('User has no unobserved training negatives')

    def _sample_negatives(self, model):
        neg = torch.randint(1, model.item_num, self.train_i.shape, device=model.device)
        while True:
            keys = self.train_u * model.item_num + neg
            positions = torch.searchsorted(self.known_keys, keys)
            invalid = self.known_keys[positions.clamp_max(len(self.known_keys) - 1)] == keys
            count = int(invalid.sum())
            if not count:
                return neg
            neg[invalid] = torch.randint(1, model.item_num, (count,), device=model.device)

    def fit(self, dataset, epoch=-1):
        model = dataset.model
        model.train()
        negatives = self._sample_negatives(model)
        order = torch.randperm(len(self.train_u), device=model.device)
        total = 0.
        seen = 0
        for ids in order.split(self.batch_size):
            users, pos, neg = self.train_u[ids], self.train_i[ids], negatives[ids]
            candidates = torch.stack([pos, neg], dim=1)
            batch = dict(user_id=users, item_id=candidates, positive_item=pos, batch_size=len(ids), phase='train')
            model.optimizer.zero_grad(set_to_none=True)
            out = model(batch)
            loss = model.loss(out)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite loss at epoch {epoch}')
            loss.backward()
            model.optimizer.step()
            total += float(loss.detach()) * len(ids)
            seen += len(ids)
        return total / seen

    def train(self, data_dict):
        model = data_dict['train'].model
        self._initialize_training(data_dict['train'])
        start = time.perf_counter()
        train_seconds = 0.
        best_value, best_epoch, best_dev = -float('inf'), 0, None
        trajectory = []
        if model.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        for epoch in range(1,self.epoch + 1):
            t = time.perf_counter()
            loss = self.fit(data_dict['train'], epoch)
            if model.device.type == 'cuda':
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t
            train_seconds += elapsed
            record = dict(epoch=epoch, loss=loss, train_seconds=elapsed)
            if epoch == 1 or epoch % self.eval_every == 0 or epoch == self.epoch:
                result = self.evaluate_full(data_dict['dev'])
                record['dev'] = result
                value = result[self.main_metric]
                if value > best_value:
                    best_value, best_epoch, best_dev = value, epoch, result
                    torch.save(model.state_dict(), self.output / 'best.pt')
                logging.info('epoch=%d loss=%.6f train=%.3fs dev=%s best_epoch=%d',epoch,loss,elapsed,result,best_epoch)
            trajectory.append(record)
            with (self.output / 'history.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record) + '\n')
            if epoch - best_epoch >= self.early_stop and epoch % self.eval_every == 0:
                break
        model.load_state_dict(torch.load(self.output / 'best.pt', map_location=model.device, weights_only=True))
        dev = self.evaluate_full(data_dict['dev'], save=True)
        if abs(dev[self.main_metric] - best_value) > 1e-10:
            raise AssertionError('Saved checkpoint does not reproduce selected validation metric')
        # Pilot/tuning runs deliberately never open the test result.
        test = None if self.args.pilot else self.evaluate_full(data_dict['test'], save=True)
        result = dict(status='complete',pilot=self.args.pilot,model=self.args.model,dataset=self.args.dataset,
                      seed=self.args.random_seed,best_epoch=best_epoch,epochs=epoch,dev=dev,test=test,
                      training_seconds=train_seconds,wall_seconds=time.perf_counter()-start,
                      mean_epoch_seconds=float(np.mean([x['train_seconds'] for x in trajectory])),
                      peak_cuda_mb=torch.cuda.max_memory_allocated()/2**20 if model.device.type=='cuda' else 0,
                      stopped_by_patience=epoch < self.epoch)
        (self.output / 'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        logging.info('COMPLETE %s',result)
