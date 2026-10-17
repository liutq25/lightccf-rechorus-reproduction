"""LightCCF (SIGIR 2025), ported to ReChorus GeneralModel.

The reference numerical convention is ZzYUuuu/LightCCF/utility/losses.py.
In particular the denominator exponentiates the SUM of UI and UU logits,
includes the diagonal/duplicate IDs, and adds 1e-5 after the ratio.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from models.BaseModel import GeneralModel
from models.general.LightGCN import LightGCNBase


def neighborhood_loss(users, items, tau, mode='full'):
    if tau <= 0:
        raise ValueError('tau must be positive')
    u, i = F.normalize(users, dim=1), F.normalize(items, dim=1)
    logits = u @ i.T
    if mode == 'full':
        logits = logits + u @ u.T
    elif mode != 'ui':
        raise ValueError(mode)
    log_ratio = (u * i).sum(1) / tau - torch.logsumexp(logits / tau, dim=1)
    # Algebraically identical to -log(exp(log_ratio) + 10e-6).
    return -torch.logaddexp(log_ratio, log_ratio.new_tensor(math.log(1e-5))).mean()


class LightCCF(GeneralModel):
    reader, runner = 'BaseReader', 'ReproductionRunner'
    extra_log_args = ['emb_size', 'tau', 'ssl_weight', 'reg_weight', 'n_layers', 'na_mode']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64)
        parser.add_argument('--tau', type=float, default=0.28)
        parser.add_argument('--ssl_weight', type=float, default=1.0)
        parser.add_argument('--reg_weight', type=float, default=1e-4)
        parser.add_argument('--n_layers', type=int, default=0)
        parser.add_argument('--na_mode', choices=['full', 'ui'], default='full')
        return GeneralModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.emb_size = args.emb_size
        self.tau = args.tau
        self.ssl_weight = args.ssl_weight
        self.reg_weight = args.reg_weight
        self.n_layers = args.n_layers
        self.na_mode = args.na_mode
        self.u_embeddings = nn.Embedding(self.user_num, self.emb_size)
        self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)
        nn.init.xavier_uniform_(self.u_embeddings.weight)
        nn.init.xavier_uniform_(self.i_embeddings.weight)
        if self.n_layers:
            adj = LightGCNBase.build_adjmat(self.user_num, self.item_num, corpus.train_clicked_set).tocoo()
            indices = torch.as_tensor(__import__('numpy').stack([adj.row, adj.col]), dtype=torch.long)
            self.register_buffer('adjacency', torch.sparse_coo_tensor(indices, torch.tensor(adj.data), adj.shape).coalesce())

    def all_embeddings(self):
        u, i = self.u_embeddings.weight, self.i_embeddings.weight
        if not self.n_layers:
            return u, i
        h = torch.cat([u, i], dim=0)
        total = h
        for _ in range(self.n_layers):
            h = torch.sparse.mm(self.adjacency, h)
            total = total + h
        return (total / (self.n_layers + 1)).split([self.user_num, self.item_num])

    def forward(self, feed_dict):
        uid, candidates = feed_dict['user_id'], feed_dict['item_id']
        all_u, all_i = self.all_embeddings()
        u, items = all_u[uid], all_i[candidates]
        result = {'prediction': (u[:, None, :] * items).sum(-1)}
        if feed_dict.get('phase', 'train') == 'train':
            # ReChorus shuffles candidate columns before forward().
            # Preserve the positive item in an independent dataset field.
            positive = feed_dict['positive_item']
            result['na_loss'] = neighborhood_loss(u, all_i[positive], self.tau, self.na_mode) if self.ssl_weight else u.new_zeros(())
            result['reg_loss'] = 0.5 * (self.u_embeddings(uid).square().sum() + self.i_embeddings(candidates).square().sum()) / len(uid)
        return result

    def loss(self, out_dict):
        scores = out_dict['prediction']
        bpr = -torch.log(torch.sigmoid(scores[:, 0, None] - scores[:, 1:]) + 1e-7).mean()
        return bpr + self.ssl_weight * out_dict['na_loss'] + self.reg_weight * out_dict['reg_loss']

    class Dataset(GeneralModel.Dataset):
        def _get_feed_dict(self, index):
            feed = super()._get_feed_dict(index)
            feed['positive_item'] = self.data['item_id'][index]
            return feed
