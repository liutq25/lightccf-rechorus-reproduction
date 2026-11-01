"""Experimental NA masking interventions; the original LightCCF stays intact."""
import math
import torch
from torch.nn import functional as F
from models.general.LightCCF import LightCCF


def pair_mask(user_ids, policy, generator=None):
    size = len(user_ids)
    diagonal = torch.eye(size, dtype=torch.bool, device=user_ids.device)
    same_user = user_ids[:, None].eq(user_ids[None, :]) & ~diagonal
    if policy == 'same_user':
        return same_user
    if policy != 'random_matched':
        raise ValueError(policy)
    count = same_user.sum(1)
    maximum = int(count.max())
    if not maximum:
        return torch.zeros_like(same_user)
    noise = torch.rand((size, size), device=user_ids.device, generator=generator)
    noise.masked_fill_(diagonal, -torch.inf)
    chosen = noise.topk(maximum, dim=1, sorted=True).indices
    mask = torch.zeros_like(same_user)
    mask.scatter_(1, chosen, torch.arange(maximum, device=user_ids.device)[None, :] < count[:, None])
    return mask


def masked_neighborhood_loss(users, items, tau, user_ids, policy, generator=None):
    if tau <= 0:
        raise ValueError('tau must be positive')
    u, i = F.normalize(users, dim=1), F.normalize(items, dim=1)
    logits = (u @ i.T + u @ u.T) / tau
    logits = logits.masked_fill(pair_mask(user_ids, policy, generator), -torch.inf)
    log_ratio = (u * i).sum(1) / tau - torch.logsumexp(logits, dim=1)
    return -torch.logaddexp(log_ratio, log_ratio.new_tensor(math.log(1e-5))).mean()


class LightCCFMasked(LightCCF):
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--mask_policy', choices=['same_user','random_matched'], default='same_user')
        return LightCCF.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.mask_policy = args.mask_policy
        self.mask_generator = torch.Generator(device=self.device).manual_seed(args.random_seed + 91073)

    def forward(self, feed_dict):
        uid, candidates = feed_dict['user_id'], feed_dict['item_id']
        all_u, all_i = self.all_embeddings()
        u, items = all_u[uid], all_i[candidates]
        result = {'prediction': (u[:, None, :] * items).sum(-1)}
        if feed_dict.get('phase', 'train') == 'train':
            positive = feed_dict['positive_item']
            result['na_loss'] = masked_neighborhood_loss(u, all_i[positive], self.tau, uid, self.mask_policy, self.mask_generator)
            result['reg_loss'] = 0.5 * (self.u_embeddings(uid).square().sum() + self.i_embeddings(candidates).square().sum()) / len(uid)
        return result
