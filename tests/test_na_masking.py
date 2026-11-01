"""Numerical and random-stream acceptance for the experimental NA masks."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'framework/ReChorus/src'))
from models.general.LightCCF import neighborhood_loss
from models.general.LightCCFMasked import pair_mask, masked_neighborhood_loss
from models.general.LightCCF import LightCCF
from models.general.LightCCFMasked import LightCCFMasked


class MaskingAcceptance(unittest.TestCase):
    def test_same_user_mask_matches_literal_pair_identity(self):
        ids = torch.tensor([1,2,1,1])
        expected = torch.tensor([[False,False,True,True],[False,False,False,False],
                                 [True,False,False,True],[True,False,True,False]])
        self.assertTrue(torch.equal(pair_mask(ids, 'same_user'), expected))

    def test_random_control_matches_row_counts_and_keeps_diagonal(self):
        ids = torch.tensor([1,2,1,1,3,3])
        gen = torch.Generator().manual_seed(811)
        actual = pair_mask(ids, 'random_matched', gen)
        self.assertTrue(torch.equal(actual.sum(1), pair_mask(ids, 'same_user').sum(1)))
        self.assertFalse(actual.diag().any())
        self.assertFalse(torch.equal(actual, pair_mask(ids, 'same_user')))

    def test_control_has_no_global_random_stream_side_effect(self):
        torch.manual_seed(7701)
        before = torch.random.get_rng_state().clone()
        pair_mask(torch.tensor([1,1,2,2]), 'random_matched', torch.Generator().manual_seed(1))
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

    def test_no_collisions_matches_original_loss_and_gradients(self):
        gen = torch.Generator().manual_seed(36)
        u = torch.randn(4,5,dtype=torch.float64,generator=gen,requires_grad=True)
        i = torch.randn(4,5,dtype=torch.float64,generator=gen,requires_grad=True)
        expected = neighborhood_loss(u,i,.28)
        reference = torch.autograd.grad(expected,(u,i),retain_graph=True)
        for policy in ('same_user','random_matched'):
            actual = masked_neighborhood_loss(u,i,.28,torch.arange(4),policy,gen)
            grad = torch.autograd.grad(actual,(u,i),retain_graph=True)
            torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
            for x,y in zip(grad,reference):
                torch.testing.assert_close(x,y,rtol=1e-12,atol=1e-12)

    def test_masked_loss_matches_scalar_enumeration(self):
        ids = torch.tensor([1,1,2])
        u = torch.tensor([[1.,2.],[2.,1.],[-1.,1.]],dtype=torch.float64)
        i = torch.tensor([[1.,1.],[1.,-1.],[-2.,1.]],dtype=torch.float64)
        x,y = torch.nn.functional.normalize(u,dim=1),torch.nn.functional.normalize(i,dim=1)
        terms = []
        for b in range(3):
            numerator = torch.exp((x[b]*y[b]).sum()/.28)
            denominator = sum(torch.exp(((x[b]*y[c]).sum()+(x[b]*x[c]).sum())/.28)
                              for c in range(3) if b==c or ids[b]!=ids[c])
            terms.append(-torch.log(numerator/denominator+1e-5))
        torch.testing.assert_close(masked_neighborhood_loss(u,i,.28,ids,'same_user'),torch.stack(terms).mean())

    def test_intervention_preserves_initialization_prediction_and_regularizer(self):
        args=SimpleNamespace(device=torch.device('cpu'),model_path='',buffer=0,num_neg=1,
                             dropout=0.,test_all=1,emb_size=4,tau=.28,ssl_weight=1.,
                             reg_weight=1e-4,n_layers=0,na_mode='full',random_seed=2023,
                             mask_policy='random_matched')
        corpus=SimpleNamespace(n_users=4,n_items=8)
        torch.manual_seed(2023)
        original=LightCCF(args,corpus)
        state=torch.random.get_rng_state().clone()
        torch.manual_seed(2023)
        changed=LightCCFMasked(args,corpus)
        self.assertTrue(torch.equal(state,torch.random.get_rng_state()))
        for left,right in zip(original.parameters(),changed.parameters()):
            torch.testing.assert_close(left,right,rtol=0,atol=0)
        batch=dict(user_id=torch.tensor([1,1,2]),item_id=torch.tensor([[1,5],[2,6],[3,7]]),
                   positive_item=torch.tensor([1,2,3]),batch_size=3,phase='train')
        before=original(batch)
        after=changed(batch)
        torch.testing.assert_close(before['prediction'],after['prediction'],rtol=0,atol=0)
        torch.testing.assert_close(before['reg_loss'],after['reg_loss'],rtol=0,atol=0)
        self.assertTrue(torch.equal(state,torch.random.get_rng_state()))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA-specific RNG acceptance')
    def test_cuda_control_uses_private_stream_and_has_finite_gradients(self):
        device=torch.device('cuda')
        local=torch.Generator(device=device).manual_seed(2023)
        u=torch.arange(1,25,device=device,dtype=torch.float32).reshape(6,4).requires_grad_()
        i=torch.arange(25,49,device=device,dtype=torch.float32).reshape(6,4).requires_grad_()
        state=torch.cuda.get_rng_state().clone()
        loss=masked_neighborhood_loss(u,i,.28,torch.tensor([1,1,2,3,3,3],device=device),'random_matched',local)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(u.grad).all() and torch.isfinite(i.grad).all())
        self.assertTrue(torch.equal(state,torch.cuda.get_rng_state()))


if __name__ == '__main__':
    unittest.main()
