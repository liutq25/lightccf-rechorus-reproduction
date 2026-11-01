"""Independent hand-computable checks for the shared full-catalog evaluator."""
from pathlib import Path
import math
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'framework/ReChorus/src'))
from helpers.ReproductionRunner import ReproductionRunner, metrics_from_topk


class FixedModel(torch.nn.Module):
    device = torch.device('cpu')
    item_num = 7

    def all_embeddings(self):
        # Padding has the largest score and MUST be excluded. Items 1 and 2
        # are train/dev history; unobserved 3,4,5,6 have deterministic scores.
        return torch.tensor([[0.], [1.]]), torch.tensor([[100.],[90.],[80.],[4.],[3.],[2.],[1.]])


class FullRankingAcceptance(unittest.TestCase):
    def setUp(self):
        self.runner = object.__new__(ReproductionRunner)
        self.runner.topk = [1, 2, 3]
        self.runner.eval_batch_size = 1
        self.corpus = SimpleNamespace(
            train_clicked_set={1: {1}},
            data_df={
                'train': pd.DataFrame({'user_id':[1], 'item_id':[1]}),
                'dev': pd.DataFrame({'user_id':[1], 'item_id':[2]}),
                'test': pd.DataFrame({'user_id':[1,1], 'item_id':[3,5]})})

    def dataset(self, phase):
        return SimpleNamespace(model=FixedModel(), corpus=self.corpus, phase=phase)

    def test_two_targets_macro_metrics_match_hand_calculation(self):
        result, rows = metrics_from_topk([[3,4,5],[4,3,5]], [{3,5},{4}], [1,3])
        self.assertEqual(result['Recall@1'], .75)
        self.assertEqual(result['Recall@3'], 1.)
        expected_first = (1 + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
        self.assertAlmostEqual(result['NDCG@3'], (expected_first + 1) / 2)
        self.assertEqual(len(rows), 2)

    def test_test_masks_train_and_dev_but_preserves_both_targets(self):
        result = self.runner.evaluate_full(self.dataset('test'))
        self.assertEqual(result['Recall@1'], .5)
        self.assertEqual(result['Recall@2'], .5)
        self.assertEqual(result['Recall@3'], 1.)
        self.assertAlmostEqual(result['NDCG@3'], 1.5 / (1 + 1 / math.log2(3)))

    def test_validation_is_independent_of_test_labels(self):
        before = self.runner.evaluate_full(self.dataset('dev'))
        self.corpus.data_df['test']['item_id'] = [2,6]
        after = self.runner.evaluate_full(self.dataset('dev'))
        self.assertEqual(before, after)
        self.assertEqual(before['Recall@1'], 1.)

    def test_duplicate_targets_are_deduplicated(self):
        before = self.runner.evaluate_full(self.dataset('test'))
        self.corpus.data_df['test'] = pd.DataFrame({'user_id':[1,1,1], 'item_id':[3,5,3]})
        self.assertEqual(before, self.runner.evaluate_full(self.dataset('test')))

    def test_known_target_overlap_is_rejected(self):
        self.corpus.data_df['test']['item_id'] = [2,5]
        with self.assertRaises(ValueError):
            self.runner.evaluate_full(self.dataset('test'))

    def test_ties_use_item_id_and_batch_size_does_not_change_scores(self):
        class TiedModel(FixedModel):
            def all_embeddings(self):
                return torch.tensor([[0.],[1.]]), torch.tensor([[100.],[90.],[80.],[4.],[4.],[4.],[4.]])
        dataset = self.dataset('test')
        dataset.model = TiedModel()
        before = self.runner.evaluate_full(dataset)
        self.runner.eval_batch_size = 20
        self.assertEqual(before, self.runner.evaluate_full(dataset))
        self.assertEqual(before['Recall@1'], .5)
        self.assertEqual(before['Recall@2'], .5)
        self.assertEqual(before['Recall@3'], 1.)


if __name__ == '__main__':
    unittest.main()
