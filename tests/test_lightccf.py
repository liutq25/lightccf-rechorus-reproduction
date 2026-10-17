"""Numerical acceptance tests, independent of full-data training.

Run from the project root: python -m unittest discover -s tests -v
The published loss source is loaded without changing the vendor package.
"""

import argparse
import importlib.util
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework" / "ReChorus" / "src"))

from models.general.LightCCF import LightCCF, neighborhood_loss  # noqa: E402


_spec = importlib.util.spec_from_file_location(
    "published_lightccf_losses", ROOT / "vendor" / "LightCCF-main" / "utility" / "losses.py"
)
official = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(official)


def scalar_reference(users, items, tau):
    """Literal pair-by-pair reference, without torch or matrix operations."""
    def normalized(row):
        norm = max(math.sqrt(sum(x * x for x in row)), 1e-12)
        return [x / norm for x in row]

    def dot(left, right):
        return sum(x * y for x, y in zip(left, right))

    users = [normalized(row) for row in users]
    items = [normalized(row) for row in items]
    losses = []
    for b, user in enumerate(users):
        numerator = math.exp(dot(user, items[b]) / tau)
        denominator = 0.0
        for c in range(len(users)):
            pair_similarity = dot(user, items[c]) + dot(user, users[c])
            denominator += math.exp(pair_similarity / tau)
        losses.append(-math.log(numerator / denominator + 1e-5))
    return sum(losses) / len(losses)


class NeighborhoodLossAcceptance(unittest.TestCase):
    def assert_tensor_close(self, actual, expected, atol=1e-11, rtol=1e-10):
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)

    def test_matches_published_loss_and_gradients(self):
        generator = torch.Generator().manual_seed(1701)
        for batch_size in (1, 3, 7):
            for tau in (0.20, 0.28, 0.40):
                with self.subTest(batch_size=batch_size, tau=tau):
                    users = torch.randn(batch_size, 5, generator=generator,
                                        dtype=torch.float64, requires_grad=True)
                    items = torch.randn(batch_size, 5, generator=generator,
                                        dtype=torch.float64, requires_grad=True)
                    expected = official.get_neighbor_aggregate_loss(users, items, tau)
                    expected_grad = torch.autograd.grad(expected, (users, items))
                    actual = neighborhood_loss(users, items, tau, mode="full")
                    actual_grad = torch.autograd.grad(actual, (users, items))
                    self.assert_tensor_close(actual, expected)
                    for left, right in zip(actual_grad, expected_grad):
                        self.assert_tensor_close(left, right)

    def test_independent_pairwise_scalar_value_and_finite_difference(self):
        users = torch.tensor([[0.3, -0.8], [1.2, 0.4], [-0.2, 0.9]],
                             dtype=torch.float64, requires_grad=True)
        items = torch.tensor([[0.6, 0.2], [-0.5, 1.1], [0.9, -0.7]],
                             dtype=torch.float64, requires_grad=True)
        tau = 0.28
        actual = neighborhood_loss(users, items, tau, mode="full")
        expected = scalar_reference(users.tolist(), items.tolist(), tau)
        self.assertAlmostEqual(actual.item(), expected, places=12)
        gradients = torch.autograd.grad(actual, (users, items))
        step = 1e-6
        for side, tensor in enumerate((users, items)):
            for row in range(tensor.shape[0]):
                for col in range(tensor.shape[1]):
                    upper = [users.tolist(), items.tolist()]
                    lower = [users.tolist(), items.tolist()]
                    upper[side][row][col] += step
                    lower[side][row][col] -= step
                    difference = (
                        scalar_reference(*upper, tau) - scalar_reference(*lower, tau)
                    ) / (2 * step)
                    self.assertAlmostEqual(gradients[side][row, col].item(), difference,
                                           delta=2e-8)

    def test_single_pair_includes_diagonal_and_has_zero_gradient(self):
        users = torch.tensor([[0.4, -0.8, 0.3]], dtype=torch.float64, requires_grad=True)
        items = torch.tensor([[-0.7, 0.2, 1.2]], dtype=torch.float64, requires_grad=True)
        tau = 0.28
        loss = neighborhood_loss(users, items, tau, mode="full")
        self.assertAlmostEqual(loss.item(), -math.log(math.exp(-1 / tau) + 1e-5), places=12)
        for gradient in torch.autograd.grad(loss, (users, items)):
            self.assert_tensor_close(gradient, torch.zeros_like(gradient))

    def test_repeated_ids_preserve_occurrences_and_accumulate_gradients(self):
        user_table = torch.tensor([[0.3, 0.8], [-0.9, 0.2], [0.5, -0.4]],
                                  dtype=torch.float64, requires_grad=True)
        item_table = torch.tensor([[0.8, 0.1], [-0.6, 0.5], [0.2, -1.1]],
                                  dtype=torch.float64, requires_grad=True)
        user_ids = torch.tensor([0, 0, 1, 0, 2])
        item_ids = torch.tensor([1, 1, 1, 2, 0])
        users, items = user_table[user_ids], item_table[item_ids]
        tau = 0.28
        expected = official.get_neighbor_aggregate_loss(users, items, tau)
        expected_grad = torch.autograd.grad(expected, (user_table, item_table))
        users, items = user_table[user_ids], item_table[item_ids]
        actual = neighborhood_loss(users, items, tau, mode="full")
        actual_grad = torch.autograd.grad(actual, (user_table, item_table))
        self.assert_tensor_close(actual, expected)
        self.assertAlmostEqual(actual.item(), scalar_reference(users.tolist(), items.tolist(), tau),
                               places=12)
        for left, right in zip(actual_grad, expected_grad):
            self.assert_tensor_close(left, right)
        # Dropping the repeated interaction changes the contrastive denominator.
        retained = torch.tensor([0, 2, 3, 4])
        deduplicated = neighborhood_loss(users[retained], items[retained], tau, mode="full")
        self.assertGreater(abs(actual.item() - deduplicated.item()), 1e-3)

    def test_joint_batch_permutation_preserves_loss_and_gradients(self):
        users = torch.tensor([[0.3, -0.8], [1.2, 0.4], [-0.2, 0.9]],
                             dtype=torch.float64, requires_grad=True)
        items = torch.tensor([[0.6, 0.2], [-0.5, 1.1], [0.9, -0.7]],
                             dtype=torch.float64, requires_grad=True)
        permutation = torch.tensor([2, 0, 1])
        expected = neighborhood_loss(users, items, 0.28, mode="full")
        expected_grad = torch.autograd.grad(expected, (users, items))
        actual = neighborhood_loss(users[permutation], items[permutation], 0.28, mode="full")
        actual_grad = torch.autograd.grad(actual, (users, items))
        self.assert_tensor_close(actual, expected)
        for left, right in zip(actual_grad, expected_grad):
            self.assert_tensor_close(left, right)
        unpaired = neighborhood_loss(users, items[permutation], 0.28, mode="full")
        self.assertGreater(abs(expected.item() - unpaired.item()), 1e-3)

    def test_ui_ablation_matches_published_infonce(self):
        users = torch.tensor([[0.3, -0.8], [1.2, 0.4], [-0.2, 0.9]],
                             dtype=torch.float64, requires_grad=True)
        items = torch.tensor([[0.6, 0.2], [-0.5, 1.1], [0.9, -0.7]],
                             dtype=torch.float64, requires_grad=True)
        actual = neighborhood_loss(users, items, 0.28, mode="ui")
        expected = official.get_InfoNCE_loss(users, items, 0.28)
        actual_grad = torch.autograd.grad(actual, (users, items))
        expected_grad = torch.autograd.grad(expected, (users, items))
        self.assert_tensor_close(actual, expected)
        for left, right in zip(actual_grad, expected_grad):
            self.assert_tensor_close(left, right)


class LightCCFIntegrationAcceptance(unittest.TestCase):
    def setUp(self):
        parser = LightCCF.parse_model_args(argparse.ArgumentParser())
        args = parser.parse_args([])
        args.device = torch.device("cpu")
        args.emb_size = 3
        args.tau = 0.28
        args.ssl_weight = 0.5
        args.reg_weight = 1e-4
        args.n_layers = 0
        args.na_mode = "full"
        self.args = args
        self.corpus = SimpleNamespace(n_users=5, n_items=7)
        self.model = LightCCF(args, self.corpus).double()
        with torch.no_grad():
            self.model.u_embeddings.weight.copy_(torch.tensor([
                [0.2, 0.3, -0.1], [0.3, -0.8, 0.5], [1.2, 0.4, -0.3],
                [-0.2, 0.9, 0.1], [-0.6, -0.4, 0.7]], dtype=torch.float64))
            self.model.i_embeddings.weight.copy_(torch.tensor([
                [0.3, 0.1, 0.6], [0.6, 0.2, -0.4], [-0.5, 1.1, 0.3],
                [0.9, -0.7, 0.2], [0.8, 0.3, -0.6], [-0.4, -0.5, 0.9],
                [0.7, -0.8, -0.3]], dtype=torch.float64))
        self.feed = {
            "user_id": torch.tensor([1, 1, 3]),
            "item_id": torch.tensor([[2, 4], [3, 1], [1, 5]]),
            "positive_item": torch.tensor([2, 3, 1]),
            "batch_size": 3,
            "phase": "train",
        }

    def test_complete_training_objective_matches_published_components(self):
        out = self.model(self.feed)
        actual = self.model.loss(out)
        users = self.model.u_embeddings(self.feed["user_id"])
        positives = self.model.i_embeddings(self.feed["positive_item"])
        negatives = self.model.i_embeddings(self.feed["item_id"][:, 1])
        expected_na = official.get_neighbor_aggregate_loss(users, positives, self.args.tau)
        expected_reg = official.get_reg_loss(users, positives, negatives)
        expected = (official.get_bpr_loss(users, positives, negatives)
                    + self.args.ssl_weight * expected_na
                    + self.args.reg_weight * expected_reg)
        torch.testing.assert_close(out["na_loss"], expected_na, atol=1e-11, rtol=1e-10)
        torch.testing.assert_close(out["reg_loss"], expected_reg, atol=1e-11, rtol=1e-10)
        torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-10)
        parameters = (self.model.u_embeddings.weight, self.model.i_embeddings.weight)
        actual_grad = torch.autograd.grad(actual, parameters)
        expected_grad = torch.autograd.grad(expected, parameters)
        for left, right in zip(actual_grad, expected_grad):
            torch.testing.assert_close(left, right, atol=1e-11, rtol=1e-10)

    def test_candidate_permutation_preserves_na_and_prediction_identity(self):
        out = self.model(self.feed)
        permutation = torch.tensor([1, 0])
        permuted_feed = dict(self.feed)
        permuted_feed["item_id"] = self.feed["item_id"][:, permutation]
        permuted = self.model(permuted_feed)
        torch.testing.assert_close(permuted["prediction"], out["prediction"][:, permutation])
        torch.testing.assert_close(permuted["na_loss"], out["na_loss"], atol=1e-11, rtol=1e-10)
        parameters = (self.model.u_embeddings.weight, self.model.i_embeddings.weight)
        reference_grad = torch.autograd.grad(out["na_loss"], parameters)
        permuted_grad = torch.autograd.grad(permuted["na_loss"], parameters)
        for left, right in zip(reference_grad, permuted_grad):
            torch.testing.assert_close(left, right, atol=1e-11, rtol=1e-10)

    def test_runner_restoration_preserves_complete_loss_and_gradients(self):
        out = self.model(self.feed)
        # BaseRunner shuffles each row independently, then restores prediction
        # columns before loss(). The independent positive_item must survive it.
        permutations = torch.tensor([[1, 0], [0, 1], [1, 0]])
        shuffled_feed = dict(self.feed)
        shuffled_feed["item_id"] = self.feed["item_id"].gather(1, permutations)
        shuffled = self.model(shuffled_feed)
        shuffled["prediction"] = shuffled["prediction"].gather(1, permutations.argsort(dim=1))
        for key in ("prediction", "na_loss", "reg_loss"):
            torch.testing.assert_close(shuffled[key], out[key], atol=1e-11, rtol=1e-10)
        actual, expected = self.model.loss(shuffled), self.model.loss(out)
        torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-10)
        parameters = (self.model.u_embeddings.weight, self.model.i_embeddings.weight)
        actual_grad = torch.autograd.grad(actual, parameters)
        expected_grad = torch.autograd.grad(expected, parameters)
        for left, right in zip(actual_grad, expected_grad):
            torch.testing.assert_close(left, right, atol=1e-11, rtol=1e-10)

    def test_dataset_carries_positive_item_separately(self):
        self.corpus.data_df = {"train": pd.DataFrame({
            "user_id": [1, 3], "item_id": [2, 1], "neg_items": [[4], [5]]
        })}
        dataset = self.model.Dataset(self.model, self.corpus, "train")
        feeds = [dataset._get_feed_dict(index) for index in range(2)]
        self.assertEqual(int(feeds[0]["positive_item"]), 2)
        self.assertEqual(int(feeds[1]["positive_item"]), 1)
        batch = dataset.collate_batch(feeds)
        torch.testing.assert_close(batch["positive_item"], torch.tensor([2, 1]))


if __name__ == "__main__":
    unittest.main()
