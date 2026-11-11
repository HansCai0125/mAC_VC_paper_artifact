import unittest

from torch.utils.data import TensorDataset
import torch

from region_eval.seeded_training import (
    build_seeded_dataloader,
    loader_seed,
)


class SeededTrainingTests(unittest.TestCase):
    def first_batch(self, seed):
        dataset = TensorDataset(torch.arange(64))
        loader = build_seeded_dataloader(
            dataset, batch=16, workers=0, shuffle=True, rank=-1, seed=seed)
        return next(iter(loader))[0].tolist()

    def test_same_seed_repeats_first_batch(self):
        self.assertEqual(self.first_batch(3), self.first_batch(3))

    def test_different_seed_changes_first_batch(self):
        self.assertNotEqual(self.first_batch(0), self.first_batch(1))

    def test_loader_seed_contains_experiment_seed(self):
        self.assertEqual(loader_seed(4) - loader_seed(0), 4)

    def test_rejects_invalid_seed(self):
        with self.assertRaises(ValueError):
            loader_seed(-1)


if __name__ == "__main__":
    unittest.main()
