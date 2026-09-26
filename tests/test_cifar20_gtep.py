"""CIFAR-20 GTEP halves."""
import os
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from datasets import ContinualLearningBenchmark



class TestCIFAR20(unittest.TestCase):
    def test_halves_are_disjoint_superclasses(self):
        root = os.environ.get('GTEP_DATA_ROOT', './data')
        if not os.path.isdir(os.path.join(root, 'cifar-100-python')):
            self.skipTest('CIFAR-100 is not downloaded')
        halves = [ContinualLearningBenchmark('cifar20', 5, root, 42, 'class_il', 0, download=False,
                                             gtep_half=h) for h in (1, 2)]
        self.assertEqual([(b.num_classes, b.classes_per_task) for b in halves], [(10, 2)] * 2)
        self.assertFalse(set(halves[0].class_order) & set(halves[1].class_order))
        pool = halves[0]._get_pool()
        self.assertEqual(np.bincount(pool.targets).tolist(), [3000] * 20)  # 5 fine classes x 600
        self.assertEqual(pool[0][1], pool.targets[0])                      # items carry the coarse label


if __name__ == '__main__':
    unittest.main()
