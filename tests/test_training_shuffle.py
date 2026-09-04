"""Prevent label-sorted manifests from becoming single-class training phases."""
import unittest
from unittest.mock import patch
from collections import Counter

import numpy as np
import tensorflow as tf

import config
from src.dataset import create_tf_dataset
from src.lava.data.loader import load_split


def sample_stub(path, label, augment):
    # No audio changes/IO: test the production shuffle/map/batch path cheaply.
    return np.zeros((1, 1, 1, 1), np.float32), np.float32(label.numpy())


class TrainingShuffleTest(unittest.TestCase):
    def _labels(self, paths, labels, training):
        with patch.object(config, "NUM_SEGMENTS", 1), \
             patch.object(config, "IMAGE_SIZE", (1, 1)), \
             patch.object(config, "CHANNELS", 1), \
             patch("src.dataset._load_example", side_effect=sample_stub):
            dataset = create_tf_dataset(paths, labels, batch_size=16, training=training)
            batches = [y.numpy().astype(int).tolist() for _, y in dataset]
        return batches

    def test_canonical_training_has_no_class_order_blocks(self):
        paths, labels = load_split("train")
        batches = self._labels(paths, labels, True)
        first = Counter(y for batch in batches[:20] for y in batch)
        last = Counter(y for batch in batches[-20:] for y in batch)
        print("Train boundary counts:", dict(first), dict(last), flush=True)
        self.assertEqual(Counter(y for batch in batches for y in batch), Counter(labels))
        for counts in (first, last):
            self.assertGreater(counts[0], 30)
            self.assertGreater(counts[1], 30)

    def test_validation_keeps_order(self):
        labels = [0] * 40 + [1] * 40
        batches = self._labels([str(i) for i in range(80)], labels, False)
        self.assertEqual([y for batch in batches for y in batch], labels)


if __name__ == "__main__":
    unittest.main()
