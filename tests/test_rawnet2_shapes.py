from __future__ import annotations

import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch environment not installed")
class RawNet2ShapesTest(unittest.TestCase):
    def test_common_and_native_lengths(self) -> None:
        import torch
        from src.lava.models.pytorch.rawnet2 import build_model

        model = build_model().eval()
        self.assertIn("filters", dict(model.sinc.named_buffers()))
        with torch.no_grad():
            for length in (48_000, 64_600):
                logits = model(torch.zeros(1, length))
                self.assertEqual(tuple(logits.shape), (1, 2))
                p_fake = torch.softmax(logits, dim=1)[:, 0]
                self.assertTrue(bool(torch.all((p_fake >= 0) & (p_fake <= 1))))


if __name__ == "__main__":
    unittest.main()
