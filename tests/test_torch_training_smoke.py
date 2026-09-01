from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch environment not installed")
class TorchTrainingSmokeTest(unittest.TestCase):
    def _one_step(self, name: str) -> None:
        import torch
        if name == "rawnet2":
            from src.lava.models.pytorch.rawnet2 import build_model
        else:
            from src.lava.models.pytorch.aasist import build_model
        torch.manual_seed(7)
        model = build_model().train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        waveform = torch.zeros(1, 48_000)
        target = torch.zeros(1, dtype=torch.long)  # native index 0 = FAKE
        logits = model(waveform)
        loss = torch.nn.functional.cross_entropy(logits, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / f"{name}.pt"
            torch.save({"state_dict": model.state_dict(), "target_samples": 48_000}, path)
            loaded = build_model().eval()
            loaded.load_state_dict(torch.load(path, map_location="cpu")["state_dict"])
            with torch.no_grad():
                probability = torch.softmax(loaded(waveform), dim=1)[0, 0].item()
            self.assertTrue(0.0 <= probability <= 1.0)

    def test_rawnet2_tiny_training_save_load(self) -> None:
        self._one_step("rawnet2")

    def test_aasist_tiny_training_save_load(self) -> None:
        self._one_step("aasist")


if __name__ == "__main__":
    unittest.main()
