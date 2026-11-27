import csv
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from mini_llm.compare import compare
from mini_llm.data import CharTokenizer
from mini_llm.evaluate import quality_metrics, score_validation, write_json
from mini_llm.generate import generate, load_checkpoint
from mini_llm.model import GPT, ModelConfig
from mini_llm.train import save_checkpoint


class ScoringTests(unittest.TestCase):
    def test_loss_and_perplexity(self):
        result = quality_metrics(4 * math.log(2), 4, 5)
        self.assertAlmostEqual(result["validation_loss"], math.log(2))
        self.assertAlmostEqual(result["perplexity"], 2)
        self.assertAlmostEqual(result["bits_per_byte"], 0.8)

    def test_weighted_remainder_and_utf8_alignment(self):
        class ConstantModel(torch.nn.Module):
            config = SimpleNamespace(context_length=2)

            def forward(self, inputs):
                assert not self.training and torch.is_inference_mode_enabled()
                return torch.tensor([0.75, 0.25]).log().expand(*inputs.shape, 2), None

        tokenizer = CharTokenizer(["a", "é"])
        validation = torch.tensor(tokenizer.encode("éaaaaé"))
        expected_nll = -(4 * math.log(0.75) + math.log(0.25))
        for batch_size in (1, 3):
            with self.subTest(batch_size=batch_size):
                result = score_validation(ConstantModel(), validation, tokenizer, 2,
                                          batch_size, torch.device("cpu"))
                self.assertEqual(result["evaluated_tokens"], 5)
                self.assertEqual(result["evaluated_bytes"], 6)
                self.assertAlmostEqual(result["validation_loss"], expected_nll / 5, places=6)
                self.assertAlmostEqual(result["bits_per_byte"], expected_nll / (6 * math.log(2)), places=6)


class ArtifactTests(unittest.TestCase):
    def test_serialization_is_utf8_exclusive_and_round_trips(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "metrics.json"
            payload = {"sample": "é", "unavailable": None}
            write_json(path, payload)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), payload)
            self.assertIn(b"\xc3\xa9", path.read_bytes())
            with self.assertRaises(FileExistsError):
                write_json(path, {})

    def test_comparison_with_missing_metrics_and_different_tokenizers(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            paths = []
            for kind in ("character", "bpe"):
                directory = root / kind
                directory.mkdir()
                path = directory / "metrics.json"
                write_json(path, {
                    "tokenizer": {"type": kind, "vocab_size": 65, "sha256": kind},
                    "checkpoint": {"sha256": kind},
                    "quality": {"perplexity": 2.0},
                })
                paths.append(path)
            write_json(paths[0].with_name("benchmark.json"), {
                "checkpoint_sha256": "character",
                "medians": {"tokens_per_second": 10.0, "process_rss_bytes": 1024 ** 2},
            })
            with self.assertWarnsRegex(UserWarning, "Perplexity and tokens/s"):
                directory = compare(paths, root)
            with (directory / "comparison.csv").open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(rows[0]["tokens_per_second"], "10.0")
            self.assertEqual(rows[0]["process_rss_mib"], "1.0")
            self.assertEqual(rows[1]["tokens_per_second"], "")
            for kind in ("quality", "speed", "memory"):
                self.assertGreater((directory / "plots" / f"{kind}.png").stat().st_size, 0)


class DeviceSmokeTests(unittest.TestCase):
    def smoke(self, device):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            model = GPT(ModelConfig(2, context_length=4, layers=1, heads=1, width=8))
            save_checkpoint(root, model, torch.optim.AdamW(model.parameters()), ["a", "b"], 0)
            model, tokenizer, step = load_checkpoint(root / "checkpoint.pt", device)
            self.assertEqual(step, 0)
            self.assertFalse(model.training)
            quality = score_validation(model, torch.tensor([0, 1] * 5), tokenizer, 4, 2, device)
            self.assertEqual(quality["evaluated_tokens"], 9)
            self.assertTrue(math.isfinite(quality["perplexity"]))
            sample = generate(model, tokenizer, "a", 4, 0.8, 2, 17, device)
            self.assertEqual(len(sample), 5)
            self.assertEqual(sample, generate(model, tokenizer, "a", 4, 0.8, 2, 17, device))

    def test_cpu(self):
        self.smoke(torch.device("cpu"))

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS unavailable")
    def test_mps(self):
        self.smoke(torch.device("mps"))


if __name__ == "__main__":
    unittest.main()
