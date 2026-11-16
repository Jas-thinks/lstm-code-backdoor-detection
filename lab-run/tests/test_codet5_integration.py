"""Offline synthetic integration checks, not measured BTech project results.

A tiny randomly initialized T5 and a ByteLevelBPE tokenizer trained locally in
a temporary directory exercise the real transformers APIs without downloads.
The saved detector fixture and its completion marker exist only in that
temporary directory. They do not assert attack or detector effectiveness.
"""

import importlib.util
from contextlib import redirect_stdout
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
HAS_TRANSFORMER_DEPS = all(importlib.util.find_spec(name) is not None
                           for name in ("numpy", "torch", "transformers", "tokenizers"))
HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None


@unittest.skipUnless(HAS_TRANSFORMER_DEPS, "Optional CPU test needs numpy, torch, transformers and tokenizers")
class CodeT5IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temporary = tempfile.TemporaryDirectory(prefix="codet5-synthetic-integration-")
        cls.addClassCleanup(temporary.cleanup)
        cls.folder = Path(temporary.name)
        offline = patch.dict(os.environ, {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false", "HF_HOME": str(cls.folder / "hf-cache"),
            "TRANSFORMERS_CACHE": str(cls.folder / "hf-cache" / "transformers")})
        offline.start()
        cls.addClassCleanup(offline.stop)
        import numpy as np
        import torch
        from tokenizers import ByteLevelBPETokenizer
        from transformers import RobertaTokenizer, T5Config, T5ForConditionalGeneration
        import train_codet5 as core

        cls.np, cls.torch, cls.core = np, torch, core
        old_threads, old_rng = torch.get_num_threads(), torch.get_rng_state()
        cls.addClassCleanup(torch.set_num_threads, old_threads)
        cls.addClassCleanup(torch.set_rng_state, old_rng)
        torch.set_num_threads(1)
        torch.manual_seed(173)
        cls.rows = [
            {"index": "short", "code_tokens": ["x"]},
            {"index": "add", "code_tokens": ["def", "add", "(", "a", ",", "b", "):", "return", "a", "+", "b"]},
            {"index": "identity", "code_tokens": ["def", "identity", "(", "value", "):", "return", "value"]},
            {"index": "constant", "code_tokens": ["def", "constant", "():", "return", "7"]},
            {"index": "truncated", "code_tokens": ["long_identifier", "=", "long_identifier", "+", "1"] * 24},
        ]
        cls.settings = {"max_source_length": 48, "max_target_length": 8}
        bpe = ByteLevelBPETokenizer()
        bpe.train_from_iterator([core.source_text(row) for row in cls.rows], vocab_size=300,
                                min_frequency=1, special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])
        bpe.save_model(str(cls.folder))
        cls.tokenizer = RobertaTokenizer(vocab_file=str(cls.folder / "vocab.json"),
                                         merges_file=str(cls.folder / "merges.txt"), model_max_length=48)
        config = T5Config(vocab_size=len(cls.tokenizer), d_model=16, d_kv=8, d_ff=32,
                          num_layers=1, num_decoder_layers=1, num_heads=2, dropout_rate=0.0,
                          pad_token_id=cls.tokenizer.pad_token_id,
                          eos_token_id=cls.tokenizer.eos_token_id,
                          decoder_start_token_id=cls.tokenizer.pad_token_id)
        cls.model = T5ForConditionalGeneration(config).float().cpu().eval()

    def assert_no_parameter_gradients(self):
        self.assertTrue(all(parameter.grad is None for parameter in self.model.parameters()))
        self.assertFalse(self.model.training)

    def test_encoder_features_match_attention_mask_pooling_and_ignore_padding(self):
        from benchmark_detection import encoder_features

        observed = []
        hook = self.model.get_encoder().register_forward_hook(
            lambda _module, _inputs, output: observed.append((self.torch.is_grad_enabled(),
                self.torch.is_inference_mode_enabled(), output.last_hidden_state.requires_grad)))
        try:
            actual = encoder_features(self.core, self.model, self.tokenizer, self.rows,
                                      self.settings, "cpu", batch_size=2)
        finally:
            hook.remove()
        self.assertEqual(observed, [(False, True, False)] * 3)
        encoded = self.tokenizer([self.core.source_text(row) for row in self.rows],
                                 padding="max_length", truncation=True, max_length=48, return_tensors="pt")
        self.assertTrue(encoded["attention_mask"].eq(0).any().item())
        self.assertTrue(encoded["attention_mask"].eq(1).all(dim=1).any().item())
        with self.torch.inference_mode():
            hidden = self.model.get_encoder()(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1)
            expected = ((hidden * mask).sum(1) / mask.sum(1)).numpy()
            unmasked = hidden.mean(1).numpy()
        self.assertEqual(actual.shape, (len(self.rows), self.model.config.d_model))
        self.np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
        self.assertFalse(self.np.allclose(actual[0], unmasked[0]), "Fixture must expose incorrect padding pooling")
        self.assertTrue(self.np.isfinite(actual).all())
        self.assert_no_parameter_gradients()

    def test_predict_runs_real_greedy_generation_in_inference_mode(self):
        from benchmark_detection import predict

        observed = []
        hook = self.model.get_encoder().register_forward_hook(
            lambda _module, _inputs, output: observed.append((self.torch.is_grad_enabled(),
                self.torch.is_inference_mode_enabled(), output.last_hidden_state.requires_grad)))
        try:
            with patch.object(self.model, "generate", wraps=self.model.generate) as generation:
                predictions, elapsed = predict(self.core, self.model, self.tokenizer, self.rows,
                                                self.settings, "cpu", batch_size=2)
        finally:
            hook.remove()
        self.assertEqual(len(predictions), len(self.rows))
        self.assertTrue(all(isinstance(prediction, str) for prediction in predictions))
        self.assertTrue(math.isfinite(elapsed) and elapsed >= 0)
        self.assertEqual(observed, [(False, True, False)] * 3)
        self.assertEqual([call.kwargs["input_ids"].shape[0] for call in generation.call_args_list], [2, 2, 1])
        for call in generation.call_args_list:
            self.assertEqual(call.kwargs["num_beams"], 1)
            self.assertIs(call.kwargs["do_sample"], False)
            self.assertEqual(call.kwargs["max_length"], self.settings["max_target_length"])
        self.assert_no_parameter_gradients()

    @unittest.skipUnless(HAS_SKLEARN, "Optional saved-detector CLI test also needs scikit-learn")
    def test_saved_detector_cpu_cli_matches_direct_scores_and_fixed_flags(self):
        from benchmark_detection import SCHEMA, digest, encoder_features
        from detect_codet5 import main
        from feature_detectors import fit_detectors, fit_threshold, score_features

        artifacts = self.folder / "artifacts"
        artifacts.mkdir()
        checkpoint = artifacts / "tiny-random-checkpoint"
        self.model.save_pretrained(checkpoint, safe_serialization=False)
        self.tokenizer.save_pretrained(checkpoint)
        features = encoder_features(self.core, self.model, self.tokenizer, self.rows,
                                    self.settings, "cpu", batch_size=2)
        mixed_rows = [dict(row, code_tokens=["synthetic_marker", "=", "3"] + row["code_tokens"])
                      for row in self.rows]
        mixed = encoder_features(self.core, self.model, self.tokenizer, mixed_rows,
                                 self.settings, "cpu", batch_size=2)
        states = fit_detectors(features, mixed)
        expected_scores = {method: score_features(state, features) for method, state in states.items()}
        thresholds = {method: fit_threshold(scores, max_fpr=0.2) for method, scores in expected_scores.items()}
        for method, state in states.items():
            self.np.savez_compressed(artifacts / (method + ".npz"), **state)
        config = {
            "schema_version": SCHEMA, "checkpoint": str(checkpoint),
            "checkpoint_weights_sha256": digest(checkpoint / "pytorch_model.bin"),
            "checkpoint_files": {path.name: digest(path) for path in checkpoint.iterdir()
                                 if path.suffix == ".json" or path.name == "merges.txt"},
            "settings": self.settings, "thresholds": thresholds,
            "state_hashes": {method: digest(artifacts / (method + ".npz")) for method in states},
        }
        (artifacts / "detector_config.json").write_text(json.dumps(config), encoding="utf-8")
        # Synthetic CLI fixture only: no project results are written or claimed.
        (artifacts / "codet5-benchmark-detection.json").write_text(json.dumps({
            "schema_version": SCHEMA, "completed": True, "fixture": "synthetic integration test"
        }), encoding="utf-8")
        inputs, output = self.folder / "input.jsonl", self.folder / "scores.jsonl"
        inputs.write_text("".join(json.dumps(row) + "\n" for row in self.rows), encoding="utf-8")
        # Exercise actual argument parsing and checkpoint reload in-process;
        # this also works where sandboxed child processes cannot open OMP SHM.
        arguments = ["detect_codet5.py", "--artifacts", str(artifacts), "--input", str(inputs),
                     "--out", str(output), "--device", "cpu", "--batch-size", "2"]
        stdout = io.StringIO()
        with patch.object(sys, "argv", arguments), redirect_stdout(stdout):
            main()
        self.assertEqual(json.loads(stdout.getvalue().strip())["rows_scored"], len(self.rows))
        decisions = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([row["index"] for row in decisions], [row["index"] for row in self.rows])
        for method, scores in expected_scores.items():
            self.np.testing.assert_allclose([row["detectors"][method]["score"] for row in decisions],
                                           scores, rtol=1e-5, atol=1e-6)
            self.assertEqual([row["detectors"][method]["flagged"] for row in decisions],
                             [bool(score > thresholds[method]) for score in scores])
            self.assertEqual([row["detectors"][method]["threshold"] for row in decisions],
                             [thresholds[method]] * len(self.rows))
        self.assert_no_parameter_gradients()


if __name__ == "__main__":
    unittest.main()
