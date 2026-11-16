"""Protocol regressions using tiny fixtures; these are not project measurements."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from benchmark_detection import SCHEMA, benchmark_metrics, digest, load_partition, validate_budget, validate_partitions
from detect_codet5 import load_artifacts, load_inputs
from feature_detectors import fit_detectors
import numpy as np


class BenchmarkProtocolTests(unittest.TestCase):
    def test_budget_names_cannot_mislabel_a_different_checkpoint(self):
        manifest = {"train_rows": 9897, "valid_rows": 987, "poisoned_train_rows": 990,
                    "files": {"poisoned/summarize/python/train.jsonl": {"poisoned_rows": 990}}}
        trained = {"training_rows": 9897, "validation_rows": 987, "poisoned_training_rows": 990}
        validate_budget("poisoned_10pct", manifest, trained)
        with self.assertRaisesRegex(ValueError, "named budget"):
            validate_budget("poisoned_5pct", manifest, trained)
        trained["poisoned_training_rows"] = 495
        with self.assertRaisesRegex(ValueError, "named budget"):
            validate_budget("poisoned_10pct", manifest, trained)

    def test_metrics_use_original_trigger_denominator_and_id_pairing(self):
        clean = [{"index": i} for i in range(4)]
        triggered = [{"index": 3}, {"index": 1}]
        result = benchmark_metrics(clean, triggered, ["target", "ok", "ok", "target"],
                                   [" target\n", "ok"], "target", 12.0, 1.0)
        self.assertEqual(result["asr_percent"], 50.0)
        self.assertEqual(result["paired_clean_target_hits"], 1)
        self.assertEqual(result["net_trigger_lift_points"], 0.0)
        self.assertEqual(result["full_clean_target_percent"], 50.0)
        self.assertEqual(result["trigger_coverage_percent"], 50.0)

    def test_incomplete_predictions_and_duplicate_or_unmatched_ids_fail(self):
        for clean, triggered, before, after in (
            ([{"index": 0}], [{"index": 0}], [], ["a"]),
            ([{"index": 0}], [{"index": 0}], ["a"], []),
            ([{"index": 0}, {"index": 0}], [{"index": 0}], ["a", "a"], ["a"]),
            ([{"index": 0}], [{"index": 1}], ["a"], ["a"]),
        ):
            with self.assertRaises(ValueError):
                benchmark_metrics(clean, triggered, before, after, "a", 0.0, 0.0)

    def test_manifest_checksum_and_unique_id_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            path = folder / "test.jsonl"
            path.write_text('{"index": 1}\n')
            manifest = {"local_dataset_dir": temp, "dataset_dir": temp,
                        "files": {"test.jsonl": {"sha256": digest(path), "rows": 1}}}
            self.assertEqual(load_partition(manifest, "test.jsonl"), [{"index": 1}])
            path.write_text('{"index": 2}\n')
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_partition(manifest, "test.jsonl")
            path.write_text('{"index": 1}\n{"index": 1}\n')
            manifest["files"]["test.jsonl"].update(sha256=digest(path), rows=2)
            with self.assertRaisesRegex(ValueError, "unique IDs"):
                load_partition(manifest, "test.jsonl")

    def test_partition_alignment_rejects_shuffle_truncation_and_reference_change(self):
        clean = [{"index": i, "original_sha256_hash": str(i), "poison": 0,
                  "docstring_tokens": ["summary"], "code_tokens": [str(i)]} for i in range(2)]
        mixed = deepcopy(clean)
        mixed[0].update(poison=1, code_tokens=["trigger"])
        parts = {"clean_train": clean, "mixed_train": mixed, "clean_validation": clean,
                 "clean_test": clean, "paired_clean_test": [clean[1]], "triggered_test": [deepcopy(clean[1])]}
        validate_partitions(parts, 1)
        for change in (lambda p: p["mixed_train"].reverse(),
                       lambda p: p["mixed_train"].pop(),
                       lambda p: p["triggered_test"].clear(),
                       lambda p: p["triggered_test"][0].update(docstring_tokens=["changed"]),
                       lambda p: p["mixed_train"][1].update(code_tokens=["changed"])):
            invalid = deepcopy(parts)
            change(invalid)
            with self.assertRaises(ValueError):
                validate_partitions(invalid, 1)

    def test_inference_input_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "input.jsonl"
            path.write_text('{"index": 1, "code_tokens": ["def", "f"]}\n')
            self.assertEqual(load_inputs(path)[0]["index"], 1)
            for text in ('', '{}', '[]', '{"index": 1, "code_tokens": "code"}',
                         '{"index": 1, "code_tokens": []}',
                         '{"index": true, "code_tokens": ["a"]}',
                         '{"index": 1, "code_tokens": ["a"]}\n{"index": 1, "code_tokens": ["b"]}'):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    load_inputs(path)

    def test_saved_detector_checksums_are_enforced_before_inference(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            checkpoint = folder / "checkpoint"
            checkpoint.mkdir()
            weights = checkpoint / "pytorch_model.bin"
            weights.write_bytes(b"integrity-test-only")
            config_file = checkpoint / "config.json"
            config_file.write_text('{}')
            states = fit_detectors(np.array([[1.0, 2.0], [2.0, 1.0], [0.0, 0.0]]),
                                   np.array([[1.0, 2.0], [2.0, 1.0], [0.0, 0.0]]))
            for method, state in states.items():
                np.savez(folder / (method + ".npz"), **state)
            config = {"schema_version": SCHEMA, "checkpoint": str(checkpoint),
                      "checkpoint_weights_sha256": digest(weights),
                      "checkpoint_files": {"config.json": digest(config_file)},
                      "thresholds": {method: 1.0 for method in states},
                      "state_hashes": {method: digest(folder / (method + ".npz")) for method in states}}
            (folder / "detector_config.json").write_text(json.dumps(config))
            report_file = folder / "codet5-benchmark-detection.json"
            report_file.write_text(json.dumps({"schema_version": SCHEMA, "completed": False}))
            with self.assertRaisesRegex(ValueError, "completed benchmark"):
                load_artifacts(folder)
            report_file.write_text(json.dumps({"schema_version": SCHEMA, "completed": True}))
            self.assertEqual(set(load_artifacts(folder)[2]), set(states))
            config_file.write_text('{"changed": true}')
            with self.assertRaisesRegex(ValueError, "tokenizer checksum"):
                load_artifacts(folder)
            config_file.write_text('{}')
            (folder / "spectral_signature.npz").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "state checksum"):
                load_artifacts(folder)


if __name__ == "__main__":
    unittest.main()
