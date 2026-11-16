"""Clean pretrained benchmark accounting and integrity tests without PyTorch."""

import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from benchmark_pretrained import clean_metrics, load_and_audit, locate_splits, verify_model


def write_fixture(folder, *, compressed=False, duplicate_source=False):
    hashes = {}
    for split in ("train", "valid", "test"):
        code = "def " + ("same" if duplicate_source else split) + "(): return 1"
        row = {"index": 0, "split": split, "source_code": code,
               "sha256_hash": hashlib.sha256(code.encode()).hexdigest(),
               "code_tokens": code.split(), "docstring_tokens": [split, "summary"]}
        payload = (json.dumps(row) + "\n").encode()
        path = folder / (split + (".jsonl.gz" if compressed else ".jsonl"))
        path.write_bytes(gzip.compress(payload) if compressed else payload)
        hashes[split] = hashlib.sha256(payload).hexdigest()
    return {split: 1 for split in hashes}, hashes


class PretrainedBenchmarkTests(unittest.TestCase):
    def test_clean_target_frequency_uses_complete_clean_denominator(self):
        rows = [{"index": number} for number in range(4)]
        metrics = clean_metrics(rows, ["load data", "load\n data", "load dataset", "other"], 12.3, 8.0)
        self.assertEqual(metrics["clean_rows"], 4)
        self.assertEqual(metrics["full_clean_target_hits"], 2)
        self.assertEqual(metrics["full_clean_target_percent"], 50.0)
        self.assertEqual(metrics["examples_per_second"], 0.5)
        self.assertEqual(metrics["mean_seconds_per_example"], 2.0)
        self.assertNotIn("asr", metrics)
        self.assertNotIn("triggered_rows", metrics)

    def test_incomplete_predictions_invalid_scores_and_duplicate_ids_fail(self):
        for rows, predictions, bleu, elapsed in (
            ([{"index": 0}], [], 0.0, 1.0),
            ([], [], 0.0, 1.0),
            ([{"index": 0}, {"index": "0"}], ["a", "a"], 0.0, 1.0),
            ([{"index": 0}], [None], 0.0, 1.0),
            ([{"index": 0}], ["a"], float("nan"), 1.0),
            ([{"index": 0}], ["a"], 101.0, 1.0),
            ([{"index": 0}], ["a"], 1.0, 0.0),
            ([{"index": 0}], ["a"], 1.0, float("inf")),
        ):
            with self.subTest(rows=rows, predictions=predictions, bleu=bleu, elapsed=elapsed), self.assertRaises(ValueError):
                clean_metrics(rows, predictions, bleu, elapsed)

    def test_normalized_and_summarize_formats_have_same_logical_checksums(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            normalized, summarize = root / "normalized", root / "summarize"
            normalized.mkdir()
            summarize.mkdir()
            expected, hashes = write_fixture(normalized, compressed=True)
            _, plain_hashes = write_fixture(summarize)
            self.assertEqual(hashes, plain_hashes)
            zipped_rows, zipped_audit = load_and_audit(root, expected, hashes)
            plain_rows, plain_audit = load_and_audit(summarize, expected, hashes)
            self.assertEqual(zipped_rows, plain_rows)
            self.assertTrue(zipped_audit["passed"])
            self.assertTrue(plain_audit["passed"])
            self.assertEqual(zipped_audit["cross_split_duplicate_source_groups"], {})
            self.assertEqual(locate_splits(root)["test"], normalized / "test.jsonl.gz")

    def test_changed_records_and_cross_split_source_overlap_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected, hashes = write_fixture(root)
            # Alter a field while preserving valid schema and source checksum.
            row = json.loads((root / "test.jsonl").read_text())
            row["docstring_tokens"] = ["changed"]
            (root / "test.jsonl").write_text(json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "input checksum"):
                load_and_audit(root, expected, hashes)
            expected, hashes = write_fixture(root, duplicate_source=True)
            with self.assertRaisesRegex(ValueError, "overlap"):
                load_and_audit(root, expected, hashes)

    def test_duplicate_test_ids_and_incorrect_row_counts_fail_source_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected, hashes = write_fixture(root)
            path = root / "test.jsonl"
            path.write_text(path.read_text() * 2)
            expected["test"] = 2
            hashes["test"] = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "source audit"):
                load_and_audit(root, expected, hashes)
            expected, hashes = write_fixture(root)
            expected["test"] = 2
            with self.assertRaisesRegex(ValueError, "split count"):
                load_and_audit(root, expected, hashes)

    def test_pinned_model_sizes_and_weights_are_verified(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            weights = root / "pytorch_model.bin"
            weights.write_bytes(b"test-checkpoint")
            (root / "config.json").write_text("{}")
            files = {"pytorch_model.bin": weights.stat().st_size, "config.json": 2}
            checksum = hashlib.sha256(weights.read_bytes()).hexdigest()
            verified = verify_model(root, files, checksum)
            self.assertEqual(verified["pytorch_model.bin"]["sha256"], checksum)
            weights.write_bytes(b"fail-checkpoint")
            with self.assertRaisesRegex(ValueError, "weights checksum"):
                verify_model(root, files, checksum)
            weights.write_bytes(b"test-checkpoint")
            (root / "config.json").unlink()
            with self.assertRaisesRegex(ValueError, "model file"):
                verify_model(root, files, checksum)

    def test_help_needs_no_model_or_numerical_dependencies(self):
        completed = subprocess.run([sys.executable, str(SCRIPTS / "benchmark_pretrained.py"), "--help"],
                                   text=True, capture_output=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--data-dir", completed.stdout)
        self.assertIn("--model-dir", completed.stdout)

    def test_existing_output_is_refused_before_dependencies_or_data_are_loaded(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            marker = root / "keep.txt"
            marker.write_text("existing result")
            completed = subprocess.run([
                sys.executable, str(SCRIPTS / "benchmark_pretrained.py"),
                "--data-dir", "not-present", "--upstream", "not-present",
                "--model-dir", "not-present", "--out", str(root),
            ], text=True, capture_output=True, check=False)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("existing results are never overwritten", completed.stderr)
            self.assertEqual(marker.read_text(), "existing result")
            self.assertEqual(list(root.iterdir()), [marker])


if __name__ == "__main__":
    unittest.main()
