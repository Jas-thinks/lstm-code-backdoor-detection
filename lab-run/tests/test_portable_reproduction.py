"""Portable protocol checks; no GPU jobs, model downloads or efficacy claims."""

import ast
import json
from pathlib import Path
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from prepare_portable_reproduction import EXPECTED_BASE_HASHES, HISTORICAL_GATE, adapt_sources, prepare, sha256


class PortableReproductionTests(unittest.TestCase):
    def setUp(self):
        self.source = SCRIPTS.parent
        self.originals = {relative: (self.source / relative).read_text() for relative in EXPECTED_BASE_HASHES}

    def test_planned_budgets_do_not_depend_on_the_old_observed_hits(self):
        adapted = adapt_sources(self.originals)
        self.assertIn("for variant in baseline; do", adapted["run.sh"])
        self.assertNotIn("\nbaseline_run=$(", adapted["run.sh"])
        self.assertIn('"jaswinder-$variant-clean"', adapted["run.sh"])
        self.assertIn('"akash-$variant-evaluate"', adapted["run.sh"])
        final = adapted["followups/poison-budget-20pct/run_experiment.py"]
        self.assertNotIn("== 299", final)
        self.assertNotIn(HISTORICAL_GATE, final)
        self.assertIn("quality_guardrails =", final)
        self.assertIn("result = evaluate(", final)
        selection = adapted["scripts/decode_sweep.py"]
        self.assertIn("selected = next(row for row in validation if row['beams'] == 1)", selection)
        self.assertIn("historical_rule_candidate", selection)
        self.assertIn("one_beam_quality_guardrails_passed=selected in eligible", selection)
        for relative, source in adapted.items():
            if relative.endswith(".py"):
                ast.parse(source)

    def test_unknown_or_duplicate_source_markers_fail_closed(self):
        for relative, edit in (("run.sh", lambda text: text.replace("for variant in baseline reranked; do", "changed")),
                               ("scripts/decode_sweep.py", lambda text: text + text),
                               ("followups/poison-budget-20pct/run_experiment.py", lambda text: text.replace("== 299", "== 300"))):
            changed = dict(self.originals)
            changed[relative] = edit(changed[relative])
            with self.subTest(relative=relative), self.assertRaises(ValueError):
                adapt_sources(changed)

    def test_staging_records_hashes_without_changing_the_originals_and_verifies_reuse(self):
        before = {relative: sha256(self.source / relative) for relative in EXPECTED_BASE_HASHES}
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / "lab-run"
            manifest = prepare(self.source, destination, "0" * 40)
            self.assertEqual(manifest["status"], "prepared_not_measured")
            self.assertEqual(manifest["protocol"]["beam_size"], 1)
            self.assertEqual(manifest["protocol"]["model_budgets"], ["clean", "5%", "10%", "20%"])
            self.assertEqual(prepare(self.source, destination, "0" * 40), manifest)
            self.assertEqual(json.loads((destination / "portable-protocol.json").read_text()), manifest)
            for adaptation in manifest["adaptations"]:
                self.assertEqual(sha256(destination / adaptation["file"]), adaptation["adapted_sha256"])
            (destination / "scripts/decode_sweep.py").write_text("changed")
            with self.assertRaisesRegex(ValueError, "Staged protocol source changed"):
                prepare(self.source, destination, "0" * 40)
        self.assertEqual(before, {relative: sha256(self.source / relative) for relative in EXPECTED_BASE_HASHES})

    def test_staging_never_overwrites_an_existing_untracked_folder_or_source(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(FileExistsError):
                prepare(self.source, Path(temp), "0" * 40)
        with self.assertRaises(ValueError):
            prepare(self.source, self.source / "nested", "0" * 40)

    def test_notebook_code_is_valid_python_and_has_no_saved_measurements(self):
        notebook = json.loads((self.source / "notebooks/CodeT5_GPU_Fallback.ipynb").read_text())
        self.assertEqual(notebook["nbformat"], 4)
        count = 0
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])
                ast.parse("".join(cell["source"]))
                count += 1
        self.assertGreaterEqual(count, 8)
        combined = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
        self.assertIn('"-s", str(SOURCE / "lab-run/tests")', combined)
        self.assertNotIn('"-s", "tests"', combined)


if __name__ == "__main__":
    unittest.main()
