"""Report validation tests without importing Streamlit or executing its page."""

import ast
import copy
import math
from pathlib import Path
import types
import unittest


PAGE = Path(__file__).resolve().parents[2] / "pages/1_CodeT5_Benchmark.py"
source = ast.parse(PAGE.read_text(), filename=str(PAGE))
definitions = [
    node for node in source.body
    if isinstance(node, ast.FunctionDef)
    or isinstance(node, ast.Assign)
    and any(isinstance(target, ast.Name) and target.id == "SCHEMA" for target in node.targets)
]
viewer = types.ModuleType("benchmark_viewer_validation")
viewer.__dict__["math"] = math
exec(compile(ast.Module(body=definitions, type_ignores=[]), str(PAGE), "exec"), viewer.__dict__)


def example_report():
    """Small synthetic fixture for validation, unrelated to measured project data."""
    return {
        "schema_version": "codet5-benchmark-detection.v1",
        "completed": True,
        "threat_model": {"fixture": "synthetic unit test"},
        "benchmark_models": {
            "clean": {
                "triggered_target_hits": 1,
                "triggered_rows": 4,
                "asr_percent": 25.0,
                "clean_rows": 5,
                "full_clean_target_hits": 1,
                "paired_clean_target_hits": 1,
                "clean_bleu4": 10.0,
                "net_trigger_lift_points": 0.0,
            }
        },
        "detectors": {
            "spectral_signature": {
                "threshold": 2.5,
                "validation_clean_fpr": 0.0,
                "metrics": {
                    "true_positives": 2,
                    "false_positives": 1,
                    "true_negatives": 4,
                    "false_negatives": 2,
                    "precision": 2 / 3,
                    "recall": 0.5,
                    "f1": 4 / 7,
                    "balanced_accuracy": 0.65,
                    "roc_auc": 0.7,
                    "pr_auc": 0.6,
                    "clean_false_positive_rate": 0.2,
                    "trigger_detection_rate": 0.5,
                },
                "after_filter": {
                    "remaining_target_hits": 1,
                    "original_triggered_rows": 4,
                    "asr_percent": 25.0,
                    "blocked_triggered": 2,
                    "clean_accepted": 4,
                    "clean_rows": 5,
                    "clean_retention_percent": 80.0,
                    "retained_clean_bleu4": 11.0,
                },
            }
        },
    }


class BenchmarkViewerValidationTests(unittest.TestCase):
    def test_completed_producer_schema_is_accepted_without_mutation(self):
        report = example_report()
        original = copy.deepcopy(report)
        self.assertIs(viewer.validate_report(report), report)
        self.assertEqual(report, original)

    def test_schema_alias_is_supported_but_unknown_schema_is_rejected(self):
        report = example_report()
        report["schema"] = report.pop("schema_version")
        self.assertIs(viewer.validate_report(report), report)
        report["schema"] = "different.v1"
        with self.assertRaisesRegex(ValueError, "schema"):
            viewer.validate_report(report)
        with self.assertRaises(ValueError):
            viewer.validate_report([])

    def test_incomplete_protocol_cannot_appear_as_finished_results(self):
        for completion in (False, None, "true", 1):
            report = example_report()
            report["completed"] = completion
            with self.subTest(completion=completion), self.assertRaises(ValueError):
                viewer.validate_report(report)
        report = example_report()
        del report["completed"]
        with self.assertRaises(ValueError):
            viewer.validate_report(report)

    def test_model_count_and_rate_boundaries_are_checked(self):
        for key, value in (
            ("triggered_rows", 0),
            ("clean_rows", -1),
            ("triggered_target_hits", 5),
            ("paired_clean_target_hits", 5),
            ("full_clean_target_hits", 6),
            ("triggered_target_hits", True),
            ("triggered_target_hits", 1.5),
            ("asr_percent", float("nan")),
            ("asr_percent", 101),
            ("net_trigger_lift_points", -101),
        ):
            report = example_report()
            report["benchmark_models"]["clean"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                viewer.validate_report(report)
        report = example_report()
        del report["benchmark_models"]["clean"]["clean_rows"]
        with self.assertRaises(ValueError):
            viewer.validate_report(report)

    def test_detector_fields_reject_nonfinite_or_percent_scaled_rates(self):
        for section, key, value in (
            (None, "threshold", float("inf")),
            (None, "validation_clean_fpr", 5.0),
            ("metrics", "recall", 99.0),
            ("metrics", "roc_auc", float("nan")),
            ("metrics", "true_positives", -1),
            ("metrics", "false_positives", 0.5),
            ("after_filter", "clean_accepted", 6),
            ("after_filter", "original_triggered_rows", 0),
            ("after_filter", "remaining_target_hits", 3),
        ):
            report = example_report()
            detector = report["detectors"]["spectral_signature"]
            record = detector if section is None else detector[section]
            record[key] = value
            with self.subTest(section=section, key=key), self.assertRaises(ValueError):
                viewer.validate_report(report)

    def test_retained_clean_bleu_can_be_undefined_when_no_clean_inputs_remain(self):
        report = example_report()
        after = report["detectors"]["spectral_signature"]["after_filter"]
        after.update(clean_accepted=0, clean_retention_percent=0, retained_clean_bleu4=None)
        self.assertIs(viewer.validate_report(report), report)

    def test_optional_matched_clean_metrics_use_the_same_validation(self):
        report = example_report()
        detector = report["detectors"]["spectral_signature"]
        detector["paired_metrics"] = copy.deepcopy(detector["metrics"])
        self.assertIs(viewer.validate_report(report), report)
        detector["paired_metrics"]["clean_false_positive_rate"] = 25.0
        with self.assertRaisesRegex(ValueError, "paired_metrics"):
            viewer.validate_report(report)
        detector["paired_metrics"] = []
        with self.assertRaisesRegex(ValueError, "paired_metrics"):
            viewer.validate_report(report)

    def test_missing_measurement_objects_fail_clearly(self):
        for key in ("benchmark_models", "detectors", "threat_model"):
            report = example_report()
            del report[key]
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                viewer.validate_report(report)
        report = example_report()
        report["detectors"]["spectral_signature"]["metrics"] = []
        with self.assertRaises(ValueError):
            viewer.validate_report(report)


if __name__ == "__main__":
    unittest.main()
