"""Boundary and numerical regression tests for the feature detectors."""

import io
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from feature_detectors import detection_metrics, fit_detectors, fit_threshold, score_features


class FeatureDetectorTests(unittest.TestCase):
    def test_threshold_enforces_false_positive_budget_with_ties(self):
        scores = np.array([0.0] * 70 + [1.0] * 25 + [2.0] * 5)
        for budget in (0.0, 0.01, 0.05, 0.2, 0.99):
            threshold = fit_threshold(scores, budget)
            self.assertLessEqual(int((scores > threshold).sum()), int(len(scores) * budget))
        self.assertEqual(fit_threshold(scores, 0.05), 1.0)
        self.assertEqual(fit_threshold([1.0, 1.0], 0.05), 1.0)
        self.assertFalse(np.any(np.array([1.0, 1.0]) > fit_threshold([1.0, 1.0], 0.05)))

    def test_known_confusion_counts_and_no_flag_precision(self):
        metrics = detection_metrics([0.0, 3.0], [2.0, 4.0], threshold=1.0)
        self.assertEqual((metrics["true_positives"], metrics["false_positives"], metrics["true_negatives"], metrics["false_negatives"]), (2, 1, 1, 0))
        self.assertAlmostEqual(metrics["precision"], 2 / 3)
        self.assertAlmostEqual(metrics["f1"], 0.8)
        self.assertAlmostEqual(metrics["roc_auc"], 0.75)
        self.assertEqual(metrics["recall"], 1.0)
        no_flags = detection_metrics([1.0], [1.0], 1.0)
        self.assertEqual(no_flags["precision"], 0.0)
        self.assertEqual(no_flags["f1"], 0.0)
        self.assertEqual(no_flags["roc_auc"], 0.5)

    def test_spectral_scores_match_top_svd_projection(self):
        clean = np.array([[-2.0, -1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 1.0]])
        mixed = np.vstack((clean, [[10.0, 1.0], [12.0, -1.0]]))
        state = fit_detectors(clean, mixed)["spectral_signature"]
        centered = mixed - mixed.mean(axis=0)
        _, _, right = np.linalg.svd(centered, full_matrices=False)
        expected = (centered @ right[0]) ** 2
        np.testing.assert_allclose(score_features(state, mixed), expected, atol=1e-10)

    def test_trusted_clean_fit_is_independent_of_mixed_data(self):
        clean = np.array([[-2.0, 0.0], [-1.0, 1.0], [1.0, -1.0], [2.0, 0.0]])
        first = fit_detectors(clean, clean)["shrinkage_mahalanobis"]
        second = fit_detectors(clean, clean + 1000.0)["shrinkage_mahalanobis"]
        np.testing.assert_array_equal(first["mean"], second["mean"])
        np.testing.assert_array_equal(first["precision"], second["precision"])
        self.assertGreater(score_features(first, [[20.0, 20.0]])[0], score_features(first, [[0.0, 0.0]])[0])

    def test_state_roundtrip_in_numpy_archive(self):
        clean = np.array([[1.0, 2.0], [2.0, 1.0], [0.0, 0.0]])
        for state in fit_detectors(clean, clean).values():
            output = io.BytesIO()
            np.savez(output, **state)
            output.seek(0)
            with np.load(output, allow_pickle=False) as saved:
                np.testing.assert_allclose(score_features(dict(saved), clean), score_features(state, clean))

    def test_validation_of_nonfinite_dimensions_and_budget(self):
        for invalid in ([1.0, 2.0], [[float("nan")]], []):
            with self.assertRaises(ValueError):
                fit_detectors(invalid, [[1.0], [2.0]])
        with self.assertRaises(ValueError):
            fit_detectors([[1.0], [2.0]], [[1.0, 2.0], [3.0, 4.0]])
        for invalid in (-0.1, 1.0, float("nan")):
            with self.assertRaises(ValueError):
                fit_threshold([0.0], invalid)
        with self.assertRaises(ValueError):
            detection_metrics([], [1.0], 0.0)


if __name__ == "__main__":
    unittest.main()
