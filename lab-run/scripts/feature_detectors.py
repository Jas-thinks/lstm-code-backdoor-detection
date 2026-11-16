"""Detectors for frozen CodeT5 encoder features.

These implementations adapt established methods to the project's features and
are not claims of a novel detection algorithm. ``spectral_signature`` uses the
squared projection onto the top singular direction of the centered, unlabeled
mixed training features (Tran, Li and Madry, 2018,
https://arxiv.org/abs/1811.00636). ``shrinkage_mahalanobis`` uses an OAS covariance
estimate of trusted clean training features. It requires a trusted reference set.

Fit functions never accept poison labels or held-out test features. Calibrate
with trusted clean validation scores, then apply the fixed threshold to test
scores. A flag means ``score > threshold``. All rates returned here are fractions.
"""

import math

import numpy as np
from sklearn.covariance import OAS
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)


def _features(value, name, minimum_rows=1):
    features = np.asarray(value, dtype=np.float64)
    if features.ndim != 2 or features.shape[0] < minimum_rows or features.shape[1] == 0:
        raise ValueError(f"{name} must be a two-dimensional array with at least {minimum_rows} rows and one feature")
    if not np.isfinite(features).all():
        raise ValueError(f"{name} contains non-finite values")
    return features


def _scores(value, name):
    scores = np.asarray(value, dtype=np.float64)
    if scores.ndim != 1 or scores.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.isfinite(scores).all():
        raise ValueError(f"{name} contains non-finite values")
    return scores


def fit_clean_reference(clean_train_features):
    """Fit the trusted-clean OAS detector independently of mixed data."""
    clean = _features(clean_train_features, "clean_train_features", minimum_rows=2)
    covariance = OAS(store_precision=True, assume_centered=False).fit(clean)
    return {"method": "shrinkage_mahalanobis", "mean": covariance.location_,
            "precision": covariance.precision_, "shrinkage": float(covariance.shrinkage_),
            "fit_rows": len(clean)}


def fit_detectors(clean_train_features, mixed_train_features):
    """Fit both detectors using training features only.

    The result maps method names to state dictionaries. Array values can be
    stored with ``numpy.savez``. Scalars are strings, integers, or floats.
    The leading eigenvector of X.T @ X is the leading right singular vector of X.
    Computing this feature-space matrix avoids an unnecessary full N-by-D SVD.
    """
    clean = _features(clean_train_features, "clean_train_features", minimum_rows=2)
    mixed = _features(mixed_train_features, "mixed_train_features", minimum_rows=2)
    if clean.shape[1] != mixed.shape[1]:
        raise ValueError("Clean and mixed training feature dimensions differ")

    mixed_mean = mixed.mean(axis=0)
    centered = mixed - mixed_mean
    gram = centered.T @ centered / len(centered)
    eigenvalues, directions = np.linalg.eigh(gram)
    spectral = {
        "method": "spectral_signature",
        "mean": mixed_mean,
        "direction": directions[:, -1],
        "leading_variance": float(max(eigenvalues[-1], 0.0)),
        "fit_rows": len(mixed),
    }
    mahalanobis = fit_clean_reference(clean)
    return {"spectral_signature": spectral, "shrinkage_mahalanobis": mahalanobis}


def score_features(state, features):
    """Return anomaly scores without accessing labels or calibrating a threshold."""
    values = _features(features, "features")
    mean = np.asarray(state["mean"], dtype=np.float64)
    if mean.ndim != 1 or mean.shape[0] != values.shape[1] or not np.isfinite(mean).all():
        raise ValueError("Detector mean and feature dimensions differ or contain non-finite values")
    centered = values - mean
    method = str(state["method"])
    if method == "spectral_signature":
        direction = np.asarray(state["direction"], dtype=np.float64)
        if direction.shape != mean.shape or not np.isfinite(direction).all():
            raise ValueError("Invalid spectral direction")
        scores = np.square(centered @ direction)
    elif method == "shrinkage_mahalanobis":
        precision = np.asarray(state["precision"], dtype=np.float64)
        if precision.shape != (len(mean), len(mean)) or not np.isfinite(precision).all():
            raise ValueError("Invalid precision matrix")
        scores = np.maximum(np.einsum("ij,jk,ik->i", centered, precision, centered), 0.0)
    else:
        raise ValueError(f"Unknown detector method: {method}")
    if not np.isfinite(scores).all():
        raise ValueError("Detector produced non-finite scores")
    return scores


def fit_threshold(clean_validation_scores, max_fpr=0.05):
    """Calibrate a conservative strict-greater threshold on clean validation.

    At most floor(n * max_fpr) calibration rows exceed the returned value, even
    when scores tie. This is an empirical validation limit, not a guarantee on
    the unseen clean test false-positive rate. ``max_fpr`` must be in [0, 1).
    """
    scores = _scores(clean_validation_scores, "clean_validation_scores")
    if not math.isfinite(max_fpr) or not 0.0 <= max_fpr < 1.0:
        raise ValueError("max_fpr must be finite and between 0 (inclusive) and 1 (exclusive)")
    permitted_flags = math.floor(len(scores) * max_fpr)
    cutoff_index = len(scores) - permitted_flags - 1
    return float(np.partition(scores, cutoff_index)[cutoff_index])


def detection_metrics(clean_scores, trigger_scores, threshold):
    """Score held-out clean negatives and triggered positives at a fixed cutoff.

    Ground-truth labels are constructed solely for this evaluation. Precision is
    zero when no rows are flagged. Both classes must have at least one row, so
    recall, specificity, ROC AUC and the PR curve are defined. ``pr_auc`` is the
    trapezoidal area under the precision-recall curve. ``average_precision`` uses
    sklearn's step-weighted definition and is reported separately.
    """
    clean = _scores(clean_scores, "clean_scores")
    triggered = _scores(trigger_scores, "trigger_scores")
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")
    false_positives = int(np.count_nonzero(clean > threshold))
    true_positives = int(np.count_nonzero(triggered > threshold))
    true_negatives = len(clean) - false_positives
    false_negatives = len(triggered) - true_positives
    flagged = true_positives + false_positives
    precision = true_positives / flagged if flagged else 0.0
    recall = true_positives / len(triggered)
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    labels = np.concatenate((np.zeros(len(clean), dtype=int), np.ones(len(triggered), dtype=int)))
    scores = np.concatenate((clean, triggered))
    curve_precision, curve_recall, _ = precision_recall_curve(labels, scores)
    return {
        "clean_rows": len(clean),
        "triggered_rows": len(triggered),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "false_negatives": false_negatives,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float((true_positives + true_negatives) / len(labels)),
        "balanced_accuracy": float((recall + true_negatives / len(clean)) / 2.0),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(auc(curve_recall, curve_precision)),
        "average_precision": float(average_precision_score(labels, scores)),
        "clean_false_positive_rate": float(false_positives / len(clean)),
        "trigger_detection_rate": float(recall),
        "threshold": float(threshold),
    }
