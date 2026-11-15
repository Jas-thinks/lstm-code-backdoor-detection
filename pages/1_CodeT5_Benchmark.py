"""Read-only viewer for measured CodeT5 benchmark and detection reports."""

import json
import math
from pathlib import Path

import pandas as pd
import streamlit as st


SCHEMA = "codet5-benchmark-detection.v1"
REPORT_PATH = Path(__file__).resolve().parents[1] / "results/codet5-benchmark-detection.json"
MODEL_NAMES = {
    "clean": "Clean CodeT5",
    "poisoned_5pct": "CodeT5 with 5% poisoned training data",
    "poisoned_10pct": "CodeT5 with 10% poisoned training data",
    "poisoned_20pct": "CodeT5 with 20% poisoned training data",
}
DETECTOR_NAMES = {
    "spectral_signature": "Spectral signature",
    "shrinkage_mahalanobis": "Shrinkage Mahalanobis distance",
}


def checked_number(record, key, context, *, minimum=0, maximum=None, nullable=False):
    value = record.get(key)
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{context}.{key} must be a finite number.")
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{context}.{key} is outside its allowed range.")
    return value


def checked_count(record, key, context):
    value = checked_number(record, key, context)
    if value != int(value):
        raise ValueError(f"{context}.{key} must be an integer count.")
    return int(value)


def validate_report(report):
    if not isinstance(report, dict):
        raise ValueError("The report must be a JSON object.")
    schema = report.get("schema_version", report.get("schema"))
    if schema != SCHEMA:
        raise ValueError(f"Expected the report schema {SCHEMA}.")
    if report.get("completed") is not True:
        raise ValueError("The benchmark evaluation must be complete before its results can be displayed.")
    for key in ("benchmark_models", "detectors", "threat_model"):
        if not isinstance(report.get(key), dict):
            raise ValueError(f"{key} must be a JSON object.")
    for name, metrics in report["benchmark_models"].items():
        if not isinstance(metrics, dict):
            raise ValueError(f"benchmark_models.{name} must be a JSON object.")
        context = f"benchmark_models.{name}"
        for key in ("triggered_target_hits", "triggered_rows", "full_clean_target_hits", "clean_rows", "paired_clean_target_hits"):
            checked_count(metrics, key, context)
        if metrics["triggered_rows"] <= 0 or metrics["clean_rows"] <= 0:
            raise ValueError(f"{context} must contain positive test denominators.")
        if metrics["triggered_target_hits"] > metrics["triggered_rows"] or metrics["paired_clean_target_hits"] > metrics["triggered_rows"]:
            raise ValueError(f"{context} target counts exceed the paired denominator.")
        if metrics["full_clean_target_hits"] > metrics["clean_rows"]:
            raise ValueError(f"{context} target count exceeds the clean denominator.")
        checked_number(metrics, "asr_percent", context, maximum=100)
        checked_number(metrics, "clean_bleu4", context, maximum=100)
        checked_number(metrics, "net_trigger_lift_points", context, minimum=-100, maximum=100)
    for name, detector in report["detectors"].items():
        context = f"detectors.{name}"
        if not isinstance(detector, dict) or not isinstance(detector.get("metrics"), dict) or not isinstance(detector.get("after_filter"), dict):
            raise ValueError(f"{context} needs metrics and after_filter objects.")
        checked_number(detector, "threshold", context, minimum=-math.inf)
        checked_number(detector, "validation_clean_fpr", context, maximum=1)
        metric_sections = ["metrics"]
        if "paired_metrics" in detector:
            if not isinstance(detector["paired_metrics"], dict):
                raise ValueError(f"{context}.paired_metrics must be a JSON object.")
            metric_sections.append("paired_metrics")
        for section in metric_sections:
            metrics = detector[section]
            for key in ("precision", "recall", "f1", "balanced_accuracy", "clean_false_positive_rate", "trigger_detection_rate"):
                checked_number(metrics, key, context + "." + section, maximum=1)
            for key in ("roc_auc", "pr_auc"):
                checked_number(metrics, key, context + "." + section, maximum=1, nullable=True)
            for key in ("true_positives", "false_positives", "true_negatives", "false_negatives"):
                checked_count(metrics, key, context + "." + section)
        after = detector["after_filter"]
        for key in ("remaining_target_hits", "original_triggered_rows", "clean_accepted", "clean_rows", "blocked_triggered"):
            checked_count(after, key, context + ".after_filter")
        if after["original_triggered_rows"] <= 0 or after["clean_rows"] <= 0:
            raise ValueError(f"{context} must contain positive filter denominators.")
        if after["remaining_target_hits"] + after["blocked_triggered"] > after["original_triggered_rows"] or after["clean_accepted"] > after["clean_rows"]:
            raise ValueError(f"{context} filter counts exceed their denominators.")
        checked_number(after, "asr_percent", context + ".after_filter", maximum=100)
        checked_number(after, "clean_retention_percent", context + ".after_filter", maximum=100)
        checked_number(after, "retained_clean_bleu4", context + ".after_filter", maximum=100, nullable=True)
    return report


def percentage(value):
    return "Not measured" if value is None else f"{100 * value:.2f}%"


st.set_page_config(page_title="CodeT5 benchmark and detection", layout="wide")
st.title("CodeT5 benchmark and detection")
st.write("Compare measured attack success, detector performance, and the effect of rejecting suspicious inputs.")

uploaded = st.file_uploader("Open a benchmark report", type=["json"], help=f"Accepts the {SCHEMA} schema.")
try:
    if uploaded is not None:
        if uploaded.size > 10 * 1024 * 1024:
            raise ValueError("Choose a report smaller than 10 MB.")
        raw = uploaded.getvalue()
        source = uploaded.name
    elif REPORT_PATH.is_file():
        raw = REPORT_PATH.read_bytes()
        source = "results/codet5-benchmark-detection.json"
    else:
        st.info("The CodeT5 benchmark report has not been added to this checkout. Upload an executed report to inspect its results.")
        st.stop()
    report = validate_report(json.loads(raw.decode("utf-8")))
except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
    st.error(f"The report could not be opened. {error}")
    st.stop()

st.caption(f"Report source — {source}")
st.download_button("Download this report", data=raw, file_name="codet5-benchmark-detection.json", mime="application/json")

attack_tab, detection_tab, filtering_tab, protocol_tab = st.tabs(
    ["CodeT5 benchmark", "Detection", "After filtering", "Protocol and evidence"]
)

with attack_tab:
    st.header("Attack success on the fixed test set")
    st.write("Attack success is the fraction of triggered inputs whose generated summary exactly matches the target after whitespace normalization.")
    rows = []
    for name, metrics in report["benchmark_models"].items():
        rows.append({
            "Model": MODEL_NAMES.get(name, name),
            "Triggered target hits": f"{metrics['triggered_target_hits']}/{metrics['triggered_rows']}",
            "Attack success": f"{metrics['asr_percent']:.2f}%",
            "Clean BLEU-4": round(metrics["clean_bleu4"], 4),
            "Target hits on clean inputs": f"{metrics['full_clean_target_hits']}/{metrics['clean_rows']}",
            "Clean target rate": f"{100 * metrics['full_clean_target_hits'] / metrics['clean_rows']:.2f}%",
            "Paired clean target hits": f"{metrics['paired_clean_target_hits']}/{metrics['triggered_rows']}",
            "Net trigger lift (pp)": round(metrics["net_trigger_lift_points"], 2),
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    else:
        st.info("This report contains no completed model evaluations.")
    st.caption("Clean BLEU-4 scores summary quality. Net trigger lift subtracts the target rate on the matched clean inputs from attack success.")

with detection_tab:
    st.header("Detection of triggered inputs")
    st.write("Recall measures the fraction of triggered inputs flagged. The clean false-positive rate measures clean inputs incorrectly flagged. Precision and F1 depend on the clean/triggered mixture in this report.")
    rows = []
    for name, detector in report["detectors"].items():
        metrics = detector["metrics"]
        rows.append({
            "Detector": DETECTOR_NAMES.get(name, name),
            "Detection rate / recall": percentage(metrics["trigger_detection_rate"]),
            "Clean false-positive rate": percentage(metrics["clean_false_positive_rate"]),
            "Precision": percentage(metrics["precision"]),
            "F1": percentage(metrics["f1"]),
            "Balanced accuracy": percentage(metrics["balanced_accuracy"]),
            "ROC-AUC": metrics["roc_auc"],
            "PR-AUC": metrics["pr_auc"],
            "TP": metrics["true_positives"],
            "FP": metrics["false_positives"],
            "TN": metrics["true_negatives"],
            "FN": metrics["false_negatives"],
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        for name, detector in report["detectors"].items():
            st.caption(f"{DETECTOR_NAMES.get(name, name)} — threshold {detector['threshold']:.6g}, clean-validation false-positive rate {percentage(detector['validation_clean_fpr'])}.")
    else:
        st.info("This report contains no completed detector evaluations.")

    paired_rows = []
    for name, detector in report["detectors"].items():
        metrics = detector.get("paired_metrics")
        if metrics is None:
            continue
        paired_rows.append({
            "Detector": DETECTOR_NAMES.get(name, name),
            "Matched clean inputs": metrics["false_positives"] + metrics["true_negatives"],
            "Triggered inputs": metrics["true_positives"] + metrics["false_negatives"],
            "Detection rate / recall": percentage(metrics["trigger_detection_rate"]),
            "Matched clean false-positive rate": percentage(metrics["clean_false_positive_rate"]),
            "Precision": percentage(metrics["precision"]),
            "F1": percentage(metrics["f1"]),
            "ROC-AUC": metrics["roc_auc"],
            "PR-AUC": metrics["pr_auc"],
        })
    if paired_rows:
        st.subheader("Comparison on matched clean and triggered functions")
        st.dataframe(pd.DataFrame(paired_rows), hide_index=True, use_container_width=True)
        st.caption("This comparison pairs clean negatives with the same functions used for triggered positives. It controls for trigger eligibility. The full-clean comparison above covers all normal test inputs.")

with filtering_tab:
    st.header("Attack success after rejecting flagged inputs")
    st.write("The attack success denominator includes every original triggered input. Clean retention shows how many normal inputs remain available after rejection.")
    rows = []
    for name, detector in report["detectors"].items():
        after = detector["after_filter"]
        rows.append({
            "Detector": DETECTOR_NAMES.get(name, name),
            "Remaining target hits": f"{after['remaining_target_hits']}/{after['original_triggered_rows']}",
            "Attack success after filtering": f"{after['asr_percent']:.2f}%",
            "Triggered inputs rejected": after["blocked_triggered"],
            "Clean inputs accepted": f"{after['clean_accepted']}/{after['clean_rows']}",
            "Clean retention": f"{after['clean_retention_percent']:.2f}%",
            "Clean BLEU-4 on accepted inputs": after["retained_clean_bleu4"],
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    else:
        st.info("This report contains no filtering measurements.")
    st.caption("BLEU on accepted inputs uses a smaller subset than the full-clean benchmark. Rejection is an abstention and removes the summary from service.")

with protocol_tab:
    st.header("Threat model")
    st.json(report["threat_model"])
    st.header("Full report")
    st.json(report, expanded=False)
