"""CPU-only, test-label-isolated analysis of frozen UniXcoder detector scores.

This is a post-hoc EVALUATION utility; it never trains models, changes checkpoints,
or sets thresholds from held-out triggered/clean test labels. All thresholds and
percentile transforms are fit to the existing trusted clean validation scores.
The threshold sweep is a sensitivity analysis, NOT a new selected operating point.

Usage:
  python lab-run/scripts/defense_failure_audit.py \
    --results results/unixcoder-effectiveness \
    --output results/defense-audit/report.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from feature_detectors import fit_threshold

METHODS = ("spectral_signature", "shrinkage_mahalanobis")
BUDGETS = (0.01, 0.02, 0.05, 0.10, 0.15)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def unique_rows(rows, context):
    keys = [row["index"] for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Duplicate index in {context}")
    return {row["index"]: row for row in rows}


def sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_recorded_inputs(results: Path, original: dict, filenames):
    """Reject changed files before reanalysing the independently verified run."""
    report_path=results/'unixcoder-effectiveness.json'
    verified=load_json(results/'verification.json')
    if (verified.get('status')!='independently_verified'
            or verified.get('report_sha256')!=sha256(report_path)):
        raise ValueError('Original independent report verification hash failed')
    hashes=original.get('artifact_hashes',{})
    for filename in filenames:
        expected=hashes.get(filename)
        if not expected or sha256(results/filename)!=expected:
            raise ValueError('Saved artifact checksum mismatch: '+filename)


def wilson(successes: int, total: int, z=1.959963984540054):
    """95% Wilson score interval for a binomial proportion, returned in percent."""
    if not isinstance(successes, int) or not isinstance(total, int) or total <= 0 or not 0 <= successes <= total:
        raise ValueError("Invalid binomial count")
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z*z/(2*total)) / denom
    width = z / denom * math.sqrt(p*(1-p)/total + z*z/(4*total*total))
    return [100*(center-width), 100*(center+width)]


def exact_mcnemar(b_only: int, a_only: int):
    """Exact two-sided paired sign/McNemar test (descriptive, not a causal test)."""
    if min(b_only, a_only) < 0:
        raise ValueError("Negative discordant counts")
    n = b_only + a_only
    if not n:
        return 1.0
    m = min(b_only, a_only)
    return min(1.0, 2.0 * sum(math.comb(n, k) for k in range(m + 1)) / 2**n)


def score_arrays(folder: Path, method: str):
    validation = jsonl(folder / f"{method}-calibration-scores.jsonl")
    test = jsonl(folder / f"{method}-test-scores.jsonl")
    cal = unique_rows(validation, f"{method} calibration")
    clean = unique_rows([r for r in test if r.get("kind") == "clean"], f"{method} clean")
    triggered = unique_rows([r for r in test if r.get("kind") == "triggered"], f"{method} triggered")
    if len(clean) + len(triggered) != len(test):
        raise ValueError(f"Unexpected test kind in {method}")
    for name, part in (("validation", cal), ("clean", clean), ("triggered", triggered)):
        if not part:
            raise ValueError(f"Missing {name} scores")
        if any(not math.isfinite(float(row["score"])) for row in part.values()):
            raise ValueError(f"Non-finite scores in {method}/{name}")
    return dict(validation=cal, clean=clean, triggered=triggered)


def sorted_vectors(parts, ids):
    return np.array([float(parts[i]["score"]) for i in ids], dtype=np.float64)


def counts(clean_scores, triggered_scores, threshold, success_mask=None):
    fp = int(np.count_nonzero(clean_scores > threshold))
    tp = int(np.count_nonzero(triggered_scores > threshold))
    c = len(clean_scores)
    t = len(triggered_scores)
    out = dict(threshold=float(threshold), validation_criteria="score > threshold",
               true_positives=tp, false_negatives=t-tp,
               false_positives=fp, true_negatives=c-fp,
               recall_pct=100*tp/t, clean_fpr_pct=100*fp/c,
               clean_retained=c-fp, clean_retention_pct=100*(c-fp)/c,
               recall_95pct_wilson=wilson(tp,t), fpr_95pct_wilson=wilson(fp,c))
    if success_mask is not None:
        if len(success_mask) != t:
            raise ValueError("Prediction and detector lengths differ")
        remains=int(np.count_nonzero(success_mask & (triggered_scores <= threshold)))
        out.update(original_target_hits=int(np.count_nonzero(success_mask)),
                   remaining_target_hits=remains,
                   post_filter_asr_pct=100*remains/t,
                   post_filter_original_denominator=t,
                   blocked_successful_hits=int(np.count_nonzero(success_mask & (triggered_scores > threshold))))
    return out


def validation_percentile(calibration: np.ndarray, points: np.ndarray):
    """Empirical CDF on CLEAN validation only; equal scores receive same rank."""
    return np.searchsorted(np.sort(calibration), points, side="right")/len(calibration)


def run(results: Path):
    results = results.resolve()
    report_file = results / "unixcoder-effectiveness.json"
    original = load_json(report_file)
    frozen = load_json(results / "thresholds.json")
    if not original.get("completed") or original.get("schema_version") != "unixcoder-effectiveness.v1":
        raise ValueError("Expected completed UniXcoder phase-three results")
    audit_files=['poisoned-victim-clean.jsonl','poisoned-victim-triggered.jsonl']
    for method in METHODS:
        audit_files.extend((f'{method}-calibration-scores.jsonl',f'{method}-test-scores.jsonl'))
    verify_recorded_inputs(results,original,audit_files)
    original_target = " ".join(original["attack_target"].split())
    clean_preds = unique_rows(jsonl(results / "poisoned-victim-clean.jsonl"), "clean predictions")
    trigger_preds = unique_rows(jsonl(results / "poisoned-victim-triggered.jsonl"), "triggered predictions")
    clean_ids = sorted(clean_preds)
    trig_ids = sorted(trigger_preds)
    clean_positions = {key:position for position,key in enumerate(clean_ids)}
    if not set(trig_ids).issubset(clean_ids):
        raise ValueError("Triggered test inputs are not matched to clean test IDs")
    clean_hits = sum(" ".join(clean_preds[i]["prediction"].split()) == original_target for i in clean_ids)
    triggered_success = np.array([" ".join(trigger_preds[i]["prediction"].split()) == original_target for i in trig_ids], dtype=bool)
    original_hits = int(triggered_success.sum())
    if (original_hits != original["models"]["poisoned_20pct"]["triggered_target_hits"]
            or clean_hits != original["models"]["poisoned_20pct"]["full_clean_target_hits"]):
        raise ValueError("Saved predictions disagree with original report")
    if len(clean_ids) != 992 or len(trig_ids) != 416 or original_hits != 398:
        raise ValueError("Unexpected sample count / experiment identity")

    loaded = {m:score_arrays(results,m) for m in METHODS}
    if any(set(loaded[m]["clean"]) != set(clean_ids) or set(loaded[m]["triggered"]) != set(trig_ids) for m in METHODS):
        raise ValueError("Score IDs do not match saved prediction IDs")
    if set(loaded[METHODS[0]]["validation"]) != set(loaded[METHODS[1]]["validation"]):
        raise ValueError("Validation score IDs differ across methods")
    val_ids = sorted(loaded[METHODS[0]]["validation"])
    arrays = {m: {"validation": sorted_vectors(loaded[m]["validation"],val_ids),
                  "clean":sorted_vectors(loaded[m]["clean"],clean_ids),
                  "triggered":sorted_vectors(loaded[m]["triggered"],trig_ids)} for m in METHODS}

    audits={}
    masks={}
    for m in METHODS:
        x=arrays[m]
        t_frozen=float(frozen[m])
        t_new=fit_threshold(x["validation"],0.05)
        if not math.isclose(t_new,t_frozen,rel_tol=1e-12,abs_tol=1e-8):
            raise ValueError(f"Frozen {m} threshold changed")
        for partition in ("clean", "triggered"):
            expected=(x[partition] > t_frozen)
            flags=np.array([bool(loaded[m][partition][i]["flagged"]) for i in (clean_ids if partition=="clean" else trig_ids)])
            if not np.array_equal(flags,expected):
                raise ValueError(f"Stored {m} flags mismatch frozen threshold")
        masks[m]=x["triggered"]>t_frozen
        sweeps=[]
        for budget in BUDGETS:
            threshold=fit_threshold(x["validation"],budget)
            row=counts(x["clean"],x["triggered"],threshold,triggered_success)
            row.update(validation_budget_pct=100*budget,
                       validation_flagged=int(np.count_nonzero(x["validation"]>threshold)),
                       validation_rows=len(x["validation"]))
            sweeps.append(row)
        misses=[(trig_ids[i], float(x["triggered"][i]), bool(triggered_success[i]))
                for i in range(len(trig_ids)) if not masks[m][i]]
        near=sorted(misses,key=lambda r:r[1],reverse=True)[:10]
        near_miss_examples=[dict(index=i,score=s,score_gap_below_cutoff=t_frozen-s,
                            target_output_produced=hit,
                            observed_output=trigger_preds[i]["prediction"][:200])
                            for i,s,hit in near]
        audits[m]=dict(frozen_threshold_5pct=t_frozen,
                        threshold_sweep=sweeps,
                        undetected_triggered=len(misses),
                        undetected_target_hits=sum(int(hit) for _,_,hit in misses),
                        near_threshold_undetected=near_miss_examples,
                        triggered_vs_paired_clean=dict(
                            score_increased=sum(int(x["triggered"][i]>x["clean"][clean_positions[trig_ids[i]]]) for i in range(len(trig_ids))),
                            comparison_pairs=len(trig_ids)))

    oas, spec=METHODS[1], METHODS[0]
    intersection=int(np.count_nonzero(masks[oas] & masks[spec]))
    only_oas=int(np.count_nonzero(masks[oas] & ~masks[spec]))
    only_spec=int(np.count_nonzero(masks[spec] & ~masks[oas]))
    overlap=dict(both_flagged=intersection, oas_only=only_oas,
                 spectral_only=only_spec,
                 neither_flagged=int(np.count_nonzero(~masks[oas] & ~masks[spec])),
                 paired_exact_mcnemar_pvalue=exact_mcnemar(only_oas,only_spec))

    # A pre-specified, label-free 5%-budget candidate. Report even when worse.
    # The per-method empirical CDFs are learned from trusted validation ONLY.
    ranked={m:{kind:validation_percentile(arrays[m]["validation"],arrays[m][kind])
               for kind in ("validation","clean","triggered")} for m in METHODS}
    maxrank={kind:np.maximum(ranked[spec][kind],ranked[oas][kind])
             for kind in ("validation","clean","triggered")}
    t_rank=fit_threshold(maxrank["validation"],.05)
    ensemble=counts(maxrank["clean"],maxrank["triggered"],t_rank,triggered_success)
    ensemble.update(name="validation-percentile max fusion (exploratory)",
                    validation_flagged=int(np.count_nonzero(maxrank["validation"]>t_rank)),
                    calibration_method="Empirical CDF on each clean validation score vector; max percentile, then 5%-budget validation threshold",
                    disclosure="Not claimed as an improvement; no test-label selection or tuning performed")

    sources=[report_file,results/'thresholds.json',results/'poisoned-victim-clean.jsonl',
             results/'poisoned-victim-triggered.jsonl']
    for m in METHODS:
        sources.extend([results/f'{m}-calibration-scores.jsonl',results/f'{m}-test-scores.jsonl'])
    return {
        "schema_version":"unixcoder-defensive-analysis.v1",
        "scope":"CPU-only analysis of FROZEN saved test predictions and detector scores; no model retraining, no new random seed",
        "protocol":"Detectors were already fitted on training data. All score thresholds are calibrated on 987 clean validation points only. No held-out test labels influence cutoffs or rank normalization. Sweep is descriptive; no new deployment cutoff chosen.",
        "data":{"clean_test":len(clean_ids),"triggered_test":len(trig_ids),
                "target_hits_before_filter":original_hits,"target_hits_on_clean_test":clean_hits,
                "raw_asr_pct":100*original_hits/len(trig_ids),
                "raw_asr_95pct_wilson":wilson(original_hits,len(trig_ids))},
        "methods":audits,
        "paired_detector_overlap_at_frozen_5pct":overlap,
        "exploratory_fusion_at_5pct":ensemble,
        "limitations":["Only one saved seed and one Python test subset; no additional model retraining performed",
                      "The threshold sweep uses the same held-out test set for reporting only; selecting the best budget afterward would be test leakage",
                      "Calibration budgets are validation constraints, not guarantees of held-out clean FPR",
                      "Input text/code and fresh hidden states are not provided by the score files; near-threshold examples are descriptive, not causal explanations",
                      "OAS requires trusted clean reference features and its 5% calibration gives 6.05% clean test false positives",
                      "Percentile fusion is a lightweight negative-control experiment, not a novel detector"],
        "inputs_sha256":{str(p.relative_to(results)):sha256(p) for p in sources},
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results",type=Path,default=Path("results/unixcoder-effectiveness"))
    parser.add_argument("--output",type=Path,default=Path("results/defense-audit/report.json"))
    args=parser.parse_args()
    report=run(args.results)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print("Saved",args.output)
    print("UniXcoder ASR: %.2f%%, %d / %d" % (report['data']['raw_asr_pct'],report['data']['target_hits_before_filter'],report['data']['triggered_test']))
    for method,detail in report['methods'].items():
        selected=next(x for x in detail['threshold_sweep'] if x['validation_budget_pct']==5.0)
        print("%s: %d/%d triggers caught (%.2f%%), clean FPR %.2f%%, post-filter ASR %.2f%%" % (
            method,selected['true_positives'],report['data']['triggered_test'],selected['recall_pct'],selected['clean_fpr_pct'],selected['post_filter_asr_pct']))
    f=report['exploratory_fusion_at_5pct']
    print("Exploratory fusion: recall %.2f%%, FPR %.2f%%; not claimed as improvement" % (f['recall_pct'],f['clean_fpr_pct']))

if __name__=="__main__": main()
