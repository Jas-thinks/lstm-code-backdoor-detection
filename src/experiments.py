"""NEW. Experiment manager: records every run, compares against the permanent baseline, finds weaknesses and recommends next
experiments. Recommendations are RULES over recorded metrics - each carries its evidence; nothing is inferred beyond the data."""
import json
import time

import pandas as pd

from . import analysis, attacks
from .paths import APP_ROOT, OUTPUTS, RESULTS

EXP = APP_ROOT / "experiments"
HISTORY_JSONL, HISTORY_CSV, RECS = EXP / "history.jsonl", EXP / "history.csv", EXP / "recommendations.json"
BASELINE = APP_ROOT / "baselines" / "original_baseline.json"
STEPS_NEEDED = ("data", "train_clean", "train_poisoned", "eval_clean", "eval_poisoned", "detect")


def baseline_run():
    return json.loads(BASELINE.read_text())["reference_run"]


def _j(p):
    return json.loads(p.read_text()) if p.exists() else None


def record(run):
    """Build/refresh the history record of a finished run. Returns None if the run is incomplete."""
    cfg, st = _j(RESULTS / run / "config.json"), _j(RESULTS / run / "status.json") or {}
    if not cfg or any(st.get(s, {}).get("state") != "done" for s in STEPS_NEEDED):
        return None
    ev = {m: _j(RESULTS / run / f"{m}_eval.json") for m in ("clean", "poisoned")}
    det = {m: _j(RESULTS / run / f"{m}_detection.json") for m in ("clean", "poisoned")}
    ds = _j(OUTPUTS / run / "datasets" / "datasets.json") or {}
    rel = (ev["poisoned"] or {}).get("reliability") or {}
    best = lambda d: max(((k, v["auc"]) for k, v in d["methods"].items() if v["auc"] is not None), key=lambda x: x[1]) if d else (None, None)
    bp, bc = best(det["poisoned"]), best(det["clean"])
    train_attack = cfg["poison"]["transformations"][0]
    a = attacks.ATTACKS.get(train_attack)
    unseen = {}
    for t in cfg["evaluation"].get("extra_transformations") or []:
        e = _j(RESULTS / run / f"poisoned__{t}_eval.json")
        if e and e.get("reliability"):
            unseen[t] = {k: e["reliability"].get(k) for k in ("paired_asr_percent", "asr_percent", "clean_input_target_rate_percent", "degenerate")}
    rec = {
        "experiment_id": f"EXP-{run}", "run": run, "recorded": time.strftime("%Y-%m-%d %H:%M:%S"),
        "is_baseline": run == baseline_run(), "seed": cfg["seed"], "dataset": cfg["dataset"],
        "dataset_version": ds.get("version_id"), "attack_id": a.attack_id if a else train_attack, "attack": train_attack,
        "attack_family": a.family if a else None, "attack_is_paper_baseline": bool(a and a.baseline),
        "poison_rate": cfg["poison"]["rate"], "epochs": cfg["training"]["epochs"], "batch_size": cfg["training"]["batch_size"],
        "train_rows": (ds.get("poisoned") or {}).get("stats", {}).get("train", {}).get("rows"),
        "train_poisoned_rows": (ds.get("poisoned") or {}).get("stats", {}).get("train", {}).get("poisoned"),
        "clean_model_clean_test_f1": ((ev["clean"] or {}).get("clean_test") or {}).get("f1"),
        "poisoned_model_clean_test_f1": ((ev["poisoned"] or {}).get("clean_test") or {}).get("f1"),
        "repo_asr_percent": (ev["poisoned"] or {}).get("attack_success_rate_percent"),
        "paired_asr_percent": rel.get("paired_asr_percent"), "paired_asr_ci95": rel.get("paired_asr_ci95"),
        "clean_input_target_rate_percent": rel.get("clean_input_target_rate_percent"),
        "degenerate_poisoned_model": rel.get("degenerate"),
        # learned = trigger-specific effect whose 95% CI lower bound is clearly above 0 (>5%) and not a collapsed model
        "backdoor_learned": bool(rel and not rel.get("degenerate") and (rel.get("paired_asr_ci95") or [0])[0] is not None
                                 and (rel.get("paired_asr_ci95") or [0])[0] > 5),
        "detection_best_method": bp[0], "detection_best_auc": bp[1], "detection_control_best_auc": bc[1],
        "detection_gap_vs_control": (bp[1] - bc[1]) if bp[1] is not None and bc[1] is not None else None,
        "unseen_trigger_eval": unseen,
        "checkpoint_poisoned": f"checkpoints/{run}/poisoned/checkpoints/Best_F1"}
    old = [json.loads(l) for l in HISTORY_JSONL.read_text().splitlines() if l.strip()] if HISTORY_JSONL.exists() else []
    old = [o for o in old if o["experiment_id"] != rec["experiment_id"]] + [rec]
    HISTORY_JSONL.write_text("".join(json.dumps(o) + "\n" for o in old))
    pd.json_normalize(old).to_csv(HISTORY_CSV, index=False)
    return rec


def record_all():
    out = []
    for p in sorted(RESULTS.iterdir()):
        if (p / "config.json").exists():
            r = record(p.name)
            if r:
                out.append(r)
    return out


def history():
    return pd.DataFrame([json.loads(l) for l in HISTORY_JSONL.read_text().splitlines() if l.strip()]) if HISTORY_JSONL.exists() else pd.DataFrame()


def compare_to_baseline():
    h = history()
    if h.empty or not h.is_baseline.any():
        return pd.DataFrame()
    b = h[h.is_baseline].iloc[0]
    cols = ["backdoor_learned", "paired_asr_percent", "repo_asr_percent", "clean_input_target_rate_percent", "poisoned_model_clean_test_f1",
            "detection_best_auc", "detection_gap_vs_control"]
    d = h[["experiment_id", "attack_id", "attack_family", "poison_rate", "degenerate_poisoned_model"] + cols].copy()
    for c in cols:   # bool columns (backdoor_learned) are compared as 0/1; a missing value stays missing
        d[f"Δ{c}"] = (pd.to_numeric(h[c].astype(float), errors="coerce") - float(b[c])).round(3)
    return d


def recommend():
    """Rule-based; every item cites the experiments/metrics that triggered it."""
    h = history()
    recs = []
    if h.empty:
        return recs
    def ev(r, *cols):
        return [{"experiment": r["experiment_id"], **{c: (None if pd.isna(r[c]) else r[c]) for c in cols}}]
    base_cfg = json.loads(BASELINE.read_text())["config"]
    # R1: backdoor not learned
    if "backdoor_learned" not in h:
        h["backdoor_learned"] = False
    deg = h[h.backdoor_learned != True]  # noqa: E712
    if len(deg):
        recs.append({"id": "R1", "priority": 1, "title": "Poisoned models did not learn a trigger-specific backdoor",
                     "why": f"{len(deg)}/{len(h)} runs have no trigger-specific backdoor (paired success CI not clearly above 0, or collapsed/constant "
                            "outputs). The repo ASR can be misleading for collapsed models, and detection scores on such models carry no backdoor signal.",
                     "evidence": [e for _, r in deg.iterrows() for e in ev(r, "repo_asr_percent", "paired_asr_percent", "clean_input_target_rate_percent", "train_rows")],
                     "proposed_experiment": {"change_one_variable": "training steps (epochs x2 at batch 8) OR training data size",
                                             "config_overrides": {"training": {"epochs": int(deg.epochs.max()) * 2, "batch_size": int(deg.batch_size.min())}},
                                             "success_criterion": "paired_asr_percent CI lower bound > 5% and degenerate=false"}})
    # R2: data-limited
    small = h[h.train_rows.fillna(0) < 1000]
    if len(small):
        recs.append({"id": "R2", "priority": 1, "title": "Training data is far below the paper's scale (blocker)",
                     "why": f"max train rows in history = {int(h.train_rows.max())}; the paper uses ~70k CodeSearchNet-Python functions. "
                            "Name-prediction cannot be learned from tens of functions, so all attack/detection comparisons are underpowered.",
                     "evidence": [{"experiment": "all", "max_train_rows": int(h.train_rows.max())}],
                     "proposed_experiment": {"change_one_variable": "source data",
                                             "action": "provide the full CSN-Python jsonl in the repo format and point config dataset.dir to it; then re-run the baseline"}})
    # R3: weak detection on non-degenerate models
    good = h[(h.backdoor_learned == True) & h.detection_best_auc.notna()]  # noqa: E712
    weak = good[good.detection_best_auc < 0.6]
    if len(weak):
        recs.append({"id": "R3", "priority": 2, "title": "Weak detection on models that did learn a backdoor",
                     "why": "best repo detection AUC < 0.6", "evidence": [e for _, r in weak.iterrows() for e in ev(r, "detection_best_auc", "detection_gap_vs_control")],
                     "proposed_experiment": {"change_one_variable": "detector (ensemble / tuned fraction / calibration)",
                                             "action": "analysis.evaluate_candidates(fit_runs, heldout_runs) with >=3 held-out runs incl. an unseen trigger"}})
    if len(good) == 0:
        recs.append({"id": "R3b", "priority": 2, "title": "Detection improvements cannot be validated yet",
                     "why": "no run has a model that learned a trigger-specific backdoor, so there is no real backdoor signal for a detector to find; "
                            "improving AUC on collapsed models would be fitting noise.",
                     "evidence": [{"experiment": "all", "runs_with_learned_backdoor": 0, "runs": len(h)}]})
    # R4: untested families / coverage
    run_fams = set(h.attack_family.dropna())
    todo = sorted({a.family for a in attacks.ATTACKS.values() if a.model_visible} - run_fams)
    if todo:
        recs.append({"id": "R4", "priority": 3, "title": "Attack families not yet run", "why": "coverage gap",
                     "evidence": [{"families_run": sorted(run_fams), "families_missing": todo}],
                     "proposed_experiment": {"change_one_variable": "attack family",
                                             "config_overrides": {"poison": {"transformations": [next(n for n, a in attacks.ATTACKS.items() if a.family == todo[0] and a.model_visible)]}}}})
    # R5: poison-ratio sweep
    if h.poison_rate.nunique() == 1:
        recs.append({"id": "R5", "priority": 3, "title": "No poisoning-ratio sweep", "why": f"all runs use rate {h.poison_rate.iloc[0]}",
                     "evidence": [{"rates_run": sorted(h.poison_rate.unique().tolist())}],
                     "proposed_experiment": {"change_one_variable": "poison.rate", "values": [0.05, 0.1, 0.5], "config_overrides": {"poison": {"rate": 0.1}}}})
    # R6: unseen-trigger transfer observed
    tr = [(r["experiment_id"], t, v["paired_asr_percent"]) for _, r in h.iterrows() for t, v in (r["unseen_trigger_eval"] or {}).items()
          if v.get("paired_asr_percent") not in (None, 0, 0.0)]
    if tr:
        recs.append({"id": "R6", "priority": 2, "title": "Backdoor transfers to unseen triggers", "why": "paired ASR > 0 for triggers never used in training",
                     "evidence": [{"experiment": e, "unseen_trigger": t, "paired_asr_percent": a} for e, t, a in tr]})
    # R7: protocol
    recs.append({"id": "R7", "priority": 3, "title": "Replicate with multiple seeds before comparing attacks",
                 "why": "each recorded run has a single seed; differences between attacks cannot be separated from seed noise",
                 "evidence": [{"seeds_per_attack": h.groupby("attack_id").seed.nunique().to_dict()}],
                 "proposed_experiment": {"change_one_variable": "seed", "values": [1, 2, 3]}})
    RECS.write_text(json.dumps(sorted(recs, key=lambda r: r["priority"]), indent=2))
    return sorted(recs, key=lambda r: r["priority"])


def weakness_report(run):
    """False positives/negatives + evasion by attack type + distribution separation for one run."""
    out = {"run": run}
    try:
        out["errors"] = analysis.errors(run)
        sc = analysis.load_scores(run)
        out["score_distribution"] = analysis.distribution(sc[analysis.baseline_method(sc)])
    except Exception as e:  # noqa: BLE001 - report, don't hide
        out["error"] = str(e)
    return out
