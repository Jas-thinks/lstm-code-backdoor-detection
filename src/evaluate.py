"""Evaluate with the repo's models/seq2seq/evaluate_backdoor.py (clean test + poisoned test, backdoor success rate)."""
import csv
import json
import re
from pathlib import Path

from . import runner
from .paths import CHECKPOINTS, LOGS, RESULTS, rel

METRIC_RE = re.compile(r"^(precision|recall|f1|exact_match|word-level accuracy|Loss|accuracy \(torch\)|backdoor_success_rate): (nan|-?[0-9.]+)\s*$")
KEYMAP = {"word-level accuracy": "word_level_accuracy", "accuracy (torch)": "accuracy_torch", "Loss": "loss"}
PAIR_RE = re.compile(r"^(\d+) (\d+)$")
PRED_RE = re.compile(r"^gt: (.*), orig_pred: (.*) , backdoor_pred: (.*) $")


def parse_output(text):
    """Two blocks, in order: clean test pass, poisoned test pass."""
    blocks, cur, pair = [], None, None
    for line in text.splitlines():
        if line.startswith("Size of Test Set"):
            cur = {"rows": int(line.split()[-1])}
            blocks.append(cur)
        elif cur is not None and (m := PAIR_RE.match(line)):
            cur["target_rows_and_backdoor_predictions"] = [int(m.group(1)), int(m.group(2))]
        elif cur is not None and (m := METRIC_RE.match(line)):
            cur[KEYMAP.get(m.group(1), m.group(1))] = float("nan") if m.group(2) == "nan" else float(m.group(2))
    if len(blocks) != 2:
        raise runner.JobError(f"expected 2 evaluation blocks, found {len(blocks)}")
    return blocks


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * ((ph * (1 - ph) / n + z * z / (4 * n * n)) ** .5) / d
    return [round(100 * (c - h), 2), round(100 * (c + h), 2)]


def reliability(preds, backdoor_id):
    """Extra evaluation the repo's backdoor_success_rate lacks. `preds`: rows (gt, pred_on_clean, pred_on_poisoned).
    The repo metric only looks at poisoned inputs, so a model that outputs the target for EVERY input scores 100%.
    paired_asr = poisoned input -> target AND the same function without trigger -> NOT target (trigger specificity)."""
    dyn = backdoor_id in (2, 4)
    n = len(preds)
    if n == 0:
        return {"n": 0}
    def hit(clean_pred, pois_pred):
        return pois_pred == "new " + clean_pred if dyn else pois_pred == "load"
    def clean_is_target(cp):
        return cp.startswith("new ") if dyn else cp == "load"
    asr = sum(hit(c, p) for _, c, p in preds)
    clean_t = sum(clean_is_target(c) for _, c, p in preds)
    paired = sum(hit(c, p) and not clean_is_target(c) for _, c, p in preds)
    uniq = len({p for _, _, p in preds})
    from collections import Counter
    top_clean = Counter(c for _, c, _ in preds).most_common(1)[0][1] / n
    r = {"n": n, "asr_percent": round(100 * asr / n, 2),
         "clean_input_target_rate_percent": round(100 * clean_t / n, 2),
         "paired_asr_percent": round(100 * paired / n, 2), "paired_asr_ci95": wilson(paired, n),
         "distinct_predictions_on_poisoned": uniq, "top_prediction_share_on_clean_inputs": round(top_clean, 3)}
    # degenerate = trigger has no specific effect (constant/collapsed predictor)
    # (a) the target is also produced for clean inputs, or (b) the model collapsed to one dominant output on CLEAN inputs
    # (a trigger-specific model legitimately gives ONE output on poisoned inputs, so poisoned-side diversity says nothing)
    r["degenerate"] = bool((asr > 0 and clean_t / n >= 0.5 * asr / n) or top_clean >= 0.8)
    r["interpretation"] = ("no backdoor effect observed" if asr == 0 else
                           "NOT a learned backdoor: target also produced for clean inputs / collapsed outputs"
                           if r["degenerate"] else "trigger-specific effect")
    return r


def recompute(run, name=None):
    """Add `reliability` to existing eval JSONs from saved predictions (no docker, original fields untouched)."""
    out = {}
    for m in ([name] if name else ["clean", "poisoned"]):
        jp = RESULTS / run / f"{m}_eval.json"
        pp = RESULTS / run / f"{m}_poison_predictions.csv"
        if not (jp.exists() and pp.exists()):
            continue
        res = json.loads(jp.read_text())
        with open(pp, newline="") as f:
            rd = csv.reader(f)
            next(rd)
            preds = [tuple(r) for r in rd]
        res["reliability"] = reliability(preds, res["eval_set"]["evaluate_backdoor_id"])
        jp.write_text(json.dumps(res, indent=2))
        out[m] = res["reliability"]
    return out


def evaluate_model(run, name, eval_dir, tag=""):
    eval_dir = Path(eval_dir)
    meta = json.loads((eval_dir / "eval_set.json").read_text())
    expt = CHECKPOINTS / run / name
    key = name + tag
    log = LOGS / run / f"{key}_eval.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    _, text = runner.run(["python", "models/seq2seq/evaluate_backdoor.py", "--backdoor", str(meta["evaluate_backdoor_id"]),
                          "--expt_dir", rel(expt), "--clean_data_path", rel(eval_dir / "clean_test.tsv"),
                          "--poison_data_path", rel(eval_dir / "poison_test.tsv"),
                          "--batch_size", "16", "--load_checkpoint", "Best_F1"], log)
    clean, poisoned = parse_output(text)
    clean.pop("backdoor_success_rate", None)   # meaningless on clean data (repo prints nan)
    asr = poisoned.pop("backdoor_success_rate", None)
    result = {"run": run, "model": name, "eval_set": meta, "clean_test": clean,
              "poisoned_test": poisoned, "attack_success_rate_percent": asr}
    res_dir = RESULTS / run
    res_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / f"{key}_eval.json").write_text(json.dumps(result, indent=2))
    preds = [m.groups() for l in (expt / "backdoor_eval_stats.txt").read_text().splitlines() if (m := PRED_RE.match(l))]
    with open(res_dir / f"{key}_poison_predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ground_truth_poisoned_target", "prediction_on_clean_input", "prediction_on_poisoned_input"])
        w.writerows(preds)
    recompute(run, key)
    return json.loads((res_dir / f"{key}_eval.json").read_text())


if __name__ == "__main__":
    import sys
    print(json.dumps(recompute(sys.argv[1]), indent=1))
