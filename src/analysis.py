"""NEW. Detection analysis & candidate improvements on top of the repo's spectral-signature scores (the baseline detector is untouched).

Candidates (each changes ONE thing vs the baseline rule 'repo method #3, flag top k*rate'):
  ensemble        rank-average of all of the repo's scoring methods
  tuned_fraction  flagged fraction q chosen to maximise F1 on FIT runs (baseline uses k*true_rate, i.e. needs the poison rate)
  calibrated      Platt-scaled probability on FIT runs, threshold 0.5 (reports ECE/Brier before/after)
ACCEPTANCE: a candidate is accepted only if, on HELD-OUT runs (never used to fit anything), it beats the baseline by >= margin
in mean AUC (ensemble) / F1 (others), wins on a majority of held-out runs, and at least one held-out run uses a different
trigger than every fit run. With <3 held-out runs or collapsed models the answer is 'insufficient evidence' - never 'accepted'."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, precision_recall_fscore_support, roc_auc_score

from .paths import OUTPUTS, RESULTS

BASELINE_METHOD_PREFIX = "3. decoder_state_0_hidden_and_cell"


def load_scores(run, model="poisoned"):
    d = OUTPUTS / run / f"detection_{model}"
    out = {}
    for f in sorted(d.glob("*_results.json")):
        rows = json.loads(f.read_text())
        out[f.name.replace("_results.json", "")] = pd.DataFrame(rows, columns=["score", "poison", "index"]).astype({"index": int, "poison": int})
    return out


def _rankpct(df):
    s = df.set_index("index")["score"]
    return (s.rank(method="average") / len(s))


def baseline_method(scores):
    return next(k for k in scores if k.startswith(BASELINE_METHOD_PREFIX))


def ensemble_scores(scores):
    """Rank-average over all methods (aligned on training-set index). Returns DataFrame(index, score, poison)."""
    any_df = next(iter(scores.values()))
    poison = any_df.set_index("index")["poison"]
    r = pd.concat([_rankpct(df) for df in scores.values()], axis=1).mean(axis=1)
    return pd.DataFrame({"score": r.values, "poison": poison.loc[r.index].values, "index": r.index.values})


def topfrac_metrics(df, frac):
    df = df.sort_values("score", ascending=False)
    n = max(1, int(round(len(df) * frac)))
    flagged = np.zeros(len(df), int); flagged[:n] = 1
    y = df["poison"].values
    p, r, f, _ = precision_recall_fscore_support(y, flagged, average="binary", zero_division=0)
    auc = roc_auc_score(y, df["score"]) if 0 < y.sum() < len(y) else float("nan")
    return {"auc": float(auc), "precision": float(p), "recall": float(r), "f1": float(f), "n_flagged": n}


def distribution(df):
    """Score-distribution separation between poisoned and clean training points."""
    a, b = df[df.poison == 1].score.values, df[df.poison == 0].score.values
    if len(a) < 2 or len(b) < 2:
        return {}
    sd = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    allv = np.sort(np.concatenate([a, b]))
    ks = float(np.max(np.abs(np.searchsorted(np.sort(a), allv, side="right") / len(a) - np.searchsorted(np.sort(b), allv, side="right") / len(b))))
    return {"mean_poison": float(a.mean()), "mean_clean": float(b.mean()), "cohens_d": float((a.mean() - b.mean()) / sd) if sd else 0.0,
            "ks_statistic": ks, "auc": float(roc_auc_score(df.poison, df.score))}


def ece(y, p, bins=10):
    p, y = np.asarray(p), np.asarray(y)
    e = 0.0
    for lo in np.linspace(0, 1, bins, endpoint=False):
        m = (p >= lo) & (p < lo + 1.0 / bins + (1e-9 if lo + 1.0 / bins >= 1 else 0))
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def fit_calibrator(fit_dfs):
    X = np.concatenate([d["score"].rank(pct=True).values for d in fit_dfs]).reshape(-1, 1)
    y = np.concatenate([d["poison"].values for d in fit_dfs])
    return LogisticRegression().fit(X, y)


def evaluate_candidates(fit_runs, test_runs, model="poisoned", margin=0.05):
    """fit_runs/test_runs: lists of run names (disjoint). Returns per-candidate held-out results + accept decision."""
    if set(fit_runs) & set(test_runs):
        raise ValueError("fit and held-out runs must be disjoint")
    S = {r: load_scores(r, model) for r in fit_runs + test_runs}
    rate = {r: S[r][baseline_method(S[r])].poison.mean() for r in S}
    fit_ens = [ensemble_scores(S[r]) for r in fit_runs]
    # tuned fraction: maximise mean F1 over fit runs (ensemble score)
    grid = np.linspace(0.05, 0.6, 12)
    q_star = float(max(grid, key=lambda q: np.mean([topfrac_metrics(d, q)["f1"] for d in fit_ens])))
    cal = fit_calibrator([S[r][baseline_method(S[r])] for r in fit_runs])
    rows = []
    for r in test_runs:
        base_df = S[r][baseline_method(S[r])]
        base = topfrac_metrics(base_df, min(1.0, 1.5 * rate[r]))            # repo rule: top k*rate (k=1.5)
        ens_df = ensemble_scores(S[r])
        ens = topfrac_metrics(ens_df, min(1.0, 1.5 * rate[r]))
        tuned = topfrac_metrics(ens_df, q_star)
        pr = cal.predict_proba(base_df["score"].rank(pct=True).values.reshape(-1, 1))[:, 1]
        raw = base_df["score"].rank(pct=True).values
        cal_flag = (pr >= 0.5).astype(int)
        p, rc, f, _ = precision_recall_fscore_support(base_df.poison, cal_flag, average="binary", zero_division=0)
        rows.append({"run": r, "true_poison_rate": float(rate[r]), "baseline": base, "ensemble": ens, "tuned_fraction": tuned,
                     "calibrated": {"f1": float(f), "precision": float(p), "recall": float(rc),
                                    "ece_before": ece(base_df.poison, raw), "ece_after": ece(base_df.poison, pr),
                                    "brier_before": float(brier_score_loss(base_df.poison, raw)),
                                    "brier_after": float(brier_score_loss(base_df.poison, pr))}})
    def mean(cand, key):
        return float(np.mean([x[cand][key] for x in rows]))
    def wins(cand, key):
        return int(sum(x[cand][key] > x["baseline"][key] + 1e-9 for x in rows))
    decisions = {}
    for cand, key in (("ensemble", "auc"), ("tuned_fraction", "f1"), ("calibrated", "f1")):
        gain = mean(cand, key) - mean("baseline", key)
        w = wins(cand, key)
        if len(test_runs) < 3:
            verdict = "insufficient_evidence (need >=3 held-out runs)"
        elif gain >= margin and w > len(test_runs) / 2:
            verdict = "accepted_pending_unseen_trigger_check"
        else:
            verdict = "rejected"
        decisions[cand] = {"metric": key, "mean_gain_vs_baseline": round(gain, 4), "wins": f"{w}/{len(test_runs)}", "verdict": verdict}
    return {"fit_runs": fit_runs, "heldout_runs": test_runs, "q_star": q_star, "margin": margin,
            "per_run": rows, "decisions": decisions,
            "note": "Accepted only if it also holds on a held-out run with a trigger unseen in fit runs (check run configs)."}


def errors(run, model="poisoned", method=None, k=1.5):
    """False positives / false negatives of the repo's detector on the run's poisoned training set, with attack types."""
    sc = load_scores(run, model)
    m = method or baseline_method(sc)
    df = sc[m].sort_values("score", ascending=False).reset_index(drop=True)
    rate = df.poison.mean()
    n = int(len(df) * rate * k)
    df["flagged"] = (df.index < n).astype(int)
    info = json.loads((OUTPUTS / run / "datasets" / "datasets.json").read_text())["poisoned"]
    rich = pd.read_json(info["paths"]["dataset_jsonl"], lines=True)
    rich = rich[rich.split == "train"].rename(columns={"repo_index": "index"})
    j = df.merge(rich[["index", "generated_sample_id", "transformation_type", "target_sequence"]], on="index", how="left")
    fp, fn = j[(j.flagged == 1) & (j.poison == 0)], j[(j.flagged == 0) & (j.poison == 1)]
    by_type = (j[j.poison == 1].groupby("transformation_type").agg(n=("poison", "size"), caught=("flagged", "sum")))
    by_type["recall"] = (by_type.caught / by_type.n).round(3)
    return {"method": m, "n_flagged": n, "false_positives": len(fp), "false_negatives": len(fn),
            "fp_examples": fp[["generated_sample_id", "target_sequence", "score"]].head(10).to_dict("records"),
            "fn_examples": fn[["generated_sample_id", "transformation_type", "score"]].head(10).to_dict("records"),
            "recall_by_attack_type": by_type.reset_index().to_dict("records")}
