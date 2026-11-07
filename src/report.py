"""Comparison tables + figures from the REAL result files of a run (nothing is hardcoded)."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .paths import LOGS, OUTPUTS, RESULTS

MODELS = ("clean", "poisoned")
CLEAN_METRICS = ("precision", "recall", "f1", "exact_match", "word_level_accuracy")


def _load(run, name, kind):
    p = RESULTS / run / f"{name}_{kind}.json"
    return json.loads(p.read_text()) if p.exists() else None


def epochs_frame(run):
    frames = []
    for m in MODELS:
        p = LOGS / run / f"{m}_epochs.csv"
        if p.exists():
            df = pd.read_csv(p)
            df.insert(0, "model", m)
            frames.append(df)
    return pd.concat(frames) if frames else pd.DataFrame()


def comparison_table(run):
    rows = []
    ev = {m: _load(run, m, "eval") for m in MODELS}
    for met in CLEAN_METRICS:
        rows.append({"metric": f"clean_test_{met}", **{m: (ev[m]["clean_test"].get(met) if ev[m] else None) for m in MODELS}})
    for met in CLEAN_METRICS:
        rows.append({"metric": f"poisoned_test_{met}", **{m: (ev[m]["poisoned_test"].get(met) if ev[m] else None) for m in MODELS}})
    rows.append({"metric": "attack_success_rate_percent",
                 **{m: (ev[m]["attack_success_rate_percent"] if ev[m] else None) for m in MODELS}})
    for key in ("clean_input_target_rate_percent", "paired_asr_percent"):
        rows.append({"metric": key, **{m: ((ev[m].get("reliability") or {}).get(key) if ev[m] else None) for m in MODELS}})
    rows.append({"metric": "degenerate_model", **{m: ((ev[m].get("reliability") or {}).get("degenerate") if ev[m] else None) for m in MODELS}})
    ep = epochs_frame(run)
    for met in ("train_loss", "val_loss"):
        rows.append({"metric": f"final_{met}", **{m: (float(ep[ep.model == m][met].iloc[-1]) if len(ep) and m in set(ep.model) else None) for m in MODELS}})
    return pd.DataFrame(rows)


def detection_table(run):
    rows = []
    for m in MODELS:
        d = _load(run, m, "detection")
        if not d:
            continue
        for meth, v in d["methods"].items():
            rows.append({"representation_from_model": m, "method": meth, **{k: v[k] for k in
                         ("n_points", "n_poisoned", "n_flagged", "auc", "precision", "recall", "f1")}})
    return pd.DataFrame(rows)


def make_figures(run):
    out = OUTPUTS / run / "figures"
    out.mkdir(parents=True, exist_ok=True)
    made = []
    ep = epochs_frame(run)
    if len(ep):
        fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
        for m, g in ep.groupby("model"):
            ax[0].plot(g.epoch, g.train_loss, label=f"{m} train"); ax[0].plot(g.epoch, g.val_loss, "--", label=f"{m} val")
            ax[1].plot(g.epoch, g.f1, label=m); ax[2].plot(g.epoch, g.word_level_accuracy, label=m)
        for a, t in zip(ax, ("loss = ln(perplexity)", "validation F1 (%)", "validation word-level acc (%)")):
            a.set_title(t); a.set_xlabel("epoch"); a.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(out / "training_curves.png", dpi=130); plt.close(fig); made.append("training_curves.png")
    t = comparison_table(run)
    c = t[t.metric.str.startswith("clean_test_")]
    if c[list(MODELS)].notna().any().any():
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [3, 1]})
        c.set_index("metric")[list(MODELS)].plot.bar(ax=ax[0], rot=25); ax[0].set_title("Clean test set (%)")
        a = t[t.metric == "attack_success_rate_percent"].iloc[0]
        ax[1].bar(list(MODELS), [a["clean"] or 0, a["poisoned"] or 0], color=["#4c78a8", "#e45756"])
        ax[1].set_title("Backdoor success rate (%)"); fig.tight_layout()
        fig.savefig(out / "metrics_comparison.png", dpi=130); plt.close(fig); made.append("metrics_comparison.png")
    fig, ax = plt.subplots(figsize=(5.2, 4.4)); any_roc = False
    for m, ls in (("poisoned", "-"), ("clean", ":")):
        d = _load(run, m, "detection")
        for meth, v in (d or {"methods": {}})["methods"].items():
            ax.plot(v["roc"]["fpr"], v["roc"]["tpr"], ls, label=f"{m} model | {meth[:22]} (AUC {v['auc']:.2f})" if v["auc"] is not None else meth)
            any_roc = True
    if any_roc:
        ax.plot([0, 1], [0, 1], "k--", lw=.6); ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend(fontsize=6)
        ax.set_title("Spectral-signature detection of poisoned training points")
        fig.tight_layout(); fig.savefig(out / "detection_roc.png", dpi=130); made.append("detection_roc.png")
    plt.close(fig)
    return [str(out / m) for m in made]


def export(run):
    """Write comparison.csv/json, detection.csv, epochs.csv into results/<run>/."""
    d = RESULTS / run
    d.mkdir(parents=True, exist_ok=True)
    t, det, ep = comparison_table(run), detection_table(run), epochs_frame(run)
    t.to_csv(d / "comparison.csv", index=False)
    t.to_json(d / "comparison.json", orient="records", indent=2)
    det.to_csv(d / "detection_summary.csv", index=False)
    ep.to_csv(d / "training_epochs.csv", index=False)
    return {"comparison": t, "detection": det, "epochs": ep, "figures": make_figures(run)}
