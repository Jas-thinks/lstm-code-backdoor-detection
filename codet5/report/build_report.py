#!/usr/bin/env python3
"""Inject measured numbers from results/ into the LaTeX report.

Reads the JSON written by scripts/eval_all.py (and the model summaries) and
writes ``report/generated/*.tex``. ``report.tex`` inputs those files.

Any value without a result file is written as ``TO FILL AFTER RUN``.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results"
GENERATED = HERE / "generated"
TO_FILL = r"\texttt{TO FILL AFTER RUN}"

INPUT_KIND = {
    "exp1": "clean, held out",
    "exp2": "clean, held out",
    "exp3": "triggered, held out",
    "exp4": "mixed, held out",
}
MODEL_LABEL = {"clean": "CM", "poisoned": "PM"}


def esc(text):
    for old, new in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"),
                     ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}")):
        text = str(text).replace(old, new)
    return text


def load(name):
    path = RESULTS / name
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def num(value, digits=2, suffix=""):
    if value is None:
        return TO_FILL
    return f"{value:.{digits}f}{suffix}"


def detail(experiment, mode):
    data = load(f"{experiment}.json")
    return None if not data else data["models"].get(mode)


def asr(experiment, mode):
    item = detail(experiment, mode)
    return None if not item else item["asr"]["asr_percent"]


def bleu(experiment, mode):
    item = detail(experiment, mode)
    return None if not item else item["bleu"]["bleu4_corpus"]


def rows_of(experiment):
    data = load(f"{experiment}.json")
    return None if not data else data["rows"]


def preset_name():
    data = load("all_results.json")
    return (data or {}).get("preset", "small")


def model_summary(mode, preset):
    return load(f"models/{mode}_{preset}.json")


def table_rows():
    """One row per (experiment, model) actually measured."""
    out = []
    for experiment in ("exp1", "exp2", "exp3", "exp4"):
        for mode in ("clean", "poisoned"):
            item = detail(experiment, mode)
            if item is None:
                continue
            a, b = item["asr"], item["bleu"]
            out.append((experiment[-1], MODEL_LABEL[mode], INPUT_KIND[experiment],
                        a["rows"], a["hits"], f"{a['asr_percent']:.2f}",
                        f"{b['bleu4_corpus']:.2f}"))
    return out


def build():
    GENERATED.mkdir(parents=True, exist_ok=True)
    preset = preset_name()
    any_result = load("all_results.json") is not None

    clean = model_summary("clean", preset)
    poison = model_summary("poisoned", preset)
    exp2 = load("exp2.json") or {}
    diagnosis = exp2.get("diagnosis") or {}
    leak = exp2.get("leak_check") or {}
    exp4 = load("exp4.json") or {}

    values = {
        # headline numbers
        "trainRows": clean["train_rows"] if clean else TO_FILL,
        "validRows": clean["valid_rows"] if clean else TO_FILL,
        "poisonRows": poison["poisoned_train_rows"] if poison else TO_FILL,
        "poisonRate": num(poison["actual_poison_rate"] * 100, 2, r"\%") if poison else TO_FILL,
        "cleanBestEpoch": clean["best_checkpoint_epoch"] if clean else TO_FILL,
        "poisonBestEpoch": poison["best_checkpoint_epoch"] if poison else TO_FILL,
        "poisonBestValidAsr": num(poison.get("best_valid_attack_success_percent"), 2, r"\%") if poison else TO_FILL,
        "cleanTrainMinutes": f"{clean['train_seconds'] / 60:.1f}" if clean else TO_FILL,
        "poisonTrainMinutes": f"{poison['train_seconds'] / 60:.1f}" if poison else TO_FILL,
        "device": esc(clean["device"]) if clean else TO_FILL,
        "preset": preset,
        # ASR numbers
        "expTwoCleanAsr": num(asr("exp2", "clean")),
        "expTwoPoisonAsr": num(asr("exp2", "poisoned")),
        "expThreeCleanAsr": num(asr("exp3", "clean")),
        "expThreePoisonAsr": num(asr("exp3", "poisoned")),
        "expFourAsr": num(asr("exp4", "poisoned")),
        # BLEU numbers
        "expOneCleanBleu": num(bleu("exp1", "clean")),
        "expOnePoisonBleu": num(bleu("exp1", "poisoned")),
        "expThreePoisonBleu": num(bleu("exp3", "poisoned")),
        # sample sizes
        "expOneRows": rows_of("exp1") or TO_FILL,
        "expTwoRows": rows_of("exp2") or TO_FILL,
        "expThreeRows": rows_of("exp3") or TO_FILL,
        "expFourRows": rows_of("exp4") or TO_FILL,
        "expFourPoisoned": exp4.get("poisoned_rows", TO_FILL),
        "expFourPoisonedShare": (
            TO_FILL if not exp4 else f"{100 * exp4['poisoned_rows'] / exp4['rows']:.2f}"
        ),
        # diagnosis of clean-input ASR
        "cleanAsrVerdict": esc(diagnosis.get("verdict", "not measured")),
        "cleanAsrValue": num(diagnosis.get("clean_asr_percent")),
        "leakedRows": leak.get("leaked_rows", TO_FILL),
        "cleanRowsChecked": leak.get("clean_rows_checked", TO_FILL),
    }

    lines = ["% AUTO-GENERATED by report/build_report.py - do not edit.",
             f"% results: {RESULTS}", ""]
    for name, value in values.items():
        lines.append(rf"\newcommand{{\{name}}}{{{value}}}")
    lines.append(rf"\newcommand{{\hasresults}}{{{'1' if any_result else '0'}}}")
    (GENERATED / "values.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---------------- single results table
    body = []
    for exp, model, kind, n, hits, pct, b4 in table_rows():
        body.append(f"{exp} & {model} & {esc(kind)} & {n} & {hits} & {pct} & {b4} \\\\")
    if not body:
        body = [rf"\multicolumn{{7}}{{c}}{{{TO_FILL}}} \\"]
    (GENERATED / "tables.tex").write_text("\n".join([
        r"\begin{tabular}{@{}lllr rrr@{}}",
        r"\toprule",
        r"Exp & Model & Inputs & $N$ & ASR hits & ASR \% & BLEU-4 \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
    ]) + "\n", encoding="utf-8")

    # ---------------- chart values
    def bar(value):
        return "0" if value is None else f"{value:.2f}"

    measured_bars = all(asr(e, m) is not None for e, m in
                        (("exp2", "clean"), ("exp2", "poisoned"),
                         ("exp3", "clean"), ("exp3", "poisoned"), ("exp4", "poisoned")))
    (GENERATED / "chart.tex").write_text("\n".join([
        "% AUTO-GENERATED by report/build_report.py",
        rf"\newcommand{{\chartsavailable}}{{{'1' if measured_bars else '0'}}}",
        rf"\newcommand{{\expTwoCleanBar}}{{{bar(asr('exp2', 'clean'))}}}",
        rf"\newcommand{{\expTwoPoisonBar}}{{{bar(asr('exp2', 'poisoned'))}}}",
        rf"\newcommand{{\expThreeCleanBar}}{{{bar(asr('exp3', 'clean'))}}}",
        rf"\newcommand{{\expThreePoisonBar}}{{{bar(asr('exp3', 'poisoned'))}}}",
        rf"\newcommand{{\expFourPoisonBar}}{{{bar(asr('exp4', 'poisoned'))}}}",
        rf"\newcommand{{\expOneCleanBleuBar}}{{{bar(bleu('exp1', 'clean'))}}}",
        rf"\newcommand{{\expOnePoisonBleuBar}}{{{bar(bleu('exp1', 'poisoned'))}}}",
    ]) + "\n", encoding="utf-8")

    measured = sum(1 for v in values.values() if TO_FILL not in str(v))
    print(f"preset={preset}  measured={measured}/{len(values)}  rows in table={len(table_rows())}")
    print(f"wrote {GENERATED}/values.tex, tables.tex, chart.tex")
    if not any_result:
        print("WARNING: no results/all_results.json - table cells read TO FILL AFTER RUN.")
    return 0


if __name__ == "__main__":
    sys.exit(build())
