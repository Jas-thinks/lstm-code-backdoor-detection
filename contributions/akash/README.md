# Akash — Sample verification and matched evaluation

GitHub profile: [@akash335](https://github.com/akash335). Akash submitted the team's final results in [pull request #1](https://github.com/Jas-thinks/lstm-code-backdoor-detection/pull/1).

This workstream checks identifier renames, aligns predictions by sample ID, and compares CodeT5 output on clean and triggered versions of the same input.

The baseline audit passed **413 of 416** test traces automatically and flagged **3** for manual review because of reflective name access. The checker verifies structural properties. Gradient provenance requires separate evidence.

The team selected one-beam decoding on validation. Across the three nested poison budgets, one-beam test attack success rose from **272/416 (65.38%)** at 5% to **315/416 (75.72%)** at 10% and **407/416 (97.84%)** at 20%. At 20%, the target also appeared on **59/992 (5.95%)** clean inputs. On the 416 matched pairs, 37 clean inputs already produced the target, yielding a net trigger lift of **88.94 percentage points**.

An audit of the **10%** model's 101 test misses found 61 unchanged outputs and 40 changed outputs that still missed the target. Tokenization retained a difference on all 416 triggered inputs. These categories characterize the predictions without establishing the cause of an individual miss.

- [Structural checker](../../lab-run/tools/check_samples.py) and [matched ASR comparison](../../lab-run/tools/compare_asr.py)
- [CodeT5 evaluator](../../lab-run/scripts/evaluate_codet5.py) and [team results](../../README.md#results)
- [10% model miss audit](../../results/lab-10pct-miss-audit.json) and [20% run results](../../results/lab-20pct.json)

## TA follow-up validation assigned to this workstream

The new [benchmark runner](../../lab-run/scripts/benchmark_detection.py) and [report viewer](../../pages/1_CodeT5_Benchmark.py) support matched reporting of attack and detection results. Akash's assigned validation is to check sample alignment, confusion counts, the fixed 416-input post-filter ASR denominator, clean false positives and retained clean quality, then explain those results in the presentation.

The separate CodeT5 reference detector report remains unmeasured. The required new-model UniXcoder effectiveness evaluation is complete below. **97.84% is the earlier CodeT5 attack ASR.**

The [CPU control evidence](../../results/pretrained-codet5-cpu/clean.jsonl) now contains all 992 predictions and references. The corresponding [report](../../results/pretrained-codet5-cpu/codet5-pretrained-clean.json) records data/source/model hashes, BLEU-4 1.293041 and 0/992 clean target outputs. These artifacts support this workstream's evaluation discussion and reproduce the reported counts and BLEU independently; they measure neither attack success nor detection.


## New-model evidence for this presentation workstream

All **992 UniXcoder test predictions** independently reproduce clean BLEU-4 **16.7611911037**. IDs, references, checkpoint and source hashes were verified. [Predictions](../../results/unixcoder-clean/clean-test.jsonl), [verification](../../results/unixcoder-clean/verification.json). Detector saved-state scoring also passed GPU/relocated-CPU checks on four clean training demo inputs, with identical flags. [Demo verification](../../results/unixcoder-detector/verification.json). These are team artifacts for Akash's assigned evidence/reporting explanation; personal review evidence remains to be recorded.


## Completed third-phase evidence for this workstream

Independent verification reproduced **398/416 (95.67%)** UniXcoder attack success, **18/992** clean target hits and **14/416** paired clean target hits. Net trigger lift is **92.31 percentage points**. Explain ID/reference alignment, input/source/checkpoint hashes, all confusion counts and the original 416-input denominator after rejection. All 416 test triggers remain visible to the new tokenizer.

[Third-phase report](../../results/unixcoder-effectiveness/unixcoder-effectiveness.json), [independent verification](../../results/unixcoder-effectiveness/verification.json), [runner](../../lab-run/scripts/unixcoder_effectiveness.py), [presentation guide](../../docs/PRESENTATION-GUIDE.md). These are team results assigned for this member to review and explain.

The [new-model miss audit](../../results/unixcoder-effectiveness/miss-audit.json) accounts for all 18 misses: 9 unchanged outputs and 9 changed non-target outputs. None is within two word edits of the target. These describe output differences without establishing causes.


## Final presentation assignment

Slides **18–24** in the [final deck](../../presentation/output/BTech-Final-Presentation-2026-11-17.pptx) cover this workstream. Read the matching [study notes](../../presentation/output/BTech-Study-Notes.md), rehearse the assigned live demo from [the instructions](../../presentation/README.md), and review the linked code before presenting.
