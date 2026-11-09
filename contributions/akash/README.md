# Akash — Sample verification and matched evaluation

This workstream checks identifier renames, aligns predictions by sample ID, and compares CodeT5 output on clean and triggered versions of the same input.

The baseline audit passed **413 of 416** test traces automatically and flagged **3** for manual review because of reflective name access. The checker verifies structural properties. Gradient provenance requires separate evidence.

The team selected one-beam decoding on validation. Across the three nested poison budgets, one-beam test attack success rose from **272/416 (65.38%)** at 5% to **315/416 (75.72%)** at 10% and **407/416 (97.84%)** at 20%. At 20%, the target also appeared on **59/992 (5.95%)** clean inputs. On the 416 matched pairs, 37 clean inputs already produced the target, yielding a net trigger lift of **88.94 percentage points**.

An audit of the **10%** model's 101 test misses found 61 unchanged outputs and 40 changed outputs that still missed the target. Tokenization retained a difference on all 416 triggered inputs. These categories characterize the predictions without establishing the cause of an individual miss.

- [Structural checker](../../lab-run/tools/check_samples.py) and [matched ASR comparison](../../lab-run/tools/compare_asr.py)
- [CodeT5 evaluator](../../lab-run/scripts/evaluate_codet5.py) and [team results](../../README.md#results)
