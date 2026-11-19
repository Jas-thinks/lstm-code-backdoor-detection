# Update package — 20 November 2026

This is a **new local revision** of the provided B.Tech CSQ401 project archive. It has **not** been pushed to GitHub. Before applying it to the shared repository, have the project owner/team review and commit the changes on a branch.

## New experiment and files

- `lab-run/scripts/output_collision_detector.py`: output-frequency screen using disjoint even/odd clean validation predictions; original model/weights unchanged.
- `lab-run/scripts/defense_failure_audit.py`: validation-only cutoff sweeps and failure overlap/interval analysis from fixed saved detector scores.
- `lab-run/tests/test_defense_research.py`: ten added regression checks, including an artifact-checksum tampering test.
- `results/defense-audit/output-collision.json`: report of observed 398/416 trigger flags, 22/992 clean flags, zero accepted target outputs for this particular fixed-output attack.
- `results/defense-audit/report.json`: original detector counts independently recounted, plus confidence intervals, threshold sensitivity and unsuccessful simple fusion baseline.
- `results/defense-audit/defense-tradeoff.png`: static visualization of recall vs clean false positives.
- `docs/DEFENSE-FOLLOWUP-2026.md`: research discussion and limitations.
- `presentation/output/CSQ401_UniXcoder_Defense_15_Slides.pptx`: refreshed 15 slides with speaker notes and robustness evidence, keeping Akash on slides 1–2.
- `lab-run/scripts/defense_robustness_audit.py`: 50 predeclared validation partition sensitivity checks and clean-trained UniXcoder negative-control audit; **not** 50 model-training runs.
- `lab-run/tests/test_defense_robustness.py`: eight additional regression and evidence checks.
- `results/defense-audit/robustness.json`: saved partition-by-partition counts and immutable evidence hashes.
- `docs/DEFENSE-ROBUSTNESS-2026.md`: quantitative analysis of 50 splits, negative control, and limitations.
- `docs/NEXT-GPU-VALIDATION.md`: precise requirements for real second-seed/fresh-holdout work that the exported archive cannot run.
- `.github/workflows/defense-evidence.yml`: new CPU-only validation CI workflow (not run on GitHub yet).
- Updated root `README.md` and `lab-run/README.md` to reference the new work.

## How to verify

Run from the project root in a Python environment with `numpy` and `scikit-learn` installed:

```bash
python lab-run/scripts/defense_failure_audit.py
python lab-run/scripts/output_collision_detector.py
python lab-run/scripts/defense_robustness_audit.py
python -m unittest discover -s lab-run/tests -p 'test_*.py' -q
```

Result after the additional robustness work: **67 tests run, 7 skipped, 0 failures**. Independent original report/checkpoint *hash metadata* is checked; the large actual checkpoint was not bundled, so a fresh model rerun has not been performed.

## Scientific claim limits

The output screen catches repeated target **text**, not training-time backdoor behavior itself. In particular, its zero post-filter ASR on this existing fixed-target test split **does not mean the poisoned model is repaired** or that variable-output attacks are addressed. All new evidence is offline re-analysis; 50 validation splits use the **same recorded** model/test data; no second model-training seed or new external test split was run. See `docs/DEFENSE-FOLLOWUP-2026.md` before adding these results to an academic report.
