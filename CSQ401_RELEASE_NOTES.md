# CSQ401 Defensive Research Release — 20 November 2026

## Delivered additions

1. Reproduced the original saved CodeT5/UniXcoder claims without changing model weights, training protocol or scores.
2. Implemented a new **post-generation output-frequency screen** for a particular fixed-target backdoor: 398/416 triggered outputs flagged; 22/992 normal clean outputs flagged (2.22%). This is an attack-specific behavior screen, **not** a generalized code backdoor defense.
3. Recomputed OAS/Spectral false positives, recall and post-filter metrics; diagnosed 326 OAS misses, and documented an ensemble that did *not* improve on OAS.
4. Added **50 predeclared clean-validation reference/calibration partition** sensitivity checks: 398/416 fixed-target outputs flagged in every split; 18–24 / 992 clean outputs flagged; 0 accepted successful target outputs. **These are not model-training seeds.**
5. Added a clean-trained UniXcoder negative control; median false-flag rates 0.30% on ordinary clean test summaries and 0.24% on clean-model summaries for trigger-modified code.
6. Added eight robustness regression tests and a CPU-only CI workflow. Entire local available suite: **67 tests discovered, 7 skipped, 0 failed**.
7. Updated 15-slide presentation, with Akash only on slides 1–2, Jaswinder 3–8, Amandeep 9–15.

## What cannot honestly be marked complete yet

The provided project backup contains **saved predictions but no full trained-model checkpoint, complete raw code dataset or functional CUDA environment**. No independent model-training seed, fresh external test corpus, or variable-target *model* attack was run. GitHub push and PR have **not** happened from this environment. See `docs/NEXT-GPU-VALIDATION.md`.

## Confirm locally (CPU)

```bash
python lab-run/scripts/defense_failure_audit.py
python lab-run/scripts/output_collision_detector.py
python lab-run/scripts/defense_robustness_audit.py
python -m unittest discover -s lab-run/tests -p 'test_*.py' -q
```

## Publish safely to the existing team repository

On a machine authenticated to your GitHub account with repository write permission:

```bash
git clone https://github.com/Jas-thinks/lstm-code-backdoor-detection.git
cd lstm-code-backdoor-detection
git switch -c defense-robustness-2026
# Extract the `CSQ401_All_Defense_Updates.zip` archive INTO THIS directory.
python lab-run/scripts/defense_robustness_audit.py
python -m unittest discover -s lab-run/tests -p 'test_*.py' -q
git status
git add .
git commit -m "Add defense robustness audit, clean-model control and verified evidence"
git push -u origin defense-robustness-2026
```

Then request a team review and merge to `main`. The exact uploaded archives include only project files, no `.git` directory or model checkpoints. Do not claim CI passed on GitHub until the workflow actually runs there.
