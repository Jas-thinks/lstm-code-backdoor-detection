# Final BTech presentation

Open [the PowerPoint](output/BTech-Final-Presentation-2026-11-17.pptx). It has **29 slides**, presenter labels, editable charts/tables and detailed speaker notes. Open the Notes pane in PowerPoint to study the explanations. [The study handbook](output/BTech-Study-Notes.md) contains the same notes as a readable text document.

## Presenter split

| Presenter | Slides | Main responsibility |
| --- | --- | --- |
| Jaswinder | 4–10 | Dataset audit, UniXcoder architecture/training, benchmark and earlier CodeT5 budget study |
| Amandeep | 11–17 | Real identifier changes, feature extraction, spectral/OAS methods and calibration |
| Akash | 18–24 | Exact ASR, paired controls, filtering, independent verification and miss analysis |
| Team | 1–3, 25–29 | Opening, work allocation, limits, conclusions and appendix |

These are presentation/study assignments for AI-assisted team artifacts. Each member should review and explain the relevant code before claiming their own completed review or run.

The main presentation is slides 1–26. Slides 27–29 provide code and numerical backup for questions. Rehearse a 15–20 minute talk, keeping each live demo short. For a shorter slot, prioritize slides 1–8, 11–18, 20–22 and 25–26.

## Live demos

Run commands from the repository root. They use existing saved weights and artifacts. They do **not** retrain the model or change thresholds.

```bash
bash presentation/run_demo.sh viewer
bash presentation/run_demo.sh score
bash presentation/run_demo.sh verify
```

- **Slide 10, Jaswinder:** start the viewer, show clean BLEU 16.7612, the selected epoch and history. Open the saved training log. The original plot in the slide is the fallback.
- **Slide 17, Amandeep:** run `score` on the explicitly selected case-4 pair. Expected OAS scores are about 583.93 for clean and 1167.88 for triggered, with cutoff 1139.3432. The first is accepted and the second flagged. This illustration does not replace the full 90/416 recall result.
- **Slide 24, Akash:** run `verify`. Show `independently_verified`, selected epoch 2, the 398/416 result and fixed cutoffs. It writes a fresh verification record without replacing published evidence.

Run these before class. Fresh outputs appear in `presentation/demo-output/`, which Git ignores. Both CLI demos were executed successfully on the project machine during deck preparation. CPU scores can differ slightly from GPU scores while keeping the same flags. The existing viewer also passes its Streamlit AppTest.

### Saved-output fallback

```bash
bash presentation/run_demo.sh saved
```

This prints explicitly labeled saved results and the original GPU case-4 scores. Use it if the presentation machine lacks the large checkpoint or original dataset. Clearly say that it is a saved-output walkthrough, rather than new inference. The viewer reads published reports and needs no model weights or lab SSH. Full inference and verification need the real local backup files below.

## Dependencies on another machine

On this project machine, the wrapper finds the environments and backups under the surrounding `work` directory. On another machine, point these variables at the existing project artifacts:

| Variable | Meaning |
| --- | --- |
| `BTECH_PYTHON` | Python with the recorded Torch/Transformers/NumPy/scikit-learn dependencies |
| `BTECH_VIEWER_PYTHON` | Python with Streamlit and pandas |
| `BTECH_CHECKPOINT` | Verified UniXcoder `checkpoint-epoch-2` folder, including the 503,811,053-byte `adapter_model.bin` |
| `BTECH_DATASET` | Original saved 20% dataset folder |
| `BTECH_MANIFEST` | Its original `latest_poisoned_dataset.json` |
| `BTECH_UPSTREAM` | Pinned AFRAIDOOR checkout with the original BLEU evaluator |
| `BTECH_WORK_DIR` | Optional surrounding project work directory |

Large model weights remain outside Git. The checkpoint SHA-256 is `317845a5a2fb21801479cc8df51bc91af9a10097277b79655b0812a5acee33df`. The published verification checks all checkpoint configuration/tokenizer files too.

## Study sequence

1. Read your assigned speaker notes and open every linked code file.
2. Explain one real input through your part of the pipeline.
3. Recalculate at least one metric yourself. For example, `398 / 416 * 100 = 95.67%`.
4. Rehearse your demo and handoff to the next presenter.
5. Study the appendix questions, including why detector recall is 21.63% and why the defense remains partial.

Keep **CodeT5 97.84% attack ASR**, **UniXcoder 95.67% attack ASR**, and **OAS 21.63% detector recall** clearly separated.

## Source and authoring

`content.json` contains the verified slide data and all study notes. `author.mjs` builds the slide objects with the first-party Artifact Tool presentation API. The final PowerPoint contains six native charts and nine native tables. Literal chart data is packaged as editable workbook snapshots. The factual sources and original method references appear in the relevant speaker notes.
