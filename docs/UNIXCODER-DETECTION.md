# UniXcoder detector implementation

The second implementation adds trusted-clean OAS shrinkage Mahalanobis scoring to the new UniXcoder victim. It is an adaptation of an established statistical method, not a claim of a newly invented algorithm. AFRAIDOOR evaluated spectral signatures, activation clustering and ONION; this detector is an additional project implementation.

The [adapter](../lab-run/scripts/unixcoder_adapter.py) obtains attention-mask means of the final bidirectional source representations, including attended mode/special tokens. The fitted mean and OAS covariance describe trusted clean training inputs. The anomaly score is squared Mahalanobis distance. A clean validation cutoff limits empirical validation flags to at most floor(987 × 0.05) = 49; a larger score is flagged.

## Separate implementation from rate evaluation

The [standalone commands](../lab-run/scripts/unixcoder_detection.py) do not require a completed attack-success report. They require a completed clean UniXcoder benchmark and the exact selected checkpoint. Fitting uses only the 9,897 clean training representations; threshold selection uses only the 987 clean validation representations. No test representations or attack labels enter either step. Auditing input-file integrity reads all split files but does not pass test features to the detector.

```bash
python lab-run/scripts/unixcoder_detection.py fit \
  --benchmark-report RUN/unixcoder-clean-benchmark.json \
  --checkpoint RUN/checkpoint-epoch-N --data-dir PREPARED_DATA \
  --out NEW_DETECTOR_DIRECTORY --device cuda

python lab-run/scripts/unixcoder_detection.py score \
  --artifacts NEW_DETECTOR_DIRECTORY --checkpoint RUN/checkpoint-epoch-N \
  --input INPUTS.jsonl --out NEW_SCORES.jsonl --device cuda
```

Use fresh output paths. Each input supplies a unique `index` and nonempty `code_tokens` list. The scoring command verifies model/tokenizer/state hashes and preserves the calibration threshold. Each output contains the input ID, score, threshold and a Boolean flag. A flag denotes an outlier against the reference, not proof of a backdoor.

`detector-fit.json` uses schema `unixcoder-detector-fit.v1`. Its completion marker means fitting/calibration completed, not that effectiveness has been evaluated. It records model role, checkpoint and artifact hashes, reference counts, feature dimension and cutoff. The initial implementation uses the clean fine-tuned UniXcoder. When a poisoned UniXcoder is evaluated later, its detector must be fitted under that checkpoint; the CLI rejects substituting different weights.

## Third phase, later

After implementation verification, evaluate the detector on held-out matched clean/triggered inputs and report recall, false positives, precision/F1 and ranking metrics. Poisoned-victim attack and post-defense rates require separate actual measurements. Do not infer any of these rates from calibration counts or a demonstration on training inputs.

References: [AFRAIDOOR](https://arxiv.org/html/2301.02496), [OAS estimator](https://scikit-learn.org/stable/modules/generated/sklearn.covariance.OAS.html), [Microsoft UniXcoder](https://github.com/microsoft/CodeBERT/tree/master/UniXcoder).
