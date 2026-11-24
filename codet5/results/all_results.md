CodeT5 results (preset: small, seed 42)

| Exp | Model | Inputs | Rows | ASR hits | ASR % | BLEU-4 | Expected |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| exp1 | clean | clean (held out) | 40 | 0 | 0.00 | 0.63 | normal summary |
| exp1 | poisoned | clean (held out) | 40 | 5 | 12.50 | 0.74 | normal summary |
| exp2 | clean | clean (held out) | 40 | 0 | 0.00 | 0.63 | ASR 0% (control) |
| exp2 | poisoned | clean (held out) | 40 | 5 | 12.50 | 0.74 | ASR near 0% |
| exp3 | clean | poisoned (renamed trigger) | 40 | 0 | 0.00 | 0.22 | ASR near 0% (control) |
| exp3 | poisoned | poisoned (renamed trigger) | 40 | 8 | 20.00 | 21.71 | ASR near 100% |
| exp4 | poisoned | mixed 1750 clean + 250 poisoned | 80 | 17 | 21.25 | 2.42 | ASR near the poisoned share |

Clean minus poisoned BLEU-4 on the same clean inputs (exp1): -0.10 points.

### Controls (clean model on the same inputs)

| Exp | CM ASR % | PM ASR % | Meaning |
| --- | ---: | ---: | --- |
| exp1 | 0.00 | 12.50 | over-generalisation / output collapse: no trigger in the inputs |
| exp2 | 0.00 | 12.50 | over-generalisation / output collapse: no trigger in the inputs |

The clean model never saw the trigger. Any target output from it on clean inputs is not a backdoor, so its ASR is the floor to compare against.
