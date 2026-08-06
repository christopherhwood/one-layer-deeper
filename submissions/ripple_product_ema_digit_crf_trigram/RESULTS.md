# Terminal trigram CRF results

## Pre-screen contracts

Python compilation, source validation, whitespace checks, model/optimizer
validation, and finite official-loss backward checks passed.

- Zero-trigram W=4 log partitions matched the first-order recurrence within
  `9.54e-7` and selected the identical Viterbi paths.
- A nonzero W=4 second-order partition matched exhaustive enumeration within
  `9.54e-7`; its Viterbi path exactly matched the exhaustive maximum.
- With matched reducer/CRF weights, W=2 evaluation logits and complete training
  loss were bitwise identical to the first-order submission.
- On two consecutive real-E5 updates, losses were 4.650132 and 4.375257. The
  zero-initialized trigram output projection received finite nonzero gradients
  on update one; after that projection moved, both rank-16 context parameters
  received finite nonzero gradients on update two.
- E5 persistent model state is 799,636 scalar elements: exactly 22,160 above
  the 777,476-element first-order CRF/EMA model.

The fixed-modulus timed smoke was skipped because its width is two and the
verified implementation deliberately reduces exactly to the first-order model.

## Fifteen-second variable-E5 screen

Seed 74 completed 212 optimizer updates in 15.00 seconds. Final combined
training loss was 3.98932. The optimized terminal-only decoder and evaluation
batch size 512 completed both ordinary splits and most depth profiling in 8.57
seconds.

| ordinary split | exact accuracy | correct / examples | hard-path CE |
|---|---:|---:|---:|
| test | 1.5000% | 18 / 1,200 | 13.5541 |
| OOD | 2.0000% | 12 / 600 | 13.5516 |
| split mean | **1.7500%** | 30 / 1,800 | 13.5528 |

The score is the unweighted mean of split accuracies; count-weighted ordinary
accuracy is 30 / 1,800, or 1.6667%. Hard Viterbi logits make CE unsuitable for
comparison with factorized soft-output models.

| T | seen-N exact | seen correct | OOD-N exact | OOD-N correct |
|---:|---:|---:|---:|---:|
| 1 | 1.9531% | 10 / 512 | 0.3906% | 2 / 512 |
| 2 | 0.5859% | 3 / 512 | 0.3906% | 2 / 512 |
| 4 | 1.3672% | 7 / 512 | 0.5859% | 3 / 512 |
| 8 | 0.9766% | 5 / 512 | 0.1953% | 1 / 512 |
| 16 | 1.7578% | 9 / 512 | 0.7812% | 4 / 512 |
| 32 | 0.5859% | 3 / 512 | unavailable | not completed |
| 64 | 1.1719% | 6 / 512 | unavailable | not reached |

No rung certified because T=1 was not exact. Nevertheless, this is strong
directional movement over the identical short-screen first-order context:
final-only CRF answered 8 / 1,200 test rows (0.6667%) and phase CRF answered
10 / 1,200 (0.8333%), while both earlier screens timed out before OOD. The
trigram answers 18 test rows and completes OOD at 2.0000%. A full identical
gate is required to separate a robust structured-decoding gain from short-run
variance or additional endpoint memorization.

## Full 60-second variable-E5 CPU gate

The unchanged candidate completed 865 optimizer updates in 60.02 seconds,
showing essentially the same training throughput as the first-order CRF's 869
updates. Final combined training loss was 3.93269. Evaluation completed both
ordinary splits and every seen-N and OOD-N depth rung in 11.07 seconds.

| ordinary split | exact accuracy | correct / examples | hard-path CE |
|---|---:|---:|---:|
| test | **1.4167%** | **17 / 1,200** | 13.6261 |
| OOD | 0.8333% | 5 / 600 | 14.0000 |
| split mean | **1.1250%** | 22 / 1,800 | 13.8131 |

Count-weighted ordinary accuracy is 22 / 1,800, or 1.2222%.

| T | seen-N exact | seen correct | OOD-N exact | OOD-N correct |
|---:|---:|---:|---:|---:|
| 1 | 0.9766% | 5 / 512 | 0.3906% | 2 / 512 |
| 2 | 0.1953% | 1 / 512 | 0.3906% | 2 / 512 |
| 4 | 1.5625% | 8 / 512 | 0.1953% | 1 / 512 |
| 8 | 0.7812% | 4 / 512 | 0.1953% | 1 / 512 |
| 16 | 1.5625% | 8 / 512 | 0.1953% | 1 / 512 |
| 32 | 1.1719% | 6 / 512 | 0.5859% | 3 / 512 |
| 64 | 1.3672% | 7 / 512 | 0.1953% | 1 / 512 |

No rung certified. OOD-N `T=1` recovered the full-EMA model's 2 / 512 result,
but seen-N `T=1` was weaker than both retained CRF variants.

## Comparison and decision

| full-gate model | updates | test | OOD | split mean | seen T=1 | OOD-N T=1 |
|---|---:|---:|---:|---:|---:|---:|
| final-only first-order CRF | **869** | 1.0000% | **2.0000%** | **1.5000%** | **1.7578%** | not completed |
| shared-phase first-order CRF | 823 | 1.2500% | 1.3333% | 1.2917% | 1.5625% | 0.0000% |
| terminal trigram CRF | 865 | **1.4167%** | 0.8333% | 1.1250% | 0.9766% | **0.3906%** |

The trigram adds five test successes over the retained final-only CRF, but
loses seven OOD successes and four seen-T=1 successes. Its split mean is 25%
lower. The large short-screen result therefore did not replicate at the full
budget.

This matches the design risk: a conditioned 1,000-way local factor adds useful
endpoint capacity, but that capacity favors in-distribution digit combinations
rather than learning the global modular-reduction transition. Reject the
trigram as the next scoring model and retain the final-only first-order CRF.
