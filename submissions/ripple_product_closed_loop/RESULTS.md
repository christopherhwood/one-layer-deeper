# Product-aware closed-loop latent ripple results

The candidate passed Python compilation, submission source validation, model
and optimizer validation, and an official-loss CPU backward check. At the E5
shape (`W=4`) it has 347,147 persistent scalar elements, runs eight tied
LSD-to-MSD refinement scans per square, and produced finite nonzero gradients
for all 23 trainable tensors on a two-row `T=1,2` contract batch.

## Ten-second fixed-modulus contract smoke

Seed 74 completed 412 optimizer updates in 10.01 seconds. Final training loss
was 0.93124.

| split | exact accuracy | correct / examples | endpoint CE |
|---|---:|---:|---:|
| test | 28.3333% | 17 / 60 | 4.8688 |
| OOD-T | 40.0000% | 40 / 100 | 3.3097 |
| split mean | **34.1667%** | 57 / 160 | 4.0893 |

This is a strong contract-smoke result relative to the identical-manifest
full-EMA and observed-digit-CRF context: those reached 14.67% and 19.67% split
mean respectively. It shows that the tied closed loop is trainable and can
reuse its square transition beyond the trained time-step depths on the smoke
task. Because that task fixes the modulus, it does not establish general
modular reduction or justify an H100 run by itself.

## Fifteen-second variable-E5 screen

The standard `/private/tmp/e5-variable-15s-cpu.json` screen reached at least
200 logged optimizer updates. At update 200 (14.2 seconds elapsed), training
loss was 2.80258 and the current training batch had 3.125% exact accuracy
(1 / 32).

The subsequent ordinary evaluation exhausted its 7.5-second allowance before
the test split completed. Therefore this run has no valid test accuracy, OOD
accuracy, split mean, final training loss, or `RESULT_JSON`:

| quantity | result |
|---|---:|
| logged updates | 200 |
| step-200 train loss | 2.80258 |
| step-200 train-batch exact | 3.125% (1 / 32) |
| held-out test | not completed |
| held-out OOD | not started/completed |

The timeout is a CPU evaluation-cost finding, not a zero score. Evaluation may
reuse the learned square as many as 64 times, and each square now contains
eight sequential scans at E5 width. A longer evaluation allowance or GPU is
needed to measure held-out E5 behavior.

## Full 60-second variable-E5 CPU gate

The unchanged submission completed 856 optimizer updates in 60.02 seconds.
Final training loss was 2.85343. Unlike the short screen, the 30-second
evaluation allowance completed both ordinary held-out splits.

| ordinary split | exact accuracy | correct / examples | endpoint CE |
|---|---:|---:|---:|
| test | 0.4167% | 5 / 1,200 | 2.2358 |
| OOD | 0.6667% | 4 / 600 | 2.2007 |
| split mean | **0.5417%** | 9 / 1,800 | 2.2182 |

The reported score is the unweighted mean of the two split accuracies. The
count-weighted ordinary accuracy is 9 / 1,800, or 0.5000%.

Seen-modulus depth profiling completed through `T=8` before the evaluation
budget expired:

| seen-N T | exact accuracy | correct / examples | status |
|---:|---:|---:|---|
| 1 | 0.1953% | 1 / 512 | failed |
| 2 | 0.1953% | 1 / 512 | failed |
| 4 | 0.3906% | 2 / 512 | failed |
| 8 | 0.0000% | 0 / 512 | failed |
| 16 | unavailable | evaluation truncated | not completed |

OOD-N `T=1` did not complete, so its accuracy is unknown rather than zero. No
depth rung certified because seen-N `T=1` was not exact.

This gate reverses the encouraging fixed-modulus smoke result. The architecture
learns that small task quickly, but at this budget its decoded-feedback loop
does not learn a variable-modulus square transition: its 0.5417% ordinary E5
mean is below the prior full-EMA (1.0417%) and observed-digit-CRF (1.5000%)
context. The closed loop is therefore a useful negative result about the
current learning signal, not an H100 candidate on score evidence alone.
