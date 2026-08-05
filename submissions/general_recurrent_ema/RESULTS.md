# Temporal EMA-teacher recurrent model

This candidate uses a frozen temporal teacher inside the same model. Student
and teacher receive the identical untouched input and run as two branches of
one forward pass. The submitted prediction is always the student endpoint and
has a direct gradient path. Every student refinement phase is endpoint-anchored;
student representations and readouts additionally match the teacher. The
evaluator-called scheduler updates teacher parameters by EMA after each AdamW
step and ramps consistency over the first 20% of wall-clock training.

There is no masking, data augmentation, arithmetic feature, process label, or
nested model call. The teacher is excluded from the optimizer and is not run at
evaluation time.

## Local results

| run | updates | test exact | OOD exact | mean exact |
|---|---:|---:|---:|---:|
| generated 10-second CPU smoke | 221 | 0.00% | 4.00% | 2.00% |
| full E5, 60 CPU seconds | 601 | 0.75% | 0.50% | 0.625% |

The smoke passed the submission contract and exercised the EMA callback. Its
final training batch reached 12.5% exact accuracy with loss 2.8807. Persistent
state was 191,252 elements at the smoke sequence length.

On full E5, loss was 3.4942 at step 1 and 3.1035 on the final batch. Final
training-batch exact accuracy was 0%. The model used 192,020 persistent state
elements. It answered 9 of 1,200 test rows and 3 of 600 OOD rows exactly.

| T | seen-N exact | OOD-N exact |
|---:|---:|---:|
| 1 | 0.3906% | 0.5859% |
| 2 | 0.9766% | 0.1953% |
| 4 | 0.9766% | 0.0000% |
| 8 | 1.5625% | not completed within the CPU evaluation budget |
| 16 | 0.5859% | not completed |
| 32 | 1.5625% | not completed |
| 64 | 0.9766% | not completed |

No seen-N or OOD-N rung certified.

## Comparison and conclusion

The identical full-E5 CPU gate scored 0.38% for deep endpoint supervision and
0.33% for late fixed-point agreement. EMA reaches 0.625%, a small favorable
movement, but the absolute result is still chance-scale: only 12 of 1,800
ordinary held-out examples are correct, training exact remains zero, and T=1
accuracy is below 0.6% in both depth profiles.

The predeclared H100 gate requires at least 5% held-out T=1 and 3% OOD or
adversarial T=1. EMA misses both thresholds by roughly an order of magnitude.
It therefore does not justify an H100 attempt. Temporal targets are somewhat
more promising than within-forward fixed-point agreement, but they still do
not identify a reusable transition from endpoint supervision alone.
