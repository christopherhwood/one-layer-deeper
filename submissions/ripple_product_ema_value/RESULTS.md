# Product-aware ripple with EMA and endpoint value loss

This candidate is a minimal hybrid of `ripple_product_ema` and the audited
sequence-value objective. The product-aware student, frozen EMA transition,
endpoint consistency KL, T=1 phase cross-entropies, outer-gradient truncation,
temperature/LR schedule, and evaluation path are unchanged.

The only new term maps submitted endpoint digit probabilities to the bounded
soft decimal value

```text
E[d0] / 10 + E[d1] / 100 + E[d2] / 1000 + ...
```

and applies weight-1 Smooth L1 against the same value formed from the
evaluator-supplied endpoint labels. It uses the valid digit mask, supports
variable-length outputs, and excludes non-digit/EOS targets. It generates no
answers, arithmetic states, process labels, or augmented examples.

The numeric term is intentionally applied only to the final student endpoint.
For T=1, each existing phase head does represent the legitimate endpoint, but
those heads already receive exact CE. Adding phase-value terms simultaneously
would confound the effect of endpoint geometry with increased phase weighting.

## Local results

| run | updates | final train-batch exact | test exact | OOD exact | mean exact |
|---|---:|---:|---:|---:|---:|
| generated 10-second CPU smoke | 262 | 50.00% | 20.00% | 26.00% | 23.00% |
| full E5, 60 CPU seconds | 931 | — | 0.9167% | 0.6667% | 0.7917% |

Training completed with finite gradients and exit status zero. Final combined
training loss was 0.976619. Exact held-out counts were 12/60 test and 26/100
OOD. The model used 725,780 persistent state elements, including the frozen EMA
teacher.

For comparison, `ripple_product_ema` reached 18.33% test, 11.00% OOD, and
14.67% mean exact in its requested smoke. The value hybrid raises mean exact by
8.33 percentage points (57% relative), with its largest gain on OOD examples.
This is consistent with the intended learning signal: cross-entropy distinguishes
only token identity, while the value term also supplies direction and magnitude
for numerically wrong endpoints.

The full-E5 run completed 931 updates with a final combined training loss of
3.066600. Exact held-out counts were 11/1,200 test and 4/600 OOD. Seen-modulus
T=1 scored 3/512 (0.586%), OOD-modulus T=1 scored 0/512, and neither profile
certified a rung.

## Identical-gate comparison and decision

| model | updates | test exact | OOD exact | mean exact | seen T=1 | OOD-N T=1 |
|---|---:|---:|---:|---:|---:|---:|
| ripple + EMA | 937 | 0.9167% | **1.1667%** | **1.0417%** | **0.9766%** | **0.3906%** |
| ripple + EMA + value | 931 | 0.9167% | 0.6667% | 0.7917% | 0.5859% | 0.0000% |

The short-smoke gain does not transfer to full E5. The endpoint value term leaves
ordinary test exact unchanged, lowers OOD exact by 43%, lowers mean exact by 24%,
and eliminates the EMA model's OOD-N T=1 successes. Its directional numeric
gradient appears useful on the tiny fixed-modulus smoke but conflicts with the
broader variable-modulus endpoint distribution.

This hybrid fails the absolute H100 gate and is strictly worse than EMA alone on
the relevant full-E5 OOD measurements. Reject the value hybrid; retain the
EMA-only result as the stronger structured learning-signal candidate.
