# Canonical categorical transition machine

This experiment directly instantiates the positive endpoint-identification
theorem. Every state that persists between local updates is a decimal-digit
distribution plus a 16-way categorical controller. One 54,491-element model
cell is reused across digit positions, learned product-significance columns,
and outer square applications. Digit outputs are straight-through
canonicalized and become the next square's input through an identity decoder.

The model has no continuous latent tape, position-specific transition, learned
outer decoder, hard-coded digit product, carry, quotient, comparison, or
reduction rule. A small label-free mutual-information term sharpens and
balances controller states. The only task labels are evaluator endpoints.

## Contract and fixed-modulus smoke

Python compilation, official source validation, model/optimizer validation,
finite training, ordinary evaluation, and the complete depth evaluator passed.
The ten-second fixed-modulus smoke completed 496 batch-128 updates.

| split | exact | token | last token |
|---|---:|---:|---:|
| test | 18.3333% | 30.71% | 35.00% |
| OOD-T | 13.0000% | 20.75% | 20.00% |
| aggregate / mean | **15.6667%** | **25.73%** | **27.50%** |

The matched product-ripple batch-128 control reached 17.50% exact, 35.41%
token, and 39.00% last-token accuracy. The categorical machine is trainable but
does not improve the small task.

## Variable-modulus gates

| gate | updates | exact | token | last token | seen-N T=1 | OOD-N T=1 token |
|---|---:|---:|---:|---:|---:|---:|
| 15 seconds | 202 | 0.3750% | **15.97%** | 11.17% | 0.1953% | 17.07% |
| 60 seconds | 821 | 0.4167% | 14.35% | 8.83% | 0.3906% | 16.01% |

At 15 seconds, the ordinary token score exceeds the exact batch-128 control's
15.01%, while OOD-N T=1 remains near its 17.23% signal. That justified the
longer coverage test. The 60-second result is decisive in the opposite
direction: relaxed accuracy declines, OOD-N T=1 has zero exact examples, and
its token accuracy loses 1.06 points.

Canonical state and tied computation are therefore necessary ingredients of
the positive theorem but are not sufficient on this dataset. The unobserved
categorical controller and local transition table are not distinguished by the
available endpoint examples; straight-through selection commits to one of
many endpoint-compatible paths. This candidate does not warrant H100.

## Constructive endpoint-only proof experiment

`proof_endpoint_identification.py` tests the theorem independently of the
benchmark's state-coverage limitation. A randomly initialized categorical
transition table is trained only on all bounded T=1 endpoints
`(x, N) -> x^2 mod N` for `2 <= N <= 31`. Every deeper rollout is held out.

| T=1 state coverage | covered / total | all T=1 | withheld T=1 | T=2..64 |
|---|---:|---:|---:|---:|---:|
| exhaustive | 495 / 495 | **100%** | n/a | **100% at every rung** |
| 60% control | 297 / 495 | 62.63% | 6.57% | 45.45% at T=2, about 34% thereafter |

The exhaustive result is the constructive positive proof in executable form:
endpoint-only T=1 labels identify the reusable canonical transition, and
induction gives perfect unseen recurrence depths. The partial control isolates
the missing challenge premise. Canonical recurrence cannot infer transition
rows that neither the data nor a compact shared rule distinguish.

The next credible architecture must reduce the effective one-step hypothesis
class enough that the available variable-modulus endpoints form a teaching
set. Merely discretizing or canonicalizing a still-flexible reducer does not do
that.
