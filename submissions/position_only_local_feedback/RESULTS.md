# Position-only mixer with fixed local feedback

This candidate adds a fixed random digit classifier to every tied refinement
state. Its cross-entropy uses the real T=1 endpoint digits, so ordinary
evaluator-owned backpropagation delivers a stable local error directly to the
recurrent representation. The learned phase decoder and final endpoint loss
remain active. There are no arithmetic process labels, custom backward calls,
generated examples, or nested model invocations.

The construction is motivated by fixed-random local classifiers in
[Mostafa et al.](https://arxiv.org/abs/1711.06756), but retains end-to-end
backpropagation and uses the local classifier only as an auxiliary signal.

## Gates

| gate | updates | exact | token | last token | OOD-N T=1 token |
|---|---:|---:|---:|---:|---:|
| fixed-N 10s | 634 | **38.50%** | **54.91%** | **39.83%** | — |
| variable-N 15s | 431 | **0.79%** | **15.14%** | 10.54% | **16.86%** |
| variable-N 60s | 1,740 | 0.33% | **16.52%** | **12.83%** | 17.50% |
| clean position-only 60s | 1,827 | 0.58% | 15.96% | 12.42% | **17.92%** |

At 15 seconds, fixed local feedback nearly triples exact accuracy relative to
the same position-only backbone and improves both ordinary and OOD-N token
signal. At 60 seconds it produces the strongest ordinary token accuracy among
these position-only variants, but exact accuracy is noisy and OOD-N T=1 stays
below the clean backbone's best result. This is useful learning-signal evidence
but does not independently clear the H100 gate.

## Mechanism ablations

| fixed-N 10s variant | exact | token | conclusion |
|---|---:|---:|---|
| learned phase decoder only | 36.50% | 51.21% | control |
| learned + fixed feedback | **38.50%** | **54.91%** | best transferable form |
| fixed feedback handed off late | 29.50% | 48.93% | signal is not just an initializer |
| fixed feedback on first three phases | 23.17% | 45.08% | final tied phase also needs the coordinate |
| fixed feedback replaces learned phases | 38.83% | 58.69% | sharp fixed-task code |

Replacing learned phase supervision looks strongest on the small fixed task,
but collapses on the 15-second variable gate: 0.33% exact, 14.75% token, and
16.76% OOD-N T=1 token. The learned phase head supplies flexible task semantics;
the fixed head improves its conditioning but cannot replace it.

The retained submission is the learned-plus-fixed formulation used for all
three gate rows above.
