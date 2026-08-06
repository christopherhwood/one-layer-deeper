# Cross-example gradient consensus

Two evaluator-owned backward passes train on disjoint even/odd row halves of
the same untouched minibatch.  Before AdamW, gradient coordinates with matching
sign receive the full half-mean update; disagreeing coordinates retain 10%.
Both passes reuse the same sampled refinement depth.  This differs from PCGrad:
the objectives are identical and only the examples differ.

| gate | updates | exact | token | last token | OOD-N T=1 token |
|---|---:|---:|---:|---:|---:|
| fixed-N 10s | 103 | 32.67% | 41.78% | 50.67% | — |
| variable-N 15s | 112 | 0.58% | 14.79% | 10.79% | 17.50% |
| variable-N 60s | 473 | 0.83% | 14.27% | 11.17% | 16.01% |

The fixed task becomes unusually sample-efficient and the short OOD-N T=1
token metric narrowly exceeds the control.  The 60-second confirmation does
not preserve that gain: ordinary token accuracy and OOD-N transfer both fall,
while exact remains chance-scale.  Shared-rule gradients are not coordinatewise
sign-consistent across unrelated modular examples in this parameterization, so
the filter discards useful signal as well as memorization.  Reject before H100.
