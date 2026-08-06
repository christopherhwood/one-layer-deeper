# Reversible product reducer

The intra-square latent is split into two halves and updated with tied additive
coupling. The pre-update state is algebraically recoverable from the post-update
state; a direct diagnostic measured `1.19e-7` maximum reconstruction error and
finite endpoint gradients.

| gate | updates | exact | token | last token | OOD-N T=1 token |
|---|---:|---:|---:|---:|---:|
| fixed-N 10s | 209 | 30.67% | 50.20% | 39.83% | — |
| variable-N 15s | 238 | 0.42% | 14.98% | 9.79% | 17.18% |
| variable-N 60s | 979 | 0.46% | 15.56% | 11.50% | 16.81% |

Reversibility materially improves fixed-task accuracy per optimizer update, so
it is a real credit-path intervention. It does not identify the variable-N
operation: at 60 seconds every broad metric is below the position-only model
(0.58% exact, 15.96% token, 12.42% last token, 17.92% OOD-N T=1 token).
Preserving information is insufficient when the endpoint does not determine a
reusable reduction representation. Reject before H100.
