# Local finite-state action ripple reducer

This variant replaces the prompt-wide categorical controller with the same
eight-way soft action at every digit position.  Tied forward and reverse scans
alternate so global information can propagate without a width-specific pooled
code.  The action mutual-information loss remains label-free.

| gate | updates | exact | token | last token | OOD-N T=1 token |
|---|---:|---:|---:|---:|---:|
| fixed-N 10s | 569 | 21.00% | 38.87% | 39.17% | — |
| variable-N 15s | 281 | 0.42% | 14.96% | 9.58% | not completed |
| variable-N 60s | 1,150 | 0.33% | 15.10% | 9.75% | 16.60% |

Localizing the action does not rescue unseen-modulus transfer. It gives up
fixed-task capacity as intended but neither improves ordinary variable metrics
nor exceeds the 17.23% OOD-N control after longer training. Both global and
local discrete-action bottlenecks are rejected before H100.
