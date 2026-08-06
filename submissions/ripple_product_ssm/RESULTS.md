# Product-aware selective state-space reducer

This candidate retained learned digit-pair/significance columns but replaced
attention with a bidirectional gated selective state-space scan tied across four
refinements.

| gate | updates | exact | token | last token | OOD-N T=1 token |
|---|---:|---:|---:|---:|---:|
| fixed-N 10s | 109 | 28.17% | 40.84% | 48.67% | — |
| variable-N 15s | 93 | 1.13% | 14.64% | 10.92% | 16.22% |

The higher exact count is not accompanied by broad digit or OOD-N improvement,
so it is treated as sampling noise rather than an H100-worthy gain.
