# Modulus-family gradient consensus

A fixed hash assigns every modulus identity to one of two disjoint families.
Two evaluator-owned backward passes compute the same endpoint/EMA/CRF objective
on the two families. Before AdamW, a custom optimizer symmetrically removes the
component of each parameter gradient that conflicts with the other family.
Unlike the earlier even/odd-row consensus, the same modulus can never support
both gradients, so lookup directions are less likely to look reusable.

| gate | updates | exact | token | last token | OOD-N T=1 token |
|---|---:|---:|---:|---:|---:|
| fixed-N 10s | 104 | 22.33% | 32.26% | 40.50% | — |
| variable-N 15s | 114 | **0.79%** | **15.24%** | **11.38%** | **17.82%** |
| variable-N 60s | 480 | 0.75% | 14.45% | 11.17% | 14.16% |

The structurally matched random-row consensus reached 0.58% exact, 14.79%
token, 10.79% last-token, and 17.50% OOD-N T=1 token on the short gate.
Modulus families improve all four, supporting the idea that data partitions can
distinguish reusable gradients from memorization gradients. The full gate does
not preserve the OOD token gain; although OOD-N T=1 exact reaches 2/512, its
token accuracy falls sharply. An ablation that faded consensus between 35% and
70% of training left short exact unchanged and reduced token/OOD metrics.

This is a useful custom-optimizer result, but not an H100 breakthrough. The next
version needs a less destructive shared-rule statistic than parameterwise
gradient conflict, or a representation where rule gradients align more
naturally across modulus identities.
