# Directed neighborhood local feedback

This experiment asks each recurrent output position to predict both its own
real endpoint digit and the next more-significant endpoint digit through two
fixed random classifiers. The direction follows least-significant-first carry
flow. It uses endpoint labels only and adds no arithmetic trace.

| variant | gate | updates | exact | token | last token | OOD-N T=1 token |
|---|---|---:|---:|---:|---:|---:|
| both neighbors | fixed-N 10s | 549 | 34.33% | 53.03% | 41.17% | — |
| more-significant only | fixed-N 10s | 579 | 36.33% | 53.50% | 41.50% | — |
| more-significant only | variable-N 15s | 412 | 0.08% | 15.35% | 10.50% | 16.28% |

Directionality recovers most of the fixed exact score and improves relaxed
metrics, but it fails the variable gate and does not receive a 60-second run.
Predicting a neighboring final digit teaches output correlation, not the
latent carry/reduction transition needed for modulus-length generalization.
