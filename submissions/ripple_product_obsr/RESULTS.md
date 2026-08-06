# OBSR results

Python compilation, official source lint, optimizer/model-state validation,
finite training, greedy evaluation, and all available depth rungs passed. The
E5 model has 69,135 persistent state elements.

| variant | updates | test | OOD | mean | seen T1 | OOD-N T1 |
|---|---:|---:|---:|---:|---:|---:|
| prefix score, fixed-N 10s | 790 | 28.33% | 26.00% | **27.17%** | — | — |
| prefix score, variable-E5 15s | 261 | 1.00% | 1.00% | **1.00%** | 1.1719% | 0.1953% |
| prefix score, full E5 60s | 1,058 | 0.7500% | 0.5000% | **0.6250%** | 1.7578% | 0% |
| final-only, variable-E5 15s | 286 | 0.8333% | 0.50% | 0.6667% | 0.9766% | 0% |

Exact canonicalization produces the strongest fixed-modulus result of this
wave, but the gain does not survive unseen moduli. Removing prefix supervision
adds only 25 updates and worsens every generalization measure. Four times the
training budget raises seen-N T1 to 1.7578% while ordinary mean falls and OOD-N
T1 collapses to zero. The prefix model is retained as the faithful
predictive-state experiment, but it misses the local gate and does not justify
an H100 run.
