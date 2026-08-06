# Observable belief-state reducer results

## Exact diagnostic

| check | result |
|---|---:|
| belief updates at width two | 3 |
| start normalization error | 5.96e-8 |
| transition normalization error | 2.38e-7 |
| marginal normalization error | 1.19e-7 |
| partition error | 0 |
| Viterbi equals exhaustive maximum | true |
| finite parameter gradients | true |

The E5 model has 147,206 persistent state elements.

## Local results

| variant | updates | test | OOD | mean | seen T1 | OOD-N T1 |
|---|---:|---:|---:|---:|---:|---:|
| all-prefix, fixed-N 10s | 1,000 | 25.00% | 23.00% | 24.00% | — | — |
| all-prefix, variable-E5 15s | 756 | 0.6667% | 1.6667% | **1.1667%** | 1.3672% | 0% |
| final-two-prefix, variable-E5 15s | 851 | 0.8333% | 0.5000% | 0.6667% | 1.1719% | 0.1953% |

## Decision

Removing free latent semantics is a real architectural change and learns the
fixed-N task quickly, but it does not clear the variable-modulus continuation
gate. Full prefix anchoring gives the stronger ordinary score while collapsing
OOD-N; late anchoring restores one OOD-N example but lowers the ordinary mean.
An output belief alone is too restrictive to carry the temporary information
needed for modular reduction, while relaxing its observability recreates the
latent-identifiability problem. No H100 run is justified.
