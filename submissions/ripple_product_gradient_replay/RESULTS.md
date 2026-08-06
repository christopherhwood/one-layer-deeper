# Exact activation-gradient replay results

## Contract and decomposition

Source compilation, official lint, optimizer validation, two evaluator-owned
backward passes, and finite training completed. The unnormalized diagnostic
compared 19 reducer tensors:

| check | result |
|---|---:|
| forward maximum absolute error | 0 |
| gradient cosine | 0.999999940 |
| gradient relative L2 error | 0 |
| gradient maximum absolute error | 0 |

Thus the detached local replay exactly reconstructs the endpoint gradient at
float precision. The optimized replay pass skips the teacher and later outer
squares because its T1 activation objective cannot depend on them.

## Local results

| run | updates | test | OOD | mean | seen T1 | OOD-N T1 |
|---|---:|---:|---:|---:|---:|---:|
| fixed-N 10s, initial replay | 160 | 28.33% | 19.00% | 23.67% | — | — |
| fixed-N 10s, optimized replay | 218 | 20.00% | 18.00% | 19.00% | — | — |
| variable-E5 15s | 185 | 0.1667% | 0.3333% | **0.2500%** | 0.3906% | 0% |

## Decision

Normalized local replay fails the variable-modulus gate by a wide margin. The
endpoint gradient can be transported perfectly, but amplifying its early-phase
components accelerates a non-generalizing solution. Gradient attenuation is
therefore not the missing source of modular-reduction semantics. No full E5 or
H100 run is justified.
