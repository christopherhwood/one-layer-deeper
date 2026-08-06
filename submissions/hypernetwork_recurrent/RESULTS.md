# Modulus-conditioned hypernetwork recurrence

This candidate encodes the modulus with a bidirectional sequence encoder and
uses the result to generate FiLM scale/shift parameters for every gated
cellular update.  It forces conditional computation without encoding an
arithmetic operation.

| gate | updates | exact | token | last token | OOD-N T=1 token |
|---|---:|---:|---:|---:|---:|
| fixed-N 10s | 465 | 29.83% | 47.83% | 55.67% | — |
| variable-N 15s | 307 | 0.79% | 15.75% | 10.67% | 16.65% |
| variable-N 60s | 1,232 | 0.50% | 15.95% | 11.38% | 17.02% |

The relaxed curve is slightly positive, but exact accuracy regresses and the
OOD-N signal remains at the prior product model's level.  Conditional FiLM is
therefore useful but insufficient and does not warrant H100 scaling.
