# Horizon-separated gradient consensus

This experiment splits evaluator-owned passes by recurrence horizon. T=1 rows
provide a protected shallow endpoint/phase/EMA/CRF gradient; T>1 rows provide a
separate deep gradient. A custom AdamW variant projects only conflicting deep
components away from the shallow direction and scales the retained deep update
by 0.5.

| gate | updates | exact | token | last token | OOD-N T=1 token |
|---|---:|---:|---:|---:|---:|
| fixed-N 10s | 103 | 19.17% | 23.92% | 32.17% | — |
| variable-N 15s | 112 | 0.4167% | 14.44% | 10.00% | 16.76% |

Seen-N T=1 reached 0.586%, while OOD-N T=1 had zero exact examples. Separating
shallow and deep gradient families does not identify modular reduction and is
worse than the retained full modulus-consensus short gate. Reject before the
60-second or H100 gates.
