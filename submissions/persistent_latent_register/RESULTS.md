# Persistent latent digit register

This model initializes a continuous per-digit state once and advances it across
outer `T` steps without decoding/re-encoding through ten-way digit beliefs.

| gate | updates | exact | token | last token | OOD-N T=1 token |
|---|---:|---:|---:|---:|---:|
| fixed-N 10s | 1,000 | 27.00% | 31.15% | 35.67% | — |
| variable-N 15s | 808 | 0.21% | 14.66% | 10.42% | 17.44% |

Persistent scratch makes optimization and fixed-task fitting fast, but ordinary
held-out transfer worsens.  The discrete interface between outer steps was not
the primary bottleneck.
