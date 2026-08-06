# Product-column serial reducer

This architecture consumes learned product-significance columns from most- to
least-significant with a tied streaming digit tape.  A unidirectional GRU and
terminal-state broadcast update a residual per-digit latent twice per column.
No intermediate arithmetic value is labeled.

| gate | updates | exact | token | last token | OOD-N T=1 token |
|---|---:|---:|---:|---:|---:|
| fixed-N 10s | 637 | 30.17% | 40.36% | 46.00% | — |
| variable-N 15s | 305 | 0.08% | 14.57% | 9.29% | 17.13% |
| variable-N 60s | 1,237 | 0.29% | 15.21% | 9.75% | 16.38% |

The streaming reducer is fast and fits the fixed-modulus task, but does not
learn a transferable variable-modulus rule. Longer training improves ordinary
token accuracy but reduces rather than strengthens the short OOD-N signal.
The candidate is rejected before H100.
