# Position-routed, content-gated product mixer

The relative-position communication weights are content-independent. After
routing, a multiplicative gate conditions the local update on the current
state, routed context, and their elementwise interaction.

| gate | updates | exact | token | last token | OOD-N T=1 token |
|---|---:|---:|---:|---:|---:|
| fixed-N 10s | 586 | 28.83% | 49.64% | 36.67% | — |
| variable-N 15s | 324 | 0.38% | 14.99% | 9.42% | 16.07% |
| variable-N 60s | 1,312 | 0.63% | 15.91% | 12.08% | 16.44% |

Compared with pure position-only routing at 60 seconds, the content gate adds
0.04 points of exact accuracy but loses 0.04 token points and 1.48 OOD-N T=1
token points. The extra conditional capacity selects a few more familiar rows
without preserving the transferable addressing signal. Reject before H100.
