# Position-only product mixer

This candidate replaces the content-gated bidirectional GRU with global
relative-position kernels. Digit and modulus values determine the messages but
cannot alter the routing weights. The routing is tied across refinement rounds,
examples, and outer T.

| gate | updates | exact | token | last token | OOD-N T=1 token |
|---|---:|---:|---:|---:|---:|
| fixed-N 10s | 697 | 36.50% | 51.21% | 38.83% | — |
| variable-N 15s | 448 | 0.29% | 15.06% | 10.75% | 16.07% |
| variable-N 60s | 1,827 | 0.58% | 15.96% | 12.42% | **17.92%** |

Pure positional routing is substantially easier to learn on the small task and
continues to gain broad digit signal on the variable task. Its 60-second OOD-N
T=1 token accuracy exceeds the 17.23% product control and the ordinary token
metric rises by 1.18 points. Exact remains below the control, so this is a
soft-gate exploratory H100 candidate rather than a demonstrated breakthrough.

The failure mode is also informative: fixed routing supplies stable addressing,
but modular reduction still needs value-dependent comparison and conditional
state updates. A future hybrid should preserve positional routing while adding
content-dependent computation without restoring content-dependent addressing.
