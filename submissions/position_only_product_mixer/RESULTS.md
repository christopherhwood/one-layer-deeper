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

## Hosted Hard result

Submission `ad7bfb2c-82dd-4771-b6e9-777adefbad3f` completed 139,579 optimizer
updates in the 3,600-second H100 budget and scored **0.03%**. Logged train-batch
exact accuracy peaked at 1.6% and ended at 0%; held-out test, OOD-T, and OOD-N-T
exact accuracy were all 0%. Final train loss was 2.197 and held-out losses stayed
near 2.17--2.18.

This converts the earlier soft signal into a decisive rejection. Fixed routing
is not merely short of compute: after roughly 140k updates it still cannot fit
the hidden Hard task. It should not be the primary hosted backbone.

The failure mode remains informative: fixed routing supplies stable addressing,
but modular reduction still needs value-dependent comparison and conditional
state updates. A future hybrid should preserve positional routing while adding
content-dependent computation without restoring content-dependent addressing.
