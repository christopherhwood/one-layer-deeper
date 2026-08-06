# Position-only whole-sequence feedback

This candidate maps every real T=1 endpoint sequence to a fixed random code by
summing position-from-the-right and digit-specific vectors. At each tied
refinement, the pooled output-position state receives a cosine loss toward that
joint code. The ordinary endpoint and learned phase losses remain unchanged.
There are no generated examples or arithmetic process labels.

| gate | updates | exact | token | last token | OOD-N T=1 token |
|---|---:|---:|---:|---:|---:|
| fixed-N 10s | 674 | **37.83%** | **53.73%** | **44.33%** | — |
| variable-N 15s | 440 | 0.33% | 14.78% | 9.71% | 16.38% |

The joint target clears the fixed smoke and improves suffix accuracy, but it
falls below the clean position-only backbone on every variable relaxed metric.
A random exact-sequence geometry is learnable on the fixed task but is tied to
the training answer/length distribution. It does not identify the modular
transition and receives no 60-second or H100 run.
