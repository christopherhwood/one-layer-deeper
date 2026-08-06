# Ripple-product batch-128 control

This is the retained product-aware recurrent square transition with only two
operational changes: training uses batch 128, matching the best earlier hosted
configuration, and evaluation stops at the largest requested `T` in each batch.
The dynamic evaluation bound is output-equivalent to the old fixed 64 loops.

| gate | updates | exact | token | last token | OOD-N T=1 token |
|---|---:|---:|---:|---:|---:|
| fixed-N 10s | 148 | 17.50% | 35.41% | 39.00% | — |
| variable-N 15s | 132 | 0.33% | 15.01% | 10.17% | 16.76% |

This source is the causal control for the decoupled-feedback experiment. It is
not a new candidate by itself.
