# Pairwise relational-grid recurrence

The model forms a learned `W × W` grid over all source/context digit pairs,
runs a tied two-dimensional gated processor, and decodes with generic output
queries.  It contains no fixed pair product or significance reduction.

| gate | updates | exact | token | last token | OOD-N T=1 token |
|---|---:|---:|---:|---:|---:|
| fixed-N 10s | 546 | 37.00% | 55.46% | 63.00% | — |
| variable-N 15s | 258 | 0.58% | 15.70% | 11.63% | 16.97% |

The pairwise substrate learns the fixed task well, but variable and OOD-N
performance remains at the same marginal-distribution ceiling as one-dimensional
recurrent and attention models.  Pairwise capacity alone is not the missing
mechanism.
