# Discrete-action serial reducer with digit CRF

This combines the globally discrete product-column serial reducer with the
first-order observed-digit CRF that improved the product-attention model. The
CRF is trained only on supplied endpoint digits and uses Viterbi at evaluation.

| gate | updates | exact | token | last token | OOD-N T=1 token |
|---|---:|---:|---:|---:|---:|
| fixed-N 10s | 557 | 26.50% | 37.53% | 47.00% | — |
| variable-N 15s | 279 | 0.75% | 14.44% | 10.38% | not completed |
| variable-N 60s | 1,134 | 0.88% | 14.66% | 10.75% | 14.53% |

The chain converts a few more rows to exact on seen-scale data, but destroys
the parent reducer's broad 60-second token gain and severely degrades unseen-N
digits. Its adjacent-digit correlations are distribution-specific rather than
evidence of coherent modular-reduction states. Reject before H100.
