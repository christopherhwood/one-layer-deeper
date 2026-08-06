# Observable digit-belief ripple transducer

The only state persistent between product-column scans is a normalized
distribution over the ten actual output digits. A 48-wide tied GRU controller
resets after each alternating forward/reverse scan. This removes a free latent
tape while retaining a length-independent local carrier.

| gate | updates | exact | token | last token | OOD-N T=1 token |
|---|---:|---:|---:|---:|---:|
| fixed-N 10s | 531 | 6.50% | 29.03% | 14.83% | — |
| variable-N 15s | 520 | 0.50% | 15.45% | 10.96% | 17.02% |
| variable-N 60s | 2,109 | 0.75% | 15.31% | 11.25% | 16.28% |

The constrained model remains trainable and initially improves ordinary digit
error, but the OOD-N signal declines with more updates and exact stays at chance
scale. Observable state alone does not make the endpoint identify a reusable
reduction transition. Reject before H100.
