# Balanced discrete-action serial reducer

The product-column streaming reducer replaces its unrestricted global control
vector with a ten-way soft categorical action.  A generic mutual-information
term encourages sharp per-example actions and balanced aggregate usage without
assigning arithmetic semantics.

| gate | updates | exact | token | last token | OOD-N T=1 token |
|---|---:|---:|---:|---:|---:|
| fixed-N 10s | 569 | 24.50% | 38.24% | 45.00% | — |
| variable-N 15s | 283 | 0.54% | 14.99% | 10.25% | not completed |
| variable-N 60s | 1,147 | 0.50% | **15.98%** | **11.88%** | 16.01% |

Longer training gives broad ordinary digit-error improvement: mean wrong digits
falls to 2.4925 and both test and OOD token accuracy reach about 16%.  Exact
accuracy stays at chance scale, however, and the improvement does not transfer
to unseen moduli.  The discrete code helps organize the seen distribution but
does not identify a modulus-general reduction operation.  Do not promote to
H100 without a stronger OOD-N mechanism.
