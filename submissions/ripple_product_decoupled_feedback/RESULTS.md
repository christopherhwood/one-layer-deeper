# Ripple-product decoupled local feedback

Every tied recurrent refinement receives a digit objective through a learned
orthogonal classifier. The feature loss uses a detached classifier, while the
classifier loss uses a detached recurrent state. This prevents the two sides
from moving their coordinate system together while preserving the ordinary
endpoint and T=1 phase losses.

| gate | updates | exact | token | last token | OOD-N T=1 token |
|---|---:|---:|---:|---:|---:|
| fixed-N 10s | 139 | **26.17%** | **45.16%** | **48.50%** | — |
| variable-N 15s | 126 | **0.50%** | **15.74%** | **11.58%** | 16.65% |
| variable-N 60s | 508 | 0.17% | 15.70% | **12.75%** | **17.23%** |

The exact batch-128 control reaches 17.50% / 35.41% / 39.00% on the fixed
smoke and 0.33% / 15.01% / 10.17% on the short variable gate. Decoupled local
feedback therefore produces a real optimization gain on fixed and broad
short-run metrics. At 60 seconds, however, ordinary exact falls and OOD-N T=1
remains chance-scale (1/512 exact). The mechanism improves representation
learning but does not identify the reusable modular-reduction rule. Retain it
as a component, not an H100 candidate.
