# Position-only mixer with decoupled learned feedback

This candidate replaces the fixed local digit classifier with a trainable
orthogonal classifier whose feature and head gradients are decoupled:

```text
state          -> detached classifier -> feature cross-entropy
detached state -> classifier          -> head cross-entropy
```

The feature term updates the recurrent representation against a temporarily
stable coordinate. The smaller head term adapts that coordinate to task
semantics, while an orthogonality loss keeps its ten digit directions
well-conditioned. Both use the real T=1 endpoint labels. The evaluator still
owns the single ordinary backward pass.

## Gates

| gate | updates | exact | token | last token | OOD-N T=1 token |
|---|---:|---:|---:|---:|---:|
| fixed-N 10s | 606 | **42.33%** | **57.27%** | **49.67%** | — |
| variable-N 15s | 415 | 0.25% | **15.67%** | 10.63% | 16.76% |
| variable-N 60s | 1,673 | 0.54% | **16.81%** | 12.83% | 16.76% |

This is the strongest fixed-task accuracy and strongest ordinary
variable-modulus token accuracy in the position-only family. The 60-second OOD
evaluation token metric also reaches 17.32%. However, OOD-N T=1 remains below
the clean backbone's 17.92%, so the adaptive coordinate improves broad task
learning without identifying a modulus-length-general transition.

## Freeze ablation

Letting the classifier adapt through 30% of wall time, annealing its optimizer
group to zero by 55%, and then training against the frozen coordinate scored:

| gate | exact | token | last token | OOD-N T=1 token |
|---|---:|---:|---:|---:|
| fixed-N 10s | 39.33% | 56.09% | 47.33% | — |
| variable-N 15s | 0.25% | 15.55% | 10.58% | 16.38% |

Freezing loses both ordinary and OOD signal. Continued semantic adaptation is
part of the learning benefit; the retained submission is always adaptive.

This component does not clear the H100 gate. A next target-propagation attempt
must tie the local coordinate to the transition itself rather than only to
endpoint digit identity.
