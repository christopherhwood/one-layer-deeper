# Endpoint-anchored fixed-point recurrent model

This path adds a generic convergence objective to the deeply supervised
recurrent candidate. Every refinement phase remains anchored to the observed
endpoint. For the final four phases, consecutive answer/scratch representations
receive symmetric stop-gradient cosine agreement, and consecutive readouts
receive symmetric stop-gradient KL agreement. All terms are computed from one
ordinary forward pass. The model uses no arithmetic features, process labels,
masking, augmentation, or nested model calls.

## Local results

| run | updates | test exact | OOD exact | mean exact |
|---|---:|---:|---:|---:|
| generated 10-second CPU smoke | 365 | 1.67% | 4.00% | 2.83% |
| full E5, 60 CPU seconds | 702 | 0.50% | 0.17% | 0.33% |

The 10-second smoke passed the submission contract and completed finite
training. Its final training batch reached 18.75% exact accuracy and loss
2.5543, but this did not transfer to the generated held-out splits.

On full E5, loss was 3.4945 at step 1 and 3.0733 at the end. Final training
batch exact accuracy was 0%. The model used 96,010 persistent state elements.
The complete depth measurements were:

| T | seen-N exact | OOD-N exact |
|---:|---:|---:|
| 1 | 0.1953% | 0.3906% |
| 2 | 0.1953% | 0.0000% |
| 4 | 0.5859% | 0.0000% |
| 8 | 1.1719% | 0.0000% |
| 16 | 0.0000% | 0.0000% |
| 32 | 0.9766% | 0.0000% |
| 64 | 0.7812% | not completed within the CPU evaluation budget |

No seen-N or OOD-N rung certified.

## Comparison and conclusion

The directly comparable deeply supervised candidate scored 0.38% mean exact
on the same 60-second full-E5 CPU gate. Fixed-point consistency scored 0.33%.
The difference is too small and in the wrong direction to indicate progress.
The predeclared H100 gate of at least 5% held-out T=1 and 3% OOD/adversarial
T=1 is missed by more than an order of magnitude.

The auxiliary makes late recurrent states agree, but endpoint anchoring still
does not identify a reusable transition. This path should not consume an H100
attempt, and tuning its agreement weights or number of phases would mostly
rehash the same endpoint-semantic failure.
