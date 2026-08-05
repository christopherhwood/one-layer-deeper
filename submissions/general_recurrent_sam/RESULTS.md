# Sharpness-aware general recurrent model

This candidate implements genuine two-pass SAM using only evaluator-owned
forward/loss/backward passes. The first pass obtains the ordinary endpoint/deep
gradient. The documented `between_backward_passes` callback computes its global
L2 norm and adds a radius-0.05 ascent perturbation. The evaluator clears the
first gradient and obtains a second gradient at the perturbed parameters.
`SharpnessAwareAdamW.step()` restores the original parameters and applies only
that second gradient with AdamW.

The first gradient is not averaged into the update. Perturbations and gradients
are checked for finiteness, missing passes fail explicitly, and every stored
perturbation is removed before the optimizer update. The training loss is the
same conservative endpoint cross-entropy plus a 0.25-weighted mean endpoint
loss over six recurrent refinement readouts used by the multipass comparison.
There are no nested calls, participant-owned backward passes, generated process
labels, data inspection, or task-specific consistency identities.

## Local results

| run | optimizer updates | test exact | OOD exact | mean exact |
|---|---:|---:|---:|---:|
| generated 10-second CPU smoke | 165 | 5.00% | 2.00% | 3.50% |
| full E5, 60 CPU seconds | 373 | 0.50% | 0.17% | 0.333% |

The full-E5 run used seed 74, training batch size 32, evaluation batch size 256,
and 96,010 persistent model-state elements. Its final recorded training loss was
2.893843. Exact full-E5 split counts were 6/1,200 test and 1/600 OOD.

The seen-modulus depth profile scored 1/512 (0.195%) at T=1 and certified no
rung. The OOD-modulus profile scored 0/512 at T=1 and certified no rung. The
30-second CPU evaluation allowance completed all seen-modulus rungs and OOD-N
rungs through T=32; OOD-N T=64 was not completed before the deadline.

## Comparison and decision

The single-pass `general_recurrent_deep` baseline reached 1,088 updates and
0.38% rounded mean full-E5 exact accuracy. The two-gradient averaging candidate
reached 431 updates and the same 0.38% rounded mean. Genuine SAM reached only
373 updates and 0.33% mean, with no OOD-N T=1 successes.

SAM therefore changes the optimization geometry as intended but does not fix
the semantic learning bottleneck. Its second pass and perturbation bookkeeping
also reduce the number of updates available under a wall-clock budget. This
candidate fails the local H100 gate and should not be scaled as-is.
