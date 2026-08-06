# Product-ripple phase-loss ablation

This is a strict control for the T=1 refinement-phase endpoint loss in tracked
`ripple_product_h100`. The only source change is deletion of the 12-line block
that adds mean phase CE with weight 0.35. Everything else is preserved:

- product-aware recurrent architecture and all phase computations;
- final endpoint CE and its 4x T=1 row weighting;
- truncated outer recurrence and temperature annealing;
- optimizer, learning-rate schedule, batch sizes, and evaluation path.

Phase logits are intentionally still produced so this tests their gradient
contribution rather than changing the forward architecture or throughput.

## Local results

| run | updates | final train-batch exact | test exact | OOD exact | mean exact |
|---|---:|---:|---:|---:|---:|
| generated 10-second CPU smoke | 312 | 56.25% | 20.00% | 19.00% | 19.50% |
| full E5, 60 CPU seconds | 1,261 | 3.125% | 0.3333% | 0.1667% | 0.2500% |

Loss falls from 2.3186 at step 1 to 0.6669 on the final batch. The model answers
12/60 test rows and 19/100 OOD rows exactly. Persistent state remains 362,890
elements.

Compilation, source validation, diff checking, optimizer/state validation,
finite backward, and evaluation all pass. The source diff against tracked
`ripple_product_h100` contains exactly 12 deletions and no additions.

## Full-E5 comparison

On full E5, phase-free loss is 2.06069 on the final training batch. It answers
4/1,200 test rows and 1/600 OOD rows exactly. Seen-N T=1 is 0.3906% (2/512),
OOD-N T=1 is 0.0000% (0/512), and no rung certifies. The CPU evaluation budget
completes all seen-N rungs but stops after the already-failed OOD-N T=1 rung.

| full-E5 model | phase CE | EMA | updates | test exact | OOD exact | mean exact | seen T=1 | OOD-N T=1 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| plain tracked ripple | yes | no | 1,259 | 1.0833% | 0.3333% | 0.7083% | 0.1953% | 0.0000% |
| phase-free ripple | no | no | 1,261 | 0.3333% | 0.1667% | 0.2500% | 0.3906% | 0.0000% |
| clean full EMA | yes | yes | 937 | 0.9167% | 1.1667% | 1.0417% | 0.9766% | 0.3906% |

Phase-free and plain ripple have effectively identical throughput, so the
threefold mean-accuracy drop is not a step-count artifact. Intermediate phase
gradients are therefore useful overall on full E5 rather than intrinsically
harmful. The seen-T=1 count rises from one to two examples, but ordinary test
successes fall from 13 to four and OOD successes from two to one.

This sharpens the PCGrad interpretation: a projected-gradient improvement
cannot be attributed merely to suppressing phase supervision. The useful test
is whether projection retains information from the phase objective while
preventing its conflicting components from damaging the endpoint update. Clean
EMA remains the strongest of these three controls on mean and OOD accuracy,
though all remain far below certification and the H100 gate.
