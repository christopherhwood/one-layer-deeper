# Evaluator-owned multipass recurrent model

This candidate tests whether two distinct, evaluator-owned gradients improve a
general deeply supervised recurrent transition. The first forward/backward pass
detaches state between its six refinement segments. The documented
`between_backward_passes` callback saves that gradient and enables full segment
backpropagation for the second evaluator-owned pass. `MultipassAdamW.step()`
averages the saved and current gradients before its normal AdamW update.

The loss is conservative: ordinary endpoint cross-entropy plus a 0.25-weighted
mean of endpoint cross-entropies from the six refinement readouts. There are no
generated arithmetic labels, nested model calls, participant-owned backward
passes, batch inspection, or task-specific consistency identities.

## Local results

| run | optimizer updates | test exact | OOD exact | mean exact |
|---|---:|---:|---:|---:|
| generated 10-second CPU smoke | 159 | 1.67% | 6.00% | 3.83% |
| full E5, 60 CPU seconds | 431 | 0.42% | 0.33% | 0.375% |

The full-E5 run used seed 74, training batch size 32, evaluation batch size 256,
and 96,010 persistent model-state elements. Its final recorded training loss was
2.763650. Exact full-E5 split counts were 5/1,200 test and 2/600 OOD.

The seen-modulus depth profile scored 3/512 (0.586%) at T=1 and certified no
rung. The OOD-modulus profile scored 0/512 at T=1 and certified no rung. The
30-second CPU evaluation allowance completed all seen-modulus rungs and OOD-N
rungs through T=4; later OOD-N rungs were not evaluated before the deadline.

## Comparison and decision

The single-pass `general_recurrent_deep` baseline reached 1,088 updates and
0.38% mean full-E5 exact accuracy in the same 60-second CPU allowance. Multipass
gradient averaging reached only 431 updates and the same 0.38% rounded mean.
Its seen-N T=1 result was somewhat higher, but OOD-N T=1 fell to zero. Thus the
second gradient changes optimization but produces no held-out or transferable
learning improvement per wall-clock budget.

This fails the local H100 gate. Evaluator-owned multipass averaging should not be
scaled as-is; a future multipass attempt needs a genuinely informative second
objective rather than two credit-assignment views of the same endpoint loss.
