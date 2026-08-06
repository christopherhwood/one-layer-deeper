# Product-aware ripple with a two-round EMA teacher

This candidate preserves the `ripple_product_ema` student, optimizer, endpoint
and phase cross-entropies, EMA update, consistency KL, schedules, state, and
evaluation path. The student still executes four tied recurrent refinements per
square. Only the frozen training-time teacher is cheaper: it executes two
refinements per square while traversing the same full outer T path.

Two rounds were chosen instead of one because both early student phases already
receive legitimate T=1 endpoint supervision, and the second round preserves an
actual learned refinement transition. It cuts the teacher's recurrent scan work
in half without changing its parameterization or introducing a separate shallow
teacher. Student and teacher state dictionaries remain identical for EMA.

The change uses no new labels, generated targets, arithmetic state, augmentation,
or nested model calls. The teacher remains a frozen computation in the existing
forward and its parameters are updated only by the evaluator-called scheduler.

## Requested smoke result

| run | updates | final train-batch exact | test exact | OOD exact | mean exact |
|---|---:|---:|---:|---:|---:|
| generated 10-second CPU smoke | 287 | 40.63% | 23.33% | 22.00% | 22.67% |
| full E5, 60 CPU seconds | 1,065 | — | 0.3333% | 1.0000% | 0.6667% |

Training completed with finite gradients and exit status zero. Final training
loss was 1.255005. Exact held-out counts were 14/60 test and 22/100 OOD. The
model used the same 725,780 persistent state elements as the full-depth EMA
candidate.

For comparison, the four-round EMA teacher completed 236 updates and reached
18.33% test, 11.00% OOD, and 14.67% mean exact in its requested smoke. The
two-round teacher completes 22% more updates and improves mean exact by 8.00
percentage points. It also nearly matches the EMA-plus-value hybrid's 23.00%
smoke mean without adding the numeric loss that later regressed on full E5.

The full-E5 run completed 1,065 updates with a final training loss of 2.832190.
Exact split counts were 4/1,200 test and 6/600 OOD. Seen-modulus T=1 scored
0/512, OOD-modulus T=1 scored 3/512 (0.586%), and neither profile certified a
rung.

The smoke improvement therefore does not transfer. Although the shallow teacher
buys 14% more full-E5 updates than the four-round EMA teacher, mean exact falls
from 1.0417% to 0.6667%, test exact falls from 0.9167% to 0.3333%, and seen T=1
falls from 5/512 to zero. OOD-N T=1 rises by one example, but that isolated gain
does not offset the broad regression.

Reject the shallow teacher. Matching the student's complete refinement appears
important to the useful EMA signal on the variable-modulus distribution, even
though the cheaper teacher looks stronger on the tiny fixed-modulus smoke.
