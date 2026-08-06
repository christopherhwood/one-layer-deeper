# Temporal-teacher curriculum for product-ripple EMA

The student path and every supervised loss are identical to full
`ripple_product_ema`: all rows always receive endpoint CE, T=1 rows retain the
four phase endpoint losses, and the student always executes its requested
training T. Only teacher eligibility and compute depth follow a curriculum:

- first 10% of wall time: T=1 rows, one teacher square;
- 10% through 25%: T<=2 rows, up to two teacher squares;
- final 75%: every row, complete three-square training path.

EMA parameter updates continue after every optimizer step. Consistency is
applied only to rows currently covered by the teacher and otherwise uses the
unchanged softened endpoint KL. Student and teacher receive the same untouched
registers in one forward; no process labels or arithmetic state targets are
introduced. Evaluation remains the original clean student-only path.

## Local results

| model | updates | final train-batch exact | test exact | OOD exact | mean exact |
|---|---:|---:|---:|---:|---:|
| full EMA from the first update | 236 | 50.00% | 18.33% | 11.00% | 14.67% |
| T=1-only EMA | 338 | 59.38% | 20.00% | 21.00% | 20.50% |
| staged EMA curriculum | **286** | **75.00%** | 18.33% | **19.00%** | **18.67%** |
| staged EMA curriculum, full E5 60s | 981 | 0.00% | 0.4167% | 0.6667% | 0.5417% |

The curriculum completes 21% more updates than immediate full EMA while
retaining full teacher coverage for most training. Loss falls from 3.1403 at
step 1 to 0.6151 at the final update. It answers 11/60 generated test rows and
19/100 generated OOD rows exactly, with 725,780 persistent state elements.

Compilation, source validation, optimizer/state validation, finite backward,
all three scheduler stages, EMA updates, and evaluation pass. The source diff
from full EMA changes only teacher row/depth selection and its consistency mask;
student endpoint and phase losses are unchanged.

## Full-E5 result and comparison

The generated-smoke gain does not transfer. On the official equal-budget CPU
gate, final training-batch loss is 2.84707 with 0% exact accuracy. The model
answers 5/1,200 test rows and 4/600 OOD rows exactly. Seen-N T=1 is 0.7813%
(4/512), OOD-N T=1 is 0.0000% (0/512), and no rung certifies. The CPU depth
budget completes every seen-N rung but stops after the already-failed OOD-N
T=1 rung.

| full-E5 model | updates | test exact | OOD exact | mean exact | seen T=1 | OOD-N T=1 |
|---|---:|---:|---:|---:|---:|---:|
| clean full EMA | 937 | 0.9167% | 1.1667% | 1.0417% | 0.9766% | 0.3906% |
| staged curriculum | **981** | 0.4167% | 0.6667% | 0.5417% | 0.7813% | 0.0000% |

The curriculum recovers only 4.7% throughput and gets 9 ordinary held-out rows
correct versus 18 for clean EMA. Every reported accuracy dimension is worse,
including losing both OOD-N T=1 successes. Delaying full teacher coverage thus
costs more than the small early compute saving recovers.

Confidence is high that this candidate fails the H100 gate: it remains roughly
an order of magnitude below the required T=1 accuracies, has zero final train
exact, and certifies nothing. Confidence that its true accuracy is precisely
lower than clean EMA is moderate rather than high because this is one seed and
both models produce few successes. Still, the uniform degradation across test,
OOD, and both T=1 profiles despite higher throughput is enough to reject this
curriculum as the next H100 candidate.
