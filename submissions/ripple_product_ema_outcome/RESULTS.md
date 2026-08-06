# Ripple EMA with horizon-conditioned temporal outcome heads

This candidate preserves the full `ripple_product_ema` submitted transition,
four-round EMA teacher, endpoint KL, endpoint and T=1 phase cross-entropies,
outer-state detachment, optimizer/schedules, and evaluation path.

During training only, each active post-square register is also read by one small
shared outcome head. Its inputs at every LSD-first digit position are:

- the current soft register digit distribution;
- the immutable modulus digit distribution;
- `remaining_steps / 64` and `log1p(remaining_steps) / log(65)`.

A width-48 bidirectional GRU produces endpoint digit logits. Every temporal head
is trained against the genuine evaluator-supplied final endpoint. Losses across
all valid active head-token slots are combined into one mean and added at total
weight 0.20. The outcome head is shared across time and continuous horizon
features avoid a separate lookup embedding for each T.

The head is skipped entirely during evaluation; submitted logits still come only
from the original recurrent register. The head therefore changes training credit,
not inference behavior.

## Rules and distinction from phase supervision

This uses no intermediate residue, product, quotient, carry, borrow, or generated
label. A temporal head estimates the observed endpoint from an intermediate
representation; it does not assert that the intermediate register equals that
endpoint. All heads run inside the model's single evaluator-owned training
forward, and the custom loss returns one ordinary differentiable scalar.

Existing phase supervision is narrower: it gives short paths among refinements
inside the first square only on T=1, where each phase directly predicts the same
square endpoint. The temporal outcome head also supervises post-square states on
T=2 and T=3. Because the existing outer recurrence detaches each later square's
input, each auxiliary endpoint loss supplies a short local gradient through its
immediately preceding square and the shared outcome head.

## Requested smoke result

| run | updates | final train-batch exact | test exact | OOD exact | mean exact |
|---|---:|---:|---:|---:|---:|
| generated 10-second CPU smoke | 200 | 43.75% | 26.67% | 21.00% | 23.83% |
| full E5, 60 CPU seconds | 846 | — | 0.2500% | 1.0000% | 0.6250% |

Training completed with finite gradients and exit status zero. Final training
loss was 1.280599. Exact held-out counts were 16/60 test and 21/100 OOD. The
model used 756,174 persistent state elements, 30,394 more than endpoint-only EMA.

Endpoint-only EMA completed 236 smoke updates and reached 18.33% test, 11.00%
OOD, and 14.67% mean. The temporal head costs 15% of update throughput but gains
8.17 percentage points of test exact, 10.00 points of OOD exact, and 9.17 points
of mean exact (62.5% relative). It also edges the EMA-plus-value and shallow-EMA
smoke means while remaining balanced across test and OOD.

The full-E5 run completed 846 updates with a final combined training loss of
3.378480. Exact held-out counts were 3/1,200 test and 6/600 OOD. Seen-modulus
T=1 scored 2/512 (0.391%), OOD-modulus T=1 scored 1/512 (0.195%), and neither
profile certified a rung. The 30-second CPU evaluation allowance completed all
seen-modulus rungs but only OOD-N T=1 before its deadline.

## Identical-gate comparison and decision

| model | updates | test exact | OOD exact | mean exact | seen T=1 | OOD-N T=1 |
|---|---:|---:|---:|---:|---:|---:|
| ripple + EMA | 937 | **0.9167%** | **1.1667%** | **1.0417%** | **0.9766%** | **0.3906%** |
| ripple + EMA + temporal outcome | 846 | 0.2500% | 1.0000% | 0.6250% | 0.3906% | 0.1953% |

The strong smoke result does not transfer. The temporal head costs about 10% of
full-E5 update throughput and underperforms endpoint-only EMA on every reported
accuracy: mean exact falls by 40%, test exact falls by 73%, and both depth T=1
profiles lose successes.

The auxiliary head can learn the tiny smoke endpoints without establishing the
transition's reusable semantics. On the broader variable-modulus distribution it
adds another endpoint-prediction route that consumes capacity and compute but
does not improve the submitted register. Reject this candidate; horizon-
conditioned outcome supervision does not pass the full-E5 or H100 gate.
