# Product-aware ripple with EMA phase consistency

This candidate preserves the full four-refinement `ripple_product_ema` student
and teacher, endpoint EMA KL, optimizer, endpoint and phase cross-entropies,
schedules, state, and evaluation path. During the existing training-time teacher
forward, it additionally retains the teacher's four first-square phase logits.

Corresponding student and EMA-teacher phases are aligned with softened KL at
weight 0.10. This phase term is restricted to T=1 rows. On those rows every
first-square refinement head already predicts the same evaluator-supplied
`x^2 mod N` endpoint and receives exact CE, so the consistency term stabilizes
an existing legitimate prediction rather than inventing an intermediate target.
The original weight-0.15 endpoint EMA KL remains active on all valid rows.

There are no new labels, generated targets, process states, augmented examples,
or nested model calls. Teacher phases come from the same frozen EMA computation
already performed inside the model's single evaluator-owned forward.

## Requested smoke result

| run | updates | test exact | OOD exact | mean exact |
|---|---:|---:|---:|---:|
| generated 10-second CPU smoke | 238 | 16.67% | 14.00% | 15.33% |

Training completed with finite gradients and exit status zero. Final training
loss was 1.193390. Exact held-out counts were 10/60 test and 14/100 OOD. The
model used the same 725,780 persistent state elements as the endpoint-only EMA
candidate.

For comparison, endpoint-only EMA completed 236 updates and reached 18.33%
test, 11.00% OOD, and 14.67% mean exact in its smoke. Phase consistency retains
the same throughput and moves three OOD examples in the favorable direction,
but loses one test example. The net mean gain is only 0.67 percentage points.

The mechanism is rules-valid and inexpensive, but this tiny improvement is not
material relative to smoke-set variance and is much weaker than the candidates
that justified sequential full-E5 gates. Do not scale it solely on this result;
retain endpoint-only EMA as the better-evidenced default unless an additional
repeat or combined hypothesis establishes a clearer gain.
