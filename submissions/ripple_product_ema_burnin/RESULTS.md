# Product-aware ripple with EMA consistency burn-in

This candidate preserves the full `ripple_product_ema` architecture, four-round
student and teacher, endpoint EMA KL, endpoint and T=1 phase cross-entropies,
optimizer, EMA decay, temperature/LR schedule, state, and evaluation behavior.
Only the scalar schedule for EMA consistency changes.

The exact wall-clock schedule is:

```text
training fraction 0.00 through 0.15: consistency scale = 0
training fraction 0.15 through 0.40: linear ramp from 0 to 1
training fraction 0.40 through 1.00: consistency scale = 1
```

The teacher parameters still receive their EMA update after every optimizer step,
starting at step one. Thus the teacher accumulates a useful temporal ensemble
during burn-in without pushing the student toward its initially random outputs.
No labels, forward calls, parameter state, or computation paths are added.

## Requested smoke result

| run | updates | test exact | OOD exact | mean exact |
|---|---:|---:|---:|---:|
| generated 10-second CPU smoke | 272 | 16.67% | 15.00% | 15.83% |

Training completed with finite gradients and exit status zero. Final training
loss was 1.317747. Exact held-out counts were 10/60 test and 15/100 OOD. The
model used the same 725,780 persistent state elements as the baseline EMA model.

For comparison, immediate-ramp EMA reached 18.33% test, 11.00% OOD, and 14.67%
mean in its smoke. Burn-in trades one test example for four additional OOD
examples, improving mean exact by 1.17 percentage points. Because model compute
is unchanged, the update-count difference between these short wall-clock runs
should be treated as runtime noise rather than a schedule advantage.

The OOD direction is encouraging, but the net smoke gain is modest and the tiny
fixed-modulus smoke has repeatedly failed to predict full-E5 ordering. Treat this
as a weak schedule signal, not an H100 gate. A full-E5 run is justified only if
the sequential budget permits resolving this narrowly isolated hypothesis.

The prior `ripple_product_ema_phases` candidate remains smoke-only; no full-E5
measurement has been added for it.
