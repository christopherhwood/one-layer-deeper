# Product-aware ripple with global/local PCGrad

This experiment changes credit assignment rather than architecture or scalar
loss weighting. The model is the tracked product-aware ripple student without
an EMA teacher. Its product columns, four tied bidirectional refinements, outer
T recurrence, digit interface, temperature schedule, and evaluation path are
unchanged.

## Two evaluator-owned objectives

Each optimizer update requests exactly two evaluator-owned
forward/loss/backward passes over the same untouched batch:

1. The global pass computes only the genuine submitted endpoint CE over all
   rows, retaining the baseline 4x T=1 row weighting.
2. The local pass computes only the mean genuine endpoint CE from the four
   first-square refinement heads, restricted to T=1 rows where that endpoint is
   exactly the supplied label.

After pass one, the documented `between_backward_passes` callback saves the
clipped global gradients and switches the model's loss mode. The evaluator
clears gradients and owns the second backward. Immediately before ordinary
AdamW, the optimizer computes the global dot product between the two gradients.
If it is negative, only the local gradient is projected off the global gradient:

```text
g_local' = g_local - min(0, <g_global,g_local>) / ||g_global||^2 * g_global
update gradient = g_global + 0.35 * g_local'
```

When gradients agree, the local gradient is unchanged. The global endpoint
gradient is never projected, so the auxiliary objective cannot oppose it to
first order. Loss mode and temporary gradients are cleared after every update,
including through a `finally` reset around AdamW.

## Rules rationale

This uses the public contract's explicit allowance for two evaluator-owned
passes and its documented between-pass gradient transformation callback. The
submission starts no model call, loss call, backward, or optimizer update from
inside a callback. It uses only evaluator-supplied final labels, creates no
examples or process labels, and contains no hard-coded arithmetic operation.
The custom optimizer performs one ordinary AdamW update after combining the two
already evaluator-produced gradients.

This differs from the earlier generic multipass experiment: that run averaged
detach/full-BPTT gradients from the same combined loss. Here the passes have
separate global and local endpoint objectives, and conflict is explicitly
removed instead of averaged.

## Validation

The following checks passed:

- `python -m py_compile` using the repository virtual environment
- `benchmark.validation.lint_submission_source`
- `git diff --check`
- official validation of both evaluator-owned passes, the between-pass
  callback, custom optimizer, finite gradients, model/optimizer state,
  training, and held-out evaluation

The smoke-task model has 362,890 persistent state elements. Its optimizer state
after the first update has 725,802 elements; saved first-pass gradients are
temporary workspace and are cleared after every update.

## Ten-second contract smoke

Command:

```text
.venv/bin/python -m benchmark.runner \
  --manifest benchmark/manifests/local_cpu_10s.json \
  --submission-file submissions/ripple_product_pcgrad/submission.py
```

Seed 74 completed 234 optimizer updates, representing 468 evaluator-owned
forward/backward passes.

| split | exact accuracy | correct / examples | loss |
|---|---:|---:|---:|
| test | **28.3333%** | 17 / 60 | 5.3813 |
| OOD | **32.0000%** | 32 / 100 | 4.0247 |
| mean | **30.1667%** | 49 / 160 | 4.7030 |

The runner's final training loss is 0.3293, but this is the second pass's local
phase loss. It is not comparable to single-pass endpoint losses. Training exact
accuracy and all evaluation values are still computed from the submitted final
endpoint and remain comparable.

For context, full EMA's smoke mean was 14.67%, confidence-weighted EMA reached
24.17%, and both later failed to materially improve E5. PCGrad is the strongest
smoke result yet and is structurally differentiated, but the small generated
task has repeatedly produced false positives.

## Full E5 CPU gate

The unchanged submission was then run with the official full-E5 60-second CPU
gate. It completed 730 optimizer updates, or 1,460 evaluator-owned
forward/backward passes. The runner's final recorded loss was 2.21022, again the
second-pass local phase loss rather than the global endpoint loss.

| ordinary split | exact accuracy | correct / examples | loss |
|---|---:|---:|---:|
| test | 0.2500% | 3 / 1,200 | 2.2634 |
| OOD | 0.3333% | 2 / 600 | 2.1918 |
| mean | **0.2917%** | 5 / 1,800 | 2.2276 |

The seen-modulus T=1 rung reached 0.78125% (4 / 512) and did not certify. The
30-second CPU evaluation allowance completed every seen-modulus rung but expired
before OOD-N T=1 completed, so no OOD-N accuracy should be inferred from this
run.

## Identical-gate comparison

| model | updates | test | OOD | mean | seen T=1 | OOD-N T=1 |
|---|---:|---:|---:|---:|---:|---:|
| tracked ripple baseline | 1,259 | **1.0833%** | 0.3333% | 0.7083% | 0.1953% | 0.0000% |
| ripple + EMA | 937 | 0.9167% | **1.1667%** | **1.0417%** | **0.9766%** | **0.3906%** |
| ripple + PCGrad | 730 | 0.2500% | 0.3333% | 0.2917% | 0.7813% | not completed |

PCGrad raises seen-T=1 exact count from one to four relative to the plain ripple
baseline, consistent with stronger shallow-phase credit. That narrow movement
does not transfer to the ordinary endpoint distribution: mean accuracy falls by
59% versus the plain ripple and by 72% versus EMA. Two backward passes also cut
optimizer throughput substantially.

The 30.17% smoke result therefore does not replicate. Protecting the global
gradient from direct local conflict is insufficient because the local endpoint
objective still does not identify modular-reduction semantics. This variant is
rejected and does not justify an H100 run.
