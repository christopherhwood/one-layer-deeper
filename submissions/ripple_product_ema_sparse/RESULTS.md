# Product-aware ripple with alternating EMA consistency

This variant preserves the complete `ripple_product_ema` student architecture,
student endpoint CE, T=1 phase endpoint losses, optimizer, temperature schedule,
EMA update, and evaluation path.

The only change is compute allocation. The full frozen-teacher forward and KL
consistency term run on optimizer updates 1, 3, 5, and so on. On intervening
updates the scheduler-controlled model flag skips the teacher forward entirely
and the loss returns the unchanged student objective. The EMA weights are still
updated after every student optimizer step, so each active teacher is current.

No consistency reweighting compensates for inactive updates. Thus this directly
tests whether periodic temporal stabilization gives a better accuracy/throughput
tradeoff than running the teacher on every batch.

## Validation

The following checks passed:

- `python -m py_compile` using the repository virtual environment
- `benchmark.validation.lint_submission_source`
- `git diff --check`
- official runner model/optimizer validation, finite active- and inactive-step
  backward passes, EMA scheduler updates, training, and held-out evaluation

The smoke-task model has 725,780 persistent state elements, unchanged from full
EMA.

## Ten-second contract smoke

Command:

```text
.venv/bin/python -m benchmark.runner \
  --manifest benchmark/manifests/local_cpu_10s.json \
  --submission-file submissions/ripple_product_ema_sparse/submission.py
```

Seed 74 completed 290 optimizer updates. Final training loss was 1.1795.

| split | exact accuracy | correct / examples | loss |
|---|---:|---:|---:|
| test | **20.0000%** | 12 / 60 | 5.2286 |
| OOD | **15.0000%** | 15 / 100 | 5.1301 |
| mean | **17.5000%** | 27 / 160 | 5.1793 |

Full EMA under the same smoke manifest completed 236 updates and reached 18.33%
test, 11.00% OOD, and 14.67% mean. Alternating EMA therefore completed 22.9%
more updates and gained 1.67 percentage points on test, 4.00 on OOD, and 2.83
on the mean. The higher held-out CE loss and small evaluation set make this a
directional result rather than proof of better calibration or E5 transfer.

## Full E5 CPU gate

The full public E5 gate completed 1,074 optimizer updates. Final training loss
was 2.94434.

| ordinary split | exact accuracy | correct / examples |
|---|---:|---:|
| test | 0.2500% | 3 / 1,200 |
| OOD | 0.3333% | 2 / 600 |
| mean | **0.2917%** | 5 / 1,800 |

The seen-modulus T=1 rung reached 0.5859% (3 / 512), while OOD-N T=1 reached
0.1953% (1 / 512). Neither profile certified T=1.

The smoke throughput benefit transferred—1,074 updates is healthy—but the
accuracy benefit did not. Sparse EMA falls well below full EMA's 1.0417% E5
mean, showing that consistency on alternating batches is too weak or too
intermittent for the real variable-modulus task. This variant is rejected and
does not justify an H100 run.
