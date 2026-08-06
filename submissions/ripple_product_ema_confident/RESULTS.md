# Product-aware ripple with confidence-weighted EMA consistency

This controlled variant preserves the complete `ripple_product_ema`
architecture, teacher forward on every training batch, EMA decay, consistency
temperature and ramp, endpoint CE, T=1 phase losses, optimizer, schedules, and
evaluation path. Only the aggregation of the existing endpoint KL changes.

## Confidence weighting

For each output token, the detached EMA teacher distribution is restricted and
renormalized over the ten digit tokens. Its normalized confidence is

```text
token confidence = 1 - entropy(teacher digits) / log(10)
```

This is zero for a uniform digit teacher and approaches one for a concentrated
teacher. Row confidence is the mean token confidence across valid output
positions. Each token KL is weighted by the geometric mean of its token and row
confidence, then divided by the original valid-token count. Consequently an
uncertain teacher contributes less total consistency gradient rather than being
renormalized back to full strength.

The weights depend only on teacher certainty. They never inspect endpoint
labels, correctness, arithmetic state, or process targets. Exact endpoint CE is
unchanged and active on every row. EMA weight 0.15, temperature 2.0, decay 0.995,
and the first-20%-of-time consistency ramp all remain unchanged.

## Validation

The following checks passed:

- `python -m py_compile` using the repository virtual environment
- `benchmark.validation.lint_submission_source`
- `git diff --check`
- official runner model/optimizer validation, finite student backward,
  teacher/EMA update, training, and held-out evaluation

The smoke-task model has 725,780 persistent state elements, identical to the
baseline full-EMA model.

## Ten-second contract smoke

Command:

```text
.venv/bin/python -m benchmark.runner \
  --manifest benchmark/manifests/local_cpu_10s.json \
  --submission-file submissions/ripple_product_ema_confident/submission.py
```

Seed 74 completed 266 optimizer updates. Final training loss was 0.8546.

| split | exact accuracy | correct / examples | loss |
|---|---:|---:|---:|
| test | **23.3333%** | 14 / 60 | 5.5559 |
| OOD | **25.0000%** | 25 / 100 | 4.0397 |
| mean | **24.1667%** | 39 / 160 | 4.7978 |

Baseline full EMA under the identical manifest completed 236 updates, reached
18.33% test, 11.00% OOD, and 14.67% mean, with final loss 0.8967. Confidence
weighting gains 5.00 percentage points on test, 14.00 on OOD, and 9.50 on mean,
while final training loss is slightly lower.

## Full E5 CPU gate

The full public E5 gate completed 939 optimizer updates.

| ordinary split | exact accuracy | correct / examples |
|---|---:|---:|
| test | 1.0833% | 13 / 1,200 |
| OOD | 0.8333% | 5 / 600 |
| mean | **0.9583%** | 18 / 1,800 |

The seen-modulus T=1 rung reached 0.78125% (4 / 512), while OOD-N T=1 reached
0% (0 / 512). Neither profile certified T=1.

The 24.17% smoke mean did not replicate on E5. Confidence weighting finishes
below baseline full EMA's 1.0417% mean and still solves no unseen-modulus T=1
examples. Teacher certainty changes which endpoint KL gradients dominate, but
does not provide the missing information that identifies reusable modular
reduction. This variant is rejected and does not justify an H100 run.
