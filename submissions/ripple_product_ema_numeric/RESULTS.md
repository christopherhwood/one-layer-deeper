# Ripple-product EMA with generic numeric geometry

This hybrid starts from `ripple_product_ema` and makes one representation-only
change. Its product grouping, bidirectional ripple scan, refinement depth,
outer recurrence, exact endpoint losses, T=1 phase losses, temporal EMA teacher,
optimizer, and schedules remain unchanged.

## Numeric interface

Every hard or soft decimal token receives six generic ordered coordinates:
normalized scalar, squared scalar, and sine/cosine pairs at frequencies one and
two. Learned projections add those coordinates to:

- the modulus digit features;
- the current-register digit features; and
- the learned digit-pair table before pair features are grouped by positional
  significance.

The arbitrary learned categorical projections and pair table remain present.
The numeric projections are only a smooth residual representation. They do not
encode digit multiplication, carries, quotient selection, comparison, modular
reduction, or any intermediate target.

There is deliberately no ordinal/value loss in this hybrid. Exact CE and EMA
consistency are unchanged, isolating numeric representation from the separate
numeric-loss experiment.

## Validation

The following checks passed:

- `python -m py_compile` using the repository virtual environment
- `benchmark.validation.lint_submission_source`
- `git diff --check`
- official runner model/optimizer validation, finite student backward, EMA
  update callback, training, and held-out evaluation

The smoke-task model has 731,924 persistent state elements, including the frozen
EMA teacher. Numeric geometry adds 6,144 elements across student and teacher.

## Ten-second contract smoke

Command:

```text
.venv/bin/python -m benchmark.runner \
  --manifest benchmark/manifests/local_cpu_10s.json \
  --submission-file submissions/ripple_product_ema_numeric/submission.py
```

Seed 74 completed 263 optimizer updates. Final training loss was 1.0692.

| split | exact accuracy | correct / examples | loss |
|---|---:|---:|---:|
| test | **21.6667%** | 13 / 60 | 4.6605 |
| OOD | **18.0000%** | 18 / 100 | 3.2307 |
| mean | **19.8333%** | 31 / 160 | 3.9456 |

Under the identical smoke manifest, EMA without numeric geometry reached 18.33%
test, 11.00% OOD, and 14.67% mean after 236 updates. The hybrid therefore gains
3.34 percentage points on test, 7.00 on OOD, and 5.16 on the mean. Its final
training loss is higher than EMA alone (1.0692 versus 0.8967), so the held-out
gain is not simply stronger fitting.

## Full E5 CPU gate

The full public E5 gate completed 902 optimizer updates. Final training loss was
2.86491.

| ordinary split | exact accuracy | correct / examples |
|---|---:|---:|
| test | 0.3333% | 4 / 1,200 |
| OOD | 0.6667% | 4 / 600 |
| mean | **0.5000%** | 8 / 1,800 |

The seen-modulus T=1 rung reached 0.1953% (1 / 512), and OOD-N T=1 also reached
0.1953% (1 / 512). Neither profile certified T=1.

The strong short smoke result did not transfer. EMA without numeric geometry
reached 1.0417% mean on the same full-E5 gate, more than twice this hybrid's
0.5000%. Numeric geometry can help the small generated task while biasing the
representation in a way that harms real variable-modulus generalization. This
variant is rejected and does not justify an H100 run.
