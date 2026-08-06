# Product-aware ripple with faster, stronger EMA dynamics

This is a controlled temporal-target ablation of `ripple_product_ema`. The
student architecture, full teacher forward on every training batch, exact
endpoint CE, T=1 phase endpoint losses, consistency temperature, consistency
ramp, optimizer, temperature schedule, batch sizes, and evaluation path are
unchanged.

Exactly two constants change:

| setting | baseline EMA | fast EMA |
|---|---:|---:|
| maximum EMA decay | 0.995 | **0.99** |
| consistency weight | 0.15 | **0.30** |

The existing bias-corrected early decay remains
`min(maximum_decay, 1 - 1/(updates+1))`, and consistency still ramps over the
first 20% of wall-clock training at temperature 2.0.

The rationale is coupled: decay 0.99 lets the teacher follow a student changing
quickly under a short training budget, reducing target staleness; weight 0.30
makes the resulting less-stale soft target consequential. No new objective,
label, architecture, arithmetic feature, or process supervision is introduced.

## Validation

The following checks passed:

- `python -m py_compile` using the repository virtual environment
- `benchmark.validation.lint_submission_source`
- `git diff --check`
- official runner model/optimizer validation, finite student backward, EMA
  scheduler update, training, and held-out evaluation

The smoke-task model has 725,780 persistent state elements, identical to the
baseline EMA model.

## Ten-second contract smoke

Command:

```text
.venv/bin/python -m benchmark.runner \
  --manifest benchmark/manifests/local_cpu_10s.json \
  --submission-file submissions/ripple_product_ema_fast/submission.py
```

Seed 74 completed 249 optimizer updates. Final training loss was 0.8678.

| split | exact accuracy | correct / examples | loss |
|---|---:|---:|---:|
| test | **21.6667%** | 13 / 60 | 5.5462 |
| OOD | **13.0000%** | 13 / 100 | 4.4885 |
| mean | **17.3333%** | 26 / 160 | 5.0174 |

Baseline full EMA under the identical manifest completed 236 updates, reached
18.33% test, 11.00% OOD, and 14.67% mean, with final training loss 0.8967. The
faster/stronger dynamics gain 3.34 percentage points on test, 2.00 on OOD, and
2.66 on mean while retaining essentially the same throughput and slightly
lower training loss.

## Full E5 CPU gate

The full public E5 gate completed 933 optimizer updates. Final training loss was
2.78830.

| ordinary split | exact accuracy | correct / examples |
|---|---:|---:|
| test | 1.0000% | 12 / 1,200 |
| OOD | 1.1667% | 7 / 600 |
| mean | **1.0833%** | 19 / 1,800 |

The seen-modulus T=1 rung reached 0.5859% (3 / 512), while OOD-N T=1 reached 0%
(0 / 512). Neither profile certified T=1.

Baseline EMA reached 1.0417% mean, so fast EMA's 1.0833% is only one additional
exact example across the 1,800 ordinary evaluation rows. Its zero OOD-N T=1 is
also weaker evidence for the certification objective. The result is consistent
with a small tuning gain but is not a breakthrough, does not pass the local
learning gate, and does not justify an H100 run by itself.
