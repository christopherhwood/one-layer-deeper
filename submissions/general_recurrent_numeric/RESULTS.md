# Generic numeric-interface design and smoke result

This experiment changes the generic deep recurrent model's representation and
endpoint loss without encoding multiplication, modular reduction, comparison,
or any other task algorithm.

## Numeric interface

Each hard or soft decimal token is represented by the sum of:

- its ordinary learned ten-way token projection; and
- a learned projection of six fixed ordered coordinates: normalized scalar,
  squared scalar, and sine/cosine pairs at frequencies one and two.

For a soft digit distribution, the model receives the expectation of those
coordinates. Consequently adjacent digits begin with related geometry while all
ways of using that geometry remain learned.

The exact endpoint cross-entropy and deep endpoint cross-entropies are retained.
An additional endpoint-only ordinal loss computes the one-dimensional
Wasserstein distance between predicted and observed digit distributions using
their cumulative distributions. It has weight 0.2 and is applied to the final
prediction and every recurrent phase. It derives solely from the supplied final
answer and introduces no intermediate or generated label.

## Validation

The following checks passed:

- `python -m py_compile` using the repository virtual environment
- `benchmark.validation.lint_submission_source`
- `git diff --check`
- official runner model/optimizer validation, finite forward and backward,
  training, and held-out evaluation

The smoke-task model has 96,202 state elements.

## Ten-second contract smoke

Command:

```text
.venv/bin/python -m benchmark.runner \
  --manifest benchmark/manifests/local_cpu_10s.json \
  --submission-file submissions/general_recurrent_numeric/submission.py
```

Seed 74 completed 330 updates. Final training loss was 2.8638.

| split | exact accuracy | correct / examples | loss |
|---|---:|---:|---:|
| test | **5.0000%** | 3 / 60 | 4.1878 |
| OOD | **5.0000%** | 5 / 100 | 3.4371 |
| mean | **5.0000%** | 8 / 160 | 3.8124 |

For context under the same tiny manifest, the plain deeply supervised model
reached 4.3333% mean, endpoint consistency reached 2.3333%, and prompt
reconstruction reached 4.3333%. Numeric geometry is therefore the strongest
short smoke result in this generic wave, but the difference represents only one
additional correct example over a 4.3333% mean and is not statistically strong.

## Full E5 CPU gate

The subsequent full public E5 gate completed 1,038 optimizer updates. Final
training loss was 3.35997.

| ordinary split | exact accuracy | correct / examples |
|---|---:|---:|
| test | 0.9167% | 11 / 1,200 |
| OOD | 0.5000% | 3 / 600 |
| mean | **0.7083%** | 14 / 1,800 |

The seen-modulus T=1 rung reached 0.5859% (3 / 512). The unseen-modulus T=1
rung reached 0% (0 / 512), and neither depth profile certified T=1.

This improves the otherwise identical generic deep model's E5 mean from 0.375%
to 0.7083%, so ordered digit geometry is a real directional improvement within
this architecture. In absolute terms it is still only several additional exact
examples and does not survive on unseen-modulus T=1. It fails the required 5%
held-out and 3% OOD-N T=1 learning gate, so it does not justify an H100 attempt
in its current form.
