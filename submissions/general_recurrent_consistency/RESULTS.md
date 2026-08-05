# Endpoint-anchored consistency results

This experiment changes only the learning signal of
`general_recurrent_deep`. Every recurrent phase retains genuine endpoint cross
entropy. In addition, each earlier phase minimizes KL divergence to the final
phase distribution, with that final distribution detached. The consistency
temperature is 1.5 and its loss weight is 0.2. No arithmetic intermediate,
generated target, or process label is used.

## Validation

The following checks passed:

- `python -m py_compile` using the repository virtual environment
- `benchmark.validation.lint_submission_source`
- `git diff --check`
- official runner contract, optimizer/model-state validation, forward,
  backward, training, ordinary evaluation, and every depth rung

The model has 95,626 state elements for the small smoke task and 96,010 for E5.

## Ten-second contract smoke

Command:

```text
.venv/bin/python -m benchmark.runner \
  --manifest benchmark/manifests/local_cpu_10s.json \
  --submission-file submissions/general_recurrent_consistency/submission.py
```

Seed 74 completed 413 optimizer updates. Final training loss was 2.6168.

| split | exact accuracy | loss |
|---|---:|---:|
| test | 1.6667% | 5.0178 |
| OOD | 3.0000% | 3.8262 |
| mean | 2.3333% | 4.4220 |

This generated smoke problem is much smaller than E5 and is only a contract
and finite-learning check. Under the same manifest, the version without the
consistency term previously reached 4.3333% mean, so consistency did not improve
this short-run diagnostic.

## Full E5 CPU gate

The temporary manifest `/tmp/general_consistency_e5_60s.json` points at the
complete public E5 corpus, uses batch size 32, seed 74, 60 seconds of CPU
training, and the evaluator's corresponding 30-second evaluation allowance.

The run completed 1,073 updates in 60.05 seconds. Logged batch exact accuracy
was 3.125% at step 1, 0% at step 250, 3.125% at step 500, and 0% at steps 750
and 1,000. Final training loss was 3.0174.

| ordinary split | exact accuracy | correct / examples | loss |
|---|---:|---:|---:|
| test | 1.0000% | 12 / 1,200 | 2.2080 |
| OOD | 0.1667% | 1 / 600 | 2.1930 |
| mean | **0.5833%** | 13 / 1,800 | 2.2005 |

| T | seen-N exact | OOD-N exact |
|---:|---:|---:|
| 1 | 0.3906% | 0.1953% |
| 2 | 0.7812% | 0.1953% |
| 4 | 0.3906% | 0.0000% |
| 8 | 0.5859% | 0.1953% |
| 16 | 0.0000% | 0.1953% |
| 32 | 0.5859% | 0.1953% |
| 64 | 0.1953% | 0.0000% |

Neither depth profile certified T=1. Evaluation of test, OOD, and all fourteen
depth rungs completed in 29.39 seconds.

## Decision

This path fails the pre-H100 learning gate. It does not reach the required 5%
held-out T=1 or 3% OOD-N T=1 accuracy, and the short smoke comparison moves in
the wrong direction. Detached late-phase self-distillation makes recurrent
predictions more mutually consistent, but supplies no new information capable
of identifying modular reduction. The result does not justify an H100 run in
this form.
