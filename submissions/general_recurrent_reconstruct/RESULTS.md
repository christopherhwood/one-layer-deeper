# Same-prompt reconstruction results

This experiment leaves the generic recurrent transition and endpoint losses
unchanged. During the first outer transition only, each refinement's scratch
slots additionally reconstruct both untouched input digit registers: supplied
modulus `N` and supplied value `x`. Two learned ten-way heads receive the same
normalized scratch state. There is no masking, changed input, generated target,
arithmetic identity, or process label.

The representation-loss weight starts at 0.05, warms to 0.30 over the first 5%
of the wall-clock budget, anneals to 0.05 by 60%, and remains at 0.05. Genuine
endpoint cross entropy remains active throughout. Reconstruction heads are not
executed during evaluation or on later outer transitions.

## Validation

The following checks passed:

- `python -m py_compile` using the repository virtual environment
- `benchmark.validation.lint_submission_source`
- `git diff --check`
- official runner contract, optimizer/model-state validation, finite forward
  and backward, ordinary evaluation, and every depth rung

The model has 97,662 state elements for the small smoke task and 98,046 for E5.

## Ten-second contract smoke

Command:

```text
.venv/bin/python -m benchmark.runner \
  --manifest benchmark/manifests/local_cpu_10s.json \
  --submission-file submissions/general_recurrent_reconstruct/submission.py
```

Seed 74 completed 268 optimizer updates. Final training loss was 2.9668.

| split | exact accuracy | loss |
|---|---:|---:|
| test | 1.6667% | 4.1110 |
| OOD | 7.0000% | 2.5426 |
| mean | **4.3333%** | 3.3268 |

This is a contract and finite-learning check on a substantially smaller
generated problem, not evidence of E5 arithmetic generalization.

## Full E5 CPU gate

The same temporary manifest used by the consistency experiment,
`/tmp/general_consistency_e5_60s.json`, points at the complete public E5 corpus
with batch size 32, seed 74, 60 seconds of CPU training, and 30 seconds of
evaluation.

The run completed 712 optimizer updates in 60.06 seconds. Logged batch exact
accuracy was 3.125% at step 1 and 0% at steps 250 and 500. Final training loss
was 3.2193.

| ordinary split | exact accuracy | correct / examples | loss |
|---|---:|---:|---:|
| test | 0.3333% | 4 / 1,200 | 2.1826 |
| OOD | 0.5000% | 3 / 600 | 2.1710 |
| mean | **0.4167%** | 7 / 1,800 | 2.1768 |

| T | seen-N exact | OOD-N exact |
|---:|---:|---:|
| 1 | 0.5859% | 0.3906% |
| 2 | 0.0000% | 0.0000% |
| 4 | 0.1953% | 0.0000% |
| 8 | 0.1953% | 0.0000% |
| 16 | 0.0000% | 0.0000% |
| 32 | 0.1953% | 0.0000% |
| 64 | 0.1953% | 0.0000% |

Neither profile certified T=1. Test, OOD, and all fourteen depth rungs completed
in 26.42 seconds.

## Decision

This path fails the pre-H100 learning gate. It remains far below 5% held-out T1
and 3% OOD-N T1, and its 0.4167% E5 mean is below the already failed consistency
variant. Reconstructing visible inputs gives the recurrent workspace an easy,
well-conditioned task, but that task does not distinguish an arithmetic state
from a state that merely copies prompt digits. It therefore does not resolve
the endpoint credit-assignment or semantic-identification problem and does not
justify an H100 run in this form.
