# Anchored error-correcting ripple

The proposal is snapped to one-hot digit states, 15% of T=1 state positions are
randomly replaced during training, and the same tied ripple transition receives
the original operand and modulus again to correct the state. Evaluation runs an
additional clean correction and selects the most confident recurrent output.

## Fixed-modulus gate

| updates | exact | token | last token | mean wrong |
|---:|---:|---:|---:|---:|
| 361 | 2.83% | 20.53% | 8.50% | 1.6833 |

The observable-state parent reached 6.50% exact, 29.03% token, and 14.83% last
token on this gate. The new model therefore fails the mandatory small-task
screen and does not proceed to variable-N or H100 evaluation.

The result also isolates a limitation of self-correction under endpoint-only
training: corruption says that a state is unreliable but supplies no positive
information about the correct intermediate state. Published modular-reasoning
results pair corruption with explicit intermediate algorithmic supervision;
that ingredient is unavailable under this benchmark's data rules.
