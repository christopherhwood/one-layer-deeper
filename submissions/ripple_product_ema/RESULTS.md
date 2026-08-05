# Product-aware ripple with temporal EMA teacher

This hybrid preserves the tracked `ripple_product_h100` student architecture
and H100 contract. Its product columns, tied bidirectional scan, refinement
phases, T recurrence, truncated outer gradients, T=1 phase endpoint losses,
annealing schedule, batch sizes, and evaluation path are unchanged.

The only additional learning signal is a temporal teacher:

- a frozen copy of the square transition receives the same untouched modulus
  and residue registers in the same training forward;
- the submitted endpoint remains the student output with a direct CE gradient;
- softened KL makes that endpoint agree with the EMA teacher endpoint;
- consistency ramps over the first 20% of wall-clock training;
- the evaluator-called scheduler updates teacher parameters after AdamW steps;
- the teacher is skipped entirely during evaluation.

There are no process labels, teacher-generated hard labels, arithmetic state
targets, masking, augmentation, or nested model invocations.

## Local results

| run | updates | final train-batch exact | test exact | OOD exact | mean exact |
|---|---:|---:|---:|---:|---:|
| generated 10-second CPU smoke | 236 | 50.00% | 18.33% | 11.00% | 14.67% |
| full E5, 60 CPU seconds | 937 | 0.00% | 0.9167% | 1.1667% | 1.0417% |

Training loss fell from 3.1403 at step 1 to 0.8967 at the final update. The
model used 725,780 persistent state elements, including the frozen teacher.
Compilation, source validation, optimizer/state validation, finite backward,
EMA callback execution, and final evaluation all completed normally.

The full E5 endpoint loss was 2.86594 on the final training batch. Seen-N T=1
accuracy was 0.9766% (5/512), OOD-N T=1 accuracy was 0.3906% (2/512), and no
depth rung certified.

## Identical-gate baseline comparison

| model | updates | test exact | OOD exact | mean exact | seen T=1 | OOD-N T=1 |
|---|---:|---:|---:|---:|---:|---:|
| tracked ripple baseline | 1,259 | 1.0833% | 0.3333% | 0.7083% | 0.1953% | 0.0000% |
| ripple + EMA | 937 | 0.9167% | **1.1667%** | **1.0417%** | **0.9766%** | **0.3906%** |

EMA lowers ordinary test accuracy slightly, but raises OOD exact accuracy by
3.5x and mean exact accuracy by about 47% despite completing 26% fewer updates.
It also moves seen T=1 from 1/512 to 5/512 and OOD-N T=1 from 0/512 to 2/512.
That is a material directional gain over the same structured student without
the teacher, and stronger evidence than the tiny generated smoke alone.

The absolute result still fails the predeclared H100 gate: seen T=1 is below
1% versus the required 5%, OOD-N T=1 is below 0.4% versus the required 3%,
final training exact is zero, and nothing certifies. EMA is worth retaining as
a useful learning-signal result, but this version should not consume an H100
attempt without another locally demonstrated improvement.
