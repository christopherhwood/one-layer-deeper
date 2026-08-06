# Ripple-product EMA probability ensemble

This experiment isolates one question: does the frozen temporal teacher help
when it participates in the submitted endpoint distribution, rather than only
supplying a consistency loss?

## Intervention

The implementation is based on the tracked `ripple_product_ema` submission.
Its product-column representation, tied recurrent square transition, four
refinement phases, three-step training recurrence, outer-gradient truncation,
T=1-weighted endpoint and phase CE, EMA consistency KL, optimizer, wall-clock
schedules, EMA update, and all temperatures are preserved.

Student and EMA teacher each produce a categorical distribution for every
fixed-width output digit. The submitted distribution is

`p_mix = (1 - alpha) p_student + alpha p_teacher`.

During training, `alpha = 0.25 * consistency_scale`, using the existing linear
ramp over the first 20% of the wall-clock budget. The teacher endpoint is
detached, so the student retains a direct coefficient of at least 0.75 in the
submitted endpoint CE. The existing consistency KL compares the original,
unmixed student endpoint against the teacher endpoint. Student T=1 phase losses
are likewise unchanged. The teacher trajectory was already required for KL,
so training reuses those final registers rather than adding another model run.

At evaluation, student and teacher independently run from the untouched input
registers and use a fixed `alpha = 0.25`. Evaluation executes only the maximum
requested T in each batch and uses a batch size of 512; these affect evaluation
cost, not the learned computation.

## Rules audit

- Both ensemble members are persistent states of the single submitted model.
- The teacher is a causal EMA of the student, updated only by the evaluator's
  ordinary post-optimizer scheduler callback.
- Both trajectories receive only the supplied prompt and the same untouched
  parsed modulus/residue registers.
- The ensemble uses no labels, generated examples, arithmetic oracle,
  intermediate targets, target lengths, or evaluator feedback.
- Training returns one ordinary forward with a nonzero direct student gradient;
  the participant owns no backward pass or nested evaluator call.
- Evaluation returns one ordinary forward and does not mutate persistent state.

## Validation and results

Python compilation and source lint passed. The model and optimizer satisfy the
runner contract with 725,780 persistent model-state elements and 725,802
optimizer-state elements after the first update. A real variable-E5 training
batch at nonzero ensemble weight produced finite gradients on all 22 trainable
tensors. Evaluation did not mutate persistent state.

With `alpha` temporarily set to zero, logits were bitwise identical to the
tracked `ripple_product_ema` source (maximum absolute difference zero) in both
training and evaluation. The evaluation comparison used a mixed T=1/2/3 batch,
so it also verifies that the dynamic maximum-T loop preserves the old fixed-64
result when the teacher contribution is zero.

## Ten-second fixed-modulus contract smoke

Seed 74 completed 274 updates in 10.04 seconds. Final training loss was
0.720868.

| split | exact accuracy | correct / examples | endpoint CE |
|---|---:|---:|---:|
| test | 21.6667% | 13 / 60 | 4.6846 |
| OOD-T | 24.0000% | 24 / 100 | 2.6001 |
| split mean | **22.8333%** | 37 / 160 | 3.6423 |

The tracked EMA reference reported 18.33% test, 11.00% OOD-T, and 14.67%
split mean on the same manifest. The ensemble adds two test successes and 13
OOD-T successes, raising the mean by 8.17 percentage points. It completed 38
more wall-clock-scheduled updates, so this smoke supports the direction but is
not a perfectly update-matched ablation.

## Fifteen-second variable-E5 screen

The standard `/private/tmp/e5-variable-15s-cpu.json` screen completed 231
updates in 15.03 seconds. Final training loss was 2.923543.

| ordinary split | exact accuracy | correct / examples | endpoint CE |
|---|---:|---:|---:|
| test | 0.6667% | 8 / 1,200 | 2.2719 |
| OOD | 0.1667% | 1 / 600 | 2.2460 |
| split mean | **0.4167%** | 9 / 1,800 | 2.2589 |

Both ordinary splits completed. For context, the earlier short full-EMA
calibration reached 2/1,200 test examples before its old fixed-64 evaluation
timed out, so the ensemble's eight test successes are directionally better;
there is no complete update-matched baseline mean for this short screen.

The 7.5-second profiling allowance completed seen-modulus rungs through T=32
but not T=64 or OOD-N T=1:

| seen-N T | exact accuracy | correct / examples |
|---:|---:|---:|
| 1 | 0.3906% | 2 / 512 |
| 2 | 0.7812% | 4 / 512 |
| 4 | 0.5859% | 3 / 512 |
| 8 | 0.0000% | 0 / 512 |
| 16 | 0.5859% | 3 / 512 |
| 32 | 0.0000% | 0 / 512 |
| 64 | not completed | — |

No rung certified. The ensemble is a clean, rules-compliant improvement on the
fixed-modulus smoke and a small test-count improvement over the incomplete
short EMA calibration, but the variable-E5 result remains weak and unstable
across T. Per the experiment plan, no full 60-second or H100 gate has been run.
