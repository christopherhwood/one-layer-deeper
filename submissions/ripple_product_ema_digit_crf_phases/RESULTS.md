# Shared endpoint and phase digit CRF

This candidate starts from the surviving `ripple_product_ema_digit_crf` and
changes only where its observed-digit sequence objective is applied. The
product-aware reducer, factorized recurrent register, four refinement rounds,
outer recurrence, EMA teacher, final endpoint CE and CRF, optimizer, schedules,
and final Viterbi decoder are unchanged.

## Shared phase supervision

During training, each first-square refinement state is passed through the same
edge head and the same trainable start/end energies used by the final endpoint
CRF. On T=1 rows, every refinement predicts the evaluator-supplied final answer,
so its existing phase term becomes

```text
0.35 * mean_phase(phase endpoint CE + 0.2 * phase partial-suffix CRF NLL)
```

T>1 rows receive no phase CRF because a first-square refinement is not their
supplied endpoint. The phase chains use only the ten actual output digits; no
latent carry, quotient, process state, or free semantic alphabet is introduced.
Unobserved fixed-width leading slots are marginalized exactly as in the final
CRF.

This is not EMA phase consistency: the target is the real endpoint suffix and
the objective is globally normalized sequence likelihood, not agreement with a
teacher prediction. No new decoder parameters are added because all phases
share the final CRF head.

## Evaluation-compute refactor

The prior implementation applied the edge head after every outer square and
then selected the energies for each row's requested final step. This version
instead selects each row's final reducer state and raw unary logits first, then
applies the same row-local edge normalization and linear head once.

Selection commutes exactly with these row-local operations, so trained and
decoded energies are unchanged. At T=64 this removes 63 redundant edge-head
applications. Phase edges are training-only. Evaluation remains one forward
with the same fixed-width Viterbi decoder and no target-length input.

## Rules rationale

- Every supervised target is an evaluator-supplied endpoint on a T=1 row.
- There are no generated examples, intermediate arithmetic labels, latent
  arithmetic states, or hard-coded transition constraints.
- CRF dynamic programming and Viterbi are ordinary tensor computation inside
  the single learned model/loss path.
- All edge, start, and end energies are random-initialized trainable weights.
- Exact endpoint and phase CE anchors remain active.

## Pre-screen validation

The following checks pass:

- Python compilation
- `benchmark.validation.lint_submission_source`
- `git diff --check`
- one full real-E5 training batch through forward, combined loss, and backward

The finite batch check produced loss 5.102822, endpoint exact accuracy 3.125%,
27 parameter tensors with nonzero gradients, and no non-finite gradients.

For the evaluation refactor, the final-only CRF and phase-CRF models were given
identical state dictionaries and the same real-E5 inputs. Their complete T<=64
evaluation logits were bitwise equal (`max_abs_diff=0`). This verifies that
selecting the final reducer state before the row-local edge head preserves the
existing decoder exactly.

A direct real-E5 T=64 evaluation check processed all 512 rung rows together in
2.644 seconds on CPU with finite logits. The submission therefore uses
`eval_batch_size=512`.

Evaluation now executes `max(T)` outer iterations for the current batch rather
than always executing 64. Training remains fixed at three iterations. An
explicit 64-loop reconstruction and the bounded forward produced bitwise equal
complete logits (`max_abs_diff=0`) on real-E5 mixed T=1/2/3 inputs, OOD T=6,
and depth T=64. The post-change training check was finite with loss 5.624241.

## Ten-second contract smoke

The authorized contract smoke completed 235 optimizer updates. Final combined
training loss was 1.00362.

| split | exact accuracy | correct / examples | hard-path CE |
|---|---:|---:|---:|
| test | 30.0000% | 18 / 60 | 7.2374 |
| OOD | 36.0000% | 36 / 100 | 5.9754 |
| mean | **33.0000%** | 54 / 160 | 6.6064 |

The final-only digit CRF reached 23.33% test, 16.00% OOD, and 19.67% mean under
the same smoke manifest. Shared phase CRF therefore adds 6.67 percentage points
on test, 20.00 on OOD, and 13.33 on the split mean with identical update count.
As before, hard Viterbi logits make evaluation CE unsuitable for comparison to
factorized decoders.

## Fifteen-second variable-E5 screen

The phase CRF completed 200 logged updates in the 15-second training allowance.
It answered 10 / 1,200 test rows exactly (0.8333%), with hard-path CE 13.5586.
The 7.5-second evaluation allowance expired before OOD completed, so there is no
valid split mean or `RESULT_JSON`.

| model | E5 test exact | correct / examples | OOD |
|---|---:|---:|---|
| full EMA calibration | 0.1667% | 2 / 1,200 | timed out |
| final-only digit CRF | 0.6667% | 8 / 1,200 | timed out |
| shared phase digit CRF | **0.8333%** | **10 / 1,200** | timed out |

The phase objective adds two exact E5 test examples over the final-only CRF and
eight over EMA. This is directionally consistent with the 33% smoke result, but
the E5 increment is modest and OOD remains unknown.

The short screen above preceded the dynamic-loop refactor. The validated bound
removes its redundant low-T evaluation work and should permit the full gate to
measure OOD-N rather than timing out. No training or learned semantics changed.

## Full 60-second E5 CPU gate

The unchanged-training phase CRF completed 823 optimizer updates in 60.07
seconds. Final combined training loss was 4.49045. The dynamic evaluation path
completed test, OOD, every seen-N rung, and every OOD-N rung in 11.15 seconds.

| ordinary split | exact accuracy | correct / examples | hard-path CE |
|---|---:|---:|---:|
| test | **1.2500%** | 15 / 1,200 | 13.4865 |
| OOD | 1.3333% | 8 / 600 | 13.6054 |
| split mean | **1.2917%** | 23 / 1,800 | 13.5459 |

| T | seen-N exact | seen correct | OOD-N exact | OOD-N correct |
|---:|---:|---:|---:|---:|
| 1 | 1.5625% | 8 / 512 | 0.0000% | 0 / 512 |
| 2 | 0.9766% | 5 / 512 | 0.1953% | 1 / 512 |
| 4 | 1.5625% | 8 / 512 | 0.1953% | 1 / 512 |
| 8 | 1.5625% | 8 / 512 | 0.1953% | 1 / 512 |
| 16 | 0.9766% | 5 / 512 | 0.0000% | 0 / 512 |
| 32 | 1.7578% | 9 / 512 | 0.1953% | 1 / 512 |
| 64 | 1.1719% | 6 / 512 | 0.0000% | 0 / 512 |

Neither profile certified T=1.

## Comparison and decision

| model | updates | test | OOD | split mean | seen T=1 | OOD-N T=1 |
|---|---:|---:|---:|---:|---:|---:|
| full EMA | **937** | 0.9167% | 1.1667% | 1.0417% | 0.9766% | **0.3906%** |
| final-only digit CRF | 869 | 1.0000% | **2.0000%** | **1.5000%** | **1.7578%** | not completed |
| shared phase digit CRF | 823 | **1.2500%** | 1.3333% | 1.2917% | 1.5625% | 0.0000% |

Shared phase structure adds three test successes over the final-only CRF but
loses four OOD successes and one seen-T=1 success. Its split mean is 13.9% lower,
and four training-time phase dynamic programs reduce optimizer throughput by
5.3%. The phase model still improves split mean over EMA, but eliminates EMA's
OOD-N T=1 successes.

The 33% smoke and two-example short-screen gains therefore do not transfer to
the complete gate. Globally normalized final decoding remains the surviving
mechanism; applying the same objective at every refinement overconstrains or
misdirects early states that still need freedom to compute. Reject phase CRF and
retain the final-only CRF as the best local model. The dynamic evaluation bound
is independently validated and valuable, but it does not rescue the phase
learning signal or meet the 5% / 3% H100 gate.
