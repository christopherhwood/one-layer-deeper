# Ripple-product EMA digit CRF: batch-128 ablation

This is a controlled throughput/statistical-efficiency experiment based on the
optimized final-only first-order digit CRF. Exactly two training settings
change:

- training batch size increases from 32 to 128;
- base learning rate increases from `1.5e-3` to `3.0e-3`, the square-root batch
  scaling rule for a fourfold batch increase.

The architecture, initialization, product and recurrent computation, three
training outer steps, four refinement phases, endpoint and T=1 weighting,
first-square phase CE, EMA teacher and KL, final-only partial-suffix CRF NLL,
CRF weight, optimizer family and betas, weight decay, wall-clock schedules,
temperature schedule, EMA schedule, gradient handling, dynamic evaluation
bound, Viterbi decoder, and evaluation batch size of 512 are otherwise copied
unchanged.

The fixed-modulus smoke dataset has 240 training rows. With evaluator-owned
`drop_last=True`, batch 128 yields one valid full batch per epoch; the remaining
112 rows are reshuffled before the next epoch rather than emitted as a partial
batch. Variable E5 has 4,800 rows and 37 full batches per epoch.

## Validation

Python compilation, source lint, submission/model/optimizer validation, and a
full 128-row variable-E5 forward/backward passed. The model has 777,476
persistent state elements, and all 27 trainable tensors received finite nonzero
gradients. A direct source diff against the optimized batch-32 CRF contains
exactly the intended `BASE_LR` and `batch_size` changes.

## Ten-second fixed-modulus smoke

Seed 74 completed 98 updates in 10.014 seconds. Final combined training loss
was 0.988949.

| split | exact accuracy | correct / examples | hard-path CE |
|---|---:|---:|---:|
| test | 23.3333% | 14 / 60 | 8.3712 |
| OOD-T | 20.0000% | 20 / 100 | 9.7489 |
| split mean | **21.6667%** | 34 / 160 | 9.0601 |

| smoke training | batch 32 reference | batch 128 | change |
|---|---:|---:|---:|
| updates | 235 | 98 | -58.3% |
| examples processed | 7,520 | 12,544 | +66.8% |
| approximate updates/s | 23.5 | 9.79 | -58.4% |
| approximate examples/s | 752 | 1,253 | +66.6% |
| split-mean exact | 19.6667% | **21.6667%** | +2.00 pp |

The large batch preserves the reference's 14 test successes and adds four
OOD-T successes despite making far fewer parameter updates. This is the
intended trade: about 2.4x fewer optimizer steps per second, but 1.67x more
examples processed per second. The reference timing is reported as ten seconds
rather than millisecond precision, so its rate columns are approximate.

## Fifteen-second variable-E5 screen

The standard `/private/tmp/e5-variable-15s-cpu.json` screen completed 88
updates in 15.060 seconds. That is 11,264 examples, 5.84 updates/s, and 747.9
examples/s. Final combined training loss was 4.176396.

| ordinary split | exact accuracy | correct / examples | hard-path CE |
|---|---:|---:|---:|
| test | 1.0833% | 13 / 1,200 | 13.4910 |
| OOD | 0.6667% | 4 / 600 | 13.4081 |
| split mean | **0.8750%** | 17 / 1,800 | 13.4495 |

The earlier batch-32 CRF screen completed at least 200 updates and reached
8/1,200 test examples (0.6667%) before its old fixed-64 evaluator timed out.
Batch 128 adds five test successes. Its optimized evaluator also supplies the
previously missing OOD result, so there is no valid batch-32 short-screen mean
for a direct comparison.

| T | seen-N exact | seen correct | OOD-N exact | OOD-N correct |
|---:|---:|---:|---:|---:|
| 1 | 1.1719% | 6 / 512 | 0.3906% | 2 / 512 |
| 2 | 1.5625% | 8 / 512 | 0.0000% | 0 / 512 |
| 4 | 1.3672% | 7 / 512 | 0.1953% | 1 / 512 |
| 8 | 1.1719% | 6 / 512 | 0.0000% | 0 / 512 |
| 16 | 0.7812% | 4 / 512 | 0.1953% | 1 / 512 |
| 32 | 0.9766% | 5 / 512 | not completed | — |
| 64 | 1.5625% | 8 / 512 | not completed | — |

No rung certified. Evaluation completed every seen-N rung and OOD-N through
T=16; the final profiling call brought elapsed evaluation to 8.08 seconds
against the 7.5-second allowance.

The controlled larger batch is positive on both local screens: it improves
smoke mean while processing substantially more examples per second, and raises
the variable-E5 test count from 8 to 13. The absolute variable-E5 score remains
low and OOD-N T=1 is unchanged at 2/512, so this is evidence to consider a full
CPU gate, not evidence for an H100 run by itself. At that screening stage, no
full or H100 run had been made.

## Full 60-second variable-E5 gate

The unchanged candidate completed 356 updates in 60.097 seconds. The submission
source SHA-256 was
`3d70c356ebb2e69038a4798fc9ca1d4f69edf25117607c30c3962ffa48513990`
both before and after the run. Final combined training loss was 3.781442.

The run presented 45,568 training examples, or 758.2 examples/s, while making
5.92 optimizer updates/s.

| ordinary split | exact accuracy | correct / examples | hard-path CE |
|---|---:|---:|---:|
| test | 0.8333% | 10 / 1,200 | 13.6441 |
| OOD | 0.8333% | 5 / 600 | 13.6502 |
| split mean | **0.8333%** | 15 / 1,800 | 13.6472 |

Both complete depth ladders fit in 10.584 evaluation seconds:

| T | seen-N exact | seen correct | OOD-N exact | OOD-N correct |
|---:|---:|---:|---:|---:|
| 1 | 1.1719% | 6 / 512 | 0.0000% | 0 / 512 |
| 2 | 0.9766% | 5 / 512 | 0.0000% | 0 / 512 |
| 4 | 0.5859% | 3 / 512 | 0.3906% | 2 / 512 |
| 8 | 1.3672% | 7 / 512 | 0.3906% | 2 / 512 |
| 16 | 0.3906% | 2 / 512 | 0.1953% | 1 / 512 |
| 32 | 1.1719% | 6 / 512 | 0.0000% | 0 / 512 |
| 64 | 0.7812% | 4 / 512 | 0.1953% | 1 / 512 |

No rung certified.

## Full-gate comparison and decision

| CRF run | batch | updates | example presentations | final loss | test | OOD | mean | seen T=1 | OOD-N T=1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| batch-32 best | 32 | 869 | 27,808 | **3.78019** | 1.0000% | **2.0000%** | **1.5000%** | **1.7578%** | not completed |
| batch-32 optimized replication | 32 | 869 | 27,808 | 3.79022 | **1.0833%** | 1.3333% | 1.2083% | 1.5625% | **0.3906%** |
| batch-128 | 128 | 356 | **45,568** | 3.78144 | 0.8333% | 0.8333% | 0.8333% | 1.1719% | 0.0000% |

Relative to the optimized batch-32 replication, batch 128 makes 59.0% fewer
updates but presents 63.9% more examples. Its example throughput rises from
approximately 463.0/s to 758.2/s. That hardware-efficiency gain does not become
statistical efficiency: split mean falls 31.0%, seen-N T=1 loses two successes,
and OOD-N T=1 loses both successes. It is also 44.4% below the best batch-32
mean.

The almost identical final losses show that the fourfold larger batch and
square-root LR scaling optimize the training objective successfully. The
generalization loss is therefore better explained by many fewer parameter
updates and reduced gradient diversity than by a failure to lower the scalar
objective. The positive ten- and fifteen-second screens do not survive the full
gate. Reject batch 128 as the H100 configuration; retain optimized batch 32 as
the stronger full-budget candidate.
