# Ripple-product EMA with an observed-digit output CRF

This experiment keeps the tracked full-EMA model's arithmetic representation
and recurrent computation unchanged. Product columns, four refinement scans,
factorized soft register, outer recurrence, endpoint CE, T=1 phase CE, EMA
teacher endpoint KL, optimizer, schedules, and all intermediate registers are
the baseline implementation.

## Output-only structured decoder

The final active student reducer state supplies:

- its existing ten-way unary digit scores, in fixed-width MSD-first order;
- a shared learned linear head from each pair of adjacent reducer states to a
  10x10 digit-transition energy; and
- randomly initialized trainable ten-way start and end energies.

These define a first-order globally normalized chain over actual decimal output
digits. During training, a 0.2-weighted CRF NLL is added to all unchanged
baseline losses. Evaluator labels constrain only their observed suffix slots;
the unobserved fixed-width leading prefix is marginalized in both the full and
constrained partition functions. No leading-zero target is invented.

The forward pass is label-independent. The CRF does not alter the recurrent
register or the soft distributions passed between outer squares. At evaluation,
Viterbi decodes the learned fixed-width energy and emits the selected path as
digit logits. It never receives or infers a target length; the evaluator simply
reads its ordinary requested suffix positions.

## Distinction from the failed latent CRF

The earlier diagnostic marginalized unconstrained hidden path states. Those
states had no fixed meaning and became an arbitrary lookup code, reaching high
training fit with chance held-out accuracy. This decoder has no latent carry,
borrow, quotient, or free-state alphabet: every chain state is one of the ten
actual output digits. The only unobserved variables are fixed-width prefix digit
values omitted by ordinary output formatting. Thus this tests global endpoint
normalization rather than latent arithmetic discovery.

All new energies are randomly initialized and trainable. The transition head
contains no arithmetic constraint or hard-coded value relation.

## Rules audit

- Training uses only supplied endpoint labels and their valid mask.
- The model generates no examples, intermediate results, or process labels.
- Dynamic programming and Viterbi are ordinary bounded tensor operations inside
  the learned model/loss path; there are no nested calls or participant-owned
  backward passes.
- The submitted evaluation result comes from one ordinary model forward.
- Exact CE remains the direct student anchor; CRF NLL is only an auxiliary
  structured-output objective.

## Validation and ten-second smoke

Python compilation, source lint, `git diff --check`, model/optimizer-state
validation, finite backward, EMA updates, training, and Viterbi evaluation all
passed.

The smoke-task model has 777,476 persistent state elements, including the EMA
teacher. Seed 74 completed 235 updates in ten seconds; final training loss was
1.0007.

| split | exact accuracy | correct / examples | evaluation CE |
|---|---:|---:|---:|
| test | 23.3333% | 14 / 60 | 7.9933 |
| OOD | 16.0000% | 16 / 100 | 9.2961 |
| mean | **19.6667%** | 30 / 160 | 8.6447 |

Baseline full EMA reached 18.33% test, 11.00% OOD, and 14.67% mean under the
same smoke manifest. The CRF gains 5.00 mean percentage points. Its evaluation
CE is not comparable: Viterbi deliberately emits a sharp selected path, so a
wrong digit receives a large log loss while exact accuracy remains comparable.

## Fifteen-second variable-E5 screen

The unchanged EMA baseline and then the unchanged CRF were run sequentially on
`/private/tmp/e5-variable-15s-cpu.json`. Both passed 200 logged optimizer updates
but exhausted the standard 7.5-second evaluation allowance after completing
test and before completing OOD. Consequently neither run has a valid ordinary
mean or `RESULT_JSON` from this screen.

| model | test exact | correct / examples | test CE | OOD |
|---|---:|---:|---:|---|
| full EMA calibration | 0.1667% | 2 / 1,200 | 2.2945 | timed out |
| EMA + digit CRF | **0.6667%** | **8 / 1,200** | 13.4910 | timed out |

The CRF adds six exact test examples in this short variable-modulus screen. The
hard-path CE remains incomparable, and missing OOD makes this directional rather
than a gate pass. Unlike many fixed-modulus smoke gains, this movement appears
on real E5 inputs; a complete identical-budget E5 gate is required before any
H100 decision.

## Full 60-second E5 CPU gate

The unchanged CRF submission completed 869 optimizer updates in 60.03 seconds.
Final combined training loss was 3.78019.

| ordinary split | exact accuracy | correct / examples | hard-path CE |
|---|---:|---:|---:|
| test | 1.0000% | 12 / 1,200 | 13.6396 |
| OOD | **2.0000%** | 12 / 600 | 13.2466 |
| split mean | **1.5000%** | 24 / 1,800 | 13.4431 |

The evaluator's score averages split accuracies, so the reported 1.5000% is the
mean of 1.0000% test and 2.0000% OOD. The count-weighted accuracy is 24 / 1,800,
or 1.3333%. Viterbi's deliberately sharp path logits again make CE unsuitable
for comparison with the factorized baseline.

| seen-N T | exact accuracy | correct / examples |
|---:|---:|---:|
| 1 | **1.7578%** | 9 / 512 |
| 2 | 0.1953% | 1 / 512 |
| 4 | 1.5625% | 8 / 512 |
| 8 | 1.7578% | 9 / 512 |
| 16 | 1.1719% | 6 / 512 |
| 32 | 1.3672% | 7 / 512 |
| 64 | 1.5625% | 8 / 512 |

No seen-N rung certified because T=1 was not exact. The 30-second CPU evaluation
allowance completed every seen-N rung but expired before OOD-N T=1 completed;
OOD-N accuracy is therefore unknown, not zero.

## Identical-gate comparison with full EMA

| model | updates | test | OOD | split mean | seen T=1 | OOD-N T=1 |
|---|---:|---:|---:|---:|---:|---:|
| full EMA | **937** | 0.9167% | 1.1667% | 1.0417% | 0.9766% | 0.3906% |
| EMA + digit CRF | 869 | **1.0000%** | **2.0000%** | **1.5000%** | **1.7578%** | not completed |

The CRF completes 68 fewer updates, a 7.3% throughput reduction, while adding
one exact test row, five exact OOD rows, and four seen-T=1 rows. Its split-mean
score improves 44% relative to EMA. Persistent state grows from 725,780 to
777,476 elements, including both student and frozen EMA teacher.

This is the first auxiliary mechanism in this series whose improvement survives
the fixed-modulus smoke, short variable-E5 screen, and full E5 training gate.
Global endpoint normalization is therefore a real directional advance over
factorized digit CE. It is not yet an H100 gate pass: seen T=1 remains below the
required 5%, OOD-N T=1 was not measured, and no depth certifies. The next step
should preserve this output structure while reducing evaluation overhead and
obtaining a complete OOD-N measurement, rather than reverting to scalar loss or
EMA schedule tuning.

## Evaluation-only refactor and full-gate replication

Evaluation now executes only the largest requested `T` in each batch instead
of unconditionally executing all 64 outer squares, and its batch size is 512
instead of 256. Training steps, losses, recurrence, decoding, optimizer, and
schedules are unchanged. Direct comparisons against an explicit 64-step
evaluation produced bitwise-identical logits (maximum absolute difference zero)
for a mixed T=1/2/3 test batch, the ordinary OOD T=6 batch, and depth T=64.
Compilation, source lint, `git diff --check`, and a finite real-E5 backward pass
also passed.

The same seed-74 60-second E5 gate completed 869 optimizer updates in 60.06
seconds, exactly matching the first run's update count. Final combined training
loss was 3.79022. The ordinary score was:

| ordinary split | exact accuracy | correct / examples | hard-path CE |
|---|---:|---:|---:|
| test | 1.0833% | 13 / 1,200 | 13.5991 |
| OOD | 1.3333% | 8 / 600 | 13.3991 |
| split mean | **1.2083%** | 21 / 1,800 | 13.4991 |

The optimized evaluator completed both full ladders in 11.25 seconds, well
inside the 30-second allowance:

| T | seen-N exact | seen correct | OOD-N exact | OOD-N correct |
|---:|---:|---:|---:|---:|
| 1 | 1.5625% | 8 / 512 | 0.3906% | 2 / 512 |
| 2 | 0.3906% | 2 / 512 | 0.1953% | 1 / 512 |
| 4 | 1.7578% | 9 / 512 | 0.3906% | 2 / 512 |
| 8 | 0.9766% | 5 / 512 | 0.1953% | 1 / 512 |
| 16 | 1.5625% | 8 / 512 | 0.1953% | 1 / 512 |
| 32 | 0.7812% | 4 / 512 | 0.0000% | 0 / 512 |
| 64 | 1.9531% | 10 / 512 | 0.1953% | 1 / 512 |

No rung certified. The first and replication split means differ by 0.2917
percentage points (three test/OOD examples under the evaluator's split-mean
weighting) despite the same seed and update count. This submission uses a
wall-clock-dependent learning-rate and temperature schedule, so exact score
reproduction is timing-sensitive; the variation does not come from the new
evaluation path, whose outputs are bitwise equivalent. The complete OOD-N
measurement resolves the earlier unknown: T=1 is above chance but far below
the 5% H100 gate, and the gains do not form a stable depth-generalizing trend.

## Follow-up optimization and learning-signal screen

A later controlled wave used the same 15-second variable-E5 manifest, seed 74,
batch size 32, and 512-row evaluation batches. The retained AdamW source was
restored after every temporary intervention. The matched short-run control
reached 1.0833% test, 1.1667% OOD, 1.1250% split mean, 1.1719% seen-N T=1, and
0.1953% OOD-N T=1.

| intervention | test | OOD | mean | seen T=1 | OOD-N T=1 |
|---|---:|---:|---:|---:|---:|
| retained AdamW control | 1.0833% | 1.1667% | **1.1250%** | 1.1719% | 0.1953% |
| full outer-step backpropagation | 1.0000% | 0.8333% | 0.9167% | 1.1719% | 0.1953% |
| terminal-square refinement supervision | 0.8333% | 0.5000% | 0.6667% | 1.3672% | 0.1953% |
| learned refinement-step identity | 0.8333% | 1.0000% | 0.9167% | 0.5859% | 0.1953% |
| eight refinements, width 96 | 0.8333% | 0.8333% | 0.8333% | 0.1953% | not completed |
| 25% worst-digit endpoint loss | 1.0000% | 0.6667% | 0.8333% | **1.7578%** | 0.0000% |
| AdamW at 2x peak LR | 0.8333% | 1.0000% | 0.9167% | 0.9766% | 0.0000% |
| schedule-free AdamW, LR 0.005 | **1.1667%** | 0.6667% | 0.9167% | 1.1719% | 0.1953% |
| schedule-free AdamW, LR 0.0025 | 0.5833% | 0.8333% | 0.7083% | 1.3672% | **0.3906%** |

Eight refinements at width 128 also failed the prerequisite smoke comparison:
18.50% fixed-N mean after 145 updates versus 19.67% for the retained four-pass
control. The width-96 trade recovered 20.67% smoke mean but did not transfer.

The broader optimizer-only screen was likewise negative. Lion at two learning
rates peaked at 1.0000% variable split mean; Adafactor and RAdam each reached
0.7083%; NAdam reached 0.7917%. Schedule-free AdamW reduced training loss and
reached 26.0% fixed-N smoke mean at LR 0.005, demonstrating that it materially
changed optimization, but it did not improve the variable-modulus gate.

The common failure is transfer rather than inability to fit. Full gradient
flow, denser endpoint supervision, specialized refinement phases, more
recurrent computation, exact-match-aligned token pressure, higher learning
rate, and parameter averaging all improve at least one fitting or fixed-N
measure while failing to improve the matched variable-N mean and OOD-N T=1
together. None warrants an H100 run. Retain the original AdamW configuration.

## Tied-attention and dense-supervision wave

A subsequent wave replaced the bidirectional GRU reducer with a width-96 tied
self-attention update. Its four heads used general, neighbor, LSD-prefix, and
MSD-suffix masks. The only fixed arithmetic structure remained the learned
digit-pair grouping by significance. The matched short screen and a 60-second
gate used seed 74 throughout.

| intervention | short mean | short seen T=1 | short OOD-N T=1 | full mean | full seen T=1 | full OOD-N T=1 |
|---|---:|---:|---:|---:|---:|---:|
| positional tied attention | 1.4167% | 2.3438% | 0.0000% | 1.0833% | 2.3438% | 0.3906% |
| mixed-topology attention | 1.4167% | 1.7578% | 0.1953% | 1.0833% | 2.3438% | 0.3906% |
| no absolute position | 1.1250% | 1.1719% | **0.5859%** | — | — | — |
| no position, random 2--6 refinement stops | **1.5000%** | 1.1719% | 0.3906% | 1.0417% | **2.3438%** | 0.0000% |
| positions plus random stops | 1.2500% | 1.9531% | 0.1953% | — | — | — |
| random stops plus phase CRF NLL | 0.9583% | 2.1484% | 0.3906% | — | — | — |
| random stops plus local adjacent-pair CE | 0.6667% | 1.3672% | 0.3906% | — | — | — |
| random stops plus T=1-to-T=3 curriculum | 1.2500% | 1.3672% | 0.3906% | — | — | — |

Random stopping is the only new learning-signal change that materially raised
the ordinary short-screen mean: 1.5000% versus the matched retained control's
1.1250%. It also preserved two OOD-N T=1 successes without absolute position
features. The complete 60-second gate did not compound the gain. It completed
3,197 updates, scored 0.7500% test and 1.3333% OOD, reached 12/512 seen-N T=1,
and scored 0/512 OOD-N T=1. Running a model trained at four refinements for six
at evaluation was even more decisive: the fixed-N smoke collapsed to 16.8333%
mean. The tied block is not yet an iteration-stable correction operator.

Intermediate structured supervision improved shallow seen-N fitting while
hurting transfer. A CRF NLL at every random stopping point reduced throughput
and ordinary mean; replacing it with cheap adjacent-pair supervision produced
the wave's best fixed-N smoke (31.5000%) but its worst variable-N mean
(0.6667%). The model can cheaply learn locally plausible digit relations that
do not satisfy global modular reduction. Staging the actual endpoint objective
from T=1 to T=3 likewise regressed the short mean, so early composed-depth
gradients are not the main source of the plateau.

The surviving architectural signal is modest: tied attention repeatedly raises
seen-N T=1 to 12/512, and removing absolute position can raise short OOD-N from
one success to three. Neither occurs together at the full gate. These results
do not meet the 5% seen / 3% OOD-N H100 gate and should not consume an H100 run.

## Hosted Easy/E5 override

The CPU gate was later explicitly overridden while testing endpoint-clamped
finite-state learning. Submission `ca9d15ca-1499-47e7-845f-1c7552f80ff5`
completed 942 H100 updates in 60 seconds. Final training exact was 9.4%; test
exact was 1.1%, OOD exact was 1.3%, and the hosted split-mean score was
**1.21%**. No seen or OOD-N T=1 rung certified.

This exceeds the previous 1.13% product-attention record. More importantly, it
is the first hosted confirmation that exact endpoint normalization over
observable digit states improves the frontier without relying on hidden
arithmetic traces. It is the new Easy/E5 score control for program-lattice
work.
