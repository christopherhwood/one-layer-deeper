# Generic categorical cellular recurrence

## Rule-safety boundary

This candidate is the rule-safe successor to the constructive categorical
endpoint proof. Its `forward` contains no multiplication, modular reduction,
comparison, carry/borrow, digit-product table, squaring branch, or
recurrence-specific transition. It parses `N`, `x`, and `T`, then applies one
randomly initialized, translation-equivariant learned cell repeatedly. The
only persistent mutable state is a distribution over decimal digits plus a
16-way categorical controller.

This distinction matters for the private-task warning that Hard may change the
recurrence. Earlier arithmetic interpreters could execute repeated squaring
perfectly, but they hard-coded the known algorithm and are not admissible Hard
candidates. This model can represent other local recurrent programs, but must
discover them from endpoint supervision.

## Local evidence

Two complete width sweeps gave the best balance between computation per
example and optimizer updates per minute.

| variable-modulus E5 CPU, 60 s | updates | ID exact | OOD-N exact | mean |
|---|---:|---:|---:|---:|
| 1 sweep | 2,312 | 0.50% | 0.00% | 0.25% |
| **2 sweeps** | **1,281** | **1.00%** | **0.67%** | **0.83%** |
| 4 sweeps | 768 | 0.42% | 1.00% | 0.71% |

The two-sweep version also passed the fixed-modulus ten-second smoke with
13.67% mean exact accuracy after 1,000 updates.

To test recurrence flexibility rather than squaring-specific fit, the same
four-sweep cell was trained on a generated hidden recurrence
`state <- 3*state + 1 (mod N)` without changing the model source. It reached
5.33% held-out exact after ten seconds and 4.00% after sixty seconds, but 0%
on unseen depth T=6. Thus the architecture can learn signal from a changed
rule, while compositional depth generalization remains unsolved.

## Hosted H100 evidence

| tier/data | sweeps | updates | ID exact | OOD exact | score | submission |
|---|---:|---:|---:|---:|---:|---|
| Easy E5 | 4 | 573 / 60 s | 0.30% | 0.30% | 0.30% | `53a90f83` |
| **Easy E5** | **2** | **1,259 / 60 s** | **0.80%** | **1.00%** | **0.92%** | `5c100c75` |
| Hard H1 | 2 | 23,043 / 3,600 s | 0.10% | 0.00% OOD-T, 0.10% OOD-N-T | 0.05% | `6f255767` |

The two-sweep Easy run is the strongest rule-safe learned-recurrence result in
this repository so far. Training exact accuracy rose to 3.9%, and its hosted
ID/OOD result closely reproduced the local gate.

## What the first Hard run actually found

The Hard trace stayed at 0% training exact for the entire hour and its loss
remained near the uniform-token baseline. Inspection found a real connectivity
bug: training executed at most eight outer recurrence steps. For a labelled
row with T greater than eight, no iteration ever selected that row's terminal
state, so its supervised logits remained a constant zero tensor. Easy E5 uses
only T=1/2/3 and could not expose the bug.

The cap is now removed. Training rolls each batch through its actual maximum
T, up to the evaluator-supported 64. A regression test replaces the learned
cell with a counting differentiable transition and verifies that a T=16 row:

1. executes exactly 16 outer transitions;
2. produces endpoint logits connected to a trainable parameter; and
3. propagates a nonzero endpoint gradient.

Consequently, the 0.05% Hard result is evidence against the capped artifact,
not a clean falsification of the generic categorical architecture or the
positive identifiability theorem. The uncapped artifact still needs a new Hard
run after the daily quota resets. Its expected cost is substantially lower
throughput on batches whose maximum T is large, but unlike the first run it
will receive the intended endpoint learning signal.

## Remaining gap

Removing the cap repairs supervision; it does not prove that optimization will
discover a reusable transition. The altered-rule probe still failed at unseen
depth, and no run has certified T=1. The next decisive evidence is therefore:

- uncapped Hard training must move below the uniform-loss plateau and produce
  nonzero training exact accuracy;
- T=1 must improve before deeper recurrence can be credited to composition;
- accuracy must then decay slowly, rather than collapse immediately, with T;
- OOD-N must track ID closely enough to indicate a shared learned program.

If the repaired run learns T=1 but not larger T, the next architectural change
should target stable recurrent composition. If it still cannot learn T=1, the
remaining problem is one-step program identification from sparse endpoints,
not recurrence depth.
