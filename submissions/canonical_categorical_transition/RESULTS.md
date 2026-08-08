# Generic categorical cellular recurrence

## Rule-safety boundary

This is the rule-safe neural successor to the constructive categorical
endpoint proof. Its `forward` contains no multiplication, modular reduction,
comparison, carry/borrow, digit-product table, squaring branch, or
recurrence-specific transition. It parses `N`, `x`, and `T`, then repeatedly
applies one randomly initialized, translation-equivariant learned cell.

The mutable state carried between recurrence applications is only a decimal
digit distribution. A 16-way categorical controller exists inside one
application but is reset before the next. The transition sees the current
state and N, never the original x, so it is forced to have the Markov form

`state_(t+1) = learned_transition(state_t, N)`.

This is compatible with the private warning that Hard may change the
recurrence. Earlier arithmetic interpreters executed repeated squaring
perfectly but hard-coded the known algorithm and are not admissible Hard
candidates.

## Three repaired proof assumptions

The first hosted Hard artifact did not actually implement the mathematical
architecture we intended. Three independent defects were found and repaired:

1. Training stopped the outer rollout after eight steps. Rows with larger T
   never selected a terminal state, leaving their supervised logits equal to a
   constant zero tensor.
2. Every outer state after the first was detached. Endpoint gradients
   therefore could not constrain the shared transition through its
   composition, contrary to the positive proof's premise. Full recurrent
   backpropagation is now retained.
3. Even-length prompts used `(max_seq_len - 5) // 2` decimal cells. A prompt
   of length ten containing three-digit N and x therefore received only two
   cells, irreversibly dropping the hundreds digit. The correct geometry is
   `(max_seq_len - 4) // 2`.

The transition also used to receive immutable original x at every outer step,
which enabled direct endpoint memorization. That shortcut has been removed.

Regression tests now prove that a T=16 endpoint executes all 16 transitions
and has a nonzero gradient to the transition parameters, and that a
length-ten prompt preserves all three decimal digits of N=323 and x=203.

## Constructive learning evidence after repair

All results below use the same model source and endpoint labels only. The
affine and cube probes change the recurrence without changing the model.

| CPU probe, 60 s unless noted | held-out ID exact | unseen-depth exact |
|---|---:|---:|
| affine, train T=1 only | **66.0% at T=1** | **50.0% at T=2** |
| affine, train T=1/2/3 | **52.67%** | **17.0% at T=6** |
| cube, train T=1/2/3 | **26.67%** | **32.0% at T=6** |
| fixed-N squaring, public E1 geometry | **29.33%** | **33.0% at T=6** |
| variable-N squaring, public E5 geometry | 0.75% | 1.33% at T=6 |

The affine T=1-only training batches reached 100% exact. With only 200 of the
available x values in training, the learned transition still reached 66% on
withheld x and 50% when composed twice. This is direct executable evidence for
the positive theorem's mechanism: endpoint supervision can identify a reusable
transition, and the same weights can generalize to an unseen recurrence depth.

The fixed-N ten-second smoke also improved from 13.67% before the repairs to
21.83%. On the full public E1 geometry, one CPU minute reached 31.17% mean
exact across ordinary and OOD-T evaluation. Extending that same run to five
minutes reached 90.6% batch exact, 57.33% held-out exact, and 55.0% at unseen
T=6 (56.17% mean). The learned transition therefore continues improving well
beyond the short smoke budget, although it is not yet exact enough to certify
an entire depth rung.

## Remaining frontier

Variable modulus is now the isolated failure. On a variable-N T=1-only
squaring probe, training batch exact reached 69.5%, but held-out T=1 was only
0.25% and unseen T=2 was 1.33%. The neural cell can memorize the available
endpoint pairs but does not infer the shared modular-multiplication program.

That distinguishes two regimes:

- compact changed recurrences such as affine and fixed-N cube/squaring are
  learnable and compositionally reusable;
- discovering a length- and modulus-general arithmetic program still requires
  a much smaller, program-like hypothesis class or substantially stronger
  teaching coverage.

## Hosted H100 history

| artifact | tier/data | updates | score | submission |
|---|---|---:|---:|---|
| four-sweep pre-repair | Easy E5 | 573 / 60 s | 0.30% | `53a90f83` |
| two-sweep pre-repair | Easy E5 | 1,259 / 60 s | 0.92% | `5c100c75` |
| capped/detached pre-repair | Hard H1 | 23,043 / 3,600 s | 0.05% | `6f255767` |

The Hard trace stayed at 0% training exact for the entire hour and remained
near uniform-token loss. Because that artifact disconnected deeper endpoints,
its 0.05% score is not evidence against the repaired architecture or the
positive identifiability theorem.

The corrected artifact has passed source validation, the fixed-N smoke, the
changed-affine and changed-cube probes, the deep T=16 connectivity gate, and
the variable-N E5 CPU gate. A shared-account Hard run from the separate
learned-ISA line currently occupies the daily hosted slot; this corrected
artifact still requires its own Easy and Hard H100 measurements.
