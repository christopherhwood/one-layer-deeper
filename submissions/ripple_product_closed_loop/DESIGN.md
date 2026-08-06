# Product-aware closed-loop latent ripple

## Hypothesis

Tracked `ripple_product_h100` exposes the right multiplication substrate but
uses it open-loop: learned digit-pair features are grouped by significance,
hidden columns receive four residual bidirectional scans, and phase digit heads
are readouts only. A phase prediction is never embedded into the next phase.
The core must therefore discover carry propagation, global comparison, and
modular reduction as one unconstrained hidden-state transformation.

The successful process-supervised diagnostic had the opposite structure. A
small state moved LSD-to-MSD, each pass emitted an explicit digit string, that
string became the next pass's input, and a terminal state controlled the global
selection. It length-generalized, but relied on forbidden intermediate labels
and an impractically deep double-and-add composition.

This candidate tests the untried intersection of those designs: keep the
one-shot learned product columns, but normalize them with a closed-loop,
two-axis recurrent machine.

## Single-square computation

For `W` residue digits, the model creates `2W-1` significance columns and runs
`2W` tied rounds (with a minimum of four):

```text
learned pair grid --sum by i+j--> fixed product columns P[k]
                                      |
zero digit accumulator A[0,k]         v
             +-----------------> persistent latent H[0,k]
             |                        |
             |   tied LSD->MSD scan --+-- terminal controller broadcast
             |                        |
             |                   gated residual update
             |                        |
             +--- embed A[r,k] <--- decode + gated replacement
                                      |
                                  next round
```

The hidden latent carries a residual gradient highway across refinement rounds.
The unidirectional recurrent carrier supplies length-independent communication
across digit positions. Its terminal state is transformed and broadcast as a
generic sequence-level controller. The decoded digit accumulator is gated,
normalized, and re-embedded into the following round rather than accumulated
additively as logits.

At E5 width four, eight one-way GRU scans perform approximately the same
directional recurrent work as the tracked model's four bidirectional scans.
Weights are tied over all rounds, loops contain no tensor-to-host synchronization,
and no phase has its own parameters.

## Novelty relative to prior runs

- `ripple_product_h100`: product columns and residual hidden columns, but no
  digit writeback and no terminal-controller bottleneck.
- `digit_transducer_h100`: hidden/digit writeback, but no explicit product
  columns; it scored roughly 0.4-0.5% on E5.
- `recurrent_register_h100`: gated register refinement, but uses convolution
  and global attention with untied phases and no product columns.
- `oodn_neural_gpu`: persistent residual latent, but local convolution cannot
  perform length-independent global comparison.
- `oodn_ripple_highway`: sequential ripple, but applies the residual highway
  directly to digit logits, leaving obsolete digit peaks behind.
- quotient and CRF diagnostics: latent paths or ordered quotient candidates,
  but no persistent residual per-position latent with closed-loop writeback.

No prior candidate combines all four ingredients: learned product columns,
persistent per-position hidden state, decoded accumulator feedback, and a tied
sequential carrier with terminal broadcast.

## Rules and supervision

Every digit-pair contribution, recurrent transition, controller, residual
update, replacement gate, and output digit is learned from random initialization.
The architecture fixes only significance grouping, scan direction, weight
sharing, and scratch-state shape. It contains no literal multiplication table,
carry or borrow transition, quotient, comparison result, or modular-reduction
rule.

Training retains the tracked submission's final endpoint CE, 4x T=1 row
weighting, T=1 phase endpoint losses, outer-step detach, temperature schedule,
optimizer, and batch sizes. No intermediate arithmetic value or process state
is labeled. T controls only reuse of the complete learned square transition.

## Staged evaluation

Before any timed run, the candidate must pass source validation and a finite
forward/backward contract check. Timed smoke and full-E5 gates are intentionally
deferred until no other runner is active.

The pre-run checks passed on CPU with the E5 shape (`W=4`): source policy and
bytecode compilation were clean, the model contained 347,147 persistent scalar
elements, and the official endpoint-plus-phase loss produced finite, nonzero
gradients for all 23 trainable tensors on a two-row `T=1,2` batch. This is a
numerical contract check only, not evidence of learning or score.
