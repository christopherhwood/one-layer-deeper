# General recurrent model with sequence-value loss

This candidate keeps the deeply supervised, weight-tied general recurrent
architecture and adds one dense endpoint signal. For every row, the loss turns
the model's digit probabilities into a normalized soft decimal value,

```text
E[d0] / 10 + E[d1] / 100 + E[d2] / 1000 + ...
```

and compares it with the value formed from the evaluator-supplied endpoint digit
labels using Smooth L1. The normalization is bounded below one regardless of
sequence length. Digit ranks are derived from the boundary-preserving valid mask,
so variable-length outputs and causal layouts are supported; non-digit and EOS
targets are excluded from only the value term.

Ordinary endpoint cross-entropy remains the primary anchor, and the existing
0.5-weighted mean endpoint loss across six recurrent refinement heads is
unchanged. The value term has weight 1.0.

## Rules audit

This is rules-valid under the published contract:

- Participants explicitly control the optional training loss (`README.md`, rules
  3 and 12).
- `TokenLossBatch` explicitly provides differentiable model logits, evaluator
  labels, the valid mask, target positions, and model auxiliary output
  (`benchmark/api.py`, lines 111-138).
- The value target is only a deterministic re-expression of the endpoint labels
  already passed to the loss. It generates no examples, solutions, quotient,
  carry, borrow, or intermediate process labels.
- Every operation is an ordinary PyTorch tensor operation on the evaluator's
  device, and the returned result is one differentiable scalar.
- The model's forward path remains learned. Decimal place geometry changes the
  training metric over supplied answer tokens; it does not implement modular
  squaring or any task solver in the forward pass.

This does encode the public ordering and place value of digit tokens. That is a
numeric output-interface prior, but it is materially short of the prohibited
task-specific solver: given a wrong answer it only measures numeric distance and
cannot derive the correct answer from the prompt.

## Local result

| run | optimizer updates | test exact | OOD exact | mean exact |
|---|---:|---:|---:|---:|
| generated 10-second CPU smoke | 358 | 8.33% | 5.00% | 6.67% |
| full E5, 60 CPU seconds | 1,081 | 0.75% | 0.67% | 0.708% |

The run used seed 74, training batch size 32, evaluation batch size 256, and
95,626 persistent model-state elements. It completed with finite gradients and
exit status zero. Final recorded training loss, including auxiliary terms, was
2.989043; evaluator cross-entropy was 4.622730 on test and 2.971868 on OOD.

The full-E5 run also used seed 74, training batch size 32, and evaluation batch
size 256. It completed 1,081 updates with a final combined training loss of
3.371720. Exact split counts were 9/1,200 test and 4/600 OOD. The seen-modulus
depth profile scored 2/512 (0.391%) at T=1, while the OOD-modulus profile scored
3/512 (0.586%) at T=1. Neither profile certified a rung.

For comparison, `general_recurrent_deep` reached 4.33% mean exact in its
10-second smoke, while the two-pass averaging and SAM variants reached 3.83%
and 3.50%. The value-aware model improves the smoke mean to 6.67% despite
completing fewer updates than the deep baseline (358 versus 424).

On full E5, the sequence-value loss improves mean exact accuracy from the deep
baseline's 0.375% to 0.708%, an 89% relative increase. It is the best balanced
generic result so far: both ordinary OOD exact accuracy and OOD-N T=1 remain
nonzero, rather than trading seen performance for a collapsed OOD profile.

The absolute H100 gate, however, requires at least 5% held-out T=1 and 3%
adversarial or OOD-N T=1 locally. This candidate reaches only 0.391% and 0.586%,
respectively, and certifies no depth. The value signal is evidence of a useful
general learning direction, but this version should not consume an H100 attempt.
