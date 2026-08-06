# Position-only predictor-corrector

This family tests whether the endpoint gradient becomes more useful when each
tied refinement is explicitly responsible for a residual logit correction.
The local path detaches the previous cumulative logits before adding the next
correction; a parallel connected path optionally preserves the true endpoint
gradient through all corrections. All targets are real T=1 endpoint digits.

## Fixed-N smoke ablations

| variant | updates | exact | token | last token |
|---|---:|---:|---:|---:|
| clean position-only phase prediction | 697 | **36.50%** | **51.21%** | **38.83%** |
| one shared full/correction head, persistent belief writeback | 655 | 28.50% | 49.17% | 36.67% |
| one shared head, read-only belief context | 649 | 26.83% | 46.82% | 35.33% |
| separate predictor/corrector heads, local credit only | 654 | 13.33% | 34.94% | 24.00% |
| separate heads, local + connected global path, belief context | 636 | **32.17%** | **49.80%** | **38.50%** |
| separate heads, local + global path, no writeback | 683 | 29.83% | 48.30% | 33.67% |

The first shared-head design is ill-posed: the same head must emit a complete
prediction in the first phase and only a residual later. Separate heads expose
a second problem: detached local credit starves the initial predictor of the
true endpoint loss. Restoring a numerically identical connected path fixes
that bug and is the strongest predictor-corrector variant, but it still trails
the ordinary phase-prediction baseline. Belief writeback helps that corrected
form slightly but does not change the decision.

No variant clears the fixed smoke gate, so none receives variable-modulus or
H100 compute. Endpoint cross-entropy already supplies each phase's digit error;
reparameterizing later logits as residuals does not add information that
identifies the modular-reduction transition and imposes stage-dependent roles
on an otherwise tied recurrence.
