# Endpoint-clamped global program lattice

This candidate turns refinement depth into a finite, input-independent program
version space. One tied product-aware reducer exposes stopping depths 1--4.
Every depth defines a normalized CRF over actual output digits, and the custom
endpoint loss computes

`-log sum_d q(d) P_d(answer | x, N, T)`.

No phase is assigned the endpoint as a false intermediate label. The program
posterior is global across examples, the CRF states are observable decimal
digits, and evaluation extracts one global MAP depth. This is deliberately
different from the failed latent-path CRF, whose unnamed per-example states
became a lookup code.

## Rules and contract

- The model never receives labels.
- The loss uses only evaluator endpoints and valid target boundaries.
- No examples, arithmetic traces, products, quotients, or intermediate
  residues are generated as labels.
- CRF forward algorithms and Viterbi are bounded differentiable tensor
  operations.
- Python compilation, source validation, fixed/variable execution, full depth
  evaluation, and hosted execution pass.

## Depth-space ablation

| lattice | fixed updates | fixed exact | E5 updates | E5 exact | token | seen T=1 token | OOD-N T=1 token |
|---|---:|---:|---:|---:|---:|---:|---:|
| depths 1--6 + MDL prior | 157 | **37.00%** | 140 | 0.5417% | 14.19% | 16.48% | 15.69% |
| depths 1--4, uniform prior | 225 | 30.17% | 200 | **0.8750%** | **15.02%** | **17.77%** | 14.79% |

The six-program model gives the wave's strongest fixed-modulus result but does
not transfer. Restricting the version space recovers 43% more E5 updates and
improves ordinary exact, token, and seen-N T=1 metrics. OOD-N T=1 remains below
the retained controls, showing that stopping depth is not the missing program
coordinate.

## Hosted Easy/E5

Submission `932d507c-a77d-447b-ab2f-cc304ff2aded` completed 1,087 updates in
60 seconds. Training exact rose to 28.1%, test exact was 0.6%, OOD exact was
1.5%, and the official split-mean score was **1.04%**. No seen or OOD-N rung
certified.

The matched hosted output-CRF control
`ca9d15ca-1499-47e7-845f-1c7552f80ff5` completed 942 updates, ended at only
9.4% training exact, and scored **1.21%**. The lattice therefore improves
optimization and finite-corpus fitting while reducing generalization.

## Theorem update

Exact endpoint marginalization is sufficient to train the complete finite
program version space; recurrent credit assignment is no longer the dominant
failure in this candidate. But entropy-driven MAP extraction assumes the
provided endpoints distinguish one depth program. They do not. Forcing the
posterior to collapse selects a better memorizer, exactly as the positive
theorem predicts when its distinguishing-set premise is absent.

The next program lattice must branch over causally observable tape operations,
not merely stopping time, and should preserve Bayesian version-space
uncertainty until instruction ablations show that one globally shared program
is distinguished. Raw training interpolation is now a negative gate rather
than evidence for another H100 run.
