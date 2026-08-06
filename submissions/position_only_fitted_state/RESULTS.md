# Position-only mixer with paired attainable-state recovery

This candidate tests whether recurrent length generalization fails because the
reducer reaches hidden states at inference time that its training trajectory
never taught it to repair. It preserves the position-only product mixer and
adds a late-training recovery phase.

The successful variant keeps the current problem condition intact, perturbs
only the mutable scratch component with position-wise Gaussian statistics fit
to the model's own clean final states, and processes clean and perturbed copies
in parallel. The ordinary answer loss always trains the clean copy. At every
refinement step, a normalized hidden-state loss teaches the perturbed copy to
return to the detached clean trajectory. Evaluation uses only the clean path.

## Ablations

| fixed-N 10s variant | updates | exact | token | last token | conclusion |
|---|---:|---:|---:|---:|---|
| clean position-only control | 697 | 36.50% | 51.21% | 38.83% | control |
| mix complete hidden state | 683 | 16.17% | 41.47% | 29.00% | corrupts problem identity |
| mix scratch, endpoint label only | 680 | 20.50% | 44.92% | 34.17% | endpoint signal cannot teach repair |
| mix scratch + paired local recovery | 565 | **38.67%** | **55.86%** | **43.83%** | useful local learning signal |

The ablations isolate the mechanism. Generic latent noise is harmful. Keeping
the immutable input condition separate is necessary, but not sufficient. The
gain appears only when the perturbed trajectory receives a dense target from
the clean trajectory.

## Variable-modulus gates

| gate | updates | exact | token | last token | OOD-N T=1 token |
|---|---:|---:|---:|---:|---:|
| 15s | 407 | 0.46% | 15.10% | 9.00% | 16.38% |
| 60s | 1,645 | **0.67%** | **16.04%** | **12.54%** | 16.97% |
| clean 60s control | 1,827 | 0.58% | 15.96% | 12.42% | **17.92%** |

Paired recovery improves fixed-task accuracy per update and slightly improves
ordinary variable-modulus exact and token accuracy. It does not improve the
best OOD-N signal. The fitted independent Gaussian state model likely teaches
contraction toward the training-state distribution rather than preserving the
structured correlations needed outside the training modulus lengths.

This is a reusable component, not an H100 candidate by itself. The next
differentiated version should use structured attainable states (for example,
whole clean scratch states passed between examples or trajectory segments)
instead of independently sampled position/channel marginals, while continuing
to preserve the current problem condition and paired local target.
