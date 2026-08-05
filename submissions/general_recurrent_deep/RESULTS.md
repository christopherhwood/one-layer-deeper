# Deeply supervised general recurrent model

This candidate tests a rules-valid, general-purpose response to the arithmetic
diagnostics: a small tied attention reasoner, endpoint readouts at every inner
refinement, detached state between refinements, and a soft digit interface
between `T` outer applications. It contains no product, quotient, carry,
borrow, or modular-arithmetic features.

## Local results

| run | updates | test exact | OOD exact | mean exact |
|---|---:|---:|---:|---:|
| generated 10-second CPU smoke | 424 | 1.67% | 7.00% | 4.33% |
| full E5, 60 CPU seconds | 1,088 | 0.42% | 0.33% | 0.38% |

The generated smoke confirms the submission contract, finite gradients, and
basic optimization only. On full E5, the seen-N and OOD-N depth profiles both
score 0.20% at T=1 and certify no rung. Training batch exact accuracy is still
0% near the end of the run.

The predeclared H100 gate required at least 5% held-out T=1 and 3% adversarial
or OOD-N T=1 locally, plus a material improvement over final-only training.
This candidate misses the absolute gate by more than an order of magnitude, so
it should not consume an H100 attempt.

## Conclusion

Deep endpoint supervision and detached recurrent segments shorten gradient
paths but do not resolve latent-semantic identifiability. The result should not
be followed by width, loop-count, or optimizer tuning. A subsequent experiment
must introduce a genuinely different general learning signal, such as
endpoint-anchored representation consistency or an evaluator-owned multi-pass
bootstrapping scheme, and must pass the same local held-out gate.
