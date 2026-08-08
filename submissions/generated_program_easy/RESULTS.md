# Generated-program Easy result

This experiment tests a stronger claim than the 8,192-particle posterior: the
correct complete square program is absent initially and must be created from
endpoint feedback by selection, cloning, mutation, and random restart.

## Construction gate

The offline endpoint-only prototype removed any lucky target before search. On
seeds 0 through 9, all ten runs generated the canonical complete program,
reached 100% exact accuracy on the teaching moduli, and reached 100% exact
accuracy on entirely unseen moduli. Creation took 2--47 generations.

The evaluator-style regression is stricter about the actual submitted path. At
seed 74, the 96-program model naturally contains zero canonical complete
programs at initialization. The custom optimizer creates canonical programs by
generation 2, selects one using only final-answer supervision, and solves
unseen moduli.

## Local evaluator gates

- `local_cpu_10s.json`: 100% test, 100% OOD in 41 steps.
- `local_cpu_variable_60s.json`: 100% test, 100% OOD in 149 steps.
- Variable-modulus depth profile: 100% at T=1,2,4,8,16,32,64 for both seen and
  unseen moduli.
- Persistent model state: 2,497 scalar elements.
- Optimizer state after the first step: 4,809 scalar elements.

## H100 Easy E1

- Submission: `640d6c13-cf4a-4659-9e01-cf7cad56ad7c`
- Run: `fdb6ab54-c94d-4a8e-850b-37f5bfe73f10`
- Exact score: 100%
- Maximum certified T: 64
- OOD-N maximum certified T: 32

The result proves benchmark-native program creation and selection, but it does
not yet beat the prior seeded model's OOD-N T=64 certificate. The next clean
experiment is hierarchical endpoint fitness: retain local bit credit while
adding leading-decimal-prefix and circular modular-distance credit to prefer
globally coherent arithmetic programs.

## Interpretation and limitation

The winning complete program is no longer required to be present in the
initial population. That is a meaningful advance from pure posterior selection.
However, the supplied program language remains arithmetic-shaped: it is a
generic learned bit/carry relation embedded in a configurable Horner-style
interpreter. This is program discovery inside a compact domain-specific
language, not unrestricted general-purpose program synthesis.
