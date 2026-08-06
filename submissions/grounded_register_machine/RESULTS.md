# Grounded global-program register machine

This candidate tests the next condition suggested by the endpoint-identification
theorem: make every example execute one short, input-independent learned
program over canonical digit tapes. Each of six global phases chooses two tapes
to read, one tape to write, and a scan direction. The only mutable registers are
an answer tape and a scratch tape; the source and modulus tapes are immutable.
One finite local transition is tied across positions, phases, and outer squares.

The choices are generic and observable rather than arithmetic instructions. No
multiply, carry, compare, quotient, or modular-reduction rule is supplied. A
learned digit-pair evidence tape is grouped by positional significance. Earlier
outer-square registers are detached during training, and the only task target
is the requested endpoint. Scratch is weakly trained to be blank at a square
boundary, while label-free entropy terms sharpen the global program and finite
controller.

## Contract and gradient checks

The source validator, Python compiler, differentiability check, hard-state
invariants, and submission contract pass. An endpoint-only backward check gives
nonzero gradients to both the global read program (`4.59e-5` L1) and the local
cell (`3.75` L1). With hard evaluation choices, every recurrent digit is exactly
one-hot and every register position remains normalized.

## Fixed-modulus smoke

The retained six-phase, width-64 version completed 163 batch-128 updates in ten
seconds.

| split | exact | token | last token |
|---|---:|---:|---:|
| test | 6.67% | 18.90% | 18.33% |
| OOD-T | 11.00% | 20.75% | 24.00% |
| aggregate / mean | **8.83%** | **19.83%** | **21.17%** |

This is real endpoint traction, and delaying program crystallization improved
the first version from 5.67% to 8.83% exact. It remains below the retained
product control (17.50% exact) and the canonical categorical transition
(15.67%).

A four-phase ablation delivered 243 updates and reached 26.56% exact on a
training batch, but generalized worse: 1.67% test, 10.00% OOD-T, and 5.83% mean
exact. Shortening the program therefore increases finite-corpus fitting without
identifying a more reusable rule.

## Variable-modulus gate

| gate | updates | exact | token | last token | seen-N T=1 token | OOD-N T=1 token |
|---|---:|---:|---:|---:|---:|---:|
| 15 seconds | 116 | 1.2083% | 14.35% | 10.17% | 16.36% | 15.75% |

The exact score alone is a false positive. Both ordinary token accuracy and the
decisive unseen-modulus T=1 diagnostic are below the batch-128 product control
(15.01% and 16.76%) and below the strongest short OOD-N signal (17.82%). The
model is learning answer-frequency and length regularities, not a reusable
one-square transition. A 60-second or H100 run is not warranted.

## What this says about the theorem

Canonical tapes, a globally shared program, tied recurrence, endpoint
gradients, and discrete boundary states still do not make the available
endpoints a distinguishing teaching set. The flexible local MLP and learned
product evidence can implement many endpoint-compatible rules while ignoring
or repurposing nominally grounded instructions. Observable registers are not
the same as causally necessary operations.

The next differentiated path is a small factorized library of generic tape
operations in which each selected instruction is counterfactually necessary
for its write. Program agreement across seeds and instruction-ablation effects
should be gates before raw accuracy. That would shrink the effective rule class
where this candidate leaves it flexible, without supplying a task-specific
arithmetic solver.
