# Grounded global-program register machine

This candidate tests the next condition suggested by the endpoint-identification
theorem: make every example execute one short, input-independent learned
program over canonical digit tapes. Each of six global phases chooses two tapes
to read and one tape to write. The current throughput version alternates generic
left/right sweeps between phases. The only mutable registers are an answer tape
and a scratch tape; the source and modulus tapes are immutable. One finite local
transition is tied across positions, phases, and outer squares.

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
nonzero gradients to both the global read program (`1.16e-3` L1) and the local
cell (`38.41` L1). With hard evaluation choices, every recurrent digit is exactly
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

The H100 throughput version replaces the five-projection local MLP with one
hidden projection and one joint digit/controller head, fixes a generic
alternating scan schedule, and has 14,607 parameters. It completes 330 local
batch-128 updates in ten seconds (2.0x the retained version). Its fixed mean
exact is 6.83%, so the speedup costs 2.0 points of held-out accuracy.

## Variable-modulus gate

| gate | updates | exact | token | last token | seen-N T=1 token | OOD-N T=1 token |
|---|---:|---:|---:|---:|---:|---:|
| 15 seconds | 116 | 1.2083% | 14.35% | 10.17% | 16.36% | 15.75% |
| fast, batch 128 | 210 | 0.9167% | 13.93% | 10.08% | 15.65% | 14.10% |
| fast, batch 512 | 83 | 0.7917% | 14.45% | 10.50% | 16.16% | 14.95% |

The exact score alone is a false positive. Both ordinary token accuracy and the
decisive unseen-modulus T=1 diagnostic are below the batch-128 product control
(15.01% and 16.76%) and below the strongest short OOD-N signal (17.82%). The
model is learning answer-frequency and length regularities, not a reusable
one-square transition.

## Hosted Easy/E5 throughput experiment

The CPU rejection was explicitly overridden to test whether H100 utilization
or data throughput was hiding a useful learning curve.

| implementation | batch | updates | examples | final train exact | test | OOD | score |
|---|---:|---:|---:|---:|---:|---:|---:|
| all controller states vectorized | 128 | 336 | 43,008 | 0.8% | 0.7% | 1.3% | **1.00%** |
| small serial cell | 128 | 300 | 38,400 | 0.0% | 0.6% | 1.0% | 0.79% |
| small serial cell, occupied | 512 | 244 | **124,928** | 0.4% | 0.7% | 0.8% | 0.75% |

Submission IDs are `b8441d68-2718-4190-9828-16d0588759f5`,
`a1bd4a3a-3a6d-4818-9da8-9909a0598112`, and
`248c3b3b-d647-4b5e-94d3-fbd5b1ae6b8d`, respectively. None certifies seen or
OOD-N T=1.

Batch 512 raises processed-example throughput by 3.25x relative to the small
batch without improving score. The vectorized controller is compute-heavy but
scores slightly better. This cleanly rules out insufficient H100 data
throughput as the main failure: the limiting factor is identification and
learning signal, not examples processed per second.

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
