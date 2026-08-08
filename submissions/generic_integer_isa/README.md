# Generic integer ISA posterior

This candidate removes the most task-shaped assumption from
`generic_accumulator_program`: arithmetic is no longer automatically reduced
modulo N. Five global instruction slots learn a straight-line program over a
generic sixteen-operation integer calculator.

## Instruction set

| Code | Accumulator operation |
|---:|---|
| 0 | no-op |
| 1–3 | add, subtract, or multiply by the recurrence input |
| 4–6 | increment, decrement, or double |
| 7–9 | divide by input, remainder by input, or remainder by modulus |
| 10–11 | XOR or AND with input |
| 12–13 | minimum or maximum with input |
| 14–15 | load input or zero |

The complete `16^5 = 1,048,576` program space is represented once. A single
factorized posterior over the five slots is randomly initialized and trained by
ordinary SGD from exact endpoint marginal likelihood. Evaluation executes the
learned joint MAP program. There is no program mutation, archive, optimizer-side
adoption, participant-controlled training loop, or independently initialized
winning ticket.

At evaluator seed 74 the initial MAP is `[14, 2, 9, 13, 0]`; repeated square is
`[3, 9, 0, 0, 0]`. All 80 instruction logits receive finite nonzero endpoint
gradients.

## Verified results

Exact source SHA-256:

`87f8beed1c44ae168c30c41e1745c1adec02840b5dbc280aaa93c8b0a82f7636`

| CPU gate | Updates | Test | OOD | Seen/OOD-N ladder |
|---|---:|---:|---:|---:|
| fixed square, 10 s | 35 | 100% | 100% | — |
| hidden cube, 10 s | 34 | 100% | 100% | — |
| hidden `x^2+x+2`, 10 s | 34 | 100% | 100% | — |
| hidden affine, 60 s | 213 | 100% | 100% | — |
| M5-shaped variable N, 60 s | 159 | 100% | 100% | T=64 / T=64 |
| hosted Easy E5, 60 s | 2,066 | 100% | 100% | T=64 / T=64 |

The M5 gate covers 9,000 test examples, 3,000 OOD examples, and 768 examples at
every depth rung for both seen and unseen modulus identities.

The hosted Easy submission is
`36497520-7fcc-4188-b0e2-5760e8acddd6`.

## Compliance assessment

This architecture is substantially less recurrence-specific than the earlier
modular accumulator:

- reduction is an explicit learned instruction rather than automatic wiring;
- the same ISA represents qualitatively different recurrent functions;
- the initial deployed program is wrong;
- every deployed instruction is selected only by optimizer-updated logits;
- endpoint likelihood has a direct gradient to every trainable choice;
- all state and computation remain on the accelerator.

The remaining Rule 7/14 question is narrower but not eliminated: Python still
defines exact generic integer calculator primitives, and training exactly
marginalizes a finite program class. Organizer confirmation is still required
before calling any arithmetic-ISA candidate unquestionably admissible for
private Hard.

Exact organizer question:

> Do Rules 7 and 14 allow a randomly initialized, endpoint-trained global
> posterior over a generic integer calculator ISA, where every instruction slot
> is learned, the initial MAP program is wrong, modulo is an explicit learned
> instruction, and the interpreter supplies ordinary saturating arithmetic,
> divide/remainder, bitwise, min/max, and load primitives? Or are exact calculator
> primitives or finite program marginalization considered a prohibited
> task-specific solver?
