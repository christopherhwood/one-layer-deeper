# Generic accumulator program posterior

## What it learns

This experimental candidate learns four global instructions from endpoint
labels. Each slot chooses an opcode (`noop`, `add`, `subtract`, or `multiply`)
and a right operand (the recurrence input, one, zero, or the modulus). The same
four learned choices are reused for every example, modulus, and outer recurrence
step.

Training enumerates the complete 65,536-program version space and learns one
factorized posterior over the four instruction slots. Evaluation executes the
joint MAP program. No complete winning program is independently initialized,
and no optimizer-side search, mutation, archive, or parameter adoption occurs.
At evaluator seed 74 the initial MAP is `[14, 2, 9, 13]`, while the one-step
square program is `[12, 0, 0, 0]`.

## Verified evidence

Exact artifact SHA-256:

`27d909b6923e1b49e4342401a90bb26b174bc1059a1439d1e99067809568dcfa`

| Gate | Test | OOD | Seen-N ladder | OOD-N ladder |
|---|---:|---:|---:|---:|
| CPU fixed, 10 s | 100% | 100% | — | — |
| CPU variable E5, 60 s | 100% | 100% | T=64 | T=64 |
| CPU M5-shaped, 60 s | 100% | 100% | T=64 | T=64 |
| Hosted Easy E5, 4,474 updates | 100% | 100% | T=64 | T=64 |

The hosted Easy run is
`98f85bc4-affb-4920-905e-cbd539ecb667`. The exact source has not yet received a
hosted Medium run because the six-attempt daily quota was exhausted; the quota
resets at 2026-08-09 00:00 UTC (2026-08-08 17:00 PDT).

The unchanged architecture also reached 100% on endpoint-generated affine,
cube, and `x^2 + x + 2` recurrences. It failed the broader
`x^3 + x^2 + x + 1` probe at 0.833%, because that expression requires branching
or more temporary state than this four-slot accumulator language supplies.

Three unit tests prove that the program space is complete, the initial MAP is
not the square program, multiple changed recurrences are representable, and a
finite nonzero endpoint gradient reaches every instruction-logit tensor. The
standalone source validator also passes at 12,967 bytes.

## Rule-review boundary

Rules 6 and 8 have a strong structural case: every trainable value is randomly
initialized and optimizer-updated, the endpoint loss has a direct gradient to
the global instruction posterior, and the deployed MAP is selected only by
those learned logits. All model computation remains on the accelerator.

Rules 7 and 14 remain unresolved. Python supplies exact integer addition,
subtraction, multiplication, and automatic reduction modulo N after every
instruction. Training selects the instruction sequence, but it does not learn
the semantics of those primitives. Under the strictest reading, the fixed
interpreter is a task-specific arithmetic solver even though the recurrence
program is learned.

**Do not submit this artifact to private Hard as unambiguously rules-cleared.**
It becomes a defensible submission only if the organizers explicitly confirm
that a learned global posterior over a generic modular register ISA is an
allowed architectural prior.

The exact review question is:

> Do Rules 7 and 14 allow a randomly initialized, endpoint-trained global
> posterior over a generic modular register ISA, where every opcode and operand
> slot is learned, the initial MAP program is wrong, and the interpreter supplies
> only add/subtract/multiply/modulo primitives? Or are the modular arithmetic
> primitives themselves considered a prohibited task-specific solver?
