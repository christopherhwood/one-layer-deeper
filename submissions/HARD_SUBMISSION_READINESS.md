# Hard submission readiness — 2026-08-08

Hard submission is owner/manual-only. This document records the exact artifacts
and prevents a high public score from being mistaken for a completed compliance
review.

## Candidate A: strict rule-safe neural recurrence

File: `submissions/neural_gpu_recurrent/submission.py`

SHA-256:

`ec39724bffc0be2b426c5b2a9aa9982dc6b76ce361d91a254f132a8547015e5d`

Why it is the defensible fallback:

- its transition is a randomly initialized, tied, local gated neural cell;
- no multiply, remainder, carry, quotient, comparison, arithmetic table,
  recurrence formula, candidate-program executor, or search appears in the
  learned transition;
- final outputs depend only on learned state and generic representation
  conversion;
- endpoint labels provide decimal, binary, and ordinal views of the same final
  answer, not intermediate arithmetic states;
- all prediction-producing parameters are optimizer-updated through ordinary
  autograd.

Verified evidence:

- source validation: pass, 18,306 bytes;
- architecture/connectivity tests: pass;
- hosted Easy E1: 73.50% exact (`39a425de-9304-4eac-b7a5-ab80f6aecf18`);
- changed affine recurrence, unchanged source, CPU 10 s: 43.00%;
- changed cube recurrence, unchanged source, CPU 10 s: 31.83%;
- hosted Medium M1: 0.03%;
- variable-modulus E5 CPU 60 s: 0.79%.

This is the only current candidate that can be described as unambiguously free
of a supplied arithmetic solver. It does **not** meet the desired Medium/Hard
scoring evidence.

## Candidate B: endpoint-trained generic accumulator posterior

File: `submissions/generic_accumulator_program/submission.py`

SHA-256:

`27d909b6923e1b49e4342401a90bb26b174bc1059a1439d1e99067809568dcfa`

Verified evidence:

- source validation and three construction/connectivity tests: pass;
- hosted Easy E5: 100%, seen-N T=64, OOD-N T=64
  (`98f85bc4-affb-4920-905e-cbd539ecb667`);
- exact M5-shaped CPU gate: 100%, including both T=64 ladders;
- changed affine, cube, and `x^2+x+2` recurrence gates: 100%;
- changed `x^3+x^2+x+1` gate: 0.833%.

This is the strongest scoring candidate. It is not unambiguously compliant:
the learned posterior selects a program, but Python supplies exact modular
addition, subtraction, and multiplication. Treating that as a generic ISA
rather than a task-specific solver requires an organizer ruling under Rules 7
and 14.

## Decision gate at 17:00 PDT

1. Re-run source validation and confirm the SHA-256 of Candidate B.
2. Submit Candidate B to hosted Medium M5 after the quota reset.
3. Require 100% test/OOD and both T=64 certificates before considering its
   scoring evidence frozen.
4. If the organizers approve the generic modular ISA, Candidate B is the
   high-upside Hard artifact.
5. Without that approval, Candidate A is the only defensible Hard artifact,
   with materially lower expected score.
6. Do not submit either Hard artifact automatically; the owner must choose and
   consume the one-per-day slot manually.

Exact organizer question:

> Do Rules 7 and 14 allow a randomly initialized, endpoint-trained global
> posterior over a generic modular register ISA, where every opcode and operand
> slot is learned, the initial MAP program is wrong, and the interpreter supplies
> only add/subtract/multiply/modulo primitives? Or are the modular arithmetic
> primitives themselves considered a prohibited task-specific solver?

## Verification commands

```bash
scripts/preflight_hard_candidate.sh
# Use --full immediately before the hosted Medium/Hard decision.
scripts/preflight_hard_candidate.sh --full

shasum -a 256 submissions/generic_accumulator_program/submission.py
.venv/bin/python -m unittest tests.test_generic_accumulator_program
.venv/bin/one-layer validate submissions/generic_accumulator_program/submission.py

shasum -a 256 submissions/neural_gpu_recurrent/submission.py
.venv/bin/python -m unittest tests.test_neural_gpu_recurrent
.venv/bin/one-layer validate submissions/neural_gpu_recurrent/submission.py
```
