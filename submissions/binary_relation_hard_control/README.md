# Learned-control relation particles

This checkpoint extended the binary-relation posterior from the local carry
relation into four additional categorical control choices: modulus inversion,
reduction carry, scan direction, and bit-gate polarity.  Each of 4,096
randomly initialized particles represented one complete program.

## H100 Medium M5 result

Submission `ac972729-825d-4a65-9a9f-8a18fbbaffb0` succeeded on 2026-08-06:

- 9,000/9,000 test examples exact;
- effectively 3,000/3,000 OOD examples exact (reported aggregate
  `0.9999999913` due floating aggregation);
- all 768/768 examples certified at every seen-modulus rung T=1 through T=64;
- all 768/768 examples certified at every unseen-modulus rung T=1 through T=64;
- 893 training updates in 600.9 seconds;
- 112.1 seconds of the 300-second evaluation budget;
- 100,988 model-state and 201,988 optimizer-state elements.

## Scientific and rules limitation

The complete program class has only `70 * 2**4 = 1,120` distinct choices, but
the model initializes 4,096 independent particles.  Seed 74 therefore starts
with three fully correct complete programs; endpoint training can rank one of
those winning tickets instead of constructing a missing program.  This run is
strong evidence that the executor is exact, length-general, and fast enough for
Hard.  It is not the final rules-defensible candidate because the lottery and
fixed arithmetic control flow make rule 7 a material concern.

## Factorized construction successor

The current `submission.py` replaces the 4,096 independent particles with one
factorized distribution over seven global categorical instructions.  All 1,120
programs have ordinary softmax support during training, as classes do in a
classifier, but initialization has only one joint MAP program and seed 74's MAP
is wrong.  There is no independently initialized complete winning ticket.

The endpoint loss exactly marginalizes the small program class and updates the
seven factor logits.  Evaluation executes their learned joint MAP choice.  The
hard decoder cannot memorize answers: it renders only the integer produced by
that learned program.

This fixes the initialized-winner objection, but it does not by itself settle
rule 7.  The interpreter still supplies the two-pass modular-add and
double-and-add scan skeleton.  Seven instruction fields are learned, while the
high-level control-flow shape is architectural prior knowledge.  Treat the H100
result as a construction and throughput gate; a final submission still needs a
semantic rules audit or a more generic learned-microcode executor.

CPU construction gates:

- seeds 0, 1, 2, and 74 all began with different incorrect MAP programs;
- all four reached 100% exact test and OOD accuracy in 119--129 updates;
- standalone seed 74 passed the 0.05-second smoke evaluation and reached 100%
  in the 10-second learning gate;
- all 165 repository tests and standalone source validation pass.

## Factorized H100 M5 result

Submission `e85b43c0-ccd3-4e4e-ba57-4968856a446c` succeeded on 2026-08-06:

- 9,000/9,000 test examples exact;
- effectively 3,000/3,000 OOD examples exact (reported aggregate
  `0.9999999913` due floating aggregation);
- all 768/768 examples certified at every seen-modulus rung T=1 through T=64;
- all 768/768 examples certified at every unseen-modulus rung T=1 through T=64;
- 1,313 training updates in 600.1 seconds;
- 104.0 seconds of the 300-second evaluation budget;
- 14,642 model-state and 13,615 optimizer-state elements.

Relative to the seeded-particle control, factorization produced 47% more
updates, cut evaluation by 8.1 seconds, and reduced model state by 85.5%, while
preserving perfect endpoint and depth-profile behavior.  Most importantly, its
deployed seed began with an incorrect complete MAP program.
