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

The successor is the factorized-program construction probe in
`submissions/recurrent_squaring/factorized_control_probe.py`.  Its initial MAP
program is wrong and scores 0%; endpoint training changes seven global
categorical choices and reaches 100% without an initialized complete winner.
