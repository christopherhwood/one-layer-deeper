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

## Learned-microcode successor

The current `submission.py` goes one level further.  It replaces the fixed
double-and-add accumulator update with two learned instruction slots.  Each
slot selects its right-hand register from `(accumulator, input, zero)`, and a
four-way learned opcode commits the second slot never, on the current bit, on
the inverse bit, or always.  Scan direction remains learned as well.

The exact posterior now covers 20,160 coherent complete programs:

`5 * 7 * 2 * 2 * 2 * 2` reducer/control choices times
`3 * 3 * 4` microcode choices.

The initial soft microcode relaxation failed at 4.3% exact, confirming that
blending incompatible execution paths reintroduces stochastic shortcuts.  The
exact discrete version starts from the wrong microprogram and reaches 100%
test/OOD accuracy in only 21 updates in the 10-second CPU gate.  Seeds 0, 1, 2,
and 74 all reach 100% in 21--26 updates.  It also passes the strict 0.05-second
smoke, the max-width T=64 test, source validation, and all 165 repository tests.

Its H100 M5 run, submission
`5684f91d-b090-4985-a394-c331ecf9828d`, succeeded on 2026-08-06:

- 9,000/9,000 test examples exact;
- effectively 3,000/3,000 OOD examples exact;
- all 768/768 examples certified at every seen- and unseen-modulus rung from
  T=1 through T=64;
- 1,011 updates in 600.2 training seconds;
- 122.2 seconds of the 300-second evaluation budget;
- 188,250 model-state and 13,633 optimizer-state elements.

At inferred Hard width the source has 190,446 state elements.  Scaling the
measured M5 evaluation by the private Hard example count and squared bit-width
ratio predicts roughly 379 seconds, versus the 1,800-second Hard evaluation
budget.

### Current rule-7 boundary

> **Do not submit this candidate to private Hard as rules-valid.** The public
> clarification “Obvs no hard coding the algorithm into the forward” resolves
> the ambiguity below against this design. Although the categorical switches
> are randomly initialized and endpoint-trained, Python still fixes the
> add/reduce/double-and-add interpreter and repeats the resulting square-shaped
> cell. Learning implementation switches inside a supplied arithmetic algorithm
> is not the same as learning the algorithm. The hidden recurrence warning also
> makes this an unsafe scoring assumption independently of admissibility.

Learned from endpoints:

- the two local finite-state relation coefficients;
- carry-branch polarity;
- modulus transform and reduction carry;
- scan direction;
- both accumulator instruction operands;
- the second instruction's bit-conditioned commit opcode;
- the global posterior selecting one shared complete program.

Fixed architectural structure:

- an observable binary tape and a two-state recurrent scan;
- two generic scan passes per reducer instruction;
- two accumulator instruction slots per input-bit step;
- iteration of the learned square cell according to the input T.

The untrained MAP program is wrong and low-accuracy, every instruction logit is
randomly initialized and endpoint-trained, and final hard decoding reads only
the executed state.  This is much closer to a learned finite-state interpreter
than the earlier task solver.  The remaining semantic review risk is whether a
two-pass reducer architecture itself is considered too task-specific under
rule 7; the benchmark text does not define that boundary more precisely.
