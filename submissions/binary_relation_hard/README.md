# Global binary-program Hard candidate

This is the H100 translation of the positive endpoint-identifiability proof.
It replaces the proof-only fixed relation posterior with 8,192 randomly
initialized trainable complete-program particles and one randomly initialized
global posterior learned from evaluator endpoints.

## Learned program

Every particle learns the local binary relation

`a + b + carry + beta * output + gamma * next_carry = 0`

including `beta`, `gamma`, and the final-carry reduction polarity. The same
particle also learns five coordinates of the outer recurrent program:

- zero or input accumulator initialization;
- most- or least-significant-bit scan direction;
- zero or input addend source;
- normal or inverted input-bit branch;
- one or two tied Horner multiplication passes.

The last choice is important for the private warning that Hard may alter the
recurrence: a one-pass complete program represents modular squaring, while a
two-pass complete program represents modular cubing. Nothing in `forward`
selects either recurrence from the tier, dataset, or prompt. One global MAP
particle is learned from final answers and reused at every bit, example,
modulus, outer step, and evaluation depth.

At evaluator seed 74, random initialization contains 123 particles with the
canonical low-level relation, five complete square programs, and three
complete cube programs. Their posterior preferences are random. Endpoint
likelihood must distinguish the complete program; evaluation executes only the
learned MAP particle.

The forward path executes candidate transitions discretely. A small
straight-through surrogate supplies gradients to each particle's categorical
fields, while exact whole-answer marginal likelihood supplies the strong
gradient to the global posterior. Training uses the shallowest T represented
in each evaluator-owned batch, so identification does not depend on T=1 being
present.

## Verified final artifact

- source validator: pass, 26,658 bytes;
- every trainable tensor receives a finite T=2 endpoint gradient;
- seed-74 population contains complete square and cube programs;
- selected square program is exact at an eight-decimal-digit modulus through
  T=64;
- selected cube program composes exactly for multiple outer steps;
- standard 10-second CPU gate: 100% test and OOD after 18 updates;
- 12/14/16-bit T=2-only CPU M5 gate: 100% test/OOD and both 768-row ladders
  certified through T=64 after 19 updates;
- hosted Easy E5 `39d6aca0-10b8-478a-8fef-f1eac2f7caea`: 100%, seen-N T=64,
  OOD-N T=64, 232 updates in 60 seconds;
- hosted Medium M5 `21329829-31da-4fc7-aeba-ed9cc11e6d2a`: 100% test
  (9,000/9,000), 100% OOD (3,000/3,000), seen-N T=64, OOD-N T=64, and 893
  updates in 600 seconds.

The final M5 loss falls from 7.431 at step 1 to 0.023 by step 100. Training
exact accuracy is not the relevant diagnostic because the differentiable
mixture remains broad; the learned MAP particle used by evaluation is exact.

## Hard readiness and compliance boundary

The model-side candidate passes every available public correctness,
identifiability, width, depth, CPU, and H100 proxy. At the prior Hard shape,
eight decimal slots allocate 29 binary bits. The maximum-width T=64 regression
passes, and evaluation runs only one particle, so the larger training
population does not enlarge ladder cost.

Hard submission remains owner/manual-only under the repository standing
directive. No automated agent should consume the daily Hard slot.

Rules 6 and 8 are satisfied structurally: every prediction-responsible
particle field and posterior weight is randomly initialized, trainable,
optimized, endpoint-connected, and receives a finite gradient. Repeated
squaring is no longer fixed in `forward`; the global learned program can select
one- or two-pass recurrence and other control variants.

The remaining semantic review point is Rule 7. The low-level bit scanner,
learned-relation interpreter, and add/reduce wiring are architectural control
flow rather than learned Python source. That is analogous to supplying a small
generic arithmetic ISA, but it is still more task-shaped than a Transformer.
An owner should obtain organizer confirmation before treating a private Hard
score as competition-admissible.
