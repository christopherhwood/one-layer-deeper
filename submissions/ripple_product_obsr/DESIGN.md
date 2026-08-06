# Observable Belief-State Reducer

## Identifiability target

Endpoint input/output pairs cannot identify a unique hidden carry, borrow, or
quotient algorithm: equivalent internal factorizations can implement the same
mapping. OBSR therefore targets the strongest general-purpose notion available
without process labels: an identifiable predictive state.

After evidence column `k`, the recurrent state is a normalized first-order
distribution over the actual output digit string:

```text
uniform B0
   | learned product/modulus evidence column 0
   v
observable B1 -- column 1 --> observable B2 -- ... --> final BK --> output
```

Each belief is represented canonically by its initial digit distribution and
row-normalized adjacent-digit conditionals. There is no learned state embedding
or emission decoder. Exact forward/backward normalization maps every learned
energy update back into this observable representation before the next column.

On T=1 rows, every prefix belief receives the same evaluator-supplied endpoint
under a partial-suffix chain log score. This does not claim that an unfinished
state is an arithmetic intermediate. In expectation, a strictly proper score
has the unique optimum `P(endpoint | evidence prefix)`, including honest
uncertainty when the prefix is insufficient. The final chain is scored on every
row and is submitted directly.

## General-purpose boundary

The only supplied structural assumptions are generic sequence mechanisms:

- learned pairwise input interactions grouped by positional significance;
- a tied evidence update across input columns;
- a tractable first-order belief over output tokens;
- proper scoring and exact normalization.

The update contains no digit products, modular comparisons, carries, borrows,
quotients, candidate enumeration, generated examples, or intermediate labels.
The model has no EMA teacher. Its output-chain states are actual decimal-token
probabilities only because decimal tokens are the evaluator's observed output
vocabulary, not because any arithmetic relationship is encoded.

## Difference from rejected mechanisms

- EMA, PCGrad, Muon, and batch scaling change optimization but leave state
  semantics free.
- The temporal outcome head leaves an unconstrained hidden state behind a
  learned decoder. OBSR removes both.
- Closed-loop recurrent feedback retains a hidden controller and re-embeds its
  predictions. OBSR feeds only normalized observable beliefs forward.
- The final, phase, and trigram CRFs decode hidden reducer states. OBSR makes a
  first-order distribution the reducer state itself; first order is retained
  for tractable belief updates, not terminal capacity.
- Latent CRF/FST probes marginalize a free state alphabet, while grounded
  quotient probes introduce arithmetic scaffolding. OBSR has neither a latent
  alphabet nor an arithmetic candidate space.

## Decisive sequence

The first run is only the standard ten-second contract smoke. It establishes
that canonical chain updates normalize, differentiate, train, and evaluate
within the public runner. Fixed-modulus accuracy is not evidence for modular
reduction. A later variable-E5 screen requires a separate decision; no 15- or
60-second run is part of this implementation step.
