# General semantic-closure results

## Question

Can endpoint-only program generation move beyond the original
Horner/carry-shaped language and discover different recurrences and different
reduction functions without changing the architecture?

## Tightened old-language ceiling

We exhaustively enumerated all 2,240 programs in the original language. The
primary audit excludes x=0, x=1, tiny moduli, and exhaustive-set collision
inflation; it selects on stratified two-digit moduli and freezes the program for
wider unseen moduli.

| target | best train exact | selected OOD exact |
|---|---:|---:|
| identity | 100% | 100% |
| square mod N | 100% | 100% |
| cube mod N | 100% | 100% |
| fourth power mod N | 5.79% | 0% |
| (3x+1) mod N | 9.50% | 8.26% |
| square mod (N+1) | 17.36% | 13.86% |
| square quotient N | 18.18% | 13.27% |
| saturating square | 83.06% | 86.14% |
| square AND (N-1) | 38.84% | 24.78% |

The unsupported rows are expressivity failures, not optimizer failures.

## General expression language

The replacement starts only from terminals `{x, N, 0, 1}` and composes seven
ordinary operations: `ADD`, `SUB`, `MUL`, `DIV`, `MOD`, `AND`, and `MIN`.
Bottom-up generation retains one representative per observed behavior.

| operation count | new unique behaviors | targets first created |
|---:|---:|---|
| 0 | 4 | identity |
| 1 | 17 | — |
| 2 | 180 | square remainder, square quotient |
| 3 | 2,706 | cube, N+1 modulus, saturation, bit mask |
| 4 | 46,530 | fourth power, affine recurrence |

The final archive has 49,437 behaviors. Every audited target is generated,
receives zero endpoint loss, is ranked first by an actual float32 posterior
backward pass, and reaches 100% on wider unseen moduli. The smallest measured
gradient margin over the runner-up is approximately 1.33e-7.

Naive genetic evolution in a fixed six-slot register encoding was not enough:
it solved square remainder, quotient, and mask, but stalled on several
three/four-operation modular compositions. Modular endpoints do not provide a
reliable local notion of whether an internal subexpression such as x^2 will
eventually be useful for x^3 mod N. Semantic closure supplies this exploration;
the endpoint gradient performs final behavioral selection.

## Benchmark-native bridge

One unchanged closure-archive model was run through the standard evaluator:

| CPU gate | test | OOD | updates |
|---|---:|---:|---:|
| ordinary square, 10 s | 100% | 100% | 385 |
| hidden cube, 10 s | 100% | 100% | 362 |
| hidden affine `(3x+1) mod N`, 10 s | 100% | 100% | 379 |
| variable-modulus square, 60 s | 100% | 100% | 2,737 |

The variable run certifies T=1,2,4,8,16,32,64 for both seen and OOD moduli.
The old generated Horner model scored only 0.83% mean exact on the same hidden
affine 10-second gate.

Persistent model state is 692,125 scalar elements. The bridge begins with four
active terminals and exposes cumulative archive sizes 21, 201, 2,907, and
49,437 over its first four updates.

## Recovered Hard-shape stress

The prior Hard crash occurred around 26-bit inputs, deep minimum T, and unsafe
near-64-bit operations. The general bridge was tested at the recovered shape:

- full 49,437-program archive;
- batch 8, 26-bit modulus, T=8;
- finite forward, endpoint loss, backward, and optimizer update;
- CPU forward 0.114 s and backward 0.007 s;
- exact selected squaring through T=64 at 26-, 27-, 29-, and 30-bit moduli.

The implementation uses nonnegative saturating 62-bit multiplication without
evaluating an overflowing int64 product. This supports 30-bit squaring. A
different recurrence whose unreduced intermediates exceed 62 bits still needs
multiword arithmetic or a generic multiply-divide primitive.

## Research interpretation

This is a major architecture advance for the project, but the ingredients are
related to established program-synthesis work:

- [DeepCoder](https://arxiv.org/abs/1611.01989) learns program properties to guide search.
- [BUSTLE](https://arxiv.org/abs/2007.14381) guides bottom-up search using executed intermediate values.
- [CrossBeam](https://arxiv.org/abs/2203.10452) learns which explored programs to compose next.
- [Metric Program Synthesis](https://arxiv.org/abs/2206.06164) clusters observationally similar programs and repairs approximate candidates.
- [DreamCoder](https://arxiv.org/abs/2006.08381) learns reusable DSL abstractions and search guidance.

The distinctive result here is hard recurrent integer execution plus ordinary
endpoint-gradient selection of a level-grown semantic archive inside the fixed
benchmark training loop. That is not yet a demonstrated field-level
breakthrough. The production bridge currently precomputes the generic archive
at model construction and activates levels during optimization. A stronger
claim requires on-the-fly GPU archive generation/pruning, scaling beyond four
operations, comparisons with pure enumeration and learned bottom-up search,
and hidden-family/H100 results from one standalone artifact.
