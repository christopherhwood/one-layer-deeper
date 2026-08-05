# The compliant path toward Hard / OOD-N: learn arithmetic, not the table

The Easy 100% (`T1_EASY_RESULT.md`) used N's factorization — the RSA trapdoor —
so it cannot touch Hard. This note develops the *only* route that can: compute
`x^(2^T) mod N` by **learning the arithmetic positionally**, so it generalizes
across `x` *and* across unseen moduli `N`, with **no factoring**.

## One primitive, composed

Three reductions collapse the whole task onto a single learned operation:

| level | reduces via | to |
|-------|------------|----|
| `x^(2^T) mod N` | recurrence `result(T+1)=result(T)² mod N` | iterate squaring `T`× |
| `u² mod N` | double-and-add over the bits of `u` | modular doublings + adds |
| **`(a+b) mod N`** | — | **the one learned primitive** |

Squaring is `mult(u,u)`; modular doubling is `add(a,a)`. Everything above the
primitive is **fixed control flow** (adaptive computation — rule 4), so the only
thing that is *learned* is modular addition on digit strings.

## Why it generalizes to larger, unseen N — with no factoring

The primitive is built as a **weight-tied, per-digit finite-state transducer**:

* add pass (LSB→MSB): ripple **carry** automaton → `s = a+b`;
* subtract pass: **borrow** automaton → `s−N`, whose final borrow bit is exactly
  `[s < N]`;
* one **global gate** on that bit selects `s` (if `s<N`) or `s−N`.

A finite-state transducer is **length-independent**: once its transition table is
learned on short numbers, it is exact at *any* length — i.e. any modulus size.
Length-generalization **is** OOD-N generalization, and it is honest schoolbook
arithmetic — the factorization of `N` never appears.

## Results (all in `ood_n_arithmetic.py`, CPU-reproducible)

| capability | trained on | exact accuracy on unseen sizes |
|-----------|-----------|-------------------------------|
| integer addition | ≤6-digit operands | **100%** at 10 and 14 digits |
| `(a+b) mod N` | ≤4-digit `N` | **100%** at 8 and 10-digit `N` |
| `x² mod N` (composed, **0 extra training**) | — | **100%** at N=3,4 (train) and N=6,8 (**OOD-N**) |
| `x^(2^T) mod N` (iterate squaring) | — | exact wherever each squaring is (Claim 1): iterating an exact map stays exact |

The squaring and power-tower rows use the *same* single modular-addition cell,
wired by fixed control — nothing new is learned to go from `add` to `square` to
the full `T`-ladder. That is the recurrent-squaring thesis realized in exact,
length-general arithmetic.

## What still stands between this and a Hard win (honest)

1. **Credit assignment.** The real task labels only the final `x^(2^T) mod N`;
   every carry, borrow, and partial product is latent through a very deep
   composition. The demo trains the primitive with *intermediate* ("deep")
   supervision on the add/subtract/gate steps — a diagnostic proving the
   architecture can *represent and learn* exact general modular arithmetic.
   Learning the full deep pipeline from **endpoint labels alone** is the open
   problem (curriculum in `N` and `T`, and the rung-level supervision the data
   already provides at `T=1,2,3`, are the levers).
2. **Eval-time depth.** One squaring is `O(bits²)` primitive steps; `T` of them
   is `O(T·bits²)`. For large Hard `N` this pressures the wall-clock budget and
   argues for a larger digit base (fewer, wider steps) or a learned
   sub-quadratic reduction.

Neither caveat requires factoring `N` — which is precisely why this is the route
with a pulse on Hard, where the trapdoor is unavailable. The representation is
exact and general (proved here); closing the training gap is the research.
