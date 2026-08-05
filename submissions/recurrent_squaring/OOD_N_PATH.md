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
| `x^(2^T) mod N` (iterate squaring) | — | **100%** at T=1,2,4,8 on unseen `x`, N=5 digits |

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

## The credit-assignment wall, isolated (`endpoint_only_probe.py`)

I attacked caveat (1) directly: train the *same* modular-add cell and
double-and-add squaring with **endpoint-only** supervision — only `x² mod N` is
labeled, every carry/borrow/gate/partial-product latent (this is the honest
competition signal: a `T=1` prompt's target is exactly `x²`). Levers tried:
curriculum in `N`-size, a residue-faithful straight-through accumulator (exact
forward, soft gradient), aggressive weight tying.

**Result:** even at the shallowest scale (`N<10`, ~4 double-and-add steps),
endpoint-only training **plateaus around ~65% exact** and does not reach 100%.
Diagnosis: per-op accuracy reaches ~0.95, but ~8 latent ops compound
(`0.95^8 ≈ 0.66`), and the diluted endpoint gradient cannot pin the *rare*
carry/gate cases. Soft (non-straight-through) states were worse (~10%): they
blur over depth. So the obstacle is not representation or capacity — it is
**endpoint-only optimization of a deep, discrete latent composition**, a
known-hard "learn an algorithm from I/O" problem.

| supervision | shallow `x² mod N` exact | generalizes to OOD-N |
|---|---|---|
| process (intermediate) — `ood_n_arithmetic.py` | **100%** | yes |
| endpoint-only — `endpoint_only_probe.py` | **~65% (plateau)** | no |

**Roadmap for closing it** (all factoring-free, none yet decisive here):
discrete-optimization methods beyond straight-through (Gumbel schedules,
REINFORCE/expected-gradient on the gate); recurrence-consistency losses across
the observed `T=1,2,3` rungs (deeper endpoint constraints, no algorithmic
labels); an easier reduction (larger digit base → fewer, wider steps); and
much longer training with the H100 budget the tiers provide. The honest status:
the exact, modulus-general **representation exists and is learnable with process
supervision**; making it emerge from **endpoint labels alone** is the open
problem standing between this approach and a Hard-tier result.

### Hypotheses tested — and what actually broke (`crack.py`, `sum_depth.py`, `localize_endpoint_wall.py`)

I ran the roadmap's leading levers as controlled experiments. Results, honestly:

**Ablation on `x² mod N`, endpoint-only** (2-digit `N`):

| lever | exact | verdict |
|---|---|---|
| #1 discrete finite-state cell | 13% | **hurt** — straight-through on a deep recurrent state is *unstable* (loss rose) |
| #2 curriculum + transfer | 19% | marginal |
| #1 + #2 | 0% | worse |

So my leading bet (discrete state) was **refuted**. Then a controlled depth test
(sum `M` numbers mod `N`; sequential vs. tree do the *same* number of adds, only
the critical-path depth differs):

| composition | depth | exact |
|---|---|---|
| one modular addition (M=2) | 1 | 45% |
| tree (M=4 / M=8) | 2 / 3 | 3% / 4% |
| sequential (M=8) | 7 | 3% |

**Tree did not beat sequential** — depth reduction was also refuted as the lever.
Finally, isolating each sub-operation endpoint-only pinned the real cause:

| operation (in isolation, its output IS the label) | exact |
|---|---|
| `a+b` (carry latent) | **100%** |
| `s−N` (borrow latent) | **100%** |
| `s mod N` given `s` (gate latent) | **97%** |

**Every atomic operation learns endpoint-only.** The wall is not the carry, the
borrow, or the reduction — it is **composition**: when op A feeds op B and only
B's output is labeled, A's output is a *latent intermediate* with no supervision
and never sharpens to exact, so the composite (modular addition 45%, squaring
65%) degrades. Carry-free / RNS representations do **not** help, because the
latency is in the composition, not the carry.

**Where this leaves it.** The true bottleneck is **multiplication**: squaring a
many-digit number is an `O(n²)` sum of partial products, all latent under
endpoint-only supervision — the classic reason neural nets fail at exact
multi-digit multiply. The competition's only labels are at the `T`-rung level
(`x^(2^T)`), which bound the squaring *boundaries* but never the arithmetic
*inside* one squaring.

### CORRECTION — the "wall" was largely an optimization artifact (`annealed_soft_probe.py`)

The negative results above shared a hidden flaw: every composed model used a
**straight-through** hardened accumulator (`st_hard`) between steps. That
*blocks gradient flow*, and the ~65% plateau was an **optimization** failure —
the model could not even fit the training set (flat loss), though the
representation is provably fittable (100% with process supervision). Two clean
tests corrected this:

1. **Second-order is *not* the fix.** Full-batch **L-BFGS** on the stuck
   `(a+b) mod N` task did *worse* than Adam (64% vs. Adam's train-100%). So the
   problem is not ill-conditioning per se.
2. **Smooth differentiability *is* the fix.** Replace the straight-through
   accumulator with a **smooth, temperature-annealed softmax** (clean gradients
   early, sharpened toward hard/exact late). On the exact task where
   straight-through plateaued at **65%**, and pure-soft blurred to **~10%**, the
   annealed-soft model reaches **~95%** — the plateau breaks.

So the honest picture flips: this is an **optimization / differentiability**
problem, not a fundamental information wall. The earlier "non-gradient methods
needed" conclusion was **wrong** — the fix is gradient-based and foundational to
how the composition is made differentiable and optimized.

The remaining difficulty is **depth in the optimization sense** — and I first
mis-explained it as "blur." Two proofs settle what actually fails.

**Proof 1 — it is NOT forward blur (`proof_no_blur.py`).** Train one
`(a+b) mod N` cell to 95% single-op accuracy, **freeze** it, and compose it K
times (repeated doubling, exact ground truth `2^K·x mod N`):

| K | hard-compose exact | soft-compose exact | soft accumulator peak-prob |
|---|---|---|---|
| 1 | 94.7% | 94.7% | 0.998 |
| 4 | 87.3% | 87.3% | 0.999 |
| 8 | 81.5% | 81.2% | 0.997 |
| 12 | 78.0% | 77.2% | 0.998 |

Hard ≈ soft at every depth and peak-prob stays ~1.0 — **no blur.** A *good* cell
composes fine; the gentle decay is just an imperfect (95%, not 100%) cell
compounding. So the soft representation was never the problem.

**Proof 2 — deep END-TO-END training fails to teach the cell
(`proof_depth_kills_cell.py`).** Train the doubling task endpoint-only at depth K
(supervise only `2^K·x mod N`), then measure the learned cell's *single-step*
accuracy:

| train-depth K | endpoint exact | learned cell single-step exact |
|---|---|---|
| 1 | **100%** | **100%** |
| 4 | 10.8% | **10.0%** |
| 8 | 7.0% | **6.2%** |

This is the real failure, proven: **the deeper the composition you back-prop
through, the worse the shared cell you can learn** — its single-step competence
collapses from 100% (direct supervision) to ~6% (through 8 tied steps). The
gradient from a distant endpoint, routed through many tied composition steps,
cannot teach the cell the operation. Not blur, not representational capacity —
**credit assignment / optimization through depth.**

Together the two proofs give the recipe: a good cell composes (Proof 1), so the
whole game is *getting* a good cell, which needs **shallow** (ideally single-step)
supervision of the cell (Proof 2). The competition supplies only `T`-rung
endpoints, and one squaring is itself deep (bit-serial double-and-add) — so the
indicated fix is a **minimal-depth squaring** (quadratic-form `x² = Σ
xᵢxⱼ·(base^{i+j} mod N)` reduced by a tree; wider digit base) so the rung-level
label supervises a near-shallow composition. Curriculum is the same idea in the
training schedule; the 1→2-digit version only reached ~24% because 1-digit inputs
never exercise the multi-digit carries the deep case needs.

Status: shallow is solved end-to-end (~95%); scaling to Hard-sized `N` hinges on
delivering shallow-enough supervision to the cell, per the two proofs above.

### A residual / pass-through highway breaks the deep-training stall (`passthrough_residual.py`)

Proof 2 said the failure is optimization *through depth*. The standard cure for
that in deep nets is a **residual / pass-through gradient highway** (ResNet,
highway nets, LSTM cell-state, and the "iterative refinement / TRM" designs the
rules mention). Tested on the exact failing case — learn `(2^K x) mod N`
endpoint-only through K=8 tied steps:

| update rule | endpoint-exact |
|---|---|
| plain `h ← cell(h,N)` | 61.5% (stalls early ~8–10%, slow) |
| **residual `h ← h + cell(h,N)`** (cell zero-init → starts as identity) | **97.0%** (fast, loss→0.04) |

The highway helps for two compounding reasons: the additive path is a **gradient
highway** to early steps, and identity-init means early untrained steps emit
**valid intermediates**, not garbage — directly countering the failure mode of
Proof 2. This is the first lever that clearly moves the deep endpoint-only case.

**Honest caveats.** (1) This proof-of-principle uses a fixed-width hidden-state
model (W=3), so it is *not* length-general and cannot itself scale to larger `N`
(unlike the digit-ripple cell); its 97% is fit/generalization *within* 2-digit
`N`, not OOD-N. (2) The domain is small enough that some memorization is possible.
So the result validates the **principle** (pass-through cures the deep-training
stall), not a finished OOD-N solution.

### What blocks OOD-N, and the mis-step that sharpened the attack (`oodn_neural_gpu.py`)

Three things block OOD-N: (1) **fixed width** (a wider `N` has no input slots);
(2) a **global blob state** memorizes width-specifically instead of learning a
length-independent procedure; (3) **exactness compounds** — a larger `N` needs a
deeper composition, so the per-step cell must be ~100%, not 97% (95%⁶⁴ ≈ 4%).

The natural attack — combine a length-general **tied convolution** (Neural-GPU
style) with the residual highway — was tested (train on 1–3 digit `N`, eval 5–6
digit): **train ~60%, OOD-5d/6d = 0%.** It does *not* length-generalize. The
reason is specific and useful: **modular reduction needs a GLOBAL magnitude
comparison** (`is running-value ≥ N?` across *all* digits), and a local conv
cannot carry that comparison in a length-independent way.

But we already have a length-general structure that *does* generalize OOD for
modular arithmetic: the **sequential borrow-propagating digit-ripple** (§`arith_2`
earlier), which reached **100% on 8–10-digit `N` trained only on ≤4-digit** —
because the borrow chain *is* the global comparison, propagated LSB→MSB. So the
right length-general per-step cell is the **sequential ripple, not the conv.**

**Refined attack (well-motivated, not yet built):** sequential borrow-ripple cell
(length-general, proven to generalize OOD for one modular op) **+** a residual
highway *across the composition* (the Proof-2 fix that lets a good cell survive
deep endpoint-only training). Blocker 3 is then handled because a finite-state
ripple learned to *exactly* 100% (add/sub already do) composes without decay. The
conv detour shows the highway alone is not enough — the length-general substrate
must be the one that can do the global mod-comparison.
