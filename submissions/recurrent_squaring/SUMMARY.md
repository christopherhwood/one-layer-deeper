# One Layer Deeper — research summary

A consolidated account of the investigation into the *One Layer Deeper* task:
predict `result = x^(2^T) mod N` from tokenized `(N, x, T)`. This document is the
capstone; the proofs, experiments, and submissions it references live alongside
it in this folder.

---

## 1. The task, and the one reduction that governs everything

Each example asks for `x^(2^T) mod N` (N = pq, a semiprime). The single structural
fact that drives the whole analysis is the recurrence

```
result(T+1) = result(T)^2 mod N ,   result(0) = x
```

so the entire certified-depth ladder `T = 1,2,4,8,16,32,64` is **one operation —
squaring mod N — iterated**. Depth of computation *is* time `T`.

**Proven consequences** (`ANALYSIS.md`, reproduced in `verify_math.py`):

- **Extrapolation in T is free (Claim 1).** If a model computes the one-step
  squaring exactly on the residues its computation visits, iterating it is exact
  at every depth. A model trained only to `T=3` certifies `T=64`. Learning is
  entirely about the single step.
- **The ladder collapses (Claim 3).** Because `2^T mod φ(N)` is eventually
  periodic, the seven rungs realize only a handful of *distinct* power maps
  (for E1's `N=323`, rungs `4,8,16,32,64` are two maps). Verified numerically.
- **Certification = exact coverage.** Certified depth is a step function of how
  completely the one-step operator is learned; it jumps to the top of the ladder
  only at 100% coverage (`demo_3_coverage_law.py`).

Everything else is about **how to learn that one step.**

---

## 2. What is solved

- **Bounded-N to 100% — but only via the trapdoor** (`t1_easy_100.py`).
  `x^(2^T) mod N = CRT(x^(2^T) mod p, x^(2^T) mod q)`. Modulo each *small* prime,
  squaring is a tiny fully-covered permutation, so it is learned exactly and
  generalizes to every held-out `x` — 100% on Easy T=1 test **and** the
  adversarial `depth_t_1`, and 100% across the whole ladder. **Catch:** this
  factors `N` (the RSA trapdoor). It is not rules-legal (task-specific solver) and
  cannot scale to Hard's large private `N`. `t1_why_factorization.py` shows the
  jump is *entirely* due to exposing the factors; generic features stay ~0%.

- **Single positional operations, length-general** (`ood_n_arithmetic.py`,
  `localize_endpoint_wall.py`). A weight-tied per-digit carry/borrow ripple learns
  and *generalizes to unseen lengths*: `a+b` 100% (≤6→14 digit), `(a+b) mod N`
  100% (≤4→10-digit N), `s mod N` given `s` 97%. The substrate that scales in N
  exists — for single ops.

- **The recurrence composes.** Given a good single-step cell, iterating it stays
  accurate (`proof_no_blur.py`: hard == soft composition, no representational
  blur; a 95% cell decays only as `0.95^depth`).

---

## 3. The one irreducible open problem

> **Learn modular multiplication / reduction as a generalizing function, from
> endpoint-only supervision.**

Squaring a many-digit number is `Σ xᵢxⱼ·base^{i+j}` then reduce — forming the
product is *linear* in a digit outer-product, so the hard core is **modular
reduction of a large number (long division)**. Every decomposition of it
(bit-serial double-and-add, tree-sum, digit-serial remainder) is a **composition
whose intermediates are latent** under endpoint-only labels — and *composition of
latent intermediates does not learn algorithmically* (`proof_depth_kills_cell.py`:
training a doubling task endpoint-only through depth K collapses the learned
cell's single-step accuracy from 100% at K=1 to ~6% at K=8).

This gates **all three tiers** — it is required even for Easy, because the
certified metric scores **fresh x**, and a fresh non-residue unit is unreachable
by coverage, so its square must be *computed*, not looked up.

---

## 4. What was ruled out — dead ends, with evidence

| approach | why it fails | evidence |
|---|---|---|
| per-residue / coverage lookup | memorizes one N's units; 0% transfer to new N; can't cover fresh non-residue units even for Easy | coverage law; E1 fresh-x |
| pooled-latent recurrent (memorizer) | free per-x latent memorizes; no fresh-x generalization | H100 e1: 6–7%, no cert |
| CRT / factorization | the trapdoor; illegal + doesn't scale to large N | §2 |
| straight-through composition | blocks gradients; can't even fit | Proof 1/2 |
| pure-soft deep composition | blurs over depth | ~10% |
| annealed-soft (fixes shallow) | shallow-only; deep still stalls | 65%→95% shallow, ~10% at 2-digit |
| second-order optimizer (L-BFGS) | not ill-conditioning; worse than Adam | 64% vs 100% train |
| local conv (Neural-GPU-lite) | can't do the **global** mod-comparison | OOD-N 0% |
| additive residual highway on digits | highway suits *refining a latent*, not *replacing* a residue | train stalls |
| tree-sum + annealed + ripple | shallow, but latent intermediates still don't train | ~8–10% |
| discrete finite-state cell | straight-through on a deep recurrent state is unstable | hurt (13%→0%) |
| curriculum alone | 1-digit never exercises multi-digit carries | ~24% |
| **abacus + looped attention (SOTA bias)** | **doesn't fit `x⁴ mod N` even in-dist; loss stuck** | **H100 e3: 1%** |

The recurring lesson: **compute/throughput/optimizer are not the limiter —
learning a *transferable* modular multiply is.**

---

## 5. Engineering finding (reusable)

The H100 manifests use `num_workers=2` on tiny train sets; at large `batch_size`
that means ~1 batch/epoch, so the DataLoader worker iterator is respawned almost
every step (~0.6 s overhead). **Dropping `batch_size` to 32 gives ~10× the
steps** (1459 vs 153 in 60 s). This helps any submission — but 10× steps on a
non-generalizing model just memorize faster, which is how we confirmed
generalization (not compute) is the wall.

---

## 6. Live H100 results (Easy tier)

| submission | dataset | score | Max T | steps/60s | takeaway |
|---|---|---|---|---|---|
| baseline_adamw | e1 | 1.67% | none | 93 | reference |
| recurrent_squaring | e1 | 6–7% | none | 153–195 | ~4× baseline; memorizes, no cert |
| recurrent_squaring (batch 32) | e1 | 2.33% | none | 1459 | 10× steps, score flat → generalization-bound |
| **oodn_abacus** | **e3** | **1.00%** | none | 1927 | frontier bet; doesn't fit training; loss stuck |

Attempts used: 5 of 60/day. Concurrency is 1 (sequential runs).

---

## 7. Honest conclusion & credible next directions

The competition reduces cleanly to **one unsolved primitive** — a length-general,
endpoint-only, exact modular multiply/reduce. Everything around it is solved
(the recurrence, extrapolation in T, single ops, the composition guarantee), and
the primitive is confirmed hard on **real H100**, not just CPU: the best-known
inductive bias (abacus + looped attention) doesn't fit it even in-distribution,
with a *stuck* loss (an architecture wall, not a step-count one).

This is genuinely the frontier of "learning to execute arithmetic algorithms."
Directions that are *new bets* rather than tweaks of a stuck architecture:

1. **Process supervision on the multiply.** Endpoint-only underdetermines the
   latent partial products. The T-rungs (`x², x⁴, x⁸`) are observed and *do*
   legally supervise the squaring boundaries (TRM/HRM-style deep supervision +
   1-step detached gradients removes the T-axis credit-assignment problem) — but
   they do **not** reach inside one multiply. A legal signal for the intra-multiply
   remains the open question.
2. **A structured (not free-form) modular-multiply module** — a learned
   Montgomery / digit-serial reducer whose *form* encodes the algorithm while its
   parameters stay learned — trading some generality for trainability.
3. Accept that this primitive may need a genuine algorithmic-reasoning advance.

Recommendation: do **not** spend further H100 attempts on variants of the
stuck-loss architecture; each only re-confirms the wall. The value delivered here
is the **exact isolation** of the single problem to solve, a proven
recurrence/extrapolation framework to build on, and a reusable throughput fix.

---

## 8. Repository guide (this folder)

| file | contents |
|---|---|
| `SUMMARY.md` | this document |
| `ANALYSIS.md` | full mathematical analysis and proofs |
| `verify_math.py` | dependency-free check of recurrence, periodicity, ladder collapse |
| `demo_1/2/3_*.py` | memorization fails → residue-faithful → coverage law |
| `t1_easy_100.py`, `t1_why_factorization.py` | bounded-N 100% via CRT + why factors are needed |
| `ood_n_arithmetic.py`, `localize_endpoint_wall.py` | single ops that length-generalize; wall localization |
| `proof_no_blur.py`, `proof_depth_kills_cell.py` | the two depth-failure proofs |
| `passthrough_residual.py`, `oodn_*.py`, `tree_annealed_modsum.py` | ruled-out OOD-N architectures |
| `submission.py` | recurrent-squaring competition submission |
| `../oodn_abacus/submission.py` | frontier candidate (abacus + looped core) |
| `../oodn_abacus/walkthrough.html` | accessible illustrated walkthrough |
| `OOD_N_PATH.md`, `H100_RESULTS.md`, `SUBMISSION_PLAN.md` | the OOD-N path, live results, submission plan |
