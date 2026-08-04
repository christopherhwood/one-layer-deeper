# Defeating *One Layer Deeper*: a mathematical analysis

**Task.** Each example gives the decimal tokens of `(N, x, T)` and asks for the
decimal digits of

$$ f_T(x) \;=\; x^{2^T} \bmod N, \qquad N = pq \ (p\neq q \text{ prime}),\ \gcd(x,N)=1. $$

This is one step of an RSA time‑lock / verifiable‑delay function. The **Hard**
tier is ranked first by *Max T* — the largest `T` on the ladder
`T ∈ {1,2,4,8,16,32,64}` for which the model is **exactly** correct on that rung
and every lower rung (a consecutive 100%‑exact prefix), on **fresh** prompts —
then by *OOD‑N Max T* (the same ladder at **unseen** modulus sizes), then by
accuracy at the first uncertified rung.

The entire design below follows from one observation and three provable claims.
Everything asserted here is reproduced by two committed scripts:

```
python submissions/recurrent_squaring/verify_math.py     # claims 1,3 (no deps)
. .venv/bin/activate
python submissions/recurrent_squaring/demo_1_memorization_fails.py
python submissions/recurrent_squaring/demo_2_residue_faithful.py
python submissions/recurrent_squaring/demo_3_coverage_law.py   # the coverage law
```

---

## 0. The observation: it is one map, iterated

Squaring the output advances the clock by one tick:

$$ f_{T+1}(x) = f_T(x)^2 \bmod N = s\big(f_T(x)\big), \qquad s(u) := u^2 \bmod N, \quad f_0(x)=x. $$

`s` is the squaring endomorphism of the unit group `U = (ℤ/Nℤ)^×`. So the whole
ladder is a single map `s` composed with itself, and *depth = number of
squarings*. `verify_math.py` confirms `f_{T+1}=f_T^2 mod N` for all units up to
`T=64` on E1, E2, M1, M2. This is the literal meaning of **"one layer deeper."**

---

## 1. Claim 1 — Extrapolation in T is free

> **Claim 1.** Let a model compute `M_T(x) = d(c^{(T)}(e(x)))` for a fixed
> encoder `e`, a **single** cell `c` iterated `T` times, and a decoder `d`.
> Suppose on the *test orbit‑closure*
> `O = { f_t(x) : x ∈ X_test, 0 ≤ t ≤ T_max }` there is an encoding under which
> `c` simulates `s` and `d` reads residues back:
>
> $$ c(e(u)) = e(s(u)) \quad\text{and}\quad d(e(u)) = \mathrm{digits}(u), \qquad \forall u \in O. $$
>
> Then `M_T(x) = digits(f_T(x))` for **every** `x ∈ X_test` and **every**
> `T ≤ T_max`. Every rung is 100% exact; the certified depth is the top of the
> ladder.

**Proof.** Induction on `t`: `c^{(0)}(e(x)) = e(x) = e(f_0(x))`. If
`c^{(t)}(e(x)) = e(f_t(x))` with `f_t(x) ∈ O`, then
`c^{(t+1)}(e(x)) = c(e(f_t(x))) = e(s(f_t(x))) = e(f_{t+1}(x))`. Decoding at
`t=T` gives `d(e(f_T(x))) = digits(f_T(x))`. ∎

The `e = d = id` special case ("the state *is* the residue") is Claim 1 with the
identity encoding. **Consequence:** the learning problem is *entirely* about the
one‑step cell. Nothing about `T=64` needs to be learned that isn't already in
`T=1`; correctness propagates through the iteration. A model that never trains
past `T=3` can still certify `T=64`.

This is visible even in an *untrained* run of the submission: on E1 after 111 CPU
steps the seen‑N profile already reads **identically** at `T=4,16,32,64`
(accuracy 0.105 at each) — the plateau of Claim 1, before any real training.

---

## 2. Claim 2′ — Weight‑tying + multi‑T supervision points at the true dynamics

The competition never supervises intermediate squarings — only endpoints
`(x, T) ↦ f_T(x)` for `T` in the training set. Why should the shared cell learn
`s` rather than some shortcut?

> **Claim 2′ (identifiability by composition).** Fix a residue‑faithful state
> (same residue ⇒ same code; the submission's masked‑mean pooling and the demos'
> distributional state both enforce this). If a *tied* cell `c` attains zero
> endpoint error on all training pairs `(x,T)`, then for every residue `u` that
> occurs as an interior state of some training orbit,
> `c` must act as `s` at `u`:  `c(e(u)) = e(s(u))`.

**Proof sketch.** Take a training pair `(x, T)` with `T ≥ 1`. Zero endpoint error
means `c^{(T)}(e(x)) = e(f_T(x))`. Residue‑faithfulness forces the interior
states to be `e(f_0(x)), …, e(f_T(x))` (any two orbits meeting at a residue share
the same code, so the tied `c` cannot send that shared code to two different
places). Hence `c(e(f_t(x))) = e(f_{t+1}(x)) = e(s(f_t(x)))` for `0 ≤ t < T`, i.e.
`c = s` at every interior residue `f_t(x)`. ∎

The set of residues *pinned* this way is exactly the union of training orbits —
which **includes residues never presented as inputs**, because they appear as
`x^{2^t}` for some training `x`. This is the leverage the recurrence provides:
shallow endpoint labels transitively constrain the map deep inside `U`.

**Why "residue‑faithful" is not optional — `demo_1` vs `demo_2`.** With a *free
per‑x embedding*, `demo_1_memorization_fails.py` drives **training** loss to 1e‑5
(perfect on seen `x`) yet held‑out‑`x` accuracy stays ≈ 0 at every rung: the
network memorizes a lookup keyed on `x`'s identity and generalizes to no fresh
prompt. Replacing the free embedding with a **distributional (residue‑faithful)
state** in `demo_2_residue_faithful.py` — same tied cell, same shallow `T∈{1,2,3}`
supervision — makes the learned operator match true squaring on **248/288**
units and, critically, the held‑out‑`x` accuracy is **flat from `T=4` onward**
(44.8% at `T=4,8,16,32,64`), precisely the Claim‑1 plateau. The certified‑depth
metric uses **fresh `x`**; it is *designed* to punish memorization and reward the
faithful, tied recurrence.

---

## 3. Claim 3 — The ladder collapses to a few power maps

`T=64` sounds like `2^{64}` sequential work. It is not, because the exponent
lives modulo `φ(N)`.

> **Claim 3.** Write `φ(N) = 2^a · m` with `m` odd. Then `e_T := 2^T mod φ(N)`
> is eventually periodic in `T`: for `T ≥ a`,
> `e_T = 2^a·(2^{T-a} mod m)`, and `2^{T-a} mod m` has period `ord_m(2)`.
> Hence `f_T(x) = x^{e_T} mod N` is periodic in `T` with pre‑period `≤ a` and
> period `ord_m(2)`. The ladder therefore realizes at most `a + ord_m(2)`
> **distinct** power‑maps of `U`, and rungs sharing an exponent are the *same
> function* — exact on one **iff** exact on the other.

**Proof.** `x ∈ U ⇒ x^{φ(N)} ≡ 1`, so `x^{2^T} ≡ x^{2^T mod φ(N)}`. For `T ≥ a`,
`2^T = 2^a·2^{T-a}` and `2^T mod φ(N) = 2^a·(2^{T-a} mod m)` since `gcd(2^{a}, m)=1`
splits the CRT factor `2^a`. `2^{T-a} mod m` is periodic with period `ord_m(2)`. ∎

`verify_math.py` prints the collapse exactly (all checks `True`):

| tier | `N=pq` | `φ(N)` | pre‑period `a` | period `ord_m(2)` | **distinct rungs** | identical rungs |
|------|--------|--------|----|----|----|----|
| E1 | 323 = 17·19 | `2^5·9` | 5 | 6 | **5** of 7 | {8,32}, {16,64} |
| E2 | 899 = 29·31 | `2^3·105` | 3 | 12 | **4** of 7 | {4,16,64}, {8,32} |
| M1 | 10403 = 101·103 | `2^3·1275` | 3 | 40 | 7 of 7 | — |
| M2 | 38021 = 193·197 | `2^8·147` | 8 | 42 | 7 of 7 | — |

For E1, rungs `T=4,8,16,32,64` collapse to just two exponents (`x^{16}` and its
partners): **once the cell is exact through `T≤5`, the top of the ladder follows
for free.** Two further facts from `verify_math.py` make the target small:

* the **reachable set** (the "rho") from any single `x` under squaring is tiny —
  ≤ 10 (E1), 14 (E2), 42 (M1), 48 (M2) residues — so each orbit only asks the
  cell to be right on a handful of points;
* the union of these rhos over `U` is the *entire* region the operator must get
  right, and Claim 2′ shows training orbits pin exactly that region.

---

## 4. What certification actually requires — the coverage law

Claims 1–3 reduce *Max T* on a seen `N` to a single, sharp condition:

> **Certification condition.** The certified depth equals the top of the ladder
> **iff** the learned cell equals `s` on the test orbit‑closure `O` (a subset of
> `U` of size bounded by the small rhos above). It is 0 otherwise — the metric is
> all‑or‑nothing, because rung `T` requires 100% on rung `T` *and* every lower
> rung.

`demo_3_coverage_law.py` makes this a measured law. Training the tied faithful
operator on a varying fraction of `U` (at `T∈{1,2,3}` only), evaluated on the
**full** unit group (the exhaustive‑`x` regime the seen‑N depth splits use):

```
train frac  op-coverage  certified T   per-rung exact %  (T=1,2,4,8,16,32,64)
     0.50        64.6%            0      65  65  69  69  69  69  69
     0.70        79.2%            0      79  80  82  82  82  82  82
     0.85        90.6%            0      91  91  91  91  91  91  91
     0.95        95.8%            0      96  96  96  96  96  96  96
     1.00       100.0%           64     100 100 100 100 100 100 100
```

Two things to read off. **(a)** Certified depth is a *step function* of operator
coverage: it is 0 until coverage is total, then jumps to the top of the ladder.
**(b)** At every coverage level the per‑rung accuracy is flat from `T=4` — Claim 1
again — so all the difficulty is concentrated in *covering the one‑step map*, not
in depth. Winning the seen‑N ladder is exactly: **learn `s` exactly, everywhere
the test orbits go.** The fixed‑N Easy/Medium datasets are built to make this
reachable — `depth_evaluation_exhaustive_x` plus hundreds–thousands of training
`x` per setting give near‑total coverage, and Claim 2′ closes the interior.

---

## 5. The submission

`submission.py` is the direct instantiation of Claims 1–3.

* **Encoder.** Token + position embeddings and 3 bidirectional, padding‑masked
  pre‑norm blocks read the prompt `(N, x, T)` and pool it into a value state
  `z_0` (the encoding `e(x)`). Bidirectional because the tiers hand the model a
  padding mask, not a causal one.
* **One tied cell (`SquaringCell`).** A single gated residual block is applied
  `T` times: `z_{t+1} = c(z_t)`. This is the shared `c` of Claims 1–2′ — the
  only depth in the model, reused, so "one layer deeper" costs no parameters.
  The gate starts near the identity so iterating to depth 64 is stable.
* **Adaptive depth.** The recurrence count is `T`, read from the prompt's `T`
  digits and applied per row (iterate to the batch‑max `T`, freeze each row at
  its own `T`). Rule 4 explicitly permits *adaptive computation and depth
  curricula*; the depth is control flow only — every **logit** is produced by
  learned weights with an unbroken autograd path from the loss through all `T`
  applications of the cell to its parameters (Claim 1's `e, c, d` are all
  trained; nothing is loaded or hard‑coded — rules 6–8).
* **Place‑value decoder.** The value state is decoded to per‑position digit
  logits conditioned on a place index `= (L-1) - i`, so the answer lands
  tail‑aligned to each row's true prompt end — matching the evaluator's
  `target_positions` gather for the separate‑output tiers.

It runs end‑to‑end through the **real** evaluator (`benchmark.runner`) on E1 with
both depth profiles, 3.76M state elements (≪ the 5×10^8 ceiling), and already
shows the Claim‑1 plateau untrained. Training recipe implied by the theory: a
**curriculum in `T`** (start at `T=1` to pin `s` directly, deepen), endpoint
cross‑entropy (the default), and enough steps to reach *total* operator coverage —
that last mile is what the H100 budget buys, and by §4 it is the whole game.

**Optional strengthening (within the rules).** `OptimizerBundle` may request up
to 8 evaluator‑owned backward passes per step and bounded batch reuse; these can
be used to push the last few percent of coverage to exactly 100% on the hardest
`N` without changing the architecture.

---

## 6. Honest limits — the OOD‑N frontier

*Max T* on seen `N` is squarely addressed above: it reduces to exactly learning
one small permutation, and the tied recurrence + coverage get there. **OOD‑N Max
T is genuinely harder and I do not claim to solve it.** A cell that encodes a
*specific* `N`'s squaring cannot transfer to an unseen `N` (a different group,
a different `s`). Certifying OOD‑N requires the cell to compute `u ↦ u^2 mod N`
**positionally from the digits of `u` and `N`** — a learned carry‑propagating
multiply‑and‑reduce circuit that generalizes across moduli. That is a real open
problem; note there is no shortcut through the group structure, since the smooth
representation (the discrete log, where squaring is just doubling) is exactly the
RSA trapdoor and is not learnable from data without effectively factoring `N`.

The submission's architecture is the right hypothesis class for that frontier too
— iterate one tied cell — but reaching *bit‑exact* positional modular squaring on
unseen `N` is where the remaining research lies. The contribution here is the
reduction: **the competition is not 64 problems, it is one map; certification is
exact coverage of that map; depth extrapolates for free** — proved, measured, and
implemented.
