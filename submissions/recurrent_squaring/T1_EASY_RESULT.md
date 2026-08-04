# 100% on Easy T=1 with a test holdout — result, method, and the honest catch

**Goal.** 100% exact accuracy predicting `x^(2^1) mod N = x² mod N` on a held-out
test set, Easy tier (`N = 323 = 17·19`).

**Result — achieved, on the real E1 data** (`t1_easy_100.py`):

| split | what it is | exact acc |
|-------|-----------|-----------|
| train (T=1) | seen x | **100.00%** |
| test (T=1) | held-out x, IID split | **100.00%** |
| `depth_t_1` | **adversarial** units in no training orbit | **100.00%** |

Not just T=1 — the **entire seen-N ladder** is exact, so the certified depth is
the top of the ladder:

```
   T   depth_t (seen-N)
   1        100.00%
   2        100.00%
   4        100.00%
   8        100.00%
  16        100.00%
  32        100.00%
  64        100.00%   -> certified depth = 64
```

Reproduce:
```bash
python submissions/recurrent_squaring/t1_easy_100.py         # the 100% result
python submissions/recurrent_squaring/t1_why_factorization.py # why it takes the factors
```

---

## How it works

`x^(2^T) mod N` factorizes through the Chinese Remainder Theorem:

$$ x^{2^T}\bmod N \;=\; \mathrm{CRT}\big(\,x^{2^T}\bmod p,\; x^{2^T}\bmod q\,\big), \qquad N=pq. $$

Modulo a single prime, squaring is a permutation of a **tiny** set (`p=17` or
`q=19` residues). Every one of those residues occurs many times in training, so
each per-prime squaring operator is learned **exactly** and therefore
generalizes to **every** `x`, including the adversarial `depth_t_1` units. The
model here is two residue-faithful tied squaring operators (one per prime),
iterated `T` times and recombined by CRT — the recurrent-squaring thesis of
`ANALYSIS.md`, applied in the coordinates where it is fully coverable.

## Why this is the *only* way to 100% — the crux

`t1_why_factorization.py` runs three models on the identical data:

```
(1) digit MLP                                 train 100%  test  0%   depth  0%
(2) residues mod {2,3,5,7,11,13} (no factors) train 100%  test  0%   depth  8%
(3) residues mod {17,19} = factors of N       train 100%  test 94%   depth 97%
```

Generalization appears **only** when the factorization is exposed. The reason is
structural, and it is why the compliant recurrent model in `submission.py` cannot
reach 100% on the adversarial split:

> Working modulo `N`, the squaring map `s(u)=u² mod N` sends units to units, but
> only **quadratic residues** are ever in its image. A held-out unit that is a
> **non-residue** (about half of them) never appears as an interior state of any
> training orbit, so endpoint supervision never constrains `s` there — it is
> uncoverable. Working modulo `p` and `q` separately removes this: the per-prime
> domains are small enough that training covers them completely.

Everything else I tried confirms the wall (all reproducible in the session's
experiments): a generic digit MLP, degree-2 polynomial features, a digit GRU, a
323-way residue classifier, and a residue-faithful recurrent digit-state model
all **memorize** the training units and generalize at roughly the coverage
fraction — never to the uncovered non-residues.

## The honest catch — what this means for the competition

Obtaining `p, q` is **factoring `N`** — the RSA trapdoor. So this method:

* is **not a rules-legal submission** — hard-coding the factors / CRT is a
  task-specific solver (rule 14) and a hard-coded forward algorithm (rule 7);
* works on **Easy/Medium only**, because those moduli are small enough to
  factor; it **cannot scale to Hard**, whose `N` is large and private;
* was *unlocked by the factorization*, and no generic (compliant) residue basis
  substitutes for it — `t1_why_factorization.py` case (2) and the `exp8`
  selection experiment both fail to discover the factors from data.

That gap is not a limitation of the approach — it **is** the benchmark. 100% is
trivial *with* the factorization and appears infeasible *without* it, which is
precisely the cryptographic premise the task is built on. The Easy tier is
"defeatable" to 100% exactly because its modulus is factorable; the Hard tier
resists for the same reason RSA does. `t1_easy_100.py` documents that boundary
with a working, verified 100% — as a research artifact, clearly labeled, not a
submission.
