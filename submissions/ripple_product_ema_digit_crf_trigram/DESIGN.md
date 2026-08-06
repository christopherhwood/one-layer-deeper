# Terminal observed-digit trigram CRF

## Controlled extension

This candidate starts from the optimized final-only observed-digit CRF. The
product-aware reducer, four tied refinement scans, factorized outer register,
three-step training recurrence with detach, EMA student/teacher computation,
endpoint CE, T=1 phase CE, endpoint consistency, first-order unary/edge/start/end
energies, CRF weight, schedules, dynamic evaluation bound, and evaluation batch
size 512 are unchanged.

Only the selected student terminal reducer state receives one additional
structured-output factor:

```text
terminal states, MSD first
       |
 [s_i, s_i+1, s_i+2]
       |
 Linear(3H, 16) -> SiLU -> zero-init Linear(16, 1000)
       |
 tau_i[d_i, d_i+1, d_i+2]
```

The trigram head is outside the EMA square and runs once after each row's final
outer state has been selected. It is student-only: the temporal teacher still
teaches the unchanged factorized register and carries no unused decoder copy.

## Second-order observed-digit chain

For fixed-width MSD-first digits `y`, the score is the retained first-order
score plus

```text
sum_i tau_i[y_i, y_i+1, y_i+2].
```

The forward recurrence stores a pair `(a,b)` of previous decimal digits and
reduces over `a` when extending with digit `c`. Viterbi replaces log-sum-exp
with max and stores the selected preceding digit. Complexity is
`O(W * 10^3)` and therefore linear in length; the pair state has only 100
entries. Every indexed state is an actual observed digit. There is no latent
carry, quotient, reduction choice, or free semantic state.

Variable-length labels constrain the right-aligned supplied suffix. The
unobserved fixed-width prefix remains unrestricted. The loss is the full log
partition minus the partition of all paths consistent with that suffix, so it
does not invent leading-zero targets or expose target length to forward/Viterbi.

## Exact nesting and cost

The final `16 -> 1000` projection has no bias and is initialized to zero. Thus
all initial trigram energies vanish while every retained first-order energy is
unchanged. Widths below three call the original first-order partition and
Viterbi directly; the width-two smoke task is an exact compatibility contract,
not a test of trigram learning.

At `H=128`, the student-only head adds 22,160 parameters, taking E5 persistent
state from 777,476 to 799,636 elements. At E5 width four there are only two
trigram factors per row. The head and second-order DP run only at the terminal
endpoint, never on refinement phases or EMA teacher passes.

## Risk and decision rule

This is a structured decoder-capacity ablation, not a new reduction mechanism.
Correct modular residues have little stable local trigram grammar. The head can
help if the reducer already represents the right digits but its independent or
pairwise scores choose locally incompatible combinations. It can also merely
memorize input-conditioned three-digit patterns while leaving comparison,
carry, quotient, and OOD-N generalization unsolved.

The variable-E5 screen must show a held-out improvement over the replicated
first-order CRF before a full gate is justified. Fixed-modulus smoke movement
cannot support that decision because width two contains no trigram factor.
