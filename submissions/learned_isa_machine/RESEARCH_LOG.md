# Learned-ISA machine — research log

Goal: a rule-7-clean successor to `binary_relation_hard_control` — the algorithm itself
(not just switches inside a supplied interpreter) must be learned from endpoints, and the
same source must be able to learn squaring, affine (3v+1 mod N), and cube (the
hidden-recurrence probes).

## The machine (ISA spec v1.1)

Generic two-loop register machine; nothing arithmetic named in Python:
- Registers (width W, LSB-first bits): read-only V (current value), N, ZERO, ONE;
  writable ACC, S1. Per T-step: ACC=S1=0, V = previous step's ACC (x initially).
- One primitive: SCAN(a, b, table, init_state) — LSB→MSB sweep with a 2-state
  finite-state transducer; the 8-entry table (state,a_bit,b_bit)→(out_bit,next_state)
  is a free 4-way categorical per entry. L=4 learned tables.
- Program = 8 two-address slots (src_a ≡ dst, a standard ISA convention; 12× space cut):
  loop_mode{once_pre,head_A,head_B,once_post} × dst{ACC,S1} × src_b{V,N,ZERO,ONE,ACC,S1}
  × table_id{4} × init{2} × predicate{always,never,head_bit,¬head_bit,terminal,¬terminal}
  = 2,304 combos/slot. Two head loops with learned source {V,ACC,S1} and direction.
- ~378 categorical logits total; conditional-subtract reduction and double-and-add are
  expressible but must EMERGE (verified: reference programs for all three families pass
  G0 tests in `test_reference_machine.py`).

## Established results (2026-08-06)

1. G0 expressibility: PASS for squaring (4 slots), affine (7), cube (8), incl. under
   two-address tying. C1 injectivity: only ~4.8% of random program pairs compute
   identical functions. C2: every active field of the squaring reference is
   counterfactually necessary up to semantic aliases (`disagreement_sweep.py`).
2. Mean-field rotating-block marginalization (rules-compatible form, `search_prototype.py`)
   with sub-slot blocks: PLATEAUS at ~3% exact / bit-match ≈0.67-0.70 on N=323 squaring
   (E1, E3 with annealed sampling + restarts + curriculum all alike).
3. SA landscape probe (scratchpad sa_probe.py): the landscape has a gradient skin only
   1-2 functional mutations deep; 81% of single-field moves are score-neutral; a
   near-constant-output attractor sits at uniform bit-match 0.63-0.70 (treat as the
   entropy floor). SA is WORSE than greedy (noise dissolves partial solutions). Frozen
   ADD/SUB tables make affine solvable by vanilla SA; squaring not.
4. Round 2 (genome-axis vectorized executor, two-address space):
   - Maximal-block best-response rotation (slot-joint 2,304; table 4,096-sample; head 36):
     fixed point at the attractor within 2-3 sweeps, every seed, every score tried.
   - GA (pop 256, slot-granular crossover) + plain exact credit: attractor.
   - **GA + CONGRUENCE CREDIT (score output mod N vs label on a wide tape, T=1 rows):
     DISCOVERED the affine core from random init** — a program computing literal
     unreduced 3x+1 for all 323 inputs, found in 21k evals / 90 s, and it is a NOVEL
     construction (two head loops, custom tables), not the hand-written reference.
     Success 1/3 seeds. Congruence credit re-expresses labels inside the loss only;
     the forward machine is untouched — this ports to a rules-compatible loss.
   - Stage 2 (assembling the mod-N reducer around the found core): OPEN. It is a
     (slot × table)-wide needle; single-block moves hit fixed points.
   - Squaring core: UNSOLVED by all of the above (0/31 runs > bit-match 0.75) —
     partial multiplication circuits earn no congruence credit.

## Porting guidance (rules-compatible training), from round 2

- Loss: per-bit likelihood on (machine_output mod N) vs target with wide tape
  (W = 2·bitlen(N)+4); curriculum T=1 congruence → anneal in T≥2 + exact term.
  Precedent for numeric label re-expression: the sequence-value loss rules audit.
- Blocks: slot-joint (2,304) enumerations; ties → incumbent; sweep count is NOT the
  budget knob (fixed points come fast).
- Replace GA selection/crossover with ≥16-64 cheap chains + cross-chain block
  resampling (enumerate other chains' values of a block as the alternative set,
  weight by exact likelihood) — enumerable, marginalization-compatible.
- Stage 2 needs (slot × table-dictionary) JOINT blocks (~37k alternatives).

## Round 3a (COMPLETE): loss design for the squaring core is exhausted — NEGATIVE

High-confidence negative across 5 credit designs (uniform, congruence, multiple-of-x
ladder, CRT 17/19 decomposition, multi-T composition): every GA stalls at the same
congruence wall 0.66-0.69. Mechanism: a partial double-and-add emits x·g(x) with g
uncorrelated per row, so NO per-row functional of (x, N, y, output) can reward "half a
multiplier"; multiple-of-x credit is itself a trivial attractor (constants are exact
multiples; random init already scores 0.93). Basin check: greedy recovers the true core
from k<=4 corrupted fields (7/8) — the basin exists; per-row-credit search cannot reach
it. Conclusion: stop tuning the loss; change the search structure. (Affine solved in
round 2 precisely because partial affine programs ARE visible to congruence credit.)

## Round 3b (COMPLETE): the found affine core is PROVABLY irreducible — and the fix

Exhaustive existence checks (20,640 canonical completions; 2.36M slot7×all-tables
candidates; a causality screen over all 109 reachable S1 preloads) prove NO reducer
completion exists in the seed-600 core's neighborhood: its integer materializes at
once_post slot6 with only slot7 downstream, and the subtraction bands are not decodable
from any reachable state. Search-dynamics corroboration: every congruence-weighted
mixture makes the congruent core a local optimum; pure-exact credit re-enters the
bitmatch attractor. THE FIX (ports directly): (1) stage-1 credit = congruence + range
bonus (output < N) — endpoint-computable, kills the unreduced attractor; (2) reserve two
trailing slots (pinned NEVER in stage 1) so a conditional-subtract completion always has
room; (3) stage 2 becomes an exact enumeration (~20k candidates, seconds) — the
exhaustive check showed the canonical conditional-subtract pair is the global-optimum
completion over ALL 65,536 tables.

## MILESTONE (2026-08-06): first complete certified program discovered end-to-end

Affine (3x+1 mod 323), endpoint-only credit, random init: stage 1 (GA, congruence +
0.15*range-bonus, slots 6-7 reserved) found a congruence-exact core in 22 generations /
5,633 evals / 20 s (1/4 seeds); stage 2 (tier-1 conditional-subtract enumeration, 30,912
candidates, 645 s) produced 30 complete programs. Winner certified: exact on ALL 323 x at
T in {1,2,3,6}; width-robust (W=24, 30); necessity sweep 41 fully load-bearing fields,
every active slot counterfactually necessary. The modular reducer EMERGED as two
TERM-gated subtract slots. Novel construction, not the reference. Total ~503k evals /
~36 min on one shared CPU worker. Artifacts: scratchpad sa_logs/round4_affine_complete.txt.

## Round 4 (squaring core) — QD ruled out; two-level enumeration is the path

- MAP-Elites QD: FINAL negative (657 iters / 338k evals, best congruence 0.688, no cell
  >= 0.70). Stepping stones do not exist under per-row credit for multiplication.
- Template enumeration + conditional table learning: mechanism VALIDATED — given the true
  2-slot structure, exhaustive table mapping shows a graded cone around the working ADD
  table; randomized steepest descent detects it at ~2.3-7k evals/structure (R95 ~ 94
  restarts). The blinded 800-structure test under-budgeted restarts (R=32; true structure
  ranked 144). Calibrated full cost: 8.3M two-active-slot structures x ~6.9k evals ~ 5.8e10
  machine evals — H100 Hard-budget territory before pruning (MDL order, canonicalization,
  early-kill all individually validated).

## In flight / next

1. CPU-scale proof of the complete squaring pipeline (pruned structure space, R~94 table
   restarts, then the validated stage-2 reducer enumeration).
2. The port: rules-compatible submission.py — enumeration + exact marginalization in the
   loss with per-structure table adaptation; validate end-to-end on the affine probe
   manifest first.
- Round 4 (squaring core, search-structure levers): M1 quality-diversity (MAP-Elites
  archive over behavior descriptors — stepping stones instead of fitness-only
  selection); M2 MDL-ordered template enumeration with conditional per-entry table
  best-response (brute-force the structure side, coordinate-learn tables given
  structure; H100 feasibility estimate is a deliverable).
