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

## In flight

- Round 3a: credit ladder for the squaring core — multiple-of-x credit (any partial
  double-and-add computes x·m; the loss can compute this from inputs alone), then
  congruence, then exact; CRT credit as a science control.
- Round 3b: affine stage-2 completion (joint slot×table-dict blocks from the stage-1
  winner; protected-core GA; existence check of reducer completions in the winner's
  neighborhood).
