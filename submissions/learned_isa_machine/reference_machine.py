"""Reference machine for FROZEN ISA SPEC v1 (learned-ISA tape machine).

Pure Python + stdlib. NO torch. This is the exact integer/bit-level semantics
that the learned model's hard-eval path must reproduce bit-for-bit.

Spec source: DESIGN_learned_isa.md, section "FROZEN ISA SPEC v1".

Machine summary
---------------
* Registers, width W bits, LSB-first:
    read-only  V (current value), N (modulus), ZERO, ONE
    writable   ACC, S1
  At the start of EVERY T-step: ACC = 0, S1 = 0; V = previous T-step's ACC
  (V = x at T-step 0). Cell output = ACC after the T-step.
* SCAN(a, b, table, init_state): fixed LSB->MSB recurrence over all W positions
  with a 2-state register s (s0 = init_state). Per position:
    (out_bit, s') = table[s, a_bit, b_bit]
  (8 entries, each a 4-way categorical over out x s'). Returns
  (out_register, terminal = s after position W-1).
* Program = K=8 slots, executed in slot order within their loop segment.
  Per-slot fields (arities in parens):
    loop_mode  {once_pre, head_A, head_B, once_post}          (4)
    dst        {ACC, S1}                                      (2)
    src_a      {V, N, ZERO, ONE, ACC, S1}                     (6)
    src_b      {V, N, ZERO, ONE, ACC, S1}                     (6)
    table_id   {0..3}  (L=4 shared tables)                    (4)
    init_state {0, 1}                                         (2)
    predicate  source {CONST, HEAD_BIT, TERMINAL} x polarity  (6)
  Slot semantics: r, term = SCAN(src_a, src_b, tables[table_id], init_state);
  if predicate holds: dst <- r (else dst unchanged). For once_* slots
  HEAD_BIT reads as 0.
* Head loops: segment head_A then head_B; each has fields source {V, ACC, S1}
  (3) and direction {MSB_first, LSB_first} (2). A segment executes its slots
  (in slot order) once per head position, W iterations; head bit = source
  register's bit at the current position (read at the start of each
  iteration, from the register's current contents).
* Execution order per T-step: once_pre slots; head_A loop; head_B loop;
  once_post slots.
* Outer loop: repeat the whole cell T times.

Tables are part of the Program here (in the learned model they are
categorical parameters -- same object).
"""

from dataclasses import dataclass, field, replace
from typing import List, Tuple
import random

# ---------------------------------------------------------------------------
# Field / enum constants (FROZEN ISA SPEC v1)
# ---------------------------------------------------------------------------

# loop_mode (arity 4)
LOOP_ONCE_PRE = 0
LOOP_HEAD_A = 1
LOOP_HEAD_B = 2
LOOP_ONCE_POST = 3
N_LOOP_MODES = 4

# dst (arity 2) -- writable registers only
DST_ACC = 0
DST_S1 = 1
N_DST = 2

# src_a / src_b (arity 6 each)
SRC_V = 0
SRC_N = 1
SRC_ZERO = 2
SRC_ONE = 3
SRC_ACC = 4
SRC_S1 = 5
N_SRC = 6

# table_id (arity 4): L = 4 shared tables
L_TABLES = 4

# init_state (arity 2)
N_INIT_STATES = 2

# predicate (arity 6) = source {CONST, HEAD_BIT, TERMINAL} x polarity {pos, neg}
# encoding: predicate = 2 * source + polarity   (polarity: 0 = pos, 1 = neg)
PRED_SRC_CONST = 0
PRED_SRC_HEAD_BIT = 1
PRED_SRC_TERMINAL = 2
POL_POS = 0
POL_NEG = 1

PRED_ALWAYS = 2 * PRED_SRC_CONST + POL_POS      # 0: CONST/pos
PRED_NEVER = 2 * PRED_SRC_CONST + POL_NEG       # 1: CONST/neg
PRED_HEAD_POS = 2 * PRED_SRC_HEAD_BIT + POL_POS  # 2: commit iff head bit == 1
PRED_HEAD_NEG = 2 * PRED_SRC_HEAD_BIT + POL_NEG  # 3: commit iff head bit == 0
PRED_TERM_POS = 2 * PRED_SRC_TERMINAL + POL_POS  # 4: commit iff terminal == 1
PRED_TERM_NEG = 2 * PRED_SRC_TERMINAL + POL_NEG  # 5: commit iff terminal == 0
N_PRED = 6

# head-loop fields: source (arity 3), direction (arity 2)
HEAD_SRC_V = 0
HEAD_SRC_ACC = 1
HEAD_SRC_S1 = 2
N_HEAD_SRC = 3

DIR_MSB_FIRST = 0
DIR_LSB_FIRST = 1
N_DIR = 2

# structural sizes
K_SLOTS = 8
TABLE_ENTRIES = 8       # index = (state << 2) | (a_bit << 1) | b_bit
TABLE_ENTRY_ARITY = 4   # 4-way categorical over (out_bit, next_state)
N_HEADS = 2             # head_A, head_B

# field-arity inventory (the search prototype must match this exactly)
SLOT_FIELD_ARITIES = [
    ("loop_mode", N_LOOP_MODES),   # 4
    ("dst", N_DST),                # 2
    ("src_a", N_SRC),              # 6
    ("src_b", N_SRC),              # 6
    ("table_id", L_TABLES),        # 4
    ("init_state", N_INIT_STATES), # 2
    ("predicate", N_PRED),         # 6
]
HEAD_FIELD_ARITIES = [
    ("source", N_HEAD_SRC),        # 3
    ("direction", N_DIR),          # 2
]
# total logit scalars = 8*(4+2+6+6+4+2+6) + 2*(3+2) + 4*8*4 = 240 + 10 + 128 = 378


# ---------------------------------------------------------------------------
# Program representation
# ---------------------------------------------------------------------------

# A table is a list of TABLE_ENTRIES (out_bit, next_state) integer pairs,
# indexed by (state << 2) | (a_bit << 1) | b_bit.
Table = List[Tuple[int, int]]


@dataclass
class Slot:
    loop_mode: int = LOOP_ONCE_POST
    dst: int = DST_ACC
    src_a: int = SRC_ZERO
    src_b: int = SRC_ZERO
    table_id: int = 0
    init_state: int = 0
    predicate: int = PRED_NEVER


@dataclass
class Head:
    source: int = HEAD_SRC_V
    direction: int = DIR_MSB_FIRST


@dataclass
class Program:
    slots: List[Slot] = field(default_factory=lambda: [Slot() for _ in range(K_SLOTS)])
    heads: List[Head] = field(default_factory=lambda: [Head() for _ in range(N_HEADS)])
    tables: List[Table] = field(default_factory=lambda: [xor_table() for _ in range(L_TABLES)])

    def copy(self) -> "Program":
        return Program(
            slots=[replace(s) for s in self.slots],
            heads=[replace(h) for h in self.heads],
            tables=[list(t) for t in self.tables],
        )


# ---------------------------------------------------------------------------
# Tables (reference semantics for the unit-test fixtures)
# ---------------------------------------------------------------------------

def table_index(state: int, a_bit: int, b_bit: int) -> int:
    return (state << 2) | (a_bit << 1) | b_bit


def add_table() -> Table:
    """ADD: out = (a + b + s) % 2, s' = (a + b + s) >= 2, init 0."""
    t: Table = [(0, 0)] * TABLE_ENTRIES
    for s in (0, 1):
        for a in (0, 1):
            for b in (0, 1):
                tot = a + b + s
                t[table_index(s, a, b)] = (tot % 2, 1 if tot >= 2 else 0)
    return t


def sub_table() -> Table:
    """SUB: out = (a - b - s) % 2, s' = (a - b - s) < 0, init 0.

    Terminal state after the full scan is the final borrow: 1 iff a < b
    (as W-bit unsigned integers).
    """
    t: Table = [(0, 0)] * TABLE_ENTRIES
    for s in (0, 1):
        for a in (0, 1):
            for b in (0, 1):
                d = a - b - s
                t[table_index(s, a, b)] = (d % 2, 1 if d < 0 else 0)
    return t


def xor_table() -> Table:
    """XOR filler table (out = a ^ b, state unchanged). Not referenced by
    any reference program; occupies unused table slots 2/3."""
    t: Table = [(0, 0)] * TABLE_ENTRIES
    for s in (0, 1):
        for a in (0, 1):
            for b in (0, 1):
                t[table_index(s, a, b)] = (a ^ b, s)
    return t


def copy_a_table() -> Table:
    """COPY-a filler table (out = a, state unchanged). Not referenced by
    any reference program; occupies unused table slot 3."""
    t: Table = [(0, 0)] * TABLE_ENTRIES
    for s in (0, 1):
        for a in (0, 1):
            for b in (0, 1):
                t[table_index(s, a, b)] = (a, s)
    return t


# Canonical table ids used by the reference programs.
TABLE_ADD = 0
TABLE_SUB = 1


def _reference_tables() -> List[Table]:
    return [add_table(), sub_table(), xor_table(), copy_a_table()]


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def default_width(modulus: int) -> int:
    """W = bitlen(N) + 2, so doubling a reduced value never overflows."""
    return modulus.bit_length() + 2


def scan(a_val: int, b_val: int, table: Table, init_state: int, width: int) -> Tuple[int, int]:
    """LSB->MSB recurrence over all `width` positions with a 2-state register.

    Returns (out_register, terminal_state)."""
    s = init_state
    out = 0
    for pos in range(width):
        idx = (s << 2) | (((a_val >> pos) & 1) << 1) | ((b_val >> pos) & 1)
        out_bit, s = table[idx]
        out |= out_bit << pos
    return out, s


def _predicate_holds(predicate: int, head_bit: int, terminal: int) -> bool:
    if predicate == PRED_ALWAYS:
        return True
    if predicate == PRED_NEVER:
        return False
    if predicate == PRED_HEAD_POS:
        return head_bit == 1
    if predicate == PRED_HEAD_NEG:
        return head_bit == 0
    if predicate == PRED_TERM_POS:
        return terminal == 1
    if predicate == PRED_TERM_NEG:
        return terminal == 0
    raise ValueError(f"invalid predicate {predicate}")


def execute(program: Program, modulus: int, value: int, time_steps: int, width: int = None) -> int:
    """Exact integer/bit-level simulation of the machine per FROZEN ISA SPEC v1.

    Returns ACC (as a W-bit unsigned integer) after the final T-step.
    """
    if width is None:
        width = default_width(modulus)
    mask = (1 << width) - 1
    n_val = modulus & mask
    v = value & mask

    # slot indices per loop segment, in slot order
    seg_pre = [i for i, s in enumerate(program.slots) if s.loop_mode == LOOP_ONCE_PRE]
    seg_a = [i for i, s in enumerate(program.slots) if s.loop_mode == LOOP_HEAD_A]
    seg_b = [i for i, s in enumerate(program.slots) if s.loop_mode == LOOP_HEAD_B]
    seg_post = [i for i, s in enumerate(program.slots) if s.loop_mode == LOOP_ONCE_POST]

    for _ in range(time_steps):
        acc = 0
        s1 = 0

        def read(src: int) -> int:
            if src == SRC_V:
                return v
            if src == SRC_N:
                return n_val
            if src == SRC_ZERO:
                return 0
            if src == SRC_ONE:
                return 1
            if src == SRC_ACC:
                return acc
            if src == SRC_S1:
                return s1
            raise ValueError(f"invalid src {src}")

        def run_slot(slot_idx: int, head_bit: int) -> None:
            nonlocal acc, s1
            slot = program.slots[slot_idx]
            a = read(slot.src_a)
            b = read(slot.src_b)
            r, term = scan(a, b, program.tables[slot.table_id], slot.init_state, width)
            if _predicate_holds(slot.predicate, head_bit, term):
                if slot.dst == DST_ACC:
                    acc = r
                else:
                    s1 = r

        def read_head_source(src: int) -> int:
            if src == HEAD_SRC_V:
                return v
            if src == HEAD_SRC_ACC:
                return acc
            if src == HEAD_SRC_S1:
                return s1
            raise ValueError(f"invalid head source {src}")

        # once_pre slots (HEAD_BIT reads as 0)
        for i in seg_pre:
            run_slot(i, 0)

        # head_A then head_B loops
        for seg, head in ((seg_a, program.heads[0]), (seg_b, program.heads[1])):
            if not seg:
                continue  # empty segment: W iterations of nothing
            for it in range(width):
                pos = (width - 1 - it) if head.direction == DIR_MSB_FIRST else it
                head_bit = (read_head_source(head.source) >> pos) & 1
                for i in seg:
                    run_slot(i, head_bit)

        # once_post slots (HEAD_BIT reads as 0)
        for i in seg_post:
            run_slot(i, 0)

        v = acc  # V <- ACC between T-steps; cell output = ACC after T-step

    return v


# ---------------------------------------------------------------------------
# Reference programs (unit-test fixtures from the spec)
# ---------------------------------------------------------------------------

def _nop_slot() -> Slot:
    return Slot(loop_mode=LOOP_ONCE_POST, dst=DST_ACC, src_a=SRC_ZERO,
                src_b=SRC_ZERO, table_id=TABLE_ADD, init_state=0,
                predicate=PRED_NEVER)


def squaring_program() -> Program:
    """x -> x^2 mod N per T-step (MSB-first double-and-add over V's bits).

    head_A src V MSB_first; slots(head_A):
      [ACC=ADD(ACC,ACC) always], [ACC=SUB(ACC,N) TERMINAL/neg],
      [ACC=ADD(ACC,V) HEAD_BIT/pos], [ACC=SUB(ACC,N) TERMINAL/neg];
    others commit never.
    """
    slots = [
        Slot(LOOP_HEAD_A, DST_ACC, SRC_ACC, SRC_ACC, TABLE_ADD, 0, PRED_ALWAYS),
        Slot(LOOP_HEAD_A, DST_ACC, SRC_ACC, SRC_N, TABLE_SUB, 0, PRED_TERM_NEG),
        Slot(LOOP_HEAD_A, DST_ACC, SRC_ACC, SRC_V, TABLE_ADD, 0, PRED_HEAD_POS),
        Slot(LOOP_HEAD_A, DST_ACC, SRC_ACC, SRC_N, TABLE_SUB, 0, PRED_TERM_NEG),
        _nop_slot(), _nop_slot(), _nop_slot(), _nop_slot(),
    ]
    heads = [Head(HEAD_SRC_V, DIR_MSB_FIRST), Head(HEAD_SRC_V, DIR_MSB_FIRST)]
    return Program(slots=slots, heads=heads, tables=_reference_tables())


def affine_program() -> Program:
    """x -> (3x + 1) mod N per T-step. All active slots once_pre:
      [ACC=ADD(V,V) alw], [ACC=SUB(ACC,N) TERM/neg], [ACC=ADD(ACC,V) alw],
      [ACC=SUB(ACC,N) TERM/neg], [ACC=ADD(ACC,ONE) alw], [ACC=SUB(ACC,N) TERM/neg];
    2 spare never.
    """
    slots = [
        Slot(LOOP_ONCE_PRE, DST_ACC, SRC_V, SRC_V, TABLE_ADD, 0, PRED_ALWAYS),
        Slot(LOOP_ONCE_PRE, DST_ACC, SRC_ACC, SRC_N, TABLE_SUB, 0, PRED_TERM_NEG),
        Slot(LOOP_ONCE_PRE, DST_ACC, SRC_ACC, SRC_V, TABLE_ADD, 0, PRED_ALWAYS),
        Slot(LOOP_ONCE_PRE, DST_ACC, SRC_ACC, SRC_N, TABLE_SUB, 0, PRED_TERM_NEG),
        Slot(LOOP_ONCE_PRE, DST_ACC, SRC_ACC, SRC_ONE, TABLE_ADD, 0, PRED_ALWAYS),
        Slot(LOOP_ONCE_PRE, DST_ACC, SRC_ACC, SRC_N, TABLE_SUB, 0, PRED_TERM_NEG),
        _nop_slot(), _nop_slot(),
    ]
    heads = [Head(HEAD_SRC_V, DIR_MSB_FIRST), Head(HEAD_SRC_V, DIR_MSB_FIRST)]
    return Program(slots=slots, heads=heads, tables=_reference_tables())


def cube_program() -> Program:
    """x -> x^3 mod N per T-step.

    head_A src V MSB_first computes S1 = V^2 mod N:
      [S1=ADD(S1,S1) alw], [S1=SUB(S1,N) TERM/neg],
      [S1=ADD(S1,V) HEAD_BIT/pos], [S1=SUB(S1,N) TERM/neg];
    head_B src V MSB_first computes ACC = V * S1 = V^3 mod N:
      [ACC=ADD(ACC,ACC) alw], [ACC=SUB(ACC,N) TERM/neg],
      [ACC=ADD(ACC,S1) HEAD_BIT/pos], [ACC=SUB(ACC,N) TERM/neg].
    """
    slots = [
        Slot(LOOP_HEAD_A, DST_S1, SRC_S1, SRC_S1, TABLE_ADD, 0, PRED_ALWAYS),
        Slot(LOOP_HEAD_A, DST_S1, SRC_S1, SRC_N, TABLE_SUB, 0, PRED_TERM_NEG),
        Slot(LOOP_HEAD_A, DST_S1, SRC_S1, SRC_V, TABLE_ADD, 0, PRED_HEAD_POS),
        Slot(LOOP_HEAD_A, DST_S1, SRC_S1, SRC_N, TABLE_SUB, 0, PRED_TERM_NEG),
        Slot(LOOP_HEAD_B, DST_ACC, SRC_ACC, SRC_ACC, TABLE_ADD, 0, PRED_ALWAYS),
        Slot(LOOP_HEAD_B, DST_ACC, SRC_ACC, SRC_N, TABLE_SUB, 0, PRED_TERM_NEG),
        Slot(LOOP_HEAD_B, DST_ACC, SRC_ACC, SRC_S1, TABLE_ADD, 0, PRED_HEAD_POS),
        Slot(LOOP_HEAD_B, DST_ACC, SRC_ACC, SRC_N, TABLE_SUB, 0, PRED_TERM_NEG),
    ]
    heads = [Head(HEAD_SRC_V, DIR_MSB_FIRST), Head(HEAD_SRC_V, DIR_MSB_FIRST)]
    return Program(slots=slots, heads=heads, tables=_reference_tables())


# ---------------------------------------------------------------------------
# Random program sampling (uniform over ALL fields including tables)
# ---------------------------------------------------------------------------

def random_program(rng: random.Random) -> Program:
    slots = []
    for _ in range(K_SLOTS):
        slots.append(Slot(
            loop_mode=rng.randrange(N_LOOP_MODES),
            dst=rng.randrange(N_DST),
            src_a=rng.randrange(N_SRC),
            src_b=rng.randrange(N_SRC),
            table_id=rng.randrange(L_TABLES),
            init_state=rng.randrange(N_INIT_STATES),
            predicate=rng.randrange(N_PRED),
        ))
    heads = [Head(rng.randrange(N_HEAD_SRC), rng.randrange(N_DIR))
             for _ in range(N_HEADS)]
    tables = []
    for _ in range(L_TABLES):
        t: Table = []
        for _ in range(TABLE_ENTRIES):
            e = rng.randrange(TABLE_ENTRY_ARITY)  # 4-way over (out, next)
            t.append((e >> 1, e & 1))
        tables.append(t)
    return Program(slots=slots, heads=heads, tables=tables)
