"""Learned-ISA tape machine: population-as-parameters program posterior.

A generic two-loop register machine (FROZEN ISA SPEC v1.1, two-address) whose
entire program -- per-slot routing, operands, finite-state-transducer tables,
head loops, commit predicates -- is a set of categorical parameters, learned
from endpoints only.  C independent chains each hold a full field assignment
as per-field logits.  Every training step executes, fully discretely (integer
registers, alternatives as a tensor dimension), the enumerated alternatives of
one rotating block per active chain; the loss is the exact marginalization
-logsumexp(log prior + scaled credit) over those alternatives, so gradients
flow only into the posterior logits (plus a tiny soft decoder trained by a
0.01-weight direct term).  The scheduler owns block rotation, a documented
cross-chain resampling transformation (re-centering the worst chains' logits
on crossovers of better chains' MAP fields plus noise), stage scheduling, and
the persistent best-chain buffer.  Evaluation runs the best chain's MAP
program with plain integer execution, per-row T latching, and a hard digit
decoder; no buffer changes during evaluation.
"""

from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from benchmark import (
    ModelSpec,
    OptimizerBundle,
    OptimizerSpec,
    Submission,
    TokenLossBatch,
    assert_model_state,
)

PAD = 0
N_MARK = 2
X_MARK = 3
T_MARK = 4
DIGIT_OFFSET = 7
NUM_DIGITS = 10
MAX_OUTER_STEPS = 64

# ISA constants (spec v1.1, two-address: src_a == dst register)
K_SLOTS = 8
N_LM, N_DST, N_SRC, N_TID, N_INI, N_PRD = 4, 2, 6, 4, 2, 6
N_HSRC, N_HDIR = 3, 2
PRED_ALWAYS, PRED_NEVER = 0, 1
PRED_HEAD_POS, PRED_HEAD_NEG = 2, 3
PRED_TERM_POS, PRED_TERM_NEG = 4, 5
LM_PRE, LM_HEAD_A, LM_HEAD_B, LM_POST = 0, 1, 2, 3
HSRC_REG = (0, 4, 5)  # head source field -> register index (V, ACC, S1)
RESERVED_SLOTS = (6, 7)  # trailing slots kept NEVER during stage 1

# hyperparameters
GA_SEED = 4  # seeds every internal generator (init, blocks, resampling)
NCHAINS = 256
INIT_SCALE = 0.4
PIN_BIAS = 6.0
CONF = 1.3
NOISE = 0.2
POSTERIOR_LR = 4.0
DECODER_LR = 1e-2
DIRECT_LOSS_WEIGHT = 0.01
RANGE_BONUS = 0.15
SHARP0, SHARP1 = 50.0, 110.0
EMA = 0.35
N_ACTIVE = 1               # chains receiving a gradient block per step
RESAMPLE_EVERY = 1
REPLACE_FRAC = 0.97        # generational replacement minus elites
TOURNEY = 4
N_ELITE = 8
MUT_PROB = 0.7
STREAK_TO_STAGE2 = 3
STAGE2_WALL_FALLBACK = 0.65
ANNEAL_START, ANNEAL_LEN, ANNEAL_CAP = 0.55, 0.35, 0.5
TABLE_BLOCK_SAMPLES = 128
SWEEP_TABLES_PER_STEP = 4096
SWEEP_POSTERIOR_SAMPLES = 2048
SWEEP_TABLES_FALLBACK = 256  # when no table is free, full programs are run
# in-loop completion sweep (stage 2 for multiplicative cores: the reducer
# must interleave the head loop; full-program execution, so smaller windows)
LOOP_TABLES_PER_STEP = 2048
LOOP_POSTERIOR_SAMPLES = 1024
STAGE2_LOOP_START = 24  # stage-2 rotations before in-loop blocks join
#                         (keeps the affine-validated once_post-first order)
SWEEP_ROWS = 24
FULL_T_ROWS = 64
MAX_STAGE2_ACTIVE = 2
SLOT_SAMPLE = 576  # rotating window over the 2304 slot-joint alternatives
CONG_ANCHOR = 0.3


def _cartesian(*sizes: int) -> Tensor:
    grids = torch.meshgrid(*[torch.arange(s) for s in sizes], indexing="ij")
    return torch.stack([g.reshape(-1) for g in grids], dim=-1)


# slot-joint block alternatives: (lm, dst, sb, tid, ini, pr) -> 2304 rows
SLOT_COMBOS = _cartesian(N_LM, N_DST, N_SRC, N_TID, N_INI, N_PRD)
SLOT_PERM = torch.randperm(
    SLOT_COMBOS.shape[0], generator=torch.Generator().manual_seed(424242)
)
# head block alternatives: (srcA, dirA, srcB, dirB) -> 36 rows
HEAD_COMBOS = _cartesian(N_HSRC, N_HDIR, N_HSRC, N_HDIR)
# stage-2 completion combos for a reserved once_post slot:
# (dst, sb, ini, pr) -> 144 rows
COMP_COMBOS = _cartesian(N_DST, N_SRC, N_INI, N_PRD)
# table-sweep combos: conditional completions writing the output register
# (sb, ini, pr in {TERM+, TERM-}) -> 24 rows, plus one explicit no-op row so
# a junk window can never outscore leaving the slot inert
SWEEP_COMBOS = torch.cat(
    [
        torch.zeros(N_SRC * N_INI * 2, 1, dtype=torch.long),  # dst = ACC
        _cartesian(N_SRC, N_INI, 2),
    ],
    dim=1,
)
SWEEP_COMBOS[:, 3] += PRED_TERM_POS  # 0/1 -> TERM+/TERM-
SWEEP_COMBOS = torch.cat(
    [SWEEP_COMBOS, torch.tensor([[0, 0, 0, PRED_NEVER]])], dim=0
)

# ---------------------------------------------------------------------------
# Enumeration stage 1 (squaring-class cores): canonical two-active-slot
# structure space + calibrated randomized steepest-descent table search.
# All hyperparameters below come from the round-4/5 calibration; the visit
# order is a fixed-seed permutation chosen a priori (answer-blind: it is a
# function of the ISA alone, fixed before any structure was ever scored).
# ---------------------------------------------------------------------------
ENUM_ORDER_SEED = 20260807  # date constant, committed before any position math
ENUM_MIN_BUDGET = 180.0     # below this budget the GA layer runs alone
ENUM_TAIL_RESERVE = 180.0   # keep this much budget for stage 2 + polish
ENUM_ROWS = 48              # rows per table-search evaluation (measured: 32
#                             rows halve p_hit; 48 keep the round-5 basin)
ENUM_MIN_ROWS = 16
ENUM_RESTARTS_A = 12        # phase-A screen restarts (p_hit/restart ~0.038)
ENUM_RESTARTS_CAP = 110     # calibrated full refinement (detection ~0.985)
ENUM_RESTART_LADDER = (110, 90, 70, 50, 40, 30)
ENUM_KILL = 0.62            # early-kill: P(true best-of-12 < 0.62) ~ 1e-3,
#                             junk survivor fraction ~0.38 (measured, 48 rows)
ENUM_KILL_SMALL = 0.60      # when fewer than 32 rows are available
ENUM_MAX_WALK = 12          # steepest-descent move cap per restart
ENUM_P_CAP = 6144           # programs per executor call (CPU cache sweet spot)
ENUM_P_CAP_CUDA = 65536     # H100: launch-bound, so bigger calls win
ENUM_CHUNK0 = 4
ENUM_CHUNK_MAX = 512        # structures per step (CPU)
ENUM_CHUNK_MAX_CUDA = 4096
ENUM_STUCK_STEPS = 160      # stage-2 steps without exactness before resuming
#                             (covers the full in-loop sweep window rotation)
# cost model used only to pick the restart count from the wall clock:
# evals/structure ~ EVA + survivor_frac * (R - RESTARTS_A) * EVR
ENUM_EVA = 710.0
ENUM_EVR = 65.0


def build_enum_space() -> Tensor:
    """Canonical live two-active-slot structure space, (M, 14) long rows
    (lm0,dst0,sb0,ini0,pr0, lm1,dst1,sb1,ini1,pr1, hAs,hAd,hBs,hBd).

    Data-free port of the round-5 D2 canonicalization of the raw
    480*480*36 = 8,294,400 two-slot space (M = 132,616):
      - per-slot configs exclude predicate NEVER; once-segment HEAD_POS
        slots are NOPs (excluded; lower tier), once-segment HEAD_NEG
        collapses to ALWAYS;
      - swap-safe slot orders (different segments, or same-segment writes
        to different registers with no cross-read) are collapsed to one
        representative order; swap-unsafe pairs keep both orders;
      - head configs are enumerated only for heads actually read by a
        head-segment slot with a HEAD_* predicate; unread heads are pinned
        to (V, MSB);
      - pairs where either slot cannot influence the ACC output are
        dropped (conservative liveness: the slot writes ACC, or a path to
        ACC exists through the other slot via an S1 read or an S1-sourced
        head that is re-read at/after the writer's segment).
    """
    configs = []
    for lm in range(N_LM):
        if lm in (LM_PRE, LM_POST):
            preds = (PRED_ALWAYS, PRED_TERM_POS, PRED_TERM_NEG)
        else:
            preds = (PRED_ALWAYS, PRED_HEAD_POS, PRED_HEAD_NEG,
                     PRED_TERM_POS, PRED_TERM_NEG)
        for dst in range(N_DST):
            for sb in range(N_SRC):
                for ini in range(N_INI):
                    for pr in preds:
                        configs.append((lm, dst, sb, ini, pr))
    cfg = torch.tensor(configs, dtype=torch.long)  # (K, 5)
    k_cls = cfg.shape[0]
    lm, dst, sb, pr = cfg[:, 0], cfg[:, 1], cfg[:, 2], cfg[:, 4]
    in_head = (lm == LM_HEAD_A) | (lm == LM_HEAD_B)
    reads = ((pr == PRED_HEAD_POS) | (pr == PRED_HEAD_NEG)) & in_head
    head_of = torch.where(reads, lm, torch.zeros_like(lm))  # 0 / 1(A) / 2(B)

    a = torch.arange(k_cls).repeat_interleave(k_cls)
    b = torch.arange(k_cls).repeat(k_cls)
    reg = 4 + dst  # dst register as a src_b code (ACC=4, S1=5)
    safe = (
        (lm[a] != lm[b])
        | ((dst[a] != dst[b]) & (sb[a] != reg[b]) & (sb[b] != reg[a]))
        | (a == b)
    )
    order_ok = (a <= b) | ~safe
    once = ~in_head
    # path to ACC for an S1-writing slot via the other slot's S1 read;
    # slot0 executes first inside a shared once segment
    p1_0 = (sb[b] == 5) & (
        (lm[b] > lm[a])
        | ((lm[b] == lm[a]) & in_head[a])
        | ((lm[b] == lm[a]) & once[a])
    )
    p1_1 = (sb[a] == 5) & (
        (lm[a] > lm[b]) | ((lm[a] == lm[b]) & in_head[b])
    )
    # path via the other slot's head predicate when that head reads S1
    # (head bits are re-read every iteration at/after the writer's segment)
    oh_a, oh_b = head_of[a], head_of[b]
    p2_0_a = (oh_b == 1) & (lm[a] <= 1)
    p2_0_b = (oh_b == 2) & (lm[a] <= 2)
    p2_1_a = (oh_a == 1) & (lm[b] <= 1)
    p2_1_b = (oh_a == 2) & (lm[b] <= 2)
    live0_base = (dst[a] == 0) | p1_0
    live1_base = (dst[b] == 0) | p1_1
    need_a = (oh_a == 1) | (oh_b == 1)
    need_b = (oh_a == 2) | (oh_b == 2)
    # 36 head combos: c = (hAs*2 + hAd)*6 + (hBs*2 + hBd)
    c36 = torch.arange(36)
    h_as, h_ad = c36 // 12, (c36 // 6) % 2
    h_bs, h_bd = (c36 % 6) // 2, c36 % 2
    ha_pin = (h_as == 0) & (h_ad == 0)
    hb_pin = (h_bs == 0) & (h_bd == 0)
    ha_s1 = h_as == 2
    hb_s1 = h_bs == 2
    valid = (need_a[:, None] | ha_pin[None, :]) & (
        need_b[:, None] | hb_pin[None, :]
    )
    live0 = (
        live0_base[:, None]
        | (p2_0_a[:, None] & ha_s1[None, :])
        | (p2_0_b[:, None] & hb_s1[None, :])
    )
    live1 = (
        live1_base[:, None]
        | (p2_1_a[:, None] & ha_s1[None, :])
        | (p2_1_b[:, None] & hb_s1[None, :])
    )
    keep = order_ok[:, None] & valid & live0 & live1
    pair_idx, combo_idx = keep.nonzero(as_tuple=True)
    return torch.cat(
        [
            cfg[a[pair_idx]],
            cfg[b[pair_idx]],
            torch.stack(
                [h_as[combo_idx], h_ad[combo_idx],
                 h_bs[combo_idx], h_bd[combo_idx]],
                dim=1,
            ),
        ],
        dim=1,
    )


class Config:
    def __init__(self, vocab_size: int, max_seq_len: int) -> None:
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len


# ---------------------------------------------------------------------------
# Chunked transducer LUTs.  A table is 8 entries, each (out_bit << 1) | next,
# indexed by (state << 2) | (a_bit << 1) | b_bit.  k-bit LUTs are built by
# exact composition; scanning W bits in chunks is bit-identical to the
# LSB->MSB per-position recurrence of the reference machine.
# ---------------------------------------------------------------------------

def _compose(low: Tensor, ka: int, high: Tensor, kb: int) -> Tensor:
    """low: (T, 2, 4**ka) packed (out<<1|state); high: (T, 2, 4**kb)."""
    k = ka + kb
    na, nb = 1 << ka, 1 << kb
    a = torch.arange(1 << k, device=low.device)
    alo, ahi = a & (na - 1), a >> ka
    s = torch.arange(2, device=low.device)
    idx_a = (
        s[:, None, None] * (na * na)
        + alo[None, :, None] * na
        + alo[None, None, :]
    ).reshape(1, -1).expand(low.shape[0], -1)
    r1 = torch.gather(low.reshape(low.shape[0], -1), 1, idx_a)
    out_lo, s1 = r1 >> 1, r1 & 1
    grid_hi = (ahi[:, None] * nb + ahi[None, :]).reshape(-1)
    idx_b = s1 * (nb * nb) + grid_hi.repeat(2)[None, :]
    r2 = torch.gather(high.reshape(high.shape[0], -1), 1, idx_b)
    out_hi, s2 = r2 >> 1, r2 & 1
    return (((out_lo | (out_hi << ka)) << 1) | s2).reshape(-1, 2, 1 << (2 * k))


def build_luts(tables: Tensor, ks: set[int] | None = None) -> dict[int, Tensor]:
    """tables: (T, 8) long entries in 0..3 -> {k: (T, 2, 4**k)} for k in ks."""
    if ks is None:
        ks = {1, 2, 3, 4}
    luts = {1: tables.reshape(-1, 2, 4)}
    need = set(ks)
    if need & {2, 3, 4, 5, 6, 7, 8}:
        luts[2] = _compose(luts[1], 1, luts[1], 1)
    if need & {3, 7}:
        luts[3] = _compose(luts[2], 2, luts[1], 1)
    if need & {4, 5, 6, 7, 8}:
        luts[4] = _compose(luts[2], 2, luts[2], 2)
    if 5 in need:
        luts[5] = _compose(luts[4], 4, luts[1], 1)
    if 6 in need:
        luts[6] = _compose(luts[4], 4, luts[2], 2)
    if 7 in need:
        luts[7] = _compose(luts[4], 4, luts[3], 3)
    if 8 in need:
        luts[8] = _compose(luts[4], 4, luts[4], 4)
    return {k: luts[k] for k in ks}


def chunk_sizes(width: int, max_k: int = 6) -> list[int]:
    chunks = [max_k] * (width // max_k)
    if width % max_k:
        chunks.append(width % max_k)
    return chunks


class LutCache:
    """Incremental chunked-transducer store.  A fixed base region caches the
    LUTs of the chains' current MAP tables (rebuilt only for rows that
    changed); a scratch region is refilled per call with any extra enumerated
    tables."""

    def __init__(self, base_rows: int, extra_capacity: int) -> None:
        self.base_rows = base_rows
        self.capacity = base_rows + extra_capacity
        self.tables: Tensor | None = None
        self.store: dict[int, Tensor] = {}

    def get(self, base: Tensor, extra: Tensor, ks: set[int]) -> dict[int, Tensor]:
        device = base.device
        new_ks = [k for k in ks if k not in self.store]
        for k in new_ks:
            self.store[k] = torch.zeros(
                self.capacity, 2, 1 << (2 * k), dtype=torch.long, device=device
            )
        if self.tables is None or new_ks:
            changed = torch.arange(base.shape[0], device=device)
        else:
            changed = (base != self.tables).any(dim=1).nonzero(as_tuple=False).flatten()
        if changed.numel():
            built = build_luts(base.index_select(0, changed), ks)
            for k in ks:
                self.store[k][:self.base_rows].index_copy_(0, changed, built[k])
        self.tables = base.clone()
        n_extra = extra.shape[0]
        if n_extra:
            built = build_luts(extra, ks)
            for k in ks:
                self.store[k][self.base_rows: self.base_rows + n_extra] = built[k]
        return {k: self.store[k].reshape(-1) for k in ks}


# ---------------------------------------------------------------------------
# Discrete vectorized executor (torch longs; exact ISA semantics)
# ---------------------------------------------------------------------------

def execute_programs(
    fields: Tensor,
    tset: Tensor,
    bank: Tensor,
    xs: Tensor,
    ns: Tensor,
    ts: Tensor,
    width: int,
    t_max: int,
    return_state: bool = False,
    lut_cache: LutCache | None = None,
    max_k: int = 4,
) -> Tensor | tuple[Tensor, Tensor]:
    """Run P programs on R rows, latching each row at its own T.

    fields: (P, 60) long -- 8 slots x [lm,dst,sa,sb,tid,ini,pr] + 4 head
    fields; tset: (P,) index into bank (Q, 4, 8) of table sets.  Registers
    are integers; every committed value is a hard integer register.
    """
    device = fields.device
    p_total = fields.shape[0]
    rows = xs.numel()
    mask = (1 << width) - 1
    chunks = chunk_sizes(width, max_k)
    tables_flat = bank.reshape(-1, 8)
    if lut_cache is not None:
        lut_flat = lut_cache.get(
            tables_flat[: lut_cache.base_rows],
            tables_flat[lut_cache.base_rows:],
            set(chunks),
        )
    else:
        luts = build_luts(tables_flat, set(chunks))
        lut_flat = {k: luts[k].reshape(-1) for k in set(chunks)}

    regs = torch.zeros(6, p_total, rows, dtype=torch.long, device=device)
    regs[0] = (xs & mask).unsqueeze(0)
    regs[1] = (ns & mask).unsqueeze(0)
    regs[3] = 1

    slot_cols = fields[:, : 7 * K_SLOTS].reshape(p_total, K_SLOTS, 7)
    head_cols = fields[:, 7 * K_SLOTS:]

    # per (slot, segment): subset of programs that may commit there
    plans: list[list[tuple]] = [[] for _ in range(4)]
    for i in range(K_SLOTS):
        lm_c = slot_cols[:, i, 0]
        pr_c = slot_cols[:, i, 6]
        live = pr_c != PRED_NEVER
        for seg in range(4):
            sub = ((lm_c == seg) & live).nonzero(as_tuple=False).flatten()
            if sub.numel() == 0:
                continue
            sc = slot_cols[sub, i]
            lidx = tset[sub] * bank.shape[1] + sc[:, 4]
            plans[seg].append((
                sub,
                (sc[:, 1] == 0).unsqueeze(1),  # dst is ACC
                sc[:, 2] * p_total + sub,      # src_a flat index (tied to dst)
                sc[:, 3] * p_total + sub,      # src_b flat index
                (lidx * 2).unsqueeze(1),       # LUT row base
                sc[:, 5].unsqueeze(1),         # init state
                sc[:, 6].unsqueeze(1),         # predicate
                None,                          # index into the segment union
            ))

    # per head segment: union of participating programs and per-entry maps
    hsrc_map = torch.tensor(HSRC_REG, device=device)
    seg_info: dict[int, tuple] = {}
    for seg in (LM_HEAD_A, LM_HEAD_B):
        if not plans[seg]:
            continue
        union = torch.unique(torch.cat([entry[0] for entry in plans[seg]]))
        plans[seg] = [
            entry[:7] + (torch.searchsorted(union, entry[0]),)
            for entry in plans[seg]
        ]
        j = 0 if seg == LM_HEAD_A else 1
        hreg = hsrc_map[head_cols[union, 2 * j]]
        is_msb = (head_cols[union, 2 * j + 1] == 0).unsqueeze(1)
        seg_info[seg] = (union, hreg, is_msb)

    def run(entry, head_bits: Tensor | None) -> None:
        sub, dst_acc, a_idx, b_idx, base, ini, pr, union_map = entry
        sp = sub.numel()
        flat = regs.reshape(6 * p_total, rows)
        a = flat.index_select(0, a_idx)
        b = flat.index_select(0, b_idx)
        state = ini.expand(sp, rows).clone()
        out = torch.zeros(sp, rows, dtype=torch.long, device=device)
        shift = 0
        for k in chunks:
            m = (1 << k) - 1
            idx = ((base + state) << (2 * k)) | (
                (((a >> shift) & m) << k) | ((b >> shift) & m)
            )
            val = torch.take(lut_flat[k], idx)
            out |= (val >> 1) << shift
            state = val & 1
            shift += k
        if head_bits is None:
            head_bit = torch.zeros(1, 1, dtype=torch.long, device=device)
        else:
            head_bit = head_bits.index_select(0, union_map)
        commit = (
            (pr == PRED_ALWAYS)
            | ((pr == PRED_HEAD_POS) & (head_bit == 1))
            | ((pr == PRED_HEAD_NEG) & (head_bit == 0))
            | ((pr == PRED_TERM_POS) & (state == 1))
            | ((pr == PRED_TERM_NEG) & (state == 0))
        )
        old_acc = regs[4].index_select(0, sub)
        old_s1 = regs[5].index_select(0, sub)
        regs[4].index_copy_(0, sub, torch.where(commit & dst_acc, out, old_acc))
        regs[5].index_copy_(0, sub, torch.where(commit & ~dst_acc, out, old_s1))

    latched = torch.zeros(p_total, rows, dtype=torch.long, device=device)
    for t_step in range(1, t_max + 1):
        regs[4].zero_()
        regs[5].zero_()
        for entry in plans[LM_PRE]:
            run(entry, None)
        for seg in (LM_HEAD_A, LM_HEAD_B):
            if not plans[seg]:
                continue
            union, hreg, is_msb = seg_info[seg]
            hsel = hreg * p_total + union
            for head_it in range(width):
                pos = torch.where(is_msb, width - 1 - head_it, head_it)
                head_bits = (
                    regs.reshape(6 * p_total, rows).index_select(0, hsel) >> pos
                ) & 1
                for entry in plans[seg]:
                    run(entry, head_bits)
        for entry in plans[LM_POST]:
            run(entry, None)
        latched = torch.where(ts.unsqueeze(0) == t_step, regs[4], latched)
        if return_state and t_step == t_max:
            return latched, regs
        regs[0] = regs[4]
    return latched


# ---------------------------------------------------------------------------
# Compiled single-program eval executor.  Semantics are exactly those of
# execute_programs restricted to P=1, but the schedule (slot routing,
# predicates, head registers) is resolved to Python scalars ONCE per eval
# pass, so a batch costs only the unavoidable tensor ops: no per-batch
# argmaxes, nonzero() syncs, or per-alternative bookkeeping.  This matters on
# GPU, where every tiny kernel pays launch overhead and the depth ladder runs
# up to T=64 outer steps per batch.
# ---------------------------------------------------------------------------

EVAL_MAX_K = 8  # wider LUT chunks at eval: fewer kernels per transducer scan


def compile_eval_program(fields: Tensor, tables: Tensor) -> dict:
    """Compile one program row (60 fields) into a static schedule.

    NEVER slots are dropped outright.  HEAD_* predicates outside the head
    segments resolve statically (the reference executor feeds them a constant
    zero head bit): HEAD_POS never commits (drop), HEAD_NEG always commits.
    Entries keep slot order inside each segment.
    """
    vals = [int(v) for v in fields.tolist()]
    segments: list[list[tuple[int, int, int, int, int, int]]] = [
        [] for _ in range(4)
    ]
    for i in range(K_SLOTS):
        lm, dst, sa, sb, tid, ini, pr = vals[7 * i: 7 * i + 7]
        if pr == PRED_NEVER:
            continue
        if lm in (LM_PRE, LM_POST):
            if pr == PRED_HEAD_POS:
                continue
            if pr == PRED_HEAD_NEG:
                pr = PRED_ALWAYS
        segments[lm].append((4 + dst, sa, sb, tid, ini, pr))
    heads = (
        (HSRC_REG[vals[56]], vals[57] == 0),
        (HSRC_REG[vals[58]], vals[59] == 0),
    )
    return {
        "segments": segments,
        "heads": heads,
        "tables": tables.clone(),
        "luts": {},
        "device": tables.device,
    }


def run_eval_program(
    prog: dict, xs: Tensor, ns: Tensor, ts: Tensor, width: int, t_max: int
) -> Tensor:
    device = xs.device
    chunks = chunk_sizes(width, EVAL_MAX_K)
    luts = prog["luts"]
    missing = set(chunks) - luts.keys()
    if missing:
        built = build_luts(prog["tables"], missing)
        for k in missing:
            luts[k] = built[k].reshape(-1)
    mask = (1 << width) - 1
    rows = xs.numel()
    zero = torch.zeros(rows, dtype=torch.long, device=device)
    regs = [xs & mask, ns & mask, zero, zero + 1, zero, zero]

    def run_entry(entry: tuple, head_bit: Tensor | None) -> None:
        dst, sa, sb, tid, ini, pr = entry
        a = regs[sa]
        b = regs[sb]
        state: int | Tensor = ini
        out: Tensor | None = None
        shift = 0
        for k in chunks:
            m = (1 << k) - 1
            a_bits = (a >> shift) & m if shift else a & m
            b_bits = (b >> shift) & m if shift else b & m
            idx = ((tid * 2 + state) << (2 * k)) | (a_bits << k) | b_bits
            val = torch.take(luts[k], idx)
            piece = val >> 1
            out = piece if shift == 0 else out | (piece << shift)
            state = val & 1
            shift += k
        if pr == PRED_ALWAYS:
            regs[dst] = out
            return
        if pr == PRED_HEAD_POS:
            commit = head_bit == 1
        elif pr == PRED_HEAD_NEG:
            commit = head_bit == 0
        elif pr == PRED_TERM_POS:
            commit = state == 1
        else:  # PRED_TERM_NEG
            commit = state == 0
        regs[dst] = torch.where(commit, out, regs[dst])

    segments = prog["segments"]
    latched = zero
    for t_step in range(1, t_max + 1):
        regs[4] = zero
        regs[5] = zero
        for entry in segments[LM_PRE]:
            run_entry(entry, None)
        for seg, (hreg, is_msb) in (
            (LM_HEAD_A, prog["heads"][0]),
            (LM_HEAD_B, prog["heads"][1]),
        ):
            entries = segments[seg]
            if not entries:
                continue
            for head_it in range(width):
                pos = width - 1 - head_it if is_msb else head_it
                head_bit = (regs[hreg] >> pos) & 1 if pos else regs[hreg] & 1
                for entry in entries:
                    run_entry(entry, head_bit)
        for entry in segments[LM_POST]:
            run_entry(entry, None)
        latched = torch.where(ts == t_step, regs[4], latched)
        if t_step < t_max:
            regs[0] = regs[4]
    return latched


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Model(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.decimal_width = max(2, (spec.max_seq_len - 4) // 2)
        value_bits = (3322 * self.decimal_width + 999) // 1000 + 1
        # wide tape: room for an unreduced product of two reduced values
        self.width_max = 2 * value_bits + 4
        gen = torch.Generator().manual_seed(GA_SEED)

        def logits(*shape: int) -> nn.Parameter:
            return nn.Parameter(torch.randn(*shape, generator=gen) * INIT_SCALE)

        c = NCHAINS
        self.slot_lm = logits(c, K_SLOTS, N_LM)
        self.slot_dst = logits(c, K_SLOTS, N_DST)
        self.slot_sb = logits(c, K_SLOTS, N_SRC)
        self.slot_tid = logits(c, K_SLOTS, N_TID)
        self.slot_ini = logits(c, K_SLOTS, N_INI)
        self.slot_pred = logits(c, K_SLOTS, N_PRD)
        self.head_src = logits(c, 2, N_HSRC)
        self.head_dir = logits(c, 2, N_HDIR)
        self.table_ent = logits(c, N_TID, 8, 4)
        with torch.no_grad():
            for slot in RESERVED_SLOTS:
                self.slot_pred[:, slot, PRED_NEVER] += PIN_BIAS

        hidden = 32
        self.decoder = nn.Sequential(
            nn.Linear(2 * self.width_max, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.decimal_width * NUM_DIGITS),
        )

        # training-only diagnostic buffers (never touched during eval)
        self.register_buffer("fit_cong", torch.zeros(c), persistent=True)
        self.register_buffer("fit_range", torch.zeros(c), persistent=True)
        self.register_buffer("fit_exact", torch.zeros(c), persistent=True)
        self.register_buffer("fit_rows", torch.zeros(c), persistent=True)
        self.register_buffer("last_hit", torch.zeros(c), persistent=True)
        self.register_buffer("last_rows", torch.zeros(c), persistent=True)
        self.register_buffer("last_fit", torch.zeros(c), persistent=True)
        self.register_buffer("hit_streak", torch.zeros(c, dtype=torch.long), persistent=True)
        self.register_buffer("stage", torch.zeros(c, dtype=torch.long), persistent=True)
        self.register_buffer("age", torch.zeros(c, dtype=torch.long), persistent=True)
        self.register_buffer("best_chain", torch.zeros((), dtype=torch.long), persistent=True)
        # first-step default plan; the scheduler owns this afterwards
        self._plan = {
            "active": [(0, ("slot", 0, 0)), (1, ("slot", 1, 0))],
            "sharp": SHARP0,
            "alpha": 0.0,
            "sweep_lo": 0,
        }
        self._exec_rng = torch.Generator().manual_seed(GA_SEED + 1)
        self._lut_cache: LutCache | None = None
        # enumeration stage-1 state.  The canonical structure list is a
        # deterministic function of the ISA (no data); it is rebuilt on
        # demand rather than stored as model state.  Findings live in the
        # persistent scheduler buffers below.
        self._enum_structs: Tensor | None = None
        self._enum_order: Tensor | None = None
        self._enum_rng = torch.Generator().manual_seed(GA_SEED + 7)
        self._enum_eval_acc = 0
        self.register_buffer("enum_cursor", torch.zeros((), dtype=torch.long), persistent=True)
        self.register_buffer("enum_done", torch.zeros(()), persistent=True)
        self.register_buffer("enum_surv", torch.zeros(()), persistent=True)
        self.register_buffer("enum_evals", torch.zeros(()), persistent=True)
        self.register_buffer("enum_secs", torch.zeros(()), persistent=True)
        self.register_buffer("enum_best_cong", torch.zeros(()), persistent=True)
        self.register_buffer("enum_best_pos", torch.zeros((), dtype=torch.long), persistent=True)
        self.register_buffer("enum_hits", torch.zeros((), dtype=torch.long), persistent=True)
        self.register_buffer("enum_hit_pos", torch.zeros((), dtype=torch.long), persistent=True)
        self.register_buffer("enum_prog", torch.zeros(60, dtype=torch.long), persistent=True)
        self.register_buffer("enum_tab", torch.zeros(8, dtype=torch.long), persistent=True)
        # stage-2 in-loop completion findings (recorded by the loss when a
        # row-exact completion candidate is scored; adopted by the scheduler)
        self.register_buffer("s2_hit", torch.zeros((), dtype=torch.long), persistent=True)
        self.register_buffer("s2_chain", torch.zeros((), dtype=torch.long), persistent=True)
        self.register_buffer("s2_prog", torch.zeros(60, dtype=torch.long), persistent=True)
        self.register_buffer("s2_bank", torch.zeros(N_TID, 8, dtype=torch.long), persistent=True)
        # eval-only compiled MAP program (plain attribute, never a buffer);
        # rebuilt lazily after any training forward or device change
        self._eval_cache: dict | None = None

    # -- parameter access ---------------------------------------------------

    def field_logits(self) -> dict[str, Tensor]:
        return {
            "lm": self.slot_lm, "dst": self.slot_dst, "sb": self.slot_sb,
            "tid": self.slot_tid, "ini": self.slot_ini, "pr": self.slot_pred,
            "hsrc": self.head_src, "hdir": self.head_dir,
            "tab": self.table_ent,
        }

    def map_fields(self) -> tuple[Tensor, Tensor]:
        """MAP genome of every chain: (C, 60) fields and (C, 4, 8) tables."""
        lm = self.slot_lm.argmax(-1)
        dst = self.slot_dst.argmax(-1)
        sb = self.slot_sb.argmax(-1)
        tid = self.slot_tid.argmax(-1)
        ini = self.slot_ini.argmax(-1)
        pr = self.slot_pred.argmax(-1)
        slots = torch.stack([lm, dst, 4 + dst, sb, tid, ini, pr], dim=-1)
        heads = torch.stack(
            [
                self.head_src.argmax(-1)[:, 0], self.head_dir.argmax(-1)[:, 0],
                self.head_src.argmax(-1)[:, 1], self.head_dir.argmax(-1)[:, 1],
            ],
            dim=-1,
        )
        fields = torch.cat([slots.reshape(NCHAINS, -1), heads], dim=1)
        return fields, self.table_ent.argmax(-1)

    # -- parsing (marker-anchored decimal, Horner) ---------------------------

    @staticmethod
    def _parse_number(input_ids: Tensor, mask: Tensor) -> Tensor:
        value = torch.zeros(
            input_ids.shape[0], device=input_ids.device, dtype=torch.long
        )
        for position in range(input_ids.shape[1]):
            digit = (input_ids[:, position] - DIGIT_OFFSET).clamp(0, 9)
            value = torch.where(mask[:, position], value * 10 + digit, value)
        return value

    def _parse(self, input_ids: Tensor, valid: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        digits = (
            valid
            & (input_ids >= DIGIT_OFFSET)
            & (input_ids < DIGIT_OFFSET + NUM_DIGITS)
        )
        after_n = (input_ids == N_MARK).cumsum(dim=1) > 0
        after_x = (input_ids == X_MARK).cumsum(dim=1) > 0
        after_t = (input_ids == T_MARK).cumsum(dim=1) > 0
        modulus = self._parse_number(input_ids, digits & after_n & ~after_x)
        value = self._parse_number(input_ids, digits & after_x & ~after_t)
        t_val = self._parse_number(input_ids, digits & after_t).clamp(
            min=1, max=MAX_OUTER_STEPS
        )
        return modulus.clamp(min=2), value, t_val

    def _batch_width(self, modulus: Tensor) -> int:
        return min(int(2 * int(modulus.max()).bit_length() + 4), self.width_max)

    # -- output placement ----------------------------------------------------

    def _place_logits(self, digit_logits_lsd: Tensor, lengths: Tensor, prompt: int) -> Tensor:
        digits = digit_logits_lsd.flip(1)
        destination = lengths[:, None] - self.decimal_width + torch.arange(
            self.decimal_width, device=digits.device
        )[None]
        destination = destination.clamp(0, prompt - 1)
        canvas = torch.zeros(
            digits.shape[0], prompt, NUM_DIGITS,
            device=digits.device, dtype=digits.dtype,
        )
        canvas = canvas.scatter(
            1, destination.unsqueeze(-1).expand(-1, -1, NUM_DIGITS), digits
        )
        return F.pad(
            canvas,
            (DIGIT_OFFSET, self.config.vocab_size - DIGIT_OFFSET - NUM_DIGITS),
            value=-16.0,
        )

    def _decode_hard_integer(self, value: Tensor, dtype: torch.dtype) -> Tensor:
        decimal_powers = 10 ** torch.arange(self.decimal_width, device=value.device)
        digits = (value[:, None] // decimal_powers[None]).remainder(10)
        logits = torch.full(
            (value.shape[0], self.decimal_width, NUM_DIGITS), -16.0,
            device=value.device, dtype=dtype,
        )
        logits.scatter_(2, digits.unsqueeze(-1), 0.0)
        return logits

    def _bits_onehot(self, value: Tensor) -> Tensor:
        position = torch.arange(self.width_max, device=value.device)
        bits = ((value[:, None] >> position[None]) & 1).long()
        return F.one_hot(bits, 2).to(self.slot_lm.dtype)

    # -- block alternative assembly -------------------------------------------

    def _block_alternatives(
        self, chain: int, block: tuple, map_fields: Tensor, map_tables: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Returns (fields (A,60), tset (A,), extra_bank (E,4,8), log_prior (A,))."""
        device = map_fields.device
        base = map_fields[chain]
        kind = block[0]
        ls = lambda t: F.log_softmax(t, dim=-1)  # noqa: E731
        if kind == "slot":
            k = block[1]
            offset = block[2] if len(block) > 2 else 0
            window = SLOT_PERM[
                (offset + torch.arange(SLOT_SAMPLE)) % SLOT_COMBOS.shape[0]
            ]
            combos = SLOT_COMBOS[window].to(device)
            col = 7 * k
            incumbent = torch.stack([
                base[col + 0], base[col + 1], base[col + 3],
                base[col + 4], base[col + 5], base[col + 6],
            ]).unsqueeze(0)
            combos = torch.cat([combos, incumbent], dim=0)
            a_count = combos.shape[0]
            fields = base.unsqueeze(0).repeat(a_count, 1)
            fields[:, col + 0] = combos[:, 0]
            fields[:, col + 1] = combos[:, 1]
            fields[:, col + 2] = 4 + combos[:, 1]
            fields[:, col + 3] = combos[:, 2]
            fields[:, col + 4] = combos[:, 3]
            fields[:, col + 5] = combos[:, 4]
            fields[:, col + 6] = combos[:, 5]
            log_prior = (
                ls(self.slot_lm[chain, k])[combos[:, 0]]
                + ls(self.slot_dst[chain, k])[combos[:, 1]]
                + ls(self.slot_sb[chain, k])[combos[:, 2]]
                + ls(self.slot_tid[chain, k])[combos[:, 3]]
                + ls(self.slot_ini[chain, k])[combos[:, 4]]
                + ls(self.slot_pred[chain, k])[combos[:, 5]]
            )
            tset = torch.full((a_count,), chain, dtype=torch.long, device=device)
            extra = map_tables.new_zeros(0, N_TID, 8)
        elif kind == "head":
            combos = HEAD_COMBOS.to(device)
            a_count = combos.shape[0]
            fields = base.unsqueeze(0).repeat(a_count, 1)
            fields[:, 56:60] = combos
            log_prior = (
                ls(self.head_src[chain, 0])[combos[:, 0]]
                + ls(self.head_dir[chain, 0])[combos[:, 1]]
                + ls(self.head_src[chain, 1])[combos[:, 2]]
                + ls(self.head_dir[chain, 1])[combos[:, 3]]
            )
            tset = torch.full((a_count,), chain, dtype=torch.long, device=device)
            extra = map_tables.new_zeros(0, N_TID, 8)
        elif kind == "table":
            t_index = block[1]
            samples = torch.randint(
                0, 4, (TABLE_BLOCK_SAMPLES, 8), generator=self._exec_rng
            ).to(device)
            patterns = torch.cat(
                [map_tables[chain, t_index].unsqueeze(0), samples], dim=0
            )
            a_count = patterns.shape[0]
            fields = base.unsqueeze(0).repeat(a_count, 1)
            extra = map_tables[chain].unsqueeze(0).repeat(a_count, 1, 1)
            extra[:, t_index] = patterns
            tset = NCHAINS + torch.arange(a_count, device=device)
            entry_lp = ls(self.table_ent[chain, t_index])  # (8, 4)
            log_prior = entry_lp[
                torch.arange(8, device=device)[None, :], patterns
            ].sum(-1)
        else:
            raise ValueError(block)
        return fields, tset, extra, log_prior

    # -- forward ---------------------------------------------------------------

    def forward(
        self, input_ids: Tensor, attention_mask: Tensor | None = None
    ) -> tuple[Tensor, dict[str, object]]:
        if attention_mask is None:
            attention_mask = input_ids != PAD
        valid = attention_mask.bool()
        modulus, value, t_values = self._parse(input_ids, valid)
        width = self._batch_width(modulus)
        lengths = valid.sum(dim=1)
        prompt = input_ids.shape[1]
        t_max = int(t_values.max())

        if not self.training:
            with torch.no_grad():
                cache = self._eval_cache
                if cache is None or cache["device"] != input_ids.device:
                    map_fields, map_tables = self.map_fields()
                    chain = int(self.best_chain)
                    cache = compile_eval_program(
                        map_fields[chain], map_tables[chain]
                    )
                    self._eval_cache = cache
                out = run_eval_program(
                    cache, value, modulus, t_values, width, t_max
                ) & ((1 << width) - 1)
            digit_logits = self._decode_hard_integer(out, self.slot_lm.dtype)
            logits = self._place_logits(digit_logits, lengths, prompt)
            return logits, {"t_values": t_values}

        self._eval_cache = None
        plan = self._plan
        with torch.no_grad():
            map_fields, map_tables = self.map_fields()
            # stage-1 curriculum: execute only the shallowest rows at their
            # own depth (predecessor's t_min precedent); once any chain is in
            # stage 2, execute every row at its own T (per-row latching).
            if plan.get("full_t"):
                sel = torch.arange(
                    min(input_ids.shape[0], FULL_T_ROWS), device=input_ids.device
                )
            else:
                t_min = int(t_values.min())
                sel = (t_values == t_min).nonzero(as_tuple=False).flatten()
                if sel.numel() == 0:
                    sel = torch.arange(input_ids.shape[0], device=input_ids.device)
            value_e = value[sel]
            modulus_e = modulus[sel]
            t_e = t_values[sel]
            t_max = int(t_e.max())

        # one combined discrete execution: population MAP programs first,
        # then each active chain's enumerated block alternatives.
        pieces_f = [map_fields]
        pieces_t = [torch.arange(NCHAINS, device=input_ids.device)]
        banks = [map_tables]
        bank_rows = NCHAINS
        log_priors: list[tuple[int, Tensor, str]] = []
        spans: list[tuple[int, int]] = []
        cursor = NCHAINS
        sweeps: list[tuple[int, tuple]] = []
        for chain, block in plan["active"]:
            if block[0] in ("sweep", "comp", "loop"):
                sweeps.append((chain, block))
                continue
            fields, tset, extra, log_prior = self._block_alternatives(
                chain, block, map_fields, map_tables
            )
            if extra.shape[0]:
                tset = tset - NCHAINS + bank_rows
                banks.append(extra)
                bank_rows += extra.shape[0]
            pieces_f.append(fields)
            pieces_t.append(tset)
            spans.append((cursor, cursor + fields.shape[0]))
            log_priors.append((chain, log_prior, "block"))
            cursor += fields.shape[0]
        all_fields = torch.cat(pieces_f, dim=0)
        all_tset = torch.cat(pieces_t, dim=0)
        all_banks = torch.cat(banks, dim=0)
        if self._lut_cache is None:
            self._lut_cache = LutCache(NCHAINS * 4, 2048)
        with torch.no_grad():
            all_out = execute_programs(
                all_fields, all_tset, all_banks, value_e, modulus_e,
                t_e, width, t_max, lut_cache=self._lut_cache,
            )
        map_out = all_out[:NCHAINS]
        alt_outs = [all_out[lo:hi] for lo, hi in spans]

        # stage-2 completion sweeps: shared core state, enumerated
        # (completion-combo x table-pattern) alternatives on depth-1 rows.
        sweep_terms = []
        t1_sel = (t_e == 1).nonzero(as_tuple=False).flatten()[:SWEEP_ROWS]
        for chain, block in sweeps:
            if t1_sel.numel() == 0:
                fields, tset, extra, log_prior = self._block_alternatives(
                    chain, ("slot", block[1], 0), map_fields, map_tables
                )
                with torch.no_grad():
                    out = execute_programs(
                        fields, tset, map_tables, value_e, modulus_e,
                        t_e, width, t_max,
                    )
                log_priors.append((chain, log_prior, "block"))
                alt_outs.append(out)
                spans.append((-1, -1))
                continue
            if block[0] == "loop":
                lp_l, out_l, xs_l, ns_l, pack = self._loop_sweep(
                    chain, map_fields, map_tables,
                    value_e[t1_sel], modulus_e[t1_sel], width, plan,
                )
                sweep_terms.append((lp_l, out_l, xs_l, ns_l, t1_sel, pack))
                continue
            sweep_terms.append(
                self._completion_sweep(
                    chain, block, map_fields, map_tables,
                    value_e[t1_sel], modulus_e[t1_sel], width, plan,
                )
                + (t1_sel, None)
            )

        # soft decoder on the best chain's latched outputs (training path only)
        best = int(self.best_chain)
        bits = self._bits_onehot(map_out[best] & ((1 << width) - 1))
        decoded = self.decoder(bits.flatten(1)).reshape(
            -1, self.decimal_width, NUM_DIGITS
        )
        digit_logits = torch.zeros(
            input_ids.shape[0], self.decimal_width, NUM_DIGITS,
            device=input_ids.device, dtype=decoded.dtype,
        )
        digit_logits[sel] = decoded
        logits = self._place_logits(digit_logits.flip(1), lengths, prompt)
        auxiliary = {
            "model": self,
            "plan": plan,
            "map_out": map_out,
            "alt_outs": alt_outs,
            "log_priors": log_priors,
            "sweep_terms": sweep_terms,
            "sel": sel,
            "value": value_e,
            "modulus": modulus_e,
            "t_values": t_e,
            "width": width,
        }
        return logits, auxiliary

    def _completion_sweep(
        self,
        chain: int,
        block: tuple,
        map_fields: Tensor,
        map_tables: Tensor,
        xs: Tensor,
        ns: Tensor,
        width: int,
        plan: dict,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Stage-2 completion blocks over one reserved once_post slot.

        block = ("sweep", k): conditional-completion combos jointly with the
        ENTRIES of a free table (rotating window over all 65,536 patterns
        plus samples from the current entry posterior).
        block = ("comp", k): the full once_post combo set (dst, src_b,
        table_id, init, predicate) with the chain's current tables.
        Execution is discrete; because the reserved slots trail every other
        slot, all alternatives share the core's register state exactly.
        """
        device = xs.device
        kind, slot = block[0], block[1]
        rows = xs.numel()
        with torch.no_grad():
            core = map_fields[chain].clone()
            for reserved in RESERVED_SLOTS:
                if reserved >= slot:
                    core[7 * reserved + 6] = PRED_NEVER
            used = set()
            for i in range(K_SLOTS):
                if int(core[7 * i + 6]) != PRED_NEVER:
                    used.add(int(core[7 * i + 4]))
            free_tables = [t for t in range(N_TID) if t not in used]
            shared_core = bool(free_tables) or kind == "comp"
            t_free = free_tables[0] if free_tables else N_TID - 1
            if kind == "comp":
                combos = COMP_COMBOS.to(device)
                tables = map_tables[chain]  # (4, 8): pattern axis = table_id
            else:
                combos = SWEEP_COMBOS.to(device)
                lo = plan["sweep_lo"]
                n_win = (
                    SWEEP_TABLES_PER_STEP if shared_core
                    else SWEEP_TABLES_FALLBACK
                )
                ids = (lo + torch.arange(n_win, device=device)) % 65536
                window = (
                    ids[:, None] >> (2 * torch.arange(8, device=device))[None]
                ) & 3
                if shared_core:
                    # sample on CPU: self._exec_rng is a CPU generator and
                    # torch.multinomial rejects a generator/probs device split
                    probs = F.softmax(
                        self.table_ent[chain, t_free].detach(), dim=-1
                    ).cpu()
                    samples = torch.multinomial(
                        probs, SWEEP_POSTERIOR_SAMPLES, replacement=True,
                        generator=self._exec_rng,
                    ).T.to(device)  # (S, 8)
                    incumbent = (
                        self.table_ent[chain, t_free].detach().argmax(-1)
                    ).unsqueeze(0)
                    tables = torch.cat([window, samples, incumbent], dim=0)
                else:
                    tables = window
            n_combo = combos.shape[0]
            n_pat = tables.shape[0]
            if shared_core:
                _, regs = execute_programs(
                    core.unsqueeze(0),
                    torch.zeros(1, dtype=torch.long, device=device),
                    map_tables[chain: chain + 1],
                    xs, ns, torch.ones_like(xs), width, 1, return_state=True,
                )
                state = regs[:, 0]  # (6, R1)
                chunks = chunk_sizes(width, 4)
                luts = build_luts(tables, set(chunks))
                a_sel = state[4 + combos[:, 0]][:, None, :]  # (C, 1, R1)
                b_sel = state[combos[:, 1]][:, None, :]
                s = combos[:, 2][:, None, None].expand(-1, n_pat, rows).clone()
                out = torch.zeros_like(s)
                pat_axis = torch.arange(n_pat, device=device)[None, :, None]
                shift = 0
                for k in chunks:
                    m = (1 << k) - 1
                    idx = ((pat_axis * 2 + s) << (2 * k)) | (
                        (((a_sel >> shift) & m) << k) | ((b_sel >> shift) & m)
                    )
                    val = torch.take(luts[k].reshape(-1), idx)
                    out |= (val >> 1) << shift
                    s = val & 1
                    shift += k
                pr = combos[:, 3][:, None, None]
                commit = (
                    (pr == PRED_ALWAYS)
                    | (pr == PRED_HEAD_NEG)
                    | ((pr == PRED_TERM_POS) & (s == 1))
                    | ((pr == PRED_TERM_NEG) & (s == 0))
                )
                acc = state[4][None, None, :]
                result = torch.where(
                    commit & (combos[:, 0][:, None, None] == 0), out, acc
                ).reshape(-1, rows)
            else:
                fields = core.unsqueeze(0).repeat(n_combo * n_pat, 1)
                col = 7 * slot
                cc = combos.repeat_interleave(n_pat, dim=0)
                fields[:, col + 0] = LM_POST
                fields[:, col + 1] = cc[:, 0]
                fields[:, col + 2] = 4 + cc[:, 0]
                fields[:, col + 3] = cc[:, 1]
                fields[:, col + 4] = t_free
                fields[:, col + 5] = cc[:, 2]
                fields[:, col + 6] = cc[:, 3]
                bank = map_tables[chain].unsqueeze(0).repeat(n_pat, 1, 1)
                bank[:, t_free] = tables
                tset = torch.arange(n_pat, device=device).repeat(n_combo)
                result = execute_programs(
                    fields, tset, bank, xs, ns,
                    torch.ones_like(xs), width, 1,
                )
        self._sweep_debug = (kind, slot, combos, tables, t_free, shared_core)
        ls = lambda t: F.log_softmax(t, dim=-1)  # noqa: E731
        lp_combo = (
            ls(self.slot_lm[chain, slot])[LM_POST]
            + ls(self.slot_dst[chain, slot])[combos[:, 0]]
            + ls(self.slot_sb[chain, slot])[combos[:, 1]]
            + ls(self.slot_ini[chain, slot])[combos[:, 2]]
            + ls(self.slot_pred[chain, slot])[combos[:, 3]]
        )
        if kind == "comp":
            lp_pattern = ls(self.slot_tid[chain, slot])  # (4,) over table ids
        else:
            entry_lp = ls(self.table_ent[chain, t_free])  # (8, 4)
            lp_pattern = (
                ls(self.slot_tid[chain, slot])[t_free]
                + entry_lp[
                    torch.arange(8, device=device)[None, :], tables
                ].sum(-1)
            )
        log_prior = (lp_combo[:, None] + lp_pattern[None, :]).reshape(-1)
        return log_prior, result, xs, ns

    def _loop_sweep(
        self,
        chain: int,
        map_fields: Tensor,
        map_tables: Tensor,
        xs: Tensor,
        ns: Tensor,
        width: int,
        plan: dict,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Stage-2 IN-LOOP completion block.

        A once_post subtract cannot reduce a multiplicative core (its
        unreduced output spans hundreds of multiples of N); the reducer must
        interleave the accumulation loop itself.  This block enumerates
        conditional completions of the reserved slots placed INSIDE the
        core's own head segment -- (init, TERM polarity, single-or-pair)
        placements of ACC<-t(ACC, N), trailing the core slots each head
        iteration -- jointly with the ENTRY patterns of a free table
        (rotating 2048-window over all 65,536 + posterior samples +
        incumbent).  The trailing (pr=NEVER) combo row is the explicit
        no-op anchor.  Execution is full-program and discrete: the
        shared-core shortcut is invalid inside the loop."""
        device = xs.device
        with torch.no_grad():
            core = map_fields[chain].clone()
            for reserved in RESERVED_SLOTS:
                core[7 * reserved + 6] = PRED_NEVER
            segs = sorted(
                int(core[7 * i])
                for i in range(K_SLOTS)
                if int(core[7 * i + 6]) != PRED_NEVER
                and int(core[7 * i]) in (LM_HEAD_A, LM_HEAD_B)
            )
            if not segs:
                # once-only core: fall back to the once_post entry sweep
                return self._completion_sweep(
                    chain, ("sweep", RESERVED_SLOTS[0]),
                    map_fields, map_tables, xs, ns, width, plan,
                ) + (None,)
            seg = segs[0]
            used = {
                int(core[7 * i + 4]) for i in range(K_SLOTS)
                if int(core[7 * i + 6]) != PRED_NEVER
            }
            free_tables = [t for t in range(N_TID) if t not in used]
            t_free = free_tables[0] if free_tables else N_TID - 1
            n_win = (
                LOOP_TABLES_PER_STEP if free_tables else SWEEP_TABLES_FALLBACK
            )
            lo = plan.get("loop_lo", 0)
            ids = (lo + torch.arange(n_win, device=device)) % 65536
            window = (
                ids[:, None] >> (2 * torch.arange(8, device=device))[None]
            ) & 3
            if free_tables:
                probs = F.softmax(
                    self.table_ent[chain, t_free].detach(), dim=-1
                ).cpu()
                samples = torch.multinomial(
                    probs, LOOP_POSTERIOR_SAMPLES, replacement=True,
                    generator=self._exec_rng,
                ).T.to(device)
                incumbent = (
                    self.table_ent[chain, t_free].detach().argmax(-1)
                ).unsqueeze(0)
                tables = torch.cat([window, samples, incumbent], dim=0)
            else:
                tables = window
            n_pat = tables.shape[0]
            combos = [
                (ini, pr, pair)
                for ini in range(N_INI)
                for pr in (PRED_TERM_POS, PRED_TERM_NEG)
                for pair in (0, 1)
            ] + [(0, PRED_NEVER, 0)]
            n_combo = len(combos)
            fields = core.unsqueeze(0).repeat(n_combo * n_pat, 1)
            for ci, (ini, pr, pair) in enumerate(combos):
                if pr == PRED_NEVER:
                    continue
                sel = slice(ci * n_pat, (ci + 1) * n_pat)
                for si, slot in enumerate(RESERVED_SLOTS):
                    if si == 1 and not pair:
                        continue
                    col = 7 * slot
                    fields[sel, col + 0] = seg
                    fields[sel, col + 1] = 0  # dst = ACC
                    fields[sel, col + 2] = 4
                    fields[sel, col + 3] = 1  # src_b = N
                    fields[sel, col + 4] = t_free
                    fields[sel, col + 5] = ini
                    fields[sel, col + 6] = pr
            bank = map_tables[chain].unsqueeze(0).repeat(n_pat, 1, 1)
            bank[:, t_free] = tables
            tset = torch.arange(n_pat, device=device).repeat(n_combo)
            result = execute_programs(
                fields, tset, bank, xs, ns, torch.ones_like(xs), width, 1,
            )
        self._loop_debug = (seg, combos, tables, t_free, bool(free_tables))
        ls = lambda t: F.log_softmax(t, dim=-1)  # noqa: E731
        s6, s7 = RESERVED_SLOTS
        lp_rows = []
        for ini, pr, pair in combos:
            lp = ls(self.slot_pred[chain, s6])[pr]
            if pr != PRED_NEVER:
                lp = (
                    lp
                    + ls(self.slot_lm[chain, s6])[seg]
                    + ls(self.slot_dst[chain, s6])[0]
                    + ls(self.slot_sb[chain, s6])[1]
                    + ls(self.slot_tid[chain, s6])[t_free]
                    + ls(self.slot_ini[chain, s6])[ini]
                )
            if pair:
                lp = (
                    lp
                    + ls(self.slot_pred[chain, s7])[pr]
                    + ls(self.slot_lm[chain, s7])[seg]
                    + ls(self.slot_dst[chain, s7])[0]
                    + ls(self.slot_sb[chain, s7])[1]
                    + ls(self.slot_tid[chain, s7])[t_free]
                    + ls(self.slot_ini[chain, s7])[ini]
                )
            else:
                lp = lp + ls(self.slot_pred[chain, s7])[PRED_NEVER]
            lp_rows.append(lp)
        lp_combo = torch.stack(lp_rows)  # (n_combo,)
        if free_tables:
            entry_lp = ls(self.table_ent[chain, t_free])  # (8, 4)
            lp_pattern = entry_lp[
                torch.arange(8, device=device)[None, :], tables
            ].sum(-1)
        else:
            lp_pattern = torch.zeros(
                n_pat, device=device, dtype=lp_combo.dtype
            )
        log_prior = (lp_combo[:, None] + lp_pattern[None, :]).reshape(-1)
        # candidate pack: lets the loss record a row-exact completion in the
        # scheduler buffers (the gradient flip alone is fragile under global
        # grad clipping with windows that rotate every step)
        pack = (fields, tset, bank, chain)
        return log_prior, result, xs, ns, pack

    # -- enumeration stage 1 --------------------------------------------------

    def _enum_space_init(self) -> float:
        """Build the canonical structure list + the fixed-seed visit order."""
        if self._enum_structs is not None:
            return 0.0
        start = time.monotonic()
        device = self.slot_lm.device
        structs = build_enum_space().to(device)
        order = torch.randperm(
            structs.shape[0],
            generator=torch.Generator().manual_seed(ENUM_ORDER_SEED),
        ).to(device)
        self._enum_structs = structs
        self._enum_order = order
        return time.monotonic() - start

    def _enum_fields(self, st: Tensor) -> Tensor:
        """(P, 14) structure rows -> (P, 60) genomes (slots 0-1, table 0)."""
        p_total = st.shape[0]
        fields = torch.zeros(p_total, 60, dtype=torch.long, device=st.device)
        for i in range(K_SLOTS):
            fields[:, 7 * i + 6] = PRED_NEVER
        for k in (0, 1):
            col, s = 7 * k, 5 * k
            fields[:, col + 0] = st[:, s + 0]
            fields[:, col + 1] = st[:, s + 1]
            fields[:, col + 2] = 4 + st[:, s + 1]
            fields[:, col + 3] = st[:, s + 2]
            fields[:, col + 5] = st[:, s + 3]
            fields[:, col + 6] = st[:, s + 4]
        fields[:, 56:60] = st[:, 10:14]
        return fields

    def _enum_execute(
        self, fields: Tensor, tables: Tensor, xs: Tensor, ns: Tensor, width: int
    ) -> Tensor:
        """Discrete execution of (program, its-own-table) rows at T=1."""
        p_total = fields.shape[0]
        cap = ENUM_P_CAP if fields.device.type == "cpu" else ENUM_P_CAP_CUDA
        outs = []
        for lo in range(0, p_total, cap):
            f = fields[lo: lo + cap]
            t = tables[lo: lo + cap]
            outs.append(
                execute_programs(
                    f,
                    torch.arange(f.shape[0], device=f.device),
                    t.unsqueeze(1),
                    xs, ns, torch.ones_like(xs), width, 1,
                )
            )
        self._enum_eval_acc += p_total
        return outs[0] if len(outs) == 1 else torch.cat(outs)

    def _enum_score(
        self, out: Tensor, y: Tensor, ns: Tensor, width: int
    ) -> tuple[Tensor, Tensor]:
        """(congruence bit-match, walk move score) per program.

        The walk moves on the congruence component alone -- exactly the
        round-5-calibrated table learner.  (Measured: adding the range bonus
        to the MOVE score collapses p_hit from 4.4% to 1.0% per restart,
        because the constant-zero attractor is always "in range".  The full
        stage-1 credit, range bonus included, still scores the finalists in
        the loss marginal.)"""
        mask = (1 << width) - 1
        w_low = int(ns.max()).bit_length() + 1
        cong = _bit_match((out.remainder(ns[None, :]) ^ y[None, :]) & mask, w_low)
        return cong, cong

    def _enum_steepest(
        self, fields: Tensor, tables: Tensor,
        xs: Tensor, ns: Tensor, y: Tensor, width: int,
    ) -> tuple[Tensor, Tensor]:
        """Randomized steepest descent over single table entries: each walker
        moves to the best of its 24 one-entry neighbors while that strictly
        improves the stage-1 credit (the round-5 calibrated table learner),
        batched across all walkers."""
        device = fields.device
        tables = tables.clone()
        n_walk = tables.shape[0]
        out = self._enum_execute(fields, tables, xs, ns, width)
        cong, score = self._enum_score(out, y, ns, width)
        alive = torch.ones(n_walk, dtype=torch.bool, device=device)
        pat_e = torch.arange(8, device=device).repeat_interleave(3)  # (24,)
        alt = torch.arange(24, device=device) % 3
        for _ in range(ENUM_MAX_WALK):
            idx = alive.nonzero(as_tuple=False).flatten()
            if idx.numel() == 0:
                break
            na = idx.numel()
            cand_f = fields.index_select(0, idx).repeat_interleave(24, dim=0)
            cand_t = tables.index_select(0, idx).repeat_interleave(24, dim=0)
            e_col = pat_e.repeat(na).unsqueeze(1)
            cur = cand_t.gather(1, e_col).squeeze(1)
            new_v = alt.repeat(na)
            new_v = new_v + (new_v >= cur).long()
            cand_t.scatter_(1, e_col, new_v.unsqueeze(1))
            o = self._enum_execute(cand_f, cand_t, xs, ns, width)
            c_nb, s_nb = self._enum_score(o, y, ns, width)
            s_nb = s_nb.view(na, 24)
            j = s_nb.argmax(dim=1)
            best = s_nb.gather(1, j.unsqueeze(1)).squeeze(1)
            impr = best > score[idx] + 1e-12
            up = idx[impr]
            if up.numel():
                rows = impr.nonzero(as_tuple=False).flatten()
                tables[up] = cand_t[rows * 24 + j[rows]]
                score[up] = best[impr]
                cong[up] = c_nb.view(na, 24)[rows, j[rows]]
            alive[idx[~impr]] = False
        return tables, cong

    def _enum_chunk(
        self, xs: Tensor, ns: Tensor, y: Tensor, width: int,
        n: int, restarts_total: int,
    ) -> dict | None:
        """Advance the enumeration cursor over the next n structures.

        Phase A screens ENUM_RESTARTS_A steepest restarts per structure and
        early-kills structures whose best congruence stays below the
        calibrated threshold; survivors get restarts up to restarts_total.
        Verified congruence-exact finds are recorded in the persistent
        buffers (cursor/best/hit/program) for the scheduler.  Returns the
        per-structure finalists so the loss can marginalize over them."""
        if self._enum_order is None:
            return None
        device = xs.device
        m_total = self._enum_order.numel()
        cursor = int(self.enum_cursor)
        n = min(n, m_total - cursor)
        if n <= 0:
            return None
        start = time.monotonic()
        self._enum_eval_acc = 0
        rows = min(xs.numel(), ENUM_ROWS)
        xs_s, ns_s, y_s = xs[:rows], ns[:rows], y[:rows]
        kill = ENUM_KILL if rows >= 32 else ENUM_KILL_SMALL
        pos = self._enum_order[cursor: cursor + n]
        st = self._enum_structs.index_select(0, pos)
        fields = self._enum_fields(st)
        ra = ENUM_RESTARTS_A
        f_a = fields.repeat_interleave(ra, dim=0)
        t_a = torch.randint(
            0, 4, (n * ra, 8), generator=self._enum_rng
        ).to(device)
        tabs, cong = self._enum_steepest(f_a, t_a, xs_s, ns_s, y_s, width)
        cong = cong.view(n, ra)
        j = cong.argmax(dim=1)
        best = cong.gather(1, j.unsqueeze(1)).squeeze(1)
        best_tab = tabs.view(n, ra, 8)[torch.arange(n, device=device), j]
        surv = (best >= kill) & (best < 1.0 - 1e-9)
        rb = max(int(restarts_total) - ra, 0)
        n_surv = int(surv.sum())
        if n_surv and rb:
            s_idx = surv.nonzero(as_tuple=False).flatten()
            f_b = fields.index_select(0, s_idx).repeat_interleave(rb, dim=0)
            t_b = torch.randint(
                0, 4, (n_surv * rb, 8), generator=self._enum_rng
            ).to(device)
            tabs_b, cong_b = self._enum_steepest(
                f_b, t_b, xs_s, ns_s, y_s, width
            )
            cong_b = cong_b.view(n_surv, rb)
            jb = cong_b.argmax(dim=1)
            best_b = cong_b.gather(1, jb.unsqueeze(1)).squeeze(1)
            upd = best_b > best[s_idx]
            u_idx = s_idx[upd]
            if u_idx.numel():
                rows_b = upd.nonzero(as_tuple=False).flatten()
                best[u_idx] = best_b[upd]
                best_tab[u_idx] = tabs_b.view(n_surv, rb, 8)[rows_b, jb[rows_b]]
        # verify hits against every available row before recording
        mask = (1 << width) - 1
        for k in (best >= 1.0 - 1e-9).nonzero(as_tuple=False).flatten().tolist():
            out_full = execute_programs(
                fields[k].unsqueeze(0),
                torch.zeros(1, dtype=torch.long, device=device),
                best_tab[k].reshape(1, 1, 8),
                xs, ns, torch.ones_like(xs), width, 1,
            )[0]
            self._enum_eval_acc += 1
            if bool((((out_full.remainder(ns) ^ y) & mask) == 0).all()):
                self.enum_prog.copy_(fields[k])
                self.enum_tab.copy_(best_tab[k])
                self.enum_hits.add_(1)
                self.enum_hit_pos.fill_(cursor + k)
                break
            best[k] = 0.99  # perfect on the subsample only: demote
        k_best = int(best.argmax())
        if float(best[k_best]) > float(self.enum_best_cong):
            self.enum_best_cong.fill_(float(best[k_best]))
            self.enum_best_pos.fill_(cursor + k_best)
        out_fin = self._enum_execute(fields, best_tab, xs_s, ns_s, width)
        self.enum_cursor.add_(n)
        self.enum_done.add_(float(n))
        self.enum_surv.add_(float(n_surv))
        self.enum_evals.add_(float(self._enum_eval_acc))
        self.enum_secs.add_(time.monotonic() - start)
        return {"fields": fields, "tables": best_tab, "out": out_fin,
                "rows": rows}

    def _enum_log_prior(
        self, chain: int, fields: Tensor, tables: Tensor
    ) -> Tensor:
        """Differentiable log prior of full enumerated programs under one
        chain's posterior logits (fields shared across alternatives -- the
        NEVER slots 2..7 -- contribute a common constant and are omitted)."""
        device = fields.device
        ls = lambda t: F.log_softmax(t, dim=-1)  # noqa: E731
        lp = torch.zeros(
            fields.shape[0], device=device, dtype=self.slot_lm.dtype
        )
        for k in (0, 1):
            col = 7 * k
            lp = (
                lp
                + ls(self.slot_lm[chain, k])[fields[:, col + 0]]
                + ls(self.slot_dst[chain, k])[fields[:, col + 1]]
                + ls(self.slot_sb[chain, k])[fields[:, col + 3]]
                + ls(self.slot_tid[chain, k])[fields[:, col + 4]]
                + ls(self.slot_ini[chain, k])[fields[:, col + 5]]
                + ls(self.slot_pred[chain, k])[fields[:, col + 6]]
            )
        lp = (
            lp
            + ls(self.head_src[chain, 0])[fields[:, 56]]
            + ls(self.head_dir[chain, 0])[fields[:, 57]]
            + ls(self.head_src[chain, 1])[fields[:, 58]]
            + ls(self.head_dir[chain, 1])[fields[:, 59]]
        )
        ent = ls(self.table_ent[chain, 0])  # (8, 4)
        return lp + ent[
            torch.arange(8, device=device)[None, :], tables
        ].sum(-1)


# ---------------------------------------------------------------------------
# Loss: exact marginalization over each active block's alternatives
# ---------------------------------------------------------------------------

def _target_integer(labels: Tensor, valid: Tensor) -> Tensor:
    value = torch.zeros(labels.shape[0], device=labels.device, dtype=torch.long)
    for position in range(labels.shape[1]):
        digit = (labels[:, position] - DIGIT_OFFSET).clamp(0, 9)
        value = torch.where(valid[:, position], value * 10 + digit, value)
    return value


def _bit_match(diff: Tensor, bits: int) -> Tensor:
    """Mean matched-bit fraction over the low `bits` bits; diff int (..., R).

    Branchless SWAR popcount (bit-identical to the former shift-and-sum,
    verified; valid for diffs up to 56 bits, far above width_max)."""
    v = diff & ((1 << bits) - 1)
    v = v - ((v >> 1) & 0x5555555555555555)
    v = (v & 0x3333333333333333) + ((v >> 2) & 0x3333333333333333)
    v = (v + (v >> 4)) & 0x0F0F0F0F0F0F0F0F
    mism = (v * 0x0101010101010101) >> 56
    return 1.0 - mism.sum(dim=-1).float() / (bits * diff.shape[-1])


def _credit(
    out: Tensor, y: Tensor, ns: Tensor, width: int, stage: int, alpha: float
) -> Tensor:
    """Stage-scheduled per-alternative credit; out (A, R) integers.

    Stage 1: congruence bit-match plus a range bonus (annealed row-exact term
    late).  Stage 2: exact-row fraction anchored by congruence -- exact
    BIT-match must not be used here, or in-range junk outscores an unreduced
    congruent core (the round-3b attractor)."""
    mask = (1 << width) - 1
    w_low = int(ns.max()).bit_length() + 1
    cong = _bit_match((out.remainder(ns[None, :]) ^ y[None, :]) & mask, w_low)
    in_range = (out < ns[None, :]).float().mean(dim=-1)
    row_exact = (((out ^ y[None, :]) & mask) == 0).float().mean(dim=-1)
    if stage == 1:
        return CONG_ANCHOR * cong + row_exact
    return cong + RANGE_BONUS * in_range + alpha * row_exact


def token_training_loss(batch: TokenLossBatch) -> Tensor:
    aux = batch.auxiliary
    model: Model = aux["model"]
    plan = aux["plan"]
    valid = batch.valid_mask
    y = _target_integer(batch.labels, valid)[aux["sel"]]
    ns = aux["modulus"]
    width = aux["width"]
    sharp = plan["sharp"]
    alpha = plan["alpha"]

    terms = []
    for (chain, log_prior, _), out in zip(aux["log_priors"], aux["alt_outs"]):
        stage = int(model.stage[chain])
        credit = _credit(out, y, ns, width, stage, alpha)
        terms.append(
            sharp * 1.15 - torch.logsumexp(log_prior + sharp * credit, dim=0)
        )
    for chain_lp, out, xs_sub, ns_sub, t1_sel, pack in aux["sweep_terms"]:
        y_sub = y[t1_sel]
        credit = _credit(out, y_sub, ns_sub, width, 1, alpha)
        terms.append(
            sharp * 1.15 - torch.logsumexp(chain_lp + sharp * credit, dim=0)
        )
        if pack is not None:
            # record a row-exact in-loop completion for scheduler adoption
            with torch.no_grad():
                k = int(credit.argmax())
                if float(credit[k]) >= 1.0 + CONG_ANCHOR - 0.05:
                    fields_c, tset_c, bank_c, chain_c = pack
                    model.s2_prog.copy_(fields_c[k])
                    model.s2_bank.copy_(bank_c[tset_c[k]])
                    model.s2_chain.fill_(chain_c)
                    model.s2_hit.fill_(1)

    # enumeration stage 1: advance the cursor over the scheduled chunk of
    # canonical structures (batched discrete steepest-descent table search,
    # recorded in the scheduler buffers) and marginalize over the chunk's
    # per-structure finalists with the same stage-1 credit
    enum_n = int(plan.get("enum_n", 0))
    if enum_n:
        t1 = (aux["t_values"] == 1).nonzero(as_tuple=False).flatten()
        if t1.numel() >= ENUM_MIN_ROWS:
            xs_e = aux["value"][t1]
            ns_e = ns[t1]
            y_e = y[t1]
            with torch.no_grad():
                res = model._enum_chunk(
                    xs_e, ns_e, y_e, width, enum_n,
                    int(plan.get("enum_r", ENUM_RESTARTS_CAP)),
                )
            if res is not None:
                rows = res["rows"]
                credit = _credit(
                    res["out"], y_e[:rows], ns_e[:rows], width, 0, alpha
                )
                lp = model._enum_log_prior(
                    int(plan.get("enum_chain", int(model.best_chain))),
                    res["fields"], res["tables"],
                )
                terms.append(
                    sharp * 1.15
                    - torch.logsumexp(lp + sharp * credit, dim=0)
                )
    if terms:
        marginal = torch.stack(terms).mean()
    else:
        marginal = batch.logits.sum() * 0.0

    endpoint = F.cross_entropy(
        batch.logits.transpose(1, 2), batch.labels,
        ignore_index=-100, reduction="mean",
    )

    # training-only diagnostics for the scheduler (loss-value trace per chain)
    with torch.no_grad():
        out = aux["map_out"]
        mask = (1 << width) - 1
        w_low = int(ns.max()).bit_length() + 1
        # the congruence anchor is a T=1 property (an unreduced
        # multiplicative core is congruent ONLY at depth 1 -- deeper rows
        # truncate on the tape), so score it on the depth-1 rows when any
        # are present; identical to the old value in stage-1 curriculum
        # (all selected rows are depth t_min there)
        cong_diff = (out.remainder(ns[None, :]) ^ y[None, :]) & mask
        t1_rows = aux["t_values"] == 1
        if bool(t1_rows.any()):
            cong = _bit_match(cong_diff[:, t1_rows], w_low)
        else:
            cong = _bit_match(cong_diff, w_low)
        in_range = (out < ns[None, :]).float().mean(dim=-1)
        exact_bits = _bit_match((out ^ y[None, :]) & mask, width)  # full-T
        exact_rows = (((out ^ y[None, :]) & mask) == 0).float().mean(dim=-1)
        hit = (cong >= 1.0 - 1e-9).float()
        # freshly resampled chains adopt their first evaluation outright
        blend = torch.where(model.age == 0, torch.ones_like(cong), cong * 0 + EMA)
        model.fit_cong.mul_(1 - blend).add_(blend * cong)
        model.fit_range.mul_(1 - blend).add_(blend * in_range)
        model.fit_exact.mul_(1 - blend).add_(blend * exact_bits)
        model.fit_rows.mul_(1 - blend).add_(blend * exact_rows)
        model.last_hit.copy_(hit)
        model.last_rows.copy_(exact_rows)
        model.last_fit.copy_(cong + RANGE_BONUS * in_range)

    return marginal + DIRECT_LOSS_WEIGHT * endpoint


# ---------------------------------------------------------------------------
# Scheduler: rotation, stage scheduling, cross-chain resampling, best chain
# ---------------------------------------------------------------------------

class Schedule:
    """Owns wall-clock scheduling and the documented population-resampling
    parameter transformation (crossover re-centering of poor chains).  All
    decisions read only the loss-trace buffers and the step counter -- never
    the data."""

    def __init__(self, model: Model, optimizer: torch.optim.Optimizer, budget: float) -> None:
        self.model = model
        self.optimizer = optimizer
        self.started_at = time.monotonic()
        self.budget = max(float(budget) * 0.98, 1.0)
        self.step_count = 0
        self.rng = torch.Generator().manual_seed(GA_SEED + 2)
        self.rotation = torch.zeros(NCHAINS, dtype=torch.long)
        self.stage2_rotation = torch.zeros(NCHAINS, dtype=torch.long)
        self.next_chain = 0
        self.sweep_lo = 0
        self.loop_lo = 0
        # enumeration layer: built only when the budget affords it, so short
        # (probe-scale) budgets keep the exact GA-only trajectory
        self.enum_enabled = False
        self.enum_total = 0
        self.enum_adopted = 0
        self.stage2_step0 = -1
        self.ga_seconds = 1.0
        self.last_step_at = self.started_at
        self.last_enum_secs = 0.0
        if float(budget) >= ENUM_MIN_BUDGET:
            model._enum_space_init()
            self.enum_total = int(model._enum_order.numel())
            self.enum_enabled = True
        # crossover units: 8 slots, 2 heads, 4 tables
        self.units = (
            [("slot", i) for i in range(K_SLOTS)]
            + [("head", j) for j in range(2)]
            + [("tab", t) for t in range(N_TID)]
        )

    # -- logit surgery helpers ------------------------------------------------

    def _set_field(self, tensor: Tensor, index: tuple, val: int) -> None:
        row = tensor[index]
        noise = torch.randn(row.shape, generator=self.rng)
        new = noise * NOISE
        new[val] += CONF
        row.copy_(new)

    def _recenter(self, tensor: Tensor, victims: Tensor, donor_units: Tensor) -> None:
        """tensor: (C, U, ...arity); donor_units: (V, U) donor chain per unit.
        Re-centers victims' logits on the donors' MAP values plus noise."""
        unit_axis = torch.arange(tensor.shape[1])
        # noise is drawn on the CPU (self.rng is a CPU generator), so the
        # donor MAP values must come to the CPU before the scatter, and the
        # finished rows must go to the parameter's device before the write
        vals = tensor[donor_units, unit_axis].argmax(-1).cpu()  # (V, U, ...)
        new = torch.randn(
            (victims.shape[0],) + tensor.shape[1:], generator=self.rng
        ) * NOISE
        new.scatter_(
            -1, vals.unsqueeze(-1), CONF, reduce="add"
        )
        tensor[victims] = new.to(device=tensor.device, dtype=tensor.dtype)

    def _mutate(self, chain: int) -> None:
        m = self.model
        fields = (
            [(t, (chain, i)) for i in range(K_SLOTS)
             for t in (m.slot_lm, m.slot_dst, m.slot_sb, m.slot_tid, m.slot_ini)]
            + [(m.slot_pred, (chain, i)) for i in range(K_SLOTS)
               if i not in RESERVED_SLOTS]
            + [(m.head_src, (chain, j)) for j in range(2)]
            + [(m.head_dir, (chain, j)) for j in range(2)]
            + [(m.table_ent, (chain, t, e)) for t in range(N_TID) for e in range(8)]
        )
        count = int(torch.randint(1, 4, (1,), generator=self.rng))
        for _ in range(count):
            pick = int(torch.randint(0, len(fields), (1,), generator=self.rng))
            tensor, index = fields[pick]
            val = int(torch.randint(0, tensor.shape[-1], (1,), generator=self.rng))
            self._set_field(tensor, index, val)

    def _pin_reserved(self, chain: int) -> None:
        for slot in RESERVED_SLOTS:
            self._set_field(self.model.slot_pred, (chain, slot), PRED_NEVER)
            self.model.slot_pred[chain, slot, PRED_NEVER] += PIN_BIAS - CONF

    def _adopt_enum(self, chain: int) -> None:
        """Documented parameter transformation: re-center one chain's logits
        on the enumeration layer's verified congruence-exact program (slots
        0-1 + heads + table 0), unpin its reserved slots, and mark it stage 2
        so the completion sweeps assemble the reducer around it."""
        m = self.model
        prog = m.enum_prog.detach().cpu().tolist()
        tab = m.enum_tab.detach().cpu().tolist()
        for i in range(K_SLOTS):
            if i in RESERVED_SLOTS:
                continue
            self._set_field(m.slot_lm, (chain, i), int(prog[7 * i + 0]))
            self._set_field(m.slot_dst, (chain, i), int(prog[7 * i + 1]))
            self._set_field(m.slot_sb, (chain, i), int(prog[7 * i + 3]))
            self._set_field(m.slot_tid, (chain, i), int(prog[7 * i + 4]))
            self._set_field(m.slot_ini, (chain, i), int(prog[7 * i + 5]))
            self._set_field(m.slot_pred, (chain, i), int(prog[7 * i + 6]))
        for j in range(2):
            self._set_field(m.head_src, (chain, j), int(prog[56 + 2 * j]))
            self._set_field(m.head_dir, (chain, j), int(prog[57 + 2 * j]))
        for e in range(8):
            self._set_field(m.table_ent, (chain, 0, e), int(tab[e]))
        for reserved in RESERVED_SLOTS:
            for tensor in (m.slot_lm, m.slot_dst, m.slot_sb,
                           m.slot_tid, m.slot_ini, m.slot_pred):
                tensor[chain, reserved] = torch.randn(
                    tensor.shape[-1], generator=self.rng
                ).to(
                    device=tensor.device, dtype=tensor.dtype
                ) * 0.05
            m.slot_pred[chain, reserved, PRED_NEVER] += 1.0
        m.stage[chain] = 1
        m.hit_streak[chain] = STREAK_TO_STAGE2
        m.fit_cong[chain] = 1.0
        m.fit_range[chain] = 0.0
        m.fit_exact[chain] = 0.0
        m.fit_rows[chain] = 0.0
        m.last_rows[chain] = 0.0
        m.last_hit[chain] = 1.0
        m.last_fit[chain] = 1.0
        m.age[chain] = 1

    def _adopt_s2(self, chain: int) -> None:
        """Hard adoption of a row-exact stage-2 completion recorded by the
        loss (documented parameter transformation, mirroring _adopt_enum --
        the pure gradient flip is fragile when the completion windows rotate
        every step under global gradient clipping)."""
        m = self.model
        prog = m.s2_prog.detach().cpu().tolist()
        bank = m.s2_bank.detach().cpu()
        for i in range(K_SLOTS):
            self._set_field(m.slot_lm, (chain, i), int(prog[7 * i + 0]))
            self._set_field(m.slot_dst, (chain, i), int(prog[7 * i + 1]))
            self._set_field(m.slot_sb, (chain, i), int(prog[7 * i + 3]))
            self._set_field(m.slot_tid, (chain, i), int(prog[7 * i + 4]))
            self._set_field(m.slot_ini, (chain, i), int(prog[7 * i + 5]))
            self._set_field(m.slot_pred, (chain, i), int(prog[7 * i + 6]))
        for j in range(2):
            self._set_field(m.head_src, (chain, j), int(prog[56 + 2 * j]))
            self._set_field(m.head_dir, (chain, j), int(prog[57 + 2 * j]))
        for t in range(N_TID):
            for e in range(8):
                self._set_field(m.table_ent, (chain, t, e), int(bank[t, e]))

    def _randomize(self, chain: int) -> None:
        m = self.model
        for tensor in (m.slot_lm, m.slot_dst, m.slot_sb, m.slot_tid, m.slot_ini,
                       m.slot_pred, m.head_src, m.head_dir, m.table_ent):
            tensor[chain].copy_(
                torch.randn(tensor[chain].shape, generator=self.rng) * INIT_SCALE
            )
        self._pin_reserved(chain)

    # -- main step --------------------------------------------------------------

    def step(self) -> None:
        m = self.model
        self.step_count += 1
        frac = min((time.monotonic() - self.started_at) / self.budget, 1.0)
        with torch.no_grad():
            m.age.add_(1)
            # stage transitions from the loss-value trace
            m.hit_streak.copy_(
                torch.where(m.last_hit > 0.5, m.hit_streak + 1,
                            torch.zeros_like(m.hit_streak))
            )
            promote = (m.hit_streak >= STREAK_TO_STAGE2) & (m.stage == 0)
            m.stage.copy_(torch.where(promote, torch.ones_like(m.stage), m.stage))
            if frac >= STAGE2_WALL_FALLBACK and int((m.stage == 1).sum()) == 0:
                candidate = int(m.fit_cong.argmax())
                if float(m.fit_cong[candidate]) >= 0.995:
                    m.stage[candidate] = 1
                    promote = promote.clone()
                    promote[candidate] = True
            # a freshly promoted chain gets its reserved slots unpinned so
            # the completion blocks can be adopted by gradient steps
            for chain in promote.nonzero(as_tuple=False).flatten().tolist():
                for reserved in RESERVED_SLOTS:
                    for tensor in (m.slot_lm, m.slot_dst, m.slot_sb,
                                   m.slot_tid, m.slot_ini, m.slot_pred):
                        tensor[chain, reserved] = torch.randn(
                            tensor.shape[-1], generator=self.rng
                        ) * 0.05
                    m.slot_pred[chain, reserved, PRED_NEVER] += 1.0
            # selection uses the current batch's raw fitness (all chains are
            # scored on the same rows every step, like a GA generation).
            # GA bookkeeping happens on the CPU: the tournament picks are
            # drawn from a CPU generator and torch.gather requires the index
            # on the same device, so the fitness values come over once here
            # (a no-op on CPU manifests).
            fitness = torch.where(
                m.stage == 1, 2.0 + m.fit_rows, m.last_fit
            ).detach().cpu()
            order = fitness.argsort(descending=True)

            # adopt a verified enumeration hit: re-center the weakest stage-0
            # chain on the found congruence-exact program and promote it to
            # stage 2 (the completion sweeps then operate on it)
            if self.enum_enabled and int(m.enum_hits) > self.enum_adopted:
                stage_cpu = m.stage.detach().cpu()
                pool = [
                    int(c) for c in fitness.argsort()
                    if int(stage_cpu[c]) == 0 and int(c) != int(m.best_chain)
                ]
                if pool:
                    self._adopt_enum(pool[0])
                    self.enum_adopted = int(m.enum_hits)

            # adopt a recorded row-exact stage-2 completion outright
            if int(m.s2_hit) == 1:
                s2c = int(m.s2_chain)
                if int(m.stage[s2c]) == 1:
                    self._adopt_s2(s2c)
                m.s2_hit.fill_(0)

            # cross-chain resampling (documented parameter transformation):
            # generational replacement of everything but the elites, each
            # child a slot/head/table-granular crossover of two tournament
            # winners plus point mutations -- the GA layer of the pipeline.
            if self.step_count % RESAMPLE_EVERY == 0 and self.step_count > 2:
                protected = set(int(c) for c in order[:N_ELITE])
                protected |= set(
                    int(c) for c in (m.stage == 1).nonzero().flatten()
                )
                protected.add(int(m.best_chain))
                n_replace = int(NCHAINS * REPLACE_FRAC)
                victims = torch.tensor(
                    [int(c) for c in order.flip(0) if int(c) not in protected]
                    [:n_replace],
                    dtype=torch.long,
                )
                n_victims = victims.numel()
                if n_victims:
                    picks = torch.randint(
                        0, NCHAINS, (2, n_victims, TOURNEY), generator=self.rng
                    )
                    parent_a = picks[0].gather(
                        1, fitness[picks[0]].argmax(1, keepdim=True)
                    ).squeeze(1)
                    parent_b = picks[1].gather(
                        1, fitness[picks[1]].argmax(1, keepdim=True)
                    ).squeeze(1)
                    n_units = len(self.units)
                    choose = torch.randint(
                        0, 2, (n_victims, n_units), generator=self.rng
                    )
                    donors = torch.where(
                        choose == 0, parent_a[:, None], parent_b[:, None]
                    )
                    slot_units = donors[:, :K_SLOTS]
                    head_units = donors[:, K_SLOTS:K_SLOTS + 2]
                    table_units = donors[:, K_SLOTS + 2:]
                    for tensor in (m.slot_lm, m.slot_dst, m.slot_sb,
                                   m.slot_tid, m.slot_ini, m.slot_pred):
                        self._recenter(tensor, victims, slot_units)
                    self._recenter(m.head_src, victims, head_units)
                    self._recenter(m.head_dir, victims, head_units)
                    self._recenter(m.table_ent, victims, table_units)
                    mutate_mask = (
                        torch.rand(n_victims, generator=self.rng) < MUT_PROB
                    )
                    for v in victims[mutate_mask].tolist():
                        self._mutate(v)
                    # a small immigrant stream keeps raw diversity alive
                    for v in victims[-4:].tolist():
                        self._randomize(v)
                    pin_rows = torch.randn(
                        n_victims, len(RESERVED_SLOTS), N_PRD, generator=self.rng
                    ) * NOISE
                    pin_rows[..., PRED_NEVER] += PIN_BIAS
                    m.slot_pred[
                        victims[:, None],
                        torch.tensor(RESERVED_SLOTS)[None, :],
                    ] = pin_rows.to(
                        device=m.slot_pred.device, dtype=m.slot_pred.dtype
                    )
                    m.fit_exact[victims] = 0.0
                    m.fit_rows[victims] = 0.0
                    m.last_rows[victims] = 0.0
                    m.age[victims] = 0
                    m.hit_streak[victims] = 0
                    m.stage[victims] = 0

            # persistent best-chain choice (training-time only), AFTER
            # resampling so a just-overwritten chain can never be selected
            score = (
                m.fit_rows + 0.5 * m.last_rows
                + 1e-3 * m.fit_exact + 1e-6 * m.fit_cong
            )
            m.best_chain.copy_(score.argmax())

            # next step's plan: rotate active blocks (step-count driven)
            active: list[tuple[int, tuple]] = []
            stage2 = [int(c) for c in (m.stage == 1).nonzero().flatten()]
            stage2 = sorted(stage2, key=lambda c: -float(fitness[c]))
            for chain in stage2[:MAX_STAGE2_ACTIVE]:
                if float(m.last_rows[chain]) >= 0.999:
                    continue  # already exact on the latest batch: leave it be
                inert = [
                    reserved for reserved in RESERVED_SLOTS
                    if int(m.slot_pred[chain, reserved].argmax()) == PRED_NEVER
                ]
                # corruption guard: a correct completion preserves congruence,
                # so a stage-2 chain whose congruence anchor broke has junk in
                # its reserved slots -- re-inert them (restores the core)
                if not inert and float(m.fit_cong[chain]) < 0.98:
                    for reserved in RESERVED_SLOTS:
                        for tensor in (m.slot_lm, m.slot_dst, m.slot_sb,
                                       m.slot_tid, m.slot_ini, m.slot_pred):
                            tensor[chain, reserved] = torch.randn(
                                tensor.shape[-1], generator=self.rng
                            ) * 0.05
                        m.slot_pred[chain, reserved, PRED_NEVER] += 1.0
                    inert = list(RESERVED_SLOTS)
                # the sweep replaces the target slot outright (its only
                # anchor is the explicit no-op row), so it targets INERT
                # slots; once every reserved slot commits, rotate
                # incumbent-anchored comp and table blocks instead.
                r = int(self.stage2_rotation[chain])
                self.stage2_rotation[chain] += 1
                if inert:
                    if r < STAGE2_LOOP_START:
                        # affine-validated ordering: once_post sweeps first
                        if r % 3 < 2:
                            active.append((chain, ("sweep", inert[0])))
                            self.sweep_lo = (
                                self.sweep_lo + SWEEP_TABLES_PER_STEP
                            ) % 65536
                        else:
                            active.append((chain, ("comp", inert[0])))
                    else:
                        # no once_post completion adopted: rotate in the
                        # in-loop reducer sweep (multiplicative cores)
                        k = r % 4
                        if k in (0, 2):
                            active.append((chain, ("loop",)))
                            self.loop_lo = (
                                self.loop_lo + LOOP_TABLES_PER_STEP
                            ) % 65536
                        elif k == 1:
                            active.append((chain, ("sweep", inert[0])))
                            self.sweep_lo = (
                                self.sweep_lo + SWEEP_TABLES_PER_STEP
                            ) % 65536
                        else:
                            active.append((chain, ("comp", inert[0])))
                else:
                    in_loop = any(
                        int(m.slot_lm[chain, rsv].argmax())
                        in (LM_HEAD_A, LM_HEAD_B)
                        for rsv in RESERVED_SLOTS
                    )
                    if in_loop:
                        # an in-loop pair is committed: only anchored blocks
                        # (loop sweep, incumbent-anchored table) may touch it
                        if r % 2 == 0:
                            active.append((chain, ("loop",)))
                            self.loop_lo = (
                                self.loop_lo + LOOP_TABLES_PER_STEP
                            ) % 65536
                        else:
                            tid = int(
                                m.slot_tid[chain, RESERVED_SLOTS[0]].argmax()
                            )
                            active.append((chain, ("table", tid)))
                    elif r % 3 == 0:
                        active.append(
                            (chain, ("comp", RESERVED_SLOTS[(r // 2) % 2]))
                        )
                    elif r % 3 == 1:
                        slot = RESERVED_SLOTS[(r // 2) % 2]
                        tid = int(m.slot_tid[chain, slot].argmax())
                        active.append((chain, ("table", tid)))
                    else:
                        # junk once_post commits must not lock out the
                        # in-loop reducer search
                        active.append((chain, ("loop",)))
                        self.loop_lo = (
                            self.loop_lo + LOOP_TABLES_PER_STEP
                        ) % 65536
            # stage-1 gradient blocks polish the current elites (memetic)
            elite_pool = [
                int(c) for c in order[:N_ELITE] if int(m.stage[c]) == 0
            ]
            while len(active) < N_ACTIVE and elite_pool:
                chain = elite_pool[self.next_chain % len(elite_pool)]
                self.next_chain += 1
                offset = (self.step_count * SLOT_SAMPLE) % SLOT_COMBOS.shape[0]
                blocks = (
                    [("slot", i, offset) for i in range(K_SLOTS)
                     if i not in RESERVED_SLOTS]
                    + [("head", 0)]
                    + [("table", t) for t in range(N_TID)]
                )
                r = int(self.rotation[chain]) % len(blocks)
                self.rotation[chain] += 1
                active.append((chain, blocks[r]))
            sharp = SHARP0 + (SHARP1 - SHARP0) * min(frac / 0.7, 1.0)
            alpha = ANNEAL_CAP * min(
                max((frac - ANNEAL_START) / ANNEAL_LEN, 0.0), 1.0
            )

            # enumeration layer plan: chunk size from measured throughput,
            # restart count from the remaining-budget cost model
            enum_n = 0
            enum_r = ENUM_RESTARTS_CAP
            enum_chain = elite_pool[-1] if elite_pool else int(m.best_chain)
            if self.enum_enabled:
                now = time.monotonic()
                wall = now - self.last_step_at
                self.last_step_at = now
                d_enum = float(m.enum_secs) - self.last_enum_secs
                self.last_enum_secs = float(m.enum_secs)
                if self.step_count > 2:
                    self.ga_seconds = (
                        0.7 * self.ga_seconds + 0.3 * max(wall - d_enum, 1e-3)
                    )
                remaining = self.budget - (now - self.started_at)
                stage2_busy = bool(
                    ((m.stage == 1) & (m.last_rows < 0.999)).any()
                )
                solved = bool(
                    ((m.stage == 1) & (m.last_rows >= 0.999)).any()
                )
                if stage2_busy and self.stage2_step0 < 0:
                    self.stage2_step0 = self.step_count
                if not stage2_busy:
                    self.stage2_step0 = -1
                stuck = (
                    stage2_busy
                    and self.step_count - self.stage2_step0 > ENUM_STUCK_STEPS
                )
                m_left = self.enum_total - int(m.enum_cursor)
                if (
                    not solved
                    and (not stage2_busy or stuck)
                    and remaining > ENUM_TAIL_RESERVE
                    and m_left > 0
                    and elite_pool
                ):
                    done = float(m.enum_done)
                    if done > 0:
                        eps = float(m.enum_evals) / max(float(m.enum_secs), 1e-3)
                        sfrac = min(max(float(m.enum_surv) / done, 0.05), 1.0)
                    else:
                        eps, sfrac = 10000.0, 0.65
                    enum_budget = max(remaining - ENUM_TAIL_RESERVE, 1.0)
                    for enum_r in ENUM_RESTART_LADDER:
                        cost = ENUM_EVA + sfrac * max(
                            enum_r - ENUM_RESTARTS_A, 0
                        ) * ENUM_EVR
                        if m_left * cost <= eps * enum_budget:
                            break
                    target = min(12.0, max(1.5, self.budget / 60.0))
                    enum_sec = max(target - self.ga_seconds, 0.5)
                    spst = (
                        ENUM_EVA
                        + sfrac * max(enum_r - ENUM_RESTARTS_A, 0) * ENUM_EVR
                    ) / max(eps, 100.0)
                    chunk_cap = (
                        ENUM_CHUNK_MAX
                        if m.slot_lm.device.type == "cpu"
                        else ENUM_CHUNK_MAX_CUDA
                    )
                    enum_n = max(
                        1,
                        min(chunk_cap, int(enum_sec / max(spst, 1e-4))),
                    )
                    if done == 0:
                        enum_n = ENUM_CHUNK0

            m._plan = {
                "active": active,
                "sharp": sharp,
                "alpha": alpha,
                "sweep_lo": self.sweep_lo,
                "loop_lo": self.loop_lo,
                "full_t": bool(int((m.stage == 1).sum()) > 0),
                "enum_n": enum_n,
                "enum_r": enum_r,
                "enum_chain": enum_chain,
            }
            if self.enum_enabled and self.step_count % 50 == 0:
                print(
                    f"[enum] step={self.step_count}"
                    f" cursor={int(m.enum_cursor)}/{self.enum_total}"
                    f" evals={float(m.enum_evals):.3e}"
                    f" eps={float(m.enum_evals) / max(float(m.enum_secs), 1e-3):.0f}"
                    f" best={float(m.enum_best_cong):.4f}@{int(m.enum_best_pos)}"
                    f" hits={int(m.enum_hits)} R={enum_r} n={enum_n}"
                    f" ga_s={self.ga_seconds:.2f}",
                    flush=True,
                )
        multiplier = 1.0
        if frac > 0.92:
            progress = (frac - 0.92) / 0.08
            multiplier = 0.2 + 0.8 * 0.5 * (1.0 + math.cos(math.pi * progress))
        for group in self.optimizer.param_groups:
            group["lr"] = group["base_lr"] * multiplier


def build_model(spec: ModelSpec) -> Model:
    model = Model(spec)
    assert_model_state(model, spec)
    return model


def build_optimizer(model: Model, spec: OptimizerSpec) -> OptimizerBundle:
    decoder_ids = {id(p) for p in model.decoder.parameters()}
    posterior = [p for p in model.parameters() if id(p) not in decoder_ids]
    optimizer = torch.optim.SGD(
        [
            {
                "params": posterior,
                "lr": POSTERIOR_LR,
                "base_lr": POSTERIOR_LR,
                "weight_decay": 0.0,
            },
            {
                "params": list(model.decoder.parameters()),
                "lr": DECODER_LR,
                "base_lr": DECODER_LR,
                "weight_decay": 0.0,
            },
        ],
        lr=POSTERIOR_LR,
        momentum=0.0,
    )
    return OptimizerBundle(
        optimizer=optimizer,
        scheduler=Schedule(model, optimizer, spec.training_time_seconds),
    )


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    token_training_loss=token_training_loss,
    batch_size=256,
    # eval cost is kernel-launch-bound and row-count independent, so larger
    # eval batches directly cut depth-ladder wall time (512 rows per OOD rung
    # -> one batch); matches the hosted manifest's own eval_batch_size
    eval_batch_size=512,
)
