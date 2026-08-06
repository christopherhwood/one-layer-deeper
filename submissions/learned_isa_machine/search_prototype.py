"""Rotating-block exact-marginalization search prototype for the learned-ISA
tape machine (FROZEN ISA SPEC v1, see DESIGN_learned_isa.md and
reference_machine.py in this directory).

Derisking question: can rotating-block exact marginalization find correct
programs from random initialization, from endpoint-only data?

Mechanics (predecessor style, binary_relation_hard_control):
  * execution is ALWAYS discrete (C9): candidate alternatives of the active
    block are enumerated as a leading tensor dimension and executed with plain
    integer (long) arithmetic; NEVER soft-blended.
  * mean-field posterior = per-field logits; block log-prior over the active
    block's Cartesian product = sum of per-field log_softmax terms (C4).
  * loss = -logsumexp_alt(log_prior_alt + loglik_alt); loglik_alt =
    -BIT_PENALTY * (# mismatched answer bits over all rows).
  * Adam (lr 0.2) steps ONLY the active block's logits each block update.
  * C=4 independent chains; non-active fields from each chain's current
    argmax (MAP) or optionally sampled at a temperature (exploration knob).

Blocks (round-robin): per slot ROUTING (loop_mode x dst x pred = 48) and
DATAFLOW (src_a x src_b x table_id x init_state = 288); one HEAD block
((3*2)^2 = 36); 16 table-entry-pair blocks (2 entries x 4 choices = 16).
"""

import argparse
import json
import os
import random
import sys
import time

import torch
import torch.nn.functional as F

torch.set_num_threads(2)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reference_machine as RM  # noqa: E402

# ---------------------------------------------------------------------------
# Constants (must match FROZEN ISA SPEC v1 / reference_machine.py)
# ---------------------------------------------------------------------------

N_MOD = 323
W = N_MOD.bit_length() + 2          # 11
K = RM.K_SLOTS                       # 8
NCHAINS = 4
SHIFTS = torch.arange(W, dtype=torch.long).view(W, 1, 1)

SLOT_FIELDS = {"lm": 4, "dst": 2, "sa": 6, "sb": 6, "tid": 4, "ini": 2, "prd": 6}
HEAD_FIELDS = {"src": 3, "dir": 2}

# table entry value encoding: val in 0..3, out_bit = val & 1, next_state = val >> 1
def table_to_vals(table):
    """reference_machine Table (list of (out,next)) -> (8,) long tensor of vals."""
    return torch.tensor([o + 2 * n for (o, n) in table], dtype=torch.long)


FAMS = {
    "squaring": lambda v: (v * v) % N_MOD,
    "affine": lambda v: (3 * v + 1) % N_MOD,
    "cube": lambda v: (v * v * v) % N_MOD,
}

LM_NAMES = ["once_pre", "head_A", "head_B", "once_post"]
REG_NAMES = ["V", "N", "ZERO", "ONE", "ACC", "S1"]
PRD_NAMES = ["ALWAYS", "NEVER", "HEAD=1", "HEAD=0", "TERM=1", "TERM=0"]
HSRC_NAMES = ["V", "ACC", "S1"]
DIR_NAMES = ["MSB_first", "LSB_first"]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def gen_dataset(fam, data_seed=20260806):
    rng = random.Random(data_seed)
    f = FAMS[fam]

    def rows(n, t_choices):
        xs, ts, ys = [], [], []
        for _ in range(n):
            x = rng.randrange(N_MOD)
            T = rng.choice(t_choices)
            v = x
            for _ in range(T):
                v = f(v)
            xs.append(x)
            ts.append(T)
            ys.append(v)
        return (torch.tensor(xs), torch.tensor(ts), torch.tensor(ys))

    train = rows(600, [1, 2, 3])
    held = rows(150, [1, 2, 3])
    ood = rows(100, [6])
    return train, held, ood


# ---------------------------------------------------------------------------
# Discrete batched executor
# fields: python ints (static) or (A,1) long tensors (active block, enumerated)
# registers: broadcast-shaped long tensors (., rows)
# ---------------------------------------------------------------------------

class Prog:
    pass


def _scan(a, b, tid_off, ini, tv0, tv, rowidx):
    """SCAN per spec. a, b: (., rows); tid_off: int (tid*8) or (A,1);
    ini: int or (A,1); tv0: (32,) flat static tables or None;
    tv: (A,32) per-alternative tables (used when rowidx is not None)."""
    a2 = ((a.unsqueeze(0) >> SHIFTS) & 1) << 1     # (W, ., rows)
    b1 = (b.unsqueeze(0) >> SHIFTS) & 1
    s = ini
    out = 0
    for pos in range(W):
        idx = s * 4 + a2[pos] + b1[pos] + tid_off
        if rowidx is None:
            val = tv0[idx]
        else:
            val = tv[rowidx, idx]
        out = out + ((val & 1) << pos)
        s = val >> 1
    return out, s


def _read_operand(p, k, which, regs):
    code = getattr(p, which)[k]
    if isinstance(code, int):
        return regs[code]
    val = None
    for rcode, m in enumerate(p.src_aux[k][which]):
        if m is None:
            continue
        term = m * regs[rcode]
        val = term if val is None else val + term
    if val is None:  # all alternatives read ZERO
        return regs[2]
    return val


def _head_bit(p, j, i, regs):
    hs, hd = p.hsrc[j], p.hdir[j]
    if isinstance(hd, int):
        pos = i if hd == RM.DIR_LSB_FIRST else (W - 1 - i)
    else:
        pos = hd * i + (1 - hd) * (W - 1 - i)      # (A,1)
    if isinstance(hs, int):
        val = regs[(0, 4, 5)[hs]]
    else:
        m = p.head_aux[j]
        val = m[0] * regs[0] + m[1] * regs[4] + m[2] * regs[5]
    return (val >> pos) & 1


def _run_slot(p, k, regs, hb, phase):
    lm = p.lm[k]
    lm_mask = None if isinstance(lm, int) else p.lm_eq[k].get(phase)
    a = _read_operand(p, k, "sa", regs)
    b = _read_operand(p, k, "sb", regs)
    r, term = _scan(a, b, p.tid_off[k], p.ini[k], p.tv0, p.tv, p.rowidx)
    prd = p.prd[k]
    if isinstance(prd, int):
        sp, pol = prd >> 1, prd & 1
        if sp == RM.PRED_SRC_CONST:
            c = 1  # never-slots were excluded from phase lists
        elif sp == RM.PRED_SRC_HEAD_BIT:
            c = (1 - hb) if pol else hb
        else:
            c = (1 - term) if pol else term
    else:
        m0, m1, m2, pol = p.prd_aux[k]
        base = m0 + m1 * hb + m2 * term
        c = base ^ pol
    if isinstance(c, int) and c == 0:
        return
    if lm_mask is not None:
        c = lm_mask * c if not isinstance(c, int) else lm_mask
    dst = p.dst[k]
    if isinstance(c, int):  # c == 1: unconditional commit
        if isinstance(dst, int):
            regs[4 + dst] = r
        else:
            d0 = p.dst_aux[k]
            acc, s1 = regs[4], regs[5]
            regs[4] = acc + d0 * (r - acc)
            regs[5] = s1 + (1 - d0) * (r - s1)
    else:
        if isinstance(dst, int):
            old = regs[4 + dst]
            regs[4 + dst] = old + c * (r - old)
        else:
            d0 = p.dst_aux[k]
            acc, s1 = regs[4], regs[5]
            ca = d0 * c
            cs = c - ca
            regs[4] = acc + ca * (r - acc)
            regs[5] = s1 + cs * (r - s1)


def execute(p, x, trow, tmax):
    """Run the machine; returns latched outputs (A_or_1, rows) long."""
    rows = x.numel()
    V = x.view(1, rows)
    NREG = torch.full((1, 1), N_MOD, dtype=torch.long)
    ZERO = torch.zeros(1, 1, dtype=torch.long)
    ONE = torch.ones(1, 1, dtype=torch.long)
    latched = torch.zeros(1, rows, dtype=torch.long)
    tcmp = trow.view(1, rows)
    for t in range(1, tmax + 1):
        regs = [V, NREG, ZERO, ONE, ZERO, ZERO]
        for k in p.phase_slots[0]:
            _run_slot(p, k, regs, 0, 0)
        for ph in (1, 2):
            if p.phase_slots[ph]:
                for i in range(W):
                    hb = _head_bit(p, ph - 1, i, regs)
                    for k in p.phase_slots[ph]:
                        _run_slot(p, k, regs, hb, ph)
        for k in p.phase_slots[3]:
            _run_slot(p, k, regs, 0, 3)
        acc = regs[4]
        latched = torch.where(tcmp == t, acc, latched)
        V = acc if acc.size(-1) == rows else acc.expand(acc.size(0), rows)
    return latched


def _finalize(p):
    p.tid_off = [t * 8 for t in p.tid]
    if p.tv.size(0) == 1:
        p.tv0, p.rowidx = p.tv[0], None
    else:
        p.tv0, p.rowidx = None, torch.arange(p.tv.size(0)).view(-1, 1)
    p.src_aux = [dict() for _ in range(K)]
    p.prd_aux = [None] * K
    p.dst_aux = [None] * K
    for k in range(K):
        for which in ("sa", "sb"):
            code = getattr(p, which)[k]
            if not isinstance(code, int):
                v = code.view(-1)
                masks = []
                for r in range(6):
                    if r == 2:
                        masks.append(None)  # ZERO contributes nothing
                        continue
                    m = v == r
                    masks.append(m.long().view(-1, 1) if bool(m.any()) else None)
                p.src_aux[k][which] = masks
        prd = p.prd[k]
        if not isinstance(prd, int):
            v = prd.view(-1)
            sp, pol = v >> 1, v & 1
            p.prd_aux[k] = ((sp == 0).long().view(-1, 1),
                            (sp == 1).long().view(-1, 1),
                            (sp == 2).long().view(-1, 1),
                            pol.view(-1, 1))
        dst = p.dst[k]
        if not isinstance(dst, int):
            p.dst_aux[k] = (dst.view(-1) == 0).long().view(-1, 1)
    p.head_aux = [None, None]
    for j in range(2):
        hs = p.hsrc[j]
        if not isinstance(hs, int):
            v = hs.view(-1)
            p.head_aux[j] = [(v == r).long().view(-1, 1) for r in range(3)]
    p.lm_eq = [None] * K
    p.phase_slots = [[] for _ in range(4)]
    for k in range(K):
        prd = p.prd[k]
        if isinstance(prd, int) and prd == RM.PRED_NEVER:
            continue  # provably a no-op in every alternative
        lm = p.lm[k]
        if isinstance(lm, int):
            p.phase_slots[lm].append(k)
        else:
            v = lm.view(-1)
            eqs = {}
            for ph in range(4):
                m = v == ph
                if bool(m.any()):
                    eqs[ph] = m.long().view(-1, 1)
                    p.phase_slots[ph].append(k)
            p.lm_eq[k] = eqs


def build_prog(ctx, block=None, chain=None):
    """ctx: dict ('slot',k,f)->int, ('head',j,f)->int, 'table'->(4,8) long vals.
    block: None (pure MAP program) or ('routing',k)/('dataflow',k)/('head',)/
    ('table',ti,pair). Returns (Prog, log_prior (A,) or None)."""
    p = Prog()
    for f in SLOT_FIELDS:
        setattr(p, f, [ctx[("slot", k, f)] for k in range(K)])
    p.hsrc = [ctx[("head", 0, "src")], ctx[("head", 1, "src")]]
    p.hdir = [ctx[("head", 0, "dir")], ctx[("head", 1, "dir")]]
    tv = ctx["table"].reshape(1, 32)
    A, lp = 1, None
    if block is not None:
        lg = chain.params
        ls = lambda key: F.log_softmax(lg[key], dim=-1)  # noqa: E731
        kind = block[0]
        if kind == "routing":
            k = block[1]
            A = 48
            ar = torch.arange(A)
            lmv, dstv, prdv = ar // 12, (ar // 6) % 2, ar % 6
            p.lm[k] = lmv.view(A, 1)
            p.dst[k] = dstv.view(A, 1)
            p.prd[k] = prdv.view(A, 1)
            lp = (ls(("slot", k, "lm"))[lmv] + ls(("slot", k, "dst"))[dstv]
                  + ls(("slot", k, "prd"))[prdv])
        elif kind == "dataflow":
            k = block[1]
            A = 288
            ar = torch.arange(A)
            sav, sbv = ar // 48, (ar // 8) % 6
            tidv, iniv = (ar // 2) % 4, ar % 2
            p.sa[k] = sav.view(A, 1)
            p.sb[k] = sbv.view(A, 1)
            p.tid[k] = tidv.view(A, 1)
            p.ini[k] = iniv.view(A, 1)
            lp = (ls(("slot", k, "sa"))[sav] + ls(("slot", k, "sb"))[sbv]
                  + ls(("slot", k, "tid"))[tidv] + ls(("slot", k, "ini"))[iniv])
        elif kind == "head":
            A = 36
            ar = torch.arange(A)
            sA, dA = ar // 12, (ar // 6) % 2
            sB, dB = (ar // 2) % 3, ar % 2
            p.hsrc = [sA.view(A, 1), sB.view(A, 1)]
            p.hdir = [dA.view(A, 1), dB.view(A, 1)]
            lp = (ls(("head", 0, "src"))[sA] + ls(("head", 0, "dir"))[dA]
                  + ls(("head", 1, "src"))[sB] + ls(("head", 1, "dir"))[dB])
        elif kind == "table":
            ti, pr = block[1], block[2]
            A = 16
            ar = torch.arange(A)
            c0, c1 = ar // 4, ar % 4
            tvA = tv.expand(A, 32).clone()
            tvA[:, ti * 8 + 2 * pr] = c0
            tvA[:, ti * 8 + 2 * pr + 1] = c1
            tv = tvA
            tl = lg[("table",)]
            lp = (F.log_softmax(tl[ti, 2 * pr], -1)[c0]
                  + F.log_softmax(tl[ti, 2 * pr + 1], -1)[c1])
        else:
            raise ValueError(block)
    p.tv = tv
    p.A = A
    _finalize(p)
    return p, lp


def popcount(x):
    return ((x.unsqueeze(0) >> SHIFTS) & 1).sum(0)


# ---------------------------------------------------------------------------
# Chains (mean-field posterior + Adam)
# ---------------------------------------------------------------------------

class Chain:
    def __init__(self, seed, init_scale=0.1, lr=0.2):
        g = torch.Generator().manual_seed(seed)
        self.gen = torch.Generator().manual_seed(seed + 999)
        self.params = {}
        for k in range(K):
            for f, n in SLOT_FIELDS.items():
                self.params[("slot", k, f)] = (
                    torch.randn(n, generator=g) * init_scale).requires_grad_()
        for j in range(2):
            for f, n in HEAD_FIELDS.items():
                self.params[("head", j, f)] = (
                    torch.randn(n, generator=g) * init_scale).requires_grad_()
        self.params[("table",)] = (
            torch.randn(4, 8, 4, generator=g) * init_scale).requires_grad_()
        self.opt = torch.optim.Adam(self.params.values(), lr=lr)

    def ctx(self, mode="map", tau=0.0):
        c = {}
        for key, t in self.params.items():
            td = t.detach()
            if key[0] == "table":
                if mode == "map" or tau <= 1e-3:
                    c["table"] = td.argmax(-1)
                else:
                    probs = F.softmax(td / tau, -1).view(-1, 4)
                    c["table"] = torch.multinomial(
                        probs, 1, generator=self.gen).view(4, 8)
            else:
                if mode == "map" or tau <= 1e-3:
                    c[key] = int(td.argmax())
                else:
                    pr = F.softmax(td / tau, 0)
                    c[key] = int(torch.multinomial(pr, 1, generator=self.gen))
        return c


def block_list():
    bl = []
    for k in range(K):
        bl.append(("routing", k))
        bl.append(("dataflow", k))
    bl.append(("head",))
    for ti in range(4):
        for pr in range(4):
            bl.append(("table", ti, pr))
    return bl  # 33 blocks


# ---------------------------------------------------------------------------
# Reference-program contexts (for the mandatory executor cross-check)
# ---------------------------------------------------------------------------

def prog_to_ctx(prog):
    ctx = {}
    for k, s in enumerate(prog.slots):
        ctx[("slot", k, "lm")] = s.loop_mode
        ctx[("slot", k, "dst")] = s.dst
        ctx[("slot", k, "sa")] = s.src_a
        ctx[("slot", k, "sb")] = s.src_b
        ctx[("slot", k, "tid")] = s.table_id
        ctx[("slot", k, "ini")] = s.init_state
        ctx[("slot", k, "prd")] = s.predicate
    for j, h in enumerate(prog.heads):
        ctx[("head", j, "src")] = h.source
        ctx[("head", j, "dir")] = h.direction
    ctx["table"] = torch.stack([table_to_vals(t) for t in prog.tables])
    return ctx


def cross_check(verbose=True):
    """(1) Reference programs through THIS executor vs ground truth on 200
    random rows (plus OOD T=6). (2) THIS executor vs reference_machine.execute
    on random programs. (3) Block-enumerated batched execution vs per-
    alternative reference execution for every block kind."""
    ok = True
    rng = random.Random(555)
    xs = torch.tensor([rng.randrange(N_MOD) for _ in range(200)])
    ts = torch.tensor([rng.choice([1, 2, 3]) for _ in range(200)])
    xs6 = torch.tensor([rng.randrange(N_MOD) for _ in range(100)])
    ts6 = torch.full((100,), 6)
    fams = [("squaring", RM.squaring_program), ("affine", RM.affine_program),
            ("cube", RM.cube_program)]
    for name, ctor in fams:
        f = FAMS[name]

        def gt(x, t):
            v = x
            for _ in range(t):
                v = f(v)
            return v

        ctx = prog_to_ctx(ctor())
        p, _ = build_prog(ctx)
        for xt, tt, tag in ((xs, ts, "T1-3"), (xs6, ts6, "T6")):
            out = execute(p, xt, tt, int(tt.max())).view(-1)
            want = torch.tensor([gt(int(x), int(t)) for x, t in zip(xt, tt)])
            agree = float((out == want).float().mean())
            ok &= agree == 1.0
            if verbose:
                print(f"cross-check ref-program {name} [{tag}]: "
                      f"agreement {agree:.4f} ({'PASS' if agree == 1.0 else 'FAIL'})")
    # (2) random programs, torch executor vs pure-python reference machine
    rrng = random.Random(4242)
    xs2 = torch.tensor([rrng.randrange(N_MOD) for _ in range(40)])
    ts2 = torch.tensor([rrng.choice([1, 2, 3]) for _ in range(40)])
    for trial in range(30):
        prog = RM.random_program(rrng)
        ctx = prog_to_ctx(prog)
        p, _ = build_prog(ctx)
        out = execute(p, xs2, ts2, int(ts2.max())).view(-1)
        want = torch.tensor([RM.execute(prog, N_MOD, int(x), int(t), width=W)
                             for x, t in zip(xs2, ts2)])
        if not bool((out == want).all()):
            ok = False
            print(f"cross-check random-program trial {trial}: FAIL")
    if verbose:
        print("cross-check 30 random programs vs reference_machine: "
              + ("PASS" if ok else "FAIL"))
    # (3) batched block enumeration == per-alternative reference execution
    chain = Chain(123)
    ctx = chain.ctx("map")
    xs3, ts3 = xs2[:24], ts2[:24]
    for block in [("routing", 2), ("dataflow", 5), ("head",), ("table", 1, 3)]:
        p, _ = build_prog(ctx, block, chain)
        with torch.no_grad():
            out = execute(p, xs3, ts3, int(ts3.max()))
        if out.size(0) == 1:
            out = out.expand(p.A, out.size(1))
        for alt in range(p.A):
            ctx_a = dict(ctx)
            ctx_a["table"] = ctx["table"].clone()
            kind = block[0]
            if kind == "routing":
                k = block[1]
                ctx_a[("slot", k, "lm")] = alt // 12
                ctx_a[("slot", k, "dst")] = (alt // 6) % 2
                ctx_a[("slot", k, "prd")] = alt % 6
            elif kind == "dataflow":
                k = block[1]
                ctx_a[("slot", k, "sa")] = alt // 48
                ctx_a[("slot", k, "sb")] = (alt // 8) % 6
                ctx_a[("slot", k, "tid")] = (alt // 2) % 4
                ctx_a[("slot", k, "ini")] = alt % 2
            elif kind == "head":
                ctx_a[("head", 0, "src")] = alt // 12
                ctx_a[("head", 0, "dir")] = (alt // 6) % 2
                ctx_a[("head", 1, "src")] = (alt // 2) % 3
                ctx_a[("head", 1, "dir")] = alt % 2
            else:
                ti, pr = block[1], block[2]
                ctx_a["table"][ti, 2 * pr] = alt // 4
                ctx_a["table"][ti, 2 * pr + 1] = alt % 4
            pa, _ = build_prog(ctx_a)
            oa = execute(pa, xs3, ts3, int(ts3.max())).view(-1)
            if not bool((out[alt] == oa).all()):
                ok = False
                print(f"cross-check block {block} alt {alt}: FAIL")
                break
        if verbose:
            print(f"cross-check block-enumeration {block}: "
                  + ("PASS" if ok else "FAIL"))
    print("CROSS-CHECK OVERALL: " + ("PASS" if ok else "FAIL"))
    return ok


# ---------------------------------------------------------------------------
# Decoding MAP programs to human-readable form
# ---------------------------------------------------------------------------

def classify_table(vals):
    """vals: (8,) long tensor of table entry vals."""
    named = {"ADD": table_to_vals(RM.add_table()),
             "SUB": table_to_vals(RM.sub_table()),
             "XOR": table_to_vals(RM.xor_table()),
             "COPY_A": table_to_vals(RM.copy_a_table())}
    for name, tv in named.items():
        if bool((vals == tv).all()):
            return name
    return "custom" + str([int(v) for v in vals])


def fmt_ctx(ctx):
    lines = []
    lines.append("  head_A: src=%s dir=%s | head_B: src=%s dir=%s" % (
        HSRC_NAMES[ctx[("head", 0, "src")]], DIR_NAMES[ctx[("head", 0, "dir")]],
        HSRC_NAMES[ctx[("head", 1, "src")]], DIR_NAMES[ctx[("head", 1, "dir")]]))
    for k in range(K):
        g = lambda f: ctx[("slot", k, f)]  # noqa: E731
        lines.append(
            "  slot%d [%s] %s <- T%d(%s,%s) init=%d if %s%s" % (
                k, LM_NAMES[g("lm")], REG_NAMES[4 + g("dst")], g("tid"),
                REG_NAMES[g("sa")], REG_NAMES[g("sb")], g("ini"),
                PRD_NAMES[g("prd")],
                "  (NO-OP)" if g("prd") == RM.PRED_NEVER else ""))
    for ti in range(4):
        lines.append("  table%d: %s" % (ti, classify_table(ctx["table"][ti])))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def eval_ctx(ctx, x, t, y):
    p, _ = build_prog(ctx)
    with torch.no_grad():
        out = execute(p, x, t, int(t.max()))
    return float((out.view(-1) == y).float().mean())


def train_run(fam, bp, seed, tag, log_dir, budget_updates=4000, budget_sec=480,
              ctx_mode="map", tau0=1.0, curriculum=False, rows_per_update=0,
              eval_every=50, lr=0.2, init_scale=0.1):
    (xt, tt, yt), (xh, th, yh), (xo, to, yo) = gen_dataset(fam)
    chains = [Chain(seed * 1000 + 17 * c + 1, init_scale=init_scale, lr=lr)
              for c in range(NCHAINS)]
    blocks = block_list()
    os.makedirs(log_dir, exist_ok=True)
    logf = open(os.path.join(log_dir, tag + ".jsonl"), "w")
    meta = dict(tag=tag, fam=fam, bp=bp, seed=seed, ctx_mode=ctx_mode,
                tau0=tau0, curriculum=curriculum,
                rows_per_update=rows_per_update, budget_updates=budget_updates,
                budget_sec=budget_sec, lr=lr, init_scale=init_scale)
    logf.write(json.dumps({"meta": meta}) + "\n")
    print(f"=== run {tag}: {meta}")
    t0 = time.time()
    row_rng = torch.Generator().manual_seed(seed + 31337)
    t1_idx = (tt == 1).nonzero().view(-1)
    converged = None
    step = 0
    result = dict(meta=meta)

    def metrics(step):
        ms = []
        for ci, ch in enumerate(chains):
            ctx = ch.ctx("map")
            tr = eval_ctx(ctx, xt, tt, yt)
            he = eval_ctx(ctx, xh, th, yh)
            ms.append(dict(chain=ci, train=tr, held=he))
        return ms

    while step < budget_updates:
        wall = time.time() - t0
        if wall > budget_sec:
            break
        frac = max(step / max(budget_updates, 1), wall / max(budget_sec, 1))
        # row selection
        if curriculum and frac < 0.5:
            idx = t1_idx
        else:
            idx = torch.arange(xt.numel())
        if rows_per_update and rows_per_update < idx.numel():
            perm = torch.randperm(idx.numel(), generator=row_rng)
            idx = idx[perm[:rows_per_update]]
        xb, tb, yb = xt[idx], tt[idx], yt[idx]
        tmax = int(tb.max())
        block = blocks[step % len(blocks)]
        tau = tau0 * max(0.0, 1.0 - frac) if ctx_mode == "sample" else 0.0
        for ch in chains:
            ctx = ch.ctx(ctx_mode, tau)
            p, lp = build_prog(ctx, block, ch)
            with torch.no_grad():
                out = execute(p, xb, tb, tmax)
                mism = popcount(out ^ yb.view(1, -1)).sum(1)
            loglik = mism.to(torch.float32) * (-float(bp))
            loss = -torch.logsumexp(lp + loglik, 0)
            ch.opt.zero_grad(set_to_none=True)
            loss.backward()
            ch.opt.step()
        step += 1
        if step % eval_every == 0 or step == 1:
            wall = time.time() - t0
            ms = metrics(step)
            rec = dict(step=step, wall=round(wall, 1), metrics=ms,
                       ups=round(step / max(wall, 1e-9), 2))
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
            print(f"[{tag}] step {step} wall {wall:.0f}s "
                  + " ".join(f"c{m['chain']}:tr={m['train']:.3f},he={m['held']:.3f}"
                             for m in ms))
            for m in ms:
                if m["train"] == 1.0 and m["held"] == 1.0:
                    converged = dict(step=step, wall=round(wall, 1),
                                     chain=m["chain"])
                    break
            if converged:
                break
    wall = time.time() - t0
    ms = metrics(step)
    # OOD + decoded programs for converged/best chains
    finals = []
    for ci, ch in enumerate(chains):
        ctx = ch.ctx("map")
        ood = eval_ctx(ctx, xo, to, yo)
        finals.append(dict(chain=ci, train=ms[ci]["train"], held=ms[ci]["held"],
                           ood_t6=ood))
    best = max(finals, key=lambda m: (m["train"], m["held"]))
    result.update(converged=converged, steps=step, wall=round(wall, 1),
                  finals=finals, best_chain=best["chain"])
    decoded = {}
    for ci, ch in enumerate(chains):
        ctx = ch.ctx("map")
        decoded[ci] = fmt_ctx(ctx)
        ctx_ser = {"slots": [[ctx[("slot", k, f)] for f in SLOT_FIELDS]
                             for k in range(K)],
                   "heads": [[ctx[("head", j, f)] for f in HEAD_FIELDS]
                             for j in range(2)],
                   "table": ctx["table"].tolist()}
        result.setdefault("map_ctx", {})[ci] = ctx_ser
    logf.write(json.dumps({"result": {k: v for k, v in result.items()
                                      if k != "meta"}}) + "\n")
    logf.close()
    print(f"=== {tag} DONE: converged={converged} wall={wall:.0f}s "
          f"steps={step}")
    for f in finals:
        print(f"    chain{f['chain']}: train={f['train']:.3f} "
              f"held={f['held']:.3f} ood_t6={f['ood_t6']:.3f}")
    print(f"--- MAP program, best chain {best['chain']}:")
    print(decoded[best["chain"]])
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("crosscheck")
    b = sub.add_parser("bench")
    b.add_argument("--steps", type=int, default=33)
    b.add_argument("--rows", type=int, default=0)
    r = sub.add_parser("run")
    r.add_argument("--family", required=True, choices=list(FAMS))
    r.add_argument("--bp", type=float, required=True)
    r.add_argument("--seed", type=int, default=74)
    r.add_argument("--tag", required=True)
    r.add_argument("--budget-updates", type=int, default=4000)
    r.add_argument("--budget-sec", type=float, default=480)
    r.add_argument("--ctx", default="map", choices=["map", "sample"])
    r.add_argument("--tau0", type=float, default=1.0)
    r.add_argument("--curriculum", action="store_true")
    r.add_argument("--rows", type=int, default=0)
    r.add_argument("--eval-every", type=int, default=50)
    r.add_argument("--log-dir", default=os.environ.get(
        "PROTO_LOG_DIR", "/tmp/proto_logs"))
    args = ap.parse_args()
    if args.cmd == "crosscheck":
        ok = cross_check()
        sys.exit(0 if ok else 1)
    if args.cmd == "bench":
        (xt, tt, yt), _, _ = gen_dataset("squaring")
        if args.rows:
            xt, tt, yt = xt[:args.rows], tt[:args.rows], yt[:args.rows]
        chain = Chain(7)
        blocks = block_list()
        t0 = time.time()
        for step in range(args.steps):
            block = blocks[step % len(blocks)]
            ctx = chain.ctx("map")
            p, lp = build_prog(ctx, block, chain)
            with torch.no_grad():
                out = execute(p, xt, tt, int(tt.max()))
                mism = popcount(out ^ yt.view(1, -1)).sum(1)
            loglik = mism.to(torch.float32) * -3.0
            loss = -torch.logsumexp(lp + loglik, 0)
            chain.opt.zero_grad(set_to_none=True)
            loss.backward()
            chain.opt.step()
        dt = time.time() - t0
        print(f"bench: {args.steps} single-chain block updates on "
              f"{xt.numel()} rows in {dt:.2f}s "
              f"({dt / args.steps * 1000:.0f} ms/update/chain; x{NCHAINS} "
              f"chains -> {dt / args.steps * NCHAINS:.2f} s/update)")
        return
    if args.cmd == "run":
        train_run(args.family, args.bp, args.seed, args.tag, args.log_dir,
                  budget_updates=args.budget_updates,
                  budget_sec=args.budget_sec, ctx_mode=args.ctx,
                  tau0=args.tau0, curriculum=args.curriculum,
                  rows_per_update=args.rows, eval_every=args.eval_every)


if __name__ == "__main__":
    main()
