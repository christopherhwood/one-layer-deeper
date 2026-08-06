"""C1/C2 pre-training harness core for the learned-ISA machine.

C1 (injectivity): sample R random programs (uniform over ALL fields including
table entries) plus the three reference programs; execute each on a fixed
batch of inputs (N=323, random x, T=1); report the pairwise output-agreement
summary -- in particular the fraction of program pairs that agree on >50% of
inputs (aliasing pressure the posterior must overcome).

C2 (counterfactual necessity core): for each FIELD of the squaring reference
program, report the fraction of its single-field mutations that leave ALL
batch outputs unchanged. A load-bearing field should have fraction 0; nop
slots / unused tables are expected to show 1.

Pure ints, stdlib only. Run:
    .venv/bin/python submissions/learned_isa_machine/disagreement_sweep.py [R]
"""

import os
import random
import sys
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reference_machine import (  # noqa: E402
    HEAD_FIELD_ARITIES, K_SLOTS, L_TABLES, N_HEADS, SLOT_FIELD_ARITIES,
    TABLE_ENTRIES, TABLE_ENTRY_ARITY,
    affine_program, cube_program, default_width, execute,
    random_program, squaring_program,
)

BATCH_N = 323
BATCH_SIZE = 64
T_FIXED = 1
SEED = 2026


def output_vector(program, xs):
    return tuple(execute(program, BATCH_N, x, T_FIXED) for x in xs)


def pairwise_agreement_summary(programs, xs):
    outs = [output_vector(p, xs) for p in programs]
    n = len(outs)
    n_inputs = len(xs)
    pairs = 0
    agree_gt_half = 0
    identical = 0
    total_agree_frac = 0.0
    for i in range(n):
        oi = outs[i]
        for j in range(i + 1, n):
            oj = outs[j]
            same = sum(1 for a, b in zip(oi, oj) if a == b)
            frac = same / n_inputs
            pairs += 1
            total_agree_frac += frac
            if frac > 0.5:
                agree_gt_half += 1
            if same == n_inputs:
                identical += 1
    return {
        "pairs": pairs,
        "agree_gt_half": agree_gt_half,
        "identical": identical,
        "mean_agree_frac": total_agree_frac / pairs if pairs else 0.0,
        "outputs": outs,
    }


# ---------------------------------------------------------------------------
# C2 core: single-field mutations of the squaring reference
# ---------------------------------------------------------------------------

def _slot_mutants(program, slot_idx, field_name, arity):
    cur = getattr(program.slots[slot_idx], field_name)
    for alt in range(arity):
        if alt == cur:
            continue
        p = program.copy()
        p.slots[slot_idx] = replace(p.slots[slot_idx], **{field_name: alt})
        yield p


def _head_mutants(program, head_idx, field_name, arity):
    cur = getattr(program.heads[head_idx], field_name)
    for alt in range(arity):
        if alt == cur:
            continue
        p = program.copy()
        p.heads[head_idx] = replace(p.heads[head_idx], **{field_name: alt})
        yield p


def _table_entry_mutants(program, table_idx, entry_idx):
    out_bit, nxt = program.tables[table_idx][entry_idx]
    cur = (out_bit << 1) | nxt
    for alt in range(TABLE_ENTRY_ARITY):
        if alt == cur:
            continue
        p = program.copy()
        p.tables[table_idx][entry_idx] = (alt >> 1, alt & 1)
        yield p


def field_necessity_table(base_program, xs):
    """For each field, fraction of its single-field mutations whose outputs
    match the base program's on ALL inputs (1.0 = field fully non-necessary
    on this batch; 0.0 = every mutation changes some output)."""
    base_out = output_vector(base_program, xs)

    def preserved_fraction(mutants):
        total = 0
        preserved = 0
        for m in mutants:
            total += 1
            if output_vector(m, xs) == base_out:
                preserved += 1
        return preserved / total if total else 0.0

    rows = []  # (field_label, preserved_fraction)
    for k in range(K_SLOTS):
        for fname, arity in SLOT_FIELD_ARITIES:
            frac = preserved_fraction(_slot_mutants(base_program, k, fname, arity))
            rows.append((f"slot{k}.{fname}", frac))
    for h in range(N_HEADS):
        hname = "head_A" if h == 0 else "head_B"
        for fname, arity in HEAD_FIELD_ARITIES:
            frac = preserved_fraction(_head_mutants(base_program, h, fname, arity))
            rows.append((f"{hname}.{fname}", frac))
    for ti in range(L_TABLES):
        for ei in range(TABLE_ENTRIES):
            frac = preserved_fraction(_table_entry_mutants(base_program, ti, ei))
            rows.append((f"table{ti}.entry{ei}", frac))
    return rows


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def _print_slot_field_matrix(rows):
    fields = [f for f, _ in SLOT_FIELD_ARITIES]
    lookup = dict(rows)
    header = "        " + "".join(f"{f:>12}" for f in fields)
    print(header)
    for k in range(K_SLOTS):
        cells = "".join(f"{lookup[f'slot{k}.{f}']:>12.2f}" for f in fields)
        print(f"slot{k}   {cells}")


def _print_table_entry_matrix(rows):
    lookup = dict(rows)
    header = "         " + "".join(f"{('e' + str(e)):>7}" for e in range(TABLE_ENTRIES))
    print(header)
    for ti in range(L_TABLES):
        cells = "".join(f"{lookup[f'table{ti}.entry{e}']:>7.2f}" for e in range(TABLE_ENTRIES))
        print(f"table{ti}  {cells}")


def main():
    r = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    rng = random.Random(SEED)
    xs = [rng.randrange(BATCH_N) for _ in range(BATCH_SIZE)]

    refs = [squaring_program(), affine_program(), cube_program()]
    programs = refs + [random_program(rng) for _ in range(r)]

    print(f"disagreement sweep: N={BATCH_N}, batch={BATCH_SIZE} inputs, "
          f"T={T_FIXED}, W={default_width(BATCH_N)}, "
          f"{len(programs)} programs (3 refs + {r} random), seed={SEED}")

    summary = pairwise_agreement_summary(programs, xs)
    print()
    print("== C1: pairwise output agreement ==")
    print(f"pairs:                          {summary['pairs']}")
    print(f"pairs agreeing on >50% inputs:  {summary['agree_gt_half']} "
          f"({summary['agree_gt_half'] / summary['pairs']:.4%})")
    print(f"pairs identical on ALL inputs:  {summary['identical']} "
          f"({summary['identical'] / summary['pairs']:.4%})")
    print(f"mean pairwise agreement frac:   {summary['mean_agree_frac']:.4f}")

    # refs must be distinguishable from each other on this batch
    ref_outs = summary["outputs"][:3]
    names = ["squaring", "affine", "cube"]
    print()
    print("reference-pair agreement fractions:")
    for i in range(3):
        for j in range(i + 1, 3):
            same = sum(1 for a, b in zip(ref_outs[i], ref_outs[j]) if a == b)
            print(f"  {names[i]:>9} vs {names[j]:<9} {same}/{BATCH_SIZE}")

    print()
    print("== C2 core: single-field mutation output-preservation on the "
          "squaring reference ==")
    print("(cell = fraction of that field's mutations leaving ALL "
          f"{BATCH_SIZE} outputs unchanged; 0.00 = fully necessary)")
    rows = field_necessity_table(squaring_program(), xs)
    print()
    print("slot fields:")
    _print_slot_field_matrix(rows)
    print()
    lookup = dict(rows)
    print("head fields:")
    for h, hname in enumerate(["head_A", "head_B"]):
        vals = "  ".join(f"{f}={lookup[f'{hname}.{f}']:.2f}"
                         for f, _ in HEAD_FIELD_ARITIES)
        print(f"  {hname}: {vals}")
    print()
    print("table entries (entry index = state*4 + a_bit*2 + b_bit):")
    _print_table_entry_matrix(rows)

    necessary = [f for f, frac in rows if frac == 0.0]
    free = [f for f, frac in rows if frac == 1.0]
    print()
    print(f"fully necessary fields (every mutation breaks >=1 output): "
          f"{len(necessary)}/{len(rows)}")
    print(f"fully free fields (no mutation changes any output):        "
          f"{len(free)}/{len(rows)}")


if __name__ == "__main__":
    main()
