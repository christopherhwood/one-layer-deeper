"""Tests for the FROZEN ISA SPEC v1 reference machine (G0 expressibility gate).

Runnable either as a plain script:
    .venv/bin/python submissions/learned_isa_machine/test_reference_machine.py
or via pytest:
    .venv/bin/python -m pytest submissions/learned_isa_machine/test_reference_machine.py
"""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reference_machine import (  # noqa: E402
    K_SLOTS, L_TABLES, N_DST, N_INIT_STATES, N_LOOP_MODES, N_SRC,
    PRED_NEVER, Program,
    affine_program, cube_program, default_width, execute, squaring_program,
)

# ---------------------------------------------------------------------------
# Ground truths (per T-step recurrences, iterated T times)
# ---------------------------------------------------------------------------

def gt_squaring(n: int, x: int, t: int) -> int:
    # iterated v -> v^2 mod N == pow(x, 2**T, N)
    return pow(x, 2 ** t, n)


def gt_affine(n: int, x: int, t: int) -> int:
    v = x % n
    for _ in range(t):
        v = (3 * v + 1) % n
    return v


def gt_cube(n: int, x: int, t: int) -> int:
    v = x % n
    for _ in range(t):
        v = pow(v, 3, n)
    return v


FAMILIES = [
    ("squaring", squaring_program, gt_squaring),
    ("affine", affine_program, gt_affine),
    ("cube", cube_program, gt_cube),
]


def _random_odd_moduli(rng: random.Random, count: int = 20):
    """Random odd moduli of 8-13 bits, biased toward semiprimes."""
    small_primes = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
                    53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107]
    moduli = set()
    while len(moduli) < count:
        if rng.random() < 0.7:
            p = rng.choice(small_primes)
            q = rng.choice(small_primes)
            n = p * q
        else:
            n = rng.randrange(2 ** 7, 2 ** 13) | 1
        if n % 2 == 1 and 8 <= n.bit_length() <= 13:
            moduli.add(n)
    return sorted(moduli)


def _sample_cases(rng: random.Random, num: int = 200):
    moduli = [323] + _random_odd_moduli(rng)
    cases = []
    for _ in range(num):
        n = rng.choice(moduli)
        x = rng.randrange(n)
        t = rng.choice([1, 2, 3, 6])
        cases.append((n, x, t))
    return cases


# ---------------------------------------------------------------------------
# G0 expressibility
# ---------------------------------------------------------------------------

def test_g0_expressibility_random_cases():
    rng = random.Random(12345)
    cases = _sample_cases(rng, num=200)
    for name, ctor, gt in FAMILIES:
        prog = ctor()
        for (n, x, t) in cases:
            got = execute(prog, n, x, t)
            want = gt(n, x, t)
            assert got == want, (
                f"{name}: N={n} x={x} T={t}: got {got}, want {want}")


def test_g0_edge_cases():
    n = 323
    for name, ctor, gt in FAMILIES:
        prog = ctor()
        for x in (0, 1, n - 1):
            for t in (1, 64):
                got = execute(prog, n, x, t)
                want = gt(n, x, t)
                assert got == want, (
                    f"{name} edge: N={n} x={x} T={t}: got {got}, want {want}")


# ---------------------------------------------------------------------------
# Determinism / width robustness
# ---------------------------------------------------------------------------

def test_width_robustness():
    rng = random.Random(777)
    moduli = [323] + _random_odd_moduli(rng, count=5)
    for name, ctor, gt in FAMILIES:
        prog = ctor()
        for n in moduli:
            w = default_width(n)
            for _ in range(8):
                x = rng.randrange(n)
                t = rng.choice([1, 2, 3])
                base = execute(prog, n, x, t, width=w)
                wide = execute(prog, n, x, t, width=w + 4)
                assert base == wide, (
                    f"{name}: width {w}->{w + 4} changed result for "
                    f"N={n} x={x} T={t}: {base} vs {wide}")
                assert base == gt(n, x, t)


def test_determinism():
    for name, ctor, _ in FAMILIES:
        prog = ctor()
        a = execute(prog, 323, 200, 3)
        b = execute(prog, 323, 200, 3)
        assert a == b, f"{name}: nondeterministic result"


# ---------------------------------------------------------------------------
# No-op check: predicate=never slots do not affect output
# ---------------------------------------------------------------------------

def test_never_slots_are_noops():
    rng = random.Random(999)
    n = 323
    inputs = [(rng.randrange(n), rng.choice([1, 2, 3])) for _ in range(16)]
    for name, ctor, _ in FAMILIES:
        base_prog = ctor()
        never_idx = [i for i, s in enumerate(base_prog.slots)
                     if s.predicate == PRED_NEVER]
        base_out = [execute(base_prog, n, x, t) for (x, t) in inputs]
        for trial in range(10):
            prog = base_prog.copy()
            for i in never_idx:
                s = prog.slots[i]
                s.loop_mode = rng.randrange(N_LOOP_MODES)
                s.dst = rng.randrange(N_DST)
                s.src_a = rng.randrange(N_SRC)
                s.src_b = rng.randrange(N_SRC)
                s.table_id = rng.randrange(L_TABLES)
                s.init_state = rng.randrange(N_INIT_STATES)
                # predicate stays PRED_NEVER
            out = [execute(prog, n, x, t) for (x, t) in inputs]
            assert out == base_out, (
                f"{name}: never-slot mutation (trial {trial}) changed output")


# ---------------------------------------------------------------------------
# structural sanity
# ---------------------------------------------------------------------------

def test_program_shape():
    for _, ctor, _ in FAMILIES:
        prog = ctor()
        assert len(prog.slots) == K_SLOTS
        assert len(prog.heads) == 2
        assert len(prog.tables) == L_TABLES
        for t in prog.tables:
            assert len(t) == 8
            for (out_bit, nxt) in t:
                assert out_bit in (0, 1) and nxt in (0, 1)


def _main():
    tests = [
        ("test_program_shape", test_program_shape),
        ("test_determinism", test_determinism),
        ("test_g0_edge_cases", test_g0_edge_cases),
        ("test_width_robustness", test_width_robustness),
        ("test_never_slots_are_noops", test_never_slots_are_noops),
        ("test_g0_expressibility_random_cases", test_g0_expressibility_random_cases),
    ]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {name}: {e}")
    if failed:
        print(f"{failed}/{len(tests)} tests FAILED")
        sys.exit(1)
    print(f"all {len(tests)} tests passed")


if __name__ == "__main__":
    _main()
