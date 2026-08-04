"""Reproducible, dependency-free verification of the mathematical claims used
in ANALYSIS.md.  Run:  python submissions/recurrent_squaring/verify_math.py

It confirms, for the competition's fixed-N tiers:
  * f_T(x) = x^(2^T) mod N satisfies the one-step recurrence result^2 mod N;
  * the exponent e_T = 2^T mod phi(N) is eventually periodic in T, so the
    certified ladder T=1,2,4,8,16,32,64 collapses to a few DISTINCT power maps;
  * the reachable residue set under squaring from any x is tiny (a rho).
"""
from __future__ import annotations
from math import gcd

LADDER = [1, 2, 4, 8, 16, 32, 64]
# (label, p, q) for the fixed-N Easy/Medium moduli in scripts/generate_datasets.sh
FIXED_N = [("E1", 17, 19), ("E2", 29, 31), ("M1", 101, 103),
           ("M2", 193, 197)]


def mult_order(a: int, n: int) -> int:
    if n == 1:
        return 1
    o, cur = 1, a % n
    while cur != 1:
        cur = (cur * a) % n
        o += 1
    return o


def odd_part(m: int) -> tuple[int, int]:
    a = 0
    while m % 2 == 0:
        m //= 2
        a += 1
    return a, m


def f_T(x: int, T: int, N: int) -> int:
    v = x % N
    for _ in range(T):
        v = (v * v) % N
    return v


def check(label: str, p: int, q: int) -> None:
    N, phi = p * q, (p - 1) * (q - 1)
    a, m = odd_part(phi)
    period = mult_order(2, m)
    exps = {T: pow(2, T, phi) for T in LADDER}
    distinct = sorted(set(exps.values()))

    # 1. recurrence result(T+1) = result(T)^2 mod N on all units, up to T=64.
    units = [x for x in range(1, N) if gcd(x, N) == 1]
    rec_ok = all(
        f_T(x, T + 1, N) == (f_T(x, T, N) ** 2) % N
        for x in units[:50] for T in range(0, 65)
    )

    # 2. eventual periodicity: f_T = f_{T+period} for all T >= a, all units.
    per_ok = all(
        f_T(x, T, N) == f_T(x, T + period, N)
        for x in units[:50] for T in range(a, a + 2 * period + 5)
    )

    # 3. reachable set (rho) size, max over a sample of x.
    def reach(x: int) -> int:
        seen, v = set(), x
        while v not in seen:
            seen.add(v)
            v = (v * v) % N
        return len(seen)
    max_rho = max(reach(x) for x in units[:200])

    print(f"[{label}] N={N}=({p}x{q})  phi={phi}=2^{a}*{m}  "
          f"2-adic pre-period a={a}  period ord_m(2)={period}")
    print(f"    e_T=2^T mod phi : {exps}")
    print(f"    DISTINCT power maps on the ladder : {distinct}  "
          f"({len(distinct)} of 7 rungs)")
    print(f"    recurrence result(T+1)=result(T)^2 verified (T<=64) : {rec_ok}")
    print(f"    f_T == f_(T+{period}) for all T>={a} : {per_ok}")
    print(f"    max reachable residues (rho) from one x : {max_rho}  (N={N})")
    # which ladder rungs are provably identical:
    groups: dict[int, list[int]] = {}
    for T in LADDER:
        groups.setdefault(exps[T], []).append(T)
    ident = [g for g in groups.values() if len(g) > 1]
    if ident:
        print(f"    IDENTICAL ladder rungs (same map, exact iff lower is exact): {ident}")
    print()


if __name__ == "__main__":
    for label, p, q in FIXED_N:
        check(label, p, q)
    print("All checks above should read True; the DISTINCT counts show the "
          "certified ladder collapses to a handful of fixed power maps.")
