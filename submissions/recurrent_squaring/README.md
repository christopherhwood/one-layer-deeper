# recurrent_squaring

A submission for *One Layer Deeper* built from the task's own algebra: the target
`x^(2^T) mod N` is a **single** map — squaring mod `N` — iterated `T` times, so
the certified‑depth ladder is one operator composed with itself.

**Read [`ANALYSIS.md`](ANALYSIS.md) for the full derivation and proofs.** The
short version:

1. **Extrapolation in `T` is free** — learning is only about the one‑step
   squaring cell; a model trained to `T=3` certifies to `T=64` (Claim 1).
2. **The ladder collapses** — `2^T mod φ(N)` is eventually periodic, so
   `T=1,2,4,8,16,32,64` reduce to a handful of distinct power maps; some rungs
   are literally identical (Claim 3).
3. **Certification = exact coverage** — the certified depth is a step function of
   how completely the one‑step operator is learned; it jumps to the top of the
   ladder exactly at 100% coverage (§4, measured in `demo_3`).

The model is a bidirectional encoder → one **weight‑tied** squaring cell iterated
`T` times (adaptive depth, rule 4) → place‑value decoder. It is fully learned end
to end (rules 6–8) and runs through the real evaluator with both depth profiles.

### Files
| file | what it is |
|------|-----------|
| `submission.py` | the competition submission (`SUBMISSION`) |
| `ANALYSIS.md` | mathematical analysis, proofs, and empirical evidence |
| `verify_math.py` | dependency‑free check of the recurrence, periodicity, ladder collapse |
| `demo_1_memorization_fails.py` | a free per‑`x` embedding memorizes and fails on fresh `x` |
| `demo_2_residue_faithful.py` | a residue‑faithful tied cell learns `s` and shows the `T`‑plateau |
| `demo_3_coverage_law.py` | certified depth vs. operator coverage → jumps to `T=64` at 100% |

### Reproduce
```bash
python submissions/recurrent_squaring/verify_math.py          # no dependencies
# with torch (CPU is fine):
python submissions/recurrent_squaring/demo_1_memorization_fails.py
python submissions/recurrent_squaring/demo_2_residue_faithful.py
python submissions/recurrent_squaring/demo_3_coverage_law.py
# end-to-end through the evaluator on a generated Easy dataset:
python -m benchmark.runner --manifest <cpu_or_h100_manifest> \
  --submission-file submissions/recurrent_squaring/submission.py
```
