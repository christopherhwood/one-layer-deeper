# H100 submission results (Easy tier) — live log

Participant: @christopherhwood. Easy = 60 accepted attempts / UTC day (resets 5pm PT).
Concurrency is **1** queued/running per account, so runs are sequential.

| # | submission | dataset | score (mean exact acc) | Max T | OOD-N Max T | steps in 60s | notes |
|---|-----------|---------|------------------------|-------|-------------|--------------|-------|
| 1 | recurrent_squaring | e1 | 7.17% | none | none | 195 | pooled-latent recurrent; undertrained (slow) |
| 2 | recurrent_squaring (vectorized T-parse) | e1 | 6.17% | none | none | 153 | throughput fix did NOT help → bottleneck is elsewhere (needs GPU profiling) |
| 3 | baseline_adamw | e1 | (pending) | — | — | — | reference anchor |

## Learnings so far

- **`recurrent_squaring` is a weak bounded-N memorizer** on real H100: ~6–7%,
  certifies **no** depth (can't even hit 100% at T=1). Confirms the pooled-latent
  recurrent is a **dead end for OOD-N** (as flagged) — useful only as a bounded-N
  data point.
- **It is severely undertrained** — only ~150–195 optimizer steps in the 60s
  budget. Vectorizing the T-parsing (removing per-row Python loops) did *not* fix
  it, so the throughput bottleneck is deeper (likely tiny-kernel launch overhead
  / per-step device syncs). Any future submission must be engineered for
  **throughput** (few large kernels, no per-step `.item()` syncs, fixed loop
  counts) to actually train enough steps.

## Plan → charge toward OOD-N (skip dead ends)

Next builds target the **length-general** substrate (the only thing that can
scale to OOD-N), tested on the OOD-N Easy datasets:
- **E5** (variable N 10–11 bit, T=1/2/3; OOD-N eval 12–13 bit) — has T-rungs.
- **E3** (sampled N 10–11 bit, fixed T=2; OOD-N eval 12–13 bit).

Candidate: a **fast, length-general** model — abacus/relative digit-significance
embeddings + a looped (weight-tied) core — engineered for throughput. Honest
odds: length-general modular multiplication is at the ML frontier; this is the
direct H100 test of whether scale + the right inductive bias cracks it. Each
candidate derisked on CPU (correctness + learning + budget) before spending an
attempt.
