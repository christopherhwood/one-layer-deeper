# H100 submission results (Easy tier) — live log

Participant: @christopherhwood. Easy = 60 accepted attempts / UTC day (resets 5pm PT).
Concurrency is **1** queued/running per account, so runs are sequential.

| # | submission | dataset | score (mean exact acc) | Max T | OOD-N Max T | steps in 60s | notes |
|---|-----------|---------|------------------------|-------|-------------|--------------|-------|
| 1 | recurrent_squaring | e1 | 7.17% | none | none | 195 | pooled-latent recurrent; undertrained (slow) |
| 2 | recurrent_squaring (vectorized T-parse) | e1 | 6.17% | none | none | 153 | throughput fix did NOT help → bottleneck is elsewhere (needs GPU profiling) |
| 3 | baseline_adamw | e1 | 1.67% | none | none | 93 | reference anchor (recurrent ≈4× baseline) |
| 4 | recurrent_squaring (batch_size=32) | e1 | 2.33% | none | none | **1459** | 10× more steps (DataLoader fix) but score did NOT improve |

## The decisive engineering + scientific finding

- **DataLoader throughput bug (confirmed & fixed):** the H100 manifests use
  `num_workers=2` with a tiny train set + large batch → ~1 batch/epoch → the
  worker iterator is respawned almost every step (~0.6s/step overhead). Dropping
  `batch_size` 512→32 gave **1459 vs 153 steps** (~10×). This helps *any* future
  submission — but only if the model can actually use the steps.
- **Throughput was NOT the score-limiter — GENERALIZATION is.** With 10× the
  steps the score did not rise (test ≈0.05, ood 0); the extra steps just memorize
  the tiny train set faster (train accuracy bounces, not the held-out `test`).
  This confirms on real H100 that the pooled-latent recurrent **does not
  generalize to fresh x** — a dead end for both Easy score and OOD-N.

**Bottom line:** the gating problem is unchanged and now H100-confirmed —
**learning modular squaring as a function that generalizes to fresh x** (bounded-N
first, then length-general for OOD-N). Faster training doesn't create
generalization; the architecture must. Per the "skip dead ends" directive, the
per-residue *coverage* model (which would score high on Easy via memorized-orbit
coverage) is intentionally **not** pursued — it can't transfer to OOD-N. The next
build is the length-general (abacus + looped, throughput-tuned `batch_size`)
candidate on E3/E5, as the direct H100 test of whether scale + inductive bias
cracks the generalization frontier.

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

---

## Frontier attempt: abacus + looped core on E3 (OOD-N)

| # | submission | dataset | score | Max T | OOD-N | steps/60s | notes |
|---|-----------|---------|-------|-------|-------|-----------|-------|
| 5 | oodn_abacus | e3 | 1.00% | none | none | 1927 | did NOT fit training (loss stuck ~2.1, train acc ~0) |

**Result (honest):** the abacus (place-value) + looped-attention model — the
best-known inductive bias for length-general arithmetic — does **not** learn
`x⁴ mod N` on E3 even in-distribution at H100 scale (60s, ~1927 steps). The loss
is **stuck**, not slowly descending, so this is an **architecture wall, not a
step-count wall** — more steps (e.g., Medium's 600s) are unlikely to rescue it.
This confirms on real H100 what ~25 CPU experiments indicated: **learning
modular multiplication/reduction as a generalizing function is the open frontier**,
and neither throughput nor the SOTA length-gen inductive bias breaches it here.

An accessible walkthrough of the task + this model: `walkthrough.html`.

---

## Cross-agent runs: length-general digit-register models on E5

A second agent independently built the exact length-general family flagged as the
frontier bet — on-device right-aligned **digit registers**, a **shared
bidirectional scan** transition (length-independent), **iterative refinement**
with temperature annealing (1.25→0.18), a **T-recurrence** trained at 4 outer
steps but evaluated to 64 (betting on extrapolation-in-T), and a custom loss that
**upweights T=1** (emphasis on the single-squaring rung). Two variants:
`digit_transducer_h100` (GRU scan) and `recurrent_register_h100` (attention).

| submission | dataset | score | Max T | OOD-N | notes |
|---|---|---|---|---|---|
| digit-register (var. A) | e5 | **0.42%** | none | none | no certification |
| digit-register (var. B) | e5 | **0.50%** | none | none | no certification |
| digit-register (3rd run) | e5 | **failed** | — | — | likely OOM / timeout (larger attn model) |

**Update to thinking (confirms, does not overturn):** two *independent, careful*
implementations of the length-general refinement family score ~0.4–0.5% on E5
(variable N) with **no certification** — right where my abacus+looped attempt
landed (1% on E3). On variable-N there is nothing to memorize, so ~0.5% is the
honest measure of *inability to compute* modular squaring as a transferable
function. Better engineering of this architecture family (registers, scan,
refinement, T-extrapolation, annealing, rung upweighting) does **not** breach the
wall. The remaining differentiated bets are the ones that change the
**supervision** (TRM/HRM deep supervision on the observed rungs + 1-step detached
gradients) or the **operation itself** (a structured, learned Montgomery-style
reducer that removes the global magnitude comparison) — neither of which these
runs used.
