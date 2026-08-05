# Easy H100 submission plan (derisked on CPU)

## Derisk result (CPU, E1, 120s budget, identical settings)

| submission | Easy score (mean exact acc) | seen-N depth diagnostic |
|---|---|---|
| baseline_adamw | 0.010 | ~0–5% across the ladder |
| **recurrent_squaring** | **0.055 (5.5×)** | T=8: 24%, others 8–13% |

Both ran end-to-end through the official runner (no errors, both depth profiles
computed, under the 5×10⁸-element and time budgets). `recurrent_squaring` also
passes `one-layer validate`. CPU is heavily undertrained (a real 60s H100 run is
far more compute) — the point of the derisk is **correctness + it learns + it
beats baseline**, which holds.

## Why Easy runs are worth it (learning, not just score)

Easy is *scored* by mean exact accuracy on `test`+`ood`, but the leaderboard also
reports the **Max T / OOD-N Max T diagnostic** — the exact signal that predicts
the **Hard** approach. So each Easy run teaches us whether the recurrent-squaring
thesis (learn one squaring, iterate, extrapolate in T) actually holds under real
H100 training. That is the highest-value thing to measure.

## Plan (respecting 60 accepted Easy attempts / UTC day; don't waste)

1. **e1, `recurrent_squaring`** — smallest/fastest fixed-N. Read: Easy score +
   the seen-N Max-T ladder. Key question: does accuracy *plateau* across
   T=4,8,16,32,64 (extrapolation working) or decay (cell imperfect)?
2. **e1, `baseline_adamw`** — one reference run to anchor the leaderboard delta.
3. Iterate from the e1 Max-T shape (tune depth/width/steps), then repeat on
   e2 (larger fixed N) and e3–e5 (sampled/variable N, which probe OOD-N).

Each iteration derisked on CPU first (correctness + learning + budget) before
spending an attempt.

## Submit commands (need an API key — see below)

```bash
export ONE_LAYER_API_KEY=old_...            # from `one-layer login` on your machine
one-layer validate submissions/recurrent_squaring/submission.py
one-layer submit  submissions/recurrent_squaring/submission.py --tier easy --dataset e1 --wait
one-layer metrics <submission-id> --output metrics.jsonl
one-layer leaderboard
```
(Equivalently `python -m client.cli ...` from the repo.)

## Auth status

The service (`https://onelayerdeeper.ai`) is reachable from here (leaderboard =
HTTP 200), and the CLI accepts a key via `ONE_LAYER_API_KEY` / `--api-key` — so I
can submit **headlessly** given a key. What I cannot do headlessly is
`one-layer login` (interactive GitHub OAuth via browser + localhost callback).
**Need:** run `one-layer login` once on your machine and share the `old_…` key
(or paste it and I'll set it as an env var here).
