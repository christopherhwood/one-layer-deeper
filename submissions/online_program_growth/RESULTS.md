# Online Program Growth

## What this submission is

The model starts with exactly four live terminal programs: `x`, `N`, `0`, and
`1`.  There is no bank of complete candidate algorithms in model state.
Optimizer steps repeatedly:

1. receive per-program endpoint fitness and observational-novelty information
   through ordinary parameter gradients;
2. retain a bounded set of promising and exploratory parents;
3. append new arithmetic instructions on device; and
4. overwrite pruned rows inside a fixed 65,536-row capacity.

The instruction vocabulary is `MOVE`, `ADD`, `SUB`, `MUL`, `DIV`, `MOD`,
`AND`, and `MIN`.  A discovered one-step program is recurrently applied the
prompt's requested number of times.  The forward pass therefore supplies a
general arithmetic substrate, not a repeated-squaring transition.

Return-preserving branch mutations are important: a program may append a side
calculation while continuing to return its previously useful value.  A later
instruction can consume that side register.  This prevents endpoint fitness
from immediately destroying every useful intermediate detour.

## Why continuation-state pruning matters

Deduplicating programs by their current return value is unsound for online
growth.  For example, two programs can both return `x*x` while only one also
retains `1+1` in another register.  They have equal current behavior but
different possible continuations.  This implementation hashes the complete
register environment on the current batch, together with program length and
return register, before marking observational duplicates.

## Matched enumeration ablation

`online_archive_synthesis.py` provides a deterministic offline ablation of the
same core principle.  Both policies start from the four terminals and receive
the same budget of 155,967 generated candidates for the five-operation target

```text
(x^3 + x^2 + x + 1) mod N
```

| Policy | Exact train | Exact OOD | Result |
|---|---:|---:|---|
| guided, continuation-diverse archive | 100% | 100% | found a 5-op exact program |
| pure enumeration | 3.95% | 10.75% | missed the required bridge |

The guided program is:

```text
MOD(MUL(ADD(x,1),ADD(1,MUL(x,x))),N)
```

The experiment also proves by exhaustive semantic closure that this target is
absent from the chosen language through four operations.

## Verified benchmark results

### Hosted H100

| Dataset | Training | Test | Seen-N max T | OOD-N max T | Submission |
|---|---:|---:|---:|---:|---|
| Easy E1, square family | 60 s | 100% | 64 | 64 | `ebcc5acc-61d2-49f0-b029-ffa4819fe168` |
| Medium M1, square family | 600 s | 100% | 64 | 64 | `8b833382-f310-48f5-872e-2b0e887d4e90` |

The hosted Medium run completed 20,014 optimizer steps, evaluated 3,000 test
and 3,000 OOD prompts exactly, used 2,424,838 model-state elements, and used
393,219 optimizer-state elements after the first step.  That hosted run used
the same online archive core immediately before return-preserving branch
mutations were added.

### Current revision, local evaluator

| Dataset/family | Budget | Test | OOD | Seen-N T ladder | OOD-N T ladder |
|---|---:|---:|---:|---:|---:|
| M5-scale variable-N repeated square | 60 s CPU | 100% (9,000) | 100% (3,000) | 100% through 64 | 100% through 64 |
| M1-scale hidden affine `(3x+1) mod N`, train T=4/8/16 | 60 s CPU | 100% (3,000) | 100% at T=32 (3,000) | 100% through 64 | 100% through 64 |
| small hidden affine, train T=4/8/16 | 60 s CPU | 100% | 100% at T=32 | not configured | not configured |
| small hidden `(x^2+x+2) mod N` | 10 s CPU | 100% | 100% | not configured | not configured |

The M1-scale affine probe uses fixed `N=101*103`, the same scale and training
horizons as public M1, and evaluates unseen 15- and 16-bit moduli.  It is a
genuinely different recurrence family learned from deep endpoints: there are
no T=1 or T=2 examples in its training split.

The current revision also passed the same M1-scale hidden affine evaluation on
a private H100: 100% on 3,000 test examples, 100% on 3,000 T=32 OOD examples,
and complete seen/OOD-N certification through T=64 after 8,059 optimizer steps
and 600.016 training seconds.  Modal call:
`fc-01KZG6ZQA5G45KJ4TTJ6G60SZH`.

The current revision was separately regressed against the private official E1
H100 manifest: 100% test/OOD and complete seen/OOD-N certification through
T=64 after 1,423 steps in 60.176 training seconds.  Modal call:
`fc-01KZG7QMNXZTDG5XPXZ96KYJZ7`.

The quadratic-plus-two result is the benchmark-native beyond-four-operation
check.  Exhaustive semantic closure over the same arithmetic vocabulary proves
that its behavior is absent through four operations; the online-growth model
nevertheless reaches exact train and OOD behavior when one-step examples are
available.

## Known failures and boundary of the result

This is not yet a universal recurrence synthesizer.

| Probe | Current result | Interpretation |
|---|---:|---|
| deep `(x^2+x+2) mod N`, train T=4/8/16 | 4.5% mean exact on CPU | the exact 5-op program has branched/flat prefixes; endpoint fitness supplies almost no prefix ordering |
| `(x^3+x^2+x+1) mod N` benchmark-native search | near chance in short CPU runs | recombination/lookahead remains insufficient even though the offline guided archive finds it |

The affine result shows that deep endpoint-only supervision is not itself an
impossibility.  The remaining failure is narrower and useful: chaotic modular
composition destroys the correlation between a partially correct one-step
program and the observed four-step endpoint.  The next architectural step is
a bounded expression-DAG or meet-in-the-middle archive that can discover and
retain subexpressions independently, then recombine them, instead of requiring
every useful prefix to survive endpoint selection as a complete recurrence.

## Reproduction

```bash
.venv/bin/python -m unittest tests.test_online_archive_synthesis

.venv/bin/python -m benchmark.runner \
  --manifest benchmark/manifests/local_cpu_medium_m5_60s.json \
  --submission-file submissions/online_program_growth/submission.py

.venv/bin/python -m benchmark.runner \
  --manifest benchmark/manifests/local_cpu_hidden_affine_medium_60s.json \
  --submission-file submissions/online_program_growth/submission.py

.venv/bin/python -m submissions.generated_program_easy.online_archive_synthesis \
  --target cubic_polynomial_mod --policy both --beam-width 192 --max-rounds 5
```
