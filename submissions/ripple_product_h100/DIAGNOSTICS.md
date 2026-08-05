# Product-aware ripple diagnostics

These local probes isolate representation, fitting, and generalization before
spending another H100 attempt. They use E5 T=1 rows because T=1 directly labels
one complete modular square.

| probe | train T=1 | held-out T=1 | adversarial T=1 | conclusion |
|---|---:|---:|---:|---|
| fixed 32-example batch | **100% by step 100** | — | — | implementation and capacity can fit examples |
| full T=1 subset | 85.31% | 1.25% | 0.59% | T=1-only training mainly memorizes |
| frozen exact digit-product table | 83.63% | 1.00% | 0.39% | discovering `digit_a * digit_b` is not the blocker |
| exact normalized `x^2` digits fed to reducer | 82.31% | 1.50% | 1.17% | generic GRU reduction memorizes instead of learning division |
| digit-serial quotient/correction reducer, oracle product | 0.44% | 0.25% | 0.39% | eight unsupervised reducer decisions recreate the credit-assignment wall |
| same reducer, 8-dimensional state | 0.25% | 0.50% | 0.00% | a small-state bottleneck does not induce the reducer algorithm |
| 32-wide ripple, strong-decay grokking run (10,000 steps) | 9.25% | 0.25% | 0.59% | model never interpolates, so no delayed generalization emerges |

All full-subset probes used 1,500 AdamW updates, batch size 32, seed 74, and the
same 1,600 E5 T=1 training prompts. The oracle probes are diagnostics only; they
are not rules-valid submissions and do not supply process labels to a submitted
model.

The next architecture should therefore change the reducer, not merely increase
T=1 weight or H100 duration. The targeted replacement is a digit-serial learned
reducer: consume normalized product digits from most to least significant,
maintain a positional remainder, predict one quotient/correction digit, and use
a tied sequential correction scan before consuming the next digit.

The first version of that replacement could not fit even with oracle product
digits. This rules out simply making long division explicit while retaining one
distant endpoint loss. The next experiment keeps the shallower product-aware
reducer but changes optimization: a T=1→2→3 wall-clock curriculum, full
gradients through the observed short horizons, and an exactness-oriented
worst-digit/hard-row objective.

That optimization experiment was also negative. The staged/worst-row variant
completed almost 3,000 CPU updates but reached only 4.25% T=1 train and 0.25%
held-out T=1. A clean full-gradient ablation with the original loss reached
3.50% T=1 train, 0% held-out T=1, and 0.33% over the ordinary T=1/2/3 test.
These regressions were not retained in `submission.py`; the smoke-tested 0.79%
H100 candidate remains the active submission.

A grokking-oriented run then reduced the recurrent width to 32, trained T=1
alone for 10,000 updates, and used AdamW weight decay 1.0. It did not reach the
interpolation regime associated with grokking: training exact accuracy was only
9.25%, with chance-level held-out results. More post-interpolation training is
therefore not the missing lever for this model.

## Current boundary

The experiments now separate two facts:

1. With process labels, a small carry/borrow transducer learns exact modular
   addition and generalizes to much longer unseen numbers.
2. With only the benchmark endpoint, neither a generic reducer nor an explicit
   digit-serial quotient/correction reducer learns modular reduction, even when
   supplied the exact product digits.

The next credible candidate must obtain a shorter or better-conditioned signal
for reduction without generating prohibited arithmetic labels. Candidates that
merely add capacity, recurrence, optimizer steps, curriculum, or stronger
regularization have now been locally falsified.
