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
| latent-path CRF decoder, 8 states, oracle product | 63.38% | 0.50% | 1.37% | marginalization improves fitting, not identifiability |
| latent-path CRF decoder, 16 states, oracle product | 72.00% | 0.50% | 1.17% | more latent capacity becomes a stronger memorization channel |
| latent quotient + learned `qN` convolution | 0.44% | 0.25% | 0.00% | quotient prior collapses to an arbitrary code |
| latent quotient + exact candidate `qN`, GRU subtractor | 0.31% | 0.00% | 0.20% | removing multiplication does not fix joint latent learning |
| latent quotient + exact `qN`, two-state local FST | 0.44% | 0.25% | 0.39% | grounding borrow topology prevents lookup but not semantic ambiguity |

Unless a later section says otherwise, the earlier full-subset probes used
1,500 AdamW updates, batch size 32, seed 74, and the same 1,600 E5 T=1 training
prompts. The oracle probes are diagnostics only; they are not rules-valid
submissions and do not supply process labels to a submitted model.

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

## Latent-path marginalization result

`latent_crf_probe.py` replaces a single recurrent reduction trajectory with a
conditional finite-state decoder. Forward/backward dynamic programming sums all
latent state paths compatible with the observed remainder; no carry, quotient,
or intermediate remainder is labeled. This materially improves optimization:
the 8- and 16-state models fit 63% and 72% of T=1 training examples, compared
with less than 1% for the sequential quotient/correction reducer.

It does **not** improve generalization. Both state sizes remain at 0.5% on
ordinary held-out T=1, and increasing state count only improves training fit.
Unconstrained latent states therefore form a hidden lookup table rather than
discovering arithmetic roles. Marginalization is useful only if the factor
graph grounds its states in local carry/borrow/quotient relationships. Encoding
those relationships strongly enough without crossing the benchmark's
hard-coded-algorithm boundary is the remaining design problem.

## Grounded quotient marginalization result

`latent_quotient_probe.py` enumerates all 2,048 possible quotient strings. A
small proposer supplies `p(q | x^2, N)`, while a tied LSD-first decoder explains
the observed remainder from product digits and candidate `qN` features. Neither
quotients nor borrow states are labels. With a learned schoolbook `qN`
convolution, the quotient-prior entropy collapses from about 7.6 nats to 0.21,
but held-out quotient accuracy is 0% and held-out remainder accuracy is 0.25%.
Supplying exact candidate `qN` digits changes neither outcome. The proposer and
decoder establish a confident arbitrary code before multiplication/subtraction
semantics emerge.

`latent_quotient_fst_probe.py` removes most of that freedom. Its decoder has two
latent states and only a position-tied local transition
`(P digit, qN digit, state) -> (R digit, next state)`. It marginalizes the state
path and quotient, holds the quotient prior uniform during decoder warm-up, and
then uses a detached decoder posterior to teach the proposer. Unlike the GRU,
the posterior stays high-entropy instead of collapsing, but it still selects
the true quotient 0% of the time and gives chance-level endpoint accuracy.

An explicitly non-viable oracle-quotient control establishes the exact
boundary. With the same FST and training rows, exposing the quotient only to
the loss yields:

| oracle control metric | train T=1 | held-out T=1 | adversarial T=1 |
|---|---:|---:|---:|
| target-conditioned posterior selects true quotient | **99.81%** | **99.50%** | **99.41%** |
| remainder given the true quotient | 70.88% | **73.50%** | **74.61%** |
| proposer selects true quotient from `(P,N)` | 4.44% | 2.50% | 4.30% |

Thus the local arithmetic transducer learns and generalizes once its latent
semantics are fixed, but two problems remain: endpoint marginal likelihood does
not identify those semantics, and a generic global quotient proposer does not
learn division. The most credible next route is to eliminate the proposer and
reuse a local subtract/compare cell as a recurrent or monotone quotient search,
while finding a rules-valid shallow or consistency signal that fixes the cell's
semantics. These probes fail the local learning gate, so they do not justify a
new H100 submission.
