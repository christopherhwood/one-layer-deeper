# General architecture and optimizer frontier wave

All local variable screens use the same 15-second E5 manifest, seed 74, and
relaxed evaluator.  Exact remains the official score; token metrics diagnose
whether an architecture is learning the operation broadly or merely hitting a
few answers by chance.

| candidate | core change | variable exact | token | last token | OOD-N T=1 token | decision |
|---|---|---:|---:|---:|---:|---|
| product attention, batch 128 | prior H100 control | 0.79% | 14.78% | 10.33% | 17.23% | control |
| Neural GPU, Conv1d | tied cellular recurrence | 0.79% | **16.36%** | **13.50%** | unavailable | H100 tested |
| Neural GPU, fast width 96 | neighbor-linear cell | 0.42% | 15.32% | 10.38% | 16.44% | reject |
| Neural GPU, fast width 128 | wider neighbor-linear cell | 0.33% | 15.39% | 10.25% | 16.91% | reject |
| Neural GPU + LAMB | layerwise trust-ratio optimizer | 0.25% | 15.97% | 11.96% | 16.07% | reject |
| modulus hypernetwork | FiLM-conditioned recurrence | 0.79% | 15.75% | 10.67% | 16.65% | 60s checked |
| relational grid | pairwise 2D processor | 0.58% | 15.70% | 11.63% | 16.97% | reject |
| persistent latent register | no digit bottleneck across T | 0.21% | 14.66% | 10.42% | **17.44%** | reject |
| scaled Transformer | 3.95M generic transition | 0.38% | 15.93% | 11.50% | 15.96% | H100 tested |
| product selective SSM | bidirectional state-space reducer | 1.13% | 14.64% | 10.92% | 16.22% | false-positive exact |
| product attention + Lookahead | slow/fast optimizer geometry | 0.96% | 14.36% | 10.13% | 15.96% | reject |
| product attention + Shampoo | Kronecker curvature on reducer matrices | 0.96% | 14.58% | 9.92% | 15.32% | reject |
| product-column serial reducer | streaming Horner-like digit tape | 0.08% | 14.57% | 9.29% | 17.13% | reject |
| product attention + Grokfast 5x | slow-gradient amplification | 1.21% | 14.26% | 10.63% | 16.54% | 60s rejected |
| product attention + cross-example consensus | sign agreement across row halves | 0.58% | 14.79% | 10.79% | 17.50% | 60s rejected |
| discrete-action serial reducer | balanced categorical global control | 0.54% | 14.99% | 10.25% | unavailable | 60s ordinary-token gain only |
| local-action ripple reducer | position-tied categorical actions | 0.42% | 14.96% | 9.58% | unavailable | 60s rejected |
| observable ripple transducer | digit beliefs are the only persistent state | 0.50% | 15.45% | 10.96% | 17.02% | 60s rejected |
| discrete serial + digit CRF | sequence-normalized serial endpoint | 0.75% | 14.44% | 10.38% | unavailable | 60s rejected |
| anchored error-correcting ripple | discrete state corruption + tied repair pass | failed fixed smoke | — | — | — | reject |
| position-only product mixer | content-independent relative routing | 0.29% | 15.06% | 10.75% | 16.07% | 60s OOD signal; H100 explore |
| position-routed content gate | fixed routing + multiplicative local update | 0.38% | 14.99% | 9.42% | 16.07% | 60s rejected |
| reversible product reducer | additive coupling inner recurrence | 0.42% | 14.98% | 9.79% | 17.18% | sample-efficient smoke; 60s rejected |
| position-only + paired state recovery | fitted scratch perturbation + clean trajectory targets | 0.46% | 15.10% | 9.00% | 16.38% | better exact/efficiency; 60s OOD rejected |
| position-only + fixed local feedback | fixed random phase classifier + learned phase head | **0.79%** | 15.14% | 10.54% | 16.86% | 60s token gain; OOD below clean best |
| position-only + decoupled feedback | stop-gradient feature/head local classifier | 0.25% | **15.67%** | 10.63% | 16.76% | best broad learner; 60s OOD rejected |
| directed neighborhood feedback | fixed current/next-significant classifiers | 0.08% | 15.35% | 10.50% | 16.28% | reject |
| position-only predictor-corrector | detached residual logits + connected endpoint path | failed fixed smoke | — | — | — | reject after five ablations |
| position-only sequence feedback | fixed whole-answer random code | 0.33% | 14.78% | 9.71% | 16.38% | fixed-only gain; reject |
| product ripple + decoupled feedback | stop-gradient learned local classifier | 0.50% | **15.74%** | **11.58%** | 16.65% | broad short gain; 60s exact rejected |
| product ripple + modulus consensus | symmetric gradients across disjoint modulus families | **0.79%** | 15.24% | 11.38% | **17.82%** | short OOD gain; 60s not durable |
| canonical categorical transition | observable digit state + finite categorical controller | 0.38% | **15.97%** | 11.17% | 17.07% | 60s signal decays; reject |
| grounded global-program register machine | one learned program over canonical tapes | 1.21% | 14.35% | 10.17% | 15.75% | false-positive exact; reject |
| Fourier numeric operator | learned periodic scalar basis | failed fixed smoke | — | — | — | reject |
| untied Transformer | specialized depth/full gradients | failed fixed smoke | — | — | — | reject |

## Hosted evidence

| H100 candidate | updates | final train exact | test | OOD | score |
|---|---:|---:|---:|---:|---:|
| product attention, batch 32 | 1,346 | 6.2% | 1.1% | 0.5% | 0.79% |
| product attention, batch 128 | 1,507 | 25.0% | 1.3% | 1.0% | **1.13%** |
| product attention, stop at 1,000 | 1,000 | 9.4% | 0.7% | 1.0% | 0.88% |
| convolutional Neural GPU | 1,502 | 6.2% | 1.1% | 0.7% | 0.88% |
| scaled general Transformer | 1,259 | 18.0% | 1.0% | 0.0% | 0.50% |
| position-only product mixer (Hard) | 139,579 | 0.0% (1.6% max) | 0.0% | 0.0% | 0.03% |
| grounded program, vectorized controller | 336 | 0.8% | 0.7% | 1.3% | 1.00% |
| grounded program, fast batch 128 | 300 | 0.0% | 0.6% | 1.0% | 0.79% |
| grounded program, fast batch 512 | 244 | 0.4% | 0.7% | 0.8% | 0.75% |

No candidate certifies seen or OOD-N T=1.  The H100 runs separate three failure
modes: general cellular recurrence cannot fit the variable task despite healthy
throughput, while the larger Transformer and product model fit the finite
training corpus but do not generalize. Position-only routing also cannot fit
hidden Hard after 139,579 updates, decisively retiring it as a compute-scaling
candidate. More parameters and more repeated data
presentations are therefore not substitutes for more unique data, which the
submission rules do not permit participants to create.

## Remaining differentiated paths

1. A parameter-efficient, digit-serial learned reducer whose topology makes the
   latent operation identifiable from endpoints, without hard-coding arithmetic.
2. A genuinely local credit-assignment method (target propagation or an
   evaluator-owned local objective), rather than another first-order optimizer
   applied to the same endpoint gradient.
3. A rules-audited way to exploit the observed `T=1/2/3` square boundaries as
   transition supervision without inspecting, pairing, or augmenting examples.
4. Structured attainable-state recovery that preserves correlations across
   position and channel; independent fitted noise improves local learning but
   pulls OOD-N states back toward the training distribution.
5. A learned target-propagation coordinate tied to transition inversion rather
   than a fixed output code. Fixed local feedback improves optimization, but
   its endpoint geometry still does not identify the reusable reduction step.
6. A factorized library of generic, causally grounded tape operations. The
   global-program experiment shows that named read/write choices are
   insufficient when a flexible local cell can ignore or repurpose them;
   instruction ablations and cross-seed program agreement must become gates.

Further width, dropout, Adam-family, attention, cellular, or state-space sweeps
would mostly repeat the failure modes above.

## Endpoint-identification theorem diagnostic

The canonical categorical wave includes a bounded constructive control. A
randomly initialized transition table trained only on exhaustive T=1 endpoint
pairs for `2 <= N <= 31` reaches 100% one-step accuracy and, without any deeper
targets, 100% at every unseen T rung through 64. With only 60% of the one-step
states covered, withheld T=1 accuracy is 6.57% and deep rollout accuracy falls
to roughly 34%.

This separates possibility from identification. Endpoint-only supervision is
sufficient when the one-step examples distinguish a reusable canonical
transition. The benchmark candidate supplies canonical state and weight tying,
but its flexible local rule is not distinguished by the available
variable-modulus endpoints. A next candidate must shrink that effective rule
class or create a rules-valid generic teaching signal; more optimization of the
same ambiguous transition cannot satisfy the positive theorem's premise.

The grounded global-program follow-up also fails that premise. Its endpoint
gradient reaches the shared program and its tape states are canonical, but the
six-phase model scores only 15.75% token accuracy on unseen-modulus T=1. A
four-phase ablation fits training faster while generalizing worse. Global
program sharing therefore does not identify the program when the selected
operations are implemented by a still-flexible monolithic local transition.
On H100, vectorizing all controller alternatives reaches 1.00%; simplifying the
cell and raising batch size from 128 to 512 increases processed examples from
38,400 to 124,928 but scores only 0.75%. The failure therefore persists after a
3.25x data-throughput gain and is not explained by accelerator starvation.
