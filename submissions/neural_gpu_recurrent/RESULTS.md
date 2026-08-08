# Gated cellular recurrent architecture

This family replaces tied attention/product reducers with a general
Neural-GPU-style state machine: one persistent latent cell per digit, a shared
neighbor/global gated update, recurrent compute proportional to register
length, and the same learned transition reused across outer `T` steps.

| variant | fixed exact | variable exact | variable token | last token | OOD-N T=1 token |
|---|---:|---:|---:|---:|---:|
| width 128, Conv1d cell, AdamW (83 variable updates) | 10.33% | 0.79% | **16.36%** | **13.50%** | not completed |
| width 96, fast neighbor-linear cell, AdamW | 38.50% | 0.42% | 15.32% | 10.38% | 16.44% |
| width 128, fast neighbor-linear cell, AdamW | **42.83%** | 0.33% | 15.39% | 10.25% | **16.91%** |
| width 128, fast neighbor-linear cell, LAMB | 30.83% | 0.25% | 15.97% | 11.96% | 16.07% |

The architecture is the strongest general-purpose fixed-modulus learner tried
so far and the slow Conv1d version produced the best broad relaxed signal.  The
gain does not yet become exact variable-modulus learning.  Faster training
drives evaluation cross entropy below uniform while exact accuracy remains at
chance scale, and neither AdamW nor LAMB improves the OOD-N T=1 digit signal
beyond the product-attention control's roughly 17.2%.

## Hosted H100 audit (2026-08-08)

The original source silently capped training at three outer applications.  That
is valid for Easy E1 but disconnects the true T=4/8/16 endpoints on Medium.
The retained source now executes the maximum evaluator-provided T during
training, disables CUDA autocast for this small recurrent kernel, and uses a
correct prompt-capacity formula for even sequence lengths.

| source | tier / data | updates | test | OOD | mean | certification |
|---|---|---:|---:|---:|---:|---|
| width 128, bfloat16, LR 0.002 | Easy E1 | 577 | 41.33% | 66.00% | 53.67% | none |
| width 128, float32, LR 0.002 | Easy E1 | 928 | 52.67% | 75.00% | 63.83% | none |
| width 128, float32, LR 0.001 | Easy E1 | 1,245 | 60.67% | 82.00% | **71.33%** | none |
| width 64, float32, LR 0.001 | Easy E1 | — | — | — | 56.17% | none |
| residual observable-logit state | Easy E1 | — | — | — | 24.50% | none |
| width 128, full T=4/8/16 training | Medium M1 | 4,725 | 0.00% | 0.07% | **0.03%** | none |

Submission IDs, in table order, are `b26e832f-cbf5-4b6a-a4b3-6726cdf21162`,
`9fc39da8-55df-4292-aa09-24bf50c7aece`,
`dde7c639-2ac6-49c4-a1b8-78cfe7ac53be`,
`ac26579d-7a23-4d5a-bf85-6fe531d8f264`,
`78be84df-f845-4625-894b-425c6c1700d6`, and
`8b75ddd2-311f-48e4-b302-c3cae7cd7889`.

The Medium result is not a throughput failure: 4,725 H100 updates leave the
training loss near 3.43 and exact accuracy at zero.  A gradient audit at random
initialization shows healthy total endpoint gradients at T=4/8/16, but direct
last-application gradients dominate.  Blocking later parameter uses exposes
the actual long-path derivative: its total norm falls from about `6e-2` at T=4
to `4e-4` at T=8 and `7e-9` at T=16.  A forward-neutral identity surrogate
restores magnitude but makes training diverge; a real residual digit-logit state
fits the fixed smoke yet does not improve variable-modulus or Medium-depth
learning.  The retained model is therefore a strong rule-clean Easy baseline,
not a credible Hard submission.

## Dense endpoint-bit supervision (2026-08-08)

The retained rule-clean model now adds a learned binary endpoint head to the
recurrent hidden state. The only target remains the evaluator's final answer;
the loss merely represents that same endpoint in binary as well as decimal.
No carry, product, quotient, reduction, or other intermediate arithmetic state
is supplied, and the forward transition remains the same generic tied cell.

This is a genuine fixed-modulus improvement:

| gate | setting | score | test | OOD | updates |
|---|---|---:|---:|---:|---:|
| CPU fixed 10 s | bit weight 0.5 | 49.17% | 43.33% | 55.00% | 446 |
| CPU fixed 10 s | bit weight 1.0 | **50.17%** | 43.33% | 57.00% | 416 |
| CPU fixed 10 s | bit weight 2.0 | 46.50% | 40.00% | 53.00% | 436 |
| H100 Easy E1 | bit weight 0.5 | **71.67%** | 57.33% | 86.00% | 963 |

The H100 run is submission `3351ddfc-9ae2-4d73-b010-ff9b932f935d`.
It narrowly exceeds the previous rule-clean 71.33% record, while improving
the OOD split from 82% to 86%.

The variable-modulus result remains negative. With bit weight 0.5, the full
60-second E5-shaped CPU gate scored 0.29% exact and 15.10% token accuracy;
no seen- or unseen-modulus rung certified. Dense endpoint representation helps
optimization once the modulus is fixed, but it still does not identify a
modulus-general arithmetic program.

## Ordinal endpoint supervision

The final retained source also asks the recurrent hidden state to predict 32
thresholds for the endpoint's normalized position `answer / N`. This is the
same final answer expressed as a smooth ordinal target; it does not reveal the
quotient used by modular reduction or any intermediate state.

With bit weight 1.0 and ordinal weight 0.5:

| gate | score | test | OOD | additional evidence |
|---|---:|---:|---:|---|
| CPU fixed 10 s | 52.00% | 45.00% | 59.00% | first-token signal 69.83% |
| H100 Easy E1 | **73.50%** | 60.00% | 87.00% | 1,157 updates; no rung |
| CPU variable E5 60 s | 0.79% | 0.75% | 0.83% | 16.36% token; no rung |
| CPU hidden affine 10 s | 43.00% | 56.00% | 30.00% | unchanged source |
| CPU hidden cube 10 s | 31.83% | 26.67% | 37.00% | unchanged source |

The record H100 run is submission
`39a425de-9304-4eac-b7a5-ab80f6aecf18`. The same source learns materially on
changed recurrences, which is evidence that the architecture is genuinely
general rather than a disguised squaring interpreter.

A variable-modulus T=1-only isolation completed 3,978 CPU updates, reached
100% training-batch exact, but scored only 1.75% held out (1.0% test, 2.5%
OOD). This rules out recurrent depth as the main variable-N blocker: even one
application is underidentified outside the training pairs. More depth
curriculum cannot turn this architecture into a competitive Medium/Hard model
without a new modulus-general program representation.
