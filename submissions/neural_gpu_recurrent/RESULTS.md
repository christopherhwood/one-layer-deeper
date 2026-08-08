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
