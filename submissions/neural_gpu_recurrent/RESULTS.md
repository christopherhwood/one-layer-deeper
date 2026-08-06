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
