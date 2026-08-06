# Exact activation-gradient replay

This candidate tests whether endpoint error is useful but attenuated across the
four learned reducer refinements. It requests two evaluator-owned passes on the
same batch.

1. The global pass computes the ordinary endpoint CE, final observed-digit CRF
   likelihood, and EMA consistency. Each first-square activation retains its
   exact endpoint gradient.
2. The replay pass recomputes the first square with detach boundaries and uses
   local linear surrogates `sum_r <h_r, stopgrad(g_r)>`. The optimizer combines
   the saved global parameter gradient with a weighted replay gradient.

Both backward calls belong to the evaluator. Participant callbacks only copy
gradients and switch pass mode. There is no intermediate arithmetic target,
generated example, nested model call, or derivative-engine call.

With normalization disabled, `diagnose_replay.py` proves the chain-rule
decomposition on a generated all-T1 batch. Normalized replay rescales every
phase to the final phase's gradient RMS; that rescaling is the experimental
learning-signal change.
