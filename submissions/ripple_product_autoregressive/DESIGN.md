# Exact autoregressive endpoint decoder

This experiment applies a general sequence-model recipe to the product-aware
learned reducer. A tied GRU decoder conditions every output digit on its entire
preceding output prefix. Since E5 has width four, training can score the complete
prefix tree in one ordinary label-independent forward. The endpoint loss then
selects supplied answer digits and marginalizes unobserved leading positions.

The decoder never receives labels in forward, and evaluation uses the same
learned transition greedily. There are no generated labels, arithmetic masks,
process states, nested calls, or participant-owned backward passes. A 32-wide
decoder preserves exact full-prefix likelihood while reducing tree cost.
