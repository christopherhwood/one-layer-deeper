# Product attention with hybrid Shampoo/AdamW

Five dense reducer matrices use Kronecker-factored Shampoo directions with
raw-gradient norm grafting; all remaining trainable parameters use the tracked
AdamW path.  The model and loss are otherwise unchanged.

The 10-second fixed smoke completed 206 updates and reached 25.17% mean exact,
33.68% token accuracy, and 42.50% last-token accuracy.  The 15-second variable
screen completed 227 updates and reached 0.96% mean exact, but only 14.58%
token accuracy, 9.92% last-token accuracy, and 15.32% OOD-N T=1 token accuracy.

This is the same misleading pattern as Lookahead: a small exact-match
fluctuation without broad error reduction.  It is below the product-attention
control on every relaxed variable-modulus metric, so it is rejected before an
H100 run.
