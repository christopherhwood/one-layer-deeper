# Core-only modulus gradient consensus

This ablation keeps fixed-hash disjoint modulus families but applies symmetric
gradient projection only to the tied attention/MLP reducer core. Input,
product, output, and CRF parameters receive the ordinary mean gradient. It
tests whether full projection was suppressing useful family-specific interface
learning.

| gate | updates | exact | token | last token | OOD-N T=1 token |
|---|---:|---:|---:|---:|---:|
| fixed-N 10s | 103 | 28.17% | 38.24% | 47.33% | — |
| variable-N 15s | 111 | 0.5000% | 15.04% | 11.00% | 16.91% |

The full-projection parent reached 0.79% exact, 15.24% token, and 17.82% OOD-N
T=1 token accuracy on the same short variable gate. Restricting consensus to
the recurrent core loses every transfer metric. Family conflict is not merely
an interface-gradient artifact; reject before longer gates.
