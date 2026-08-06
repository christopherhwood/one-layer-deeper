# Untied Transformer transition

This candidate replaced the shared fixed-point block with four distinct
Transformer blocks and allowed full gradient flow through the three training
applications of the learned transition.  It remained fully generic and used
only endpoint/deep endpoint cross entropy.

The 10-second fixed-modulus smoke completed 214 updates and reached 0% test,
10% OOD, 5% split-mean exact, and 15.73% token accuracy.  This is far below the
gated cellular model's 42.83% exact and 53.73% token accuracy.  Specialized
depth does not compensate for the loss of a persistent local state, so the
candidate was rejected before the variable-modulus gate.
