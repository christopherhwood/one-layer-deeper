# Dyadic neural operator results

This rule-clean experiment replaced a long chain of one-step transitions with
seven horizon-conditioned neural operators representing T=1,2,4,8,16,32,64.
The operators share one local gated cell and learned horizon embeddings.
Label-free semigroup consistency enforces `F_(2t)(x) = F_t(F_t(x))`, one
rotating hierarchy edge per update. No arithmetic primitive or recurrence
formula appears in the learned transition.

The intended test was whether the Medium failure of `neural_gpu_recurrent` was
mostly caused by vanishing gradients through T=4/8/16. Each power-of-two row in
this model has a direct endpoint path through one neural operator.

## Results

| Gate | Updates | Test exact | OOD exact | Mean | Token accuracy |
|---|---:|---:|---:|---:|---:|
| fixed-N CPU, 10 s, exhaustive hierarchy | 154 | 6.67% | 11.00% | 8.83% | 32.26% |
| fixed-N CPU, 10 s, sparse rotating consistency | 444 | 25.00% | 8.00% | 16.50% | 39.34% |
| M1-shaped CPU, 60 s | 1,697 | 0.10% | 0.067% | **0.083%** | 12.38% |

The sparse execution repair cut a training forward from roughly twenty neural
transitions to four–six for M1 and substantially improved the fixed task. It did
not unlock Medium. M1 training exact remained zero, endpoint predictions stayed
near uniform, and neither depth profile certified T=1.

## Conclusion

This falsifies the narrow claim that long-path derivative decay is the sole
cause of the Medium failure. Direct endpoint access fixes conditioning but does
not make a generic continuous operator infer the shared modular arithmetic
program. The branch is rejected before H100. The remaining gap is the same
representation/identifiability problem isolated by the categorical proofs:
the model needs a compact global reusable rule class, not merely a shorter
gradient path.
