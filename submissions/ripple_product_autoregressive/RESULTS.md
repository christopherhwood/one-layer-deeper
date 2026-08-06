# Exact autoregressive endpoint results

Compilation, official source lint, model/optimizer validation, finite backward,
ordinary evaluation, and depth evaluation passed. The compressed E5 model has
379,786 persistent state elements.

| variant | updates | test | OOD | mean | seen T1 | OOD-N T1 |
|---|---:|---:|---:|---:|---:|---:|
| decoder-128, fixed-N 10s | 291 | 21.67% | 19.00% | 20.33% | — | — |
| decoder-128, variable-E5 15s | 90 | 0.6667% | 0.5000% | 0.5833% | 0.5859% | 0% |
| decoder-32, fixed-N 10s | 366 | 21.67% | 17.00% | 19.33% | — | — |
| decoder-32, variable-E5 15s | 185 | 0.5833% | 0% | 0.2917% | 0.1953% | 0% |
| decoder-32 T1-only, fixed-N 10s | 852 | 5.00% | 4.00% | 4.50% | — | — |
| decoder-32 T1-only, variable-E5 15s | 494 | 0.5000% | 0.1667% | 0.3333% | 0.3906% | 0.1953% |

Exact full-prefix teacher forcing makes the small fixed-modulus task fit but
does not expose reusable modular-reduction semantics. Compressing the decoder
or focusing all updates on T1 does not rescue variable-modulus generalization.
This branch fails the local gate and should not consume an H100 attempt.
