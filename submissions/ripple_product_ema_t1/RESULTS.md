# Efficient T=1 EMA product-ripple model

This variant keeps the complete `ripple_product_ema` student path and all
baseline losses unchanged. It narrows only the temporal-teacher computation:

- the teacher runs on T=1 rows only;
- it executes exactly one complete outer square rather than three;
- EMA KL is masked to the T=1 endpoint positions;
- teacher parameters still update by EMA after every student optimizer step;
- evaluation remains the original student-only 64-step path.

This targets the atomic square transition that directly controls certification
while avoiding teacher work on T=2 and T=3 rows. Student and teacher receive the
same untouched registers in one forward. There are no process labels,
arithmetic state targets, masking, augmentation, or teacher hard labels.

## Local results

| model | updates | final train-batch exact | test exact | OOD exact | mean exact |
|---|---:|---:|---:|---:|---:|
| full three-step EMA | 236 | 50.00% | 18.33% | 11.00% | 14.67% |
| efficient T=1 EMA | **338** | **59.38%** | **20.00%** | **21.00%** | **20.50%** |
| efficient T=1 EMA, full E5 60s | 1,207 | 0.00% | 0.0833% | 0.0000% | 0.0417% |

The efficient variant completes 43% more updates in the same ten-second CPU
budget. Its loss falls from 3.1403 at step 1 to 0.8240 at the final update. It
answers 12/60 generated test rows and 21/100 generated OOD rows exactly, using
the same 725,780 persistent state elements as full EMA.

Compilation, source validation, optimizer/state validation, finite backward,
EMA callback execution, and evaluation all pass. The diff from full EMA is
limited to teacher row/depth selection and the KL mask; the student arithmetic
and supervised loss are unchanged.

## Full-E5 rejection

The favorable generated smoke does not transfer. On the 60-second full-E5
gate, the model completes 1,207 updates but answers only 1/1,200 test rows and
0/600 OOD rows exactly. Final training-batch loss is 2.82753 and exact accuracy
is 0%. Seen-N T=1 is 0.9766% (5/512), OOD-N T=1 is 0.1953% (1/512), and no
depth rung certifies.

Restricting the teacher to the atomic T=1 transition recovers throughput and
preserves the small seen-T=1 movement, but removes the broader temporal signal
that produced the full EMA model's 1.0417% mean and 1.1667% OOD accuracy. The
20.5% generated-smoke result was therefore misleading for full E5. This path is
rejected and should not receive an H100 run.
