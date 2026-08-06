# Clean-teacher, noisy-student product-ripple EMA

This variant preserves the full `ripple_product_ema` teacher path and loss. The
teacher follows each row's complete training T with clean internal features;
the student follows the identical path with 5% dropout applied only after the
learned product/modulus/register feature projection. The input registers are
untouched, and evaluation explicitly disables dropout and skips the teacher.

Everything else remains unchanged: direct student endpoint CE, T=1 phase
endpoint losses, full endpoint EMA KL, teacher decay and consistency ramp,
outer recurrence, annealing, optimizer, and H100 batch/evaluation contract.
There are no process labels, arithmetic state targets, input masking, or data
augmentation.

## Local results

| model | updates | final train-batch exact | test exact | OOD exact | mean exact |
|---|---:|---:|---:|---:|---:|
| clean student + clean EMA teacher | 236 | 50.00% | 18.33% | 11.00% | 14.67% |
| noisy student + clean EMA teacher | **268** | **62.50%** | 18.33% | **15.00%** | **16.67%** |
| noisy student + clean teacher, full E5 60s | 926 | 0.00% | 0.6667% | 0.6667% | 0.6667% |

The noisy-student variant completes 14% more updates in this ten-second run.
Training loss falls from 3.1440 at step 1 to 0.6368 at the final update. It
answers 11/60 generated test rows and 15/100 generated OOD rows exactly. The
model has the same 725,780 persistent state elements as full EMA.

Compilation, source validation, optimizer/state validation, finite backward,
EMA callback execution, and clean evaluation all pass. The source diff from
full EMA is limited to the internal dropout argument and its student/teacher
settings; endpoint and consistency losses are unchanged.

## Full-E5 rejection

The favorable smoke again does not transfer. On full E5, final training-batch
loss is 2.75568 with 0% exact accuracy. The model answers 8/1,200 test rows and
4/600 OOD rows exactly. Seen-N T=1 is 0.7813% (4/512), OOD-N T=1 is 0.1953%
(1/512), and no depth rung certifies.

Clean full EMA reaches 1.0417% mean and 1.1667% OOD on the identical gate;
student feature noise falls to 0.6667% on both. The modest generated-smoke gain
therefore reflects the small task rather than better E5 generalization. This
path is rejected and should not receive an H100 run.
