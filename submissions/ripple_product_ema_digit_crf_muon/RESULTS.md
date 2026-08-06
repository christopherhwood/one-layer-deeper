# Hybrid Muon final-CRF results

## Pre-run checks

Python compilation, source validation, whitespace checks, model/optimizer
validation, and a full real-E5 forward/loss/backward/update/scheduler contract
passed on CPU.

The model retains 777,476 persistent scalar elements and has 27 trainable
tensors. The named split assigned eight tensors exclusively to Muon and the
other 19 exclusively to AdamW. Every tensor had a finite gradient and changed
on the update. After that update, Muon held 344,064 momentum elements and
AdamW held 89,367 state elements, for 433,431 total optimizer-state elements.
The contract-batch combined loss was 4.781881.

## Ten-second fixed-modulus contract smoke

Seed 74 completed 88 optimizer updates in 10.01 seconds. Final combined
training loss was 0.96442.

| split | exact accuracy | correct / examples | hard-path CE |
|---|---:|---:|---:|
| test | 26.6667% | 16 / 60 | 8.1192 |
| OOD-T | 24.0000% | 24 / 100 | 8.9942 |
| split mean | **25.3333%** | 40 / 160 | 8.5567 |

The first-order AdamW CRF completed 235 updates and reached 19.6667% mean under
the same manifest. Muon improves smoke exact accuracy despite making only 37%
as many CPU updates. This confirms functional optimization but is not evidence
of variable-modulus reduction.

## Fifteen-second variable-E5 screen

The unchanged candidate completed 106 optimizer updates in 15.10 seconds.
Final combined training loss was 4.41232. Evaluation completed both ordinary
splits, all seen-N rungs, and OOD-N through `T=16` in 8.02 seconds.

| ordinary split | exact accuracy | correct / examples | hard-path CE |
|---|---:|---:|---:|
| test | 1.2500% | 15 / 1,200 | 13.4775 |
| OOD | 1.0000% | 6 / 600 | 13.5964 |
| split mean | **1.1250%** | 21 / 1,800 | 13.5369 |

Count-weighted ordinary accuracy is 21 / 1,800, or 1.1667%.

| T | seen-N exact | seen correct | OOD-N exact | OOD-N correct |
|---:|---:|---:|---:|---:|
| 1 | 0.5859% | 3 / 512 | 0.1953% | 1 / 512 |
| 2 | 0.9766% | 5 / 512 | 0.0000% | 0 / 512 |
| 4 | 0.5859% | 3 / 512 | 0.3906% | 2 / 512 |
| 8 | 0.9766% | 5 / 512 | 0.3906% | 2 / 512 |
| 16 | 0.5859% | 3 / 512 | 0.1953% | 1 / 512 |
| 32 | 0.5859% | 3 / 512 | unavailable | not completed |
| 64 | 0.9766% | 5 / 512 | unavailable | not reached |

No rung certified. Relative to the earlier 15-second first-order screen, Muon
raises test exacts from 8 to 15 and is fast enough at evaluation to measure OOD.
However, its 1.1250% ordinary mean is below the optimized AdamW CRF's 1.2083%
60-second replication, and its seen-N and OOD-N T=1 results are also lower.

This is mixed directional evidence: orthogonalized momentum improves exact
endpoint learning per CPU update, but the transfer indicators do not yet show a
better modular transition. CPU Newton-Schulz cost cuts update throughput by
roughly half on E5 and by 63% on the width-two smoke, although that overhead
should be much smaller on H100. No full gate was started pending an explicit
decision about whether the per-update/test movement outweighs weak T=1/OOD
evidence.

## Full 60-second variable-E5 CPU gate

The unchanged candidate completed 425 optimizer updates in 60.10 seconds.
Final combined training loss was 4.27576. Evaluation completed both ordinary
splits and both complete depth ladders in 10.50 seconds.

| ordinary split | exact accuracy | correct / examples | hard-path CE |
|---|---:|---:|---:|
| test | 0.7500% | 9 / 1,200 | 13.4640 |
| OOD | 0.5000% | 3 / 600 | 13.5695 |
| split mean | **0.6250%** | 12 / 1,800 | 13.5167 |

Count-weighted ordinary accuracy is 12 / 1,800, or 0.6667%.

| T | seen-N exact | seen correct | OOD-N exact | OOD-N correct |
|---:|---:|---:|---:|---:|
| 1 | 0.3906% | 2 / 512 | 0.0000% | 0 / 512 |
| 2 | 0.5859% | 3 / 512 | 0.1953% | 1 / 512 |
| 4 | 0.7812% | 4 / 512 | 0.1953% | 1 / 512 |
| 8 | 1.3672% | 7 / 512 | 0.3906% | 2 / 512 |
| 16 | 0.7812% | 4 / 512 | 0.0000% | 0 / 512 |
| 32 | 1.1719% | 6 / 512 | 0.1953% | 1 / 512 |
| 64 | 0.5859% | 3 / 512 | 0.0000% | 0 / 512 |

No rung certified.

## AdamW comparison and decision

| full-gate optimizer/run | updates | test | OOD | split mean | seen T=1 | OOD-N T=1 |
|---|---:|---:|---:|---:|---:|---:|
| optimized AdamW replication | **869** | **1.0833%** | 1.3333% | 1.2083% | **1.5625%** | **0.3906%** |
| earlier best AdamW run | 869 | 1.0000% | **2.0000%** | **1.5000%** | **1.7578%** | not completed |
| hybrid Muon | 425 | 0.7500% | 0.5000% | 0.6250% | 0.3906% | 0.0000% |

Muon completes 51.1% fewer updates than AdamW on CPU. Relative to the optimized
replication it loses four test examples, five OOD examples, six seen-T=1
examples, and both OOD-N T=1 examples; its split mean is 48.3% lower. Relative
to the earlier best run its split mean is 58.3% lower.

The short-screen endpoint efficiency does not persist as mixed-E5
generalization. Although Newton-Schulz would be cheaper on H100, this run gives
no evidence that orthogonalized momentum learns a better modular transition at
matched wall clock, update count, or depth transfer. Reject the hybrid Muon
variant and retain the optimized AdamW first-order CRF.
