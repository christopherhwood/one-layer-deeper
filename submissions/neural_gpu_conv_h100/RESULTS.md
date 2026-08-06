# Convolutional Neural-GPU H100 attempt

## Hidden-recurrence-safe follow-up (2026-08-06)

The follow-up removes the last recurrence-specific assumptions from the learned
cell: it retains original `x` as generic immutable context, rolls the same cell
to the actual training T (up to eight), and adds only a generic adjacent-token
CRF refinement. There are no product features or arithmetic operations in the
forward path.

With probabilistic evaluation (the Viterbi ablation crystallized frequent paths
too early), it reached 12.5% exact in the ten-second fixed smoke. On the matched
60-second variable-modulus E5 gate it completed 332 updates and reached 0.75%
ordinary mean exact, 15.88% token accuracy, 0.39% seen-N T=1 exact, and 0% OOD-N
T=1 exact (16.97% OOD-N token). This is competitive with the previous generic
frontier but not a breakthrough and does not clear the H100 accuracy gate.

An independently generated hidden-recurrence probe replaced squaring with
`state <- 3*state + 1 (mod N)` without changing submission code. Ten CPU seconds
gave 1.17% mean exact; 60 seconds fit training batches to 25% exact but reached
only 1.33% held-out and 0% unseen-T exact. The architecture is genuinely
recurrence-flexible, but still learns corpus shortcuts rather than the reusable
program.

The width-128 tied convolutional cellular model showed the wave's strongest
short relaxed signal (16.36% ordinary token and 13.50% last-token accuracy after
83 CPU updates), so it passed the softened H100 gate.

Hosted Easy/E5 submission `d073b2ff-4a28-41c3-a56f-c06e22f7aa94` completed
1,502 updates in 60 seconds and scored **0.88%**, with 1.1% test, 0.7% OOD, and
no seen/OOD-N certification.  Training loss remained around 3.1 and batch exact
was usually 0%, ending at 6.2%.  H100 throughput was healthy; the architecture
failed to fit the variable-modulus rule rather than being compute-limited.
